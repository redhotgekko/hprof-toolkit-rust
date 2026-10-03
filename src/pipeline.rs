//! High-level pipeline for building every index for an hprof dump.
//!
//! [`build_indexes`] is the real pipeline: it takes the hprof bytes, an
//! [`IndexStore`] to write into, [`IndexOptions`] and a [`Progress`] sink.
//! It never touches the filesystem itself, so the whole pipeline runs
//! unchanged against [`crate::index::MemStore`] in tests.
//!
//! [`build_all_indexes`] is the filesystem convenience wrapper used by the
//! CLI and examples: it memory-maps the hprof and builds into
//! `{hprof_stem}.indexes/` next to it.
//!
//! ## Completion tracking
//!
//! A step is considered done only when the store's manifest
//! ([`crate::index::Manifest`], entry `index.json`) lists every output of
//! that step **and** the entries exist.  The manifest is rewritten after
//! each step commits, so an interrupted build resumes at the first
//! unfinished step, and a manifest for a different hprof or an older
//! [`crate::index::FORMAT_VERSION`] causes a full rebuild.
//!
//! ## Index entries
//!
//! All entries are created inside the store (a directory on disk).  For
//! `./heap.dump` the directory is `./heap.indexes/`:
//!
//! ```text
//!   index.json            manifest
//!   record_index.bin      top-level record index
//!   object_store.bin      sorted combined sub-record index
//!   instances_by_class.bin objects grouped by class key
//!   class_histogram.bin   per-class counts and shallow bytes, largest first
//!   utf8.bin              UTF-8 name index
//!   loadclass.bin         load-class index
//!   refs.bin              object reference index
//!   frames.bin            HPROF_FRAME index
//!   traces.bin            HPROF_TRACE index
//!   start_threads.bin     HPROF_START_THREAD index
//!   end_threads.bin       HPROF_END_THREAD index
//!   unload_classes.bin    HPROF_UNLOAD_CLASS index
//!   root_*.bin            nine per-type GC root indexes
//!   dominators.bin        dominator tree (optional, see IndexOptions::retained)
//!   retained.bin          retained heap sizes (optional)
//!   retained_by_size.bin  retained sizes ranked largest-first (optional)
//!   dominator_children.bin dominator tree keyed by parent (optional)
//!   array_*.bin           nine per-kind array size indexes (largest first)
//! ```
//!
//! `heap_index/…` and `*.part.*` entries are build intermediates and are
//! removed once the entry they feed is committed.

use crate::array_index::{ArrayKind, build_array_size_indexes};
use crate::aux_index::{
    build_end_thread_index, build_frame_index, build_start_thread_index, build_trace_index,
    build_unload_class_index,
};
use crate::class_index::build_class_indexes;
use crate::dominator::{build_dominator_and_retained, check_memory};
use crate::heap_index::index_heap_dumps;
use crate::heap_index::sub_record::SUB_INDEX_ENTRY_SIZE;
use crate::heap_query::build_name_indexes;
use crate::hprof::HprofError;
use crate::index::{ByteSource, FsStore, HprofIdentity, IndexStore, Manifest, StoreWriter, names};
use crate::object_store::combine_sort_and_split;
use crate::progress::Progress;
use crate::record_index::index_hprof;
use crate::ref_index::REF_ENTRY_SIZE;
use crate::ref_index::build_reference_index;
use crate::root_index::RootIndexReader;
use std::fmt::Write as _;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ── IndexOptions ──────────────────────────────────────────────────────────────

/// What to build.
#[derive(Debug, Clone)]
pub struct IndexOptions {
    /// Build the dominator tree and retained-size indexes.  This is the only
    /// step whose memory use scales with the heap: about 72 bytes per object plus 12
    /// per reference.
    pub retained: bool,
    /// Rebuild every index even if the manifest says it is done.
    pub force: bool,
    /// Memory budget in bytes for the dominator step.  `None` (the default)
    /// uses the memory currently available on the machine.  The step is
    /// refused up front, with an actionable error, when its estimate exceeds
    /// the budget.
    pub max_memory: Option<u64>,
    /// Where the indexes live.  `None` (the default) uses
    /// [`default_index_dir`]: `{stem}.indexes` next to the dump.
    pub index_dir: Option<PathBuf>,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            retained: true,
            force: false,
            max_memory: None,
            index_dir: None,
        }
    }
}

impl IndexOptions {
    /// The index directory these options select for `hprof_path`.
    pub fn dir_for(&self, hprof_path: &Path) -> PathBuf {
        self.index_dir
            .clone()
            .unwrap_or_else(|| default_index_dir(hprof_path))
    }
}

/// The default index directory for `hprof_path`: `{stem}.indexes` next to it.
pub fn default_index_dir(hprof_path: &Path) -> PathBuf {
    let stem = hprof_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "dump".to_string());
    let parent = hprof_path.parent().unwrap_or(Path::new("."));
    parent.join(format!("{stem}.indexes"))
}

// ── Filesystem wrappers ───────────────────────────────────────────────────────

/// Build all index files for `hprof_path` with default options.
///
/// Indexes go into `{hprof_stem}.indexes/` next to the dump.  Each step is
/// skipped if the manifest says it is done.  Progress is reported through
/// `progress`.  Returns the index directory.
pub fn build_all_indexes(
    hprof_path: &Path,
    progress: &dyn Progress,
) -> Result<PathBuf, HprofError> {
    build_all_indexes_with(hprof_path, &IndexOptions::default(), progress)
}

/// [`build_all_indexes`] with explicit [`IndexOptions`].
pub fn build_all_indexes_with(
    hprof_path: &Path,
    opts: &IndexOptions,
    progress: &dyn Progress,
) -> Result<PathBuf, HprofError> {
    let dir = opts.dir_for(hprof_path);
    let store = FsStore::open_or_create(&dir)?;
    let hprof = ByteSource::map_file(hprof_path)?;
    build_indexes(hprof.as_ref(), &store, opts, progress)?;
    Ok(dir)
}

// ── Step bookkeeping ──────────────────────────────────────────────────────────

/// Tracks which steps are done through the manifest.
struct Steps<'s> {
    store: &'s dyn IndexStore,
    manifest: Manifest,
    force: bool,
}

impl Steps<'_> {
    /// `true` when every entry in `outputs` was committed by a finished step
    /// and still exists.
    fn done(&self, outputs: &[&str]) -> bool {
        !self.force
            && outputs
                .iter()
                .all(|n| self.manifest.is_built(n) && self.store.exists(n))
    }

    /// Record `outputs` as committed and persist the manifest.
    fn finished(&mut self, outputs: &[&str]) -> Result<(), HprofError> {
        self.manifest.mark_built(outputs);
        self.manifest.write(self.store)
    }
}

// ── Pipeline ──────────────────────────────────────────────────────────────────

/// Build every index for `hprof` into `store`.
///
/// Steps run in dependency order; every output is written through a staged
/// [`StoreWriter`] and committed only on success, and the manifest is
/// updated after each step, so an interrupted build never leaves a
/// truncated index under its final name and resumes where it stopped.
pub fn build_indexes(
    hprof: &[u8],
    store: &dyn IndexStore,
    opts: &IndexOptions,
    progress: &dyn Progress,
) -> Result<(), HprofError> {
    let identity = HprofIdentity::of(hprof)?;
    let manifest = match Manifest::read(store)? {
        Some(m) if !opts.force && m.matches(&identity) => m,
        Some(m) if !opts.force => {
            let why = m
                .mismatch_reason(&identity)
                .unwrap_or_else(|| "manifest mismatch".to_owned());
            progress.step("Manifest", &format!("rebuilding everything: {why}"));
            Manifest::new(identity)
        }
        _ => Manifest::new(identity),
    };
    let mut steps = Steps {
        store,
        manifest,
        force: opts.force,
    };
    let secs = |t: Instant| format!("{:.1}s", t.elapsed().as_secs_f64());

    // ── Record index ──────────────────────────────────────────────────────────
    const RECORD: &str = "Record index";
    if steps.done(&[names::RECORD_INDEX]) {
        progress.step(RECORD, "skipping");
    } else {
        let t = Instant::now();
        let mut w = store.create(names::RECORD_INDEX)?;
        let n = index_hprof(hprof, &mut w)?;
        w.commit()?;
        steps.finished(&[names::RECORD_INDEX])?;
        progress.step(RECORD, &format!("{n} records ({})", secs(t)));
    }
    let record_index = store.open(names::RECORD_INDEX)?;

    // ── Object store + root indexes ───────────────────────────────────────────
    // The per-segment heap index is an intermediate: built, concatenated,
    // sorted, fanned out into the root indexes, then removed.
    const STORE: &str = "Object store + root indexes";
    let mut store_outputs: Vec<&str> = vec![names::OBJECT_STORE];
    store_outputs.extend(names::ROOTS);
    if steps.done(&store_outputs) {
        progress.step(STORE, "skipping");
    } else {
        const HEAP: &str = "Heap index";
        let t = Instant::now();
        store.remove_prefix(names::HEAP_INDEX_PREFIX)?;
        let n = index_heap_dumps(hprof, record_index.as_ref(), store)?;
        progress.step(HEAP, &format!("{n} sub-records ({})", secs(t)));

        let t = Instant::now();
        let mut combined = store.create(names::OBJECT_STORE)?;
        let mut roots: Vec<Box<dyn StoreWriter>> = names::ROOTS
            .iter()
            .map(|n| store.create(n))
            .collect::<Result<_, _>>()?;
        let counts = {
            let mut writers: Vec<&mut dyn Write> = roots
                .iter_mut()
                .map(|b| b.as_mut() as &mut dyn Write)
                .collect();
            combine_sort_and_split(store, &mut *combined, writers.as_mut_slice())?
        };
        combined.commit()?;
        for w in roots {
            w.commit()?;
        }
        store.remove_prefix(names::HEAP_INDEX_PREFIX)?;
        steps.finished(&store_outputs)?;
        let r = &counts.roots;
        progress.step(
            STORE,
            &format!(
                "{} entries; roots: unknown={}, jni_global={}, jni_local={}, java_frame={}, \
                 native_stack={}, sticky_class={}, thread_block={}, monitor_used={}, \
                 thread_obj={} ({})",
                counts.total,
                r.root_unknown,
                r.root_jni_global,
                r.root_jni_local,
                r.root_java_frame,
                r.root_native_stack,
                r.root_sticky_class,
                r.root_thread_block,
                r.root_monitor_used,
                r.root_thread_obj,
                secs(t)
            ),
        );
    }
    let object_store = store.open(names::OBJECT_STORE)?;

    // ── Per-class indexes ─────────────────────────────────────────────────────
    const CLASSES: &str = "Class indexes";
    if steps.done(&[names::INSTANCES_BY_CLASS, names::CLASS_HISTOGRAM]) {
        progress.step(CLASSES, "skipping");
    } else {
        let t = Instant::now();
        let (objects, classes) = build_class_indexes(hprof, object_store.as_ref(), store)?;
        steps.finished(&[names::INSTANCES_BY_CLASS, names::CLASS_HISTOGRAM])?;
        progress.step(
            CLASSES,
            &format!("{objects} objects in {classes} class keys ({})", secs(t)),
        );
    }

    // ── Name indexes ──────────────────────────────────────────────────────────
    const NAMES: &str = "Name indexes";
    if steps.done(&[names::UTF8, names::LOAD_CLASS]) {
        progress.step(NAMES, "skipping");
    } else {
        let t = Instant::now();
        let mut utf8 = store.create(names::UTF8)?;
        let mut lc = store.create(names::LOAD_CLASS)?;
        let (utf8_n, lc_n) =
            build_name_indexes(hprof, record_index.as_ref(), &mut *utf8, &mut *lc)?;
        utf8.commit()?;
        lc.commit()?;
        steps.finished(&[names::UTF8, names::LOAD_CLASS])?;
        progress.step(
            NAMES,
            &format!("{utf8_n} UTF-8 names, {lc_n} classes ({})", secs(t)),
        );
    }

    // ── Reference index ───────────────────────────────────────────────────────
    const REFS: &str = "Reference index";
    if steps.done(&[names::REFS]) {
        progress.step(REFS, "skipping");
    } else {
        let t = Instant::now();
        let n = build_reference_index(hprof, object_store.as_ref(), store)?;
        steps.finished(&[names::REFS])?;
        progress.step(REFS, &format!("{n} references ({})", secs(t)));
    }

    // ── Dominator tree + retained heap sizes ─────────────────────────────────
    const DOM: &str = "Dominator tree + retained sizes";
    if !opts.retained {
        progress.step(DOM, "skipped (retained heap disabled)");
    } else if steps.done(&names::RETAINED_SET) {
        progress.step(DOM, "skipping");
    } else {
        // Fail fast if this step's memory would not fit (see dominator docs).
        let objects = (object_store.len() / SUB_INDEX_ENTRY_SIZE) as u64;
        let references = (store.open(names::REFS)?.len() / REF_ENTRY_SIZE) as u64;
        let estimate = check_memory(objects, references, opts.max_memory)?;
        progress.step(
            DOM,
            &format!(
                "{objects} objects, {references} references: estimated peak memory {} MB",
                estimate / (1024 * 1024)
            ),
        );
        let t = Instant::now();
        let root_sources: Vec<ByteSource> = names::ROOTS
            .iter()
            .map(|n| store.open(n))
            .collect::<Result<_, _>>()?;
        for s in &root_sources {
            RootIndexReader::from_ref(s.as_ref())?;
        }
        let root_readers: [RootIndexReader<'_>; 9] =
            std::array::from_fn(|i| RootIndexReader::from_slice(root_sources[i].as_ref()));
        let (dom_n, ret_n) =
            build_dominator_and_retained(hprof, object_store.as_ref(), &root_readers, store)?;
        steps.finished(&names::RETAINED_SET)?;
        progress.step(
            DOM,
            &format!(
                "{dom_n} dominator entries, {ret_n} retained entries ({})",
                secs(t)
            ),
        );
    }

    // ── Auxiliary indexes ─────────────────────────────────────────────────────
    const AUX: &str = "Auxiliary indexes";
    let aux_outputs = [
        names::FRAMES,
        names::TRACES,
        names::START_THREADS,
        names::END_THREADS,
        names::UNLOAD_CLASSES,
    ];
    if steps.done(&aux_outputs) {
        progress.step(AUX, "skipping");
    } else {
        let t = Instant::now();
        let ri = record_index.as_ref();
        let mut counts = [0u64; 5];
        type AuxBuilder = fn(&[u8], &[u8], &mut dyn StoreWriter) -> Result<u64, HprofError>;
        let builders: [AuxBuilder; 5] = [
            build_frame_index,
            build_trace_index,
            build_start_thread_index,
            build_end_thread_index,
            build_unload_class_index,
        ];
        for (i, (name, build)) in aux_outputs.iter().zip(builders.iter()).enumerate() {
            let mut w = store.create(name)?;
            counts[i] = build(hprof, ri, &mut *w)?;
            w.commit()?;
        }
        steps.finished(&aux_outputs)?;
        progress.step(
            AUX,
            &format!(
                "{} frames, {} traces, {} threads, {} ends, {} unloads ({})",
                counts[0],
                counts[1],
                counts[2],
                counts[3],
                counts[4],
                secs(t)
            ),
        );
    }

    // ── Array size indexes ────────────────────────────────────────────────────
    const ARRAYS: &str = "Array size indexes";
    if steps.done(&names::ARRAYS) {
        progress.step(ARRAYS, "skipping");
    } else {
        let t = Instant::now();
        let counts = build_array_size_indexes(hprof, object_store.as_ref(), store)?;
        steps.finished(&names::ARRAYS)?;
        let mut summary = String::new();
        for (kind, count) in ArrayKind::ALL.iter().zip(counts.iter()) {
            if *count > 0 {
                write!(summary, " {}={count}", kind.slug()).unwrap_or(());
            }
        }
        progress.step(ARRAYS, &format!("({}){summary}", secs(t)));
    }

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{MANIFEST_NAME, MemStore};
    use crate::progress::NoProgress;
    use crate::test_util::standard_heap;
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicIsize, Ordering};

    fn all_final_names() -> Vec<&'static str> {
        let mut v = vec![
            MANIFEST_NAME,
            names::RECORD_INDEX,
            names::OBJECT_STORE,
            names::INSTANCES_BY_CLASS,
            names::CLASS_HISTOGRAM,
            names::UTF8,
            names::LOAD_CLASS,
            names::REFS,
            names::FRAMES,
            names::TRACES,
            names::START_THREADS,
            names::END_THREADS,
            names::UNLOAD_CLASSES,
        ];
        v.extend(names::RETAINED_SET);
        v.extend(names::ROOTS);
        v.extend(names::ARRAYS);
        v
    }

    /// Every entry except the manifest (whose timestamps vary).
    fn content_without_manifest(store: &MemStore) -> BTreeMap<String, Vec<u8>> {
        let mut s = store.snapshot();
        s.remove(MANIFEST_NAME);
        s
    }

    #[test]
    fn pipeline_builds_every_entry_in_memory_and_removes_intermediates() {
        let hprof = standard_heap();
        let store = MemStore::new();
        build_indexes(&hprof, &store, &IndexOptions::default(), &NoProgress).unwrap();

        let mut expected = all_final_names();
        expected.sort();
        assert_eq!(store.list("").unwrap(), expected);
        assert!(store.list(names::HEAP_INDEX_PREFIX).unwrap().is_empty());

        let manifest = Manifest::read(&store).unwrap().unwrap();
        for name in all_final_names() {
            if name != MANIFEST_NAME {
                assert!(manifest.is_built(name), "{name} not in manifest");
            }
        }
        assert!(manifest.matches(&HprofIdentity::of(&hprof).unwrap()));
    }

    #[test]
    fn pipeline_skips_retained_when_disabled_and_adds_it_later() {
        let hprof = standard_heap();
        let store = MemStore::new();
        let opts = IndexOptions {
            retained: false,
            force: false,
            ..IndexOptions::default()
        };
        build_indexes(&hprof, &store, &opts, &NoProgress).unwrap();
        assert!(!store.exists(names::DOMINATORS));
        assert!(!store.exists(names::RETAINED));
        assert!(store.exists(names::REFS));

        // A later run with retained enabled builds only the missing step.
        let log = Log::default();
        build_indexes(&hprof, &store, &IndexOptions::default(), &log).unwrap();
        let lines = log.lines();
        assert!(store.exists(names::RETAINED));
        assert!(
            lines
                .iter()
                .filter(|l| !l.ends_with("skipping"))
                .all(|l| l.starts_with("Dominator tree")),
            "{lines:?}"
        );
    }

    #[test]
    fn pipeline_second_run_is_a_no_op_and_force_rebuilds() {
        let hprof = standard_heap();
        let store = MemStore::new();
        build_indexes(&hprof, &store, &IndexOptions::default(), &NoProgress).unwrap();
        let before = content_without_manifest(&store);

        let log = Log::default();
        build_indexes(&hprof, &store, &IndexOptions::default(), &log).unwrap();
        let lines = log.lines();
        assert!(lines.iter().all(|l| l.ends_with("skipping")), "{lines:?}");
        assert_eq!(content_without_manifest(&store), before);

        let opts = IndexOptions {
            retained: true,
            force: true,
            ..IndexOptions::default()
        };
        let log = Log::default();
        build_indexes(&hprof, &store, &opts, &log).unwrap();
        let lines = log.lines();
        assert!(lines.iter().all(|l| !l.ends_with("skipping")), "{lines:?}");
        assert_eq!(content_without_manifest(&store), before);
    }

    #[test]
    fn entries_not_in_the_manifest_are_rebuilt() {
        let hprof = standard_heap();
        let store = MemStore::new();
        build_indexes(&hprof, &store, &IndexOptions::default(), &NoProgress).unwrap();
        // Forge a manifest that forgets the reference index: the file exists
        // but must be rebuilt because completion is tracked in the manifest.
        let mut m = Manifest::read(&store).unwrap().unwrap();
        m.built.remove(names::REFS);
        m.write(&store).unwrap();
        let log = Log::default();
        build_indexes(&hprof, &store, &IndexOptions::default(), &log).unwrap();
        let rebuilt: Vec<String> = log
            .lines()
            .into_iter()
            .filter(|l| !l.ends_with("skipping"))
            .collect();
        assert_eq!(rebuilt.len(), 1, "{rebuilt:?}");
        assert!(rebuilt[0].starts_with("Reference index"));
    }

    #[test]
    fn manifest_for_another_hprof_triggers_full_rebuild() {
        let hprof = standard_heap();
        // A different length, and the same length dumped at another time.
        let edits: [fn(&mut Manifest); 2] = [|m| m.hprof_len += 1, |m| m.hprof_timestamp_ms += 1];
        for edit in edits {
            let store = MemStore::new();
            build_indexes(&hprof, &store, &IndexOptions::default(), &NoProgress).unwrap();
            let mut m = Manifest::read(&store).unwrap().unwrap();
            edit(&mut m);
            m.write(&store).unwrap();
            let log = Log::default();
            build_indexes(&hprof, &store, &IndexOptions::default(), &log).unwrap();
            let lines = log.lines();
            assert!(
                lines[0].starts_with("Manifest: rebuilding everything"),
                "{lines:?}"
            );
            assert!(
                lines[1..].iter().all(|l| !l.ends_with("skipping")),
                "{lines:?}"
            );
        }
    }

    /// Interrupt the build by failing the n-th byte write, then finish it with
    /// a healthy store: the result must equal an uninterrupted build.
    #[test]
    fn interrupted_builds_resume_to_the_same_result() {
        let hprof = standard_heap();
        let reference = MemStore::new();
        build_indexes(&hprof, &reference, &IndexOptions::default(), &NoProgress).unwrap();
        let expected = content_without_manifest(&reference);

        // How many `write` calls does a full build make?  (Counted through
        // the same wrapper with a budget that never runs out.)
        let counter = Arc::new(AtomicIsize::new(isize::MAX));
        let counting = FailingStore {
            inner: MemStore::new(),
            budget: Arc::clone(&counter),
        };
        build_indexes(&hprof, &counting, &IndexOptions::default(), &NoProgress).unwrap();
        let total_writes = isize::MAX - counter.load(Ordering::SeqCst);
        assert!(
            total_writes > 20,
            "fixture too small: {total_writes} writes"
        );

        // Crash at a spread of points, including the first write of the last
        // step and the very last write.
        let budgets = [
            1,
            3,
            total_writes / 8,
            total_writes / 4,
            total_writes / 2,
            (total_writes * 3) / 4,
            total_writes - 1,
        ];
        for fail_after_writes in budgets {
            let store = MemStore::new();
            let failing = FailingStore {
                inner: store.clone(),
                budget: Arc::new(AtomicIsize::new(fail_after_writes)),
            };
            let interrupted =
                build_indexes(&hprof, &failing, &IndexOptions::default(), &NoProgress);
            assert!(
                interrupted.is_err(),
                "build with write budget {fail_after_writes}/{total_writes} should fail"
            );
            // Resume with a healthy store: every step whose manifest entry is
            // missing is redone, leftover parts are cleaned up, and the
            // result equals the reference (an extra entry would show up as a
            // difference).
            build_indexes(&hprof, &store, &IndexOptions::default(), &NoProgress).unwrap();
            assert_eq!(
                content_without_manifest(&store),
                expected,
                "resumed build differs from reference (budget {fail_after_writes}/{total_writes})"
            );
        }
    }

    #[test]
    fn dominator_step_is_refused_when_it_would_not_fit_and_can_be_retried() {
        let hprof = standard_heap();
        let store = MemStore::new();
        let tiny = IndexOptions {
            max_memory: Some(1),
            ..IndexOptions::default()
        };
        let err = build_indexes(&hprof, &store, &tiny, &NoProgress)
            .expect_err("a 1-byte budget cannot fit the dominator step");
        assert!(
            matches!(err, HprofError::InsufficientMemory { needed, available: 1, .. } if needed > 1),
            "{err:?}"
        );
        let msg = err.to_string();
        assert!(msg.contains("--no-retained"), "{msg}");
        // Everything before the refused step was built and recorded, nothing
        // of the retained family exists, and no scratch entry leaked.
        assert!(store.exists(names::REFS));
        assert!(!store.exists(names::DOMINATORS));
        assert!(store.list(".part.").unwrap().is_empty());

        // Disabling the step avoids the check entirely...
        let skip = IndexOptions {
            retained: false,
            max_memory: Some(1),
            ..IndexOptions::default()
        };
        build_indexes(&hprof, &store, &skip, &NoProgress).unwrap();
        // ...and a sufficient budget lets the same store finish the job.
        let roomy = IndexOptions {
            max_memory: Some(u64::MAX),
            ..IndexOptions::default()
        };
        build_indexes(&hprof, &store, &roomy, &NoProgress).unwrap();
        assert!(store.exists(names::RETAINED_BY_SIZE));
    }

    #[test]
    fn index_directory_defaults_next_to_the_dump_and_can_be_overridden() {
        let dump = Path::new("/tmp/heap.dump");
        assert_eq!(default_index_dir(dump), Path::new("/tmp/heap.indexes"));
        assert_eq!(
            IndexOptions::default().dir_for(dump),
            Path::new("/tmp/heap.indexes")
        );
        let custom = IndexOptions {
            index_dir: Some(PathBuf::from("/fast/idx")),
            ..IndexOptions::default()
        };
        assert_eq!(custom.dir_for(dump), Path::new("/fast/idx"));
    }

    // ── Test doubles ──────────────────────────────────────────────────────────

    /// Progress sink that records every line.
    #[derive(Default)]
    struct Log(std::sync::Mutex<Vec<String>>);

    impl Log {
        fn lines(&self) -> Vec<String> {
            self.0.lock().unwrap().clone()
        }
    }

    impl Progress for Log {
        fn step(&self, stage: &str, detail: &str) {
            self.0.lock().unwrap().push(format!("{stage}: {detail}"));
        }
    }

    /// A store whose writers fail once a shared budget of `write` calls is
    /// exhausted — simulates a crash part-way through a build.
    struct FailingStore {
        inner: MemStore,
        budget: Arc<AtomicIsize>,
    }

    impl IndexStore for FailingStore {
        fn open(&self, name: &str) -> Result<ByteSource, HprofError> {
            self.inner.open(name)
        }
        fn create(&self, name: &str) -> Result<Box<dyn StoreWriter>, HprofError> {
            Ok(Box::new(FailingWriter {
                inner: self.inner.create(name)?,
                budget: Arc::clone(&self.budget),
            }))
        }
        fn exists(&self, name: &str) -> bool {
            self.inner.exists(name)
        }
        fn remove(&self, name: &str) -> Result<(), HprofError> {
            self.inner.remove(name)
        }
        fn list(&self, prefix: &str) -> Result<Vec<String>, HprofError> {
            self.inner.list(prefix)
        }
    }

    struct FailingWriter {
        inner: Box<dyn StoreWriter>,
        budget: Arc<AtomicIsize>,
    }

    impl Write for FailingWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if self.budget.fetch_sub(1, Ordering::SeqCst) <= 0 {
                return Err(std::io::Error::other("simulated crash"));
            }
            self.inner.write(buf)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            self.inner.flush()
        }
    }

    impl StoreWriter for FailingWriter {
        fn as_mut_bytes(&mut self) -> Result<&mut [u8], HprofError> {
            self.inner.as_mut_bytes()
        }
        fn commit(self: Box<Self>) -> Result<(), HprofError> {
            self.inner.commit()
        }
    }
}
