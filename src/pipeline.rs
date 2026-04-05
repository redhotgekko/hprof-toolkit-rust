//! High-level pipeline for building all binary index files for an hprof dump.
//!
//! [`IndexPaths`] holds the paths to every index file, all derived from the
//! hprof file path.  [`build_all_indexes`] builds every index that does not
//! yet exist, printing progress to stdout.
//!
//! ## Index directory
//!
//! All indexes are created inside `{hprof_stem}.indexes/`, placed in the same
//! directory as the hprof file.  For example, `./heap.dump` produces:
//!
//! ```text
//! ./heap.indexes/
//!   record_index.bin      top-level record index
//!   heap_index/           heap sub-record index files
//!   object_store.bin      sorted combined sub-record index
//!   utf8.bin              UTF-8 name index
//!   loadclass.bin         load-class index
//!   refs.bin              object reference index
//!   frames.bin            HPROF_FRAME index
//!   traces.bin            HPROF_TRACE index
//!   start_threads.bin     HPROF_START_THREAD index
//!   end_threads.bin       HPROF_END_THREAD index
//!   unload_classes.bin    HPROF_UNLOAD_CLASS index
//!   root_unknown.bin      GC_ROOT_UNKNOWN index
//!   root_jni_global.bin   GC_ROOT_JNI_GLOBAL index
//!   root_jni_local.bin    GC_ROOT_JNI_LOCAL index
//!   root_java_frame.bin   GC_ROOT_JAVA_FRAME index
//!   root_native_stack.bin GC_ROOT_NATIVE_STACK index
//!   root_sticky_class.bin GC_ROOT_STICKY_CLASS index
//!   root_thread_block.bin GC_ROOT_THREAD_BLOCK index
//!   root_monitor_used.bin GC_ROOT_MONITOR_USED index
//!   root_thread_obj.bin   GC_ROOT_THREAD_OBJ index
//!   array_boolean.bin     boolean[] size index (largest first)
//!   array_char.bin        char[]    size index (largest first)
//!   array_float.bin       float[]   size index (largest first)
//!   array_double.bin      double[]  size index (largest first)
//!   array_byte.bin        byte[]    size index (largest first)
//!   array_short.bin       short[]   size index (largest first)
//!   array_int.bin         int[]     size index (largest first)
//!   array_long.bin        long[]    size index (largest first)
//!   array_object.bin      Object[]  size index (largest first)
//! ```

use crate::array_index::{ArrayKind, build_array_size_indexes};
use crate::aux_index::{
    build_end_thread_index, build_frame_index, build_start_thread_index, build_trace_index,
    build_unload_class_index,
};
use crate::dominator::build_dominator_and_retained;
use crate::heap_index::index_heap_dumps;
use crate::heap_query::build_name_indexes;
use crate::hprof::{HprofError, map_file};
use crate::object_store::combine_sort_and_split;
use crate::record_index::index_hprof;
use crate::ref_index::build_reference_index;
use crate::root_index::RootIndexReader;
use crate::vfs::{ByteSource, MMapReader, SubIndexDir};
use std::fmt::Write as _;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct IndexData {
    /// The index directory (`{hprof_stem}.indexes/`).
    pub dir: ByteSource,
    /// Top-level record index.
    pub record_index: ByteSource,
    /// Heap sub-record index directory.
    pub heap_index_dir: ByteSource,
    /// Sorted combined sub-record index.
    pub object_store: ByteSource,
    /// UTF-8 name index.
    pub utf8: ByteSource,
    /// Load-class index.
    pub load_class: ByteSource,
    /// Object reference index.
    pub refs: ByteSource,
    /// HPROF_FRAME index.
    pub frames: ByteSource,
    /// HPROF_TRACE index.
    pub traces: ByteSource,
    /// HPROF_START_THREAD index.
    pub start_threads: ByteSource,
    /// HPROF_END_THREAD index.
    pub end_threads: ByteSource,
    /// HPROF_UNLOAD_CLASS index.
    pub unload_classes: ByteSource,
    /// GC_ROOT_UNKNOWN index.
    pub root_unknown: ByteSource,
    /// GC_ROOT_JNI_GLOBAL index.
    pub root_jni_global: ByteSource,
    /// GC_ROOT_JNI_LOCAL index.
    pub root_jni_local: ByteSource,
    /// GC_ROOT_JAVA_FRAME index.
    pub root_java_frame: ByteSource,
    /// GC_ROOT_NATIVE_STACK index.
    pub root_native_stack: ByteSource,
    /// GC_ROOT_STICKY_CLASS index.
    pub root_sticky_class: ByteSource,
    /// GC_ROOT_THREAD_BLOCK index.
    pub root_thread_block: ByteSource,
    /// GC_ROOT_MONITOR_USED index.
    pub root_monitor_used: ByteSource,
    /// GC_ROOT_THREAD_OBJ index.
    pub root_thread_obj: ByteSource,
    /// Dominator tree index (`dominators.bin`).
    pub dominators: ByteSource,
    /// Retained heap size index (`retained.bin`).
    pub retained: ByteSource,
}

// ── IndexPaths ────────────────────────────────────────────────────────────────

/// Paths to all binary index files for a single hprof dump.
///
/// Construct via [`IndexPaths::for_hprof`]; build via [`build_all_indexes`].
pub struct IndexPaths {
    /// The index directory (`{hprof_stem}.indexes/`).
    pub dir: PathBuf,
    /// Top-level record index.
    pub record_index: PathBuf,
    /// Heap sub-record index directory.
    pub heap_index_dir: PathBuf,
    /// Sorted combined sub-record index.
    pub object_store: PathBuf,
    /// UTF-8 name index.
    pub utf8: PathBuf,
    /// Load-class index.
    pub load_class: PathBuf,
    /// Object reference index.
    pub refs: PathBuf,
    /// HPROF_FRAME index.
    pub frames: PathBuf,
    /// HPROF_TRACE index.
    pub traces: PathBuf,
    /// HPROF_START_THREAD index.
    pub start_threads: PathBuf,
    /// HPROF_END_THREAD index.
    pub end_threads: PathBuf,
    /// HPROF_UNLOAD_CLASS index.
    pub unload_classes: PathBuf,
    /// GC_ROOT_UNKNOWN index.
    pub root_unknown: PathBuf,
    /// GC_ROOT_JNI_GLOBAL index.
    pub root_jni_global: PathBuf,
    /// GC_ROOT_JNI_LOCAL index.
    pub root_jni_local: PathBuf,
    /// GC_ROOT_JAVA_FRAME index.
    pub root_java_frame: PathBuf,
    /// GC_ROOT_NATIVE_STACK index.
    pub root_native_stack: PathBuf,
    /// GC_ROOT_STICKY_CLASS index.
    pub root_sticky_class: PathBuf,
    /// GC_ROOT_THREAD_BLOCK index.
    pub root_thread_block: PathBuf,
    /// GC_ROOT_MONITOR_USED index.
    pub root_monitor_used: PathBuf,
    /// GC_ROOT_THREAD_OBJ index.
    pub root_thread_obj: PathBuf,
    /// Dominator tree index (`dominators.bin`).
    pub dominators: PathBuf,
    /// Retained heap size index (`retained.bin`).
    pub retained: PathBuf,
}

impl IndexPaths {
    /// Return the path to the array size index file for `kind`.
    pub fn array_size(&self, kind: ArrayKind) -> PathBuf {
        self.dir.join(kind.file_name())
    }

    /// Derive all index paths from `hprof_path`.
    ///
    /// The index directory is placed adjacent to the hprof file and named
    /// `{stem}.indexes`.  For `./heap.dump` this gives
    /// `./heap.indexes/`.
    pub fn for_hprof(hprof_path: &Path) -> Self {
        let stem = hprof_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "dump".to_string());
        let parent = hprof_path.parent().unwrap_or(Path::new("."));
        let dir = parent.join(format!("{stem}.indexes"));
        Self {
            record_index: dir.join("record_index.bin"),
            heap_index_dir: dir.join("heap_index"),
            object_store: dir.join("object_store.bin"),
            utf8: dir.join("utf8.bin"),
            load_class: dir.join("loadclass.bin"),
            refs: dir.join("refs.bin"),
            frames: dir.join("frames.bin"),
            traces: dir.join("traces.bin"),
            start_threads: dir.join("start_threads.bin"),
            end_threads: dir.join("end_threads.bin"),
            unload_classes: dir.join("unload_classes.bin"),
            root_unknown: dir.join("root_unknown.bin"),
            root_jni_global: dir.join("root_jni_global.bin"),
            root_jni_local: dir.join("root_jni_local.bin"),
            root_java_frame: dir.join("root_java_frame.bin"),
            root_native_stack: dir.join("root_native_stack.bin"),
            root_sticky_class: dir.join("root_sticky_class.bin"),
            root_thread_block: dir.join("root_thread_block.bin"),
            root_monitor_used: dir.join("root_monitor_used.bin"),
            root_thread_obj: dir.join("root_thread_obj.bin"),
            dominators: dir.join("dominators.bin"),
            retained: dir.join("retained.bin"),
            dir,
        }
    }

    pub fn to_data(&self) -> Result<IndexData, HprofError> {
        use crate::vfs::MMapReader;
        Ok(IndexData {
            dir: ByteSource::VecSource(Vec::new()),
            record_index: self.record_index.open_mmap()?,
            heap_index_dir: ByteSource::VecSource(Vec::new()),
            object_store: self.object_store.open_mmap()?,
            utf8: self.utf8.open_mmap()?,
            load_class: self.load_class.open_mmap()?,
            refs: self.refs.open_mmap()?,
            frames: self.frames.open_mmap()?,
            traces: self.traces.open_mmap()?,
            start_threads: self.start_threads.open_mmap()?,
            end_threads: self.end_threads.open_mmap()?,
            unload_classes: self.unload_classes.open_mmap()?,
            root_unknown: self.root_unknown.open_mmap()?,
            root_jni_global: self.root_jni_global.open_mmap()?,
            root_jni_local: self.root_jni_local.open_mmap()?,
            root_java_frame: self.root_java_frame.open_mmap()?,
            root_native_stack: self.root_native_stack.open_mmap()?,
            root_sticky_class: self.root_sticky_class.open_mmap()?,
            root_thread_block: self.root_thread_block.open_mmap()?,
            root_monitor_used: self.root_monitor_used.open_mmap()?,
            root_thread_obj: self.root_thread_obj.open_mmap()?,
            dominators: self.dominators.open_mmap()?,
            retained: self.retained.open_mmap()?,
        })
    }
}

// ── Pipeline ──────────────────────────────────────────────────────────────────

/// Build all index files for `hprof_path`, printing progress to stdout.
///
/// Each index is skipped if its output file (or directory) already exists.
/// Returns the populated [`IndexPaths`] so the caller can open readers or pass
/// paths to HTML generation.
pub fn build_all_indexes(hprof_path: &Path) -> Result<IndexPaths, HprofError> {
    let mut p = IndexPaths::for_hprof(hprof_path);
    std::fs::create_dir_all(&p.dir)?;
    let hprof_pb: PathBuf = hprof_path.to_path_buf();

    // Memory-map the hprof file once; reused across all build steps.
    let hprof_mmap = map_file(&hprof_pb)?;
    let hprof_bytes: &[u8] = hprof_mmap.as_ref();

    // ── Record index ──────────────────────────────────────────────────────────
    if p.record_index.exists() {
        println!("Record index: skipping");
    } else {
        print!("Record index: indexing records … ");
        let t = Instant::now();
        let n = index_hprof(hprof_bytes, &mut p.record_index)?;
        println!("{n} records  ({:.1}s)", t.elapsed().as_secs_f64());
    }

    // ── Heap index ────────────────────────────────────────────────────────────
    if p.heap_index_dir.exists() {
        println!("Heap index: skipping");
    } else {
        print!("Heap index: indexing heap sub-records … ");
        let t = Instant::now();
        let heap_dir = SubIndexDir::fs(p.heap_index_dir.clone());
        let n = index_heap_dumps(&hprof_pb, &p.record_index, &heap_dir)?;
        println!("{n} sub-records  ({:.1}s)", t.elapsed().as_secs_f64());
    }

    // ── Object store + root indexes ───────────────────────────────────────────
    // Both are built in one pass: combine → sort → fan-out roots.
    let store_and_roots_done = p.object_store.exists()
        && p.root_unknown.exists()
        && p.root_jni_global.exists()
        && p.root_jni_local.exists()
        && p.root_java_frame.exists()
        && p.root_native_stack.exists()
        && p.root_sticky_class.exists()
        && p.root_thread_block.exists()
        && p.root_monitor_used.exists()
        && p.root_thread_obj.exists();

    if store_and_roots_done {
        println!("Object store + root indexes: skipping");
    } else {
        print!("Object store + root indexes: combining, sorting, splitting … ");
        let t = Instant::now();
        let heap_dir = SubIndexDir::fs(p.heap_index_dir.clone());
        let mut w_unknown = BufWriter::new(File::create(&p.root_unknown)?);
        let mut w_jni_global = BufWriter::new(File::create(&p.root_jni_global)?);
        let mut w_jni_local = BufWriter::new(File::create(&p.root_jni_local)?);
        let mut w_java_frame = BufWriter::new(File::create(&p.root_java_frame)?);
        let mut w_native_stack = BufWriter::new(File::create(&p.root_native_stack)?);
        let mut w_sticky_class = BufWriter::new(File::create(&p.root_sticky_class)?);
        let mut w_thread_block = BufWriter::new(File::create(&p.root_thread_block)?);
        let mut w_monitor_used = BufWriter::new(File::create(&p.root_monitor_used)?);
        let mut w_thread_obj = BufWriter::new(File::create(&p.root_thread_obj)?);
        let counts = combine_sort_and_split(
            &heap_dir,
            &mut p.object_store,
            &mut [
                &mut w_unknown,
                &mut w_jni_global,
                &mut w_jni_local,
                &mut w_java_frame,
                &mut w_native_stack,
                &mut w_sticky_class,
                &mut w_thread_block,
                &mut w_monitor_used,
                &mut w_thread_obj,
            ],
        )?;
        println!(
            "{} entries; roots: unknown={}, jni_global={}, jni_local={}, java_frame={}, \
             native_stack={}, sticky_class={}, thread_block={}, monitor_used={}, \
             thread_obj={}  ({:.1}s)",
            counts.total,
            counts.roots.root_unknown,
            counts.roots.root_jni_global,
            counts.roots.root_jni_local,
            counts.roots.root_java_frame,
            counts.roots.root_native_stack,
            counts.roots.root_sticky_class,
            counts.roots.root_thread_block,
            counts.roots.root_monitor_used,
            counts.roots.root_thread_obj,
            t.elapsed().as_secs_f64()
        );
    }

    // ── Name indexes ──────────────────────────────────────────────────────────
    if p.utf8.exists() && p.load_class.exists() {
        println!("Name indexes: skipping");
    } else {
        print!("Name indexes: building … ");
        let t = Instant::now();
        let record_index_mmap = map_file(&p.record_index)?;
        let (utf8_n, lc_n) = build_name_indexes(
            hprof_bytes,
            record_index_mmap.as_ref(),
            &mut p.utf8,
            &mut p.load_class,
        )?;
        println!(
            "{utf8_n} UTF-8 names, {lc_n} classes  ({:.1}s)",
            t.elapsed().as_secs_f64()
        );
    }

    // ── Reference index ───────────────────────────────────────────────────────
    if p.refs.exists() {
        println!("Reference index: skipping");
    } else {
        print!("Reference index: building … ");
        let t = Instant::now();
        let object_store_mmap = map_file(&p.object_store)?;
        let utf8_mmap = map_file(&p.utf8)?;
        let lc_mmap = map_file(&p.load_class)?;
        let n = build_reference_index(
            hprof_bytes,
            object_store_mmap.as_ref(),
            utf8_mmap.as_ref(),
            lc_mmap.as_ref(),
            &mut p.refs,
        )?;
        println!("{n} references  ({:.1}s)", t.elapsed().as_secs_f64());
    }

    // ── Dominator tree + retained heap sizes ─────────────────────────────────
    let dominator_done = p.dominators.exists() && p.retained.exists();
    if dominator_done {
        println!("Dominator tree + retained sizes: skipping");
    } else {
        print!("Dominator tree + retained sizes: building … ");
        let t = Instant::now();
        let root_mmaps = [
            p.root_unknown.open_mmap()?,
            p.root_jni_global.open_mmap()?,
            p.root_jni_local.open_mmap()?,
            p.root_java_frame.open_mmap()?,
            p.root_native_stack.open_mmap()?,
            p.root_sticky_class.open_mmap()?,
            p.root_thread_block.open_mmap()?,
            p.root_monitor_used.open_mmap()?,
            p.root_thread_obj.open_mmap()?,
        ];
        let root_readers = [
            RootIndexReader::from_ref(root_mmaps[0].as_ref())?,
            RootIndexReader::from_ref(root_mmaps[1].as_ref())?,
            RootIndexReader::from_ref(root_mmaps[2].as_ref())?,
            RootIndexReader::from_ref(root_mmaps[3].as_ref())?,
            RootIndexReader::from_ref(root_mmaps[4].as_ref())?,
            RootIndexReader::from_ref(root_mmaps[5].as_ref())?,
            RootIndexReader::from_ref(root_mmaps[6].as_ref())?,
            RootIndexReader::from_ref(root_mmaps[7].as_ref())?,
            RootIndexReader::from_ref(root_mmaps[8].as_ref())?,
        ];
        let (dom_n, ret_n) = build_dominator_and_retained(
            &hprof_pb,
            &p.object_store,
            &root_readers,
            &mut p.dominators,
            &mut p.retained,
        )?;
        println!(
            "{dom_n} dominator entries, {ret_n} retained entries  ({:.1}s)",
            t.elapsed().as_secs_f64()
        );
    }

    // ── Auxiliary indexes ─────────────────────────────────────────────────────
    let aux_done = p.frames.exists()
        && p.traces.exists()
        && p.start_threads.exists()
        && p.end_threads.exists()
        && p.unload_classes.exists();

    if aux_done {
        println!("Auxiliary indexes: skipping");
    } else {
        print!("Auxiliary indexes: building … ");
        let t = Instant::now();
        let frames = build_frame_index(hprof_bytes, &p.record_index, &mut p.frames)?;
        let traces = build_trace_index(hprof_bytes, &p.record_index, &mut p.traces)?;
        let start_threads =
            build_start_thread_index(hprof_bytes, &p.record_index, &mut p.start_threads)?;
        let end_threads = build_end_thread_index(hprof_bytes, &p.record_index, &mut p.end_threads)?;
        let unload_classes =
            build_unload_class_index(hprof_bytes, &p.record_index, &mut p.unload_classes)?;
        println!(
            "{frames} frames, {traces} traces, {start_threads} threads, \
             {end_threads} ends, {unload_classes} unloads  ({:.1}s)",
            t.elapsed().as_secs_f64()
        );
    }

    // ── Array size indexes ────────────────────────────────────────────────────
    let arrays_done = ArrayKind::ALL.iter().all(|k| p.array_size(*k).exists());
    if arrays_done {
        println!("Array size indexes: skipping");
    } else {
        print!("Array size indexes: building … ");
        let t = Instant::now();
        let mut array_outputs = [
            p.dir.join(ArrayKind::Boolean.file_name()),
            p.dir.join(ArrayKind::Char.file_name()),
            p.dir.join(ArrayKind::Float.file_name()),
            p.dir.join(ArrayKind::Double.file_name()),
            p.dir.join(ArrayKind::Byte.file_name()),
            p.dir.join(ArrayKind::Short.file_name()),
            p.dir.join(ArrayKind::Int.file_name()),
            p.dir.join(ArrayKind::Long.file_name()),
            p.dir.join(ArrayKind::Object.file_name()),
        ];
        let object_store_mmap2 = map_file(&p.object_store)?;
        let counts =
            build_array_size_indexes(hprof_bytes, object_store_mmap2.as_ref(), &mut array_outputs)?;
        let mut summary = String::new();
        for (kind, count) in ArrayKind::ALL.iter().zip(counts.iter()) {
            if *count > 0 {
                write!(summary, " {}={count}", kind.slug()).unwrap_or(());
            }
        }
        println!("({:.1}s){summary}", t.elapsed().as_secs_f64());
    }

    Ok(p)
}
