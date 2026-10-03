//! Unified query API for heap dump analysis.
//!
//! [`HeapQuery`] is the single entry point for ad-hoc heap dump analysis.
//! It opens the object store, name, reference, GC-root, class, array, thread
//! and retained-size indexes from one index store and answers questions from
//! them.
//!
//! All data is accessed via memory-mapped files; no dump data is loaded into
//! process memory.  Object lookup is O(log n) via binary search; parallel
//! iteration uses rayon.
//!
//! ## Quick start
//!
//! ```no_run
//! use hprof_toolkit::prelude::*;
//!
//! // Builds the indexes on first use, then just opens them.
//! let heap = HeapQuery::open("heap.hprof")?;
//!
//! // Look up a single object by ID.
//! if let Some(record) = heap.object(0xDEAD_BEEF)? {
//!     match record {
//!         SubRecord::InstanceDump(inst) => {
//!             let name = heap.class_name(inst.class_id).unwrap_or_default();
//!             println!("{name}: {:?}", heap.instance_fields(&inst)?);
//!         }
//!         SubRecord::ClassDump(cd) => {
//!             println!("class: {}", heap.class_name(cd.class_id).unwrap_or_default());
//!         }
//!         _ => {}
//!     }
//! }
//!
//! // Visit all instances in parallel (rayon).
//! // (`ParallelIterator` comes with the prelude.)
//! heap.par_instances().try_for_each(|inst| {
//!     let _ = heap.class_name(inst?.class_id);
//!     Ok::<(), HprofError>(())
//! })?;
//! # Ok::<(), HprofError>(())
//! ```
//!
//! Tests (and anything else that has the dump in memory) use
//! [`HeapQuery::from_store`] with an [`crate::index::MemStore`] instead; no
//! file is involved.

use crate::array_index::{ArrayKind, ArraySizeIter, ArraySizeReader};
use crate::aux_query::{
    AuxRecordIndex, Frame, FrameIter, ResolvedFrame, ResolvedThread, StartThread, StartThreadIter,
    Trace, TraceIter,
};
use crate::class_index::{HistogramRecord, InstanceByClassEntry};
use crate::class_key::ClassKey;
use crate::dominator::{DomChildEntry, DominatorIndex, RetainedBySizeEntry, RetainedIndex};
use crate::heap_index::sub_record::SubIndexEntry;
use crate::heap_index::sub_record::TAG_INSTANCE_DUMP;
use crate::heap_parser::{ClassDump, InstanceDump, SubIndexIter, SubRecord};
use crate::heap_query::name_index::LoadClassEntry;
use crate::heap_query::{Field, HprofIndex};
use crate::hprof::{HprofError, HprofHeader};
use crate::index::{
    ByteSource, FsStore, HprofIdentity, IndexStore, Manifest, RecordFile, RecordIter, names,
};
use crate::pipeline::{IndexOptions, build_all_indexes_with, default_index_dir};
use crate::progress::{NoProgress, Progress};
use crate::ref_index::RefIndex;
use crate::resolved::Value;
use crate::root_index::{GcRootType, RootIndexReader, RootIter};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Borrow the bytes behind a shared [`ByteSource`].
fn bytes(src: &Arc<ByteSource>) -> &[u8] {
    (**src).as_ref()
}

// ── HeapQuery ─────────────────────────────────────────────────────────────────

/// Unified query API over a heap dump and all its index files.
///
/// Owns every index a question can need (objects, names, references, GC roots,
/// classes, arrays, threads, retained sizes) behind one interface.
///
/// All data is accessed via memory-mapped files; no dump data is loaded into
/// process memory.
pub struct HeapQuery {
    /// The hprof bytes, shared with the auxiliary index.
    hprof_data: Arc<ByteSource>,
    hprof_header: HprofHeader,
    combined_data: ByteSource,
    /// UTF-8 name index, shared with the auxiliary index.
    utf8_data: Arc<ByteSource>,
    lc_data: ByteSource,
    aux: AuxRecordIndex,
    /// Per-type GC root data, indexed by [`GcRootType::index()`].
    roots: [ByteSource; 9],
    /// Back-reference index data.
    refs: ByteSource,
    /// Per-kind array size index data, indexed by [`ArrayKind::index()`].
    array_sizes: [ByteSource; 9],
    /// Objects grouped by class key (`instances_by_class.bin`).
    instances_by_class: ByteSource,
    /// Per-class counts and shallow sizes (`class_histogram.bin`).
    class_histogram: ByteSource,
    /// The dominator/retained index family (optional — present only when
    /// the pipeline ran with `IndexOptions::retained`).
    retained_set: Option<RetainedSet>,
    /// Class id ⇄ name maps, built when the query is opened (O(classes),
    /// never O(objects)).
    class_names: ClassNames,
}

/// Every loaded class's dot-notation name, both directions.
#[derive(Default)]
struct ClassNames {
    by_id: HashMap<u64, Arc<str>>,
    /// The lowest class id wins when several classes share a name (as with
    /// the same class loaded by two class loaders).
    by_name: HashMap<Arc<str>, u64>,
}

/// The four entries produced by the dominator step, see
/// [`crate::index::names::RETAINED_SET`].
struct RetainedSet {
    dominators: ByteSource,
    retained: ByteSource,
    by_size: ByteSource,
    children: ByteSource,
}

// ── Paging ────────────────────────────────────────────────────────────────────

/// A window into a ranked or grouped result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Page {
    /// Number of items to skip.
    pub offset: usize,
    /// Maximum number of items to return.
    pub limit: usize,
}

impl Page {
    pub fn new(offset: usize, limit: usize) -> Self {
        Self { offset, limit }
    }

    /// The first `limit` items.
    pub fn first(limit: usize) -> Self {
        Self { offset: 0, limit }
    }
}

/// One page of results plus what the caller needs to page further.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageResult<T> {
    /// The items in this page, in result order.
    pub items: Vec<T>,
    /// Total number of items across all pages.
    pub total: usize,
    /// `true` when `offset + items.len() < total`.
    pub has_more: bool,
}

// ── Bounded scans ─────────────────────────────────────────────────────────────

/// A window over an id-ordered scan that has to read object contents (for
/// example [`HeapQuery::search_strings`]).
///
/// Unlike [`Page`], the number of matches is not known up front, so a scan
/// examines at most `max_scan` entries from `cursor`, stops early once it
/// has `max_results` matches, and reports in [`ScanResult::next_cursor`]
/// where to continue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanWindow {
    /// Index of the first entry to examine (a previous
    /// [`ScanResult::next_cursor`]).
    pub cursor: usize,
    /// Entries to examine at most in this call.
    pub max_scan: usize,
    /// Stop after this many matches.
    pub max_results: usize,
}

impl ScanWindow {
    /// Default `max_scan`.
    pub const DEFAULT_MAX_SCAN: usize = 100_000;
    /// Largest `max_scan` the front ends accept.
    pub const MAX_SCAN_LIMIT: usize = 5_000_000;
    /// Default `max_results`.
    pub const DEFAULT_MAX_RESULTS: usize = 20;
    /// Largest `max_results` the front ends accept.
    pub const MAX_RESULTS_LIMIT: usize = 200;

    pub fn new(cursor: usize, max_scan: usize, max_results: usize) -> Self {
        Self {
            cursor,
            max_scan,
            max_results,
        }
    }

    /// From the start, with the default budget.
    pub fn first() -> Self {
        Self::new(0, Self::DEFAULT_MAX_SCAN, Self::DEFAULT_MAX_RESULTS)
    }

    /// The same budget, continuing from `cursor`.
    pub fn from(self, cursor: usize) -> Self {
        Self { cursor, ..self }
    }
}

impl Default for ScanWindow {
    fn default() -> Self {
        Self::first()
    }
}

/// What one [`ScanWindow`] found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanResult<T> {
    /// The matches, in scan order.
    pub items: Vec<T>,
    /// Entries examined in this call.
    pub scanned: usize,
    /// Entries in the whole scan.
    pub total: usize,
    /// Where to continue, or `None` when the scan reached the end.
    pub next_cursor: Option<usize>,
}

impl<T> ScanResult<T> {
    /// A scan over nothing.
    pub(crate) fn empty() -> Self {
        Self {
            items: Vec::new(),
            scanned: 0,
            total: 0,
            next_cursor: None,
        }
    }

    /// The window that continues this scan, if it is not finished.
    pub fn next_window(&self, window: ScanWindow) -> Option<ScanWindow> {
        self.next_cursor.map(|c| window.from(c))
    }
}

/// A `java.lang.String` found by [`HeapQuery::search_strings`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StringMatch {
    /// The String object's id.
    pub object_id: u64,
    /// Length of the whole string in characters.
    pub length_chars: usize,
    /// The first [`StringMatch::PREVIEW_CHARS`] characters.
    pub preview: String,
}

impl StringMatch {
    /// How much of a matched string [`StringMatch::preview`] carries.
    pub const PREVIEW_CHARS: usize = 200;
}

impl<T> PageResult<T> {
    /// No items at all.
    pub(crate) fn empty() -> Self {
        Self {
            items: Vec::new(),
            total: 0,
            has_more: false,
        }
    }

    /// Take `page` out of an iterator whose total length is only known by
    /// exhausting it (a filtered list).  The items outside the page are
    /// counted, not kept.
    pub(crate) fn from_unsized(page: Page, iter: impl Iterator<Item = T>) -> Self {
        let mut total = 0;
        let mut items = Vec::new();
        for item in iter {
            if total >= page.offset && items.len() < page.limit {
                items.push(item);
            }
            total += 1;
        }
        let has_more = page.offset + items.len() < total;
        Self {
            items,
            total,
            has_more,
        }
    }

    /// Take `page` out of an iterator whose total length is known.
    fn from_iter(total: usize, page: Page, iter: impl Iterator<Item = T>) -> Self {
        let items: Vec<T> = iter.skip(page.offset).take(page.limit).collect();
        let has_more = page.offset + items.len() < total;
        Self {
            items,
            total,
            has_more,
        }
    }
}

impl HeapQuery {
    /// Open the heap dump at `hprof_path`, building any missing indexes first.
    ///
    /// Indexes live in `{stem}.indexes/` next to the dump and are reused on
    /// later runs.  Only the *cheap* indexes are built implicitly; the
    /// dominator tree and retained sizes (the one step whose memory grows
    /// with the heap) are used if they already exist but never started here —
    /// use [`Self::open_with`] with `IndexOptions { retained: true, .. }` or
    /// run `hprof-toolkit index` to build them.  Nothing is printed; see
    /// [`Self::open_with`] for progress.
    ///
    /// ```no_run
    /// use hprof_toolkit::HeapQuery;
    ///
    /// let heap = HeapQuery::open("heap.hprof")?;
    /// println!("{} objects", heap.object_count());
    /// # Ok::<(), hprof_toolkit::HprofError>(())
    /// ```
    pub fn open(hprof_path: impl AsRef<Path>) -> Result<Self, HprofError> {
        let opts = IndexOptions {
            retained: false,
            ..IndexOptions::default()
        };
        Self::open_with(hprof_path, &opts, &NoProgress)
    }

    /// [`Self::open`] with explicit [`IndexOptions`] and a [`Progress`] sink.
    pub fn open_with(
        hprof_path: impl AsRef<Path>,
        opts: &IndexOptions,
        progress: &dyn Progress,
    ) -> Result<Self, HprofError> {
        let hprof_path = hprof_path.as_ref();
        let dir = build_all_indexes_with(hprof_path, opts, progress)?;
        Self::open_existing_in(hprof_path, &dir)
    }

    /// Fail with [`HprofError::NotIndexed`] unless `store` holds a completed
    /// build of every mandatory index.
    pub(crate) fn check_indexed(store: &dyn IndexStore) -> Result<(), HprofError> {
        let manifest = Manifest::read(store)?.ok_or(HprofError::NotIndexed("hprof"))?;
        match names::mandatory()
            .into_iter()
            .find(|n| !manifest.is_built(n) || !store.exists(n))
        {
            Some(missing) => Err(HprofError::NotIndexed(missing)),
            None => Ok(()),
        }
    }

    /// Open the heap dump at `hprof_path` using indexes that must already
    /// exist; never builds anything.
    ///
    /// Fails with [`HprofError::NotIndexed`] when the index directory, its
    /// manifest or any mandatory index is missing.
    pub fn open_existing(hprof_path: impl AsRef<Path>) -> Result<Self, HprofError> {
        let hprof_path = hprof_path.as_ref();
        Self::open_existing_in(hprof_path, &default_index_dir(hprof_path))
    }

    /// [`Self::open_existing`] with the indexes in `dir` instead of next to
    /// the dump.
    pub fn open_existing_in(hprof_path: &Path, dir: &Path) -> Result<Self, HprofError> {
        if !dir.is_dir() {
            return Err(HprofError::NotIndexed("hprof"));
        }
        let store = FsStore::open_or_create(dir)?;
        Self::check_indexed(&store)?;
        Self::from_store(ByteSource::map_file(hprof_path)?, &store)
    }

    /// Open a [`HeapQuery`] over `hprof` and the indexes in `store`.
    ///
    /// This is the one real constructor; it works identically for a
    /// filesystem store (memory-mapped files) and an in-memory store.  All
    /// mandatory indexes must exist in the store; the dominator and retained
    /// indexes are optional and detected via [`IndexStore::exists`].
    pub fn from_store(hprof: ByteSource, store: &dyn IndexStore) -> Result<Self, HprofError> {
        if let Some(manifest) = Manifest::read(store)? {
            let identity = HprofIdentity::of(hprof.as_ref())?;
            if let Some(reason) = manifest.mismatch_reason(&identity) {
                return Err(HprofError::Corrupt(reason));
            }
        }
        let hprof = Arc::new(hprof);
        let utf8 = Arc::new(store.open(names::UTF8)?);
        let aux = AuxRecordIndex::open(
            Arc::clone(&hprof),
            store.open(names::FRAMES)?,
            store.open(names::TRACES)?,
            store.open(names::START_THREADS)?,
            store.open(names::END_THREADS)?,
            store.open(names::UNLOAD_CLASSES)?,
            Arc::clone(&utf8),
        )?;
        let open_nine = |entries: &[&str; 9]| -> Result<[ByteSource; 9], HprofError> {
            let v: Vec<ByteSource> = entries
                .iter()
                .map(|n| store.open(n))
                .collect::<Result<_, _>>()?;
            v.try_into()
                .map_err(|_| HprofError::Internal("expected 9 index entries".to_owned()))
        };
        let retained_set = if names::RETAINED_SET.iter().all(|n| store.exists(n)) {
            Some(RetainedSet {
                dominators: store.open(names::DOMINATORS)?,
                retained: store.open(names::RETAINED)?,
                by_size: store.open(names::RETAINED_BY_SIZE)?,
                children: store.open(names::DOMINATOR_CHILDREN)?,
            })
        } else {
            None
        };
        Self::create(
            hprof,
            store.open(names::OBJECT_STORE)?,
            utf8,
            store.open(names::LOAD_CLASS)?,
            aux,
            store.open(names::REFS)?,
            open_nine(&names::ROOTS)?,
            open_nine(&names::ARRAYS)?,
            store.open(names::INSTANCES_BY_CLASS)?,
            store.open(names::CLASS_HISTOGRAM)?,
            retained_set,
        )
    }

    /// Shared inner constructor: validates all readers from their [`ByteSource`]s
    /// and stores them.
    #[allow(clippy::too_many_arguments)]
    fn create(
        hprof_data: Arc<ByteSource>,
        combined_data: ByteSource,
        utf8_data: Arc<ByteSource>,
        lc_data: ByteSource,
        aux: AuxRecordIndex,
        refs_data: ByteSource,
        root_data: [ByteSource; 9],
        array_data: [ByteSource; 9],
        instances_by_class: ByteSource,
        class_histogram: ByteSource,
        retained_set: Option<RetainedSet>,
    ) -> Result<Self, HprofError> {
        let hprof_header = HprofIndex::from_ref(
            bytes(&hprof_data),
            combined_data.as_ref(),
            bytes(&utf8_data),
            lc_data.as_ref(),
        )?
        .hprof_header();
        RefIndex::from_ref(refs_data.as_ref())?;
        for s in &root_data {
            RootIndexReader::from_ref(s.as_ref())?;
        }
        for s in &array_data {
            ArraySizeReader::from_ref(s.as_ref())?;
        }
        RecordFile::<InstanceByClassEntry>::new(instances_by_class.as_ref())?;
        RecordFile::<HistogramRecord>::new(class_histogram.as_ref())?;
        if let Some(set) = &retained_set {
            DominatorIndex::from_ref(set.dominators.as_ref())?;
            RetainedIndex::from_ref(set.retained.as_ref())?;
            RecordFile::<RetainedBySizeEntry>::new(set.by_size.as_ref())?;
            RecordFile::<DomChildEntry>::new(set.children.as_ref())?;
        }
        let mut query = Self {
            hprof_data,
            hprof_header,
            combined_data,
            utf8_data,
            lc_data,
            aux,
            roots: root_data,
            refs: refs_data,
            array_sizes: array_data,
            instances_by_class,
            class_histogram,
            retained_set,
            class_names: ClassNames::default(),
        };
        query.class_names = query.load_class_names()?;
        Ok(query)
    }

    // ── Private reader helpers ────────────────────────────────────────────────

    fn hprof_index(&self) -> HprofIndex<'_> {
        HprofIndex::from_slice(
            bytes(&self.hprof_data),
            self.combined_data.as_ref(),
            bytes(&self.utf8_data),
            self.lc_data.as_ref(),
            &self.hprof_header,
        )
    }

    fn root_reader(&self, rt: GcRootType) -> RootIndexReader<'_> {
        RootIndexReader::from_slice(self.roots[rt.index()].as_ref())
    }

    pub(crate) fn ref_index(&self) -> RefIndex<'_> {
        RefIndex::from_slice(self.refs.as_ref())
    }

    fn array_size_reader(&self, kind: ArrayKind) -> ArraySizeReader<'_> {
        ArraySizeReader::from_slice(self.array_sizes[kind.index()].as_ref())
    }

    fn retained_index(&self) -> Option<RetainedIndex<'_>> {
        self.retained_set
            .as_ref()
            .map(|s| RetainedIndex::from_slice(s.retained.as_ref()))
    }

    fn dominator_index(&self) -> Option<DominatorIndex<'_>> {
        self.retained_set
            .as_ref()
            .map(|s| DominatorIndex::from_slice(s.dominators.as_ref()))
    }

    fn retained_by_size(&self) -> Option<RecordFile<'_, RetainedBySizeEntry>> {
        self.retained_set
            .as_ref()
            .map(|s| RecordFile::from_slice(s.by_size.as_ref()))
    }

    fn dominator_children(&self) -> Option<RecordFile<'_, DomChildEntry>> {
        self.retained_set
            .as_ref()
            .map(|s| RecordFile::from_slice(s.children.as_ref()))
    }

    // ── Basic accessors ───────────────────────────────────────────────────────

    /// The parsed hprof file header (version, id size, timestamp).
    pub fn hprof_header(&self) -> &HprofHeader {
        &self.hprof_header
    }

    /// Size of object identifiers in this dump, in bytes (4 or 8).
    pub fn id_size(&self) -> u32 {
        self.hprof_index().id_size()
    }

    /// Total number of sub-records in the combined index (classes, instances,
    /// arrays, and GC roots).
    pub fn object_count(&self) -> usize {
        self.hprof_index().object_count()
    }

    // ── Array size indexes ────────────────────────────────────────────────────

    /// Iterate arrays of `kind` in descending byte-size order (largest first).
    pub fn iter_arrays_by_size(&self, kind: ArrayKind) -> ArraySizeIter<'_> {
        self.array_size_reader(kind).iter()
    }

    /// Total number of arrays indexed for `kind`.
    pub fn array_count(&self, kind: ArrayKind) -> usize {
        self.array_size_reader(kind).len()
    }

    // ── Object lookup by ID ───────────────────────────────────────────────────

    /// The sub-record with object id `object_id`, whatever its kind.
    ///
    /// `None` when the id is not in the dump.  O(log n) binary search.
    pub fn object(&self, object_id: u64) -> Result<Option<SubRecord<'_>>, HprofError> {
        self.hprof_index().find_object(object_id)
    }

    /// The `INSTANCE_DUMP` with object id `object_id`.
    ///
    /// `None` when the id is absent or names something that is not an
    /// instance (a class, an array, a GC root).
    pub fn instance(&self, object_id: u64) -> Result<Option<InstanceDump<'_>>, HprofError> {
        Ok(match self.hprof_index().find_instance(object_id)? {
            Some(SubRecord::InstanceDump(inst)) => Some(inst),
            _ => None,
        })
    }

    /// The `CLASS_DUMP` of class object `class_id`.
    ///
    /// `None` when the id is absent or is not a class.
    pub fn class(&self, class_id: u64) -> Result<Option<ClassDump<'_>>, HprofError> {
        Ok(match self.hprof_index().find_class_dump(class_id)? {
            Some(SubRecord::ClassDump(cd)) => Some(cd),
            _ => None,
        })
    }

    // ── Sequential iteration ──────────────────────────────────────────────────

    /// Iterate over all sub-records in ascending object-ID order.
    ///
    /// Yields parsed [`SubRecord`] values on demand; use `match` to identify
    /// the type:
    ///
    /// ```no_run
    /// # use hprof_toolkit::prelude::*;
    /// # let heap = HeapQuery::open("heap.hprof")?;
    /// let mut instances = 0;
    /// for result in heap.objects() {
    ///     match result? {
    ///         SubRecord::InstanceDump(_) => instances += 1,
    ///         SubRecord::ClassDump(_) | SubRecord::ObjArrayDump(_) | SubRecord::PrimArrayDump(_) => {}
    ///         _ => {} // GC roots
    ///     }
    /// }
    /// # Ok::<(), HprofError>(())
    /// ```
    pub fn objects(&self) -> ObjectIter<'_> {
        ObjectIter {
            inner: RecordFile::<SubIndexEntry>::from_slice(self.combined_data.as_ref()).iter(),
            query: self,
        }
    }

    /// Iterate raw [`SubIndexEntry`] values from the combined index in ascending
    /// object-ID order.
    ///
    /// Unlike [`Self::objects`], no parsing is performed; callers receive
    /// the lightweight `(tag, object_id, position)` tuple and can choose to
    /// parse selectively with [`Self::parse_entry`].
    pub(crate) fn iter_entries(&self) -> SubIndexIter<'_> {
        RecordFile::<SubIndexEntry>::from_slice(self.combined_data.as_ref()).iter()
    }

    /// Parse the sub-record described by `entry` directly from the hprof mmap.
    ///
    /// Use this when you already hold a [`SubIndexEntry`] (e.g. from iterating
    /// via [`Self::iter_entries`]) and want to avoid a redundant binary search.
    pub(crate) fn parse_entry<'a>(
        &'a self,
        entry: &SubIndexEntry,
    ) -> Result<SubRecord<'a>, HprofError> {
        self.hprof_index().parse_entry(entry)
    }

    // ── Parallel iteration ────────────────────────────────────────────────────

    /// Every sub-record, as a rayon parallel iterator.
    ///
    /// Items are parsed on demand by the worker that receives them; nothing
    /// is collected.  Use the ordinary rayon adaptors:
    ///
    /// ```no_run
    /// # use hprof_toolkit::prelude::*;
    /// # let heap = HeapQuery::open("heap.hprof")?;
    /// let arrays = heap
    ///     .par_objects()
    ///     .filter(|r| matches!(r, Ok(SubRecord::PrimArrayDump(_))))
    ///     .count();
    /// # Ok::<(), HprofError>(())
    /// ```
    pub fn par_objects(&self) -> impl ParallelIterator<Item = Result<SubRecord<'_>, HprofError>> {
        (0..self.hprof_index().object_count())
            .into_par_iter()
            .filter_map(move |i| self.hprof_index().parse_at(i).transpose())
    }

    /// Every `INSTANCE_DUMP`, as a rayon parallel iterator.
    pub fn par_instances(
        &self,
    ) -> impl ParallelIterator<Item = Result<InstanceDump<'_>, HprofError>> {
        self.par_objects().filter_map(|r| match r {
            Ok(SubRecord::InstanceDump(inst)) => Some(Ok(inst)),
            Ok(_) => None,
            Err(e) => Some(Err(e)),
        })
    }

    /// Every `CLASS_DUMP`, as a rayon parallel iterator.
    pub fn par_classes(&self) -> impl ParallelIterator<Item = Result<ClassDump<'_>, HprofError>> {
        self.par_objects().filter_map(|r| match r {
            Ok(SubRecord::ClassDump(cd)) => Some(Ok(cd)),
            Ok(_) => None,
            Err(e) => Some(Err(e)),
        })
    }

    /// Every instance of the class object `class_id`, in ascending object-id
    /// order.  Reads only that class's range of the per-class index.
    ///
    /// Subclass instances are *not* included: an instance belongs to exactly
    /// one class.
    pub fn instances_of(
        &self,
        class_id: u64,
    ) -> impl Iterator<Item = Result<InstanceDump<'_>, HprofError>> {
        self.class_entries(ClassKey::Class(class_id))
            .filter_map(move |e| self.instance_from_entry(&e))
    }

    /// [`Self::instances_of`] as a rayon parallel iterator.
    pub fn par_instances_of(
        &self,
        class_id: u64,
    ) -> impl ParallelIterator<Item = Result<InstanceDump<'_>, HprofError>> {
        self.instances_by_class()
            .par_range(ClassKey::Class(class_id).to_u64())
            .filter_map(move |e| self.instance_from_entry(&e))
    }

    pub(crate) fn instance_from_entry(
        &self,
        e: &InstanceByClassEntry,
    ) -> Option<Result<InstanceDump<'_>, HprofError>> {
        match self
            .hprof_index()
            .parse_entry(&e.sub_index_entry(TAG_INSTANCE_DUMP))
        {
            Ok(SubRecord::InstanceDump(inst)) => Some(Ok(inst)),
            Ok(_) => None,
            Err(e) => Some(Err(e)),
        }
    }

    // ── Per-class index ───────────────────────────────────────────────────────

    fn instances_by_class(&self) -> RecordFile<'_, InstanceByClassEntry> {
        RecordFile::from_slice(self.instances_by_class.as_ref())
    }

    /// Every object grouped under `key`, in ascending object-id order.
    ///
    /// A binary-search range over `instances_by_class.bin`; nothing outside
    /// the class is touched.  Parse an entry with
    /// [`Self::parse_entry`] and [`InstanceByClassEntry::sub_index_entry`].
    pub(crate) fn class_entries(&self, key: ClassKey) -> RecordIter<'_, InstanceByClassEntry> {
        self.instances_by_class().range(key.to_u64())
    }

    /// Number of objects grouped under `key`.  O(log n).
    pub fn instance_count(&self, key: ClassKey) -> usize {
        self.class_entries(key).len()
    }

    /// The class histogram, largest instance count first.
    ///
    /// Read straight from `class_histogram.bin`: no scan, O(classes) total.
    pub(crate) fn histogram(&self) -> RecordIter<'_, HistogramRecord> {
        RecordFile::<HistogramRecord>::from_slice(self.class_histogram.as_ref()).iter()
    }

    /// Number of distinct class keys with at least one object.
    pub(crate) fn histogram_len(&self) -> usize {
        RecordFile::<HistogramRecord>::from_slice(self.class_histogram.as_ref()).len()
    }

    /// Every loaded class as `(class_id, name_id)` from the load-class index,
    /// ascending by class id.  Includes classes with no instances.
    pub fn class_ids(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        RecordFile::<LoadClassEntry>::from_slice(self.lc_data.as_ref())
            .iter()
            .map(|e| (e.class_id, e.class_name_id))
    }

    // ── Name resolution ───────────────────────────────────────────────────────

    /// Look up the UTF-8 string for `name_id`.
    pub fn lookup_name(&self, name_id: u64) -> Result<Option<String>, HprofError> {
        self.hprof_index().lookup_name(name_id)
    }

    /// The class object whose dot-notation name is exactly `name`
    /// (e.g. `"java.lang.String"`).
    ///
    /// A hash probe: the names are read once when the query is opened.  When
    /// several classes share a name (one class loaded by two class loaders)
    /// the lowest class id is returned.
    pub fn find_class_by_name(&self, name: &str) -> Option<u64> {
        self.class_names.by_name.get(name).copied()
    }

    /// The dot-notation class name for `class_id` (e.g. `"java.lang.String"`).
    ///
    /// `None` when `class_id` is not a loaded class.  A hash probe and a
    /// pointer clone.
    pub fn class_name(&self, class_id: u64) -> Option<Arc<str>> {
        self.class_names.by_id.get(&class_id).cloned()
    }

    /// [`Self::class_name`] for display: the name, or `0x…` hex of the id when
    /// the class is unknown.
    pub fn class_label(&self, class_id: u64) -> String {
        match self.class_name(class_id) {
            Some(name) => name.to_string(),
            None => format!("0x{class_id:x}"),
        }
    }

    /// Read every class name once: O(classes) memory.
    fn load_class_names(&self) -> Result<ClassNames, HprofError> {
        let mut by_id = HashMap::new();
        let mut by_name: HashMap<Arc<str>, u64> = HashMap::new();
        for (class_id, name_id) in self.class_ids() {
            let Some(raw) = self.lookup_name(name_id)? else {
                continue;
            };
            let name: Arc<str> = raw.replace('/', ".").into();
            by_name.entry(name.clone()).or_insert(class_id);
            by_id.insert(class_id, name);
        }
        Ok(ClassNames { by_id, by_name })
    }

    /// The runtime type name of the object at `object_id`, for display:
    /// `java.util.ArrayList`, `Class` for a class object, `int[]`,
    /// `java.lang.String[]`.
    ///
    /// `Object` when the id is null, not in the dump or has an unknown class;
    /// `?` when its record cannot be read.  Never fails, so it can label rows
    /// in a listing.
    pub fn object_type_name(&self, object_id: u64) -> String {
        const UNKNOWN: &str = "Object";
        if object_id == 0 {
            return UNKNOWN.to_owned();
        }
        match self.object(object_id) {
            Err(_) => "?".to_owned(),
            Ok(None) => UNKNOWN.to_owned(),
            Ok(Some(SubRecord::InstanceDump(inst))) => self
                .class_name(inst.class_id)
                .map_or_else(|| UNKNOWN.to_owned(), |n| n.to_string()),
            Ok(Some(SubRecord::ClassDump(_))) => "Class".to_owned(),
            Ok(Some(SubRecord::ObjArrayDump(arr))) => {
                self.key_name(ClassKey::ObjArray(arr.array_class_id))
            }
            Ok(Some(SubRecord::PrimArrayDump(arr))) => {
                self.key_name(ClassKey::PrimArray(arr.element_type))
            }
            Ok(Some(_)) => UNKNOWN.to_owned(),
        }
    }

    // ── Field resolution ──────────────────────────────────────────────────────

    /// Resolve the instance fields for an [`InstanceDump`], traversing the
    /// full class hierarchy.
    pub fn instance_fields(&self, instance: &InstanceDump<'_>) -> Result<Vec<Field>, HprofError> {
        self.hprof_index().instance_fields(instance)
    }

    /// The text of the String instance `inst`, when it is one already in hand
    /// (skips the lookup [`Self::string`] does).  Handles `char[]` and compact
    /// `byte[]` strings.
    pub(crate) fn string_of(&self, inst: &InstanceDump<'_>) -> Result<String, HprofError> {
        Ok(match self.hprof_index().resolve_string(inst)? {
            Value::String(_, s) => s,
            _ => String::new(),
        })
    }

    /// Attempt to resolve `object_id` as a primitive Java wrapper value.
    ///
    /// Handles `String`, `Integer`, `Long`, `Double`, `Float`, `Short`,
    /// `Byte`, `Boolean`, and `Character`.  Anything else returns
    /// [`Value::Object`].
    pub fn resolve_value(&self, object_id: u64) -> Result<Value, HprofError> {
        self.hprof_index().resolve_value(object_id)
    }

    // ── Auxiliary record lookup ───────────────────────────────────────────────

    /// Find a `HPROF_TRACE` record by `trace_serial`.
    pub(crate) fn find_trace(&self, trace_serial: u32) -> Result<Option<Trace>, HprofError> {
        self.aux.find_trace(trace_serial)
    }

    /// Resolve all name IDs in `frame` to strings.
    pub(crate) fn resolve_frame(&self, frame: &Frame) -> Result<ResolvedFrame, HprofError> {
        self.aux.resolve_frame(frame)
    }

    /// Resolve all name IDs in `thread` to strings.
    pub(crate) fn resolve_thread(
        &self,
        thread: &StartThread,
    ) -> Result<ResolvedThread, HprofError> {
        self.aux.resolve_thread(thread)
    }

    /// Parse every frame in `trace` and return them in order.
    pub(crate) fn trace_frames(&self, trace: &Trace) -> Result<Vec<Frame>, HprofError> {
        self.aux.trace_frames(trace)
    }

    /// Returns `true` if a `HPROF_END_THREAD` record exists for `thread_serial`.
    pub(crate) fn was_thread_ended(&self, thread_serial: u32) -> bool {
        self.aux.was_thread_ended(thread_serial)
    }

    /// Returns `true` if a `HPROF_UNLOAD_CLASS` record exists for `class_serial`.
    pub fn was_class_unloaded(&self, class_serial: u32) -> bool {
        self.aux.was_class_unloaded(class_serial)
    }

    // ── Auxiliary record iteration ────────────────────────────────────────────

    /// Iterate all `HPROF_FRAME` records in ascending `frame_id` order.
    pub(crate) fn iter_frames(&self) -> FrameIter<'_> {
        self.aux.iter_frames()
    }

    /// Iterate all `HPROF_TRACE` records in ascending `trace_serial` order.
    pub(crate) fn iter_traces(&self) -> TraceIter<'_> {
        self.aux.iter_traces()
    }

    /// Iterate all `HPROF_START_THREAD` records in ascending `thread_serial` order.
    pub(crate) fn iter_threads(&self) -> StartThreadIter<'_> {
        self.aux.iter_start_threads()
    }

    // ── GC root access ────────────────────────────────────────────────────────

    /// Iterate all root entries for the given `root_type` in ascending
    /// `object_id` order.
    pub(crate) fn iter_roots(&self, root_type: GcRootType) -> RootIter<'_> {
        self.root_reader(root_type).iter()
    }

    /// Returns `true` if `object_id` appears in any of the nine GC root indexes.
    pub fn is_gc_root(&self, object_id: u64) -> bool {
        GcRootType::ALL
            .iter()
            .any(|&rt| self.root_reader(rt).find(object_id).is_some())
    }

    /// Return all root types for which `object_id` has a root entry.
    pub fn root_types_of(&self, object_id: u64) -> Vec<GcRootType> {
        GcRootType::ALL
            .iter()
            .filter(|&&rt| self.root_reader(rt).find(object_id).is_some())
            .copied()
            .collect()
    }

    // ── Reference index ───────────────────────────────────────────────────────

    /// Total number of reference records in the reference index.
    pub fn ref_count(&self) -> usize {
        self.ref_index().len()
    }

    // ── Retained heap / dominator tree ────────────────────────────────────────

    /// Returns `true` if the dominator tree and retained heap size indexes are
    /// available (i.e. were built by the indexing pipeline).
    pub fn has_retained_heap(&self) -> bool {
        self.retained_set.is_some()
    }

    /// The objects with the largest retained heap sizes, largest first.
    ///
    /// Each item is `(object_id, retained_bytes)`.  Served from the
    /// pre-sorted `retained_by_size.bin`, so a page costs O(page), not
    /// O(objects).  Returns `None` when the retained heap index was not built.
    pub fn retained_top(&self, page: Page) -> Option<PageResult<(u64, u64)>> {
        let file = self.retained_by_size()?;
        Some(PageResult::from_iter(
            file.len(),
            page,
            file.iter().map(|e| (e.object_id, e.retained_bytes)),
        ))
    }

    /// The objects immediately dominated by `dominator_id`, with their
    /// retained sizes, in ascending object-id order.
    ///
    /// Pass [`crate::VIRTUAL_ROOT_ID`] (`0`) for the objects that
    /// are dominated only by the virtual root (the GC roots).  Served from
    /// `dominator_children.bin` by binary search; a page costs
    /// O(log n + page).  Returns `None` when the retained heap index was not
    /// built.
    pub fn dominated_by(&self, dominator_id: u64, page: Page) -> Option<PageResult<(u64, u64)>> {
        let children = self.dominator_children()?;
        let retained = self.retained_index()?;
        let range = children.range(dominator_id);
        Some(PageResult::from_iter(
            range.len(),
            page,
            range.map(|e| (e.object_id, retained.find(e.object_id).unwrap_or(0))),
        ))
    }

    /// Return the retained heap size in bytes for `object_id`.
    ///
    /// The retained size is the total memory that would be freed if this object
    /// were garbage-collected — i.e. the shallow size of this object plus the
    /// retained sizes of all objects it exclusively dominates.
    ///
    /// Returns `None` when:
    /// * The retained heap index was not built (run the indexing pipeline).
    /// * `object_id` is not a live (reachable) heap object.
    pub fn retained_size(&self, object_id: u64) -> Option<u64> {
        self.retained_index()?.find(object_id)
    }

    /// Return the `object_id` of the immediate dominator of `object_id`.
    ///
    /// Object A dominates object B if every path from any GC root to B passes
    /// through A.  The immediate dominator is the closest such A.
    ///
    /// Returns [`crate::dominator::VIRTUAL_ROOT_ID`] (`0`) when `object_id` is
    /// itself a GC root (dominated only by the synthetic virtual root).
    ///
    /// Returns `None` when:
    /// * The dominator index was not built.
    /// * `object_id` is not a live (reachable) heap object.
    pub fn dominator_of(&self, object_id: u64) -> Option<u64> {
        self.dominator_index()?.find(object_id)
    }

    // ── Object resolution ─────────────────────────────────────────────────────

    /// Fully resolve an [`InstanceDump`] into a [`crate::resolved::ResolvedInstance`].
    ///
    /// All fields (including inherited ones) are resolved; object-typed fields
    /// that point to known wrapper types (String, Integer, Long, …) are
    /// unwrapped into rich [`crate::resolved::Value`] variants.
    pub fn resolve_instance(
        &self,
        inst: &InstanceDump<'_>,
    ) -> Result<crate::resolved::ResolvedInstance, HprofError> {
        crate::resolved::ResolvedInstance::from_dump(self, inst)
    }

    /// Fully resolve a [`ClassDump`] into a [`crate::resolved::ResolvedClass`].
    ///
    /// Resolves class name, super-class name, static field values (with wrapper
    /// type unwrapping), and instance field descriptors.
    pub fn resolve_class(
        &self,
        cd: &ClassDump<'_>,
    ) -> Result<crate::resolved::ResolvedClass, HprofError> {
        crate::resolved::ResolvedClass::from_dump(self, cd)
    }
}

// ── ObjectIter ────────────────────────────────────────────────────────────────

/// Iterator over all sub-records in a [`HeapQuery`].
///
/// Yields [`SubRecord`] values in ascending object-ID order.  Records are
/// parsed on demand from the memory-mapped hprof file; no data is buffered.
pub struct ObjectIter<'a> {
    query: &'a HeapQuery,
    inner: SubIndexIter<'a>,
}

impl<'a> Iterator for ObjectIter<'a> {
    type Item = Result<SubRecord<'a>, HprofError>;

    fn next(&mut self) -> Option<Self::Item> {
        let entry = self.inner.next()?;
        Some(self.query.hprof_index().parse_entry(&entry))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aux_query::LineNumber;
    use crate::heap_parser::FieldValue;
    use crate::index::MemStore;
    use crate::resolved::Value;
    use crate::test_util::{build_in_memory, build_in_memory_with, standard_heap, std_ids::*};

    fn query() -> HeapQuery {
        build_in_memory(&standard_heap())
    }

    // ── Open / completeness checks ────────────────────────────────────────────

    #[test]
    fn a_complete_store_passes_the_indexed_check() {
        let (store, _) = build_in_memory_with(&standard_heap(), &IndexOptions::default());
        HeapQuery::check_indexed(&store).unwrap();
    }

    #[test]
    fn an_empty_store_is_not_indexed() {
        let err = HeapQuery::check_indexed(&MemStore::new()).unwrap_err();
        assert!(matches!(err, HprofError::NotIndexed(_)), "{err}");
    }

    #[test]
    fn a_store_missing_one_mandatory_entry_names_it() {
        let (store, _) = build_in_memory_with(&standard_heap(), &IndexOptions::default());
        store.remove(names::REFS).unwrap();
        match HeapQuery::check_indexed(&store) {
            Err(HprofError::NotIndexed(name)) => assert_eq!(name, names::REFS),
            other => panic!("expected NotIndexed(refs), got {other:?}"),
        }
    }

    #[test]
    fn the_retained_family_is_not_mandatory() {
        let opts = IndexOptions {
            retained: false,
            ..IndexOptions::default()
        };
        let (store, _) = build_in_memory_with(&standard_heap(), &opts);
        HeapQuery::check_indexed(&store).unwrap();
    }

    // ── Heap query tests ──────────────────────────────────────────────────────

    #[test]
    fn find_by_object_id() {
        let query = query();
        let record = query.object(INTEGER_42).unwrap().unwrap();
        assert!(matches!(record, SubRecord::InstanceDump(_)));
    }

    #[test]
    fn find_missing_returns_none() {
        let query = query();
        assert!(query.object(0xDEAD).unwrap().is_none());
    }

    #[test]
    fn class_returns_the_class_dump_only_for_classes() {
        let query = query();
        let cd = query.class(INTEGER_CLASS).unwrap().unwrap();
        assert_eq!(cd.class_id, INTEGER_CLASS);
        // An instance id, an unknown id: not classes.
        assert!(query.class(INTEGER_42).unwrap().is_none());
        assert!(query.class(0xDEAD).unwrap().is_none());
    }

    #[test]
    fn instance_returns_the_instance_dump_only_for_instances() {
        let query = query();
        let inst = query.instance(INTEGER_42).unwrap().unwrap();
        assert_eq!(inst.object_id, INTEGER_42);
        assert_eq!(inst.class_id, INTEGER_CLASS);
        // A class id, an array id, an unknown id: not instances.
        assert!(query.instance(INTEGER_CLASS).unwrap().is_none());
        assert!(query.instance(CHARS_HI).unwrap().is_none());
        assert!(query.instance(0xDEAD).unwrap().is_none());
    }

    #[test]
    fn class_name_resolved() {
        let query = query();
        assert_eq!(
            query.class_name(INTEGER_CLASS),
            Some("java.lang.Integer".into())
        );
    }

    #[test]
    fn instance_fields_resolved() {
        let query = query();
        let inst = query.instance(INTEGER_42).unwrap().unwrap();
        let fields = query.instance_fields(&inst).unwrap();
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].name, "value");
        assert_eq!(fields[0].value, FieldValue::Int(42));
    }

    #[test]
    fn resolve_value_integer_wrapper() {
        let query = query();
        let val = query.resolve_value(INTEGER_42).unwrap();
        assert!(matches!(val, Value::BoxedInt(INTEGER_42, 42)));
    }

    #[test]
    fn resolve_value_string() {
        let query = query();
        let val = query.resolve_value(STRING_HI).unwrap();
        assert!(matches!(val, Value::String(STRING_HI, ref s) if s == "hi"));
    }

    #[test]
    fn object_count_matches_fixture() {
        let query = query();
        // 5 classes + 4 objects + 3 roots
        assert_eq!(query.object_count(), 12);
    }

    #[test]
    fn iter_objects_yields_records_in_id_order() {
        let query = query();
        let records: Vec<_> = query.objects().collect::<Result<_, _>>().unwrap();
        assert_eq!(records.len(), query.object_count());
        let ids: Vec<u64> = query.iter_entries().map(|e| e.object_id).collect();
        assert!(ids.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn par_objects_visits_every_record_once() {
        let query = query();
        let n = query
            .par_objects()
            .map(|r| r.map(|_| 1u64))
            .sum::<Result<u64, _>>()
            .unwrap();
        assert_eq!(n, query.object_count() as u64);
    }

    #[test]
    fn par_instances_yields_only_instances() {
        let query = query();
        let mut ids: Vec<u64> = query
            .par_instances()
            .map(|r| r.map(|i| i.object_id))
            .collect::<Result<_, _>>()
            .unwrap();
        ids.sort_unstable();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&INTEGER_42) && ids.contains(&LIST));
    }

    #[test]
    fn par_classes_yields_only_classes() {
        let query = query();
        let n = query
            .par_classes()
            .map(|r| r.map(|_| 1u64))
            .sum::<Result<u64, _>>()
            .unwrap();
        assert_eq!(n, 5);
    }

    #[test]
    fn resolving_instances_from_the_iterator() {
        let query = query();
        let mut names: Vec<String> = query
            .par_instances()
            .map(|r| Ok(query.resolve_instance(&r?)?.class_name))
            .collect::<Result<_, HprofError>>()
            .unwrap();
        names.sort();
        assert_eq!(
            names,
            vec![
                "java.lang.Integer",
                "java.lang.String",
                "java.util.ArrayList"
            ]
        );
    }

    #[test]
    fn resolved_instance_fields_come_through_the_iterator() {
        let query = query();
        let mut values: Vec<Value> = Vec::new();
        for inst in query.objects().filter_map(|r| match r {
            Ok(SubRecord::InstanceDump(i)) => Some(i),
            _ => None,
        }) {
            values.extend(
                query
                    .resolve_instance(&inst)
                    .unwrap()
                    .fields
                    .into_iter()
                    .map(|f| f.value),
            );
        }
        assert_eq!(values.len(), 3);
        assert!(values.contains(&Value::Int(42)));
        assert!(values.contains(&Value::Object(CHARS_HI)));
        assert!(values.contains(&Value::BoxedInt(INTEGER_42, 42)));
    }

    #[test]
    fn resolving_classes_from_the_iterator() {
        let query = query();
        let mut names: Vec<String> = query
            .par_classes()
            .map(|r| Ok(query.resolve_class(&r?)?.class_name))
            .collect::<Result<_, HprofError>>()
            .unwrap();
        names.sort();
        assert_eq!(
            names,
            vec![
                "[C",
                "java.lang.Integer",
                "java.lang.Object",
                "java.lang.String",
                "java.util.ArrayList"
            ]
        );
        let integer = query
            .resolve_class(&query.class(INTEGER_CLASS).unwrap().unwrap())
            .unwrap();
        assert_eq!(
            integer.super_class_name.as_deref(),
            Some("java.lang.Object")
        );
    }

    // ── find_class_by_name / instances_of tests ───────────────────────────────

    #[test]
    fn find_class_by_name_returns_class_id() {
        let query = query();
        let class_id = query.find_class_by_name("java.lang.Integer");
        assert_eq!(class_id, Some(INTEGER_CLASS));
    }

    #[test]
    fn class_names_use_dots_and_round_trip() {
        let query = query();
        assert_eq!(query.find_class_by_name("java/lang/Object"), None);
        assert_eq!(
            query.find_class_by_name("java.lang.Object"),
            Some(OBJECT_CLASS)
        );
        assert_eq!(
            query.class_name(OBJECT_CLASS).as_deref(),
            Some("java.lang.Object")
        );
        assert_eq!(query.class_name(0xDEAD), None);
        assert_eq!(query.class_label(0xDEAD), "0xdead");
        assert_eq!(query.class_label(INTEGER_CLASS), "java.lang.Integer");
        for (id, _) in query.class_ids() {
            let name = query.class_name(id).unwrap();
            assert_eq!(query.find_class_by_name(&name), Some(id));
        }
    }

    #[test]
    fn object_type_names_cover_every_kind_and_never_fail() {
        let query = query();
        assert_eq!(query.object_type_name(LIST), "java.util.ArrayList");
        assert_eq!(query.object_type_name(INTEGER_CLASS), "Class");
        assert_eq!(query.object_type_name(CHARS_HI), "char[]");
        assert_eq!(query.object_type_name(0), "Object");
        assert_eq!(query.object_type_name(0xDEAD), "Object");
    }

    #[test]
    fn find_class_by_name_unknown_returns_none() {
        let query = query();
        let class_id = query.find_class_by_name("does.not.Exist");
        assert_eq!(class_id, None);
    }

    #[test]
    fn instances_of_yields_the_classes_own_instances() {
        let query = query();
        let ints: Vec<u64> = query
            .instances_of(INTEGER_CLASS)
            .map(|r| r.unwrap().object_id)
            .collect();
        assert_eq!(ints, vec![INTEGER_42]);
        let par: Vec<u64> = query
            .par_instances_of(INTEGER_CLASS)
            .map(|r| r.map(|i| i.object_id))
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(par, vec![INTEGER_42]);
    }

    #[test]
    fn instances_of_a_class_without_instances_or_an_unknown_class_is_empty() {
        let query = query();
        assert_eq!(query.instances_of(OBJECT_CLASS).count(), 0);
        assert_eq!(query.instances_of(0xDEAD).count(), 0);
        assert_eq!(query.par_instances_of(0xDEAD).count(), 0);
    }

    #[test]
    fn instances_of_resolves_field_values() {
        let query = query();
        let inst = query.instances_of(INTEGER_CLASS).next().unwrap().unwrap();
        let fields = query.resolve_instance(&inst).unwrap().fields;
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].value, Value::Int(42));
    }

    // ── Aux record tests ──────────────────────────────────────────────────────

    #[test]
    fn find_frame_by_id() {
        let query = query();
        let frame = query.iter_frames().next().unwrap().unwrap();
        assert_eq!(frame.frame_id, FRAME_MAIN);
        assert_eq!(frame.class_serial, 1);

        let resolved = query.resolve_frame(&frame).unwrap();
        assert_eq!(resolved.method_name, "main");
        assert_eq!(resolved.method_signature, "()V");
        assert_eq!(resolved.source_file, "MyClass.java");
        assert_eq!(resolved.line_number, LineNumber::Line(7));
    }

    #[test]
    fn find_trace_by_serial() {
        let query = query();
        let trace = query.find_trace(TRACE_SERIAL).unwrap().unwrap();
        assert_eq!(trace.trace_serial, TRACE_SERIAL);
        assert_eq!(trace.thread_serial, THREAD_SERIAL);

        let frames = query.trace_frames(&trace).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].frame_id, FRAME_MAIN);
    }

    #[test]
    fn find_thread_by_serial() {
        let query = query();
        let thread = query.iter_threads().next().unwrap().unwrap();
        assert_eq!(thread.thread_serial, THREAD_SERIAL);
        assert_eq!(thread.thread_id, THREAD_OBJ);

        let resolved = query.resolve_thread(&thread).unwrap();
        assert_eq!(resolved.thread_name, "main");
    }

    #[test]
    fn was_thread_ended_true() {
        let query = query();
        assert!(query.was_thread_ended(THREAD_SERIAL));
        assert!(!query.was_thread_ended(99));
    }

    #[test]
    fn iter_frames_yields_all() {
        let query = query();
        let frames: Vec<_> = query.iter_frames().collect::<Result<_, _>>().unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].frame_id, FRAME_MAIN);
    }

    #[test]
    fn iter_traces_yields_all() {
        let query = query();
        let traces: Vec<_> = query.iter_traces().collect::<Result<_, _>>().unwrap();
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].trace_serial, TRACE_SERIAL);
    }

    #[test]
    fn iter_threads_yields_all() {
        let query = query();
        let threads: Vec<_> = query.iter_threads().collect::<Result<_, _>>().unwrap();
        assert_eq!(threads.len(), 1);
        assert_eq!(threads[0].thread_id, THREAD_OBJ);
    }

    // ── GC root tests ─────────────────────────────────────────────────────────

    #[test]
    fn is_gc_root_false_for_plain_instance() {
        let query = query();
        assert!(!query.is_gc_root(INTEGER_42));
    }

    #[test]
    fn is_gc_root_true_for_frame_root_and_sticky_class() {
        let query = query();
        assert!(query.is_gc_root(LIST));
        assert!(query.is_gc_root(INTEGER_CLASS));
        assert_eq!(query.root_types_of(LIST), vec![GcRootType::JavaFrame]);
    }

    #[test]
    fn root_kinds_are_reported_per_object() {
        let query = query();
        assert!(
            !query
                .root_types_of(INTEGER_42)
                .contains(&GcRootType::StickyClass)
        );
        assert!(
            query
                .root_types_of(INTEGER_CLASS)
                .contains(&GcRootType::StickyClass)
        );
    }

    #[test]
    fn iter_roots_counts_per_type() {
        let query = query();
        assert_eq!(query.iter_roots(GcRootType::JniGlobal).count(), 0);
        assert_eq!(query.iter_roots(GcRootType::StickyClass).count(), 2);
        assert_eq!(query.iter_roots(GcRootType::JavaFrame).count(), 1);
    }

    #[test]
    fn root_types_of_empty_for_non_root() {
        let query = query();
        assert!(query.root_types_of(INTEGER_42).is_empty());
    }

    // ── Reference index tests ─────────────────────────────────────────────────

    #[test]
    fn ref_count_counts_reference_records() {
        assert_eq!(query().ref_count(), 2);
    }

    // ── Retained heap tests ───────────────────────────────────────────────────

    #[test]
    fn retained_heap_available_after_full_build() {
        let query = query();
        assert!(query.has_retained_heap());
        // ArrayList instance: 8 bytes of field data + Integer (4 bytes).
        assert_eq!(query.retained_size(LIST), Some(12));
        assert_eq!(query.retained_size(INTEGER_42), Some(4));
        assert_eq!(query.dominator_of(INTEGER_42), Some(LIST));
        assert_eq!(
            query.dominator_of(LIST),
            Some(crate::dominator::VIRTUAL_ROOT_ID)
        );
        // Unreachable objects have no retained entry.
        assert_eq!(query.retained_size(STRING_HI), None);
        assert!(query.retained_top(Page::first(1)).unwrap().total > 0);
    }

    #[test]
    fn instances_of_and_histogram_come_from_the_class_index() {
        let query = query();
        let ints: Vec<u64> = query
            .class_entries(ClassKey::Class(INTEGER_CLASS))
            .map(|e| e.object_id)
            .collect();
        assert_eq!(ints, vec![INTEGER_42]);
        assert_eq!(query.instance_count(ClassKey::Class(STRING_CLASS)), 1);
        assert_eq!(query.instance_count(ClassKey::Class(OBJECT_CLASS)), 0);
        assert_eq!(query.instance_count(ClassKey::PrimArray(5)), 1);
        assert_eq!(query.histogram_len(), 4);
        assert!(query.histogram().all(|r| r.instance_count == 1));
        let classes: Vec<u64> = query.class_ids().map(|(id, _)| id).collect();
        assert_eq!(
            classes,
            vec![
                INTEGER_CLASS,
                OBJECT_CLASS,
                STRING_CLASS,
                CHAR_ARRAY_CLASS,
                ARRAYLIST_CLASS
            ]
        );

        // Parsing through the entry gives the real record.
        let e = query
            .class_entries(ClassKey::Class(ARRAYLIST_CLASS))
            .next()
            .unwrap();
        let rec = query
            .parse_entry(&e.sub_index_entry(TAG_INSTANCE_DUMP))
            .unwrap();
        assert!(matches!(rec, SubRecord::InstanceDump(ref i) if i.object_id == LIST));
    }

    #[test]
    fn retained_top_and_dominated_by_are_paged() {
        let query = query();
        // Reachable: LIST (12), INTEGER_42 (4), the two sticky classes (0).
        let top = query.retained_top(Page::first(2)).unwrap();
        assert_eq!(top.total, 4);
        assert!(top.has_more);
        assert_eq!(top.items, vec![(LIST, 12), (INTEGER_42, 4)]);
        let rest = query.retained_top(Page::new(2, 10)).unwrap();
        assert_eq!(rest.items.len(), 2);
        assert!(!rest.has_more);
        assert!(rest.items.iter().all(|&(_, r)| r == 0));

        // Children of the virtual root are the GC roots, in id order.
        let roots = query
            .dominated_by(crate::dominator::VIRTUAL_ROOT_ID, Page::first(10))
            .unwrap();
        assert_eq!(
            roots.items,
            vec![(INTEGER_CLASS, 0), (OBJECT_CLASS, 0), (LIST, 12)]
        );
        assert!(!roots.has_more);
        let under_list = query.dominated_by(LIST, Page::first(10)).unwrap();
        assert_eq!(under_list.items, vec![(INTEGER_42, 4)]);
        let leaf = query.dominated_by(INTEGER_42, Page::first(10)).unwrap();
        assert!(leaf.items.is_empty());
        assert_eq!(leaf.total, 0);
        // Paging past the end is empty, not an error.
        let past = query.dominated_by(LIST, Page::new(5, 10)).unwrap();
        assert!(past.items.is_empty());
        assert!(!past.has_more);
    }

    #[test]
    fn retained_heap_absent_when_not_built() {
        let opts = IndexOptions {
            retained: false,
            force: false,
            ..IndexOptions::default()
        };
        let (_, query) = build_in_memory_with(&standard_heap(), &opts);
        assert!(!query.has_retained_heap());
        assert_eq!(query.retained_size(LIST), None);
        assert_eq!(query.dominator_of(LIST), None);
        assert!(query.retained_top(Page::first(1)).is_none());
        assert!(query.dominated_by(LIST, Page::first(1)).is_none());
        // Everything else still works.
        assert_eq!(query.refs_to(INTEGER_42, Page::first(5)).items, vec![LIST]);
    }
}
