//! Unified query API for heap dump analysis.
//!
//! [`HeapQuery`] is the single entry point for ad-hoc heap dump analysis.
//! It wraps the heap-object API ([`crate::heap_query::HprofIndex`]) and the
//! auxiliary record API ([`crate::aux_query::AuxRecordIndex`]) and opens both
//! from a single [`crate::pipeline::IndexPaths`].
//!
//! All data is accessed via memory-mapped files; no dump data is loaded into
//! process memory.  Object lookup is O(log n) via binary search; parallel
//! iteration uses rayon.
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use hprof_toolkit::query::HeapQuery;
//! use hprof_toolkit::pipeline::IndexPaths;
//! use hprof_toolkit::heap_parser::SubRecord;
//! use std::path::Path;
//!
//! let hprof = Path::new("heap.dump");
//! let paths = IndexPaths::for_hprof(hprof);
//! let query  = HeapQuery::open(hprof, &paths)?;
//!
//! // Look up a single object by ID
//! if let Some(record) = query.find(0xDEAD_BEEF)? {
//!     match record {
//!         SubRecord::InstanceDump(inst) => {
//!             let name = query.class_name(inst.class_id)?.unwrap_or_default();
//!             let fields = query.instance_fields(&inst)?;
//!             println!("{name}: {fields:?}");
//!         }
//!         SubRecord::ClassDump(cd) => {
//!             println!("class: {}", query.class_name(cd.class_id)?.unwrap_or_default());
//!         }
//!         _ => {}
//!     }
//! }
//!
//! // Iterate all instances in parallel
//! query.par_instances(|inst| {
//!     let _ = query.class_name(inst.class_id)?;
//!     Ok(())
//! })?;
//! ```

use crate::array_index::{ArrayKind, ArraySizeIter, ArraySizeReader};
use crate::aux_query::{
    AuxRecordIndex, Frame, FrameIter, ResolvedFrame, ResolvedThread, StartThread, StartThreadIter,
    Trace, TraceIter,
};
use crate::dominator::{DominatorIndex, RetainedIndex};
use crate::heap_index::sub_record::SubIndexEntry;
use crate::heap_parser::{ClassDump, InstanceDump, SubIndexIter, SubRecord};
use crate::heap_query::{HprofIndex, JavaValue, ResolvedField};
use crate::hprof::{HprofError, HprofHeader};
use crate::pipeline::IndexPaths;
use crate::ref_index::RefIndex;
use crate::root_index::{GcRootType, RootIndexEntry, RootIndexReader, RootIter};
use crate::vfs::{ByteSource, MMapReader};
use std::path::Path;

// ── RootPathResult ────────────────────────────────────────────────────────────

/// Maximum number of BFS nodes visited before [`HeapQuery::path_to_root`] gives up.
pub const ROOT_PATH_SEARCH_LIMIT: usize = 10_000;

/// Result of [`HeapQuery::path_to_root`].
#[derive(Debug, Clone)]
pub enum RootPathResult {
    /// A path was found.  `path[0]` is a GC root; `path.last()` is the target.
    Found(Vec<u64>),
    /// The BFS hit [`ROOT_PATH_SEARCH_LIMIT`] visited nodes before finding a root.
    LimitReached,
    /// The reference graph was exhausted without reaching any GC root.
    NotReachable,
}

// ── HeapQuery ─────────────────────────────────────────────────────────────────

/// Unified query API over a heap dump and all its index files.
///
/// Wraps both the heap-object API ([`HprofIndex`]) and the auxiliary record
/// API ([`AuxRecordIndex`]) and exposes a single, intuitive interface.
///
/// All data is accessed via memory-mapped files; no dump data is loaded into
/// process memory.
pub struct HeapQuery {
    hprof_data: ByteSource,
    hprof_header: HprofHeader,
    combined_data: ByteSource,
    utf8_data: ByteSource,
    lc_data: ByteSource,
    aux: AuxRecordIndex,
    /// Per-type GC root data, indexed by [`GcRootType::index()`].
    roots: [ByteSource; 9],
    /// Back-reference index data.
    refs: ByteSource,
    /// Per-kind array size index data, indexed by [`ArrayKind::index()`].
    array_sizes: [ByteSource; 9],
    /// Dominator tree index data (optional — present only when built).
    dominator: Option<ByteSource>,
    /// Retained heap size index data (optional — present only when built).
    retained: Option<ByteSource>,
}

impl HeapQuery {
    /// Open the heap dump and all index files described by `paths`.
    ///
    /// All index files must already exist (run [`crate::pipeline::build_all_indexes`]
    /// first).
    pub fn open(hprof_path: &Path, paths: &IndexPaths) -> Result<Self, HprofError> {
        let hprof_pb = hprof_path.to_path_buf();
        let hprof_data = hprof_pb.open_mmap()?;
        let combined_data = paths.object_store.open_mmap()?;
        let utf8_data = paths.utf8.open_mmap()?;
        let lc_data = paths.load_class.open_mmap()?;
        let hprof_header = HprofIndex::from_ref(
            hprof_data.as_ref(),
            combined_data.as_ref(),
            utf8_data.as_ref(),
            lc_data.as_ref(),
        )?
        .hprof_header();

        let roots = [
            {
                let s = paths.root_unknown.open_mmap()?;
                RootIndexReader::from_ref(s.as_ref())?;
                s
            },
            {
                let s = paths.root_jni_global.open_mmap()?;
                RootIndexReader::from_ref(s.as_ref())?;
                s
            },
            {
                let s = paths.root_jni_local.open_mmap()?;
                RootIndexReader::from_ref(s.as_ref())?;
                s
            },
            {
                let s = paths.root_java_frame.open_mmap()?;
                RootIndexReader::from_ref(s.as_ref())?;
                s
            },
            {
                let s = paths.root_native_stack.open_mmap()?;
                RootIndexReader::from_ref(s.as_ref())?;
                s
            },
            {
                let s = paths.root_sticky_class.open_mmap()?;
                RootIndexReader::from_ref(s.as_ref())?;
                s
            },
            {
                let s = paths.root_thread_block.open_mmap()?;
                RootIndexReader::from_ref(s.as_ref())?;
                s
            },
            {
                let s = paths.root_monitor_used.open_mmap()?;
                RootIndexReader::from_ref(s.as_ref())?;
                s
            },
            {
                let s = paths.root_thread_obj.open_mmap()?;
                RootIndexReader::from_ref(s.as_ref())?;
                s
            },
        ];
        let refs = {
            let s = paths.refs.open_mmap()?;
            RefIndex::from_ref(s.as_ref())?;
            s
        };
        let array_sizes = [
            {
                let s = paths.array_size(ArrayKind::Boolean).open_mmap()?;
                ArraySizeReader::from_ref(s.as_ref())?;
                s
            },
            {
                let s = paths.array_size(ArrayKind::Char).open_mmap()?;
                ArraySizeReader::from_ref(s.as_ref())?;
                s
            },
            {
                let s = paths.array_size(ArrayKind::Float).open_mmap()?;
                ArraySizeReader::from_ref(s.as_ref())?;
                s
            },
            {
                let s = paths.array_size(ArrayKind::Double).open_mmap()?;
                ArraySizeReader::from_ref(s.as_ref())?;
                s
            },
            {
                let s = paths.array_size(ArrayKind::Byte).open_mmap()?;
                ArraySizeReader::from_ref(s.as_ref())?;
                s
            },
            {
                let s = paths.array_size(ArrayKind::Short).open_mmap()?;
                ArraySizeReader::from_ref(s.as_ref())?;
                s
            },
            {
                let s = paths.array_size(ArrayKind::Int).open_mmap()?;
                ArraySizeReader::from_ref(s.as_ref())?;
                s
            },
            {
                let s = paths.array_size(ArrayKind::Long).open_mmap()?;
                ArraySizeReader::from_ref(s.as_ref())?;
                s
            },
            {
                let s = paths.array_size(ArrayKind::Object).open_mmap()?;
                ArraySizeReader::from_ref(s.as_ref())?;
                s
            },
        ];

        Ok(Self {
            hprof_data,
            hprof_header,
            combined_data,
            utf8_data,
            lc_data,
            aux: AuxRecordIndex::open(
                hprof_pb.open_mmap()?,
                paths.frames.open_mmap()?,
                paths.traces.open_mmap()?,
                paths.start_threads.open_mmap()?,
                paths.end_threads.open_mmap()?,
                paths.unload_classes.open_mmap()?,
                paths.utf8.open_mmap()?,
            )?,
            roots,
            refs,
            array_sizes,
            dominator: if paths.dominators.exists() {
                let src = paths.dominators.open_mmap()?;
                DominatorIndex::from_ref(src.as_ref())?;
                Some(src)
            } else {
                None
            },
            retained: if paths.retained.exists() {
                let src = paths.retained.open_mmap()?;
                RetainedIndex::from_ref(src.as_ref())?;
                Some(src)
            } else {
                None
            },
        })
    }

    /// Open a [`HeapQuery`] from in-memory sources.
    ///
    /// All sources implement [`MMapReader`] (e.g. `Vec<u8>`, `PathBuf`).
    /// `root_sources` must be 9 sources in the canonical GC-root order
    /// (unknown, jni_global, jni_local, java_frame, native_stack, sticky_class,
    /// thread_block, monitor_used, thread_obj).
    /// `array_sources` must be 9 sources in [`ArrayKind::ALL`] order
    /// (boolean, char, float, double, byte, short, int, long, object).
    #[allow(clippy::too_many_arguments)]
    pub fn from_sources(
        hprof_source: &[u8],
        combined_source: &[u8],
        utf8_source: &[u8],
        lc_source: &[u8],
        frame_source: &[u8],
        trace_source: &[u8],
        start_thread_source: &[u8],
        end_thread_source: &[u8],
        unload_class_source: &[u8],
        refs_source: &[u8],
        root_sources: [&[u8]; 9],
        array_sources: [&[u8]; 9],
    ) -> Result<Self, HprofError> {
        let hprof_data = ByteSource::from(hprof_source.to_vec());
        let combined_data = ByteSource::from(combined_source.to_vec());
        let utf8_data = ByteSource::from(utf8_source.to_vec());
        let lc_data = ByteSource::from(lc_source.to_vec());
        let hprof_header = HprofIndex::from_ref(
            hprof_data.as_ref(),
            combined_data.as_ref(),
            utf8_data.as_ref(),
            lc_data.as_ref(),
        )?
        .hprof_header();

        let make_root = |s: &[u8]| -> Result<ByteSource, HprofError> {
            RootIndexReader::from_ref(s)?;
            Ok(ByteSource::from(s.to_vec()))
        };
        let make_array = |s: &[u8]| -> Result<ByteSource, HprofError> {
            ArraySizeReader::from_ref(s)?;
            Ok(ByteSource::from(s.to_vec()))
        };
        let refs = {
            RefIndex::from_ref(refs_source)?;
            ByteSource::from(refs_source.to_vec())
        };

        Ok(Self {
            hprof_data,
            hprof_header,
            combined_data,
            utf8_data,
            lc_data,
            aux: AuxRecordIndex::open(
                ByteSource::from(hprof_source.to_vec()),
                ByteSource::from(frame_source.to_vec()),
                ByteSource::from(trace_source.to_vec()),
                ByteSource::from(start_thread_source.to_vec()),
                ByteSource::from(end_thread_source.to_vec()),
                ByteSource::from(unload_class_source.to_vec()),
                ByteSource::from(utf8_source.to_vec()),
            )?,
            roots: [
                make_root(root_sources[0])?,
                make_root(root_sources[1])?,
                make_root(root_sources[2])?,
                make_root(root_sources[3])?,
                make_root(root_sources[4])?,
                make_root(root_sources[5])?,
                make_root(root_sources[6])?,
                make_root(root_sources[7])?,
                make_root(root_sources[8])?,
            ],
            refs,
            array_sizes: [
                make_array(array_sources[0])?,
                make_array(array_sources[1])?,
                make_array(array_sources[2])?,
                make_array(array_sources[3])?,
                make_array(array_sources[4])?,
                make_array(array_sources[5])?,
                make_array(array_sources[6])?,
                make_array(array_sources[7])?,
                make_array(array_sources[8])?,
            ],
            dominator: None,
            retained: None,
        })
    }

    // ── Private reader helpers ────────────────────────────────────────────────

    fn hprof_index(&self) -> HprofIndex<'_> {
        HprofIndex::from_slice(
            self.hprof_data.as_ref(),
            self.combined_data.as_ref(),
            self.utf8_data.as_ref(),
            self.lc_data.as_ref(),
            self.hprof_header.clone(),
        )
    }

    fn root_reader(&self, rt: GcRootType) -> RootIndexReader<'_> {
        RootIndexReader::from_slice(self.roots[rt.index()].as_ref())
    }

    fn ref_index(&self) -> RefIndex<'_> {
        RefIndex::from_slice(self.refs.as_ref())
    }

    fn array_size_reader(&self, kind: ArrayKind) -> ArraySizeReader<'_> {
        ArraySizeReader::from_slice(self.array_sizes[kind.index()].as_ref())
    }

    fn retained_index(&self) -> Option<RetainedIndex<'_>> {
        self.retained
            .as_ref()
            .map(|src| RetainedIndex::from_slice(src.as_ref()))
    }

    // ── Basic accessors ───────────────────────────────────────────────────────

    /// hprof identifier size in bytes (4 or 8).
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

    /// Find any sub-record by its object ID.
    ///
    /// Returns `None` when the ID is not present. O(log n) binary search.
    pub fn find(&self, object_id: u64) -> Result<Option<SubRecord<'_>>, HprofError> {
        self.hprof_index().find_object(object_id)
    }

    /// Find a `CLASS_DUMP` sub-record by class ID.
    pub fn find_class(&self, class_id: u64) -> Result<Option<SubRecord<'_>>, HprofError> {
        self.hprof_index().find_class_dump(class_id)
    }

    /// Find an `INSTANCE_DUMP` sub-record by object ID.
    pub fn find_instance(&self, object_id: u64) -> Result<Option<SubRecord<'_>>, HprofError> {
        self.hprof_index().find_instance(object_id)
    }

    // ── Sequential iteration ──────────────────────────────────────────────────

    /// Iterate over all sub-records in ascending object-ID order.
    ///
    /// Yields parsed [`SubRecord`] values on demand; use `match` to identify
    /// the type:
    ///
    /// ```rust,ignore
    /// for result in query.iter_objects() {
    ///     match result? {
    ///         SubRecord::ClassDump(cd)    => { /* class dump */ }
    ///         SubRecord::InstanceDump(i)  => { /* instance  */ }
    ///         SubRecord::ObjArrayDump(a)  => { /* obj array */ }
    ///         SubRecord::PrimArrayDump(a) => { /* prim array */ }
    ///         _                           => { /* GC root   */ }
    ///     }
    /// }
    /// ```
    pub fn iter_objects(&self) -> ObjectIter<'_> {
        ObjectIter {
            inner: SubIndexIter::new(self.combined_data.as_ref()),
            query: self,
        }
    }

    /// Iterate raw [`SubIndexEntry`] values from the combined index in ascending
    /// object-ID order.
    ///
    /// Unlike [`Self::iter_objects`], no parsing is performed; callers receive
    /// the lightweight `(tag, object_id, position)` tuple and can choose to
    /// parse selectively with [`Self::parse_entry`].
    pub fn iter_entries(&self) -> SubIndexIter<'_> {
        SubIndexIter::new(self.combined_data.as_ref())
    }

    /// Parse the sub-record described by `entry` directly from the hprof mmap.
    ///
    /// Use this when you already hold a [`SubIndexEntry`] (e.g. from iterating
    /// via [`Self::iter_entries`]) and want to avoid a redundant binary search.
    pub fn parse_entry<'a>(&'a self, entry: &SubIndexEntry) -> Result<SubRecord<'a>, HprofError> {
        self.hprof_index().parse_entry(entry)
    }

    // ── Parallel iteration ────────────────────────────────────────────────────

    /// Execute `f` over every sub-record in parallel using rayon.
    ///
    /// Processing stops as soon as any invocation of `f` returns an `Err`;
    /// that error is returned from `par_for_each`.
    ///
    /// ```rust,ignore
    /// query.par_for_each(|record| {
    ///     match record {
    ///         SubRecord::InstanceDump(inst) => { /* ... */ }
    ///         _ => {}
    ///     }
    ///     Ok(())
    /// })?;
    /// ```
    pub fn par_for_each<F>(&self, f: F) -> Result<(), HprofError>
    where
        F: for<'a> Fn(SubRecord<'a>) -> Result<(), HprofError> + Send + Sync,
    {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        (0..self.hprof_index().object_count())
            .into_par_iter()
            .try_for_each(|i| {
                if let Some(record) = self.hprof_index().parse_at(i)? {
                    f(record)
                } else {
                    Ok(())
                }
            })
    }

    /// Execute `f` over every `CLASS_DUMP` record in parallel using rayon.
    ///
    /// Non-class records are silently skipped.
    pub fn par_classes<F>(&self, f: F) -> Result<(), HprofError>
    where
        F: for<'a> Fn(ClassDump<'a>) -> Result<(), HprofError> + Send + Sync,
    {
        self.par_for_each(|record| {
            if let SubRecord::ClassDump(cd) = record {
                f(cd)
            } else {
                Ok(())
            }
        })
    }

    /// Execute `f` over every `INSTANCE_DUMP` record in parallel using rayon.
    ///
    /// Non-instance records are silently skipped.
    pub fn par_instances<F>(&self, f: F) -> Result<(), HprofError>
    where
        F: for<'a> Fn(InstanceDump<'a>) -> Result<(), HprofError> + Send + Sync,
    {
        self.par_for_each(|record| {
            if let SubRecord::InstanceDump(inst) = record {
                f(inst)
            } else {
                Ok(())
            }
        })
    }

    /// Execute `f` over every fully-resolved instance in parallel using rayon.
    ///
    /// Each [`InstanceDump`] is resolved to a [`crate::resolved::ResolvedInstance`]
    /// before being passed to `f`; non-instance records are silently skipped.
    /// Use this instead of [`Self::par_instances`] when you need resolved field
    /// values (class name, field names, wrapper-type unwrapping).
    ///
    /// ```rust,ignore
    /// query.par_resolved_instances(|inst| {
    ///     for field in &inst.fields {
    ///         if let Value::String(_, s) = &field.value {
    ///             println!("{} - {s}", field.name);
    ///         }
    ///     }
    ///     Ok(())
    /// })?;
    /// ```
    pub fn par_resolved_instances<F>(&self, f: F) -> Result<(), HprofError>
    where
        F: Fn(crate::resolved::ResolvedInstance) -> Result<(), HprofError> + Send + Sync,
    {
        self.par_instances(|inst| f(crate::resolved::ResolvedInstance::from_dump(self, &inst)?))
    }

    /// Execute `f` over every fully-resolved class in parallel using rayon.
    ///
    /// Each [`ClassDump`] is resolved to a [`crate::resolved::ResolvedClass`]
    /// before being passed to `f`; non-class records are silently skipped.
    /// Use this instead of [`Self::par_classes`] when you need resolved names
    /// and static field values.
    ///
    /// ```rust,ignore
    /// query.par_resolved_classes(|cd| {
    ///     println!("{}: {} static fields", cd.class_name, cd.static_fields.len());
    ///     Ok(())
    /// })?;
    /// ```
    pub fn par_resolved_classes<F>(&self, f: F) -> Result<(), HprofError>
    where
        F: Fn(crate::resolved::ResolvedClass) -> Result<(), HprofError> + Send + Sync,
    {
        self.par_classes(|cd| f(crate::resolved::ResolvedClass::from_dump(self, &cd)?))
    }

    /// Execute `f` over every `INSTANCE_DUMP` whose class name exactly matches
    /// `class_name`, in parallel using rayon.
    ///
    /// Resolves the class ID once via [`Self::find_class_by_name`], then
    /// filters instances by an O(1) integer comparison per record.  Returns
    /// `Ok(())` immediately (without calling `f`) when `class_name` is not
    /// found in the heap dump.
    ///
    /// ```rust,ignore
    /// query.par_instances_of("com.example.Foo", |inst| {
    ///     let fields = query.instance_fields(&inst)?;
    ///     println!("{fields:?}");
    ///     Ok(())
    /// })?;
    /// ```
    pub fn par_instances_of<F>(&self, class_name: &str, f: F) -> Result<(), HprofError>
    where
        F: for<'a> Fn(InstanceDump<'a>) -> Result<(), HprofError> + Send + Sync,
    {
        let class_id = match self.find_class_by_name(class_name)? {
            Some(id) => id,
            None => return Ok(()),
        };
        self.par_instances(|inst| {
            if inst.class_id == class_id {
                f(inst)
            } else {
                Ok(())
            }
        })
    }

    /// Execute `f` over every fully-resolved instance whose class name exactly
    /// matches `class_name`, in parallel using rayon.
    ///
    /// Combines [`Self::par_instances_of`] with resolution to
    /// [`crate::resolved::ResolvedInstance`].  Returns `Ok(())` immediately
    /// when `class_name` is not found.
    ///
    /// ```rust,ignore
    /// query.par_resolved_instances_of("com.example.Foo", |inst| {
    ///     for field in &inst.fields {
    ///         println!("  {}: {:?}", field.name, field.value);
    ///     }
    ///     Ok(())
    /// })?;
    /// ```
    pub fn par_resolved_instances_of<F>(&self, class_name: &str, f: F) -> Result<(), HprofError>
    where
        F: Fn(crate::resolved::ResolvedInstance) -> Result<(), HprofError> + Send + Sync,
    {
        self.par_instances_of(class_name, |inst| {
            f(crate::resolved::ResolvedInstance::from_dump(self, &inst)?)
        })
    }

    // ── Name resolution ───────────────────────────────────────────────────────

    /// Look up the UTF-8 string for `name_id`.
    pub fn lookup_name(&self, name_id: u64) -> Result<Option<String>, HprofError> {
        self.hprof_index().lookup_name(name_id)
    }

    /// Find a class by its dot-notation name (e.g. `"java.lang.String"`).
    ///
    /// Returns the `class_id` of the first matching class, or `None` if no
    /// class with that name exists in the heap dump.  Scans the load-class
    /// index linearly (O(n_classes)), so call this once and then filter
    /// instance records by the returned `class_id`.
    pub fn find_class_by_name(&self, name: &str) -> Result<Option<u64>, HprofError> {
        self.hprof_index().find_class_by_name(name)
    }

    /// Return the dot-notation class name for `class_id`
    /// (e.g. `"java.lang.String"`).
    pub fn class_name(&self, class_id: u64) -> Result<Option<String>, HprofError> {
        self.hprof_index().class_name(class_id)
    }

    /// Resolve the runtime type name of the object at `object_id`.
    pub fn object_type_name(&self, object_id: u64) -> Result<String, HprofError> {
        self.hprof_index().object_type_name(object_id)
    }

    // ── Field resolution ──────────────────────────────────────────────────────

    /// Resolve the instance fields for an [`InstanceDump`], traversing the
    /// full class hierarchy.
    pub fn instance_fields(
        &self,
        instance: &InstanceDump<'_>,
    ) -> Result<Vec<ResolvedField>, HprofError> {
        self.hprof_index().instance_fields(instance)
    }

    /// Attempt to resolve `object_id` as a primitive Java wrapper value.
    ///
    /// Handles `String`, `Integer`, `Long`, `Double`, `Float`, `Short`,
    /// `Byte`, `Boolean`, and `Character`.  Anything else returns
    /// [`JavaValue::Object`].
    pub fn resolve_value(&self, object_id: u64) -> Result<JavaValue, HprofError> {
        self.hprof_index().resolve_value(object_id)
    }

    // ── Auxiliary record lookup ───────────────────────────────────────────────

    /// Find a `HPROF_FRAME` record by `frame_id`.
    pub fn find_frame(&self, frame_id: u64) -> Result<Option<Frame>, HprofError> {
        self.aux.find_frame(frame_id)
    }

    /// Find a `HPROF_TRACE` record by `trace_serial`.
    pub fn find_trace(&self, trace_serial: u32) -> Result<Option<Trace>, HprofError> {
        self.aux.find_trace(trace_serial)
    }

    /// Find a `HPROF_START_THREAD` record by `thread_serial`.
    pub fn find_thread(&self, thread_serial: u32) -> Result<Option<StartThread>, HprofError> {
        self.aux.find_start_thread(thread_serial)
    }

    /// Resolve all name IDs in `frame` to strings.
    pub fn resolve_frame(&self, frame: &Frame) -> Result<ResolvedFrame, HprofError> {
        self.aux.resolve_frame(frame)
    }

    /// Resolve all name IDs in `thread` to strings.
    pub fn resolve_thread(&self, thread: &StartThread) -> Result<ResolvedThread, HprofError> {
        self.aux.resolve_thread(thread)
    }

    /// Parse every frame in `trace` and return them in order.
    pub fn trace_frames(&self, trace: &Trace) -> Result<Vec<Frame>, HprofError> {
        self.aux.trace_frames(trace)
    }

    /// Returns `true` if a `HPROF_END_THREAD` record exists for `thread_serial`.
    pub fn was_thread_ended(&self, thread_serial: u32) -> bool {
        self.aux.was_thread_ended(thread_serial)
    }

    /// Returns `true` if a `HPROF_UNLOAD_CLASS` record exists for `class_serial`.
    pub fn was_class_unloaded(&self, class_serial: u32) -> bool {
        self.aux.was_class_unloaded(class_serial)
    }

    // ── Auxiliary record iteration ────────────────────────────────────────────

    /// Iterate all `HPROF_FRAME` records in ascending `frame_id` order.
    pub fn iter_frames(&self) -> FrameIter<'_> {
        self.aux.iter_frames()
    }

    /// Iterate all `HPROF_TRACE` records in ascending `trace_serial` order.
    pub fn iter_traces(&self) -> TraceIter<'_> {
        self.aux.iter_traces()
    }

    /// Iterate all `HPROF_START_THREAD` records in ascending `thread_serial` order.
    pub fn iter_threads(&self) -> StartThreadIter<'_> {
        self.aux.iter_start_threads()
    }

    // ── GC root access ────────────────────────────────────────────────────────

    /// Find the root index entry for `object_id` in the given `root_type` file.
    ///
    /// Returns `Some(entry)` if `object_id` appears as a root of that type,
    /// `None` otherwise.  O(log n) binary search.
    pub fn find_root(&self, object_id: u64, root_type: GcRootType) -> Option<RootIndexEntry> {
        self.root_reader(root_type).find(object_id)
    }

    /// Iterate all root entries for the given `root_type` in ascending
    /// `object_id` order.
    pub fn iter_roots(&self, root_type: GcRootType) -> RootIter<'_> {
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

    /// Return the IDs of all objects that hold a direct reference to `object_id`.
    ///
    /// Results are bounded by [`crate::ref_index::MAX_BACK_REFS`].  Uses the
    /// pre-built reference index for O(log n) lookup.
    pub fn refs_to(&self, object_id: u64) -> Vec<u64> {
        self.ref_index().find(object_id)
    }

    /// Walk backwards through the reference graph from `object_id` to find
    /// the shortest path to any GC root.
    ///
    /// Returns [`RootPathResult::Found`] with a vec where `path[0]` is the GC
    /// root and `path.last()` is `object_id`.  Returns
    /// [`RootPathResult::LimitReached`] if the BFS visited
    /// [`ROOT_PATH_SEARCH_LIMIT`] nodes without finding a root, or
    /// [`RootPathResult::NotReachable`] if the graph was exhausted.
    ///
    /// **Caveat:** [`Self::refs_to`] is capped at
    /// [`crate::ref_index::MAX_BACK_REFS`] per node.  Paths that pass through
    /// high-fanin objects may not be found.
    pub fn path_to_root(&self, object_id: u64) -> RootPathResult {
        use std::collections::{HashMap, HashSet, VecDeque};

        if self.is_gc_root(object_id) {
            return RootPathResult::Found(vec![object_id]);
        }

        let mut visited: HashSet<u64> = HashSet::new();
        // parent[v] = u means v references u, so u is one step closer to object_id
        let mut parent: HashMap<u64, u64> = HashMap::new();
        let mut queue: VecDeque<u64> = VecDeque::new();

        visited.insert(object_id);
        queue.push_back(object_id);

        while let Some(current) = queue.pop_front() {
            for referrer in self.refs_to(current) {
                if !visited.insert(referrer) {
                    continue;
                }
                parent.insert(referrer, current);

                if self.is_gc_root(referrer) {
                    // Reconstruct path: root → … → object_id
                    let mut path = vec![referrer];
                    let mut node = referrer;
                    while node != object_id {
                        match parent.get(&node).copied() {
                            Some(next) => {
                                node = next;
                                path.push(node);
                            }
                            None => break,
                        }
                    }
                    return RootPathResult::Found(path);
                }

                if visited.len() >= ROOT_PATH_SEARCH_LIMIT {
                    return RootPathResult::LimitReached;
                }
                queue.push_back(referrer);
            }
        }

        RootPathResult::NotReachable
    }

    /// Total number of reference records in the reference index.
    pub fn ref_count(&self) -> usize {
        self.ref_index().len()
    }

    // ── Retained heap / dominator tree ────────────────────────────────────────

    /// Returns `true` if the dominator tree and retained heap size indexes are
    /// available (i.e. were built by the indexing pipeline).
    pub fn has_retained_heap(&self) -> bool {
        self.retained.is_some() && self.dominator.is_some()
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
    fn dominator_index(&self) -> Option<DominatorIndex<'_>> {
        self.dominator
            .as_ref()
            .map(|src| DominatorIndex::from_slice(src.as_ref()))
    }

    pub fn dominator_of(&self, object_id: u64) -> Option<u64> {
        self.dominator_index()?.find(object_id)
    }

    /// Iterate all `(object_id, retained_bytes)` pairs from the retained heap
    /// index in ascending `object_id` order.
    ///
    /// Returns `None` when the retained heap index was not built.
    pub fn iter_retained(&self) -> Option<crate::dominator::RetainedIter<'_>> {
        self.retained_index().map(|r| r.iter())
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
    use crate::array_index::build_array_size_indexes;
    use crate::aux_index::{
        build_end_thread_index, build_frame_index, build_start_thread_index, build_trace_index,
        build_unload_class_index,
    };
    use crate::aux_query::LineNumber;
    use crate::heap_index::index_heap_dumps;
    use crate::heap_parser::FieldValue;
    use crate::heap_query::build_name_indexes;
    use crate::object_store::combine_sort_and_split;
    use crate::record_index::index_hprof;
    use crate::ref_index::build_reference_index;
    use crate::vfs::SubIndexDir;
    use std::sync::atomic::{AtomicU64, Ordering};

    // ── Test hprof builder ────────────────────────────────────────────────────

    fn write_record(buf: &mut Vec<u8>, tag: u8, body: &[u8]) {
        buf.push(tag);
        buf.extend_from_slice(&0u32.to_be_bytes());
        buf.extend_from_slice(&(body.len() as u32).to_be_bytes());
        buf.extend_from_slice(body);
    }

    /// Build a minimal hprof combining heap dump objects and aux records:
    ///
    /// Heap content (id_size = 8):
    ///   UTF8(1,"value"), UTF8(2,"java/lang/Integer"), UTF8(3,"java/lang/Object"),
    ///   UTF8(4,"main"),  UTF8(5,"()V"),               UTF8(6,"MyClass.java")
    ///   LOAD_CLASS(class_id=0x200, name_id=2)
    ///   LOAD_CLASS(class_id=0x300, name_id=3)
    ///   HEAP_DUMP_SEGMENT:
    ///     CLASS_DUMP(0x200, super=0x300, 1 int field "value")
    ///     CLASS_DUMP(0x300, super=0)
    ///     INSTANCE_DUMP(0x100, class=0x200, data=[0,0,0,42])
    ///
    /// Aux records:
    ///   FRAME(id=0x10, method=4, sig=5, src=6, class_serial=1, line=7)
    ///   TRACE(serial=1, thread_serial=1, frames=[0x10])
    ///   START_THREAD(serial=1, thread_id=0xABC, trace=1, name=4)
    ///   END_THREAD(serial=1)
    fn build_test_hprof() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"JAVA PROFILE 1.0.2\0");
        buf.extend_from_slice(&8u32.to_be_bytes()); // id_size = 8
        buf.extend_from_slice(&0u64.to_be_bytes()); // timestamp

        let utf8 = |buf: &mut Vec<u8>, id: u64, s: &[u8]| {
            let mut body = Vec::new();
            body.extend_from_slice(&id.to_be_bytes());
            body.extend_from_slice(s);
            write_record(buf, 0x01, &body);
        };
        utf8(&mut buf, 1, b"value");
        utf8(&mut buf, 2, b"java/lang/Integer");
        utf8(&mut buf, 3, b"java/lang/Object");
        utf8(&mut buf, 4, b"main");
        utf8(&mut buf, 5, b"()V");
        utf8(&mut buf, 6, b"MyClass.java");

        // LOAD_CLASS: class_id=0x200, name_id=2
        let mut lc = Vec::new();
        lc.extend_from_slice(&1u32.to_be_bytes()); // class_serial
        lc.extend_from_slice(&0x200u64.to_be_bytes()); // class_id
        lc.extend_from_slice(&0u32.to_be_bytes()); // stack_serial
        lc.extend_from_slice(&2u64.to_be_bytes()); // name_id
        write_record(&mut buf, 0x02, &lc);

        let mut lc = Vec::new();
        lc.extend_from_slice(&2u32.to_be_bytes()); // class_serial
        lc.extend_from_slice(&0x300u64.to_be_bytes()); // class_id
        lc.extend_from_slice(&0u32.to_be_bytes());
        lc.extend_from_slice(&3u64.to_be_bytes()); // name_id
        write_record(&mut buf, 0x02, &lc);

        // HEAP_DUMP_SEGMENT
        let mut seg = Vec::new();

        // CLASS_DUMP(0x200, super=0x300, instance_size=4, 1 field: name_id=1, type=int)
        seg.push(0x20u8);
        seg.extend_from_slice(&0x200u64.to_be_bytes()); // class_id
        seg.extend_from_slice(&0u32.to_be_bytes()); // stack_serial
        seg.extend_from_slice(&0x300u64.to_be_bytes()); // super
        seg.extend_from_slice(&[0u8; 8 * 5]); // loader+signers+domain+res1+res2
        seg.extend_from_slice(&4u32.to_be_bytes()); // instance_size
        seg.extend_from_slice(&0u16.to_be_bytes()); // cp_count
        seg.extend_from_slice(&0u16.to_be_bytes()); // statics_count
        seg.extend_from_slice(&1u16.to_be_bytes()); // instance_fields_count
        seg.extend_from_slice(&1u64.to_be_bytes()); // field name_id=1 ("value")
        seg.push(10u8); // type = int

        // CLASS_DUMP(0x300, super=0, 0 fields) — java.lang.Object
        seg.push(0x20u8);
        seg.extend_from_slice(&0x300u64.to_be_bytes());
        seg.extend_from_slice(&0u32.to_be_bytes());
        seg.extend_from_slice(&0u64.to_be_bytes()); // super = null
        seg.extend_from_slice(&[0u8; 8 * 5]);
        seg.extend_from_slice(&0u32.to_be_bytes());
        seg.extend_from_slice(&0u16.to_be_bytes());
        seg.extend_from_slice(&0u16.to_be_bytes());
        seg.extend_from_slice(&0u16.to_be_bytes());

        // INSTANCE_DUMP(0x100, class=0x200, data=[0,0,0,42])
        seg.push(0x21u8);
        seg.extend_from_slice(&0x100u64.to_be_bytes()); // object_id
        seg.extend_from_slice(&0u32.to_be_bytes()); // stack_serial
        seg.extend_from_slice(&0x200u64.to_be_bytes()); // class_id
        seg.extend_from_slice(&4u32.to_be_bytes()); // data_len
        seg.extend_from_slice(&42i32.to_be_bytes()); // value = 42

        write_record(&mut buf, 0x1C, &seg); // HEAP_DUMP_SEGMENT

        // FRAME record
        let mut frame_body = Vec::new();
        frame_body.extend_from_slice(&0x10u64.to_be_bytes()); // frame_id
        frame_body.extend_from_slice(&4u64.to_be_bytes()); // method_name_id ("main")
        frame_body.extend_from_slice(&5u64.to_be_bytes()); // method_sig_id  ("()V")
        frame_body.extend_from_slice(&6u64.to_be_bytes()); // source_file_id ("MyClass.java")
        frame_body.extend_from_slice(&1u32.to_be_bytes()); // class_serial
        frame_body.extend_from_slice(&7i32.to_be_bytes()); // line_number
        write_record(&mut buf, 0x04, &frame_body);

        // TRACE record
        let mut trace_body = Vec::new();
        trace_body.extend_from_slice(&1u32.to_be_bytes()); // trace_serial
        trace_body.extend_from_slice(&1u32.to_be_bytes()); // thread_serial
        trace_body.extend_from_slice(&1u32.to_be_bytes()); // num_frames
        trace_body.extend_from_slice(&0x10u64.to_be_bytes());
        write_record(&mut buf, 0x05, &trace_body);

        // START_THREAD record
        let mut st_body = Vec::new();
        st_body.extend_from_slice(&1u32.to_be_bytes()); // thread_serial
        st_body.extend_from_slice(&0xABCu64.to_be_bytes()); // thread_id
        st_body.extend_from_slice(&1u32.to_be_bytes()); // stack_trace_serial
        st_body.extend_from_slice(&4u64.to_be_bytes()); // thread_name_id ("main")
        st_body.extend_from_slice(&0u64.to_be_bytes()); // group_name_id
        st_body.extend_from_slice(&0u64.to_be_bytes()); // parent_group_id
        write_record(&mut buf, 0x0A, &st_body);

        // END_THREAD record
        write_record(&mut buf, 0x0B, &1u32.to_be_bytes());

        buf
    }

    /// Run the full build pipeline and open a [`HeapQuery`] entirely in memory.
    fn build_query(hprof_data: &[u8]) -> HeapQuery {
        let hprof = hprof_data.to_vec();
        let mut p1 = Vec::new();
        let p2d = SubIndexDir::mem();
        let mut p4 = Vec::new();
        let mut utf8 = Vec::new();
        let mut lc = Vec::new();
        let mut frames = Vec::new();
        let mut traces = Vec::new();
        let mut st = Vec::new();
        let mut et = Vec::new();
        let mut uc = Vec::new();
        let mut refs = Vec::new();
        // 9 separate root buffers (one per GC root type in canonical order).
        let mut r0: Vec<u8> = Vec::new();
        let mut r1: Vec<u8> = Vec::new();
        let mut r2: Vec<u8> = Vec::new();
        let mut r3: Vec<u8> = Vec::new();
        let mut r4: Vec<u8> = Vec::new();
        let mut r5: Vec<u8> = Vec::new();
        let mut r6: Vec<u8> = Vec::new();
        let mut r7: Vec<u8> = Vec::new();
        let mut r8: Vec<u8> = Vec::new();
        let mut arrays: [Vec<u8>; 9] = std::array::from_fn(|_| Vec::new());

        index_hprof(&hprof, &mut p1).unwrap();
        index_heap_dumps(&hprof, &p1, &p2d).unwrap();

        combine_sort_and_split(
            &p2d,
            &mut p4,
            &mut [
                &mut r0, &mut r1, &mut r2, &mut r3, &mut r4, &mut r5, &mut r6, &mut r7, &mut r8,
            ],
        )
        .unwrap();

        build_name_indexes(&hprof, &p1, &mut utf8, &mut lc).unwrap();
        build_reference_index(&hprof, &p4, &utf8, &lc, &mut refs).unwrap();
        build_frame_index(&hprof, &p1, &mut frames).unwrap();
        build_trace_index(&hprof, &p1, &mut traces).unwrap();
        build_start_thread_index(&hprof, &p1, &mut st).unwrap();
        build_end_thread_index(&hprof, &p1, &mut et).unwrap();
        build_unload_class_index(&hprof, &p1, &mut uc).unwrap();

        build_array_size_indexes(&hprof, &p4, &mut arrays).unwrap();

        HeapQuery::from_sources(
            &hprof,
            &p4,
            &utf8,
            &lc,
            &frames,
            &traces,
            &st,
            &et,
            &uc,
            &refs,
            [&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7, &r8],
            [
                &arrays[0], &arrays[1], &arrays[2], &arrays[3], &arrays[4], &arrays[5], &arrays[6],
                &arrays[7], &arrays[8],
            ],
        )
        .unwrap()
    }

    // ── Heap query tests ──────────────────────────────────────────────────────

    #[test]
    fn find_by_object_id() {
        let query = build_query(&build_test_hprof());
        let record = query.find(0x100).unwrap().unwrap();
        assert!(matches!(record, SubRecord::InstanceDump(_)));
    }

    #[test]
    fn find_missing_returns_none() {
        let query = build_query(&build_test_hprof());
        assert!(query.find(0xDEAD).unwrap().is_none());
    }

    #[test]
    fn find_class_returns_class_dump() {
        let query = build_query(&build_test_hprof());
        let record = query.find_class(0x200).unwrap().unwrap();
        assert!(matches!(record, SubRecord::ClassDump(_)));
    }

    #[test]
    fn find_instance_returns_instance_dump() {
        let query = build_query(&build_test_hprof());
        let record = query.find_instance(0x100).unwrap().unwrap();
        assert!(matches!(record, SubRecord::InstanceDump(_)));
    }

    #[test]
    fn class_name_resolved() {
        let query = build_query(&build_test_hprof());
        assert_eq!(
            query.class_name(0x200).unwrap(),
            Some("java.lang.Integer".to_string())
        );
    }

    #[test]
    fn instance_fields_resolved() {
        let query = build_query(&build_test_hprof());
        let SubRecord::InstanceDump(inst) = query.find_instance(0x100).unwrap().unwrap() else {
            panic!("expected InstanceDump");
        };
        let fields = query.instance_fields(&inst).unwrap();
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].name, "value");
        assert_eq!(fields[0].value, FieldValue::Int(42));
    }

    #[test]
    fn resolve_value_integer_wrapper() {
        let query = build_query(&build_test_hprof());
        let val = query.resolve_value(0x100).unwrap();
        assert!(matches!(val, JavaValue::Integer(0x100, 42)));
    }

    #[test]
    fn object_count_nonzero() {
        let query = build_query(&build_test_hprof());
        assert!(query.object_count() > 0);
    }

    #[test]
    fn iter_objects_yields_records() {
        let query = build_query(&build_test_hprof());
        let records: Vec<_> = query.iter_objects().collect::<Result<_, _>>().unwrap();
        // Should contain at least the 2 class dumps and 1 instance dump.
        assert!(records.len() >= 3);
        let has_instance = records
            .iter()
            .any(|r| matches!(r, SubRecord::InstanceDump(_)));
        let has_class = records.iter().any(|r| matches!(r, SubRecord::ClassDump(_)));
        assert!(has_instance);
        assert!(has_class);
    }

    #[test]
    fn par_for_each_visits_all_records() {
        let query = build_query(&build_test_hprof());
        let count = AtomicU64::new(0);
        query
            .par_for_each(|_record| {
                count.fetch_add(1, Ordering::Relaxed);
                Ok(())
            })
            .unwrap();
        assert_eq!(count.load(Ordering::Relaxed), query.object_count() as u64);
    }

    #[test]
    fn par_instances_visits_only_instances() {
        let query = build_query(&build_test_hprof());
        let count = AtomicU64::new(0);
        query
            .par_instances(|_inst| {
                count.fetch_add(1, Ordering::Relaxed);
                Ok(())
            })
            .unwrap();
        // Test hprof has exactly 1 INSTANCE_DUMP.
        assert_eq!(count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn par_classes_visits_only_classes() {
        let query = build_query(&build_test_hprof());
        let count = AtomicU64::new(0);
        query
            .par_classes(|_cd| {
                count.fetch_add(1, Ordering::Relaxed);
                Ok(())
            })
            .unwrap();
        // Test hprof has 2 CLASS_DUMP records.
        assert_eq!(count.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn par_resolved_instances_visits_resolved_instances() {
        use std::sync::Mutex;
        let query = build_query(&build_test_hprof());
        let names: Mutex<Vec<String>> = Mutex::new(Vec::new());
        query
            .par_resolved_instances(|inst| {
                names.lock().unwrap().push(inst.class_name.clone());
                Ok(())
            })
            .unwrap();
        let mut names = names.into_inner().unwrap();
        names.sort();
        // Test hprof has 1 INSTANCE_DUMP with class java.lang.Integer.
        assert_eq!(names, vec!["java.lang.Integer"]);
    }

    #[test]
    fn par_resolved_instances_fields_are_resolved() {
        use crate::resolved::Value;
        use std::sync::Mutex;
        let query = build_query(&build_test_hprof());
        let values: Mutex<Vec<Value>> = Mutex::new(Vec::new());
        query
            .par_resolved_instances(|inst| {
                for f in &inst.fields {
                    values.lock().unwrap().push(f.value.clone());
                }
                Ok(())
            })
            .unwrap();
        let values = values.into_inner().unwrap();
        // INSTANCE_DUMP(0x100, class=Integer, value=42) → one Int(42) field.
        assert_eq!(values, vec![Value::Int(42)]);
    }

    #[test]
    fn par_resolved_classes_visits_resolved_classes() {
        use std::sync::Mutex;
        let query = build_query(&build_test_hprof());
        let names: Mutex<Vec<String>> = Mutex::new(Vec::new());
        query
            .par_resolved_classes(|cd| {
                names.lock().unwrap().push(cd.class_name.clone());
                Ok(())
            })
            .unwrap();
        let mut names = names.into_inner().unwrap();
        names.sort();
        // Test hprof has CLASS_DUMP for Integer and Object.
        assert_eq!(names, vec!["java.lang.Integer", "java.lang.Object"]);
    }

    #[test]
    fn par_resolved_classes_super_class_name_resolved() {
        use std::sync::Mutex;
        let query = build_query(&build_test_hprof());
        let super_names: Mutex<Vec<Option<String>>> = Mutex::new(Vec::new());
        query
            .par_resolved_classes(|cd| {
                if cd.class_name == "java.lang.Integer" {
                    super_names
                        .lock()
                        .unwrap()
                        .push(cd.super_class_name.clone());
                }
                Ok(())
            })
            .unwrap();
        let super_names = super_names.into_inner().unwrap();
        assert_eq!(super_names, vec![Some("java.lang.Object".to_string())]);
    }

    // ── find_class_by_name / par_instances_of tests ───────────────────────────

    #[test]
    fn find_class_by_name_returns_class_id() {
        let query = build_query(&build_test_hprof());
        let class_id = query.find_class_by_name("java.lang.Integer").unwrap();
        assert_eq!(class_id, Some(0x200));
    }

    #[test]
    fn find_class_by_name_unknown_returns_none() {
        let query = build_query(&build_test_hprof());
        let class_id = query.find_class_by_name("does.not.Exist").unwrap();
        assert_eq!(class_id, None);
    }

    #[test]
    fn par_instances_of_visits_matching_instances() {
        let query = build_query(&build_test_hprof());
        let count = AtomicU64::new(0);
        query
            .par_instances_of("java.lang.Integer", |_inst| {
                count.fetch_add(1, Ordering::Relaxed);
                Ok(())
            })
            .unwrap();
        // One Integer instance in the test hprof.
        assert_eq!(count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn par_instances_of_skips_non_matching_instances() {
        let query = build_query(&build_test_hprof());
        let count = AtomicU64::new(0);
        query
            .par_instances_of("java.lang.Object", |_inst| {
                count.fetch_add(1, Ordering::Relaxed);
                Ok(())
            })
            .unwrap();
        // No Object instances directly (only Integer which extends Object).
        assert_eq!(count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn par_instances_of_unknown_class_is_noop() {
        let query = build_query(&build_test_hprof());
        let count = AtomicU64::new(0);
        query
            .par_instances_of("no.such.Class", |_inst| {
                count.fetch_add(1, Ordering::Relaxed);
                Ok(())
            })
            .unwrap();
        assert_eq!(count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn par_resolved_instances_of_yields_resolved() {
        use crate::resolved::Value;
        use std::sync::Mutex;
        let query = build_query(&build_test_hprof());
        let values: Mutex<Vec<Value>> = Mutex::new(Vec::new());
        query
            .par_resolved_instances_of("java.lang.Integer", |inst| {
                for f in &inst.fields {
                    values.lock().unwrap().push(f.value.clone());
                }
                Ok(())
            })
            .unwrap();
        // Integer instance has one int field "value" = 42.
        assert_eq!(values.into_inner().unwrap(), vec![Value::Int(42)]);
    }

    // ── Aux record tests ──────────────────────────────────────────────────────

    #[test]
    fn find_frame_by_id() {
        let query = build_query(&build_test_hprof());
        let frame = query.find_frame(0x10).unwrap().unwrap();
        assert_eq!(frame.frame_id, 0x10);
        assert_eq!(frame.class_serial, 1);

        let resolved = query.resolve_frame(&frame).unwrap();
        assert_eq!(resolved.method_name, "main");
        assert_eq!(resolved.method_signature, "()V");
        assert_eq!(resolved.source_file, "MyClass.java");
        assert_eq!(resolved.line_number, LineNumber::Line(7));
    }

    #[test]
    fn find_trace_by_serial() {
        let query = build_query(&build_test_hprof());
        let trace = query.find_trace(1).unwrap().unwrap();
        assert_eq!(trace.trace_serial, 1);
        assert_eq!(trace.thread_serial, 1);

        let frames = query.trace_frames(&trace).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].frame_id, 0x10);
    }

    #[test]
    fn find_thread_by_serial() {
        let query = build_query(&build_test_hprof());
        let thread = query.find_thread(1).unwrap().unwrap();
        assert_eq!(thread.thread_id, 0xABC);

        let resolved = query.resolve_thread(&thread).unwrap();
        assert_eq!(resolved.thread_name, "main");
    }

    #[test]
    fn was_thread_ended_true() {
        let query = build_query(&build_test_hprof());
        assert!(query.was_thread_ended(1));
        assert!(!query.was_thread_ended(99));
    }

    #[test]
    fn iter_frames_yields_all() {
        let query = build_query(&build_test_hprof());
        let frames: Vec<_> = query.iter_frames().collect::<Result<_, _>>().unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].frame_id, 0x10);
    }

    #[test]
    fn iter_traces_yields_all() {
        let query = build_query(&build_test_hprof());
        let traces: Vec<_> = query.iter_traces().collect::<Result<_, _>>().unwrap();
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].trace_serial, 1);
    }

    #[test]
    fn iter_threads_yields_all() {
        let query = build_query(&build_test_hprof());
        let threads: Vec<_> = query.iter_threads().collect::<Result<_, _>>().unwrap();
        assert_eq!(threads.len(), 1);
        assert_eq!(threads[0].thread_id, 0xABC);
    }

    // ── GC root tests ─────────────────────────────────────────────────────────

    #[test]
    fn is_gc_root_false_for_plain_instance() {
        // The test hprof has no GC roots; instance 0x100 is not a root.
        let query = build_query(&build_test_hprof());
        assert!(!query.is_gc_root(0x100));
    }

    #[test]
    fn find_root_returns_none_for_absent_id() {
        let query = build_query(&build_test_hprof());
        assert!(query.find_root(0x100, GcRootType::StickyClass).is_none());
    }

    #[test]
    fn iter_roots_empty_when_no_roots() {
        let query = build_query(&build_test_hprof());
        let count = query.iter_roots(GcRootType::JniGlobal).count();
        assert_eq!(count, 0);
    }

    #[test]
    fn root_types_of_empty_for_non_root() {
        let query = build_query(&build_test_hprof());
        assert!(query.root_types_of(0x100).is_empty());
    }

    // ── Reference index tests ─────────────────────────────────────────────────

    #[test]
    fn refs_to_returns_vec() {
        // The test hprof has INSTANCE_DUMP(0x100, class=0x200).
        // Instance 0x100 holds no object-reference fields (only an int field),
        // so refs_to(0x100) should be empty; but the ref_count should be >= 0.
        let query = build_query(&build_test_hprof());
        let _ = query.refs_to(0x100); // just ensure it doesn't panic
        let _ = query.ref_count(); // same
    }
}
