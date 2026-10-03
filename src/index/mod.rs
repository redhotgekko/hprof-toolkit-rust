//! Index storage layer.
//!
//! Every index file the toolkit produces is a flat sequence of fixed-size
//! records.  This module owns *where those bytes live* — see [`store`] — so
//! that the rest of the crate never names `std::fs` directly and every build
//! and query path can run against an in-memory backend in tests.

pub mod manifest;
pub(crate) mod record_file;
pub mod store;

pub use manifest::{FORMAT_VERSION, HprofIdentity, MANIFEST_NAME, Manifest};
pub use record_file::RecordIter;
pub(crate) use record_file::{Entry, RecordFile, RecordWriter, read_u32_le, read_u64_le};
pub use store::{ByteSource, FsStore, IndexStore, MemStore, StoreWriter};

/// Canonical entry names inside an [`IndexStore`].
///
/// These are the file names under `{stem}.indexes/` on disk and the map keys
/// in memory.  Keep them in one place so builders and readers cannot drift.
pub mod names {
    pub const RECORD_INDEX: &str = "record_index.bin";
    /// Prefix for the per-segment heap sub-record index entries (build
    /// intermediates, removed once [`OBJECT_STORE`] exists).
    pub const HEAP_INDEX_PREFIX: &str = "heap_index/";
    pub const OBJECT_STORE: &str = "object_store.bin";
    pub const UTF8: &str = "utf8.bin";
    pub const LOAD_CLASS: &str = "loadclass.bin";
    pub const REFS: &str = "refs.bin";
    pub const FRAMES: &str = "frames.bin";
    pub const TRACES: &str = "traces.bin";
    pub const START_THREADS: &str = "start_threads.bin";
    pub const END_THREADS: &str = "end_threads.bin";
    pub const UNLOAD_CLASSES: &str = "unload_classes.bin";
    pub const DOMINATORS: &str = "dominators.bin";
    pub const RETAINED: &str = "retained.bin";
    /// `(retained_bytes, object_id)` sorted by retained size, largest first.
    pub const RETAINED_BY_SIZE: &str = "retained_by_size.bin";
    /// `(dominator_id, object_id)` sorted by dominator then object id.
    pub const DOMINATOR_CHILDREN: &str = "dominator_children.bin";

    /// `(class_key, object_id, position)` sorted by `(class_key, object_id)`.
    pub const INSTANCES_BY_CLASS: &str = "instances_by_class.bin";
    /// `(instance_count, shallow_bytes, class_key)` sorted by count, largest first.
    pub const CLASS_HISTOGRAM: &str = "class_histogram.bin";

    /// Every entry the dominator/retained step produces.
    pub const RETAINED_SET: [&str; 4] =
        [DOMINATORS, RETAINED, RETAINED_BY_SIZE, DOMINATOR_CHILDREN];

    /// Per-type GC root index names, in [`crate::root_index::GcRootType::ALL`] order.
    pub const ROOTS: [&str; 9] = [
        "root_unknown.bin",
        "root_jni_global.bin",
        "root_jni_local.bin",
        "root_java_frame.bin",
        "root_native_stack.bin",
        "root_sticky_class.bin",
        "root_thread_block.bin",
        "root_monitor_used.bin",
        "root_thread_obj.bin",
    ];

    /// Every entry a [`crate::query::HeapQuery`] needs in order to open
    /// (everything except the optional retained-heap family).
    pub fn mandatory() -> Vec<&'static str> {
        let mut v = vec![
            RECORD_INDEX,
            OBJECT_STORE,
            INSTANCES_BY_CLASS,
            CLASS_HISTOGRAM,
            UTF8,
            LOAD_CLASS,
            REFS,
            FRAMES,
            TRACES,
            START_THREADS,
            END_THREADS,
            UNLOAD_CLASSES,
        ];
        v.extend(ROOTS);
        v.extend(ARRAYS);
        v
    }

    /// Name of temporary part `k` of the entry `base`, used by parallel
    /// builders that write one part per worker and then concatenate them.
    pub fn part(base: &str, k: usize) -> String {
        format!("{base}.part.{k:04}")
    }

    /// Prefix matching every part of `base` (for cleanup via
    /// [`super::IndexStore::remove_prefix`]).
    pub fn part_prefix(base: &str) -> String {
        format!("{base}.part.")
    }

    /// Per-kind array size index names, in [`crate::array_index::ArrayKind::ALL`] order.
    pub const ARRAYS: [&str; 9] = [
        "array_boolean.bin",
        "array_char.bin",
        "array_float.bin",
        "array_double.bin",
        "array_byte.bin",
        "array_short.bin",
        "array_int.bin",
        "array_long.bin",
        "array_object.bin",
    ];
}
