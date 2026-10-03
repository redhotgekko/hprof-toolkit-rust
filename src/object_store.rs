//! The object store: every heap sub-record index entry, sorted by object id.
//!
//! Built by concatenating the per-segment intermediates written by
//! [`crate::heap_index::index_heap_dumps`] (streamed straight from the store,
//! never read into memory), sorting the result in place, and fanning GC-root
//! entries out into the nine per-type root indexes in the same pass.

use crate::heap_index::sub_record::{
    SUB_INDEX_ENTRY_SIZE, SubIndexEntry, TAG_ROOT_JAVA_FRAME, TAG_ROOT_JNI_GLOBAL,
    TAG_ROOT_JNI_LOCAL, TAG_ROOT_MONITOR_USED, TAG_ROOT_NATIVE_STACK, TAG_ROOT_STICKY_CLASS,
    TAG_ROOT_THREAD_BLOCK, TAG_ROOT_THREAD_OBJ, TAG_ROOT_UNKNOWN,
};
use crate::hprof::HprofError;
use crate::index::{IndexStore, StoreWriter, names};
use crate::root_index::{RootIndexCounts, RootIndexEntry};
use std::io::Write;

/// Counts returned by [`combine_sort_and_split`].
pub struct CombinedCounts {
    /// Total entries written to the combined object store.
    pub total: u64,
    /// Per-type counts for the nine GC root index files.
    pub roots: RootIndexCounts,
}

/// Combines, sorts, and splits heap sub-index data in one logical step.
///
/// 1. Streams every entry under [`names::HEAP_INDEX_PREFIX`] in `store`
///    into `combined` (each intermediate is opened as a byte source and
///    copied; nothing is buffered in process memory).
/// 2. Sorts `combined` in-place by `object_id`.
/// 3. Makes a single sequential pass over the sorted data, routing each
///    GC-root entry to the appropriate per-type root index writer.
///
/// `root_writers` must hold exactly 9 writers, one per GC root type in the
/// canonical order defined by [`crate::root_index::GcRootType::ALL`]:
/// unknown, jni_global, jni_local, java_frame, native_stack, sticky_class,
/// thread_block, monitor_used, thread_obj.
///
/// Each root writer receives sorted entries (sorted because `combined` is
/// already sorted by `object_id`).  The caller is responsible for committing
/// the writers and for removing the intermediates afterwards.
pub fn combine_sort_and_split(
    store: &dyn IndexStore,
    combined: &mut dyn StoreWriter,
    root_writers: &mut [&mut dyn Write],
) -> Result<CombinedCounts, HprofError> {
    if root_writers.len() != 9 {
        return Err(HprofError::Internal(format!(
            "combine_sort_and_split needs 9 root writers, got {}",
            root_writers.len()
        )));
    }

    let total = concatenate_intermediates(store, combined)?;
    if total > 1 {
        crate::sort::parallel_introsort(combined.as_mut_bytes()?, SUB_INDEX_ENTRY_SIZE, 8);
    }
    let roots = fan_out_roots(combined.as_mut_bytes()?, root_writers)?;
    Ok(CombinedCounts { total, roots })
}

// ── Internals ─────────────────────────────────────────────────────────────────

/// Stream every heap-index intermediate in `store` into `combined`.
///
/// Only complete 24-byte records are written; any partial trailing bytes in
/// a source are silently skipped (they should not occur with valid data).
fn concatenate_intermediates(
    store: &dyn IndexStore,
    combined: &mut dyn StoreWriter,
) -> Result<u64, HprofError> {
    let mut total = 0u64;

    for name in store.list(names::HEAP_INDEX_PREFIX)? {
        let source = store.open(&name)?;
        let bytes = source.as_ref();
        let aligned = (bytes.len() / SUB_INDEX_ENTRY_SIZE) * SUB_INDEX_ENTRY_SIZE;
        combined.write_all(&bytes[..aligned])?;
        total += (aligned / SUB_INDEX_ENTRY_SIZE) as u64;
    }
    combined.flush()?;
    Ok(total)
}

/// Read the sorted combined data and route GC-root entries to nine per-type
/// output writers.  Non-root entries (class dumps, instance dumps, array dumps)
/// are skipped.  Because `combined_data` is already sorted by `object_id`
/// each output writer receives sorted entries automatically.
fn fan_out_roots(
    combined_data: &[u8],
    writers: &mut [&mut dyn Write],
) -> Result<RootIndexCounts, HprofError> {
    if !combined_data.len().is_multiple_of(SUB_INDEX_ENTRY_SIZE) {
        return Err(HprofError::InvalidIndexFile);
    }

    let mut counts = RootIndexCounts {
        root_unknown: 0,
        root_jni_global: 0,
        root_jni_local: 0,
        root_java_frame: 0,
        root_native_stack: 0,
        root_sticky_class: 0,
        root_thread_block: 0,
        root_monitor_used: 0,
        root_thread_obj: 0,
    };

    for arr in combined_data.as_chunks::<SUB_INDEX_ENTRY_SIZE>().0 {
        let sub = SubIndexEntry::from_bytes(arr);
        let entry = RootIndexEntry {
            object_id: sub.object_id,
            position: sub.position,
        };
        let bytes = entry.to_bytes();

        let (slot, counter): (usize, &mut u64) = match sub.tag {
            TAG_ROOT_UNKNOWN => (0, &mut counts.root_unknown),
            TAG_ROOT_JNI_GLOBAL => (1, &mut counts.root_jni_global),
            TAG_ROOT_JNI_LOCAL => (2, &mut counts.root_jni_local),
            TAG_ROOT_JAVA_FRAME => (3, &mut counts.root_java_frame),
            TAG_ROOT_NATIVE_STACK => (4, &mut counts.root_native_stack),
            TAG_ROOT_STICKY_CLASS => (5, &mut counts.root_sticky_class),
            TAG_ROOT_THREAD_BLOCK => (6, &mut counts.root_thread_block),
            TAG_ROOT_MONITOR_USED => (7, &mut counts.root_monitor_used),
            TAG_ROOT_THREAD_OBJ => (8, &mut counts.root_thread_obj),
            _ => continue,
        };
        writers[slot].write_all(&bytes)?;
        *counter += 1;
    }

    for w in writers.iter_mut() {
        w.flush()?;
    }

    Ok(counts)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heap_index::sub_record::{TAG_ROOT_JNI_GLOBAL, TAG_ROOT_STICKY_CLASS};
    use crate::index::Entry;
    use crate::index::MemStore;
    use crate::root_index::ROOT_INDEX_ENTRY_SIZE;

    /// Build a raw sub-index byte vector from `(tag, object_id)` pairs.
    fn make_sub_index_bytes(entries: &[(u8, u64)]) -> Vec<u8> {
        let mut data = Vec::with_capacity(entries.len() * SUB_INDEX_ENTRY_SIZE);
        for &(tag, id) in entries {
            let entry = SubIndexEntry {
                tag,
                object_id: id,
                position: id * 10,
            };
            data.extend_from_slice(&entry.to_bytes());
        }
        data
    }

    /// Read object IDs from a combined index in order.
    fn read_object_ids(data: &[u8]) -> Vec<u64> {
        data.as_chunks::<SUB_INDEX_ENTRY_SIZE>()
            .0
            .iter()
            .map(|arr| SubIndexEntry::from_bytes(arr).object_id)
            .collect()
    }

    fn read_root_object_ids(data: &[u8]) -> Vec<u64> {
        data.as_chunks::<ROOT_INDEX_ENTRY_SIZE>()
            .0
            .iter()
            .map(|arr| RootIndexEntry::from_bytes(arr).object_id)
            .collect()
    }

    /// Put two heap-index intermediates into a fresh `MemStore`.
    fn store_with_intermediates(a: &[(u8, u64)], b: &[(u8, u64)]) -> MemStore {
        let store = MemStore::new();
        for (name, entries) in [
            ("heap_index/00/HPROF_HEAP_DUMP_1", a),
            ("heap_index/00/HPROF_HEAP_DUMP_SEGMENT_2", b),
        ] {
            let mut w = store.create(name).unwrap();
            w.write_all(&make_sub_index_bytes(entries)).unwrap();
            w.commit().unwrap();
        }
        store
    }

    /// Run `combine_sort_and_split` in memory; returns (counts, combined, roots).
    fn run(store: &MemStore) -> (CombinedCounts, Vec<u8>, [Vec<u8>; 9]) {
        let out = MemStore::new();
        let mut combined = out.create("combined").unwrap();
        let mut roots: [Vec<u8>; 9] = std::array::from_fn(|_| Vec::new());
        let counts = {
            let mut writers: Vec<&mut dyn Write> =
                roots.iter_mut().map(|v| v as &mut dyn Write).collect();
            combine_sort_and_split(store, &mut *combined, writers.as_mut_slice()).unwrap()
        };
        combined.commit().unwrap();
        let bytes = out.open("combined").unwrap().as_ref().to_vec();
        (counts, bytes, roots)
    }

    #[test]
    fn combine_merges_and_sorts_two_intermediates() {
        let store = store_with_intermediates(
            &[
                (TAG_ROOT_STICKY_CLASS, 3),
                (TAG_ROOT_STICKY_CLASS, 1),
                (TAG_ROOT_STICKY_CLASS, 5),
            ],
            &[(TAG_ROOT_STICKY_CLASS, 4), (TAG_ROOT_STICKY_CLASS, 2)],
        );
        let (counts, combined, _) = run(&store);
        assert_eq!(counts.total, 5);
        assert_eq!(read_object_ids(&combined), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn combine_single_intermediate() {
        let store = store_with_intermediates(
            &[
                (TAG_ROOT_STICKY_CLASS, 2),
                (TAG_ROOT_STICKY_CLASS, 1),
                (TAG_ROOT_STICKY_CLASS, 3),
            ],
            &[],
        );
        let (counts, combined, _) = run(&store);
        assert_eq!(counts.total, 3);
        assert_eq!(read_object_ids(&combined), vec![1, 2, 3]);
    }

    #[test]
    fn combine_with_no_intermediates_is_empty() {
        let store = MemStore::new();
        let (counts, combined, roots) = run(&store);
        assert_eq!(counts.total, 0);
        assert!(combined.is_empty());
        assert!(roots.iter().all(|r| r.is_empty()));
    }

    #[test]
    fn combine_sort_and_split_produces_sorted_root_files() {
        let store = store_with_intermediates(
            &[(TAG_ROOT_STICKY_CLASS, 30), (TAG_ROOT_STICKY_CLASS, 10)],
            &[(TAG_ROOT_JNI_GLOBAL, 20)],
        );
        let (counts, _, roots) = run(&store);

        assert_eq!(counts.total, 3);
        assert_eq!(counts.roots.root_sticky_class, 2);
        assert_eq!(counts.roots.root_jni_global, 1);
        assert_eq!(counts.roots.root_unknown, 0);

        // sticky_class buffer (slot 5) must be sorted: [10, 30]
        assert_eq!(read_root_object_ids(&roots[5]), vec![10, 30]);
        // jni_global buffer (slot 1) has a single entry
        assert_eq!(read_root_object_ids(&roots[1]), vec![20]);
        // everything else empty
        for (i, r) in roots.iter().enumerate() {
            if i != 1 && i != 5 {
                assert!(r.is_empty(), "slot {i} should be empty");
            }
        }
    }

    #[test]
    fn wrong_number_of_root_writers_is_an_error() {
        let store = MemStore::new();
        let mut combined = MemStore::new().create("combined").unwrap();
        let mut only: Vec<u8> = Vec::new();
        let mut writers: Vec<&mut dyn Write> = vec![&mut only];
        let err = combine_sort_and_split(&store, &mut *combined, writers.as_mut_slice())
            .err()
            .expect("expected an error");
        assert!(matches!(err, HprofError::Internal(_)));
    }
}
