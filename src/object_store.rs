use crate::heap_index::sub_record::{
    SUB_INDEX_ENTRY_SIZE, SubIndexEntry, TAG_ROOT_JAVA_FRAME, TAG_ROOT_JNI_GLOBAL,
    TAG_ROOT_JNI_LOCAL, TAG_ROOT_MONITOR_USED, TAG_ROOT_NATIVE_STACK, TAG_ROOT_STICKY_CLASS,
    TAG_ROOT_THREAD_BLOCK, TAG_ROOT_THREAD_OBJ, TAG_ROOT_UNKNOWN,
};
use crate::hprof::HprofError;
use crate::root_index::{RootIndexCounts, RootIndexEntry};
use crate::vfs::{MMapReader, MMapWriter, SubIndexDir};
use std::io::Write;

/// Combines all heap index sub-index files in `dir` into a single buffer
/// at `combined`, then sorts it in-place by `object_id`.
///
/// The resulting data has the same fixed-size record format as the heap index
/// sub-index files ([`SubIndexEntry`], 24 bytes each, little-endian) and is
/// sorted ascending by `object_id`, enabling O(log n) binary search via
/// [`find_by_object_id`].
///
/// Returns the total number of entries written.
pub fn combine_and_sort_sub_index(
    dir: &SubIndexDir,
    combined: &mut impl MMapWriter,
) -> Result<u64, HprofError> {
    let all_bytes = dir.all_file_bytes()?;

    // Step 1: concatenate all sub-index byte slices into combined.
    let total = concatenate_byte_slices(&all_bytes, combined)?;

    // Step 2: sort combined in-place by object_id.
    if total > 1 {
        let mut map = combined.create_mut_mmap()?;
        let data = map.as_mut();
        crate::sort::parallel_introsort(data, SUB_INDEX_ENTRY_SIZE, 8);
    }

    Ok(total)
}

/// Counts returned by [`combine_sort_and_split`].
pub struct CombinedCounts {
    /// Total entries written to the combined object store.
    pub total: u64,
    /// Per-type counts for the nine GC root index files.
    pub roots: RootIndexCounts,
}

/// Combines, sorts, and splits heap sub-index data in one logical step.
///
/// 1. Stream-concatenates all heap sub-record index files in `dir`
///    into `combined`.
/// 2. Sorts `combined` in-place by `object_id`.
/// 3. Makes a single sequential pass over the sorted data, routing each
///    GC-root entry to the appropriate per-type root index file writer.
///
/// `root_writers` must be an array of exactly 9 writers, one per GC root type
/// in the canonical order defined by [`crate::root_index::GcRootType::ALL`]:
/// unknown, jni_global, jni_local, java_frame, native_stack, sticky_class,
/// thread_block, monitor_used, thread_obj.
///
/// Each root writer receives sorted entries (sorted because `combined` is
/// already sorted by `object_id`).
pub fn combine_sort_and_split(
    dir: &SubIndexDir,
    combined: &mut impl MMapWriter,
    root_writers: &mut [&mut dyn Write; 9],
) -> Result<CombinedCounts, HprofError> {
    let all_bytes = dir.all_file_bytes()?;
    let total = concatenate_byte_slices(&all_bytes, combined)?;
    if total > 1 {
        let mut map = combined.create_mut_mmap()?;
        let data = map.as_mut();
        crate::sort::parallel_introsort(data, SUB_INDEX_ENTRY_SIZE, 8);
    }
    let mut combined_source = combined.create_mut_mmap()?;
    let roots = fan_out_roots(combined_source.as_mut(), root_writers)?;
    Ok(CombinedCounts { total, roots })
}

/// Binary-searches the sorted combined index for the first entry whose
/// `object_id` equals `target`.
///
/// Returns `Ok(Some(entry))` if found, `Ok(None)` if not present.
/// The combined data must have been produced by [`combine_and_sort_sub_index`].
pub fn find_by_object_id(
    combined: &impl MMapReader,
    target: u64,
) -> Result<Option<SubIndexEntry>, HprofError> {
    let source = combined.open_mmap()?;
    let data: &[u8] = source.as_ref();
    if !data.len().is_multiple_of(SUB_INDEX_ENTRY_SIZE) {
        return Err(HprofError::InvalidIndexFile);
    }
    let n = data.len() / SUB_INDEX_ENTRY_SIZE;

    // Leftmost binary search: find first index where object_id >= target.
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if read_object_id_at(data, mid) < target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    if lo < n && read_object_id_at(data, lo) == target {
        let start = lo * SUB_INDEX_ENTRY_SIZE;
        let bytes: [u8; SUB_INDEX_ENTRY_SIZE] = data[start..start + SUB_INDEX_ENTRY_SIZE]
            .try_into()
            .map_err(|_| HprofError::InvalidIndexFile)?;
        Ok(Some(SubIndexEntry::from_bytes(&bytes)))
    } else {
        Ok(None)
    }
}

// ── Internals ─────────────────────────────────────────────────────────────────

/// Concatenate multiple byte slices (each a sub-index file) into `combined`.
///
/// Only complete 24-byte records are written; any partial trailing bytes in
/// a source slice are silently skipped (they should not occur with valid data).
fn concatenate_byte_slices(
    slices: &[Vec<u8>],
    combined: &mut impl MMapWriter,
) -> Result<u64, HprofError> {
    let mut writer = combined.create_writer()?;
    let mut total = 0u64;

    for slice in slices {
        let aligned = (slice.len() / SUB_INDEX_ENTRY_SIZE) * SUB_INDEX_ENTRY_SIZE;
        writer.write_all(&slice[..aligned])?;
        total += (aligned / SUB_INDEX_ENTRY_SIZE) as u64;
    }
    writer.flush()?;
    Ok(total)
}

/// Read the `object_id` field (bytes 8..16, little-endian) of entry `idx`
/// from the flat byte slice.
fn read_object_id_at(data: &[u8], idx: usize) -> u64 {
    let start = idx * SUB_INDEX_ENTRY_SIZE + 8;
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&data[start..start + 8]);
    u64::from_le_bytes(bytes)
}

/// Read the sorted combined data and route GC-root entries to nine per-type
/// output writers.  Non-root entries (class dumps, instance dumps, array dumps)
/// are skipped.  Because `combined_data` is already sorted by `object_id`
/// each output writer receives sorted entries automatically.
///
/// `writers` must have exactly 9 elements in the canonical GC root order:
/// [unknown, jni_global, jni_local, java_frame, native_stack, sticky_class,
///  thread_block, monitor_used, thread_obj].
fn fan_out_roots(
    combined_data: &[u8],
    writers: &mut [&mut dyn Write; 9],
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

    for chunk in combined_data.chunks_exact(SUB_INDEX_ENTRY_SIZE) {
        let arr: [u8; SUB_INDEX_ENTRY_SIZE] =
            chunk.try_into().map_err(|_| HprofError::InvalidIndexFile)?;
        let sub = SubIndexEntry::from_bytes(&arr);
        let entry = RootIndexEntry {
            object_id: sub.object_id,
            position: sub.position,
        };
        let bytes = entry.to_bytes();

        #[allow(clippy::arithmetic_side_effects)]
        match sub.tag {
            TAG_ROOT_UNKNOWN => {
                writers[0].write_all(&bytes)?;
                counts.root_unknown += 1;
            }
            TAG_ROOT_JNI_GLOBAL => {
                writers[1].write_all(&bytes)?;
                counts.root_jni_global += 1;
            }
            TAG_ROOT_JNI_LOCAL => {
                writers[2].write_all(&bytes)?;
                counts.root_jni_local += 1;
            }
            TAG_ROOT_JAVA_FRAME => {
                writers[3].write_all(&bytes)?;
                counts.root_java_frame += 1;
            }
            TAG_ROOT_NATIVE_STACK => {
                writers[4].write_all(&bytes)?;
                counts.root_native_stack += 1;
            }
            TAG_ROOT_STICKY_CLASS => {
                writers[5].write_all(&bytes)?;
                counts.root_sticky_class += 1;
            }
            TAG_ROOT_THREAD_BLOCK => {
                writers[6].write_all(&bytes)?;
                counts.root_thread_block += 1;
            }
            TAG_ROOT_MONITOR_USED => {
                writers[7].write_all(&bytes)?;
                counts.root_monitor_used += 1;
            }
            TAG_ROOT_THREAD_OBJ => {
                writers[8].write_all(&bytes)?;
                counts.root_thread_obj += 1;
            }
            _ => {}
        }
    }

    for w in writers.iter_mut() {
        w.flush()?;
    }

    Ok(counts)
}

/// Thin wrapper so existing unit tests can call `heapsort` by its old name.
#[cfg(test)]
fn heapsort(data: &mut [u8], n: usize) {
    crate::sort::heapsort_for_tests(data, n, SUB_INDEX_ENTRY_SIZE, 8);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heap_index::sub_record::TAG_ROOT_STICKY_CLASS;
    use crate::vfs::SubIndexDir;

    /// Build a raw sub-index Vec from a slice of object IDs.
    fn make_sub_index_bytes(ids: &[u64]) -> Vec<u8> {
        let mut data = Vec::with_capacity(ids.len() * SUB_INDEX_ENTRY_SIZE);
        for &id in ids {
            let entry = SubIndexEntry {
                tag: TAG_ROOT_STICKY_CLASS,
                object_id: id,
                position: id,
            };
            data.extend_from_slice(&entry.to_bytes());
        }
        data
    }

    /// Read object IDs from a combined index Vec in order.
    fn read_object_ids(data: &[u8]) -> Vec<u64> {
        data.chunks_exact(SUB_INDEX_ENTRY_SIZE)
            .map(|chunk| {
                let arr: [u8; SUB_INDEX_ENTRY_SIZE] = chunk.try_into().unwrap();
                SubIndexEntry::from_bytes(&arr).object_id
            })
            .collect()
    }

    /// Build a fake heap index output `SubIndexDir::mem()` with two sub-index files.
    fn make_heap_index_dir(file_a_ids: &[u64], file_b_ids: &[u64]) -> SubIndexDir {
        let dir = SubIndexDir::mem();
        dir.write_sub_file("00/HPROF_HEAP_DUMP_1", make_sub_index_bytes(file_a_ids))
            .unwrap();
        dir.write_sub_file(
            "00/HPROF_HEAP_DUMP_SEGMENT_2",
            make_sub_index_bytes(file_b_ids),
        )
        .unwrap();
        dir
    }

    #[test]
    fn heapsort_sorts_entries() {
        let ids = [5u64, 3, 8, 1, 9, 2, 7, 4, 6];
        let mut data = Vec::new();
        for &id in &ids {
            let entry = SubIndexEntry {
                tag: 0,
                object_id: id,
                position: id * 10,
            };
            data.extend_from_slice(&entry.to_bytes());
        }
        heapsort(&mut data, ids.len());
        let sorted: Vec<u64> = data
            .chunks_exact(SUB_INDEX_ENTRY_SIZE)
            .map(|c| {
                let arr: [u8; SUB_INDEX_ENTRY_SIZE] = c.try_into().unwrap();
                SubIndexEntry::from_bytes(&arr).object_id
            })
            .collect();
        assert_eq!(sorted, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn heapsort_preserves_fields() {
        let mut data = Vec::new();
        let entries = [
            SubIndexEntry {
                tag: 0x21,
                object_id: 42,
                position: 1000,
            },
            SubIndexEntry {
                tag: 0x20,
                object_id: 7,
                position: 2000,
            },
        ];
        for e in &entries {
            data.extend_from_slice(&e.to_bytes());
        }
        heapsort(&mut data, 2);
        let arr: [u8; SUB_INDEX_ENTRY_SIZE] = data[..SUB_INDEX_ENTRY_SIZE].try_into().unwrap();
        let first = SubIndexEntry::from_bytes(&arr);
        assert_eq!(first.object_id, 7);
        assert_eq!(first.tag, 0x20);
        assert_eq!(first.position, 2000);
    }

    #[test]
    fn combine_and_sort_merges_two_files() {
        let heap_index_dir = make_heap_index_dir(&[3, 1, 5], &[4, 2]);

        let mut combined = Vec::new();
        let count = combine_and_sort_sub_index(&heap_index_dir, &mut combined).unwrap();
        assert_eq!(count, 5);

        let ids = read_object_ids(&combined);
        assert_eq!(ids, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn combine_single_file() {
        let heap_index_dir = make_heap_index_dir(&[2, 1, 3], &[]);

        let mut combined = Vec::new();
        let count = combine_and_sort_sub_index(&heap_index_dir, &mut combined).unwrap();
        assert_eq!(count, 3);

        let ids = read_object_ids(&combined);
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn find_by_object_id_returns_entry() {
        let heap_index_dir = make_heap_index_dir(&[10, 20, 30], &[15, 25]);
        let mut combined = Vec::new();
        combine_and_sort_sub_index(&heap_index_dir, &mut combined).unwrap();

        let entry = find_by_object_id(&combined, 20).unwrap();
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().object_id, 20);
    }

    #[test]
    fn find_by_object_id_not_found() {
        let heap_index_dir = make_heap_index_dir(&[10, 20, 30], &[]);
        let mut combined = Vec::new();
        combine_and_sort_sub_index(&heap_index_dir, &mut combined).unwrap();

        let entry = find_by_object_id(&combined, 99).unwrap();
        assert!(entry.is_none());
    }

    #[test]
    fn find_by_object_id_first_occurrence() {
        let heap_index_dir = make_heap_index_dir(&[5, 10], &[5, 20]);
        let mut combined = Vec::new();
        combine_and_sort_sub_index(&heap_index_dir, &mut combined).unwrap();

        let entry = find_by_object_id(&combined, 5).unwrap();
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().object_id, 5);
    }

    // ── combine_sort_and_split tests ──────────────────────────────────────────

    fn read_root_object_ids(data: &[u8]) -> Vec<u64> {
        use crate::root_index::{ROOT_INDEX_ENTRY_SIZE, RootIndexEntry};
        data.chunks_exact(ROOT_INDEX_ENTRY_SIZE)
            .map(|c| {
                let arr: [u8; ROOT_INDEX_ENTRY_SIZE] = c.try_into().unwrap();
                RootIndexEntry::from_bytes(&arr).object_id
            })
            .collect()
    }

    #[test]
    fn combine_sort_and_split_produces_sorted_root_files() {
        use crate::heap_index::sub_record::TAG_ROOT_JNI_GLOBAL;

        let heap_index_dir = SubIndexDir::mem();

        // file_a: sticky_class ids 30, 10; file_b: jni_global id 20
        let mut buf_a = Vec::new();
        for &id in &[30u64, 10] {
            buf_a.extend_from_slice(
                &SubIndexEntry {
                    tag: TAG_ROOT_STICKY_CLASS,
                    object_id: id,
                    position: id * 10,
                }
                .to_bytes(),
            );
        }
        let buf_b = SubIndexEntry {
            tag: TAG_ROOT_JNI_GLOBAL,
            object_id: 20,
            position: 200,
        }
        .to_bytes();
        heap_index_dir
            .write_sub_file("00/HPROF_HEAP_DUMP_1", buf_a)
            .unwrap();
        heap_index_dir
            .write_sub_file("00/HPROF_HEAP_DUMP_SEGMENT_2", buf_b.to_vec())
            .unwrap();

        let mut r_unknown = Vec::new();
        let mut r_jni_global = Vec::new();
        let mut r_jni_local = Vec::new();
        let mut r_java_frame = Vec::new();
        let mut r_native_stack = Vec::new();
        let mut r_sticky_class = Vec::new();
        let mut r_thread_block = Vec::new();
        let mut r_monitor_used = Vec::new();
        let mut r_thread_obj = Vec::new();

        let mut combined = Vec::new();
        let counts = combine_sort_and_split(
            &heap_index_dir,
            &mut combined,
            &mut [
                &mut r_unknown,
                &mut r_jni_global,
                &mut r_jni_local,
                &mut r_java_frame,
                &mut r_native_stack,
                &mut r_sticky_class,
                &mut r_thread_block,
                &mut r_monitor_used,
                &mut r_thread_obj,
            ],
        )
        .unwrap();

        assert_eq!(counts.total, 3);
        assert_eq!(counts.roots.root_sticky_class, 2);
        assert_eq!(counts.roots.root_jni_global, 1);
        assert_eq!(counts.roots.root_unknown, 0);

        // sticky_class buffer must be sorted: [10, 30]
        let sticky_ids = read_root_object_ids(&r_sticky_class);
        assert_eq!(sticky_ids, vec![10, 30]);

        // jni_global buffer has a single entry
        let jni_ids = read_root_object_ids(&r_jni_global);
        assert_eq!(jni_ids, vec![20]);
    }
}
