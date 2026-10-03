//! Object reference index.
//!
//! Scans every heap object in the object store index and records every
//! object-reference field value as a `(to_object_id, from_object_id)` pair.
//! The resulting file is sorted by `to_object_id` so that all objects that
//! hold a reference to a given object can be found with O(log n) binary search.
//!
//! ## File format
//!
//! Fixed 16-byte records, little-endian:
//!
//! ```text
//! bytes 0..8   to_object_id    (the object being pointed to)
//! bytes 8..16  from_object_id  (the object holding the reference)
//! ```
//!
//! ## Parallelism and memory
//!
//! The combined index is split into `rayon::current_num_threads()` equal
//! chunks.  Each chunk is processed by a rayon task that streams its
//! `(to, from)` pairs into its own temporary store part (`refs.bin.part.k`).
//! The parts are then concatenated by streaming copy into `refs.bin`, sorted
//! in place on the store, and removed.  Nothing proportional to the number of
//! references is ever held in process memory.

use crate::class_layout::{ClassCache, build_class_cache, for_each_instance_ref};
use crate::heap_index::sub_record::{
    SUB_INDEX_ENTRY_SIZE, SubIndexEntry, TAG_CLASS_DUMP, TAG_INSTANCE_DUMP, TAG_OBJ_ARRAY_DUMP,
};
use crate::heap_parser::record::FieldValue;
use crate::heap_parser::{SubIndexReader, SubRecord, parse_sub_record};
use crate::hprof::{HprofError, HprofFile};
use crate::index::{Entry, IndexStore, RecordFile, RecordIter, RecordWriter, names, read_u64_le};
use rayon::prelude::*;
use std::io::Write;

/// Byte size of one reference index record.
pub const REF_ENTRY_SIZE: usize = 16;

// ── Entry ─────────────────────────────────────────────────────────────────────

/// One `(to, from)` reference record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefEntry {
    /// The object being pointed to (sort key).
    pub to_object_id: u64,
    /// The object holding the reference.
    pub from_object_id: u64,
}

impl Entry for RefEntry {
    const SIZE: usize = REF_ENTRY_SIZE;
    const KEY_OFFSET: usize = 0;

    fn from_bytes(b: &[u8]) -> Self {
        Self {
            to_object_id: read_u64_le(b, 0),
            from_object_id: read_u64_le(b, 8),
        }
    }

    fn write_to(&self, out: &mut [u8]) {
        out[0..8].copy_from_slice(&self.to_object_id.to_le_bytes());
        out[8..16].copy_from_slice(&self.from_object_id.to_le_bytes());
    }

    fn key(&self) -> u64 {
        self.to_object_id
    }
}

// ── Public builder ────────────────────────────────────────────────────────────

/// Scan all heap objects and build the reference index
/// ([`names::REFS`]) in `store`.
///
/// For every non-null object-reference field in every `INSTANCE_DUMP`,
/// `CLASS_DUMP` (static fields), and `OBJ_ARRAY_DUMP`, one 16-byte record
/// `(to_object_id, from_object_id)` is written.
///
/// The result is sorted ascending by `to_object_id` so that
/// [`RefIndex::find`] can perform O(log n) binary search.
///
/// Returns the total number of reference records written.
pub fn build_reference_index(
    hprof_source: &[u8],
    combined_mmap: &[u8],
    store: &dyn IndexStore,
) -> Result<u64, HprofError> {
    let hprof = HprofFile::from_ref(hprof_source)?;
    // Field layouts of every class, read once: O(classes) memory.
    let class_cache = build_class_cache(&hprof, &SubIndexReader::from_ref(combined_mmap)?)?;

    if !combined_mmap.len().is_multiple_of(SUB_INDEX_ENTRY_SIZE) {
        return Err(HprofError::InvalidIndexFile);
    }

    // Leftover parts from an interrupted run are never valid.
    store.remove_prefix(&names::part_prefix(names::REFS))?;

    // Split the combined index into one chunk per rayon thread.
    let n_entries = combined_mmap.len() / SUB_INDEX_ENTRY_SIZE;
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_entries = (n_entries / n_threads).max(1);
    let chunks: Vec<&[u8]> = combined_mmap
        .chunks(chunk_entries * SUB_INDEX_ENTRY_SIZE)
        .collect();

    // Each task streams its pairs into its own store part.
    let part_names: Vec<String> = chunks
        .par_iter()
        .enumerate()
        .map(|(k, chunk)| -> Result<String, HprofError> {
            let name = names::part(names::REFS, k);
            let mut out = RecordWriter::<RefEntry>::new(store.create(&name)?);
            for entry in RecordFile::<SubIndexEntry>::from_slice(chunk).iter() {
                write_refs_for_entry(&hprof, &class_cache, &entry, &mut out)?;
            }
            out.finish()?;
            Ok(name)
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Concatenate the parts by streaming copy, sort in place, commit, clean up.
    let mut out = store.create(names::REFS)?;
    let mut total = 0u64;
    for name in &part_names {
        let part = store.open(name)?;
        let bytes = part.as_ref();
        out.write_all(bytes)?;
        total += (bytes.len() / REF_ENTRY_SIZE) as u64;
    }
    if total > 1 {
        crate::sort::parallel_introsort(out.as_mut_bytes()?, REF_ENTRY_SIZE, 0);
    }
    out.commit()?;
    for name in &part_names {
        store.remove(name)?;
    }

    Ok(total)
}

// ── Public reader ─────────────────────────────────────────────────────────────

/// Read-only handle to a sorted reference index file.
#[derive(Clone, Copy)]
pub struct RefIndex<'a> {
    file: RecordFile<'a, RefEntry>,
}

impl<'a> RefIndex<'a> {
    /// Create a validated reader from a byte slice.
    pub fn from_ref(data: &'a [u8]) -> Result<Self, HprofError> {
        Ok(Self {
            file: RecordFile::new(data)?,
        })
    }

    /// Create a reader from a slice already known to be valid.
    pub(crate) fn from_slice(data: &'a [u8]) -> Self {
        Self {
            file: RecordFile::from_slice(data),
        }
    }

    /// Every record whose `to_object_id == to_id`, uncapped, in file order.
    pub fn referrers(&self, to_id: u64) -> RecordIter<'a, RefEntry> {
        self.file.range(to_id)
    }

    /// Total number of reference records in this index.
    pub fn len(&self) -> usize {
        self.file.len()
    }
}

// ── Per-entry reference extraction ───────────────────────────────────────────

fn write_refs_for_entry(
    hprof: &HprofFile,
    class_cache: &ClassCache,
    entry: &SubIndexEntry,
    out: &mut RecordWriter<RefEntry>,
) -> Result<(), HprofError> {
    let from_object_id = entry.object_id;
    let mut push = |to_object_id: u64| {
        out.push(&RefEntry {
            to_object_id,
            from_object_id,
        })
    };

    match entry.tag {
        TAG_INSTANCE_DUMP => {
            if let SubRecord::InstanceDump(inst) = parse_sub_record(hprof, entry)? {
                let id_size = hprof.id_size() as usize;
                for_each_instance_ref(&inst, class_cache, id_size, &mut push)?;
            }
        }
        TAG_CLASS_DUMP => {
            if let SubRecord::ClassDump(cd) = parse_sub_record(hprof, entry)? {
                for sf in cd.static_fields() {
                    let sf = sf?;
                    if let FieldValue::Object(to_id) = sf.value
                        && to_id != 0
                    {
                        push(to_id)?;
                    }
                }
            }
        }
        TAG_OBJ_ARRAY_DUMP => {
            if let SubRecord::ObjArrayDump(arr) = parse_sub_record(hprof, entry)? {
                for to_id in arr.elements() {
                    if to_id != 0 {
                        push(to_id)?;
                    }
                }
            }
        }
        _ => {}
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::MemStore;
    use crate::pipeline::{IndexOptions, build_indexes};
    use crate::progress::NoProgress;
    use crate::test_util::{standard_heap, std_ids::*};

    fn make_ref_bytes(pairs: &[(u64, u64)]) -> Vec<u8> {
        let mut data = Vec::with_capacity(pairs.len() * REF_ENTRY_SIZE);
        for &(to_id, from_id) in pairs {
            data.extend_from_slice(&to_id.to_le_bytes());
            data.extend_from_slice(&from_id.to_le_bytes());
        }
        data
    }

    #[test]
    fn ref_index_referrers_returns_matching_from_ids() {
        // Sorted pairs: (to=1,from=10), (to=2,from=20), (to=2,from=30), (to=3,from=40)
        let data = make_ref_bytes(&[(1, 10), (2, 20), (2, 30), (3, 40)]);
        let idx = RefIndex::from_ref(&data).unwrap();

        let from = |to| {
            idx.referrers(to)
                .map(|e| e.from_object_id)
                .collect::<Vec<_>>()
        };
        assert_eq!(from(2), vec![20, 30]);
        assert_eq!(from(1), vec![10]);
        assert_eq!(from(3), vec![40]);
        assert!(from(99).is_empty());
        assert_eq!(idx.referrers(2).count(), 2);
        assert_eq!(idx.len(), 4);
    }

    #[test]
    fn referrers_are_not_capped() {
        let pairs: Vec<(u64, u64)> = (0..500).map(|i| (7, i)).collect();
        let data = make_ref_bytes(&pairs);
        let idx = RefIndex::from_ref(&data).unwrap();
        assert_eq!(idx.referrers(7).count(), 500);
    }

    #[test]
    fn ref_index_find_empty_file() {
        let empty: Vec<u8> = vec![];
        let idx = RefIndex::from_ref(&empty).unwrap();
        assert_eq!(idx.referrers(1).count(), 0);
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn ref_entry_round_trips() {
        let mut buf = [0u8; REF_ENTRY_SIZE];
        RefEntry {
            to_object_id: 0xAABBCCDD_00112233,
            from_object_id: 0x11223344_AABBCCDD,
        }
        .write_to(&mut buf);
        let e = RefEntry::from_bytes(&buf);
        assert_eq!(e.to_object_id, 0xAABBCCDD_00112233);
        assert_eq!(e.from_object_id, 0x11223344_AABBCCDD);
        assert_eq!(e.key(), e.to_object_id);
    }

    #[test]
    fn builder_streams_parts_and_leaves_only_the_final_entry() {
        let hprof = standard_heap();
        let store = MemStore::new();
        // Build the prerequisites without the reference index itself.
        let opts = IndexOptions {
            retained: false,
            force: false,
            ..IndexOptions::default()
        };
        build_indexes(&hprof, &store, &opts, &NoProgress).unwrap();
        store.remove(names::REFS).unwrap();
        // A stale part from an "interrupted" run must be cleaned up.
        store
            .create(&names::part(names::REFS, 7))
            .unwrap()
            .commit()
            .unwrap();

        let combined = store.open(names::OBJECT_STORE).unwrap();
        let n = build_reference_index(&hprof, combined.as_ref(), &store).unwrap();

        // LIST → INTEGER_42 and STRING_HI → CHARS_HI.
        assert_eq!(n, 2);
        assert!(
            store
                .list(&names::part_prefix(names::REFS))
                .unwrap()
                .is_empty()
        );
        let refs = store.open(names::REFS).unwrap();
        let idx = RefIndex::from_ref(refs.as_ref()).unwrap();
        assert_eq!(
            idx.referrers(INTEGER_42)
                .map(|e| e.from_object_id)
                .collect::<Vec<_>>(),
            vec![LIST]
        );
        assert_eq!(
            idx.referrers(CHARS_HI)
                .map(|e| e.from_object_id)
                .collect::<Vec<_>>(),
            vec![STRING_HI]
        );
        // Sorted by to_object_id.
        let keys: Vec<u64> = idx.referrers(INTEGER_42).map(|e| e.to_object_id).collect();
        assert_eq!(keys, vec![INTEGER_42]);
    }
}
