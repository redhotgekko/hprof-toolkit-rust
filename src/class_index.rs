//! Per-class indexes: `instances_by_class.bin` and `class_histogram.bin`.
//!
//! Both are built in one parallel pass over the object store so that
//! "instances of class X" is a binary-search range and the histogram needs no
//! scan at all at query time.
//!
//! ## `instances_by_class.bin` — 24 bytes per object
//!
//! ```text
//!  0..8   u64  class_key   (see [`ClassKey`]; primary sort key)
//!  8..16  u64  object_id   (secondary sort key)
//! 16..24  u64  position    (byte offset of the sub-record in the hprof file)
//! ```
//!
//! Sorted by `(class_key, object_id)`.  GC-root records and class dumps are
//! not included; every instance, object array and primitive array is.
//!
//! ## `class_histogram.bin` — 24 bytes per class key
//!
//! ```text
//!  0..8   u64  instance_count  (sort key, descending)
//!  8..16  u64  shallow_bytes   (instance field bytes / array element bytes; no headers)
//! 16..24  u64  class_key
//! ```
//!
//! Sorted by `(instance_count, class_key)` descending.  Only keys with at
//! least one object appear; the number of records is bounded by the number
//! of classes.
//!
//! ## Memory
//!
//! Each rayon chunk streams its instance entries into a store part and keeps
//! an `O(classes)` map of counts; the maps are merged, the parts concatenated
//! and sorted on the store.  Nothing proportional to the number of objects is
//! held in process memory.

use crate::class_key::ClassKey;
use crate::heap_index::sub_record::{
    SUB_INDEX_ENTRY_SIZE, SubIndexEntry, TAG_INSTANCE_DUMP, TAG_OBJ_ARRAY_DUMP, TAG_PRIM_ARRAY_DUMP,
};
use crate::heap_parser::{SubRecord, parse_sub_record};
use crate::hprof::{HprofError, HprofFile};
use crate::index::{Entry, IndexStore, RecordFile, RecordWriter, names, read_u64_le};
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::Write;

// ── Entries ───────────────────────────────────────────────────────────────────

/// One `instances_by_class.bin` record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InstanceByClassEntry {
    pub class_key: u64,
    pub object_id: u64,
    pub position: u64,
}

impl InstanceByClassEntry {
    /// The object-store entry this record points at (for parsing).
    pub fn sub_index_entry(&self, tag: u8) -> SubIndexEntry {
        SubIndexEntry {
            tag,
            object_id: self.object_id,
            position: self.position,
        }
    }
}

impl Entry for InstanceByClassEntry {
    const SIZE: usize = 24;
    const KEY_OFFSET: usize = 0;

    fn from_bytes(b: &[u8]) -> Self {
        Self {
            class_key: read_u64_le(b, 0),
            object_id: read_u64_le(b, 8),
            position: read_u64_le(b, 16),
        }
    }

    fn write_to(&self, out: &mut [u8]) {
        out[0..8].copy_from_slice(&self.class_key.to_le_bytes());
        out[8..16].copy_from_slice(&self.object_id.to_le_bytes());
        out[16..24].copy_from_slice(&self.position.to_le_bytes());
    }

    fn key(&self) -> u64 {
        self.class_key
    }
}

/// One `class_histogram.bin` record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HistogramRecord {
    pub instance_count: u64,
    pub shallow_bytes: u64,
    pub class_key: u64,
}

impl HistogramRecord {
    /// The record's class key, decoded.
    pub fn key(&self) -> ClassKey {
        ClassKey::from_u64(self.class_key)
    }
}

impl Entry for HistogramRecord {
    const SIZE: usize = 24;
    const KEY_OFFSET: usize = 0;

    fn from_bytes(b: &[u8]) -> Self {
        Self {
            instance_count: read_u64_le(b, 0),
            shallow_bytes: read_u64_le(b, 8),
            class_key: read_u64_le(b, 16),
        }
    }

    fn write_to(&self, out: &mut [u8]) {
        out[0..8].copy_from_slice(&self.instance_count.to_le_bytes());
        out[8..16].copy_from_slice(&self.shallow_bytes.to_le_bytes());
        out[16..24].copy_from_slice(&self.class_key.to_le_bytes());
    }

    fn key(&self) -> u64 {
        self.instance_count
    }
}

// ── Builder ───────────────────────────────────────────────────────────────────

/// Build both per-class indexes into `store`.
///
/// Returns `(objects indexed, distinct class keys)`.
pub fn build_class_indexes(
    hprof_bytes: &[u8],
    combined: &[u8],
    store: &dyn IndexStore,
) -> Result<(u64, u64), HprofError> {
    let hprof = HprofFile::from_ref(hprof_bytes)?;
    let id_size = u64::from(hprof.id_size());
    if !combined.len().is_multiple_of(SUB_INDEX_ENTRY_SIZE) {
        return Err(HprofError::InvalidIndexFile);
    }

    store.remove_prefix(&names::part_prefix(names::INSTANCES_BY_CLASS))?;

    let n_entries = combined.len() / SUB_INDEX_ENTRY_SIZE;
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_entries = (n_entries / n_threads).max(1);
    let chunks: Vec<&[u8]> = combined
        .chunks(chunk_entries * SUB_INDEX_ENTRY_SIZE)
        .collect();

    // (count, shallow_bytes) per class key.
    type Counts = HashMap<u64, (u64, u64)>;

    let results: Vec<(String, Counts)> = chunks
        .par_iter()
        .enumerate()
        .map(|(k, chunk)| -> Result<(String, Counts), HprofError> {
            let name = names::part(names::INSTANCES_BY_CLASS, k);
            let mut out = RecordWriter::<InstanceByClassEntry>::new(store.create(&name)?);
            let mut counts: Counts = HashMap::new();
            for entry in RecordFile::<SubIndexEntry>::from_slice(chunk).iter() {
                if !matches!(
                    entry.tag,
                    TAG_INSTANCE_DUMP | TAG_OBJ_ARRAY_DUMP | TAG_PRIM_ARRAY_DUMP
                ) {
                    continue;
                }
                let record = parse_sub_record(&hprof, &entry)?;
                let (key, shallow) = match &record {
                    SubRecord::InstanceDump(i) => {
                        (ClassKey::Class(i.class_id), i.data.len() as u64)
                    }
                    SubRecord::ObjArrayDump(a) => (
                        ClassKey::ObjArray(a.array_class_id),
                        u64::from(a.num_elements) * id_size,
                    ),
                    SubRecord::PrimArrayDump(a) => {
                        (ClassKey::PrimArray(a.element_type), a.data.len() as u64)
                    }
                    _ => continue,
                };
                let class_key = key.to_u64();
                out.push(&InstanceByClassEntry {
                    class_key,
                    object_id: entry.object_id,
                    position: entry.position,
                })?;
                let c = counts.entry(class_key).or_insert((0, 0));
                c.0 += 1;
                c.1 += shallow;
            }
            out.finish()?;
            Ok((name, counts))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Concatenate the parts, sort by (class_key, object_id), clean up.
    let mut out = store.create(names::INSTANCES_BY_CLASS)?;
    let mut total = 0u64;
    for (name, _) in &results {
        let part = store.open(name)?;
        out.write_all(part.as_ref())?;
        total += (part.len() / InstanceByClassEntry::SIZE) as u64;
    }
    if total > 1 {
        crate::sort::parallel_introsort_by_keys(
            out.as_mut_bytes()?,
            InstanceByClassEntry::SIZE,
            0,
            Some(8),
            false,
        );
    }
    out.commit()?;
    for (name, _) in &results {
        store.remove(name)?;
    }

    // Merge the per-chunk maps (O(classes)) and write the histogram.
    let mut merged: Counts = HashMap::new();
    for (_, counts) in results {
        for (key, (n, bytes)) in counts {
            let c = merged.entry(key).or_insert((0, 0));
            c.0 += n;
            c.1 += bytes;
        }
    }
    let mut hist = RecordWriter::<HistogramRecord>::new(store.create(names::CLASS_HISTOGRAM)?);
    for (class_key, (instance_count, shallow_bytes)) in &merged {
        hist.push(&HistogramRecord {
            instance_count: *instance_count,
            shallow_bytes: *shallow_bytes,
            class_key: *class_key,
        })?;
    }
    // Descending by count, ties broken by class key so the output is
    // byte-deterministic regardless of HashMap iteration order.
    let classes = hist.finish_sorted_by(Some(16), true)?;

    Ok((total, classes))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::MemStore;
    use crate::pipeline::{IndexOptions, build_indexes};
    use crate::progress::NoProgress;
    use crate::test_util::{standard_heap, std_ids::*};

    fn built() -> (Vec<u8>, MemStore) {
        let hprof = standard_heap();
        let store = MemStore::new();
        let opts = IndexOptions {
            retained: false,
            force: false,
            ..IndexOptions::default()
        };
        build_indexes(&hprof, &store, &opts, &NoProgress).unwrap();
        (hprof, store)
    }

    #[test]
    fn instances_by_class_is_a_sorted_range_per_key() {
        let (_, store) = built();
        let src = store.open(names::INSTANCES_BY_CLASS).unwrap();
        let file = RecordFile::<InstanceByClassEntry>::new(src.as_ref()).unwrap();
        // 3 instances + 1 char[]
        assert_eq!(file.len(), 4);
        let keys: Vec<u64> = file.iter().map(|e| e.class_key).collect();
        assert!(keys.windows(2).all(|w| w[0] <= w[1]));

        let integers: Vec<u64> = file
            .range(ClassKey::Class(INTEGER_CLASS).to_u64())
            .map(|e| e.object_id)
            .collect();
        assert_eq!(integers, vec![INTEGER_42]);
        let chars: Vec<u64> = file
            .range(ClassKey::PrimArray(5).to_u64())
            .map(|e| e.object_id)
            .collect();
        assert_eq!(chars, vec![CHARS_HI]);
        assert!(
            file.range(ClassKey::Class(OBJECT_CLASS).to_u64())
                .next()
                .is_none()
        );
        assert!(
            store
                .list(&names::part_prefix(names::INSTANCES_BY_CLASS))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn histogram_counts_and_shallow_bytes() {
        let (_, store) = built();
        let src = store.open(names::CLASS_HISTOGRAM).unwrap();
        let file = RecordFile::<HistogramRecord>::new(src.as_ref()).unwrap();
        let rows: Vec<(ClassKey, u64, u64)> = file
            .iter()
            .map(|r| (r.key(), r.instance_count, r.shallow_bytes))
            .collect();
        assert_eq!(rows.len(), 4);
        assert!(rows.iter().all(|r| r.1 == 1));
        let find = |k: ClassKey| rows.iter().find(|r| r.0 == k).map(|r| r.2);
        assert_eq!(find(ClassKey::Class(INTEGER_CLASS)), Some(4));
        assert_eq!(find(ClassKey::Class(STRING_CLASS)), Some(8));
        assert_eq!(find(ClassKey::Class(ARRAYLIST_CLASS)), Some(8));
        assert_eq!(find(ClassKey::PrimArray(5)), Some(4)); // "hi" = 2 chars
    }

    #[test]
    fn histogram_is_sorted_by_count_descending() {
        let hprof = crate::test_util::HprofBuilder::new(8)
            .utf8(1, "A")
            .utf8(2, "B")
            .load_class(1, 0x10, 1)
            .load_class(2, 0x20, 2)
            .class_dump(crate::test_util::ClassSpec::new(0x10))
            .class_dump(crate::test_util::ClassSpec::new(0x20))
            .instance(0x100, 0x10, &[])
            .instance(0x101, 0x20, &[])
            .instance(0x102, 0x20, &[])
            .instance(0x103, 0x20, &[])
            .int_array(0x200, &[1, 2])
            .int_array(0x201, &[3])
            .build();
        let store = MemStore::new();
        let opts = IndexOptions {
            retained: false,
            force: false,
            ..IndexOptions::default()
        };
        build_indexes(&hprof, &store, &opts, &NoProgress).unwrap();
        let src = store.open(names::CLASS_HISTOGRAM).unwrap();
        let file = RecordFile::<HistogramRecord>::new(src.as_ref()).unwrap();
        let rows: Vec<(ClassKey, u64, u64)> = file
            .iter()
            .map(|r| (r.key(), r.instance_count, r.shallow_bytes))
            .collect();
        assert_eq!(
            rows,
            vec![
                (ClassKey::Class(0x20), 3, 0),
                (ClassKey::PrimArray(10), 2, 12),
                (ClassKey::Class(0x10), 1, 0),
            ]
        );
    }
}
