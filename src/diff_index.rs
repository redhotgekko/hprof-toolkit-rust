//! Binary diff index files for heap dump comparisons.
//!
//! [`build_diff_indexes`] performs a single O(n + m) merge-walk over two
//! sorted combined sub-record indexes and writes three entries into an
//! [`IndexStore`] (a directory on disk, or memory in tests):
//!
//! * `removed.bin` — objects present only in dump 1 (garbage-collected)
//! * `added.bin`   — objects present only in dump 2 (newly allocated)
//! * `common.bin`  — objects present in both dumps (with a `changed` flag)
//!
//! All files use **fixed-size records sorted by `object_id`** (ascending),
//! enabling O(log n) binary search and chunked concurrent processing.
//!
//! ## Record layouts
//!
//! **`removed.bin` / `added.bin`** — 24 bytes per record:
//! ```text
//!  0.. 1  u8        tag     (heap sub-record type, e.g. 0x21 = INSTANCE_DUMP)
//!  1.. 8  [u8; 7]   padding (zeros)
//!  8..16  u64 LE    object_id
//! 16..24  u64 LE    position (byte offset of sub-record in the respective hprof)
//! ```
//!
//! **`common.bin`** — 32 bytes per record:
//! ```text
//!  0.. 1  u8        tag
//!  1.. 2  u8        changed  (0 = raw bytes identical, 1 = bytes differ)
//!  2.. 8  [u8; 6]   padding (zeros)
//!  8..16  u64 LE    object_id
//! 16..24  u64 LE    position1 (byte offset in hprof1)
//! 24..32  u64 LE    position2 (byte offset in hprof2)
//! ```

use crate::heap_index::sub_record::SubIndexEntry;
use crate::heap_parser::{SubRecord, is_object_tag};
use crate::hprof::HprofError;
use crate::index::{Entry, IndexStore, RecordWriter, read_u64_le};
use crate::query::HeapQuery;
use std::path::{Path, PathBuf};

/// Entry name of the objects only in the first dump.
pub const REMOVED: &str = "removed.bin";
/// Entry name of the objects only in the second dump.
pub const ADDED: &str = "added.bin";
/// Entry name of the objects in both dumps.
pub const COMMON: &str = "common.bin";

// ── Record sizes ──────────────────────────────────────────────────────────────

/// Byte size of a serialized [`DiffEntry`] (removed / added record).
pub const DIFF_ENTRY_SIZE: usize = 24;
/// Byte size of a serialized [`CommonEntry`] (common record).
pub const COMMON_ENTRY_SIZE: usize = 32;

// ── DiffEntry (removed / added) ───────────────────────────────────────────────

/// A single record in `removed.bin` or `added.bin`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiffEntry {
    /// Heap sub-record type (e.g. [`TAG_INSTANCE_DUMP`]).
    pub tag: u8,
    /// Object identifier.
    pub object_id: u64,
    /// Byte offset of the sub-record in its hprof file.
    pub position: u64,
}

impl DiffEntry {
    pub fn to_bytes(self) -> [u8; DIFF_ENTRY_SIZE] {
        let mut buf = [0u8; DIFF_ENTRY_SIZE];
        buf[0] = self.tag;
        // buf[1..8] = 0 (padding)
        buf[8..16].copy_from_slice(&self.object_id.to_le_bytes());
        buf[16..24].copy_from_slice(&self.position.to_le_bytes());
        buf
    }
}

// ── CommonEntry ───────────────────────────────────────────────────────────────

/// A single record in `common.bin`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommonEntry {
    /// Heap sub-record type.
    pub tag: u8,
    /// `true` if the raw data bytes of the two records differ.
    pub changed: bool,
    /// Object identifier.
    pub object_id: u64,
    /// Byte offset of the sub-record in hprof 1.
    pub position1: u64,
    /// Byte offset of the sub-record in hprof 2.
    pub position2: u64,
}

impl CommonEntry {
    pub fn to_bytes(self) -> [u8; COMMON_ENTRY_SIZE] {
        let mut buf = [0u8; COMMON_ENTRY_SIZE];
        buf[0] = self.tag;
        buf[1] = self.changed as u8;
        // buf[2..8] = 0 (padding)
        buf[8..16].copy_from_slice(&self.object_id.to_le_bytes());
        buf[16..24].copy_from_slice(&self.position1.to_le_bytes());
        buf[24..32].copy_from_slice(&self.position2.to_le_bytes());
        buf
    }
}

// ── Location on disk ──────────────────────────────────────────────────────────

/// Directory that holds the diff entries of `hprof1` against `hprof2`:
/// `{stem1}_vs_{stem2}.diff_indexes`, next to `hprof1`.
pub fn diff_dir_for_hprofs(hprof1: &Path, hprof2: &Path) -> PathBuf {
    let stem = |p: &Path, fallback: &str| {
        p.file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| fallback.to_string())
    };
    let parent = hprof1.parent().unwrap_or(Path::new("."));
    parent.join(format!(
        "{}_vs_{}.diff_indexes",
        stem(hprof1, "dump1"),
        stem(hprof2, "dump2")
    ))
}

// ── Build counts ─────────────────────────────────────────────────────────────

/// Record counts returned by [`build_diff_indexes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DiffIndexCounts {
    pub removed: u64,
    pub added: u64,
    pub common: u64,
    pub common_changed: u64,
}

// ── Builder ───────────────────────────────────────────────────────────────────

/// Build the three diff entries in `store`, unless all three already exist.
///
/// Performs a single O(n + m) merge-walk of the two sorted combined indexes.
/// For each common object, the raw data bytes of both records are compared to
/// set the `changed` flag.  Output is streamed; memory use is constant.
///
/// Returns the counts of written entries, or zero counts when the entries
/// were already present.
pub fn build_diff_indexes(
    query1: &HeapQuery,
    query2: &HeapQuery,
    store: &dyn IndexStore,
) -> Result<DiffIndexCounts, HprofError> {
    if [REMOVED, ADDED, COMMON].iter().all(|n| store.exists(n)) {
        return Ok(DiffIndexCounts::default());
    }

    let mut w_removed = RecordWriter::<DiffEntry>::new(store.create(REMOVED)?);
    let mut w_added = RecordWriter::<DiffEntry>::new(store.create(ADDED)?);
    let mut w_common = RecordWriter::<CommonEntry>::new(store.create(COMMON)?);
    let mut counts = DiffIndexCounts::default();

    let mut iter1 = query1.iter_entries().filter(|e| is_object_tag(e.tag));
    let mut iter2 = query2.iter_entries().filter(|e| is_object_tag(e.tag));
    let mut cur1: Option<SubIndexEntry> = iter1.next();
    let mut cur2: Option<SubIndexEntry> = iter2.next();

    let as_diff = |e: &SubIndexEntry| DiffEntry {
        tag: e.tag,
        object_id: e.object_id,
        position: e.position,
    };

    loop {
        let order = match (&cur1, &cur2) {
            (None, None) => break,
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (Some(a), Some(b)) => a.object_id.cmp(&b.object_id),
        };
        match (order, cur1, cur2) {
            (std::cmp::Ordering::Less, Some(e1), _) => {
                w_removed.push(&as_diff(&e1))?;
                counts.removed += 1;
                cur1 = iter1.next();
            }
            (std::cmp::Ordering::Greater, _, Some(e2)) => {
                w_added.push(&as_diff(&e2))?;
                counts.added += 1;
                cur2 = iter2.next();
            }
            (_, Some(e1), Some(e2)) => {
                let changed = records_changed(query1, &e1, query2, &e2)?;
                w_common.push(&CommonEntry {
                    tag: e1.tag,
                    changed,
                    object_id: e1.object_id,
                    position1: e1.position,
                    position2: e2.position,
                })?;
                counts.common += 1;
                counts.common_changed += u64::from(changed);
                cur1 = iter1.next();
                cur2 = iter2.next();
            }
            _ => break,
        }
    }

    w_removed.finish()?;
    w_added.finish()?;
    w_common.finish()?;
    Ok(counts)
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Compare the data bytes of two records at the same object_id.
///
/// Returns `true` when the records differ in a meaningful way:
/// * `InstanceDump`   — field data bytes (`inst.data`)
/// * `ClassDump`      — full raw sub-record bytes (`cd.raw_bytes()`)
/// * `ObjArrayDump`   — element bytes (`arr.elements_raw()`)
/// * `PrimArrayDump`  — element bytes (`arr.data`)
fn records_changed(
    query1: &HeapQuery,
    entry1: &SubIndexEntry,
    query2: &HeapQuery,
    entry2: &SubIndexEntry,
) -> Result<bool, HprofError> {
    let r1 = query1.parse_entry(entry1)?;
    let r2 = query2.parse_entry(entry2)?;
    Ok(match (&r1, &r2) {
        (SubRecord::InstanceDump(i1), SubRecord::InstanceDump(i2)) => i1.data != i2.data,
        (SubRecord::ClassDump(c1), SubRecord::ClassDump(c2)) => c1.raw_bytes() != c2.raw_bytes(),
        (SubRecord::ObjArrayDump(a1), SubRecord::ObjArrayDump(a2)) => {
            a1.elements_raw() != a2.elements_raw()
        }
        (SubRecord::PrimArrayDump(a1), SubRecord::PrimArrayDump(a2)) => a1.data != a2.data,
        _ => true,
    })
}

// ── Entry impls ───────────────────────────────────────────────────────────────

impl Entry for DiffEntry {
    const SIZE: usize = DIFF_ENTRY_SIZE;
    const KEY_OFFSET: usize = 8;

    fn from_bytes(b: &[u8]) -> Self {
        Self {
            tag: b[0],
            object_id: read_u64_le(b, 8),
            position: read_u64_le(b, 16),
        }
    }

    fn write_to(&self, out: &mut [u8]) {
        out.copy_from_slice(&self.to_bytes());
    }

    fn key(&self) -> u64 {
        self.object_id
    }
}

impl Entry for CommonEntry {
    const SIZE: usize = COMMON_ENTRY_SIZE;
    const KEY_OFFSET: usize = 8;

    fn from_bytes(b: &[u8]) -> Self {
        Self {
            tag: b[0],
            changed: b[1] != 0,
            object_id: read_u64_le(b, 8),
            position1: read_u64_le(b, 16),
            position2: read_u64_le(b, 24),
        }
    }

    fn write_to(&self, out: &mut [u8]) {
        out.copy_from_slice(&self.to_bytes());
    }

    fn key(&self) -> u64 {
        self.object_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heap_index::sub_record::{
        TAG_CLASS_DUMP, TAG_INSTANCE_DUMP, TAG_OBJ_ARRAY_DUMP, TAG_PRIM_ARRAY_DUMP,
    };

    #[test]
    fn diff_entry_round_trip() {
        let entry = DiffEntry {
            tag: TAG_INSTANCE_DUMP,
            object_id: 0x1234_5678_9abc_def0,
            position: 0x0042_0000,
        };
        let bytes = entry.to_bytes();
        let decoded = DiffEntry::from_bytes(&bytes);
        assert_eq!(decoded, entry);
        assert_eq!(&bytes[1..8], &[0u8; 7], "padding must be zero");
    }

    #[test]
    fn common_entry_round_trip() {
        let entry = CommonEntry {
            tag: TAG_CLASS_DUMP,
            changed: true,
            object_id: 0xdead_beef_0001_0002,
            position1: 0x1000,
            position2: 0x2000,
        };
        let bytes = entry.to_bytes();
        let decoded = CommonEntry::from_bytes(&bytes);
        assert_eq!(decoded, entry);
        assert_eq!(&bytes[2..8], &[0u8; 6], "padding must be zero");
    }

    #[test]
    fn entries_read_back_through_record_file() {
        use crate::index::RecordFile;
        let diffs = [
            DiffEntry {
                tag: TAG_INSTANCE_DUMP,
                object_id: 10,
                position: 100,
            },
            DiffEntry {
                tag: TAG_PRIM_ARRAY_DUMP,
                object_id: 20,
                position: 200,
            },
        ];
        let data: Vec<u8> = diffs.iter().flat_map(|e| e.to_bytes()).collect();
        let file = RecordFile::<DiffEntry>::new(&data).unwrap();
        assert_eq!(file.iter().collect::<Vec<_>>(), diffs);
        assert_eq!(file.find(20), Some(diffs[1]));

        let commons = [
            CommonEntry {
                tag: TAG_OBJ_ARRAY_DUMP,
                changed: false,
                object_id: 5,
                position1: 50,
                position2: 500,
            },
            CommonEntry {
                tag: TAG_INSTANCE_DUMP,
                changed: true,
                object_id: 15,
                position1: 150,
                position2: 1500,
            },
        ];
        let data: Vec<u8> = commons.iter().flat_map(|e| e.to_bytes()).collect();
        let file = RecordFile::<CommonEntry>::new(&data).unwrap();
        assert_eq!(file.iter().collect::<Vec<_>>(), commons);
    }

    #[test]
    fn diff_directory_sits_next_to_the_first_dump() {
        let dir = diff_dir_for_hprofs(Path::new("/d/a.hprof"), Path::new("/e/b.hprof"));
        assert_eq!(dir, Path::new("/d/a_vs_b.diff_indexes"));
    }
}
