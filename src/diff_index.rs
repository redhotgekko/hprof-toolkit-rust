//! Binary diff index files for heap dump comparisons.
//!
//! [`build_diff_indexes`] performs a single O(n + m) merge-walk over two
//! sorted combined sub-record indexes and writes three binary output files:
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

use crate::heap_index::sub_record::{
    SubIndexEntry, TAG_CLASS_DUMP, TAG_INSTANCE_DUMP, TAG_OBJ_ARRAY_DUMP, TAG_PRIM_ARRAY_DUMP,
};
use crate::heap_parser::SubRecord;
use crate::hprof::HprofError;
use crate::query::HeapQuery;
use crate::vfs::MMapReader;
use std::io::Write;
use std::path::{Path, PathBuf};

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

    pub fn from_bytes(bytes: &[u8; DIFF_ENTRY_SIZE]) -> Self {
        Self {
            tag: bytes[0],
            object_id: u64::from_le_bytes(bytes[8..16].try_into().unwrap_or([0u8; 8])),
            position: u64::from_le_bytes(bytes[16..24].try_into().unwrap_or([0u8; 8])),
        }
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

    pub fn from_bytes(bytes: &[u8; COMMON_ENTRY_SIZE]) -> Self {
        Self {
            tag: bytes[0],
            changed: bytes[1] != 0,
            object_id: u64::from_le_bytes(bytes[8..16].try_into().unwrap_or([0u8; 8])),
            position1: u64::from_le_bytes(bytes[16..24].try_into().unwrap_or([0u8; 8])),
            position2: u64::from_le_bytes(bytes[24..32].try_into().unwrap_or([0u8; 8])),
        }
    }
}

// ── DiffIndexPaths ────────────────────────────────────────────────────────────

/// Paths to the three diff index files and their parent directory.
///
/// The directory is placed adjacent to `hprof1` and named
/// `{stem1}_vs_{stem2}.diff_indexes`.  For `./heap1.dump` vs `./heap2.dump`
/// this gives `./heap1_vs_heap2.diff_indexes/`.
pub struct DiffIndexPaths {
    pub dir: PathBuf,
    pub removed: PathBuf,
    pub added: PathBuf,
    pub common: PathBuf,
}

impl DiffIndexPaths {
    /// Derive diff index paths from the two hprof file paths.
    pub fn for_hprofs(hprof1: &Path, hprof2: &Path) -> Self {
        let stem1 = hprof1
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "dump1".to_string());
        let stem2 = hprof2
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "dump2".to_string());
        let parent = hprof1.parent().unwrap_or(Path::new("."));
        let dir = parent.join(format!("{stem1}_vs_{stem2}.diff_indexes"));
        Self {
            removed: dir.join("removed.bin"),
            added: dir.join("added.bin"),
            common: dir.join("common.bin"),
            dir,
        }
    }

    /// Returns `true` when all three index files already exist.
    pub fn all_exist(&self) -> bool {
        self.removed.exists() && self.added.exists() && self.common.exists()
    }
}

// ── Build counts ─────────────────────────────────────────────────────────────

/// Record counts returned by [`build_diff_indexes`].
#[derive(Debug, Clone, Copy)]
pub struct DiffIndexCounts {
    pub removed: u64,
    pub added: u64,
    pub common: u64,
    pub common_changed: u64,
}

// ── Builder ───────────────────────────────────────────────────────────────────

/// Build the three diff index files, skipping if they already exist.
///
/// Performs a single O(n + m) merge-walk of the two sorted combined indexes.
/// For each common object, the raw data bytes of both records are compared to
/// set the `changed` flag.
///
/// Returns the counts of written entries. If all three files exist, returns
/// zero counts without re-building.
pub fn build_diff_indexes(
    query1: &HeapQuery,
    query2: &HeapQuery,
    paths: &DiffIndexPaths,
) -> Result<DiffIndexCounts, HprofError> {
    if paths.all_exist() {
        return Ok(DiffIndexCounts {
            removed: 0,
            added: 0,
            common: 0,
            common_changed: 0,
        });
    }

    std::fs::create_dir_all(&paths.dir)?;

    let mut w_removed = std::io::BufWriter::new(std::fs::File::create(&paths.removed)?);
    let mut w_added = std::io::BufWriter::new(std::fs::File::create(&paths.added)?);
    let mut w_common = std::io::BufWriter::new(std::fs::File::create(&paths.common)?);

    let mut counts = DiffIndexCounts {
        removed: 0,
        added: 0,
        common: 0,
        common_changed: 0,
    };

    let mut iter1 = query1.iter_entries().filter(|e| is_object_tag(e.tag));
    let mut iter2 = query2.iter_entries().filter(|e| is_object_tag(e.tag));
    let mut cur1: Option<SubIndexEntry> = iter1.next();
    let mut cur2: Option<SubIndexEntry> = iter2.next();

    loop {
        match (cur1, cur2) {
            (None, None) => break,

            (Some(e1), None) => {
                write_diff_entry(&mut w_removed, &e1)?;
                counts.removed += 1;
                cur1 = iter1.next();
                cur2 = None;
            }

            (None, Some(e2)) => {
                write_diff_entry(&mut w_added, &e2)?;
                counts.added += 1;
                cur1 = None;
                cur2 = iter2.next();
            }

            (Some(e1), Some(e2)) => match e1.object_id.cmp(&e2.object_id) {
                std::cmp::Ordering::Less => {
                    write_diff_entry(&mut w_removed, &e1)?;
                    counts.removed += 1;
                    cur1 = iter1.next();
                    cur2 = Some(e2);
                }
                std::cmp::Ordering::Greater => {
                    write_diff_entry(&mut w_added, &e2)?;
                    counts.added += 1;
                    cur1 = Some(e1);
                    cur2 = iter2.next();
                }
                std::cmp::Ordering::Equal => {
                    let changed = records_changed(query1, &e1, query2, &e2)?;
                    let entry = CommonEntry {
                        tag: e1.tag,
                        changed,
                        object_id: e1.object_id,
                        position1: e1.position,
                        position2: e2.position,
                    };
                    w_common.write_all(&entry.to_bytes())?;
                    counts.common += 1;
                    if changed {
                        counts.common_changed += 1;
                    }
                    cur1 = iter1.next();
                    cur2 = iter2.next();
                }
            },
        }
    }

    w_removed.flush()?;
    w_added.flush()?;
    w_common.flush()?;

    Ok(counts)
}

// ── Private helpers ───────────────────────────────────────────────────────────

fn is_object_tag(tag: u8) -> bool {
    matches!(
        tag,
        TAG_INSTANCE_DUMP | TAG_CLASS_DUMP | TAG_OBJ_ARRAY_DUMP | TAG_PRIM_ARRAY_DUMP
    )
}

fn write_diff_entry(w: &mut impl Write, entry: &SubIndexEntry) -> Result<(), HprofError> {
    let de = DiffEntry {
        tag: entry.tag,
        object_id: entry.object_id,
        position: entry.position,
    };
    w.write_all(&de.to_bytes())?;
    Ok(())
}

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

// ── DiffEntryReader ───────────────────────────────────────────────────────────

/// Memory-mapped reader for `removed.bin` or `added.bin`.
pub struct DiffEntryReader {
    data: Vec<u8>,
}

impl DiffEntryReader {
    /// Open a diff entry file (removed or added) for reading.
    pub fn open(path: &Path) -> Result<Self, HprofError> {
        let mmap = path.to_path_buf().open_mmap()?;
        let bytes = mmap.as_ref().to_vec();
        if bytes.len() % DIFF_ENTRY_SIZE != 0 {
            return Err(HprofError::InvalidIndexFile);
        }
        Ok(Self { data: bytes })
    }

    /// Number of entries in the file.
    pub fn len(&self) -> usize {
        self.data.len() / DIFF_ENTRY_SIZE
    }

    /// Returns `true` if there are no entries.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Return the entry at position `i`, or `None` when `i >= len()`.
    pub fn entry_at(&self, i: usize) -> Option<DiffEntry> {
        let start = i * DIFF_ENTRY_SIZE;
        let end = start + DIFF_ENTRY_SIZE;
        if end > self.data.len() {
            return None;
        }
        let arr: &[u8; DIFF_ENTRY_SIZE] = self.data[start..end].try_into().ok()?;
        Some(DiffEntry::from_bytes(arr))
    }

    /// Iterate all entries in ascending `object_id` order.
    pub fn iter(&self) -> DiffEntryIter<'_> {
        DiffEntryIter {
            data: &self.data,
            pos: 0,
        }
    }
}

/// Iterator over [`DiffEntry`] values.
pub struct DiffEntryIter<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Iterator for DiffEntryIter<'_> {
    type Item = DiffEntry;

    fn next(&mut self) -> Option<Self::Item> {
        let end = self.pos + DIFF_ENTRY_SIZE;
        if end > self.data.len() {
            return None;
        }
        let arr: &[u8; DIFF_ENTRY_SIZE] = self.data[self.pos..end].try_into().ok()?;
        self.pos = end;
        Some(DiffEntry::from_bytes(arr))
    }
}

// ── CommonEntryReader ─────────────────────────────────────────────────────────

/// Memory-mapped reader for `common.bin`.
pub struct CommonEntryReader {
    data: Vec<u8>,
}

impl CommonEntryReader {
    /// Open the common diff index file for reading.
    pub fn open(path: &Path) -> Result<Self, HprofError> {
        let mmap = path.to_path_buf().open_mmap()?;
        let bytes = mmap.as_ref().to_vec();
        if bytes.len() % COMMON_ENTRY_SIZE != 0 {
            return Err(HprofError::InvalidIndexFile);
        }
        Ok(Self { data: bytes })
    }

    /// Number of entries in the file.
    pub fn len(&self) -> usize {
        self.data.len() / COMMON_ENTRY_SIZE
    }

    /// Returns `true` if there are no entries.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Return the entry at position `i`, or `None` when `i >= len()`.
    pub fn entry_at(&self, i: usize) -> Option<CommonEntry> {
        let start = i * COMMON_ENTRY_SIZE;
        let end = start + COMMON_ENTRY_SIZE;
        if end > self.data.len() {
            return None;
        }
        let arr: &[u8; COMMON_ENTRY_SIZE] = self.data[start..end].try_into().ok()?;
        Some(CommonEntry::from_bytes(arr))
    }

    /// Iterate all entries in ascending `object_id` order.
    pub fn iter(&self) -> CommonEntryIter<'_> {
        CommonEntryIter {
            data: &self.data,
            pos: 0,
        }
    }
}

/// Iterator over [`CommonEntry`] values.
pub struct CommonEntryIter<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Iterator for CommonEntryIter<'_> {
    type Item = CommonEntry;

    fn next(&mut self) -> Option<Self::Item> {
        let end = self.pos + COMMON_ENTRY_SIZE;
        if end > self.data.len() {
            return None;
        }
        let arr: &[u8; COMMON_ENTRY_SIZE] = self.data[self.pos..end].try_into().ok()?;
        self.pos = end;
        Some(CommonEntry::from_bytes(arr))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
    fn diff_entry_reader_iter() {
        let entries = vec![
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
        let mut data = Vec::new();
        for e in &entries {
            data.extend_from_slice(&e.to_bytes());
        }
        let reader = DiffEntryReader { data };
        let collected: Vec<DiffEntry> = reader.iter().collect();
        assert_eq!(collected, entries);
    }

    #[test]
    fn common_entry_reader_iter() {
        let entries = vec![
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
        let mut data = Vec::new();
        for e in &entries {
            data.extend_from_slice(&e.to_bytes());
        }
        let reader = CommonEntryReader { data };
        let collected: Vec<CommonEntry> = reader.iter().collect();
        assert_eq!(collected, entries);
    }
}
