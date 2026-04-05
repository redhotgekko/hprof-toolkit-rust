pub mod record;

pub use record::{
    ClassDump, CpEntry, CpEntryIter, FieldValue, InstanceDump, InstanceFieldDescriptor,
    InstanceFieldIter, ObjArrayDump, ObjArrayElemIter, PrimArrayDump, RootJavaFrame, RootJniGlobal,
    RootJniLocal, RootMonitorUsed, RootNativeStack, RootStickyClass, RootThreadBlock,
    RootThreadObj, RootUnknown, StaticField, StaticFieldIter, SubRecord, parse_sub_record,
};

use crate::heap_index::sub_record::{SUB_INDEX_ENTRY_SIZE, SubIndexEntry};
use crate::hprof::HprofError;
use std::path::{Path, PathBuf};

// ── SubIndexReader ────────────────────────────────────────────────────────────

/// Memory-mapped reader for a single heap index sub-index file.
///
/// Each file was produced by [`crate::heap_index::index_heap_dumps`] for one
/// `HPROF_HEAP_DUMP` / `HPROF_HEAP_DUMP_SEGMENT` record.
pub struct SubIndexReader<'a> {
    data: &'a [u8],
}

impl<'a> SubIndexReader<'a> {
    pub fn from_ref(bytes: &'a [u8]) -> Result<Self, HprofError> {
        if !bytes.len().is_multiple_of(SUB_INDEX_ENTRY_SIZE) {
            return Err(HprofError::InvalidIndexFile);
        }
        Ok(Self { data: bytes })
    }

    fn as_slice(&self) -> &[u8] {
        self.data
    }

    /// Number of sub-index entries in this file.
    pub fn len(&self) -> usize {
        self.as_slice().len() / SUB_INDEX_ENTRY_SIZE
    }

    /// Returns `true` if the file contains no entries.
    pub fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    /// Iterate over all [`SubIndexEntry`] values in file order.
    ///
    /// The returned iterator borrows from the underlying data slice (`'a`),
    /// so it can outlive the `SubIndexReader` struct itself.
    pub fn iter(&self) -> SubIndexIter<'a> {
        SubIndexIter {
            data: self.data,
            pos: 0,
        }
    }

    /// Return the [`SubIndexEntry`] at position `i`, or `None` when
    /// `i >= len()`.
    pub fn entry_at(&self, i: usize) -> Option<SubIndexEntry> {
        let data = self.as_slice();
        let start = i * SUB_INDEX_ENTRY_SIZE;
        let end = start + SUB_INDEX_ENTRY_SIZE;
        if end > data.len() {
            return None;
        }
        let arr: &[u8; SUB_INDEX_ENTRY_SIZE] = data[start..end].try_into().ok()?;
        Some(SubIndexEntry::from_bytes(arr))
    }

    /// Binary-search for the first entry whose `object_id == target`.
    ///
    /// **Requires:** the file must be sorted ascending by `object_id`, as
    /// produced by [`crate::object_store::combine_and_sort_sub_index`].
    pub fn find_by_object_id(&self, target: u64) -> Option<SubIndexEntry> {
        self.find_by_object_id_and_tag(target, None)
    }

    /// Binary-search for the first entry matching `object_id == target` with
    /// the given `tag` (or any tag if `tag` is `None`).
    ///
    /// After locating the insertion point via binary search, scans forward
    /// through entries that share `target` until the desired tag is found.
    ///
    /// **Requires:** sorted ascending by `object_id`.
    pub fn find_by_object_id_and_tag(&self, target: u64, tag: Option<u8>) -> Option<SubIndexEntry> {
        let data = self.as_slice();
        let n = self.len();

        // Leftmost binary search: first index where object_id >= target.
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if sub_entry_object_id(data, mid) < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        // Linear scan through entries that share the target object_id.
        while lo < n && sub_entry_object_id(data, lo) == target {
            let start = lo * SUB_INDEX_ENTRY_SIZE;
            let bytes: [u8; SUB_INDEX_ENTRY_SIZE] =
                data[start..start + SUB_INDEX_ENTRY_SIZE].try_into().ok()?;
            let entry = SubIndexEntry::from_bytes(&bytes);
            if tag.is_none_or(|t| t == entry.tag) {
                return Some(entry);
            }
            lo += 1;
        }
        None
    }
}

/// Read the `object_id` field from entry `idx` in a flat sub-index byte slice.
fn sub_entry_object_id(data: &[u8], idx: usize) -> u64 {
    let start = idx * SUB_INDEX_ENTRY_SIZE + 8;
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&data[start..start + 8]);
    u64::from_le_bytes(bytes)
}

// ── SubIndexIter ──────────────────────────────────────────────────────────────

/// Iterator over [`SubIndexEntry`] values in a [`SubIndexReader`].
pub struct SubIndexIter<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> SubIndexIter<'a> {
    /// Construct an iterator directly from a validated data slice.
    ///
    /// The caller must ensure `data.len()` is a multiple of
    /// [`SUB_INDEX_ENTRY_SIZE`]; this is guaranteed when the slice comes from
    /// a [`ByteSource`] that was validated at index-open time.
    pub(crate) fn new(data: &'a [u8]) -> Self {
        debug_assert!(data.len().is_multiple_of(SUB_INDEX_ENTRY_SIZE));
        SubIndexIter { data, pos: 0 }
    }
}

impl Iterator for SubIndexIter<'_> {
    type Item = SubIndexEntry;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos + SUB_INDEX_ENTRY_SIZE > self.data.len() {
            return None;
        }
        let arr: &[u8; SUB_INDEX_ENTRY_SIZE] = self.data[self.pos..self.pos + SUB_INDEX_ENTRY_SIZE]
            .try_into()
            .ok()?;
        let entry = SubIndexEntry::from_bytes(arr);
        self.pos += SUB_INDEX_ENTRY_SIZE;
        Some(entry)
    }
}

// ── Directory helpers ─────────────────────────────────────────────────────────

/// Return paths to all heap index sub-index files under `output_dir`.
///
/// Files are stored in two-character hex subdirectories (e.g. `output_dir/1f/HPROF_HEAP_DUMP_SEGMENT_31`).
/// Matches any file whose name starts with `HPROF_HEAP_DUMP`, which covers
/// both `HPROF_HEAP_DUMP_<pos>` and `HPROF_HEAP_DUMP_SEGMENT_<pos>`.
pub fn sub_index_paths(output_dir: &Path) -> Result<Vec<PathBuf>, HprofError> {
    let mut paths = Vec::new();
    for subdir_entry in std::fs::read_dir(output_dir)? {
        let subdir_entry = subdir_entry?;
        if !subdir_entry.file_type()?.is_dir() {
            continue;
        }
        for file_entry in std::fs::read_dir(subdir_entry.path())? {
            let file_entry = file_entry?;
            if file_entry
                .file_name()
                .to_string_lossy()
                .starts_with("HPROF_HEAP_DUMP")
            {
                paths.push(file_entry.path());
            }
        }
    }
    Ok(paths)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heap_index::index_heap_dumps;
    use crate::heap_index::sub_record::TAG_ROOT_STICKY_CLASS;
    use crate::record_index::index_hprof;
    use crate::vfs::SubIndexDir;

    fn sub_index_readers(all_bytes: &[Vec<u8>]) -> Result<Vec<SubIndexReader<'_>>, HprofError> {
        all_bytes
            .iter()
            .map(|data| data.as_ref())
            .map(SubIndexReader::from_ref)
            .collect()
    }

    fn minimal_hprof_two_sticky_classes() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"JAVA PROFILE 1.0.2\0");
        buf.extend_from_slice(&8u32.to_be_bytes());
        buf.extend_from_slice(&0u64.to_be_bytes());
        // HEAP_DUMP_SEGMENT: two ROOT_STICKY_CLASS sub-records
        let mut body = Vec::new();
        body.push(TAG_ROOT_STICKY_CLASS);
        body.extend_from_slice(&1u64.to_be_bytes()); // class_id = 1
        body.push(TAG_ROOT_STICKY_CLASS);
        body.extend_from_slice(&2u64.to_be_bytes()); // class_id = 2
        buf.push(0x1C);
        buf.extend_from_slice(&0u32.to_be_bytes());
        buf.extend_from_slice(&(body.len() as u32).to_be_bytes());
        buf.extend_from_slice(&body);
        buf
    }

    #[test]
    fn sub_index_reader_iter() {
        // Run record_index + heap_index, then read the sub-index with SubIndexReader.
        let hprof_data = minimal_hprof_two_sticky_classes();
        let mut idx_buf = Vec::new();
        let out_dir = SubIndexDir::mem();

        index_hprof(&hprof_data, &mut idx_buf).unwrap();
        index_heap_dumps(&hprof_data, &idx_buf, &out_dir).unwrap();

        let all_bytes = out_dir.all_file_bytes().unwrap();
        let readers = sub_index_readers(&all_bytes).unwrap();
        assert_eq!(readers.len(), 1);

        let reader = &readers[0];
        assert_eq!(reader.len(), 2);

        let entries: Vec<_> = reader.iter().collect();
        assert_eq!(entries[0].tag, TAG_ROOT_STICKY_CLASS);
        assert_eq!(entries[0].object_id, 1);
        assert_eq!(entries[1].tag, TAG_ROOT_STICKY_CLASS);
        assert_eq!(entries[1].object_id, 2);
    }

    #[test]
    fn parse_sub_record_via_reader() {
        let hprof_data = minimal_hprof_two_sticky_classes();
        let mut idx_buf = Vec::new();
        let out_dir = SubIndexDir::mem();

        index_hprof(&hprof_data, &mut idx_buf).unwrap();
        index_heap_dumps(&hprof_data, &idx_buf, &out_dir).unwrap();

        let hprof = crate::hprof::HprofFile::from_ref(&hprof_data).unwrap();
        let all_bytes = out_dir.all_file_bytes().unwrap();
        let readers = sub_index_readers(&all_bytes).unwrap();
        let reader = &readers[0];

        let class_ids: Vec<u64> = reader
            .iter()
            .map(|entry| {
                let rec = parse_sub_record(&hprof, &entry).unwrap();
                if let SubRecord::RootStickyClass(r) = rec {
                    r.class_id
                } else {
                    0
                }
            })
            .collect();

        assert_eq!(class_ids, vec![1, 2]);
    }
}
