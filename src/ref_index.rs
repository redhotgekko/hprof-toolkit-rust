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
//! ## Parallelism
//!
//! The combined index is split into `rayon::current_num_threads()` equal
//! chunks.  Each chunk is processed by a rayon task that writes its results
//! to a separate temporary file.  The temporary files are then concatenated
//! and sorted in-place using the generic parallel introsort from `crate::sort`.

use crate::heap_index::sub_record::{
    SUB_INDEX_ENTRY_SIZE, SubIndexEntry, TAG_CLASS_DUMP, TAG_INSTANCE_DUMP, TAG_OBJ_ARRAY_DUMP,
};
use crate::heap_parser::SubRecord;
use crate::heap_parser::record::FieldValue;
use crate::heap_query::HprofIndex;
use crate::heap_query::resolve::read_field_value;
use crate::hprof::HprofError;
use crate::vfs::{MMapReader, MMapWriter};
use rayon::prelude::*;
use std::io::Write;
use std::path::Path;

/// Byte size of one reference index record.
pub const REF_ENTRY_SIZE: usize = 16;

/// Maximum number of back-references shown per page.
pub const MAX_BACK_REFS: usize = 50;

// ── Public builder ────────────────────────────────────────────────────────────

/// Scan all heap objects and build a reference index, writing to `output`.
///
/// For every non-null object-reference field in every `INSTANCE_DUMP`,
/// `CLASS_DUMP` (static fields), and `OBJ_ARRAY_DUMP`, one 16-byte record
/// `(to_object_id, from_object_id)` is appended to the output.
///
/// The combined output is sorted ascending by `to_object_id` so that
/// [`RefIndex::find`] can perform O(log n) binary search.
///
/// Returns the total number of reference records written.
pub fn build_reference_index(
    hprof_source: &impl MMapReader,
    combined_source: &impl MMapReader,
    utf8_source: &impl MMapReader,
    load_class_source: &impl MMapReader,
    output: &mut impl MMapWriter,
) -> Result<u64, HprofError> {
    let index = HprofIndex::open(
        hprof_source,
        combined_source,
        utf8_source,
        load_class_source,
    )?;

    let combined_bs = combined_source.open_mmap()?;
    let combined_mmap: &[u8] = combined_bs.as_ref();
    if !combined_mmap.len().is_multiple_of(SUB_INDEX_ENTRY_SIZE) {
        return Err(HprofError::InvalidIndexFile);
    }

    // Split the combined index into one chunk per rayon thread.
    let n_entries = combined_mmap.len() / SUB_INDEX_ENTRY_SIZE;
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_entries = (n_entries / n_threads).max(1);

    let chunks: Vec<&[u8]> = combined_mmap
        .chunks(chunk_entries * SUB_INDEX_ENTRY_SIZE)
        .collect();

    // Process each chunk in parallel; each task collects into an in-memory buffer.
    let chunk_bufs: Vec<Vec<u8>> = chunks
        .par_iter()
        .map(|chunk| -> Result<Vec<u8>, HprofError> {
            let mut buf = Vec::new();
            for raw in chunk.chunks_exact(SUB_INDEX_ENTRY_SIZE) {
                let arr: [u8; SUB_INDEX_ENTRY_SIZE] =
                    raw.try_into().map_err(|_| HprofError::InvalidIndexFile)?;
                let entry = SubIndexEntry::from_bytes(&arr);
                write_refs_for_entry(&index, &entry, &mut buf)?;
            }
            Ok(buf)
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Concatenate all chunk buffers into the output writer.
    let total = concatenate_ref_bufs(&chunk_bufs, output)?;

    // Sort by to_object_id (offset 0).
    if total > 1 {
        let mut mmap = output.create_mut_mmap()?;
        crate::sort::parallel_introsort(mmap.as_mut(), REF_ENTRY_SIZE, 0);
    }

    Ok(total)
}

// ── Public reader ─────────────────────────────────────────────────────────────

/// Read-only handle to a sorted reference index file.
pub struct RefIndex {
    data: crate::vfs::ByteSource,
}

impl RefIndex {
    /// Open `path` as a read-only memory-mapped reference index.
    pub fn open(path: &Path) -> Result<Self, HprofError> {
        let mmap = crate::hprof::map_file(path)?;
        if mmap.len() % REF_ENTRY_SIZE != 0 {
            return Err(HprofError::InvalidIndexFile);
        }
        Ok(Self {
            data: crate::vfs::ByteSource::MMapSource(mmap),
        })
    }

    /// Create a reader from raw bytes.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, HprofError> {
        if !bytes.len().is_multiple_of(REF_ENTRY_SIZE) {
            return Err(HprofError::InvalidIndexFile);
        }
        Ok(Self {
            data: crate::vfs::ByteSource::VecSource(bytes),
        })
    }

    /// Return all `from_object_id` values that hold a reference to `to_id`.
    ///
    /// Uses leftmost binary search followed by a linear scan through equal
    /// keys.  Results are bounded by [`MAX_BACK_REFS`].
    pub fn find(&self, to_id: u64) -> Vec<u64> {
        let data = self.data.as_ref();
        let n = data.len() / REF_ENTRY_SIZE;

        // Leftmost binary search.
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if read_to_id(data, mid) < to_id {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        let mut results = Vec::new();
        let mut i = lo;
        while i < n && read_to_id(data, i) == to_id {
            if results.len() >= MAX_BACK_REFS {
                break;
            }
            results.push(read_from_id(data, i));
            i += 1;
        }
        results
    }

    /// Total number of reference records in this index.
    pub fn len(&self) -> usize {
        self.data.as_ref().len() / REF_ENTRY_SIZE
    }

    /// Returns `true` if the index contains no records.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ── Per-entry reference extraction ───────────────────────────────────────────

fn write_refs_for_entry(
    index: &HprofIndex,
    entry: &SubIndexEntry,
    writer: &mut impl Write,
) -> Result<(), HprofError> {
    let from_id = entry.object_id;

    match entry.tag {
        TAG_INSTANCE_DUMP => {
            if let SubRecord::InstanceDump(inst) = index.parse_entry(entry)? {
                for to_id in extract_instance_refs(index, &inst)? {
                    write_ref(writer, to_id, from_id)?;
                }
            }
        }
        TAG_CLASS_DUMP => {
            if let SubRecord::ClassDump(cd) = index.parse_entry(entry)? {
                for sf in cd.static_fields() {
                    let sf = sf?;
                    if let FieldValue::Object(to_id) = sf.value
                        && to_id != 0
                    {
                        write_ref(writer, to_id, from_id)?;
                    }
                }
            }
        }
        TAG_OBJ_ARRAY_DUMP => {
            if let SubRecord::ObjArrayDump(arr) = index.parse_entry(entry)? {
                for to_id in arr.elements() {
                    if to_id != 0 {
                        write_ref(writer, to_id, from_id)?;
                    }
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// Walk the class hierarchy of `inst` and collect every non-null object
/// reference value from the instance data bytes.
///
/// Only field types are needed (not names), so this is lighter than
/// `HprofIndex::instance_fields`.
fn extract_instance_refs(
    index: &HprofIndex,
    inst: &crate::heap_parser::InstanceDump<'_>,
) -> Result<Vec<u64>, HprofError> {
    let id_size = index.id_size() as usize;
    let mut chain: Vec<Vec<u8>> = Vec::new();
    let mut class_id = inst.class_id;

    while class_id != 0 {
        // Collect field types and super_class_id then drop the ClassDump borrow
        // before the next find_class_dump call (ClassDump borrows from index).
        let (types, super_id) = match index.find_class_dump(class_id)? {
            Some(SubRecord::ClassDump(cd)) => {
                let types: Vec<u8> = cd
                    .instance_fields()
                    .filter_map(|r| r.ok())
                    .map(|fd| fd.field_type)
                    .collect();
                let super_id = cd.super_class_id;
                (types, super_id)
            }
            _ => break,
        };
        chain.push(types);
        class_id = super_id;
    }

    let mut refs = Vec::new();
    let mut offset = 0usize;
    for types in &chain {
        for &ft in types {
            let (value, consumed) = read_field_value(inst.data, offset, ft, id_size)?;
            offset += consumed;
            if let FieldValue::Object(id) = value
                && id != 0
            {
                refs.push(id);
            }
        }
    }
    Ok(refs)
}

// ── File I/O ──────────────────────────────────────────────────────────────────

fn write_ref(writer: &mut impl Write, to_id: u64, from_id: u64) -> Result<(), HprofError> {
    let mut buf = [0u8; REF_ENTRY_SIZE];
    buf[0..8].copy_from_slice(&to_id.to_le_bytes());
    buf[8..16].copy_from_slice(&from_id.to_le_bytes());
    writer.write_all(&buf)?;
    Ok(())
}

fn concatenate_ref_bufs(bufs: &[Vec<u8>], output: &mut impl MMapWriter) -> Result<u64, HprofError> {
    let mut writer = output.create_writer()?;
    let mut total = 0u64;
    for buf in bufs {
        let aligned = (buf.len() / REF_ENTRY_SIZE) * REF_ENTRY_SIZE;
        writer.write_all(&buf[..aligned])?;
        total += (aligned / REF_ENTRY_SIZE) as u64;
    }
    writer.flush()?;
    Ok(total)
}

fn read_to_id(data: &[u8], idx: usize) -> u64 {
    let start = idx * REF_ENTRY_SIZE;
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&data[start..start + 8]);
    u64::from_le_bytes(bytes)
}

fn read_from_id(data: &[u8], idx: usize) -> u64 {
    let start = idx * REF_ENTRY_SIZE + 8;
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&data[start..start + 8]);
    u64::from_le_bytes(bytes)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ref_bytes(pairs: &[(u64, u64)]) -> Vec<u8> {
        let mut data = Vec::with_capacity(pairs.len() * REF_ENTRY_SIZE);
        for &(to_id, from_id) in pairs {
            data.extend_from_slice(&to_id.to_le_bytes());
            data.extend_from_slice(&from_id.to_le_bytes());
        }
        data
    }

    #[test]
    fn ref_index_find_returns_matching_from_ids() {
        // Sorted pairs: (to=1,from=10), (to=2,from=20), (to=2,from=30), (to=3,from=40)
        let data = make_ref_bytes(&[(1, 10), (2, 20), (2, 30), (3, 40)]);
        let idx = RefIndex::from_bytes(data).unwrap();

        assert_eq!(idx.find(2), vec![20, 30]);
        assert_eq!(idx.find(1), vec![10]);
        assert_eq!(idx.find(3), vec![40]);
        assert_eq!(idx.find(99), vec![] as Vec<u64>);
    }

    #[test]
    fn ref_index_find_empty_file() {
        let idx = RefIndex::from_bytes(vec![]).unwrap();
        assert!(idx.find(1).is_empty());
        assert!(idx.is_empty());
    }

    #[test]
    fn write_ref_round_trips() {
        let mut buf = Vec::new();
        write_ref(&mut buf, 0xAABBCCDD_00112233, 0x11223344_AABBCCDD).unwrap();
        assert_eq!(buf.len(), REF_ENTRY_SIZE);
        assert_eq!(
            u64::from_le_bytes(buf[0..8].try_into().unwrap()),
            0xAABBCCDD_00112233
        );
        assert_eq!(
            u64::from_le_bytes(buf[8..16].try_into().unwrap()),
            0x11223344_AABBCCDD
        );
    }
}
