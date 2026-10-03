//! Auxiliary record indexes.
//!
//! Builds fixed-size binary index files for the remaining top-level hprof
//! record types that are not yet indexed by previous phases:
//!
//! | Record              | Tag  | Key                |
//! |---------------------|------|--------------------|
//! | `HPROF_UNLOAD_CLASS`| 0x03 | class_serial (u32) |
//! | `HPROF_FRAME`       | 0x04 | frame_id (ID)      |
//! | `HPROF_TRACE`       | 0x05 | trace_serial (u32) |
//! | `HPROF_START_THREAD`| 0x0A | thread_serial (u32)|
//! | `HPROF_END_THREAD`  | 0x0B | thread_serial (u32)|
//!
//! All index files share a common 16-byte record format (little-endian):
//!
//! ```text
//! bytes 0..8   key           u64  primary identifier or serial number
//! bytes 8..16  hprof_offset  u64  byte offset of the record tag in the hprof file
//! ```
//!
//! Each file is sorted by `key`, enabling O(log n) binary search.

use crate::hprof::record::{RecordTag, read_id};
use crate::hprof::{HprofError, HprofFile};
use crate::index::StoreWriter;
use crate::index::{Entry, RecordFile, read_u64_le};
use crate::record_index::entry::IndexEntry;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Byte size of one auxiliary record index entry.
pub const AUX_ENTRY_SIZE: usize = 16;

/// Byte size of a top-level record header (tag + time_offset + body_length).
const RECORD_HEADER_SIZE: usize = 9;

// ── Entry ─────────────────────────────────────────────────────────────────────

/// One `(key, hprof_offset)` record shared by all five auxiliary indexes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuxEntry {
    /// Primary identifier or serial number, zero-extended to u64 (sort key).
    pub key: u64,
    /// Byte offset of the record tag byte in the hprof file.
    pub hprof_offset: u64,
}

impl Entry for AuxEntry {
    const SIZE: usize = AUX_ENTRY_SIZE;
    const KEY_OFFSET: usize = 0;

    fn from_bytes(b: &[u8]) -> Self {
        Self {
            key: read_u64_le(b, 0),
            hprof_offset: read_u64_le(b, 8),
        }
    }

    fn write_to(&self, out: &mut [u8]) {
        out[0..8].copy_from_slice(&self.key.to_le_bytes());
        out[8..16].copy_from_slice(&self.hprof_offset.to_le_bytes());
    }

    fn key(&self) -> u64 {
        self.key
    }
}

// ── Generic reader ────────────────────────────────────────────────────────────

/// Read-only handle to a sorted auxiliary index file.
///
/// All five index types share the same 16-byte record layout, so a single
/// generic reader covers them all.
#[derive(Clone, Copy)]
pub struct AuxIndexReader<'a> {
    file: RecordFile<'a, AuxEntry>,
}

impl<'a> AuxIndexReader<'a> {
    pub fn from_ref(source: &'a [u8]) -> Result<Self, HprofError> {
        Ok(Self {
            file: RecordFile::new(source)?,
        })
    }

    /// Construct from a slice that is already known to be a valid index.
    pub(crate) fn from_slice(source: &'a [u8]) -> Self {
        Self {
            file: RecordFile::from_slice(source),
        }
    }

    /// Return the `hprof_offset` stored for `key`, or `None` if not found.
    pub fn find(&self, key: u64) -> Option<u64> {
        self.file.find(key).map(|e| e.hprof_offset)
    }

    /// Return the `(key, hprof_offset)` pair at position `idx` in the sorted index.
    ///
    /// Returns `(0, 0)` when `idx >= self.len()`; callers bound `idx` by `len()`.
    pub fn entry_at(&self, idx: usize) -> (u64, u64) {
        let e = self.file.get(idx).unwrap_or(AuxEntry {
            key: 0,
            hprof_offset: 0,
        });
        (e.key, e.hprof_offset)
    }

    /// Total number of records in this index.
    pub fn len(&self) -> usize {
        self.file.len()
    }
}

// ── Index builders ────────────────────────────────────────────────────────────

/// Build an index of `HPROF_FRAME` records sorted by `frame_id`.
///
/// Record body layout:
/// ```text
/// frame_id(ID)  method_name_id(ID)  method_sig_id(ID)  source_file_id(ID)
/// class_serial(u32)  line_number(i32)
/// ```
///
/// Returns the number of entries written.
pub fn build_frame_index(
    hprof_source: &[u8],
    record_index_source: &[u8],
    output: &mut dyn StoreWriter,
) -> Result<u64, HprofError> {
    let hprof = HprofFile::from_ref(hprof_source)?;
    let id_size = hprof.id_size() as usize;
    build_aux_index(
        record_index_source,
        RecordTag::Frame,
        output,
        |body_start| read_id(hprof.data(), body_start, id_size),
    )
}

/// Build an index of `HPROF_TRACE` records sorted by `trace_serial`.
///
/// Record body layout:
/// ```text
/// trace_serial(u32)  thread_serial(u32)  num_frames(u32)
/// [frame_id(ID); num_frames]
/// ```
///
/// Returns the number of entries written.
pub fn build_trace_index(
    hprof_source: &[u8],
    record_index_source: &[u8],
    output: &mut dyn StoreWriter,
) -> Result<u64, HprofError> {
    let hprof = HprofFile::from_ref(hprof_source)?;
    build_aux_index(
        record_index_source,
        RecordTag::Trace,
        output,
        |body_start| read_u32_be(hprof.data(), body_start).map(u64::from),
    )
}

/// Build an index of `HPROF_START_THREAD` records sorted by `thread_serial`.
///
/// Record body layout:
/// ```text
/// thread_serial(u32)  thread_id(ID)  stack_trace_serial(u32)
/// thread_name_id(ID)  thread_group_name_id(ID)  parent_group_name_id(ID)
/// ```
///
/// Returns the number of entries written.
pub fn build_start_thread_index(
    hprof_source: &[u8],
    record_index_source: &[u8],
    output: &mut dyn StoreWriter,
) -> Result<u64, HprofError> {
    let hprof = HprofFile::from_ref(hprof_source)?;
    build_aux_index(
        record_index_source,
        RecordTag::StartThread,
        output,
        |body_start| read_u32_be(hprof.data(), body_start).map(u64::from),
    )
}

/// Build an index of `HPROF_END_THREAD` records sorted by `thread_serial`.
///
/// Record body layout: `thread_serial(u32)`.
///
/// Returns the number of entries written.
pub fn build_end_thread_index(
    hprof_source: &[u8],
    record_index_source: &[u8],
    output: &mut dyn StoreWriter,
) -> Result<u64, HprofError> {
    let hprof = HprofFile::from_ref(hprof_source)?;
    build_aux_index(
        record_index_source,
        RecordTag::EndThread,
        output,
        |body_start| read_u32_be(hprof.data(), body_start).map(u64::from),
    )
}

/// Build an index of `HPROF_UNLOAD_CLASS` records sorted by `class_serial`.
///
/// Record body layout: `class_serial(u32)`.
///
/// Returns the number of entries written.
pub fn build_unload_class_index(
    hprof_source: &[u8],
    record_index_source: &[u8],
    output: &mut dyn StoreWriter,
) -> Result<u64, HprofError> {
    let hprof = HprofFile::from_ref(hprof_source)?;
    build_aux_index(
        record_index_source,
        RecordTag::UnloadClass,
        output,
        |body_start| read_u32_be(hprof.data(), body_start).map(u64::from),
    )
}

// ── Common helpers ────────────────────────────────────────────────────────────

/// Shared builder: for every record-index entry with `tag`, compute the key
/// from its body via `key_of(body_start)`, then write all `(key, offset)`
/// pairs sorted by key.
///
/// The number of auxiliary records is bounded by frames/threads, not heap
/// objects, so collecting them in a `Vec` is within the memory rule.
fn build_aux_index(
    record_index_source: &[u8],
    tag: RecordTag,
    output: &mut dyn StoreWriter,
    key_of: impl Fn(usize) -> Result<u64, HprofError>,
) -> Result<u64, HprofError> {
    let tag = u8::from(tag);
    let mut entries: Vec<AuxEntry> = Vec::new();
    for entry in RecordFile::<IndexEntry>::new(record_index_source)?.iter() {
        if entry.tag != tag {
            continue;
        }
        let body_start = entry.position as usize + RECORD_HEADER_SIZE;
        entries.push(AuxEntry {
            key: key_of(body_start)?,
            hprof_offset: entry.position,
        });
    }

    entries.sort_unstable_by_key(|e| e.key);
    let mut buf = [0u8; AUX_ENTRY_SIZE];
    for e in &entries {
        e.write_to(&mut buf);
        output.write_all(&buf)?;
    }
    output.flush()?;
    Ok(entries.len() as u64)
}

/// Bounds-checked big-endian u32 read.
fn read_u32_be(data: &[u8], off: usize) -> Result<u32, HprofError> {
    if off + 4 > data.len() {
        return Err(HprofError::UnexpectedEof(off));
    }
    Ok(u32::from_be_bytes([
        data[off],
        data[off + 1],
        data[off + 2],
        data[off + 3],
    ]))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record_index::index_hprof;
    use crate::test_util::{HprofBuilder, built_bytes};

    /// Linear search over a raw aux index buffer; returns `hprof_offset` for `key`.
    fn aux_find(buf: &[u8], key: u64) -> Option<u64> {
        buf.as_chunks::<AUX_ENTRY_SIZE>()
            .0
            .iter()
            .map(|c| AuxEntry::from_bytes(c))
            .find(|e| e.key == key)
            .map(|e| e.hprof_offset)
    }

    fn record_index(hprof: &[u8]) -> Vec<u8> {
        let mut p1 = Vec::new();
        index_hprof(hprof, &mut p1).unwrap();
        p1
    }

    #[test]
    fn frame_index_round_trip() {
        let hprof = HprofBuilder::new(8)
            .frame(0x100, 0, 0, 0, 1, 1)
            .frame(0x200, 0, 0, 0, 2, 1)
            .frame(0x50, 0, 0, 0, 3, 1) // out of order to test sort
            .build();
        let p1 = record_index(&hprof);

        let (n, out_buf) = built_bytes(|w| build_frame_index(&hprof, &p1, w));
        assert_eq!(n, 3);
        assert_eq!(out_buf.len(), 3 * AUX_ENTRY_SIZE);
        assert!(aux_find(&out_buf, 0x50).is_some());
        assert!(aux_find(&out_buf, 0x100).is_some());
        assert!(aux_find(&out_buf, 0x200).is_some());
        assert!(aux_find(&out_buf, 0x999).is_none());
        // sorted ascending by key
        let reader = AuxIndexReader::from_ref(&out_buf).unwrap();
        let keys: Vec<u64> = reader.file.iter().map(|e| e.key).collect();
        assert_eq!(keys, vec![0x50, 0x100, 0x200]);
        assert_eq!(reader.entry_at(0).0, 0x50);
    }

    #[test]
    fn trace_index_round_trip() {
        let hprof = HprofBuilder::new(8)
            .trace(10, 1, &[0x100, 0x200])
            .trace(5, 1, &[]) // lower serial, tests sort
            .build();
        let p1 = record_index(&hprof);

        let (n, out_buf) = built_bytes(|w| build_trace_index(&hprof, &p1, w));
        assert_eq!(n, 2);
        assert!(aux_find(&out_buf, 5).is_some());
        assert!(aux_find(&out_buf, 10).is_some());
        assert!(aux_find(&out_buf, 99).is_none());
    }

    #[test]
    fn start_thread_index_round_trip() {
        let hprof = HprofBuilder::new(8)
            .start_thread(3, 0xABC, 0, 0, 0, 0)
            .start_thread(1, 0xDEF, 0, 0, 0, 0)
            .build();
        let p1 = record_index(&hprof);

        let (n, out_buf) = built_bytes(|w| build_start_thread_index(&hprof, &p1, w));
        assert_eq!(n, 2);
        assert!(aux_find(&out_buf, 1).is_some());
        assert!(aux_find(&out_buf, 3).is_some());
        assert!(aux_find(&out_buf, 2).is_none());
    }

    #[test]
    fn end_thread_index_round_trip() {
        let hprof = HprofBuilder::new(8).end_thread(7).build();
        let p1 = record_index(&hprof);

        let (n, out_buf) = built_bytes(|w| build_end_thread_index(&hprof, &p1, w));
        assert_eq!(n, 1);
        assert!(aux_find(&out_buf, 7).is_some());
        assert!(aux_find(&out_buf, 1).is_none());
    }

    #[test]
    fn unload_class_index_round_trip() {
        let hprof = HprofBuilder::new(8)
            .unload_class(42)
            .unload_class(10)
            .build();
        let p1 = record_index(&hprof);

        let (n, out_buf) = built_bytes(|w| build_unload_class_index(&hprof, &p1, w));
        assert_eq!(n, 2);
        assert!(aux_find(&out_buf, 10).is_some());
        assert!(aux_find(&out_buf, 42).is_some());
        assert!(aux_find(&out_buf, 99).is_none());
    }

    #[test]
    fn empty_index_is_empty() {
        let hprof = HprofBuilder::new(8).build(); // no records
        let p1 = record_index(&hprof);

        let (n, out_buf) = built_bytes(|w| build_frame_index(&hprof, &p1, w));
        assert_eq!(n, 0);
        assert!(out_buf.is_empty());
        assert!(aux_find(&out_buf, 1).is_none());
        let reader = AuxIndexReader::from_ref(&out_buf).unwrap();
        assert!(reader.file.is_empty());
        assert_eq!(reader.find(1), None);
    }

    #[test]
    fn find_returns_correct_hprof_offset() {
        // Verify that the stored offset points at the record tag byte.
        let mut b = HprofBuilder::new(8);
        let offset_before = b.next_record_position();
        let hprof = b.frame(0xABCD, 0, 0, 0, 1, 1).build();
        let p1 = record_index(&hprof);

        let ((), out_buf) = built_bytes(|w| build_frame_index(&hprof, &p1, w).map(drop));
        let stored_offset = aux_find(&out_buf, 0xABCD).unwrap();
        assert_eq!(
            stored_offset, offset_before,
            "offset should point at tag byte"
        );
        assert_eq!(hprof[stored_offset as usize], 0x04);
        let reader = AuxIndexReader::from_ref(&out_buf).unwrap();
        assert_eq!(reader.find(0xABCD), Some(offset_before));
    }

    #[test]
    fn id_size_four_frames_are_indexed() {
        let hprof = HprofBuilder::new(4).frame(0x77, 1, 2, 3, 1, 9).build();
        let p1 = record_index(&hprof);
        let (n, out_buf) = built_bytes(|w| build_frame_index(&hprof, &p1, w));
        assert_eq!(n, 1);
        assert!(aux_find(&out_buf, 0x77).is_some());
    }
}
