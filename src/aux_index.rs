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
//! Each file is sorted in-place by `key`, enabling O(log n) binary search.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use hprof_toolkit::aux_index::{
//!     build_frame_index, build_trace_index,
//!     build_start_thread_index, build_end_thread_index,
//!     build_unload_class_index,
//!     FrameIndexReader, TraceIndexReader,
//!     StartThreadIndexReader, EndThreadIndexReader,
//!     UnloadClassIndexReader,
//! };
//! ```

use crate::hprof::record::RecordTag;
use crate::hprof::{HprofError, HprofFile};
use crate::record_index::entry::{INDEX_ENTRY_SIZE, IndexEntry};
use crate::vfs::{ByteSource, MMapReader, MMapWriter};
use std::io::Write;
use std::path::Path;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Byte size of one auxiliary record index entry.
pub const AUX_ENTRY_SIZE: usize = 16;

/// Byte size of a top-level record header (tag + time_offset + body_length).
const RECORD_HEADER_SIZE: usize = 9;

// ── Generic reader ────────────────────────────────────────────────────────────

/// Read-only handle to a sorted auxiliary index file.
///
/// All five index types share the same 16-byte record layout, so a single
/// generic reader covers them all.
pub struct AuxIndexReader {
    data: ByteSource,
}

impl AuxIndexReader {
    /// Open `path` as a read-only memory-mapped auxiliary index.
    pub fn open(path: &Path) -> Result<Self, HprofError> {
        let mmap = crate::hprof::map_file(path)?;
        if mmap.len() % AUX_ENTRY_SIZE != 0 {
            return Err(HprofError::InvalidIndexFile);
        }
        Ok(Self {
            data: ByteSource::MMapSource(mmap),
        })
    }

    /// Create a reader from raw bytes.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, HprofError> {
        if !bytes.len().is_multiple_of(AUX_ENTRY_SIZE) {
            return Err(HprofError::InvalidIndexFile);
        }
        Ok(Self {
            data: ByteSource::VecSource(bytes),
        })
    }

    /// Return the `hprof_offset` stored for `key`, or `None` if not found.
    ///
    /// Uses leftmost binary search on the sorted key field (bytes 0..8).
    pub fn find(&self, key: u64) -> Option<u64> {
        let data = self.data.as_ref();
        let n = data.len() / AUX_ENTRY_SIZE;
        let lo = leftmost_search(data, n, key);
        if lo < n && read_u64_le(data, lo * AUX_ENTRY_SIZE) == key {
            Some(read_u64_le(data, lo * AUX_ENTRY_SIZE + 8))
        } else {
            None
        }
    }

    /// Return the `(key, hprof_offset)` pair at position `idx` in the sorted index.
    ///
    /// Panics in debug builds if `idx >= self.len()`.
    pub fn entry_at(&self, idx: usize) -> (u64, u64) {
        let data = self.data.as_ref();
        let base = idx * AUX_ENTRY_SIZE;
        (read_u64_le(data, base), read_u64_le(data, base + 8))
    }

    /// Total number of records in this index.
    pub fn len(&self) -> usize {
        self.data.as_ref().len() / AUX_ENTRY_SIZE
    }

    /// Returns `true` if the index contains no records.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Typed alias for a frame index reader (sorted by `frame_id`).
pub type FrameIndexReader = AuxIndexReader;

/// Typed alias for a trace index reader (sorted by `trace_serial`).
pub type TraceIndexReader = AuxIndexReader;

/// Typed alias for a start-thread index reader (sorted by `thread_serial`).
pub type StartThreadIndexReader = AuxIndexReader;

/// Typed alias for an end-thread index reader (sorted by `thread_serial`).
pub type EndThreadIndexReader = AuxIndexReader;

/// Typed alias for an unload-class index reader (sorted by `class_serial`).
pub type UnloadClassIndexReader = AuxIndexReader;

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
    hprof_source: &impl MMapReader,
    record_index_source: &impl MMapReader,
    output: &mut impl MMapWriter,
) -> Result<u64, HprofError> {
    let hprof = HprofFile::from_source(hprof_source.open_mmap()?)?;
    let record_index = record_index_source.open_mmap()?;
    let id_size = hprof.header.id_size as usize;
    let hprof_data = hprof.data();
    let tag = u8::from(RecordTag::Frame);

    let mut entries: Vec<(u64, u64)> = Vec::new();
    for_each_entry(record_index.as_ref(), tag, |entry| {
        let body_start = entry.position as usize + RECORD_HEADER_SIZE;
        let frame_id = read_id_be(hprof_data, body_start, id_size)?;
        entries.push((frame_id, entry.position));
        Ok(())
    })?;

    entries.sort_unstable_by_key(|&(key, _)| key);
    let mut writer = output.create_writer()?;
    for &(key, offset) in &entries {
        write_entry(&mut writer, key, offset)?;
    }
    writer.flush()?;
    Ok(entries.len() as u64)
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
    hprof_source: &impl MMapReader,
    record_index_source: &impl MMapReader,
    output: &mut impl MMapWriter,
) -> Result<u64, HprofError> {
    let hprof = HprofFile::from_source(hprof_source.open_mmap()?)?;
    let record_index = record_index_source.open_mmap()?;
    let hprof_data = hprof.data();
    let tag = u8::from(RecordTag::Trace);

    let mut entries: Vec<(u64, u64)> = Vec::new();
    for_each_entry(record_index.as_ref(), tag, |entry| {
        let body_start = entry.position as usize + RECORD_HEADER_SIZE;
        let trace_serial = read_u32_be(hprof_data, body_start)? as u64;
        entries.push((trace_serial, entry.position));
        Ok(())
    })?;

    entries.sort_unstable_by_key(|&(key, _)| key);
    let mut writer = output.create_writer()?;
    for &(key, offset) in &entries {
        write_entry(&mut writer, key, offset)?;
    }
    writer.flush()?;
    Ok(entries.len() as u64)
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
    hprof_source: &impl MMapReader,
    record_index_source: &impl MMapReader,
    output: &mut impl MMapWriter,
) -> Result<u64, HprofError> {
    let hprof = HprofFile::from_source(hprof_source.open_mmap()?)?;
    let record_index = record_index_source.open_mmap()?;
    let hprof_data = hprof.data();
    let tag = u8::from(RecordTag::StartThread);

    let mut entries: Vec<(u64, u64)> = Vec::new();
    for_each_entry(record_index.as_ref(), tag, |entry| {
        let body_start = entry.position as usize + RECORD_HEADER_SIZE;
        let thread_serial = read_u32_be(hprof_data, body_start)? as u64;
        entries.push((thread_serial, entry.position));
        Ok(())
    })?;

    entries.sort_unstable_by_key(|&(key, _)| key);
    let mut writer = output.create_writer()?;
    for &(key, offset) in &entries {
        write_entry(&mut writer, key, offset)?;
    }
    writer.flush()?;
    Ok(entries.len() as u64)
}

/// Build an index of `HPROF_END_THREAD` records sorted by `thread_serial`.
///
/// Record body layout: `thread_serial(u32)`.
///
/// Returns the number of entries written.
pub fn build_end_thread_index(
    hprof_source: &impl MMapReader,
    record_index_source: &impl MMapReader,
    output: &mut impl MMapWriter,
) -> Result<u64, HprofError> {
    let hprof = HprofFile::from_source(hprof_source.open_mmap()?)?;
    let record_index = record_index_source.open_mmap()?;
    let hprof_data = hprof.data();
    let tag = u8::from(RecordTag::EndThread);

    let mut entries: Vec<(u64, u64)> = Vec::new();
    for_each_entry(record_index.as_ref(), tag, |entry| {
        let body_start = entry.position as usize + RECORD_HEADER_SIZE;
        let thread_serial = read_u32_be(hprof_data, body_start)? as u64;
        entries.push((thread_serial, entry.position));
        Ok(())
    })?;

    entries.sort_unstable_by_key(|&(key, _)| key);
    let mut writer = output.create_writer()?;
    for &(key, offset) in &entries {
        write_entry(&mut writer, key, offset)?;
    }
    writer.flush()?;
    Ok(entries.len() as u64)
}

/// Build an index of `HPROF_UNLOAD_CLASS` records sorted by `class_serial`.
///
/// Record body layout: `class_serial(u32)`.
///
/// Returns the number of entries written.
pub fn build_unload_class_index(
    hprof_source: &impl MMapReader,
    record_index_source: &impl MMapReader,
    output: &mut impl MMapWriter,
) -> Result<u64, HprofError> {
    let hprof = HprofFile::from_source(hprof_source.open_mmap()?)?;
    let record_index = record_index_source.open_mmap()?;
    let hprof_data = hprof.data();
    let tag = u8::from(RecordTag::UnloadClass);

    let mut entries: Vec<(u64, u64)> = Vec::new();
    for_each_entry(record_index.as_ref(), tag, |entry| {
        let body_start = entry.position as usize + RECORD_HEADER_SIZE;
        let class_serial = read_u32_be(hprof_data, body_start)? as u64;
        entries.push((class_serial, entry.position));
        Ok(())
    })?;

    entries.sort_unstable_by_key(|&(key, _)| key);
    let mut writer = output.create_writer()?;
    for &(key, offset) in &entries {
        write_entry(&mut writer, key, offset)?;
    }
    writer.flush()?;
    Ok(entries.len() as u64)
}

// ── Common helpers ────────────────────────────────────────────────────────────

/// Iterate over record index entries that match `tag`, calling `f` for each.
fn for_each_entry(
    record_index_data: &[u8],
    tag: u8,
    mut f: impl FnMut(&IndexEntry) -> Result<(), HprofError>,
) -> Result<(), HprofError> {
    let n = record_index_data.len() / INDEX_ENTRY_SIZE;
    for i in 0..n {
        let start = i * INDEX_ENTRY_SIZE;
        let bytes: [u8; INDEX_ENTRY_SIZE] = record_index_data[start..start + INDEX_ENTRY_SIZE]
            .try_into()
            .map_err(|_| HprofError::InvalidIndexFile)?;
        let entry = IndexEntry::from_bytes(&bytes);
        if entry.tag == tag {
            f(&entry)?;
        }
    }
    Ok(())
}

/// Write a single 16-byte (key, hprof_offset) entry.
fn write_entry(w: &mut impl Write, key: u64, hprof_offset: u64) -> Result<(), HprofError> {
    let mut buf = [0u8; AUX_ENTRY_SIZE];
    buf[0..8].copy_from_slice(&key.to_le_bytes());
    buf[8..16].copy_from_slice(&hprof_offset.to_le_bytes());
    w.write_all(&buf)?;
    Ok(())
}

// ── Binary search ─────────────────────────────────────────────────────────────

fn leftmost_search(data: &[u8], n: usize, target: u64) -> usize {
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if read_u64_le(data, mid * AUX_ENTRY_SIZE) < target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

// ── Byte helpers ──────────────────────────────────────────────────────────────

fn read_u64_le(data: &[u8], off: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&data[off..off + 8]);
    u64::from_le_bytes(b)
}

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

fn read_id_be(data: &[u8], off: usize, id_size: usize) -> Result<u64, HprofError> {
    match id_size {
        4 => {
            if off + 4 > data.len() {
                return Err(HprofError::UnexpectedEof(off));
            }
            Ok(u32::from_be_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]) as u64)
        }
        8 => {
            if off + 8 > data.len() {
                return Err(HprofError::UnexpectedEof(off));
            }
            Ok(u64::from_be_bytes([
                data[off],
                data[off + 1],
                data[off + 2],
                data[off + 3],
                data[off + 4],
                data[off + 5],
                data[off + 6],
                data[off + 7],
            ]))
        }
        _ => Err(HprofError::InvalidIdSize(id_size as u32)),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record_index::index_hprof;

    /// Linear search over a raw aux index buffer; returns `hprof_offset` for `key`.
    fn aux_find(buf: &[u8], key: u64) -> Option<u64> {
        let n = buf.len() / AUX_ENTRY_SIZE;
        for i in 0..n {
            let base = i * AUX_ENTRY_SIZE;
            let k = u64::from_le_bytes(buf[base..base + 8].try_into().unwrap());
            if k == key {
                return Some(u64::from_le_bytes(
                    buf[base + 8..base + 16].try_into().unwrap(),
                ));
            }
        }
        None
    }

    // ── Minimal hprof builders ────────────────────────────────────────────────

    fn hprof_header(id_size: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"JAVA PROFILE 1.0.2\0");
        buf.extend_from_slice(&id_size.to_be_bytes());
        buf.extend_from_slice(&0u64.to_be_bytes()); // timestamp
        buf
    }

    fn write_record(buf: &mut Vec<u8>, tag: u8, body: &[u8]) {
        buf.push(tag);
        buf.extend_from_slice(&0u32.to_be_bytes()); // time_offset
        buf.extend_from_slice(&(body.len() as u32).to_be_bytes());
        buf.extend_from_slice(body);
    }

    /// HPROF_FRAME: frame_id(8) method_name_id(8) method_sig_id(8)
    ///              source_file_id(8) class_serial(4) line_number(4)
    fn frame_body(frame_id: u64, class_serial: u32) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&frame_id.to_be_bytes());
        b.extend_from_slice(&0u64.to_be_bytes()); // method_name_id
        b.extend_from_slice(&0u64.to_be_bytes()); // method_sig_id
        b.extend_from_slice(&0u64.to_be_bytes()); // source_file_id
        b.extend_from_slice(&class_serial.to_be_bytes());
        b.extend_from_slice(&1i32.to_be_bytes()); // line_number
        b
    }

    /// HPROF_TRACE: trace_serial(4) thread_serial(4) num_frames(4) [frame_ids]
    fn trace_body(trace_serial: u32, thread_serial: u32, frame_ids: &[u64]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&trace_serial.to_be_bytes());
        b.extend_from_slice(&thread_serial.to_be_bytes());
        b.extend_from_slice(&(frame_ids.len() as u32).to_be_bytes());
        for &id in frame_ids {
            b.extend_from_slice(&id.to_be_bytes());
        }
        b
    }

    /// HPROF_START_THREAD: thread_serial(4) thread_id(8) stack_trace_serial(4)
    ///                     thread_name_id(8) group_name_id(8) parent_group_id(8)
    fn start_thread_body(thread_serial: u32, thread_id: u64) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&thread_serial.to_be_bytes());
        b.extend_from_slice(&thread_id.to_be_bytes());
        b.extend_from_slice(&0u32.to_be_bytes()); // stack_trace_serial
        b.extend_from_slice(&0u64.to_be_bytes()); // thread_name_id
        b.extend_from_slice(&0u64.to_be_bytes()); // group_name_id
        b.extend_from_slice(&0u64.to_be_bytes()); // parent_group_id
        b
    }

    fn end_thread_body(thread_serial: u32) -> Vec<u8> {
        thread_serial.to_be_bytes().to_vec()
    }

    fn unload_class_body(class_serial: u32) -> Vec<u8> {
        class_serial.to_be_bytes().to_vec()
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[test]
    fn frame_index_round_trip() {
        let mut hprof = hprof_header(8);
        write_record(&mut hprof, 0x04, &frame_body(0x100, 1));
        write_record(&mut hprof, 0x04, &frame_body(0x200, 2));
        write_record(&mut hprof, 0x04, &frame_body(0x50, 3)); // out of order to test sort

        let mut p1 = Vec::new();
        index_hprof(&hprof, &mut p1).unwrap();

        let mut out_buf = Vec::new();
        let n = build_frame_index(&hprof, &p1, &mut out_buf).unwrap();
        assert_eq!(n, 3);
        assert_eq!(out_buf.len(), 3 * AUX_ENTRY_SIZE);
        // sorted: 0x50, 0x100, 0x200
        assert!(aux_find(&out_buf, 0x50).is_some());
        assert!(aux_find(&out_buf, 0x100).is_some());
        assert!(aux_find(&out_buf, 0x200).is_some());
        assert!(aux_find(&out_buf, 0x999).is_none());
    }

    #[test]
    fn trace_index_round_trip() {
        let mut hprof = hprof_header(8);
        write_record(&mut hprof, 0x05, &trace_body(10, 1, &[0x100, 0x200]));
        write_record(&mut hprof, 0x05, &trace_body(5, 1, &[])); // lower serial, tests sort

        let mut p1 = Vec::new();
        index_hprof(&hprof, &mut p1).unwrap();

        let mut out_buf = Vec::new();
        let n = build_trace_index(&hprof, &p1, &mut out_buf).unwrap();
        assert_eq!(n, 2);
        assert!(aux_find(&out_buf, 5).is_some());
        assert!(aux_find(&out_buf, 10).is_some());
        assert!(aux_find(&out_buf, 99).is_none());
    }

    #[test]
    fn start_thread_index_round_trip() {
        let mut hprof = hprof_header(8);
        write_record(&mut hprof, 0x0A, &start_thread_body(3, 0xABC));
        write_record(&mut hprof, 0x0A, &start_thread_body(1, 0xDEF));

        let mut p1 = Vec::new();
        index_hprof(&hprof, &mut p1).unwrap();

        let mut out_buf = Vec::new();
        let n = build_start_thread_index(&hprof, &p1, &mut out_buf).unwrap();
        assert_eq!(n, 2);
        assert!(aux_find(&out_buf, 1).is_some());
        assert!(aux_find(&out_buf, 3).is_some());
        assert!(aux_find(&out_buf, 2).is_none());
    }

    #[test]
    fn end_thread_index_round_trip() {
        let mut hprof = hprof_header(8);
        write_record(&mut hprof, 0x0B, &end_thread_body(7));

        let mut p1 = Vec::new();
        index_hprof(&hprof, &mut p1).unwrap();

        let mut out_buf = Vec::new();
        let n = build_end_thread_index(&hprof, &p1, &mut out_buf).unwrap();
        assert_eq!(n, 1);
        assert!(aux_find(&out_buf, 7).is_some());
        assert!(aux_find(&out_buf, 1).is_none());
    }

    #[test]
    fn unload_class_index_round_trip() {
        let mut hprof = hprof_header(8);
        write_record(&mut hprof, 0x03, &unload_class_body(42));
        write_record(&mut hprof, 0x03, &unload_class_body(10));

        let mut p1 = Vec::new();
        index_hprof(&hprof, &mut p1).unwrap();

        let mut out_buf = Vec::new();
        let n = build_unload_class_index(&hprof, &p1, &mut out_buf).unwrap();
        assert_eq!(n, 2);
        assert!(aux_find(&out_buf, 10).is_some());
        assert!(aux_find(&out_buf, 42).is_some());
        assert!(aux_find(&out_buf, 99).is_none());
    }

    #[test]
    fn empty_index_is_empty() {
        let hprof = hprof_header(8); // no records
        let mut p1 = Vec::new();
        index_hprof(&hprof, &mut p1).unwrap();

        let mut out_buf = Vec::new();
        let n = build_frame_index(&hprof, &p1, &mut out_buf).unwrap();
        assert_eq!(n, 0);
        assert!(out_buf.is_empty());
        assert!(aux_find(&out_buf, 1).is_none());
    }

    #[test]
    fn find_returns_correct_hprof_offset() {
        // Verify that the stored offset points at the record tag byte.
        let mut hprof = hprof_header(8);
        let offset_before = hprof.len() as u64;
        write_record(&mut hprof, 0x04, &frame_body(0xABCD, 1));

        let mut p1 = Vec::new();
        index_hprof(&hprof, &mut p1).unwrap();

        let mut out_buf = Vec::new();
        build_frame_index(&hprof, &p1, &mut out_buf).unwrap();
        let stored_offset = aux_find(&out_buf, 0xABCD).unwrap();
        assert_eq!(
            stored_offset, offset_before,
            "offset should point at tag byte"
        );
        // The byte at that offset should be the FRAME tag 0x04.
        assert_eq!(hprof[stored_offset as usize], 0x04);
    }
}
