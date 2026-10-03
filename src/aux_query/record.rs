//! Parsed record types for auxiliary hprof records.
//!
//! Each struct corresponds to one top-level hprof record type in the
//! auxiliary record indexes.  All fields are owned values; the hprof mmap is only touched
//! during parsing and never held across call boundaries.

use crate::hprof::HprofError;

// ── Record headers ────────────────────────────────────────────────────────────

/// Byte size of the 9-byte top-level record header.
pub(crate) const RECORD_HEADER_SIZE: usize = 9;

// ── Parsed records ────────────────────────────────────────────────────────────

/// A parsed `HPROF_FRAME` record.
///
/// Mirrors the hprof record; not every field has a reader yet.
///
/// Names and signatures are stored as `name_id` references; use
/// [`crate::aux_query::AuxRecordIndex::lookup_name`] to resolve them.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Frame {
    /// Unique identifier for this stack frame.
    pub frame_id: u64,
    /// UTF-8 name_id of the method name.
    pub method_name_id: u64,
    /// UTF-8 name_id of the method descriptor / signature.
    pub method_sig_id: u64,
    /// UTF-8 name_id of the source file name, or 0 if unknown.
    pub source_file_id: u64,
    /// Serial number of the class containing this method.
    pub class_serial: u32,
    /// Source line number, or a negative sentinel:
    /// * `-1` unknown location  * `-2` compiled method  * `-3` native method.
    pub line_number: i32,
}

/// A parsed `HPROF_TRACE` record.
///
/// `frame_ids` is bounded by `num_frames` in the record — a small number
/// even for deep stack traces — so storing it as a `Vec` is safe.
///
/// Mirrors the hprof record; not every field has a reader yet.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Trace {
    /// Unique serial number for this stack trace.
    pub trace_serial: u32,
    /// Serial number of the thread that produced this trace.
    pub thread_serial: u32,
    /// Ordered list of [`Frame::frame_id`] values, outermost first.
    pub frame_ids: Vec<u64>,
}

/// A parsed `HPROF_START_THREAD` record.
///
/// Mirrors the hprof record; not every field has a reader yet.
///
/// Name IDs are resolved via
/// [`crate::aux_query::AuxRecordIndex::lookup_name`].
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct StartThread {
    /// Unique serial number for this thread.
    pub thread_serial: u32,
    /// Object ID of the `java.lang.Thread` instance.
    pub thread_id: u64,
    /// Serial number of the stack trace at the time of thread start.
    pub stack_trace_serial: u32,
    /// UTF-8 name_id of the thread name.
    pub thread_name_id: u64,
    /// UTF-8 name_id of the thread group name.
    pub thread_group_name_id: u64,
    /// UTF-8 name_id of the parent thread group name.
    pub thread_parent_group_name_id: u64,
}

// ── Resolved types ────────────────────────────────────────────────────────────

/// The source line number of a stack frame, with the hprof sentinels decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineNumber {
    /// A positive source line number.
    Line(u32),
    /// No line information was available (`0`).
    NoInfo,
    /// The method location is unknown (`-1`).
    Unknown,
    /// The method was compiled; no line info (`-2`).
    Compiled,
    /// The method is native (`-3`).
    Native,
}

impl LineNumber {
    pub(crate) fn from_raw(v: i32) -> Self {
        match v {
            n if n > 0 => Self::Line(n as u32),
            0 => Self::NoInfo,
            -1 => Self::Unknown,
            -2 => Self::Compiled,
            -3 => Self::Native,
            _ => Self::Unknown,
        }
    }
}

/// A [`Frame`] with all name IDs resolved to `String` values.
#[derive(Debug, Clone)]
pub struct ResolvedFrame {
    /// Method name (resolved from UTF-8 index).
    pub method_name: String,
    /// Method descriptor / signature (resolved from UTF-8 index).
    pub method_signature: String,
    /// Source file name (resolved from UTF-8 index), or empty if unknown.
    pub source_file: String,
    /// Serial number of the class containing this method.
    pub class_serial: u32,
    /// Source line information.
    pub line_number: LineNumber,
}

/// A [`StartThread`] with the thread name resolved to a `String`.
#[derive(Debug, Clone)]
pub struct ResolvedThread {
    /// Thread name (resolved from UTF-8 index).
    pub thread_name: String,
    /// Thread group name (resolved from UTF-8 index).
    pub thread_group_name: String,
}

// ── Parsers ───────────────────────────────────────────────────────────────────

/// Parse a `HPROF_FRAME` record whose tag byte is at `record_offset` in `data`.
///
/// Body layout (all big-endian):
/// ```text
/// frame_id(ID)  method_name_id(ID)  method_sig_id(ID)  source_file_id(ID)
/// class_serial(u32)  line_number(i32)
/// ```
pub fn parse_frame(data: &[u8], record_offset: usize, id_size: usize) -> Result<Frame, HprofError> {
    let body = record_offset + RECORD_HEADER_SIZE;
    let min_len = 4 * id_size + 4 + 4;
    if body + min_len > data.len() {
        return Err(HprofError::UnexpectedEof(body));
    }
    let mut off = body;
    let frame_id = read_id(data, off, id_size)?;
    off += id_size;
    let method_name_id = read_id(data, off, id_size)?;
    off += id_size;
    let method_sig_id = read_id(data, off, id_size)?;
    off += id_size;
    let source_file_id = read_id(data, off, id_size)?;
    off += id_size;
    let class_serial = read_u32_be(data, off);
    off += 4;
    let line_number = read_u32_be(data, off) as i32;
    Ok(Frame {
        frame_id,
        method_name_id,
        method_sig_id,
        source_file_id,
        class_serial,
        line_number,
    })
}

/// Parse a `HPROF_TRACE` record whose tag byte is at `record_offset` in `data`.
///
/// Body layout (all big-endian):
/// ```text
/// trace_serial(u32)  thread_serial(u32)  num_frames(u32)
/// [frame_id(ID); num_frames]
/// ```
pub fn parse_trace(data: &[u8], record_offset: usize, id_size: usize) -> Result<Trace, HprofError> {
    let body = record_offset + RECORD_HEADER_SIZE;
    if body + 12 > data.len() {
        return Err(HprofError::UnexpectedEof(body));
    }
    let trace_serial = read_u32_be(data, body);
    let thread_serial = read_u32_be(data, body + 4);
    let num_frames = read_u32_be(data, body + 8) as usize;
    let frames_start = body + 12;
    if frames_start + num_frames * id_size > data.len() {
        return Err(HprofError::UnexpectedEof(frames_start));
    }
    let mut frame_ids = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        frame_ids.push(read_id(data, frames_start + i * id_size, id_size)?);
    }
    Ok(Trace {
        trace_serial,
        thread_serial,
        frame_ids,
    })
}

/// Parse a `HPROF_START_THREAD` record whose tag byte is at `record_offset`.
///
/// Body layout (all big-endian):
/// ```text
/// thread_serial(u32)  thread_id(ID)  stack_trace_serial(u32)
/// thread_name_id(ID)  group_name_id(ID)  parent_group_id(ID)
/// ```
pub fn parse_start_thread(
    data: &[u8],
    record_offset: usize,
    id_size: usize,
) -> Result<StartThread, HprofError> {
    let body = record_offset + RECORD_HEADER_SIZE;
    let min_len = 4 + id_size + 4 + 3 * id_size;
    if body + min_len > data.len() {
        return Err(HprofError::UnexpectedEof(body));
    }
    let thread_serial = read_u32_be(data, body);
    let mut off = body + 4;
    let thread_id = read_id(data, off, id_size)?;
    off += id_size;
    let stack_trace_serial = read_u32_be(data, off);
    off += 4;
    let thread_name_id = read_id(data, off, id_size)?;
    off += id_size;
    let thread_group_name_id = read_id(data, off, id_size)?;
    off += id_size;
    let thread_parent_group_name_id = read_id(data, off, id_size)?;
    Ok(StartThread {
        thread_serial,
        thread_id,
        stack_trace_serial,
        thread_name_id,
        thread_group_name_id,
        thread_parent_group_name_id,
    })
}

// ── Byte helpers ──────────────────────────────────────────────────────────────

fn read_u32_be(data: &[u8], off: usize) -> u32 {
    u32::from_be_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
}

pub(crate) fn read_id(data: &[u8], off: usize, id_size: usize) -> Result<u64, HprofError> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::HprofBuilder;

    #[test]
    fn parse_frame_id8() {
        let mut b = HprofBuilder::new(8);
        let off = b.next_record_position() as usize;
        let data = b.frame(0xABCD, 1, 2, 3, 42, 10).build();
        let frame = parse_frame(&data, off, 8).unwrap();
        assert_eq!(frame.frame_id, 0xABCD);
        assert_eq!(frame.method_name_id, 1);
        assert_eq!(frame.class_serial, 42);
        assert_eq!(frame.line_number, 10);
    }

    #[test]
    fn parse_trace_id8() {
        let mut b = HprofBuilder::new(8);
        let off = b.next_record_position() as usize;
        let data = b.trace(99, 1, &[0x100, 0x200]).build();
        let trace = parse_trace(&data, off, 8).unwrap();
        assert_eq!(trace.trace_serial, 99);
        assert_eq!(trace.thread_serial, 1);
        assert_eq!(trace.frame_ids, vec![0x100, 0x200]);
    }

    #[test]
    fn parse_trace_empty_frames() {
        let mut b = HprofBuilder::new(8);
        let off = b.next_record_position() as usize;
        let data = b.trace(5, 1, &[]).build();
        let trace = parse_trace(&data, off, 8).unwrap();
        assert!(trace.frame_ids.is_empty());
    }

    #[test]
    fn parse_start_thread_id8() {
        let mut b = HprofBuilder::new(8);
        let off = b.next_record_position() as usize;
        let data = b.start_thread(7, 0xDEAD, 3, 10, 20, 30).build();
        let st = parse_start_thread(&data, off, 8).unwrap();
        assert_eq!(st.thread_serial, 7);
        assert_eq!(st.thread_id, 0xDEAD);
        assert_eq!(st.stack_trace_serial, 3);
        assert_eq!(st.thread_name_id, 10);
        assert_eq!(st.thread_group_name_id, 20);
        assert_eq!(st.thread_parent_group_name_id, 30);
    }

    #[test]
    fn parse_frame_id4() {
        let mut b = HprofBuilder::new(4);
        let off = b.next_record_position() as usize;
        let data = b.frame(0xABCD, 1, 2, 3, 42, -3).build();
        let frame = parse_frame(&data, off, 4).unwrap();
        assert_eq!(frame.frame_id, 0xABCD);
        assert_eq!(frame.source_file_id, 3);
        assert_eq!(LineNumber::from_raw(frame.line_number), LineNumber::Native);
    }

    #[test]
    fn line_number_from_raw() {
        assert_eq!(LineNumber::from_raw(42), LineNumber::Line(42));
        assert_eq!(LineNumber::from_raw(0), LineNumber::NoInfo);
        assert_eq!(LineNumber::from_raw(-1), LineNumber::Unknown);
        assert_eq!(LineNumber::from_raw(-2), LineNumber::Compiled);
        assert_eq!(LineNumber::from_raw(-3), LineNumber::Native);
    }
}
