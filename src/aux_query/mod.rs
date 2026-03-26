//! High-level API for auxiliary hprof records.
//!
//! [`AuxRecordIndex`] is the entry point for accessing the auxiliary record
//! index files. It provides:
//!
//! * **Lookup by key** — `find_frame`, `find_trace`, `find_start_thread`,
//!   `was_thread_ended`, `was_class_unloaded`.
//! * **Name resolution** — `resolve_frame`, `resolve_thread` convert raw ID
//!   fields to `String` values using the UTF-8 name index.
//! * **Iteration** — `iter_frames`, `iter_traces`, `iter_start_threads`
//!   stream all records in key order without buffering them in memory.
//!
//! No data from the hprof file is ever loaded into memory: every lookup
//! fetches the `hprof_offset` from the mmap'd index file and then parses
//! the record directly from the mmap'd hprof file.
//!
//! ## Example
//!
//! ```rust,ignore
//! use hprof_toolkit::aux_query::AuxRecordIndex;
//!
//! let idx = AuxRecordIndex::open(
//!     hprof_path,
//!     frame_path, trace_path,
//!     start_thread_path, end_thread_path, unload_class_path,
//!     utf8_path,
//! )?;
//!
//! if let Some(frame) = idx.find_frame(0xDEAD)? {
//!     let resolved = idx.resolve_frame(&frame)?;
//!     println!("{}.{}:{}", resolved.source_file, resolved.method_name, resolved.line_number);
//! }
//!
//! for result in idx.iter_start_threads() {
//!     let thread = result?;
//!     let resolved = idx.resolve_thread(&thread)?;
//!     println!("Thread {}: {}", thread.thread_serial, resolved.thread_name);
//! }
//! ```

pub mod record;
pub mod thread_names;

pub use record::{Frame, LineNumber, ResolvedFrame, ResolvedThread, StartThread, Trace};

use crate::aux_index::AuxIndexReader;
use crate::heap_query::name_index::Utf8IndexReader;
use crate::hprof::{HprofError, HprofFile};
use crate::vfs::MMapReader;
use record::{parse_frame, parse_start_thread, parse_trace};

// ── AuxRecordIndex ────────────────────────────────────────────────────────────

/// High-level API for the auxiliary record indexes.
///
/// All data is accessed via memory-mapped files; no heap dump content is
/// loaded into process memory.
pub struct AuxRecordIndex {
    hprof: HprofFile,
    frames: AuxIndexReader,
    traces: AuxIndexReader,
    start_threads: AuxIndexReader,
    end_threads: AuxIndexReader,
    unload_classes: AuxIndexReader,
    utf8: Utf8IndexReader,
}

impl AuxRecordIndex {
    /// Open the hprof file and all auxiliary index files.
    ///
    /// Each index file must have been produced by the corresponding
    /// `build_*_index` function in [`crate::aux_index`].
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        hprof_source: &impl MMapReader,
        frame_source: &impl MMapReader,
        trace_source: &impl MMapReader,
        start_thread_source: &impl MMapReader,
        end_thread_source: &impl MMapReader,
        unload_class_source: &impl MMapReader,
        utf8_source: &impl MMapReader,
    ) -> Result<Self, HprofError> {
        Ok(Self {
            hprof: HprofFile::from_source(hprof_source.open_mmap()?)?,
            frames: AuxIndexReader::from_bytes(frame_source.open_mmap()?.as_ref().to_vec())?,
            traces: AuxIndexReader::from_bytes(trace_source.open_mmap()?.as_ref().to_vec())?,
            start_threads: AuxIndexReader::from_bytes(
                start_thread_source.open_mmap()?.as_ref().to_vec(),
            )?,
            end_threads: AuxIndexReader::from_bytes(
                end_thread_source.open_mmap()?.as_ref().to_vec(),
            )?,
            unload_classes: AuxIndexReader::from_bytes(
                unload_class_source.open_mmap()?.as_ref().to_vec(),
            )?,
            utf8: Utf8IndexReader::from_bytes(utf8_source.open_mmap()?.as_ref().to_vec())?,
        })
    }

    /// Return the hprof identifier size in bytes (4 or 8).
    pub fn id_size(&self) -> u32 {
        self.hprof.header.id_size
    }

    // ── Name resolution ───────────────────────────────────────────────────────

    /// Look up the UTF-8 string for `name_id` in the name index.
    ///
    /// Returns `None` when `name_id` is not present.
    pub fn lookup_name(&self, name_id: u64) -> Result<Option<String>, HprofError> {
        self.utf8.lookup(&self.hprof, name_id)
    }

    // ── Frame lookups ─────────────────────────────────────────────────────────

    /// Find and parse the `HPROF_FRAME` record for `frame_id`.
    ///
    /// Returns `None` when the frame is not in the index.
    pub fn find_frame(&self, frame_id: u64) -> Result<Option<Frame>, HprofError> {
        match self.frames.find(frame_id) {
            Some(offset) => Ok(Some(parse_frame(
                self.hprof.data(),
                offset as usize,
                self.hprof.header.id_size as usize,
            )?)),
            None => Ok(None),
        }
    }

    /// Resolve all name IDs in `frame` to `String` values.
    pub fn resolve_frame(&self, frame: &Frame) -> Result<ResolvedFrame, HprofError> {
        let method_name = self.lookup_name(frame.method_name_id)?.unwrap_or_default();
        let method_signature = self.lookup_name(frame.method_sig_id)?.unwrap_or_default();
        let source_file = self.lookup_name(frame.source_file_id)?.unwrap_or_default();
        Ok(ResolvedFrame {
            frame_id: frame.frame_id,
            method_name,
            method_signature,
            source_file,
            class_serial: frame.class_serial,
            line_number: LineNumber::from_raw(frame.line_number),
        })
    }

    /// Iterate over all frames in the index in ascending `frame_id` order.
    pub fn iter_frames(&self) -> FrameIter<'_> {
        FrameIter {
            index: self,
            pos: 0,
        }
    }

    // ── Trace lookups ─────────────────────────────────────────────────────────

    /// Find and parse the `HPROF_TRACE` record for `trace_serial`.
    ///
    /// Returns `None` when the trace is not in the index.
    pub fn find_trace(&self, trace_serial: u32) -> Result<Option<Trace>, HprofError> {
        match self.traces.find(trace_serial as u64) {
            Some(offset) => Ok(Some(parse_trace(
                self.hprof.data(),
                offset as usize,
                self.hprof.header.id_size as usize,
            )?)),
            None => Ok(None),
        }
    }

    /// Parse every frame in `trace` and return them in order.
    ///
    /// Frames missing from the index (e.g. pruned by the JVM) are silently
    /// skipped.
    pub fn trace_frames(&self, trace: &Trace) -> Result<Vec<Frame>, HprofError> {
        let mut out = Vec::with_capacity(trace.frame_ids.len());
        for &fid in &trace.frame_ids {
            if let Some(frame) = self.find_frame(fid)? {
                out.push(frame);
            }
        }
        Ok(out)
    }

    /// Iterate over all traces in the index in ascending `trace_serial` order.
    pub fn iter_traces(&self) -> TraceIter<'_> {
        TraceIter {
            index: self,
            pos: 0,
        }
    }

    // ── Thread lookups ────────────────────────────────────────────────────────

    /// Find and parse the `HPROF_START_THREAD` record for `thread_serial`.
    ///
    /// Returns `None` when the thread is not in the index.
    pub fn find_start_thread(&self, thread_serial: u32) -> Result<Option<StartThread>, HprofError> {
        match self.start_threads.find(thread_serial as u64) {
            Some(offset) => Ok(Some(parse_start_thread(
                self.hprof.data(),
                offset as usize,
                self.hprof.header.id_size as usize,
            )?)),
            None => Ok(None),
        }
    }

    /// Resolve all name IDs in `thread` to `String` values.
    pub fn resolve_thread(&self, thread: &StartThread) -> Result<ResolvedThread, HprofError> {
        let thread_name = self.lookup_name(thread.thread_name_id)?.unwrap_or_default();
        let thread_group_name = self
            .lookup_name(thread.thread_group_name_id)?
            .unwrap_or_default();
        let thread_parent_group_name = self
            .lookup_name(thread.thread_parent_group_name_id)?
            .unwrap_or_default();
        Ok(ResolvedThread {
            thread_serial: thread.thread_serial,
            thread_id: thread.thread_id,
            stack_trace_serial: thread.stack_trace_serial,
            thread_name,
            thread_group_name,
            thread_parent_group_name,
        })
    }

    /// Returns `true` if a `HPROF_END_THREAD` record exists for `thread_serial`.
    pub fn was_thread_ended(&self, thread_serial: u32) -> bool {
        self.end_threads.find(thread_serial as u64).is_some()
    }

    /// Iterate over all start-thread records in ascending `thread_serial` order.
    pub fn iter_start_threads(&self) -> StartThreadIter<'_> {
        StartThreadIter {
            index: self,
            pos: 0,
        }
    }

    // ── Unload-class lookup ───────────────────────────────────────────────────

    /// Returns `true` if a `HPROF_UNLOAD_CLASS` record exists for `class_serial`.
    pub fn was_class_unloaded(&self, class_serial: u32) -> bool {
        self.unload_classes.find(class_serial as u64).is_some()
    }

    // ── Heap-based thread name resolution ─────────────────────────────────────

    /// Build a map from `thread_serial` → thread name by reading
    /// `java.lang.Thread` heap objects.
    ///
    /// Uses `GC_ROOT_THREAD_OBJ` sub-records to correlate `thread_serial`
    /// with a heap object, then follows the `name` field chain through the
    /// heap.  Returns an empty map when no thread-object roots are present.
    ///
    /// `record_index_source` — record index (for scanning heap dumps).
    /// `object_store_source` — object store index (for object lookup).
    pub fn collect_thread_names_from_heap(
        &self,
        record_index_source: &impl MMapReader,
        object_store_source: &impl MMapReader,
    ) -> Result<std::collections::HashMap<u32, String>, HprofError> {
        thread_names::collect_thread_names(
            &self.hprof,
            record_index_source,
            object_store_source,
            &self.utf8,
        )
    }
}

// ── Iterators ─────────────────────────────────────────────────────────────────

/// Iterator over all [`Frame`] records in ascending `frame_id` order.
pub struct FrameIter<'a> {
    index: &'a AuxRecordIndex,
    pos: usize,
}

impl Iterator for FrameIter<'_> {
    type Item = Result<Frame, HprofError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.index.frames.len() {
            return None;
        }
        let (_key, hprof_offset) = self.index.frames.entry_at(self.pos);
        self.pos += 1;
        Some(parse_frame(
            self.index.hprof.data(),
            hprof_offset as usize,
            self.index.hprof.header.id_size as usize,
        ))
    }
}

/// Iterator over all [`Trace`] records in ascending `trace_serial` order.
pub struct TraceIter<'a> {
    index: &'a AuxRecordIndex,
    pos: usize,
}

impl Iterator for TraceIter<'_> {
    type Item = Result<Trace, HprofError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.index.traces.len() {
            return None;
        }
        let (_key, hprof_offset) = self.index.traces.entry_at(self.pos);
        self.pos += 1;
        Some(parse_trace(
            self.index.hprof.data(),
            hprof_offset as usize,
            self.index.hprof.header.id_size as usize,
        ))
    }
}

/// Iterator over all [`StartThread`] records in ascending `thread_serial` order.
pub struct StartThreadIter<'a> {
    index: &'a AuxRecordIndex,
    pos: usize,
}

impl Iterator for StartThreadIter<'_> {
    type Item = Result<StartThread, HprofError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.index.start_threads.len() {
            return None;
        }
        let (_key, hprof_offset) = self.index.start_threads.entry_at(self.pos);
        self.pos += 1;
        Some(parse_start_thread(
            self.index.hprof.data(),
            hprof_offset as usize,
            self.index.hprof.header.id_size as usize,
        ))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aux_index::{
        build_end_thread_index, build_frame_index, build_start_thread_index, build_trace_index,
        build_unload_class_index,
    };
    use crate::heap_query::build_name_indexes;
    use crate::record_index::index_hprof;

    // ── Minimal hprof builder ─────────────────────────────────────────────────

    fn write_record(buf: &mut Vec<u8>, tag: u8, body: &[u8]) {
        buf.push(tag);
        buf.extend_from_slice(&0u32.to_be_bytes());
        buf.extend_from_slice(&(body.len() as u32).to_be_bytes());
        buf.extend_from_slice(body);
    }

    /// Build a minimal hprof with:
    /// - UTF8(1, "main"), UTF8(2, "MyClass"), UTF8(3, "()V"), UTF8(4, "MyClass.java")
    /// - FRAME(id=0x10, method_name=1, sig=3, src=4, class_serial=99, line=5)
    /// - TRACE(serial=1, thread_serial=1, frames=[0x10])
    /// - START_THREAD(serial=1, thread_id=0xABC, trace_serial=1, name=1)
    /// - END_THREAD(serial=1)
    /// - UNLOAD_CLASS(serial=77)
    fn build_test_hprof() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"JAVA PROFILE 1.0.2\0");
        buf.extend_from_slice(&8u32.to_be_bytes()); // id_size = 8
        buf.extend_from_slice(&0u64.to_be_bytes()); // timestamp

        // UTF8 records
        let utf8 = |buf: &mut Vec<u8>, id: u64, s: &[u8]| {
            let mut body = Vec::new();
            body.extend_from_slice(&id.to_be_bytes());
            body.extend_from_slice(s);
            write_record(buf, 0x01, &body);
        };
        utf8(&mut buf, 1, b"main");
        utf8(&mut buf, 2, b"MyClass");
        utf8(&mut buf, 3, b"()V");
        utf8(&mut buf, 4, b"MyClass.java");

        // FRAME record
        let mut frame_body = Vec::new();
        frame_body.extend_from_slice(&0x10u64.to_be_bytes()); // frame_id
        frame_body.extend_from_slice(&1u64.to_be_bytes()); // method_name_id
        frame_body.extend_from_slice(&3u64.to_be_bytes()); // method_sig_id
        frame_body.extend_from_slice(&4u64.to_be_bytes()); // source_file_id
        frame_body.extend_from_slice(&99u32.to_be_bytes()); // class_serial
        frame_body.extend_from_slice(&5i32.to_be_bytes()); // line_number
        write_record(&mut buf, 0x04, &frame_body);

        // TRACE record
        let mut trace_body = Vec::new();
        trace_body.extend_from_slice(&1u32.to_be_bytes()); // trace_serial
        trace_body.extend_from_slice(&1u32.to_be_bytes()); // thread_serial
        trace_body.extend_from_slice(&1u32.to_be_bytes()); // num_frames
        trace_body.extend_from_slice(&0x10u64.to_be_bytes()); // frame_id
        write_record(&mut buf, 0x05, &trace_body);

        // START_THREAD record
        let mut st_body = Vec::new();
        st_body.extend_from_slice(&1u32.to_be_bytes()); // thread_serial
        st_body.extend_from_slice(&0xABCu64.to_be_bytes()); // thread_id
        st_body.extend_from_slice(&1u32.to_be_bytes()); // stack_trace_serial
        st_body.extend_from_slice(&1u64.to_be_bytes()); // thread_name_id
        st_body.extend_from_slice(&2u64.to_be_bytes()); // group_name_id
        st_body.extend_from_slice(&0u64.to_be_bytes()); // parent_group_id
        write_record(&mut buf, 0x0A, &st_body);

        // END_THREAD record
        write_record(&mut buf, 0x0B, &1u32.to_be_bytes());

        // UNLOAD_CLASS record
        write_record(&mut buf, 0x03, &77u32.to_be_bytes());

        buf
    }

    /// Build all auxiliary indexes and open an [`AuxRecordIndex`].
    fn build_all(hprof_data: &[u8]) -> AuxRecordIndex {
        let hprof = hprof_data.to_vec();
        let mut p1 = Vec::new();
        let mut frame_buf = Vec::new();
        let mut trace_buf = Vec::new();
        let mut st_buf = Vec::new();
        let mut et_buf = Vec::new();
        let mut uc_buf = Vec::new();
        let mut utf8_buf = Vec::new();
        let mut lc_buf = Vec::new();

        index_hprof(&hprof, &mut p1).unwrap();
        build_frame_index(&hprof, &p1, &mut frame_buf).unwrap();
        build_trace_index(&hprof, &p1, &mut trace_buf).unwrap();
        build_start_thread_index(&hprof, &p1, &mut st_buf).unwrap();
        build_end_thread_index(&hprof, &p1, &mut et_buf).unwrap();
        build_unload_class_index(&hprof, &p1, &mut uc_buf).unwrap();
        build_name_indexes(&hprof, &p1, &mut utf8_buf, &mut lc_buf).unwrap();

        AuxRecordIndex::open(
            &hprof, &frame_buf, &trace_buf, &st_buf, &et_buf, &uc_buf, &utf8_buf,
        )
        .unwrap()
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[test]
    fn find_frame_and_resolve() {
        let hprof_data = build_test_hprof();
        let idx = build_all(&hprof_data);

        let frame = idx.find_frame(0x10).unwrap().unwrap();
        assert_eq!(frame.frame_id, 0x10);
        assert_eq!(frame.class_serial, 99);
        assert_eq!(frame.line_number, 5);

        let resolved = idx.resolve_frame(&frame).unwrap();
        assert_eq!(resolved.method_name, "main");
        assert_eq!(resolved.method_signature, "()V");
        assert_eq!(resolved.source_file, "MyClass.java");
        assert_eq!(resolved.line_number, LineNumber::Line(5));
    }

    #[test]
    fn find_frame_missing_returns_none() {
        let hprof_data = build_test_hprof();
        let idx = build_all(&hprof_data);
        assert!(idx.find_frame(0xDEAD).unwrap().is_none());
    }

    #[test]
    fn find_trace_and_frames() {
        let hprof_data = build_test_hprof();
        let idx = build_all(&hprof_data);

        let trace = idx.find_trace(1).unwrap().unwrap();
        assert_eq!(trace.trace_serial, 1);
        assert_eq!(trace.thread_serial, 1);
        assert_eq!(trace.frame_ids, vec![0x10]);

        let frames = idx.trace_frames(&trace).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].frame_id, 0x10);
    }

    #[test]
    fn find_trace_missing_returns_none() {
        let hprof_data = build_test_hprof();
        let idx = build_all(&hprof_data);
        assert!(idx.find_trace(999).unwrap().is_none());
    }

    #[test]
    fn find_start_thread_and_resolve() {
        let hprof_data = build_test_hprof();
        let idx = build_all(&hprof_data);

        let thread = idx.find_start_thread(1).unwrap().unwrap();
        assert_eq!(thread.thread_serial, 1);
        assert_eq!(thread.thread_id, 0xABC);
        assert_eq!(thread.stack_trace_serial, 1);

        let resolved = idx.resolve_thread(&thread).unwrap();
        assert_eq!(resolved.thread_name, "main");
        assert_eq!(resolved.thread_group_name, "MyClass");
    }

    #[test]
    fn was_thread_ended() {
        let hprof_data = build_test_hprof();
        let idx = build_all(&hprof_data);
        assert!(idx.was_thread_ended(1));
        assert!(!idx.was_thread_ended(99));
    }

    #[test]
    fn was_class_unloaded() {
        let hprof_data = build_test_hprof();
        let idx = build_all(&hprof_data);
        assert!(idx.was_class_unloaded(77));
        assert!(!idx.was_class_unloaded(1));
    }

    #[test]
    fn iter_frames_yields_all() {
        let hprof_data = build_test_hprof();
        let idx = build_all(&hprof_data);
        let frames: Vec<_> = idx.iter_frames().collect::<Result<_, _>>().unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].frame_id, 0x10);
    }

    #[test]
    fn iter_traces_yields_all() {
        let hprof_data = build_test_hprof();
        let idx = build_all(&hprof_data);
        let traces: Vec<_> = idx.iter_traces().collect::<Result<_, _>>().unwrap();
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].trace_serial, 1);
    }

    #[test]
    fn iter_start_threads_yields_all() {
        let hprof_data = build_test_hprof();
        let idx = build_all(&hprof_data);
        let threads: Vec<_> = idx.iter_start_threads().collect::<Result<_, _>>().unwrap();
        assert_eq!(threads.len(), 1);
        assert_eq!(threads[0].thread_serial, 1);
    }

    #[test]
    fn lookup_name_works() {
        let hprof_data = build_test_hprof();
        let idx = build_all(&hprof_data);
        assert_eq!(idx.lookup_name(1).unwrap(), Some("main".to_string()));
        assert_eq!(idx.lookup_name(9999).unwrap(), None);
    }
}
