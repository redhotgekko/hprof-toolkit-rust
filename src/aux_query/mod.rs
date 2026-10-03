//! High-level API for auxiliary hprof records.
//!
//! [`AuxRecordIndex`] is the entry point for accessing the auxiliary record
//! index files. It provides:
//!
//! * **Lookup by key** — `find_frame`, `find_trace`,
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
//! This type is internal.  Use the [`crate::query::HeapQuery`] methods
//! (`find_frame`, `iter_threads`, `threads`, `thread_stack`, …), which wrap
//! it and are what the examples and tests exercise.

pub mod record;

pub use record::{Frame, LineNumber, ResolvedFrame, ResolvedThread, StartThread, Trace};

use crate::aux_index::AuxIndexReader;
use crate::heap_query::name_index::Utf8IndexReader;
use crate::hprof::{HprofError, HprofFile, HprofHeader};
use crate::index::ByteSource;
use record::{parse_frame, parse_start_thread, parse_trace};
use std::sync::Arc;

// ── AuxRecordIndex ────────────────────────────────────────────────────────────

/// High-level API for the auxiliary record indexes.
///
/// All data is stored in owned [`ByteSource`] buffers (either memory maps or
/// in-memory vecs); no heap dump content is loaded into process memory beyond
/// what is already in the owned buffers.  The hprof bytes and the UTF-8 name
/// index are shared with [`crate::query::HeapQuery`] rather than mapped twice.
pub(crate) struct AuxRecordIndex {
    hprof_data: Arc<ByteSource>,
    hprof_header: HprofHeader,
    frame_data: ByteSource,
    trace_data: ByteSource,
    start_thread_data: ByteSource,
    end_thread_data: ByteSource,
    unload_class_data: ByteSource,
    utf8_data: Arc<ByteSource>,
}

impl AuxRecordIndex {
    /// Open the auxiliary record indexes.
    ///
    /// Each index must have been produced by the corresponding
    /// `build_*_index` function in [`crate::aux_index`].
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        hprof_source: Arc<ByteSource>,
        frame_source: ByteSource,
        trace_source: ByteSource,
        start_thread_source: ByteSource,
        end_thread_source: ByteSource,
        unload_class_source: ByteSource,
        utf8_source: Arc<ByteSource>,
    ) -> Result<Self, HprofError> {
        let hprof_header = HprofHeader::parse((*hprof_source).as_ref())?;
        // Validate all indexes at construction time.
        AuxIndexReader::from_ref(frame_source.as_ref())?;
        AuxIndexReader::from_ref(trace_source.as_ref())?;
        AuxIndexReader::from_ref(start_thread_source.as_ref())?;
        AuxIndexReader::from_ref(end_thread_source.as_ref())?;
        AuxIndexReader::from_ref(unload_class_source.as_ref())?;
        Utf8IndexReader::from_ref((*utf8_source).as_ref())?;
        Ok(Self {
            hprof_data: hprof_source,
            hprof_header,
            frame_data: frame_source,
            trace_data: trace_source,
            start_thread_data: start_thread_source,
            end_thread_data: end_thread_source,
            unload_class_data: unload_class_source,
            utf8_data: utf8_source,
        })
    }

    // ── Short-lived reader helpers ────────────────────────────────────────────

    fn hprof_bytes(&self) -> &[u8] {
        (*self.hprof_data).as_ref()
    }

    fn utf8_bytes(&self) -> &[u8] {
        (*self.utf8_data).as_ref()
    }

    fn hprof_file(&self) -> HprofFile<'_> {
        HprofFile::from_parts(self.hprof_bytes(), &self.hprof_header)
    }

    fn utf8_reader(&self) -> Result<Utf8IndexReader<'_>, HprofError> {
        Utf8IndexReader::from_ref(self.utf8_bytes())
    }

    fn frames(&self) -> AuxIndexReader<'_> {
        AuxIndexReader::from_slice(self.frame_data.as_ref())
    }

    fn traces(&self) -> AuxIndexReader<'_> {
        AuxIndexReader::from_slice(self.trace_data.as_ref())
    }

    fn start_threads(&self) -> AuxIndexReader<'_> {
        AuxIndexReader::from_slice(self.start_thread_data.as_ref())
    }

    fn end_threads(&self) -> AuxIndexReader<'_> {
        AuxIndexReader::from_slice(self.end_thread_data.as_ref())
    }

    fn unload_classes(&self) -> AuxIndexReader<'_> {
        AuxIndexReader::from_slice(self.unload_class_data.as_ref())
    }

    // ── Name resolution ───────────────────────────────────────────────────────

    /// Look up the UTF-8 string for `name_id` in the name index.
    ///
    /// Returns `None` when `name_id` is not present.
    pub fn lookup_name(&self, name_id: u64) -> Result<Option<String>, HprofError> {
        let hprof = self.hprof_file();
        self.utf8_reader()?.lookup(&hprof, name_id)
    }

    // ── Frame lookups ─────────────────────────────────────────────────────────

    /// Find and parse the `HPROF_FRAME` record for `frame_id`.
    ///
    /// Returns `None` when the frame is not in the index.
    pub fn find_frame(&self, frame_id: u64) -> Result<Option<Frame>, HprofError> {
        match self.frames().find(frame_id) {
            Some(offset) => Ok(Some(parse_frame(
                self.hprof_bytes(),
                offset as usize,
                self.hprof_header.id_size as usize,
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
            hprof_data: self.hprof_bytes(),
            id_size: self.hprof_header.id_size as usize,
            frames: self.frames(),
            pos: 0,
        }
    }

    // ── Trace lookups ─────────────────────────────────────────────────────────

    /// Find and parse the `HPROF_TRACE` record for `trace_serial`.
    ///
    /// Returns `None` when the trace is not in the index.
    pub fn find_trace(&self, trace_serial: u32) -> Result<Option<Trace>, HprofError> {
        match self.traces().find(trace_serial as u64) {
            Some(offset) => Ok(Some(parse_trace(
                self.hprof_bytes(),
                offset as usize,
                self.hprof_header.id_size as usize,
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
            hprof_data: self.hprof_bytes(),
            id_size: self.hprof_header.id_size as usize,
            traces: self.traces(),
            pos: 0,
        }
    }

    // ── Thread lookups ────────────────────────────────────────────────────────

    /// Resolve all name IDs in `thread` to `String` values.
    pub fn resolve_thread(&self, thread: &StartThread) -> Result<ResolvedThread, HprofError> {
        let thread_name = self.lookup_name(thread.thread_name_id)?.unwrap_or_default();
        let thread_group_name = self
            .lookup_name(thread.thread_group_name_id)?
            .unwrap_or_default();
        Ok(ResolvedThread {
            thread_name,
            thread_group_name,
        })
    }

    /// Returns `true` if a `HPROF_END_THREAD` record exists for `thread_serial`.
    pub fn was_thread_ended(&self, thread_serial: u32) -> bool {
        self.end_threads().find(thread_serial as u64).is_some()
    }

    /// Iterate over all start-thread records in ascending `thread_serial` order.
    pub fn iter_start_threads(&self) -> StartThreadIter<'_> {
        StartThreadIter {
            hprof_data: self.hprof_bytes(),
            id_size: self.hprof_header.id_size as usize,
            start_threads: self.start_threads(),
            pos: 0,
        }
    }

    // ── Unload-class lookup ───────────────────────────────────────────────────

    /// Returns `true` if a `HPROF_UNLOAD_CLASS` record exists for `class_serial`.
    pub fn was_class_unloaded(&self, class_serial: u32) -> bool {
        self.unload_classes().find(class_serial as u64).is_some()
    }
}

// ── Iterators ─────────────────────────────────────────────────────────────────

/// Iterator over all [`Frame`] records in ascending `frame_id` order.
pub struct FrameIter<'a> {
    hprof_data: &'a [u8],
    id_size: usize,
    frames: AuxIndexReader<'a>,
    pos: usize,
}

impl Iterator for FrameIter<'_> {
    type Item = Result<Frame, HprofError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.frames.len() {
            return None;
        }
        let (_key, hprof_offset) = self.frames.entry_at(self.pos);
        self.pos += 1;
        Some(parse_frame(
            self.hprof_data,
            hprof_offset as usize,
            self.id_size,
        ))
    }
}

/// Iterator over all [`Trace`] records in ascending `trace_serial` order.
pub struct TraceIter<'a> {
    hprof_data: &'a [u8],
    id_size: usize,
    traces: AuxIndexReader<'a>,
    pos: usize,
}

impl Iterator for TraceIter<'_> {
    type Item = Result<Trace, HprofError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.traces.len() {
            return None;
        }
        let (_key, hprof_offset) = self.traces.entry_at(self.pos);
        self.pos += 1;
        Some(parse_trace(
            self.hprof_data,
            hprof_offset as usize,
            self.id_size,
        ))
    }
}

/// Iterator over all [`StartThread`] records in ascending `thread_serial` order.
pub struct StartThreadIter<'a> {
    hprof_data: &'a [u8],
    id_size: usize,
    start_threads: AuxIndexReader<'a>,
    pos: usize,
}

impl Iterator for StartThreadIter<'_> {
    type Item = Result<StartThread, HprofError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.start_threads.len() {
            return None;
        }
        let (_key, hprof_offset) = self.start_threads.entry_at(self.pos);
        self.pos += 1;
        Some(parse_start_thread(
            self.hprof_data,
            hprof_offset as usize,
            self.id_size,
        ))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{IndexStore, MemStore, names};
    use crate::pipeline::{IndexOptions, build_indexes};
    use crate::progress::NoProgress;
    use crate::test_util::HprofBuilder;

    /// UTF8: 1 "main", 2 "MyClass", 3 "()V", 4 "MyClass.java";
    /// FRAME 0x10 (method=1, sig=3, src=4, class_serial=99, line=5);
    /// TRACE 1 (thread 1, frames [0x10]); START_THREAD 1 (id 0xABC, trace 1,
    /// name=1, group=2); END_THREAD 1; UNLOAD_CLASS 77.
    fn test_heap() -> Vec<u8> {
        HprofBuilder::new(8)
            .utf8(1, "main")
            .utf8(2, "MyClass")
            .utf8(3, "()V")
            .utf8(4, "MyClass.java")
            .frame(0x10, 1, 3, 4, 99, 5)
            .trace(1, 1, &[0x10])
            .start_thread(1, 0xABC, 1, 1, 2, 0)
            .end_thread(1)
            .unload_class(77)
            .build()
    }

    /// Build all indexes in memory and open an [`AuxRecordIndex`].
    fn build_all(hprof_data: &[u8]) -> AuxRecordIndex {
        let store = MemStore::new();
        let opts = IndexOptions {
            retained: false,
            force: false,
            ..IndexOptions::default()
        };
        build_indexes(hprof_data, &store, &opts, &NoProgress).unwrap();
        AuxRecordIndex::open(
            Arc::new(ByteSource::from(hprof_data.to_vec())),
            store.open(names::FRAMES).unwrap(),
            store.open(names::TRACES).unwrap(),
            store.open(names::START_THREADS).unwrap(),
            store.open(names::END_THREADS).unwrap(),
            store.open(names::UNLOAD_CLASSES).unwrap(),
            Arc::new(store.open(names::UTF8).unwrap()),
        )
        .unwrap()
    }

    #[test]
    fn find_frame_and_resolve() {
        let idx = build_all(&test_heap());

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
        let idx = build_all(&test_heap());
        assert!(idx.find_frame(0xDEAD).unwrap().is_none());
    }

    #[test]
    fn find_trace_and_frames() {
        let idx = build_all(&test_heap());

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
        let idx = build_all(&test_heap());
        assert!(idx.find_trace(999).unwrap().is_none());
    }

    #[test]
    fn start_thread_iterates_and_resolves() {
        let idx = build_all(&test_heap());

        let thread = idx.iter_start_threads().next().unwrap().unwrap();
        assert_eq!(thread.thread_serial, 1);
        assert_eq!(thread.thread_id, 0xABC);
        assert_eq!(thread.stack_trace_serial, 1);

        let resolved = idx.resolve_thread(&thread).unwrap();
        assert_eq!(resolved.thread_name, "main");
        assert_eq!(resolved.thread_group_name, "MyClass");
    }

    #[test]
    fn was_thread_ended() {
        let idx = build_all(&test_heap());
        assert!(idx.was_thread_ended(1));
        assert!(!idx.was_thread_ended(99));
    }

    #[test]
    fn was_class_unloaded() {
        let idx = build_all(&test_heap());
        assert!(idx.was_class_unloaded(77));
        assert!(!idx.was_class_unloaded(1));
    }

    #[test]
    fn iter_frames_yields_all() {
        let idx = build_all(&test_heap());
        let frames: Vec<_> = idx.iter_frames().collect::<Result<_, _>>().unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].frame_id, 0x10);
    }

    #[test]
    fn iter_traces_yields_all() {
        let idx = build_all(&test_heap());
        let traces: Vec<_> = idx.iter_traces().collect::<Result<_, _>>().unwrap();
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].trace_serial, 1);
    }

    #[test]
    fn iter_start_threads_yields_all() {
        let idx = build_all(&test_heap());
        let threads: Vec<_> = idx.iter_start_threads().collect::<Result<_, _>>().unwrap();
        assert_eq!(threads.len(), 1);
        assert_eq!(threads[0].thread_serial, 1);
    }

    #[test]
    fn lookup_name_works() {
        let idx = build_all(&test_heap());
        assert_eq!(idx.lookup_name(1).unwrap(), Some("main".to_string()));
        assert_eq!(idx.lookup_name(9999).unwrap(), None);
    }
}
