//! List all threads that were live at dump time with their stack traces.
//!
//! Reads `GC_ROOT_THREAD_OBJ` heap sub-records, which carry the
//! `stack_trace_serial` for the thread's *current* call stack at dump time.
//! `HPROF_START_THREAD` records are only used for the thread name — their
//! `stack_trace_serial` points to the trace at thread startup, which is
//! usually empty.
//!
//! ```text
//! cargo run --example thread_stacks
//! ```
//!
//! Output:
//! ```text
//! Thread 1 "main" (serial=1, id=0x1)
//!   java.lang.Thread.run(Thread.java:834)
//!
//! Thread 2 "Server thread" (serial=2, id=0x2)  [ended]
//!   ...
//! ```

use hprof_toolkit::{
    aux_query::record::LineNumber,
    heap_index::sub_record::{SubIndexEntry, TAG_ROOT_THREAD_OBJ},
    heap_parser::SubRecord,
    hprof::HprofError,
    pipeline::build_all_indexes,
    query::HeapQuery,
    root_index::GcRootType,
};
use std::path::Path;

fn main() -> Result<(), HprofError> {
    let path = Path::new("./heap.dump");
    let indexes = build_all_indexes(path)?;
    let query = HeapQuery::open(path, &indexes)?;

    for root_entry in query.iter_roots(GcRootType::ThreadObject) {
        // Parse the GC_ROOT_THREAD_OBJ sub-record from the hprof to get
        // thread_serial and the dump-time stack_trace_serial.
        let sub_entry = SubIndexEntry {
            tag: TAG_ROOT_THREAD_OBJ,
            object_id: root_entry.object_id,
            position: root_entry.position,
        };
        let Ok(SubRecord::RootThreadObj(root)) = query.parse_entry(&sub_entry) else {
            continue;
        };

        // Thread name from HPROF_START_THREAD, falling back to serial number.
        let thread_name = match query.find_thread(root.thread_serial)? {
            Some(thread) => query.resolve_thread(&thread)?.thread_name,
            None => format!("thread-{}", root.thread_serial),
        };

        let ended = if query.was_thread_ended(root.thread_serial) {
            "  [ended]"
        } else {
            ""
        };

        println!(
            "Thread {} {:?} (serial={}, id=0x{:x}){}",
            root.thread_serial, thread_name, root.thread_serial, root.thread_object_id, ended,
        );

        if let Ok(Some(trace)) = query.find_trace(root.stack_trace_serial) {
            for frame in query.trace_frames(&trace)? {
                let rf = query.resolve_frame(&frame)?;
                let location = match rf.line_number {
                    LineNumber::Line(n) => format!("{}:{n}", rf.source_file),
                    LineNumber::Native => format!("{} [native]", rf.source_file),
                    LineNumber::Compiled => format!("{} [compiled]", rf.source_file),
                    _ => rf.source_file.clone(),
                };
                println!(
                    "  {}{}  ({})",
                    rf.method_name, rf.method_signature, location
                );
            }
        }

        println!();
    }

    Ok(())
}
