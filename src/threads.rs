//! Threads and their stack traces.
//!
//! A heap dump describes threads in three places: `HPROF_START_THREAD`
//! records (name and group, present in most dumps), `GC_ROOT_THREAD_OBJ`
//! sub-records (object id and the stack trace at dump time, present in all
//! dumps), and the `java.lang.Thread` objects themselves.  [`HeapQuery::threads`]
//! merges the three so callers see one list.

use crate::aux_query::LineNumber;
use crate::heap_index::sub_record::{SubIndexEntry, TAG_ROOT_THREAD_OBJ};
use crate::heap_parser::{FieldValue, SubRecord};
use crate::hprof::{BasicType, HprofError};
use crate::query::HeapQuery;
use crate::resolved::Value;
use crate::root_index::GcRootType;
use crate::search::Matcher;
use std::cmp::Reverse;
use std::collections::BTreeMap;

/// One stack frame of a thread.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameInfo {
    /// Method name (`"run"`).
    pub method: String,
    /// JVM method signature (`"()V"`).
    pub signature: String,
    /// Source file name, empty when unknown.
    pub source_file: String,
    /// Line number, or why there is none (native, compiled, …).
    pub line: LineNumber,
    /// Serial of the class declaring the method (see `HPROF_LOAD_CLASS`).
    pub class_serial: u32,
}

impl FrameInfo {
    /// `File.java:42`, `File.java [native]`, or just the file name.
    pub fn location(&self) -> String {
        match self.line {
            LineNumber::Line(n) => format!("{}:{n}", self.source_file),
            LineNumber::Native => format!("{} [native]", self.source_file),
            LineNumber::Compiled => format!("{} [compiled]", self.source_file),
            LineNumber::Unknown | LineNumber::NoInfo => self.source_file.clone(),
        }
    }
}

/// A thread that existed when the heap was dumped.
#[derive(Debug, Clone, PartialEq)]
pub struct ThreadInfo {
    /// The thread's serial number; the key for [`HeapQuery::thread`].
    pub serial: u32,
    /// Object id of the `java.lang.Thread` instance, when the dump has one.
    pub object_id: Option<u64>,
    /// Thread name; `thread-<serial>` when nothing better is recorded.
    pub name: String,
    /// Thread group name, when recorded.
    pub group: Option<String>,
    /// `true` when the dump also records the thread as ended.
    pub ended: bool,
    /// Serial of the thread's stack trace (the one recorded at dump time
    /// when available), if any.
    pub stack_trace_serial: Option<u32>,
}

impl HeapQuery {
    /// Every thread in the dump, ascending by serial.
    pub fn threads(&self) -> Result<Vec<ThreadInfo>, HprofError> {
        let mut by_serial: BTreeMap<u32, ThreadInfo> = BTreeMap::new();

        for thread in self.iter_threads() {
            let thread = thread?;
            let resolved = self.resolve_thread(&thread)?;
            by_serial.insert(
                thread.thread_serial,
                ThreadInfo {
                    serial: thread.thread_serial,
                    object_id: Some(thread.thread_id),
                    name: resolved.thread_name,
                    group: Some(resolved.thread_group_name).filter(|g| !g.is_empty()),
                    ended: self.was_thread_ended(thread.thread_serial),
                    stack_trace_serial: Some(thread.stack_trace_serial),
                },
            );
        }

        for root in self.iter_roots(GcRootType::ThreadObject) {
            let entry = SubIndexEntry {
                tag: TAG_ROOT_THREAD_OBJ,
                object_id: root.object_id,
                position: root.position,
            };
            let SubRecord::RootThreadObj(r) = self.parse_entry(&entry)? else {
                continue;
            };
            let info = match by_serial.entry(r.thread_serial) {
                std::collections::btree_map::Entry::Occupied(o) => o.into_mut(),
                std::collections::btree_map::Entry::Vacant(v) => v.insert(ThreadInfo {
                    serial: r.thread_serial,
                    object_id: None,
                    name: self
                        .thread_name_from_object(r.thread_object_id)
                        .unwrap_or_else(|| format!("thread-{}", r.thread_serial)),
                    group: None,
                    ended: self.was_thread_ended(r.thread_serial),
                    stack_trace_serial: None,
                }),
            };
            info.object_id = Some(r.thread_object_id);
            info.stack_trace_serial = Some(r.stack_trace_serial);
        }

        Ok(by_serial.into_values().collect())
    }

    /// The threads whose name or group matches `matcher`.
    ///
    /// In serial order, or best match first when the matcher ranks
    /// ([`Matcher::ranks`], fuzzy mode).
    pub fn threads_matching(&self, matcher: &Matcher) -> Result<Vec<ThreadInfo>, HprofError> {
        let mut scored: Vec<(Reverse<u32>, ThreadInfo)> = self
            .threads()?
            .into_iter()
            .filter_map(|t| {
                let name = matcher.score(&t.name);
                let group = t.group.as_deref().and_then(|g| matcher.score(g));
                let best = match (name, group) {
                    (Some(a), Some(b)) => Some(a.max(b)),
                    (a, b) => a.or(b),
                };
                best.map(|score| (Reverse(score), t))
            })
            .collect();
        if matcher.ranks() {
            scored.sort_by_key(|(score, t)| (*score, t.serial));
        }
        Ok(scored.into_iter().map(|(_, t)| t).collect())
    }

    /// The thread with serial `serial`, if the dump has one.
    pub fn thread(&self, serial: u32) -> Result<Option<ThreadInfo>, HprofError> {
        Ok(self.threads()?.into_iter().find(|t| t.serial == serial))
    }

    /// The thread's stack, outermost frame first; empty when the dump has no
    /// trace for it.
    pub fn thread_stack(&self, thread: &ThreadInfo) -> Result<Vec<FrameInfo>, HprofError> {
        let Some(serial) = thread.stack_trace_serial else {
            return Ok(Vec::new());
        };
        let Some(trace) = self.find_trace(serial)? else {
            return Ok(Vec::new());
        };
        self.trace_frames(&trace)?
            .iter()
            .map(|frame| {
                let rf = self.resolve_frame(frame)?;
                Ok(FrameInfo {
                    method: rf.method_name,
                    signature: rf.method_signature,
                    source_file: rf.source_file,
                    line: rf.line_number,
                    class_serial: rf.class_serial,
                })
            })
            .collect()
    }

    /// The `name` field of the `java.lang.Thread` instance `thread_object_id`.
    fn thread_name_from_object(&self, thread_object_id: u64) -> Option<String> {
        let inst = self.instance(thread_object_id).ok()??;
        let name_id = self
            .instance_fields(&inst)
            .ok()?
            .into_iter()
            .find(|f| f.name == "name" && f.ty == BasicType::Object)
            .and_then(|f| match f.value {
                FieldValue::Object(id) if id != 0 => Some(id),
                _ => None,
            })?;
        match self.resolve_value(name_id).ok()? {
            Value::String(_, s) => Some(s),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::heap_parser::FieldValue;
    use crate::test_util::{ClassSpec, HprofBuilder, build_in_memory, standard_heap, ty};

    #[test]
    fn threads_matching_filters_on_name_or_group_and_ranks_fuzzy_matches() {
        use crate::search::{Matcher, SearchQuery};
        let heap = build_in_memory(&standard_heap());
        let m = |q: SearchQuery| Matcher::new(&q).unwrap();
        let found = heap.threads_matching(&m(SearchQuery::fuzzy("mn"))).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "main");
        assert!(
            heap.threads_matching(&m(SearchQuery::contains("xyz")))
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            heap.threads_matching(&m(SearchQuery::exact("MAIN")))
                .unwrap()
                .len(),
            1
        );

        // Several threads: fuzzy order is by score, then serial; the group
        // name counts too.
        let bytes = HprofBuilder::new(8)
            .utf8(1, "worker-pool-1")
            .utf8(2, "main")
            .utf8(3, "pool")
            .utf8(4, "wp")
            .start_thread(1, 0xA1, 0, 1, 3, 0)
            .start_thread(2, 0xA2, 0, 2, 0, 0)
            .start_thread(3, 0xA3, 0, 4, 0, 0)
            .build();
        let heap = build_in_memory(&bytes);
        let names = |q: SearchQuery| -> Vec<String> {
            heap.threads_matching(&m(q))
                .unwrap()
                .into_iter()
                .map(|t| t.name)
                .collect()
        };
        assert_eq!(names(SearchQuery::fuzzy("wp")), ["wp", "worker-pool-1"]);
        assert_eq!(names(SearchQuery::contains("pool")), ["worker-pool-1"]);
        assert_eq!(names(SearchQuery::contains("o")), ["worker-pool-1"]);
    }

    #[test]
    fn a_start_thread_record_gives_name_serial_object_and_ended_flag() {
        let heap = build_in_memory(&standard_heap());
        let threads = heap.threads().unwrap();
        assert_eq!(threads.len(), 1);
        let t = &threads[0];
        assert_eq!(t.serial, 1);
        assert_eq!(t.name, "main");
        assert_eq!(t.object_id, Some(0xABC));
        assert_eq!(t.group, None);
        assert!(t.ended);
        assert_eq!(t.stack_trace_serial, Some(1));
        assert_eq!(heap.thread(1).unwrap().as_ref(), Some(t));
        assert_eq!(heap.thread(99).unwrap(), None);
    }

    #[test]
    fn thread_stack_resolves_frames() {
        let heap = build_in_memory(&standard_heap());
        let t = heap.thread(1).unwrap().unwrap();
        let frames = heap.thread_stack(&t).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].method, "main");
        assert_eq!(frames[0].signature, "()V");
        assert_eq!(frames[0].location(), "MyClass.java:7");
    }

    /// No HPROF_START_THREAD: the thread comes from GC_ROOT_THREAD_OBJ and its
    /// name from the `java.lang.Thread.name` string in the heap.
    #[test]
    fn root_only_threads_take_their_name_from_the_thread_object() {
        const THREAD_CLASS: u64 = 0x500;
        const STRING_CLASS: u64 = 0x510;
        const THREAD: u64 = 0x600;
        const NAME: u64 = 0x610;
        const CHARS: u64 = 0x620;
        let bytes = HprofBuilder::new(8)
            .utf8(1, "name")
            .utf8(2, "java/lang/Thread")
            .utf8(3, "java/lang/String")
            .utf8(4, "value")
            .load_class(1, THREAD_CLASS, 2)
            .load_class(2, STRING_CLASS, 3)
            .class_dump(
                ClassSpec::new(THREAD_CLASS)
                    .instance_size(8)
                    .field(1, ty::OBJECT),
            )
            .class_dump(
                ClassSpec::new(STRING_CLASS)
                    .instance_size(8)
                    .field(4, ty::OBJECT),
            )
            .char_array(CHARS, "worker-7")
            .instance_values(NAME, STRING_CLASS, &[FieldValue::Object(CHARS)])
            .instance_values(THREAD, THREAD_CLASS, &[FieldValue::Object(NAME)])
            .root_thread_obj(THREAD, 5, 9)
            .build();
        let heap = build_in_memory(&bytes);
        let threads = heap.threads().unwrap();
        assert_eq!(threads.len(), 1);
        assert_eq!(threads[0].serial, 5);
        assert_eq!(threads[0].name, "worker-7");
        assert_eq!(threads[0].object_id, Some(THREAD));
        assert_eq!(threads[0].stack_trace_serial, Some(9));
        assert!(!threads[0].ended);
        // Trace 9 is not in the dump: an empty stack, not an error.
        assert!(heap.thread_stack(&threads[0]).unwrap().is_empty());
    }

    #[test]
    fn an_unnamed_root_thread_falls_back_to_its_serial() {
        let bytes = HprofBuilder::new(8).root_thread_obj(0x1, 3, 0).build();
        let heap = build_in_memory(&bytes);
        assert_eq!(heap.threads().unwrap()[0].name, "thread-3");
    }
}
