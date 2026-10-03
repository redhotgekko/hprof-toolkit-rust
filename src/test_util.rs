//! Shared, file-free test fixtures.
//!
//! * [`HprofBuilder`] assembles a syntactically valid hprof byte vector from
//!   a handful of records, so tests describe a heap in a few lines instead
//!   of hand-encoding big-endian fields.
//! * [`build_in_memory`] runs the **real** index pipeline against a
//!   [`MemStore`] and opens a [`HeapQuery`] on it — no file is touched.
//! * [`standard_heap`] is the fixture most modules share.
//!
//! Nothing in this module is compiled outside `cfg(test)`.

#![allow(dead_code)] // fixtures: not every helper is used by every test module

use crate::heap_parser::FieldValue;
use crate::hprof::HprofError;
use crate::index::{ByteSource, IndexStore, MemStore, StoreWriter};
use crate::pipeline::{IndexOptions, build_indexes};
use crate::progress::NoProgress;
use crate::query::HeapQuery;

// ── Tag constants (hprof spec) ────────────────────────────────────────────────

const TAG_UTF8: u8 = 0x01;
const TAG_LOAD_CLASS: u8 = 0x02;
const TAG_UNLOAD_CLASS: u8 = 0x03;
const TAG_FRAME: u8 = 0x04;
const TAG_TRACE: u8 = 0x05;
const TAG_START_THREAD: u8 = 0x0A;
const TAG_END_THREAD: u8 = 0x0B;
const TAG_HEAP_DUMP_SEGMENT: u8 = 0x1C;
const TAG_HEAP_DUMP_END: u8 = 0x2C;

/// hprof basic type codes used by [`ClassSpec::field`].
pub(crate) mod ty {
    pub const OBJECT: u8 = 2;
    pub const BOOLEAN: u8 = 4;
    pub const CHAR: u8 = 5;
    pub const FLOAT: u8 = 6;
    pub const DOUBLE: u8 = 7;
    pub const BYTE: u8 = 8;
    pub const SHORT: u8 = 9;
    pub const INT: u8 = 10;
    pub const LONG: u8 = 11;
}

// ── ClassSpec ─────────────────────────────────────────────────────────────────

/// Description of one `CLASS_DUMP` sub-record.
pub(crate) struct ClassSpec {
    class_id: u64,
    stack_serial: u32,
    super_id: u64,
    instance_size: u32,
    statics: Vec<(u64, FieldValue)>,
    fields: Vec<(u64, u8)>,
}

impl ClassSpec {
    pub fn new(class_id: u64) -> Self {
        Self {
            class_id,
            stack_serial: 0,
            super_id: 0,
            instance_size: 0,
            statics: Vec::new(),
            fields: Vec::new(),
        }
    }

    pub fn super_class(mut self, super_id: u64) -> Self {
        self.super_id = super_id;
        self
    }

    pub fn instance_size(mut self, size: u32) -> Self {
        self.instance_size = size;
        self
    }

    /// Add a static field with a concrete value.
    pub fn static_field(mut self, name_id: u64, value: FieldValue) -> Self {
        self.statics.push((name_id, value));
        self
    }

    /// Add an instance field descriptor (`type_code` from [`ty`]).
    pub fn field(mut self, name_id: u64, type_code: u8) -> Self {
        self.fields.push((name_id, type_code));
        self
    }
}

// ── HprofBuilder ──────────────────────────────────────────────────────────────

/// Builds an hprof byte vector record by record.
///
/// Heap sub-records are accumulated into the current `HEAP_DUMP_SEGMENT`,
/// which is flushed automatically before any top-level record is written and
/// at [`build`](Self::build).  Call [`new_segment`](Self::new_segment) to
/// split sub-records across several segments.
pub(crate) struct HprofBuilder {
    id_size: u32,
    buf: Vec<u8>,
    seg: Vec<u8>,
}

impl HprofBuilder {
    /// Start a dump with the given identifier size (4 or 8).
    pub fn new(id_size: u32) -> Self {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"JAVA PROFILE 1.0.2\0");
        buf.extend_from_slice(&id_size.to_be_bytes());
        buf.extend_from_slice(&0u64.to_be_bytes()); // timestamp
        Self {
            id_size,
            buf,
            seg: Vec::new(),
        }
    }

    /// Byte offset at which the first record starts (31 for the fixed header).
    pub fn data_offset(&self) -> usize {
        19 + 4 + 8
    }

    fn push_id(out: &mut Vec<u8>, id_size: u32, v: u64) {
        if id_size == 4 {
            out.extend_from_slice(&(v as u32).to_be_bytes());
        } else {
            out.extend_from_slice(&v.to_be_bytes());
        }
    }

    fn flush_segment(&mut self) {
        if self.seg.is_empty() {
            return;
        }
        let seg = std::mem::take(&mut self.seg);
        self.write_record(TAG_HEAP_DUMP_SEGMENT, &seg);
    }

    fn write_record(&mut self, tag: u8, body: &[u8]) {
        self.buf.push(tag);
        self.buf.extend_from_slice(&0u32.to_be_bytes()); // time offset
        self.buf
            .extend_from_slice(&(body.len() as u32).to_be_bytes());
        self.buf.extend_from_slice(body);
    }

    fn record(mut self, tag: u8, body: &[u8]) -> Self {
        self.flush_segment();
        self.write_record(tag, body);
        self
    }

    /// Byte position the *next* top-level record will be written at.
    ///
    /// Useful for asserting index positions.  Flushes any pending segment.
    pub fn next_record_position(&mut self) -> u64 {
        self.flush_segment();
        self.buf.len() as u64
    }

    // ── Top-level records ─────────────────────────────────────────────────────

    pub fn utf8(self, id: u64, s: &str) -> Self {
        let mut body = Vec::new();
        Self::push_id(&mut body, self.id_size, id);
        body.extend_from_slice(s.as_bytes());
        self.record(TAG_UTF8, &body)
    }

    pub fn load_class(self, serial: u32, class_id: u64, name_id: u64) -> Self {
        let mut body = Vec::new();
        body.extend_from_slice(&serial.to_be_bytes());
        Self::push_id(&mut body, self.id_size, class_id);
        body.extend_from_slice(&0u32.to_be_bytes()); // stack trace serial
        Self::push_id(&mut body, self.id_size, name_id);
        self.record(TAG_LOAD_CLASS, &body)
    }

    pub fn unload_class(self, serial: u32) -> Self {
        self.record(TAG_UNLOAD_CLASS, &serial.to_be_bytes())
    }

    pub fn frame(
        self,
        frame_id: u64,
        method_name_id: u64,
        method_sig_id: u64,
        source_file_id: u64,
        class_serial: u32,
        line: i32,
    ) -> Self {
        let mut body = Vec::new();
        Self::push_id(&mut body, self.id_size, frame_id);
        Self::push_id(&mut body, self.id_size, method_name_id);
        Self::push_id(&mut body, self.id_size, method_sig_id);
        Self::push_id(&mut body, self.id_size, source_file_id);
        body.extend_from_slice(&class_serial.to_be_bytes());
        body.extend_from_slice(&line.to_be_bytes());
        self.record(TAG_FRAME, &body)
    }

    pub fn trace(self, serial: u32, thread_serial: u32, frame_ids: &[u64]) -> Self {
        let mut body = Vec::new();
        body.extend_from_slice(&serial.to_be_bytes());
        body.extend_from_slice(&thread_serial.to_be_bytes());
        body.extend_from_slice(&(frame_ids.len() as u32).to_be_bytes());
        for &f in frame_ids {
            Self::push_id(&mut body, self.id_size, f);
        }
        self.record(TAG_TRACE, &body)
    }

    pub fn start_thread(
        self,
        serial: u32,
        thread_id: u64,
        trace_serial: u32,
        name_id: u64,
        group_name_id: u64,
        parent_group_name_id: u64,
    ) -> Self {
        let mut body = Vec::new();
        body.extend_from_slice(&serial.to_be_bytes());
        Self::push_id(&mut body, self.id_size, thread_id);
        body.extend_from_slice(&trace_serial.to_be_bytes());
        Self::push_id(&mut body, self.id_size, name_id);
        Self::push_id(&mut body, self.id_size, group_name_id);
        Self::push_id(&mut body, self.id_size, parent_group_name_id);
        self.record(TAG_START_THREAD, &body)
    }

    pub fn end_thread(self, serial: u32) -> Self {
        self.record(TAG_END_THREAD, &serial.to_be_bytes())
    }

    /// Append an explicit `HEAP_DUMP_END` record.
    pub fn heap_dump_end(self) -> Self {
        self.record(TAG_HEAP_DUMP_END, &[])
    }

    /// Close the current heap segment so later sub-records go into a new one.
    pub fn new_segment(mut self) -> Self {
        self.flush_segment();
        self
    }

    // ── Heap sub-records ──────────────────────────────────────────────────────

    fn sub(mut self, tag: u8, f: impl FnOnce(&mut Vec<u8>, u32)) -> Self {
        self.seg.push(tag);
        f(&mut self.seg, self.id_size);
        self
    }

    pub fn class_dump(self, spec: ClassSpec) -> Self {
        self.sub(0x20, |seg, id_size| {
            Self::push_id(seg, id_size, spec.class_id);
            seg.extend_from_slice(&spec.stack_serial.to_be_bytes());
            Self::push_id(seg, id_size, spec.super_id);
            for _ in 0..5 {
                // loader, signers, domain, reserved1, reserved2
                Self::push_id(seg, id_size, 0);
            }
            seg.extend_from_slice(&spec.instance_size.to_be_bytes());
            seg.extend_from_slice(&0u16.to_be_bytes()); // constant pool
            seg.extend_from_slice(&(spec.statics.len() as u16).to_be_bytes());
            for (name_id, value) in &spec.statics {
                Self::push_id(seg, id_size, *name_id);
                seg.push(type_code_of(value));
                encode_value(seg, id_size, value);
            }
            seg.extend_from_slice(&(spec.fields.len() as u16).to_be_bytes());
            for (name_id, type_code) in &spec.fields {
                Self::push_id(seg, id_size, *name_id);
                seg.push(*type_code);
            }
        })
    }

    /// `INSTANCE_DUMP` with raw field bytes.
    pub fn instance(self, object_id: u64, class_id: u64, data: &[u8]) -> Self {
        self.sub(0x21, |seg, id_size| {
            Self::push_id(seg, id_size, object_id);
            seg.extend_from_slice(&0u32.to_be_bytes());
            Self::push_id(seg, id_size, class_id);
            seg.extend_from_slice(&(data.len() as u32).to_be_bytes());
            seg.extend_from_slice(data);
        })
    }

    /// `INSTANCE_DUMP` whose field data is a sequence of typed values, in
    /// class-hierarchy order (own fields first).
    pub fn instance_values(self, object_id: u64, class_id: u64, values: &[FieldValue]) -> Self {
        let mut data = Vec::new();
        for v in values {
            encode_value(&mut data, self.id_size, v);
        }
        self.instance(object_id, class_id, &data)
    }

    /// `PRIM_ARRAY_DUMP` with raw big-endian element bytes.
    pub fn prim_array(self, array_id: u64, elem_type: u8, num_elements: u32, data: &[u8]) -> Self {
        self.sub(0x23, |seg, id_size| {
            Self::push_id(seg, id_size, array_id);
            seg.extend_from_slice(&0u32.to_be_bytes());
            seg.extend_from_slice(&num_elements.to_be_bytes());
            seg.push(elem_type);
            seg.extend_from_slice(data);
        })
    }

    pub fn int_array(self, array_id: u64, values: &[i32]) -> Self {
        let mut data = Vec::new();
        for v in values {
            data.extend_from_slice(&v.to_be_bytes());
        }
        self.prim_array(array_id, ty::INT, values.len() as u32, &data)
    }

    pub fn char_array(self, array_id: u64, s: &str) -> Self {
        let units: Vec<u16> = s.encode_utf16().collect();
        let mut data = Vec::new();
        for u in &units {
            data.extend_from_slice(&u.to_be_bytes());
        }
        self.prim_array(array_id, ty::CHAR, units.len() as u32, &data)
    }

    pub fn byte_array(self, array_id: u64, bytes: &[u8]) -> Self {
        self.prim_array(array_id, ty::BYTE, bytes.len() as u32, bytes)
    }

    /// `OBJ_ARRAY_DUMP`; `array_class_id` is the array's own class (as in a
    /// real dump, e.g. the class named `[Ljava.lang.String;`).
    pub fn obj_array(self, array_id: u64, array_class_id: u64, elems: &[u64]) -> Self {
        self.sub(0x22, |seg, id_size| {
            Self::push_id(seg, id_size, array_id);
            seg.extend_from_slice(&0u32.to_be_bytes());
            seg.extend_from_slice(&(elems.len() as u32).to_be_bytes());
            Self::push_id(seg, id_size, array_class_id);
            for &e in elems {
                Self::push_id(seg, id_size, e);
            }
        })
    }

    // ── GC roots ──────────────────────────────────────────────────────────────

    pub fn root_unknown(self, object_id: u64) -> Self {
        self.sub(0xFF, |seg, id_size| Self::push_id(seg, id_size, object_id))
    }

    pub fn root_jni_global(self, object_id: u64, jni_ref_id: u64) -> Self {
        self.sub(0x01, |seg, id_size| {
            Self::push_id(seg, id_size, object_id);
            Self::push_id(seg, id_size, jni_ref_id);
        })
    }

    pub fn root_jni_local(self, object_id: u64, thread_serial: u32, frame: u32) -> Self {
        self.sub(0x02, |seg, id_size| {
            Self::push_id(seg, id_size, object_id);
            seg.extend_from_slice(&thread_serial.to_be_bytes());
            seg.extend_from_slice(&frame.to_be_bytes());
        })
    }

    pub fn root_java_frame(self, object_id: u64, thread_serial: u32, frame: u32) -> Self {
        self.sub(0x03, |seg, id_size| {
            Self::push_id(seg, id_size, object_id);
            seg.extend_from_slice(&thread_serial.to_be_bytes());
            seg.extend_from_slice(&frame.to_be_bytes());
        })
    }

    pub fn root_native_stack(self, object_id: u64, thread_serial: u32) -> Self {
        self.sub(0x04, |seg, id_size| {
            Self::push_id(seg, id_size, object_id);
            seg.extend_from_slice(&thread_serial.to_be_bytes());
        })
    }

    pub fn root_sticky_class(self, class_id: u64) -> Self {
        self.sub(0x05, |seg, id_size| Self::push_id(seg, id_size, class_id))
    }

    pub fn root_thread_block(self, object_id: u64, thread_serial: u32) -> Self {
        self.sub(0x06, |seg, id_size| {
            Self::push_id(seg, id_size, object_id);
            seg.extend_from_slice(&thread_serial.to_be_bytes());
        })
    }

    pub fn root_monitor_used(self, object_id: u64) -> Self {
        self.sub(0x07, |seg, id_size| Self::push_id(seg, id_size, object_id))
    }

    pub fn root_thread_obj(self, object_id: u64, thread_serial: u32, trace_serial: u32) -> Self {
        self.sub(0x08, |seg, id_size| {
            Self::push_id(seg, id_size, object_id);
            seg.extend_from_slice(&thread_serial.to_be_bytes());
            seg.extend_from_slice(&trace_serial.to_be_bytes());
        })
    }

    /// Finish: flush the pending segment and return the bytes.
    pub fn build(mut self) -> Vec<u8> {
        self.flush_segment();
        self.buf
    }
}

fn type_code_of(v: &FieldValue) -> u8 {
    match v {
        FieldValue::Object(_) => ty::OBJECT,
        FieldValue::Bool(_) => ty::BOOLEAN,
        FieldValue::Char(_) => ty::CHAR,
        FieldValue::Float(_) => ty::FLOAT,
        FieldValue::Double(_) => ty::DOUBLE,
        FieldValue::Byte(_) => ty::BYTE,
        FieldValue::Short(_) => ty::SHORT,
        FieldValue::Int(_) => ty::INT,
        FieldValue::Long(_) => ty::LONG,
    }
}

fn encode_value(out: &mut Vec<u8>, id_size: u32, v: &FieldValue) {
    match v {
        FieldValue::Object(id) => HprofBuilder::push_id(out, id_size, *id),
        FieldValue::Bool(b) => out.push(u8::from(*b)),
        FieldValue::Char(c) => out.extend_from_slice(&c.to_be_bytes()),
        FieldValue::Float(f) => out.extend_from_slice(&f.to_bits().to_be_bytes()),
        FieldValue::Double(d) => out.extend_from_slice(&d.to_bits().to_be_bytes()),
        FieldValue::Byte(b) => out.push(*b as u8),
        FieldValue::Short(s) => out.extend_from_slice(&s.to_be_bytes()),
        FieldValue::Int(i) => out.extend_from_slice(&i.to_be_bytes()),
        FieldValue::Long(l) => out.extend_from_slice(&l.to_be_bytes()),
    }
}

/// Run a builder that writes one store entry and return its result and the
/// committed bytes.  Nothing touches a file.
pub(crate) fn built_bytes<R>(
    build: impl FnOnce(&mut dyn StoreWriter) -> Result<R, HprofError>,
) -> (R, Vec<u8>) {
    let store = MemStore::new();
    let mut writer = store.create("entry").expect("create");
    let result = build(&mut *writer).expect("builder failed");
    writer.commit().expect("commit");
    let bytes = store.open("entry").expect("open").as_ref().to_vec();
    (result, bytes)
}

// ── Pipeline helpers ──────────────────────────────────────────────────────────

/// Run the full index pipeline (including retained heap) into a fresh
/// [`MemStore`] and open a [`HeapQuery`] on it.  Never touches a file.
pub(crate) fn build_in_memory(hprof: &[u8]) -> HeapQuery {
    build_in_memory_with(hprof, &IndexOptions::default()).1
}

/// Like [`build_in_memory`] but returns the store too and takes options.
pub(crate) fn build_in_memory_with(hprof: &[u8], opts: &IndexOptions) -> (MemStore, HeapQuery) {
    let store = MemStore::new();
    build_indexes(hprof, &store, opts, &NoProgress).expect("in-memory index build failed");
    let query = HeapQuery::from_store(ByteSource::from(hprof.to_vec()), &store)
        .expect("HeapQuery::from_store failed");
    (store, query)
}

// ── Standard fixture ──────────────────────────────────────────────────────────

/// Name ids used by [`standard_heap`].
pub(crate) mod std_names {
    pub const COUNT: u64 = 1;
    pub const JAVA_LANG_OBJECT: u64 = 2;
    pub const JAVA_LANG_INTEGER: u64 = 3;
    pub const VALUE: u64 = 4;
    pub const MY_FIELD: u64 = 5;
    pub const JAVA_LANG_STRING: u64 = 6;
    pub const CHAR_ARRAY: u64 = 7;
    pub const JAVA_UTIL_ARRAYLIST: u64 = 8;
    pub const MAIN: u64 = 9;
    pub const SIG_V: u64 = 10;
    pub const MY_CLASS_JAVA: u64 = 11;
}

/// Object ids used by [`standard_heap`].
pub(crate) mod std_ids {
    // classes
    pub const INTEGER_CLASS: u64 = 0x10;
    pub const OBJECT_CLASS: u64 = 0x20;
    pub const STRING_CLASS: u64 = 0x30;
    pub const CHAR_ARRAY_CLASS: u64 = 0x40;
    pub const ARRAYLIST_CLASS: u64 = 0x50;
    // objects
    pub const INTEGER_42: u64 = 0x100;
    pub const CHARS_HI: u64 = 0x200;
    pub const STRING_HI: u64 = 0x300;
    pub const LIST: u64 = 0x400;
    // aux
    pub const FRAME_MAIN: u64 = 0x1000;
    pub const THREAD_OBJ: u64 = 0xABC;
    pub const THREAD_SERIAL: u32 = 1;
    pub const TRACE_SERIAL: u32 = 1;
}

/// A small heap that exercises every index:
///
/// ```text
/// classes:   Integer(0x10 ⊂ Object)  String(0x30 ⊂ Object)  char[](0x40)
///            ArrayList(0x50 ⊂ Object; static count:int = 7; field myField:Object)
/// objects:   0x100 Integer{value=42}
///            0x200 char[] "hi"
///            0x300 String{value → 0x200}
///            0x400 ArrayList{myField → 0x100}
/// GC roots:  sticky class 0x10, sticky class 0x20, java frame → 0x400 (thread 1)
/// aux:       frame 0x1000 "main()V" MyClass.java:7, trace 1, thread 1 "main" (ended)
/// ```
///
/// Reachability: 0x400 → 0x100 via `myField`; 0x300 and 0x200 are unreachable.
pub(crate) fn standard_heap() -> Vec<u8> {
    use std_ids::*;
    use std_names::*;
    HprofBuilder::new(8)
        .utf8(COUNT, "count")
        .utf8(JAVA_LANG_OBJECT, "java/lang/Object")
        .utf8(JAVA_LANG_INTEGER, "java/lang/Integer")
        .utf8(VALUE, "value")
        .utf8(MY_FIELD, "myField")
        .utf8(JAVA_LANG_STRING, "java/lang/String")
        .utf8(CHAR_ARRAY, "[C")
        .utf8(JAVA_UTIL_ARRAYLIST, "java/util/ArrayList")
        .utf8(MAIN, "main")
        .utf8(SIG_V, "()V")
        .utf8(MY_CLASS_JAVA, "MyClass.java")
        .load_class(1, INTEGER_CLASS, JAVA_LANG_INTEGER)
        .load_class(2, OBJECT_CLASS, JAVA_LANG_OBJECT)
        .load_class(3, STRING_CLASS, JAVA_LANG_STRING)
        .load_class(4, CHAR_ARRAY_CLASS, CHAR_ARRAY)
        .load_class(5, ARRAYLIST_CLASS, JAVA_UTIL_ARRAYLIST)
        .class_dump(
            ClassSpec::new(INTEGER_CLASS)
                .super_class(OBJECT_CLASS)
                .instance_size(4)
                .field(VALUE, ty::INT),
        )
        .class_dump(ClassSpec::new(OBJECT_CLASS))
        .class_dump(
            ClassSpec::new(STRING_CLASS)
                .super_class(OBJECT_CLASS)
                .instance_size(8)
                .field(VALUE, ty::OBJECT),
        )
        .class_dump(ClassSpec::new(CHAR_ARRAY_CLASS).super_class(OBJECT_CLASS))
        .class_dump(
            ClassSpec::new(ARRAYLIST_CLASS)
                .super_class(OBJECT_CLASS)
                .instance_size(8)
                .static_field(COUNT, FieldValue::Int(7))
                .field(MY_FIELD, ty::OBJECT),
        )
        .instance_values(INTEGER_42, INTEGER_CLASS, &[FieldValue::Int(42)])
        .char_array(CHARS_HI, "hi")
        .instance_values(STRING_HI, STRING_CLASS, &[FieldValue::Object(CHARS_HI)])
        .instance_values(LIST, ARRAYLIST_CLASS, &[FieldValue::Object(INTEGER_42)])
        .root_sticky_class(INTEGER_CLASS)
        .root_sticky_class(OBJECT_CLASS)
        .root_java_frame(LIST, THREAD_SERIAL, 0)
        .frame(FRAME_MAIN, MAIN, SIG_V, MY_CLASS_JAVA, 1, 7)
        .trace(TRACE_SERIAL, THREAD_SERIAL, &[FRAME_MAIN])
        .start_thread(THREAD_SERIAL, THREAD_OBJ, TRACE_SERIAL, MAIN, 0, 0)
        .end_thread(THREAD_SERIAL)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hprof::HprofFile;

    #[test]
    fn builder_produces_parseable_records() {
        let bytes = standard_heap();
        let hprof = HprofFile::from_ref(&bytes).unwrap();
        assert_eq!(hprof.id_size(), 8);
        let headers: Vec<_> = hprof.record_headers().map(|r| r.unwrap()).collect();
        // 11 utf8 + 5 load_class + 1 segment + frame + trace + start + end = 21
        assert_eq!(headers.len(), 21);
    }

    #[test]
    fn segments_split_on_demand() {
        let bytes = HprofBuilder::new(8)
            .root_unknown(1)
            .new_segment()
            .root_unknown(2)
            .build();
        let hprof = HprofFile::from_ref(&bytes).unwrap();
        assert_eq!(hprof.record_headers().count(), 2);
    }

    #[test]
    fn id_size_four_is_encoded_with_four_byte_ids() {
        let bytes = HprofBuilder::new(4).root_unknown(0x1234).build();
        let hprof = HprofFile::from_ref(&bytes).unwrap();
        let rec = hprof.record_headers().next().unwrap().unwrap();
        // subtag(1) + id(4)
        assert_eq!(rec.body_length, 5);
    }
}
