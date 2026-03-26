//! Fully-resolved representations of instances and classes.
//!
//! This module converts raw [`ClassDump`] and [`InstanceDump`] records into
//! fully resolved structs where:
//!
//! * Every name ID is resolved to a `String` via the UTF-8 name index.
//! * Every object-typed field is resolved through
//!   [`HeapQuery::resolve_value`], so common wrapper types (String, Integer,
//!   Long, …) are returned as rich enum variants while the object ID is
//!   preserved.
//! * Instance field descriptors in a class dump are mapped to the typed
//!   [`FieldType`] enum rather than raw hprof type codes.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use hprof_toolkit::query::HeapQuery;
//! use hprof_toolkit::resolved::{ResolvedInstance, ResolvedClass};
//! use hprof_toolkit::heap_parser::SubRecord;
//!
//! for result in query.iter_objects() {
//!     match result? {
//!         SubRecord::InstanceDump(inst) => {
//!             let resolved = ResolvedInstance::from_dump(&query, &inst)?;
//!             println!("{}: {:?}", resolved.class_name, resolved.fields);
//!         }
//!         SubRecord::ClassDump(cd) => {
//!             let resolved = ResolvedClass::from_dump(&query, &cd)?;
//!             println!("class {} ({} static fields)", resolved.class_name, resolved.static_fields.len());
//!         }
//!         _ => {}
//!     }
//! }
//! ```

use crate::heap_parser::{
    ClassDump, InstanceDump, ObjArrayDump, PrimArrayDump, RootJavaFrame, RootJniGlobal,
    RootJniLocal, RootMonitorUsed, RootNativeStack, RootStickyClass, RootThreadBlock,
    RootThreadObj, RootUnknown,
};
use crate::heap_query::JavaValue;
use crate::hprof::HprofError;
use crate::query::HeapQuery;

// ── Value ─────────────────────────────────────────────────────────────────────

/// A fully resolved field value.
///
/// Primitive types are stored directly.  Object-typed fields are resolved via
/// [`HeapQuery::resolve_value`]:  common Java wrapper classes become rich
/// variants (preserving the object ID), while arbitrary object references
/// become [`Value::Object`].
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    // ── Primitive instance field types ────────────────────────────────────────
    Bool(bool),
    Char(u16),
    Float(f32),
    Double(f64),
    Byte(i8),
    Short(i16),
    Int(i32),
    Long(i64),

    // ── Null object reference ─────────────────────────────────────────────────
    Null,

    // ── Resolved Java wrapper types (object_id, wrapped_value) ───────────────
    /// `java.lang.String`
    String(u64, std::string::String),
    /// `java.lang.Integer`
    Integer(u64, i32),
    /// `java.lang.Long`
    BoxedLong(u64, i64),
    /// `java.lang.Double`
    BoxedDouble(u64, f64),
    /// `java.lang.Float`
    BoxedFloat(u64, f32),
    /// `java.lang.Short`
    BoxedShort(u64, i16),
    /// `java.lang.Byte`
    BoxedByte(u64, i8),
    /// `java.lang.Boolean`
    BoxedBoolean(u64, bool),
    /// `java.lang.Character`
    BoxedCharacter(u64, u16),

    // ── Unresolved object reference ───────────────────────────────────────────
    /// An object reference that is not a recognised wrapper type.
    Object(u64),
}

// ── FieldType ─────────────────────────────────────────────────────────────────

/// The declared type of an instance or static field.
///
/// Maps the hprof type codes (2, 4–11) to named variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    Object,
    Bool,
    Char,
    Float,
    Double,
    Byte,
    Short,
    Int,
    Long,
}

impl FieldType {
    /// Convert an hprof type code to a [`FieldType`].
    ///
    /// Returns `None` for unknown codes.
    pub fn from_type_code(code: u8) -> Option<Self> {
        match code {
            2 => Some(Self::Object),
            4 => Some(Self::Bool),
            5 => Some(Self::Char),
            6 => Some(Self::Float),
            7 => Some(Self::Double),
            8 => Some(Self::Byte),
            9 => Some(Self::Short),
            10 => Some(Self::Int),
            11 => Some(Self::Long),
            _ => None,
        }
    }
}

// ── ResolvedInstance ──────────────────────────────────────────────────────────

/// A fully resolved instance dump.
///
/// All fields (including those inherited from superclasses) are resolved to
/// [`Value`] variants.  Object-typed fields pointing to known wrapper types
/// are unwrapped; everything else is [`Value::Object`].
#[derive(Debug, Clone)]
pub struct ResolvedInstance {
    /// The object's heap ID.
    pub object_id: u64,
    /// The class's heap ID.
    pub class_id: u64,
    /// Dot-notation class name (e.g. `"java.util.HashMap"`).
    pub class_name: std::string::String,
    /// Stack trace serial from the `INSTANCE_DUMP` record.
    pub stack_trace_serial: u32,
    /// All instance fields in class-hierarchy order (superclass fields last).
    pub fields: Vec<InstanceField>,
}

impl ResolvedInstance {
    /// Resolve an [`InstanceDump`] into a fully populated [`ResolvedInstance`].
    ///
    /// Traverses the class hierarchy to collect all fields, then resolves each
    /// value.  Object-typed fields that point to wrapper instances (String,
    /// Integer, Long, …) are resolved to their corresponding [`Value`] variants.
    pub fn from_dump(query: &HeapQuery, inst: &InstanceDump<'_>) -> Result<Self, HprofError> {
        let class_name = query.class_name(inst.class_id)?.unwrap_or_default();
        let raw_fields = query.instance_fields(inst)?;
        let fields = raw_fields
            .into_iter()
            .map(|f| {
                let value = field_value_to_value(query, f.value)?;
                Ok(InstanceField {
                    name: f.name,
                    value,
                })
            })
            .collect::<Result<Vec<_>, HprofError>>()?;

        Ok(Self {
            object_id: inst.object_id,
            class_id: inst.class_id,
            class_name,
            stack_trace_serial: inst.stack_trace_serial,
            fields,
        })
    }
}

// ── ResolvedClass ─────────────────────────────────────────────────────────────

/// A fully resolved class dump.
///
/// Static field values are resolved (wrapper types unwrapped).  Instance field
/// descriptors describe the layout of instances of this class; they carry the
/// field name and declared type but no value (values live in
/// [`ResolvedInstance::fields`]).
#[derive(Debug, Clone)]
pub struct ResolvedClass {
    /// The class's heap ID.
    pub class_id: u64,
    /// Dot-notation class name.
    pub class_name: std::string::String,
    /// Heap ID of the immediate superclass (0 = none).
    pub super_class_id: u64,
    /// Dot-notation name of the immediate superclass, when resolvable.
    pub super_class_name: Option<std::string::String>,
    /// Stack trace serial from the `CLASS_DUMP` record.
    pub stack_trace_serial: u32,
    /// Total size (in bytes) of one instance of this class.
    pub instance_size: u32,
    /// Resolved static fields declared on this class.
    pub static_fields: Vec<ResolvedStaticField>,
    /// Instance field descriptors declared on this class (not inherited).
    pub instance_fields: Vec<FieldDescriptor>,
}

impl ResolvedClass {
    /// Resolve a [`ClassDump`] into a fully populated [`ResolvedClass`].
    pub fn from_dump(query: &HeapQuery, cd: &ClassDump<'_>) -> Result<Self, HprofError> {
        let class_name = query.class_name(cd.class_id)?.unwrap_or_default();

        let super_class_name = if cd.super_class_id != 0 {
            query.class_name(cd.super_class_id)?
        } else {
            None
        };

        // Resolve static fields.
        let mut static_fields = Vec::new();
        for sf_result in cd.static_fields() {
            let sf = sf_result?;
            let name = query.lookup_name(sf.name_id)?.unwrap_or_default();
            let value = field_value_to_value(query, sf.value)?;
            static_fields.push(ResolvedStaticField { name, value });
        }

        // Resolve instance field descriptors for this class only.
        let mut instance_fields = Vec::new();
        for fd_result in cd.instance_fields() {
            let fd = fd_result?;
            let name = query.lookup_name(fd.name_id)?.unwrap_or_default();
            let field_type = FieldType::from_type_code(fd.field_type).unwrap_or(FieldType::Object);
            instance_fields.push(FieldDescriptor { name, field_type });
        }

        Ok(Self {
            class_id: cd.class_id,
            class_name,
            super_class_id: cd.super_class_id,
            super_class_name,
            stack_trace_serial: cd.stack_trace_serial,
            instance_size: cd.instance_size,
            static_fields,
            instance_fields,
        })
    }
}

// ── ResolvedObjArray ──────────────────────────────────────────────────────────

/// A fully resolved OBJ_ARRAY_DUMP.
///
/// The element class name is resolved via the load-class → UTF-8 name index
/// chain.  Each element object ID is resolved through
/// [`HeapQuery::resolve_value`] so wrapper types (String, Integer, …) become
/// rich [`Value`] variants.
#[derive(Debug, Clone)]
pub struct ResolvedObjArray {
    /// The array's heap ID.
    pub array_id: u64,
    /// Stack trace serial from the `OBJ_ARRAY_DUMP` record.
    pub stack_trace_serial: u32,
    /// Number of elements in the array.
    pub num_elements: u32,
    /// Heap ID of the element class (e.g. the class object for `String`).
    pub element_class_id: u64,
    /// Dot-notation element class name (e.g. `"java.lang.String"`).
    pub element_class_name: std::string::String,
    /// Resolved element values in array order.
    pub elements: Vec<Value>,
}

impl ResolvedObjArray {
    /// Resolve an [`ObjArrayDump`] into a fully populated [`ResolvedObjArray`].
    ///
    /// Resolves the element class name and then resolves each element object ID
    /// through [`HeapQuery::resolve_value`].
    pub fn from_dump(query: &HeapQuery, arr: &ObjArrayDump<'_>) -> Result<Self, HprofError> {
        let element_class_name = query.class_name(arr.element_class_id)?.unwrap_or_default();
        let elements = arr
            .elements()
            .map(|id| {
                let jv = query.resolve_value(id)?;
                Ok(java_value_to_value(jv))
            })
            .collect::<Result<Vec<_>, HprofError>>()?;
        Ok(Self {
            array_id: arr.array_id,
            stack_trace_serial: arr.stack_trace_serial,
            num_elements: arr.num_elements,
            element_class_id: arr.element_class_id,
            element_class_name,
            elements,
        })
    }
}

// ── ResolvedRoot ──────────────────────────────────────────────────────────────

/// A fully resolved GC root sub-record.
///
/// Each variant preserves all scalar fields from the raw record and adds a
/// resolved type name for the referenced object.  For most root kinds this is
/// the runtime class name of the object at `object_id` (via
/// [`HeapQuery::object_type_name`]).  For [`ResolvedRoot::StickyClass`] it is
/// the dot-notation class name of the pinned class itself.
#[derive(Debug, Clone)]
pub enum ResolvedRoot {
    /// `HPROF_GC_ROOT_UNKNOWN` — object kept alive by an unknown GC root.
    Unknown {
        object_id: u64,
        /// Runtime type name of the referenced object.
        object_type_name: std::string::String,
    },
    /// `HPROF_GC_ROOT_JNI_GLOBAL` — object held by a JNI global reference.
    JniGlobal {
        object_id: u64,
        /// The JNI global reference handle.
        jni_global_ref_id: u64,
        object_type_name: std::string::String,
    },
    /// `HPROF_GC_ROOT_JNI_LOCAL` — object held by a JNI local reference.
    JniLocal {
        object_id: u64,
        thread_serial: u32,
        frame_number: u32,
        object_type_name: std::string::String,
    },
    /// `HPROF_GC_ROOT_JAVA_FRAME` — object referenced from a Java stack frame.
    JavaFrame {
        object_id: u64,
        thread_serial: u32,
        frame_number: u32,
        object_type_name: std::string::String,
    },
    /// `HPROF_GC_ROOT_NATIVE_STACK` — object referenced from native code.
    NativeStack {
        object_id: u64,
        thread_serial: u32,
        object_type_name: std::string::String,
    },
    /// `HPROF_GC_ROOT_STICKY_CLASS` — a system/bootstrap class pinned by the VM.
    StickyClass {
        class_id: u64,
        /// Dot-notation name of the pinned class.
        class_name: std::string::String,
    },
    /// `HPROF_GC_ROOT_THREAD_BLOCK` — object referenced from a thread block.
    ThreadBlock {
        object_id: u64,
        thread_serial: u32,
        object_type_name: std::string::String,
    },
    /// `HPROF_GC_ROOT_MONITOR_USED` — object used as a monitor (synchronized).
    MonitorUsed {
        object_id: u64,
        object_type_name: std::string::String,
    },
    /// `HPROF_GC_ROOT_THREAD_OBJ` — a `java.lang.Thread` instance.
    ThreadObj {
        thread_object_id: u64,
        thread_serial: u32,
        stack_trace_serial: u32,
        object_type_name: std::string::String,
    },
}

impl ResolvedRoot {
    /// Resolve a [`RootUnknown`] record.
    pub fn from_unknown(query: &HeapQuery, root: &RootUnknown) -> Result<Self, HprofError> {
        let object_type_name = query.object_type_name(root.object_id)?;
        Ok(Self::Unknown {
            object_id: root.object_id,
            object_type_name,
        })
    }

    /// Resolve a [`RootJniGlobal`] record.
    pub fn from_jni_global(query: &HeapQuery, root: &RootJniGlobal) -> Result<Self, HprofError> {
        let object_type_name = query.object_type_name(root.object_id)?;
        Ok(Self::JniGlobal {
            object_id: root.object_id,
            jni_global_ref_id: root.jni_global_ref_id,
            object_type_name,
        })
    }

    /// Resolve a [`RootJniLocal`] record.
    pub fn from_jni_local(query: &HeapQuery, root: &RootJniLocal) -> Result<Self, HprofError> {
        let object_type_name = query.object_type_name(root.object_id)?;
        Ok(Self::JniLocal {
            object_id: root.object_id,
            thread_serial: root.thread_serial,
            frame_number: root.frame_number,
            object_type_name,
        })
    }

    /// Resolve a [`RootJavaFrame`] record.
    pub fn from_java_frame(query: &HeapQuery, root: &RootJavaFrame) -> Result<Self, HprofError> {
        let object_type_name = query.object_type_name(root.object_id)?;
        Ok(Self::JavaFrame {
            object_id: root.object_id,
            thread_serial: root.thread_serial,
            frame_number: root.frame_number,
            object_type_name,
        })
    }

    /// Resolve a [`RootNativeStack`] record.
    pub fn from_native_stack(
        query: &HeapQuery,
        root: &RootNativeStack,
    ) -> Result<Self, HprofError> {
        let object_type_name = query.object_type_name(root.object_id)?;
        Ok(Self::NativeStack {
            object_id: root.object_id,
            thread_serial: root.thread_serial,
            object_type_name,
        })
    }

    /// Resolve a [`RootStickyClass`] record.
    ///
    /// Resolves the class name via the load-class → UTF-8 name index chain.
    pub fn from_sticky_class(
        query: &HeapQuery,
        root: &RootStickyClass,
    ) -> Result<Self, HprofError> {
        let class_name = query.class_name(root.class_id)?.unwrap_or_default();
        Ok(Self::StickyClass {
            class_id: root.class_id,
            class_name,
        })
    }

    /// Resolve a [`RootThreadBlock`] record.
    pub fn from_thread_block(
        query: &HeapQuery,
        root: &RootThreadBlock,
    ) -> Result<Self, HprofError> {
        let object_type_name = query.object_type_name(root.object_id)?;
        Ok(Self::ThreadBlock {
            object_id: root.object_id,
            thread_serial: root.thread_serial,
            object_type_name,
        })
    }

    /// Resolve a [`RootMonitorUsed`] record.
    pub fn from_monitor_used(
        query: &HeapQuery,
        root: &RootMonitorUsed,
    ) -> Result<Self, HprofError> {
        let object_type_name = query.object_type_name(root.object_id)?;
        Ok(Self::MonitorUsed {
            object_id: root.object_id,
            object_type_name,
        })
    }

    /// Resolve a [`RootThreadObj`] record.
    pub fn from_thread_obj(query: &HeapQuery, root: &RootThreadObj) -> Result<Self, HprofError> {
        let object_type_name = query.object_type_name(root.thread_object_id)?;
        Ok(Self::ThreadObj {
            thread_object_id: root.thread_object_id,
            thread_serial: root.thread_serial,
            stack_trace_serial: root.stack_trace_serial,
            object_type_name,
        })
    }
}

// ── ResolvedPrimArray ─────────────────────────────────────────────────────────

/// The parsed elements of a primitive array.
///
/// Each variant holds all elements decoded from the big-endian raw bytes in the
/// hprof file.  The variant corresponds to the hprof element type code stored
/// in [`PrimArrayDump::element_type`].
#[derive(Debug, Clone, PartialEq)]
pub enum PrimArrayElements {
    /// `boolean[]` — element type code 4.
    Bool(Vec<bool>),
    /// `char[]` — element type code 5 (UTF-16 code units).
    Char(Vec<u16>),
    /// `float[]` — element type code 6.
    Float(Vec<f32>),
    /// `double[]` — element type code 7.
    Double(Vec<f64>),
    /// `byte[]` — element type code 8.
    Byte(Vec<i8>),
    /// `short[]` — element type code 9.
    Short(Vec<i16>),
    /// `int[]` — element type code 10.
    Int(Vec<i32>),
    /// `long[]` — element type code 11.
    Long(Vec<i64>),
}

/// A fully resolved primitive array dump.
///
/// The raw big-endian bytes from the hprof file are decoded into a typed
/// [`PrimArrayElements`] variant.  No object-ID resolution is needed since
/// primitive arrays never contain references.
#[derive(Debug, Clone)]
pub struct ResolvedPrimArray {
    /// The array's heap ID.
    pub array_id: u64,
    /// Stack trace serial from the `PRIM_ARRAY_DUMP` record.
    pub stack_trace_serial: u32,
    /// Number of elements in the array.
    pub num_elements: u32,
    /// Decoded array elements.
    pub elements: PrimArrayElements,
}

impl ResolvedPrimArray {
    /// Resolve a [`PrimArrayDump`] into a fully populated [`ResolvedPrimArray`].
    ///
    /// Decodes the raw big-endian bytes in `arr.data` according to
    /// `arr.element_type`.  Returns [`HprofError::UnknownPrimitiveType`] for
    /// unrecognised element type codes.
    pub fn from_dump(_query: &HeapQuery, arr: &PrimArrayDump<'_>) -> Result<Self, HprofError> {
        let elements = parse_prim_elements(arr.data, arr.element_type, arr.num_elements)?;
        Ok(Self {
            array_id: arr.array_id,
            stack_trace_serial: arr.stack_trace_serial,
            num_elements: arr.num_elements,
            elements,
        })
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Decode big-endian raw bytes into a [`PrimArrayElements`] variant.
fn parse_prim_elements(
    data: &[u8],
    element_type: u8,
    num_elements: u32,
) -> Result<PrimArrayElements, HprofError> {
    let n = num_elements as usize;
    match element_type {
        4 => {
            // boolean: 1 byte each (0 = false, anything else = true)
            Ok(PrimArrayElements::Bool(
                data[..n].iter().map(|&b| b != 0).collect(),
            ))
        }
        5 => {
            // char: 2 bytes big-endian each
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                let off = i * 2;
                v.push(u16::from_be_bytes([data[off], data[off + 1]]));
            }
            Ok(PrimArrayElements::Char(v))
        }
        6 => {
            // float: 4 bytes big-endian each
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                let off = i * 4;
                let bits =
                    u32::from_be_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
                v.push(f32::from_bits(bits));
            }
            Ok(PrimArrayElements::Float(v))
        }
        7 => {
            // double: 8 bytes big-endian each
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                let off = i * 8;
                let bits = u64::from_be_bytes([
                    data[off],
                    data[off + 1],
                    data[off + 2],
                    data[off + 3],
                    data[off + 4],
                    data[off + 5],
                    data[off + 6],
                    data[off + 7],
                ]);
                v.push(f64::from_bits(bits));
            }
            Ok(PrimArrayElements::Double(v))
        }
        8 => {
            // byte: 1 byte each (signed)
            Ok(PrimArrayElements::Byte(
                data[..n].iter().map(|&b| b as i8).collect(),
            ))
        }
        9 => {
            // short: 2 bytes big-endian each (signed)
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                let off = i * 2;
                v.push(i16::from_be_bytes([data[off], data[off + 1]]));
            }
            Ok(PrimArrayElements::Short(v))
        }
        10 => {
            // int: 4 bytes big-endian each (signed)
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                let off = i * 4;
                v.push(i32::from_be_bytes([
                    data[off],
                    data[off + 1],
                    data[off + 2],
                    data[off + 3],
                ]));
            }
            Ok(PrimArrayElements::Int(v))
        }
        11 => {
            // long: 8 bytes big-endian each (signed)
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                let off = i * 8;
                v.push(i64::from_be_bytes([
                    data[off],
                    data[off + 1],
                    data[off + 2],
                    data[off + 3],
                    data[off + 4],
                    data[off + 5],
                    data[off + 6],
                    data[off + 7],
                ]));
            }
            Ok(PrimArrayElements::Long(v))
        }
        other => Err(HprofError::UnknownPrimitiveType(other)),
    }
}

// ── Supporting types ──────────────────────────────────────────────────────────

/// A resolved instance field: name and value.
#[derive(Debug, Clone, PartialEq)]
pub struct InstanceField {
    /// Resolved field name.
    pub name: std::string::String,
    /// Resolved field value.
    pub value: Value,
}

/// A resolved static field declared on a class: name and value.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedStaticField {
    /// Resolved field name.
    pub name: std::string::String,
    /// Resolved field value.
    pub value: Value,
}

/// An instance field descriptor from a class dump: name and declared type.
///
/// This describes the layout of instances of the class; the actual values live
/// in the `INSTANCE_DUMP` records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldDescriptor {
    /// Resolved field name.
    pub name: std::string::String,
    /// Declared field type.
    pub field_type: FieldType,
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Convert a [`crate::heap_parser::FieldValue`] to a [`Value`].
///
/// Primitive variants are converted directly.  Object references are resolved
/// through [`HeapQuery::resolve_value`] to unwrap common wrapper types.
fn field_value_to_value(
    query: &HeapQuery,
    fv: crate::heap_parser::FieldValue,
) -> Result<Value, HprofError> {
    match fv {
        crate::heap_parser::FieldValue::Bool(v) => Ok(Value::Bool(v)),
        crate::heap_parser::FieldValue::Char(v) => Ok(Value::Char(v)),
        crate::heap_parser::FieldValue::Float(v) => Ok(Value::Float(v)),
        crate::heap_parser::FieldValue::Double(v) => Ok(Value::Double(v)),
        crate::heap_parser::FieldValue::Byte(v) => Ok(Value::Byte(v)),
        crate::heap_parser::FieldValue::Short(v) => Ok(Value::Short(v)),
        crate::heap_parser::FieldValue::Int(v) => Ok(Value::Int(v)),
        crate::heap_parser::FieldValue::Long(v) => Ok(Value::Long(v)),
        crate::heap_parser::FieldValue::Object(id) => {
            let jv = query.resolve_value(id)?;
            Ok(java_value_to_value(jv))
        }
    }
}

/// Convert a [`JavaValue`] (from wrapper type resolution) to a [`Value`].
fn java_value_to_value(jv: JavaValue) -> Value {
    match jv {
        JavaValue::Null => Value::Null,
        JavaValue::String(id, s) => Value::String(id, s),
        JavaValue::Integer(id, v) => Value::Integer(id, v),
        JavaValue::Long(id, v) => Value::BoxedLong(id, v),
        JavaValue::Double(id, v) => Value::BoxedDouble(id, v),
        JavaValue::Float(id, v) => Value::BoxedFloat(id, v),
        JavaValue::Short(id, v) => Value::BoxedShort(id, v),
        JavaValue::Byte(id, v) => Value::BoxedByte(id, v),
        JavaValue::Boolean(id, v) => Value::BoxedBoolean(id, v),
        JavaValue::Character(id, v) => Value::BoxedCharacter(id, v),
        JavaValue::Object(id) => Value::Object(id),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array_index::build_array_size_indexes;
    use crate::aux_index::{
        build_end_thread_index, build_frame_index, build_start_thread_index, build_trace_index,
        build_unload_class_index,
    };
    use crate::heap_index::index_heap_dumps;
    use crate::heap_parser::SubRecord;
    use crate::heap_query::build_name_indexes;
    use crate::object_store::combine_sort_and_split;
    use crate::record_index::index_hprof;
    use crate::ref_index::build_reference_index;
    use crate::vfs::{MMapReader, SubIndexDir};

    // ── Minimal hprof builder ─────────────────────────────────────────────────

    fn write_record(buf: &mut Vec<u8>, tag: u8, body: &[u8]) {
        buf.push(tag);
        buf.extend_from_slice(&0u32.to_be_bytes());
        buf.extend_from_slice(&(body.len() as u32).to_be_bytes());
        buf.extend_from_slice(body);
    }

    /// Build a minimal hprof with:
    ///
    /// - UTF8(1,"count"), UTF8(2,"java/lang/Object"), UTF8(3,"java/lang/Integer"),
    ///   UTF8(4,"value"), UTF8(5,"myField"), UTF8(6,"java/lang/String"),
    ///   UTF8(7,"[C"), UTF8(8,"java/util/ArrayList")
    /// - LOAD_CLASS for Integer (0x10), Object (0x20), String (0x30),
    ///   char[] (0x40), ArrayList (0x50)
    /// - HEAP_DUMP_SEGMENT:
    ///   CLASS_DUMP(0x10 = Integer, super=0x20, 1 int field "value")
    ///   CLASS_DUMP(0x20 = Object, super=0, 0 fields)
    ///   CLASS_DUMP(0x30 = String, super=0x20, 1 obj field "value" → char[])
    ///   CLASS_DUMP(0x40 = char[], super=0x20, 0 instance fields)
    ///   CLASS_DUMP(0x50 = ArrayList, super=0x20,
    ///   static: "count"(int)=7,
    ///   instance: "myField"(obj))
    ///   INSTANCE_DUMP(0x100, class=0x10, data=int(42))    -- Integer(42)
    ///   PRIM_ARRAY_DUMP(0x200, type=char, "hi")           -- char[] backing String
    ///   INSTANCE_DUMP(0x300, class=0x30, value→0x200)      -- String "hi"
    ///   INSTANCE_DUMP(0x400, class=0x50, myField→0x100)   -- ArrayList{myField=Integer(42)}
    fn build_test_hprof() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"JAVA PROFILE 1.0.2\0");
        buf.extend_from_slice(&8u32.to_be_bytes()); // id_size = 8
        buf.extend_from_slice(&0u64.to_be_bytes());

        let utf8 = |buf: &mut Vec<u8>, id: u64, s: &[u8]| {
            let mut body = Vec::new();
            body.extend_from_slice(&id.to_be_bytes());
            body.extend_from_slice(s);
            write_record(buf, 0x01, &body);
        };
        utf8(&mut buf, 1, b"count");
        utf8(&mut buf, 2, b"java/lang/Object");
        utf8(&mut buf, 3, b"java/lang/Integer");
        utf8(&mut buf, 4, b"value");
        utf8(&mut buf, 5, b"myField");
        utf8(&mut buf, 6, b"java/lang/String");
        utf8(&mut buf, 7, b"[C");
        utf8(&mut buf, 8, b"java/util/ArrayList");

        let load_class = |buf: &mut Vec<u8>, serial: u32, class_id: u64, name_id: u64| {
            let mut body = Vec::new();
            body.extend_from_slice(&serial.to_be_bytes());
            body.extend_from_slice(&class_id.to_be_bytes());
            body.extend_from_slice(&0u32.to_be_bytes());
            body.extend_from_slice(&name_id.to_be_bytes());
            write_record(buf, 0x02, &body);
        };
        load_class(&mut buf, 1, 0x10, 3); // Integer
        load_class(&mut buf, 2, 0x20, 2); // Object
        load_class(&mut buf, 3, 0x30, 6); // String
        load_class(&mut buf, 4, 0x40, 7); // char[]
        load_class(&mut buf, 5, 0x50, 8); // ArrayList

        let mut seg = Vec::new();

        // ── CLASS_DUMP(0x10 = Integer, super=0x20, 1 int field "value") ──
        seg.push(0x20u8);
        seg.extend_from_slice(&0x10u64.to_be_bytes()); // class_id
        seg.extend_from_slice(&0u32.to_be_bytes()); // stack_serial
        seg.extend_from_slice(&0x20u64.to_be_bytes()); // super
        seg.extend_from_slice(&[0u8; 8 * 5]); // loader+..
        seg.extend_from_slice(&4u32.to_be_bytes()); // instance_size
        seg.extend_from_slice(&0u16.to_be_bytes()); // cp_count
        seg.extend_from_slice(&0u16.to_be_bytes()); // statics_count
        seg.extend_from_slice(&1u16.to_be_bytes()); // instance_fields_count
        seg.extend_from_slice(&4u64.to_be_bytes()); // name_id=4 ("value")
        seg.push(10u8); // type = int

        // ── CLASS_DUMP(0x20 = Object, super=0) ──
        seg.push(0x20u8);
        seg.extend_from_slice(&0x20u64.to_be_bytes());
        seg.extend_from_slice(&0u32.to_be_bytes());
        seg.extend_from_slice(&0u64.to_be_bytes()); // no super
        seg.extend_from_slice(&[0u8; 8 * 5]);
        seg.extend_from_slice(&0u32.to_be_bytes());
        seg.extend_from_slice(&0u16.to_be_bytes());
        seg.extend_from_slice(&0u16.to_be_bytes());
        seg.extend_from_slice(&0u16.to_be_bytes());

        // ── CLASS_DUMP(0x30 = String, super=0x20, 1 obj field "value") ──
        seg.push(0x20u8);
        seg.extend_from_slice(&0x30u64.to_be_bytes());
        seg.extend_from_slice(&0u32.to_be_bytes());
        seg.extend_from_slice(&0x20u64.to_be_bytes()); // super = Object
        seg.extend_from_slice(&[0u8; 8 * 5]);
        seg.extend_from_slice(&8u32.to_be_bytes()); // instance_size = 8 (one id)
        seg.extend_from_slice(&0u16.to_be_bytes()); // cp_count
        seg.extend_from_slice(&0u16.to_be_bytes()); // statics_count
        seg.extend_from_slice(&1u16.to_be_bytes()); // instance_fields_count
        seg.extend_from_slice(&4u64.to_be_bytes()); // name_id=4 ("value")
        seg.push(2u8); // type = object

        // ── CLASS_DUMP(0x40 = char[], super=0x20, 0 fields) ──
        seg.push(0x20u8);
        seg.extend_from_slice(&0x40u64.to_be_bytes());
        seg.extend_from_slice(&0u32.to_be_bytes());
        seg.extend_from_slice(&0x20u64.to_be_bytes());
        seg.extend_from_slice(&[0u8; 8 * 5]);
        seg.extend_from_slice(&0u32.to_be_bytes());
        seg.extend_from_slice(&0u16.to_be_bytes());
        seg.extend_from_slice(&0u16.to_be_bytes());
        seg.extend_from_slice(&0u16.to_be_bytes());

        // ── CLASS_DUMP(0x50 = ArrayList, super=0x20) ──
        //    static: "count" (int) = 7
        //    instance: "myField" (object)
        seg.push(0x20u8);
        seg.extend_from_slice(&0x50u64.to_be_bytes());
        seg.extend_from_slice(&0u32.to_be_bytes());
        seg.extend_from_slice(&0x20u64.to_be_bytes()); // super = Object
        seg.extend_from_slice(&[0u8; 8 * 5]);
        seg.extend_from_slice(&8u32.to_be_bytes()); // instance_size = 8 (one id)
        seg.extend_from_slice(&0u16.to_be_bytes()); // cp_count = 0
        seg.extend_from_slice(&1u16.to_be_bytes()); // statics_count = 1
        // static field: name_id=1 ("count"), type=int(10), value=7
        seg.extend_from_slice(&1u64.to_be_bytes()); // name_id
        seg.push(10u8); // type = int
        seg.extend_from_slice(&7i32.to_be_bytes()); // value = 7
        seg.extend_from_slice(&1u16.to_be_bytes()); // instance_fields_count = 1
        // instance field: name_id=5 ("myField"), type=object(2)
        seg.extend_from_slice(&5u64.to_be_bytes()); // name_id
        seg.push(2u8); // type = object

        // ── INSTANCE_DUMP(0x100, class=0x10 Integer, data=42) ──
        seg.push(0x21u8);
        seg.extend_from_slice(&0x100u64.to_be_bytes()); // object_id
        seg.extend_from_slice(&0u32.to_be_bytes());
        seg.extend_from_slice(&0x10u64.to_be_bytes()); // class_id = Integer
        seg.extend_from_slice(&4u32.to_be_bytes()); // data_len = 4
        seg.extend_from_slice(&42i32.to_be_bytes()); // value = 42

        // ── PRIM_ARRAY_DUMP(0x200, type=char, ['h','i']) ──
        seg.push(0x23u8);
        seg.extend_from_slice(&0x200u64.to_be_bytes()); // array_id
        seg.extend_from_slice(&0u32.to_be_bytes()); // stack_serial
        seg.extend_from_slice(&2u32.to_be_bytes()); // num_elements = 2
        seg.push(5u8); // element_type = char
        seg.extend_from_slice(&('h' as u16).to_be_bytes());
        seg.extend_from_slice(&('i' as u16).to_be_bytes());

        // ── INSTANCE_DUMP(0x300, class=0x30 String, value→0x200) ──
        seg.push(0x21u8);
        seg.extend_from_slice(&0x300u64.to_be_bytes()); // object_id
        seg.extend_from_slice(&0u32.to_be_bytes());
        seg.extend_from_slice(&0x30u64.to_be_bytes()); // class_id = String
        seg.extend_from_slice(&8u32.to_be_bytes()); // data_len = 8 (one id)
        seg.extend_from_slice(&0x200u64.to_be_bytes()); // value field → char[] 0x200

        // ── INSTANCE_DUMP(0x400, class=0x50 ArrayList, myField→0x100) ──
        seg.push(0x21u8);
        seg.extend_from_slice(&0x400u64.to_be_bytes()); // object_id
        seg.extend_from_slice(&0u32.to_be_bytes());
        seg.extend_from_slice(&0x50u64.to_be_bytes()); // class_id = ArrayList
        seg.extend_from_slice(&8u32.to_be_bytes()); // data_len = 8 (one id)
        seg.extend_from_slice(&0x100u64.to_be_bytes()); // myField → Integer 0x100

        write_record(&mut buf, 0x1C, &seg); // HEAP_DUMP_SEGMENT
        buf
    }

    /// Build all indexes and return a [`HeapQuery`] entirely in memory.
    fn build_query(hprof_data: &[u8]) -> HeapQuery {
        let hprof = hprof_data.to_vec();
        let mut p1 = Vec::new();
        let p2d = SubIndexDir::mem();
        let mut p4 = Vec::new();
        let mut utf8 = Vec::new();
        let mut lc = Vec::new();
        let mut frames = Vec::new();
        let mut traces = Vec::new();
        let mut st = Vec::new();
        let mut et = Vec::new();
        let mut uc = Vec::new();
        let mut refs = Vec::new();
        // 9 separate root buffers (one per GC root type in canonical order).
        let mut r0: Vec<u8> = Vec::new();
        let mut r1: Vec<u8> = Vec::new();
        let mut r2: Vec<u8> = Vec::new();
        let mut r3: Vec<u8> = Vec::new();
        let mut r4: Vec<u8> = Vec::new();
        let mut r5: Vec<u8> = Vec::new();
        let mut r6: Vec<u8> = Vec::new();
        let mut r7: Vec<u8> = Vec::new();
        let mut r8: Vec<u8> = Vec::new();
        let mut arrays: [Vec<u8>; 9] = std::array::from_fn(|_| Vec::new());

        index_hprof(&hprof, &mut p1).unwrap();
        index_heap_dumps(&hprof, &p1, &p2d).unwrap();

        combine_sort_and_split(
            &p2d,
            &mut p4,
            &mut [
                &mut r0, &mut r1, &mut r2, &mut r3, &mut r4, &mut r5, &mut r6, &mut r7, &mut r8,
            ],
        )
        .unwrap();

        build_name_indexes(&hprof, &p1, &mut utf8, &mut lc).unwrap();
        build_reference_index(&hprof, &p4, &utf8, &lc, &mut refs).unwrap();
        build_frame_index(&hprof, &p1, &mut frames).unwrap();
        build_trace_index(&hprof, &p1, &mut traces).unwrap();
        build_start_thread_index(&hprof, &p1, &mut st).unwrap();
        build_end_thread_index(&hprof, &p1, &mut et).unwrap();
        build_unload_class_index(&hprof, &p1, &mut uc).unwrap();

        build_array_size_indexes(&hprof, &p4, &mut arrays).unwrap();

        HeapQuery::from_sources(
            &hprof,
            &p4,
            &utf8,
            &lc,
            &frames,
            &traces,
            &st,
            &et,
            &uc,
            &refs,
            [
                &r0 as &dyn MMapReader,
                &r1,
                &r2,
                &r3,
                &r4,
                &r5,
                &r6,
                &r7,
                &r8,
            ],
            [
                &arrays[0] as &dyn MMapReader,
                &arrays[1],
                &arrays[2],
                &arrays[3],
                &arrays[4],
                &arrays[5],
                &arrays[6],
                &arrays[7],
                &arrays[8],
            ],
        )
        .unwrap()
    }

    // ── ResolvedInstance tests ────────────────────────────────────────────────

    #[test]
    fn resolved_instance_class_name() {
        let query = build_query(&build_test_hprof());
        let SubRecord::InstanceDump(inst) = query.find_instance(0x400).unwrap().unwrap() else {
            panic!("expected InstanceDump");
        };
        let resolved = ResolvedInstance::from_dump(&query, &inst).unwrap();
        assert_eq!(resolved.object_id, 0x400);
        assert_eq!(resolved.class_name, "java.util.ArrayList");
    }

    #[test]
    fn resolved_instance_primitive_field() {
        let query = build_query(&build_test_hprof());
        let SubRecord::InstanceDump(inst) = query.find_instance(0x100).unwrap().unwrap() else {
            panic!("expected InstanceDump");
        };
        let resolved = ResolvedInstance::from_dump(&query, &inst).unwrap();
        assert_eq!(resolved.class_name, "java.lang.Integer");
        // Integer has one int field "value" = 42 (stored directly as Int)
        assert_eq!(resolved.fields.len(), 1);
        assert_eq!(resolved.fields[0].name, "value");
        assert_eq!(resolved.fields[0].value, Value::Int(42));
    }

    #[test]
    fn resolved_instance_object_field_resolved_to_integer() {
        let query = build_query(&build_test_hprof());
        let SubRecord::InstanceDump(inst) = query.find_instance(0x400).unwrap().unwrap() else {
            panic!("expected InstanceDump");
        };
        let resolved = ResolvedInstance::from_dump(&query, &inst).unwrap();
        // ArrayList.myField → Integer(0x100, 42)
        assert_eq!(resolved.fields.len(), 1);
        assert_eq!(resolved.fields[0].name, "myField");
        assert_eq!(resolved.fields[0].value, Value::Integer(0x100, 42));
    }

    #[test]
    fn resolved_instance_string_field() {
        let query = build_query(&build_test_hprof());
        let SubRecord::InstanceDump(inst) = query.find_instance(0x300).unwrap().unwrap() else {
            panic!("expected InstanceDump");
        };
        let resolved = ResolvedInstance::from_dump(&query, &inst).unwrap();
        assert_eq!(resolved.class_name, "java.lang.String");
        // String.value is the char[] backing array (0x200 — a PRIM_ARRAY_DUMP).
        // resolve_value(0x200) cannot unwrap a prim array as a String; it returns
        // Value::Object.  String resolution happens at the STRING INSTANCE level
        // (0x300), not the char-array level.
        assert_eq!(resolved.fields.len(), 1);
        assert_eq!(resolved.fields[0].name, "value");
        assert_eq!(resolved.fields[0].value, Value::Object(0x200));
    }

    /// When an object field in some other instance points to a String INSTANCE,
    /// resolve_value resolves it all the way to Value::String.
    #[test]
    fn object_field_pointing_to_string_instance_is_resolved() {
        let query = build_query(&build_test_hprof());
        // The ArrayList (0x400) has "myField" → Integer (0x100).
        // Verify that a String INSTANCE (0x300) is resolved correctly when
        // accessed via resolve_value directly (simulating a field reference).
        let java_val = query.resolve_value(0x300).unwrap();
        assert!(
            matches!(java_val, crate::heap_query::JavaValue::String(0x300, ref s) if s == "hi"),
            "expected String(0x300, \"hi\"), got {java_val:?}"
        );
        // And the corresponding Value conversion:
        let value = super::java_value_to_value(java_val);
        assert_eq!(value, Value::String(0x300, "hi".to_string()));
    }

    // ── ResolvedClass tests ───────────────────────────────────────────────────

    #[test]
    fn resolved_class_names() {
        let query = build_query(&build_test_hprof());
        let SubRecord::ClassDump(cd) = query.find_class(0x50).unwrap().unwrap() else {
            panic!("expected ClassDump");
        };
        let resolved = ResolvedClass::from_dump(&query, &cd).unwrap();
        assert_eq!(resolved.class_id, 0x50);
        assert_eq!(resolved.class_name, "java.util.ArrayList");
        assert_eq!(
            resolved.super_class_name,
            Some("java.lang.Object".to_string())
        );
    }

    #[test]
    fn resolved_class_static_fields() {
        let query = build_query(&build_test_hprof());
        let SubRecord::ClassDump(cd) = query.find_class(0x50).unwrap().unwrap() else {
            panic!("expected ClassDump");
        };
        let resolved = ResolvedClass::from_dump(&query, &cd).unwrap();
        assert_eq!(resolved.static_fields.len(), 1);
        assert_eq!(resolved.static_fields[0].name, "count");
        assert_eq!(resolved.static_fields[0].value, Value::Int(7));
    }

    #[test]
    fn resolved_class_instance_field_descriptors() {
        let query = build_query(&build_test_hprof());
        let SubRecord::ClassDump(cd) = query.find_class(0x50).unwrap().unwrap() else {
            panic!("expected ClassDump");
        };
        let resolved = ResolvedClass::from_dump(&query, &cd).unwrap();
        assert_eq!(resolved.instance_fields.len(), 1);
        assert_eq!(resolved.instance_fields[0].name, "myField");
        assert_eq!(resolved.instance_fields[0].field_type, FieldType::Object);
    }

    #[test]
    fn field_type_from_type_code() {
        assert_eq!(FieldType::from_type_code(2), Some(FieldType::Object));
        assert_eq!(FieldType::from_type_code(4), Some(FieldType::Bool));
        assert_eq!(FieldType::from_type_code(10), Some(FieldType::Int));
        assert_eq!(FieldType::from_type_code(11), Some(FieldType::Long));
        assert_eq!(FieldType::from_type_code(99), None);
    }

    // ── ResolvedPrimArray tests ───────────────────────────────────────────────

    #[test]
    fn resolved_prim_array_metadata() {
        let query = build_query(&build_test_hprof());
        let SubRecord::PrimArrayDump(arr) = query.find(0x200).unwrap().unwrap() else {
            panic!("expected PrimArrayDump");
        };
        let resolved = ResolvedPrimArray::from_dump(&query, &arr).unwrap();
        assert_eq!(resolved.array_id, 0x200);
        assert_eq!(resolved.stack_trace_serial, 0);
        assert_eq!(resolved.num_elements, 2);
    }

    #[test]
    fn resolved_prim_array_char_elements() {
        let query = build_query(&build_test_hprof());
        let SubRecord::PrimArrayDump(arr) = query.find(0x200).unwrap().unwrap() else {
            panic!("expected PrimArrayDump");
        };
        let resolved = ResolvedPrimArray::from_dump(&query, &arr).unwrap();
        assert_eq!(
            resolved.elements,
            PrimArrayElements::Char(vec!['h' as u16, 'i' as u16])
        );
    }

    #[test]
    fn resolved_prim_array_int_roundtrip() {
        // Build a minimal hprof with a single int[] of [1, 2, 3].
        let mut buf = Vec::new();
        buf.extend_from_slice(b"JAVA PROFILE 1.0.2\0");
        buf.extend_from_slice(&8u32.to_be_bytes()); // id_size = 8
        buf.extend_from_slice(&0u64.to_be_bytes());

        let mut seg = Vec::new();
        seg.push(0x23u8); // PRIM_ARRAY_DUMP
        seg.extend_from_slice(&0x500u64.to_be_bytes()); // array_id
        seg.extend_from_slice(&0u32.to_be_bytes()); // stack_serial
        seg.extend_from_slice(&3u32.to_be_bytes()); // num_elements = 3
        seg.push(10u8); // element_type = int
        seg.extend_from_slice(&1i32.to_be_bytes());
        seg.extend_from_slice(&2i32.to_be_bytes());
        seg.extend_from_slice(&3i32.to_be_bytes());

        buf.push(0x1Cu8); // HEAP_DUMP_SEGMENT
        buf.extend_from_slice(&0u32.to_be_bytes());
        buf.extend_from_slice(&(seg.len() as u32).to_be_bytes());
        buf.extend_from_slice(&seg);

        let query = build_query(&buf);
        let SubRecord::PrimArrayDump(arr) = query.find(0x500).unwrap().unwrap() else {
            panic!("expected PrimArrayDump");
        };
        let resolved = ResolvedPrimArray::from_dump(&query, &arr).unwrap();
        assert_eq!(resolved.num_elements, 3);
        assert_eq!(resolved.elements, PrimArrayElements::Int(vec![1, 2, 3]));
    }

    #[test]
    fn resolved_prim_array_unknown_type_error() {
        let data = &[0u8; 4];
        let result = super::parse_prim_elements(data, 99, 1);
        assert!(matches!(result, Err(HprofError::UnknownPrimitiveType(99))));
    }

    // ── ResolvedObjArray tests ────────────────────────────────────────────────

    /// Build a minimal hprof with:
    /// - UTF8(1,"java/lang/Integer"), UTF8(2,"value"), UTF8(3,"java/lang/Object")
    /// - LOAD_CLASS(0x10=Integer, name_id=1), LOAD_CLASS(0x20=Object, name_id=3)
    /// - HEAP_DUMP_SEGMENT:
    ///   CLASS_DUMP(0x10=Integer, super=0x20, 1 int field "value")
    ///   CLASS_DUMP(0x20=Object, super=0)
    ///   INSTANCE_DUMP(0x100, class=0x10, value=99)
    ///   OBJ_ARRAY_DUMP(0x200, elem_class=0x10, elements=[0x100, 0])
    fn build_obj_array_hprof() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"JAVA PROFILE 1.0.2\0");
        buf.extend_from_slice(&8u32.to_be_bytes());
        buf.extend_from_slice(&0u64.to_be_bytes());

        let utf8 = |buf: &mut Vec<u8>, id: u64, s: &[u8]| {
            let mut body = Vec::new();
            body.extend_from_slice(&id.to_be_bytes());
            body.extend_from_slice(s);
            buf.push(0x01);
            buf.extend_from_slice(&0u32.to_be_bytes());
            buf.extend_from_slice(&(body.len() as u32).to_be_bytes());
            buf.extend_from_slice(&body);
        };
        utf8(&mut buf, 1, b"java/lang/Integer");
        utf8(&mut buf, 2, b"value");
        utf8(&mut buf, 3, b"java/lang/Object");

        let load_class = |buf: &mut Vec<u8>, serial: u32, class_id: u64, name_id: u64| {
            let mut body = Vec::new();
            body.extend_from_slice(&serial.to_be_bytes());
            body.extend_from_slice(&class_id.to_be_bytes());
            body.extend_from_slice(&0u32.to_be_bytes());
            body.extend_from_slice(&name_id.to_be_bytes());
            buf.push(0x02);
            buf.extend_from_slice(&0u32.to_be_bytes());
            buf.extend_from_slice(&(body.len() as u32).to_be_bytes());
            buf.extend_from_slice(&body);
        };
        load_class(&mut buf, 1, 0x10, 1); // Integer
        load_class(&mut buf, 2, 0x20, 3); // Object

        let mut seg = Vec::new();

        // CLASS_DUMP(0x10=Integer, super=0x20, instance_size=4, 1 int field "value")
        seg.push(0x20u8);
        seg.extend_from_slice(&0x10u64.to_be_bytes());
        seg.extend_from_slice(&0u32.to_be_bytes());
        seg.extend_from_slice(&0x20u64.to_be_bytes());
        seg.extend_from_slice(&[0u8; 8 * 5]);
        seg.extend_from_slice(&4u32.to_be_bytes());
        seg.extend_from_slice(&0u16.to_be_bytes()); // cp=0
        seg.extend_from_slice(&0u16.to_be_bytes()); // statics=0
        seg.extend_from_slice(&1u16.to_be_bytes()); // fields=1
        seg.extend_from_slice(&2u64.to_be_bytes()); // name_id=2 "value"
        seg.push(10u8); // int

        // CLASS_DUMP(0x20=Object, super=0, no fields)
        seg.push(0x20u8);
        seg.extend_from_slice(&0x20u64.to_be_bytes());
        seg.extend_from_slice(&0u32.to_be_bytes());
        seg.extend_from_slice(&0u64.to_be_bytes());
        seg.extend_from_slice(&[0u8; 8 * 5]);
        seg.extend_from_slice(&0u32.to_be_bytes());
        seg.extend_from_slice(&0u16.to_be_bytes());
        seg.extend_from_slice(&0u16.to_be_bytes());
        seg.extend_from_slice(&0u16.to_be_bytes());

        // INSTANCE_DUMP(0x100, class=0x10 Integer, value=99)
        seg.push(0x21u8);
        seg.extend_from_slice(&0x100u64.to_be_bytes());
        seg.extend_from_slice(&0u32.to_be_bytes());
        seg.extend_from_slice(&0x10u64.to_be_bytes());
        seg.extend_from_slice(&4u32.to_be_bytes());
        seg.extend_from_slice(&99i32.to_be_bytes());

        // OBJ_ARRAY_DUMP(0x200, elem_class=0x10, 2 elements: [0x100, 0])
        seg.push(0x22u8);
        seg.extend_from_slice(&0x200u64.to_be_bytes()); // array_id
        seg.extend_from_slice(&0u32.to_be_bytes()); // stack_serial
        seg.extend_from_slice(&2u32.to_be_bytes()); // num_elements
        seg.extend_from_slice(&0x10u64.to_be_bytes()); // element_class_id
        seg.extend_from_slice(&0x100u64.to_be_bytes()); // elem[0] → Integer(99)
        seg.extend_from_slice(&0u64.to_be_bytes()); // elem[1] → null

        buf.push(0x1Cu8); // HEAP_DUMP_SEGMENT
        buf.extend_from_slice(&0u32.to_be_bytes());
        buf.extend_from_slice(&(seg.len() as u32).to_be_bytes());
        buf.extend_from_slice(&seg);

        buf
    }

    #[test]
    fn resolved_obj_array_metadata() {
        let query = build_query(&build_obj_array_hprof());
        let SubRecord::ObjArrayDump(arr) = query.find(0x200).unwrap().unwrap() else {
            panic!("expected ObjArrayDump");
        };
        let resolved = ResolvedObjArray::from_dump(&query, &arr).unwrap();
        assert_eq!(resolved.array_id, 0x200);
        assert_eq!(resolved.num_elements, 2);
        assert_eq!(resolved.element_class_name, "java.lang.Integer");
    }

    #[test]
    fn resolved_obj_array_elements_resolved() {
        let query = build_query(&build_obj_array_hprof());
        let SubRecord::ObjArrayDump(arr) = query.find(0x200).unwrap().unwrap() else {
            panic!("expected ObjArrayDump");
        };
        let resolved = ResolvedObjArray::from_dump(&query, &arr).unwrap();
        // elem[0] is Integer(0x100, 99), elem[1] is null
        assert_eq!(resolved.elements[0], Value::Integer(0x100, 99));
        assert_eq!(resolved.elements[1], Value::Null);
    }

    // ── ResolvedRoot tests ────────────────────────────────────────────────────
    //
    // The raw root structs are constructed directly; HeapQuery is used only for
    // object_type_name / class_name resolution against the existing test objects.

    #[test]
    fn resolved_root_unknown() {
        let query = build_query(&build_test_hprof());
        // 0x400 is java.util.ArrayList in the test hprof
        let root = crate::heap_parser::RootUnknown { object_id: 0x400 };
        let resolved = ResolvedRoot::from_unknown(&query, &root).unwrap();
        assert!(matches!(
            resolved,
            ResolvedRoot::Unknown {
                object_id: 0x400,
                ..
            }
        ));
        if let ResolvedRoot::Unknown {
            object_type_name, ..
        } = &resolved
        {
            assert_eq!(object_type_name, "java.util.ArrayList");
        }
    }

    #[test]
    fn resolved_root_jni_global() {
        let query = build_query(&build_test_hprof());
        let root = crate::heap_parser::RootJniGlobal {
            object_id: 0x100,
            jni_global_ref_id: 0xABCD,
        };
        let resolved = ResolvedRoot::from_jni_global(&query, &root).unwrap();
        if let ResolvedRoot::JniGlobal {
            object_id,
            jni_global_ref_id,
            object_type_name,
        } = &resolved
        {
            assert_eq!(*object_id, 0x100);
            assert_eq!(*jni_global_ref_id, 0xABCD);
            assert_eq!(object_type_name, "java.lang.Integer");
        } else {
            panic!("expected JniGlobal");
        }
    }

    #[test]
    fn resolved_root_jni_local() {
        let query = build_query(&build_test_hprof());
        let root = crate::heap_parser::RootJniLocal {
            object_id: 0x100,
            thread_serial: 1,
            frame_number: 2,
        };
        let resolved = ResolvedRoot::from_jni_local(&query, &root).unwrap();
        if let ResolvedRoot::JniLocal {
            thread_serial,
            frame_number,
            ..
        } = &resolved
        {
            assert_eq!(*thread_serial, 1);
            assert_eq!(*frame_number, 2);
        } else {
            panic!("expected JniLocal");
        }
    }

    #[test]
    fn resolved_root_java_frame() {
        let query = build_query(&build_test_hprof());
        let root = crate::heap_parser::RootJavaFrame {
            object_id: 0x100,
            thread_serial: 3,
            frame_number: 7,
        };
        let resolved = ResolvedRoot::from_java_frame(&query, &root).unwrap();
        if let ResolvedRoot::JavaFrame {
            thread_serial,
            frame_number,
            ..
        } = &resolved
        {
            assert_eq!(*thread_serial, 3);
            assert_eq!(*frame_number, 7);
        } else {
            panic!("expected JavaFrame");
        }
    }

    #[test]
    fn resolved_root_native_stack() {
        let query = build_query(&build_test_hprof());
        let root = crate::heap_parser::RootNativeStack {
            object_id: 0x400,
            thread_serial: 5,
        };
        let resolved = ResolvedRoot::from_native_stack(&query, &root).unwrap();
        if let ResolvedRoot::NativeStack {
            thread_serial,
            object_type_name,
            ..
        } = &resolved
        {
            assert_eq!(*thread_serial, 5);
            assert_eq!(object_type_name, "java.util.ArrayList");
        } else {
            panic!("expected NativeStack");
        }
    }

    #[test]
    fn resolved_root_sticky_class() {
        let query = build_query(&build_test_hprof());
        // 0x10 is the Integer class in the test hprof
        let root = crate::heap_parser::RootStickyClass { class_id: 0x10 };
        let resolved = ResolvedRoot::from_sticky_class(&query, &root).unwrap();
        if let ResolvedRoot::StickyClass {
            class_id,
            class_name,
        } = &resolved
        {
            assert_eq!(*class_id, 0x10);
            assert_eq!(class_name, "java.lang.Integer");
        } else {
            panic!("expected StickyClass");
        }
    }

    #[test]
    fn resolved_root_thread_block() {
        let query = build_query(&build_test_hprof());
        let root = crate::heap_parser::RootThreadBlock {
            object_id: 0x400,
            thread_serial: 9,
        };
        let resolved = ResolvedRoot::from_thread_block(&query, &root).unwrap();
        if let ResolvedRoot::ThreadBlock { thread_serial, .. } = &resolved {
            assert_eq!(*thread_serial, 9);
        } else {
            panic!("expected ThreadBlock");
        }
    }

    #[test]
    fn resolved_root_monitor_used() {
        let query = build_query(&build_test_hprof());
        let root = crate::heap_parser::RootMonitorUsed { object_id: 0x300 };
        let resolved = ResolvedRoot::from_monitor_used(&query, &root).unwrap();
        if let ResolvedRoot::MonitorUsed {
            object_type_name, ..
        } = &resolved
        {
            assert_eq!(object_type_name, "java.lang.String");
        } else {
            panic!("expected MonitorUsed");
        }
    }

    #[test]
    fn resolved_root_thread_obj() {
        let query = build_query(&build_test_hprof());
        let root = crate::heap_parser::RootThreadObj {
            thread_object_id: 0x400,
            thread_serial: 2,
            stack_trace_serial: 42,
        };
        let resolved = ResolvedRoot::from_thread_obj(&query, &root).unwrap();
        if let ResolvedRoot::ThreadObj {
            thread_serial,
            stack_trace_serial,
            ..
        } = &resolved
        {
            assert_eq!(*thread_serial, 2);
            assert_eq!(*stack_trace_serial, 42);
        } else {
            panic!("expected ThreadObj");
        }
    }

    #[test]
    fn resolved_root_unknown_object_id_not_found_falls_back_to_object() {
        let query = build_query(&build_test_hprof());
        // Use an object_id that does not exist in the index.
        let root = crate::heap_parser::RootUnknown { object_id: 0xDEAD };
        let resolved = ResolvedRoot::from_unknown(&query, &root).unwrap();
        if let ResolvedRoot::Unknown {
            object_type_name, ..
        } = &resolved
        {
            assert_eq!(object_type_name, "Object");
        } else {
            panic!("expected Unknown");
        }
    }
}
