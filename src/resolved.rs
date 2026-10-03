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
//! ```no_run
//! use hprof_toolkit::prelude::*;
//!
//! let heap = HeapQuery::open("heap.hprof")?;
//! for record in heap.objects() {
//!     match record?.resolve(&heap)? {
//!         Resolved::Instance(i) => println!("{}: {:?}", i.class_name, i.fields),
//!         Resolved::Class(c) => println!("class {} ({} statics)", c.class_name, c.static_fields.len()),
//!         _ => {}
//!     }
//! }
//! # Ok::<(), HprofError>(())
//! ```

use crate::class_key::ClassKey;
use crate::heap_parser::{ClassDump, InstanceDump, ObjArrayDump, PrimArrayDump, SubRecord};
use crate::hprof::{BasicType, HprofError};
use crate::query::{HeapQuery, ScanResult, ScanWindow, StringMatch};
use crate::search::{Matcher, SearchMode};

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
    BoxedInt(u64, i32),
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

/// The declared type of a field: an alias of the crate-wide
/// [`BasicType`].
pub type FieldType = crate::hprof::BasicType;

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
        let class_name = query
            .class_name(inst.class_id)
            .map(|n| n.to_string())
            .unwrap_or_default();
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
        let class_name = query
            .class_name(cd.class_id)
            .map(|n| n.to_string())
            .unwrap_or_default();

        let super_class_name = if cd.super_class_id != 0 {
            query.class_name(cd.super_class_id).map(|n| n.to_string())
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
            let field_type = FieldType::from_code(fd.field_type).unwrap_or(FieldType::Object);
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
/// The array class name is resolved via the load-class → UTF-8 name index
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
    /// Heap ID of the array's class object (`[Ljava.lang.String;`).
    pub array_class_id: u64,
    /// JVM name of the array class (e.g. `"[Ljava.lang.String;"`); use
    /// [`HeapQuery::key_name`] with `ClassKey::ObjArray(array_class_id)` for
    /// the Java form `java.lang.String[]`.
    pub array_class_name: std::string::String,
    /// Resolved element values in array order.
    pub elements: Vec<Value>,
}

impl ResolvedObjArray {
    /// Resolve an [`ObjArrayDump`] into a fully populated [`ResolvedObjArray`].
    ///
    /// Resolves the array class name and then resolves each element object ID
    /// through [`HeapQuery::resolve_value`].
    pub fn from_dump(query: &HeapQuery, arr: &ObjArrayDump<'_>) -> Result<Self, HprofError> {
        let array_class_name = query
            .class_name(arr.array_class_id)
            .map(|n| n.to_string())
            .unwrap_or_default();
        let elements = arr
            .elements()
            .map(|id| query.resolve_value(id))
            .collect::<Result<Vec<_>, HprofError>>()?;
        Ok(Self {
            array_id: arr.array_id,
            stack_trace_serial: arr.stack_trace_serial,
            num_elements: arr.num_elements,
            array_class_id: arr.array_class_id,
            array_class_name,
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
    /// The id of the rooted object (the class for a sticky-class root, the
    /// thread object for a thread-object root).
    pub fn object_id(&self) -> u64 {
        match self {
            Self::Unknown { object_id, .. }
            | Self::JniGlobal { object_id, .. }
            | Self::JniLocal { object_id, .. }
            | Self::JavaFrame { object_id, .. }
            | Self::NativeStack { object_id, .. }
            | Self::ThreadBlock { object_id, .. }
            | Self::MonitorUsed { object_id, .. } => *object_id,
            Self::StickyClass { class_id, .. } => *class_id,
            Self::ThreadObj {
                thread_object_id, ..
            } => *thread_object_id,
        }
    }

    /// Type name of the rooted object (the class name for a sticky class).
    pub fn type_name(&self) -> &str {
        match self {
            Self::Unknown {
                object_type_name, ..
            }
            | Self::JniGlobal {
                object_type_name, ..
            }
            | Self::JniLocal {
                object_type_name, ..
            }
            | Self::JavaFrame {
                object_type_name, ..
            }
            | Self::NativeStack {
                object_type_name, ..
            }
            | Self::ThreadBlock {
                object_type_name, ..
            }
            | Self::MonitorUsed {
                object_type_name, ..
            }
            | Self::ThreadObj {
                object_type_name, ..
            } => object_type_name,
            Self::StickyClass { class_name, .. } => class_name,
        }
    }

    /// Resolve a GC-root sub-record, naming the rooted object's type.
    ///
    /// Returns `Ok(None)` when `record` is not one of the nine root kinds.
    pub fn from_sub_record(
        query: &HeapQuery,
        record: &SubRecord<'_>,
    ) -> Result<Option<Self>, HprofError> {
        let type_name = |id: u64| query.object_type_name(id);
        Ok(Some(match record {
            SubRecord::RootUnknown(r) => Self::Unknown {
                object_id: r.object_id,
                object_type_name: type_name(r.object_id),
            },
            SubRecord::RootJniGlobal(r) => Self::JniGlobal {
                object_id: r.object_id,
                jni_global_ref_id: r.jni_global_ref_id,
                object_type_name: type_name(r.object_id),
            },
            SubRecord::RootJniLocal(r) => Self::JniLocal {
                object_id: r.object_id,
                thread_serial: r.thread_serial,
                frame_number: r.frame_number,
                object_type_name: type_name(r.object_id),
            },
            SubRecord::RootJavaFrame(r) => Self::JavaFrame {
                object_id: r.object_id,
                thread_serial: r.thread_serial,
                frame_number: r.frame_number,
                object_type_name: type_name(r.object_id),
            },
            SubRecord::RootNativeStack(r) => Self::NativeStack {
                object_id: r.object_id,
                thread_serial: r.thread_serial,
                object_type_name: type_name(r.object_id),
            },
            SubRecord::RootStickyClass(r) => Self::StickyClass {
                class_id: r.class_id,
                class_name: query.class_label(r.class_id),
            },
            SubRecord::RootThreadBlock(r) => Self::ThreadBlock {
                object_id: r.object_id,
                thread_serial: r.thread_serial,
                object_type_name: type_name(r.object_id),
            },
            SubRecord::RootMonitorUsed(r) => Self::MonitorUsed {
                object_id: r.object_id,
                object_type_name: type_name(r.object_id),
            },
            SubRecord::RootThreadObj(r) => Self::ThreadObj {
                thread_object_id: r.thread_object_id,
                thread_serial: r.thread_serial,
                stack_trace_serial: r.stack_trace_serial,
                object_type_name: type_name(r.thread_object_id),
            },
            _ => return Ok(None),
        }))
    }
}

// ── Resolved (any sub-record) ─────────────────────────────────────────────────

/// Any sub-record, fully resolved.  Produced by [`SubRecord::resolve`].
#[derive(Debug, Clone)]
pub enum Resolved {
    Instance(ResolvedInstance),
    Class(ResolvedClass),
    ObjArray(ResolvedObjArray),
    PrimArray(ResolvedPrimArray),
    Root(ResolvedRoot),
}

impl SubRecord<'_> {
    /// Resolve this record: names looked up, wrapper objects unwrapped,
    /// primitive arrays decoded.  One call for any kind of object.
    ///
    /// ```no_run
    /// # use hprof_toolkit::prelude::*;
    /// # let heap = HeapQuery::open("heap.hprof")?;
    /// if let Some(record) = heap.object(0x1234)? {
    ///     match record.resolve(&heap)? {
    ///         Resolved::Instance(i) => println!("{} with {} fields", i.class_name, i.fields.len()),
    ///         other => println!("{other:?}"),
    ///     }
    /// }
    /// # Ok::<(), HprofError>(())
    /// ```
    pub fn resolve(&self, query: &HeapQuery) -> Result<Resolved, HprofError> {
        Ok(match self {
            SubRecord::InstanceDump(i) => {
                Resolved::Instance(ResolvedInstance::from_dump(query, i)?)
            }
            SubRecord::ClassDump(c) => Resolved::Class(ResolvedClass::from_dump(query, c)?),
            SubRecord::ObjArrayDump(a) => {
                Resolved::ObjArray(ResolvedObjArray::from_dump(query, a)?)
            }
            SubRecord::PrimArrayDump(a) => Resolved::PrimArray(ResolvedPrimArray::from_dump(a)?),
            root => match ResolvedRoot::from_sub_record(query, root)? {
                Some(r) => Resolved::Root(r),
                None => {
                    return Err(HprofError::Internal(
                        "sub-record is neither an object nor a GC root".to_owned(),
                    ));
                }
            },
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

impl PrimArrayElements {
    /// Number of decoded elements.
    pub fn len(&self) -> usize {
        match self {
            Self::Bool(v) => v.len(),
            Self::Char(v) => v.len(),
            Self::Float(v) => v.len(),
            Self::Double(v) => v.len(),
            Self::Byte(v) => v.len(),
            Self::Short(v) => v.len(),
            Self::Int(v) => v.len(),
            Self::Long(v) => v.len(),
        }
    }

    /// `true` when no elements were decoded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Each element rendered as text.  With `quote_chars`, `char` elements
    /// appear as `'a'`; without, as `a`.  Invalid UTF-16 units show as U+FFFD.
    pub fn to_strings(&self, quote_chars: bool) -> Vec<String> {
        fn all<T: ToString>(v: &[T]) -> Vec<String> {
            v.iter().map(T::to_string).collect()
        }
        match self {
            Self::Bool(v) => all(v),
            Self::Char(v) => v
                .iter()
                .map(|&u| {
                    let c = char::from_u32(u32::from(u)).unwrap_or('\u{FFFD}');
                    if quote_chars {
                        format!("'{c}'")
                    } else {
                        c.to_string()
                    }
                })
                .collect(),
            Self::Float(v) => all(v),
            Self::Double(v) => all(v),
            Self::Byte(v) => all(v),
            Self::Short(v) => all(v),
            Self::Int(v) => all(v),
            Self::Long(v) => all(v),
        }
    }
}

/// A window of a primitive array's elements; see [`HeapQuery::prim_array`].
#[derive(Debug, Clone, PartialEq)]
pub struct PrimArrayWindow {
    /// The array's object id.
    pub array_id: u64,
    /// Element type (never [`BasicType::Object`]).
    pub element_type: BasicType,
    /// Length of the whole array.
    pub total: usize,
    /// Index of the first decoded element.
    pub offset: usize,
    /// The decoded elements, at most the requested limit.
    pub elements: PrimArrayElements,
}

impl PrimArrayWindow {
    /// `true` when elements exist beyond this window.
    pub fn has_more(&self) -> bool {
        self.offset + self.elements.len() < self.total
    }
}

impl HeapQuery {
    /// Decode a window of the primitive array `array_id`, reading only the
    /// bytes inside the window.  `None` when `array_id` is not a primitive
    /// array.
    pub fn prim_array(
        &self,
        array_id: u64,
        page: crate::query::Page,
    ) -> Result<Option<PrimArrayWindow>, HprofError> {
        let Some(SubRecord::PrimArrayDump(arr)) = self.object(array_id)? else {
            return Ok(None);
        };
        let elements = parse_prim_window(arr.data, arr.element_type, page.offset, page.limit)?;
        Ok(Some(PrimArrayWindow {
            array_id,
            element_type: BasicType::from_code_or_err(arr.element_type)?,
            total: arr.num_elements as usize,
            offset: page.offset.min(arr.num_elements as usize),
            elements,
        }))
    }

    /// The text of the `java.lang.String` `string_id`; `None` when the id is
    /// not a String.  Handles both `char[]` and compact `byte[]` strings.
    pub fn string(&self, string_id: u64) -> Result<Option<String>, HprofError> {
        Ok(match self.resolve_value(string_id)? {
            Value::String(_, s) => Some(s),
            _ => None,
        })
    }

    /// `java.lang.String` objects whose text matches `matcher`, scanning
    /// strings in id order within `window`.
    ///
    /// Reads string contents, so it is bounded: at most `window.max_scan`
    /// strings are decoded per call and the result says where to continue.
    /// Fuzzy matching is refused (`InvalidArgument`): a subsequence matches
    /// almost any text and cannot rank across a bounded window.
    ///
    /// ```no_run
    /// use hprof_toolkit::prelude::*;
    ///
    /// let heap = HeapQuery::open("heap.hprof")?;
    /// let m = Matcher::new(&SearchQuery::regex(r"^jdbc:"))?;
    /// let mut window = ScanWindow::first();
    /// loop {
    ///     let found = heap.search_strings(&m, window)?;
    ///     for s in &found.items {
    ///         println!("0x{:x} {}", s.object_id, s.preview);
    ///     }
    ///     match found.next_window(window) {
    ///         Some(next) => window = next,
    ///         None => break,
    ///     }
    /// }
    /// # Ok::<(), HprofError>(())
    /// ```
    pub fn search_strings(
        &self,
        matcher: &Matcher,
        window: ScanWindow,
    ) -> Result<ScanResult<StringMatch>, HprofError> {
        if matcher.mode() == SearchMode::Fuzzy {
            return Err(HprofError::InvalidArgument(
                "fuzzy search is not available for string contents; use contains, exact or regex"
                    .to_owned(),
            ));
        }
        let Some(string_class) = self.find_class_by_name("java.lang.String") else {
            return Ok(ScanResult::empty());
        };
        let key = ClassKey::Class(string_class);
        let total = self.instance_count(key);
        let mut items = Vec::new();
        let mut scanned = 0usize;
        // Skip on the index entries (O(1)) and decode only the strings scanned.
        for entry in self
            .class_entries(key)
            .skip(window.cursor)
            .take(window.max_scan)
        {
            scanned += 1;
            let Some(inst) = self.instance_from_entry(&entry).transpose()? else {
                continue;
            };
            let text = self.string_of(&inst)?;
            if matcher.is_match(&text) {
                items.push(StringMatch {
                    object_id: inst.object_id,
                    length_chars: text.chars().count(),
                    preview: text.chars().take(StringMatch::PREVIEW_CHARS).collect(),
                });
                if items.len() >= window.max_results {
                    break;
                }
            }
        }
        let next = window.cursor + scanned;
        Ok(ScanResult {
            items,
            scanned,
            total,
            next_cursor: (next < total).then_some(next),
        })
    }
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
    pub fn from_dump(arr: &PrimArrayDump<'_>) -> Result<Self, HprofError> {
        let elements = parse_prim_window(arr.data, arr.element_type, 0, arr.num_elements as usize)?;
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
/// Decode elements `offset..offset + limit` (clamped to the array) of a
/// primitive array whose raw big-endian bytes are `data`.
fn parse_prim_window(
    data: &[u8],
    element_type: u8,
    offset: usize,
    limit: usize,
) -> Result<PrimArrayElements, HprofError> {
    let ty = BasicType::from_code_or_err(element_type)?;
    if !ty.is_primitive() {
        return Err(HprofError::UnknownPrimitiveType(element_type));
    }
    let size = ty.size(0);
    let total = data.len() / size;
    let start = offset.min(total);
    let end = start.saturating_add(limit).min(total);
    let window = &data[start * size..end * size];

    fn chunks<const N: usize, T>(window: &[u8], f: impl Fn([u8; N]) -> T) -> Vec<T> {
        window.as_chunks::<N>().0.iter().map(|c| f(*c)).collect()
    }

    Ok(match ty {
        BasicType::Boolean => PrimArrayElements::Bool(window.iter().map(|&b| b != 0).collect()),
        BasicType::Char => PrimArrayElements::Char(chunks(window, u16::from_be_bytes)),
        BasicType::Float => PrimArrayElements::Float(chunks(window, f32::from_be_bytes)),
        BasicType::Double => PrimArrayElements::Double(chunks(window, f64::from_be_bytes)),
        BasicType::Byte => PrimArrayElements::Byte(window.iter().map(|&b| b as i8).collect()),
        BasicType::Short => PrimArrayElements::Short(chunks(window, i16::from_be_bytes)),
        BasicType::Int => PrimArrayElements::Int(chunks(window, i32::from_be_bytes)),
        BasicType::Long => PrimArrayElements::Long(chunks(window, i64::from_be_bytes)),
        BasicType::Object => unreachable!("rejected above"),
    })
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
        crate::heap_parser::FieldValue::Object(id) => query.resolve_value(id),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heap_parser::{FieldValue, SubRecord};
    use crate::test_util::{ClassSpec, HprofBuilder, build_in_memory, standard_heap, ty};

    fn query() -> HeapQuery {
        build_in_memory(&standard_heap())
    }

    #[test]
    fn search_strings_finds_matching_strings() {
        use crate::search::SearchQuery;
        use crate::test_util::std_ids::STRING_HI;
        let q = query();
        let m = Matcher::new(&SearchQuery::regex("^h.$")).unwrap();
        let r = q.search_strings(&m, ScanWindow::first()).unwrap();
        assert_eq!(r.total, 1);
        assert_eq!(r.scanned, 1);
        assert_eq!(r.next_cursor, None);
        assert_eq!(r.items.len(), 1);
        assert_eq!(r.items[0].object_id, STRING_HI);
        assert_eq!(r.items[0].length_chars, 2);
        assert_eq!(r.items[0].preview, "hi");

        let miss = Matcher::new(&SearchQuery::contains("bye")).unwrap();
        assert!(
            q.search_strings(&miss, ScanWindow::first())
                .unwrap()
                .items
                .is_empty()
        );

        let fuzzy = Matcher::new(&SearchQuery::fuzzy("hi")).unwrap();
        assert!(matches!(
            q.search_strings(&fuzzy, ScanWindow::first()),
            Err(HprofError::InvalidArgument(_))
        ));
    }

    #[test]
    fn search_strings_resumes_from_the_cursor_and_stops_at_max_results() {
        use crate::search::SearchQuery;
        const STRING_CLASS: u64 = 0x10;
        let mut b = HprofBuilder::new(8)
            .utf8(1, "value")
            .utf8(2, "java/lang/String")
            .load_class(1, STRING_CLASS, 2)
            .class_dump(
                ClassSpec::new(STRING_CLASS)
                    .instance_size(8)
                    .field(1, ty::OBJECT),
            );
        for i in 0..30u64 {
            let text = if i % 10 == 5 {
                format!("needle {i}")
            } else {
                format!("s-{i:02}")
            };
            b = b.char_array(0x1000 + i, &text).instance_values(
                0x2000 + i,
                STRING_CLASS,
                &[FieldValue::Object(0x1000 + i)],
            );
        }
        let q = build_in_memory(&b.build());
        let m = Matcher::new(&SearchQuery::contains("NEEDLE")).unwrap();

        // Two windows of 10: the first finds string 5, the second string 15.
        let window = ScanWindow::new(0, 10, 20);
        let first = q.search_strings(&m, window).unwrap();
        assert_eq!(first.total, 30);
        assert_eq!(first.scanned, 10);
        assert_eq!(first.next_cursor, Some(10));
        assert_eq!(first.items.len(), 1);
        assert_eq!(first.items[0].object_id, 0x2005);
        let second = q
            .search_strings(&m, first.next_window(window).unwrap())
            .unwrap();
        assert_eq!(second.items[0].object_id, 0x200f);
        assert_eq!(second.next_cursor, Some(20));

        // max_results stops the scan early, and the cursor points just past
        // the last string examined.
        let capped = q.search_strings(&m, ScanWindow::new(0, 30, 1)).unwrap();
        assert_eq!(capped.items.len(), 1);
        assert_eq!(capped.scanned, 6);
        assert_eq!(capped.next_cursor, Some(6));

        // The last window reaches the end.
        let last = q.search_strings(&m, ScanWindow::new(20, 100, 20)).unwrap();
        assert_eq!(last.next_cursor, None);
        assert_eq!(last.items[0].object_id, 0x2019);
    }

    // ── ResolvedInstance tests ────────────────────────────────────────────────

    #[test]
    fn resolved_instance_class_name() {
        let query = query();
        let inst = query.instance(0x400).unwrap().unwrap();
        let resolved = ResolvedInstance::from_dump(&query, &inst).unwrap();
        assert_eq!(resolved.object_id, 0x400);
        assert_eq!(resolved.class_name, "java.util.ArrayList");
    }

    #[test]
    fn resolved_instance_primitive_field() {
        let query = query();
        let inst = query.instance(0x100).unwrap().unwrap();
        let resolved = ResolvedInstance::from_dump(&query, &inst).unwrap();
        assert_eq!(resolved.class_name, "java.lang.Integer");
        assert_eq!(resolved.fields.len(), 1);
        assert_eq!(resolved.fields[0].name, "value");
        assert_eq!(resolved.fields[0].value, Value::Int(42));
    }

    #[test]
    fn resolved_instance_object_field_resolved_to_integer() {
        let query = query();
        let inst = query.instance(0x400).unwrap().unwrap();
        let resolved = ResolvedInstance::from_dump(&query, &inst).unwrap();
        assert_eq!(resolved.fields.len(), 1);
        assert_eq!(resolved.fields[0].name, "myField");
        assert_eq!(resolved.fields[0].value, Value::BoxedInt(0x100, 42));
    }

    #[test]
    fn resolved_instance_string_field() {
        let query = query();
        let inst = query.instance(0x300).unwrap().unwrap();
        let resolved = ResolvedInstance::from_dump(&query, &inst).unwrap();
        assert_eq!(resolved.class_name, "java.lang.String");
        // String.value is the char[] backing array (a PRIM_ARRAY_DUMP), which
        // resolve_value cannot unwrap on its own; String resolution happens at
        // the String instance level, not the char-array level.
        assert_eq!(resolved.fields.len(), 1);
        assert_eq!(resolved.fields[0].name, "value");
        assert_eq!(resolved.fields[0].value, Value::Object(0x200));
    }

    #[test]
    fn object_field_pointing_to_string_instance_is_resolved() {
        let query = query();
        let value = query.resolve_value(0x300).unwrap();
        assert_eq!(value, Value::String(0x300, "hi".to_string()));
    }

    // ── ResolvedClass tests ───────────────────────────────────────────────────

    #[test]
    fn resolved_class_names() {
        let query = query();
        let cd = query.class(0x50).unwrap().unwrap();
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
        let query = query();
        let cd = query.class(0x50).unwrap().unwrap();
        let resolved = ResolvedClass::from_dump(&query, &cd).unwrap();
        assert_eq!(resolved.static_fields.len(), 1);
        assert_eq!(resolved.static_fields[0].name, "count");
        assert_eq!(resolved.static_fields[0].value, Value::Int(7));
    }

    #[test]
    fn resolved_class_instance_field_descriptors() {
        let query = query();
        let cd = query.class(0x50).unwrap().unwrap();
        let resolved = ResolvedClass::from_dump(&query, &cd).unwrap();
        assert_eq!(resolved.instance_fields.len(), 1);
        assert_eq!(resolved.instance_fields[0].name, "myField");
        assert_eq!(resolved.instance_fields[0].field_type, FieldType::Object);
    }

    #[test]
    fn field_type_from_type_code() {
        assert_eq!(FieldType::from_code(2), Some(FieldType::Object));
        assert_eq!(FieldType::from_code(4), Some(FieldType::Boolean));
        assert_eq!(FieldType::from_code(10), Some(FieldType::Int));
        assert_eq!(FieldType::from_code(11), Some(FieldType::Long));
        assert_eq!(FieldType::from_code(99), None);
    }

    // ── ResolvedPrimArray tests ───────────────────────────────────────────────

    #[test]
    fn resolved_prim_array_metadata() {
        let query = query();
        let SubRecord::PrimArrayDump(arr) = query.object(0x200).unwrap().unwrap() else {
            panic!("expected PrimArrayDump");
        };
        let resolved = ResolvedPrimArray::from_dump(&arr).unwrap();
        assert_eq!(resolved.array_id, 0x200);
        assert_eq!(resolved.stack_trace_serial, 0);
        assert_eq!(resolved.num_elements, 2);
    }

    #[test]
    fn resolved_prim_array_char_elements() {
        let query = query();
        let SubRecord::PrimArrayDump(arr) = query.object(0x200).unwrap().unwrap() else {
            panic!("expected PrimArrayDump");
        };
        let resolved = ResolvedPrimArray::from_dump(&arr).unwrap();
        assert_eq!(
            resolved.elements,
            PrimArrayElements::Char(vec![u16::from(b'h'), u16::from(b'i')])
        );
    }

    #[test]
    fn resolved_prim_array_int_roundtrip() {
        let hprof = HprofBuilder::new(8).int_array(0x500, &[1, 2, 3]).build();
        let query = build_in_memory(&hprof);
        let SubRecord::PrimArrayDump(arr) = query.object(0x500).unwrap().unwrap() else {
            panic!("expected PrimArrayDump");
        };
        let resolved = ResolvedPrimArray::from_dump(&arr).unwrap();
        assert_eq!(resolved.num_elements, 3);
        assert_eq!(resolved.elements, PrimArrayElements::Int(vec![1, 2, 3]));
    }

    #[test]
    fn resolved_prim_array_unknown_type_error() {
        let data = &[0u8; 4];
        let result = super::parse_prim_window(data, 99, 0, 1);
        assert!(matches!(result, Err(HprofError::UnknownPrimitiveType(99))));
    }

    // ── ResolvedObjArray tests ────────────────────────────────────────────────

    /// Integer(0x10 ⊂ Object 0x20); INSTANCE 0x100 = Integer(99);
    /// OBJ_ARRAY 0x200 of Integer = [0x100, null].
    fn obj_array_heap() -> Vec<u8> {
        HprofBuilder::new(8)
            .utf8(1, "java/lang/Integer")
            .utf8(2, "value")
            .utf8(3, "java/lang/Object")
            .utf8(4, "[Ljava/lang/Integer;")
            .load_class(1, 0x10, 1)
            .load_class(2, 0x20, 3)
            .load_class(3, 0x30, 4)
            .class_dump(
                ClassSpec::new(0x10)
                    .super_class(0x20)
                    .instance_size(4)
                    .field(2, ty::INT),
            )
            .class_dump(ClassSpec::new(0x20))
            .class_dump(ClassSpec::new(0x30))
            .instance_values(0x100, 0x10, &[FieldValue::Int(99)])
            .obj_array(0x200, 0x30, &[0x100, 0])
            .build()
    }

    #[test]
    fn resolved_obj_array_metadata() {
        let query = build_in_memory(&obj_array_heap());
        let SubRecord::ObjArrayDump(arr) = query.object(0x200).unwrap().unwrap() else {
            panic!("expected ObjArrayDump");
        };
        let resolved = ResolvedObjArray::from_dump(&query, &arr).unwrap();
        assert_eq!(resolved.array_id, 0x200);
        assert_eq!(resolved.num_elements, 2);
        assert_eq!(resolved.array_class_id, 0x30);
        assert_eq!(resolved.array_class_name, "[Ljava.lang.Integer;");
        assert_eq!(
            query.key_name(crate::class_key::ClassKey::ObjArray(0x30)),
            "java.lang.Integer[]"
        );
    }

    #[test]
    fn resolved_obj_array_elements_resolved() {
        let query = build_in_memory(&obj_array_heap());
        let SubRecord::ObjArrayDump(arr) = query.object(0x200).unwrap().unwrap() else {
            panic!("expected ObjArrayDump");
        };
        let resolved = ResolvedObjArray::from_dump(&query, &arr).unwrap();
        assert_eq!(resolved.elements[0], Value::BoxedInt(0x100, 99));
        assert_eq!(resolved.elements[1], Value::Null);
    }

    // ── ResolvedRoot tests ────────────────────────────────────────────────────
    //
    // The raw root structs are constructed directly; HeapQuery is used only for
    // object_type_name / class_name resolution against the fixture objects.

    fn resolve_root(query: &HeapQuery, record: SubRecord<'_>) -> ResolvedRoot {
        ResolvedRoot::from_sub_record(query, &record)
            .unwrap()
            .expect("a root record")
    }

    #[test]
    fn resolved_root_unknown() {
        let query = query();
        let root = crate::heap_parser::RootUnknown { object_id: 0x400 };
        let resolved = resolve_root(&query, SubRecord::RootUnknown(root));
        let ResolvedRoot::Unknown {
            object_id,
            object_type_name,
        } = &resolved
        else {
            panic!("expected Unknown");
        };
        assert_eq!(*object_id, 0x400);
        assert_eq!(object_type_name, "java.util.ArrayList");
    }

    #[test]
    fn resolved_root_jni_global() {
        let query = query();
        let root = crate::heap_parser::RootJniGlobal {
            object_id: 0x100,
            jni_global_ref_id: 0xABCD,
        };
        let resolved = resolve_root(&query, SubRecord::RootJniGlobal(root));
        let ResolvedRoot::JniGlobal {
            object_id,
            jni_global_ref_id,
            object_type_name,
        } = &resolved
        else {
            panic!("expected JniGlobal");
        };
        assert_eq!(*object_id, 0x100);
        assert_eq!(*jni_global_ref_id, 0xABCD);
        assert_eq!(object_type_name, "java.lang.Integer");
    }

    #[test]
    fn resolved_root_jni_local() {
        let query = query();
        let root = crate::heap_parser::RootJniLocal {
            object_id: 0x100,
            thread_serial: 1,
            frame_number: 2,
        };
        let resolved = resolve_root(&query, SubRecord::RootJniLocal(root));
        let ResolvedRoot::JniLocal {
            thread_serial,
            frame_number,
            ..
        } = &resolved
        else {
            panic!("expected JniLocal");
        };
        assert_eq!(*thread_serial, 1);
        assert_eq!(*frame_number, 2);
    }

    #[test]
    fn resolved_root_java_frame() {
        let query = query();
        let root = crate::heap_parser::RootJavaFrame {
            object_id: 0x100,
            thread_serial: 3,
            frame_number: 7,
        };
        let resolved = resolve_root(&query, SubRecord::RootJavaFrame(root));
        let ResolvedRoot::JavaFrame {
            thread_serial,
            frame_number,
            ..
        } = &resolved
        else {
            panic!("expected JavaFrame");
        };
        assert_eq!(*thread_serial, 3);
        assert_eq!(*frame_number, 7);
    }

    #[test]
    fn resolved_root_native_stack() {
        let query = query();
        let root = crate::heap_parser::RootNativeStack {
            object_id: 0x400,
            thread_serial: 5,
        };
        let resolved = resolve_root(&query, SubRecord::RootNativeStack(root));
        let ResolvedRoot::NativeStack {
            thread_serial,
            object_type_name,
            ..
        } = &resolved
        else {
            panic!("expected NativeStack");
        };
        assert_eq!(*thread_serial, 5);
        assert_eq!(object_type_name, "java.util.ArrayList");
    }

    #[test]
    fn resolved_root_sticky_class() {
        let query = query();
        let root = crate::heap_parser::RootStickyClass { class_id: 0x10 };
        let resolved = resolve_root(&query, SubRecord::RootStickyClass(root));
        let ResolvedRoot::StickyClass {
            class_id,
            class_name,
        } = &resolved
        else {
            panic!("expected StickyClass");
        };
        assert_eq!(*class_id, 0x10);
        assert_eq!(class_name, "java.lang.Integer");
    }

    #[test]
    fn resolved_root_thread_block() {
        let query = query();
        let root = crate::heap_parser::RootThreadBlock {
            object_id: 0x400,
            thread_serial: 9,
        };
        let resolved = resolve_root(&query, SubRecord::RootThreadBlock(root));
        let ResolvedRoot::ThreadBlock { thread_serial, .. } = &resolved else {
            panic!("expected ThreadBlock");
        };
        assert_eq!(*thread_serial, 9);
    }

    #[test]
    fn resolved_root_monitor_used() {
        let query = query();
        let root = crate::heap_parser::RootMonitorUsed { object_id: 0x300 };
        let resolved = resolve_root(&query, SubRecord::RootMonitorUsed(root));
        let ResolvedRoot::MonitorUsed {
            object_type_name, ..
        } = &resolved
        else {
            panic!("expected MonitorUsed");
        };
        assert_eq!(object_type_name, "java.lang.String");
    }

    #[test]
    fn resolved_root_thread_obj() {
        let query = query();
        let root = crate::heap_parser::RootThreadObj {
            thread_object_id: 0x400,
            thread_serial: 2,
            stack_trace_serial: 42,
        };
        let resolved = resolve_root(&query, SubRecord::RootThreadObj(root));
        let ResolvedRoot::ThreadObj {
            thread_serial,
            stack_trace_serial,
            ..
        } = &resolved
        else {
            panic!("expected ThreadObj");
        };
        assert_eq!(*thread_serial, 2);
        assert_eq!(*stack_trace_serial, 42);
    }

    #[test]
    fn resolved_root_unknown_object_id_not_found_falls_back_to_object() {
        let query = query();
        let root = crate::heap_parser::RootUnknown { object_id: 0xDEAD };
        let resolved = resolve_root(&query, SubRecord::RootUnknown(root));
        let ResolvedRoot::Unknown {
            object_type_name, ..
        } = &resolved
        else {
            panic!("expected Unknown");
        };
        assert_eq!(object_type_name, "Object");
    }

    // ── SubRecord::resolve ────────────────────────────────────────────────────

    #[test]
    fn sub_record_resolve_covers_every_object_kind() {
        let query = query();
        let inst = query
            .object(0x400)
            .unwrap()
            .unwrap()
            .resolve(&query)
            .unwrap();
        assert!(matches!(inst, Resolved::Instance(ref i) if i.class_name == "java.util.ArrayList"));
        let class = query
            .object(0x50)
            .unwrap()
            .unwrap()
            .resolve(&query)
            .unwrap();
        assert!(matches!(class, Resolved::Class(_)));
        let prim = query
            .object(0x200)
            .unwrap()
            .unwrap()
            .resolve(&query)
            .unwrap();
        assert!(matches!(prim, Resolved::PrimArray(_)));
    }

    #[test]
    fn from_sub_record_ignores_non_roots_and_resolves_roots() {
        let query = query();
        let inst = query.object(0x400).unwrap().unwrap();
        assert!(
            ResolvedRoot::from_sub_record(&query, &inst)
                .unwrap()
                .is_none()
        );
        let root = SubRecord::RootUnknown(crate::heap_parser::RootUnknown { object_id: 0x400 });
        assert!(matches!(
            root.resolve(&query).unwrap(),
            Resolved::Root(ResolvedRoot::Unknown {
                object_id: 0x400,
                ..
            })
        ));
    }

    // ── Windowed primitive arrays and strings ─────────────────────────────────

    #[test]
    fn prim_array_windows_are_clamped_and_report_more() {
        use crate::query::Page;
        let bytes = HprofBuilder::new(8)
            .int_array(0x10, &[10, 20, 30, 40, 50])
            .byte_array(0x20, &[1, 0xFF])
            .build();
        let heap = crate::test_util::build_in_memory(&bytes);

        let w = heap.prim_array(0x10, Page::new(1, 2)).unwrap().unwrap();
        assert_eq!(w.elements, PrimArrayElements::Int(vec![20, 30]));
        assert_eq!((w.total, w.offset), (5, 1));
        assert!(w.has_more());

        let tail = heap.prim_array(0x10, Page::new(3, 100)).unwrap().unwrap();
        assert_eq!(tail.elements, PrimArrayElements::Int(vec![40, 50]));
        assert!(!tail.has_more());

        let past = heap.prim_array(0x10, Page::new(99, 5)).unwrap().unwrap();
        assert!(past.elements.is_empty());
        assert_eq!(past.offset, 5);

        let bytes = heap.prim_array(0x20, Page::first(10)).unwrap().unwrap();
        assert_eq!(bytes.elements, PrimArrayElements::Byte(vec![1, -1]));
        assert_eq!(bytes.element_type, BasicType::Byte);

        // Not a primitive array / not present.
        assert!(heap.prim_array(0x99, Page::first(1)).unwrap().is_none());
    }

    #[test]
    fn prim_array_elements_render_as_text() {
        let chars = PrimArrayElements::Char(vec![u16::from(b'h'), u16::from(b'i')]);
        assert_eq!(chars.to_strings(true), ["'h'", "'i'"]);
        assert_eq!(chars.to_strings(false), ["h", "i"]);
        assert_eq!(
            PrimArrayElements::Bool(vec![true, false]).to_strings(true),
            ["true", "false"]
        );
        assert_eq!(PrimArrayElements::Long(vec![-5]).to_strings(true), ["-5"]);
    }

    #[test]
    fn string_returns_text_only_for_strings() {
        let heap = query();
        assert_eq!(heap.string(0x300).unwrap().as_deref(), Some("hi"));
        assert_eq!(heap.string(0x100).unwrap(), None);
        assert_eq!(heap.string(0).unwrap(), None);
    }
}
