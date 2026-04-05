use crate::heap_index::sub_record::{
    SubIndexEntry, TAG_CLASS_DUMP, TAG_INSTANCE_DUMP, TAG_OBJ_ARRAY_DUMP, TAG_PRIM_ARRAY_DUMP,
    TAG_ROOT_JAVA_FRAME, TAG_ROOT_JNI_GLOBAL, TAG_ROOT_JNI_LOCAL, TAG_ROOT_MONITOR_USED,
    TAG_ROOT_NATIVE_STACK, TAG_ROOT_STICKY_CLASS, TAG_ROOT_THREAD_BLOCK, TAG_ROOT_THREAD_OBJ,
    TAG_ROOT_UNKNOWN,
};
use crate::hprof::HprofFile;
use crate::hprof::error::HprofError;
use crate::hprof::record::{read_id, read_u16_be, read_u32_be, read_u64_be};

// ── Value type ────────────────────────────────────────────────────────────────

/// A typed hprof field value.
///
/// Type codes are the same used in CLASS_DUMP constant-pool and static fields:
/// 2=object, 4=bool, 5=char, 6=float, 7=double, 8=byte, 9=short, 10=int, 11=long.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FieldValue {
    Object(u64),
    Bool(bool),
    Char(u16),
    Float(f32),
    Double(f64),
    Byte(i8),
    Short(i16),
    Int(i32),
    Long(i64),
}

// ── Root GC record structs (all fields fully parsed) ─────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootUnknown {
    pub object_id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootJniGlobal {
    pub object_id: u64,
    pub jni_global_ref_id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootJniLocal {
    pub object_id: u64,
    pub thread_serial: u32,
    pub frame_number: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootJavaFrame {
    pub object_id: u64,
    pub thread_serial: u32,
    pub frame_number: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootNativeStack {
    pub object_id: u64,
    pub thread_serial: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootStickyClass {
    pub class_id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootThreadBlock {
    pub object_id: u64,
    pub thread_serial: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootMonitorUsed {
    pub object_id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootThreadObj {
    pub thread_object_id: u64,
    pub thread_serial: u32,
    pub stack_trace_serial: u32,
}

// ── ClassDump ─────────────────────────────────────────────────────────────────

/// Lazily-iterable constant-pool entry.
#[derive(Debug, Clone, PartialEq)]
pub struct CpEntry {
    /// Constant pool index.
    pub index: u16,
    pub value: FieldValue,
}

/// Iterator over the constant pool entries of a [`ClassDump`].
pub struct CpEntryIter<'a> {
    raw: &'a [u8],
    pos: usize,
    remaining: usize,
    id_size: usize,
}

impl Iterator for CpEntryIter<'_> {
    type Item = Result<CpEntry, HprofError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let result = (|| -> Result<CpEntry, HprofError> {
            require(self.raw, self.pos, 3)?; // cp_index(2) + type(1)
            let index = read_u16_be(self.raw, self.pos);
            let type_id = self.raw[self.pos + 2];
            let (value, vsz) = parse_field_value(self.raw, self.pos + 3, type_id, self.id_size)?;
            self.pos += 3 + vsz;
            self.remaining -= 1;
            Ok(CpEntry { index, value })
        })();
        Some(result)
    }
}

/// Lazily-iterable static field.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticField {
    /// Name string ID.
    pub name_id: u64,
    pub value: FieldValue,
}

/// Iterator over the static fields of a [`ClassDump`].
pub struct StaticFieldIter<'a> {
    raw: &'a [u8],
    pos: usize,
    remaining: usize,
    id_size: usize,
}

impl Iterator for StaticFieldIter<'_> {
    type Item = Result<StaticField, HprofError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let result = (|| -> Result<StaticField, HprofError> {
            require(self.raw, self.pos, self.id_size + 1)?; // name_id + type
            let name_id = read_id(self.raw, self.pos, self.id_size)?;
            let type_id = self.raw[self.pos + self.id_size];
            let (value, vsz) =
                parse_field_value(self.raw, self.pos + self.id_size + 1, type_id, self.id_size)?;
            self.pos += self.id_size + 1 + vsz;
            self.remaining -= 1;
            Ok(StaticField { name_id, value })
        })();
        Some(result)
    }
}

/// Lazily-iterable instance field descriptor (name + type; no value stored in CLASS_DUMP).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InstanceFieldDescriptor {
    /// Name string ID.
    pub name_id: u64,
    /// hprof type code (2=object, 4=bool, 5=char, 6=float, 7=double, 8=byte, 9=short, 10=int, 11=long).
    pub field_type: u8,
}

/// Iterator over the instance field descriptors of a [`ClassDump`].
pub struct InstanceFieldIter<'a> {
    raw: &'a [u8],
    pos: usize,
    remaining: usize,
    id_size: usize,
}

impl Iterator for InstanceFieldIter<'_> {
    type Item = Result<InstanceFieldDescriptor, HprofError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let result = (|| -> Result<InstanceFieldDescriptor, HprofError> {
            require(self.raw, self.pos, self.id_size + 1)?;
            let name_id = read_id(self.raw, self.pos, self.id_size)?;
            let field_type = self.raw[self.pos + self.id_size];
            self.pos += self.id_size + 1;
            self.remaining -= 1;
            Ok(InstanceFieldDescriptor {
                name_id,
                field_type,
            })
        })();
        Some(result)
    }
}

/// A fully-parsed CLASS_DUMP sub-record.
///
/// Scalar fields are eagerly parsed. Constant pool, static fields, and instance
/// field descriptors are provided as lazy iterators — no allocations are made.
#[derive(Debug)]
pub struct ClassDump<'a> {
    pub class_id: u64,
    pub stack_trace_serial: u32,
    pub super_class_id: u64,
    pub class_loader_id: u64,
    pub signers_id: u64,
    pub domain_id: u64,
    pub reserved1_id: u64,
    pub reserved2_id: u64,
    pub instance_size: u32,

    // Section metadata: counts + offsets within `raw`
    cp_count: u16,
    /// Offset within `raw` where the cp entries begin (after the cp_count u16).
    cp_data_offset: usize,
    statics_count: u16,
    /// Offset within `raw` where the static field entries begin.
    statics_data_offset: usize,
    instance_fields_count: u16,
    /// Offset within `raw` where the instance field entries begin.
    fields_data_offset: usize,

    /// Slice of the hprof mmap from the subtag byte through the end of this sub-record.
    raw: &'a [u8],
    id_size: usize,
}

impl<'a> ClassDump<'a> {
    pub fn constant_pool(&self) -> CpEntryIter<'a> {
        CpEntryIter {
            raw: self.raw,
            pos: self.cp_data_offset,
            remaining: self.cp_count as usize,
            id_size: self.id_size,
        }
    }

    pub fn static_fields(&self) -> StaticFieldIter<'a> {
        StaticFieldIter {
            raw: self.raw,
            pos: self.statics_data_offset,
            remaining: self.statics_count as usize,
            id_size: self.id_size,
        }
    }

    pub fn instance_fields(&self) -> InstanceFieldIter<'a> {
        InstanceFieldIter {
            raw: self.raw,
            pos: self.fields_data_offset,
            remaining: self.instance_fields_count as usize,
            id_size: self.id_size,
        }
    }

    /// Raw bytes of this sub-record starting from the subtag byte.
    ///
    /// Use this for byte-level equality comparison between two CLASS_DUMP
    /// records for the same class across two heap snapshots.
    pub fn raw_bytes(&self) -> &[u8] {
        self.raw
    }
}

// ── InstanceDump ──────────────────────────────────────────────────────────────

/// A fully-parsed INSTANCE_DUMP sub-record.
///
/// `data` is a borrowed slice of the raw instance field bytes (big-endian).
/// Interpret the bytes using the field layout from the associated CLASS_DUMP.
#[derive(Debug)]
pub struct InstanceDump<'a> {
    pub object_id: u64,
    pub stack_trace_serial: u32,
    pub class_id: u64,
    /// Raw instance field data (big-endian). Interpret using CLASS_DUMP field descriptors.
    pub data: &'a [u8],
}

// ── ObjArrayDump ──────────────────────────────────────────────────────────────

/// Iterator over object-reference elements of an [`ObjArrayDump`].
pub struct ObjArrayElemIter<'a> {
    raw: &'a [u8],
    pos: usize,
    remaining: usize,
    id_size: usize,
}

impl Iterator for ObjArrayElemIter<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        // Bounds are guaranteed by ObjArrayDump construction.
        let id = match self.id_size {
            4 => read_u32_be(self.raw, self.pos) as u64,
            8 => read_u64_be(self.raw, self.pos),
            _ => return None,
        };
        self.pos += self.id_size;
        self.remaining -= 1;
        Some(id)
    }
}

/// A fully-parsed OBJ_ARRAY_DUMP sub-record.
#[derive(Debug)]
pub struct ObjArrayDump<'a> {
    pub array_id: u64,
    pub stack_trace_serial: u32,
    pub num_elements: u32,
    pub element_class_id: u64,
    elements_raw: &'a [u8],
    id_size: usize,
}

impl<'a> ObjArrayDump<'a> {
    /// Iterate over element object IDs.
    pub fn elements(&self) -> ObjArrayElemIter<'a> {
        ObjArrayElemIter {
            raw: self.elements_raw,
            pos: 0,
            remaining: self.num_elements as usize,
            id_size: self.id_size,
        }
    }

    /// Raw element bytes (concatenated big-endian object IDs).
    pub fn elements_raw(&self) -> &[u8] {
        self.elements_raw
    }
}

// ── PrimArrayDump ─────────────────────────────────────────────────────────────

/// A fully-parsed PRIM_ARRAY_DUMP sub-record.
///
/// `data` is the raw big-endian element bytes. Use `element_type` to interpret:
/// 4=bool(1B), 5=char(2B), 6=float(4B), 7=double(8B),
/// 8=byte(1B), 9=short(2B), 10=int(4B), 11=long(8B).
#[derive(Debug)]
pub struct PrimArrayDump<'a> {
    pub array_id: u64,
    pub stack_trace_serial: u32,
    pub num_elements: u32,
    pub element_type: u8,
    /// Raw element bytes in big-endian format.
    pub data: &'a [u8],
}

// ── SubRecord ─────────────────────────────────────────────────────────────────

/// A fully-parsed heap dump sub-record.
///
/// Variants that contain borrowed data (`'a`) hold references into the hprof
/// memory map — no heap allocations are made.
#[derive(Debug)]
pub enum SubRecord<'a> {
    RootUnknown(RootUnknown),
    RootJniGlobal(RootJniGlobal),
    RootJniLocal(RootJniLocal),
    RootJavaFrame(RootJavaFrame),
    RootNativeStack(RootNativeStack),
    RootStickyClass(RootStickyClass),
    RootThreadBlock(RootThreadBlock),
    RootMonitorUsed(RootMonitorUsed),
    RootThreadObj(RootThreadObj),
    ClassDump(ClassDump<'a>),
    InstanceDump(InstanceDump<'a>),
    ObjArrayDump(ObjArrayDump<'a>),
    PrimArrayDump(PrimArrayDump<'a>),
}

// ── Public parse entry point ──────────────────────────────────────────────────

/// Parse the full sub-record identified by `entry` from the hprof mmap.
///
/// The returned `SubRecord<'a>` borrows from `hprof` — no data is copied or
/// stored on the heap.
pub fn parse_sub_record<'a>(
    hprof: &HprofFile<'a>,
    entry: &SubIndexEntry,
) -> Result<SubRecord<'a>, HprofError> {
    let data = hprof.data();
    let id = hprof.header.id_size as usize;
    let pos = entry.position as usize;

    if pos >= data.len() {
        return Err(HprofError::UnexpectedEof(pos));
    }

    match data[pos] {
        TAG_ROOT_UNKNOWN => Ok(SubRecord::RootUnknown(RootUnknown {
            object_id: read_id(data, pos + 1, id)?,
        })),
        TAG_ROOT_JNI_GLOBAL => Ok(SubRecord::RootJniGlobal(RootJniGlobal {
            object_id: read_id(data, pos + 1, id)?,
            jni_global_ref_id: read_id(data, pos + 1 + id, id)?,
        })),
        TAG_ROOT_JNI_LOCAL => {
            require(data, pos + 1 + id, 8)?;
            Ok(SubRecord::RootJniLocal(RootJniLocal {
                object_id: read_id(data, pos + 1, id)?,
                thread_serial: read_u32_be(data, pos + 1 + id),
                frame_number: read_u32_be(data, pos + 1 + id + 4),
            }))
        }
        TAG_ROOT_JAVA_FRAME => {
            require(data, pos + 1 + id, 8)?;
            Ok(SubRecord::RootJavaFrame(RootJavaFrame {
                object_id: read_id(data, pos + 1, id)?,
                thread_serial: read_u32_be(data, pos + 1 + id),
                frame_number: read_u32_be(data, pos + 1 + id + 4),
            }))
        }
        TAG_ROOT_NATIVE_STACK => {
            require(data, pos + 1 + id, 4)?;
            Ok(SubRecord::RootNativeStack(RootNativeStack {
                object_id: read_id(data, pos + 1, id)?,
                thread_serial: read_u32_be(data, pos + 1 + id),
            }))
        }
        TAG_ROOT_STICKY_CLASS => Ok(SubRecord::RootStickyClass(RootStickyClass {
            class_id: read_id(data, pos + 1, id)?,
        })),
        TAG_ROOT_THREAD_BLOCK => {
            require(data, pos + 1 + id, 4)?;
            Ok(SubRecord::RootThreadBlock(RootThreadBlock {
                object_id: read_id(data, pos + 1, id)?,
                thread_serial: read_u32_be(data, pos + 1 + id),
            }))
        }
        TAG_ROOT_MONITOR_USED => Ok(SubRecord::RootMonitorUsed(RootMonitorUsed {
            object_id: read_id(data, pos + 1, id)?,
        })),
        TAG_ROOT_THREAD_OBJ => {
            require(data, pos + 1 + id, 8)?;
            Ok(SubRecord::RootThreadObj(RootThreadObj {
                thread_object_id: read_id(data, pos + 1, id)?,
                thread_serial: read_u32_be(data, pos + 1 + id),
                stack_trace_serial: read_u32_be(data, pos + 1 + id + 4),
            }))
        }
        TAG_CLASS_DUMP => parse_class_dump(data, pos, id),
        TAG_INSTANCE_DUMP => parse_instance_dump(data, pos, id),
        TAG_OBJ_ARRAY_DUMP => parse_obj_array_dump(data, pos, id),
        TAG_PRIM_ARRAY_DUMP => parse_prim_array_dump(data, pos, id),
        other => Err(HprofError::UnknownSubRecordTag(other, pos)),
    }
}

// ── Private parsing helpers ───────────────────────────────────────────────────

fn parse_class_dump<'a>(
    data: &'a [u8],
    pos: usize,
    id: usize,
) -> Result<SubRecord<'a>, HprofError> {
    // Fixed prefix layout (relative to pos):
    //   0: subtag(1) + class_id(id) + stack_serial(4) + super_id(id) + loader_id(id)
    //   + signers_id(id) + domain_id(id) + reserved1(id) + reserved2(id) + instance_size(4)
    //   = 9 + 7*id bytes, then cp_count(2).
    let fixed_prefix_end = pos + 9 + 7 * id;
    require(data, fixed_prefix_end, 2)?; // 2 for cp_count

    let class_id = read_id(data, pos + 1, id)?;
    let stack_trace_serial = read_u32_be(data, pos + 1 + id);
    let super_class_id = read_id(data, pos + 1 + id + 4, id)?;
    let class_loader_id = read_id(data, pos + 1 + 2 * id + 4, id)?;
    let signers_id = read_id(data, pos + 1 + 3 * id + 4, id)?;
    let domain_id = read_id(data, pos + 1 + 4 * id + 4, id)?;
    let reserved1_id = read_id(data, pos + 1 + 5 * id + 4, id)?;
    let reserved2_id = read_id(data, pos + 1 + 6 * id + 4, id)?;
    let instance_size = read_u32_be(data, pos + 1 + 7 * id + 4);

    // Scan constant pool to find where statics begin.
    let cp_count_abs = fixed_prefix_end;
    let cp_count = read_u16_be(data, cp_count_abs);
    let mut abs_off = cp_count_abs + 2; // first cp entry
    for _ in 0..cp_count {
        require(data, abs_off, 3)?; // cp_index(2) + type(1)
        abs_off += 2;
        let type_id = data[abs_off];
        abs_off += 1;
        abs_off += type_byte_size(type_id, id)?;
        require(data, abs_off, 0)?;
    }

    // Scan statics to find where instance fields begin.
    require(data, abs_off, 2)?;
    let statics_count = read_u16_be(data, abs_off);
    let statics_data_abs = abs_off + 2;
    abs_off = statics_data_abs;
    for _ in 0..statics_count {
        require(data, abs_off, id + 1)?; // name_id + type
        abs_off += id;
        let type_id = data[abs_off];
        abs_off += 1;
        abs_off += type_byte_size(type_id, id)?;
        require(data, abs_off, 0)?;
    }

    require(data, abs_off, 2)?;
    let fields_count = read_u16_be(data, abs_off);
    let fields_data_abs = abs_off + 2;
    let rec_end = fields_data_abs + fields_count as usize * (id + 1);
    require(data, rec_end, 0)?;

    // All offsets below are relative to raw[0] = data[pos].
    let raw = &data[pos..rec_end];
    Ok(SubRecord::ClassDump(ClassDump {
        class_id,
        stack_trace_serial,
        super_class_id,
        class_loader_id,
        signers_id,
        domain_id,
        reserved1_id,
        reserved2_id,
        instance_size,
        cp_count,
        cp_data_offset: cp_count_abs + 2 - pos,
        statics_count,
        statics_data_offset: statics_data_abs - pos,
        instance_fields_count: fields_count,
        fields_data_offset: fields_data_abs - pos,
        raw,
        id_size: id,
    }))
}

fn parse_instance_dump<'a>(
    data: &'a [u8],
    pos: usize,
    id: usize,
) -> Result<SubRecord<'a>, HprofError> {
    // subtag(1) + object_id(id) + stack_serial(4) + class_id(id) + data_len(4) + data(data_len)
    let data_len_off = pos + 1 + id + 4 + id;
    require(data, data_len_off, 4)?;
    let object_id = read_id(data, pos + 1, id)?;
    let stack_trace_serial = read_u32_be(data, pos + 1 + id);
    let class_id = read_id(data, pos + 1 + id + 4, id)?;
    let data_len = read_u32_be(data, data_len_off) as usize;
    let data_start = data_len_off + 4;
    require(data, data_start, data_len)?;
    Ok(SubRecord::InstanceDump(InstanceDump {
        object_id,
        stack_trace_serial,
        class_id,
        data: &data[data_start..data_start + data_len],
    }))
}

fn parse_obj_array_dump<'a>(
    data: &'a [u8],
    pos: usize,
    id: usize,
) -> Result<SubRecord<'a>, HprofError> {
    // subtag(1) + array_id(id) + stack_serial(4) + num_elements(4) + elem_class_id(id) + elements(num*id)
    let num_off = pos + 1 + id + 4;
    let elem_class_off = num_off + 4;
    let elems_off = elem_class_off + id;
    require(data, elem_class_off, id)?;
    let array_id = read_id(data, pos + 1, id)?;
    let stack_trace_serial = read_u32_be(data, pos + 1 + id);
    let num_elements = read_u32_be(data, num_off);
    let element_class_id = read_id(data, elem_class_off, id)?;
    let elems_len = num_elements as usize * id;
    require(data, elems_off, elems_len)?;
    Ok(SubRecord::ObjArrayDump(ObjArrayDump {
        array_id,
        stack_trace_serial,
        num_elements,
        element_class_id,
        elements_raw: &data[elems_off..elems_off + elems_len],
        id_size: id,
    }))
}

fn parse_prim_array_dump<'a>(
    data: &'a [u8],
    pos: usize,
    id: usize,
) -> Result<SubRecord<'a>, HprofError> {
    // subtag(1) + array_id(id) + stack_serial(4) + num_elements(4) + elem_type(1) + data(num*elem_sz)
    let type_off = pos + 1 + id + 4 + 4;
    require(data, type_off, 1)?;
    let array_id = read_id(data, pos + 1, id)?;
    let stack_trace_serial = read_u32_be(data, pos + 1 + id);
    let num_elements = read_u32_be(data, pos + 1 + id + 4);
    let element_type = data[type_off];
    let elem_sz = type_byte_size(element_type, id)?;
    let data_off = type_off + 1;
    let data_len = num_elements as usize * elem_sz;
    require(data, data_off, data_len)?;
    Ok(SubRecord::PrimArrayDump(PrimArrayDump {
        array_id,
        stack_trace_serial,
        num_elements,
        element_type,
        data: &data[data_off..data_off + data_len],
    }))
}

/// Parse a typed field value at `offset` within `data`.
///
/// Returns the parsed `FieldValue` and the number of bytes consumed.
fn parse_field_value(
    data: &[u8],
    offset: usize,
    type_id: u8,
    id_size: usize,
) -> Result<(FieldValue, usize), HprofError> {
    match type_id {
        2 => {
            let id = read_id(data, offset, id_size)?;
            Ok((FieldValue::Object(id), id_size))
        }
        4 => {
            require(data, offset, 1)?;
            Ok((FieldValue::Bool(data[offset] != 0), 1))
        }
        5 => {
            require(data, offset, 2)?;
            Ok((FieldValue::Char(read_u16_be(data, offset)), 2))
        }
        6 => {
            require(data, offset, 4)?;
            Ok((
                FieldValue::Float(f32::from_bits(read_u32_be(data, offset))),
                4,
            ))
        }
        7 => {
            require(data, offset, 8)?;
            Ok((
                FieldValue::Double(f64::from_bits(read_u64_be(data, offset))),
                8,
            ))
        }
        8 => {
            require(data, offset, 1)?;
            Ok((FieldValue::Byte(data[offset] as i8), 1))
        }
        9 => {
            require(data, offset, 2)?;
            Ok((FieldValue::Short(read_u16_be(data, offset) as i16), 2))
        }
        10 => {
            require(data, offset, 4)?;
            Ok((FieldValue::Int(read_u32_be(data, offset) as i32), 4))
        }
        11 => {
            require(data, offset, 8)?;
            Ok((FieldValue::Long(read_u64_be(data, offset) as i64), 8))
        }
        other => Err(HprofError::UnknownPrimitiveType(other)),
    }
}

/// Return the byte size of a value with the given hprof type code.
fn type_byte_size(type_id: u8, id_size: usize) -> Result<usize, HprofError> {
    match type_id {
        2 => Ok(id_size),
        4 | 8 => Ok(1),
        5 | 9 => Ok(2),
        6 | 10 => Ok(4),
        7 | 11 => Ok(8),
        other => Err(HprofError::UnknownPrimitiveType(other)),
    }
}

/// Assert that `data[offset..offset+len]` is in-bounds.
fn require(data: &[u8], offset: usize, len: usize) -> Result<(), HprofError> {
    if offset + len > data.len() {
        Err(HprofError::UnexpectedEof(offset))
    } else {
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Minimal hprof helpers ────────────────────────────────────────────────

    /// Build a minimal hprof with one HEAP_DUMP_SEGMENT containing `body`.
    fn make_hprof(id_size: u32, body: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"JAVA PROFILE 1.0.2\0"); // 19 bytes
        buf.extend_from_slice(&id_size.to_be_bytes()); // 4 bytes
        buf.extend_from_slice(&0u64.to_be_bytes()); // 8 bytes → data_offset = 31
        buf.push(0x1C); // HEAP_DUMP_SEGMENT tag
        buf.extend_from_slice(&0u32.to_be_bytes()); // time_offset
        buf.extend_from_slice(&(body.len() as u32).to_be_bytes());
        buf.extend_from_slice(body);
        buf
    }

    /// data_offset = 31; first sub-record body_start = 31 + 9 = 40.
    const BODY_START: u64 = 40;

    fn open_hprof(data: &[u8]) -> HprofFile<'_> {
        HprofFile::from_ref(data).unwrap()
    }

    // ── Root record tests ────────────────────────────────────────────────────

    #[test]
    fn parse_root_sticky_class_id8() {
        let mut body = vec![0x05u8]; // TAG_ROOT_STICKY_CLASS
        body.extend_from_slice(&42u64.to_be_bytes());

        let hprof_data = make_hprof(8, &body);
        let hprof = open_hprof(&hprof_data);

        let entry = SubIndexEntry {
            tag: 0x05,
            object_id: 42,
            position: BODY_START,
        };
        let rec = parse_sub_record(&hprof, &entry).unwrap();
        if let SubRecord::RootStickyClass(r) = rec {
            assert_eq!(r.class_id, 42);
        } else {
            panic!("expected RootStickyClass");
        }
    }

    #[test]
    fn parse_root_jni_local_id8() {
        let mut body = vec![0x02u8]; // TAG_ROOT_JNI_LOCAL
        body.extend_from_slice(&7u64.to_be_bytes()); // object_id = 7
        body.extend_from_slice(&3u32.to_be_bytes()); // thread_serial = 3
        body.extend_from_slice(&99u32.to_be_bytes()); // frame_number = 99

        let hprof_data = make_hprof(8, &body);
        let hprof = open_hprof(&hprof_data);

        let entry = SubIndexEntry {
            tag: 0x02,
            object_id: 7,
            position: BODY_START,
        };
        let rec = parse_sub_record(&hprof, &entry).unwrap();
        if let SubRecord::RootJniLocal(r) = rec {
            assert_eq!(r.object_id, 7);
            assert_eq!(r.thread_serial, 3);
            assert_eq!(r.frame_number, 99);
        } else {
            panic!("expected RootJniLocal");
        }
    }

    // ── InstanceDump ─────────────────────────────────────────────────────────

    #[test]
    fn parse_instance_dump_id8() {
        let mut body = vec![0x21u8]; // TAG_INSTANCE_DUMP
        body.extend_from_slice(&100u64.to_be_bytes()); // object_id
        body.extend_from_slice(&0u32.to_be_bytes()); // stack_serial
        body.extend_from_slice(&200u64.to_be_bytes()); // class_id
        body.extend_from_slice(&4u32.to_be_bytes()); // data_len = 4
        body.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]); // data

        let hprof_data = make_hprof(8, &body);
        let hprof = open_hprof(&hprof_data);

        let entry = SubIndexEntry {
            tag: 0x21,
            object_id: 100,
            position: BODY_START,
        };
        let rec = parse_sub_record(&hprof, &entry).unwrap();
        if let SubRecord::InstanceDump(inst) = rec {
            assert_eq!(inst.object_id, 100);
            assert_eq!(inst.class_id, 200);
            assert_eq!(inst.data, &[0xDE, 0xAD, 0xBE, 0xEF]);
        } else {
            panic!("expected InstanceDump");
        }
    }

    // ── ClassDump ────────────────────────────────────────────────────────────

    #[test]
    fn parse_class_dump_with_static_and_instance_fields() {
        // id_size = 8, minimal CLASS_DUMP:
        // Fixed prefix + 0 cp entries + 1 static (int=10) + 1 instance field (long=11)
        const ID: usize = 8;
        let mut body = vec![0x20u8]; // TAG_CLASS_DUMP
        body.extend_from_slice(&1u64.to_be_bytes()); // class_id = 1
        body.extend_from_slice(&0u32.to_be_bytes()); // stack_serial
        body.extend_from_slice(&2u64.to_be_bytes()); // super_class_id = 2
        body.extend_from_slice(&[0u8; ID]); // class_loader_id
        body.extend_from_slice(&[0u8; ID]); // signers_id
        body.extend_from_slice(&[0u8; ID]); // domain_id
        body.extend_from_slice(&[0u8; ID]); // reserved1
        body.extend_from_slice(&[0u8; ID]); // reserved2
        body.extend_from_slice(&16u32.to_be_bytes()); // instance_size = 16
        // cp_count = 0
        body.extend_from_slice(&0u16.to_be_bytes());
        // statics_count = 1
        body.extend_from_slice(&1u16.to_be_bytes());
        // static: name_id=99, type=10(int), value=42
        body.extend_from_slice(&99u64.to_be_bytes());
        body.push(10); // int
        body.extend_from_slice(&42i32.to_be_bytes());
        // instance_fields_count = 1
        body.extend_from_slice(&1u16.to_be_bytes());
        // instance field: name_id=88, type=11(long)
        body.extend_from_slice(&88u64.to_be_bytes());
        body.push(11); // long

        let hprof_data = make_hprof(8, &body);
        let hprof = open_hprof(&hprof_data);

        let entry = SubIndexEntry {
            tag: 0x20,
            object_id: 1,
            position: BODY_START,
        };
        let rec = parse_sub_record(&hprof, &entry).unwrap();
        if let SubRecord::ClassDump(cd) = rec {
            assert_eq!(cd.class_id, 1);
            assert_eq!(cd.super_class_id, 2);
            assert_eq!(cd.instance_size, 16);

            let statics: Vec<_> = cd.static_fields().map(|r| r.unwrap()).collect();
            assert_eq!(statics.len(), 1);
            assert_eq!(statics[0].name_id, 99);
            assert_eq!(statics[0].value, FieldValue::Int(42));

            let fields: Vec<_> = cd.instance_fields().map(|r| r.unwrap()).collect();
            assert_eq!(fields.len(), 1);
            assert_eq!(fields[0].name_id, 88);
            assert_eq!(fields[0].field_type, 11);
        } else {
            panic!("expected ClassDump");
        }
    }

    // ── ObjArrayDump ─────────────────────────────────────────────────────────

    #[test]
    fn parse_obj_array_dump_id8() {
        let mut body = vec![0x22u8]; // TAG_OBJ_ARRAY_DUMP
        body.extend_from_slice(&10u64.to_be_bytes()); // array_id
        body.extend_from_slice(&0u32.to_be_bytes()); // stack_serial
        body.extend_from_slice(&3u32.to_be_bytes()); // num_elements = 3
        body.extend_from_slice(&20u64.to_be_bytes()); // element_class_id
        body.extend_from_slice(&100u64.to_be_bytes()); // element[0]
        body.extend_from_slice(&200u64.to_be_bytes()); // element[1]
        body.extend_from_slice(&300u64.to_be_bytes()); // element[2]

        let hprof_data = make_hprof(8, &body);
        let hprof = open_hprof(&hprof_data);

        let entry = SubIndexEntry {
            tag: 0x22,
            object_id: 10,
            position: BODY_START,
        };
        let rec = parse_sub_record(&hprof, &entry).unwrap();
        if let SubRecord::ObjArrayDump(arr) = rec {
            assert_eq!(arr.num_elements, 3);
            assert_eq!(arr.element_class_id, 20);
            let elems: Vec<u64> = arr.elements().collect();
            assert_eq!(elems, vec![100, 200, 300]);
        } else {
            panic!("expected ObjArrayDump");
        }
    }

    // ── PrimArrayDump ─────────────────────────────────────────────────────────

    #[test]
    fn parse_prim_array_dump_ints() {
        let mut body = vec![0x23u8]; // TAG_PRIM_ARRAY_DUMP
        body.extend_from_slice(&50u64.to_be_bytes()); // array_id
        body.extend_from_slice(&0u32.to_be_bytes()); // stack_serial
        body.extend_from_slice(&2u32.to_be_bytes()); // num_elements = 2
        body.push(10); // element_type = int
        body.extend_from_slice(&1i32.to_be_bytes()); // element[0] = 1
        body.extend_from_slice(&(-1i32).to_be_bytes()); // element[1] = -1

        let hprof_data = make_hprof(8, &body);
        let hprof = open_hprof(&hprof_data);

        let entry = SubIndexEntry {
            tag: 0x23,
            object_id: 50,
            position: BODY_START,
        };
        let rec = parse_sub_record(&hprof, &entry).unwrap();
        if let SubRecord::PrimArrayDump(arr) = rec {
            assert_eq!(arr.num_elements, 2);
            assert_eq!(arr.element_type, 10); // int
            assert_eq!(arr.data.len(), 8); // 2 * 4 bytes
        } else {
            panic!("expected PrimArrayDump");
        }
    }
}
