use crate::hprof::error::HprofError;
use crate::hprof::record::{read_u16_be, read_u32_be, read_u64_be};

// ── Sub-record tag constants (HPROF_GC_* from heapDumper.cpp) ────────────────

pub const TAG_ROOT_UNKNOWN: u8 = 0xFF;
pub const TAG_ROOT_JNI_GLOBAL: u8 = 0x01;
pub const TAG_ROOT_JNI_LOCAL: u8 = 0x02;
pub const TAG_ROOT_JAVA_FRAME: u8 = 0x03;
pub const TAG_ROOT_NATIVE_STACK: u8 = 0x04;
pub const TAG_ROOT_STICKY_CLASS: u8 = 0x05;
pub const TAG_ROOT_THREAD_BLOCK: u8 = 0x06;
pub const TAG_ROOT_MONITOR_USED: u8 = 0x07;
pub const TAG_ROOT_THREAD_OBJ: u8 = 0x08;
pub const TAG_CLASS_DUMP: u8 = 0x20;
pub const TAG_INSTANCE_DUMP: u8 = 0x21;
pub const TAG_OBJ_ARRAY_DUMP: u8 = 0x22;
pub const TAG_PRIM_ARRAY_DUMP: u8 = 0x23;

// ── Heap sub-record index entry ───────────────────────────────────────────────

/// Size in bytes of a serialized [`SubIndexEntry`].
pub const SUB_INDEX_ENTRY_SIZE: usize = 24;

/// A fixed-size entry in the heap sub-record index.
///
/// Binary layout (all little-endian):
/// ```text
///  0..1    u8        subtag
///  1..8    [u8; 7]   padding (zeros)
///  8..16   u64       object_id  (first id-sized field; zero-extended to 8 bytes)
///  16..24  u64       position   (byte offset of the subtag byte in the hprof file)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubIndexEntry {
    pub tag: u8,
    pub object_id: u64,
    /// Byte offset of the subtag byte within the hprof file.
    pub position: u64,
}

impl SubIndexEntry {
    pub fn to_bytes(self) -> [u8; SUB_INDEX_ENTRY_SIZE] {
        let mut buf = [0u8; SUB_INDEX_ENTRY_SIZE];
        buf[0] = self.tag;
        // buf[1..8] = 0 (padding)
        buf[8..16].copy_from_slice(&self.object_id.to_le_bytes());
        buf[16..24].copy_from_slice(&self.position.to_le_bytes());
        buf
    }

    pub fn from_bytes(bytes: &[u8; SUB_INDEX_ENTRY_SIZE]) -> Self {
        let tag = bytes[0];
        let object_id = u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]);
        let position = u64::from_le_bytes([
            bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23],
        ]);
        Self {
            tag,
            object_id,
            position,
        }
    }
}

// ── Sub-record scanning ───────────────────────────────────────────────────────

/// Iterate over sub-records within a heap dump body slice.
///
/// `body`       — slice of `data` covering only the heap dump body (no record header).
/// `body_start` — absolute byte offset of `body[0]` within the hprof file.
/// `id_size`    — identifier size from the hprof file header (4 or 8).
pub struct SubRecordScanner<'a> {
    body: &'a [u8],
    pos: usize,
    body_start: u64,
    id_size: usize,
}

impl<'a> SubRecordScanner<'a> {
    pub fn new(body: &'a [u8], body_start: u64, id_size: u32) -> Result<Self, HprofError> {
        if id_size != 4 && id_size != 8 {
            return Err(HprofError::InvalidIdSize(id_size));
        }
        Ok(Self {
            body,
            pos: 0,
            body_start,
            id_size: id_size as usize,
        })
    }
}

impl Iterator for SubRecordScanner<'_> {
    type Item = Result<SubIndexEntry, HprofError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.body.len() {
            return None;
        }
        let result = parse_sub_record(self.body, self.pos, self.body_start, self.id_size);
        match result {
            Ok((entry, size)) => {
                self.pos += size;
                Some(Ok(entry))
            }
            Err(e) => Some(Err(e)),
        }
    }
}

/// Parse a single sub-record at `pos` within `body`.
///
/// Returns the `SubIndexEntry` and the total byte size of the sub-record.
fn parse_sub_record(
    body: &[u8],
    pos: usize,
    body_start: u64,
    id_size: usize,
) -> Result<(SubIndexEntry, usize), HprofError> {
    if pos >= body.len() {
        return Err(HprofError::UnexpectedEof(pos));
    }
    let tag = body[pos];
    let object_id = read_id(body, pos + 1, id_size)?;
    let size = sub_record_size(body, pos, id_size)?;
    let entry = SubIndexEntry {
        tag,
        object_id,
        position: body_start + pos as u64,
    };
    Ok((entry, size))
}

/// Returns the total byte size (including the subtag byte) of the sub-record at `pos`.
pub fn sub_record_size(body: &[u8], pos: usize, id_size: usize) -> Result<usize, HprofError> {
    if pos >= body.len() {
        return Err(HprofError::UnexpectedEof(pos));
    }
    let tag = body[pos];
    let id = id_size;

    let size = match tag {
        // ROOT_UNKNOWN: subtag + object_id
        TAG_ROOT_UNKNOWN => 1 + id,
        // ROOT_JNI_GLOBAL: subtag + object_id + jni_global_ref_id
        TAG_ROOT_JNI_GLOBAL => 1 + 2 * id,
        // ROOT_JNI_LOCAL: subtag + object_id + thread_serial(u32) + frame_number(u32)
        TAG_ROOT_JNI_LOCAL => 1 + id + 8,
        // ROOT_JAVA_FRAME: subtag + object_id + thread_serial(u32) + frame_number(u32)
        TAG_ROOT_JAVA_FRAME => 1 + id + 8,
        // ROOT_NATIVE_STACK: subtag + object_id + thread_serial(u32)
        TAG_ROOT_NATIVE_STACK => 1 + id + 4,
        // ROOT_STICKY_CLASS: subtag + class_id
        TAG_ROOT_STICKY_CLASS => 1 + id,
        // ROOT_THREAD_BLOCK: subtag + object_id + thread_serial(u32)
        TAG_ROOT_THREAD_BLOCK => 1 + id + 4,
        // ROOT_MONITOR_USED: subtag + object_id
        TAG_ROOT_MONITOR_USED => 1 + id,
        // ROOT_THREAD_OBJ: subtag + thread_object_id + thread_serial(u32) + stack_trace_serial(u32)
        TAG_ROOT_THREAD_OBJ => 1 + id + 8,

        // CLASS_DUMP: complex variable-length (constant pool + static + instance fields)
        TAG_CLASS_DUMP => class_dump_size(body, pos, id)?,

        // INSTANCE_DUMP: subtag + object_id + stack_serial(u32) + class_id + data_len(u32) + data
        TAG_INSTANCE_DUMP => {
            let data_len_off = pos + 1 + id + 4 + id;
            if data_len_off + 4 > body.len() {
                return Err(HprofError::UnexpectedEof(data_len_off));
            }
            1 + id + 4 + id + 4 + read_u32_be(body, data_len_off) as usize
        }

        // OBJ_ARRAY_DUMP: subtag + array_id + stack_serial(u32) + num_elements(u32) + elem_class_id + elements
        TAG_OBJ_ARRAY_DUMP => {
            let num_off = pos + 1 + id + 4;
            if num_off + 4 > body.len() {
                return Err(HprofError::UnexpectedEof(num_off));
            }
            let num = read_u32_be(body, num_off) as usize;
            1 + id + 4 + 4 + id + num * id
        }

        // PRIM_ARRAY_DUMP: subtag + array_id + stack_serial(u32) + num_elements(u32) + elem_type(u8) + data
        TAG_PRIM_ARRAY_DUMP => {
            let num_off = pos + 1 + id + 4;
            if num_off + 5 > body.len() {
                return Err(HprofError::UnexpectedEof(num_off));
            }
            let num = read_u32_be(body, num_off) as usize;
            let elem_type = body[num_off + 4];
            let elem_size = value_size(elem_type, id)?;
            1 + id + 4 + 4 + 1 + num * elem_size
        }

        other => return Err(HprofError::UnknownSubRecordTag(other, pos)),
    };

    if pos + size > body.len() {
        return Err(HprofError::UnexpectedEof(pos));
    }
    Ok(size)
}

/// Compute the total byte size of a CLASS_DUMP sub-record.
///
/// Fixed prefix:
///   subtag(1) + class_id(id) + stack_serial(4) + super_id(id) + loader_id(id)
///   + signers_id(id) + domain_id(id) + reserved1(id) + reserved2(id) + instance_size(4)
///     = 9 + 7*id bytes
fn class_dump_size(body: &[u8], pos: usize, id_size: usize) -> Result<usize, HprofError> {
    let id = id_size;
    // Start of variable section (after fixed prefix)
    let mut off = pos + 9 + 7 * id;

    // Constant pool
    if off + 2 > body.len() {
        return Err(HprofError::UnexpectedEof(off));
    }
    let cp_count = read_u16_be(body, off) as usize;
    off += 2;
    for _ in 0..cp_count {
        // cp_index(2) + type(1) + value
        if off + 3 > body.len() {
            return Err(HprofError::UnexpectedEof(off));
        }
        off += 2; // cp_index
        let prim_type = body[off];
        off += 1;
        let sz = value_size(prim_type, id)?;
        if off + sz > body.len() {
            return Err(HprofError::UnexpectedEof(off));
        }
        off += sz;
    }

    // Static fields
    if off + 2 > body.len() {
        return Err(HprofError::UnexpectedEof(off));
    }
    let statics_count = read_u16_be(body, off) as usize;
    off += 2;
    for _ in 0..statics_count {
        // name_id(id) + type(1) + value
        if off + id + 1 > body.len() {
            return Err(HprofError::UnexpectedEof(off));
        }
        off += id;
        let prim_type = body[off];
        off += 1;
        let sz = value_size(prim_type, id)?;
        if off + sz > body.len() {
            return Err(HprofError::UnexpectedEof(off));
        }
        off += sz;
    }

    // Instance fields: name_id(id) + type(1) each
    if off + 2 > body.len() {
        return Err(HprofError::UnexpectedEof(off));
    }
    let fields_count = read_u16_be(body, off) as usize;
    off += 2;
    let fields_size = fields_count * (id + 1);
    if off + fields_size > body.len() {
        return Err(HprofError::UnexpectedEof(off));
    }
    off += fields_size;

    Ok(off - pos)
}

/// Size in bytes of a value of the given hprof primitive type.
///
/// Type codes from heapDumper.cpp:
///   2=object(id_size), 4=bool(1), 5=char(2), 6=float(4), 7=double(8),
///   8=byte(1), 9=short(2), 10=int(4), 11=long(8)
pub fn value_size(type_id: u8, id_size: usize) -> Result<usize, HprofError> {
    match type_id {
        2 => Ok(id_size),
        4 => Ok(1),
        5 => Ok(2),
        6 => Ok(4),
        7 => Ok(8),
        8 => Ok(1),
        9 => Ok(2),
        10 => Ok(4),
        11 => Ok(8),
        other => Err(HprofError::UnknownPrimitiveType(other)),
    }
}

/// Read an id-sized value at `offset` within `data`, zero-extending to u64.
fn read_id(data: &[u8], offset: usize, id_size: usize) -> Result<u64, HprofError> {
    match id_size {
        4 => {
            if offset + 4 > data.len() {
                return Err(HprofError::UnexpectedEof(offset));
            }
            Ok(read_u32_be(data, offset) as u64)
        }
        8 => {
            if offset + 8 > data.len() {
                return Err(HprofError::UnexpectedEof(offset));
            }
            Ok(read_u64_be(data, offset))
        }
        _ => Err(HprofError::InvalidIdSize(id_size as u32)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sub_index_entry_round_trip() {
        let entry = SubIndexEntry {
            tag: TAG_INSTANCE_DUMP,
            object_id: 0xCAFE_BABE_1234_5678,
            position: 0x0000_0001_0000_0000,
        };
        let bytes = entry.to_bytes();
        assert_eq!(bytes.len(), SUB_INDEX_ENTRY_SIZE);
        assert_eq!(SubIndexEntry::from_bytes(&bytes), entry);
    }

    #[test]
    fn sub_index_entry_padding_is_zero() {
        let entry = SubIndexEntry {
            tag: 0x05,
            object_id: 1,
            position: 2,
        };
        let bytes = entry.to_bytes();
        assert!(bytes[1..8].iter().all(|&b| b == 0));
    }

    #[test]
    fn fixed_size_sub_records_id8() {
        // ROOT_STICKY_CLASS: subtag(1) + class_id(8) = 9
        let body: Vec<u8> = std::iter::once(TAG_ROOT_STICKY_CLASS)
            .chain(std::iter::repeat_n(0u8, 8))
            .collect();
        assert_eq!(sub_record_size(&body, 0, 8).unwrap(), 9);
    }

    #[test]
    fn instance_dump_size_id8() {
        // INSTANCE_DUMP: 1 + 8(obj) + 4(serial) + 8(class) + 4(len) + data_len
        let data_len: u32 = 12;
        let mut body = vec![TAG_INSTANCE_DUMP];
        body.extend_from_slice(&[0u8; 8]); // object_id
        body.extend_from_slice(&0u32.to_be_bytes()); // stack_serial
        body.extend_from_slice(&[0u8; 8]); // class_id
        body.extend_from_slice(&data_len.to_be_bytes()); // data_length
        body.extend_from_slice(&[0xAA; 12]); // data
        let expected = 1 + 8 + 4 + 8 + 4 + 12;
        assert_eq!(sub_record_size(&body, 0, 8).unwrap(), expected);
    }

    #[test]
    fn prim_array_dump_size_id8() {
        // PRIM_ARRAY_DUMP: 1 + 8(id) + 4(serial) + 4(num) + 1(type=10/int) + 3*4(data)
        let mut body = vec![TAG_PRIM_ARRAY_DUMP];
        body.extend_from_slice(&[0u8; 8]); // array_id
        body.extend_from_slice(&0u32.to_be_bytes()); // stack_serial
        body.extend_from_slice(&3u32.to_be_bytes()); // num_elements = 3
        body.push(10); // element_type = int (4 bytes each)
        body.extend_from_slice(&[0u8; 12]); // 3 * 4 bytes
        let expected = 1 + 8 + 4 + 4 + 1 + 12;
        assert_eq!(sub_record_size(&body, 0, 8).unwrap(), expected);
    }

    #[test]
    fn class_dump_empty_fields_id8() {
        // CLASS_DUMP with 0 cp entries, 0 statics, 0 instance fields
        const ID: usize = 8;
        let mut body = vec![TAG_CLASS_DUMP];
        body.extend_from_slice(&[0u8; ID]); // class_id
        body.extend_from_slice(&[0u8; 4]); // stack_serial
        body.extend_from_slice(&[0u8; 6 * ID]); // super + loader + signers + domain + res1 + res2
        body.extend_from_slice(&[0u8; 4]); // instance_size
        body.extend_from_slice(&0u16.to_be_bytes()); // cp_count = 0
        body.extend_from_slice(&0u16.to_be_bytes()); // statics_count = 0
        body.extend_from_slice(&0u16.to_be_bytes()); // instance_fields_count = 0
        let expected = body.len();
        assert_eq!(sub_record_size(&body, 0, 8).unwrap(), expected);
    }

    #[test]
    fn scanner_yields_all_sub_records() {
        // Two ROOT_STICKY_CLASS sub-records (id_size=8)
        let mut body = Vec::new();
        body.push(TAG_ROOT_STICKY_CLASS);
        body.extend_from_slice(&1u64.to_be_bytes());
        body.push(TAG_ROOT_STICKY_CLASS);
        body.extend_from_slice(&2u64.to_be_bytes());

        let entries: Vec<_> = SubRecordScanner::new(&body, 1000, 8)
            .unwrap()
            .map(|r| r.unwrap())
            .collect();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].tag, TAG_ROOT_STICKY_CLASS);
        assert_eq!(entries[0].object_id, 1);
        assert_eq!(entries[0].position, 1000);
        assert_eq!(entries[1].object_id, 2);
        assert_eq!(entries[1].position, 1009);
    }

    #[test]
    fn unknown_subtag_returns_error() {
        let body = [0xFEu8, 0, 0, 0, 0, 0, 0, 0, 0]; // unknown tag
        let mut scanner = SubRecordScanner::new(&body, 0, 8).unwrap();
        assert!(scanner.next().unwrap().is_err());
    }
}
