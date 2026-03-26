//! Field value and wrapper type resolution helpers.

use crate::heap_parser::FieldValue;
use crate::hprof::HprofError;
use crate::hprof::record::{read_u16_be, read_u32_be, read_u64_be};

// ── ResolvedField ─────────────────────────────────────────────────────────────

/// A single instance field with its resolved name and raw hprof value.
#[derive(Debug, Clone)]
pub struct ResolvedField {
    /// Field name (from the UTF-8 name index).
    pub name: String,
    /// hprof type code (2=object, 4=bool, 5=char, 6=float, 7=double,
    /// 8=byte, 9=short, 10=int, 11=long).
    pub field_type: u8,
    /// Raw field value parsed from the instance data bytes.
    pub value: FieldValue,
}

// ── JavaValue ─────────────────────────────────────────────────────────────────

/// A Java value, potentially unwrapping common Java wrapper types.
///
/// The first element of each tuple is always the `object_id` of the wrapper
/// instance, preserving the link back to the original heap object.
#[derive(Debug, Clone)]
pub enum JavaValue {
    /// `java.lang.String` instance: `(object_id, string_content)`.
    String(u64, std::string::String),
    /// `java.lang.Boolean` instance: `(object_id, value)`.
    Boolean(u64, bool),
    /// `java.lang.Byte` instance: `(object_id, value)`.
    Byte(u64, i8),
    /// `java.lang.Short` instance: `(object_id, value)`.
    Short(u64, i16),
    /// `java.lang.Character` instance: `(object_id, UTF-16 code unit)`.
    Character(u64, u16),
    /// `java.lang.Integer` instance: `(object_id, value)`.
    Integer(u64, i32),
    /// `java.lang.Long` instance: `(object_id, value)`.
    Long(u64, i64),
    /// `java.lang.Float` instance: `(object_id, value)`.
    Float(u64, f32),
    /// `java.lang.Double` instance: `(object_id, value)`.
    Double(u64, f64),
    /// Generic object reference (not a recognised wrapper type).
    Object(u64),
    /// Null reference (`object_id == 0`).
    Null,
}

// ── Field-value parser ────────────────────────────────────────────────────────

/// Parse a typed field value at `offset` within `data`.
///
/// Returns `(value, bytes_consumed)`.
pub fn read_field_value(
    data: &[u8],
    offset: usize,
    field_type: u8,
    id_size: usize,
) -> Result<(FieldValue, usize), HprofError> {
    match field_type {
        2 => {
            // Object reference
            if offset + id_size > data.len() {
                return Err(HprofError::UnexpectedEof(offset));
            }
            let id = match id_size {
                4 => read_u32_be(data, offset) as u64,
                8 => read_u64_be(data, offset),
                _ => return Err(HprofError::InvalidIdSize(id_size as u32)),
            };
            Ok((FieldValue::Object(id), id_size))
        }
        4 => {
            if offset + 1 > data.len() {
                return Err(HprofError::UnexpectedEof(offset));
            }
            Ok((FieldValue::Bool(data[offset] != 0), 1))
        }
        5 => {
            if offset + 2 > data.len() {
                return Err(HprofError::UnexpectedEof(offset));
            }
            Ok((FieldValue::Char(read_u16_be(data, offset)), 2))
        }
        6 => {
            if offset + 4 > data.len() {
                return Err(HprofError::UnexpectedEof(offset));
            }
            Ok((
                FieldValue::Float(f32::from_bits(read_u32_be(data, offset))),
                4,
            ))
        }
        7 => {
            if offset + 8 > data.len() {
                return Err(HprofError::UnexpectedEof(offset));
            }
            Ok((
                FieldValue::Double(f64::from_bits(read_u64_be(data, offset))),
                8,
            ))
        }
        8 => {
            if offset + 1 > data.len() {
                return Err(HprofError::UnexpectedEof(offset));
            }
            Ok((FieldValue::Byte(data[offset] as i8), 1))
        }
        9 => {
            if offset + 2 > data.len() {
                return Err(HprofError::UnexpectedEof(offset));
            }
            Ok((FieldValue::Short(read_u16_be(data, offset) as i16), 2))
        }
        10 => {
            if offset + 4 > data.len() {
                return Err(HprofError::UnexpectedEof(offset));
            }
            Ok((FieldValue::Int(read_u32_be(data, offset) as i32), 4))
        }
        11 => {
            if offset + 8 > data.len() {
                return Err(HprofError::UnexpectedEof(offset));
            }
            Ok((FieldValue::Long(read_u64_be(data, offset) as i64), 8))
        }
        other => Err(HprofError::UnknownPrimitiveType(other)),
    }
}

// ── String decoding ───────────────────────────────────────────────────────────

/// Decode a `byte[]` array as a Java compact string.
///
/// For LATIN-1 (`coder == 0`): each byte is a Unicode code point.
/// For UTF-16 (`coder == 1`): pairs of bytes are big-endian UTF-16 code units.
pub fn decode_string_bytes(bytes: &[u8], coder: u8) -> std::string::String {
    if coder == 0 {
        // LATIN-1: direct byte-to-char mapping.
        bytes.iter().map(|&b| b as char).collect()
    } else {
        // UTF-16 big-endian.
        let units: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|c| u16::from_be_bytes([c[0], c[1]]))
            .collect();
        std::string::String::from_utf16_lossy(&units)
    }
}

/// Decode a `char[]` array (element_type = 5) stored as big-endian UTF-16.
pub fn decode_char_array(bytes: &[u8]) -> std::string::String {
    let units: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|c| u16::from_be_bytes([c[0], c[1]]))
        .collect();
    std::string::String::from_utf16_lossy(&units)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_int_field() {
        let data = 42i32.to_be_bytes();
        let (v, consumed) = read_field_value(&data, 0, 10, 8).unwrap();
        assert_eq!(v, FieldValue::Int(42));
        assert_eq!(consumed, 4);
    }

    #[test]
    fn read_object_field_id8() {
        let mut data = [0u8; 8];
        data.copy_from_slice(&0xCAFEBABEu64.to_be_bytes());
        let (v, consumed) = read_field_value(&data, 0, 2, 8).unwrap();
        assert_eq!(v, FieldValue::Object(0xCAFEBABE));
        assert_eq!(consumed, 8);
    }

    #[test]
    fn read_object_field_id4() {
        let data = 0x1234u32.to_be_bytes();
        let (v, consumed) = read_field_value(&data, 0, 2, 4).unwrap();
        assert_eq!(v, FieldValue::Object(0x1234));
        assert_eq!(consumed, 4);
    }

    #[test]
    fn decode_latin1_bytes() {
        let bytes = b"hello";
        assert_eq!(decode_string_bytes(bytes, 0), "hello");
    }

    #[test]
    fn decode_utf16_bytes() {
        // "AB" as big-endian UTF-16
        let bytes = [0x00, 0x41, 0x00, 0x42];
        assert_eq!(decode_string_bytes(&bytes, 1), "AB");
    }

    #[test]
    fn decode_char_array_utf16() {
        // "Hi" as big-endian char[]
        let bytes = [0x00, 0x48, 0x00, 0x69];
        assert_eq!(decode_char_array(&bytes), "Hi");
    }
}
