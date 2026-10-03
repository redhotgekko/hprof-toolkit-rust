//! Field value and wrapper type resolution helpers.

use crate::heap_parser::FieldValue;
use crate::hprof::record::{read_u16_be, read_u32_be, read_u64_be};
use crate::hprof::{BasicType, HprofError};

// ── Field ─────────────────────────────────────────────────────────────────────

/// A single instance field with its resolved name and raw value.
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    /// Field name (from the UTF-8 name index).
    pub name: String,
    /// Declared type of the field.
    pub ty: BasicType,
    /// Raw field value parsed from the instance data bytes.
    pub value: FieldValue,
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
    let ty = BasicType::from_code_or_err(field_type)?;
    let len = ty.size(id_size);
    if offset + len > data.len() {
        return Err(HprofError::UnexpectedEof(offset));
    }
    let value = match ty {
        BasicType::Object => FieldValue::Object(match id_size {
            4 => read_u32_be(data, offset) as u64,
            8 => read_u64_be(data, offset),
            _ => return Err(HprofError::InvalidIdSize(id_size as u32)),
        }),
        BasicType::Boolean => FieldValue::Bool(data[offset] != 0),
        BasicType::Char => FieldValue::Char(read_u16_be(data, offset)),
        BasicType::Float => FieldValue::Float(f32::from_bits(read_u32_be(data, offset))),
        BasicType::Double => FieldValue::Double(f64::from_bits(read_u64_be(data, offset))),
        BasicType::Byte => FieldValue::Byte(data[offset] as i8),
        BasicType::Short => FieldValue::Short(read_u16_be(data, offset) as i16),
        BasicType::Int => FieldValue::Int(read_u32_be(data, offset) as i32),
        BasicType::Long => FieldValue::Long(read_u64_be(data, offset) as i64),
    };
    Ok((value, len))
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
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_be_bytes(*c))
            .collect();
        std::string::String::from_utf16_lossy(&units)
    }
}

/// Decode a `char[]` array (element_type = 5) stored as big-endian UTF-16.
pub fn decode_char_array(bytes: &[u8]) -> std::string::String {
    let units: Vec<u16> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_be_bytes(*c))
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
