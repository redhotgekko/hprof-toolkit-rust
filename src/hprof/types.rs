//! The hprof basic-type table, in one place.
//!
//! Heap dumps describe field values and array elements with a one-byte type
//! code (heapDumper.cpp): `2` object reference, `4` boolean, `5` char,
//! `6` float, `7` double, `8` byte, `9` short, `10` int, `11` long.  Every
//! part of the crate that needs the size, name or descriptor letter of such
//! a code goes through [`BasicType`] instead of keeping its own table.

use super::HprofError;

/// A hprof basic type: an object reference or one of the eight primitives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BasicType {
    Object,
    Boolean,
    Char,
    Float,
    Double,
    Byte,
    Short,
    Int,
    Long,
}

impl BasicType {
    /// Every type, in type-code order.
    pub const ALL: [BasicType; 9] = [
        BasicType::Object,
        BasicType::Boolean,
        BasicType::Char,
        BasicType::Float,
        BasicType::Double,
        BasicType::Byte,
        BasicType::Short,
        BasicType::Int,
        BasicType::Long,
    ];

    /// Decode an hprof type code; `None` for codes the format does not define
    /// (including `0`, `1` and `3`, which appear only in class-file contexts).
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            2 => Some(Self::Object),
            4 => Some(Self::Boolean),
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

    /// Java name for a raw type code, or `"?"` when the code is unknown.
    pub fn name_of_code(code: u8) -> &'static str {
        Self::from_code(code).map_or("?", Self::java_name)
    }

    /// Like [`Self::from_code`] but with the crate's error for unknown codes.
    pub fn from_code_or_err(code: u8) -> Result<Self, HprofError> {
        Self::from_code(code).ok_or(HprofError::UnknownPrimitiveType(code))
    }

    /// The hprof type code.
    pub fn code(self) -> u8 {
        match self {
            Self::Object => 2,
            Self::Boolean => 4,
            Self::Char => 5,
            Self::Float => 6,
            Self::Double => 7,
            Self::Byte => 8,
            Self::Short => 9,
            Self::Int => 10,
            Self::Long => 11,
        }
    }

    /// `true` for everything except [`BasicType::Object`].
    pub fn is_primitive(self) -> bool {
        self != Self::Object
    }

    /// Size in bytes of one value; object references take `id_size` bytes.
    pub fn size(self, id_size: usize) -> usize {
        match self {
            Self::Object => id_size,
            Self::Boolean | Self::Byte => 1,
            Self::Char | Self::Short => 2,
            Self::Float | Self::Int => 4,
            Self::Double | Self::Long => 8,
        }
    }

    /// Java source name: `"boolean"`, `"int"`, …; `"Object"` for references.
    pub fn java_name(self) -> &'static str {
        match self {
            Self::Object => "Object",
            Self::Boolean => "boolean",
            Self::Char => "char",
            Self::Float => "float",
            Self::Double => "double",
            Self::Byte => "byte",
            Self::Short => "short",
            Self::Int => "int",
            Self::Long => "long",
        }
    }

    /// JVM descriptor letter: `'Z'`, `'C'`, `'F'`, `'D'`, `'B'`, `'S'`, `'I'`,
    /// `'J'`; `'L'` for references.
    pub fn descriptor_char(self) -> char {
        match self {
            Self::Object => 'L',
            Self::Boolean => 'Z',
            Self::Char => 'C',
            Self::Float => 'F',
            Self::Double => 'D',
            Self::Byte => 'B',
            Self::Short => 'S',
            Self::Int => 'I',
            Self::Long => 'J',
        }
    }

    /// The primitive named by a JVM descriptor letter (`'I'` → `Int`).
    /// `'L'` and unknown letters give `None`: only primitives are decoded.
    pub fn from_descriptor_char(c: char) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|t| t.is_primitive() && t.descriptor_char() == c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_round_trip_and_unknown_codes_are_rejected() {
        for t in BasicType::ALL {
            assert_eq!(BasicType::from_code(t.code()), Some(t));
        }
        for bad in [0u8, 1, 3, 12, 255] {
            assert_eq!(BasicType::from_code(bad), None, "code {bad}");
            assert!(matches!(
                BasicType::from_code_or_err(bad),
                Err(HprofError::UnknownPrimitiveType(c)) if c == bad
            ));
        }
    }

    #[test]
    fn sizes_follow_the_hprof_spec() {
        let sizes: Vec<usize> = BasicType::ALL.iter().map(|t| t.size(8)).collect();
        assert_eq!(sizes, [8, 1, 2, 4, 8, 1, 2, 4, 8]);
        assert_eq!(BasicType::Object.size(4), 4);
    }

    #[test]
    fn names_and_descriptor_letters() {
        assert_eq!(BasicType::Boolean.java_name(), "boolean");
        assert_eq!(BasicType::Object.java_name(), "Object");
        assert_eq!(BasicType::Long.descriptor_char(), 'J');
        assert_eq!(
            BasicType::from_descriptor_char('Z'),
            Some(BasicType::Boolean)
        );
        assert_eq!(BasicType::from_descriptor_char('J'), Some(BasicType::Long));
        assert_eq!(BasicType::from_descriptor_char('L'), None);
        assert_eq!(BasicType::from_descriptor_char('x'), None);
        for t in BasicType::ALL.into_iter().filter(|t| t.is_primitive()) {
            assert_eq!(
                BasicType::from_descriptor_char(t.descriptor_char()),
                Some(t)
            );
        }
    }
}
