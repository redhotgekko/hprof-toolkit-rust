//! [`ClassKey`]: what an object is "an instance of", as one `u64`.
//!
//! Instances have a real class object id.  Arrays have no class object of
//! their own in the heap dump (only a `LOAD_CLASS` record), so the histogram
//! and the instances-by-class index group them under **synthetic** keys:
//!
//! | Kind             | Encoding                                  |
//! |------------------|-------------------------------------------|
//! | `Class(id)`      | `id` unchanged                            |
//! | `ObjArray(arr)`  | bit 63 set, array class id in bits 0–62   |
//! | `PrimArray(t)`   | bits 63 and 62 set, hprof type code in bits 0–7 |
//!
//! 64-bit HotSpot heap addresses never set the top two bits, so the synthetic
//! keys cannot collide with real ids.  The encoding is bit-compatible with
//! the ids the HTTP UI has always used in its URLs.

use crate::heap_parser::SubRecord;

/// Bit set on every synthetic key.
pub const OBJ_ARRAY_FLAG: u64 = 1u64 << 63;
/// Bits set on primitive-array keys (`OBJ_ARRAY_FLAG` plus bit 62).
pub const PRIM_ARRAY_FLAG: u64 = (1u64 << 63) | (1u64 << 62);

/// The class-like grouping of a heap object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ClassKey {
    /// An ordinary instance of the class object `class_id`.
    Class(u64),
    /// An object array whose **array class** object (`[Ljava.lang.String;`)
    /// is the payload.  hprof records the array class in an object-array
    /// record, not the element class.
    ObjArray(u64),
    /// A primitive array; the payload is the hprof element type code
    /// (4 = boolean … 11 = long).
    PrimArray(u8),
}

impl ClassKey {
    /// Encode as a `u64` (see the module docs).
    pub fn to_u64(self) -> u64 {
        match self {
            ClassKey::Class(id) => id,
            ClassKey::ObjArray(elem) => OBJ_ARRAY_FLAG | (elem & !PRIM_ARRAY_FLAG),
            ClassKey::PrimArray(t) => PRIM_ARRAY_FLAG | u64::from(t),
        }
    }

    /// Decode a key produced by [`Self::to_u64`].
    pub fn from_u64(v: u64) -> Self {
        if v & PRIM_ARRAY_FLAG == PRIM_ARRAY_FLAG {
            ClassKey::PrimArray((v & 0xFF) as u8)
        } else if v & OBJ_ARRAY_FLAG != 0 {
            ClassKey::ObjArray(v & !OBJ_ARRAY_FLAG)
        } else {
            ClassKey::Class(v)
        }
    }

    /// `true` when `v` encodes an array key rather than a real class id.
    pub fn is_synthetic(v: u64) -> bool {
        v & OBJ_ARRAY_FLAG != 0
    }

    /// The key an object record belongs to, or `None` for GC-root records
    /// and class dumps (a class dump is not an instance of anything).
    pub fn of(record: &SubRecord<'_>) -> Option<Self> {
        match record {
            SubRecord::InstanceDump(i) => Some(ClassKey::Class(i.class_id)),
            SubRecord::ObjArrayDump(a) => Some(ClassKey::ObjArray(a.array_class_id)),
            SubRecord::PrimArrayDump(a) => Some(ClassKey::PrimArray(a.element_type)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_all_variants() {
        for key in [
            ClassKey::Class(0),
            ClassKey::Class(0x7fff_ffff_ffff_ffff),
            ClassKey::Class(0x1234_5678),
            ClassKey::ObjArray(0x30),
            ClassKey::PrimArray(4),
            ClassKey::PrimArray(11),
        ] {
            assert_eq!(ClassKey::from_u64(key.to_u64()), key, "{key:?}");
        }
    }

    #[test]
    fn encoding_matches_legacy_constants() {
        assert_eq!(ClassKey::ObjArray(0x30).to_u64(), (1u64 << 63) | 0x30);
        assert_eq!(
            ClassKey::PrimArray(8).to_u64(),
            (1u64 << 63) | (1u64 << 62) | 8
        );
        assert!(ClassKey::is_synthetic(ClassKey::ObjArray(1).to_u64()));
        assert!(ClassKey::is_synthetic(ClassKey::PrimArray(1).to_u64()));
        assert!(!ClassKey::is_synthetic(ClassKey::Class(1).to_u64()));
    }
}
