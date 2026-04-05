//! Array size index files — one per array element type (boolean[], char[], …, Object[]).
//!
//! Each file stores one 24-byte record per array, sorted by element byte size
//! descending (largest first), enabling O(1) prefix access to the N largest arrays.
//!
//! ## Entry format (24 bytes, all little-endian)
//!
//! ```text
//!  0..8    u64  object_id   (array_id from the hprof sub-record)
//!  8..16   u64  position    (byte offset of the subtag byte in the hprof file)
//! 16..24   u64  byte_size   (total bytes occupied by array elements)
//! ```
//!
//! ## Output files
//!
//! | File                | Array type  |
//! |---------------------|-------------|
//! | `array_boolean.bin` | `boolean[]` |
//! | `array_char.bin`    | `char[]`    |
//! | `array_float.bin`   | `float[]`   |
//! | `array_double.bin`  | `double[]`  |
//! | `array_byte.bin`    | `byte[]`    |
//! | `array_short.bin`   | `short[]`   |
//! | `array_int.bin`     | `int[]`     |
//! | `array_long.bin`    | `long[]`    |
//! | `array_object.bin`  | `Object[]`  |

use crate::heap_index::sub_record::{
    SUB_INDEX_ENTRY_SIZE, TAG_OBJ_ARRAY_DUMP, TAG_PRIM_ARRAY_DUMP,
};
use crate::hprof::record::read_u32_be;
use crate::hprof::{HprofError, HprofFile};
use crate::vfs::MMapWriter;
use rayon::prelude::*;
use std::io::Write;

// ── ArrayKind ─────────────────────────────────────────────────────────────────

/// The nine array kinds tracked in the array size index.
///
/// Variants correspond to all eight Java primitive array element types plus
/// a catch-all for object reference arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayKind {
    Boolean,
    Char,
    Float,
    Double,
    Byte,
    Short,
    Int,
    Long,
    Object,
}

impl ArrayKind {
    /// All nine array kinds in canonical index order (0–8).
    pub const ALL: [ArrayKind; 9] = [
        ArrayKind::Boolean,
        ArrayKind::Char,
        ArrayKind::Float,
        ArrayKind::Double,
        ArrayKind::Byte,
        ArrayKind::Short,
        ArrayKind::Int,
        ArrayKind::Long,
        ArrayKind::Object,
    ];

    /// Canonical array index (0–8), used as the key into fixed-size arrays.
    pub fn index(self) -> usize {
        match self {
            Self::Boolean => 0,
            Self::Char => 1,
            Self::Float => 2,
            Self::Double => 3,
            Self::Byte => 4,
            Self::Short => 5,
            Self::Int => 6,
            Self::Long => 7,
            Self::Object => 8,
        }
    }

    /// Parse an [`ArrayKind`] from its URL slug (e.g. `"int"`, `"object"`).
    pub fn from_slug(s: &str) -> Option<Self> {
        match s {
            "boolean" => Some(Self::Boolean),
            "char" => Some(Self::Char),
            "float" => Some(Self::Float),
            "double" => Some(Self::Double),
            "byte" => Some(Self::Byte),
            "short" => Some(Self::Short),
            "int" => Some(Self::Int),
            "long" => Some(Self::Long),
            "object" => Some(Self::Object),
            _ => None,
        }
    }

    /// URL slug for this kind (e.g. `"int"`, `"object"`).
    pub fn slug(self) -> &'static str {
        match self {
            Self::Boolean => "boolean",
            Self::Char => "char",
            Self::Float => "float",
            Self::Double => "double",
            Self::Byte => "byte",
            Self::Short => "short",
            Self::Int => "int",
            Self::Long => "long",
            Self::Object => "object",
        }
    }

    /// Java-style display name for the array type (e.g. `"int[]"`).
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Boolean => "boolean[]",
            Self::Char => "char[]",
            Self::Float => "float[]",
            Self::Double => "double[]",
            Self::Byte => "byte[]",
            Self::Short => "short[]",
            Self::Int => "int[]",
            Self::Long => "long[]",
            Self::Object => "Object[]",
        }
    }

    /// Output file name for this kind.
    pub fn file_name(self) -> &'static str {
        match self {
            Self::Boolean => "array_boolean.bin",
            Self::Char => "array_char.bin",
            Self::Float => "array_float.bin",
            Self::Double => "array_double.bin",
            Self::Byte => "array_byte.bin",
            Self::Short => "array_short.bin",
            Self::Int => "array_int.bin",
            Self::Long => "array_long.bin",
            Self::Object => "array_object.bin",
        }
    }

    /// Byte size of one element for this kind.
    ///
    /// For `Object` arrays, pass the hprof `id_size` (4 or 8).
    pub fn elem_size(self, id_size: u32) -> u64 {
        match self {
            Self::Boolean | Self::Byte => 1,
            Self::Char | Self::Short => 2,
            Self::Float | Self::Int => 4,
            Self::Double | Self::Long => 8,
            Self::Object => u64::from(id_size),
        }
    }

    /// Construct from an hprof primitive `element_type` byte.
    ///
    /// Returns `None` for unknown/object type codes.
    pub fn from_prim_element_type(et: u8) -> Option<Self> {
        match et {
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
}

// ── Entry format ──────────────────────────────────────────────────────────────

/// Byte size of one array size index entry.
pub const ARRAY_SIZE_ENTRY_SIZE: usize = 24;

/// A fixed-size entry in an array size index file.
///
/// Binary layout (all little-endian):
/// ```text
///  0..8    u64  object_id   (array_id from the hprof sub-record)
///  8..16   u64  position    (byte offset of the subtag byte in the hprof file)
/// 16..24   u64  byte_size   (total bytes occupied by array elements)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArraySizeEntry {
    /// Array object ID.
    pub object_id: u64,
    /// Byte offset of the subtag byte in the hprof file (for on-demand parsing).
    pub position: u64,
    /// Total bytes occupied by the array's elements.
    pub byte_size: u64,
}

impl ArraySizeEntry {
    pub fn to_bytes(self) -> [u8; ARRAY_SIZE_ENTRY_SIZE] {
        let mut buf = [0u8; ARRAY_SIZE_ENTRY_SIZE];
        buf[0..8].copy_from_slice(&self.object_id.to_le_bytes());
        buf[8..16].copy_from_slice(&self.position.to_le_bytes());
        buf[16..24].copy_from_slice(&self.byte_size.to_le_bytes());
        buf
    }

    pub fn from_bytes(bytes: &[u8; ARRAY_SIZE_ENTRY_SIZE]) -> Self {
        let object_id = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        let position = u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]);
        let byte_size = u64::from_le_bytes([
            bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23],
        ]);
        Self {
            object_id,
            position,
            byte_size,
        }
    }
}

// ── Builder ───────────────────────────────────────────────────────────────────

/// Scan all heap arrays in `combined_source` and write one size-sorted index
/// per array kind to the corresponding `MMapWriter` in `outputs`.
///
/// `outputs` must be an array of 9 `MMapWriter`s in [`ArrayKind::ALL`] order
/// (Boolean=0, Char=1, …, Object=8).
///
/// Arrays with the same element type are grouped together and sorted descending
/// by their total element byte size (largest first).
///
/// Returns the number of entries written for each [`ArrayKind`] in canonical
/// order (same order as [`ArrayKind::ALL`]).
pub fn build_array_size_indexes(
    hprof_source: &[u8],
    combined_mmap: &[u8],
    outputs: &mut [impl MMapWriter; 9],
) -> Result<[u64; 9], HprofError> {
    let hprof = HprofFile::from_ref(hprof_source)?;
    let id_size = hprof.header.id_size as usize;
    let hprof_data = hprof.data();

    if !combined_mmap.len().is_multiple_of(SUB_INDEX_ENTRY_SIZE) {
        return Err(HprofError::InvalidIndexFile);
    }

    // One accumulator per ArrayKind.
    let mut buckets: [Vec<ArraySizeEntry>; 9] = std::array::from_fn(|_| Vec::new());

    let n_entries = combined_mmap.len() / SUB_INDEX_ENTRY_SIZE;
    for i in 0..n_entries {
        let start = i * SUB_INDEX_ENTRY_SIZE;
        let tag = combined_mmap[start];

        match tag {
            TAG_PRIM_ARRAY_DUMP => {
                let object_id = read_le_u64(combined_mmap, start + 8);
                let position = read_le_u64(combined_mmap, start + 16) as usize;

                // PrimArrayDump layout at `position` in the hprof file:
                //   subtag(1) + array_id(id_size) + stack_serial(4) + num_elements(4) + elem_type(1)
                let num_off = position + 1 + id_size + 4;
                if num_off + 5 > hprof_data.len() {
                    continue;
                }
                let num_elements = read_u32_be(hprof_data, num_off) as u64;
                let elem_type = hprof_data[num_off + 4];

                if let Some(kind) = ArrayKind::from_prim_element_type(elem_type) {
                    let elem_size = kind.elem_size(hprof.header.id_size);
                    buckets[kind.index()].push(ArraySizeEntry {
                        object_id,
                        position: position as u64,
                        byte_size: num_elements * elem_size,
                    });
                }
            }
            TAG_OBJ_ARRAY_DUMP => {
                let object_id = read_le_u64(combined_mmap, start + 8);
                let position = read_le_u64(combined_mmap, start + 16) as usize;

                // ObjArrayDump layout at `position` in the hprof file:
                //   subtag(1) + array_id(id_size) + stack_serial(4) + num_elements(4) + elem_class_id(id_size)
                let num_off = position + 1 + id_size + 4;
                if num_off + 4 > hprof_data.len() {
                    continue;
                }
                let num_elements = read_u32_be(hprof_data, num_off) as u64;
                let byte_size = num_elements * id_size as u64;
                buckets[ArrayKind::Object.index()].push(ArraySizeEntry {
                    object_id,
                    position: position as u64,
                    byte_size,
                });
            }
            _ => {}
        }
    }

    // Sort each bucket descending by byte_size and write via outputs.
    let mut counts = [0u64; 9];
    for (i, bucket) in buckets.iter_mut().enumerate() {
        bucket.par_sort_unstable_by(|a, b| b.byte_size.cmp(&a.byte_size));

        let mut writer = outputs[i].create_writer()?;
        for entry in bucket.iter() {
            writer.write_all(&entry.to_bytes())?;
        }
        counts[i] = bucket.len() as u64;
    }

    Ok(counts)
}

/// Read a little-endian u64 from `data` at `offset`.
///
/// Panics if `offset + 8 > data.len()`.
fn read_le_u64(data: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ])
}

// ── Reader ────────────────────────────────────────────────────────────────────

/// Read-only handle to a single array size index file.
///
/// Entries are in descending `byte_size` order (largest arrays first).
pub struct ArraySizeReader<'a> {
    data: &'a [u8],
}

impl<'a> ArraySizeReader<'a> {
    /// Create a validated reader from a byte slice.
    pub fn from_ref(data: &'a [u8]) -> Result<Self, HprofError> {
        if !data.len().is_multiple_of(ARRAY_SIZE_ENTRY_SIZE) {
            return Err(HprofError::InvalidIndexFile);
        }
        Ok(Self { data })
    }

    /// Create a reader from a slice already known to be valid.
    pub(crate) fn from_slice(data: &'a [u8]) -> Self {
        debug_assert!(data.len().is_multiple_of(ARRAY_SIZE_ENTRY_SIZE));
        Self { data }
    }

    fn as_slice(&self) -> &[u8] {
        self.data
    }

    /// Total number of entries in this index.
    pub fn len(&self) -> usize {
        self.as_slice().len() / ARRAY_SIZE_ENTRY_SIZE
    }

    /// Returns `true` if this index contains no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the entry at index `idx`, or `None` if out of range.
    pub fn entry_at(&self, idx: usize) -> Option<ArraySizeEntry> {
        let data = self.as_slice();
        let start = idx * ARRAY_SIZE_ENTRY_SIZE;
        let end = start + ARRAY_SIZE_ENTRY_SIZE;
        if end > data.len() {
            return None;
        }
        let bytes: &[u8; ARRAY_SIZE_ENTRY_SIZE] = (&data[start..end]).try_into().ok()?;
        Some(ArraySizeEntry::from_bytes(bytes))
    }

    /// Iterate all entries in descending byte-size order.
    pub fn iter(&self) -> ArraySizeIter<'a> {
        ArraySizeIter {
            data: self.data,
            pos: 0,
        }
    }
}

/// Iterator over entries in an array size index file.
///
/// Yields entries in descending `byte_size` order (largest first).
pub struct ArraySizeIter<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Iterator for ArraySizeIter<'_> {
    type Item = ArraySizeEntry;

    fn next(&mut self) -> Option<Self::Item> {
        let end = self.pos + ARRAY_SIZE_ENTRY_SIZE;
        if end > self.data.len() {
            return None;
        }
        let bytes: &[u8; ARRAY_SIZE_ENTRY_SIZE] = (&self.data[self.pos..end]).try_into().ok()?;
        let entry = ArraySizeEntry::from_bytes(bytes);
        self.pos = end;
        Some(entry)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_round_trip() {
        let entry = ArraySizeEntry {
            object_id: 0xCAFE_BABE_1234_5678,
            position: 0x0000_0001_DEAD_BEEF,
            byte_size: 1_048_576,
        };
        let bytes = entry.to_bytes();
        assert_eq!(bytes.len(), ARRAY_SIZE_ENTRY_SIZE);
        assert_eq!(ArraySizeEntry::from_bytes(&bytes), entry);
    }

    #[test]
    fn all_kinds_have_unique_indexes() {
        let mut seen = [false; 9];
        for kind in ArrayKind::ALL {
            let idx = kind.index();
            assert!(!seen[idx], "duplicate index {idx} for {kind:?}");
            seen[idx] = true;
        }
    }

    #[test]
    fn slug_round_trip() {
        for kind in ArrayKind::ALL {
            assert_eq!(ArrayKind::from_slug(kind.slug()), Some(kind));
        }
    }

    #[test]
    fn prim_element_types_map_correctly() {
        assert_eq!(
            ArrayKind::from_prim_element_type(4),
            Some(ArrayKind::Boolean)
        );
        assert_eq!(ArrayKind::from_prim_element_type(8), Some(ArrayKind::Byte));
        assert_eq!(ArrayKind::from_prim_element_type(10), Some(ArrayKind::Int));
        assert_eq!(ArrayKind::from_prim_element_type(11), Some(ArrayKind::Long));
        assert_eq!(ArrayKind::from_prim_element_type(99), None);
    }
}
