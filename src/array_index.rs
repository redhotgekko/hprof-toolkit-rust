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

use crate::heap_index::sub_record::{SubIndexEntry, TAG_OBJ_ARRAY_DUMP, TAG_PRIM_ARRAY_DUMP};
use crate::hprof::record::read_u32_be;
use crate::hprof::{BasicType, HprofError, HprofFile};
use crate::index::{Entry, IndexStore, RecordFile, RecordIter, RecordWriter, names, read_u64_le};

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

    /// The hprof basic type of this kind's elements.
    pub fn basic_type(self) -> BasicType {
        match self {
            Self::Boolean => BasicType::Boolean,
            Self::Char => BasicType::Char,
            Self::Float => BasicType::Float,
            Self::Double => BasicType::Double,
            Self::Byte => BasicType::Byte,
            Self::Short => BasicType::Short,
            Self::Int => BasicType::Int,
            Self::Long => BasicType::Long,
            Self::Object => BasicType::Object,
        }
    }

    /// Byte size of one element for this kind.
    ///
    /// For `Object` arrays, pass the hprof `id_size` (4 or 8).
    pub fn elem_size(self, id_size: u32) -> u64 {
        self.basic_type().size(id_size as usize) as u64
    }

    /// Construct from an hprof primitive `element_type` byte.
    ///
    /// Returns `None` for unknown/object type codes.
    pub fn from_prim_element_type(et: u8) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|k| k.basic_type().is_primitive() && k.basic_type().code() == et)
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

/// Scan all heap arrays in `combined_mmap` and write one size-sorted index
/// per array kind into `store`, under the names in [`names::ARRAYS`]
/// ([`ArrayKind::ALL`] order).
///
/// Each entry is streamed straight into the writer for its kind as the
/// object store is scanned, then every writer is sorted in place descending
/// by byte size (largest first) and committed.  Nothing proportional to the
/// number of arrays is held in process memory.
///
/// Returns the number of entries written for each [`ArrayKind`] in canonical
/// order (same order as [`ArrayKind::ALL`]).
pub fn build_array_size_indexes(
    hprof_source: &[u8],
    combined_mmap: &[u8],
    store: &dyn IndexStore,
) -> Result<[u64; 9], HprofError> {
    let hprof = HprofFile::from_ref(hprof_source)?;
    let id_size = hprof.id_size() as usize;
    let hprof_data = hprof.data();

    let combined = RecordFile::<SubIndexEntry>::new(combined_mmap)?;

    let writers: Vec<RecordWriter<ArraySizeEntry>> = names::ARRAYS
        .iter()
        .map(|n| Ok(RecordWriter::new(store.create(n)?)))
        .collect::<Result<_, HprofError>>()?;
    let mut writers: [RecordWriter<ArraySizeEntry>; 9] = writers
        .try_into()
        .map_err(|_| HprofError::Internal("expected 9 array writers".to_owned()))?;

    for entry in combined.iter() {
        let position = entry.position as usize;
        // Both array layouts start: subtag(1) + array_id(id_size) + stack_serial(4) + num_elements(4)
        let num_off = position + 1 + id_size + 4;
        match entry.tag {
            TAG_PRIM_ARRAY_DUMP => {
                // ... then elem_type(1)
                if num_off + 5 > hprof_data.len() {
                    continue;
                }
                let num_elements = u64::from(read_u32_be(hprof_data, num_off));
                let elem_type = hprof_data[num_off + 4];
                if let Some(kind) = ArrayKind::from_prim_element_type(elem_type) {
                    let elem_size = kind.elem_size(hprof.id_size());
                    writers[kind.index()].push(&ArraySizeEntry {
                        object_id: entry.object_id,
                        position: entry.position,
                        byte_size: num_elements * elem_size,
                    })?;
                }
            }
            TAG_OBJ_ARRAY_DUMP => {
                // ... then array_class_id(id_size)
                if num_off + 4 > hprof_data.len() {
                    continue;
                }
                let num_elements = u64::from(read_u32_be(hprof_data, num_off));
                writers[ArrayKind::Object.index()].push(&ArraySizeEntry {
                    object_id: entry.object_id,
                    position: entry.position,
                    byte_size: num_elements * id_size as u64,
                })?;
            }
            _ => {}
        }
    }

    let mut counts = [0u64; 9];
    for (i, w) in writers.into_iter().enumerate() {
        counts[i] = w.finish_sorted_desc()?;
    }
    Ok(counts)
}

/// Array size entries are sorted **descending** by `byte_size`, so the
/// key-based lookups of [`RecordFile`] must not be used on them; only
/// iteration is meaningful.
impl Entry for ArraySizeEntry {
    const SIZE: usize = ARRAY_SIZE_ENTRY_SIZE;
    const KEY_OFFSET: usize = 16;

    fn from_bytes(b: &[u8]) -> Self {
        Self {
            object_id: read_u64_le(b, 0),
            position: read_u64_le(b, 8),
            byte_size: read_u64_le(b, 16),
        }
    }

    fn write_to(&self, out: &mut [u8]) {
        out.copy_from_slice(&self.to_bytes());
    }

    fn key(&self) -> u64 {
        self.byte_size
    }
}

// ── Reader ────────────────────────────────────────────────────────────────────

/// Iterator over entries in an array size index file, largest first.
pub type ArraySizeIter<'a> = RecordIter<'a, ArraySizeEntry>;

/// Read-only handle to a single array size index file.
///
/// Entries are in descending `byte_size` order (largest arrays first).
#[derive(Clone, Copy)]
pub struct ArraySizeReader<'a> {
    file: RecordFile<'a, ArraySizeEntry>,
}

impl<'a> ArraySizeReader<'a> {
    /// Create a validated reader from a byte slice.
    pub fn from_ref(data: &'a [u8]) -> Result<Self, HprofError> {
        Ok(Self {
            file: RecordFile::new(data)?,
        })
    }

    /// Create a reader from a slice already known to be valid.
    pub(crate) fn from_slice(data: &'a [u8]) -> Self {
        Self {
            file: RecordFile::from_slice(data),
        }
    }

    /// Total number of entries in this index.
    pub fn len(&self) -> usize {
        self.file.len()
    }

    /// Iterate all entries in descending byte-size order.
    pub fn iter(&self) -> ArraySizeIter<'a> {
        self.file.iter()
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
