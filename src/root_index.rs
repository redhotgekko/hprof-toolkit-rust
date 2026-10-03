//! Per-type GC root index files — entry format and reader.
//!
//! Root index files are produced by [`crate::object_store::combine_sort_and_split`]
//! as part of the combined object-store build step.
//!
//! ## Entry format (16 bytes, all little-endian)
//!
//! ```text
//!  0..8    u64  object_id   first id-sized field of the sub-record
//!  8..16   u64  position    byte offset of the subtag in the hprof file
//! ```
//!
//! ## Output files
//!
//! | File                    | Sub-record tag | [`GcRootType`] variant  |
//! |-------------------------|----------------|-------------------------|
//! | `root_unknown.bin`      | `0xFF`         | `Unknown`               |
//! | `root_jni_global.bin`   | `0x01`         | `JniGlobal`             |
//! | `root_jni_local.bin`    | `0x02`         | `JniLocal`              |
//! | `root_java_frame.bin`   | `0x03`         | `JavaFrame`             |
//! | `root_native_stack.bin` | `0x04`         | `NativeStack`           |
//! | `root_sticky_class.bin` | `0x05`         | `StickyClass`           |
//! | `root_thread_block.bin` | `0x06`         | `ThreadBlock`           |
//! | `root_monitor_used.bin` | `0x07`         | `MonitorUsed`           |
//! | `root_thread_obj.bin`   | `0x08`         | `ThreadObject`          |

use crate::hprof::HprofError;
use crate::index::{Entry, RecordFile, RecordIter, read_u64_le};

// ── GcRootType ────────────────────────────────────────────────────────────────

/// The nine GC root sub-record types defined by the hprof format.
///
/// Used to select which kind of root to query through
/// [`crate::query::HeapQuery`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GcRootType {
    Unknown,
    JniGlobal,
    JniLocal,
    JavaFrame,
    NativeStack,
    StickyClass,
    ThreadBlock,
    MonitorUsed,
    ThreadObject,
}

impl GcRootType {
    /// All nine root types in a fixed canonical order (same as the array
    /// index used internally by [`crate::query::HeapQuery`]).
    pub const ALL: [GcRootType; 9] = [
        GcRootType::Unknown,
        GcRootType::JniGlobal,
        GcRootType::JniLocal,
        GcRootType::JavaFrame,
        GcRootType::NativeStack,
        GcRootType::StickyClass,
        GcRootType::ThreadBlock,
        GcRootType::MonitorUsed,
        GcRootType::ThreadObject,
    ];

    /// Returns the canonical array index for this type (0–8).
    /// The hprof sub-record tag of this root kind.
    pub(crate) fn sub_record_tag(self) -> u8 {
        use crate::heap_index::sub_record::*;
        match self {
            Self::Unknown => TAG_ROOT_UNKNOWN,
            Self::JniGlobal => TAG_ROOT_JNI_GLOBAL,
            Self::JniLocal => TAG_ROOT_JNI_LOCAL,
            Self::JavaFrame => TAG_ROOT_JAVA_FRAME,
            Self::NativeStack => TAG_ROOT_NATIVE_STACK,
            Self::StickyClass => TAG_ROOT_STICKY_CLASS,
            Self::ThreadBlock => TAG_ROOT_THREAD_BLOCK,
            Self::MonitorUsed => TAG_ROOT_MONITOR_USED,
            Self::ThreadObject => TAG_ROOT_THREAD_OBJ,
        }
    }

    pub(crate) fn index(self) -> usize {
        match self {
            Self::Unknown => 0,
            Self::JniGlobal => 1,
            Self::JniLocal => 2,
            Self::JavaFrame => 3,
            Self::NativeStack => 4,
            Self::StickyClass => 5,
            Self::ThreadBlock => 6,
            Self::MonitorUsed => 7,
            Self::ThreadObject => 8,
        }
    }

    /// Stable lower-case identifier (`"java_frame"`), used in URLs and tool
    /// arguments.  Inverse of [`Self::from_slug`].
    pub fn slug(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::JniGlobal => "jni_global",
            Self::JniLocal => "jni_local",
            Self::JavaFrame => "java_frame",
            Self::NativeStack => "native_stack",
            Self::StickyClass => "sticky_class",
            Self::ThreadBlock => "thread_block",
            Self::MonitorUsed => "monitor_used",
            Self::ThreadObject => "thread_object",
        }
    }

    /// The root type named by `slug`, if any.
    pub fn from_slug(slug: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|t| t.slug() == slug)
    }
}

// ── Entry format ──────────────────────────────────────────────────────────────

/// Byte size of one root index entry.
pub const ROOT_INDEX_ENTRY_SIZE: usize = 16;

/// A fixed-size entry in a per-type root index file.
///
/// Binary layout (all little-endian):
/// ```text
///  0..8    u64  object_id
///  8..16   u64  position  (byte offset of the subtag byte in the hprof file)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootIndexEntry {
    pub object_id: u64,
    pub position: u64,
}

impl RootIndexEntry {
    pub fn to_bytes(self) -> [u8; ROOT_INDEX_ENTRY_SIZE] {
        let mut buf = [0u8; ROOT_INDEX_ENTRY_SIZE];
        self.write_to(&mut buf);
        buf
    }
}

impl Entry for RootIndexEntry {
    const SIZE: usize = ROOT_INDEX_ENTRY_SIZE;
    const KEY_OFFSET: usize = 0;

    fn from_bytes(b: &[u8]) -> Self {
        Self {
            object_id: read_u64_le(b, 0),
            position: read_u64_le(b, 8),
        }
    }

    fn write_to(&self, out: &mut [u8]) {
        out[0..8].copy_from_slice(&self.object_id.to_le_bytes());
        out[8..16].copy_from_slice(&self.position.to_le_bytes());
    }

    fn key(&self) -> u64 {
        self.object_id
    }
}

// ── RootIndexReader ───────────────────────────────────────────────────────────

/// Iterator over all entries in a [`RootIndexReader`], ascending `object_id`.
pub type RootIter<'a> = RecordIter<'a, RootIndexEntry>;

/// Read-only handle to a single per-type root index file.
#[derive(Copy, Clone)]
pub struct RootIndexReader<'a> {
    file: RecordFile<'a, RootIndexEntry>,
}

impl<'a> RootIndexReader<'a> {
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

    /// Binary-search for an entry with the given `object_id`.  O(log n).
    pub fn find(&self, object_id: u64) -> Option<RootIndexEntry> {
        self.file.find(object_id)
    }

    /// Iterate all entries in ascending `object_id` order.
    pub fn iter(&self) -> RootIter<'a> {
        self.file.iter()
    }
}

/// Entry counts for each root type, returned by
/// [`crate::object_store::combine_sort_and_split`].
pub struct RootIndexCounts {
    pub root_unknown: u64,
    pub root_jni_global: u64,
    pub root_jni_local: u64,
    pub root_java_frame: u64,
    pub root_native_stack: u64,
    pub root_sticky_class: u64,
    pub root_thread_block: u64,
    pub root_monitor_used: u64,
    pub root_thread_obj: u64,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_root_index(entries: &[(u64, u64)]) -> Vec<u8> {
        let mut data = Vec::new();
        for &(object_id, position) in entries {
            let e = RootIndexEntry {
                object_id,
                position,
            };
            data.extend_from_slice(&e.to_bytes());
        }
        data
    }

    #[test]
    fn round_trip_entry() {
        let e = RootIndexEntry {
            object_id: 0xDEAD_BEEF_1234_5678,
            position: 0x0000_0001_0000_0000,
        };
        assert_eq!(RootIndexEntry::from_bytes(&e.to_bytes()), e);
    }

    #[test]
    fn reader_find_found() {
        let data = make_root_index(&[(10, 1000), (20, 2000), (30, 3000)]);
        let reader = RootIndexReader::from_ref(&data).unwrap();
        let e = reader.find(20).unwrap();
        assert_eq!(e.object_id, 20);
        assert_eq!(e.position, 2000);
    }

    #[test]
    fn reader_find_not_found() {
        let data = make_root_index(&[(10, 1000), (20, 2000)]);
        let reader = RootIndexReader::from_ref(&data).unwrap();
        assert!(reader.find(99).is_none());
    }

    #[test]
    fn reader_find_empty() {
        let empty: Vec<u8> = vec![];
        let reader = RootIndexReader::from_ref(&empty).unwrap();
        assert!(reader.find(1).is_none());
        assert_eq!(reader.file.len(), 0);
        assert!(reader.file.is_empty());
    }

    #[test]
    fn reader_rejects_misaligned_data() {
        assert!(RootIndexReader::from_ref(&[0u8; 17]).is_err());
    }

    #[test]
    fn reader_iter_yields_all_in_order() {
        let data = make_root_index(&[(10, 100), (20, 200), (30, 300)]);
        let reader = RootIndexReader::from_ref(&data).unwrap();
        let entries: Vec<_> = reader.iter().collect();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].object_id, 10);
        assert_eq!(entries[1].object_id, 20);
        assert_eq!(entries[2].object_id, 30);
    }

    #[test]
    fn slugs_round_trip() {
        for t in GcRootType::ALL {
            assert_eq!(GcRootType::from_slug(t.slug()), Some(t));
        }
        assert_eq!(GcRootType::from_slug("purple"), None);
    }

    #[test]
    fn gc_root_type_all_coverage() {
        assert_eq!(GcRootType::ALL.len(), 9);
        let indices: Vec<usize> = GcRootType::ALL.iter().map(|t| t.index()).collect();
        assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
    }
}
