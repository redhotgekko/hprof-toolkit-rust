//! Per-type GC root index files — entry format, reader, and binary-search.
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
use std::path::Path;

// ── GcRootType ────────────────────────────────────────────────────────────────

/// The nine GC root sub-record types defined by the hprof format.
///
/// Used to select which per-type root index file to query via
/// [`RootIndexReader`] or [`crate::query::HeapQuery`].
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
    /// index used internally by [`HeapQuery`]).
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

    /// Human-readable name for display.
    pub fn name(self) -> &'static str {
        match self {
            Self::Unknown => "ROOT_UNKNOWN",
            Self::JniGlobal => "ROOT_JNI_GLOBAL",
            Self::JniLocal => "ROOT_JNI_LOCAL",
            Self::JavaFrame => "ROOT_JAVA_FRAME",
            Self::NativeStack => "ROOT_NATIVE_STACK",
            Self::StickyClass => "ROOT_STICKY_CLASS",
            Self::ThreadBlock => "ROOT_THREAD_BLOCK",
            Self::MonitorUsed => "ROOT_MONITOR_USED",
            Self::ThreadObject => "ROOT_THREAD_OBJ",
        }
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
        buf[0..8].copy_from_slice(&self.object_id.to_le_bytes());
        buf[8..16].copy_from_slice(&self.position.to_le_bytes());
        buf
    }

    pub fn from_bytes(bytes: &[u8; ROOT_INDEX_ENTRY_SIZE]) -> Self {
        let object_id = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        let position = u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]);
        Self {
            object_id,
            position,
        }
    }
}

// ── RootIndexReader ───────────────────────────────────────────────────────────

/// Read-only handle to a single per-type root index file.
#[derive(Copy, Clone)]
pub struct RootIndexReader<'a> {
    data: &'a [u8],
}

impl<'a> RootIndexReader<'a> {
    /// Create a validated reader from a byte slice.
    pub fn from_ref(data: &'a [u8]) -> Result<Self, HprofError> {
        if !data.len().is_multiple_of(ROOT_INDEX_ENTRY_SIZE) {
            return Err(HprofError::InvalidIndexFile);
        }
        Ok(Self { data })
    }

    /// Create a reader from a slice already known to be valid.
    ///
    /// The caller must guarantee that `data.len()` is a multiple of
    /// [`ROOT_INDEX_ENTRY_SIZE`]; checked with `debug_assert`.
    pub(crate) fn from_slice(data: &'a [u8]) -> Self {
        debug_assert!(data.len().is_multiple_of(ROOT_INDEX_ENTRY_SIZE));
        Self { data }
    }

    fn as_slice(&self) -> &[u8] {
        self.data
    }

    /// Total number of entries in this index.
    pub fn len(&self) -> usize {
        self.as_slice().len() / ROOT_INDEX_ENTRY_SIZE
    }

    /// Returns `true` if this index contains no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Binary-search for an entry with the given `object_id`.
    ///
    /// Returns `Some(entry)` if found, `None` if absent.  O(log n).
    pub fn find(&self, object_id: u64) -> Option<RootIndexEntry> {
        let n = self.len();
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.read_object_id_at(mid) < object_id {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo < n && self.read_object_id_at(lo) == object_id {
            Some(self.read_entry_at(lo))
        } else {
            None
        }
    }

    /// Iterate all entries in ascending `object_id` order.
    pub fn iter(&self) -> RootIter<'a> {
        RootIter {
            reader: *self,
            idx: 0,
        }
    }

    fn read_object_id_at(&self, idx: usize) -> u64 {
        let data = self.as_slice();
        let start = idx * ROOT_INDEX_ENTRY_SIZE;
        u64::from_le_bytes([
            data[start],
            data[start + 1],
            data[start + 2],
            data[start + 3],
            data[start + 4],
            data[start + 5],
            data[start + 6],
            data[start + 7],
        ])
    }

    fn read_entry_at(&self, idx: usize) -> RootIndexEntry {
        let data = self.as_slice();
        let start = idx * ROOT_INDEX_ENTRY_SIZE;
        let bytes: [u8; ROOT_INDEX_ENTRY_SIZE] = data[start..start + ROOT_INDEX_ENTRY_SIZE]
            .try_into()
            .unwrap_or([0u8; ROOT_INDEX_ENTRY_SIZE]);
        RootIndexEntry::from_bytes(&bytes)
    }
}

// ── RootIter ──────────────────────────────────────────────────────────────────

/// Iterator over all entries in a [`RootIndexReader`].
///
/// Yields entries in ascending `object_id` order.
pub struct RootIter<'a> {
    reader: RootIndexReader<'a>,
    idx: usize,
}

impl<'a> Iterator for RootIter<'a> {
    type Item = RootIndexEntry;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.reader.len() {
            return None;
        }
        let entry = self.reader.read_entry_at(self.idx);
        self.idx += 1;
        Some(entry)
    }
}

// ── Output path bundle ────────────────────────────────────────────────────────

/// Paths for all nine per-type root index files.
pub struct RootIndexPaths<'a> {
    pub root_unknown: &'a Path,
    pub root_jni_global: &'a Path,
    pub root_jni_local: &'a Path,
    pub root_java_frame: &'a Path,
    pub root_native_stack: &'a Path,
    pub root_sticky_class: &'a Path,
    pub root_thread_block: &'a Path,
    pub root_monitor_used: &'a Path,
    pub root_thread_obj: &'a Path,
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
        assert_eq!(reader.len(), 0);
        assert!(reader.is_empty());
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
    fn gc_root_type_all_coverage() {
        assert_eq!(GcRootType::ALL.len(), 9);
        let indices: Vec<usize> = GcRootType::ALL.iter().map(|t| t.index()).collect();
        assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
    }
}
