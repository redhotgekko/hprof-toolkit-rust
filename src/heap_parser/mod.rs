pub mod record;

pub(crate) use record::parse_sub_record;
pub use record::{
    ClassDump, CpEntry, CpEntryIter, FieldValue, InstanceDump, InstanceFieldDescriptor,
    InstanceFieldIter, ObjArrayDump, ObjArrayElemIter, PrimArrayDump, RootJavaFrame, RootJniGlobal,
    RootJniLocal, RootMonitorUsed, RootNativeStack, RootStickyClass, RootThreadBlock,
    RootThreadObj, RootUnknown, StaticField, StaticFieldIter, SubRecord,
};

use crate::heap_index::sub_record::{
    SubIndexEntry, TAG_CLASS_DUMP, TAG_INSTANCE_DUMP, TAG_OBJ_ARRAY_DUMP, TAG_PRIM_ARRAY_DUMP,
};
use crate::hprof::HprofError;
use crate::index::{RecordFile, RecordIter};

// ── SubIndexReader ────────────────────────────────────────────────────────────

/// Iterator over [`SubIndexEntry`] values in a [`SubIndexReader`].
pub(crate) type SubIndexIter<'a> = RecordIter<'a, SubIndexEntry>;

/// Reader for a heap sub-record index: either one per-segment intermediate
/// produced by [`crate::heap_index::index_heap_dumps`], or the sorted object
/// store produced by [`crate::object_store::combine_sort_and_split`].
#[derive(Clone, Copy)]
pub(crate) struct SubIndexReader<'a> {
    file: RecordFile<'a, SubIndexEntry>,
}

impl<'a> SubIndexReader<'a> {
    pub fn from_ref(bytes: &'a [u8]) -> Result<Self, HprofError> {
        Ok(Self {
            file: RecordFile::new(bytes)?,
        })
    }

    /// Iterate over all [`SubIndexEntry`] values in file order.
    ///
    /// The returned iterator borrows from the underlying data slice (`'a`),
    /// so it can outlive the `SubIndexReader` struct itself.
    pub fn iter(&self) -> SubIndexIter<'a> {
        self.file.iter()
    }

    /// Return the [`SubIndexEntry`] at position `i`, or `None` when
    /// `i >= len()`.
    pub fn entry_at(&self, i: usize) -> Option<SubIndexEntry> {
        self.file.get(i)
    }

    /// Binary-search for the entry describing the object `target`.
    ///
    /// An object that is also a GC root has several entries with the same
    /// `object_id` (one per root record plus the object record).  This
    /// returns the **object record** (class dump, instance dump or array
    /// dump) when one exists, and only falls back to a root record when the
    /// id has no object record at all — so the answer does not depend on the
    /// order in which equal ids were sorted.
    ///
    /// **Requires:** sorted ascending by `object_id` (the object store).
    pub fn find_by_object_id(&self, target: u64) -> Option<SubIndexEntry> {
        self.find_by_object_id_and_tag(target, None)
    }

    /// Binary-search for the entry matching `object_id == target` with the
    /// given `tag`.  With `tag == None` behaves like [`Self::find_by_object_id`].
    ///
    /// **Requires:** sorted ascending by `object_id`.
    pub fn find_by_object_id_and_tag(&self, target: u64, tag: Option<u8>) -> Option<SubIndexEntry> {
        let mut fallback: Option<SubIndexEntry> = None;
        for entry in self.file.range(target) {
            match tag {
                Some(t) if t == entry.tag => return Some(entry),
                Some(_) => {}
                None if is_object_tag(entry.tag) => return Some(entry),
                None => {
                    fallback.get_or_insert(entry);
                }
            }
        }
        fallback
    }
}

/// `true` for the four tags that describe a heap object rather than a GC root.
pub(crate) fn is_object_tag(tag: u8) -> bool {
    matches!(
        tag,
        TAG_CLASS_DUMP | TAG_INSTANCE_DUMP | TAG_OBJ_ARRAY_DUMP | TAG_PRIM_ARRAY_DUMP
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heap_index::index_heap_dumps;
    use crate::heap_index::sub_record::TAG_ROOT_STICKY_CLASS;
    use crate::index::{IndexStore, MemStore, names};
    use crate::record_index::index_hprof;
    use crate::test_util::HprofBuilder;

    fn two_sticky_classes() -> Vec<u8> {
        HprofBuilder::new(8)
            .root_sticky_class(1)
            .root_sticky_class(2)
            .build()
    }

    /// Index the heap dumps in memory and return every per-segment entry.
    fn sub_index_sources(hprof_data: &[u8]) -> Vec<crate::index::ByteSource> {
        let mut idx_buf = Vec::new();
        let store = MemStore::new();
        index_hprof(hprof_data, &mut idx_buf).unwrap();
        index_heap_dumps(hprof_data, &idx_buf, &store).unwrap();
        store
            .list(names::HEAP_INDEX_PREFIX)
            .unwrap()
            .iter()
            .map(|n| store.open(n).unwrap())
            .collect()
    }

    #[test]
    fn sub_index_reader_iter() {
        let hprof_data = two_sticky_classes();
        let sources = sub_index_sources(&hprof_data);
        assert_eq!(sources.len(), 1);

        let reader = SubIndexReader::from_ref(sources[0].as_ref()).unwrap();
        assert_eq!(reader.iter().count(), 2);

        let entries: Vec<_> = reader.iter().collect();
        assert_eq!(entries[0].tag, TAG_ROOT_STICKY_CLASS);
        assert_eq!(entries[0].object_id, 1);
        assert_eq!(entries[1].tag, TAG_ROOT_STICKY_CLASS);
        assert_eq!(entries[1].object_id, 2);
    }

    #[test]
    fn parse_sub_record_via_reader() {
        let hprof_data = two_sticky_classes();
        let sources = sub_index_sources(&hprof_data);
        let hprof = crate::hprof::HprofFile::from_ref(&hprof_data).unwrap();
        let reader = SubIndexReader::from_ref(sources[0].as_ref()).unwrap();

        let class_ids: Vec<u64> = reader
            .iter()
            .map(|entry| {
                let rec = parse_sub_record(&hprof, &entry).unwrap();
                if let SubRecord::RootStickyClass(r) = rec {
                    r.class_id
                } else {
                    0
                }
            })
            .collect();

        assert_eq!(class_ids, vec![1, 2]);
    }

    #[test]
    fn find_by_object_id_and_tag_scans_duplicates() {
        // Two entries share object id 5 with different tags; sorted input.
        let mut data = Vec::new();
        for (tag, id) in [(0x05u8, 5u64), (0x21, 5), (0x21, 9)] {
            data.extend_from_slice(
                &SubIndexEntry {
                    tag,
                    object_id: id,
                    position: 0,
                }
                .to_bytes(),
            );
        }
        let reader = SubIndexReader::from_ref(&data).unwrap();
        // The object record (0x21) wins over the GC-root record (0x05) that
        // shares its id, regardless of sort order.
        assert_eq!(reader.find_by_object_id(5).unwrap().tag, 0x21);
        assert_eq!(
            reader.find_by_object_id_and_tag(5, Some(0x05)).unwrap().tag,
            0x05
        );
        assert!(reader.find_by_object_id_and_tag(5, Some(0x22)).is_none());
        assert!(reader.find_by_object_id(7).is_none());
        assert_eq!(reader.entry_at(2).unwrap().object_id, 9);
        assert!(reader.entry_at(3).is_none());
    }

    #[test]
    fn root_only_id_falls_back_to_root_record() {
        let data = SubIndexEntry {
            tag: 0x05,
            object_id: 7,
            position: 1,
        }
        .to_bytes();
        let reader = SubIndexReader::from_ref(&data).unwrap();
        assert_eq!(reader.find_by_object_id(7).unwrap().tag, 0x05);
    }
}
