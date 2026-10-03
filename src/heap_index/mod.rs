pub mod sub_record;

use crate::hprof::{HprofError, HprofFile};
use crate::index::{IndexStore, RecordFile, names};
use crate::record_index::entry::IndexEntry;
use rayon::prelude::*;
use std::io::Write;

/// Parse sub-records from every `HPROF_HEAP_DUMP` / `HPROF_HEAP_DUMP_SEGMENT`
/// record and write a fixed-size binary sub-index entry for each one into
/// `store`.
///
/// Entries are named
/// `heap_index/<hh>/HPROF_HEAP_DUMP_<position>` or
/// `heap_index/<hh>/HPROF_HEAP_DUMP_SEGMENT_<position>` where `<position>`
/// is the hex byte offset of the record in the hprof file and `<hh>` its
/// first two hex digits (keeps directory sizes sane on disk).
///
/// Each entry contains zero or more 24-byte [`SubIndexEntry`] records
/// (no header — count = size / 24).  These are build intermediates: the
/// pipeline concatenates them into the object store and then removes them.
///
/// Returns the total number of sub-records written across all entries.
///
/// # Errors
/// Returns the first error encountered during parallel processing.
pub fn index_heap_dumps(
    hprof_data: &[u8],
    index_data: &[u8],
    store: &dyn IndexStore,
) -> Result<u64, HprofError> {
    let hprof = HprofFile::from_ref(hprof_data)?;
    let id_size = hprof.id_size();

    // Collect only heap dump entries from the record index (bounded by the
    // number of segments, not objects).
    let heap_dump_entries: Vec<IndexEntry> = RecordFile::<IndexEntry>::new(index_data)?
        .iter()
        .filter(|entry| {
            use crate::hprof::RecordTag;
            matches!(
                RecordTag::from(entry.tag),
                RecordTag::HeapDump | RecordTag::HeapDumpSegment
            )
        })
        .collect();

    // Process each heap dump record in parallel.
    let total: u64 = heap_dump_entries
        .par_iter()
        .map(|entry| process_heap_dump(hprof_data, id_size, entry, store))
        .try_reduce(|| 0u64, |a, b| Ok(a + b))?;

    Ok(total)
}

/// Store entry name for the heap dump record at `position`.
pub fn heap_index_entry_name(is_segment: bool, position: u64) -> String {
    let tag_name = if is_segment {
        "HPROF_HEAP_DUMP_SEGMENT"
    } else {
        "HPROF_HEAP_DUMP"
    };
    let hex_pos = format!("{position:x}");
    let subdir = if hex_pos.len() >= 2 {
        hex_pos[..2].to_string()
    } else {
        format!("{position:02x}")
    };
    format!("{}{subdir}/{tag_name}_{hex_pos}", names::HEAP_INDEX_PREFIX)
}

/// Parse all sub-records within a single heap dump record and stream them
/// into a new store entry.
fn process_heap_dump(
    hprof_data: &[u8],
    id_size: u32,
    entry: &IndexEntry,
    store: &dyn IndexStore,
) -> Result<u64, HprofError> {
    use crate::hprof::RecordTag;

    // Record layout: tag(1) + time_offset(4) + body_length(4) = 9-byte header, then body.
    let body_start = entry.position as usize + 9;
    let body_end = body_start + entry.body_length as usize;

    if body_end > hprof_data.len() {
        return Err(HprofError::UnexpectedEof(body_start));
    }
    let body = &hprof_data[body_start..body_end];

    let is_segment = entry.tag == u8::from(RecordTag::HeapDumpSegment);
    let name = heap_index_entry_name(is_segment, entry.position);

    let scanner = sub_record::SubRecordScanner::new(body, entry.position + 9, id_size)?;

    let mut writer = store.create(&name)?;
    let mut count = 0u64;
    for result in scanner {
        let sub_entry = result?;
        writer.write_all(&sub_entry.to_bytes())?;
        count += 1;
    }
    writer.commit()?;

    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heap_index::sub_record::{SUB_INDEX_ENTRY_SIZE, SubIndexEntry};
    use crate::index::MemStore;
    use crate::record_index::index_hprof;
    use crate::test_util::HprofBuilder;

    #[test]
    fn heap_indexer_produces_correct_sub_index() {
        let hprof_data = HprofBuilder::new(8)
            .root_sticky_class(1)
            .root_sticky_class(2)
            .build();

        let mut index_buf = Vec::new();
        let store = MemStore::new();

        index_hprof(&hprof_data, &mut index_buf).unwrap();
        let total = index_heap_dumps(&hprof_data, &index_buf, &store).unwrap();

        assert_eq!(total, 2, "expected 2 sub-records");

        // The HEAP_DUMP_SEGMENT is the first record; its position = data_offset = 31.
        let seg_pos = 31u64;
        let name = heap_index_entry_name(true, seg_pos);
        assert_eq!(name, "heap_index/1f/HPROF_HEAP_DUMP_SEGMENT_1f");
        let src = store.open(&name).expect("sub-index entry not found");
        let bytes = src.as_ref();
        assert_eq!(bytes.len(), 2 * SUB_INDEX_ENTRY_SIZE);

        let e0 = SubIndexEntry::from_bytes(bytes[0..24].try_into().unwrap());
        assert_eq!(e0.tag, sub_record::TAG_ROOT_STICKY_CLASS);
        assert_eq!(e0.object_id, 1);
        // body_start = 31 (record pos) + 9 (record header) = 40
        assert_eq!(e0.position, 40);

        let e1 = SubIndexEntry::from_bytes(bytes[24..48].try_into().unwrap());
        assert_eq!(e1.tag, sub_record::TAG_ROOT_STICKY_CLASS);
        assert_eq!(e1.object_id, 2);
        // sub-record 1 size = 1 (subtag) + 8 (id) = 9; so e1 starts at 40 + 9 = 49
        assert_eq!(e1.position, 49);
    }

    #[test]
    fn entry_names_use_two_hex_digit_subdir() {
        assert_eq!(
            heap_index_entry_name(false, 0x5),
            "heap_index/05/HPROF_HEAP_DUMP_5"
        );
        assert_eq!(
            heap_index_entry_name(true, 0x1234_5678),
            "heap_index/12/HPROF_HEAP_DUMP_SEGMENT_12345678"
        );
    }
}
