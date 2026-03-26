pub mod sub_record;

pub use sub_record::{SUB_INDEX_ENTRY_SIZE, SubIndexEntry};

use crate::hprof::{HprofError, HprofFile};
use crate::record_index::entry::{INDEX_ENTRY_SIZE, IndexEntry};
use crate::vfs::{MMapReader, SubIndexDir};
use rayon::prelude::*;
use std::io::Write;

/// Parse sub-records from every `HPROF_HEAP_DUMP` / `HPROF_HEAP_DUMP_SEGMENT`
/// record and write a fixed-size binary sub-index for each one.
///
/// Output files are placed in `output` and named:
///   `HPROF_HEAP_DUMP_<position>` or `HPROF_HEAP_DUMP_SEGMENT_<position>`
/// where `<position>` is the hex byte offset of the record in the hprof file.
///
/// Each output file contains zero or more 24-byte [`SubIndexEntry`] records
/// (no file header — count = file_size / 24).
///
/// Returns the total number of sub-records written across all output files.
///
/// # Errors
/// Returns the first error encountered during parallel processing.
pub fn index_heap_dumps(
    hprof_source: &impl MMapReader,
    record_index_source: &impl MMapReader,
    output: &SubIndexDir,
) -> Result<u64, HprofError> {
    let hprof = HprofFile::from_source(hprof_source.open_mmap()?)?;
    let id_size = hprof.header.id_size;
    let hprof_data: &[u8] = hprof.data();

    let index_source = record_index_source.open_mmap()?;
    let index_data: &[u8] = index_source.as_ref();
    if !index_data.len().is_multiple_of(INDEX_ENTRY_SIZE) {
        return Err(HprofError::InvalidIndexFile);
    }

    // Collect only heap dump entries from the record index.
    let heap_dump_entries: Vec<IndexEntry> = index_data
        .chunks_exact(INDEX_ENTRY_SIZE)
        .filter_map(|chunk| {
            // chunks_exact guarantees len == INDEX_ENTRY_SIZE; this always succeeds.
            let arr: &[u8; INDEX_ENTRY_SIZE] = chunk.try_into().ok()?;
            let entry = IndexEntry::from_bytes(arr);
            use crate::hprof::RecordTag;
            let tag = RecordTag::from(entry.tag);
            if matches!(tag, RecordTag::HeapDump | RecordTag::HeapDumpSegment) {
                Some(entry)
            } else {
                None
            }
        })
        .collect();

    // Process each heap dump record in parallel.
    let total: u64 = heap_dump_entries
        .par_iter()
        .map(|entry| process_heap_dump(hprof_data, id_size, entry, output))
        .try_reduce(|| 0u64, |a, b| Ok(a + b))?;

    Ok(total)
}

/// Parse all sub-records within a single heap dump record and write them to
/// a new file in `output`.
fn process_heap_dump(
    hprof_data: &[u8],
    id_size: u32,
    entry: &IndexEntry,
    output: &SubIndexDir,
) -> Result<u64, HprofError> {
    use crate::hprof::RecordTag;

    // Record layout: tag(1) + time_offset(4) + body_length(4) = 9-byte header, then body.
    let body_start = entry.position as usize + 9;
    let body_end = body_start + entry.body_length as usize;

    if body_end > hprof_data.len() {
        return Err(HprofError::UnexpectedEof(body_start));
    }
    let body = &hprof_data[body_start..body_end];

    let tag_name = if entry.tag == u8::from(RecordTag::HeapDumpSegment) {
        "HPROF_HEAP_DUMP_SEGMENT"
    } else {
        "HPROF_HEAP_DUMP"
    };
    let hex_pos = format!("{:x}", entry.position);
    let subdir_name = if hex_pos.len() >= 2 {
        hex_pos[..2].to_string()
    } else {
        format!("{:02x}", entry.position)
    };
    let file_name = format!("{}_{:x}", tag_name, entry.position);
    let rel_path = format!("{}/{}", subdir_name, file_name);

    let scanner = sub_record::SubRecordScanner::new(body, entry.position + 9, id_size)?;

    let mut buf: Vec<u8> = Vec::new();
    let mut count = 0u64;
    for result in scanner {
        let sub_entry = result?;
        buf.write_all(&sub_entry.to_bytes())?;
        count += 1;
    }

    output.write_sub_file(&rel_path, buf)?;

    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heap_index::sub_record::{
        SUB_INDEX_ENTRY_SIZE, SubIndexEntry, TAG_ROOT_STICKY_CLASS,
    };
    use crate::record_index::index_hprof;
    use crate::vfs::SubIndexDir;

    /// Minimal hprof with one HEAP_DUMP_SEGMENT containing two ROOT_STICKY_CLASS sub-records.
    fn minimal_hprof_with_heap_dump() -> Vec<u8> {
        let id_size: u32 = 8;
        let mut buf = Vec::new();

        // Header
        buf.extend_from_slice(b"JAVA PROFILE 1.0.2\0");
        buf.extend_from_slice(&id_size.to_be_bytes());
        buf.extend_from_slice(&0u64.to_be_bytes()); // timestamp

        // Sub-record 1: ROOT_STICKY_CLASS, class_id = 1
        let mut body = Vec::new();
        body.push(TAG_ROOT_STICKY_CLASS);
        body.extend_from_slice(&1u64.to_be_bytes());
        // Sub-record 2: ROOT_STICKY_CLASS, class_id = 2
        body.push(TAG_ROOT_STICKY_CLASS);
        body.extend_from_slice(&2u64.to_be_bytes());

        // HEAP_DUMP_SEGMENT record
        buf.push(0x1C); // tag
        buf.extend_from_slice(&0u32.to_be_bytes()); // time_offset
        buf.extend_from_slice(&(body.len() as u32).to_be_bytes()); // body_length
        buf.extend_from_slice(&body);

        // HEAP_DUMP_END record
        buf.push(0x2C);
        buf.extend_from_slice(&0u32.to_be_bytes());
        buf.extend_from_slice(&0u32.to_be_bytes());

        buf
    }

    #[test]
    fn heap_indexer_produces_correct_sub_index() {
        let hprof_data = minimal_hprof_with_heap_dump();

        let mut index_buf = Vec::new();
        let out_dir = SubIndexDir::mem();

        index_hprof(&hprof_data, &mut index_buf).unwrap();
        let total = index_heap_dumps(&hprof_data, &index_buf, &out_dir).unwrap();

        assert_eq!(total, 2, "expected 2 sub-records");

        // The HEAP_DUMP_SEGMENT is the first record; its position = data_offset = 31.
        let seg_pos = 31u64;
        let out_file_name = format!("HPROF_HEAP_DUMP_SEGMENT_{:x}", seg_pos);
        let rel_path = format!("1f/{}", out_file_name);
        let bytes = out_dir
            .get_file(&rel_path)
            .expect("sub-index file not found");
        assert_eq!(bytes.len(), 2 * SUB_INDEX_ENTRY_SIZE);

        let e0 = SubIndexEntry::from_bytes(bytes[0..24].try_into().unwrap());
        assert_eq!(e0.tag, TAG_ROOT_STICKY_CLASS);
        assert_eq!(e0.object_id, 1);
        // body_start = 31 (record pos) + 9 (record header) = 40
        assert_eq!(e0.position, 40);

        let e1 = SubIndexEntry::from_bytes(bytes[24..48].try_into().unwrap());
        assert_eq!(e1.tag, TAG_ROOT_STICKY_CLASS);
        assert_eq!(e1.object_id, 2);
        // sub-record 1 size = 1 (subtag) + 8 (id) = 9; so e1 starts at 40 + 9 = 49
        assert_eq!(e1.position, 49);
    }
}
