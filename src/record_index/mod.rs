pub mod entry;

pub use entry::IndexEntry;

use crate::hprof::{HprofError, HprofFile};
use std::io::Write;

/// Read every top-level record from `hprof_source` and write a fixed-size binary
/// index to `out`.
///
/// Each index entry is [`INDEX_ENTRY_SIZE`] bytes (see [`IndexEntry`]).
/// The index file contains no header — the record count is implicitly
/// `file_size / INDEX_ENTRY_SIZE`, which allows the file to be chunked for
/// concurrent processing by later indexers.
///
/// Returns the number of records indexed.
pub fn index_hprof(hprof_source: &[u8], out: &mut dyn Write) -> Result<u64, HprofError> {
    let hprof = HprofFile::from_ref(hprof_source)?;

    let mut count = 0u64;
    for result in hprof.record_headers() {
        let rec = result?;
        let entry = IndexEntry {
            tag: u8::from(rec.tag),
            body_length: rec.body_length,
            position: rec.position,
        };
        out.write_all(&entry.to_bytes())?;
        count += 1;
    }

    out.flush()?;
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hprof::RecordTag;
    use crate::index::Entry;
    use crate::record_index::entry::INDEX_ENTRY_SIZE;
    use crate::test_util::HprofBuilder;

    #[test]
    fn index_minimal_hprof() {
        // UTF8 (id=1, "hi") — body = 8-byte id + 2-byte string = 10 bytes,
        // then an empty HEAP_DUMP_END.
        let hprof_data = HprofBuilder::new(8).utf8(1, "hi").heap_dump_end().build();

        let mut index_buf = Vec::new();
        let count = index_hprof(&hprof_data, &mut index_buf).unwrap();

        assert_eq!(count, 2);
        assert_eq!(index_buf.len(), 2 * INDEX_ENTRY_SIZE);

        let e0 = IndexEntry::from_bytes(&index_buf[0..16]);
        assert_eq!(e0.tag, u8::from(RecordTag::Utf8));
        assert_eq!(e0.body_length, 10);
        // data_offset = len("JAVA PROFILE 1.0.2") + 1 (null) + 4 (id_size) + 8 (ts) = 31
        assert_eq!(e0.position, 31);

        let e1 = IndexEntry::from_bytes(&index_buf[16..32]);
        assert_eq!(e1.tag, u8::from(RecordTag::HeapDumpEnd));
        assert_eq!(e1.body_length, 0);
        // position = 31 (start) + 9 (header) + 10 (body) = 50
        assert_eq!(e1.position, 50);
    }

    #[test]
    fn index_malformed_hprof_input() {
        // Malformed data that should fail HprofFile::from_ref early on (e.g., too short)
        let malformed_data = vec![0xde, 0xad, 0xbe, 0xef];
        assert!(index_hprof(&malformed_data, &mut Vec::new()).is_err());
    }
}
