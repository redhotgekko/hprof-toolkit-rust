pub mod entry;

pub use entry::{INDEX_ENTRY_SIZE, IndexEntry};

use crate::hprof::{HprofError, HprofFile};
use crate::vfs::MMapWriter;
use std::io::Write;

/// Read every top-level record from `hprof_source` and write a fixed-size binary
/// index to `index_path`.
///
/// Each index entry is [`INDEX_ENTRY_SIZE`] bytes (see [`IndexEntry`]).
/// The index file contains no header — the record count is implicitly
/// `file_size / INDEX_ENTRY_SIZE`, which allows the file to be chunked for
/// concurrent processing by later indexers.
///
/// Returns the number of records indexed.
pub fn index_hprof(
    hprof_source: &[u8],
    index_path: &mut impl MMapWriter,
) -> Result<u64, HprofError> {
    let hprof = HprofFile::from_ref(hprof_source)?;
    let mut writer = index_path.create_writer()?;

    let mut count = 0u64;
    for result in hprof.record_headers() {
        let rec = result?;
        let entry = IndexEntry {
            tag: u8::from(rec.tag),
            body_length: rec.body_length,
            position: rec.position,
        };
        writer.write_all(&entry.to_bytes())?;
        count += 1;
    }

    writer.flush()?;
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hprof::{HprofFile, RecordTag};

    /// Build a minimal valid hprof byte sequence in memory.
    ///
    /// Records:
    ///   1. UTF8 (id=1, "hi") — body = 8-byte id + 2-byte string = 10 bytes
    ///   2. HEAP_DUMP_END     — empty body
    fn minimal_hprof() -> Vec<u8> {
        let mut buf = Vec::new();

        // Header
        buf.extend_from_slice(b"JAVA PROFILE 1.0.2\0");
        buf.extend_from_slice(&8u32.to_be_bytes()); // id_size = 8
        buf.extend_from_slice(&0u64.to_be_bytes()); // timestamp = 0

        // Record 1: UTF8, body = id(8 bytes) + "hi"(2 bytes) = 10 bytes
        buf.push(0x01); // tag UTF8
        buf.extend_from_slice(&0u32.to_be_bytes()); // time_offset
        buf.extend_from_slice(&10u32.to_be_bytes()); // body_length
        buf.extend_from_slice(&1u64.to_be_bytes()); // id
        buf.extend_from_slice(b"hi");

        // Record 2: HEAP_DUMP_END, empty body
        buf.push(0x2C); // tag HEAP_DUMP_END
        buf.extend_from_slice(&0u32.to_be_bytes()); // time_offset
        buf.extend_from_slice(&0u32.to_be_bytes()); // body_length = 0

        buf
    }

    #[test]
    fn index_minimal_hprof() {
        let hprof_data = minimal_hprof();

        let mut index_buf = Vec::new();
        let count = index_hprof(&hprof_data, &mut index_buf).unwrap();

        assert_eq!(count, 2);
        assert_eq!(index_buf.len(), 2 * INDEX_ENTRY_SIZE);

        let e0 = IndexEntry::from_bytes(index_buf[0..16].try_into().unwrap());
        assert_eq!(e0.tag, u8::from(RecordTag::Utf8));
        assert_eq!(e0.body_length, 10);
        // data_offset = len("JAVA PROFILE 1.0.2") + 1 (null) + 4 (id_size) + 8 (ts) = 31
        assert_eq!(e0.position, 31);

        let e1 = IndexEntry::from_bytes(index_buf[16..32].try_into().unwrap());
        assert_eq!(e1.tag, u8::from(RecordTag::HeapDumpEnd));
        assert_eq!(e1.body_length, 0);
        // position = 31 (start) + 9 (header) + 10 (body) = 50
        assert_eq!(e1.position, 50);
    }

    #[test]
    fn record_header_iter_matches_index() {
        let data = minimal_hprof();
        let hprof = HprofFile::from_ref(&data).unwrap();

        let headers: Vec<_> = hprof.record_headers().map(|r| r.unwrap()).collect();

        assert_eq!(headers.len(), 2);
        assert_eq!(headers[0].tag, RecordTag::Utf8);
        assert_eq!(headers[0].body_length, 10);
        assert_eq!(headers[1].tag, RecordTag::HeapDumpEnd);
        assert_eq!(headers[1].body_length, 0);
    }
}
