//! Name resolution indexes: UTF-8 strings and LOAD_CLASS records.
//!
//! Both indexes store fixed-size binary records sorted by a u64 key so that
//! lookups can be answered with an O(log n) binary search.  The files are
//! built from the record index + the hprof memory map, then sorted in place.
//!
//! ## UTF-8 index (24 bytes per entry)
//! ```text
//!  0..8   u64  name_id        (key, little-endian)
//!  8..16  u64  string_start   (byte position of string data in the hprof file)
//! 16..20  u32  string_length  (byte length of the string data)
//! 20..24  [u8;4] padding
//! ```
//!
//! ## Load-class index (16 bytes per entry)
//! ```text
//!  0..8   u64  class_id       (key, little-endian)
//!  8..16  u64  class_name_id  (UTF-8 name_id for this class)
//! ```

use crate::hprof::record::{RecordTag, read_id};
use crate::hprof::{HprofError, HprofFile};
use crate::index::StoreWriter;
use crate::index::{Entry, RecordFile, read_u32_le, read_u64_le};
use crate::record_index::entry::IndexEntry;

// ── Constants ─────────────────────────────────────────────────────────────────

pub const UTF8_ENTRY_SIZE: usize = 24;
pub const LOAD_CLASS_ENTRY_SIZE: usize = 16;

const RECORD_HEADER_SIZE: usize = 9; // tag(1) + time_offset(4) + body_length(4)

// ── Entries ───────────────────────────────────────────────────────────────────

/// One `HPROF_UTF8` record: where its string bytes live in the hprof file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Utf8Entry {
    pub name_id: u64,
    pub string_start: u64,
    pub string_length: u32,
}

impl Entry for Utf8Entry {
    const SIZE: usize = UTF8_ENTRY_SIZE;
    const KEY_OFFSET: usize = 0;

    fn from_bytes(b: &[u8]) -> Self {
        Self {
            name_id: read_u64_le(b, 0),
            string_start: read_u64_le(b, 8),
            string_length: read_u32_le(b, 16),
        }
    }

    fn write_to(&self, out: &mut [u8]) {
        out[0..8].copy_from_slice(&self.name_id.to_le_bytes());
        out[8..16].copy_from_slice(&self.string_start.to_le_bytes());
        out[16..20].copy_from_slice(&self.string_length.to_le_bytes());
        out[20..24].fill(0);
    }

    fn key(&self) -> u64 {
        self.name_id
    }
}

/// One `HPROF_LOAD_CLASS` record: class object id → class name id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadClassEntry {
    pub class_id: u64,
    pub class_name_id: u64,
}

impl Entry for LoadClassEntry {
    const SIZE: usize = LOAD_CLASS_ENTRY_SIZE;
    const KEY_OFFSET: usize = 0;

    fn from_bytes(b: &[u8]) -> Self {
        Self {
            class_id: read_u64_le(b, 0),
            class_name_id: read_u64_le(b, 8),
        }
    }

    fn write_to(&self, out: &mut [u8]) {
        out[0..8].copy_from_slice(&self.class_id.to_le_bytes());
        out[8..16].copy_from_slice(&self.class_name_id.to_le_bytes());
    }

    fn key(&self) -> u64 {
        self.class_id
    }
}

// ── Utf8IndexReader ───────────────────────────────────────────────────────────

/// Reader for the UTF-8 name index produced by [`build_utf8_index`].
#[derive(Clone, Copy)]
pub struct Utf8IndexReader<'a> {
    file: RecordFile<'a, Utf8Entry>,
}

impl<'a> Utf8IndexReader<'a> {
    pub fn from_ref(data: &'a [u8]) -> Result<Self, HprofError> {
        Ok(Self {
            file: RecordFile::new(data)?,
        })
    }

    /// Look up the string for `name_id`, reading its bytes from `hprof`.
    ///
    /// Returns `None` if `name_id` is not in the index.
    pub fn lookup(&self, hprof: &HprofFile, name_id: u64) -> Result<Option<String>, HprofError> {
        let Some(entry) = self.file.find(name_id) else {
            return Ok(None);
        };
        let start = entry.string_start as usize;
        let end = start + entry.string_length as usize;
        let hprof_data = hprof.data();
        if end > hprof_data.len() {
            return Err(HprofError::UnexpectedEof(start));
        }
        // Java uses modified UTF-8; for most identifiers this is valid UTF-8.
        Ok(Some(
            String::from_utf8_lossy(&hprof_data[start..end]).into_owned(),
        ))
    }
}

// ── LoadClassReader ───────────────────────────────────────────────────────────

/// Reader for the load-class index produced by [`build_load_class_index`].
#[derive(Clone, Copy)]
pub struct LoadClassReader<'a> {
    file: RecordFile<'a, LoadClassEntry>,
}

impl<'a> LoadClassReader<'a> {
    pub fn from_ref(bytes: &'a [u8]) -> Result<Self, HprofError> {
        Ok(Self {
            file: RecordFile::new(bytes)?,
        })
    }

    /// Return the `class_name_id` for the given `class_id`, or `None` if not
    /// found.
    pub fn find_class_name_id(&self, class_id: u64) -> Option<u64> {
        self.file.find(class_id).map(|e| e.class_name_id)
    }
}

// ── Index builders ────────────────────────────────────────────────────────────

/// Build a UTF-8 name index from the record index.
///
/// Writes one entry per `UTF8` record in the record index, then sorts the
/// output by `name_id`.  Returns the number of UTF-8 records indexed.
pub fn build_utf8_index(
    hprof: &HprofFile,
    record_index_data: impl AsRef<[u8]>,
    out: &mut dyn StoreWriter,
) -> Result<u64, HprofError> {
    let id_size = hprof.id_size() as usize;
    let hprof_data = hprof.data();
    let utf8_tag = u8::from(RecordTag::Utf8);

    let mut count = 0u64;
    let mut buf = [0u8; UTF8_ENTRY_SIZE];
    for entry in RecordFile::<IndexEntry>::new(record_index_data.as_ref())?.iter() {
        if entry.tag != utf8_tag {
            continue;
        }

        let body_length = entry.body_length as usize;
        if body_length < id_size {
            continue; // malformed record, skip
        }

        let body_start = entry.position as usize + RECORD_HEADER_SIZE;
        if body_start + body_length > hprof_data.len() {
            return Err(HprofError::UnexpectedEof(body_start));
        }

        // name_id is the first id_size bytes of the body (big-endian).
        Utf8Entry {
            name_id: read_id(hprof_data, body_start, id_size)?,
            string_start: (body_start + id_size) as u64,
            string_length: (body_length - id_size) as u32,
        }
        .write_to(&mut buf);
        out.write_all(&buf)?;
        count += 1;
    }
    out.flush()?;

    if count > 1 {
        crate::sort::parallel_introsort(out.as_mut_bytes()?, UTF8_ENTRY_SIZE, 0);
    }

    Ok(count)
}

/// Build a load-class index from the record index.
///
/// Writes one entry per `LoadClass` record, sorted by `class_id`.
/// Returns the number of load-class records indexed.
pub fn build_load_class_index(
    hprof: &HprofFile,
    record_index_data: &[u8],
    out: &mut dyn StoreWriter,
) -> Result<u64, HprofError> {
    let id_size = hprof.id_size() as usize;
    let hprof_data = hprof.data();
    let lc_tag = u8::from(RecordTag::LoadClass);
    let mut count = 0u64;

    let mut buf = [0u8; LOAD_CLASS_ENTRY_SIZE];
    for entry in RecordFile::<IndexEntry>::new(record_index_data)?.iter() {
        if entry.tag != lc_tag {
            continue;
        }

        // LOAD_CLASS body: class_serial(4) + class_id(id) + stack_serial(4) + class_name_id(id)
        let body_start = entry.position as usize + RECORD_HEADER_SIZE;
        let min_body = 4 + id_size + 4 + id_size;
        if body_start + min_body > hprof_data.len() {
            return Err(HprofError::UnexpectedEof(body_start));
        }

        LoadClassEntry {
            class_id: read_id(hprof_data, body_start + 4, id_size)?,
            class_name_id: read_id(hprof_data, body_start + 8 + id_size, id_size)?,
        }
        .write_to(&mut buf);
        out.write_all(&buf)?;
        count += 1;
    }
    out.flush()?;

    if count > 1 {
        crate::sort::parallel_introsort(out.as_mut_bytes()?, LOAD_CLASS_ENTRY_SIZE, 0);
    }
    Ok(count)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{HprofBuilder, built_bytes};

    /// UTF8 (1, "hello"), (2, "java/lang/String"); LOAD_CLASS class 0x100 → name 2.
    fn minimal_hprof() -> Vec<u8> {
        HprofBuilder::new(8)
            .utf8(1, "hello")
            .utf8(2, "java/lang/String")
            .load_class(1, 0x100, 2)
            .heap_dump_end()
            .build()
    }

    fn record_index_for(hprof_data: &[u8]) -> Vec<u8> {
        use crate::record_index::index_hprof;
        let mut idx_buf = Vec::new();
        index_hprof(hprof_data, &mut idx_buf).unwrap();
        idx_buf
    }

    #[test]
    fn utf8_index_lookup() {
        let hprof_data = minimal_hprof();
        let record_index_bytes = record_index_for(&hprof_data);
        let hprof = crate::hprof::HprofFile::from_ref(&hprof_data).unwrap();

        let (count, data) = built_bytes(|w| build_utf8_index(&hprof, &record_index_bytes, w));
        assert_eq!(count, 2);

        let reader = Utf8IndexReader::from_ref(&data).unwrap();
        assert_eq!(reader.file.len(), 2);
        assert_eq!(reader.lookup(&hprof, 1).unwrap(), Some("hello".to_string()));
        assert_eq!(
            reader.lookup(&hprof, 2).unwrap(),
            Some("java/lang/String".to_string())
        );
        assert_eq!(reader.lookup(&hprof, 999).unwrap(), None);
    }

    #[test]
    fn load_class_index_lookup() {
        let hprof_data = minimal_hprof();
        let record_index_bytes = record_index_for(&hprof_data);
        let hprof = crate::hprof::HprofFile::from_ref(&hprof_data).unwrap();

        let (count, lc_buf) =
            built_bytes(|w| build_load_class_index(&hprof, &record_index_bytes, w));
        assert_eq!(count, 1);

        let reader = LoadClassReader::from_ref(&lc_buf).unwrap();
        assert_eq!(reader.find_class_name_id(0x100), Some(2));
        assert_eq!(reader.find_class_name_id(0x200), None);
        assert_eq!(
            reader
                .file
                .iter()
                .map(|e| (e.class_id, e.class_name_id))
                .collect::<Vec<_>>(),
            vec![(0x100, 2)]
        );
    }

    #[test]
    fn utf8_index_is_sorted_by_name_id() {
        let hprof_data = HprofBuilder::new(8)
            .utf8(30, "c")
            .utf8(10, "a")
            .utf8(20, "b")
            .build();
        let record_index_bytes = record_index_for(&hprof_data);
        let hprof = crate::hprof::HprofFile::from_ref(&hprof_data).unwrap();
        let (_, data) = built_bytes(|w| build_utf8_index(&hprof, &record_index_bytes, w));
        let reader = Utf8IndexReader::from_ref(&data).unwrap();
        let ids: Vec<u64> = reader.file.iter().map(|e| e.name_id).collect();
        assert_eq!(ids, vec![10, 20, 30]);
        assert_eq!(reader.lookup(&hprof, 20).unwrap().as_deref(), Some("b"));
    }

    #[test]
    fn utf8_entry_padding_is_zero() {
        let mut buf = [0xFFu8; UTF8_ENTRY_SIZE];
        Utf8Entry {
            name_id: 1,
            string_start: 2,
            string_length: 3,
        }
        .write_to(&mut buf);
        assert_eq!(&buf[20..24], &[0, 0, 0, 0]);
        assert_eq!(Utf8Entry::from_bytes(&buf).string_length, 3);
    }
}
