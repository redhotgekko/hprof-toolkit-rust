//! Name resolution indexes: UTF-8 strings and LOAD_CLASS records.
//!
//! Both indexes store fixed-size binary records sorted by a u64 key so that
//! lookups can be answered with an O(log n) binary search.  The files are
//! built from the Phase 1 index + the hprof memory map, then sorted in-place
//! via mmap-based heapsort.
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

use crate::hprof::record::RecordTag;
use crate::hprof::{HprofError, HprofFile};
use crate::record_index::entry::{INDEX_ENTRY_SIZE, IndexEntry};
use crate::vfs::MMapWriter;
use std::io::Write;

// ── Constants ─────────────────────────────────────────────────────────────────

pub const UTF8_ENTRY_SIZE: usize = 24;
pub const LOAD_CLASS_ENTRY_SIZE: usize = 16;

const RECORD_HEADER_SIZE: usize = 9; // tag(1) + time_offset(4) + body_length(4)

// ── Utf8IndexReader ───────────────────────────────────────────────────────────

/// Memory-mapped reader for the UTF-8 name index produced by
/// [`build_utf8_index`].
pub struct Utf8IndexReader<'a> {
    mmap: &'a [u8],
}

impl<'a> Utf8IndexReader<'a> {
    pub fn from_ref(mmap: &'a [u8]) -> Result<Self, HprofError> {
        if !mmap.len().is_multiple_of(UTF8_ENTRY_SIZE) {
            return Err(HprofError::InvalidIndexFile);
        }
        Ok(Self { mmap })
    }

    /// Look up the string for `name_id`, reading its bytes from `hprof`.
    ///
    /// Returns `None` if `name_id` is not in the index.
    pub fn lookup(&self, hprof: &HprofFile, name_id: u64) -> Result<Option<String>, HprofError> {
        let data = self.mmap;
        let n = data.len() / UTF8_ENTRY_SIZE;

        let lo = leftmost_search(data, n, UTF8_ENTRY_SIZE, 0, name_id);
        if lo >= n || read_u64_le(data, lo * UTF8_ENTRY_SIZE) != name_id {
            return Ok(None);
        }

        let base = lo * UTF8_ENTRY_SIZE;
        let string_start = read_u64_le(data, base + 8) as usize;
        let string_length = read_u32_le(data, base + 16) as usize;

        let hprof_data = hprof.data();
        if string_start + string_length > hprof_data.len() {
            return Err(HprofError::UnexpectedEof(string_start));
        }

        let bytes = &hprof_data[string_start..string_start + string_length];
        // Java uses modified UTF-8; for most identifiers this is valid UTF-8.
        Ok(Some(String::from_utf8_lossy(bytes).into_owned()))
    }
}

// ── LoadClassReader ───────────────────────────────────────────────────────────

/// Memory-mapped reader for the load-class index produced by
/// [`build_load_class_index`].
pub struct LoadClassReader<'a> {
    data: &'a [u8],
}

impl<'a> LoadClassReader<'a> {
    pub fn from_ref(bytes: &'a [u8]) -> Result<Self, HprofError> {
        if !bytes.len().is_multiple_of(LOAD_CLASS_ENTRY_SIZE) {
            return Err(HprofError::InvalidIndexFile);
        }
        Ok(Self { data: bytes })
    }

    /// Return the `class_name_id` for the given `class_id`, or `None` if not
    /// found.
    pub fn find_class_name_id(&self, class_id: u64) -> Option<u64> {
        let data = self.data;
        let n = data.len() / LOAD_CLASS_ENTRY_SIZE;

        let lo = leftmost_search(data, n, LOAD_CLASS_ENTRY_SIZE, 0, class_id);
        if lo < n && read_u64_le(data, lo * LOAD_CLASS_ENTRY_SIZE) == class_id {
            Some(read_u64_le(data, lo * LOAD_CLASS_ENTRY_SIZE + 8))
        } else {
            None
        }
    }

    /// Iterate all `(class_id, class_name_id)` pairs in the index.
    ///
    /// Entries are yielded in ascending `class_id` order (the index sort
    /// order).  Used for name-based class lookup.
    pub fn iter(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        let data = self.data;
        let n = data.len() / LOAD_CLASS_ENTRY_SIZE;
        (0..n).map(move |i| {
            let base = i * LOAD_CLASS_ENTRY_SIZE;
            let class_id = read_u64_le(data, base);
            let name_id = read_u64_le(data, base + 8);
            (class_id, name_id)
        })
    }
}

// ── Index builders ────────────────────────────────────────────────────────────

/// Build a UTF-8 name index from the record index.
///
/// Writes one entry per `UTF8` record in the record index, then sorts the
/// output file by `name_id`.  Returns the number of UTF-8 records indexed.
pub fn build_utf8_index(
    hprof: &HprofFile,
    record_index_data: impl AsRef<[u8]>,
    out: &mut impl MMapWriter,
) -> Result<u64, HprofError> {
    let id_size = hprof.header.id_size as usize;
    let hprof_data = hprof.data();
    let utf8_tag = u8::from(RecordTag::Utf8);

    let mut count = 0u64;

    let mut writer = out.create_writer()?;
    let n = record_index_data.as_ref().len() / INDEX_ENTRY_SIZE;
    for i in 0..n {
        let start = i * INDEX_ENTRY_SIZE;
        let bytes: [u8; INDEX_ENTRY_SIZE] = record_index_data.as_ref()
            [start..start + INDEX_ENTRY_SIZE]
            .try_into()
            .map_err(|_| HprofError::InvalidIndexFile)?;
        let entry = IndexEntry::from_bytes(&bytes);

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
        let name_id = read_id_be(hprof_data, body_start, id_size)?;
        let string_start = (body_start + id_size) as u64;
        let string_length = (body_length - id_size) as u32;

        writer.write_all(&name_id.to_le_bytes())?;
        writer.write_all(&string_start.to_le_bytes())?;
        writer.write_all(&string_length.to_le_bytes())?;
        writer.write_all(&[0u8; 4])?; // padding
        count += 1;
    }
    writer.flush()?;
    drop(writer);

    if count > 1 {
        let mut map = out.create_mut_mmap()?;
        sort_file(&mut map, UTF8_ENTRY_SIZE, 0);
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
    out: &mut impl MMapWriter,
) -> Result<u64, HprofError> {
    let id_size = hprof.header.id_size as usize;
    let hprof_data = hprof.data();
    let lc_tag = u8::from(RecordTag::LoadClass);
    let mut count = 0u64;

    let mut writer = out.create_writer()?;

    let n = record_index_data.len() / INDEX_ENTRY_SIZE;
    for i in 0..n {
        let start = i * INDEX_ENTRY_SIZE;
        let bytes: [u8; INDEX_ENTRY_SIZE] = record_index_data[start..start + INDEX_ENTRY_SIZE]
            .try_into()
            .map_err(|_| HprofError::InvalidIndexFile)?;
        let entry = IndexEntry::from_bytes(&bytes);

        if entry.tag != lc_tag {
            continue;
        }

        // LOAD_CLASS body: class_serial(4) + class_id(id) + stack_serial(4) + class_name_id(id)
        let body_start = entry.position as usize + RECORD_HEADER_SIZE;
        let min_body = 4 + id_size + 4 + id_size;
        if body_start + min_body > hprof_data.len() {
            return Err(HprofError::UnexpectedEof(body_start));
        }

        let class_id = read_id_be(hprof_data, body_start + 4, id_size)?;
        let class_name_id = read_id_be(hprof_data, body_start + 8 + id_size, id_size)?;

        writer.write_all(&class_id.to_le_bytes())?;
        writer.write_all(&class_name_id.to_le_bytes())?;
        count += 1;
    }
    writer.flush()?;

    drop(writer);

    if count > 1 {
        let mut map = out.create_mut_mmap()?;
        sort_file(&mut map, LOAD_CLASS_ENTRY_SIZE, 0);
    }
    Ok(count)
}

// ── Sorting helpers ───────────────────────────────────────────────────────────

/// Sort a fixed-size record file in-place by a u64 key at `key_offset`.
fn sort_file(array: &mut impl AsMut<[u8]>, record_size: usize, key_offset: usize) {
    let mmap = array.as_mut();
    let n = mmap.len() / record_size;
    heapsort(mmap, n, record_size, key_offset);
}

fn heapsort(data: &mut [u8], n: usize, rec: usize, key: usize) {
    if n < 2 {
        return;
    }
    let mut i = n / 2;
    while i > 0 {
        i -= 1;
        sift_down(data, i, n, rec, key);
    }
    let mut end = n;
    while end > 1 {
        end -= 1;
        swap_recs(data, 0, end, rec);
        sift_down(data, 0, end, rec, key);
    }
}

fn sift_down(data: &mut [u8], mut root: usize, n: usize, rec: usize, key: usize) {
    loop {
        let left = 2 * root + 1;
        if left >= n {
            break;
        }
        let right = left + 1;
        let largest = if right < n
            && read_u64_le(data, right * rec + key) > read_u64_le(data, left * rec + key)
        {
            right
        } else {
            left
        };
        if read_u64_le(data, largest * rec + key) <= read_u64_le(data, root * rec + key) {
            break;
        }
        swap_recs(data, root, largest, rec);
        root = largest;
    }
}

fn swap_recs(data: &mut [u8], i: usize, j: usize, rec: usize) {
    if i == j {
        return;
    }
    let (a, b) = if i < j { (i, j) } else { (j, i) };
    let (left, right) = data.split_at_mut(b * rec);
    left[a * rec..a * rec + rec].swap_with_slice(&mut right[..rec]);
}

// ── Common binary search ──────────────────────────────────────────────────────

/// Leftmost binary search: return first index where key >= target.
pub fn leftmost_search(
    data: &[u8],
    n: usize,
    rec_size: usize,
    key_off: usize,
    target: u64,
) -> usize {
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if read_u64_le(data, mid * rec_size + key_off) < target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

// ── Byte helpers ──────────────────────────────────────────────────────────────

pub fn read_u64_le(data: &[u8], off: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&data[off..off + 8]);
    u64::from_le_bytes(b)
}

fn read_u32_le(data: &[u8], off: usize) -> u32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&data[off..off + 4]);
    u32::from_le_bytes(b)
}

fn read_id_be(data: &[u8], off: usize, id_size: usize) -> Result<u64, HprofError> {
    match id_size {
        4 => {
            if off + 4 > data.len() {
                return Err(HprofError::UnexpectedEof(off));
            }
            Ok(u32::from_be_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]) as u64)
        }
        8 => {
            if off + 8 > data.len() {
                return Err(HprofError::UnexpectedEof(off));
            }
            Ok(u64::from_be_bytes([
                data[off],
                data[off + 1],
                data[off + 2],
                data[off + 3],
                data[off + 4],
                data[off + 5],
                data[off + 6],
                data[off + 7],
            ]))
        }
        _ => Err(HprofError::InvalidIdSize(id_size as u32)),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal hprof with UTF8 + LOAD_CLASS + HEAP_DUMP_END records.
    ///
    /// id_size = 8.
    /// UTF8 entries: (id=1, "hello"), (id=2, "java/lang/String")
    /// LOAD_CLASS: class_serial=1, class_id=0x100, stack_serial=0, name_id=2
    fn minimal_hprof() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"JAVA PROFILE 1.0.2\0");
        buf.extend_from_slice(&8u32.to_be_bytes()); // id_size
        buf.extend_from_slice(&0u64.to_be_bytes()); // timestamp

        // UTF8 record 1: id=1, "hello" (5 bytes) → body = 8+5 = 13
        buf.push(0x01);
        buf.extend_from_slice(&0u32.to_be_bytes());
        buf.extend_from_slice(&13u32.to_be_bytes());
        buf.extend_from_slice(&1u64.to_be_bytes());
        buf.extend_from_slice(b"hello");

        // UTF8 record 2: id=2, "java/lang/String" (16 bytes) → body = 8+16 = 24
        buf.push(0x01);
        buf.extend_from_slice(&0u32.to_be_bytes());
        buf.extend_from_slice(&24u32.to_be_bytes());
        buf.extend_from_slice(&2u64.to_be_bytes());
        buf.extend_from_slice(b"java/lang/String");

        // LOAD_CLASS: class_serial=1, class_id=0x100, stack_serial=0, name_id=2
        // body = 4 + 8 + 4 + 8 = 24
        buf.push(0x02);
        buf.extend_from_slice(&0u32.to_be_bytes());
        buf.extend_from_slice(&24u32.to_be_bytes());
        buf.extend_from_slice(&1u32.to_be_bytes()); // class_serial
        buf.extend_from_slice(&0x100u64.to_be_bytes()); // class_id
        buf.extend_from_slice(&0u32.to_be_bytes()); // stack_serial
        buf.extend_from_slice(&2u64.to_be_bytes()); // class_name_id

        // HEAP_DUMP_END
        buf.push(0x2C);
        buf.extend_from_slice(&0u32.to_be_bytes());
        buf.extend_from_slice(&0u32.to_be_bytes());

        buf
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

        let mut data: Vec<u8> = vec![];
        let count = build_utf8_index(&hprof, &record_index_bytes, &mut data).unwrap();
        assert_eq!(count, 2);

        let reader = Utf8IndexReader::from_ref(&data).unwrap();
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

        let mut lc_buf = Vec::new();
        let count = build_load_class_index(&hprof, &record_index_bytes, &mut lc_buf).unwrap();
        assert_eq!(count, 1);

        let reader = LoadClassReader::from_ref(&lc_buf).unwrap();
        assert_eq!(reader.find_class_name_id(0x100), Some(2));
        assert_eq!(reader.find_class_name_id(0x200), None);
    }

    #[test]
    fn heapsort_generic() {
        // Two 16-byte records with u64 key at offset 0.
        let mut data = Vec::new();
        data.extend_from_slice(&5u64.to_le_bytes());
        data.extend_from_slice(&0xAAAAu64.to_le_bytes()); // payload
        data.extend_from_slice(&1u64.to_le_bytes());
        data.extend_from_slice(&0xBBBBu64.to_le_bytes());
        heapsort(&mut data, 2, 16, 0);
        assert_eq!(read_u64_le(&data, 0), 1);
        assert_eq!(read_u64_le(&data, 8), 0xBBBB);
        assert_eq!(read_u64_le(&data, 16), 5);
        assert_eq!(read_u64_le(&data, 24), 0xAAAA);
    }
}
