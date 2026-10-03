pub mod error;
pub mod header;
pub mod record;
pub mod types;

pub use error::HprofError;
pub use header::HprofHeader;
pub use record::{RecordHeader, RecordTag};
pub use types::BasicType;

/// A memory-mapped hprof file.
///
/// Keeps the mmap alive for the lifetime of this struct. All data is accessed
/// through the mmap slice — no heap dump content is loaded into memory.
pub struct HprofFile<'a> {
    mmap: &'a [u8],
    id_size: u32,
    data_offset: usize,
}

impl<'a> HprofFile<'a> {
    pub fn from_ref(data: &'a [u8]) -> Result<Self, HprofError> {
        let header = HprofHeader::parse(data)?;
        Ok(Self::from_parts(data, &header))
    }

    /// Construct from a pre-parsed header (avoids re-parsing on every access).
    ///
    /// Copies only the two scalars the file needs, so this is free: it is
    /// called for every object lookup.
    pub fn from_parts(data: &'a [u8], header: &HprofHeader) -> Self {
        Self {
            mmap: data,
            id_size: header.id_size,
            data_offset: header.data_offset,
        }
    }

    /// Identifier size in bytes (4 or 8).
    pub fn id_size(&self) -> u32 {
        self.id_size
    }

    /// Return the full file contents as a byte slice.
    ///
    /// The returned slice has lifetime `'a` — the same lifetime as the
    /// underlying data — not the lifetime of `&self`.  This allows callers
    /// holding a `&HprofFile<'a>` (without tying the reference lifetime to
    /// `'a`) to still produce values that borrow from the data for `'a`.
    pub fn data(&self) -> &'a [u8] {
        self.mmap
    }

    /// Iterate over record headers in file order without loading record bodies.
    pub fn record_headers(&self) -> RecordHeaderIter<'_> {
        RecordHeaderIter {
            data: self.mmap,
            pos: self.data_offset,
        }
    }
}

/// Iterator over top-level hprof record headers.
///
/// Yields `RecordHeader` values in file order. Record bodies are skipped;
/// they can be read on demand via the mmap offset in `RecordHeader::position`.
pub struct RecordHeaderIter<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Iterator for RecordHeaderIter<'a> {
    type Item = Result<RecordHeader, HprofError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.data.len() {
            return None;
        }
        match RecordHeader::parse_at(self.data, self.pos) {
            Ok(rec) => {
                let next_pos = self.pos + 9 + rec.body_length as usize;
                if next_pos > self.data.len() {
                    return Some(Err(HprofError::UnexpectedEof(self.pos)));
                }
                self.pos = next_pos;
                Some(Ok(rec))
            }
            Err(e) => Some(Err(e)),
        }
    }
}
