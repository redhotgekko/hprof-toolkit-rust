pub mod error;
pub mod header;
pub mod record;

pub use error::HprofError;
pub use header::HprofHeader;
pub use record::{RecordHeader, RecordTag};

use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

/// Open a file read-only and memory-map it.
///
/// Used by both `HprofFile::open` and the heap parser's `SubIndexReader`.
#[allow(unsafe_code)]
pub fn map_file(path: &Path) -> Result<Mmap, HprofError> {
    let file = File::open(path)?;
    // SAFETY: file opened read-only and not modified while mapped.
    let mmap = unsafe { Mmap::map(&file) }?;
    Ok(mmap)
}

/// A memory-mapped hprof file.
///
/// Keeps the mmap alive for the lifetime of this struct. All data is accessed
/// through the mmap slice — no heap dump content is loaded into memory.
pub struct HprofFile<'a> {
    mmap: &'a [u8],
    pub header: HprofHeader,
}

impl<'a> HprofFile<'a> {
    pub fn from_ref(data: &'a [u8]) -> Result<Self, HprofError> {
        let header = HprofHeader::parse(data)?;
        Ok(Self { mmap: data, header })
    }

    /// Construct from pre-parsed header (avoids re-parsing on every access).
    pub fn from_parts(data: &'a [u8], header: HprofHeader) -> Self {
        Self { mmap: data, header }
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
            pos: self.header.data_offset,
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
