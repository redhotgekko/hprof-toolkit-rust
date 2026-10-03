/// Size in bytes of a serialized [`IndexEntry`].
pub const INDEX_ENTRY_SIZE: usize = 16;

/// A fixed-size entry in the top-level record index.
///
/// Binary layout (all little-endian):
/// ```text
///  0..1   u8   tag
///  1..4   [u8; 3] padding (zeros)
///  4..8   u32  body_length
///  8..16  u64  position  (byte offset of the tag byte in the hprof file)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexEntry {
    pub tag: u8,
    pub body_length: u32,
    /// Byte offset of the tag byte within the hprof file.
    pub position: u64,
}

impl IndexEntry {
    pub fn to_bytes(self) -> [u8; INDEX_ENTRY_SIZE] {
        let mut buf = [0u8; INDEX_ENTRY_SIZE];
        buf[0] = self.tag;
        // buf[1..4] = 0 (padding)
        buf[4..8].copy_from_slice(&self.body_length.to_le_bytes());
        buf[8..16].copy_from_slice(&self.position.to_le_bytes());
        buf
    }
}

/// Record-index entries are unsorted (hprof record order); the key is the
/// file position, which is at least strictly increasing.
impl crate::index::Entry for IndexEntry {
    const SIZE: usize = INDEX_ENTRY_SIZE;
    const KEY_OFFSET: usize = 8;

    fn from_bytes(b: &[u8]) -> Self {
        Self {
            tag: b[0],
            body_length: crate::index::read_u32_le(b, 4),
            position: crate::index::read_u64_le(b, 8),
        }
    }

    fn write_to(&self, out: &mut [u8]) {
        out.copy_from_slice(&self.to_bytes());
    }

    fn key(&self) -> u64 {
        self.position
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Entry;

    #[test]
    fn round_trip() {
        let entry = IndexEntry {
            tag: 0x1C,
            body_length: 4096,
            position: 0xDEAD_BEEF_CAFE_1234,
        };
        let bytes = entry.to_bytes();
        assert_eq!(bytes.len(), INDEX_ENTRY_SIZE);
        let decoded = IndexEntry::from_bytes(&bytes);
        assert_eq!(decoded, entry);
    }

    #[test]
    fn padding_is_zero() {
        let entry = IndexEntry {
            tag: 0x01,
            body_length: 0,
            position: 0,
        };
        let bytes = entry.to_bytes();
        assert_eq!(bytes[1], 0);
        assert_eq!(bytes[2], 0);
        assert_eq!(bytes[3], 0);
    }
}
