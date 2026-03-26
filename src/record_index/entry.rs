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

    pub fn from_bytes(bytes: &[u8; INDEX_ENTRY_SIZE]) -> Self {
        let tag = bytes[0];
        let body_length = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let position = u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]);
        Self {
            tag,
            body_length,
            position,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
