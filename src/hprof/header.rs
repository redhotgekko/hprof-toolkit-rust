use super::error::HprofError;
use super::record::{read_u32_be, read_u64_be};

/// Parsed hprof file header.
pub struct HprofHeader {
    /// e.g. "JAVA PROFILE 1.0.2"
    pub version: String,
    /// Size of object identifiers in bytes (4 or 8).
    pub id_size: u32,
    /// Dump creation timestamp (milliseconds since epoch).
    pub timestamp_ms: u64,
    /// Byte offset of the first record.
    pub data_offset: usize,
}

impl HprofHeader {
    /// Parse the hprof file header from the beginning of `data`.
    ///
    /// Layout:
    ///   - null-terminated version string
    ///   - u32 id_size (big-endian)
    ///   - u64 timestamp_ms (big-endian)
    pub fn parse(data: &[u8]) -> Result<Self, HprofError> {
        let null_pos = data
            .iter()
            .position(|&b| b == 0)
            .ok_or(HprofError::InvalidHeader(
                "no null terminator in version string",
            ))?;

        let version = String::from_utf8_lossy(&data[..null_pos]).into_owned();
        if !version.starts_with("JAVA PROFILE") {
            return Err(HprofError::InvalidHeader("not a valid hprof file"));
        }

        let mut offset = null_pos + 1;

        if offset + 4 > data.len() {
            return Err(HprofError::UnexpectedEof(offset));
        }
        let id_size = read_u32_be(data, offset);
        offset += 4;

        if offset + 8 > data.len() {
            return Err(HprofError::UnexpectedEof(offset));
        }
        let timestamp_ms = read_u64_be(data, offset);
        offset += 8;

        Ok(Self {
            version,
            id_size,
            timestamp_ms,
            data_offset: offset,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_header(version: &str, id_size: u32, timestamp_ms: u64) -> Vec<u8> {
        let mut buf = version.as_bytes().to_vec();
        buf.push(0); // null terminator
        buf.extend_from_slice(&id_size.to_be_bytes());
        buf.extend_from_slice(&timestamp_ms.to_be_bytes());
        buf
    }

    #[test]
    fn parse_valid_header_1_0_2() {
        let data = minimal_header("JAVA PROFILE 1.0.2", 8, 1_700_000_000_000);
        let hdr = HprofHeader::parse(&data).unwrap();
        assert_eq!(hdr.version, "JAVA PROFILE 1.0.2");
        assert_eq!(hdr.id_size, 8);
        assert_eq!(hdr.timestamp_ms, 1_700_000_000_000);
        // data_offset = len("JAVA PROFILE 1.0.2") + 1 (null) + 4 (id_size) + 8 (ts)
        assert_eq!(hdr.data_offset, 18 + 1 + 4 + 8);
    }

    #[test]
    fn parse_valid_header_1_0_1() {
        let data = minimal_header("JAVA PROFILE 1.0.1", 4, 0);
        let hdr = HprofHeader::parse(&data).unwrap();
        assert_eq!(hdr.version, "JAVA PROFILE 1.0.1");
        assert_eq!(hdr.id_size, 4);
    }

    #[test]
    fn parse_invalid_magic() {
        let data = minimal_header("NOT A PROFILE", 8, 0);
        assert!(HprofHeader::parse(&data).is_err());
    }

    #[test]
    fn parse_truncated() {
        // Only the version string, no id_size/timestamp
        let mut data = b"JAVA PROFILE 1.0.2".to_vec();
        data.push(0);
        assert!(HprofHeader::parse(&data).is_err());
    }
}
