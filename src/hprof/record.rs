use super::error::HprofError;

/// Tags defined in heapDumper.cpp (OpenJDK).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordTag {
    Utf8,
    LoadClass,
    UnloadClass,
    Frame,
    Trace,
    AllocSites,
    HeapSummary,
    StartThread,
    EndThread,
    HeapDump,
    HeapDumpSegment,
    HeapDumpEnd,
    CpuSamples,
    ControlSettings,
    Unknown(u8),
}

impl From<u8> for RecordTag {
    fn from(b: u8) -> Self {
        match b {
            0x01 => Self::Utf8,
            0x02 => Self::LoadClass,
            0x03 => Self::UnloadClass,
            0x04 => Self::Frame,
            0x05 => Self::Trace,
            0x06 => Self::AllocSites,
            0x07 => Self::HeapSummary,
            0x0A => Self::StartThread,
            0x0B => Self::EndThread,
            0x0C => Self::HeapDump,
            0x1C => Self::HeapDumpSegment,
            0x2C => Self::HeapDumpEnd,
            0x0D => Self::CpuSamples,
            0x0E => Self::ControlSettings,
            other => Self::Unknown(other),
        }
    }
}

impl From<RecordTag> for u8 {
    fn from(tag: RecordTag) -> u8 {
        match tag {
            RecordTag::Utf8 => 0x01,
            RecordTag::LoadClass => 0x02,
            RecordTag::UnloadClass => 0x03,
            RecordTag::Frame => 0x04,
            RecordTag::Trace => 0x05,
            RecordTag::AllocSites => 0x06,
            RecordTag::HeapSummary => 0x07,
            RecordTag::StartThread => 0x0A,
            RecordTag::EndThread => 0x0B,
            RecordTag::HeapDump => 0x0C,
            RecordTag::HeapDumpSegment => 0x1C,
            RecordTag::HeapDumpEnd => 0x2C,
            RecordTag::CpuSamples => 0x0D,
            RecordTag::ControlSettings => 0x0E,
            RecordTag::Unknown(b) => b,
        }
    }
}

/// The fixed 9-byte header preceding every record body.
#[derive(Debug, Clone, Copy)]
pub struct RecordHeader {
    pub tag: RecordTag,
    pub time_offset_ms: u32,
    pub body_length: u32,
    /// Byte offset of the tag byte within the hprof file.
    pub position: u64,
}

impl RecordHeader {
    /// Parse a record header at `pos` within `data`.
    ///
    /// A record header is 9 bytes: tag(1) + time_offset(4) + body_length(4).
    pub fn parse_at(data: &[u8], pos: usize) -> Result<Self, HprofError> {
        if pos + 9 > data.len() {
            return Err(HprofError::UnexpectedEof(pos));
        }
        let tag = RecordTag::from(data[pos]);
        let time_offset_ms = read_u32_be(data, pos + 1);
        let body_length = read_u32_be(data, pos + 5);
        Ok(Self {
            tag,
            time_offset_ms,
            body_length,
            position: pos as u64,
        })
    }
}

/// Read an identifier at `offset`, zero-extended to u64. `id_size` must be 4 or 8.
pub(crate) fn read_id(data: &[u8], offset: usize, id_size: usize) -> Result<u64, HprofError> {
    match id_size {
        4 => {
            if offset + 4 > data.len() {
                return Err(HprofError::UnexpectedEof(offset));
            }
            Ok(read_u32_be(data, offset) as u64)
        }
        8 => {
            if offset + 8 > data.len() {
                return Err(HprofError::UnexpectedEof(offset));
            }
            Ok(read_u64_be(data, offset))
        }
        _ => Err(HprofError::InvalidIdSize(id_size as u32)),
    }
}

/// Read 2 bytes as big-endian u16. Caller must ensure `data[offset..offset+2]` is valid.
pub(crate) fn read_u16_be(data: &[u8], offset: usize) -> u16 {
    u16::from_be_bytes([data[offset], data[offset + 1]])
}

/// Read 4 bytes as big-endian u32. Caller must ensure `data[offset..offset+4]` is valid.
pub(crate) fn read_u32_be(data: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

/// Read 8 bytes as big-endian u64. Caller must ensure `data[offset..offset+8]` is valid.
pub(crate) fn read_u64_be(data: &[u8], offset: usize) -> u64 {
    u64::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_tag_round_trip() {
        let tags = [
            RecordTag::Utf8,
            RecordTag::HeapDump,
            RecordTag::HeapDumpSegment,
            RecordTag::HeapDumpEnd,
            RecordTag::Unknown(0xFF),
        ];
        for tag in tags {
            assert_eq!(RecordTag::from(u8::from(tag)), tag);
        }
    }

    #[test]
    fn parse_record_header() {
        // tag=0x01, time_offset=0x00000000, body_length=0x00000005
        let data = [0x01u8, 0, 0, 0, 0, 0, 0, 0, 5, b'h', b'e', b'l', b'l', b'o'];
        let hdr = RecordHeader::parse_at(&data, 0).unwrap();
        assert_eq!(hdr.tag, RecordTag::Utf8);
        assert_eq!(hdr.time_offset_ms, 0);
        assert_eq!(hdr.body_length, 5);
        assert_eq!(hdr.position, 0);
    }

    #[test]
    fn parse_record_header_eof() {
        let data = [0x01u8, 0, 0]; // only 3 bytes, need 9
        assert!(RecordHeader::parse_at(&data, 0).is_err());
    }
}
