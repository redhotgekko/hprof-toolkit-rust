use thiserror::Error;

#[derive(Error, Debug)]
pub enum HprofError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid hprof header: {0}")]
    InvalidHeader(&'static str),
    #[error("Unexpected end of file at offset {0}")]
    UnexpectedEof(usize),
    #[error("Unknown heap dump sub-record tag {0:#04x} at body offset {1}")]
    UnknownSubRecordTag(u8, usize),
    #[error("Unknown primitive type {0}")]
    UnknownPrimitiveType(u8),
    #[error("Invalid identifier size {0}, expected 4 or 8")]
    InvalidIdSize(u32),
    #[error("Index file is malformed (size is not a multiple of entry size)")]
    InvalidIndexFile,
}
