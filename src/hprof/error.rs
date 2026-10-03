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
    /// An index entry is corrupt: its size is not a multiple of its record
    /// size.  Only the index layer constructs this.
    #[error("Index file is malformed (size is not a multiple of entry size)")]
    InvalidIndexFile,
    /// An optional index (e.g. retained heap) has not been built.
    #[error("{0} index has not been built; run `hprof-toolkit index` first")]
    NotIndexed(&'static str),
    /// The index store is unusable for this hprof or this build (wrong
    /// format version, built for another dump, …).  Rebuild with
    /// `hprof-toolkit index --force`.
    #[error("index store is unusable: {0}; rebuild with `hprof-toolkit index --force`")]
    Corrupt(String),
    /// A build step would need more memory than is available.
    #[error(
        "{step} needs about {} MB of memory but only {} MB is available; \
         index with the retained heap disabled (--no-retained) or raise the \
         limit (IndexOptions::max_memory)",
        .needed / (1024 * 1024),
        .available / (1024 * 1024)
    )]
    InsufficientMemory {
        step: &'static str,
        needed: u64,
        available: u64,
    },
    /// The heap is larger than a build step can handle (for example more
    /// references than the dominator builder's `u32` indexes can address).
    #[error("{0}; index with --no-retained to skip the retained sizes")]
    TooLarge(String),
    /// A caller-supplied argument was unusable (bad hex id, unknown root type, …).
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    /// A bug or a poisoned lock; never caused by user input or bad data.
    #[error("internal error: {0}")]
    Internal(String),
}
