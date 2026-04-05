// unsafe_code is denied crate-wide. The single exception is the memmap2
// wrapper in hprof::file, which needs one unsafe block for Mmap::map.
// We cannot use #![forbid(unsafe_code)] because forbid cannot be overridden
// by inner #[allow] attributes.
#![deny(unsafe_code)]

pub mod array_index;
pub mod aux_index;
pub mod aux_query;
pub mod diff;
pub mod diff_index;
pub mod dominator;
pub mod heap_index;
pub mod heap_parser;
pub mod heap_query;
pub mod hprof;
pub mod object_store;
pub mod pipeline;
pub mod query;
pub mod record_index;
pub mod ref_index;
pub mod resolved;
pub mod root_index;
pub mod server;
pub mod sort;
pub mod vfs;
