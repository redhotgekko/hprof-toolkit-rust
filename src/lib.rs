//! # hprof-toolkit
//!
//! Analyse Java heap dumps (`.hprof`) that are too large to load into memory.
//!
//! Everything is read through memory-mapped files: a one-off *indexing* pass
//! writes fixed-size, sorted binary index files next to the dump, and every
//! query afterwards is a binary search over those files.  Process memory
//! stays modest no matter how big the heap is (the dominator/retained-size
//! step is the one documented exception, see [`pipeline::IndexOptions`]).
//!
//! ## Quick start
//!
//! [`HeapQuery`] is the single entry point.  Open a dump (building its
//! indexes on first use) and ask questions:
//!
//! ```no_run
//! use hprof_toolkit::prelude::*;
//!
//! let heap = HeapQuery::open("heap.hprof")?;
//!
//! // The biggest classes by instance count, straight from a precomputed index.
//! for row in heap.class_histogram(Page::first(10)).items {
//!     println!("{:>10}  {}", row.instance_count, row.class_name);
//! }
//!
//! // Look one object up by id and resolve its fields.
//! if let Some(SubRecord::InstanceDump(inst)) = heap.object(0x1a2b3c)? {
//!     let resolved = heap.resolve_instance(&inst)?;
//!     println!("{}: {:?}", resolved.class_name, resolved.fields);
//! }
//!
//! // Why is it alive?
//! let path = heap.path_to_root(0x1a2b3c, &RootPathLimits::default());
//! if path.outcome == PathOutcome::Found {
//!     println!("{} hops to a GC root", path.steps.len() - 1);
//! }
//! # Ok::<(), HprofError>(())
//! ```
//!
//! ## Where things are
//!
//! * [`query::HeapQuery`] — objects, classes, names, roots, references,
//!   retained sizes, threads and stack traces.
//! * [`resolved`] — owned, name-resolved views of instances, classes,
//!   arrays and GC roots.
//! * [`pipeline`] — building the indexes explicitly ([`pipeline::IndexOptions`]
//!   selects what to build; [`progress::Progress`] reports how it is going).
//! * [`diff`] — comparing two snapshots of the same JVM.
//! * [`server`] — the HTTP UI and MCP endpoint used by the `hprof-toolkit`
//!   binary.
//! * [`index`] — where index files live ([`index::IndexStore`]): the same
//!   code runs against a directory of files and against an in-memory store,
//!   which is what the crate's own tests use.
//!
//! The builders and readers of the on-disk index files are private; the files
//! themselves are documented in `INDEX_FILE_FORMATS.md`.

// unsafe_code is denied crate-wide. The only exception is the memmap2 wrapper
// in index::store, which needs unsafe blocks for Mmap::map / MmapMut::map_mut.  We cannot use #![forbid(unsafe_code)]
// because forbid cannot be overridden by inner #[allow] attributes.
#![deny(unsafe_code)]

// Public modules: the analysis API (`query`, `graph`, `classes`, `threads`,
// `diff`, `resolved`), the format types it returns (`hprof`, `heap_parser`,
// `class_key`), building and storage (`pipeline`, `progress`, `index`) and
// the front ends (`server`).  Everything else is the index layer: builders
// and readers of the on-disk formats, documented in INDEX_FILE_FORMATS.md but
// not part of the API.
mod array_index;
mod aux_index;
mod aux_query;
mod class_index;
pub mod class_key;
pub(crate) mod class_layout;
pub mod classes;
pub mod diff;
mod diff_index;
mod dominator;
pub mod graph;
mod heap_index;
pub mod heap_parser;
mod heap_query;
pub mod hprof;
pub mod index;
mod object_store;
pub mod pipeline;
pub mod progress;
pub mod query;
mod record_index;
mod ref_index;
pub mod resolved;
mod root_index;
pub mod search;
pub mod server;
mod sort;
pub mod threads;

#[cfg(test)]
pub(crate) mod test_util;

pub use dominator::VIRTUAL_ROOT_ID;
pub use hprof::HprofError;
pub use query::HeapQuery;

/// The names most analysis programs need, in one import:
/// `use hprof_toolkit::prelude::*;`.
pub mod prelude {
    pub use crate::VIRTUAL_ROOT_ID;
    pub use crate::array_index::{ArrayKind, ArraySizeEntry};
    pub use crate::aux_query::LineNumber;
    pub use crate::class_key::ClassKey;
    pub use crate::classes::{ClassSummary, HistogramEntry};
    pub use crate::diff::{DiffObject, DiffSummary, HeapDiff};
    pub use crate::graph::{Edge, PathOutcome, PathStep, Reference, RootPath, RootPathLimits};
    pub use crate::heap_parser::{
        ClassDump, FieldValue, InstanceDump, ObjArrayDump, PrimArrayDump, SubRecord,
    };
    pub use crate::heap_query::Field;
    pub use crate::hprof::{BasicType, HprofError};
    pub use crate::pipeline::IndexOptions;
    pub use crate::progress::{NoProgress, Progress, StderrProgress};
    pub use crate::query::{HeapQuery, Page, PageResult, ScanResult, ScanWindow, StringMatch};
    pub use crate::resolved::{
        PrimArrayElements, PrimArrayWindow, Resolved, ResolvedClass, ResolvedInstance,
        ResolvedObjArray, ResolvedPrimArray, ResolvedRoot, Value,
    };
    pub use crate::root_index::GcRootType;
    pub use crate::search::{Matcher, SearchMode, SearchQuery};
    pub use crate::threads::{FrameInfo, ThreadInfo};
    /// Re-exported so `heap.par_instances().map(..)` works with one import.
    pub use rayon::iter::{IntoParallelIterator, ParallelIterator};
}
