//! Higher-level diff API: compute a [`DiffSummary`] from the pre-built diff
//! index files produced by [`crate::diff_index::build_diff_indexes`].
//!
//! This module does NOT perform the merge-walk itself — that is handled by
//! [`crate::diff_index::build_diff_indexes`].  Instead, it reads the three
//! binary index files (`removed.bin`, `added.bin`, `common.bin`) and
//! accumulates per-class counts to produce the summary table.

use crate::diff_index::{CommonEntryReader, DiffEntryReader, DiffIndexPaths};
use crate::heap_parser::SubRecord;
use crate::hprof::HprofError;
use crate::query::HeapQuery;
use crate::vfs::MMapReader;
use std::collections::HashMap;

// ── Synthetic class-ID flags (mirrors server/mod.rs) ─────────────────────────

const SYNTHETIC_OBJ_ARRAY: u64 = 1u64 << 63;
const SYNTHETIC_PRIM_ARRAY: u64 = (1u64 << 63) | (1u64 << 62);

// ── Public types ──────────────────────────────────────────────────────────────

/// Per-class instance counts from a two-snapshot diff.
#[derive(Debug, Clone)]
pub struct ClassDiffEntry {
    /// Class ID (real heap address or synthetic ID for arrays).
    pub class_id: u64,
    /// Dot-notation class name (e.g. `"java.lang.String"`).
    pub class_name: String,
    /// Instances present only in dump 1 (garbage-collected by dump 2).
    pub count_removed: u64,
    /// Instances present only in dump 2 (allocated between the two dumps).
    pub count_added: u64,
    /// Instances with the same object ID in both dumps (survived), unchanged.
    pub count_common_unchanged: u64,
    /// Instances with the same object ID in both dumps (survived), with changed data.
    pub count_common_changed: u64,
}

impl ClassDiffEntry {
    /// Total instances in both "common" categories.
    pub fn count_common(&self) -> u64 {
        self.count_common_unchanged + self.count_common_changed
    }

    /// Total instances in dump 1 (`count_removed + count_common()`).
    pub fn count_before(&self) -> u64 {
        self.count_removed + self.count_common()
    }

    /// Total instances in dump 2 (`count_added + count_common()`).
    pub fn count_after(&self) -> u64 {
        self.count_added + self.count_common()
    }

    /// Net change: positive = growth, negative = shrinkage.
    pub fn net_change(&self) -> i64 {
        self.count_added as i64 - self.count_removed as i64
    }
}

/// Summary of the differences between two heap snapshots.
#[derive(Debug)]
pub struct DiffSummary {
    /// Total meaningful objects in dump 1.
    pub total_before: u64,
    /// Total meaningful objects in dump 2.
    pub total_after: u64,
    /// Objects present only in dump 2 (added).
    pub total_added: u64,
    /// Objects present only in dump 1 (removed).
    pub total_removed: u64,
    /// Objects present in both dumps (common), with identical raw bytes.
    pub total_common_unchanged: u64,
    /// Objects present in both dumps (common), with differing raw bytes.
    pub total_common_changed: u64,
    /// Per-class breakdown sorted by `(count_added + count_removed)` descending.
    pub by_class: Vec<ClassDiffEntry>,
}

impl DiffSummary {
    /// Total objects in common (changed + unchanged).
    pub fn total_common(&self) -> u64 {
        self.total_common_unchanged + self.total_common_changed
    }
}

// ── Internal accumulator ──────────────────────────────────────────────────────

#[derive(Default)]
struct RawCounts {
    removed: u64,
    added: u64,
    common_unchanged: u64,
    common_changed: u64,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute the diff summary by reading the pre-built diff index files.
///
/// Opens `removed.bin`, `added.bin`, and `common.bin` from `paths`, then
/// iterates each file to accumulate per-class counts.  Class names are
/// resolved using `query1` (for removed/common) and `query2` (for added).
///
/// This is much cheaper than the merge-walk — it reads the compact 24/32-byte
/// index records rather than the raw hprof sub-record data.
pub fn compute_diff_summary(
    query1: &HeapQuery,
    query2: &HeapQuery,
    paths: &DiffIndexPaths,
) -> Result<DiffSummary, HprofError> {
    let removed_mmap = paths.removed.open_mmap()?;
    let added_mmap = paths.added.open_mmap()?;
    let common_mmap = paths.common.open_mmap()?;
    let removed_reader = DiffEntryReader::from_ref(removed_mmap.as_ref())?;
    let added_reader = DiffEntryReader::from_ref(added_mmap.as_ref())?;
    let common_reader = CommonEntryReader::from_ref(common_mmap.as_ref())?;

    let mut counts: HashMap<u64, RawCounts> = HashMap::new();
    let mut total_added = 0u64;
    let mut total_removed = 0u64;
    let mut total_common_unchanged = 0u64;
    let mut total_common_changed = 0u64;

    // ── removed ───────────────────────────────────────────────────────────────
    for entry in removed_reader.iter() {
        if let Some(k) = class_key_from_query(query1, &entry)? {
            counts.entry(k).or_default().removed += 1;
            total_removed += 1;
        }
    }

    // ── added ─────────────────────────────────────────────────────────────────
    for entry in added_reader.iter() {
        if let Some(k) = class_key_from_query(query2, &entry)? {
            counts.entry(k).or_default().added += 1;
            total_added += 1;
        }
    }

    // ── common ────────────────────────────────────────────────────────────────
    for entry in common_reader.iter() {
        use crate::diff_index::DiffEntry;
        let stub = DiffEntry {
            tag: entry.tag,
            object_id: entry.object_id,
            position: entry.position1,
        };
        if let Some(k) = class_key_from_query(query1, &stub)? {
            let raw = counts.entry(k).or_default();
            if entry.changed {
                raw.common_changed += 1;
                total_common_changed += 1;
            } else {
                raw.common_unchanged += 1;
                total_common_unchanged += 1;
            }
        }
    }

    // ── Resolve names and build output ────────────────────────────────────────
    let mut by_class: Vec<ClassDiffEntry> = counts
        .into_iter()
        .map(|(class_id, raw)| ClassDiffEntry {
            class_name: resolve_class_name(class_id, query1, query2),
            class_id,
            count_removed: raw.removed,
            count_added: raw.added,
            count_common_unchanged: raw.common_unchanged,
            count_common_changed: raw.common_changed,
        })
        .collect();

    by_class.sort_by(|a, b| {
        let a_change = a.count_added + a.count_removed;
        let b_change = b.count_added + b.count_removed;
        b_change
            .cmp(&a_change)
            .then_with(|| a.class_name.cmp(&b.class_name))
    });

    let total_before = total_removed + total_common_unchanged + total_common_changed;
    let total_after = total_added + total_common_unchanged + total_common_changed;

    Ok(DiffSummary {
        total_before,
        total_after,
        total_added,
        total_removed,
        total_common_unchanged,
        total_common_changed,
        by_class,
    })
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Extract the class-bucketing key from a [`DiffEntry`] by parsing the record
/// from the mmap.
fn class_key_from_query(
    query: &HeapQuery,
    entry: &crate::diff_index::DiffEntry,
) -> Result<Option<u64>, HprofError> {
    use crate::heap_index::sub_record::SubIndexEntry;
    let sub_entry = SubIndexEntry {
        tag: entry.tag,
        object_id: entry.object_id,
        position: entry.position,
    };
    let record = query.parse_entry(&sub_entry)?;
    Ok(match record {
        SubRecord::InstanceDump(inst) => Some(inst.class_id),
        SubRecord::ClassDump(cd) => Some(cd.class_id),
        SubRecord::ObjArrayDump(arr) => Some(SYNTHETIC_OBJ_ARRAY | arr.element_class_id),
        SubRecord::PrimArrayDump(arr) => Some(SYNTHETIC_PRIM_ARRAY | u64::from(arr.element_type)),
        _ => None,
    })
}

/// Resolve a display name for `class_id`.
fn resolve_class_name(class_id: u64, query1: &HeapQuery, query2: &HeapQuery) -> String {
    if class_id & (1u64 << 63) != 0 {
        if class_id & (1u64 << 62) != 0 {
            let elem_type = (class_id & 0xFF) as u8;
            return format!("{}[]", prim_type_name(elem_type));
        }
        let elem_class_id = class_id & !(1u64 << 63);
        let elem = query1
            .class_name(elem_class_id)
            .ok()
            .flatten()
            .or_else(|| query2.class_name(elem_class_id).ok().flatten())
            .unwrap_or_else(|| format!("0x{elem_class_id:x}"));
        return format!("{elem}[]");
    }
    query1
        .class_name(class_id)
        .ok()
        .flatten()
        .or_else(|| query2.class_name(class_id).ok().flatten())
        .unwrap_or_else(|| format!("0x{class_id:x}"))
}

fn prim_type_name(element_type: u8) -> &'static str {
    match element_type {
        4 => "boolean",
        5 => "char",
        6 => "float",
        7 => "double",
        8 => "byte",
        9 => "short",
        10 => "int",
        11 => "long",
        _ => "unknown",
    }
}
