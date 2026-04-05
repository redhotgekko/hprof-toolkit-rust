//! Dominator tree and retained heap size computation.
//!
//! Implements the Cooper et al. (2001) iterative dominator algorithm to compute
//! the immediate dominator of every reachable heap object, then accumulates
//! retained heap sizes bottom-up.
//!
//! ## Output files
//!
//! * `dominators.bin` — 16-byte records `(object_id: u64, dominator_id: u64)`,
//!   sorted by `object_id`.  `dominator_id = 0` means the object is directly
//!   dominated by the virtual GC root (i.e. it is itself a GC root or only
//!   reachable from the synthetic root node).
//! * `retained.bin`   — 16-byte records `(object_id: u64, retained_bytes: u64)`,
//!   sorted by `object_id`.
//!
//! ## Memory usage
//!
//! Requires O(N + E) RAM where N is the number of heap objects and E is the
//! total number of object references.  For production use on very large heap
//! dumps (> 100 M objects), a file-backed implementation is recommended.

use crate::heap_index::sub_record::{
    SubIndexEntry, TAG_CLASS_DUMP, TAG_INSTANCE_DUMP, TAG_OBJ_ARRAY_DUMP, TAG_PRIM_ARRAY_DUMP,
};
use crate::heap_parser::record::FieldValue;
use crate::heap_parser::{SubIndexReader, SubRecord, parse_sub_record};
use crate::heap_query::resolve::read_field_value;
use crate::hprof::{HprofError, HprofFile};
use crate::root_index::RootIndexReader;
use crate::vfs::{MMapReader, MMapWriter};
use rayon::prelude::*;
use std::collections::VecDeque;
use std::io::Write;

// ── Entry format ──────────────────────────────────────────────────────────────

/// Byte size of one entry in `dominators.bin`.
pub const DOM_ENTRY_SIZE: usize = 16;

/// Byte size of one entry in `retained.bin`.
pub const RETAINED_ENTRY_SIZE: usize = 16;

/// The dominator_id written for objects that are dominated only by the virtual
/// GC root (i.e. direct GC roots themselves).  Zero is not a valid Java object
/// ID in any hprof file.
pub const VIRTUAL_ROOT_ID: u64 = 0;

// ── Public build function ─────────────────────────────────────────────────────

/// Build `dominators.bin` and `retained.bin` for a heap dump.
///
/// Requires the combined object store index and all nine GC root index readers
/// to be available.  The two output files are written sorted by `object_id`.
///
/// Returns `(dominator_entry_count, retained_entry_count)`.
///
/// **Memory usage:** O(N + E) where N = number of heap objects and E = number
/// of object references.  For very large heaps this may require several GB of
/// RAM; consider adding a `--skip-retained-heap` flag for production use.
pub fn build_dominator_and_retained(
    hprof_source: &impl MMapReader,
    combined_source: &impl MMapReader,
    root_readers: &[RootIndexReader<'_>; 9],
    dominators_out: &mut impl MMapWriter,
    retained_out: &mut impl MMapWriter,
) -> Result<(u64, u64), HprofError> {
    // Open indexes.
    let mmap = hprof_source.open_mmap()?;
    let hprof = HprofFile::from_ref(mmap.as_ref())?;
    let combined_bytes = combined_source.open_mmap()?;
    let combined_slice = combined_bytes.as_ref();
    let combined = SubIndexReader::from_ref(combined_slice)?;

    // ── Pass 1 (sequential): build class field-layout cache ───────────────────
    // Eliminates O(depth × log N) binary searches per instance in pass 2.
    let class_cache = build_class_cache(&hprof, &combined)?;

    // ── Streaming two-pass CSR build ──────────────────────────────────────────
    let (compact_ids, shallow_sizes, fwd_off, fwd_edges) =
        build_forward_csr_streaming(&hprof, &combined, &class_cache)?;
    let n = compact_ids.len();

    // ── GC roots as compact indices ───────────────────────────────────────────
    let gc_roots_compact = collect_gc_roots_compact(root_readers, &compact_ids);

    // ── Color nodes by exclusive GC root ─────────────────────────────────────
    let color = color_nodes_by_root(n, &gc_roots_compact, &fwd_off, &fwd_edges);

    // ── Partition-parallel Cooper et al. dominator algorithm ─────────────────
    // Each exclusive partition runs in parallel; MULTI partition runs after.
    let global_doms =
        run_partitioned_dominators(n, &gc_roots_compact, &fwd_off, &fwd_edges, &color);

    // ── Retained sizes via Kahn's topological algorithm ───────────────────────
    let retained = compute_retained_from_doms(n, &global_doms, &shallow_sizes);

    // ── Write output files ────────────────────────────────────────────────────
    let (dom_count, ret_count) = write_outputs(
        n,
        &compact_ids,
        &global_doms,
        &retained,
        dominators_out,
        retained_out,
    )?;

    Ok((dom_count, ret_count))
}

// ── Public reader: DominatorIndex ─────────────────────────────────────────────

/// Read-only handle to a sorted `dominators.bin` index file.
///
/// Each entry stores `(object_id, dominator_id)`.  A `dominator_id` of
/// [`VIRTUAL_ROOT_ID`] (`0`) means the object is a direct GC root.
pub struct DominatorIndex<'a> {
    data: &'a [u8],
}

impl<'a> DominatorIndex<'a> {
    pub fn from_ref(bytes: &'a [u8]) -> Result<Self, HprofError> {
        if !bytes.len().is_multiple_of(DOM_ENTRY_SIZE) {
            return Err(HprofError::InvalidIndexFile);
        }
        Ok(Self { data: bytes })
    }

    pub(crate) fn from_slice(bytes: &'a [u8]) -> Self {
        debug_assert!(bytes.len().is_multiple_of(DOM_ENTRY_SIZE));
        Self { data: bytes }
    }

    fn as_slice(&self) -> &[u8] {
        self.data
    }

    /// Total number of entries.
    pub fn len(&self) -> usize {
        self.as_slice().len() / DOM_ENTRY_SIZE
    }

    /// Returns `true` if the index contains no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the `dominator_id` of `object_id`, or `None` if not found.
    ///
    /// O(log n) binary search.
    pub fn find(&self, object_id: u64) -> Option<u64> {
        let data = self.as_slice();
        let n = self.len();
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let key = read_u64_le(data, mid * DOM_ENTRY_SIZE);
            if key < object_id {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo < n && read_u64_le(data, lo * DOM_ENTRY_SIZE) == object_id {
            Some(read_u64_le(data, lo * DOM_ENTRY_SIZE + 8))
        } else {
            None
        }
    }
}

// ── Public reader: RetainedIndex ──────────────────────────────────────────────

/// Read-only handle to a sorted `retained.bin` index file.
///
/// Each entry stores `(object_id, retained_bytes)`.
pub struct RetainedIndex<'a> {
    data: &'a [u8],
}

impl<'a> RetainedIndex<'a> {
    /// Create a validated reader from a byte slice.
    pub fn from_ref(data: &'a [u8]) -> Result<Self, HprofError> {
        if !data.len().is_multiple_of(RETAINED_ENTRY_SIZE) {
            return Err(HprofError::InvalidIndexFile);
        }
        Ok(Self { data })
    }

    /// Create a reader from a slice already known to be valid.
    pub(crate) fn from_slice(data: &'a [u8]) -> Self {
        debug_assert!(data.len().is_multiple_of(RETAINED_ENTRY_SIZE));
        Self { data }
    }

    fn as_slice(&self) -> &[u8] {
        self.data
    }

    /// Total number of entries.
    pub fn len(&self) -> usize {
        self.as_slice().len() / RETAINED_ENTRY_SIZE
    }

    /// Returns `true` if the index contains no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the retained heap size in bytes for `object_id`, or `None`.
    ///
    /// O(log n) binary search.
    pub fn find(&self, object_id: u64) -> Option<u64> {
        let data = self.as_slice();
        let n = self.len();
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let key = read_u64_le(data, mid * RETAINED_ENTRY_SIZE);
            if key < object_id {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo < n && read_u64_le(data, lo * RETAINED_ENTRY_SIZE) == object_id {
            Some(read_u64_le(data, lo * RETAINED_ENTRY_SIZE + 8))
        } else {
            None
        }
    }

    /// Iterate all entries in ascending `object_id` order.
    ///
    /// Yields `(object_id, retained_bytes)` pairs.
    pub fn iter(&self) -> RetainedIter<'a> {
        RetainedIter {
            data: self.data,
            pos: 0,
            len: self.len(),
        }
    }
}

/// Iterator over `(object_id, retained_bytes)` entries in a [`RetainedIndex`].
pub struct RetainedIter<'a> {
    data: &'a [u8],
    pos: usize,
    len: usize,
}

impl Iterator for RetainedIter<'_> {
    type Item = (u64, u64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.len {
            return None;
        }
        let offset = self.pos * RETAINED_ENTRY_SIZE;
        let object_id = read_u64_le(self.data, offset);
        let retained_bytes = read_u64_le(self.data, offset + 8);
        self.pos += 1;
        Some((object_id, retained_bytes))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.pos;
        (remaining, Some(remaining))
    }
}

// ── Private: class field-layout cache ────────────────────────────────────────

/// Per-class instance field metadata, used to walk field data without repeated
/// binary searches into the combined index.
struct ClassLayout {
    super_class_id: u64,
    /// hprof field-type code for each instance field in declaration order.
    /// Type 2 = object reference (id_size bytes); others are primitive types.
    field_types: Vec<u8>,
}

/// Sorted-array map from class_id → ClassLayout; binary-search lookup.
///
/// Avoids any hash-map overhead: build once, share read-only across threads.
struct ClassCache {
    ids: Vec<u64>,
    layouts: Vec<ClassLayout>,
}

impl ClassCache {
    fn get(&self, class_id: u64) -> Option<&ClassLayout> {
        self.ids
            .binary_search(&class_id)
            .ok()
            .map(|i| &self.layouts[i])
    }
}

/// Scan the combined index once, building a [`ClassCache`] for every CLASS_DUMP.
///
/// This cache eliminates the O(depth × log N) binary searches that the
/// previous implementation performed for each instance dump.
fn build_class_cache(
    hprof: &HprofFile,
    combined: &SubIndexReader,
) -> Result<ClassCache, HprofError> {
    let mut pairs: Vec<(u64, ClassLayout)> = Vec::new();
    for entry in combined.iter() {
        if entry.tag != TAG_CLASS_DUMP {
            continue;
        }
        let rec = parse_sub_record(hprof, &entry)?;
        let cd = match rec {
            SubRecord::ClassDump(c) => c,
            _ => continue,
        };
        let field_types: Result<Vec<u8>, HprofError> = cd
            .instance_fields()
            .map(|r| r.map(|fd| fd.field_type))
            .collect();
        pairs.push((
            cd.class_id,
            ClassLayout {
                super_class_id: cd.super_class_id,
                field_types: field_types?,
            },
        ));
    }
    pairs.sort_unstable_by_key(|&(id, _)| id);
    pairs.dedup_by_key(|(id, _)| *id);
    let ids = pairs.iter().map(|&(id, _)| id).collect();
    let layouts = pairs.into_iter().map(|(_, l)| l).collect();
    Ok(ClassCache { ids, layouts })
}

// ── Private: streaming two-pass CSR builder ───────────────────────────────────
//
// **Pass 1** (parallel): for every heap record, read its outgoing references
//   to count them exactly, storing only (object_id, shallow, out_degree,
//   entry_idx) — 24 bytes per object.  No Vec<u64> per object.
//
// **Pass 2** (sequential): re-read each record, appending (to_id, from_compact)
//   pairs directly into a single flat edge buffer — no per-object allocation.
//
// After Pass 2: sort the buffer by `to_id`, then merge with `compact_ids` in a
// two-pointer scan that writes directly into the pre-allocated `fwd_edges` —
// eliminating the intermediate "translated" Vec of the old approach.
//
// Peak memory: 24·N + 16·E  bytes
//    vs old:   40·N + 28·E  bytes  (Vec<RawNode> headers + raw_edges + translated)

// (compact_ids, shallow_sizes, fwd_offsets, fwd_edges)
type ForwardCsr = (Vec<u64>, Vec<u64>, Vec<u32>, Vec<u32>);
// (object_id, shallow_size, out_degree, entry_index)
type NodeMeta = (u64, u64, u32, u32);

fn build_forward_csr_streaming(
    hprof: &HprofFile,
    combined: &SubIndexReader,
    class_cache: &ClassCache,
) -> Result<ForwardCsr, HprofError> {
    let id_size = hprof.header.id_size as usize;
    let entries: Vec<SubIndexEntry> = combined.iter().collect();
    let entry_count = entries.len();

    // ── Pass 1 (parallel): count exact out-degrees ────────────────────────────
    // Yields (object_id, shallow_size, out_degree, original_entry_index).
    let pass1: Vec<Result<Option<NodeMeta>, HprofError>> = entries
        .par_iter()
        .enumerate()
        .map(|(k, entry)| -> Result<Option<NodeMeta>, HprofError> {
            match entry.tag {
                TAG_INSTANCE_DUMP => {
                    let rec = parse_sub_record(hprof, entry)?;
                    let inst = match rec {
                        SubRecord::InstanceDump(i) => i,
                        _ => return Ok(None),
                    };
                    let shallow = inst.data.len() as u64;
                    let count = count_instance_refs(&inst, class_cache, id_size)? as u32;
                    Ok(Some((inst.object_id, shallow, count, k as u32)))
                }
                TAG_CLASS_DUMP => {
                    let rec = parse_sub_record(hprof, entry)?;
                    let cd = match rec {
                        SubRecord::ClassDump(c) => c,
                        _ => return Ok(None),
                    };
                    let mut count = 0u32;
                    for sf_res in cd.static_fields() {
                        if let FieldValue::Object(id) = sf_res?.value
                            && id != 0
                        {
                            count += 1;
                        }
                    }
                    Ok(Some((cd.class_id, 0u64, count, k as u32)))
                }
                TAG_OBJ_ARRAY_DUMP => {
                    let rec = parse_sub_record(hprof, entry)?;
                    let arr = match rec {
                        SubRecord::ObjArrayDump(a) => a,
                        _ => return Ok(None),
                    };
                    let shallow = arr.num_elements as u64 * id_size as u64;
                    let count = arr.elements().filter(|&id| id != 0).count() as u32;
                    Ok(Some((arr.array_id, shallow, count, k as u32)))
                }
                TAG_PRIM_ARRAY_DUMP => {
                    let rec = parse_sub_record(hprof, entry)?;
                    let arr = match rec {
                        SubRecord::PrimArrayDump(a) => a,
                        _ => return Ok(None),
                    };
                    let shallow = arr.num_elements as u64 * prim_elem_byte_size(arr.element_type);
                    Ok(Some((arr.array_id, shallow, 0u32, k as u32)))
                }
                _ => Ok(None),
            }
        })
        .collect();

    // Sort by object_id, assign compact indices.
    let mut meta: Vec<(u64, u64, u32, u32)> = Vec::with_capacity(entry_count);
    for r in pass1 {
        if let Some(m) = r? {
            meta.push(m);
        }
    }
    meta.sort_unstable_by_key(|&(id, _, _, _)| id);
    meta.dedup_by_key(|(id, _, _, _)| *id);
    let n = meta.len();

    let compact_ids: Vec<u64> = meta.iter().map(|&(id, _, _, _)| id).collect();
    let shallow_sizes: Vec<u64> = meta.iter().map(|&(_, sh, _, _)| sh).collect();

    let mut fwd_off: Vec<u32> = Vec::with_capacity(n + 1);
    let mut fwd_running = 0u32;
    fwd_off.push(fwd_running);
    for &(_, _, deg, _) in &meta {
        fwd_running = fwd_running.saturating_add(deg);
        fwd_off.push(fwd_running);
    }

    // Map original entry index → compact index for O(1) lookup in Pass 2.
    let mut entry_to_compact: Vec<u32> = vec![u32::MAX; entry_count];
    for (ci, &(_, _, _, ek)) in meta.iter().enumerate() {
        entry_to_compact[ek as usize] = ci as u32;
    }
    drop(meta); // free 24·N bytes

    // ── Pass 2 (sequential): emit (to_id, from_compact) directly ─────────────
    // Pre-sized to the exact total from Pass 1 — no reallocation, no per-object
    // allocation.
    let total_edges = fwd_off[n] as usize;
    let mut raw_edges: Vec<(u64, u32)> = Vec::with_capacity(total_edges);

    for (k, entry) in entries.iter().enumerate() {
        let from_compact = entry_to_compact[k];
        if from_compact == u32::MAX {
            continue;
        }
        match entry.tag {
            TAG_INSTANCE_DUMP => {
                let rec = parse_sub_record(hprof, entry)?;
                let inst = match rec {
                    SubRecord::InstanceDump(i) => i,
                    _ => continue,
                };
                collect_instance_ref_pairs(
                    &inst,
                    class_cache,
                    id_size,
                    from_compact,
                    &mut raw_edges,
                )?;
            }
            TAG_CLASS_DUMP => {
                let rec = parse_sub_record(hprof, entry)?;
                let cd = match rec {
                    SubRecord::ClassDump(c) => c,
                    _ => continue,
                };
                for sf_res in cd.static_fields() {
                    if let FieldValue::Object(id) = sf_res?.value
                        && id != 0
                    {
                        raw_edges.push((id, from_compact));
                    }
                }
            }
            TAG_OBJ_ARRAY_DUMP => {
                let rec = parse_sub_record(hprof, entry)?;
                let arr = match rec {
                    SubRecord::ObjArrayDump(a) => a,
                    _ => continue,
                };
                for id in arr.elements() {
                    if id != 0 {
                        raw_edges.push((id, from_compact));
                    }
                }
            }
            _ => {}
        }
    }
    drop(entry_to_compact);
    drop(entries);

    // ── Sort + two-pointer merge → fill fwd_edges directly ───────────────────
    // Eliminates the old "translated: Vec<(u32,u32)>" buffer entirely.
    raw_edges.sort_unstable_by_key(|&(to_id, _)| to_id);

    let mut fwd_edges: Vec<u32> = vec![0u32; total_edges];
    let mut cursor: Vec<u32> = fwd_off[..n].to_vec();

    let mut ci = 0usize; // pointer into compact_ids (ascending)
    let mut ei = 0usize; // pointer into raw_edges (ascending by to_id)
    while ei < raw_edges.len() && ci < n {
        let to_id = raw_edges[ei].0;
        match compact_ids[ci].cmp(&to_id) {
            std::cmp::Ordering::Less => ci += 1,
            std::cmp::Ordering::Greater => ei += 1, // to_id not in graph
            std::cmp::Ordering::Equal => {
                let to_compact = ci as u32;
                while ei < raw_edges.len() && raw_edges[ei].0 == to_id {
                    let from = raw_edges[ei].1 as usize;
                    let pos = cursor[from] as usize;
                    let limit = fwd_off[from + 1] as usize;
                    if pos < limit {
                        fwd_edges[pos] = to_compact;
                        cursor[from] += 1;
                    }
                    ei += 1;
                }
            }
        }
    }

    Ok((compact_ids, shallow_sizes, fwd_off, fwd_edges))
}

/// Count non-null outgoing object references in an instance dump without
/// allocating a collection.  Used by Pass 1 to build exact CSR out-degrees.
fn count_instance_refs(
    inst: &crate::heap_parser::InstanceDump<'_>,
    class_cache: &ClassCache,
    id_size: usize,
) -> Result<usize, HprofError> {
    let mut count = 0usize;
    let mut offset = 0usize;
    let mut curr = inst.class_id;
    while curr != 0 {
        let layout = match class_cache.get(curr) {
            Some(l) => l,
            None => break,
        };
        for &ft in &layout.field_types {
            let (value, sz) = read_field_value(inst.data, offset, ft, id_size)?;
            offset += sz;
            if matches!(value, FieldValue::Object(id) if id != 0) {
                count += 1;
            }
        }
        curr = layout.super_class_id;
    }
    Ok(count)
}

/// Append `(to_id, from_compact)` pairs for every non-null outgoing object
/// reference in an instance dump into an existing flat buffer.
/// Used by Pass 2; no intermediate allocation per object.
fn collect_instance_ref_pairs(
    inst: &crate::heap_parser::InstanceDump<'_>,
    class_cache: &ClassCache,
    id_size: usize,
    from_compact: u32,
    out: &mut Vec<(u64, u32)>,
) -> Result<(), HprofError> {
    let mut offset = 0usize;
    let mut curr = inst.class_id;
    while curr != 0 {
        let layout = match class_cache.get(curr) {
            Some(l) => l,
            None => break,
        };
        for &ft in &layout.field_types {
            let (value, sz) = read_field_value(inst.data, offset, ft, id_size)?;
            offset += sz;
            if let FieldValue::Object(id) = value
                && id != 0
            {
                out.push((id, from_compact));
            }
        }
        curr = layout.super_class_id;
    }
    Ok(())
}

fn prim_elem_byte_size(type_id: u8) -> u64 {
    match type_id {
        4 => 1,  // bool
        5 => 2,  // char
        6 => 4,  // float
        7 => 8,  // double
        8 => 1,  // byte
        9 => 2,  // short
        10 => 4, // int
        11 => 8, // long
        _ => 1,
    }
}

// ── Test-only: RawNode + build_forward_csr ────────────────────────────────────
// Used by make_compact_csr in the test module to construct small graphs inline.
// Not compiled in release builds.

/// One raw heap object: its ID, shallow size, and outbound reference IDs.
#[cfg(test)]
struct RawNode {
    object_id: u64,
    shallow: u64,
    out_ids: Vec<u64>,
}

/// Sort nodes by `object_id`, assign compact indices, and build a forward CSR.
/// Only compiled for tests; production code uses [`build_forward_csr_streaming`].
#[cfg(test)]
fn build_forward_csr(nodes: &mut Vec<RawNode>) -> (Vec<u64>, Vec<u64>, Vec<u32>, Vec<u32>) {
    nodes.sort_unstable_by_key(|n| n.object_id);
    nodes.dedup_by(|a, b| a.object_id == b.object_id);
    let n = nodes.len();

    let compact_ids: Vec<u64> = nodes.iter().map(|n| n.object_id).collect();
    let shallow_sizes: Vec<u64> = nodes.iter().map(|n| n.shallow).collect();

    let total_raw_edges: usize = nodes.iter().map(|n| n.out_ids.len()).sum();
    let mut raw_edges: Vec<(u64, u32)> = Vec::with_capacity(total_raw_edges);
    for (i, node) in nodes.iter().enumerate() {
        for &to_id in &node.out_ids {
            raw_edges.push((to_id, i as u32));
        }
    }
    raw_edges.sort_unstable_by_key(|&(to_id, _)| to_id);

    let mut fwd_off: Vec<u32> = Vec::with_capacity(n + 1);
    fwd_off.push(0u32);
    let mut cursor_out: Vec<u32> = vec![0u32; n];

    let mut ci = 0usize;
    let mut ei = 0usize;
    while ei < raw_edges.len() && ci < n {
        let to_id = raw_edges[ei].0;
        match compact_ids[ci].cmp(&to_id) {
            std::cmp::Ordering::Less => ci += 1,
            std::cmp::Ordering::Greater => ei += 1,
            std::cmp::Ordering::Equal => {
                while ei < raw_edges.len() && raw_edges[ei].0 == to_id {
                    cursor_out[raw_edges[ei].1 as usize] += 1;
                    ei += 1;
                }
            }
        }
    }
    for i in 0..n {
        fwd_off.push(fwd_off[i] + cursor_out[i]);
    }

    let edge_count = fwd_off[n] as usize;
    let mut fwd_edges: Vec<u32> = vec![0u32; edge_count];
    let mut cursor: Vec<u32> = fwd_off[..n].to_vec();

    raw_edges.sort_unstable_by_key(|&(to_id, _)| to_id);
    let mut ci = 0usize;
    let mut ei = 0usize;
    while ei < raw_edges.len() && ci < n {
        let to_id = raw_edges[ei].0;
        match compact_ids[ci].cmp(&to_id) {
            std::cmp::Ordering::Less => ci += 1,
            std::cmp::Ordering::Greater => ei += 1,
            std::cmp::Ordering::Equal => {
                let to_compact = ci as u32;
                while ei < raw_edges.len() && raw_edges[ei].0 == to_id {
                    let from = raw_edges[ei].1 as usize;
                    let pos = cursor[from] as usize;
                    fwd_edges[pos] = to_compact;
                    cursor[from] += 1;
                    ei += 1;
                }
            }
        }
    }

    (compact_ids, shallow_sizes, fwd_off, fwd_edges)
}

// ── Private: GC root collection ───────────────────────────────────────────────

/// Collect unique GC root object IDs as compact u32 indices.
///
/// IDs that don't appear in the combined index are silently dropped.
fn collect_gc_roots_compact(root_readers: &[RootIndexReader; 9], compact_ids: &[u64]) -> Vec<u32> {
    let mut seen_roots: Vec<bool> = vec![false; compact_ids.len()];
    let mut roots: Vec<u32> = Vec::new();
    for reader in root_readers {
        for entry in reader.iter() {
            if let Ok(idx) = compact_ids.binary_search(&entry.object_id)
                && !seen_roots[idx]
            {
                seen_roots[idx] = true;
                roots.push(idx as u32);
            }
        }
    }
    roots
}

// ── Private: RPO computation (compact) ───────────────────────────────────────

/// Compute the Reverse Post-Order (RPO) of the heap graph rooted at the
/// virtual GC root, working entirely in compact u32 index space.
///
/// Returns:
/// * `rpo_order[i]` — compact index of the node at RPO position `i`.
///   Position 0 is the virtual root (represented as `u32::MAX` in the returned
///   Vec; callers must handle this sentinel).
/// * `rpo_of[compact_idx]` — RPO position of that node (`u32::MAX` = unreachable).
fn compute_rpo_compact(
    n: usize,
    gc_roots: &[u32],
    fwd_off: &[u32],
    fwd_edges: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    // rpo_of[i] = u32::MAX means "not yet visited / unreachable"
    let mut rpo_of: Vec<u32> = vec![u32::MAX; n];
    let mut post_order: Vec<u32> = Vec::with_capacity(n + 1);

    // Stack entries: (node, child_cursor)
    // node = u32::MAX → virtual root; otherwise compact index.
    let mut stack: Vec<(u32, u32)> = vec![(u32::MAX, 0)];
    // Mark virtual root as visited with a sentinel RPO slot.
    // We use a separate visited array for compact indices.
    let mut visited: Vec<bool> = vec![false; n];

    while let Some(top) = stack.last_mut() {
        let (node, ref mut ci) = *top;

        let next_child: Option<u32> = if node == u32::MAX {
            // Virtual root: children = gc_roots
            gc_roots.get(*ci as usize).copied()
        } else {
            let start = fwd_off[node as usize] as usize;
            let end = fwd_off[node as usize + 1] as usize;
            let pos = start + *ci as usize;
            if pos < end {
                Some(fwd_edges[pos])
            } else {
                None
            }
        };

        if let Some(child) = next_child {
            *ci += 1;
            if (child as usize) < n && !visited[child as usize] {
                visited[child as usize] = true;
                stack.push((child, 0));
            }
        } else {
            let finished = stack.pop().map(|(n, _)| n).unwrap_or(u32::MAX);
            post_order.push(finished);
        }
    }

    // Reverse post-order: first entry is virtual root (u32::MAX), rest are
    // compact indices in dominator-algorithm order.
    let rpo_order: Vec<u32> = post_order.into_iter().rev().collect();

    // Fill rpo_of from rpo_order (skip virtual root at position 0).
    for (pos, &compact_idx) in rpo_order.iter().enumerate() {
        if compact_idx != u32::MAX {
            rpo_of[compact_idx as usize] = pos as u32;
        }
    }

    (rpo_order, rpo_of)
}

// ── Private: backward CSR (predecessor lists in RPO space) ───────────────────

/// Build a backward CSR in RPO space for the Cooper et al. algorithm.
///
/// For each reachable node `b` (RPO index 1..n_rpo), `pred_off[b]..pred_off[b+1]`
/// gives its predecessor RPO indices in `pred_edges`.
fn build_backward_csr(
    n_rpo: usize,
    gc_roots: &[u32],
    fwd_off: &[u32],
    fwd_edges: &[u32],
    rpo_of: &[u32],
    n_compact: usize,
) -> (Vec<u32>, Vec<u32>) {
    // Count in-degrees (in RPO space).
    let mut in_degree: Vec<u32> = vec![0u32; n_rpo];

    // Virtual root (RPO 0) → each GC root
    for &gc_root in gc_roots {
        let rpo = rpo_of[gc_root as usize];
        if rpo != u32::MAX {
            in_degree[rpo as usize] += 1;
        }
    }
    // Real edges
    for from_compact in 0..n_compact {
        let from_rpo = rpo_of[from_compact];
        if from_rpo == u32::MAX {
            continue;
        }
        let start = fwd_off[from_compact] as usize;
        let end = fwd_off[from_compact + 1] as usize;
        for &to_compact in &fwd_edges[start..end] {
            let to_rpo = rpo_of[to_compact as usize];
            if to_rpo != u32::MAX {
                in_degree[to_rpo as usize] += 1;
            }
        }
    }

    // Build offset array.
    let mut pred_off: Vec<u32> = Vec::with_capacity(n_rpo + 1);
    pred_off.push(0);
    for i in 0..n_rpo {
        pred_off.push(pred_off[i].saturating_add(in_degree[i]));
    }

    let total = pred_off[n_rpo] as usize;
    let mut pred_edges: Vec<u32> = vec![0u32; total];
    let mut cursor: Vec<u32> = pred_off[..n_rpo].to_vec();

    // Fill virtual-root predecessors.
    for &gc_root in gc_roots {
        let to_rpo = rpo_of[gc_root as usize];
        if to_rpo != u32::MAX {
            let pos = cursor[to_rpo as usize] as usize;
            pred_edges[pos] = 0; // virtual root is RPO 0
            cursor[to_rpo as usize] += 1;
        }
    }
    // Fill real-edge predecessors.
    for from_compact in 0..n_compact {
        let from_rpo = rpo_of[from_compact];
        if from_rpo == u32::MAX {
            continue;
        }
        let start = fwd_off[from_compact] as usize;
        let end = fwd_off[from_compact + 1] as usize;
        for &to_compact in &fwd_edges[start..end] {
            let to_rpo = rpo_of[to_compact as usize];
            if to_rpo != u32::MAX {
                let pos = cursor[to_rpo as usize] as usize;
                pred_edges[pos] = from_rpo;
                cursor[to_rpo as usize] += 1;
            }
        }
    }

    (pred_off, pred_edges)
}

// ── Private: Cooper et al. (2001) algorithm ───────────────────────────────────

/// Run the iterative dominator algorithm (Cooper et al. 2001) on a CSR
/// predecessor graph in RPO space.
///
/// Returns `doms[i]` = RPO index of node `i`'s immediate dominator.
/// `doms[0] = 0` (virtual root dominates itself).
/// Unreachable nodes have `doms[i] = u32::MAX`.
fn run_dominator(n: usize, pred_off: &[u32], pred_edges: &[u32]) -> Vec<u32> {
    const UNDEF: u32 = u32::MAX;
    let mut doms: Vec<u32> = vec![UNDEF; n];
    if n == 0 {
        return doms;
    }
    doms[0] = 0;

    let mut changed = true;
    while changed {
        changed = false;
        for b in 1..n {
            let p_start = pred_off[b] as usize;
            let p_end = pred_off[b + 1] as usize;
            if p_start == p_end {
                continue; // no predecessors → unreachable
            }
            let mut new_idom = UNDEF;
            for &p in &pred_edges[p_start..p_end] {
                if doms[p as usize] == UNDEF {
                    continue;
                }
                if new_idom == UNDEF {
                    new_idom = p;
                } else {
                    new_idom = intersect(p, new_idom, &doms);
                }
            }
            if new_idom != UNDEF && doms[b] != new_idom {
                doms[b] = new_idom;
                changed = true;
            }
        }
    }
    doms
}

/// Walk up both dominator chains until they meet, returning the common ancestor.
fn intersect(b1: u32, b2: u32, doms: &[u32]) -> u32 {
    const UNDEF: u32 = u32::MAX;
    let mut f1 = b1;
    let mut f2 = b2;
    while f1 != f2 {
        while f1 > f2 {
            match doms.get(f1 as usize) {
                Some(&d) if d != UNDEF => f1 = d,
                _ => return f2,
            }
        }
        while f2 > f1 {
            match doms.get(f2 as usize) {
                Some(&d) if d != UNDEF => f2 = d,
                _ => return f1,
            }
        }
    }
    f1
}

// ── Partition-parallel dominator computation ──────────────────────────────────
//
// Strategy:
//   1. Multi-source BFS to color each node with the compact index of its
//      unique reachable GC root, or MULTI_COLOR if reachable from > 1 root.
//   2. Run Cooper et al. in parallel over each exclusive partition.
//   3. Run Cooper et al. on the compressed MULTI subgraph (GC-root proxy
//      nodes + MULTI nodes) to dominate the shared nodes.
//   4. Compute retained sizes via Kahn's topological algorithm on the
//      assembled global dominator tree.

/// Node is directly dominated by the virtual GC root.
const VROOT_COMPACT: u32 = u32::MAX - 2;

/// Node is reachable from more than one GC root.
const MULTI_COLOR: u32 = u32::MAX;

/// Node has not yet been reached by the coloring BFS.
const COLOR_UNVISITED: u32 = u32::MAX - 1;

/// Assign each reachable node the compact index of its unique GC root, or
/// [`MULTI_COLOR`] if reachable from multiple roots.
///
/// Each node transitions at most twice (unvisited → single → MULTI), giving
/// O(N + E) total work.
fn color_nodes_by_root(n: usize, gc_roots: &[u32], fwd_off: &[u32], fwd_edges: &[u32]) -> Vec<u32> {
    let mut color: Vec<u32> = vec![COLOR_UNVISITED; n];
    let mut in_queue: Vec<bool> = vec![false; n];
    let mut queue: VecDeque<u32> = VecDeque::with_capacity(n.min(1 << 16));

    for &root in gc_roots {
        let r = root as usize;
        if color[r] == COLOR_UNVISITED {
            color[r] = root;
        } else if color[r] != root {
            color[r] = MULTI_COLOR;
        }
        if !in_queue[r] {
            in_queue[r] = true;
            queue.push_back(root);
        }
    }

    while let Some(node) = queue.pop_front() {
        in_queue[node as usize] = false;
        let node_color = color[node as usize];

        let start = fwd_off[node as usize] as usize;
        let end = fwd_off[node as usize + 1] as usize;
        for &child in &fwd_edges[start..end] {
            let ci = child as usize;
            let child_color = color[ci];
            let new_color = if child_color == COLOR_UNVISITED {
                node_color
            } else if child_color == MULTI_COLOR || child_color == node_color {
                continue; // stable
            } else {
                MULTI_COLOR
            };
            color[ci] = new_color;
            if !in_queue[ci] {
                in_queue[ci] = true;
                queue.push_back(child);
            }
        }
    }

    color
}

/// Run Cooper et al. on one exclusive partition (all nodes reachable only
/// from `root_compact`).  Returns `(compact_idx, dom_compact_idx)` pairs;
/// [`VROOT_COMPACT`] in the dom slot means "directly dominated by VROOT".
fn run_exclusive_partition(
    root_compact: u32,
    local_compact_list: &[u32],
    fwd_off: &[u32],
    fwd_edges: &[u32],
) -> Vec<(u32, u32)> {
    // Sort for O(log N) binary-search lookups during edge translation.
    let mut sorted: Vec<u32> = local_compact_list.to_vec();
    sorted.sort_unstable();
    let local_n = sorted.len();

    let local_of = |c: u32| -> Option<u32> { sorted.binary_search(&c).ok().map(|i| i as u32) };

    let local_root = match local_of(root_compact) {
        Some(l) => l,
        None => return vec![],
    };

    // Build local forward CSR (intra-partition edges only).
    let mut out_degree: Vec<u32> = vec![0; local_n];
    for (li, &compact) in sorted.iter().enumerate() {
        let start = fwd_off[compact as usize] as usize;
        let end = fwd_off[compact as usize + 1] as usize;
        for &to in &fwd_edges[start..end] {
            if local_of(to).is_some() {
                out_degree[li] += 1;
            }
        }
    }
    let mut local_off: Vec<u32> = Vec::with_capacity(local_n + 1);
    local_off.push(0);
    for i in 0..local_n {
        local_off.push(local_off[i] + out_degree[i]);
    }
    let edge_total = local_off[local_n] as usize;
    let mut local_edges: Vec<u32> = vec![0; edge_total];
    let mut cursor: Vec<u32> = local_off[..local_n].to_vec();
    for (li, &compact) in sorted.iter().enumerate() {
        let start = fwd_off[compact as usize] as usize;
        let end = fwd_off[compact as usize + 1] as usize;
        for &to in &fwd_edges[start..end] {
            if let Some(to_local) = local_of(to) {
                let pos = cursor[li] as usize;
                local_edges[pos] = to_local;
                cursor[li] += 1;
            }
        }
    }

    let gc_roots_local = vec![local_root];
    let (rpo_order, rpo_of) =
        compute_rpo_compact(local_n, &gc_roots_local, &local_off, &local_edges);
    let n_rpo = rpo_order.len();
    let (pred_off, pred_edges) = build_backward_csr(
        n_rpo,
        &gc_roots_local,
        &local_off,
        &local_edges,
        &rpo_of,
        local_n,
    );
    let doms = run_dominator(n_rpo, &pred_off, &pred_edges);

    let mut results: Vec<(u32, u32)> = Vec::with_capacity(n_rpo.saturating_sub(1));
    for i in 1..n_rpo {
        let local_idx = rpo_order[i] as usize;
        if local_idx >= local_n {
            continue;
        }
        let dom_rpo = doms[i];
        if dom_rpo == u32::MAX {
            continue;
        }
        let dom_compact = if dom_rpo == 0 {
            VROOT_COMPACT
        } else {
            let dom_local = rpo_order[dom_rpo as usize] as usize;
            sorted[dom_local]
        };
        results.push((sorted[local_idx], dom_compact));
    }
    results
}

/// Run Cooper et al. on the MULTI subgraph.
///
/// Local indices: `0..K` are GC-root proxy nodes (one per root); `K..K+M` are
/// the MULTI nodes.  An edge `proxy[R] → M` is emitted for every exclusive
/// node with color `R` that has an edge to MULTI node `M` in the full graph.
fn run_multi_partition(
    gc_roots: &[u32],
    multi_nodes: &[u32], // sorted compact indices of MULTI nodes
    n: usize,
    fwd_off: &[u32],
    fwd_edges: &[u32],
    color: &[u32],
) -> Vec<(u32, u32)> {
    let k = gc_roots.len();
    let m = multi_nodes.len();
    let local_n = k + m;

    // O(1) lookup: compact index of a GC root → proxy local index (0..K).
    // Uses a flat Vec of size n instead of a hash map.
    let mut proxy_of: Vec<u32> = vec![u32::MAX; n];
    for (i, &root) in gc_roots.iter().enumerate() {
        proxy_of[root as usize] = i as u32;
    }

    let multi_local =
        |c: u32| -> Option<u32> { multi_nodes.binary_search(&c).ok().map(|i| (k + i) as u32) };

    // Count out-degrees.
    let mut out_degree: Vec<u32> = vec![0; local_n];
    for v in 0..n {
        let v_color = color[v];
        if v_color == COLOR_UNVISITED {
            continue;
        }
        let from_local = if v_color == MULTI_COLOR {
            match multi_local(v as u32) {
                Some(l) => l as usize,
                None => continue,
            }
        } else {
            let p = proxy_of[v_color as usize];
            if p == u32::MAX {
                continue;
            }
            p as usize
        };
        let start = fwd_off[v] as usize;
        let end = fwd_off[v + 1] as usize;
        for &to in &fwd_edges[start..end] {
            if multi_local(to).is_some() {
                out_degree[from_local] += 1;
            }
        }
    }

    let mut local_off: Vec<u32> = Vec::with_capacity(local_n + 1);
    local_off.push(0);
    for i in 0..local_n {
        local_off.push(local_off[i] + out_degree[i]);
    }
    let edge_total = local_off[local_n] as usize;
    let mut local_edges: Vec<u32> = vec![0; edge_total];
    let mut cursor: Vec<u32> = local_off[..local_n].to_vec();

    for v in 0..n {
        let v_color = color[v];
        if v_color == COLOR_UNVISITED {
            continue;
        }
        let from_local = if v_color == MULTI_COLOR {
            match multi_local(v as u32) {
                Some(l) => l as usize,
                None => continue,
            }
        } else {
            let p = proxy_of[v_color as usize];
            if p == u32::MAX {
                continue;
            }
            p as usize
        };
        let start = fwd_off[v] as usize;
        let end = fwd_off[v + 1] as usize;
        for &to in &fwd_edges[start..end] {
            if let Some(to_local) = multi_local(to) {
                let pos = cursor[from_local] as usize;
                local_edges[pos] = to_local;
                cursor[from_local] += 1;
            }
        }
    }

    let gc_roots_local: Vec<u32> = (0..k as u32).collect();
    let (rpo_order, rpo_of) =
        compute_rpo_compact(local_n, &gc_roots_local, &local_off, &local_edges);
    let n_rpo = rpo_order.len();
    let (pred_off, pred_edges) = build_backward_csr(
        n_rpo,
        &gc_roots_local,
        &local_off,
        &local_edges,
        &rpo_of,
        local_n,
    );
    let doms = run_dominator(n_rpo, &pred_off, &pred_edges);

    let mut results: Vec<(u32, u32)> = Vec::with_capacity(m);
    for i in 1..n_rpo {
        let local_idx = rpo_order[i] as usize;
        if local_idx < k {
            continue; // proxy — dominated by VROOT, handled globally
        }
        let multi_offset = local_idx - k;
        if multi_offset >= m {
            continue;
        }
        let dom_rpo = doms[i];
        if dom_rpo == u32::MAX {
            continue;
        }
        let dom_compact = if dom_rpo == 0 {
            VROOT_COMPACT
        } else {
            let dom_local = rpo_order[dom_rpo as usize] as usize;
            if dom_local < k {
                // Dominated by a GC-root proxy → dominated by that GC root.
                gc_roots[dom_local]
            } else {
                multi_nodes[dom_local - k]
            }
        };
        results.push((multi_nodes[multi_offset], dom_compact));
    }
    results
}

/// Run dominator computation on all partitions in parallel, then assemble
/// the global dominator array indexed by compact index.
///
/// `global_doms[i]` = compact index of `i`'s immediate dominator,
/// [`VROOT_COMPACT`] for GC roots, `u32::MAX` for unreachable nodes.
fn run_partitioned_dominators(
    n: usize,
    gc_roots: &[u32],
    fwd_off: &[u32],
    fwd_edges: &[u32],
    color: &[u32],
) -> Vec<u32> {
    let mut global_doms: Vec<u32> = vec![u32::MAX; n];

    // Partition nodes by color using a sort (avoids HashMap).
    let mut exclusive_pairs: Vec<(u32, u32)> = Vec::new(); // (root_compact, node_compact)
    let mut multi_nodes: Vec<u32> = Vec::new();
    for (i, &c) in color.iter().enumerate().take(n) {
        match c {
            COLOR_UNVISITED => {}
            MULTI_COLOR => multi_nodes.push(i as u32),
            _ => exclusive_pairs.push((c, i as u32)),
        }
    }
    exclusive_pairs.sort_unstable_by_key(|&(c, _)| c);
    multi_nodes.sort_unstable();

    // Slice sorted list into per-root groups.
    let mut groups: Vec<(u32, Vec<u32>)> = Vec::new();
    let mut gi = 0;
    while gi < exclusive_pairs.len() {
        let root = exclusive_pairs[gi].0;
        let end = exclusive_pairs[gi..].partition_point(|&(c, _)| c == root) + gi;
        let nodes: Vec<u32> = exclusive_pairs[gi..end].iter().map(|&(_, n)| n).collect();
        groups.push((root, nodes));
        gi = end;
    }

    // Run exclusive partitions in parallel via rayon.
    let exclusive_results: Vec<Vec<(u32, u32)>> = groups
        .par_iter()
        .map(|(root_compact, nodes)| {
            run_exclusive_partition(*root_compact, nodes, fwd_off, fwd_edges)
        })
        .collect();
    for pairs in exclusive_results {
        for (compact_idx, dom_compact) in pairs {
            global_doms[compact_idx as usize] = dom_compact;
        }
    }

    // Run MULTI partition.
    if !multi_nodes.is_empty() {
        let multi_pairs = run_multi_partition(gc_roots, &multi_nodes, n, fwd_off, fwd_edges, color);
        for (compact_idx, dom_compact) in multi_pairs {
            global_doms[compact_idx as usize] = dom_compact;
        }
    }

    // GC roots are always dominated by the virtual root.
    for &root in gc_roots {
        global_doms[root as usize] = VROOT_COMPACT;
    }

    global_doms
}

/// Compute retained heap sizes via Kahn's topological algorithm on the
/// dominator tree.
///
/// Processes leaves first, accumulating sizes up the tree in O(N) time.
fn compute_retained_from_doms(n: usize, global_doms: &[u32], shallow_sizes: &[u64]) -> Vec<u64> {
    let mut retained: Vec<u64> = shallow_sizes.to_vec();

    // Count direct dominator-tree children per node.
    let mut child_count: Vec<u32> = vec![0; n];
    for &dom in global_doms.iter().take(n) {
        if (dom as usize) < n {
            child_count[dom as usize] += 1;
        }
    }

    // Seed queue with reachable leaves.
    let mut queue: VecDeque<u32> = VecDeque::new();
    for i in 0..n {
        if global_doms[i] != u32::MAX && child_count[i] == 0 {
            queue.push_back(i as u32);
        }
    }

    while let Some(v) = queue.pop_front() {
        let dom = global_doms[v as usize];
        if (dom as usize) >= n {
            continue; // VROOT_COMPACT or unreachable
        }
        let rv = retained[v as usize];
        retained[dom as usize] = retained[dom as usize].saturating_add(rv);
        child_count[dom as usize] -= 1;
        if child_count[dom as usize] == 0 {
            queue.push_back(dom);
        }
    }

    retained
}

// ── Private: output writing ───────────────────────────────────────────────────

fn write_outputs(
    n: usize,
    compact_ids: &[u64],
    global_doms: &[u32],
    retained: &[u64],
    dominators_out: &mut impl MMapWriter,
    retained_out: &mut impl MMapWriter,
) -> Result<(u64, u64), HprofError> {
    let mut dom_entries: Vec<(u64, u64)> = Vec::with_capacity(n);
    let mut ret_entries: Vec<(u64, u64)> = Vec::with_capacity(n);

    for i in 0..n {
        let dom = global_doms[i];
        if dom == u32::MAX {
            continue; // unreachable
        }
        let obj_id = compact_ids[i];
        let dom_id = if dom == VROOT_COMPACT {
            VIRTUAL_ROOT_ID
        } else {
            compact_ids[dom as usize]
        };
        dom_entries.push((obj_id, dom_id));
        ret_entries.push((obj_id, retained[i]));
    }

    dom_entries.sort_unstable_by_key(|&(id, _)| id);
    ret_entries.sort_unstable_by_key(|&(id, _)| id);

    let dom_count = dom_entries.len() as u64;
    let ret_count = ret_entries.len() as u64;

    let mut dom_writer = dominators_out.create_writer()?;
    for (obj_id, dom_id) in &dom_entries {
        let mut buf = [0u8; DOM_ENTRY_SIZE];
        buf[0..8].copy_from_slice(&obj_id.to_le_bytes());
        buf[8..16].copy_from_slice(&dom_id.to_le_bytes());
        dom_writer.write_all(&buf)?;
    }
    dom_writer.flush()?;

    let mut ret_writer = retained_out.create_writer()?;
    for (obj_id, ret) in &ret_entries {
        let mut buf = [0u8; RETAINED_ENTRY_SIZE];
        buf[0..8].copy_from_slice(&obj_id.to_le_bytes());
        buf[8..16].copy_from_slice(&ret.to_le_bytes());
        ret_writer.write_all(&buf)?;
    }
    ret_writer.flush()?;

    Ok((dom_count, ret_count))
}

// ── Private: byte utilities ───────────────────────────────────────────────────

fn read_u64_le(data: &[u8], offset: usize) -> u64 {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&data[offset..offset + 8]);
    u64::from_le_bytes(bytes)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heap_index::index_heap_dumps;
    use crate::heap_query::build_name_indexes;
    use crate::object_store::combine_sort_and_split;
    use crate::record_index::index_hprof;
    use crate::root_index::RootIndexReader;
    use crate::vfs::SubIndexDir;

    // ── Test hprof builder ────────────────────────────────────────────────────

    /// Build a minimal hprof file with the following object graph:
    ///
    /// ```text
    /// GC_ROOT_STICKY_CLASS → Class(0x100) [static field → Instance(0x200)]
    /// GC_ROOT_STICKY_CLASS → Class(0x300) [superclass of 0x100]
    /// Instance(0x200) [class=0x100, instance field → PrimArray(0x400)]
    /// PrimArray(0x400) [int[], 3 elements]
    /// ```
    ///
    /// Dominator tree:
    /// ```text
    /// VROOT → Class(0x100), Class(0x300)
    /// Class(0x100) → Instance(0x200)
    /// Instance(0x200) → PrimArray(0x400)
    /// ```
    ///
    /// Retained sizes (shallow using data.len() for instances, elem*size for arrays):
    /// - PrimArray(0x400): shallow = 3 * 4 = 12
    /// - Instance(0x200): shallow = 8 (one object-ref field = 8 bytes id_size)
    ///   retained = 8 + 12 = 20
    /// - Class(0x100): shallow = 0
    ///   retained = 0 + 20 = 20
    /// - Class(0x300): shallow = 0, retained = 0
    fn build_test_hprof() -> Vec<u8> {
        let id_size: u32 = 8;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"JAVA PROFILE 1.0.2\0");
        buf.extend_from_slice(&id_size.to_be_bytes());
        buf.extend_from_slice(&0u64.to_be_bytes()); // timestamp

        // Helper: write a top-level record
        let write_record = |buf: &mut Vec<u8>, tag: u8, body: &[u8]| {
            buf.push(tag);
            buf.extend_from_slice(&0u32.to_be_bytes()); // timestamp_delta
            buf.extend_from_slice(&(body.len() as u32).to_be_bytes());
            buf.extend_from_slice(body);
        };

        // UTF8 strings for class names
        // name_id=1 → "MyClass"
        let mut utf8_body = Vec::new();
        utf8_body.extend_from_slice(&1u64.to_be_bytes()); // name_id
        utf8_body.extend_from_slice(b"MyClass");
        write_record(&mut buf, 0x01, &utf8_body);

        // name_id=2 → "java/lang/Object"
        let mut utf8_body2 = Vec::new();
        utf8_body2.extend_from_slice(&2u64.to_be_bytes());
        utf8_body2.extend_from_slice(b"java/lang/Object");
        write_record(&mut buf, 0x01, &utf8_body2);

        // LOAD_CLASS: class_serial=1, class_id=0x100, stack_trace=1, name_id=1
        let mut lc_body = Vec::new();
        lc_body.extend_from_slice(&1u32.to_be_bytes()); // class_serial
        lc_body.extend_from_slice(&0x100u64.to_be_bytes()); // class_id
        lc_body.extend_from_slice(&1u32.to_be_bytes()); // stack_trace_serial
        lc_body.extend_from_slice(&1u64.to_be_bytes()); // class_name_id
        write_record(&mut buf, 0x02, &lc_body);

        // LOAD_CLASS: class_serial=2, class_id=0x300, name_id=2
        let mut lc_body2 = Vec::new();
        lc_body2.extend_from_slice(&2u32.to_be_bytes());
        lc_body2.extend_from_slice(&0x300u64.to_be_bytes());
        lc_body2.extend_from_slice(&1u32.to_be_bytes());
        lc_body2.extend_from_slice(&2u64.to_be_bytes());
        write_record(&mut buf, 0x02, &lc_body2);

        // HEAP_DUMP_SEGMENT containing all sub-records
        let mut heap = Vec::new();

        // ROOT_STICKY_CLASS(0x100) — marks Class 0x100 as a GC root
        heap.push(0x05u8); // TAG_ROOT_STICKY_CLASS
        heap.extend_from_slice(&0x100u64.to_be_bytes());

        // ROOT_STICKY_CLASS(0x300) — marks Class 0x300 as a GC root
        heap.push(0x05u8);
        heap.extend_from_slice(&0x300u64.to_be_bytes());

        // CLASS_DUMP(class_id=0x300, super=0, instance_size=0, no fields)
        {
            heap.push(0x20u8); // TAG_CLASS_DUMP
            heap.extend_from_slice(&0x300u64.to_be_bytes()); // class_id
            heap.extend_from_slice(&0u32.to_be_bytes()); // stack_serial
            heap.extend_from_slice(&0u64.to_be_bytes()); // super_class_id = 0
            heap.extend_from_slice(&0u64.to_be_bytes()); // class_loader_id
            heap.extend_from_slice(&0u64.to_be_bytes()); // signers_id
            heap.extend_from_slice(&0u64.to_be_bytes()); // domain_id
            heap.extend_from_slice(&0u64.to_be_bytes()); // reserved1
            heap.extend_from_slice(&0u64.to_be_bytes()); // reserved2
            heap.extend_from_slice(&0u32.to_be_bytes()); // instance_size
            heap.extend_from_slice(&0u16.to_be_bytes()); // cp_count = 0
            heap.extend_from_slice(&0u16.to_be_bytes()); // statics_count = 0
            heap.extend_from_slice(&0u16.to_be_bytes()); // instance_fields_count = 0
        }

        // CLASS_DUMP(class_id=0x100, super=0x300, one instance field: obj-ref)
        // one static field: Object → 0x200
        {
            heap.push(0x20u8);
            heap.extend_from_slice(&0x100u64.to_be_bytes()); // class_id
            heap.extend_from_slice(&0u32.to_be_bytes()); // stack_serial
            heap.extend_from_slice(&0x300u64.to_be_bytes()); // super_class_id
            heap.extend_from_slice(&0u64.to_be_bytes()); // class_loader_id
            heap.extend_from_slice(&0u64.to_be_bytes()); // signers_id
            heap.extend_from_slice(&0u64.to_be_bytes()); // domain_id
            heap.extend_from_slice(&0u64.to_be_bytes()); // reserved1
            heap.extend_from_slice(&0u64.to_be_bytes()); // reserved2
            heap.extend_from_slice(&8u32.to_be_bytes()); // instance_size = 8
            heap.extend_from_slice(&0u16.to_be_bytes()); // cp_count = 0
            // 1 static field: name_id=1, type=2 (object), value=0x200
            heap.extend_from_slice(&1u16.to_be_bytes()); // statics_count = 1
            heap.extend_from_slice(&1u64.to_be_bytes()); // static field name_id
            heap.push(2u8); // type = Object
            heap.extend_from_slice(&0x200u64.to_be_bytes()); // value = 0x200
            // 1 instance field: name_id=1, type=2 (object-ref, to PrimArray 0x400)
            heap.extend_from_slice(&1u16.to_be_bytes()); // instance_fields_count = 1
            heap.extend_from_slice(&1u64.to_be_bytes()); // field name_id
            heap.push(2u8); // type = Object
        }

        // INSTANCE_DUMP(object_id=0x200, class=0x100)
        // data = object-ref 0x400 (the prim array)
        {
            heap.push(0x21u8); // TAG_INSTANCE_DUMP
            heap.extend_from_slice(&0x200u64.to_be_bytes()); // object_id
            heap.extend_from_slice(&0u32.to_be_bytes()); // stack_serial
            heap.extend_from_slice(&0x100u64.to_be_bytes()); // class_id
            heap.extend_from_slice(&8u32.to_be_bytes()); // data_length = 8 (one id-ref)
            heap.extend_from_slice(&0x400u64.to_be_bytes()); // field value = 0x400
        }

        // PRIM_ARRAY_DUMP(array_id=0x400, int[], 3 elements)
        {
            heap.push(0x23u8); // TAG_PRIM_ARRAY_DUMP
            heap.extend_from_slice(&0x400u64.to_be_bytes()); // array_id
            heap.extend_from_slice(&0u32.to_be_bytes()); // stack_serial
            heap.extend_from_slice(&3u32.to_be_bytes()); // num_elements = 3
            heap.push(10u8); // element_type = int (4 bytes)
            heap.extend_from_slice(&[0u8; 12]); // 3 × 4 bytes
        }

        write_record(&mut buf, 0x1Cu8, &heap);
        buf
    }

    fn build_all_indexes(hprof: &Vec<u8>) -> (Vec<u8>, Vec<u8>, Vec<u8>, [Vec<u8>; 9]) {
        let mut record_idx = Vec::new();
        let heap_dir = SubIndexDir::mem();
        index_hprof(hprof, &mut record_idx).unwrap();
        index_heap_dumps(hprof, &record_idx, &heap_dir).unwrap();

        let mut object_store = Vec::new();
        let mut r_unknown = Vec::new();
        let mut r_jni_global = Vec::new();
        let mut r_jni_local = Vec::new();
        let mut r_java_frame = Vec::new();
        let mut r_native_stack = Vec::new();
        let mut r_sticky = Vec::new();
        let mut r_thread_block = Vec::new();
        let mut r_monitor_used = Vec::new();
        let mut r_thread_obj = Vec::new();
        combine_sort_and_split(
            &heap_dir,
            &mut object_store,
            &mut [
                &mut r_unknown,
                &mut r_jni_global,
                &mut r_jni_local,
                &mut r_java_frame,
                &mut r_native_stack,
                &mut r_sticky,
                &mut r_thread_block,
                &mut r_monitor_used,
                &mut r_thread_obj,
            ],
        )
        .unwrap();

        let mut utf8 = Vec::new();
        let mut lc = Vec::new();
        build_name_indexes(hprof, &record_idx, &mut utf8, &mut lc).unwrap();

        let root_bytes = [
            r_unknown,
            r_jni_global,
            r_jni_local,
            r_java_frame,
            r_native_stack,
            r_sticky,
            r_thread_block,
            r_monitor_used,
            r_thread_obj,
        ];
        (object_store, utf8, lc, root_bytes)
    }

    #[test]
    fn dominator_and_retained_basic() {
        let hprof = build_test_hprof();
        let (object_store, _utf8, _lc, root_bytes) = build_all_indexes(&hprof);

        let root_readers = [
            RootIndexReader::from_ref(&root_bytes[0]).unwrap(),
            RootIndexReader::from_ref(&root_bytes[1]).unwrap(),
            RootIndexReader::from_ref(&root_bytes[2]).unwrap(),
            RootIndexReader::from_ref(&root_bytes[3]).unwrap(),
            RootIndexReader::from_ref(&root_bytes[4]).unwrap(),
            RootIndexReader::from_ref(&root_bytes[5]).unwrap(),
            RootIndexReader::from_ref(&root_bytes[6]).unwrap(),
            RootIndexReader::from_ref(&root_bytes[7]).unwrap(),
            RootIndexReader::from_ref(&root_bytes[8]).unwrap(),
        ];

        let mut dominators = Vec::new();
        let mut retained = Vec::new();

        let (dom_count, ret_count) = build_dominator_and_retained(
            &hprof,
            &object_store,
            &root_readers,
            &mut dominators,
            &mut retained,
        )
        .unwrap();

        assert!(dom_count > 0, "expected dominator entries");
        assert_eq!(dom_count, ret_count);

        let dom_idx = DominatorIndex::from_ref(&dominators).unwrap();
        let ret_idx = RetainedIndex::from_ref(&retained).unwrap();

        // Class(0x100) and Class(0x300) are GC roots → dominated by VROOT
        assert_eq!(dom_idx.find(0x100), Some(VIRTUAL_ROOT_ID));
        assert_eq!(dom_idx.find(0x300), Some(VIRTUAL_ROOT_ID));

        // Instance(0x200) is dominated by Class(0x100) (via static field)
        assert_eq!(dom_idx.find(0x200), Some(0x100));

        // PrimArray(0x400) is dominated by Instance(0x200)
        assert_eq!(dom_idx.find(0x400), Some(0x200));

        // PrimArray(0x400): 3 int elements = 12 bytes shallow = 12 retained
        assert_eq!(ret_idx.find(0x400), Some(12));

        // Instance(0x200): 8 bytes shallow + 12 retained from 0x400 = 20
        assert_eq!(ret_idx.find(0x200), Some(20));

        // Class(0x100): 0 shallow + 20 retained from 0x200 = 20
        assert_eq!(ret_idx.find(0x100), Some(20));

        // Class(0x300): no children → 0 retained
        assert_eq!(ret_idx.find(0x300), Some(0));
    }

    /// Build a compact forward CSR for a simple graph described as a list of
    /// directed edges `(from_object_id, to_object_id)` and a list of all
    /// node object IDs (including isolated nodes).
    ///
    /// Returns `(compact_ids, fwd_off, fwd_edges)` suitable for the compact
    /// RPO / dominator functions.
    fn make_compact_csr(
        all_ids: &[u64],
        edges: &[(u64, u64)],
    ) -> (Vec<u64>, Vec<u64>, Vec<u32>, Vec<u32>) {
        let mut nodes: Vec<RawNode> = all_ids
            .iter()
            .map(|&id| RawNode {
                object_id: id,
                shallow: 0,
                out_ids: edges
                    .iter()
                    .filter(|&&(f, _)| f == id)
                    .map(|&(_, t)| t)
                    .collect(),
            })
            .collect();
        build_forward_csr(&mut nodes)
    }

    #[test]
    fn rpo_computation_linear_chain() {
        // VROOT → A(id=1) → B(id=2) → C(id=3)
        let (compact_ids, _shallow, fwd_off, fwd_edges) =
            make_compact_csr(&[1, 2, 3], &[(1, 2), (2, 3)]);

        // compact_ids[0]=1, [1]=2, [2]=3  (sorted by ID)
        let gc_roots_compact: Vec<u32> = vec![0u32]; // id=1 → compact index 0
        let (rpo_order, rpo_of) = compute_rpo_compact(3, &gc_roots_compact, &fwd_off, &fwd_edges);

        // rpo_order[0] = virtual root sentinel
        assert_eq!(rpo_order[0], u32::MAX, "RPO[0] should be virtual root");
        // rpo_order[1..] = compact indices in DFS order: 0, 1, 2
        assert_eq!(rpo_order[1], 0); // compact idx 0 = id 1
        assert_eq!(rpo_order[2], 1);
        assert_eq!(rpo_order[3], 2);

        assert_eq!(rpo_of[0], 1); // compact idx 0 → RPO pos 1
        assert_eq!(rpo_of[1], 2);
        assert_eq!(rpo_of[2], 3);

        let _ = compact_ids;
    }

    #[test]
    fn dominator_linear_chain() {
        // VROOT → A → B → C
        let (compact_ids, _shallow, fwd_off, fwd_edges) =
            make_compact_csr(&[10, 20, 30], &[(10, 20), (20, 30)]);
        let gc_roots: Vec<u32> = vec![0]; // id=10 → compact 0
        let n = compact_ids.len();

        let (rpo_order, rpo_of) = compute_rpo_compact(n, &gc_roots, &fwd_off, &fwd_edges);
        let n_rpo = rpo_order.len();
        let (pred_off, pred_edges) =
            build_backward_csr(n_rpo, &gc_roots, &fwd_off, &fwd_edges, &rpo_of, n);
        let doms = run_dominator(n_rpo, &pred_off, &pred_edges);

        // doms[0] = 0 (VROOT dominates itself)
        // doms[1] = 0 (A ← VROOT)
        // doms[2] = 1 (B ← A)
        // doms[3] = 2 (C ← B)
        assert_eq!(doms[0], 0);
        assert_eq!(doms[1], 0);
        assert_eq!(doms[2], 1);
        assert_eq!(doms[3], 2);
    }

    #[test]
    fn retained_size_linear_chain() {
        // VROOT → A(10 bytes) → B(20 bytes) → C(30 bytes)
        // retained: C=30, B=50, A=60
        let mut nodes = vec![
            RawNode {
                object_id: 1,
                shallow: 10,
                out_ids: vec![2],
            },
            RawNode {
                object_id: 2,
                shallow: 20,
                out_ids: vec![3],
            },
            RawNode {
                object_id: 3,
                shallow: 30,
                out_ids: vec![],
            },
        ];
        let (compact_ids, shallow_sizes, fwd_off, fwd_edges) = build_forward_csr(&mut nodes);
        let gc_roots: Vec<u32> = vec![0]; // id=1 → compact 0
        let n = compact_ids.len();

        let color = color_nodes_by_root(n, &gc_roots, &fwd_off, &fwd_edges);
        let global_doms = run_partitioned_dominators(n, &gc_roots, &fwd_off, &fwd_edges, &color);
        let retained = compute_retained_from_doms(n, &global_doms, &shallow_sizes);

        // compact: id=1→0, id=2→1, id=3→2
        assert_eq!(retained[2], 30, "C retained"); // compact 2 = id=3
        assert_eq!(retained[1], 50, "B retained"); // compact 1 = id=2
        assert_eq!(retained[0], 60, "A retained"); // compact 0 = id=1

        let _ = compact_ids;
    }

    #[test]
    fn dominator_index_binary_search() {
        let mut data = Vec::new();
        for &(id, dom) in &[(10u64, 0u64), (20, 10), (30, 10)] {
            data.extend_from_slice(&id.to_le_bytes());
            data.extend_from_slice(&dom.to_le_bytes());
        }
        let idx = DominatorIndex::from_ref(&data).unwrap();
        assert_eq!(idx.find(10), Some(0));
        assert_eq!(idx.find(20), Some(10));
        assert_eq!(idx.find(30), Some(10));
        assert_eq!(idx.find(99), None);
        assert_eq!(idx.len(), 3);
    }

    #[test]
    fn retained_index_binary_search() {
        let mut data = Vec::new();
        for &(id, ret) in &[(5u64, 100u64), (15, 200), (25, 300)] {
            data.extend_from_slice(&id.to_le_bytes());
            data.extend_from_slice(&ret.to_le_bytes());
        }
        let idx = RetainedIndex::from_ref(&data).unwrap();
        assert_eq!(idx.find(5), Some(100));
        assert_eq!(idx.find(15), Some(200));
        assert_eq!(idx.find(25), Some(300));
        assert_eq!(idx.find(0), None);
        assert!(idx.find(99).is_none());
    }

    #[test]
    fn intersect_converges() {
        // Linear chain in RPO space: doms[0]=0, doms[1]=0, doms[2]=1, doms[3]=2
        let doms: Vec<u32> = vec![0, 0, 1, 2];
        assert_eq!(intersect(3, 2, &doms), 2);
        assert_eq!(intersect(3, 1, &doms), 1);
        assert_eq!(intersect(3, 0, &doms), 0);
        assert_eq!(intersect(2, 1, &doms), 1);
    }
}
