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
//! This is the one build step whose private memory grows with the heap:
//! O(N + E) where N is the number of heap objects and E the number of object
//! references.  What is resident, by phase (N objects, E references):
//!
//! | Phase | Resident |
//! |---|---|
//! | CSR build, pass 1 | ≈ 24·N (per-chunk metadata, consumed and freed in order) |
//! | CSR build, pass 2 + merge | 24·N + 4·E (the `(to, from)` pairs live in a scratch store entry, sorted on the store) |
//! | after the build | 20·N + 4·E (ids, shallow sizes, offsets, edges) |
//! | colouring + partition dominators | + up to ≈ 52·N + 8·E for the largest partition |
//!
//! [`estimate_memory_bytes`] turns that into a single conservative figure
//! (`72·N + 12·E`, ≈ 20 % above the measured peak on a 767 MB dump) that the
//! pipeline checks against available memory before starting; see
//! [`check_memory`].  Beyond that only an out-of-core algorithm helps (plan
//! task S6.1).

use crate::class_layout::{
    ClassCache, build_class_cache, count_instance_refs, for_each_instance_ref,
};
use crate::heap_index::sub_record::{
    SubIndexEntry, TAG_CLASS_DUMP, TAG_INSTANCE_DUMP, TAG_OBJ_ARRAY_DUMP, TAG_PRIM_ARRAY_DUMP,
};
use crate::heap_parser::record::FieldValue;
use crate::heap_parser::{SubIndexReader, SubRecord, parse_sub_record};
use crate::hprof::{HprofError, HprofFile};
use crate::index::{Entry, IndexStore, RecordFile, RecordWriter, names, read_u64_le};
use crate::root_index::RootIndexReader;
use rayon::prelude::*;
use std::collections::VecDeque;

// ── Entry format ──────────────────────────────────────────────────────────────

/// Byte size of one entry in `dominators.bin`.
pub const DOM_ENTRY_SIZE: usize = 16;

/// Byte size of one entry in `retained.bin`.
pub const RETAINED_ENTRY_SIZE: usize = 16;

/// The dominator_id written for objects that are dominated only by the virtual
/// GC root (i.e. direct GC roots themselves).  Zero is not a valid Java object
/// ID in any hprof file.
pub const VIRTUAL_ROOT_ID: u64 = 0;

// ── Memory estimate and guard ─────────────────────────────────────────────────

/// Bytes of private memory per object in [`estimate_memory_bytes`].
const BYTES_PER_OBJECT: u64 = 72;
/// Bytes of private memory per object reference in [`estimate_memory_bytes`].
const BYTES_PER_REFERENCE: u64 = 12;

/// Conservative estimate of the peak private memory
/// [`build_dominator_and_retained`] needs for a heap with `objects`
/// sub-records and `references` object references.
///
/// Calibrated on a 767 MB dump (9.7 M objects, 18.2 M references, measured
/// peak 726 MB vs. estimate 875 MB).
pub fn estimate_memory_bytes(objects: u64, references: u64) -> u64 {
    objects
        .saturating_mul(BYTES_PER_OBJECT)
        .saturating_add(references.saturating_mul(BYTES_PER_REFERENCE))
}

/// Memory currently available to this process, when the platform reports it.
pub fn available_memory_bytes() -> Option<u64> {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    match sys.available_memory() {
        0 => None,
        n => Some(n),
    }
}

/// Fail fast when the dominator step would not fit.
///
/// `limit` overrides the detected available memory (`None` = detect; if the
/// platform reports nothing the check passes).  Returns the estimate on
/// success so callers can report it.
pub fn check_memory(objects: u64, references: u64, limit: Option<u64>) -> Result<u64, HprofError> {
    let needed = estimate_memory_bytes(objects, references);
    if let Some(available) = limit.or_else(available_memory_bytes)
        && needed > available
    {
        return Err(HprofError::InsufficientMemory {
            step: "the dominator tree and retained sizes",
            needed,
            available,
        });
    }
    Ok(needed)
}

// ── Public build function ─────────────────────────────────────────────────────

/// Build the dominator/retained index family for a heap dump.
///
/// Requires the combined object store index and all nine GC root index readers
/// to be available.  Writes `dominators.bin`, `retained.bin`,
/// `retained_by_size.bin` and `dominator_children.bin` into `store`.
///
/// Returns `(dominator_entry_count, retained_entry_count)`.
///
/// **Memory usage:** see the module docs; call [`check_memory`] first to fail
/// fast instead of exhausting memory.
pub fn build_dominator_and_retained(
    hprof_bytes: &[u8],
    combined_slice: &[u8],
    root_readers: &[RootIndexReader<'_>; 9],
    store: &dyn IndexStore,
) -> Result<(u64, u64), HprofError> {
    // Open indexes.
    let hprof = HprofFile::from_ref(hprof_bytes)?;
    let combined = SubIndexReader::from_ref(combined_slice)?;
    let combined_file = RecordFile::<SubIndexEntry>::new(combined_slice)?;

    // A leftover scratch entry from an interrupted run is never valid.
    store.remove_prefix(&names::part_prefix(names::DOMINATORS))?;

    // ── Pass 1 (sequential): build class field-layout cache ───────────────────
    // Eliminates O(depth × log N) binary searches per instance in pass 2.
    let class_cache = build_class_cache(&hprof, &combined)?;

    // ── Streaming two-pass CSR build ──────────────────────────────────────────
    let (compact_ids, shallow_sizes, fwd_off, fwd_edges) =
        build_forward_csr_streaming(&hprof, combined_file, &class_cache, store)?;
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
    write_outputs(n, &compact_ids, &global_doms, &retained, store)
}

// ── Public reader: DominatorIndex ─────────────────────────────────────────────

/// One `dominators.bin` record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DomEntry {
    pub object_id: u64,
    /// [`VIRTUAL_ROOT_ID`] when the object is a direct GC root.
    pub dominator_id: u64,
}

impl Entry for DomEntry {
    const SIZE: usize = DOM_ENTRY_SIZE;
    const KEY_OFFSET: usize = 0;

    fn from_bytes(b: &[u8]) -> Self {
        Self {
            object_id: read_u64_le(b, 0),
            dominator_id: read_u64_le(b, 8),
        }
    }

    fn write_to(&self, out: &mut [u8]) {
        out[0..8].copy_from_slice(&self.object_id.to_le_bytes());
        out[8..16].copy_from_slice(&self.dominator_id.to_le_bytes());
    }

    fn key(&self) -> u64 {
        self.object_id
    }
}

/// One `retained.bin` record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetainedEntry {
    pub object_id: u64,
    pub retained_bytes: u64,
}

impl Entry for RetainedEntry {
    const SIZE: usize = RETAINED_ENTRY_SIZE;
    const KEY_OFFSET: usize = 0;

    fn from_bytes(b: &[u8]) -> Self {
        Self {
            object_id: read_u64_le(b, 0),
            retained_bytes: read_u64_le(b, 8),
        }
    }

    fn write_to(&self, out: &mut [u8]) {
        out[0..8].copy_from_slice(&self.object_id.to_le_bytes());
        out[8..16].copy_from_slice(&self.retained_bytes.to_le_bytes());
    }

    fn key(&self) -> u64 {
        self.object_id
    }
}

/// One `retained_by_size.bin` record: the retained index re-keyed by size,
/// sorted **descending** (largest retained size first).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetainedBySizeEntry {
    pub retained_bytes: u64,
    pub object_id: u64,
}

impl Entry for RetainedBySizeEntry {
    const SIZE: usize = 16;
    const KEY_OFFSET: usize = 0;

    fn from_bytes(b: &[u8]) -> Self {
        Self {
            retained_bytes: read_u64_le(b, 0),
            object_id: read_u64_le(b, 8),
        }
    }

    fn write_to(&self, out: &mut [u8]) {
        out[0..8].copy_from_slice(&self.retained_bytes.to_le_bytes());
        out[8..16].copy_from_slice(&self.object_id.to_le_bytes());
    }

    fn key(&self) -> u64 {
        self.retained_bytes
    }
}

/// One `dominator_children.bin` record: the dominator tree keyed by parent,
/// sorted by `(dominator_id, object_id)`, so the children of `X` are
/// `range(X)`.  Children of the virtual root have `dominator_id ==`
/// [`VIRTUAL_ROOT_ID`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DomChildEntry {
    pub dominator_id: u64,
    pub object_id: u64,
}

impl Entry for DomChildEntry {
    const SIZE: usize = 16;
    const KEY_OFFSET: usize = 0;

    fn from_bytes(b: &[u8]) -> Self {
        Self {
            dominator_id: read_u64_le(b, 0),
            object_id: read_u64_le(b, 8),
        }
    }

    fn write_to(&self, out: &mut [u8]) {
        out[0..8].copy_from_slice(&self.dominator_id.to_le_bytes());
        out[8..16].copy_from_slice(&self.object_id.to_le_bytes());
    }

    fn key(&self) -> u64 {
        self.dominator_id
    }
}

/// Read-only handle to a sorted `dominators.bin` index file.
///
/// Each entry stores `(object_id, dominator_id)`.  A `dominator_id` of
/// [`VIRTUAL_ROOT_ID`] (`0`) means the object is a direct GC root.
#[derive(Clone, Copy)]
pub struct DominatorIndex<'a> {
    file: RecordFile<'a, DomEntry>,
}

impl<'a> DominatorIndex<'a> {
    pub fn from_ref(bytes: &'a [u8]) -> Result<Self, HprofError> {
        Ok(Self {
            file: RecordFile::new(bytes)?,
        })
    }

    pub(crate) fn from_slice(bytes: &'a [u8]) -> Self {
        Self {
            file: RecordFile::from_slice(bytes),
        }
    }

    /// Return the `dominator_id` of `object_id`, or `None` if not found.
    pub fn find(&self, object_id: u64) -> Option<u64> {
        self.file.find(object_id).map(|e| e.dominator_id)
    }
}

// ── Public reader: RetainedIndex ──────────────────────────────────────────────

/// Read-only handle to a sorted `retained.bin` index file.
///
/// Each entry stores `(object_id, retained_bytes)`.
#[derive(Clone, Copy)]
pub struct RetainedIndex<'a> {
    file: RecordFile<'a, RetainedEntry>,
}

impl<'a> RetainedIndex<'a> {
    /// Create a validated reader from a byte slice.
    pub fn from_ref(data: &'a [u8]) -> Result<Self, HprofError> {
        Ok(Self {
            file: RecordFile::new(data)?,
        })
    }

    /// Create a reader from a slice already known to be valid.
    pub(crate) fn from_slice(data: &'a [u8]) -> Self {
        Self {
            file: RecordFile::from_slice(data),
        }
    }

    /// Return the retained heap size in bytes for `object_id`, or `None`.
    pub fn find(&self, object_id: u64) -> Option<u64> {
        self.file.find(object_id).map(|e| e.retained_bytes)
    }
}

// ── Private: streaming two-pass CSR builder ───────────────────────────────────
//
// **Pass 1** (parallel, chunked): for every heap record, read its outgoing
//   references to count them exactly, producing (object_id, shallow,
//   out_degree, entry_idx) — 24 bytes per object — in per-chunk vectors.
//   The object store is sorted by object id, so the chunks are already in
//   global order: they are consumed one after another straight into
//   `compact_ids` / `shallow_sizes` / `fwd_off` / `entry_to_compact`, and
//   each chunk is freed as soon as it has been consumed.  The peak of this
//   phase is therefore ≈ 24·N bytes (chunks shrinking as the outputs grow),
//   with no sort and no second copy.
//
// **Pass 2** (sequential): re-read each record and stream
//   (to_id, from_compact) pairs into a scratch entry of the index store
//   (`dominators.bin.part.0000`).  The pairs never live in process memory;
//   on disk they are sorted in place on the store (a file-backed map).
//
// **Merge**: a two-pointer scan of the sorted pairs against `compact_ids`
//   fills `fwd_edges`.  References to ids that are not in the dump
//   (dangling) leave unfilled slots, which `compact_csr` squeezes out.
//
// Resident after the build: 20·N + 4·E bytes (compact_ids, shallow_sizes,
// fwd_off, fwd_edges).

// (compact_ids, shallow_sizes, fwd_offsets, fwd_edges)
type ForwardCsr = (Vec<u64>, Vec<u64>, Vec<u32>, Vec<u32>);
// (object_id, shallow_size, out_degree, entry_index)
type NodeMeta = (u64, u64, u32, u32);

/// One `(to_id, from_compact)` reference in the scratch entry written by
/// pass 2.  Sorted ascending by `to_id`.
#[derive(Debug, Clone, Copy)]
struct EdgeEntry {
    to_id: u64,
    from: u64,
}

impl Entry for EdgeEntry {
    const SIZE: usize = 16;
    const KEY_OFFSET: usize = 0;

    fn from_bytes(b: &[u8]) -> Self {
        Self {
            to_id: read_u64_le(b, 0),
            from: read_u64_le(b, 8),
        }
    }

    fn write_to(&self, out: &mut [u8]) {
        out[0..8].copy_from_slice(&self.to_id.to_le_bytes());
        out[8..16].copy_from_slice(&self.from.to_le_bytes());
    }

    fn key(&self) -> u64 {
        self.to_id
    }
}

/// Pass-1 record for one object-store entry, or `None` for GC-root records.
fn node_meta(
    hprof: &HprofFile,
    entry: &SubIndexEntry,
    k: usize,
    class_cache: &ClassCache,
    id_size: usize,
) -> Result<Option<NodeMeta>, HprofError> {
    let k = k as u32;
    match entry.tag {
        TAG_INSTANCE_DUMP => {
            let SubRecord::InstanceDump(inst) = parse_sub_record(hprof, entry)? else {
                return Ok(None);
            };
            let shallow = inst.data.len() as u64;
            let count = count_instance_refs(&inst, class_cache, id_size)? as u32;
            Ok(Some((inst.object_id, shallow, count, k)))
        }
        TAG_CLASS_DUMP => {
            let SubRecord::ClassDump(cd) = parse_sub_record(hprof, entry)? else {
                return Ok(None);
            };
            let mut count = 0u32;
            for sf_res in cd.static_fields() {
                if let FieldValue::Object(id) = sf_res?.value
                    && id != 0
                {
                    count += 1;
                }
            }
            Ok(Some((cd.class_id, 0u64, count, k)))
        }
        TAG_OBJ_ARRAY_DUMP => {
            let SubRecord::ObjArrayDump(arr) = parse_sub_record(hprof, entry)? else {
                return Ok(None);
            };
            let shallow = arr.num_elements as u64 * id_size as u64;
            let count = arr.elements().filter(|&id| id != 0).count() as u32;
            Ok(Some((arr.array_id, shallow, count, k)))
        }
        TAG_PRIM_ARRAY_DUMP => {
            let SubRecord::PrimArrayDump(arr) = parse_sub_record(hprof, entry)? else {
                return Ok(None);
            };
            let shallow = arr.num_elements as u64 * prim_elem_byte_size(arr.element_type);
            Ok(Some((arr.array_id, shallow, 0u32, k)))
        }
        _ => Ok(None),
    }
}

fn build_forward_csr_streaming(
    hprof: &HprofFile,
    combined: RecordFile<'_, SubIndexEntry>,
    class_cache: &ClassCache,
    store: &dyn IndexStore,
) -> Result<ForwardCsr, HprofError> {
    let id_size = hprof.id_size() as usize;
    let entry_count = combined.len();
    if entry_count >= u32::MAX as usize {
        return Err(HprofError::Internal(format!(
            "object store has {entry_count} entries; the dominator builder indexes objects with u32"
        )));
    }

    // ── Pass 1 (parallel, chunked): count exact out-degrees ───────────────────
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_len = entry_count.div_ceil(n_threads * 4).max(1);
    let n_chunks = entry_count.div_ceil(chunk_len);
    let chunks: Vec<Vec<NodeMeta>> = (0..n_chunks)
        .into_par_iter()
        .map(|c| -> Result<Vec<NodeMeta>, HprofError> {
            let start = c * chunk_len;
            let end = (start + chunk_len).min(entry_count);
            let mut out = Vec::with_capacity(end - start);
            for (i, entry) in combined.iter_range(start, end).enumerate() {
                if let Some(m) = node_meta(hprof, &entry, start + i, class_cache, id_size)? {
                    out.push(m);
                }
            }
            Ok(out)
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Consume the chunks in order (they are globally sorted by object id: the
    // object store is), freeing each one as soon as it has been consumed.
    let total_meta: usize = chunks.iter().map(Vec::len).sum();
    let mut compact_ids: Vec<u64> = Vec::with_capacity(total_meta);
    let mut shallow_sizes: Vec<u64> = Vec::with_capacity(total_meta);
    let mut fwd_off: Vec<u32> = Vec::with_capacity(total_meta + 1);
    fwd_off.push(0);
    // Map original entry index → compact index for O(1) lookup in Pass 2.
    let mut entry_to_compact: Vec<u32> = vec![u32::MAX; entry_count];
    let mut running = 0u64;
    for chunk in chunks {
        for (id, shallow, degree, entry_idx) in chunk {
            if let Some(&last) = compact_ids.last() {
                if id == last {
                    continue; // duplicate id: keep the first
                }
                if id < last {
                    return Err(HprofError::Corrupt(
                        "object store is not sorted by object id".to_owned(),
                    ));
                }
            }
            entry_to_compact[entry_idx as usize] = compact_ids.len() as u32;
            compact_ids.push(id);
            shallow_sizes.push(shallow);
            running += u64::from(degree);
            check_edge_capacity(running, entry_count as u64)?;
            fwd_off.push(running as u32);
        }
    }
    let n = compact_ids.len();

    // ── Pass 2 (sequential): stream (to_id, from_compact) to the store ────────
    let total_edges = fwd_off[n] as usize;
    let scratch = names::part(names::DOMINATORS, 0);
    let mut writer = RecordWriter::<EdgeEntry>::new(store.create(&scratch)?);
    for (k, entry) in combined.iter().enumerate() {
        let from = entry_to_compact[k];
        if from == u32::MAX {
            continue;
        }
        let mut emit = |to_id: u64| {
            writer.push(&EdgeEntry {
                to_id,
                from: u64::from(from),
            })
        };
        match entry.tag {
            TAG_INSTANCE_DUMP => {
                if let SubRecord::InstanceDump(inst) = parse_sub_record(hprof, &entry)? {
                    for_each_instance_ref(&inst, class_cache, id_size, &mut emit)?;
                }
            }
            TAG_CLASS_DUMP => {
                if let SubRecord::ClassDump(cd) = parse_sub_record(hprof, &entry)? {
                    for sf_res in cd.static_fields() {
                        if let FieldValue::Object(id) = sf_res?.value
                            && id != 0
                        {
                            emit(id)?;
                        }
                    }
                }
            }
            TAG_OBJ_ARRAY_DUMP => {
                if let SubRecord::ObjArrayDump(arr) = parse_sub_record(hprof, &entry)? {
                    for id in arr.elements() {
                        if id != 0 {
                            emit(id)?;
                        }
                    }
                }
            }
            _ => {}
        }
    }
    drop(entry_to_compact);
    // Sort ascending by to_id in place on the store, then commit.
    writer.finish_sorted()?;

    // ── Merge: two-pointer scan → fill fwd_edges directly ─────────────────────
    let scratch_src = store.open(&scratch)?;
    let edges = RecordFile::<EdgeEntry>::new(scratch_src.as_ref())?;
    let mut fwd_edges: Vec<u32> = vec![0u32; total_edges];
    let mut cursor: Vec<u32> = fwd_off[..n].to_vec();

    let mut ci = 0usize; // pointer into compact_ids (ascending)
    let mut iter = edges.iter().peekable();
    while ci < n {
        let Some(&e) = iter.peek() else { break };
        match compact_ids[ci].cmp(&e.to_id) {
            std::cmp::Ordering::Less => ci += 1,
            std::cmp::Ordering::Greater => {
                iter.next(); // to_id not in the dump (dangling reference)
            }
            std::cmp::Ordering::Equal => {
                let to_compact = ci as u32;
                while let Some(&e2) = iter.peek() {
                    if e2.to_id != e.to_id {
                        break;
                    }
                    let from = e2.from as usize;
                    let pos = cursor[from] as usize;
                    let limit = fwd_off[from + 1] as usize;
                    if pos < limit {
                        fwd_edges[pos] = to_compact;
                        cursor[from] += 1;
                    }
                    iter.next();
                }
            }
        }
    }
    drop(iter);
    drop(scratch_src);
    store.remove(&scratch)?;

    compact_csr(&mut fwd_off, &mut fwd_edges, &cursor);

    Ok((compact_ids, shallow_sizes, fwd_off, fwd_edges))
}

/// Fail unless a CSR with `edges` forward edges over at most `nodes` nodes
/// can be addressed with `u32` offsets.
///
/// The backward CSR holds every forward edge plus one edge per GC root
/// (at most one per node), so `edges + nodes` must fit in a `u32`.  Without
/// this check the offsets would wrap and the dominators would be wrong with
/// no error.
fn check_edge_capacity(edges: u64, nodes: u64) -> Result<(), HprofError> {
    if edges + nodes > u64::from(u32::MAX) {
        return Err(HprofError::TooLarge(format!(
            "the heap has more than {} references, more than the in-memory dominator              builder can index (plan task S6.1 covers an out-of-core builder)",
            u32::MAX as u64 - nodes
        )));
    }
    Ok(())
}

/// Squeeze the unfilled slots out of a forward CSR in place.
///
/// Pass 1 counts every non-null reference, including references to ids that
/// are not in the dump; the merge only fills slots for references that
/// resolve.  `filled_end[i]` is the end of node `i`'s filled slots (its
/// cursor after the merge).  Without this step the unfilled slots would read
/// as edges to compact node 0.
fn compact_csr(fwd_off: &mut [u32], fwd_edges: &mut Vec<u32>, filled_end: &[u32]) {
    let n = filled_end.len();
    let filled: usize = (0..n).map(|i| (filled_end[i] - fwd_off[i]) as usize).sum();
    if filled == fwd_edges.len() {
        return; // no dangling references
    }
    let mut write = 0usize;
    for i in 0..n {
        let old_start = fwd_off[i] as usize; // not yet overwritten
        let old_end = filled_end[i] as usize;
        fwd_edges.copy_within(old_start..old_end, write);
        fwd_off[i] = write as u32;
        write += old_end - old_start;
    }
    fwd_off[n] = write as u32;
    fwd_edges.truncate(write);
    fwd_edges.shrink_to_fit();
}

fn prim_elem_byte_size(type_id: u8) -> u64 {
    crate::hprof::BasicType::from_code(type_id).map_or(1, |t| t.size(1) as u64)
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
        // Cannot overflow: `check_edge_capacity` bounds forward edges plus roots.
        pred_off.push(pred_off[i] + in_degree[i]);
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
    drop(exclusive_pairs); // 8 bytes per exclusive node, not needed again

    // Run exclusive partitions in parallel via rayon.
    let exclusive_results: Vec<Vec<(u32, u32)>> = groups
        .par_iter()
        .map(|(root_compact, nodes)| {
            run_exclusive_partition(*root_compact, nodes, fwd_off, fwd_edges)
        })
        .collect();
    drop(groups);
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

/// Write the four output entries.
///
/// `compact_ids` is sorted ascending, so `dominators.bin` and `retained.bin`
/// come out sorted by `object_id` without a sort pass; the two derived
/// entries are sorted in place on the store.
fn write_outputs(
    n: usize,
    compact_ids: &[u64],
    global_doms: &[u32],
    retained: &[u64],
    store: &dyn IndexStore,
) -> Result<(u64, u64), HprofError> {
    let mut dom_w = RecordWriter::<DomEntry>::new(store.create(names::DOMINATORS)?);
    let mut ret_w = RecordWriter::<RetainedEntry>::new(store.create(names::RETAINED)?);
    let mut by_size_w =
        RecordWriter::<RetainedBySizeEntry>::new(store.create(names::RETAINED_BY_SIZE)?);
    let mut children_w =
        RecordWriter::<DomChildEntry>::new(store.create(names::DOMINATOR_CHILDREN)?);

    for i in 0..n {
        let dom = global_doms[i];
        if dom == u32::MAX {
            continue; // unreachable
        }
        let object_id = compact_ids[i];
        let dominator_id = if dom == VROOT_COMPACT {
            VIRTUAL_ROOT_ID
        } else {
            compact_ids[dom as usize]
        };
        dom_w.push(&DomEntry {
            object_id,
            dominator_id,
        })?;
        ret_w.push(&RetainedEntry {
            object_id,
            retained_bytes: retained[i],
        })?;
        by_size_w.push(&RetainedBySizeEntry {
            retained_bytes: retained[i],
            object_id,
        })?;
        children_w.push(&DomChildEntry {
            dominator_id,
            object_id,
        })?;
    }

    let dom_count = dom_w.finish()?;
    let ret_count = ret_w.finish()?;
    by_size_w.finish_sorted_desc()?;
    children_w.finish_sorted_then_by(8)?;

    Ok((dom_count, ret_count))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heap_parser::FieldValue;
    use crate::index::{IndexStore, MemStore, names};
    use crate::pipeline::{IndexOptions, build_indexes};
    use crate::progress::NoProgress;
    use crate::root_index::RootIndexReader;
    use crate::test_util::{ClassSpec, HprofBuilder, ty};

    // ── Test hprof builder ────────────────────────────────────────────────────

    /// Object graph:
    ///
    /// ```text
    /// GC_ROOT_STICKY_CLASS → Class(0x100) [static field → Instance(0x200)]
    /// GC_ROOT_STICKY_CLASS → Class(0x300) [superclass of 0x100]
    /// Instance(0x200) [class=0x100, instance field → PrimArray(0x400)]
    /// PrimArray(0x400) [int[], 3 elements]
    /// ```
    ///
    /// Dominator tree: VROOT → {0x100, 0x300}; 0x100 → 0x200 → 0x400.
    /// Retained: 0x400 = 12, 0x200 = 8 + 12 = 20, 0x100 = 20, 0x300 = 0.
    fn build_test_heap() -> Vec<u8> {
        HprofBuilder::new(8)
            .utf8(1, "MyClass")
            .utf8(2, "java/lang/Object")
            .load_class(1, 0x100, 1)
            .load_class(2, 0x300, 2)
            .root_sticky_class(0x100)
            .root_sticky_class(0x300)
            .class_dump(ClassSpec::new(0x300))
            .class_dump(
                ClassSpec::new(0x100)
                    .super_class(0x300)
                    .instance_size(8)
                    .static_field(1, FieldValue::Object(0x200))
                    .field(1, ty::OBJECT),
            )
            .instance_values(0x200, 0x100, &[FieldValue::Object(0x400)])
            .int_array(0x400, &[0, 0, 0])
            .build()
    }

    /// Run the pipeline (without the dominator step) in memory and return the
    /// object store plus the nine root indexes the builder needs.
    fn prerequisites(hprof: &[u8]) -> (crate::index::ByteSource, Vec<crate::index::ByteSource>) {
        let store = MemStore::new();
        let opts = IndexOptions {
            retained: false,
            force: false,
            ..IndexOptions::default()
        };
        build_indexes(hprof, &store, &opts, &NoProgress).unwrap();
        let roots = names::ROOTS
            .iter()
            .map(|n| store.open(n).unwrap())
            .collect();
        (store.open(names::OBJECT_STORE).unwrap(), roots)
    }

    #[test]
    fn dominator_and_retained_basic() {
        let hprof = build_test_heap();
        let (object_store, root_bytes) = prerequisites(&hprof);
        let root_readers: [RootIndexReader<'_>; 9] =
            std::array::from_fn(|i| RootIndexReader::from_ref(root_bytes[i].as_ref()).unwrap());

        let out = MemStore::new();
        let (dom_count, ret_count) =
            build_dominator_and_retained(&hprof, object_store.as_ref(), &root_readers, &out)
                .unwrap();

        assert!(dom_count > 0, "expected dominator entries");
        assert_eq!(dom_count, ret_count);

        let dominators = out.open(names::DOMINATORS).unwrap();
        let retained = out.open(names::RETAINED).unwrap();
        let dom_idx = DominatorIndex::from_ref(dominators.as_ref()).unwrap();
        let ret_idx = RetainedIndex::from_ref(retained.as_ref()).unwrap();

        // Derived entries: largest-first ranking and children by parent.
        let by_size = out.open(names::RETAINED_BY_SIZE).unwrap();
        let ranked: Vec<(u64, u64)> = RecordFile::<RetainedBySizeEntry>::new(by_size.as_ref())
            .unwrap()
            .iter()
            .map(|e| (e.retained_bytes, e.object_id))
            .collect();
        assert_eq!(ranked[0].0, 20);
        assert_eq!(ranked[ranked.len() - 1], (0, 0x300));
        let children = out.open(names::DOMINATOR_CHILDREN).unwrap();
        let children = RecordFile::<DomChildEntry>::new(children.as_ref()).unwrap();
        let roots: Vec<u64> = children
            .range(VIRTUAL_ROOT_ID)
            .map(|e| e.object_id)
            .collect();
        assert_eq!(roots, vec![0x100, 0x300]);
        assert_eq!(
            children
                .range(0x100)
                .map(|e| e.object_id)
                .collect::<Vec<_>>(),
            vec![0x200]
        );
        assert!(children.range(0x400).next().is_none());

        // Class(0x100) and Class(0x300) are GC roots → dominated by VROOT
        assert_eq!(dom_idx.find(0x100), Some(VIRTUAL_ROOT_ID));
        assert_eq!(dom_idx.find(0x300), Some(VIRTUAL_ROOT_ID));
        // Instance(0x200) is dominated by Class(0x100) (via static field)
        assert_eq!(dom_idx.find(0x200), Some(0x100));
        // PrimArray(0x400) is dominated by Instance(0x200)
        assert_eq!(dom_idx.find(0x400), Some(0x200));

        assert_eq!(ret_idx.find(0x400), Some(12));
        assert_eq!(ret_idx.find(0x200), Some(20));
        assert_eq!(ret_idx.find(0x100), Some(20));
        assert_eq!(ret_idx.find(0x300), Some(0));
    }

    #[test]
    fn memory_estimate_is_linear_and_the_guard_uses_the_limit() {
        assert_eq!(estimate_memory_bytes(0, 0), 0);
        assert_eq!(estimate_memory_bytes(10, 0), 720);
        assert_eq!(estimate_memory_bytes(0, 10), 120);
        assert_eq!(estimate_memory_bytes(u64::MAX, u64::MAX), u64::MAX);
        // 9.7 M objects + 18.2 M references (the calibration dump).
        let est = estimate_memory_bytes(9_680_000, 18_200_000);
        assert!(est > 726 * 1024 * 1024 && est < 1200 * 1024 * 1024, "{est}");

        assert_eq!(check_memory(10, 10, Some(10_000)).unwrap(), 840);
        assert!(matches!(
            check_memory(10, 10, Some(839)),
            Err(HprofError::InsufficientMemory {
                needed: 840,
                available: 839,
                ..
            })
        ));
    }

    /// A reference to an object that is not in the dump (dangling) must not
    /// create an edge.  Regression: the CSR out-degrees counted such
    /// references but the fill pass skipped them, leaving zero-filled slots
    /// that read as edges to compact node 0 (the lowest object id).
    #[test]
    fn edge_counts_beyond_u32_are_rejected_not_wrapped() {
        assert!(check_edge_capacity(1_000, 100).is_ok());
        let max = u64::from(u32::MAX);
        assert!(check_edge_capacity(max - 100, 100).is_ok());
        assert!(matches!(
            check_edge_capacity(max - 99, 100),
            Err(HprofError::TooLarge(_))
        ));
        assert!(check_edge_capacity(max + 1, 0).is_err());
    }

    #[test]
    fn dangling_references_do_not_create_edges_to_the_lowest_object() {
        use crate::test_util::{ClassSpec, HprofBuilder, ty};
        // 0x50: unreachable instance with the lowest id (compact index 0).
        // 0x100: sticky-root class; 0x200: instance referenced by its static.
        // 0x200 has one field pointing at 0x999, which is not in the dump.
        let hprof = HprofBuilder::new(8)
            .utf8(1, "C")
            .utf8(2, "f")
            .load_class(1, 0x100, 1)
            .root_sticky_class(0x100)
            .class_dump(
                ClassSpec::new(0x100)
                    .instance_size(8)
                    .static_field(2, FieldValue::Object(0x200))
                    .field(2, ty::OBJECT),
            )
            .instance_values(0x50, 0x100, &[FieldValue::Object(0)])
            .instance_values(0x200, 0x100, &[FieldValue::Object(0x999)])
            .build();
        let query = crate::test_util::build_in_memory(&hprof);
        assert_eq!(query.dominator_of(0x200), Some(0x100));
        assert_eq!(
            query.dominator_of(0x50),
            None,
            "0x50 is unreachable; a dangling reference must not make it reachable"
        );
        assert_eq!(query.retained_size(0x50), None);
        assert_eq!(query.retained_size(0x999), None);
    }

    #[test]
    fn full_pipeline_exposes_the_same_answers_through_heap_query() {
        let hprof = build_test_heap();
        let query = crate::test_util::build_in_memory(&hprof);
        assert!(query.has_retained_heap());
        assert_eq!(query.retained_size(0x200), Some(20));
        assert_eq!(query.dominator_of(0x400), Some(0x200));
    }

    /// Build a compact forward CSR for a simple graph described as a list of
    /// directed edges `(from_object_id, to_object_id)` and a list of all
    /// node object IDs (including isolated nodes).
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

        let gc_roots_compact: Vec<u32> = vec![0u32]; // id=1 → compact index 0
        let (rpo_order, rpo_of) = compute_rpo_compact(3, &gc_roots_compact, &fwd_off, &fwd_edges);

        assert_eq!(rpo_order[0], u32::MAX, "RPO[0] should be virtual root");
        assert_eq!(rpo_order[1], 0);
        assert_eq!(rpo_order[2], 1);
        assert_eq!(rpo_order[3], 2);

        assert_eq!(rpo_of[0], 1);
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

        assert_eq!(retained[2], 30, "C retained");
        assert_eq!(retained[1], 50, "B retained");
        assert_eq!(retained[0], 60, "A retained");

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
        assert_eq!(idx.file.len(), 3);
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
