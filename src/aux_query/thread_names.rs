//! Resolves thread names from `java.lang.Thread` heap objects.
//!
//! Used when `HPROF_START_THREAD` records are absent.
//!
//! ## Strategy
//!
//! 1. Scan heap sub-records for `GC_ROOT_THREAD_OBJ` (0x08) to map
//!    `thread_serial` → `thread_object_id`.
//! 2. For each thread object, look up its `GC_INSTANCE_DUMP` in the object
//!    store index.  Walk the class hierarchy (following `super_class_id`
//!    links through `GC_CLASS_DUMP` records) to find the byte offset of the
//!    `name` instance field.
//! 3. Follow `name` → `java.lang.String` instance → `value` → `byte[]` (Java
//!    9+) or `char[]` (Java 8) and decode.

use std::collections::HashMap;

use crate::heap_index::sub_record::{
    SUB_INDEX_ENTRY_SIZE, SubIndexEntry, SubRecordScanner, TAG_CLASS_DUMP, TAG_INSTANCE_DUMP,
    TAG_PRIM_ARRAY_DUMP, TAG_ROOT_THREAD_OBJ, value_size,
};
use crate::heap_parser::record::{PrimArrayDump, SubRecord, parse_sub_record};
use crate::heap_query::name_index::Utf8IndexReader;
use crate::hprof::record::{RecordTag, read_id};
use crate::hprof::{HprofError, HprofFile};
use crate::record_index::entry::{INDEX_ENTRY_SIZE, IndexEntry};
use crate::vfs::MMapReader;

// ── Public entry point ────────────────────────────────────────────────────────

/// Build a map from `thread_serial` → resolved thread name by reading
/// `java.lang.Thread` heap objects.
///
/// Returns an empty map if no `GC_ROOT_THREAD_OBJ` sub-records are found or
/// if the object store index does not contain the relevant objects.
pub fn collect_thread_names(
    hprof: &HprofFile,
    record_index_source: &impl MMapReader,
    object_store_source: &impl MMapReader,
    utf8: &Utf8IndexReader,
) -> Result<HashMap<u32, String>, HprofError> {
    let record_index_bs = record_index_source.open_mmap()?;
    let root_objs = collect_root_thread_objs(hprof, record_index_bs.as_ref())?;
    if root_objs.is_empty() {
        return Ok(HashMap::new());
    }

    let object_store_bs = object_store_source.open_mmap()?;
    let object_store_data: &[u8] = object_store_bs.as_ref();
    let mut names = HashMap::new();

    for (thread_serial, thread_object_id) in root_objs {
        if let Some(name) =
            resolve_thread_name_from_heap(hprof, object_store_data, utf8, thread_object_id)?
            && !name.is_empty()
        {
            names.insert(thread_serial, name);
        }
    }

    Ok(names)
}

// ── Step 1: GC_ROOT_THREAD_OBJ scan ──────────────────────────────────────────

/// Scan heap dump bodies for `GC_ROOT_THREAD_OBJ` sub-records and return a
/// map from `thread_serial` → `thread_object_id`.
fn collect_root_thread_objs(
    hprof: &HprofFile,
    record_index_data: &[u8],
) -> Result<HashMap<u32, u64>, HprofError> {
    let data = hprof.data();
    let id_size = hprof.header.id_size;
    let heap_tag = u8::from(RecordTag::HeapDump);
    let heap_seg_tag = u8::from(RecordTag::HeapDumpSegment);

    let n = record_index_data.len() / INDEX_ENTRY_SIZE;
    let mut result: HashMap<u32, u64> = HashMap::new();

    for i in 0..n {
        let start = i * INDEX_ENTRY_SIZE;
        let bytes: [u8; INDEX_ENTRY_SIZE] = record_index_data[start..start + INDEX_ENTRY_SIZE]
            .try_into()
            .map_err(|_| HprofError::InvalidIndexFile)?;
        let entry = IndexEntry::from_bytes(&bytes);
        if entry.tag != heap_tag && entry.tag != heap_seg_tag {
            continue;
        }

        let header_pos = entry.position as usize;
        if header_pos + 9 > data.len() {
            continue;
        }
        let body_len = u32::from_be_bytes([
            data[header_pos + 5],
            data[header_pos + 6],
            data[header_pos + 7],
            data[header_pos + 8],
        ]) as usize;
        let body_start = header_pos + 9;
        if body_start + body_len > data.len() {
            continue;
        }

        let body = &data[body_start..body_start + body_len];
        let body_start_abs = entry.position + 9;
        let scanner = SubRecordScanner::new(body, body_start_abs, id_size)?;

        for sub_result in scanner {
            let sub = sub_result?;
            if sub.tag != TAG_ROOT_THREAD_OBJ {
                continue;
            }
            // GC_ROOT_THREAD_OBJ layout (relative to subtag byte):
            //   tag(1)  thread_object_id(id)  thread_serial(4)  stack_trace_serial(4)
            let serial_abs = sub.position as usize + 1 + id_size as usize;
            if serial_abs + 4 > data.len() {
                continue;
            }
            let thread_serial = u32::from_be_bytes([
                data[serial_abs],
                data[serial_abs + 1],
                data[serial_abs + 2],
                data[serial_abs + 3],
            ]);
            // Keep first mapping if a serial appears more than once.
            result.entry(thread_serial).or_insert(sub.object_id);
        }
    }

    Ok(result)
}

// ── Step 2: heap object traversal ────────────────────────────────────────────

/// Resolve the thread name for a single `thread_object_id` by walking:
/// `GC_INSTANCE_DUMP(Thread)` → `name` field → `GC_INSTANCE_DUMP(String)`
/// → `value` field → `GC_PRIM_ARRAY_DUMP(byte[]/char[])` → decoded text.
fn resolve_thread_name_from_heap(
    hprof: &HprofFile,
    object_store_data: &[u8],
    utf8: &Utf8IndexReader,
    thread_object_id: u64,
) -> Result<Option<String>, HprofError> {
    // Locate the Thread INSTANCE_DUMP.
    let thread_entry =
        match find_in_object_store(object_store_data, thread_object_id, TAG_INSTANCE_DUMP)? {
            Some(e) => e,
            None => return Ok(None),
        };
    let thread_sub = parse_sub_record(hprof, &thread_entry)?;
    let (thread_class_id, thread_data) = match thread_sub {
        SubRecord::InstanceDump(i) => (i.class_id, i.data),
        _ => return Ok(None),
    };

    // Find the object ID stored in the "name" field.
    let name_obj_id = match find_object_field(
        hprof,
        object_store_data,
        utf8,
        thread_class_id,
        thread_data,
        "name",
    )? {
        Some(id) if id != 0 => id,
        _ => return Ok(None),
    };

    // Locate the String INSTANCE_DUMP.
    let str_entry = match find_in_object_store(object_store_data, name_obj_id, TAG_INSTANCE_DUMP)? {
        Some(e) => e,
        None => return Ok(None),
    };
    let str_sub = parse_sub_record(hprof, &str_entry)?;
    let (str_class_id, str_data) = match str_sub {
        SubRecord::InstanceDump(i) => (i.class_id, i.data),
        _ => return Ok(None),
    };

    // Find the object ID stored in the "value" field of String.
    let value_id = match find_object_field(
        hprof,
        object_store_data,
        utf8,
        str_class_id,
        str_data,
        "value",
    )? {
        Some(id) if id != 0 => id,
        _ => return Ok(None),
    };

    // Locate the backing byte[] or char[] PRIM_ARRAY_DUMP.
    let arr_entry = match find_in_object_store(object_store_data, value_id, TAG_PRIM_ARRAY_DUMP)? {
        Some(e) => e,
        None => return Ok(None),
    };
    let name = match parse_sub_record(hprof, &arr_entry)? {
        SubRecord::PrimArrayDump(a) => decode_string_bytes(&a),
        _ => return Ok(None),
    };

    Ok(Some(name))
}

// ── Field lookup ──────────────────────────────────────────────────────────────

/// Walk the class hierarchy of `class_id` (root → leaf) and return the object
/// ID stored in the first instance field named `field_name` whose type is
/// an object reference (type code 2).
///
/// Returns `None` if no matching field is found or bounds are exceeded.
fn find_object_field(
    hprof: &HprofFile,
    object_store_data: &[u8],
    utf8: &Utf8IndexReader,
    class_id: u64,
    instance_data: &[u8],
    field_name: &str,
) -> Result<Option<u64>, HprofError> {
    let id_size = hprof.header.id_size as usize;

    // Collect class chain from leaf to root, then reverse.
    let mut chain: Vec<u64> = Vec::new();
    let mut cur = class_id;
    while cur != 0 && chain.len() < 64 {
        chain.push(cur);
        let entry = match find_in_object_store(object_store_data, cur, TAG_CLASS_DUMP)? {
            Some(e) => e,
            None => break,
        };
        let super_id = match parse_sub_record(hprof, &entry)? {
            SubRecord::ClassDump(cd) => cd.super_class_id,
            _ => break,
        };
        cur = super_id;
    }
    chain.reverse(); // root first → leaf last

    // Walk fields in layout order, accumulating byte offset.
    let mut offset = 0usize;
    for cid in chain {
        let entry = match find_in_object_store(object_store_data, cid, TAG_CLASS_DUMP)? {
            Some(e) => e,
            None => continue,
        };
        let cd = match parse_sub_record(hprof, &entry)? {
            SubRecord::ClassDump(cd) => cd,
            _ => continue,
        };

        // Collect into Vec so cd's borrow ends before the next loop iteration.
        let fields: Vec<_> = cd.instance_fields().collect::<Result<_, _>>()?;
        for field in fields {
            let sz = value_size(field.field_type, id_size)?;
            if field.field_type == 2 {
                // Object reference — check name.
                if let Some(name) = utf8.lookup(hprof, field.name_id)?
                    && name == field_name
                {
                    if offset + id_size > instance_data.len() {
                        return Ok(None);
                    }
                    let obj_id = read_id(instance_data, offset, id_size)?;
                    return Ok(Some(obj_id));
                }
            }
            offset += sz;
        }
    }

    Ok(None)
}

// ── Object store binary search ────────────────────────────────────────────────

/// Binary-search the object store for the first entry whose `object_id`
/// equals `target` **and** whose sub-record `tag` equals `desired_tag`.
///
/// Because multiple sub-record types (e.g. `GC_ROOT_THREAD_OBJ` and
/// `GC_INSTANCE_DUMP`) share the same `object_id` for the same heap object,
/// the leftmost binary-search result may have the wrong tag.  This function
/// scans forward through all consecutive entries with the same `object_id`
/// until it finds one with the requested tag, or returns `None`.
fn find_in_object_store(
    data: &[u8],
    target: u64,
    desired_tag: u8,
) -> Result<Option<SubIndexEntry>, HprofError> {
    if !data.len().is_multiple_of(SUB_INDEX_ENTRY_SIZE) {
        return Err(HprofError::InvalidIndexFile);
    }
    let n = data.len() / SUB_INDEX_ENTRY_SIZE;

    // Leftmost binary search for first index where object_id >= target.
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if obj_id_at(data, mid) < target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    // Scan forward through all entries with object_id == target.
    while lo < n && obj_id_at(data, lo) == target {
        let start = lo * SUB_INDEX_ENTRY_SIZE;
        let bytes: [u8; SUB_INDEX_ENTRY_SIZE] = data[start..start + SUB_INDEX_ENTRY_SIZE]
            .try_into()
            .map_err(|_| HprofError::InvalidIndexFile)?;
        let entry = SubIndexEntry::from_bytes(&bytes);
        if entry.tag == desired_tag {
            return Ok(Some(entry));
        }
        lo += 1;
    }
    Ok(None)
}

/// Read the `object_id` field (bytes 8..16, little-endian) at index `idx`.
fn obj_id_at(data: &[u8], idx: usize) -> u64 {
    let off = idx * SUB_INDEX_ENTRY_SIZE + 8;
    u64::from_le_bytes([
        data[off],
        data[off + 1],
        data[off + 2],
        data[off + 3],
        data[off + 4],
        data[off + 5],
        data[off + 6],
        data[off + 7],
    ])
}

// ── String decoding ───────────────────────────────────────────────────────────

/// Decode a `byte[]` (Java 9+ Latin-1) or `char[]` (Java 8 UTF-16BE) array
/// into a Rust `String`.
fn decode_string_bytes(arr: &PrimArrayDump<'_>) -> String {
    match arr.element_type {
        // char[] — UTF-16 big-endian code units
        5 => arr
            .data
            .chunks_exact(2)
            .filter_map(|ch| char::from_u32(u16::from_be_bytes([ch[0], ch[1]]) as u32))
            .collect(),
        // byte[] — Latin-1 (U+0000..U+00FF, one byte per character)
        8 => arr.data.iter().map(|&b| b as char).collect(),
        _ => String::new(),
    }
}
