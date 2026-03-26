//! High-level heap analysis API.
//!
//! This module provides [`HprofIndex`], the main entry point for adhoc heap
//! dump analysis. It layers on top of all prior phases:
//!
//! * Record index — scanned once to build the name indexes below.
//! * Object store (combined + sorted sub-record index) — random access by object ID.
//! * UTF-8 name index (`utf8.index`) — sorted by `name_id`.
//! * Load-class index (`load_class.index`) — sorted by `class_id`.
//!
//! **No in-memory caches** are maintained: every lookup goes directly to the
//! mmapped index files.
//!
//! ## Building the name indexes
//!
//! ```rust,ignore
//! use hprof_toolkit::heap_query::build_name_indexes;
//! build_name_indexes(hprof_path, record_index_path, utf8_path, load_class_path)?;
//! ```
//!
//! ## Opening the API
//!
//! ```rust,ignore
//! use hprof_toolkit::heap_query::HprofIndex;
//! let index = HprofIndex::open(hprof_path, combined_path, utf8_path, load_class_path)?;
//! ```

pub mod name_index;
pub mod resolve;

pub use name_index::{LoadClassReader, Utf8IndexReader};
pub use resolve::{JavaValue, ResolvedField};

use crate::heap_index::sub_record::{TAG_CLASS_DUMP, TAG_INSTANCE_DUMP, TAG_PRIM_ARRAY_DUMP};
use crate::heap_parser::record::FieldValue;
use crate::heap_parser::{InstanceFieldDescriptor, SubIndexReader, SubRecord, parse_sub_record};
use crate::hprof::{HprofError, HprofFile};
use crate::vfs::{MMapReader, MMapWriter};
use name_index::{build_load_class_index, build_utf8_index};
use resolve::{decode_char_array, decode_string_bytes, read_field_value};

// ── Public builder ────────────────────────────────────────────────────────────

/// Scan the record index and build UTF-8 name + load-class indexes.
///
/// Both output files are sorted by their respective key fields so that
/// [`HprofIndex`] can resolve names with O(log n) binary searches.
///
/// Returns `(utf8_count, load_class_count)`.
pub fn build_name_indexes(
    hprof_source: &impl MMapReader,
    record_index_source: &impl MMapReader,
    utf8_path: &mut impl MMapWriter,
    load_class_path: &mut impl MMapWriter,
) -> Result<(u64, u64), HprofError> {
    let hprof = HprofFile::from_source(hprof_source.open_mmap()?)?;
    let record_index_mmap = record_index_source.open_mmap()?;

    let utf8_count = build_utf8_index(&hprof, record_index_mmap.as_ref(), utf8_path)?;
    let lc_count = build_load_class_index(&hprof, record_index_mmap.as_ref(), load_class_path)?;

    Ok((utf8_count, lc_count))
}

// ── HprofIndex ────────────────────────────────────────────────────────────────

/// High-level analysis API over a heap dump and its index files.
///
/// All index files are kept as memory maps; no data is copied into
/// process-owned structures.  Each method performs O(log n) binary searches
/// against the sorted index files.
pub struct HprofIndex {
    hprof: HprofFile,
    /// Object store (combined + sorted sub-record index).
    combined: SubIndexReader,
    utf8: Utf8IndexReader,
    load_class: LoadClassReader,
}

impl HprofIndex {
    /// Open all required index files.
    ///
    /// * `hprof_source`    — the hprof file or bytes.
    /// * `combined_source` — object store index (sorted by object ID).
    /// * `utf8_source`     — UTF-8 name index (sorted by name ID).
    /// * `lc_source`       — load-class index (sorted by class ID).
    pub fn open(
        hprof_source: &impl MMapReader,
        combined_source: &impl MMapReader,
        utf8_source: &impl MMapReader,
        lc_source: &impl MMapReader,
    ) -> Result<Self, HprofError> {
        Ok(Self {
            hprof: HprofFile::from_source(hprof_source.open_mmap()?)?,
            combined: SubIndexReader::from_bytes(combined_source.open_mmap()?.as_ref().to_vec())?,
            utf8: Utf8IndexReader::from_bytes(utf8_source.open_mmap()?.as_ref().to_vec())?,
            load_class: LoadClassReader::from_bytes(lc_source.open_mmap()?.as_ref().to_vec())?,
        })
    }

    // ── Basic accessors ───────────────────────────────────────────────────────

    /// hprof identifier size in bytes (4 or 8).
    pub fn id_size(&self) -> u32 {
        self.hprof.header.id_size
    }

    /// Total number of sub-records in the combined index (classes, instances,
    /// arrays, and GC roots).
    pub fn object_count(&self) -> usize {
        self.combined.len()
    }

    /// Iterate over all sub-record index entries in the combined index.
    ///
    /// Entries are in object-ID order (ascending).  Call
    /// [`parse_sub_record`] to obtain full record details.
    pub fn iter_entries(&self) -> crate::heap_parser::SubIndexIter<'_> {
        self.combined.iter()
    }

    /// Parse the sub-record at position `i` in the combined index.
    ///
    /// Returns `None` when `i >= object_count()`.
    pub fn parse_at(&self, i: usize) -> Result<Option<SubRecord<'_>>, HprofError> {
        match self.combined.entry_at(i) {
            Some(entry) => Ok(Some(parse_sub_record(&self.hprof, &entry)?)),
            None => Ok(None),
        }
    }

    // ── Object lookup ─────────────────────────────────────────────────────────

    /// Parse the sub-record for `object_id` (any tag).
    ///
    /// Uses the object store index for O(log n) lookup.
    /// Returns `None` when the object ID is not present.
    pub fn find_object<'a>(&'a self, object_id: u64) -> Result<Option<SubRecord<'a>>, HprofError> {
        match self.combined.find_by_object_id(object_id) {
            Some(entry) => Ok(Some(parse_sub_record(&self.hprof, &entry)?)),
            None => Ok(None),
        }
    }

    /// Parse the CLASS_DUMP sub-record for `class_id`.
    ///
    /// Searches specifically for a `CLASS_DUMP` tag so that other sub-records
    /// that happen to share the same object ID (e.g. `ROOT_*` entries for
    /// class objects) are skipped.
    pub fn find_class_dump<'a>(
        &'a self,
        class_id: u64,
    ) -> Result<Option<SubRecord<'a>>, HprofError> {
        match self
            .combined
            .find_by_object_id_and_tag(class_id, Some(TAG_CLASS_DUMP))
        {
            Some(entry) => Ok(Some(parse_sub_record(&self.hprof, &entry)?)),
            None => Ok(None),
        }
    }

    /// Parse the INSTANCE_DUMP sub-record for `object_id`.
    pub fn find_instance<'a>(
        &'a self,
        object_id: u64,
    ) -> Result<Option<SubRecord<'a>>, HprofError> {
        match self
            .combined
            .find_by_object_id_and_tag(object_id, Some(TAG_INSTANCE_DUMP))
        {
            Some(entry) => Ok(Some(parse_sub_record(&self.hprof, &entry)?)),
            None => Ok(None),
        }
    }

    /// Parse the sub-record described by `entry` directly from the hprof mmap.
    ///
    /// Use this when you already hold a [`SubIndexEntry`] (e.g. from iterating
    /// the combined index) and want to avoid a redundant binary search.
    pub fn parse_entry<'a>(
        &'a self,
        entry: &crate::heap_index::sub_record::SubIndexEntry,
    ) -> Result<SubRecord<'a>, HprofError> {
        parse_sub_record(&self.hprof, entry)
    }

    // ── Name resolution ───────────────────────────────────────────────────────

    /// Look up the string for `name_id` in the UTF-8 name index.
    pub fn lookup_name(&self, name_id: u64) -> Result<Option<String>, HprofError> {
        self.utf8.lookup(&self.hprof, name_id)
    }

    /// Find a class by its dot-notation name (e.g. `"java.lang.String"`).
    ///
    /// Performs a linear scan of the load-class index, resolving each entry's
    /// name via the UTF-8 index, until a match is found.  Returns the
    /// `class_id` of the first matching class, or `None` if no class with that
    /// name is present.
    ///
    /// The `name` argument must use dot notation exactly as returned by
    /// [`Self::class_name`].  The scan is O(n_classes × log n_utf8) and is
    /// intended to be called once to obtain the class ID, after which
    /// filtering by `class_id` is an O(1) integer comparison per record.
    pub fn find_class_by_name(&self, name: &str) -> Result<Option<u64>, HprofError> {
        // Stored names use JVM internal slash notation; convert the input once.
        let slash_name = name.replace('.', "/");
        for (class_id, name_id) in self.load_class.iter() {
            if let Some(raw) = self.utf8.lookup(&self.hprof, name_id)?
                && raw == slash_name
            {
                return Ok(Some(class_id));
            }
        }
        Ok(None)
    }

    /// Return the dot-notation class name for `class_id` (e.g. `"java.lang.String"`).
    ///
    /// Uses the load-class index → UTF-8 index chain. Returns `None` when the
    /// class is not found in either index.
    pub fn class_name(&self, class_id: u64) -> Result<Option<String>, HprofError> {
        let name_id = match self.load_class.find_class_name_id(class_id) {
            Some(id) => id,
            None => return Ok(None),
        };
        let raw = match self.utf8.lookup(&self.hprof, name_id)? {
            Some(s) => s,
            None => return Ok(None),
        };
        // Normalise JVM internal format (e.g. "java/lang/String") to dots.
        Ok(Some(raw.replace('/', ".")))
    }

    /// Resolve the runtime type name of the object at `object_id`.
    ///
    /// Returns the Java class name of the referenced object:
    /// * `InstanceDump`  → class name (e.g. `java.util.ArrayList`)
    /// * `ClassDump`     → `"Class"`
    /// * `ObjArrayDump`  → `"ElementClass[]"`
    /// * `PrimArrayDump` → `"primitive[]"`
    /// * not found / null → `"Object"`
    pub fn object_type_name(&self, object_id: u64) -> Result<String, HprofError> {
        if object_id == 0 {
            return Ok("Object".to_string());
        }
        match self.find_object(object_id)? {
            None => Ok("Object".to_string()),
            Some(SubRecord::InstanceDump(inst)) => Ok(self
                .class_name(inst.class_id)?
                .unwrap_or_else(|| "Object".to_string())),
            Some(SubRecord::ClassDump(_)) => Ok("Class".to_string()),
            Some(SubRecord::ObjArrayDump(arr)) => {
                let elem = self
                    .class_name(arr.element_class_id)?
                    .unwrap_or_else(|| "Object".to_string());
                Ok(format!("{elem}[]"))
            }
            Some(SubRecord::PrimArrayDump(arr)) => {
                let prim = match arr.element_type {
                    4 => "boolean",
                    5 => "char",
                    6 => "float",
                    7 => "double",
                    8 => "byte",
                    9 => "short",
                    10 => "int",
                    11 => "long",
                    _ => "?",
                };
                Ok(format!("{prim}[]"))
            }
            Some(_) => Ok("Object".to_string()),
        }
    }

    // ── Field resolution ──────────────────────────────────────────────────────

    /// Resolve the instance fields for an [`crate::heap_parser::InstanceDump`].
    ///
    /// Traverses the class hierarchy starting from `instance.class_id`,
    /// consuming bytes from `instance.data` in class-first, super-last order.
    /// Field names are resolved via the UTF-8 index.
    ///
    /// Returns a `Vec` whose length is bounded by the total number of instance
    /// fields declared across the class hierarchy — not by heap size.
    pub fn instance_fields(
        &self,
        instance: &crate::heap_parser::InstanceDump<'_>,
    ) -> Result<Vec<ResolvedField>, HprofError> {
        let id_size = self.id_size() as usize;

        // ── Step 1: collect field descriptors for each class in the chain ──
        // We collect owned data so that each ClassDump (which borrows from the
        // hprof mmap) can be dropped before the next loop iteration.
        let mut chain: Vec<Vec<InstanceFieldDescriptor>> = Vec::new();
        let mut class_id = instance.class_id;

        while class_id != 0 {
            let sub = match self.find_class_dump(class_id)? {
                Some(s) => s,
                None => break,
            };
            let SubRecord::ClassDump(cd) = sub else {
                break;
            };
            let super_id = cd.super_class_id;
            // Collect into an owned Vec — InstanceFieldDescriptor is Copy.
            let descs: Vec<InstanceFieldDescriptor> =
                cd.instance_fields().filter_map(|r| r.ok()).collect();
            chain.push(descs);
            class_id = super_id;
        }

        // ── Step 2: parse instance.data using the collected field layout ──
        let mut fields = Vec::new();
        let mut offset = 0usize;

        for descs in &chain {
            for desc in descs {
                let (value, consumed) =
                    read_field_value(instance.data, offset, desc.field_type, id_size)?;
                offset += consumed;
                let name = self
                    .lookup_name(desc.name_id)?
                    .unwrap_or_else(|| format!("<name#{}>", desc.name_id));
                fields.push(ResolvedField {
                    name,
                    field_type: desc.field_type,
                    value,
                });
            }
        }

        Ok(fields)
    }

    // ── Java wrapper type resolution ──────────────────────────────────────────

    /// Attempt to resolve `object_id` as a primitive Java value.
    ///
    /// Handles:
    /// * `0` → [`JavaValue::Null`]
    /// * `java.lang.String` → [`JavaValue::String`] (reads the backing array)
    /// * `java.lang.Integer` / `Long` / `Double` / `Float` / `Short` / `Byte`
    ///   / `Boolean` / `Character` → the corresponding [`JavaValue`] variant
    /// * Anything else → [`JavaValue::Object`]
    ///
    /// No in-memory caches are used; every resolution is a fresh set of O(log n)
    /// binary searches.
    pub fn resolve_value(&self, object_id: u64) -> Result<JavaValue, HprofError> {
        if object_id == 0 {
            return Ok(JavaValue::Null);
        }

        let sub = match self.find_instance(object_id)? {
            Some(s) => s,
            None => return Ok(JavaValue::Object(object_id)),
        };
        let SubRecord::InstanceDump(inst) = sub else {
            return Ok(JavaValue::Object(object_id));
        };

        let class_name = match self.class_name(inst.class_id)? {
            Some(n) => n,
            None => return Ok(JavaValue::Object(object_id)),
        };

        match class_name.as_str() {
            "java.lang.String" => self.resolve_string(&inst),
            "java.lang.Integer" => Ok(self
                .resolve_single_primitive_field(&inst, "value")?
                .map(|f| match f.value {
                    FieldValue::Int(v) => JavaValue::Integer(object_id, v),
                    _ => JavaValue::Object(object_id),
                })
                .unwrap_or(JavaValue::Object(object_id))),
            "java.lang.Long" => Ok(self
                .resolve_single_primitive_field(&inst, "value")?
                .map(|f| match f.value {
                    FieldValue::Long(v) => JavaValue::Long(object_id, v),
                    _ => JavaValue::Object(object_id),
                })
                .unwrap_or(JavaValue::Object(object_id))),
            "java.lang.Double" => Ok(self
                .resolve_single_primitive_field(&inst, "value")?
                .map(|f| match f.value {
                    FieldValue::Double(v) => JavaValue::Double(object_id, v),
                    _ => JavaValue::Object(object_id),
                })
                .unwrap_or(JavaValue::Object(object_id))),
            "java.lang.Float" => Ok(self
                .resolve_single_primitive_field(&inst, "value")?
                .map(|f| match f.value {
                    FieldValue::Float(v) => JavaValue::Float(object_id, v),
                    _ => JavaValue::Object(object_id),
                })
                .unwrap_or(JavaValue::Object(object_id))),
            "java.lang.Short" => Ok(self
                .resolve_single_primitive_field(&inst, "value")?
                .map(|f| match f.value {
                    FieldValue::Short(v) => JavaValue::Short(object_id, v),
                    _ => JavaValue::Object(object_id),
                })
                .unwrap_or(JavaValue::Object(object_id))),
            "java.lang.Byte" => Ok(self
                .resolve_single_primitive_field(&inst, "value")?
                .map(|f| match f.value {
                    FieldValue::Byte(v) => JavaValue::Byte(object_id, v),
                    _ => JavaValue::Object(object_id),
                })
                .unwrap_or(JavaValue::Object(object_id))),
            "java.lang.Boolean" => Ok(self
                .resolve_single_primitive_field(&inst, "value")?
                .map(|f| match f.value {
                    FieldValue::Bool(v) => JavaValue::Boolean(object_id, v),
                    _ => JavaValue::Object(object_id),
                })
                .unwrap_or(JavaValue::Object(object_id))),
            "java.lang.Character" => Ok(self
                .resolve_single_primitive_field(&inst, "value")?
                .map(|f| match f.value {
                    FieldValue::Char(v) => JavaValue::Character(object_id, v),
                    _ => JavaValue::Object(object_id),
                })
                .unwrap_or(JavaValue::Object(object_id))),
            _ => Ok(JavaValue::Object(object_id)),
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Resolve `object_id` as a `java.lang.String`.
    ///
    /// Handles both pre-Java-9 (`char[]` backing) and Java 9+ compact strings
    /// (`byte[]` + `coder` field).
    fn resolve_string(
        &self,
        inst: &crate::heap_parser::InstanceDump<'_>,
    ) -> Result<JavaValue, HprofError> {
        let fields = self.instance_fields(inst)?;

        // Locate `value` (array reference) and optional `coder` (byte).
        let mut value_id: Option<u64> = None;
        let mut coder: u8 = 0; // default: LATIN-1 / UTF-16 auto-detect

        for f in &fields {
            match f.name.as_str() {
                "value" => {
                    if let FieldValue::Object(id) = f.value {
                        value_id = Some(id);
                    }
                }
                "coder" => {
                    if let FieldValue::Byte(c) = f.value {
                        coder = c as u8;
                    }
                }
                _ => {}
            }
        }

        let arr_id = match value_id {
            Some(id) if id != 0 => id,
            _ => return Ok(JavaValue::String(inst.object_id, String::new())),
        };

        // Look up the backing array.
        let arr_entry = match self
            .combined
            .find_by_object_id_and_tag(arr_id, Some(TAG_PRIM_ARRAY_DUMP))
        {
            Some(e) => e,
            None => return Ok(JavaValue::String(inst.object_id, String::new())),
        };

        let arr_record = parse_sub_record(&self.hprof, &arr_entry)?;
        let SubRecord::PrimArrayDump(arr) = arr_record else {
            return Ok(JavaValue::String(inst.object_id, String::new()));
        };

        let text = match arr.element_type {
            5 => decode_char_array(arr.data),          // char[]
            8 => decode_string_bytes(arr.data, coder), // byte[]
            _ => return Ok(JavaValue::String(inst.object_id, String::new())),
        };

        Ok(JavaValue::String(inst.object_id, text))
    }

    /// Find the first instance field named `field_name` and return it.
    fn resolve_single_primitive_field(
        &self,
        inst: &crate::heap_parser::InstanceDump<'_>,
        field_name: &str,
    ) -> Result<Option<ResolvedField>, HprofError> {
        let fields = self.instance_fields(inst)?;
        Ok(fields.into_iter().find(|f| f.name == field_name))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heap_index::index_heap_dumps;
    use crate::object_store::combine_and_sort_sub_index;
    use crate::record_index::index_hprof;
    use crate::vfs::SubIndexDir;

    // ── Minimal hprof builder ─────────────────────────────────────────────────

    /// Build a minimal hprof with:
    /// * UTF8(1, "value") + UTF8(2, "java/lang/Integer") + UTF8(3, "java/lang/Object")
    /// * LOAD_CLASS(class_id=0x200, name_id=2) + LOAD_CLASS(class_id=0x300, name_id=3)
    /// * HEAP_DUMP_SEGMENT containing:
    ///   - CLASS_DUMP(class_id=0x200, super=0x300, 1 instance field: name_id=1, type=int)
    ///   - CLASS_DUMP(class_id=0x300, super=0, 0 instance fields)
    ///   - INSTANCE_DUMP(object_id=0x100, class=0x200, data=[0,0,0,42])
    fn build_test_hprof() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"JAVA PROFILE 1.0.2\0");
        buf.extend_from_slice(&8u32.to_be_bytes()); // id_size = 8
        buf.extend_from_slice(&0u64.to_be_bytes()); // timestamp

        // UTF8(1, "value")  body = 8 + 5 = 13
        buf.push(0x01);
        buf.extend_from_slice(&0u32.to_be_bytes());
        buf.extend_from_slice(&13u32.to_be_bytes());
        buf.extend_from_slice(&1u64.to_be_bytes());
        buf.extend_from_slice(b"value");

        // UTF8(2, "java/lang/Integer") body = 8+17 = 25
        buf.push(0x01);
        buf.extend_from_slice(&0u32.to_be_bytes());
        buf.extend_from_slice(&25u32.to_be_bytes());
        buf.extend_from_slice(&2u64.to_be_bytes());
        buf.extend_from_slice(b"java/lang/Integer");

        // UTF8(3, "java/lang/Object") body = 8+16 = 24
        buf.push(0x01);
        buf.extend_from_slice(&0u32.to_be_bytes());
        buf.extend_from_slice(&24u32.to_be_bytes());
        buf.extend_from_slice(&3u64.to_be_bytes());
        buf.extend_from_slice(b"java/lang/Object");

        // LOAD_CLASS: class_id=0x200, name_id=2
        buf.push(0x02);
        buf.extend_from_slice(&0u32.to_be_bytes());
        buf.extend_from_slice(&24u32.to_be_bytes()); // body = 4+8+4+8 = 24
        buf.extend_from_slice(&1u32.to_be_bytes()); // class_serial
        buf.extend_from_slice(&0x200u64.to_be_bytes()); // class_id
        buf.extend_from_slice(&0u32.to_be_bytes()); // stack_serial
        buf.extend_from_slice(&2u64.to_be_bytes()); // name_id

        // LOAD_CLASS: class_id=0x300, name_id=3
        buf.push(0x02);
        buf.extend_from_slice(&0u32.to_be_bytes());
        buf.extend_from_slice(&24u32.to_be_bytes());
        buf.extend_from_slice(&2u32.to_be_bytes());
        buf.extend_from_slice(&0x300u64.to_be_bytes());
        buf.extend_from_slice(&0u32.to_be_bytes());
        buf.extend_from_slice(&3u64.to_be_bytes());

        // Build the HEAP_DUMP_SEGMENT body:
        let mut seg = Vec::new();

        // CLASS_DUMP(class_id=0x200, super=0x300, instance_size=4, 1 field: name_id=1, type=10/int)
        seg.push(0x20u8); // TAG_CLASS_DUMP
        seg.extend_from_slice(&0x200u64.to_be_bytes()); // class_id
        seg.extend_from_slice(&0u32.to_be_bytes()); // stack_serial
        seg.extend_from_slice(&0x300u64.to_be_bytes()); // super_class_id
        seg.extend_from_slice(&[0u8; 8]); // class_loader_id
        seg.extend_from_slice(&[0u8; 8]); // signers_id
        seg.extend_from_slice(&[0u8; 8]); // domain_id
        seg.extend_from_slice(&[0u8; 8]); // reserved1
        seg.extend_from_slice(&[0u8; 8]); // reserved2
        seg.extend_from_slice(&4u32.to_be_bytes()); // instance_size = 4
        seg.extend_from_slice(&0u16.to_be_bytes()); // cp_count = 0
        seg.extend_from_slice(&0u16.to_be_bytes()); // statics_count = 0
        seg.extend_from_slice(&1u16.to_be_bytes()); // instance_fields_count = 1
        seg.extend_from_slice(&1u64.to_be_bytes()); // field name_id = 1 ("value")
        seg.push(10u8); // field type = int

        // CLASS_DUMP(class_id=0x300, super=0, 0 fields) — java.lang.Object
        seg.push(0x20u8);
        seg.extend_from_slice(&0x300u64.to_be_bytes()); // class_id
        seg.extend_from_slice(&0u32.to_be_bytes());
        seg.extend_from_slice(&0u64.to_be_bytes()); // super = 0
        seg.extend_from_slice(&[0u8; 8 * 5]); // loader + signers + domain + res1 + res2
        seg.extend_from_slice(&0u32.to_be_bytes()); // instance_size
        seg.extend_from_slice(&0u16.to_be_bytes()); // cp=0
        seg.extend_from_slice(&0u16.to_be_bytes()); // statics=0
        seg.extend_from_slice(&0u16.to_be_bytes()); // fields=0

        // INSTANCE_DUMP(object_id=0x100, class=0x200, data=[0,0,0,42])
        seg.push(0x21u8); // TAG_INSTANCE_DUMP
        seg.extend_from_slice(&0x100u64.to_be_bytes()); // object_id
        seg.extend_from_slice(&0u32.to_be_bytes()); // stack_serial
        seg.extend_from_slice(&0x200u64.to_be_bytes()); // class_id
        seg.extend_from_slice(&4u32.to_be_bytes()); // data_len = 4
        seg.extend_from_slice(&42i32.to_be_bytes()); // value = 42

        // Write HEAP_DUMP_SEGMENT record
        buf.push(0x1C);
        buf.extend_from_slice(&0u32.to_be_bytes());
        buf.extend_from_slice(&(seg.len() as u32).to_be_bytes());
        buf.extend_from_slice(&seg);

        buf
    }

    /// Run the full build pipeline on `hprof_data` and open a [`HprofIndex`].
    fn full_pipeline(hprof_data: &[u8]) -> HprofIndex {
        let mut p1_idx = Vec::new();
        let p2_dir = SubIndexDir::mem();
        let mut combined = Vec::new();
        let mut utf8_idx = Vec::new();
        let mut lc_idx = Vec::new();

        index_hprof(&hprof_data.to_vec(), &mut p1_idx).unwrap();
        index_heap_dumps(&hprof_data.to_vec(), &p1_idx, &p2_dir).unwrap();
        combine_and_sort_sub_index(&p2_dir, &mut combined).unwrap();
        build_name_indexes(&hprof_data.to_vec(), &p1_idx, &mut utf8_idx, &mut lc_idx).unwrap();

        HprofIndex::open(&hprof_data.to_vec(), &combined, &utf8_idx, &lc_idx).unwrap()
    }

    #[test]
    fn class_name_resolved() {
        let hprof_data = build_test_hprof();
        let index = full_pipeline(&hprof_data);

        let name = index.class_name(0x200).unwrap();
        assert_eq!(name, Some("java.lang.Integer".to_string()));
    }

    #[test]
    fn class_name_unknown_returns_none() {
        let hprof_data = build_test_hprof();
        let index = full_pipeline(&hprof_data);

        assert_eq!(index.class_name(0xDEAD).unwrap(), None);
    }

    #[test]
    fn instance_fields_resolved() {
        let hprof_data = build_test_hprof();
        let index = full_pipeline(&hprof_data);

        let sub = index.find_instance(0x100).unwrap().unwrap();
        let SubRecord::InstanceDump(inst) = sub else {
            panic!("expected InstanceDump");
        };

        let fields = index.instance_fields(&inst).unwrap();
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].name, "value");
        assert_eq!(fields[0].value, FieldValue::Int(42));
    }

    #[test]
    fn resolve_integer_wrapper() {
        let hprof_data = build_test_hprof();
        let index = full_pipeline(&hprof_data);

        let java_val = index.resolve_value(0x100).unwrap();
        assert!(
            matches!(java_val, JavaValue::Integer(0x100, 42)),
            "expected Integer(0x100, 42), got {:?}",
            java_val
        );
    }

    #[test]
    fn resolve_null() {
        let hprof_data = build_test_hprof();
        let index = full_pipeline(&hprof_data);

        assert!(matches!(index.resolve_value(0).unwrap(), JavaValue::Null));
    }
}
