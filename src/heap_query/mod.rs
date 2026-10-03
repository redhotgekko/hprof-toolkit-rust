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
//! This module is internal plumbing.  The name indexes are built by
//! [`crate::pipeline`], and everything here is reached through
//! [`crate::query::HeapQuery`].

pub mod name_index;
pub mod resolve;

pub use name_index::{LoadClassReader, Utf8IndexReader};
pub use resolve::Field;

use crate::heap_index::sub_record::{
    SUB_INDEX_ENTRY_SIZE, TAG_CLASS_DUMP, TAG_INSTANCE_DUMP, TAG_PRIM_ARRAY_DUMP,
};
use crate::heap_parser::record::FieldValue;
use crate::heap_parser::{InstanceFieldDescriptor, SubIndexReader, SubRecord, parse_sub_record};
use crate::hprof::{BasicType, HprofError, HprofFile, HprofHeader};
use crate::index::StoreWriter;
use crate::resolved::Value;
use name_index::{build_load_class_index, build_utf8_index};
use resolve::{decode_char_array, decode_string_bytes, read_field_value};
use std::borrow::Cow;

// ── Public builder ────────────────────────────────────────────────────────────

/// Scan the record index and build UTF-8 name + load-class indexes.
///
/// Both output files are sorted by their respective key fields so that
/// [`HprofIndex`] can resolve names with O(log n) binary searches.
///
/// Returns `(utf8_count, load_class_count)`.
pub fn build_name_indexes(
    hprof_source: &[u8],
    record_index_source: &[u8],
    utf8_path: &mut dyn StoreWriter,
    load_class_path: &mut dyn StoreWriter,
) -> Result<(u64, u64), HprofError> {
    let hprof = HprofFile::from_ref(hprof_source)?;

    let utf8_count = build_utf8_index(&hprof, record_index_source, utf8_path)?;
    let lc_count = build_load_class_index(&hprof, record_index_source, load_class_path)?;

    Ok((utf8_count, lc_count))
}

// ── HprofIndex ────────────────────────────────────────────────────────────────

/// High-level analysis API over a heap dump and its index files.
///
/// All index files are accessed via borrowed byte slices.  Each method performs
/// O(log n) binary searches against the sorted index files using short-lived
/// reader temporaries.
pub(crate) struct HprofIndex<'a> {
    hprof_data: &'a [u8],
    hprof_header: Cow<'a, HprofHeader>,
    combined_data: &'a [u8],
    utf8_data: &'a [u8],
    lc_data: &'a [u8],
}

impl<'a> HprofIndex<'a> {
    /// Create a validated index from byte slices.
    ///
    /// * `hprof`    — the raw hprof file bytes.
    /// * `combined` — object store index (sorted by object ID).
    /// * `utf8`     — UTF-8 name index (sorted by name ID).
    /// * `lc`       — load-class index (sorted by class ID).
    pub fn from_ref(
        hprof: &'a [u8],
        combined: &'a [u8],
        utf8: &'a [u8],
        lc: &'a [u8],
    ) -> Result<Self, HprofError> {
        let hprof_header = Cow::Owned(HprofHeader::parse(hprof)?);
        SubIndexReader::from_ref(combined)?;
        Utf8IndexReader::from_ref(utf8)?;
        LoadClassReader::from_ref(lc)?;
        Ok(Self {
            hprof_data: hprof,
            hprof_header,
            combined_data: combined,
            utf8_data: utf8,
            lc_data: lc,
        })
    }

    /// Create an index from slices already known to be valid, with a
    /// pre-parsed header.
    pub(crate) fn from_slice(
        hprof: &'a [u8],
        combined: &'a [u8],
        utf8: &'a [u8],
        lc: &'a [u8],
        hprof_header: &'a HprofHeader,
    ) -> Self {
        Self {
            hprof_data: hprof,
            hprof_header: Cow::Borrowed(hprof_header),
            combined_data: combined,
            utf8_data: utf8,
            lc_data: lc,
        }
    }

    // ── Short-lived reader helpers ────────────────────────────────────────────

    fn hprof_file(&self) -> HprofFile<'a> {
        HprofFile::from_parts(self.hprof_data, &self.hprof_header)
    }

    fn combined_reader(&self) -> Result<SubIndexReader<'a>, HprofError> {
        SubIndexReader::from_ref(self.combined_data)
    }

    fn utf8_reader(&self) -> Result<Utf8IndexReader<'a>, HprofError> {
        Utf8IndexReader::from_ref(self.utf8_data)
    }

    fn lc_reader(&self) -> Result<LoadClassReader<'a>, HprofError> {
        LoadClassReader::from_ref(self.lc_data)
    }

    // ── Basic accessors ───────────────────────────────────────────────────────

    /// Return the parsed hprof file header.
    pub fn hprof_header(&self) -> HprofHeader {
        self.hprof_header.clone().into_owned()
    }

    /// hprof identifier size in bytes (4 or 8).
    pub fn id_size(&self) -> u32 {
        self.hprof_header.id_size
    }

    /// Total number of sub-records in the combined index (classes, instances,
    /// arrays, and GC roots).
    pub fn object_count(&self) -> usize {
        self.combined_data.len() / SUB_INDEX_ENTRY_SIZE
    }

    /// Parse the sub-record at position `i` in the combined index.
    ///
    /// Returns `None` when `i >= object_count()`.
    pub fn parse_at(&self, i: usize) -> Result<Option<SubRecord<'a>>, HprofError> {
        let hprof = self.hprof_file();
        match self.combined_reader()?.entry_at(i) {
            Some(entry) => Ok(Some(parse_sub_record(&hprof, &entry)?)),
            None => Ok(None),
        }
    }

    // ── Object lookup ─────────────────────────────────────────────────────────

    /// Parse the sub-record for `object_id` (any tag).
    ///
    /// Uses the object store index for O(log n) lookup.
    /// Returns `None` when the object ID is not present.
    pub fn find_object(&self, object_id: u64) -> Result<Option<SubRecord<'a>>, HprofError> {
        let hprof = self.hprof_file();
        match self.combined_reader()?.find_by_object_id(object_id) {
            Some(entry) => Ok(Some(parse_sub_record(&hprof, &entry)?)),
            None => Ok(None),
        }
    }

    /// Parse the CLASS_DUMP sub-record for `class_id`.
    ///
    /// Searches specifically for a `CLASS_DUMP` tag so that other sub-records
    /// that happen to share the same object ID (e.g. `ROOT_*` entries for
    /// class objects) are skipped.
    pub fn find_class_dump(&self, class_id: u64) -> Result<Option<SubRecord<'a>>, HprofError> {
        let hprof = self.hprof_file();
        match self
            .combined_reader()?
            .find_by_object_id_and_tag(class_id, Some(TAG_CLASS_DUMP))
        {
            Some(entry) => Ok(Some(parse_sub_record(&hprof, &entry)?)),
            None => Ok(None),
        }
    }

    /// Parse the INSTANCE_DUMP sub-record for `object_id`.
    pub fn find_instance(&self, object_id: u64) -> Result<Option<SubRecord<'a>>, HprofError> {
        let hprof = self.hprof_file();
        match self
            .combined_reader()?
            .find_by_object_id_and_tag(object_id, Some(TAG_INSTANCE_DUMP))
        {
            Some(entry) => Ok(Some(parse_sub_record(&hprof, &entry)?)),
            None => Ok(None),
        }
    }

    /// Parse the sub-record described by `entry` directly from the hprof mmap.
    ///
    /// Use this when you already hold a [`SubIndexEntry`] (e.g. from iterating
    /// the combined index) and want to avoid a redundant binary search.
    pub fn parse_entry(
        &self,
        entry: &crate::heap_index::sub_record::SubIndexEntry,
    ) -> Result<SubRecord<'a>, HprofError> {
        let hprof = self.hprof_file();
        parse_sub_record(&hprof, entry)
    }

    // ── Name resolution ───────────────────────────────────────────────────────

    /// Look up the string for `name_id` in the UTF-8 name index.
    pub fn lookup_name(&self, name_id: u64) -> Result<Option<String>, HprofError> {
        let hprof = self.hprof_file();
        self.utf8_reader()?.lookup(&hprof, name_id)
    }

    /// Return the dot-notation class name for `class_id` (e.g. `"java.lang.String"`).
    ///
    /// Uses the load-class index → UTF-8 index chain. Returns `None` when the
    /// class is not found in either index.
    pub fn class_name(&self, class_id: u64) -> Result<Option<String>, HprofError> {
        let name_id = match self.lc_reader()?.find_class_name_id(class_id) {
            Some(id) => id,
            None => return Ok(None),
        };
        let hprof = self.hprof_file();
        let raw = match self.utf8_reader()?.lookup(&hprof, name_id)? {
            Some(s) => s,
            None => return Ok(None),
        };
        // Normalise JVM internal format (e.g. "java/lang/String") to dots.
        Ok(Some(raw.replace('/', ".")))
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
    ) -> Result<Vec<Field>, HprofError> {
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
                fields.push(Field {
                    name,
                    ty: BasicType::from_code_or_err(desc.field_type)?,
                    value,
                });
            }
        }

        Ok(fields)
    }

    // ── Java wrapper type resolution ──────────────────────────────────────────

    /// Resolve the reference `object_id` to a [`Value`].
    ///
    /// * `0` → [`Value::Null`]
    /// * `java.lang.String` → [`Value::String`] (reads the backing array)
    /// * `Integer`, `Long`, `Double`, `Float`, `Short`, `Byte`, `Boolean`,
    ///   `Character` → the matching `Boxed…` variant
    /// * anything else → [`Value::Object`]
    ///
    /// No in-memory caches are used; every resolution is a fresh set of O(log n)
    /// binary searches.
    pub fn resolve_value(&self, object_id: u64) -> Result<Value, HprofError> {
        if object_id == 0 {
            return Ok(Value::Null);
        }
        let inst = match self.find_instance(object_id)? {
            Some(SubRecord::InstanceDump(inst)) => inst,
            _ => return Ok(Value::Object(object_id)),
        };
        let Some(class_name) = self.class_name(inst.class_id)? else {
            return Ok(Value::Object(object_id));
        };
        if class_name == "java.lang.String" {
            return self.resolve_string(&inst);
        }
        if !class_name.starts_with("java.lang.") {
            return Ok(Value::Object(object_id));
        }
        let wrapped = self
            .instance_fields(&inst)?
            .into_iter()
            .find(|f| f.name == "value")
            .map(|f| f.value);
        let id = object_id;
        Ok(match (class_name.as_str(), wrapped) {
            ("java.lang.Integer", Some(FieldValue::Int(v))) => Value::BoxedInt(id, v),
            ("java.lang.Long", Some(FieldValue::Long(v))) => Value::BoxedLong(id, v),
            ("java.lang.Double", Some(FieldValue::Double(v))) => Value::BoxedDouble(id, v),
            ("java.lang.Float", Some(FieldValue::Float(v))) => Value::BoxedFloat(id, v),
            ("java.lang.Short", Some(FieldValue::Short(v))) => Value::BoxedShort(id, v),
            ("java.lang.Byte", Some(FieldValue::Byte(v))) => Value::BoxedByte(id, v),
            ("java.lang.Boolean", Some(FieldValue::Bool(v))) => Value::BoxedBoolean(id, v),
            ("java.lang.Character", Some(FieldValue::Char(v))) => Value::BoxedCharacter(id, v),
            _ => Value::Object(id),
        })
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Resolve `object_id` as a `java.lang.String`.
    ///
    /// Handles both pre-Java-9 (`char[]` backing) and Java 9+ compact strings
    /// (`byte[]` + `coder` field).
    pub(crate) fn resolve_string(
        &self,
        inst: &crate::heap_parser::InstanceDump<'_>,
    ) -> Result<Value, HprofError> {
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
            _ => return Ok(Value::String(inst.object_id, String::new())),
        };

        // Look up the backing array.
        let arr_entry = match self
            .combined_reader()?
            .find_by_object_id_and_tag(arr_id, Some(TAG_PRIM_ARRAY_DUMP))
        {
            Some(e) => e,
            None => return Ok(Value::String(inst.object_id, String::new())),
        };

        let hprof = self.hprof_file();
        let arr_record = parse_sub_record(&hprof, &arr_entry)?;
        let SubRecord::PrimArrayDump(arr) = arr_record else {
            return Ok(Value::String(inst.object_id, String::new()));
        };

        let text = match arr.element_type {
            5 => decode_char_array(arr.data),          // char[]
            8 => decode_string_bytes(arr.data, coder), // byte[]
            _ => return Ok(Value::String(inst.object_id, String::new())),
        };

        Ok(Value::String(inst.object_id, text))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{IndexStore, MemStore, names};
    use crate::pipeline::{IndexOptions, build_indexes};
    use crate::progress::NoProgress;
    use crate::test_util::{ClassSpec, HprofBuilder, ty};

    /// Integer(0x200 ⊂ Object 0x300) with one int field "value";
    /// INSTANCE 0x100 = Integer(42).
    fn test_heap() -> Vec<u8> {
        HprofBuilder::new(8)
            .utf8(1, "value")
            .utf8(2, "java/lang/Integer")
            .utf8(3, "java/lang/Object")
            .load_class(1, 0x200, 2)
            .load_class(2, 0x300, 3)
            .class_dump(
                ClassSpec::new(0x200)
                    .super_class(0x300)
                    .instance_size(4)
                    .field(1, ty::INT),
            )
            .class_dump(ClassSpec::new(0x300))
            .instance_values(0x100, 0x200, &[FieldValue::Int(42)])
            .build()
    }

    /// Run the real pipeline in memory and keep the three entries
    /// [`HprofIndex`] needs.
    fn full_pipeline(hprof_data: &[u8]) -> Pipeline {
        let store = MemStore::new();
        let opts = IndexOptions {
            retained: false,
            force: false,
            ..IndexOptions::default()
        };
        build_indexes(hprof_data, &store, &opts, &NoProgress).unwrap();
        Pipeline {
            hprof_data: hprof_data.to_vec(),
            combined: store.open(names::OBJECT_STORE).unwrap(),
            utf8_idx: store.open(names::UTF8).unwrap(),
            lc_idx: store.open(names::LOAD_CLASS).unwrap(),
        }
    }

    struct Pipeline {
        hprof_data: Vec<u8>,
        combined: crate::index::ByteSource,
        utf8_idx: crate::index::ByteSource,
        lc_idx: crate::index::ByteSource,
    }

    impl Pipeline {
        fn create_hprof_index(&self) -> HprofIndex<'_> {
            HprofIndex::from_ref(
                &self.hprof_data,
                self.combined.as_ref(),
                self.utf8_idx.as_ref(),
                self.lc_idx.as_ref(),
            )
            .unwrap()
        }
    }

    #[test]
    fn class_name_resolved() {
        let index = full_pipeline(&test_heap());
        let name = index.create_hprof_index().class_name(0x200).unwrap();
        assert_eq!(name, Some("java.lang.Integer".to_string()));
    }

    #[test]
    fn class_name_unknown_returns_none() {
        let index = full_pipeline(&test_heap());
        assert_eq!(index.create_hprof_index().class_name(0xDEAD).unwrap(), None);
    }

    #[test]
    fn instance_fields_resolved() {
        let index = full_pipeline(&test_heap());
        let hprof_index = index.create_hprof_index();
        let sub = hprof_index.find_instance(0x100).unwrap().unwrap();
        let SubRecord::InstanceDump(inst) = sub else {
            panic!("expected InstanceDump");
        };

        let fields = index.create_hprof_index().instance_fields(&inst).unwrap();
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].name, "value");
        assert_eq!(fields[0].value, FieldValue::Int(42));
    }

    #[test]
    fn resolve_integer_wrapper() {
        let index = full_pipeline(&test_heap());
        let java_val = index.create_hprof_index().resolve_value(0x100).unwrap();
        assert!(
            matches!(java_val, Value::BoxedInt(0x100, 42)),
            "expected BoxedInt(0x100, 42), got {java_val:?}"
        );
    }

    #[test]
    fn resolve_null() {
        let index = full_pipeline(&test_heap());
        assert!(matches!(
            index.create_hprof_index().resolve_value(0).unwrap(),
            Value::Null
        ));
    }
}
