//! Class field layouts, cached once per build step.
//!
//! Walking an instance's reference fields needs the field types of its class
//! and every superclass.  Looking those up in the object store per instance is
//! a binary search per level; [`ClassCache`] reads every class dump once into
//! a sorted `O(classes)` table instead.  Both the dominator builder and the
//! reference-index builder use it.

use crate::heap_index::sub_record::TAG_CLASS_DUMP;
use crate::heap_parser::record::FieldValue;
use crate::heap_parser::{InstanceDump, SubIndexReader, SubRecord, parse_sub_record};
use crate::heap_query::resolve::read_field_value;
use crate::hprof::{HprofError, HprofFile};

/// Per-class instance field metadata, used to walk field data without repeated
/// binary searches into the combined index.
pub(crate) struct ClassLayout {
    pub super_class_id: u64,
    /// hprof field-type code for each instance field in declaration order.
    /// Type 2 = object reference (id_size bytes); others are primitive types.
    pub field_types: Vec<u8>,
}

/// Sorted-array map from class_id → ClassLayout; binary-search lookup.
///
/// Avoids any hash-map overhead: build once, share read-only across threads.
pub(crate) struct ClassCache {
    ids: Vec<u64>,
    layouts: Vec<ClassLayout>,
}

impl ClassCache {
    pub fn get(&self, class_id: u64) -> Option<&ClassLayout> {
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
pub(crate) fn build_class_cache(
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

/// Count non-null outgoing object references in an instance dump without
/// allocating a collection.  Used by Pass 1 to build exact CSR out-degrees.
pub(crate) fn count_instance_refs(
    inst: &InstanceDump<'_>,
    class_cache: &ClassCache,
    id_size: usize,
) -> Result<usize, HprofError> {
    let mut count = 0usize;
    for_each_instance_ref(inst, class_cache, id_size, &mut |_| {
        count += 1;
        Ok(())
    })?;
    Ok(count)
}

/// Call `emit(id)` for every non-null outgoing object reference of an
/// instance, in field order (own fields first, then each superclass).
pub(crate) fn for_each_instance_ref(
    inst: &InstanceDump<'_>,
    class_cache: &ClassCache,
    id_size: usize,
    emit: &mut dyn FnMut(u64) -> Result<(), HprofError>,
) -> Result<(), HprofError> {
    let mut offset = 0usize;
    let mut curr = inst.class_id;
    while curr != 0 {
        let Some(layout) = class_cache.get(curr) else {
            break;
        };
        for &ft in &layout.field_types {
            let (value, sz) = read_field_value(inst.data, offset, ft, id_size)?;
            offset += sz;
            if let FieldValue::Object(id) = value
                && id != 0
            {
                emit(id)?;
            }
        }
        curr = layout.super_class_id;
    }
    Ok(())
}
