//! Print the fully resolved fields of a specific object by ID.
//!
//! Once you have an interesting `object_id` (from `class_histogram`,
//! `find_strings`, or `path_to_root`) use this to inspect its contents,
//! including the class hierarchy and all inherited fields.
//!
//! ```text
//! cargo run --example inspect_object -- <object_id_hex>
//! ```

use hprof_toolkit::{
    heap_parser::SubRecord, hprof::HprofError, pipeline::build_all_indexes, query::HeapQuery,
    resolved::ResolvedInstance,
};
use std::path::Path;

fn main() -> Result<(), HprofError> {
    let hex = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: inspect_object <object_id_hex>");
        std::process::exit(1);
    });
    let object_id = u64::from_str_radix(hex.trim_start_matches("0x"), 16).unwrap_or_else(|_| {
        eprintln!("Invalid hex id: {hex}");
        std::process::exit(1);
    });

    let path = Path::new("./heap.dump");
    let indexes = build_all_indexes(path)?;
    let query = HeapQuery::open(path, &indexes)?;

    match query.find(object_id)? {
        Some(SubRecord::InstanceDump(inst)) => {
            let resolved = ResolvedInstance::from_dump(&query, &inst)?;
            println!("object_id : 0x{:x}", resolved.object_id);
            println!("class     : {}", resolved.class_name);
            println!("trace     : serial={}", resolved.stack_trace_serial);
            println!();
            println!("fields:");
            for field in resolved.fields {
                println!("  {:30}= {:?}", field.name, field.value);
            }
        }
        Some(SubRecord::ClassDump(cd)) => {
            let resolved = query.resolve_class(&cd)?;
            println!("object_id : 0x{:x}  (class dump)", resolved.class_id);
            println!("class     : {}", resolved.class_name);
            if let Some(super_name) = &resolved.super_class_name {
                println!("super     : {super_name}");
            }
        }
        Some(SubRecord::PrimArrayDump(arr)) => {
            println!("object_id : 0x{object_id:x}  (primitive array)");
            println!("element   : type code {}", arr.element_type);
            println!("length    : {}", arr.num_elements);
        }
        Some(SubRecord::ObjArrayDump(arr)) => {
            println!("object_id : 0x{object_id:x}  (object array)");
            println!("length    : {}", arr.num_elements);
            println!("elements  :");
            for elem in arr.elements() {
                println!("  0x{elem:x}");
            }
        }
        Some(other) => {
            println!("object_id : 0x{object_id:x}  (GC root: {other:?})");
        }
        None => {
            println!("No object found with id 0x{object_id:x}");
        }
    }

    Ok(())
}
