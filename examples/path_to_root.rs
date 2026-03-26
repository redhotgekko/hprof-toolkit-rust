//! Find the GC root path that is keeping a specific object alive.
//!
//! When you have identified a suspicious object (e.g. from `class_histogram`
//! or `find_strings`) and want to know *why* it hasn't been collected, this
//! traces the reference chain back to the nearest GC root.
//!
//! ```text
//! cargo run --example path_to_root -- <object_id_hex>
//! ```
//!
//! Example:
//! ```text
//! cargo run --example path_to_root -- 0x1a2b3c4d
//! ```
//!
//! Output:
//! ```text
//! Path to root for 0x1a2b3c4d:
//!   0x1a2b3c4d  java.util.HashMap
//!   0x00000080  [GC root: JNI global]
//! ```

use hprof_toolkit::{
    hprof::HprofError,
    pipeline::build_all_indexes,
    query::{HeapQuery, RootPathResult},
};
use std::path::Path;

fn main() -> Result<(), HprofError> {
    let hex = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: path_to_root <object_id_hex>");
        std::process::exit(1);
    });
    let object_id = u64::from_str_radix(hex.trim_start_matches("0x"), 16).unwrap_or_else(|_| {
        eprintln!("Invalid hex id: {hex}");
        std::process::exit(1);
    });

    let path = Path::new("./heap.dump");
    let indexes = build_all_indexes(path)?;
    let query = HeapQuery::open(path, &indexes)?;

    match query.path_to_root(object_id) {
        RootPathResult::Found(chain) => {
            println!("Path to root for 0x{object_id:x}:");
            for id in &chain {
                let type_name = query.object_type_name(*id)?;
                let root_marker = if query.is_gc_root(*id) {
                    let kinds = query.root_types_of(*id);
                    format!("  [GC root: {:?}]", kinds)
                } else {
                    String::new()
                };
                println!("  0x{id:x}  {type_name}{root_marker}");
            }
        }
        RootPathResult::LimitReached => {
            println!("Search limit reached — object is deeply nested or part of a large graph.");
        }
        RootPathResult::NotReachable => {
            println!(
                "Object 0x{object_id:x} is not reachable from any GC root (already collected?)."
            );
        }
    }

    Ok(())
}
