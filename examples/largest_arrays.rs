//! Print the largest arrays of each primitive type.
//!
//! Large primitive arrays (especially `byte[]` and `int[]`) are a common
//! source of memory pressure. This shows the top entries sorted by size so
//! you can decide which ones warrant further inspection.
//!
//! ```text
//! cargo run --example largest_arrays
//! ```
//!
//! Output:
//! ```text
//! --- byte (top 5) ---
//! 0x1a2b3c  12 582 912 bytes  (12.0 MiB)
//! 0x1a2b40   4 194 304 bytes   (4.0 MiB)
//! ...
//!
//! --- double (top 5) ---
//! 0x2f00a0     65 536 bytes  (64.0 KiB)
//! ...
//! ```

use hprof_toolkit::{
    array_index::ArrayKind, hprof::HprofError, pipeline::build_all_indexes, query::HeapQuery,
};
use std::path::Path;

const TOP_N: usize = 5;

fn main() -> Result<(), HprofError> {
    let path = Path::new("./heap.dump");
    let indexes = build_all_indexes(path)?;
    let query = HeapQuery::open(path, &indexes)?;

    let kinds = [
        ArrayKind::Byte,
        ArrayKind::Int,
        ArrayKind::Long,
        ArrayKind::Double,
        ArrayKind::Float,
        ArrayKind::Short,
        ArrayKind::Char,
        ArrayKind::Boolean,
    ];

    for kind in kinds {
        // iter_arrays_by_size yields entries largest-first.
        let top: Vec<_> = query.iter_arrays_by_size(kind).take(TOP_N).collect();
        if top.is_empty() {
            continue;
        }
        println!("--- {} (top {TOP_N}) ---", kind.display_name());
        for entry in top {
            println!(
                "  0x{:x}  {:>12} bytes  ({:.1} MiB)",
                entry.object_id,
                entry.byte_size,
                entry.byte_size as f64 / (1024.0 * 1024.0),
            );
        }
        println!();
    }

    Ok(())
}
