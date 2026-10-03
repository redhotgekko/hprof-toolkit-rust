//! The largest arrays of each element kind.
//!
//!     cargo run --release --example largest_arrays -- heap.hprof

use hprof_toolkit::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: largest_arrays <heap.hprof>")?;
    let heap = HeapQuery::open(path)?;

    for kind in ArrayKind::ALL {
        println!(
            "--- {} ({} arrays) ---",
            kind.display_name(),
            heap.array_count(kind)
        );
        // Entries come largest first.
        for e in heap.iter_arrays_by_size(kind).take(5) {
            println!("  0x{:x}  {:>12} bytes", e.object_id, e.byte_size);
        }
    }
    Ok(())
}
