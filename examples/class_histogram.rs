//! The classes with the most live instances, straight from the precomputed
//! histogram (the dump itself is never scanned).
//!
//!     cargo run --release --example class_histogram -- heap.hprof

use hprof_toolkit::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: class_histogram <heap.hprof>")?;
    let heap = HeapQuery::open(path)?;

    println!("{:>10} {:>12}  class", "count", "shallow B");
    for row in heap.class_histogram(Page::first(30)).items {
        println!(
            "{:>10} {:>12}  {}",
            row.instance_count, row.shallow_bytes, row.class_name
        );
    }
    Ok(())
}
