//! The objects that keep the most memory alive (retained size), and what each
//! one directly dominates. Needs the retained index: run
//! `hprof-toolkit index --hprof heap.hprof` first.
//!
//!     cargo run --release --example retained_top -- heap.hprof

use hprof_toolkit::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: retained_top <heap.hprof>")?;
    let heap = HeapQuery::open_existing(path)?;

    let top = heap
        .retained_top(Page::first(10))
        .ok_or(HprofError::NotIndexed("retained heap"))?;
    for (id, bytes) in top.items {
        println!("{bytes:>14} B  0x{id:x}  {}", heap.object_type_name(id));
        let children = heap
            .dominated_by(id, Page::first(3))
            .map(|p| p.items)
            .unwrap_or_default();
        for (child, kept) in children {
            println!(
                "{kept:>14} B    keeps 0x{child:x}  {}",
                heap.object_type_name(child)
            );
        }
    }
    Ok(())
}
