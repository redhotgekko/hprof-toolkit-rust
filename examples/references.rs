//! Who points at an object, and what does it point at? Edges are labelled
//! with the field, static or array slot that holds the reference.
//!
//!     cargo run --release --example references -- heap.hprof 0x1a2b3c

use hprof_toolkit::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (Some(path), Some(id)) = (args.next(), args.next()) else {
        eprintln!("usage: references <heap.hprof> <hex id>");
        std::process::exit(1);
    };
    let id = u64::from_str_radix(id.trim_start_matches("0x"), 16)?;
    let heap = HeapQuery::open(path)?;

    let to = heap.refs_to(id, Page::first(10));
    println!("{} reference(s) to 0x{id:x}:", to.total);
    for from in to.items {
        println!("  <- 0x{from:x}  {}", heap.object_type_name(from));
    }
    let from = heap.refs_from(id, Page::first(10))?;
    println!("{} reference(s) from 0x{id:x}:", from.total);
    for r in from.items {
        println!(
            "  {} -> 0x{:x}  {}",
            r.via,
            r.target,
            heap.object_type_name(r.target)
        );
    }
    Ok(())
}
