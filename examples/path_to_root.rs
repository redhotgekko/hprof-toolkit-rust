//! Why is this object still alive? Prints the reference chain from a GC root,
//! naming the field that holds each link.
//!
//!     cargo run --release --example path_to_root -- heap.hprof 0x1a2b3c

use hprof_toolkit::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (Some(path), Some(id)) = (args.next(), args.next()) else {
        eprintln!("usage: path_to_root <heap.hprof> <hex id>");
        std::process::exit(1);
    };
    let id = u64::from_str_radix(id.trim_start_matches("0x"), 16)?;
    let heap = HeapQuery::open(path)?;

    let path = heap.path_to_root(id, &RootPathLimits::default());
    println!(
        "{:?} after visiting {} objects",
        path.outcome, path.nodes_visited
    );
    for step in &path.steps {
        let via = step.via.as_ref().map(|e| e.to_string()).unwrap_or_default();
        println!(
            "{via:<14} 0x{:x}  {}",
            step.object_id,
            heap.object_type_name(step.object_id)
        );
    }
    Ok(())
}
