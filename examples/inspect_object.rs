//! Print everything known about one object: fields with resolved values,
//! retained size, and GC root status.
//!
//!     cargo run --release --example inspect_object -- heap.hprof 0x1a2b3c

use hprof_toolkit::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (Some(path), Some(id)) = (args.next(), args.next()) else {
        eprintln!("usage: inspect_object <heap.hprof> <hex id>");
        std::process::exit(1);
    };
    let id = u64::from_str_radix(id.trim_start_matches("0x"), 16)?;
    let heap = HeapQuery::open(path)?;

    match heap.object(id)? {
        Some(record) => println!("{:#?}", record.resolve(&heap)?),
        None => println!("no object 0x{id:x}"),
    }
    println!("retained: {:?} bytes", heap.retained_size(id));
    println!("root kinds: {:?}", heap.root_types_of(id));
    Ok(())
}
