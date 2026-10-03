//! Find `java.lang.String` objects whose text matches a regex, in parallel.
//! Only the String instances are visited (per-class index), and one
//! `Matcher` serves every rayon worker.
//!
//!     cargo run --release --example find_strings -- heap.hprof '^jdbc:'

use hprof_toolkit::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (Some(path), Some(pattern)) = (args.next(), args.next()) else {
        eprintln!("usage: find_strings <heap.hprof> <regex>");
        std::process::exit(1);
    };
    let heap = HeapQuery::open(path)?;
    let matcher = Matcher::new(&SearchQuery::regex(pattern))?;
    let Some(string_class) = heap.find_class_by_name("java.lang.String") else {
        return Ok(());
    };
    heap.par_instances_of(string_class).try_for_each(|inst| {
        let id = inst?.object_id;
        if let Some(text) = heap.string(id)?.filter(|s| matcher.is_match(s)) {
            println!("0x{id:x}  {text:?}");
        }
        Ok::<(), HprofError>(())
    })?;
    Ok(())
}
