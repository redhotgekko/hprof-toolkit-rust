//! Find loaded classes by name: substring (default), `exact`, `regex` or
//! `fuzzy`, with the instance count of each match.
//!
//!     cargo run --release --example search_classes -- heap.hprof 'java\.util\..*Map$' regex
//!     cargo run --release --example search_classes -- heap.hprof hm fuzzy

use hprof_toolkit::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (Some(path), Some(text)) = (args.next(), args.next()) else {
        eprintln!("usage: search_classes <heap.hprof> <text> [contains|exact|regex|fuzzy]");
        std::process::exit(1);
    };
    let mode = match args.next() {
        None => SearchMode::Contains,
        Some(m) => {
            SearchMode::from_name(&m).ok_or("mode must be contains, exact, regex or fuzzy")?
        }
    };
    let heap = HeapQuery::open(path)?;
    let matcher = Matcher::new(&SearchQuery::new(text, mode))?;
    let found = heap.search_classes(&matcher, Page::first(50));
    for c in &found.items {
        println!("{:>10}  {}", c.instance_count, c.name);
    }
    println!(
        "{} of {} matching classes shown",
        found.items.len(),
        found.total
    );
    Ok(())
}
