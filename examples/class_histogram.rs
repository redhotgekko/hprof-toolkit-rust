//! Count live instances by class name and print a histogram.
//!
//! This is the first thing to run when exploring an unfamiliar heap dump —
//! it tells you what is actually in there and which classes are dominant.
//!
//! ```text
//! cargo run --example class_histogram
//! ```
//!
//! Output (trimmed):
//! ```text
//!   count  class
//!  ------  -----
//!  481 920  [B
//!  213 408  java.lang.String
//!   98 304  java.util.HashMap$Node
//!      ...
//! ```

use hprof_toolkit::{hprof::HprofError, pipeline::build_all_indexes, query::HeapQuery};
use std::{collections::HashMap, path::Path, sync::Mutex};

fn main() -> Result<(), HprofError> {
    let path = Path::new("./heap.dump");
    let indexes = build_all_indexes(path)?;
    let query = HeapQuery::open(path, &indexes)?;

    // Accumulate per-class counts in parallel.
    let counts: Mutex<HashMap<String, u64>> = Mutex::new(HashMap::new());

    query.par_resolved_instances(|inst| {
        counts
            .lock()
            .map_err(|_| HprofError::InvalidIndexFile)?
            .entry(inst.class_name)
            .and_modify(|n| *n += 1)
            .or_insert(1);
        Ok(())
    })?;

    let mut counts = counts
        .into_inner()
        .map_err(|_| HprofError::InvalidIndexFile)?;
    let mut rows: Vec<(u64, String)> = counts.drain().map(|(k, v)| (v, k)).collect();
    rows.sort_unstable_by(|a, b| b.0.cmp(&a.0));

    println!("{:>10}  class", "count");
    println!("{:->10}  {}", "", "-".repeat(40));
    for (count, class) in rows.iter().take(30) {
        println!("{count:>10}  {class}");
    }

    Ok(())
}
