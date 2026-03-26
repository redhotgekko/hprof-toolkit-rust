//! Compare two heap dumps and show which classes grew or shrank.
//!
//! Both dumps must have been taken from the same JVM process (so object IDs
//! are stable). Run this after a suspected memory leak to identify the
//! accumulating class.
//!
//! ```text
//! cargo run --example diff_summary -- before.dump after.dump
//! ```
//!
//! Output:
//! ```text
//! Total objects: before=1 200 000  after=1 450 000  net=+250 000
//!
//!  net change  class
//! ----------   -----
//!    +50 000   java.util.LinkedList$Node
//!     -5 000   java.lang.ref.WeakReference
//!       ...
//! ```

use hprof_toolkit::{
    diff::compute_diff_summary, diff_index::DiffIndexPaths, hprof::HprofError,
    pipeline::build_all_indexes, query::HeapQuery,
};
use std::path::PathBuf;

fn main() -> Result<(), HprofError> {
    let args: Vec<String> = std::env::args().collect();
    let (before_path, after_path) = match (args.get(1), args.get(2)) {
        (Some(a), Some(b)) => (PathBuf::from(a), PathBuf::from(b)),
        _ => {
            eprintln!("Usage: diff_summary <before.dump> <after.dump>");
            std::process::exit(1);
        }
    };

    // Build (or reuse) all indexes for both dumps.
    eprintln!("Indexing {} …", before_path.display());
    let before_idx = build_all_indexes(&before_path)?;
    eprintln!("Indexing {} …", after_path.display());
    let after_idx = build_all_indexes(&after_path)?;

    let before_query = HeapQuery::open(&before_path, &before_idx)?;
    let after_query = HeapQuery::open(&after_path, &after_idx)?;

    // Build (or reuse) the three diff index files.
    let diff_paths = DiffIndexPaths::for_hprofs(&before_path, &after_path);
    if !diff_paths.all_exist() {
        eprintln!("Building diff indexes …");
        hprof_toolkit::diff_index::build_diff_indexes(&before_query, &after_query, &diff_paths)?;
    }

    let summary = compute_diff_summary(&before_query, &after_query, &diff_paths)?;

    println!(
        "Total objects: before={}  after={}  net={:+}",
        summary.total_before,
        summary.total_after,
        summary.total_after as i64 - summary.total_before as i64,
    );
    println!();

    // Sort by absolute net change, largest first.
    let mut rows = summary.by_class;
    rows.sort_unstable_by_key(|r| -(r.net_change().abs()));

    println!("{:>12}  class", "net change");
    println!("{:->12}  {}", "", "-".repeat(50));
    for row in rows.iter().take(30) {
        println!("{:>+12}  {}", row.net_change(), row.class_name);
    }

    Ok(())
}
