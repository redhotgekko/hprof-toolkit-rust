//! Which classes grew or shrank between two dumps of the same JVM?
//!
//!     cargo run --release --example diff_summary -- before.hprof after.hprof

use hprof_toolkit::prelude::*;
use std::{path::PathBuf, sync::Arc};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1).map(PathBuf::from);
    let (Some(before_path), Some(after_path)) = (args.next(), args.next()) else {
        eprintln!("usage: diff_summary <before.hprof> <after.hprof>");
        std::process::exit(1);
    };
    let before = Arc::new(HeapQuery::open(&before_path)?);
    let after = Arc::new(HeapQuery::open(&after_path)?);
    let diff = HeapDiff::open(before, after, &before_path, &after_path)?;

    let summary = diff.summary()?;
    println!(
        "objects: {} -> {}",
        summary.total_before, summary.total_after
    );
    let mut rows: Vec<_> = summary.by_class.iter().collect();
    rows.sort_by_key(|r| -r.net_change().abs());
    for row in rows.iter().take(30) {
        println!("{:>+10}  {}", row.net_change(), row.class_name);
    }
    Ok(())
}
