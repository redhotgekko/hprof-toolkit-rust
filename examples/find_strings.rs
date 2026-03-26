//! Search all live `java.lang.String` instances for a substring.
//!
//! Useful for hunting down which objects are holding onto a specific
//! configuration value, URL, or player name that you know should appear
//! in the heap.
//!
//! ```text
//! cargo run --example find_strings -- <needle>
//! ```
//!
//! Output:
//! ```text
//! object_id=0x1a2b3c4d  "localhost:25565"
//! object_id=0x1a2b3c4e  "localhost:19132"
//! ```

use hprof_toolkit::{
    heap_query::JavaValue, hprof::HprofError, pipeline::build_all_indexes, query::HeapQuery,
};
use std::path::Path;

fn main() -> Result<(), HprofError> {
    let needle = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: find_strings <needle>");
        std::process::exit(1);
    });

    let path = Path::new("./heap.dump");
    let indexes = build_all_indexes(path)?;
    let query = HeapQuery::open(path, &indexes)?;

    // resolve_value follows the backing byte[] + coder field to decode the string.
    query.par_resolved_instances_of("java.lang.String", |inst| {
        if let Ok(JavaValue::String(_, s)) = query.resolve_value(inst.object_id)
            && s.contains(needle.as_str())
        {
            println!("object_id=0x{:x}  {:?}", inst.object_id, s);
        }
        Ok(())
    })?;

    Ok(())
}
