//! Every thread with its stack at dump time.
//!
//!     cargo run --release --example thread_stacks -- heap.hprof

use hprof_toolkit::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: thread_stacks <heap.hprof>")?;
    let heap = HeapQuery::open(path)?;

    for thread in heap.threads()? {
        let ended = if thread.ended { "  [ended]" } else { "" };
        println!("Thread {} {:?}{ended}", thread.serial, thread.name);
        for f in heap.thread_stack(&thread)? {
            println!("  {}{}  ({})", f.method, f.signature, f.location());
        }
    }
    Ok(())
}
