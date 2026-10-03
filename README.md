*Note: This project is still under construction!*

# hprof-toolkit

Explore Java heap dumps (`.hprof`) that are too big to load into memory.

A one-off indexing pass writes fixed-size, sorted binary files next to the dump. These are used to navigate the dump.

You can use it three ways:

* **As an AI tool.** An MCP server lets a MCP client investigate a heap: histograms, retained sizes, reference chains to GC roots, threads, and diffs between two dumps.
* **As a library.** A small Rust API for ad-hoc analysis programs.
* **In a browser.** A jhat-style HTML UI on localhost, with substring, regex and fuzzy search over classes, threads and string contents.

## Quick start

```
cargo build --release

# Build the indexes once.
target/release/hprof-toolkit index path/to/dump.hprof

# Then launch a front end:
target/release/hprof-toolkit mcp   path/to/dump.hprof     # MCP over stdin/stdout
target/release/hprof-toolkit serve path/to/dump.hprof     # HTML UI on http://localhost:7000
```

`index` writes `dump.indexes/` next to the dump (or use `--index-dir`). Finished steps are recorded in a manifest, so a second run skips them and an interrupted run resumes where it stopped. `serve` and `mcp` build any missing *cheap* indexes themselves, with progress on stderr.

### Command line

| Command | What it does |
|---|---|
| `index <HPROF>` | Build every index, including retained sizes. `--no-retained` skips the dominator step, `--force` rebuilds, `--max-memory <MiB>` caps that step. |
| `diff <HPROF> --diff-hprof <HPROF>` | Build the comparison between two dumps of the same JVM process. |
| `serve <HPROF> [--diff-hprof <HPROF>] [--port N]` | HTML UI on localhost, plus MCP over HTTP at `POST /mcp`. |
| `mcp <HPROF> [--diff-hprof <HPROF>]` | MCP server on stdin/stdout. |

Common flags: `--index-dir <DIR>`, `--threads <N>`. The dump may also be given as `--hprof <path>` / `-f <path>`. Run any command with `--help`.

Retained sizes need the dominator tree, the one step whose memory grows with the heap (about 72 bytes per object plus 12 per reference). It is only started by `index`, is refused up front when the estimate exceeds available memory, and is optional: everything else works without it.

## Use it from an AI client

Add the server to Claude Code:

```
claude mcp add hprof -- /path/to/hprof-toolkit mcp /path/to/dump.hprof
```

or to any client that launches MCP servers as subprocesses (Claude Desktop, and others):

```json
{ "mcpServers": { "hprof": { "command": "/path/to/hprof-toolkit", "args": ["mcp", "/path/to/dump.hprof"] } } }
```

Build the indexes first (`index`) if you want retained sizes. Stdout carries only protocol messages; progress goes to stderr.

The server gives the model a workflow in its `initialize` instructions and an `investigate_memory_leak` prompt. Tools return JSON (`structuredContent`) plus readable text, write ids as `"0x…"`, page every list with `offset`/`limit` and report `has_more`, and turn mistakes into actionable `isError` results.

| Tool | Use |
|---|---|
| `heap_summary` | Start here: scale, GC roots, which indexes exist. |
| `class_histogram`, `find_classes`, `class_info`, `instances_of_class` | Classes and their instances. `find_classes` and `class_histogram` take a `name` with a `mode`: `contains`, `exact`, `regex` or `fuzzy`. |
| `object`, `object_string`, `array_contents` | Read one object: fields, strings, array slices. |
| `references_to`, `references_from`, `path_to_gc_root`, `gc_roots` | Why an object is alive; edges are labelled with the holding field. |
| `retained_top`, `dominated_by` | What keeps the most memory alive (needs the retained index). |
| `threads`, `thread_stack` | What was running; `threads` filters by `name` too. |
| `largest_arrays`, `search_strings` | Big arrays; search over String contents (`contains`, `exact` or `regex`) in bounded, resumable slices. |
| `heap_diff_summary`, `heap_diff_objects` | Compare two dumps (start with `--diff-hprof`). |

The HTTP server listens on localhost only and has no authentication. `POST /mcp` (served by `serve`) speaks the same protocol for clients that use MCP over HTTP: it returns plain JSON, with no sessions or streaming, answers `GET` with 405, and refuses requests whose `Origin` header is not `localhost`, `127.0.0.1` or `[::1]` (the spec requires this check, so a web page cannot drive the server through your browser).

## Use it as a library

```rust
use hprof_toolkit::prelude::*;

fn main() -> Result<(), HprofError> {
    // Builds the indexes on first use, then just opens them.
    let heap = HeapQuery::open("heap.hprof")?;

    // Biggest classes, straight from a precomputed index.
    for row in heap.class_histogram(Page::first(5)).items {
        println!("{:>10}  {}", row.instance_count, row.class_name);
    }

    // Why is this object alive?
    let path = heap.path_to_root(0x1a2b3c, &RootPathLimits::default());
    for step in &path.steps {
        println!("{:?} 0x{:x}", step.via, step.object_id);
    }
    Ok(())
}
```

`HeapQuery` is the one entry point: objects and classes, name-resolved views (`SubRecord::resolve`), parallel iteration with rayon (`par_instances_of`), references and paths, retained sizes, threads, and paged results everywhere (`Page`, `PageResult`). `HeapDiff` compares two dumps. Tests and embedders can build the same indexes in memory with `HeapQuery::from_store` and a `MemStore`; no file is involved.

Searching uses one `Matcher`, built from a `SearchQuery` in one of four modes: `contains` (the default), `exact`, `regex` or `fuzzy` (ranked subsequence, for names typed from memory). All are case-insensitive unless asked otherwise. `search_classes` and `class_histogram_filtered` apply it to class names, `threads_matching` to thread names, and `search_strings` to `java.lang.String` contents in bounded, resumable windows (`ScanWindow`, `ScanResult`). A `Matcher` is `Send + Sync`, so one can serve every rayon worker:

```rust
let m = Matcher::new(&SearchQuery::regex(r"^java\.util\..*Map$"))?;
for c in heap.search_classes(&m, Page::first(20)).items {
    println!("{:>10}  {}", c.instance_count, c.name);
}
```

The [`examples/`](examples) directory has short programs (each under 40 lines, dump path as the first argument):

| Example | Question it answers |
|---|---|
| `class_histogram` | What is in this dump and which classes dominate? |
| `retained_top` | Which objects keep the most memory alive? |
| `path_to_root` | Why has this object not been garbage collected? |
| `references` | Who points at this object, and what does it point at? |
| `inspect_object` | What are the fields, retained size and roots of one object? |
| `search_classes` | Which classes match this name, regex or fuzzy pattern? |
| `find_strings` | Where are the strings matching this regex? |
| `largest_arrays` | Which arrays are consuming the most memory? |
| `thread_stacks` | What were all threads doing at dump time? |
| `diff_summary` | Which classes grew or shrank between two snapshots? |

```
cargo run --release --example class_histogram -- dump.hprof
```

## How it works

```
 CLI  (index | diff | serve | mcp)                              src/main.rs
   │
 server: HTML UI · MCP (stdio + HTTP)                           src/server/
   │        transport, formatting and paging only
 analysis: HeapQuery · HeapDiff · resolved views ·              src/query.rs, graph.rs,
           graph, threads, classes, retained                    classes.rs, threads.rs, diff.rs
   │
 index:    IndexStore (files or memory) · sorted fixed-record   src/index/, pipeline.rs
           files · builders · manifest
   │
 format:   hprof header, records and sub-records, parsed        src/hprof/, src/heap_parser/
           in place from the mmap
```

**Indexing** scans the dump once to locate records, then builds sorted files of fixed-size records: every object (`object_store.bin`), names, back-references, GC roots, arrays by size, instances by class, a class histogram, and optionally the dominator tree and retained sizes. Big builds stream through parallel workers and sort in place; nothing proportional to the number of objects is held in RAM except the dominator step. See [INDEX_FILE_FORMATS.md](INDEX_FILE_FORMATS.md) for every file.

**Querying** is a binary search on the sorted files followed by parsing the record directly from the mmap. A per-class index means "all instances of X" is one range scan; a precomputed histogram and retained ranking mean the common questions never scan the heap.

**Storage** goes through one abstraction, `IndexStore`, with a directory backend (staged writes, atomic renames) and an in-memory backend. Every build and query path runs against both, which is why nearly all tests use no files at all. One integration test, `tests/fs_backend.rs`, proves the two agree.

## Development

```
cargo fmt
cargo clippy --all-targets -- -D warnings
cargo test
```

CI runs the three commands above (fmt as `cargo fmt --check`), and the crate needs Rust 1.95 or newer (`rust-version` in `Cargo.toml`; the floor is set by the `sysinfo` dependency). `unwrap` and `expect` are allowed only in tests. `unsafe` appears only where the index store memory-maps a file. Unit tests must not touch the filesystem; `tests/fs_backend.rs` is the single exception.

## Right to delete

I reserve the right to delete or make private this repository, and related repositories, at my own discretion without notice.

## License

Licensed under:

Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE.md) or http://www.apache.org/licenses/LICENSE-2.0)

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
licensed as above, without any additional terms or conditions.
