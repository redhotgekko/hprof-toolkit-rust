*Note: This project is still under construction!*

# hprof-toolkit

A Rust toolkit for analysing Java hprof heap dump files. Designed for very large dumps (250 GB+) where loading the file into memory is not an option — all access is via `mmap` and fixed-size binary index files.

## Usage

All operations go through the `hprof-toolkit` binary.

### Build indexes for a heap dump

```
cargo run --release -- index --hprof path/to/dump.hprof
```

Indexes are written into `dump.indexes/` next to the hprof file. Subsequent runs skip any index that already exists.

The index file formats can be found [here](INDEX_FILE_FORMATS.md).

### Build diff indexes for two dumps

```
cargo run --release -- diff --hprof before.hprof --diff-hprof after.hprof
```

Produces three files — `removed.bin`, `added.bin`, and `common.bin` — enabling object-level comparison between two snapshots of the same JVM.

### Start the HTTP / MCP server

```
cargo run --release -- serve --hprof dump.hprof [--diff-hprof after.hprof] [--port 7000]
```

Starts a jhat-style web interface and an MCP endpoint at `POST /mcp` (JSON-RPC 2.0, MCP spec 2025-03-26). Indexes are built automatically if not already present. Default port is 7000.

## Examples

The `examples/` directory contains self-contained programs that demonstrate common analysis workflows. Run any of them with:

```
cargo run --example <name>
```

| Example | Question it answers |
|---|---|
| `class_histogram` | What is in this dump and which classes dominate? |
| `find_strings` | Where is a specific string value in the heap? |
| `largest_arrays` | Which primitive arrays are consuming the most memory? |
| `inspect_object` | What are the field values of a specific object? |
| `path_to_root` | Why has this object not been garbage collected? |
| `thread_stacks` | What were all live threads doing at dump time? |
| `diff_summary` | What classes grew or shrank between two snapshots? |

## Architecture

Indexing runs in three phases, each producing binary files with fixed-size records to enable chunked concurrent processing.

**Phase 1** scans the hprof file and writes an index of all top-level record tags and their mmap positions.

**Phase 2** parses each heap dump segment concurrently and writes per-dump sub-record index files containing the subtag, hprof offset, and object ID.

**Phase 3** is the query layer. It uses the Phase 2 indexes to locate sub-records on demand and parses them directly from the mmap. No parsed records are kept in memory.

## Development

```
cargo fmt
cargo build
cargo test
cargo clippy
```

`unwrap` and `expect` are allowed only in tests. `unsafe` is not permitted anywhere in the crate (with the exception of mmap usage).

## Right to delete

I reserve the right to delete or make private this repository, and related repositories, at my own discretion without notice.

## License

Licensed under:

Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE.md) or http://www.apache.org/licenses/LICENSE-2.0)

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
licensed as above, without any additional terms or conditions.
