//! The only tests that touch the disk.
//!
//! Every other test builds and queries indexes in memory.  These prove the
//! on-disk backend behaves like the in-memory one:
//!
//! * `the_filesystem_backend_matches_the_in_memory_backend` generates a
//!   10 000-object heap dump, indexes it into a temporary directory with the
//!   filesystem store, reopens it with `open_existing`, and compares the
//!   results with the same dump indexed into a `MemStore`.
//! * `the_filesystem_store_stages_commits_and_cleans_up` exercises the store
//!   operations the crash-resume guarantee rests on: staged writes, atomic
//!   replacement, discarding, prefix removal, and a stale manifest.

use hprof_toolkit::index::{ByteSource, FsStore, IndexStore, MemStore};
use hprof_toolkit::pipeline::{IndexOptions, build_indexes};
use hprof_toolkit::prelude::*;
use std::path::PathBuf;

const CHAINS: u64 = 10;
const CHAIN_LEN: u64 = 1000;
const NODE_CLASS: u64 = 0x10;

fn node_id(chain: u64, pos: u64) -> u64 {
    0x10_000 + chain * CHAIN_LEN + pos
}

fn record(out: &mut Vec<u8>, tag: u8, body: &[u8]) {
    out.push(tag);
    out.extend_from_slice(&0u32.to_be_bytes());
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    out.extend_from_slice(body);
}

/// Ten chains of 1000 `Node { Node next }` objects.  Position 0 of each chain
/// is the head and is a GC root; every other node is referenced by its
/// predecessor only, so retained sizes shrink along the chain.
fn generate_hprof() -> Vec<u8> {
    generate_hprof_with(CHAINS)
}

/// [`generate_hprof`] with a chosen number of chains.
fn generate_hprof_with(chains: u64) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"JAVA PROFILE 1.0.2\0");
    out.extend_from_slice(&8u32.to_be_bytes());
    out.extend_from_slice(&0u64.to_be_bytes());

    for (id, name) in [(1u64, "Node"), (2, "next")] {
        let mut body = id.to_be_bytes().to_vec();
        body.extend_from_slice(name.as_bytes());
        record(&mut out, 0x01, &body);
    }
    let mut load = 1u32.to_be_bytes().to_vec();
    load.extend_from_slice(&NODE_CLASS.to_be_bytes());
    load.extend_from_slice(&0u32.to_be_bytes());
    load.extend_from_slice(&1u64.to_be_bytes());
    record(&mut out, 0x02, &load);

    let mut seg = Vec::new();
    // CLASS_DUMP: no superclass, one object field `next`.
    seg.push(0x20);
    seg.extend_from_slice(&NODE_CLASS.to_be_bytes());
    seg.extend_from_slice(&0u32.to_be_bytes());
    for _ in 0..6 {
        seg.extend_from_slice(&0u64.to_be_bytes()); // super, loader, signers, domain, 2 reserved
    }
    seg.extend_from_slice(&8u32.to_be_bytes()); // instance size
    seg.extend_from_slice(&0u16.to_be_bytes()); // constant pool
    seg.extend_from_slice(&0u16.to_be_bytes()); // statics
    seg.extend_from_slice(&1u16.to_be_bytes()); // instance fields
    seg.extend_from_slice(&2u64.to_be_bytes());
    seg.push(2); // object

    for chain in 0..chains {
        for pos in 0..CHAIN_LEN {
            let next = if pos + 1 < CHAIN_LEN {
                node_id(chain, pos + 1)
            } else {
                0
            };
            seg.push(0x21);
            seg.extend_from_slice(&node_id(chain, pos).to_be_bytes());
            seg.extend_from_slice(&0u32.to_be_bytes());
            seg.extend_from_slice(&NODE_CLASS.to_be_bytes());
            seg.extend_from_slice(&8u32.to_be_bytes());
            seg.extend_from_slice(&next.to_be_bytes());
        }
        seg.push(0xFF); // ROOT_UNKNOWN
        seg.extend_from_slice(&node_id(chain, 0).to_be_bytes());
    }
    record(&mut out, 0x1C, &seg);
    out
}

/// A fresh temporary directory, removed when the test ends, pass or fail.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("hprof-fs-backend-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("temp dir: {e}"));
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Everything the test compares.
#[derive(Debug, PartialEq)]
struct Answers {
    objects: usize,
    references: usize,
    histogram: Vec<(String, u64, u64)>,
    retained: Vec<(u64, u64)>,
    sample_retained: Vec<Option<u64>>,
    path: Vec<u64>,
    entries: Vec<String>,
}

fn answers(heap: &HeapQuery, entries: Vec<String>) -> Answers {
    let histogram = heap
        .class_histogram(Page::first(100))
        .items
        .into_iter()
        .map(|e| (e.class_name, e.instance_count, e.shallow_bytes))
        .collect();
    let deep_tail = node_id(3, CHAIN_LEN - 1);
    let path = heap.path_to_root(deep_tail, &RootPathLimits::default());
    assert_eq!(path.outcome, PathOutcome::Found);
    Answers {
        objects: heap.object_count(),
        references: heap.ref_count(),
        histogram,
        retained: heap
            .retained_top(Page::first(20))
            .map(|p| p.items)
            .unwrap_or_default(),
        sample_retained: [node_id(0, 0), node_id(0, 500), deep_tail]
            .iter()
            .map(|&id| heap.retained_size(id))
            .collect(),
        path: path.steps.iter().map(|s| s.object_id).collect(),
        entries,
    }
}

#[test]
fn the_filesystem_backend_matches_the_in_memory_backend() {
    let hprof = generate_hprof();

    // In memory.
    let mem = MemStore::new();
    build_indexes(&hprof, &mem, &IndexOptions::default(), &NoProgress).expect("memory build");
    let mem_heap =
        HeapQuery::from_store(ByteSource::from(hprof.clone()), &mem).expect("memory open");
    let mem_answers = answers(&mem_heap, mem.list("").expect("list"));

    // On disk.
    let dir = TempDir::new("equivalence");
    let path = dir.0.join("heap.hprof");
    std::fs::write(&path, &hprof).expect("write hprof");

    assert!(
        matches!(
            HeapQuery::open_existing(&path),
            Err(HprofError::NotIndexed(_))
        ),
        "nothing is built yet"
    );
    HeapQuery::open_with(&path, &IndexOptions::default(), &NoProgress).expect("disk build");
    let disk_heap = HeapQuery::open_existing(&path).expect("open_existing after build");
    let index_dir = dir.0.join("heap.indexes");
    let disk_entries = FsStore::open_or_create(&index_dir)
        .expect("fs store")
        .list("")
        .expect("list");
    let disk_answers = answers(&disk_heap, disk_entries);

    assert_eq!(disk_answers, mem_answers);

    // Sanity on the shared answers so a bug that breaks both backends still fails.
    assert_eq!(
        mem_answers.objects,
        (CHAINS * CHAIN_LEN + 1 + CHAINS) as usize
    );
    assert_eq!(
        mem_answers.histogram[0],
        ("Node".to_owned(), CHAINS * CHAIN_LEN, 80_000)
    );
    assert_eq!(mem_answers.path.len(), CHAIN_LEN as usize);
    assert_eq!(mem_answers.path[0], node_id(3, 0));
    assert_eq!(mem_answers.sample_retained[0], Some(CHAIN_LEN * 8));
    assert_eq!(mem_answers.sample_retained[1], Some((CHAIN_LEN - 500) * 8));
    assert_eq!(mem_answers.sample_retained[2], Some(8));
}

/// Names of the files directly inside `dir` (temp files included).
fn files_in(dir: &std::path::Path) -> Vec<String> {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir: {e}"));
    let mut names: Vec<String> = entries
        .map(|e| {
            let entry = e.unwrap_or_else(|e| panic!("dir entry: {e}"));
            entry.file_name().to_string_lossy().into_owned()
        })
        .collect();
    names.sort();
    names
}

#[test]
fn the_filesystem_store_stages_commits_and_cleans_up() {
    use std::io::Write;

    let dir = TempDir::new("store");
    let store = FsStore::open_or_create(dir.0.join("idx")).expect("store");
    let root = store.dir().to_path_buf();

    // A staged entry is invisible to `exists` and `list` but occupies a temp
    // file; dropping the writer removes it.
    let mut w = store.create("a.bin").expect("create");
    w.write_all(&[3, 1, 2]).expect("write");
    w.flush().expect("flush");
    assert!(!store.exists("a.bin"));
    assert!(
        store.list("").expect("list").is_empty(),
        "temp files are not entries"
    );
    assert_eq!(files_in(&root), vec!["a.bin.tmp"]);
    drop(w);
    assert!(files_in(&root).is_empty(), "dropped writer leaves nothing");

    // A writer dropped after its bytes were mapped for sorting is cleaned up too.
    let mut w = store.create("mapped.bin").expect("create");
    w.write_all(&[9, 8, 7]).expect("write");
    w.as_mut_bytes().expect("map").sort_unstable();
    drop(w);
    assert!(
        files_in(&root).is_empty(),
        "dropped mapped writer leaves nothing"
    );

    // Sort in place on the store, then commit; the result is visible.
    let mut w = store.create("a.bin").expect("create");
    w.write_all(&[3, 1, 2]).expect("write");
    w.as_mut_bytes().expect("map").sort_unstable();
    w.commit().expect("commit");
    assert_eq!(store.open("a.bin").expect("open").as_ref(), &[1, 2, 3]);
    assert_eq!(files_in(&root), vec!["a.bin"]);

    // Committing over an existing entry replaces it (Windows cannot rename
    // over a file, so the store removes it first).
    let mut w = store.create("a.bin").expect("create");
    w.write_all(&[7, 7]).expect("write");
    w.commit().expect("commit over existing");
    assert_eq!(store.open("a.bin").expect("open").as_ref(), &[7, 7]);
    // ...and until that commit the old bytes stayed readable.
    let mut w = store.create("a.bin").expect("create");
    w.write_all(&[1]).expect("write");
    assert_eq!(store.open("a.bin").expect("open").as_ref(), &[7, 7]);
    drop(w);
    assert_eq!(store.open("a.bin").expect("open").as_ref(), &[7, 7]);

    // An empty entry needs no mapping.
    let mut w = store.create("empty.bin").expect("create");
    assert!(w.as_mut_bytes().expect("map").is_empty());
    w.commit().expect("commit");
    assert!(store.open("empty.bin").expect("open").is_empty());

    // Nested names and prefix removal, including the directory itself.
    for name in ["heap_index/00/x", "heap_index/01/y", "keep/z.bin"] {
        let mut w = store.create(name).expect("create nested");
        w.write_all(&[1]).expect("write");
        w.commit().expect("commit");
    }
    assert_eq!(
        store.list("heap_index/").expect("list"),
        vec!["heap_index/00/x".to_owned(), "heap_index/01/y".to_owned()]
    );
    store.remove_prefix("heap_index/").expect("remove_prefix");
    assert!(!root.join("heap_index").exists(), "directory removed too");
    assert!(store.exists("keep/z.bin"));
    store.remove("keep/z.bin").expect("remove");
    store
        .remove("keep/z.bin")
        .expect("removing a missing entry is fine");

    // A stale manifest: indexes built for one dump must not be used for
    // another, and rebuilding fixes it.
    let path = dir.0.join("heap.hprof");
    std::fs::write(&path, generate_hprof_with(2)).expect("write hprof");
    let first = HeapQuery::open_with(&path, &IndexOptions::default(), &NoProgress).expect("build");
    assert_eq!(first.object_count(), (2 * CHAIN_LEN + 1 + 2) as usize);
    drop(first);
    std::fs::write(&path, generate_hprof_with(3)).expect("overwrite hprof");
    assert!(
        matches!(HeapQuery::open_existing(&path), Err(HprofError::Corrupt(_))),
        "old indexes are refused for a different dump"
    );
    let second =
        HeapQuery::open_with(&path, &IndexOptions::default(), &NoProgress).expect("rebuild");
    assert_eq!(second.object_count(), (3 * CHAIN_LEN + 1 + 3) as usize);
    HeapQuery::open_existing(&path).expect("open_existing after the rebuild");
}
