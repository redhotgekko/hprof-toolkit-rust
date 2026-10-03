//! [`IndexStore`]: the one place the crate touches storage.
//!
//! Two backends exist and must stay behaviourally identical:
//!
//! * [`FsStore`] — a directory on disk.  Entries are files; reads are
//!   memory-mapped; writes go to `<name>.tmp` and are renamed into place on
//!   [`StoreWriter::commit`], so a crash never leaves a half-written entry
//!   under its final name.
//! * [`MemStore`] — a `HashMap<String, Arc<Vec<u8>>>`.  Used by every unit
//!   test in the crate so that building indexes, opening a
//!   [`crate::query::HeapQuery`] and serving requests never touch the
//!   filesystem.
//!
//! Writers stage data and expose it mutably ([`StoreWriter::as_mut_bytes`])
//! so that a builder can write records sequentially and then sort them in
//! place before committing — an `MmapMut` on disk, a `&mut Vec<u8>` in memory.
//!
//! **Memory rule reminder:** `MemStore` exists for tests.  Production code
//! paths must still stream through writers; the in-memory backend does not
//! license holding whole indexes in RAM.

use crate::hprof::HprofError;
use memmap2::{Mmap, MmapMut};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

// ── ByteSource ────────────────────────────────────────────────────────────────

/// Read-only bytes from any backend.
///
/// Everything that reads an index or the hprof file itself takes one of
/// these (or a `&[u8]` borrowed from one).  Readers never know whether they
/// are looking at a memory map or a vector.
pub enum ByteSource {
    /// Owned bytes (tests, small buffers).
    Vec(Vec<u8>),
    /// A read-only memory map of a file.
    Mmap(Mmap),
    /// Bytes shared with an in-memory store (cheap to clone).
    Shared(Arc<Vec<u8>>),
}

impl ByteSource {
    /// Map a file read-only.
    pub fn map_file(path: &Path) -> Result<Self, HprofError> {
        let file = File::open(path)?;
        #[allow(unsafe_code)]
        // SAFETY: file opened read-only and not modified while mapped.
        let mmap = unsafe { Mmap::map(&file) }?;
        Ok(ByteSource::Mmap(mmap))
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.as_ref().len()
    }

    /// `true` when empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl AsRef<[u8]> for ByteSource {
    fn as_ref(&self) -> &[u8] {
        match self {
            ByteSource::Vec(v) => v.as_slice(),
            ByteSource::Mmap(m) => m.as_ref(),
            ByteSource::Shared(v) => v.as_slice(),
        }
    }
}

impl From<Vec<u8>> for ByteSource {
    fn from(v: Vec<u8>) -> Self {
        ByteSource::Vec(v)
    }
}

impl std::fmt::Debug for ByteSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self {
            ByteSource::Vec(_) => "Vec",
            ByteSource::Mmap(_) => "Mmap",
            ByteSource::Shared(_) => "Shared",
        };
        write!(f, "ByteSource::{kind}({} bytes)", self.len())
    }
}

// ── Traits ────────────────────────────────────────────────────────────────────

/// Where index entries live.  See the module docs.
///
/// Entry names are `/`-separated relative paths (e.g. `"refs.bin"`,
/// `"heap_index/1f/HPROF_HEAP_DUMP_SEGMENT_1f"`).
pub trait IndexStore: Send + Sync {
    /// Open a committed entry for reading.
    fn open(&self, name: &str) -> Result<ByteSource, HprofError>;

    /// Start writing `name`.  Nothing is visible under `name` until the
    /// returned writer is committed; dropping it discards the staged data.
    fn create(&self, name: &str) -> Result<Box<dyn StoreWriter>, HprofError>;

    /// `true` when a committed entry named `name` exists.
    fn exists(&self, name: &str) -> bool;

    /// Remove a committed entry.  Removing a missing entry is not an error.
    fn remove(&self, name: &str) -> Result<(), HprofError>;

    /// Names of all committed entries starting with `prefix`, sorted.
    fn list(&self, prefix: &str) -> Result<Vec<String>, HprofError>;

    /// Remove every committed entry starting with `prefix`.
    fn remove_prefix(&self, prefix: &str) -> Result<(), HprofError> {
        for name in self.list(prefix)? {
            self.remove(&name)?;
        }
        Ok(())
    }
}

/// A staged, uncommitted entry being written.
pub trait StoreWriter: Write + Send {
    /// Flush and expose everything written so far as a mutable slice, so the
    /// caller can sort it in place.  Further `write` calls after this are an
    /// error.
    fn as_mut_bytes(&mut self) -> Result<&mut [u8], HprofError>;

    /// Publish the staged bytes under the entry's final name.
    fn commit(self: Box<Self>) -> Result<(), HprofError>;
}

// ── FsStore ───────────────────────────────────────────────────────────────────

/// Filesystem backend: one directory, one file per entry.
#[derive(Debug, Clone)]
pub struct FsStore {
    dir: PathBuf,
}

impl FsStore {
    /// Create the directory if needed and return a store over it.
    pub fn open_or_create(dir: impl Into<PathBuf>) -> Result<Self, HprofError> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    /// The directory backing this store.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn path_of(&self, name: &str) -> PathBuf {
        let mut p = self.dir.clone();
        for seg in name.split('/') {
            p.push(seg);
        }
        p
    }

    fn walk(&self, dir: &Path, rel: &str, out: &mut Vec<String>) -> io::Result<()> {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e),
        };
        for entry in entries {
            let entry = entry?;
            let file_name = entry.file_name().to_string_lossy().into_owned();
            let child_rel = if rel.is_empty() {
                file_name.clone()
            } else {
                format!("{rel}/{file_name}")
            };
            let ft = entry.file_type()?;
            if ft.is_dir() {
                self.walk(&entry.path(), &child_rel, out)?;
            } else if !file_name.ends_with(TMP_SUFFIX) {
                out.push(child_rel);
            }
        }
        Ok(())
    }
}

const TMP_SUFFIX: &str = ".tmp";

impl IndexStore for FsStore {
    fn open(&self, name: &str) -> Result<ByteSource, HprofError> {
        ByteSource::map_file(&self.path_of(name))
    }

    fn create(&self, name: &str) -> Result<Box<dyn StoreWriter>, HprofError> {
        let final_path = self.path_of(name);
        if let Some(parent) = final_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file_name = final_path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let tmp_path = final_path.with_file_name(format!("{file_name}{TMP_SUFFIX}"));
        let file = File::create(&tmp_path)?;
        Ok(Box::new(FsWriter {
            state: FsWriterState::Writing(BufWriter::new(file)),
            tmp_path,
            final_path,
            finished: false,
        }))
    }

    fn exists(&self, name: &str) -> bool {
        self.path_of(name).is_file()
    }

    fn remove(&self, name: &str) -> Result<(), HprofError> {
        match std::fs::remove_file(self.path_of(name)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>, HprofError> {
        let mut out = Vec::new();
        self.walk(&self.dir, "", &mut out)?;
        out.retain(|n| n.starts_with(prefix));
        out.sort();
        Ok(out)
    }

    fn remove_prefix(&self, prefix: &str) -> Result<(), HprofError> {
        for name in self.list(prefix)? {
            self.remove(&name)?;
        }
        // If the prefix names a directory, remove the (now empty) tree too.
        if let Some(dir_name) = prefix.strip_suffix('/') {
            let dir = self.path_of(dir_name);
            if dir.is_dir() {
                std::fs::remove_dir_all(dir)?;
            }
        }
        Ok(())
    }
}

enum FsWriterState {
    Writing(BufWriter<File>),
    Mapped(MmapMut),
    /// Zero-length entry: nothing to map.
    Empty,
    Done,
}

struct FsWriter {
    state: FsWriterState,
    tmp_path: PathBuf,
    final_path: PathBuf,
    /// `true` once the temp file was renamed into place.  Kept apart from
    /// `state` so that a writer whose state was reset part-way through
    /// `as_mut_bytes` (a failed remap) still removes its temp file on drop.
    finished: bool,
}

impl Write for FsWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match &mut self.state {
            FsWriterState::Writing(w) => w.write(buf),
            _ => Err(io::Error::other("write after as_mut_bytes")),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match &mut self.state {
            FsWriterState::Writing(w) => w.flush(),
            _ => Ok(()),
        }
    }
}

impl StoreWriter for FsWriter {
    fn as_mut_bytes(&mut self) -> Result<&mut [u8], HprofError> {
        if let FsWriterState::Writing(w) = &mut self.state {
            w.flush()?;
            // Drop the BufWriter (closes the file) before remapping.
            self.state = FsWriterState::Done;
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&self.tmp_path)?;
            let len = file.metadata()?.len();
            if len == 0 {
                self.state = FsWriterState::Empty;
            } else {
                #[allow(unsafe_code)]
                // SAFETY: exclusive access; the file is not touched by anyone
                // else until commit.
                let map = unsafe { MmapMut::map_mut(&file) }?;
                self.state = FsWriterState::Mapped(map);
            }
        }
        match &mut self.state {
            FsWriterState::Mapped(m) => Ok(&mut m[..]),
            FsWriterState::Empty => Ok(&mut []),
            _ => Err(HprofError::Io(io::Error::other(
                "writer is no longer usable",
            ))),
        }
    }

    fn commit(mut self: Box<Self>) -> Result<(), HprofError> {
        match std::mem::replace(&mut self.state, FsWriterState::Done) {
            FsWriterState::Writing(mut w) => w.flush()?,
            FsWriterState::Mapped(m) => m.flush()?,
            FsWriterState::Empty => {}
            FsWriterState::Done => {
                return Err(HprofError::Io(io::Error::other("writer already committed")));
            }
        }
        // Windows refuses to rename over an existing file.
        if self.final_path.exists() {
            std::fs::remove_file(&self.final_path)?;
        }
        std::fs::rename(&self.tmp_path, &self.final_path)?;
        self.finished = true;
        Ok(())
    }
}

impl Drop for FsWriter {
    fn drop(&mut self) {
        // Discard staged data if never committed (including a commit that
        // failed half way).  Errors are ignored: nothing sensible can be done
        // in a destructor.
        if !self.finished {
            // Unmap and close first: Windows cannot delete a mapped file.
            self.state = FsWriterState::Done;
            let _ = std::fs::remove_file(&self.tmp_path);
        }
    }
}

// ── MemStore ──────────────────────────────────────────────────────────────────

type Entries = HashMap<String, Arc<Vec<u8>>>;
type MemMap = Arc<Mutex<Entries>>;

/// In-memory backend for tests.
#[derive(Debug, Clone, Default)]
pub struct MemStore {
    entries: MemMap,
}

impl MemStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Entries>, HprofError> {
        self.entries
            .lock()
            .map_err(|_| HprofError::Internal("MemStore mutex poisoned".to_owned()))
    }

    /// Total bytes held by all committed entries (test diagnostics).
    pub fn total_bytes(&self) -> usize {
        self.entries
            .lock()
            .map(|m| m.values().map(|v| v.len()).sum())
            .unwrap_or(0)
    }

    /// Copy of every committed entry, for comparing two stores in tests.
    #[cfg(test)]
    pub fn snapshot(&self) -> std::collections::BTreeMap<String, Vec<u8>> {
        self.entries
            .lock()
            .map(|m| {
                m.iter()
                    .map(|(k, v)| (k.clone(), v.as_ref().clone()))
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl IndexStore for MemStore {
    fn open(&self, name: &str) -> Result<ByteSource, HprofError> {
        self.lock()?
            .get(name)
            .map(|v| ByteSource::Shared(Arc::clone(v)))
            .ok_or_else(|| {
                HprofError::Io(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("no in-memory index entry named {name:?}"),
                ))
            })
    }

    fn create(&self, name: &str) -> Result<Box<dyn StoreWriter>, HprofError> {
        Ok(Box::new(MemWriter {
            buf: Vec::new(),
            name: name.to_owned(),
            entries: Arc::clone(&self.entries),
        }))
    }

    fn exists(&self, name: &str) -> bool {
        self.entries
            .lock()
            .map(|m| m.contains_key(name))
            .unwrap_or(false)
    }

    fn remove(&self, name: &str) -> Result<(), HprofError> {
        self.lock()?.remove(name);
        Ok(())
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>, HprofError> {
        let mut names: Vec<String> = self
            .lock()?
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        names.sort();
        Ok(names)
    }
}

struct MemWriter {
    buf: Vec<u8>,
    name: String,
    entries: MemMap,
}

impl Write for MemWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buf.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl StoreWriter for MemWriter {
    fn as_mut_bytes(&mut self) -> Result<&mut [u8], HprofError> {
        Ok(&mut self.buf[..])
    }

    fn commit(self: Box<Self>) -> Result<(), HprofError> {
        let MemWriter { buf, name, entries } = *self;
        entries
            .lock()
            .map_err(|_| HprofError::Internal("MemStore mutex poisoned".to_owned()))?
            .insert(name, Arc::new(buf));
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mem_store_round_trip_and_visibility() {
        let store = MemStore::new();
        assert!(!store.exists("a.bin"));
        let mut w = store.create("a.bin").unwrap();
        w.write_all(&[3, 1, 2]).unwrap();
        // Staged data is invisible until commit.
        assert!(!store.exists("a.bin"));
        assert!(store.open("a.bin").is_err());
        w.as_mut_bytes().unwrap().sort_unstable();
        w.commit().unwrap();
        assert!(store.exists("a.bin"));
        assert_eq!(store.open("a.bin").unwrap().as_ref(), &[1, 2, 3]);
    }

    #[test]
    fn mem_store_dropped_writer_leaves_nothing() {
        let store = MemStore::new();
        {
            let mut w = store.create("gone.bin").unwrap();
            w.write_all(b"xyz").unwrap();
            // dropped without commit
        }
        assert!(!store.exists("gone.bin"));
        assert!(store.list("").unwrap().is_empty());
    }

    #[test]
    fn mem_store_list_and_remove_prefix() {
        let store = MemStore::new();
        for name in ["heap_index/00/a", "heap_index/01/b", "other.bin"] {
            store.create(name).unwrap().commit().unwrap();
        }
        assert_eq!(
            store.list("heap_index/").unwrap(),
            vec!["heap_index/00/a".to_owned(), "heap_index/01/b".to_owned()]
        );
        store.remove_prefix("heap_index/").unwrap();
        assert_eq!(store.list("").unwrap(), vec!["other.bin".to_owned()]);
        // Removing a missing entry is fine.
        store.remove("missing").unwrap();
    }

    #[test]
    fn mem_store_open_is_shared_not_copied() {
        let store = MemStore::new();
        let mut w = store.create("s.bin").unwrap();
        w.write_all(&[9; 1024]).unwrap();
        w.commit().unwrap();
        let a = store.open("s.bin").unwrap();
        let b = store.open("s.bin").unwrap();
        match (a, b) {
            (ByteSource::Shared(x), ByteSource::Shared(y)) => assert!(Arc::ptr_eq(&x, &y)),
            _ => panic!("expected shared sources"),
        }
    }

    #[test]
    fn mem_store_empty_entry_has_empty_mut_bytes() {
        let store = MemStore::new();
        let mut w = store.create("e.bin").unwrap();
        assert!(w.as_mut_bytes().unwrap().is_empty());
        w.commit().unwrap();
        assert!(store.open("e.bin").unwrap().is_empty());
    }
}
