use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::{BufWriter, Cursor, Error, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use memmap2::{Mmap, MmapMut};

pub trait MMapWriter {
    fn create_writer(&mut self) -> Result<impl Write, Error>;
    fn create_mut_mmap(&mut self) -> Result<impl AsMut<[u8]>, Error>;
}

impl MMapWriter for Path {
    fn create_writer(&mut self) -> Result<impl Write, Error> {
        let out = File::create(self)?;
        Ok(BufWriter::new(out))
    }

    fn create_mut_mmap(&mut self) -> Result<impl AsMut<[u8]>, Error> {
        let file = OpenOptions::new().read(true).write(true).open(self)?;
        let map = map_mut(&file)?;
        Ok(map)
    }
}

impl MMapWriter for PathBuf {
    fn create_writer(&mut self) -> Result<impl Write, Error> {
        let out = File::create(self)?;
        Ok(BufWriter::new(out))
    }

    fn create_mut_mmap(&mut self) -> Result<impl AsMut<[u8]>, Error> {
        let file = OpenOptions::new().read(true).write(true).open(self)?;
        let map = map_mut(&file)?;
        Ok(map)
    }
}

impl MMapWriter for Vec<u8> {
    fn create_writer(&mut self) -> Result<impl Write, Error> {
        Ok(Cursor::new(self))
    }

    fn create_mut_mmap(&mut self) -> Result<impl AsMut<[u8]>, Error> {
        Ok(self)
    }
}

#[allow(unsafe_code)]
fn map_mut(file: &File) -> Result<MmapMut, Error> {
    // SAFETY: exclusive write access; file is not otherwise modified.
    let m = unsafe { MmapMut::map_mut(file) }?;
    Ok(m)
}

pub enum ByteSource {
    VecSource(Vec<u8>),
    MMapSource(Mmap),
}

impl AsRef<[u8]> for ByteSource {
    fn as_ref(&self) -> &[u8] {
        match self {
            ByteSource::VecSource(value) => value.as_ref(),
            ByteSource::MMapSource(value) => value.as_ref(),
        }
    }
}

impl From<Vec<u8>> for ByteSource {
    fn from(v: Vec<u8>) -> ByteSource {
        ByteSource::VecSource(v)
    }
}

// ── MMapReader ────────────────────────────────────────────────────────────────

/// Trait for types that can produce a read-only [`ByteSource`].
///
/// Implemented for [`Path`], [`PathBuf`] (memory-map the file), and [`Vec<u8>`]
/// (wrap the bytes directly).
pub trait MMapReader {
    fn open_mmap(&self) -> Result<ByteSource, std::io::Error>;
}

impl MMapReader for Path {
    fn open_mmap(&self) -> Result<ByteSource, std::io::Error> {
        let file = File::open(self)?;
        #[allow(unsafe_code)]
        // SAFETY: file opened read-only and not modified while mapped.
        let mmap = unsafe { Mmap::map(&file) }?;
        Ok(ByteSource::MMapSource(mmap))
    }
}

impl MMapReader for PathBuf {
    fn open_mmap(&self) -> Result<ByteSource, std::io::Error> {
        self.as_path().open_mmap()
    }
}

impl MMapReader for Vec<u8> {
    fn open_mmap(&self) -> Result<ByteSource, std::io::Error> {
        Ok(ByteSource::VecSource(self.clone()))
    }
}

// ── SubIndexDir ───────────────────────────────────────────────────────────────

/// An abstraction over a directory of heap sub-index files.
///
/// Backed by either a real filesystem directory or an in-memory HashMap,
/// enabling both production use and in-memory testing without temp files.
#[derive(Clone)]
pub struct SubIndexDir(SubIndexDirInner);

#[derive(Clone)]
enum SubIndexDirInner {
    Fs(PathBuf),
    Mem(Arc<Mutex<HashMap<String, Vec<u8>>>>),
}

impl SubIndexDir {
    /// Create a filesystem-backed directory.
    pub fn fs(path: PathBuf) -> Self {
        SubIndexDir(SubIndexDirInner::Fs(path))
    }

    /// Create an in-memory directory (for tests).
    pub fn mem() -> Self {
        SubIndexDir(SubIndexDirInner::Mem(Arc::new(Mutex::new(HashMap::new()))))
    }

    /// Write a named sub-index file. The `rel_path` may include a subdirectory
    /// component (e.g. `"1f/HPROF_HEAP_DUMP_SEGMENT_1f"`).
    pub(crate) fn write_sub_file(
        &self,
        rel_path: &str,
        data: Vec<u8>,
    ) -> Result<(), std::io::Error> {
        match &self.0 {
            SubIndexDirInner::Fs(base) => {
                let full = base.join(rel_path);
                if let Some(parent) = full.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(full, data)
            }
            SubIndexDirInner::Mem(arc) => {
                arc.lock()
                    .map_err(|_| std::io::Error::other("mutex poisoned"))?
                    .insert(rel_path.to_string(), data);
                Ok(())
            }
        }
    }

    /// Return the raw bytes of every sub-index file in this directory.
    pub(crate) fn all_file_bytes(&self) -> Result<Vec<Vec<u8>>, std::io::Error> {
        match &self.0 {
            SubIndexDirInner::Fs(base) => {
                // Walk two levels: base/<hex_prefix>/<filename>
                let mut result = Vec::new();
                for subdir_entry in std::fs::read_dir(base)? {
                    let subdir_entry = subdir_entry?;
                    if !subdir_entry.file_type()?.is_dir() {
                        continue;
                    }
                    for file_entry in std::fs::read_dir(subdir_entry.path())? {
                        let file_entry = file_entry?;
                        if file_entry
                            .file_name()
                            .to_string_lossy()
                            .starts_with("HPROF_HEAP_DUMP")
                        {
                            result.push(std::fs::read(file_entry.path())?);
                        }
                    }
                }
                Ok(result)
            }
            SubIndexDirInner::Mem(arc) => Ok(arc
                .lock()
                .map_err(|_| std::io::Error::other("mutex poisoned"))?
                .values()
                .cloned()
                .collect()),
        }
    }

    /// Read a specific named file, or `None` if it does not exist.
    pub fn get_file(&self, rel_path: &str) -> Option<Vec<u8>> {
        match &self.0 {
            SubIndexDirInner::Fs(base) => std::fs::read(base.join(rel_path)).ok(),
            SubIndexDirInner::Mem(arc) => arc.lock().ok()?.get(rel_path).cloned(),
        }
    }

    /// Returns the filesystem path, or `None` for an in-memory directory.
    pub fn path(&self) -> Option<&Path> {
        match &self.0 {
            SubIndexDirInner::Fs(p) => Some(p.as_path()),
            SubIndexDirInner::Mem(_) => None,
        }
    }
}
