//! `index.json`: what an index store contains and which hprof it was built for.
//!
//! Build completion is recognised **only** through the manifest.  A store
//! entry that exists but is not listed in `built` is treated as absent (it
//! may be a leftover from an interrupted or older run) and is rebuilt.
//!
//! The manifest also pins the on-disk layout: [`FORMAT_VERSION`] is bumped
//! whenever any entry's record layout or name changes, and a store built
//! with a different version is rebuilt from scratch (indexes are derived
//! data; there is no migration).

use crate::hprof::{HprofError, HprofHeader};
use crate::index::IndexStore;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;

/// Layout version of every entry in the store.  Bump on any layout change.
///
/// History:
/// * 1 — first manifested layout: object store, roots, names, refs,
///   dominators + retained (+ `retained_by_size`, `dominator_children`),
///   aux, arrays, `instances_by_class`, `class_histogram`.
/// * 2 — the manifest also records the hprof header's timestamp, so two dumps
///   of the same length no longer share indexes.  File layouts are unchanged.
pub const FORMAT_VERSION: u32 = 2;

/// Store entry name of the manifest.
pub const MANIFEST_NAME: &str = "index.json";

/// What identifies a heap dump: its length, identifier size and the creation
/// timestamp in its header.  All three come from the bytes, so an in-memory
/// dump has an identity too.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HprofIdentity {
    /// Byte length of the hprof.
    pub len: u64,
    /// Identifier size (4 or 8).
    pub id_size: u32,
    /// Dump creation time from the header, milliseconds since the epoch.
    pub timestamp_ms: u64,
}

impl HprofIdentity {
    /// Read the identity of `hprof` (parses only the header).
    pub fn of(hprof: &[u8]) -> Result<Self, HprofError> {
        let header = HprofHeader::parse(hprof)?;
        Ok(Self {
            len: hprof.len() as u64,
            id_size: header.id_size,
            timestamp_ms: header.timestamp_ms,
        })
    }
}

/// The manifest document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// [`FORMAT_VERSION`] the entries were written with.
    pub format_version: u32,
    /// Byte length of the hprof the indexes were built from.
    pub hprof_len: u64,
    /// Identifier size of that hprof (4 or 8).
    pub id_size: u32,
    /// Creation timestamp (ms since the epoch) from that hprof's header.
    pub hprof_timestamp_ms: u64,
    /// Entry name → Unix timestamp (seconds) of the build that committed it.
    pub built: BTreeMap<String, u64>,
}

impl Manifest {
    /// A fresh manifest for the dump `id` with nothing built.
    pub fn new(id: HprofIdentity) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            hprof_len: id.len,
            id_size: id.id_size,
            hprof_timestamp_ms: id.timestamp_ms,
            built: BTreeMap::new(),
        }
    }

    fn identity(&self) -> HprofIdentity {
        HprofIdentity {
            len: self.hprof_len,
            id_size: self.id_size,
            timestamp_ms: self.hprof_timestamp_ms,
        }
    }

    /// Read the manifest from `store`.
    ///
    /// Returns `Ok(None)` when there is no manifest **or** it cannot be
    /// parsed: either way nothing in the store can be trusted, and the
    /// caller rebuilds.
    pub fn read(store: &dyn IndexStore) -> Result<Option<Self>, HprofError> {
        if !store.exists(MANIFEST_NAME) {
            return Ok(None);
        }
        let bytes = store.open(MANIFEST_NAME)?;
        Ok(serde_json::from_slice(bytes.as_ref()).ok())
    }

    /// Write (replace) the manifest in `store`.
    pub fn write(&self, store: &dyn IndexStore) -> Result<(), HprofError> {
        let json = serde_json::to_vec_pretty(self)
            .map_err(|e| HprofError::Internal(format!("manifest serialisation: {e}")))?;
        let mut w = store.create(MANIFEST_NAME)?;
        w.write_all(&json)?;
        w.commit()
    }

    /// `true` when this manifest describes indexes usable for the dump `id`
    /// with the current layout.
    pub fn matches(&self, id: &HprofIdentity) -> bool {
        self.format_version == FORMAT_VERSION && self.identity() == *id
    }

    /// Why [`Self::matches`] would fail, as an actionable message.
    pub fn mismatch_reason(&self, id: &HprofIdentity) -> Option<String> {
        if self.format_version != FORMAT_VERSION {
            Some(format!(
                "index format version {} but this build expects {}",
                self.format_version, FORMAT_VERSION
            ))
        } else if self.identity() != *id {
            Some(format!(
                "indexes were built for a different hprof ({} bytes, id size {}, dumped at {} ms); \
                 this one is {} bytes, id size {}, dumped at {} ms",
                self.hprof_len,
                self.id_size,
                self.hprof_timestamp_ms,
                id.len,
                id.id_size,
                id.timestamp_ms
            ))
        } else {
            None
        }
    }

    /// `true` when `name` was committed by a completed build step.
    pub fn is_built(&self, name: &str) -> bool {
        self.built.contains_key(name)
    }

    /// Record that every entry in `names` was just committed.
    pub fn mark_built(&mut self, names: &[&str]) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        for n in names {
            self.built.insert((*n).to_owned(), now);
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::MemStore;

    #[test]
    fn round_trips_through_the_store() {
        let store = MemStore::new();
        assert!(Manifest::read(&store).unwrap().is_none());
        let id = HprofIdentity {
            len: 1234,
            id_size: 8,
            timestamp_ms: 99,
        };
        let mut m = Manifest::new(id);
        m.mark_built(&["a.bin", "b.bin"]);
        m.write(&store).unwrap();
        let back = Manifest::read(&store).unwrap().unwrap();
        assert_eq!(back, m);
        assert!(back.is_built("a.bin"));
        assert!(!back.is_built("c.bin"));
        assert!(back.matches(&id));
        assert!(back.mismatch_reason(&id).is_none());
        let with = |f: fn(&mut HprofIdentity)| {
            let mut i = id;
            f(&mut i);
            i
        };
        for changed in [
            with(|i| i.id_size = 4),
            with(|i| i.len += 1),
            // Same length and id size, dumped at another time.
            with(|i| i.timestamp_ms += 1),
        ] {
            assert!(!back.matches(&changed));
            assert!(
                back.mismatch_reason(&changed)
                    .unwrap()
                    .contains("different hprof")
            );
        }
    }

    #[test]
    fn unparsable_manifest_reads_as_none() {
        let store = MemStore::new();
        let mut w = store.create(MANIFEST_NAME).unwrap();
        w.write_all(b"{ not json").unwrap();
        w.commit().unwrap();
        assert!(Manifest::read(&store).unwrap().is_none());
    }

    #[test]
    fn version_mismatch_is_reported() {
        let id = HprofIdentity {
            len: 1,
            id_size: 8,
            timestamp_ms: 0,
        };
        let mut m = Manifest::new(id);
        m.format_version = FORMAT_VERSION + 1;
        assert!(!m.matches(&id));
        assert!(m.mismatch_reason(&id).unwrap().contains("format version"));
    }
}
