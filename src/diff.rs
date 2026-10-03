//! Comparing two snapshots of the same JVM.
//!
//! [`HeapDiff`] pairs two [`HeapQuery`]s with three precomputed entries
//! (objects only in the first dump, only in the second, and in both with a
//! changed flag), built once into an
//! [`IndexStore`].  The entries can live on disk ([`HeapDiff::open`]) or in
//! memory ([`HeapDiff::from_store`] with a `MemStore`, as the tests do).
//!
//! Object ids are heap addresses, so an id that appears in both dumps is
//! *probably* the same object, but the JVM may have reused the address for a
//! different one.  Compare [`HeapQuery::object_type_name`] on both sides when
//! it matters.
//!
//! ```no_run
//! use hprof_toolkit::diff::HeapDiff;
//! use hprof_toolkit::prelude::*;
//! use std::sync::Arc;
//!
//! let before = Arc::new(HeapQuery::open("before.hprof")?);
//! let after = Arc::new(HeapQuery::open("after.hprof")?);
//! let diff = HeapDiff::open(before, after, "before.hprof".as_ref(), "after.hprof".as_ref())?;
//! for c in diff.summary()?.by_class.iter().take(10) {
//!     println!("{:+}  {}", c.net_change(), c.class_name);
//! }
//! # Ok::<(), HprofError>(())
//! ```

use crate::class_key::ClassKey;
use crate::diff_index::{
    ADDED, COMMON, CommonEntry, DiffEntry, DiffIndexCounts, REMOVED, build_diff_indexes,
    diff_dir_for_hprofs,
};
use crate::heap_index::sub_record::SubIndexEntry;
use crate::heap_parser::SubRecord;
use crate::hprof::HprofError;
use crate::index::{ByteSource, FsStore, IndexStore, RecordFile};
use crate::query::{HeapQuery, Page, PageResult};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, OnceLock};

// ── Public types ──────────────────────────────────────────────────────────────

/// Per-class instance counts from a two-snapshot diff.
#[derive(Debug, Clone)]
pub struct ClassDiffEntry {
    /// The histogram bucket: a class, an object-array kind or a primitive-array kind.
    pub key: ClassKey,
    /// Display name (e.g. `"java.lang.String"`, `"int[]"`).
    pub class_name: String,
    /// Instances present only in dump 1 (garbage-collected by dump 2).
    pub count_removed: u64,
    /// Instances present only in dump 2 (allocated between the two dumps).
    pub count_added: u64,
    /// Instances with the same object ID in both dumps (survived), unchanged.
    pub count_common_unchanged: u64,
    /// Instances with the same object ID in both dumps (survived), with changed data.
    pub count_common_changed: u64,
}

impl ClassDiffEntry {
    /// Total instances in both "common" categories.
    pub fn count_common(&self) -> u64 {
        self.count_common_unchanged + self.count_common_changed
    }

    /// Total instances in dump 1 (`count_removed + count_common()`).
    pub fn count_before(&self) -> u64 {
        self.count_removed + self.count_common()
    }

    /// Total instances in dump 2 (`count_added + count_common()`).
    pub fn count_after(&self) -> u64 {
        self.count_added + self.count_common()
    }

    /// Net change: positive = growth, negative = shrinkage.
    pub fn net_change(&self) -> i64 {
        self.count_added as i64 - self.count_removed as i64
    }
}

/// Summary of the differences between two heap snapshots.
#[derive(Debug)]
pub struct DiffSummary {
    /// Total meaningful objects in dump 1.
    pub total_before: u64,
    /// Total meaningful objects in dump 2.
    pub total_after: u64,
    /// Objects present only in dump 2 (added).
    pub total_added: u64,
    /// Objects present only in dump 1 (removed).
    pub total_removed: u64,
    /// Objects present in both dumps (common), with identical raw bytes.
    pub total_common_unchanged: u64,
    /// Objects present in both dumps (common), with differing raw bytes.
    pub total_common_changed: u64,
    /// Per-class breakdown sorted by `(count_added + count_removed)` descending.
    pub by_class: Vec<ClassDiffEntry>,
}

impl DiffSummary {
    /// Total objects in common (changed + unchanged).
    pub fn total_common(&self) -> u64 {
        self.total_common_unchanged + self.total_common_changed
    }
}

/// One object in a diff list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffObject {
    /// The object id (a heap address in the dump it appears in).
    pub object_id: u64,
    /// The histogram bucket of the object, when its record could be parsed.
    pub key: Option<ClassKey>,
    /// Display name of `key`, empty when unknown.
    pub class_name: String,
}

// ── HeapDiff ──────────────────────────────────────────────────────────────────

/// Two heap dumps and the precomputed differences between them.
pub struct HeapDiff {
    before: Arc<HeapQuery>,
    after: Arc<HeapQuery>,
    removed: ByteSource,
    added: ByteSource,
    common: ByteSource,
    summary: OnceLock<DiffSummary>,
}

impl HeapDiff {
    /// Open the diff of `before` against `after` in `store`, building the
    /// three diff entries first if they are missing.
    ///
    /// An existing entry set is trusted as is: it is not checked against the
    /// dumps, so delete the store when either dump changes.
    pub fn from_store(
        before: Arc<HeapQuery>,
        after: Arc<HeapQuery>,
        store: &dyn IndexStore,
    ) -> Result<Self, HprofError> {
        build_diff_indexes(&before, &after, store)?;
        Ok(Self {
            removed: store.open(REMOVED)?,
            added: store.open(ADDED)?,
            common: store.open(COMMON)?,
            before,
            after,
            summary: OnceLock::new(),
        })
    }

    /// [`Self::from_store`] on the directory `{before}_vs_{after}.diff_indexes`
    /// next to the first dump.
    pub fn open(
        before: Arc<HeapQuery>,
        after: Arc<HeapQuery>,
        before_path: &Path,
        after_path: &Path,
    ) -> Result<Self, HprofError> {
        let store = FsStore::open_or_create(diff_dir_for_hprofs(before_path, after_path))?;
        Self::from_store(before, after, &store)
    }

    /// The first (older) dump.
    pub fn before(&self) -> &HeapQuery {
        &self.before
    }

    /// The first dump as a shared handle.
    pub fn before_arc(&self) -> Arc<HeapQuery> {
        Arc::clone(&self.before)
    }

    /// The second (newer) dump.
    pub fn after(&self) -> &HeapQuery {
        &self.after
    }

    fn removed_file(&self) -> RecordFile<'_, DiffEntry> {
        RecordFile::from_slice(self.removed.as_ref())
    }

    fn added_file(&self) -> RecordFile<'_, DiffEntry> {
        RecordFile::from_slice(self.added.as_ref())
    }

    fn common_file(&self) -> RecordFile<'_, CommonEntry> {
        RecordFile::from_slice(self.common.as_ref())
    }

    /// Counts of the three entry sets.
    pub fn counts(&self) -> DiffIndexCounts {
        DiffIndexCounts {
            removed: self.removed_file().len() as u64,
            added: self.added_file().len() as u64,
            common: self.common_file().len() as u64,
            common_changed: self.common_file().iter().filter(|e| e.changed).count() as u64,
        }
    }

    /// Per-class counts of removed, added and common objects.
    ///
    /// Computed once (one pass over the three entries, memory proportional
    /// to the number of classes) and cached.
    pub fn summary(&self) -> Result<&DiffSummary, HprofError> {
        if let Some(s) = self.summary.get() {
            return Ok(s);
        }
        let s = self.compute_summary()?;
        Ok(self.summary.get_or_init(|| s))
    }

    /// Objects only in the first dump (garbage-collected since), by object
    /// id; optionally only those in `class`.
    pub fn removed(
        &self,
        class: Option<ClassKey>,
        page: Page,
    ) -> Result<PageResult<DiffObject>, HprofError> {
        let entries = self.removed_file().iter();
        self.list(&self.before, entries, class, page)
    }

    /// Objects only in the second dump (allocated since), by object id;
    /// optionally only those in `class`.
    pub fn added(
        &self,
        class: Option<ClassKey>,
        page: Page,
    ) -> Result<PageResult<DiffObject>, HprofError> {
        let entries = self.added_file().iter();
        self.list(&self.after, entries, class, page)
    }

    /// Objects in both dumps whose data changed, by object id.
    pub fn changed(
        &self,
        class: Option<ClassKey>,
        page: Page,
    ) -> Result<PageResult<DiffObject>, HprofError> {
        self.common(Some(true), class, page)
    }

    /// Objects in both dumps: `changed` selects only changed (`Some(true)`),
    /// only unchanged (`Some(false)`) or all (`None`) objects.
    pub fn common(
        &self,
        changed: Option<bool>,
        class: Option<ClassKey>,
        page: Page,
    ) -> Result<PageResult<DiffObject>, HprofError> {
        let entries = self
            .common_file()
            .iter()
            .filter(move |e| changed.is_none_or(|c| e.changed == c))
            .map(|e| DiffEntry {
                tag: e.tag,
                object_id: e.object_id,
                position: e.position1,
            });
        self.list(&self.before, entries, class, page)
    }

    /// Whether `object_id` is in both dumps with different data: `Some(true)`
    /// changed, `Some(false)` identical, `None` not present in both.
    pub fn object_changed(&self, object_id: u64) -> Option<bool> {
        self.common_file().find(object_id).map(|e| e.changed)
    }

    /// Stream `entries`, keep those in `class`, and return one page.
    ///
    /// Memory is proportional to the page: entries are counted, not
    /// collected, and class names are resolved for the page only.  Without a
    /// class filter the total is the entry count and only the page's records
    /// are parsed.
    fn list(
        &self,
        query: &HeapQuery,
        entries: impl Iterator<Item = DiffEntry>,
        class: Option<ClassKey>,
        page: Page,
    ) -> Result<PageResult<DiffObject>, HprofError> {
        let mut total = 0usize;
        let mut items = Vec::new();
        for entry in entries {
            let key = if class.is_some() || total >= page.offset && items.len() < page.limit {
                entry_key(query, &entry)?
            } else {
                None
            };
            if class.is_some() && key != class {
                continue;
            }
            if total >= page.offset && items.len() < page.limit {
                items.push(DiffObject {
                    object_id: entry.object_id,
                    class_name: key.map(|k| self.key_name(k)).unwrap_or_default(),
                    key,
                });
            }
            total += 1;
        }
        let has_more = page.offset + items.len() < total;
        Ok(PageResult {
            items,
            total,
            has_more,
        })
    }

    /// Display name of `key`, resolved in whichever dump knows the class.
    pub fn key_name(&self, key: ClassKey) -> String {
        let known_before = match key {
            ClassKey::Class(id) | ClassKey::ObjArray(id) => self.before.class_name(id).is_some(),
            ClassKey::PrimArray(_) => true,
        };
        if known_before {
            self.before.key_name(key)
        } else {
            self.after.key_name(key)
        }
    }

    fn compute_summary(&self) -> Result<DiffSummary, HprofError> {
        #[derive(Default)]
        struct Raw {
            removed: u64,
            added: u64,
            common_unchanged: u64,
            common_changed: u64,
        }
        let mut counts: HashMap<ClassKey, Raw> = HashMap::new();
        let (mut total_added, mut total_removed) = (0u64, 0u64);
        let (mut total_unchanged, mut total_changed) = (0u64, 0u64);

        for e in self.removed_file().iter() {
            if let Some(k) = entry_key(&self.before, &e)? {
                counts.entry(k).or_default().removed += 1;
                total_removed += 1;
            }
        }
        for e in self.added_file().iter() {
            if let Some(k) = entry_key(&self.after, &e)? {
                counts.entry(k).or_default().added += 1;
                total_added += 1;
            }
        }
        for e in self.common_file().iter() {
            let stub = DiffEntry {
                tag: e.tag,
                object_id: e.object_id,
                position: e.position1,
            };
            if let Some(k) = entry_key(&self.before, &stub)? {
                let raw = counts.entry(k).or_default();
                if e.changed {
                    raw.common_changed += 1;
                    total_changed += 1;
                } else {
                    raw.common_unchanged += 1;
                    total_unchanged += 1;
                }
            }
        }

        let mut by_class: Vec<ClassDiffEntry> = counts
            .into_iter()
            .map(|(key, raw)| ClassDiffEntry {
                class_name: self.key_name(key),
                key,
                count_removed: raw.removed,
                count_added: raw.added,
                count_common_unchanged: raw.common_unchanged,
                count_common_changed: raw.common_changed,
            })
            .collect();
        by_class.sort_by(|a, b| {
            (b.count_added + b.count_removed)
                .cmp(&(a.count_added + a.count_removed))
                .then_with(|| a.class_name.cmp(&b.class_name))
        });

        Ok(DiffSummary {
            total_before: total_removed + total_unchanged + total_changed,
            total_after: total_added + total_unchanged + total_changed,
            total_added,
            total_removed,
            total_common_unchanged: total_unchanged,
            total_common_changed: total_changed,
            by_class,
        })
    }
}

/// The histogram bucket of the object a diff entry points at.  A class dump
/// is bucketed under its own id so class-level changes (statics) show up
/// against the class name.
fn entry_key(query: &HeapQuery, entry: &DiffEntry) -> Result<Option<ClassKey>, HprofError> {
    let record = query.parse_entry(&SubIndexEntry {
        tag: entry.tag,
        object_id: entry.object_id,
        position: entry.position,
    })?;
    Ok(match &record {
        SubRecord::ClassDump(cd) => Some(ClassKey::Class(cd.class_id)),
        other => ClassKey::of(other),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heap_parser::FieldValue;
    use crate::index::MemStore;
    use crate::test_util::{ClassSpec, HprofBuilder, build_in_memory, ty};

    const CLASS: u64 = 0x10;

    /// `Node { int v }` instances with the given `(id, v)` values, plus a
    /// `char[]` with the given text.
    fn heap(nodes: &[(u64, i32)], chars: Option<(u64, &str)>) -> Arc<HeapQuery> {
        let mut b = HprofBuilder::new(8)
            .utf8(1, "v")
            .utf8(2, "Node")
            .load_class(1, CLASS, 2)
            .class_dump(ClassSpec::new(CLASS).instance_size(4).field(1, ty::INT));
        for &(id, v) in nodes {
            b = b.instance_values(id, CLASS, &[FieldValue::Int(v)]);
        }
        if let Some((id, s)) = chars {
            b = b.char_array(id, s);
        }
        Arc::new(build_in_memory(&b.build()))
    }

    /// before: nodes 1(v=1) 2(v=2) 3(v=3), chars "ab" at 0x90
    /// after:  nodes 2(v=2) 3(v=99) 4(v=4), chars "ab" at 0x90 (unchanged)
    fn diff() -> HeapDiff {
        let before = heap(&[(1, 1), (2, 2), (3, 3)], Some((0x90, "ab")));
        let after = heap(&[(2, 2), (3, 99), (4, 4)], Some((0x90, "ab")));
        HeapDiff::from_store(before, after, &MemStore::new()).unwrap()
    }

    fn ids(r: &PageResult<DiffObject>) -> Vec<u64> {
        r.items.iter().map(|o| o.object_id).collect()
    }

    #[test]
    fn counts_and_summary_match_the_two_heaps() {
        let d = diff();
        let c = d.counts();
        assert_eq!(
            (c.removed, c.added, c.common, c.common_changed),
            (1, 1, 4, 1)
        );
        let s = d.summary().unwrap();
        assert_eq!((s.total_before, s.total_after), (5, 5));
        assert_eq!((s.total_removed, s.total_added), (1, 1));
        assert_eq!(s.total_common(), 4);
        assert_eq!(s.total_common_changed, 1);

        let node = s
            .by_class
            .iter()
            .find(|e| e.key == ClassKey::Class(CLASS))
            .expect("Node bucket");
        assert_eq!(node.class_name, "Node");
        assert_eq!(
            (
                node.count_removed,
                node.count_added,
                node.count_common_changed,
                node.count_common_unchanged
            ),
            (1, 1, 1, 2)
        );
        assert_eq!(
            (node.count_before(), node.count_after(), node.net_change()),
            (4, 4, 0)
        );
        // The class object is bucketed with its instances (hence 2 unchanged); char[] too.
        assert!(s.by_class.iter().any(|e| e.class_name == "char[]"));
    }

    #[test]
    fn lists_are_paged_and_filterable_by_class() {
        let d = diff();
        assert_eq!(ids(&d.removed(None, Page::first(10)).unwrap()), vec![1]);
        assert_eq!(ids(&d.added(None, Page::first(10)).unwrap()), vec![4]);
        assert_eq!(ids(&d.changed(None, Page::first(10)).unwrap()), vec![3]);

        let node = Some(ClassKey::Class(CLASS));
        let all = d.common(None, node, Page::first(10)).unwrap();
        assert_eq!(ids(&all), vec![2, 3, CLASS]);
        assert_eq!(all.items[0].class_name, "Node");
        let unchanged = d.common(Some(false), node, Page::first(10)).unwrap();
        assert_eq!(ids(&unchanged), vec![2, CLASS]);

        // Paging with a filter: total counts the whole filtered list.
        let p = d.common(None, node, Page::new(1, 1)).unwrap();
        assert_eq!((ids(&p), p.total, p.has_more), (vec![3], 3, true));
        let p = d.common(None, node, Page::new(0, 1)).unwrap();
        assert_eq!((ids(&p), p.total, p.has_more), (vec![2], 3, true));

        // A class with no entries in the list.
        let none = d
            .removed(Some(ClassKey::PrimArray(5)), Page::first(10))
            .unwrap();
        assert_eq!((none.total, none.items.len()), (0, 0));
        // Unfiltered totals do not parse anything past the page.
        let p = d.common(None, None, Page::new(3, 10)).unwrap();
        assert_eq!((p.total, p.items.len()), (4, 1));
    }

    #[test]
    fn object_changed_reports_common_objects_only() {
        let d = diff();
        assert_eq!(d.object_changed(2), Some(false));
        assert_eq!(d.object_changed(3), Some(true));
        assert_eq!(d.object_changed(1), None);
        assert_eq!(d.object_changed(4), None);
    }

    #[test]
    fn building_twice_reuses_the_stored_entries() {
        let before = heap(&[(1, 1)], None);
        let after = heap(&[(2, 2)], None);
        let store = MemStore::new();
        let first = build_diff_indexes(&before, &after, &store).unwrap();
        assert_eq!((first.removed, first.added), (1, 1));
        assert_eq!(
            build_diff_indexes(&before, &after, &store).unwrap(),
            DiffIndexCounts::default()
        );
        let d = HeapDiff::from_store(before, after, &store).unwrap();
        assert_eq!(d.counts().removed, 1);
    }

    #[test]
    fn identical_dumps_have_no_differences() {
        let a = heap(&[(1, 1), (2, 2)], None);
        let d = HeapDiff::from_store(a.clone(), a, &MemStore::new()).unwrap();
        let c = d.counts();
        assert_eq!((c.removed, c.added, c.common_changed), (0, 0, 0));
        assert!(d.removed(None, Page::first(5)).unwrap().items.is_empty());
    }
}
