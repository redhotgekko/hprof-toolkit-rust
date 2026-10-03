//! The reference graph: who points at an object, what it points at, and why
//! it is still alive.
//!
//! Incoming references come from the precomputed reference index (one binary
//! search, no cap).  Outgoing references are read from the object itself, so
//! every edge can be labelled with the field, static or array slot that holds
//! it.  Paths to a GC root are found with a breadth-first search over
//! incoming references, bounded by [`RootPathLimits`].
//!
//! **Weak references count.** `java.lang.ref.Reference.referent` is an
//! ordinary field in a heap dump, so a path that goes through a weak or soft
//! reference is reported like any other.  Look for `referent` edges before
//! concluding that an object is strongly held.

use crate::heap_index::sub_record::SubIndexEntry;
use crate::heap_parser::{FieldValue, SubRecord};
use crate::hprof::HprofError;
use crate::query::{HeapQuery, Page, PageResult};
use crate::resolved::ResolvedRoot;
use crate::root_index::GcRootType;
use std::collections::{HashMap, HashSet, VecDeque};

/// How one object holds a reference to another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Edge {
    /// An instance field, by name.
    Field(String),
    /// A static field of a class object, by name.
    Static(String),
    /// A slot of an object array, by index.
    Element(usize),
}

impl std::fmt::Display for Edge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Edge::Field(name) => write!(f, ".{name}"),
            Edge::Static(name) => write!(f, "::{name}"),
            Edge::Element(i) => write!(f, "[{i}]"),
        }
    }
}

/// An outgoing reference: the target and the edge that holds it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reference {
    /// Object id of the referenced object.
    pub target: u64,
    /// Where in the referencing object the reference lives.
    pub via: Edge,
}

/// Bounds for [`HeapQuery::path_to_root`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootPathLimits {
    /// Give up after discovering this many distinct objects.
    pub max_nodes: usize,
    /// Do not follow chains longer than this many references.
    pub max_depth: usize,
}

impl Default for RootPathLimits {
    fn default() -> Self {
        Self {
            max_nodes: 100_000,
            max_depth: 10_000,
        }
    }
}

/// How a root-path search ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathOutcome {
    /// A GC root was reached; [`RootPath::steps`] holds the chain.
    Found,
    /// A limit in [`RootPathLimits`] stopped the search before it finished.
    /// An object that is reachable may still exist; raise the limits.
    LimitReached,
    /// Every referrer was examined and none leads to a GC root: the object is
    /// unreachable garbage or only held from an unindexed source.
    NotReachable,
}

/// One object on a path to a GC root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathStep {
    /// The object.
    pub object_id: u64,
    /// How the *previous* step (one closer to the root) references this
    /// object; `None` for the root itself.
    pub via: Option<Edge>,
}

/// The result of [`HeapQuery::path_to_root`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootPath {
    /// How the search ended.
    pub outcome: PathOutcome,
    /// GC root first, target last.  Empty unless `outcome` is `Found`.
    pub steps: Vec<PathStep>,
    /// Kinds of GC root the first step is (empty unless `Found`).
    pub root_kinds: Vec<GcRootType>,
    /// Distinct objects discovered by the search, including the target.
    pub nodes_visited: usize,
}

impl HeapQuery {
    /// One page of the references *to* `object_id`, as the ids of the
    /// objects holding them.
    ///
    /// There is one item per reference, so an object that holds two
    /// references to `object_id` appears twice.  `total` is exact and there
    /// is no cap.
    pub fn refs_to(&self, object_id: u64, page: Page) -> PageResult<u64> {
        let referrers = self.ref_index().referrers(object_id);
        let total = referrers.len();
        let items: Vec<u64> = referrers
            .skip(page.offset)
            .take(page.limit)
            .map(|e| e.from_object_id)
            .collect();
        let has_more = page.offset + items.len() < total;
        PageResult {
            items,
            total,
            has_more,
        }
    }

    /// One page of the references *from* `object_id`: instance fields, class
    /// statics or object-array slots that hold a non-null reference.
    ///
    /// Empty when the object does not exist or holds no references.  Reads
    /// only the object itself, so a huge array is paged without being
    /// loaded.
    pub fn refs_from(
        &self,
        object_id: u64,
        page: Page,
    ) -> Result<PageResult<Reference>, HprofError> {
        let Some(record) = self.object(object_id)? else {
            return Ok(PageResult {
                items: Vec::new(),
                total: 0,
                has_more: false,
            });
        };
        // Instances and classes resolve all their fields at once, so collect
        // them one time; arrays are paged lazily and counted in a second pass.
        let (total, items): (usize, Vec<Reference>) = match &record {
            SubRecord::ObjArrayDump(_) => (
                self.references_of(&record)?.count(),
                self.references_of(&record)?
                    .skip(page.offset)
                    .take(page.limit)
                    .collect(),
            ),
            _ => {
                let all: Vec<Reference> = self.references_of(&record)?.collect();
                let total = all.len();
                (
                    total,
                    all.into_iter().skip(page.offset).take(page.limit).collect(),
                )
            }
        };
        let has_more = page.offset + items.len() < total;
        Ok(PageResult {
            items,
            total,
            has_more,
        })
    }

    /// Outgoing references of `record`, lazily for arrays.
    fn references_of<'a>(
        &self,
        record: &SubRecord<'a>,
    ) -> Result<Box<dyn Iterator<Item = Reference> + 'a>, HprofError> {
        Ok(match record {
            SubRecord::InstanceDump(inst) => {
                let refs: Vec<Reference> = self
                    .instance_fields(inst)?
                    .into_iter()
                    .filter_map(|f| match f.value {
                        FieldValue::Object(t) if t != 0 => Some(Reference {
                            target: t,
                            via: Edge::Field(f.name),
                        }),
                        _ => None,
                    })
                    .collect();
                Box::new(refs.into_iter())
            }
            SubRecord::ClassDump(cd) => {
                let mut refs = Vec::new();
                for sf in cd.static_fields() {
                    let sf = sf?;
                    if let FieldValue::Object(t) = sf.value
                        && t != 0
                    {
                        let name = self
                            .lookup_name(sf.name_id)?
                            .unwrap_or_else(|| format!("<name#{}>", sf.name_id));
                        refs.push(Reference {
                            target: t,
                            via: Edge::Static(name),
                        });
                    }
                }
                Box::new(refs.into_iter())
            }
            SubRecord::ObjArrayDump(arr) => Box::new(
                arr.elements()
                    .enumerate()
                    .filter(|(_, t)| *t != 0)
                    .map(|(i, target)| Reference {
                        target,
                        via: Edge::Element(i),
                    }),
            ),
            _ => Box::new(std::iter::empty()),
        })
    }

    /// The first edge by which `from` references `to`, if any.
    fn edge_between(&self, from: u64, to: u64) -> Result<Option<Edge>, HprofError> {
        let Some(record) = self.object(from)? else {
            return Ok(None);
        };
        Ok(self
            .references_of(&record)?
            .find(|r| r.target == to)
            .map(|r| r.via))
    }

    /// The shortest chain of references from a GC root to `object_id`,
    /// found by breadth-first search over incoming references.
    ///
    /// Every referrer of every visited object is examined (there is no
    /// per-node cap); the search stops at `limits.max_nodes` discovered
    /// objects or `limits.max_depth` hops.  Each step records the field,
    /// static or array slot that links it to the next.
    pub fn path_to_root(&self, object_id: u64, limits: &RootPathLimits) -> RootPath {
        let found = |steps: Vec<PathStep>, visited: usize| {
            let root_kinds = steps
                .first()
                .map(|s| self.root_types_of(s.object_id))
                .unwrap_or_default();
            RootPath {
                outcome: PathOutcome::Found,
                steps,
                root_kinds,
                nodes_visited: visited,
            }
        };
        if self.is_gc_root(object_id) {
            return found(
                vec![PathStep {
                    object_id,
                    via: None,
                }],
                1,
            );
        }

        // parent[v] = u: v references u, so u is one hop closer to object_id.
        let mut parent: HashMap<u64, u64> = HashMap::new();
        let mut visited: HashSet<u64> = HashSet::from([object_id]);
        let mut queue: VecDeque<(u64, usize)> = VecDeque::from([(object_id, 0)]);
        let mut limited = false;

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= limits.max_depth {
                limited = true;
                continue;
            }
            for entry in self.ref_index().referrers(current) {
                let referrer = entry.from_object_id;
                if !visited.insert(referrer) {
                    continue;
                }
                parent.insert(referrer, current);

                if self.is_gc_root(referrer) {
                    // Chain: root → … → object_id, labelled with the edge from
                    // each step to the next.
                    let mut ids = vec![referrer];
                    let mut node = referrer;
                    while node != object_id {
                        let Some(&next) = parent.get(&node) else {
                            break;
                        };
                        ids.push(next);
                        node = next;
                    }
                    let mut steps = Vec::with_capacity(ids.len());
                    for (i, &id) in ids.iter().enumerate() {
                        let via = if i == 0 {
                            None
                        } else {
                            self.edge_between(ids[i - 1], id).ok().flatten()
                        };
                        steps.push(PathStep { object_id: id, via });
                    }
                    return found(steps, visited.len());
                }

                if visited.len() >= limits.max_nodes {
                    return RootPath {
                        outcome: PathOutcome::LimitReached,
                        steps: Vec::new(),
                        root_kinds: Vec::new(),
                        nodes_visited: visited.len(),
                    };
                }
                queue.push_back((referrer, depth + 1));
            }
        }

        RootPath {
            outcome: if limited {
                PathOutcome::LimitReached
            } else {
                PathOutcome::NotReachable
            },
            steps: Vec::new(),
            root_kinds: Vec::new(),
            nodes_visited: visited.len(),
        }
    }

    /// One page of the GC roots of `kind`, in ascending object-id order,
    /// each resolved to its record (thread serial, frame number, JNI ref…)
    /// and the type name of the rooted object.
    pub fn gc_roots(
        &self,
        kind: GcRootType,
        page: Page,
    ) -> Result<PageResult<ResolvedRoot>, HprofError> {
        let entries = self.iter_roots(kind);
        let total = entries.len();
        let mut items = Vec::new();
        for e in entries.skip(page.offset).take(page.limit) {
            let record = self.parse_entry(&SubIndexEntry {
                tag: kind.sub_record_tag(),
                object_id: e.object_id,
                position: e.position,
            })?;
            if let Some(root) = ResolvedRoot::from_sub_record(self, &record)? {
                items.push(root);
            }
        }
        let has_more = page.offset + items.len() < total;
        Ok(PageResult {
            items,
            total,
            has_more,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{
        ClassSpec, HprofBuilder, build_in_memory, standard_heap, std_ids::*, ty,
    };

    #[test]
    fn refs_to_pages_without_a_cap() {
        let heap = build_in_memory(&standard_heap());
        let r = heap.refs_to(INTEGER_42, Page::first(10));
        assert_eq!(r.items, vec![LIST]);
        assert_eq!(r.total, 1);
        assert!(!r.has_more);
        assert_eq!(heap.refs_to(0xDEAD, Page::first(10)).total, 0);
    }

    /// One target referenced by 120 holders: more than the old cap of 50.
    #[test]
    fn a_hub_object_reports_every_referrer() {
        const HOLDER_CLASS: u64 = 0x10;
        const TARGET: u64 = 0x20;
        let mut b = HprofBuilder::new(8)
            .utf8(1, "ref")
            .utf8(2, "Holder")
            .load_class(1, HOLDER_CLASS, 2)
            .class_dump(
                ClassSpec::new(HOLDER_CLASS)
                    .instance_size(8)
                    .field(1, ty::OBJECT),
            )
            .class_dump(ClassSpec::new(0x30));
        for i in 0..120u64 {
            b = b.instance_values(0x1000 + i, HOLDER_CLASS, &[FieldValue::Object(TARGET)]);
        }
        let heap = build_in_memory(&b.build());
        let all = heap.refs_to(TARGET, Page::first(1000));
        assert_eq!(all.total, 120);
        assert_eq!(all.items.len(), 120);
        let page = heap.refs_to(TARGET, Page::new(100, 50));
        assert_eq!(page.items.len(), 20);
        assert!(!page.has_more);
        assert!(heap.refs_to(TARGET, Page::first(50)).has_more);
    }

    #[test]
    fn refs_from_labels_fields_statics_and_elements() {
        let heap = build_in_memory(&standard_heap());
        let list = heap.refs_from(LIST, Page::first(10)).unwrap();
        assert_eq!(list.total, 1);
        assert_eq!(list.items[0].target, INTEGER_42);
        assert!(matches!(list.items[0].via, Edge::Field(_)));

        // Object array: element indexes, nulls skipped, paged.
        let bytes = HprofBuilder::new(8)
            .utf8(1, "[LElem;")
            .load_class(1, 0x10, 1)
            .class_dump(ClassSpec::new(0x10))
            .obj_array(0x50, 0x10, &[0x1, 0, 0x2, 0x3])
            .build();
        let heap = build_in_memory(&bytes);
        let all = heap.refs_from(0x50, Page::first(10)).unwrap();
        assert_eq!(all.total, 3);
        assert_eq!(
            all.items
                .iter()
                .map(|r| (r.target, r.via.clone()))
                .collect::<Vec<_>>(),
            vec![
                (0x1, Edge::Element(0)),
                (0x2, Edge::Element(2)),
                (0x3, Edge::Element(3))
            ]
        );
        let tail = heap.refs_from(0x50, Page::new(2, 10)).unwrap();
        assert_eq!(tail.items.len(), 1);
        assert!(!tail.has_more);

        // Class statics.
        let standard = build_in_memory(&standard_heap());
        let statics = standard
            .refs_from(ARRAYLIST_CLASS, Page::first(10))
            .unwrap();
        assert_eq!(statics.total, 0, "no object-valued statics in this heap");
        // Missing / non-referencing objects.
        assert_eq!(heap.refs_from(0xDEAD, Page::first(5)).unwrap().total, 0);
        assert_eq!(
            standard.refs_from(CHARS_HI, Page::first(5)).unwrap().total,
            0
        );
    }

    #[test]
    fn path_to_root_labels_every_hop() {
        let heap = build_in_memory(&standard_heap());
        // LIST is a Java-frame root and holds INTEGER_42.
        let path = heap.path_to_root(INTEGER_42, &RootPathLimits::default());
        assert_eq!(path.outcome, PathOutcome::Found);
        let ids: Vec<u64> = path.steps.iter().map(|s| s.object_id).collect();
        assert_eq!(ids, vec![LIST, INTEGER_42]);
        assert_eq!(path.steps[0].via, None);
        assert!(matches!(path.steps[1].via, Some(Edge::Field(_))));
        assert_eq!(path.root_kinds, vec![GcRootType::JavaFrame]);
    }

    #[test]
    fn a_root_is_its_own_path() {
        let heap = build_in_memory(&standard_heap());
        let path = heap.path_to_root(LIST, &RootPathLimits::default());
        assert_eq!(path.outcome, PathOutcome::Found);
        assert_eq!(path.steps.len(), 1);
    }

    #[test]
    fn unreachable_objects_and_tight_limits_are_told_apart() {
        let heap = build_in_memory(&standard_heap());
        // STRING_HI is unreachable in the standard heap.
        let none = heap.path_to_root(STRING_HI, &RootPathLimits::default());
        assert_eq!(none.outcome, PathOutcome::NotReachable);
        assert!(none.steps.is_empty());

        // A chain a -> b -> c -> root, searched with max_depth 1.
        let bytes = HprofBuilder::new(8)
            .utf8(1, "next")
            .utf8(2, "Node")
            .load_class(1, 0x10, 2)
            .class_dump(ClassSpec::new(0x10).instance_size(8).field(1, ty::OBJECT))
            .instance_values(0xC, 0x10, &[FieldValue::Object(0)])
            .instance_values(0xB, 0x10, &[FieldValue::Object(0xC)])
            .instance_values(0xA, 0x10, &[FieldValue::Object(0xB)])
            .root_unknown(0xA)
            .build();
        let heap = build_in_memory(&bytes);
        let full = heap.path_to_root(0xC, &RootPathLimits::default());
        assert_eq!(
            full.steps.iter().map(|s| s.object_id).collect::<Vec<_>>(),
            vec![0xA, 0xB, 0xC]
        );
        let shallow = heap.path_to_root(
            0xC,
            &RootPathLimits {
                max_depth: 1,
                ..RootPathLimits::default()
            },
        );
        assert_eq!(shallow.outcome, PathOutcome::LimitReached);
        let few_nodes = heap.path_to_root(
            0xC,
            &RootPathLimits {
                max_nodes: 2,
                ..RootPathLimits::default()
            },
        );
        assert_eq!(few_nodes.outcome, PathOutcome::LimitReached);
    }

    #[test]
    fn gc_roots_come_back_resolved_and_paged() {
        let heap = build_in_memory(&standard_heap());
        let sticky = heap
            .gc_roots(GcRootType::StickyClass, Page::first(10))
            .unwrap();
        assert_eq!(sticky.total, 2);
        assert!(matches!(sticky.items[0], ResolvedRoot::StickyClass { .. }));
        let frames = heap
            .gc_roots(GcRootType::JavaFrame, Page::first(10))
            .unwrap();
        assert!(matches!(
            frames.items[0],
            ResolvedRoot::JavaFrame { object_id, thread_serial: 1, .. } if object_id == LIST
        ));
        let one = heap
            .gc_roots(GcRootType::StickyClass, Page::first(1))
            .unwrap();
        assert!(one.has_more);
        assert_eq!(
            heap.gc_roots(GcRootType::JniGlobal, Page::first(1))
                .unwrap()
                .total,
            0
        );
    }
}
