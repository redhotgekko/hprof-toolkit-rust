//! Class-level views: the histogram, class search and class keys.
//!
//! Everything here is built from the precomputed class indexes and the
//! class-name cache, so cost is proportional to the classes touched, never to
//! the number of objects.

use crate::class_key::ClassKey;
use crate::hprof::BasicType;
use crate::query::{HeapQuery, Page, PageResult};
use crate::search::{Matcher, SearchQuery};
use std::cmp::Reverse;
use std::sync::Arc;

/// One row of the class histogram.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistogramEntry {
    /// What the row counts: instances of a class, or a kind of array.
    pub key: ClassKey,
    /// Display name: `java.lang.String`, `int[]`, `com.example.Foo[]`.
    pub class_name: String,
    /// Number of live objects with this key.
    pub instance_count: u64,
    /// Sum of shallow sizes in bytes (instance data only, no object header).
    pub shallow_bytes: u64,
}

/// A loaded class, as found by [`HeapQuery::find_classes`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassSummary {
    /// The class object's id.
    pub class_id: u64,
    /// Dot-notation name (`[B` for `byte[]`, the JVM's own name for arrays).
    pub name: Arc<str>,
    /// Which histogram bucket holds this class's instances.
    pub key: ClassKey,
    /// Number of live objects in that bucket.
    pub instance_count: usize,
}

/// Convert a JVM array class name to Java source notation:
/// `[Ljava.lang.Object;` → `java.lang.Object[]`, `[[I` → `int[][]`.
/// Names that are not array descriptors get `[]` appended, so that
/// `display_type_name("Foo")` is `Foo[]` (a plain class name, which real dumps
/// never use for arrays; kept so a bare element-class name still renders).
pub(crate) fn display_type_name(name: &str) -> String {
    let dims = name.bytes().take_while(|&b| b == b'[').count();
    if dims == 0 {
        return format!("{name}[]");
    }
    let rest = &name[dims..];
    let element = match rest.as_bytes() {
        [c] => BasicType::from_descriptor_char(char::from(*c))
            .map_or_else(|| rest.to_owned(), |t| t.java_name().to_owned()),
        _ => rest
            .strip_prefix('L')
            .and_then(|s| s.strip_suffix(';'))
            .unwrap_or(rest)
            .to_owned(),
    };
    format!("{element}{}", "[]".repeat(dims))
}

impl HeapQuery {
    /// Display name for a class key: `java.lang.String`, `int[]`,
    /// `com.example.Foo[]`.
    pub fn key_name(&self, key: ClassKey) -> String {
        match key {
            ClassKey::Class(id) => self.class_label(id),
            // hprof records the array *class* (`[Lfoo.Bar;`) for object arrays.
            ClassKey::ObjArray(array_class) => display_type_name(&self.class_label(array_class)),
            ClassKey::PrimArray(t) => format!("{}[]", BasicType::name_of_code(t)),
        }
    }

    /// The histogram bucket that holds the instances of the class object
    /// `class_id`, given its name.
    ///
    /// Ordinary classes map to themselves.  The class objects of array types
    /// map to the array buckets, because arrays are indexed by kind rather
    /// than by class object: `[B` to the `byte[]` bucket, and object-array
    /// classes (`[Ljava.lang.String;`, `[[I`) to the bucket keyed by that
    /// very class, which is how hprof records object arrays.
    pub fn class_key_for(&self, class_name: &str, class_id: u64) -> ClassKey {
        let Some(rest) = class_name.strip_prefix('[') else {
            return ClassKey::Class(class_id);
        };
        if let [c] = rest.as_bytes()
            && let Some(t) = BasicType::from_descriptor_char(char::from(*c))
        {
            return ClassKey::PrimArray(t.code());
        }
        ClassKey::ObjArray(class_id)
    }

    /// One page of the class histogram, largest instance count first.
    ///
    /// `total` is the number of buckets.  Names are resolved for the page
    /// only.
    pub fn class_histogram(&self, page: Page) -> PageResult<HistogramEntry> {
        let total = self.histogram_len();
        let items: Vec<HistogramEntry> = self
            .histogram()
            .skip(page.offset)
            .take(page.limit)
            .map(|r| {
                let key = r.key();
                HistogramEntry {
                    key,
                    class_name: self.key_name(key),
                    instance_count: r.instance_count,
                    shallow_bytes: r.shallow_bytes,
                }
            })
            .collect();
        let has_more = page.offset + items.len() < total;
        PageResult {
            items,
            total,
            has_more,
        }
    }

    /// `(objects, shallow bytes)` summed over the whole histogram.
    pub fn histogram_totals(&self) -> (u64, u64) {
        self.histogram().fold((0, 0), |(n, b), r| {
            (n + r.instance_count, b + r.shallow_bytes)
        })
    }

    /// Loaded classes whose name contains `needle` (case-insensitive; empty
    /// matches all), ordered by name.
    ///
    /// The simple form of [`Self::search_classes`].  A needle longer than
    /// [`crate::search::MAX_PATTERN_LEN`] bytes matches nothing.
    pub fn find_classes(&self, needle: &str, page: Page) -> PageResult<ClassSummary> {
        if needle.is_empty() {
            return self.classes_where(None, page);
        }
        match Matcher::new(&SearchQuery::contains(needle)) {
            Ok(m) => self.classes_where(Some(&m), page),
            Err(_) => PageResult::empty(),
        }
    }

    /// Loaded classes matching `matcher`.
    ///
    /// Ordered by name, or best match first when the matcher ranks
    /// ([`Matcher::ranks`], fuzzy mode).  Array classes match on either
    /// their JVM name (`[Ljava.lang.String;`) or their display name
    /// (`java.lang.String[]`).  Cost is one match per loaded class; nothing
    /// proportional to the number of objects.
    pub fn search_classes(&self, matcher: &Matcher, page: Page) -> PageResult<ClassSummary> {
        self.classes_where(Some(matcher), page)
    }

    fn classes_where(&self, matcher: Option<&Matcher>, page: Page) -> PageResult<ClassSummary> {
        let mut matches: Vec<(Reverse<u32>, Arc<str>, u64)> = Vec::new();
        for (class_id, _) in self.class_ids() {
            let Some(name) = self.class_name(class_id) else {
                continue;
            };
            let score = match matcher {
                None => 0,
                Some(m) => match class_name_score(m, &name) {
                    Some(score) => score,
                    None => continue,
                },
            };
            matches.push((Reverse(score), name, class_id));
        }
        // Scores are all zero unless the matcher ranks, so this is "by name"
        // for every other mode.
        matches.sort();
        let total = matches.len();
        let items = matches
            .into_iter()
            .skip(page.offset)
            .take(page.limit)
            .map(|(_, name, class_id)| {
                let key = self.class_key_for(&name, class_id);
                ClassSummary {
                    class_id,
                    instance_count: self.instance_count(key),
                    name,
                    key,
                }
            })
            .collect::<Vec<_>>();
        let has_more = page.offset + items.len() < total;
        PageResult {
            items,
            total,
            has_more,
        }
    }

    /// The rows of the class histogram whose display name matches
    /// `matcher`, in histogram order (largest instance count first).
    ///
    /// The histogram is a ranking, so search only filters it; a fuzzy
    /// matcher does not re-order the rows.  `total` is the number of
    /// matching rows.  Cost is one name resolution and one match per
    /// histogram bucket (one per class with instances), never per object.
    pub fn class_histogram_filtered(
        &self,
        matcher: &Matcher,
        page: Page,
    ) -> PageResult<HistogramEntry> {
        let rows = self.histogram().filter_map(|r| {
            let key = r.key();
            let class_name = self.key_name(key);
            matcher.is_match(&class_name).then_some(HistogramEntry {
                key,
                class_name,
                instance_count: r.instance_count,
                shallow_bytes: r.shallow_bytes,
            })
        });
        PageResult::from_unsized(page, rows)
    }
}

/// How well `matcher` fits a loaded class's JVM name.  Array classes are
/// also tried in Java notation (`[I` as `int[]`), and the better score wins.
fn class_name_score(matcher: &Matcher, name: &str) -> Option<u32> {
    let direct = matcher.score(name);
    if !name.starts_with('[') {
        return direct;
    }
    let display = matcher.score(&display_type_name(name));
    match (direct, display) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{build_in_memory, standard_heap, std_ids::*};

    #[test]
    fn histogram_pages_carry_names_and_totals() {
        let heap = build_in_memory(&standard_heap());
        let all = heap.class_histogram(Page::first(100));
        assert_eq!(all.total, 4);
        assert!(!all.has_more);
        let names: Vec<&str> = all.items.iter().map(|e| e.class_name.as_str()).collect();
        assert!(names.contains(&"java.lang.Integer"));
        assert!(names.contains(&"char[]"));
        let (n, _) = heap.histogram_totals();
        assert_eq!(n, 4);

        let first = heap.class_histogram(Page::first(1));
        assert_eq!(first.items.len(), 1);
        assert!(first.has_more);
        let last = heap.class_histogram(Page::new(3, 10));
        assert_eq!(last.items.len(), 1);
        assert!(!last.has_more);
        assert!(heap.class_histogram(Page::new(99, 10)).items.is_empty());
    }

    #[test]
    fn find_classes_matches_case_insensitively_and_sorts_by_name() {
        let heap = build_in_memory(&standard_heap());
        let r = heap.find_classes("JAVA.LANG", Page::first(10));
        let names: Vec<&str> = r.items.iter().map(|c| &*c.name).collect();
        assert_eq!(
            names,
            ["java.lang.Integer", "java.lang.Object", "java.lang.String"]
        );
        let integer = &r.items[0];
        assert_eq!(integer.class_id, INTEGER_CLASS);
        assert_eq!(integer.instance_count, 1);
        assert_eq!(r.items[1].instance_count, 0);

        let paged = heap.find_classes("", Page::new(1, 2));
        assert_eq!(paged.total, 5);
        assert_eq!(paged.items.len(), 2);
        assert!(paged.has_more);
        assert!(heap.find_classes("nope", Page::first(5)).items.is_empty());
    }

    #[test]
    fn search_classes_supports_every_mode() {
        let heap = build_in_memory(&standard_heap());
        let m = |q: SearchQuery| Matcher::new(&q).unwrap();
        let names = |r: PageResult<ClassSummary>| -> Vec<String> {
            r.items.iter().map(|c| c.name.to_string()).collect()
        };

        let regex = heap.search_classes(&m(SearchQuery::regex(r"^java\.lang\.")), Page::first(10));
        assert_eq!(
            names(regex),
            ["java.lang.Integer", "java.lang.Object", "java.lang.String"]
        );

        let exact = heap.search_classes(&m(SearchQuery::exact("[C")), Page::first(10));
        assert_eq!(exact.total, 1);
        assert_eq!(exact.items[0].key, ClassKey::PrimArray(5));
        // The display name of an array class matches too.
        let display = heap.search_classes(&m(SearchQuery::exact("char[]")), Page::first(10));
        assert_eq!(display.items[0].class_id, CHAR_ARRAY_CLASS);

        // Fuzzy ranks: `jlint` fits java.lang.Integer best.
        let fuzzy = heap.search_classes(&m(SearchQuery::fuzzy("jlint")), Page::first(10));
        assert_eq!(names(fuzzy)[0], "java.lang.Integer");
        let fuzzy = heap.search_classes(&m(SearchQuery::fuzzy("jl")), Page::first(2));
        assert_eq!(fuzzy.total, 4); // ArrayList has a j…l subsequence too
        assert!(fuzzy.has_more);

        let cs = heap.search_classes(
            &m(SearchQuery::contains("INTEGER").case_sensitive(true)),
            Page::first(10),
        );
        assert!(cs.items.is_empty());
        assert!(
            heap.find_classes(&"x".repeat(1000), Page::first(1))
                .items
                .is_empty()
        );
    }

    #[test]
    fn filtered_histogram_keeps_histogram_order() {
        let heap = build_in_memory(&standard_heap());
        let all = heap.class_histogram(Page::first(10));
        let arrays = Matcher::new(&SearchQuery::regex(r"\[\]$")).unwrap();
        let r = heap.class_histogram_filtered(&arrays, Page::first(10));
        assert_eq!(r.total, 1);
        assert_eq!(r.items[0].class_name, "char[]");
        assert!(!r.has_more);

        // Every row matches `.`: same rows, same order as the plain histogram.
        let any = Matcher::new(&SearchQuery::regex(".")).unwrap();
        let r = heap.class_histogram_filtered(&any, Page::first(10));
        assert_eq!(r.items, all.items);
        let paged = heap.class_histogram_filtered(&any, Page::new(1, 2));
        assert_eq!(paged.items, all.items[1..3]);
        assert!(paged.has_more);
        assert_eq!(paged.total, 4);

        let none = Matcher::new(&SearchQuery::contains("nothing")).unwrap();
        assert_eq!(
            heap.class_histogram_filtered(&none, Page::first(10)).total,
            0
        );
    }

    #[test]
    fn array_class_objects_map_to_the_array_buckets() {
        let heap = build_in_memory(&standard_heap());
        // `[C` is the char[] class in the standard heap.
        let r = heap.find_classes("[C", Page::first(5));
        assert_eq!(r.items.len(), 1);
        assert_eq!(r.items[0].key, ClassKey::PrimArray(5));
        assert_eq!(r.items[0].instance_count, 1);
        assert_eq!(heap.key_name(ClassKey::PrimArray(5)), "char[]");
    }

    #[test]
    fn class_key_for_maps_array_classes_to_their_buckets() {
        let heap = build_in_memory(&standard_heap());
        assert_eq!(
            heap.class_key_for("[Ljava.lang.Integer;", 0x77),
            ClassKey::ObjArray(0x77)
        );
        assert_eq!(heap.class_key_for("[[I", 0x78), ClassKey::ObjArray(0x78));
        assert_eq!(heap.class_key_for("[I", 0x1), ClassKey::PrimArray(10));
        assert_eq!(heap.class_key_for("x.Y", 0x7), ClassKey::Class(0x7));
        // A plain class name (not a JVM array descriptor) renders with a
        // `[]` suffix.
        assert_eq!(
            heap.key_name(ClassKey::ObjArray(INTEGER_CLASS)),
            "java.lang.Integer[]"
        );
    }

    #[test]
    fn display_type_name_converts_descriptors() {
        assert_eq!(
            display_type_name("[Ljava.lang.Object;"),
            "java.lang.Object[]"
        );
        assert_eq!(display_type_name("[[I"), "int[][]");
        assert_eq!(display_type_name("[[Lfoo.Bar;"), "foo.Bar[][]");
        assert_eq!(display_type_name("[J"), "long[]");
        assert_eq!(display_type_name("foo.Bar"), "foo.Bar[]");
    }
}
