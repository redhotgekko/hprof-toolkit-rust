//! The MCP tools.
//!
//! Every tool is a thin adapter over [`HeapQuery`] / [`HeapDiff`]: it parses
//! [`Args`], calls the library, and returns a [`ToolOutput`].  Ids are
//! written as `"0x…"` strings, byte counts as exact integers, and every list
//! is paged with `total` / `has_more` / `next_offset`.

use super::args::{Args, ToolError};
use crate::array_index::ArrayKind;
use crate::class_key::ClassKey;
use crate::graph::{PathOutcome, RootPathLimits};
use crate::heap_parser::SubRecord;
use crate::hprof::BasicType;
use crate::query::{HeapQuery, Page, PageResult, ScanWindow};
use crate::resolved::{PrimArrayElements, ResolvedRoot, Value as JavaValue};
use crate::root_index::GcRootType;
use crate::search::{Matcher, SearchMode, SearchQuery};
use crate::server::AppState;
use serde_json::{Map, Value, json};

/// What a tool returns: a summary line for humans plus the JSON payload.
pub struct ToolOutput {
    pub summary: String,
    pub data: Value,
}

impl ToolOutput {
    fn new(summary: impl Into<String>, data: Value) -> Self {
        Self {
            summary: summary.into(),
            data,
        }
    }
}

type ToolResult = Result<ToolOutput, ToolError>;

/// A registered tool.
pub struct Tool {
    pub name: &'static str,
    pub description: &'static str,
    /// JSON Schema of the `arguments` object.
    pub schema: fn() -> Value,
    pub run: fn(&AppState, &Args) -> ToolResult,
}

/// Every tool, in the order `tools/list` shows them.
pub fn all() -> Vec<Tool> {
    macro_rules! tool {
        ($name:literal, $desc:literal, $schema:expr, $run:expr) => {
            Tool {
                name: $name,
                description: $desc,
                schema: $schema,
                run: $run,
            }
        };
    }
    vec![
        tool!(
            "heap_summary",
            "Start here. Object, class and reference counts, GC roots by kind, thread count, and \
             which optional indexes exist (retained sizes, second dump for diffs). Next: \
             class_histogram or retained_top.",
            || schema(json!({}), &[]),
            heap_summary
        ),
        tool!(
            "class_histogram",
            "Classes ranked by live instance count (default) or by total shallow bytes. Includes \
             array types such as `byte[]`. Shallow size counts instance data only, not the \
             object header or referenced objects. Give `name` (with `mode`) to keep only \
             matching classes instead of paging through everything; the order stays by count \
             or bytes. Next: instances_of_class, or find_classes.",
            || with_page(
                merge(
                    json!({
                        "sort_by": {"type": "string", "enum": ["instances", "shallow_bytes"],
                                    "description": "Ranking. Default `instances`."}
                    }),
                    search_props(
                        "name",
                        "Only classes whose name matches this (optional), e.g. `com.example` or, with `mode` regex, `^java\\.util\\..*Map$`.",
                        &SearchMode::ALL,
                    )
                ),
                &[]
            ),
            class_histogram
        ),
        tool!(
            "find_classes",
            "Find loaded classes by name: substring (default), the exact name, a regex, or a \
             fuzzy pattern (`mode`); case-insensitive unless `case_sensitive`. Fuzzy results \
             come best match first, the rest by name. Answers from the class index only, so it \
             is fast. Next: class_info or instances_of_class.",
            || with_page(
                search_props(
                    "name",
                    "Class name or part of it, e.g. `HashMap`; a regex or fuzzy pattern with `mode`.",
                    &SearchMode::ALL,
                ),
                &["name"]
            ),
            find_classes
        ),
        tool!(
            "class_info",
            "One class: superclass, instance size, instance count, instance field layout and \
             static field values. Give `class_id` or `class_name`.",
            || json!({"type": "object", "properties": class_selector(), "required": []}),
            class_info
        ),
        tool!(
            "instances_of_class",
            "Ids of the live instances of a class, ascending. Give `class_id` or `class_name` \
             (array types by name: `byte[]`, `java.lang.Object[]`). Does not include subclass \
             instances. Next: object on any id.",
            || with_page(class_selector(), &[]),
            instances_of_class
        ),
        tool!(
            "object",
            "Everything known about one object: type, field values (wrapper types and Strings \
             unwrapped), retained size and dominator when indexed, GC root kinds, and how many \
             objects reference it. Next: path_to_gc_root or references_to for why it is alive.",
            || json!({"type": "object", "properties": {"id": id_prop()}, "required": ["id"]}),
            object
        ),
        tool!(
            "object_string",
            "The full text of a java.lang.String object (up to `max_chars`).",
            || json!({"type": "object", "properties": {
                "id": id_prop(),
                "max_chars": {"type": "integer", "description": "Default 10000, at most 1000000."}
            }, "required": ["id"]}),
            object_string
        ),
        tool!(
            "array_contents",
            "A window of an array's elements: values for primitive arrays, element ids with \
             types for object arrays. Use offset and limit (default 100, at most 1000).",
            || json!({"type": "object", "properties": {
                "id": id_prop(),
                "offset": {"type": "integer"},
                "limit": {"type": "integer"}
            }, "required": ["id"]}),
            array_contents
        ),
        tool!(
            "references_to",
            "Objects that hold a reference to this one (incoming references), paged with no cap. \
             One item per reference. Next: object on a referrer, or path_to_gc_root.",
            || with_page(json!({"id": id_prop()}), &["id"]),
            references_to
        ),
        tool!(
            "references_from",
            "What this object references (outgoing), each labelled with the field, static or \
             array index that holds it. Paged.",
            || with_page(json!({"id": id_prop()}), &["id"]),
            references_from
        ),
        tool!(
            "path_to_gc_root",
            "The shortest reference chain from a GC root to this object, with the field that \
             links each step: the answer to \"why is this still alive?\". Weak and soft \
             references count as edges (look for `referent`). The search is bounded; the \
             outcome says whether it found a path, hit the limit, or proved none exists.",
            || json!({"type": "object", "properties": {
                "id": id_prop(),
                "max_nodes": {"type": "integer", "description": "Objects to discover before giving up. Default 100000."}
            }, "required": ["id"]}),
            path_to_gc_root
        ),
        tool!(
            "gc_roots",
            "GC roots. Without `kind`, counts per kind. With `kind`, a page of that kind's roots \
             with thread serial and frame where recorded.",
            || with_page(
                json!({"kind": {"type": "string", "enum": root_slugs(),
                                          "description": "Root kind."}}),
                &[]
            ),
            gc_roots
        ),
        tool!(
            "threads",
            "Threads in the dump: serial, name, group, thread object id, and whether it ended. \
             Give `name` (with `mode`) to keep only matching threads. Next: thread_stack.",
            || schema(
                search_props(
                    "name",
                    "Only threads whose name or group matches this (optional).",
                    &SearchMode::ALL,
                ),
                &[]
            ),
            threads
        ),
        tool!(
            "thread_stack",
            "The call stack of one thread at dump time, outermost frame first.",
            || json!({"type": "object", "properties": {
                "serial": {"type": "integer", "description": "Thread serial from `threads`."}
            }, "required": ["serial"]}),
            thread_stack
        ),
        tool!(
            "largest_arrays",
            "The biggest arrays of one element kind, by size in bytes. Kinds: boolean, char, \
             float, double, byte, short, int, long, object.",
            || with_page(
                json!({"kind": {"type": "string", "enum": array_slugs()}}),
                &["kind"]
            ),
            largest_arrays
        ),
        tool!(
            "retained_top",
            "Objects ranked by retained size: the bytes that would be freed if the object died. \
             Needs the retained index (built by `hprof-toolkit index`). Next: dominated_by to \
             see what an object keeps alive.",
            || with_page(json!({}), &[]),
            retained_top
        ),
        tool!(
            "dominated_by",
            "The objects an object directly dominates (everything reachable only through it), \
             with their retained sizes. Needs the retained index.",
            || with_page(json!({"id": id_prop()}), &["id"]),
            dominated_by
        ),
        tool!(
            "search_strings",
            "Find java.lang.String objects whose text matches: substring (default), the whole \
             text, or a regex (`mode`); case-insensitive unless `case_sensitive`. Scans strings \
             in id order, at most `max_scan` per call, and returns `next_cursor` to continue; \
             this reads string contents, so it can be slow on huge heaps.",
            || schema(
                merge(
                    search_props("text", "Text to look for.", &STRING_MODES),
                    json!({
                        "max_results": {"type": "integer", "description": "Default 20, at most 200."},
                        "max_scan": {"type": "integer", "description": "Strings to examine per call. Default 100000, at most 5000000."},
                        "cursor": {"type": "integer", "description": "`next_cursor` from the previous call."}
                    })
                ),
                &["text"]
            ),
            search_strings
        ),
        tool!(
            "heap_diff_summary",
            "Compare against the second dump (server started with --diff-hprof): per class, how \
             many instances were added, removed, or survived changed/unchanged, biggest change \
             first. Object ids are heap addresses, so a reused address can look like a survivor.",
            || with_page(json!({}), &[]),
            heap_diff_summary
        ),
        tool!(
            "heap_diff_objects",
            "List objects that were added, removed, or changed between the two dumps, optionally \
             for one class. Ids from `added` exist only in the second dump.",
            || with_page(
                json!({
                    "kind": {"type": "string", "enum": ["added", "removed", "changed", "unchanged"]},
                    "class_name": {"type": "string", "description": "Restrict to one class, e.g. `java.lang.String` or `byte[]`."}
                }),
                &["kind"]
            ),
            heap_diff_objects
        ),
    ]
}

// ── Schema helpers ────────────────────────────────────────────────────────────

fn schema(properties: Value, required: &[&str]) -> Value {
    json!({"type": "object", "properties": properties, "required": required})
}

/// The modes `search_strings` offers: no fuzzy, see
/// [`HeapQuery::search_strings`].
const STRING_MODES: [SearchMode; 3] = [SearchMode::Contains, SearchMode::Exact, SearchMode::Regex];

/// The properties every searching tool shares: the text under `text_key`,
/// `mode` (one of `modes`) and `case_sensitive`.
fn search_props(text_key: &str, text_desc: &str, modes: &[SearchMode]) -> Value {
    let names: Vec<&str> = modes.iter().map(|m| m.name()).collect();
    let mut explain = vec!["`contains` (default): substring."];
    if modes.contains(&SearchMode::Exact) {
        explain.push("`exact`: the whole text must match.");
    }
    if modes.contains(&SearchMode::Regex) {
        explain.push("`regex`: a Rust regular expression, unanchored.");
    }
    if modes.contains(&SearchMode::Fuzzy) {
        explain.push(
            "`fuzzy`: every character in order, best matches first (for names typed from memory).",
        );
    }
    let mut props = Map::new();
    props.insert(
        text_key.to_owned(),
        json!({"type": "string", "description": text_desc}),
    );
    props.insert(
        "mode".to_owned(),
        json!({"type": "string", "enum": names, "description": explain.join(" ")}),
    );
    props.insert(
        "case_sensitive".to_owned(),
        json!({"type": "boolean", "description": "Default false: every mode ignores case."}),
    );
    Value::Object(props)
}

/// Two property objects as one.
fn merge(a: Value, b: Value) -> Value {
    let mut props = a;
    if let (Some(into), Value::Object(from)) = (props.as_object_mut(), b) {
        into.extend(from);
    }
    props
}

/// The `mode` argument, when given and one of `allowed`.
fn opt_mode(args: &Args, allowed: &[SearchMode]) -> Result<Option<SearchMode>, ToolError> {
    let Some(name) = args.opt_str("mode")? else {
        return Ok(None);
    };
    match SearchMode::from_name(name) {
        Some(m) if allowed.contains(&m) => Ok(Some(m)),
        _ => {
            let list: Vec<String> = allowed.iter().map(|m| format!("`{}`", m.name())).collect();
            Err(ToolError::new(format!(
                "`mode` must be one of {}, not `{name}`.",
                list.join(", ")
            )))
        }
    }
}

/// Compile a tool's search: `text` in `mode` (the `mode` argument, else
/// `default_mode`), honouring `case_sensitive`.
fn search_matcher(
    args: &Args,
    text: &str,
    allowed: &[SearchMode],
    default_mode: SearchMode,
) -> Result<Matcher, ToolError> {
    let mode = opt_mode(args, allowed)?.unwrap_or(default_mode);
    let case_sensitive = args.bool_or("case_sensitive", false)?;
    Ok(Matcher::new(
        &SearchQuery::new(text, mode).case_sensitive(case_sensitive),
    )?)
}

/// `properties` plus `offset` and `limit`.
fn with_page(properties: Value, required: &[&str]) -> Value {
    let mut props = properties;
    if let Some(m) = props.as_object_mut() {
        m.insert(
            "offset".into(),
            json!({"type": "integer", "description": "Items to skip. Default 0."}),
        );
        m.insert(
            "limit".into(),
            json!({"type": "integer", "description": "Items to return. Default 50, at most 500."}),
        );
    }
    schema(props, required)
}

fn id_prop() -> Value {
    json!({"type": ["string", "integer"],
           "description": "Object id: hex string like \"0x1a2b3c\" (as printed by other tools) or a decimal number."})
}

fn class_selector() -> Value {
    json!({
        "class_id": id_prop(),
        "class_name": {"type": "string",
                       "description": "Dotted class name, or an array type like `byte[]`."}
    })
}

fn root_slugs() -> Vec<&'static str> {
    GcRootType::ALL.iter().map(|k| root_slug(*k)).collect()
}

fn array_slugs() -> Vec<&'static str> {
    ArrayKind::ALL.iter().map(|k| k.slug()).collect()
}

fn root_slug(k: GcRootType) -> &'static str {
    k.slug()
}

// ── JSON helpers ──────────────────────────────────────────────────────────────

fn hex(id: u64) -> String {
    format!("0x{id:x}")
}

/// `{items, total, offset, limit, has_more, next_offset}` around `items`.
fn paged(page: Page, total: usize, has_more: bool, items: Vec<Value>) -> Value {
    json!({
        "items": items,
        "total": total,
        "offset": page.offset,
        "limit": page.limit,
        "has_more": has_more,
        "next_offset": if has_more { Some(page.offset + items.len()) } else { None },
    })
}

fn type_name(q: &HeapQuery, id: u64) -> String {
    q.object_type_name(id)
}

/// A resolved field value as JSON.  Objects carry their id and type so the
/// model can follow them with `object`.
fn value_json(q: &HeapQuery, v: &JavaValue) -> Value {
    const MAX_TEXT: usize = 500;
    let boxed =
        |id: u64, ty: &str, value: Value| json!({"id": hex(id), "type": ty, "value": value});
    match v {
        JavaValue::Null => Value::Null,
        JavaValue::Bool(b) => json!(b),
        JavaValue::Char(c) => json!(
            char::from_u32(u32::from(*c))
                .map(String::from)
                .unwrap_or_default()
        ),
        JavaValue::Float(f) => json!(f),
        JavaValue::Double(d) => json!(d),
        JavaValue::Byte(b) => json!(b),
        JavaValue::Short(s) => json!(s),
        JavaValue::Int(i) => json!(i),
        JavaValue::Long(l) => json!(l),
        JavaValue::String(id, s) => {
            let truncated = s.chars().count() > MAX_TEXT;
            let text: String = s.chars().take(MAX_TEXT).collect();
            json!({"id": hex(*id), "type": "java.lang.String", "value": text, "truncated": truncated})
        }
        JavaValue::BoxedInt(id, x) => boxed(*id, "java.lang.Integer", json!(x)),
        JavaValue::BoxedLong(id, x) => boxed(*id, "java.lang.Long", json!(x)),
        JavaValue::BoxedDouble(id, x) => boxed(*id, "java.lang.Double", json!(x)),
        JavaValue::BoxedFloat(id, x) => boxed(*id, "java.lang.Float", json!(x)),
        JavaValue::BoxedShort(id, x) => boxed(*id, "java.lang.Short", json!(x)),
        JavaValue::BoxedByte(id, x) => boxed(*id, "java.lang.Byte", json!(x)),
        JavaValue::BoxedBoolean(id, x) => boxed(*id, "java.lang.Boolean", json!(x)),
        JavaValue::BoxedCharacter(id, x) => boxed(
            *id,
            "java.lang.Character",
            json!(
                char::from_u32(u32::from(*x))
                    .map(String::from)
                    .unwrap_or_default()
            ),
        ),
        JavaValue::Object(id) => json!({"id": hex(*id), "type": type_name(q, *id)}),
    }
}

fn elements_json(e: &PrimArrayElements) -> Vec<Value> {
    match e {
        PrimArrayElements::Bool(v) => v.iter().map(|x| json!(x)).collect(),
        PrimArrayElements::Char(v) => v
            .iter()
            .map(|&u| {
                json!(
                    char::from_u32(u32::from(u))
                        .map(String::from)
                        .unwrap_or_default()
                )
            })
            .collect(),
        PrimArrayElements::Float(v) => v.iter().map(|x| json!(x)).collect(),
        PrimArrayElements::Double(v) => v.iter().map(|x| json!(x)).collect(),
        PrimArrayElements::Byte(v) => v.iter().map(|x| json!(x)).collect(),
        PrimArrayElements::Short(v) => v.iter().map(|x| json!(x)).collect(),
        PrimArrayElements::Int(v) => v.iter().map(|x| json!(x)).collect(),
        PrimArrayElements::Long(v) => v.iter().map(|x| json!(x)).collect(),
    }
}

fn root_json(root: &ResolvedRoot) -> Value {
    match root {
        ResolvedRoot::Unknown {
            object_id,
            object_type_name,
        } => {
            json!({"kind": "unknown", "object_id": hex(*object_id), "type": object_type_name})
        }
        ResolvedRoot::JniGlobal {
            object_id,
            jni_global_ref_id,
            object_type_name,
        } => json!({
            "kind": "jni_global", "object_id": hex(*object_id),
            "jni_global_ref_id": hex(*jni_global_ref_id), "type": object_type_name}),
        ResolvedRoot::JniLocal {
            object_id,
            thread_serial,
            frame_number,
            object_type_name,
        } => json!({
            "kind": "jni_local", "object_id": hex(*object_id), "thread_serial": thread_serial,
            "frame_number": frame_number, "type": object_type_name}),
        ResolvedRoot::JavaFrame {
            object_id,
            thread_serial,
            frame_number,
            object_type_name,
        } => json!({
            "kind": "java_frame", "object_id": hex(*object_id), "thread_serial": thread_serial,
            "frame_number": frame_number, "type": object_type_name}),
        ResolvedRoot::NativeStack {
            object_id,
            thread_serial,
            object_type_name,
        } => json!({
            "kind": "native_stack", "object_id": hex(*object_id), "thread_serial": thread_serial,
            "type": object_type_name}),
        ResolvedRoot::StickyClass {
            class_id,
            class_name,
        } => {
            json!({"kind": "sticky_class", "object_id": hex(*class_id), "type": class_name})
        }
        ResolvedRoot::ThreadBlock {
            object_id,
            thread_serial,
            object_type_name,
        } => json!({
            "kind": "thread_block", "object_id": hex(*object_id), "thread_serial": thread_serial,
            "type": object_type_name}),
        ResolvedRoot::MonitorUsed {
            object_id,
            object_type_name,
        } => {
            json!({"kind": "monitor_used", "object_id": hex(*object_id), "type": object_type_name})
        }
        ResolvedRoot::ThreadObj {
            thread_object_id,
            thread_serial,
            stack_trace_serial,
            object_type_name,
        } => json!({
            "kind": "thread_object", "object_id": hex(*thread_object_id),
            "thread_serial": thread_serial, "stack_trace_serial": stack_trace_serial,
            "type": object_type_name}),
    }
}

// ── Class selection ───────────────────────────────────────────────────────────

/// Resolve a Java type name (`java.util.HashMap`, `byte[]`,
/// `java.lang.Object[][]`) to its histogram bucket.
fn key_for_name(q: &HeapQuery, name: &str) -> Option<ClassKey> {
    let mut base = name.trim();
    let mut dims = 0;
    while let Some(rest) = base.strip_suffix("[]") {
        base = rest;
        dims += 1;
    }
    if dims == 0 {
        return q.find_class_by_name(base).map(ClassKey::Class);
    }
    let primitive = BasicType::ALL
        .into_iter()
        .find(|t| t.is_primitive() && t.java_name() == base);
    if let (1, Some(t)) = (dims, primitive) {
        return Some(ClassKey::PrimArray(t.code()));
    }
    let element = match primitive {
        Some(t) => t.descriptor_char().to_string(),
        None => format!("L{base};"),
    };
    let descriptor = format!("{}{element}", "[".repeat(dims));
    q.find_class_by_name(&descriptor).map(ClassKey::ObjArray)
}

/// The bucket selected by `class_id` or `class_name`.
fn class_key_arg(q: &HeapQuery, args: &Args) -> Result<ClassKey, ToolError> {
    if let Some(id) = args.opt_id("class_id")? {
        return match q.class(id)? {
            Some(_) => {
                let name = q.class_label(id);
                Ok(q.class_key_for(&name, id))
            }
            None => Err(ToolError::new(format!(
                "{} is not a class object. Use find_classes to look a class up by name.",
                hex(id)
            ))),
        };
    }
    let name = args
        .opt_str("class_name")?
        .ok_or_else(|| ToolError::new("Give `class_id` or `class_name`."))?;
    key_for_name(q, name).ok_or_else(|| {
        ToolError::new(format!(
            "No class named `{name}`. Use find_classes to search by part of the name."
        ))
    })
}

// ── Tools ─────────────────────────────────────────────────────────────────────

fn heap_summary(state: &AppState, _args: &Args) -> ToolResult {
    let q = &*state.query;
    let (objects, shallow) = q.histogram_totals();
    let roots: serde_json::Map<String, Value> = GcRootType::ALL
        .iter()
        .map(|&k| (root_slug(k).to_owned(), json!(q.iter_roots(k).len())))
        .collect();
    let header = q.hprof_header();
    let classes = q.class_ids().count();
    let threads = q.threads()?.len();
    let data = json!({
        "hprof_path": state.hprof_path.display().to_string(),
        "hprof_version": header.version,
        "timestamp_ms": header.timestamp_ms,
        "id_size": q.id_size(),
        "sub_records": q.object_count(),
        "objects": objects,
        "shallow_bytes": shallow,
        "classes": classes,
        "references": q.ref_count(),
        "gc_roots": roots,
        "threads": threads,
        "indexes": {"retained_sizes": q.has_retained_heap(), "diff": state.diff.is_some()},
    });
    Ok(ToolOutput::new(
        format!(
            "{objects} objects in {} classes, {shallow} shallow bytes. Retained sizes {}available; diff {}configured.",
            classes,
            if q.has_retained_heap() { "" } else { "not " },
            if state.diff.is_some() { "" } else { "not " },
        ),
        data,
    ))
}

fn class_histogram(state: &AppState, args: &Args) -> ToolResult {
    let q = &*state.query;
    let page = args.page()?;
    let by_bytes = match args.opt_str("sort_by")? {
        None | Some("instances") => false,
        Some("shallow_bytes") => true,
        Some(other) => {
            return Err(ToolError::new(format!(
                "`sort_by` must be `instances` or `shallow_bytes`, not `{other}`."
            )));
        }
    };
    let name = args.opt_str("name")?.filter(|n| !n.is_empty());
    let matcher = match name {
        Some(n) => Some(search_matcher(
            args,
            n,
            &SearchMode::ALL,
            SearchMode::Contains,
        )?),
        None => None,
    };
    // (total, rows) where each row is (key, instances, shallow bytes).
    let (total, rows): (usize, Vec<(ClassKey, u64, u64)>) = if by_bytes {
        let mut all: Vec<_> = q
            .histogram()
            .filter(|r| {
                matcher
                    .as_ref()
                    .is_none_or(|m| m.is_match(&q.key_name(r.key())))
            })
            .collect();
        all.sort_by_key(|r| std::cmp::Reverse(r.shallow_bytes));
        let total = all.len();
        let rows = all
            .into_iter()
            .skip(page.offset)
            .take(page.limit)
            .map(|r| (r.key(), r.instance_count, r.shallow_bytes))
            .collect();
        (total, rows)
    } else {
        let r = match &matcher {
            Some(m) => q.class_histogram_filtered(m, page),
            None => q.class_histogram(page),
        };
        let rows = r
            .items
            .iter()
            .map(|e| (e.key, e.instance_count, e.shallow_bytes))
            .collect();
        (r.total, rows)
    };
    let items: Vec<Value> = rows
        .iter()
        .map(|&(key, instances, bytes)| {
            json!({
                "class_name": q.key_name(key),
                "class_id": match key { ClassKey::Class(id) => Some(hex(id)), _ => None },
                "instances": instances,
                "shallow_bytes": bytes,
            })
        })
        .collect();
    let has_more = page.offset + items.len() < total;
    let what = match name {
        Some(n) => format!("classes matching `{n}`"),
        None => "classes".to_owned(),
    };
    Ok(ToolOutput::new(
        format!(
            "{} of {total} {what}, by {}.",
            items.len(),
            if by_bytes {
                "shallow bytes"
            } else {
                "instance count"
            }
        ),
        paged(page, total, has_more, items),
    ))
}

fn find_classes(state: &AppState, args: &Args) -> ToolResult {
    let q = &*state.query;
    let name = args.str("name")?;
    let page = args.page()?;
    // `exact: true` is an older spelling of `mode: "exact"`; still honoured.
    let default_mode = if args.bool_or("exact", false)? {
        SearchMode::Exact
    } else {
        SearchMode::Contains
    };
    let matcher = search_matcher(args, name, &SearchMode::ALL, default_mode)?;
    // An exact name that the class map knows is answered without a scan.
    let fast = (matcher.mode() == SearchMode::Exact)
        .then(|| q.find_class_by_name(name))
        .flatten();
    let result = match fast {
        Some(id) => {
            let key = q.class_key_for(name, id);
            PageResult {
                items: vec![crate::classes::ClassSummary {
                    class_id: id,
                    name: name.into(),
                    key,
                    instance_count: q.instance_count(key),
                }],
                total: 1,
                has_more: false,
            }
        }
        None => q.search_classes(&matcher, page),
    };
    let items: Vec<Value> = result
        .items
        .iter()
        .map(|c| json!({"class_id": hex(c.class_id), "class_name": &*c.name, "instances": c.instance_count}))
        .collect();
    Ok(ToolOutput::new(
        format!(
            "{} class(es) match `{name}` ({}).",
            result.total,
            matcher.mode().name()
        ),
        paged(page, result.total, result.has_more, items),
    ))
}

fn class_json(q: &HeapQuery, class_id: u64) -> Result<Option<Value>, ToolError> {
    let Some(cd) = q.class(class_id)? else {
        return Ok(None);
    };
    let rc = q.resolve_class(&cd)?;
    let key = q.class_key_for(&rc.class_name, class_id);
    Ok(Some(json!({
        "kind": "class",
        "class_id": hex(class_id),
        "class_name": rc.class_name,
        "super_class": if rc.super_class_id == 0 { Value::Null } else {
            json!({"class_id": hex(rc.super_class_id), "class_name": rc.super_class_name})
        },
        "instance_size": rc.instance_size,
        "instances": q.instance_count(key),
        "instance_fields": rc.instance_fields.iter()
            .map(|f| json!({"name": f.name, "type": f.field_type.java_name()})).collect::<Vec<_>>(),
        "static_fields": rc.static_fields.iter()
            .map(|f| json!({"name": f.name, "value": value_json(q, &f.value)})).collect::<Vec<_>>(),
    })))
}

fn class_info(state: &AppState, args: &Args) -> ToolResult {
    let q = &*state.query;
    let id = match args.opt_id("class_id")? {
        Some(id) => id,
        None => {
            let name = args
                .opt_str("class_name")?
                .ok_or_else(|| ToolError::new("Give `class_id` or `class_name`."))?;
            q.find_class_by_name(name).ok_or_else(|| {
                ToolError::new(format!(
                    "No class named `{name}`. Use find_classes to search."
                ))
            })?
        }
    };
    let data = class_json(q, id)?
        .ok_or_else(|| ToolError::new(format!("{} is not a class object.", hex(id))))?;
    let name = data["class_name"].as_str().unwrap_or("?").to_owned();
    Ok(ToolOutput::new(format!("Class {name}."), data))
}

fn instances_of_class(state: &AppState, args: &Args) -> ToolResult {
    let q = &*state.query;
    let key = class_key_arg(q, args)?;
    let page = args.page()?;
    let total = q.instance_count(key);
    let items: Vec<Value> = q
        .class_entries(key)
        .skip(page.offset)
        .take(page.limit)
        .map(|e| json!({"id": hex(e.object_id)}))
        .collect();
    let has_more = page.offset + items.len() < total;
    let mut data = paged(page, total, has_more, items);
    data["class_name"] = json!(q.key_name(key));
    Ok(ToolOutput::new(
        format!("{total} instances of {}.", q.key_name(key)),
        data,
    ))
}

fn object(state: &AppState, args: &Args) -> ToolResult {
    let q = &*state.query;
    let id = args.id("id")?;
    let record = q
        .object(id)?
        .ok_or_else(|| ToolError::new(format!("No object with id {} in this dump.", hex(id))))?;

    let mut data = match &record {
        SubRecord::InstanceDump(inst) => {
            let ri = q.resolve_instance(inst)?;
            let mut d = json!({
                "kind": "instance",
                "class_name": ri.class_name,
                "class_id": hex(ri.class_id),
                "shallow_bytes": inst.data.len(),
                "fields": ri.fields.iter()
                    .map(|f| json!({"name": f.name, "value": value_json(q, &f.value)}))
                    .collect::<Vec<_>>(),
            });
            if ri.class_name == "java.lang.String" {
                let s = q.string_of(inst)?;
                let text: String = s.chars().take(200).collect();
                d["string"] = json!({"length_chars": s.chars().count(), "preview": text});
            }
            d
        }
        SubRecord::ClassDump(cd) => class_json(q, cd.class_id)?.unwrap_or(Value::Null),
        SubRecord::ObjArrayDump(arr) => json!({
            "kind": "object_array",
            "type": type_name(q, id),
            "length": arr.num_elements,
            "hint": "Use array_contents to read elements.",
        }),
        SubRecord::PrimArrayDump(arr) => json!({
            "kind": "primitive_array",
            "type": type_name(q, id),
            "element_type": BasicType::name_of_code(arr.element_type),
            "length": arr.num_elements,
            "hint": "Use array_contents to read elements.",
        }),
        root => match ResolvedRoot::from_sub_record(q, root)? {
            Some(r) => root_json(&r),
            None => json!({"kind": "unknown"}),
        },
    };
    data["id"] = json!(hex(id));
    data["gc_root_kinds"] = json!(
        q.root_types_of(id)
            .iter()
            .map(|k| root_slug(*k))
            .collect::<Vec<_>>()
    );
    data["referrers"] = json!(q.refs_to(id, Page::first(0)).total);
    data["retained_bytes"] = json!(q.retained_size(id));
    data["dominator"] = match q.dominator_of(id) {
        Some(0) | None => Value::Null,
        Some(d) => json!(hex(d)),
    };
    let summary = format!(
        "{} {}",
        hex(id),
        data["class_name"]
            .as_str()
            .or(data["type"].as_str())
            .unwrap_or("object")
    );
    Ok(ToolOutput::new(summary, data))
}

fn object_string(state: &AppState, args: &Args) -> ToolResult {
    let q = &*state.query;
    let id = args.id("id")?;
    let max = args.usize_or("max_chars", 10_000, 1_000_000)?;
    let s = q.string(id)?.ok_or_else(|| {
        ToolError::new(format!(
            "{} is not a java.lang.String. Use `object` to see what it is.",
            hex(id)
        ))
    })?;
    let length = s.chars().count();
    let text: String = s.chars().take(max).collect();
    Ok(ToolOutput::new(
        format!("String of {length} characters."),
        json!({"id": hex(id), "length_chars": length, "truncated": length > max, "text": text}),
    ))
}

fn array_contents(state: &AppState, args: &Args) -> ToolResult {
    let q = &*state.query;
    let id = args.id("id")?;
    let page = args.page_with(100, 1000)?;
    match q.object(id)? {
        Some(SubRecord::PrimArrayDump(_)) => {
            let w = q
                .prim_array(id, page)?
                .ok_or_else(|| ToolError::new("Array vanished."))?;
            Ok(ToolOutput::new(
                format!(
                    "{} elements of {}[] starting at {}.",
                    w.elements.len(),
                    w.element_type.java_name(),
                    w.offset
                ),
                json!({
                    "id": hex(id),
                    "element_type": w.element_type.java_name(),
                    "total": w.total,
                    "offset": w.offset,
                    "has_more": w.has_more(),
                    "next_offset": if w.has_more() { Some(w.offset + w.elements.len()) } else { None },
                    "elements": elements_json(&w.elements),
                }),
            ))
        }
        Some(SubRecord::ObjArrayDump(arr)) => {
            let total = arr.num_elements as usize;
            let items: Vec<Value> = arr
                .elements()
                .enumerate()
                .skip(page.offset)
                .take(page.limit)
                .map(|(i, e)| {
                    if e == 0 {
                        json!({"index": i, "id": Value::Null})
                    } else {
                        json!({"index": i, "id": hex(e), "type": type_name(q, e)})
                    }
                })
                .collect();
            let has_more = page.offset + items.len() < total;
            Ok(ToolOutput::new(
                format!("{} of {total} elements.", items.len()),
                json!({
                    "id": hex(id), "total": total, "offset": page.offset, "has_more": has_more,
                    "next_offset": if has_more { Some(page.offset + items.len()) } else { None },
                    "elements": items,
                }),
            ))
        }
        Some(_) => Err(ToolError::new(format!("{} is not an array.", hex(id)))),
        None => Err(ToolError::new(format!("No object with id {}.", hex(id)))),
    }
}

fn references_to(state: &AppState, args: &Args) -> ToolResult {
    let q = &*state.query;
    let id = args.id("id")?;
    let page = args.page()?;
    let r = q.refs_to(id, page);
    let items: Vec<Value> = r
        .items
        .iter()
        .map(|&from| json!({"id": hex(from), "type": type_name(q, from)}))
        .collect();
    Ok(ToolOutput::new(
        format!("{} reference(s) to {}.", r.total, hex(id)),
        paged(page, r.total, r.has_more, items),
    ))
}

fn references_from(state: &AppState, args: &Args) -> ToolResult {
    let q = &*state.query;
    let id = args.id("id")?;
    let page = args.page()?;
    let r = q.refs_from(id, page)?;
    let items: Vec<Value> = r
        .items
        .iter()
        .map(|x| json!({"id": hex(x.target), "type": type_name(q, x.target), "via": x.via.to_string()}))
        .collect();
    Ok(ToolOutput::new(
        format!("{} reference(s) from {}.", r.total, hex(id)),
        paged(page, r.total, r.has_more, items),
    ))
}

fn path_to_gc_root(state: &AppState, args: &Args) -> ToolResult {
    let q = &*state.query;
    let id = args.id("id")?;
    if q.object(id)?.is_none() {
        return Err(ToolError::new(format!("No object with id {}.", hex(id))));
    }
    let limits = RootPathLimits {
        max_nodes: args.usize_or("max_nodes", RootPathLimits::default().max_nodes, 5_000_000)?,
        ..RootPathLimits::default()
    };
    let p = q.path_to_root(id, &limits);
    let outcome = match p.outcome {
        PathOutcome::Found => "found",
        PathOutcome::LimitReached => "limit_reached",
        PathOutcome::NotReachable => "not_reachable",
    };
    let steps: Vec<Value> = p
        .steps
        .iter()
        .map(|s| {
            json!({
                "id": hex(s.object_id),
                "type": type_name(q, s.object_id),
                "via": s.via.as_ref().map(|e| e.to_string()),
            })
        })
        .collect();
    let summary = match p.outcome {
        PathOutcome::Found => format!(
            "{} hops from a GC root to {}.",
            p.steps.len().saturating_sub(1),
            hex(id)
        ),
        PathOutcome::LimitReached => format!(
            "Search stopped after {} objects; raise `max_nodes` to continue.",
            p.nodes_visited
        ),
        PathOutcome::NotReachable => format!("{} is not reachable from any GC root.", hex(id)),
    };
    Ok(ToolOutput::new(
        summary,
        json!({
            "outcome": outcome,
            "nodes_visited": p.nodes_visited,
            "root_kinds": p.root_kinds.iter().map(|k| root_slug(*k)).collect::<Vec<_>>(),
            "steps": steps,
        }),
    ))
}

fn gc_roots(state: &AppState, args: &Args) -> ToolResult {
    let q = &*state.query;
    let Some(kind) = args.opt_str("kind")? else {
        let counts: serde_json::Map<String, Value> = GcRootType::ALL
            .iter()
            .map(|&k| (root_slug(k).to_owned(), json!(q.iter_roots(k).len())))
            .collect();
        return Ok(ToolOutput::new(
            "GC root counts by kind.",
            json!({"counts": counts}),
        ));
    };
    let kind = GcRootType::ALL
        .into_iter()
        .find(|&k| root_slug(k) == kind)
        .ok_or_else(|| {
            ToolError::new(format!(
                "Unknown root kind `{kind}`. One of: {}.",
                root_slugs().join(", ")
            ))
        })?;
    let page = args.page()?;
    let r = q.gc_roots(kind, page)?;
    let items: Vec<Value> = r.items.iter().map(root_json).collect();
    Ok(ToolOutput::new(
        format!("{} {} root(s).", r.total, root_slug(kind)),
        paged(page, r.total, r.has_more, items),
    ))
}

fn threads(state: &AppState, args: &Args) -> ToolResult {
    let name = args.opt_str("name")?.filter(|n| !n.is_empty());
    let list = match name {
        Some(n) => {
            let matcher = search_matcher(args, n, &SearchMode::ALL, SearchMode::Contains)?;
            state.query.threads_matching(&matcher)?
        }
        None => state.query.threads()?,
    };
    let items: Vec<Value> = list
        .iter()
        .map(|t| {
            json!({
                "serial": t.serial,
                "name": t.name,
                "group": t.group,
                "object_id": t.object_id.map(hex),
                "ended": t.ended,
                "has_stack": t.stack_trace_serial.is_some(),
            })
        })
        .collect();
    let summary = match name {
        Some(n) => format!("{} thread(s) match `{n}`.", list.len()),
        None => format!("{} threads.", list.len()),
    };
    Ok(ToolOutput::new(
        summary,
        json!({"total": list.len(), "items": items}),
    ))
}

fn thread_stack(state: &AppState, args: &Args) -> ToolResult {
    let q = &*state.query;
    let serial = args.usize_or("serial", usize::MAX, u32::MAX as usize)?;
    if serial == usize::MAX {
        return Err(ToolError::new("Missing required argument `serial`."));
    }
    let serial = serial as u32;
    let thread = q.thread(serial)?.ok_or_else(|| {
        ToolError::new(format!(
            "No thread with serial {serial}. Use `threads` to list them."
        ))
    })?;
    let frames = q.thread_stack(&thread)?;
    let items: Vec<Value> = frames
        .iter()
        .map(|f| {
            json!({
                "method": f.method,
                "signature": f.signature,
                "location": f.location(),
                "class_serial": f.class_serial,
            })
        })
        .collect();
    Ok(ToolOutput::new(
        format!(
            "Thread {} \"{}\": {} frames.",
            thread.serial,
            thread.name,
            frames.len()
        ),
        json!({
            "serial": thread.serial, "name": thread.name, "ended": thread.ended,
            "object_id": thread.object_id.map(hex), "frames": items,
        }),
    ))
}

fn largest_arrays(state: &AppState, args: &Args) -> ToolResult {
    let q = &*state.query;
    let slug = args.str("kind")?;
    let kind = ArrayKind::from_slug(slug).ok_or_else(|| {
        ToolError::new(format!(
            "Unknown array kind `{slug}`. One of: {}.",
            array_slugs().join(", ")
        ))
    })?;
    let page = args.page()?;
    let total = q.array_count(kind);
    let elem = kind.elem_size(q.id_size());
    let items: Vec<Value> = q
        .iter_arrays_by_size(kind)
        .skip(page.offset)
        .take(page.limit)
        .map(|e| json!({"id": hex(e.object_id), "elements": e.byte_size.checked_div(elem).unwrap_or(0), "bytes": e.byte_size}))
        .collect();
    let has_more = page.offset + items.len() < total;
    Ok(ToolOutput::new(
        format!("{total} {slug} arrays, largest first."),
        paged(page, total, has_more, items),
    ))
}

fn retained_top(state: &AppState, args: &Args) -> ToolResult {
    let q = &*state.query;
    let page = args.page()?;
    let r = q
        .retained_top(page)
        .ok_or(crate::hprof::HprofError::NotIndexed("retained heap"))?;
    let items: Vec<Value> = r
        .items
        .iter()
        .map(|&(id, bytes)| {
            json!({
                "id": hex(id), "type": type_name(q, id), "retained_bytes": bytes,
                "dominator": match q.dominator_of(id) { Some(0) | None => Value::Null, Some(d) => json!(hex(d)) },
            })
        })
        .collect();
    Ok(ToolOutput::new(
        format!("{} objects have retained sizes; largest first.", r.total),
        paged(page, r.total, r.has_more, items),
    ))
}

fn dominated_by(state: &AppState, args: &Args) -> ToolResult {
    let q = &*state.query;
    let id = args.id("id")?;
    let page = args.page()?;
    let r = q
        .dominated_by(id, page)
        .ok_or(crate::hprof::HprofError::NotIndexed("retained heap"))?;
    let items: Vec<Value> = r
        .items
        .iter()
        .map(|&(child, bytes)| json!({"id": hex(child), "type": type_name(q, child), "retained_bytes": bytes}))
        .collect();
    Ok(ToolOutput::new(
        format!("{} object(s) directly dominated by {}.", r.total, hex(id)),
        paged(page, r.total, r.has_more, items),
    ))
}

fn search_strings(state: &AppState, args: &Args) -> ToolResult {
    let q = &*state.query;
    let needle = args.str("text")?;
    let window = ScanWindow::new(
        args.usize_or("cursor", 0, usize::MAX)?,
        args.usize_or(
            "max_scan",
            ScanWindow::DEFAULT_MAX_SCAN,
            ScanWindow::MAX_SCAN_LIMIT,
        )?,
        args.usize_or(
            "max_results",
            ScanWindow::DEFAULT_MAX_RESULTS,
            ScanWindow::MAX_RESULTS_LIMIT,
        )?,
    );
    // `ignore_case` is the older spelling of `case_sensitive`, inverted; an
    // explicit `case_sensitive` wins, and the default is to ignore case.
    let mode = opt_mode(args, &STRING_MODES)?.unwrap_or(SearchMode::Contains);
    let case_sensitive = match (
        args.opt_bool("case_sensitive")?,
        args.opt_bool("ignore_case")?,
    ) {
        (Some(cs), _) => cs,
        (None, Some(ignore)) => !ignore,
        (None, None) => false,
    };
    let matcher = Matcher::new(&SearchQuery::new(needle, mode).case_sensitive(case_sensitive))?;
    let r = q.search_strings(&matcher, window)?;
    if r.total == 0 && r.scanned == 0 && window.cursor == 0 {
        return Ok(ToolOutput::new(
            "No java.lang.String objects in this dump.",
            json!({"matches": [], "next_cursor": null}),
        ));
    }
    let matches: Vec<Value> = r
        .items
        .iter()
        .map(|s| json!({"id": hex(s.object_id), "length_chars": s.length_chars, "text": s.preview}))
        .collect();
    let next = window.cursor + r.scanned;
    Ok(ToolOutput::new(
        format!(
            "{} match(es) in strings {}..{next} of {}.{}",
            matches.len(),
            window.cursor,
            r.total,
            if r.next_cursor.is_some() {
                " More to scan: pass next_cursor."
            } else {
                ""
            }
        ),
        json!({"matches": matches, "scanned": r.scanned, "total_strings": r.total, "next_cursor": r.next_cursor}),
    ))
}

fn require_diff(state: &AppState) -> Result<&crate::diff::HeapDiff, ToolError> {
    state.diff.as_ref().map(|d| &d.heap).ok_or_else(|| {
        ToolError::new("No second heap dump is configured. Start the server with `--diff-hprof <path>` to compare two dumps.")
    })
}

fn heap_diff_summary(state: &AppState, args: &Args) -> ToolResult {
    let diff = require_diff(state)?;
    let page = args.page()?;
    let s = diff.summary()?;
    let total = s.by_class.len();
    let items: Vec<Value> = s
        .by_class
        .iter()
        .skip(page.offset)
        .take(page.limit)
        .map(|c| {
            json!({
                "class_name": c.class_name, "before": c.count_before(), "after": c.count_after(),
                "net": c.net_change(), "added": c.count_added, "removed": c.count_removed,
                "unchanged": c.count_common_unchanged, "changed": c.count_common_changed,
            })
        })
        .collect();
    let has_more = page.offset + items.len() < total;
    let mut data = paged(page, total, has_more, items);
    data["totals"] = json!({
        "before": s.total_before, "after": s.total_after, "added": s.total_added,
        "removed": s.total_removed, "unchanged": s.total_common_unchanged,
        "changed": s.total_common_changed,
    });
    Ok(ToolOutput::new(
        format!(
            "{} objects before, {} after: +{} added, -{} removed.",
            s.total_before, s.total_after, s.total_added, s.total_removed
        ),
        data,
    ))
}

fn heap_diff_objects(state: &AppState, args: &Args) -> ToolResult {
    let diff = require_diff(state)?;
    let page = args.page()?;
    let kind = args.str("kind")?;
    let class = match args.opt_str("class_name")? {
        None => None,
        Some(name) => Some(
            key_for_name(diff.before(), name)
                .or(key_for_name(diff.after(), name))
                .ok_or_else(|| {
                    ToolError::new(format!("No class named `{name}` in either dump."))
                })?,
        ),
    };
    let r = match kind {
        "added" => diff.added(class, page)?,
        "removed" => diff.removed(class, page)?,
        "changed" => diff.changed(class, page)?,
        "unchanged" => diff.common(Some(false), class, page)?,
        other => {
            return Err(ToolError::new(format!(
                "`kind` must be added, removed, changed or unchanged, not `{other}`."
            )));
        }
    };
    let items: Vec<Value> = r
        .items
        .iter()
        .map(|o| json!({"id": hex(o.object_id), "class_name": o.class_name}))
        .collect();
    let dump = if kind == "added" { "second" } else { "first" };
    Ok(ToolOutput::new(
        format!(
            "{} {kind} object(s); ids refer to the {dump} dump.",
            r.total
        ),
        paged(page, r.total, r.has_more, items),
    ))
}
