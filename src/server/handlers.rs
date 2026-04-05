//! HTTP request handlers — one function per route.
//!
//! All handlers follow the same pattern:
//! 1. Clone the shared `Arc<AppState>`.
//! 2. Move it into `tokio::task::spawn_blocking` so mmap I/O doesn't block the
//!    async runtime.
//! 3. Build an owned HTML string inside the blocking task.
//! 4. Return `Html<String>` on success or `(StatusCode, String)` on error.
//!
//! No heap-dump data (parsed `SubRecord` values, borrowed slices, etc.) is
//! stored in any `Arc` or returned across the spawn boundary — every handler
//! converts results to owned `String`s before returning.

use super::{
    AppState, SYNTHETIC_OBJ_ARRAY, SYNTHETIC_PRIM_ARRAY, class_link, esc, fmt_bytes,
    instances_key_for_class, is_synthetic, obj_link, page, parse_hex_id, prim_desc_to_elem_type,
    prim_type_name,
};
use crate::array_index::ArrayKind;
use crate::diff_index::{CommonEntryReader, DiffEntry, DiffEntryReader};
use crate::heap_index::sub_record::SubIndexEntry;
use crate::heap_parser::FieldValue;
use crate::heap_parser::SubRecord;
use crate::hprof::HprofError;
use crate::query::{ROOT_PATH_SEARCH_LIMIT, RootPathResult};
use crate::root_index::GcRootType;
use crate::vfs::MMapReader;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{Html, IntoResponse},
};
use std::sync::Arc;

// ── Handler return type ───────────────────────────────────────────────────────

type Resp = (StatusCode, Html<String>);

fn ok(html: String) -> Resp {
    (StatusCode::OK, Html(html))
}

fn err(status: StatusCode, msg: impl std::fmt::Display) -> Resp {
    (
        status,
        Html(page(
            "Error",
            &format!("<p class=\"muted\">{}</p>", esc(&msg.to_string())),
        )),
    )
}

fn internal(e: impl std::fmt::Display) -> Resp {
    err(StatusCode::INTERNAL_SERVER_ERROR, e)
}

fn not_found(msg: impl std::fmt::Display) -> Resp {
    err(StatusCode::NOT_FOUND, msg)
}

fn bad_request(msg: impl std::fmt::Display) -> Resp {
    err(StatusCode::BAD_REQUEST, msg)
}

// ── Pagination params ─────────────────────────────────────────────────────────

#[derive(serde::Deserialize, Default)]
pub struct PageParams {
    #[serde(default)]
    pub offset: usize,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    200
}

/// Query params for the diff list pages (`/diff/removed`, `/diff/added`, `/diff/common`).
#[derive(serde::Deserialize, Default)]
pub struct DiffListParams {
    /// Hex class ID to filter by (optional).
    pub class: Option<String>,
    /// For `/diff/common`: `"0"` = unchanged only, `"1"` = changed only, absent = all.
    pub changed: Option<String>,
    #[serde(default)]
    pub offset: usize,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

// ── / — Summary ───────────────────────────────────────────────────────────────

pub async fn summary(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(move || -> Result<String, HprofError> {
        let q = &state.query;
        let object_count = q.object_count();
        let id_size = q.id_size();
        let hprof = state.hprof_path.display().to_string();

        let thread_count = q.iter_threads().count();
        let trace_count = q.iter_traces().count();
        let frame_count = q.iter_frames().count();

        let mut root_counts = String::new();
        for rt in GcRootType::ALL {
            let n = q.iter_roots(rt).count();
            if n > 0 {
                root_counts.push_str(&format!(
                    "<tr><td>{}</td><td class=\"num\">{n}</td></tr>",
                    esc(gc_root_label(rt))
                ));
            }
        }

        let content = format!(
            r#"<table>
<tr><th>Property</th><th>Value</th></tr>
<tr><td>Heap dump file</td><td>{hprof}</td></tr>
<tr><td>Object ID size</td><td>{id_size} bytes</td></tr>
<tr><td>Total sub-records</td><td class="num">{object_count}</td></tr>
<tr><td>Threads</td><td class="num">{thread_count}</td></tr>
<tr><td>Stack traces</td><td class="num">{trace_count}</td></tr>
<tr><td>Stack frames</td><td class="num">{frame_count}</td></tr>
</table>
<h2>GC Root Counts</h2>
<table><tr><th>Root type</th><th>Count</th></tr>{root_counts}</table>
<h2>Quick Links</h2>
<ul>
<li><a href="/histogram">Class instance histogram</a></li>
<li><a href="/allClasses">All classes</a></li>
<li><a href="/roots">GC roots</a></li>
<li><a href="/threads">Threads and stack traces</a></li>
</ul>"#
        );
        Ok(page("Heap Summary", &content))
    })
    .await;

    match result {
        Ok(Ok(html)) => ok(html).into_response(),
        Ok(Err(e)) => internal(e).into_response(),
        Err(e) => internal(e).into_response(),
    }
}

// ── /histogram — Class histogram ──────────────────────────────────────────────

pub async fn histogram(
    State(state): State<Arc<AppState>>,
    Query(p): Query<PageParams>,
) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(move || -> Result<String, HprofError> {
        let hist = state.histogram()?;
        let total: u64 = hist.iter().map(|e| e.instance_count).sum();
        let total_bytes: u64 = hist.iter().map(|e| e.shallow_bytes).sum();
        let page_entries = hist.iter().skip(p.offset).take(p.limit);

        let mut rows = String::new();
        for entry in page_entries {
            let id_hex = format!("{:x}", entry.class_id);
            // Primitive array types link directly to the size-sorted /arrays/:kind view.
            let link = if (entry.class_id & SYNTHETIC_PRIM_ARRAY) == SYNTHETIC_PRIM_ARRAY {
                let et = (entry.class_id & 0xFF) as u8;
                let slug = ArrayKind::from_prim_element_type(et)
                    .map(|k| k.slug())
                    .unwrap_or("unknown");
                format!("<a href=\"/arrays/{slug}\">{}</a>", esc(&entry.class_name))
            } else {
                format!(
                    "<a href=\"/instances/{id_hex}\">{}</a>",
                    esc(&entry.class_name)
                )
            };
            rows.push_str(&format!(
                "<tr><td>{link}</td><td class=\"num\">{}</td><td class=\"num\">{}</td></tr>",
                entry.instance_count,
                fmt_bytes(entry.shallow_bytes)
            ));
        }

        let note = if hist.len() > p.offset + p.limit {
            format!(
                "<p><a href=\"/histogram?offset={}&limit={}\">Next {} →</a></p>",
                p.offset + p.limit,
                p.limit,
                p.limit
            )
        } else {
            String::new()
        };

        let prev = if p.offset > 0 {
            format!(
                "<p><a href=\"/histogram?offset={}&limit={}\">← Prev {}</a></p>",
                p.offset.saturating_sub(p.limit),
                p.limit,
                p.limit
            )
        } else {
            String::new()
        };

        let warn = if hist.is_empty() {
            "<p class=\"muted\">Histogram is still being computed — refresh in a moment.</p>"
                .to_owned()
        } else {
            String::new()
        };

        let content = format!(
            r#"{warn}
<p>{} classes &nbsp;·&nbsp; {total} total instances &nbsp;·&nbsp; {} total shallow bytes</p>
{prev}
<table>
<tr><th>Class</th><th>Instances</th><th>Shallow size</th></tr>
{rows}
</table>
{note}"#,
            hist.len(),
            fmt_bytes(total_bytes)
        );
        Ok(page("Histogram", &content))
    })
    .await;

    match result {
        Ok(Ok(html)) => ok(html).into_response(),
        Ok(Err(e)) => internal(e).into_response(),
        Err(e) => internal(e).into_response(),
    }
}

// ── /allClasses — Class list ──────────────────────────────────────────────────

pub async fn all_classes(
    State(state): State<Arc<AppState>>,
    Query(p): Query<PageParams>,
) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(move || -> Result<String, HprofError> {
        let hist = state.histogram()?;
        // Collect only real (non-synthetic) entries, sorted by name.
        let mut entries: Vec<_> = hist.iter().filter(|e| !is_synthetic(e.class_id)).collect();
        entries.sort_by(|a, b| a.class_name.cmp(&b.class_name));

        // Build a map from synthetic key → instance_count for O(1) array-count lookup.
        let synthetic_counts: std::collections::HashMap<u64, u64> = hist
            .iter()
            .filter(|e| is_synthetic(e.class_id))
            .map(|e| (e.class_id, e.instance_count))
            .collect();

        let mut rows = String::new();
        for entry in entries.iter().skip(p.offset).take(p.limit) {
            // For primitive-array class objects (e.g. "class [B"), show the count of
            // actual array instances from the corresponding synthetic histogram entry.
            let display_count = entry
                .class_name
                .strip_prefix("class [")
                .and_then(|rest| {
                    if rest.len() == 1 {
                        let elem_type = rest.chars().next().and_then(prim_desc_to_elem_type)?;
                        let key = SYNTHETIC_PRIM_ARRAY | u64::from(elem_type);
                        Some(*synthetic_counts.get(&key).unwrap_or(&0))
                    } else {
                        None
                    }
                })
                .unwrap_or(entry.instance_count);
            rows.push_str(&format!(
                "<tr><td>{}</td><td class=\"num\">{display_count}</td></tr>",
                class_link(entry.class_id, &entry.class_name),
            ));
        }

        let note = if entries.len() > p.offset + p.limit {
            format!(
                "<p><a href=\"/allClasses?offset={}&limit={}\">Next {} →</a></p>",
                p.offset + p.limit,
                p.limit,
                p.limit
            )
        } else {
            String::new()
        };

        let prev = if p.offset > 0 {
            format!(
                "<p><a href=\"/allClasses?offset={}&limit={}\">← Prev {}</a></p>",
                p.offset.saturating_sub(p.limit),
                p.limit,
                p.limit
            )
        } else {
            String::new()
        };

        let content = format!(
            r#"<p>{} classes</p>{prev}
<table>
<tr><th>Class name</th><th>Instance count</th></tr>
{rows}
</table>{note}"#,
            entries.len()
        );
        Ok(page("All Classes", &content))
    })
    .await;

    match result {
        Ok(Ok(html)) => ok(html).into_response(),
        Ok(Err(e)) => internal(e).into_response(),
        Err(e) => internal(e).into_response(),
    }
}

// ── /class/{id} — Class detail ────────────────────────────────────────────────

pub async fn class_detail(
    State(state): State<Arc<AppState>>,
    Path(id_str): Path<String>,
) -> impl IntoResponse {
    let class_id = match parse_hex_id(&id_str) {
        Some(id) => id,
        None => return bad_request(format!("Invalid class ID: {id_str}")).into_response(),
    };

    let result = tokio::task::spawn_blocking(move || -> Result<String, HprofError> {
        let q = &state.query;
        let record = match q.find_class(class_id)? {
            Some(r) => r,
            None => {
                return Ok(page(
                    "Class not found",
                    "<p>No class dump with that ID.</p>",
                ));
            }
        };

        let SubRecord::ClassDump(cd) = record else {
            return Ok(page("Not a class", "<p>Object is not a CLASS_DUMP.</p>"));
        };

        let class_name = q
            .class_name(cd.class_id)
            .ok()
            .flatten()
            .unwrap_or_else(|| format!("0x{:x}", cd.class_id));
        let super_name = if cd.super_class_id == 0 {
            "<span class=\"muted\">(none)</span>".to_owned()
        } else {
            let n = q
                .class_name(cd.super_class_id)
                .ok()
                .flatten()
                .unwrap_or_else(|| format!("0x{:x}", cd.super_class_id));
            class_link(cd.super_class_id, &n)
        };

        let mut statics = String::new();
        for sf in cd.static_fields() {
            let sf = sf?;
            let name = q
                .lookup_name(sf.name_id)
                .ok()
                .flatten()
                .unwrap_or_else(|| format!("?{:x}", sf.name_id));
            let val = render_field_value(&sf.value, q);
            statics.push_str(&format!("<tr><td>{}</td><td>{val}</td></tr>", esc(&name)));
        }

        let mut inst_fields = String::new();
        for fd in cd.instance_fields() {
            let fd = fd?;
            let name = q
                .lookup_name(fd.name_id)
                .ok()
                .flatten()
                .unwrap_or_else(|| format!("?{:x}", fd.name_id));
            let type_name = field_type_name(fd.field_type);
            inst_fields.push_str(&format!(
                "<tr><td>{}</td><td>{type_name}</td></tr>",
                esc(&name)
            ));
        }

        let statics_section = if statics.is_empty() {
            "<p class=\"muted\">No static fields.</p>".to_owned()
        } else {
            format!("<table><tr><th>Static field</th><th>Value</th></tr>{statics}</table>")
        };

        let inst_fields_section = if inst_fields.is_empty() {
            "<p class=\"muted\">No instance fields.</p>".to_owned()
        } else {
            format!("<table><tr><th>Instance field</th><th>Type</th></tr>{inst_fields}</table>")
        };

        let instances_id = instances_key_for_class(&class_name, cd.class_id, q);
        let content = format!(
            r#"<table>
<tr><td>Object ID</td><td>{}</td></tr>
<tr><td>Class name</td><td>{}</td></tr>
<tr><td>Superclass</td><td>{super_name}</td></tr>
<tr><td>Instance size</td><td>{} bytes</td></tr>
</table>
<p><a href="/instances/{instances_id:x}">Show instances</a></p>
<h2>Static fields</h2>
{statics_section}
<h2>Instance field layout</h2>
{inst_fields_section}"#,
            obj_link(cd.class_id),
            esc(&class_name),
            cd.instance_size,
        );
        Ok(page(&format!("Class: {class_name}"), &content))
    })
    .await;

    match result {
        Ok(Ok(html)) => ok(html).into_response(),
        Ok(Err(e)) => internal(e).into_response(),
        Err(e) => internal(e).into_response(),
    }
}

// ── /instances/{id} — Instances of a class ───────────────────────────────────

pub async fn instances_of_class(
    State(state): State<Arc<AppState>>,
    Path(id_str): Path<String>,
    Query(p): Query<PageParams>,
) -> impl IntoResponse {
    let target_id = match parse_hex_id(&id_str) {
        Some(id) => id,
        None => {
            return bad_request(format!("Invalid ID: {id_str}")).into_response();
        }
    };

    let result = tokio::task::spawn_blocking(move || -> Result<String, HprofError> {
        let q = &state.query;

        // Determine display title
        let title = if is_synthetic(target_id) {
            // Synthetic ID: derive name from histogram cache if available
            state
                .histogram_cache
                .lock()
                .ok()
                .and_then(|g| {
                    g.as_ref().and_then(|h| {
                        h.iter()
                            .find(|e| e.class_id == target_id)
                            .map(|e| e.class_name.clone())
                    })
                })
                .unwrap_or_else(|| "array type".to_owned())
        } else {
            q.class_name(target_id)
                .ok()
                .flatten()
                .unwrap_or_else(|| format!("0x{target_id:x}"))
        };

        let is_prim_array = (target_id & SYNTHETIC_PRIM_ARRAY) == SYNTHETIC_PRIM_ARRAY;
        let is_obj_array = (target_id & SYNTHETIC_OBJ_ARRAY) != 0 && !is_prim_array;
        let elem_class_id = target_id & !SYNTHETIC_OBJ_ARRAY;

        // ── Primitive arrays: use size-sorted index directly ──────────────────
        if is_prim_array {
            let prim_type = (target_id & 0xFF) as u8;
            let kind = match ArrayKind::from_prim_element_type(prim_type) {
                Some(k) => k,
                None => return Ok(page(&format!("Instances of {title}"), "<p>Unknown primitive type.</p>")),
            };
            let total = q.array_count(kind);
            let id_size = q.id_size();
            let elem_size = kind.elem_size(id_size);
            let offset = p.offset.min(total);
            let shown = p.limit.min(total.saturating_sub(offset));

            let mut rows = String::new();
            for entry in q.iter_arrays_by_size(kind).skip(offset).take(shown) {
                let num_elements = if elem_size > 0 { entry.byte_size / elem_size } else { 0 };
                rows.push_str(&format!(
                    "<tr><td>{}</td><td class=\"num\">{num_elements}</td><td class=\"num\">{}</td></tr>",
                    obj_link(entry.object_id),
                    fmt_bytes(entry.byte_size),
                ));
            }

            let note = if offset + shown < total {
                format!(
                    "<p><a href=\"/instances/{id_str}?offset={}&limit={}\">Next {} →</a></p>",
                    offset + p.limit, p.limit, p.limit
                )
            } else { String::new() };
            let prev = if offset > 0 {
                format!(
                    "<p><a href=\"/instances/{id_str}?offset={}&limit={}\">← Prev {}</a></p>",
                    offset.saturating_sub(p.limit), p.limit, p.limit
                )
            } else { String::new() };

            let content = format!(
                "<p>{total} arrays total, showing {offset}–{} by size (largest first)</p>\
                 {prev}\
                 <table><tr><th>Array</th><th>Elements</th><th>Size</th></tr>{rows}</table>\
                 {note}",
                offset + shown
            );
            return Ok(page(&format!("Instances of {title}"), &content));
        }

        // ── Object arrays: full scan, collect all, sort by size ───────────────
        if is_obj_array {
            // Collect (array_id, num_elements) for every matching object array.
            let mut all_matches: Vec<(u64, u32)> = Vec::new();
            for result in q.iter_objects() {
                if let SubRecord::ObjArrayDump(a) = result?
                    && a.element_class_id == elem_class_id
                {
                    all_matches.push((a.array_id, a.num_elements));
                }
            }
            let total = all_matches.len();
            // Sort descending by element count (proxy for size, since id_size is fixed).
            all_matches.sort_unstable_by(|a, b| b.1.cmp(&a.1));

            let id_size = q.id_size() as u64;
            let offset = p.offset.min(total);
            let shown = p.limit.min(total.saturating_sub(offset));

            let mut rows = String::new();
            for (array_id, num_elements) in all_matches.into_iter().skip(offset).take(shown) {
                let byte_size = u64::from(num_elements) * id_size;
                rows.push_str(&format!(
                    "<tr><td>{}</td><td class=\"num\">{num_elements}</td><td class=\"num\">{}</td></tr>",
                    obj_link(array_id),
                    fmt_bytes(byte_size),
                ));
            }

            let note = if offset + shown < total {
                format!(
                    "<p><a href=\"/instances/{id_str}?offset={}&limit={}\">Next {} →</a></p>",
                    offset + p.limit, p.limit, p.limit
                )
            } else { String::new() };
            let prev = if offset > 0 {
                format!(
                    "<p><a href=\"/instances/{id_str}?offset={}&limit={}\">← Prev {}</a></p>",
                    offset.saturating_sub(p.limit), p.limit, p.limit
                )
            } else { String::new() };

            let content = format!(
                "<p>{total} arrays total, showing {offset}–{} by size (largest first)</p>\
                 {prev}\
                 <table><tr><th>Array</th><th>Elements</th><th>Size</th></tr>{rows}</table>\
                 {note}",
                offset + shown
            );
            return Ok(page(&format!("Instances of {title}"), &content));
        }

        // ── Regular instances: full scan, no size ordering ────────────────────
        let mut matching: Vec<u64> = Vec::new();
        let mut total = 0usize;

        for result in q.iter_objects() {
            if let SubRecord::InstanceDump(i) = result?
                && i.class_id == target_id
            {
                total += 1;
                if total > p.offset && matching.len() < p.limit {
                    matching.push(i.object_id);
                }
            }
        }

        let mut rows = String::new();
        for id in &matching {
            rows.push_str(&format!("<tr><td>{}</td></tr>", obj_link(*id)));
        }

        let note = if total > p.offset + p.limit {
            format!(
                "<p><a href=\"/instances/{id_str}?offset={}&limit={}\">Next {} →</a></p>",
                p.offset + p.limit, p.limit, p.limit
            )
        } else { String::new() };
        let prev = if p.offset > 0 {
            format!(
                "<p><a href=\"/instances/{id_str}?offset={}&limit={}\">← Prev {}</a></p>",
                p.offset.saturating_sub(p.limit), p.limit, p.limit
            )
        } else { String::new() };

        let content = format!(
            "<p>Total matching: {total} &nbsp; showing {}-{}</p>\
             {prev}\
             <table><tr><th>Object ID</th></tr>{rows}</table>\
             {note}",
            p.offset + 1,
            p.offset + matching.len()
        );
        Ok(page(&format!("Instances of {title}"), &content))
    })
    .await;

    match result {
        Ok(Ok(html)) => ok(html).into_response(),
        Ok(Err(e)) => internal(e).into_response(),
        Err(e) => internal(e).into_response(),
    }
}

// ── /object/{id} — Object detail ─────────────────────────────────────────────

pub async fn object_detail(
    State(state): State<Arc<AppState>>,
    Path(id_str): Path<String>,
) -> impl IntoResponse {
    let object_id = match parse_hex_id(&id_str) {
        Some(id) => id,
        None => return bad_request(format!("Invalid object ID: {id_str}")).into_response(),
    };

    let result = tokio::task::spawn_blocking(move || -> Result<String, HprofError> {
        let reuse_warning = if let Some(dq) = state.diff_query.as_ref() {
            match (state.query.find(object_id)?, dq.find(object_id)?) {
                (Some(r1), Some(r2)) if object_fingerprint(&r1) != object_fingerprint(&r2) => {
                    ADDRESS_REUSE_WARNING
                }
                _ => "",
            }
        } else {
            ""
        };
        let html = render_object_page(object_id, &state.query, reuse_warning)?;
        Ok(html)
    })
    .await;

    match result {
        Ok(Ok(html)) => ok(html).into_response(),
        Ok(Err(e)) => internal(e).into_response(),
        Err(e) => internal(e).into_response(),
    }
}

/// Render a full HTML page for `object_id` using the given query.
///
/// `header_html` is inserted before the object body (e.g. a notice banner for
/// diff-dump objects).  Pass `""` when not needed.
fn render_object_page(
    object_id: u64,
    q: &crate::query::HeapQuery,
    header_html: &str,
) -> Result<String, HprofError> {
    let record = match q.find(object_id)? {
        Some(r) => r,
        None => return Ok(page("Object not found", "<p>No object with that ID.</p>")),
    };

    let refs_to = q.refs_to(object_id);
    let root_types = q.root_types_of(object_id);

    let mut refs_html = String::new();
    for from_id in &refs_to {
        let type_name = q
            .object_type_name(*from_id)
            .unwrap_or_else(|_| "?".to_owned());
        refs_html.push_str(&format!(
            "<li>{} <span class=\"muted\">{}</span></li>",
            obj_link(*from_id),
            esc(&type_name)
        ));
    }
    let refs_section = if refs_html.is_empty() {
        "<p class=\"muted\">No back-references found.</p>".to_owned()
    } else {
        format!("<ul>{refs_html}</ul>")
    };

    let root_html = if root_types.is_empty() {
        format!(
            "<p class=\"muted\">Not a GC root.</p>\
             <p><a href=\"/object/{object_id:x}/root-path\">Show path to root \u{2192}</a></p>"
        )
    } else {
        let names: Vec<_> = root_types
            .iter()
            .map(|rt| {
                format!(
                    "<a href=\"/roots/{}\">{}</a>",
                    gc_root_slug(*rt),
                    gc_root_label(*rt)
                )
            })
            .collect();
        format!("<p>{}</p>", names.join(", "))
    };

    let body = match record {
        SubRecord::InstanceDump(inst) => {
            let class_name = q
                .class_name(inst.class_id)
                .ok()
                .flatten()
                .unwrap_or_else(|| format!("0x{:x}", inst.class_id));

            let resolved_value_html = match q.resolve_value(inst.object_id) {
                Ok(jv) => render_java_value_banner(&jv),
                Err(_) => String::new(),
            };

            let fields = q.instance_fields(&inst)?;
            let mut field_rows = String::new();
            for f in &fields {
                let val = render_field_value(&f.value, q);
                let type_display = resolve_field_type(f.field_type, &f.value, q);
                field_rows.push_str(&format!(
                    "<tr><td>{}</td><td>{}</td><td>{val}</td></tr>",
                    esc(&f.name),
                    type_display
                ));
            }
            let fields_table = if field_rows.is_empty() {
                "<p class=\"muted\">No instance fields.</p>".to_owned()
            } else {
                format!(
                    "<table><tr><th>Field</th><th>Type</th><th>Value</th></tr>{field_rows}</table>"
                )
            };
            format!(
                r#"<table>
<tr><td>Type</td><td>instance</td></tr>
<tr><td>Object ID</td><td>0x{:x}</td></tr>
<tr><td>Class</td><td>{}</td></tr>
<tr><td>Instance data</td><td>{} bytes</td></tr>
</table>
{resolved_value_html}
<h2>Fields</h2>
{fields_table}"#,
                inst.object_id,
                class_link(inst.class_id, &class_name),
                inst.data.len()
            )
        }
        SubRecord::ClassDump(cd) => {
            let name = q
                .class_name(cd.class_id)
                .ok()
                .flatten()
                .unwrap_or_else(|| format!("0x{:x}", cd.class_id));
            format!(
                r#"<table>
<tr><td>Type</td><td>class</td></tr>
<tr><td>Class ID</td><td>0x{:x}</td></tr>
<tr><td>Class name</td><td>{}</td></tr>
<tr><td>Instance size</td><td>{} bytes</td></tr>
</table>
<p><a href="/class/{:x}">View full class detail →</a></p>"#,
                cd.class_id,
                esc(&name),
                cd.instance_size,
                cd.class_id,
            )
        }
        SubRecord::ObjArrayDump(arr) => {
            let elem_name = q
                .class_name(arr.element_class_id)
                .ok()
                .flatten()
                .unwrap_or_else(|| format!("0x{:x}", arr.element_class_id));
            let mut elems = String::new();
            for (i, id) in arr.elements().enumerate().take(50) {
                if id == 0 {
                    elems.push_str(&format!("<li>[{i}] null</li>"));
                } else {
                    elems.push_str(&format!("<li>[{i}] {}</li>", obj_link(id)));
                }
            }
            let more = if arr.num_elements > 50 {
                format!(
                    "<li class=\"muted\">… {} more elements</li>",
                    arr.num_elements - 50
                )
            } else {
                String::new()
            };
            format!(
                r#"<table>
<tr><td>Type</td><td>object array</td></tr>
<tr><td>Array ID</td><td>0x{:x}</td></tr>
<tr><td>Element type</td><td>{}</td></tr>
<tr><td>Length</td><td>{}</td></tr>
</table>
<h2>Elements (first 50)</h2>
<ul>{elems}{more}</ul>"#,
                arr.array_id,
                class_link(arr.element_class_id, &elem_name),
                arr.num_elements,
            )
        }
        SubRecord::PrimArrayDump(arr) => {
            let tn = prim_type_name(arr.element_type);
            let preview = decode_prim_array(arr.element_type, arr.data, arr.num_elements);
            format!(
                r#"<table>
<tr><td>Type</td><td>primitive array</td></tr>
<tr><td>Array ID</td><td>0x{:x}</td></tr>
<tr><td>Element type</td><td>{tn}</td></tr>
<tr><td>Length</td><td>{}</td></tr>
</table>
<h2>Contents</h2>
<p><a href="/object/{:x}/raw-array">raw comma-delimited values</a></p>
<pre>{}</pre>"#,
                arr.array_id,
                arr.num_elements,
                arr.array_id,
                esc(&preview)
            )
        }
        SubRecord::RootUnknown(r) => format!(
            "<p>GC root: unknown<br>Object: {}</p>",
            obj_link(r.object_id)
        ),
        SubRecord::RootJniGlobal(r) => format!(
            "<p>GC root: JNI global<br>Object: {}<br>JNI ref: {}</p>",
            obj_link(r.object_id),
            obj_link(r.jni_global_ref_id)
        ),
        SubRecord::RootJniLocal(r) => format!(
            "<p>GC root: JNI local<br>Object: {}<br>Thread serial: {}<br>Frame: {}</p>",
            obj_link(r.object_id),
            r.thread_serial,
            r.frame_number
        ),
        SubRecord::RootJavaFrame(r) => format!(
            "<p>GC root: Java frame<br>Object: {}<br>Thread serial: {}<br>Frame: {}</p>",
            obj_link(r.object_id),
            r.thread_serial,
            r.frame_number
        ),
        SubRecord::RootNativeStack(r) => format!(
            "<p>GC root: native stack<br>Object: {}<br>Thread serial: {}</p>",
            obj_link(r.object_id),
            r.thread_serial
        ),
        SubRecord::RootStickyClass(r) => format!(
            "<p>GC root: sticky class<br>Class: {}</p>",
            obj_link(r.class_id)
        ),
        SubRecord::RootThreadBlock(r) => format!(
            "<p>GC root: thread block<br>Object: {}<br>Thread serial: {}</p>",
            obj_link(r.object_id),
            r.thread_serial
        ),
        SubRecord::RootMonitorUsed(r) => format!(
            "<p>GC root: monitor used<br>Object: {}</p>",
            obj_link(r.object_id)
        ),
        SubRecord::RootThreadObj(r) => format!(
            "<p>GC root: thread object<br>Object: {}<br>Thread serial: {}<br>Trace serial: {}</p>",
            obj_link(r.thread_object_id),
            r.thread_serial,
            r.stack_trace_serial
        ),
    };

    let retained_html = if q.has_retained_heap() {
        let retained = q
            .retained_size(object_id)
            .map(fmt_bytes)
            .unwrap_or_else(|| "<span class=\"muted\">not reachable</span>".to_owned());
        let dom_html = match q.dominator_of(object_id) {
            None => "<span class=\"muted\">not reachable</span>".to_owned(),
            Some(0) => "<span class=\"muted\">(GC root — virtual root)</span>".to_owned(),
            Some(dom_id) => {
                let dom_type = q
                    .object_type_name(dom_id)
                    .unwrap_or_else(|_| "?".to_owned());
                format!(
                    "{} <span class=\"muted\">{}</span>",
                    obj_link(dom_id),
                    esc(&dom_type)
                )
            }
        };
        format!(
            "<h2>Retained heap</h2>\
             <table>\
             <tr><td>Retained size</td><td class=\"num\">{retained}</td></tr>\
             <tr><td>Immediate dominator</td><td>{dom_html}</td></tr>\
             </table>"
        )
    } else {
        String::new()
    };

    let content = format!(
        "{header_html}{body}
{retained_html}
<h2>GC root status</h2>
{root_html}
<h2>Referenced by</h2>
{refs_section}"
    );
    Ok(page(&format!("Object 0x{object_id:x}"), &content))
}

// ── /object/:id/raw-string — raw string content ───────────────────────────────

pub async fn raw_string(
    State(state): State<Arc<AppState>>,
    Path(id_str): Path<String>,
) -> impl IntoResponse {
    let object_id = match parse_hex_id(&id_str) {
        Some(id) => id,
        None => {
            return (StatusCode::BAD_REQUEST, "Invalid object ID").into_response();
        }
    };

    let result = tokio::task::spawn_blocking(move || -> Result<Option<String>, HprofError> {
        let q = &state.query;
        match q.resolve_value(object_id)? {
            crate::heap_query::JavaValue::String(_, s) => Ok(Some(s)),
            _ => Ok(None),
        }
    })
    .await;

    match result {
        Ok(Ok(Some(s))) => (
            StatusCode::OK,
            [("content-type", "text/plain; charset=utf-8")],
            s,
        )
            .into_response(),
        Ok(Ok(None)) => (StatusCode::NOT_FOUND, "Not a String object").into_response(),
        Ok(Err(e)) => internal(e).into_response(),
        Err(e) => internal(e).into_response(),
    }
}

// ── /object/:id/raw-array — raw primitive array values ────────────────────────

pub async fn raw_prim_array(
    State(state): State<Arc<AppState>>,
    Path(id_str): Path<String>,
) -> impl IntoResponse {
    let object_id = match parse_hex_id(&id_str) {
        Some(id) => id,
        None => {
            return (StatusCode::BAD_REQUEST, "Invalid object ID").into_response();
        }
    };

    let result = tokio::task::spawn_blocking(move || -> Result<Option<String>, HprofError> {
        let q = &state.query;
        match q.find(object_id)? {
            Some(SubRecord::PrimArrayDump(arr)) => Ok(Some(decode_prim_array_full(
                arr.element_type,
                arr.data,
                arr.num_elements,
            ))),
            _ => Ok(None),
        }
    })
    .await;

    match result {
        Ok(Ok(Some(csv))) => (
            StatusCode::OK,
            [("content-type", "text/plain; charset=utf-8")],
            csv,
        )
            .into_response(),
        Ok(Ok(None)) => (StatusCode::NOT_FOUND, "Not a primitive array").into_response(),
        Ok(Err(e)) => internal(e).into_response(),
        Err(e) => internal(e).into_response(),
    }
}

// ── /object/:id/root-path — path to GC root ───────────────────────────────────

pub async fn root_path_page(
    State(state): State<Arc<AppState>>,
    Path(id_str): Path<String>,
) -> impl IntoResponse {
    let object_id = match parse_hex_id(&id_str) {
        Some(id) => id,
        None => return bad_request(format!("Invalid object ID: {id_str}")).into_response(),
    };

    let result = tokio::task::spawn_blocking(move || -> Result<String, HprofError> {
        let q = &state.query;

        let path_result = q.path_to_root(object_id);

        let content = match path_result {
            RootPathResult::NotReachable => "<p class=\"muted\">No path to a GC root found. \
                 The object may be unreachable or form a reference cycle \
                 with no live root.</p>"
                .to_owned(),
            RootPathResult::LimitReached => {
                format!(
                    "<p class=\"muted\">Search limit reached ({ROOT_PATH_SEARCH_LIMIT} nodes \
                     visited) without finding a root. The path may be very long or pass \
                     through a high-fanin object whose referrers exceed the index cap.</p>"
                )
            }
            RootPathResult::Found(path) => {
                let mut steps = String::new();
                let last_idx = path.len().saturating_sub(1);
                for (i, &id) in path.iter().enumerate() {
                    let type_name = q.object_type_name(id).unwrap_or_else(|_| "?".to_owned());
                    let root_types = q.root_types_of(id);

                    let root_badge = if !root_types.is_empty() {
                        let labels: Vec<_> = root_types
                            .iter()
                            .map(|rt| {
                                format!(
                                    "<a href=\"/roots/{}\"><strong>{}</strong></a>",
                                    gc_root_slug(*rt),
                                    gc_root_label(*rt)
                                )
                            })
                            .collect();
                        format!(
                            " &nbsp; <span style=\"color:#c00\">{}</span>",
                            labels.join(", ")
                        )
                    } else {
                        String::new()
                    };

                    let target_badge = if i == last_idx {
                        " &nbsp; <span style=\"color:#060\"><strong>← target</strong></span>"
                    } else {
                        ""
                    };

                    steps.push_str(&format!(
                        "<tr>\
                           <td class=\"num\" style=\"color:#888\">{i}</td>\
                           <td>{}</td>\
                           <td>{}</td>\
                           <td>{root_badge}{target_badge}</td>\
                         </tr>",
                        obj_link(id),
                        esc(&type_name),
                    ));
                }
                format!(
                    "<p>{} hops from GC root to target.</p>\
                     <table>\
                       <tr><th>#</th><th>Object</th><th>Type</th><th></th></tr>\
                       {steps}\
                     </table>",
                    last_idx
                )
            }
        };

        Ok(page(
            &format!("Root path for 0x{object_id:x}"),
            &format!(
                "<p><a href=\"/object/{object_id:x}\">\u{2190} back to object</a></p>{content}"
            ),
        ))
    })
    .await;

    match result {
        Ok(Ok(html)) => ok(html).into_response(),
        Ok(Err(e)) => internal(e).into_response(),
        Err(e) => internal(e).into_response(),
    }
}

// ── /arrays/:kind — arrays by size ────────────────────────────────────────────

pub async fn arrays_by_kind(
    State(state): State<Arc<AppState>>,
    Path(kind_str): Path<String>,
    Query(params): Query<PageParams>,
) -> impl IntoResponse {
    let kind = match ArrayKind::from_slug(&kind_str) {
        Some(k) => k,
        None => return bad_request(format!("Unknown array type: {kind_str}")).into_response(),
    };

    let result = tokio::task::spawn_blocking(move || -> Result<String, HprofError> {
        let q = &state.query;
        let id_size = q.id_size();
        let total = q.array_count(kind);
        let elem_size = kind.elem_size(id_size);
        let display_name = kind.display_name();

        let offset = params.offset.min(total);
        let limit = params.limit;
        let shown = limit.min(total.saturating_sub(offset));

        let mut rows = String::new();
        for entry in q.iter_arrays_by_size(kind).skip(offset).take(shown) {
            let num_elements = if elem_size > 0 { entry.byte_size / elem_size } else { 0 };
            rows.push_str(&format!(
                "<tr><td>{}</td><td class=\"num\">{num_elements}</td><td class=\"num\">{}</td></tr>",
                obj_link(entry.object_id),
                fmt_bytes(entry.byte_size),
            ));
        }

        let table = if rows.is_empty() {
            format!("<p class=\"muted\">No {display_name} arrays found.</p>")
        } else {
            format!(
                "<table><tr><th>Array</th><th>Elements</th><th>Size</th></tr>{rows}</table>"
            )
        };

        let mut nav = String::new();
        if offset > 0 {
            let prev = offset.saturating_sub(limit);
            nav.push_str(&format!(
                "<a href=\"/arrays/{kind_slug}?offset={prev}&limit={limit}\">← prev</a> ",
                kind_slug = kind.slug()
            ));
        }
        if offset + shown < total {
            let next = offset + limit;
            nav.push_str(&format!(
                "<a href=\"/arrays/{kind_slug}?offset={next}&limit={limit}\">next →</a>",
                kind_slug = kind.slug()
            ));
        }

        let content = format!(
            "<p>{total} {display_name} arrays total \
             (showing {offset}–{})</p>\
             {table}\
             <p>{nav}</p>",
            offset + shown
        );
        Ok(page(&format!("{display_name} arrays by size"), &content))
    })
    .await;

    match result {
        Ok(Ok(html)) => ok(html).into_response(),
        Ok(Err(e)) => internal(e).into_response(),
        Err(e) => internal(e).into_response(),
    }
}

// ── /roots — GC root type summary ─────────────────────────────────────────────

pub async fn roots_summary(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(move || -> Result<String, HprofError> {
        let q = &state.query;
        let mut rows = String::new();
        for rt in GcRootType::ALL {
            let count = q.iter_roots(rt).count();
            let slug = gc_root_slug(rt);
            let label = gc_root_label(rt);
            rows.push_str(&format!(
                "<tr><td><a href=\"/roots/{slug}\">{label}</a></td><td class=\"num\">{count}</td></tr>"
            ));
        }
        let content = format!(
            "<table><tr><th>Root type</th><th>Count</th></tr>{rows}</table>"
        );
        Ok(page("GC Roots", &content))
    })
    .await;

    match result {
        Ok(Ok(html)) => ok(html).into_response(),
        Ok(Err(e)) => internal(e).into_response(),
        Err(e) => internal(e).into_response(),
    }
}

// ── /roots/{type} — GC roots of a type ───────────────────────────────────────

pub async fn roots_by_type(
    State(state): State<Arc<AppState>>,
    Path(root_type_str): Path<String>,
    Query(p): Query<PageParams>,
) -> impl IntoResponse {
    let rt = match parse_root_type(&root_type_str) {
        Some(rt) => rt,
        None => return not_found(format!("Unknown root type: {root_type_str}")).into_response(),
    };

    let result = tokio::task::spawn_blocking(move || -> Result<String, HprofError> {
        let q = &state.query;
        let label = gc_root_label(rt);

        let mut rows = String::new();
        let mut total = 0usize;
        for entry in q.iter_roots(rt) {
            total += 1;
            if total > p.offset && rows.len() < p.limit * 60 {
                let name = q
                    .object_type_name(entry.object_id)
                    .unwrap_or_else(|_| "?".to_owned());
                rows.push_str(&format!(
                    "<tr><td>{}</td><td>{}</td></tr>",
                    obj_link(entry.object_id),
                    esc(&name)
                ));
            }
        }

        let note = if total > p.offset + p.limit {
            format!(
                "<p><a href=\"/roots/{root_type_str}?offset={}&limit={}\">Next {} →</a></p>",
                p.offset + p.limit,
                p.limit,
                p.limit
            )
        } else {
            String::new()
        };

        let prev = if p.offset > 0 {
            format!(
                "<p><a href=\"/roots/{root_type_str}?offset={}&limit={}\">← Prev {}</a></p>",
                p.offset.saturating_sub(p.limit),
                p.limit,
                p.limit
            )
        } else {
            String::new()
        };

        let content = format!(
            "<p>Total: {total}</p>{prev}<table><tr><th>Object ID</th><th>Type</th></tr>{rows}</table>{note}"
        );
        Ok(page(&format!("GC Roots: {label}"), &content))
    })
    .await;

    match result {
        Ok(Ok(html)) => ok(html).into_response(),
        Ok(Err(e)) => internal(e).into_response(),
        Err(e) => internal(e).into_response(),
    }
}

// ── /threads — Thread list ────────────────────────────────────────────────────

pub async fn threads(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(move || -> Result<String, HprofError> {
        let q = &state.query;
        let mut rows = String::new();

        // Primary: HPROF_START_THREAD records (present in most dumps)
        let mut found_any = false;
        for result in q.iter_threads() {
            found_any = true;
            let thread = result?;
            let resolved = q.resolve_thread(&thread)?;
            let ended = if q.was_thread_ended(thread.thread_serial) { "ended" } else { "running" };
            rows.push_str(&format!(
                "<tr><td><a href=\"/thread/{}\">{}</a></td><td>{}</td><td>{}</td><td>{ended}</td></tr>",
                thread.thread_serial,
                esc(&resolved.thread_name),
                esc(&resolved.thread_group_name),
                thread.thread_serial,
            ));
        }

        // Fallback: GC_ROOT_THREAD_OBJ sub-records (present in all dumps)
        if !found_any {
            let mut thread_objs: Vec<(u32, u64)> = Vec::new(); // (thread_serial, thread_obj_id)
            for result in q.iter_objects() {
                if let SubRecord::RootThreadObj(r) = result? {
                    thread_objs.push((r.thread_serial, r.thread_object_id));
                }
            }
            thread_objs.sort_by_key(|t| t.0);
            for (serial, obj_id) in &thread_objs {
                let name = thread_name_from_object(q, *obj_id);
                rows.push_str(&format!(
                    "<tr><td><a href=\"/thread/{serial}\">{}</a></td><td class=\"muted\">—</td><td>{serial}</td><td>—</td></tr>",
                    esc(&name),
                ));
            }
        }

        let content = format!(
            "<table><tr><th>Thread name</th><th>Group</th><th>Serial</th><th>Status</th></tr>{rows}</table>"
        );
        Ok(page("Threads", &content))
    })
    .await;

    match result {
        Ok(Ok(html)) => ok(html).into_response(),
        Ok(Err(e)) => internal(e).into_response(),
        Err(e) => internal(e).into_response(),
    }
}

// ── /thread/{serial} — Thread detail ─────────────────────────────────────────

pub async fn thread_detail(
    State(state): State<Arc<AppState>>,
    Path(serial_str): Path<String>,
) -> impl IntoResponse {
    let serial: u32 = match serial_str.parse() {
        Ok(s) => s,
        Err(_) => {
            return bad_request(format!("Invalid thread serial: {serial_str}")).into_response();
        }
    };

    let result = tokio::task::spawn_blocking(move || -> Result<String, HprofError> {
        let q = &state.query;

        // (thread_name, thread_group, thread_obj_id, trace_serial)
        let thread_info: Option<(String, String, u64, u32)> =
            if let Some(t) = q.find_thread(serial)? {
                let resolved = q.resolve_thread(&t)?;
                Some((
                    resolved.thread_name,
                    resolved.thread_group_name,
                    t.thread_id,
                    t.stack_trace_serial,
                ))
            } else {
                // Fall back: scan GC_ROOT_THREAD_OBJ sub-records
                let mut found = None;
                for result in q.iter_objects() {
                    if let SubRecord::RootThreadObj(r) = result?
                        && r.thread_serial == serial
                    {
                        let name = thread_name_from_object(q, r.thread_object_id);
                        found = Some((
                            name,
                            String::new(),
                            r.thread_object_id,
                            r.stack_trace_serial,
                        ));
                        break;
                    }
                }
                found
            };

        let (thread_name, thread_group, thread_obj_id, trace_serial) = match thread_info {
            Some(t) => t,
            None => {
                return Ok(page(
                    "Thread not found",
                    "<p>No thread with that serial.</p>",
                ));
            }
        };
        let trace = q.find_trace(trace_serial)?;

        let mut frames_html = String::new();
        if let Some(trace) = &trace {
            let frames = q.trace_frames(trace)?;
            for frame in &frames {
                let rf = q.resolve_frame(frame)?;
                let line = match &rf.line_number {
                    crate::aux_query::LineNumber::Line(n) => n.to_string(),
                    crate::aux_query::LineNumber::Native => "native".to_owned(),
                    crate::aux_query::LineNumber::Compiled => "compiled".to_owned(),
                    crate::aux_query::LineNumber::Unknown
                    | crate::aux_query::LineNumber::NoInfo => "?".to_owned(),
                };
                frames_html.push_str(&format!(
                    "<tr><td>{}</td><td>{}</td><td>{}</td></tr>",
                    esc(&rf.method_name),
                    esc(&rf.source_file),
                    line
                ));
            }
        }

        let frames_section = if frames_html.is_empty() {
            "<p class=\"muted\">No stack frames available.</p>".to_owned()
        } else {
            format!(
                "<table><tr><th>Method</th><th>Source</th><th>Line</th></tr>{frames_html}</table>"
            )
        };

        let content = format!(
            r#"<table>
<tr><td>Thread name</td><td>{}</td></tr>
<tr><td>Thread group</td><td>{}</td></tr>
<tr><td>Serial</td><td>{serial}</td></tr>
<tr><td>Thread object</td><td>{}</td></tr>
<tr><td>Trace serial</td><td>{}</td></tr>
</table>
<h2>Stack trace</h2>
{frames_section}"#,
            esc(&thread_name),
            esc(&thread_group),
            obj_link(thread_obj_id),
            trace_serial,
        );
        Ok(page(&format!("Thread: {thread_name}"), &content))
    })
    .await;

    match result {
        Ok(Ok(html)) => ok(html).into_response(),
        Ok(Err(e)) => internal(e).into_response(),
        Err(e) => internal(e).into_response(),
    }
}

// ── Helper: banner for resolved Java wrapper values ───────────────────────────

/// If `jv` is a recognised wrapper type, returns an HTML callout showing the
/// resolved value.  Returns an empty string for `JavaValue::Object` (unknown
/// type) and `JavaValue::Null`.
fn render_java_value_banner(jv: &crate::heap_query::JavaValue) -> String {
    use crate::heap_query::JavaValue;

    let text = match jv {
        JavaValue::String(id, s) => {
            // Truncate very long strings so the page stays readable.
            const MAX: usize = 2000;
            let raw_link = format!(" <a href=\"/object/{id:x}/raw-string\">raw</a>");
            if s.len() > MAX {
                format!(
                    "\"{}\" <span class=\"muted\">… ({} chars total)</span>{raw_link}",
                    esc(&s[..MAX]),
                    s.chars().count()
                )
            } else {
                format!("\"{}\"  {raw_link}", esc(s))
            }
        }
        JavaValue::Boolean(_, b) => b.to_string(),
        JavaValue::Byte(_, b) => format!("{b}"),
        JavaValue::Short(_, s) => format!("{s}"),
        JavaValue::Character(_, c) => {
            let ch = char::from_u32(u32::from(*c)).unwrap_or('?');
            format!("'{}' (U+{:04X})", esc(&ch.to_string()), c)
        }
        JavaValue::Integer(_, i) => format!("{i}"),
        JavaValue::Long(_, l) => format!("{l}L"),
        JavaValue::Float(_, f) => format!("{f}f"),
        JavaValue::Double(_, d) => format!("{d}"),
        // Not a recognised wrapper — no banner.
        JavaValue::Object(_) | JavaValue::Null => return String::new(),
    };

    format!(
        "<p style=\"background:#fffbe6;border:1px solid #f0c040;padding:0.5em 1em;margin:0.5em 0\">\
         <strong>Value:</strong> {text}</p>"
    )
}

// ── Helper: render a FieldValue as HTML ──────────────────────────────────────

fn render_field_value(v: &FieldValue, q: &crate::query::HeapQuery) -> String {
    match v {
        FieldValue::Object(0) => "<span class=\"muted\">null</span>".to_owned(),
        FieldValue::Object(id) => {
            // Try to resolve as a known wrapper type
            let resolved = q.resolve_value(*id).ok();
            match resolved {
                Some(crate::heap_query::JavaValue::String(sid, s)) => {
                    format!(
                        "{} <span class=\"muted\">\"{}\"</span> <a href=\"/object/{sid:x}/raw-string\">raw</a>",
                        obj_link(sid),
                        esc(&s)
                    )
                }
                Some(crate::heap_query::JavaValue::Integer(oid, n)) => {
                    format!("{} <span class=\"muted\">= {n}</span>", obj_link(oid))
                }
                Some(crate::heap_query::JavaValue::Long(oid, n)) => {
                    format!("{} <span class=\"muted\">= {n}L</span>", obj_link(oid))
                }
                _ => obj_link(*id),
            }
        }
        FieldValue::Bool(b) => b.to_string(),
        FieldValue::Char(c) => char::from_u32(u32::from(*c))
            .map(|ch| format!("'{}' ({c})", esc(&ch.to_string())))
            .unwrap_or_else(|| c.to_string()),
        FieldValue::Float(f) => format!("{f}f"),
        FieldValue::Double(d) => format!("{d}d"),
        FieldValue::Byte(b) => b.to_string(),
        FieldValue::Short(s) => s.to_string(),
        FieldValue::Int(i) => i.to_string(),
        FieldValue::Long(l) => format!("{l}L"),
    }
}

// ── Helper: resolve display type for a field ─────────────────────────────────

/// Returns an HTML string for the type column of a field row.
///
/// For Object-typed fields with a non-null reference, looks up the actual
/// runtime class of the referenced object (one O(log n) binary search).
/// For null references or primitives, falls back to the static type name.
fn resolve_field_type(type_code: u8, value: &FieldValue, q: &crate::query::HeapQuery) -> String {
    if type_code != 2 {
        return field_type_name(type_code).to_owned();
    }
    let id = match value {
        FieldValue::Object(id) if *id != 0 => *id,
        _ => return "Object".to_owned(),
    };
    q.object_type_name(id)
        .unwrap_or_else(|_| "Object".to_owned())
}

// ── Helper: thread name from Thread object ────────────────────────────────────

/// Try to resolve a human-readable thread name from a `java.lang.Thread`
/// instance by following its `name` field to the backing String/char[].
/// Falls back to the hex object ID if the name cannot be resolved.
fn thread_name_from_object(q: &crate::query::HeapQuery, thread_obj_id: u64) -> String {
    let inst = match q.find_instance(thread_obj_id) {
        Ok(Some(SubRecord::InstanceDump(inst))) => inst,
        _ => return format!("0x{thread_obj_id:x}"),
    };
    let fields = match q.instance_fields(&inst) {
        Ok(f) => f,
        Err(_) => return format!("0x{thread_obj_id:x}"),
    };
    let name_ref = fields
        .iter()
        .find(|f| f.name == "name" && f.field_type == 2)
        .and_then(|f| match &f.value {
            FieldValue::Object(id) if *id != 0 => Some(*id),
            _ => None,
        });
    if let Some(name_id) = name_ref
        && let Ok(crate::heap_query::JavaValue::String(_, s)) = q.resolve_value(name_id)
    {
        return s;
    }
    format!("0x{thread_obj_id:x}")
}

// ── Helper: decode primitive array bytes → display string ────────────────────

/// Decode big-endian primitive array bytes into a human-readable `[v1, v2, …]`
/// string.  Shows up to `MAX_SHOW` elements; appends `…` if truncated.
fn decode_prim_array(element_type: u8, data: &[u8], num_elements: u32) -> String {
    const MAX_SHOW: usize = 256;

    macro_rules! decode_elements {
        ($size:expr, $fmt:expr, $conv:expr) => {{
            let shown = (num_elements as usize).min(MAX_SHOW);
            let mut parts: Vec<String> = data
                .chunks_exact($size)
                .take(shown)
                .map(|chunk| {
                    let arr: [u8; $size] = chunk.try_into().unwrap_or([0u8; $size]);
                    format!($fmt, $conv(arr))
                })
                .collect();
            if num_elements as usize > MAX_SHOW {
                parts.push(format!("… ({} more)", num_elements as usize - MAX_SHOW));
            }
            format!("[{}]", parts.join(", "))
        }};
    }

    match element_type {
        4 /* boolean */ => {
            let shown = (num_elements as usize).min(MAX_SHOW);
            let mut parts: Vec<String> = data
                .iter()
                .take(shown)
                .map(|b| if *b != 0 { "true".to_owned() } else { "false".to_owned() })
                .collect();
            if num_elements as usize > MAX_SHOW {
                parts.push(format!("… ({} more)", num_elements as usize - MAX_SHOW));
            }
            format!("[{}]", parts.join(", "))
        }
        5 /* char */ => {
            let shown = (num_elements as usize).min(MAX_SHOW);
            let mut parts: Vec<String> = data
                .chunks_exact(2)
                .take(shown)
                .map(|c| {
                    let cp = u16::from_be_bytes([c[0], c[1]]);
                    let ch = char::from_u32(u32::from(cp)).unwrap_or('\u{FFFD}');
                    format!("'{}'", ch)
                })
                .collect();
            if num_elements as usize > MAX_SHOW {
                parts.push(format!("… ({} more)", num_elements as usize - MAX_SHOW));
            }
            format!("[{}]", parts.join(", "))
        }
        6  /* float  */ => decode_elements!(4, "{}", |a: [u8; 4]| f32::from_be_bytes(a)),
        7  /* double */ => decode_elements!(8, "{}", |a: [u8; 8]| f64::from_be_bytes(a)),
        8  /* byte   */ => {
            let shown = (num_elements as usize).min(MAX_SHOW);
            let mut parts: Vec<String> = data
                .iter()
                .take(shown)
                .map(|b| format!("{}", *b as i8))
                .collect();
            if num_elements as usize > MAX_SHOW {
                parts.push(format!("… ({} more)", num_elements as usize - MAX_SHOW));
            }
            format!("[{}]", parts.join(", "))
        }
        9  /* short  */ => decode_elements!(2, "{}", |a: [u8; 2]| i16::from_be_bytes(a)),
        10 /* int    */ => decode_elements!(4, "{}", |a: [u8; 4]| i32::from_be_bytes(a)),
        11 /* long   */ => decode_elements!(8, "{}", |a: [u8; 8]| i64::from_be_bytes(a)),
        _ => {
            data.iter()
                .take(64)
                .map(|b| format!("{b:02x}"))
                .collect::<Vec<_>>()
                .join(" ")
        }
    }
}

// ── Helper: decode primitive array bytes → full comma-delimited string ────────

/// Decode big-endian primitive array bytes into a comma-delimited string with
/// no truncation, suitable for serving as raw text.
fn decode_prim_array_full(element_type: u8, data: &[u8], num_elements: u32) -> String {
    macro_rules! decode_all {
        ($size:expr, $fmt:expr, $conv:expr) => {{
            data.chunks_exact($size)
                .take(num_elements as usize)
                .map(|chunk| {
                    let arr: [u8; $size] = chunk.try_into().unwrap_or([0u8; $size]);
                    format!($fmt, $conv(arr))
                })
                .collect::<Vec<_>>()
                .join(",")
        }};
    }

    match element_type {
        4 /* boolean */ => data
            .iter()
            .take(num_elements as usize)
            .map(|b| if *b != 0 { "true" } else { "false" })
            .collect::<Vec<_>>()
            .join(","),
        5 /* char */ => data
            .chunks_exact(2)
            .take(num_elements as usize)
            .map(|c| {
                let cp = u16::from_be_bytes([c[0], c[1]]);
                char::from_u32(u32::from(cp))
                    .unwrap_or('\u{FFFD}')
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join(","),
        6  /* float  */ => decode_all!(4, "{}", |a: [u8; 4]| f32::from_be_bytes(a)),
        7  /* double */ => decode_all!(8, "{}", |a: [u8; 8]| f64::from_be_bytes(a)),
        8  /* byte   */ => data
            .iter()
            .take(num_elements as usize)
            .map(|b| format!("{}", *b as i8))
            .collect::<Vec<_>>()
            .join(","),
        9  /* short  */ => decode_all!(2, "{}", |a: [u8; 2]| i16::from_be_bytes(a)),
        10 /* int    */ => decode_all!(4, "{}", |a: [u8; 4]| i32::from_be_bytes(a)),
        11 /* long   */ => decode_all!(8, "{}", |a: [u8; 8]| i64::from_be_bytes(a)),
        _ => data
            .iter()
            .take(num_elements as usize)
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<_>>()
            .join(","),
    }
}

// ── Helper: hprof type code → display name ────────────────────────────────────

fn field_type_name(type_code: u8) -> &'static str {
    match type_code {
        2 => "Object",
        4 => "boolean",
        5 => "char",
        6 => "float",
        7 => "double",
        8 => "byte",
        9 => "short",
        10 => "int",
        11 => "long",
        _ => "?",
    }
}

// ── Helper: GC root type labels / slugs ──────────────────────────────────────

pub(crate) fn gc_root_label(rt: GcRootType) -> &'static str {
    match rt {
        GcRootType::Unknown => "Unknown",
        GcRootType::JniGlobal => "JNI global",
        GcRootType::JniLocal => "JNI local",
        GcRootType::JavaFrame => "Java frame",
        GcRootType::NativeStack => "Native stack",
        GcRootType::StickyClass => "Sticky class",
        GcRootType::ThreadBlock => "Thread block",
        GcRootType::MonitorUsed => "Monitor used",
        GcRootType::ThreadObject => "Thread object",
    }
}

pub(crate) fn gc_root_slug(rt: GcRootType) -> &'static str {
    match rt {
        GcRootType::Unknown => "unknown",
        GcRootType::JniGlobal => "jni_global",
        GcRootType::JniLocal => "jni_local",
        GcRootType::JavaFrame => "java_frame",
        GcRootType::NativeStack => "native_stack",
        GcRootType::StickyClass => "sticky_class",
        GcRootType::ThreadBlock => "thread_block",
        GcRootType::MonitorUsed => "monitor_used",
        GcRootType::ThreadObject => "thread_object",
    }
}

pub(crate) fn parse_root_type(s: &str) -> Option<GcRootType> {
    match s {
        "unknown" => Some(GcRootType::Unknown),
        "jni_global" => Some(GcRootType::JniGlobal),
        "jni_local" => Some(GcRootType::JniLocal),
        "java_frame" => Some(GcRootType::JavaFrame),
        "native_stack" => Some(GcRootType::NativeStack),
        "sticky_class" => Some(GcRootType::StickyClass),
        "thread_block" => Some(GcRootType::ThreadBlock),
        "monitor_used" => Some(GcRootType::MonitorUsed),
        "thread_object" => Some(GcRootType::ThreadObject),
        _ => None,
    }
}

// ── Diff navigation helpers ───────────────────────────────────────────────────

/// Get the synthetic class-bucket key for a `DiffEntry` by parsing the record.
fn diff_entry_class_key(entry: &DiffEntry, query: &crate::query::HeapQuery) -> Option<u64> {
    let sub = SubIndexEntry {
        tag: entry.tag,
        object_id: entry.object_id,
        position: entry.position,
    };
    let record = query.parse_entry(&sub).ok()?;
    match record {
        SubRecord::InstanceDump(inst) => Some(inst.class_id),
        SubRecord::ClassDump(cd) => Some(cd.class_id),
        SubRecord::ObjArrayDump(arr) => Some(SYNTHETIC_OBJ_ARRAY | arr.element_class_id),
        SubRecord::PrimArrayDump(arr) => Some(SYNTHETIC_PRIM_ARRAY | u64::from(arr.element_type)),
        _ => None,
    }
}

/// Resolve a display class name for a class-bucket key.
fn diff_class_name(class_key: u64, query: &crate::query::HeapQuery) -> String {
    if (class_key & SYNTHETIC_PRIM_ARRAY) == SYNTHETIC_PRIM_ARRAY {
        return format!("{}[]", prim_type_name((class_key & 0xFF) as u8));
    }
    if class_key & SYNTHETIC_OBJ_ARRAY != 0 {
        let elem_id = class_key & !SYNTHETIC_OBJ_ARRAY;
        let elem = query
            .class_name(elem_id)
            .ok()
            .flatten()
            .unwrap_or_else(|| format!("0x{elem_id:x}"));
        return format!("{elem}[]");
    }
    query
        .class_name(class_key)
        .ok()
        .flatten()
        .unwrap_or_else(|| format!("0x{class_key:x}"))
}

// ── Helper: object fingerprint for address-reuse detection ───────────────────

/// Returns a `(variant_tag, size)` pair that can be compared between two dumps.
/// If the fingerprints differ for the same object ID, the JVM reused that address.
fn object_fingerprint(record: &SubRecord<'_>) -> (u8, u64) {
    match record {
        SubRecord::InstanceDump(inst) => (0x21, inst.data.len() as u64),
        SubRecord::ObjArrayDump(arr) => (0x22, u64::from(arr.num_elements)),
        SubRecord::PrimArrayDump(arr) => (0x23, u64::from(arr.num_elements)),
        SubRecord::ClassDump(cd) => (0x20, u64::from(cd.instance_size)),
        _ => (0, 0),
    }
}

const ADDRESS_REUSE_WARNING: &str = "<div class=\"warn\">⚠ <strong>Address reused between dumps.</strong> \
     The object at this address differs in type or size between the two heap dumps. \
     The JVM garbage-collected the original object and allocated a new, unrelated object \
     at the same address. The two views show different objects, not the same object \
     that changed.</div>";

// ── /diff/object/:id — object detail from dump 2 ─────────────────────────────

pub async fn diff_object_detail(
    State(state): State<Arc<AppState>>,
    Path(id_str): Path<String>,
) -> impl IntoResponse {
    let object_id = match parse_hex_id(&id_str) {
        Some(id) => id,
        None => return bad_request(format!("Invalid object ID: {id_str}")).into_response(),
    };

    let result = tokio::task::spawn_blocking(move || -> Result<String, HprofError> {
        let q = match state.diff_query.as_ref() {
            Some(q) => q,
            None => {
                return Ok(page(
                    "Object (dump 2)",
                    "<p>No second heap dump configured.</p>",
                ));
            }
        };
        let reuse_warning = match (state.query.find(object_id)?, q.find(object_id)?) {
            (Some(r1), Some(r2)) if object_fingerprint(&r1) != object_fingerprint(&r2) => {
                ADDRESS_REUSE_WARNING
            }
            _ => "",
        };
        let note = format!(
            "{reuse_warning}<p class=\"muted\"><em>Viewing from <strong>dump 2</strong>. \
             <a href=\"/object/{object_id:x}\">View from dump 1 →</a></em></p>"
        );
        render_object_page(object_id, q, &note)
    })
    .await;

    match result {
        Ok(Ok(html)) => ok(html).into_response(),
        Ok(Err(e)) => internal(e).into_response(),
        Err(e) => internal(e).into_response(),
    }
}

// ── /diff/removed — removed instances list ────────────────────────────────────

pub async fn diff_removed(
    State(state): State<Arc<AppState>>,
    Query(p): Query<DiffListParams>,
) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(move || -> Result<String, HprofError> {
        let paths = match state.diff_index_paths.as_ref() {
            Some(p) => p,
            None => return Ok(page("Diff: Removed", "<p>No diff configured.</p>")),
        };
        let removed_mmap = paths.removed.open_mmap()?;
        let reader = DiffEntryReader::from_ref(removed_mmap.as_ref())?;
        let q = &state.query;
        let class_filter = p.class.as_deref().and_then(parse_hex_id);

        let mut entries: Vec<(u64, String)> = Vec::new(); // (object_id, class_name)
        for entry in reader.iter() {
            let class_key = diff_entry_class_key(&entry, q);
            if let Some(filter) = class_filter
                && class_key != Some(filter)
            {
                continue;
            }
            let class_name = class_key.map(|k| diff_class_name(k, q)).unwrap_or_default();
            entries.push((entry.object_id, class_name));
        }

        let total = entries.len();
        let offset = p.offset.min(total);
        let shown = p.limit.min(total.saturating_sub(offset));

        let mut rows = String::new();
        for (oid, class_name) in entries.iter().skip(offset).take(shown) {
            rows.push_str(&format!(
                "<tr><td>{}</td><td>{}</td></tr>",
                obj_link(*oid),
                esc(class_name),
            ));
        }

        let class_param = class_filter
            .map(|c| format!("&class={c:x}"))
            .unwrap_or_default();
        let title = class_filter
            .map(|ck| format!("Removed: {}", diff_class_name(ck, q)))
            .unwrap_or_else(|| "Removed Instances".to_owned());

        let prev = if offset > 0 {
            format!(
                "<p><a href=\"/diff/removed?offset={}&limit={}{class_param}\">← Prev {}</a></p>",
                offset.saturating_sub(p.limit),
                p.limit,
                p.limit
            )
        } else {
            String::new()
        };
        let next = if offset + shown < total {
            format!(
                "<p><a href=\"/diff/removed?offset={}&limit={}{class_param}\">Next {} →</a></p>",
                offset + p.limit,
                p.limit,
                p.limit
            )
        } else {
            String::new()
        };

        let content = format!(
            "<p>{total} removed; showing {offset}–{}</p>\
             {prev}\
             <table><tr><th>Object (dump 1)</th><th>Class</th></tr>{rows}</table>\
             {next}",
            offset + shown,
        );
        Ok(page(&title, &content))
    })
    .await;

    match result {
        Ok(Ok(html)) => ok(html).into_response(),
        Ok(Err(e)) => internal(e).into_response(),
        Err(e) => internal(e).into_response(),
    }
}

// ── /diff/added — added instances list ───────────────────────────────────────

pub async fn diff_added(
    State(state): State<Arc<AppState>>,
    Query(p): Query<DiffListParams>,
) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(move || -> Result<String, HprofError> {
        let paths = match state.diff_index_paths.as_ref() {
            Some(p) => p,
            None => return Ok(page("Diff: Added", "<p>No diff configured.</p>")),
        };
        let added_mmap = paths.added.open_mmap()?;
        let reader = DiffEntryReader::from_ref(added_mmap.as_ref())?;
        // Added objects live in dump 2; use diff_query for class resolution.
        let q = match state.diff_query.as_ref() {
            Some(q) => q,
            None => return Ok(page("Diff: Added", "<p>No diff configured.</p>")),
        };
        let class_filter = p.class.as_deref().and_then(parse_hex_id);

        let mut entries: Vec<(u64, String)> = Vec::new();
        for entry in reader.iter() {
            let class_key = diff_entry_class_key(&entry, q);
            if let Some(filter) = class_filter
                && class_key != Some(filter)
            {
                continue;
            }
            let class_name = class_key.map(|k| diff_class_name(k, q)).unwrap_or_default();
            entries.push((entry.object_id, class_name));
        }

        let total = entries.len();
        let offset = p.offset.min(total);
        let shown = p.limit.min(total.saturating_sub(offset));

        let mut rows = String::new();
        for (oid, class_name) in entries.iter().skip(offset).take(shown) {
            // Added objects are in dump 2, so link to /diff/object/:id
            let link = format!("<a href=\"/diff/object/{oid:x}\">0x{oid:x}</a>");
            rows.push_str(&format!(
                "<tr><td>{link}</td><td>{}</td></tr>",
                esc(class_name),
            ));
        }

        let class_param = class_filter
            .map(|c| format!("&class={c:x}"))
            .unwrap_or_default();
        let title = class_filter
            .map(|ck| format!("Added: {}", diff_class_name(ck, q)))
            .unwrap_or_else(|| "Added Instances".to_owned());

        let prev = if offset > 0 {
            format!(
                "<p><a href=\"/diff/added?offset={}&limit={}{class_param}\">← Prev {}</a></p>",
                offset.saturating_sub(p.limit),
                p.limit,
                p.limit
            )
        } else {
            String::new()
        };
        let next = if offset + shown < total {
            format!(
                "<p><a href=\"/diff/added?offset={}&limit={}{class_param}\">Next {} →</a></p>",
                offset + p.limit,
                p.limit,
                p.limit
            )
        } else {
            String::new()
        };

        let content = format!(
            "<p>{total} added; showing {offset}–{}</p>\
             {prev}\
             <table><tr><th>Object (dump 2)</th><th>Class</th></tr>{rows}</table>\
             {next}",
            offset + shown,
        );
        Ok(page(&title, &content))
    })
    .await;

    match result {
        Ok(Ok(html)) => ok(html).into_response(),
        Ok(Err(e)) => internal(e).into_response(),
        Err(e) => internal(e).into_response(),
    }
}

// ── /diff/common — common instances list ─────────────────────────────────────

pub async fn diff_common(
    State(state): State<Arc<AppState>>,
    Query(p): Query<DiffListParams>,
) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(move || -> Result<String, HprofError> {
        let paths = match state.diff_index_paths.as_ref() {
            Some(p) => p,
            None => return Ok(page("Diff: Common", "<p>No diff configured.</p>")),
        };
        let common_mmap = paths.common.open_mmap()?;
        let reader = CommonEntryReader::from_ref(common_mmap.as_ref())?;
        let q = &state.query;
        let class_filter = p.class.as_deref().and_then(parse_hex_id);
        // changed filter: "0" = unchanged only, "1" = changed only, absent = all
        let changed_filter: Option<bool> = match p.changed.as_deref() {
            Some("0") => Some(false),
            Some("1") => Some(true),
            _ => None,
        };

        // For common entries we need DiffEntry-compatible lookup; use the object_id
        // and position1 (dump 1 position) for class resolution.
        let mut entries: Vec<(u64, bool, String)> = Vec::new(); // (object_id, changed, class_name)
        for entry in reader.iter() {
            if let Some(cf) = changed_filter
                && entry.changed != cf
            {
                continue;
            }
            let stub = DiffEntry {
                tag: entry.tag,
                object_id: entry.object_id,
                position: entry.position1,
            };
            let class_key = diff_entry_class_key(&stub, q);
            if let Some(filter) = class_filter
                && class_key != Some(filter)
            {
                continue;
            }
            let class_name = class_key.map(|k| diff_class_name(k, q)).unwrap_or_default();
            entries.push((entry.object_id, entry.changed, class_name));
        }

        let total = entries.len();
        let offset = p.offset.min(total);
        let shown = p.limit.min(total.saturating_sub(offset));

        let mut rows = String::new();
        for (oid, changed, class_name) in entries.iter().skip(offset).take(shown) {
            let changed_badge = if *changed {
                "<span class=\"removed\">changed</span>"
            } else {
                "<span class=\"muted\">unchanged</span>"
            };
            let dump2_link = format!("<a href=\"/diff/object/{oid:x}\">dump 2</a>");
            rows.push_str(&format!(
                "<tr><td>{}</td><td>{dump2_link}</td><td>{changed_badge}</td><td>{}</td></tr>",
                obj_link(*oid),
                esc(class_name),
            ));
        }

        let changed_param = p
            .changed
            .as_deref()
            .map(|v| format!("&changed={v}"))
            .unwrap_or_default();
        let class_param = class_filter
            .map(|c| format!("&class={c:x}"))
            .unwrap_or_default();
        let extra_params = format!("{changed_param}{class_param}");

        let filter_desc = match (changed_filter, class_filter) {
            (Some(true), None) => " (changed only)".to_owned(),
            (Some(false), None) => " (unchanged only)".to_owned(),
            (None, Some(ck)) => format!(" — {}", diff_class_name(ck, q)),
            (Some(true), Some(ck)) => format!(" — {} (changed)", diff_class_name(ck, q)),
            (Some(false), Some(ck)) => format!(" — {} (unchanged)", diff_class_name(ck, q)),
            (None, None) => String::new(),
        };
        let title = format!("Common Instances{filter_desc}");

        let prev = if offset > 0 {
            format!(
                "<p><a href=\"/diff/common?offset={}&limit={}{extra_params}\">← Prev {}</a></p>",
                offset.saturating_sub(p.limit),
                p.limit,
                p.limit
            )
        } else {
            String::new()
        };
        let next = if offset + shown < total {
            format!(
                "<p><a href=\"/diff/common?offset={}&limit={}{extra_params}\">Next {} →</a></p>",
                offset + p.limit,
                p.limit,
                p.limit
            )
        } else {
            String::new()
        };

        let content = format!(
            "<style>.removed{{color:#cc0000}}</style>\
             <p>{total} entries{filter_desc}; showing {offset}–{}</p>\
             {prev}\
             <table>\
             <tr><th>Object (dump 1)</th><th>Dump 2</th><th>Status</th><th>Class</th></tr>\
             {rows}\
             </table>\
             {next}",
            offset + shown,
        );
        Ok(page(&title, &content))
    })
    .await;

    match result {
        Ok(Ok(html)) => ok(html).into_response(),
        Ok(Err(e)) => internal(e).into_response(),
        Err(e) => internal(e).into_response(),
    }
}

// ── /diff — Heap dump diff summary ───────────────────────────────────────────

pub async fn diff_summary(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(move || -> Result<String, HprofError> {
        let diff_path = match state.diff_path.as_ref() {
            Some(p) => p.display().to_string(),
            None => {
                let content = r#"<p>No second heap dump configured.</p>
<p>Restart the server with <code>--diff-hprof &lt;path&gt;</code> to enable heap diff.</p>"#;
                return Ok(page("Heap Diff", content));
            }
        };

        let summary = match state.diff() {
            Some(r) => r?,
            None => {
                return Ok(page(
                    "Heap Diff",
                    "<p>No second heap dump configured.</p>",
                ))
            }
        };

        let before_path = state.hprof_path.display().to_string();

        let mut rows = String::new();
        for entry in &summary.by_class {
            let cid = entry.class_id;
            let change = entry.net_change();
            let change_class = if change > 0 {
                "added"
            } else if change < 0 {
                "removed"
            } else {
                ""
            };
            let change_str = if change > 0 {
                format!("+{change}")
            } else {
                format!("{change}")
            };
            // Link each count to the filtered list page.
            let added_cell = if entry.count_added > 0 {
                format!("<a href=\"/diff/added?class={cid:x}\" class=\"added\">+{}</a>", entry.count_added)
            } else {
                "0".to_owned()
            };
            let removed_cell = if entry.count_removed > 0 {
                format!("<a href=\"/diff/removed?class={cid:x}\" class=\"removed\">-{}</a>", entry.count_removed)
            } else {
                "0".to_owned()
            };
            let unchanged_cell = if entry.count_common_unchanged > 0 {
                format!("<a href=\"/diff/common?changed=0&class={cid:x}\">{}</a>", entry.count_common_unchanged)
            } else {
                "0".to_owned()
            };
            let changed_cell = if entry.count_common_changed > 0 {
                format!("<a href=\"/diff/common?changed=1&class={cid:x}\" class=\"removed\">{}</a>", entry.count_common_changed)
            } else {
                "0".to_owned()
            };
            rows.push_str(&format!(
                "<tr>\
                <td>{}</td>\
                <td class=\"num\">{}</td>\
                <td class=\"num\">{}</td>\
                <td class=\"num {change_class}\">{change_str}</td>\
                <td class=\"num\">{added_cell}</td>\
                <td class=\"num\">{removed_cell}</td>\
                <td class=\"num\">{unchanged_cell}</td>\
                <td class=\"num\">{changed_cell}</td>\
                </tr>",
                esc(&entry.class_name),
                entry.count_before(),
                entry.count_after(),
            ));
        }

        let content = format!(
            r#"<style>
.added{{color:#006600}}
.removed{{color:#cc0000}}
</style>
<table>
<tr><th>Property</th><th>Dump 1 (before)</th><th>Dump 2 (after)</th></tr>
<tr><td>File</td><td>{}</td><td>{diff_path}</td></tr>
<tr><td>Total objects</td><td class="num">{}</td><td class="num">{}</td></tr>
</table>
<p class="added"><a href="/diff/added" class="added">+{} added</a></p>
<p class="removed"><a href="/diff/removed" class="removed">-{} removed</a></p>
<p><a href="/diff/common?changed=0">{} common unchanged</a> &nbsp; <a href="/diff/common?changed=1" class="removed">{} common changed</a></p>
<h2>By Class</h2>
<table>
<tr><th>Class</th><th>Before</th><th>After</th><th>Net</th><th>Added</th><th>Removed</th><th>Unchanged</th><th>Changed</th></tr>
{rows}
</table>"#,
            esc(&before_path),
            summary.total_before,
            summary.total_after,
            summary.total_added,
            summary.total_removed,
            summary.total_common_unchanged,
            summary.total_common_changed,
        );

        Ok(page("Heap Diff", &content))
    })
    .await;

    match result {
        Ok(Ok(html)) => ok(html).into_response(),
        Ok(Err(e)) => internal(e).into_response(),
        Err(e) => internal(e).into_response(),
    }
}

// ── /retained — Retained heap histogram ──────────────────────────────────────

pub async fn retained_histogram(
    State(state): State<Arc<AppState>>,
    Query(p): Query<PageParams>,
) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(move || -> Result<String, HprofError> {
        let q = &state.query;

        if !q.has_retained_heap() {
            let content = "<p class=\"muted\">Retained heap index not available. \
                           Run <code>hprof-toolkit index</code> to build it.</p>";
            return Ok(page("Retained Heap", content));
        }

        // Collect all (retained_bytes, object_id) pairs and sort descending.
        let iter = q.iter_retained().ok_or(HprofError::InvalidIndexFile)?;
        let mut entries: Vec<(u64, u64)> = iter.map(|(id, ret)| (ret, id)).collect();
        entries.sort_unstable_by(|a, b| b.0.cmp(&a.0));

        let total_count = entries.len();
        let page_entries: Vec<(u64, u64)> =
            entries.into_iter().skip(p.offset).take(p.limit).collect();

        let mut rows = String::new();
        for (retained_bytes, object_id) in &page_entries {
            let type_name = q
                .object_type_name(*object_id)
                .unwrap_or_else(|_| "?".to_owned());
            let dom_cell = match q.dominator_of(*object_id) {
                None | Some(0) => "<span class=\"muted\">(root)</span>".to_owned(),
                Some(dom_id) => obj_link(dom_id),
            };
            rows.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td class=\"num\">{}</td><td>{dom_cell}</td></tr>",
                obj_link(*object_id),
                esc(&type_name),
                fmt_bytes(*retained_bytes),
            ));
        }

        let note = if total_count > p.offset + p.limit {
            format!(
                "<p><a href=\"/retained?offset={}&limit={}\">Next {} →</a></p>",
                p.offset + p.limit,
                p.limit,
                p.limit
            )
        } else {
            String::new()
        };

        let prev = if p.offset > 0 {
            format!(
                "<p><a href=\"/retained?offset={}&limit={}\">← Prev {}</a></p>",
                p.offset.saturating_sub(p.limit),
                p.limit,
                p.limit
            )
        } else {
            String::new()
        };

        let content = format!(
            r#"<p>{total_count} reachable objects with retained size data</p>
{prev}
<table>
<tr><th>Object ID</th><th>Type</th><th>Retained size</th><th>Dominator</th></tr>
{rows}
</table>
{note}"#
        );
        Ok(page("Retained Heap", &content))
    })
    .await;

    match result {
        Ok(Ok(html)) => ok(html).into_response(),
        Ok(Err(e)) => internal(e).into_response(),
        Err(e) => internal(e).into_response(),
    }
}
