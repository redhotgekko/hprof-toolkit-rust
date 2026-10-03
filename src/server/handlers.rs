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

use super::{AppState, class_link, esc, fmt_bytes, obj_link, page, parse_hex_id};
use crate::array_index::ArrayKind;
use crate::class_key::ClassKey;
use crate::graph::{PathOutcome, RootPathLimits};
use crate::heap_index::sub_record::TAG_OBJ_ARRAY_DUMP;
use crate::heap_parser::FieldValue;
use crate::heap_parser::SubRecord;
use crate::hprof::{BasicType, HprofError};
use crate::query::{Page, ScanWindow};
use crate::root_index::GcRootType;
use crate::search::{Matcher, SearchMode, SearchQuery};
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

/// Query params for the searchable, paged lists (`/histogram`, `/allClasses`,
/// `/threads`): a [`PageParams`] plus the search fields.
#[derive(serde::Deserialize, Default)]
pub struct SearchParams {
    /// The text to search for; absent or empty means no filter.
    pub q: Option<String>,
    /// A [`SearchMode::name`]; default `contains`.
    pub mode: Option<String>,
    /// `1` for a case-sensitive search; absent (an unticked box) for not.
    pub case: Option<String>,
    #[serde(default)]
    pub offset: usize,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

impl SearchParams {
    fn page(&self) -> Page {
        Page::new(self.offset, self.limit)
    }

    fn text(&self) -> &str {
        self.q.as_deref().unwrap_or("").trim()
    }

    fn case_sensitive(&self) -> bool {
        self.case.as_deref() == Some("1")
    }

    fn mode(&self) -> Result<SearchMode, HprofError> {
        match self.mode.as_deref() {
            None | Some("") => Ok(SearchMode::Contains),
            Some(name) => SearchMode::from_name(name).ok_or_else(|| {
                HprofError::InvalidArgument(format!(
                    "unknown search mode `{name}`; use contains, exact, regex or fuzzy"
                ))
            }),
        }
    }

    /// The compiled search, or `None` when there is no text to search for.
    fn matcher(&self) -> Result<Option<Matcher>, HprofError> {
        if self.text().is_empty() {
            return Ok(None);
        }
        let query =
            SearchQuery::new(self.text(), self.mode()?).case_sensitive(self.case_sensitive());
        Matcher::new(&query).map(Some)
    }

    /// The `&q=…&mode=…&case=1` tail that paging links carry, or nothing
    /// when there is no search.
    fn query_suffix(&self) -> String {
        if self.text().is_empty() {
            return String::new();
        }
        let mut tail = format!(
            "&q={}&mode={}",
            url_encode(self.text()),
            self.mode().map_or("contains", SearchMode::name)
        );
        if self.case_sensitive() {
            tail.push_str("&case=1");
        }
        tail
    }
}

/// Query params for `/strings`: a search over a bounded, resumable scan.
#[derive(serde::Deserialize, Default)]
pub struct StringsParams {
    pub q: Option<String>,
    pub mode: Option<String>,
    pub case: Option<String>,
    #[serde(default)]
    pub cursor: usize,
    pub max_scan: Option<usize>,
    pub max_results: Option<usize>,
}

impl StringsParams {
    fn search(&self) -> SearchParams {
        SearchParams {
            q: self.q.clone(),
            mode: self.mode.clone(),
            case: self.case.clone(),
            offset: 0,
            limit: 0,
        }
    }

    fn window(&self) -> ScanWindow {
        ScanWindow::new(
            self.cursor,
            self.max_scan
                .unwrap_or(ScanWindow::DEFAULT_MAX_SCAN)
                .clamp(1, ScanWindow::MAX_SCAN_LIMIT),
            self.max_results
                .unwrap_or(ScanWindow::DEFAULT_MAX_RESULTS)
                .clamp(1, ScanWindow::MAX_RESULTS_LIMIT),
        )
    }
}

/// Percent-encode `s` for a query-string value (everything outside the
/// unreserved set).
fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(char::from(b));
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Prev/Next links for the page of `route` starting at `offset`: `shown`
/// rows of `total`, `limit` per page.  `extra` is a `&k=v…` tail every link
/// keeps (a class filter, a search).  Either link is empty when there is
/// nothing in that direction.
fn pager(
    route: &str,
    offset: usize,
    limit: usize,
    shown: usize,
    total: usize,
    extra: &str,
) -> (String, String) {
    let prev = if offset > 0 {
        format!(
            "<p><a href=\"{route}?offset={}&limit={limit}{extra}\">← Prev {limit}</a></p>",
            offset.saturating_sub(limit)
        )
    } else {
        String::new()
    };
    let next = if offset + shown < total {
        format!(
            "<p><a href=\"{route}?offset={}&limit={limit}{extra}\">Next {limit} →</a></p>",
            offset + limit
        )
    } else {
        String::new()
    };
    (prev, next)
}

/// The search form at the top of a searchable page, offering `modes`, with
/// the current search filled in and `error` (a bad pattern) shown above it.
fn search_form(
    route: &str,
    p: &SearchParams,
    modes: &[SearchMode],
    error: Option<&str>,
    hidden: &str,
) -> String {
    let current = p.mode().unwrap_or_default();
    let options: String = modes
        .iter()
        .map(|m| {
            format!(
                "<option value=\"{}\"{}>{}</option>",
                m.name(),
                if *m == current { " selected" } else { "" },
                m.name()
            )
        })
        .collect();
    let error = error.map_or_else(String::new, |e| format!("<p class=\"warn\">{}</p>", esc(e)));
    let clear = if p.text().is_empty() {
        String::new()
    } else {
        format!(" <a href=\"{route}\">clear</a>")
    };
    format!(
        "{error}<form method=\"get\" action=\"{route}\" class=\"search\">\
         <input type=\"text\" name=\"q\" value=\"{}\" size=\"40\" placeholder=\"search\"> \
         <select name=\"mode\">{options}</select> \
         <label><input type=\"checkbox\" name=\"case\" value=\"1\"{}> match case</label> \
         {hidden}<button type=\"submit\">Search</button>{clear}</form>",
        esc(p.text()),
        if p.case_sensitive() { " checked" } else { "" },
    )
}

/// A page whose search could not be compiled: 400, with the form shown
/// again so the pattern can be corrected.
fn search_error(
    title: &str,
    route: &str,
    p: &SearchParams,
    modes: &[SearchMode],
    error: &HprofError,
) -> (StatusCode, String) {
    (
        StatusCode::BAD_REQUEST,
        page(
            title,
            &search_form(route, p, modes, Some(&error.to_string()), ""),
        ),
    )
}

/// Turn a blocking task's outcome into a response: the rendered page with
/// its status, or 500.
fn rendered(
    result: Result<Result<(StatusCode, String), HprofError>, tokio::task::JoinError>,
) -> axum::response::Response {
    match result {
        Ok(Ok((status, html))) => (status, Html(html)).into_response(),
        Ok(Err(e)) => internal(e).into_response(),
        Err(e) => internal(e).into_response(),
    }
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
    Query(p): Query<SearchParams>,
) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(move || -> Result<(StatusCode, String), HprofError> {
        let q = &state.query;
        let matcher = match p.matcher() {
            Ok(m) => m,
            Err(e) => return Ok(search_error("Histogram", "/histogram", &p, &SearchMode::ALL, &e)),
        };
        let hist = match &matcher {
            Some(m) => q.class_histogram_filtered(m, p.page()),
            None => q.class_histogram(p.page()),
        };

        let mut rows = String::new();
        for entry in &hist.items {
            // Primitive array types link directly to the size-sorted /arrays/:kind view.
            let link = match entry.key {
                ClassKey::PrimArray(t) => {
                    let slug = ArrayKind::from_prim_element_type(t)
                        .map(|k| k.slug())
                        .unwrap_or("unknown");
                    format!("<a href=\"/arrays/{slug}\">{}</a>", esc(&entry.class_name))
                }
                key => format!(
                    "<a href=\"/instances/{:x}\">{}</a>",
                    key.to_u64(),
                    esc(&entry.class_name)
                ),
            };
            rows.push_str(&format!(
                "<tr><td>{link}</td><td class=\"num\">{}</td><td class=\"num\">{}</td></tr>",
                entry.instance_count,
                fmt_bytes(entry.shallow_bytes)
            ));
        }

        let heading = if matcher.is_some() {
            format!(
                "<p>{} of {} classes match “{}”</p>",
                hist.total,
                q.histogram_len(),
                esc(p.text())
            )
        } else {
            let (total, total_bytes) = q.histogram_totals();
            format!(
                "<p>{} classes &nbsp;·&nbsp; {total} total instances &nbsp;·&nbsp; {} total shallow bytes</p>",
                hist.total,
                fmt_bytes(total_bytes)
            )
        };
        let (prev, next) = pager(
            "/histogram",
            p.offset,
            p.limit,
            hist.items.len(),
            hist.total,
            &p.query_suffix(),
        );
        let hidden = format!("<input type=\"hidden\" name=\"limit\" value=\"{}\">", p.limit);
        let content = format!(
            r#"{form}
{heading}
{prev}
<table>
<tr><th>Class</th><th>Instances</th><th>Shallow size</th></tr>
{rows}
</table>
{next}"#,
            form = search_form("/histogram", &p, &SearchMode::ALL, None, &hidden),
        );
        Ok((StatusCode::OK, page("Histogram", &content)))
    })
    .await;
    rendered(result)
}

// ── /allClasses — Class list ──────────────────────────────────────────────────

pub async fn all_classes(
    State(state): State<Arc<AppState>>,
    Query(p): Query<SearchParams>,
) -> impl IntoResponse {
    let result =
        tokio::task::spawn_blocking(move || -> Result<(StatusCode, String), HprofError> {
            let q = &state.query;
            let matcher = match p.matcher() {
                Ok(m) => m,
                Err(e) => {
                    return Ok(search_error(
                        "All Classes",
                        "/allClasses",
                        &p,
                        &SearchMode::ALL,
                        &e,
                    ));
                }
            };
            let classes = match &matcher {
                Some(m) => q.search_classes(m, p.page()),
                None => q.find_classes("", p.page()),
            };

            let mut rows = String::new();
            for c in &classes.items {
                rows.push_str(&format!(
                    "<tr><td>{}</td><td class=\"num\">{}</td></tr>",
                    class_link(c.class_id, &c.name),
                    c.instance_count,
                ));
            }

            let heading = if matcher.is_some() {
                format!("<p>{} classes match “{}”</p>", classes.total, esc(p.text()))
            } else {
                format!("<p>{} classes</p>", classes.total)
            };
            let (prev, next) = pager(
                "/allClasses",
                p.offset,
                p.limit,
                classes.items.len(),
                classes.total,
                &p.query_suffix(),
            );
            let hidden = format!(
                "<input type=\"hidden\" name=\"limit\" value=\"{}\">",
                p.limit
            );
            let content = format!(
                r#"{form}
{heading}{prev}
<table>
<tr><th>Class name</th><th>Instance count</th></tr>
{rows}
</table>{next}"#,
                form = search_form("/allClasses", &p, &SearchMode::ALL, None, &hidden),
            );
            Ok((StatusCode::OK, page("All Classes", &content)))
        })
        .await;
    rendered(result)
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
        let cd = match q.class(class_id)? {
            Some(cd) => cd,
            None => {
                return Ok(page(
                    "Class not found",
                    "<p>No class dump with that ID.</p>",
                ));
            }
        };

        let class_name = q.class_label(cd.class_id);
        let super_name = if cd.super_class_id == 0 {
            "<span class=\"muted\">(none)</span>".to_owned()
        } else {
            let n = q.class_label(cd.super_class_id);
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
            let type_name = BasicType::name_of_code(fd.field_type);
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

        let instances_id = q.class_key_for(&class_name, cd.class_id).to_u64();
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

        let key = ClassKey::from_u64(target_id);
        let title = q.key_name(key);

        // ── Primitive arrays: use size-sorted index directly ──────────────────
        if let ClassKey::PrimArray(prim_type) = key {
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
                let num_elements = entry.byte_size.checked_div(elem_size).unwrap_or(0);
                rows.push_str(&format!(
                    "<tr><td>{}</td><td class=\"num\">{num_elements}</td><td class=\"num\">{}</td></tr>",
                    obj_link(entry.object_id),
                    fmt_bytes(entry.byte_size),
                ));
            }

            let (prev, note) = pager(&format!("/instances/{id_str}"), offset, p.limit, shown, total, "");

            let content = format!(
                "<p>{total} arrays total, showing {offset}–{} by size (largest first)</p>\
                 {prev}\
                 <table><tr><th>Array</th><th>Elements</th><th>Size</th></tr>{rows}</table>\
                 {note}",
                offset + shown
            );
            return Ok(page(&format!("Instances of {title}"), &content));
        }

        // ── Object arrays: per-class index range, sizes parsed per page ───────
        if let ClassKey::ObjArray(_) = key {
            let total = q.instance_count(key);
            let id_size = q.id_size() as u64;
            let offset = p.offset.min(total);
            let shown = p.limit.min(total.saturating_sub(offset));

            let mut rows = String::new();
            for entry in q.class_entries(key).skip(offset).take(shown) {
                let num_elements =
                    match q.parse_entry(&entry.sub_index_entry(TAG_OBJ_ARRAY_DUMP))? {
                        SubRecord::ObjArrayDump(a) => a.num_elements,
                        _ => 0,
                    };
                let byte_size = u64::from(num_elements) * id_size;
                rows.push_str(&format!(
                    "<tr><td>{}</td><td class=\"num\">{num_elements}</td><td class=\"num\">{}</td></tr>",
                    obj_link(entry.object_id),
                    fmt_bytes(byte_size),
                ));
            }

            let (prev, note) = pager(&format!("/instances/{id_str}"), offset, p.limit, shown, total, "");

            let content = format!(
                "<p>{total} arrays total, showing {offset}–{} (by object id)</p>\
                 {prev}\
                 <table><tr><th>Array</th><th>Elements</th><th>Size</th></tr>{rows}</table>\
                 {note}",
                offset + shown
            );
            return Ok(page(&format!("Instances of {title}"), &content));
        }

        // ── Regular instances: per-class index range ──────────────────────────
        let total = q.instance_count(key);
        let offset = p.offset.min(total);
        let shown = p.limit.min(total.saturating_sub(offset));

        let mut rows = String::new();
        for entry in q.class_entries(key).skip(offset).take(shown) {
            rows.push_str(&format!("<tr><td>{}</td></tr>", obj_link(entry.object_id)));
        }

        let (prev, note) = pager(&format!("/instances/{id_str}"), offset, p.limit, shown, total, "");

        let content = format!(
            "<p>Total matching: {total} &nbsp; showing {}-{}</p>\
             {prev}\
             <table><tr><th>Object ID</th></tr>{rows}</table>\
             {note}",
            offset + 1,
            offset + shown
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

/// How many referrers the object page lists (the total is shown as a count).
const REFS_SHOWN: usize = 50;

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
        let reuse_warning = if let Some(dq) = state.diff_query() {
            match (state.query.object(object_id)?, dq.object(object_id)?) {
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
    let record = match q.object(object_id)? {
        Some(r) => r,
        None => return Ok(page("Object not found", "<p>No object with that ID.</p>")),
    };

    let refs_page = q.refs_to(object_id, Page::first(REFS_SHOWN));
    let refs_to = &refs_page.items;
    let root_types = q.root_types_of(object_id);

    let mut refs_html = String::new();
    for from_id in refs_to {
        let type_name = q.object_type_name(*from_id);
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
                    rt.slug(),
                    gc_root_label(*rt)
                )
            })
            .collect();
        format!("<p>{}</p>", names.join(", "))
    };

    let body = match record {
        SubRecord::InstanceDump(inst) => {
            let class_name = q.class_label(inst.class_id);

            let resolved_value_html = match q.resolve_value(inst.object_id) {
                Ok(jv) => render_java_value_banner(&jv),
                Err(_) => String::new(),
            };

            let fields = q.instance_fields(&inst)?;
            let mut field_rows = String::new();
            for f in &fields {
                let val = render_field_value(&f.value, q);
                let type_display = resolve_field_type(f.ty, &f.value, q);
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
            let name = q.class_label(cd.class_id);
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
            let array_type = crate::classes::display_type_name(&q.class_label(arr.array_class_id));
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
<tr><td>Array type</td><td>{}</td></tr>
<tr><td>Length</td><td>{}</td></tr>
</table>
<h2>Elements (first 50)</h2>
<ul>{elems}{more}</ul>"#,
                arr.array_id,
                class_link(arr.array_class_id, &array_type),
                arr.num_elements,
            )
        }
        SubRecord::PrimArrayDump(arr) => {
            let tn = BasicType::name_of_code(arr.element_type);
            let preview = prim_array_preview(q, arr.array_id)?;
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
                let dom_type = q.object_type_name(dom_id);
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
            crate::resolved::Value::String(_, s) => Ok(Some(s)),
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
        Ok(q.prim_array(object_id, Page::new(0, usize::MAX))?
            .map(|w| w.elements.to_strings(false).join(",")))
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

        let limits = RootPathLimits::default();
        let path_result = q.path_to_root(object_id, &limits);

        let content = match path_result.outcome {
            PathOutcome::NotReachable => "<p class=\"muted\">No path to a GC root found. \
                 The object may be unreachable or form a reference cycle \
                 with no live root.</p>"
                .to_owned(),
            PathOutcome::LimitReached => {
                format!(
                    "<p class=\"muted\">Search limit reached ({} nodes \
                     visited) without finding a root. The path may be very long or \
                     the object may be referenced from very many places.</p>",
                    path_result.nodes_visited
                )
            }
            PathOutcome::Found => {
                let path = &path_result.steps;
                let mut steps = String::new();
                let last_idx = path.len().saturating_sub(1);
                for (i, step) in path.iter().enumerate() {
                    let id = step.object_id;
                    let type_name = q.object_type_name(id);
                    let root_types = q.root_types_of(id);

                    let root_badge = if !root_types.is_empty() {
                        let labels: Vec<_> = root_types
                            .iter()
                            .map(|rt| {
                                format!(
                                    "<a href=\"/roots/{}\"><strong>{}</strong></a>",
                                    rt.slug(),
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
            let num_elements = entry.byte_size.checked_div(elem_size).unwrap_or(0);
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

        let (prev, next) = pager(
            &format!("/arrays/{}", kind.slug()),
            offset,
            limit,
            shown,
            total,
            "",
        );

        let content = format!(
            "<p>{total} {display_name} arrays total \
             (showing {offset}–{})</p>\
             {prev}{table}{next}",
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
            let slug = rt.slug();
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
    let rt = match GcRootType::from_slug(&root_type_str) {
        Some(rt) => rt,
        None => return not_found(format!("Unknown root type: {root_type_str}")).into_response(),
    };

    let result = tokio::task::spawn_blocking(move || -> Result<String, HprofError> {
        let q = &state.query;
        let label = gc_root_label(rt);

        let roots = q.gc_roots(rt, Page::new(p.offset, p.limit))?;
        let total = roots.total;
        let mut rows = String::new();
        for root in &roots.items {
            rows.push_str(&format!(
                "<tr><td>{}</td><td>{}</td></tr>",
                obj_link(root.object_id()),
                esc(root.type_name())
            ));
        }

        let (prev, note) = pager(
            &format!("/roots/{root_type_str}"),
            p.offset,
            p.limit,
            roots.items.len(),
            total,
            "",
        );

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

pub async fn threads(
    State(state): State<Arc<AppState>>,
    Query(p): Query<SearchParams>,
) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(move || -> Result<(StatusCode, String), HprofError> {
        let matcher = match p.matcher() {
            Ok(m) => m,
            Err(e) => return Ok(search_error("Threads", "/threads", &p, &SearchMode::ALL, &e)),
        };
        let threads = match &matcher {
            Some(m) => state.query.threads_matching(m)?,
            None => state.query.threads()?,
        };
        let mut rows = String::new();
        for t in &threads {
            let group = t.group.as_deref().map_or_else(
                || "<span class=\"muted\">—</span>".to_owned(),
                esc,
            );
            let status = if t.ended { "ended" } else { "running" };
            rows.push_str(&format!(
                "<tr><td><a href=\"/thread/{s}\">{}</a></td><td>{group}</td><td>{s}</td><td>{status}</td></tr>",
                esc(&t.name),
                s = t.serial,
            ));
        }
        let heading = match &matcher {
            Some(_) if threads.is_empty() => {
                format!("<p class=\"muted\">No threads match “{}”.</p>", esc(p.text()))
            }
            Some(_) => format!("<p>{} threads match “{}”</p>", threads.len(), esc(p.text())),
            None => String::new(),
        };
        let content = format!(
            "{form}{heading}<table><tr><th>Thread name</th><th>Group</th><th>Serial</th><th>Status</th></tr>{rows}</table>",
            form = search_form("/threads", &p, &SearchMode::ALL, None, ""),
        );
        Ok((StatusCode::OK, page("Threads", &content)))
    })
    .await;
    rendered(result)
}

// ── /strings — Search java.lang.String contents ──────────────────────────────

/// The modes `/strings` offers: no fuzzy, see [`crate::query::HeapQuery::search_strings`].
const STRING_MODES: [SearchMode; 3] = [SearchMode::Contains, SearchMode::Exact, SearchMode::Regex];

pub async fn strings(
    State(state): State<Arc<AppState>>,
    Query(p): Query<StringsParams>,
) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(move || -> Result<(StatusCode, String), HprofError> {
        let q = &state.query;
        let search = p.search();
        let window = p.window();
        let hidden = format!(
            "<input type=\"hidden\" name=\"max_scan\" value=\"{}\">\
             <input type=\"hidden\" name=\"max_results\" value=\"{}\">",
            window.max_scan, window.max_results
        );
        let matcher = match search.matcher() {
            Ok(Some(m)) => m,
            Ok(None) => {
                let content = format!(
                    "{}<p class=\"muted\">Searches the text of java.lang.String objects, \
                     {} strings per page.</p>",
                    search_form("/strings", &search, &STRING_MODES, None, &hidden),
                    window.max_scan
                );
                return Ok((StatusCode::OK, page("Strings", &content)));
            }
            Err(e) => return Ok(search_error("Strings", "/strings", &search, &STRING_MODES, &e)),
        };
        let found = match q.search_strings(&matcher, window) {
            Ok(found) => found,
            Err(e @ HprofError::InvalidArgument(_)) => {
                return Ok(search_error("Strings", "/strings", &search, &STRING_MODES, &e));
            }
            Err(e) => return Err(e),
        };

        let mut rows = String::new();
        for s in &found.items {
            rows.push_str(&format!(
                "<tr><td>{}</td><td class=\"num\">{}</td><td>{}</td></tr>",
                obj_link(s.object_id),
                s.length_chars,
                esc(&s.preview),
            ));
        }
        let table = if rows.is_empty() {
            "<p class=\"muted\">No matches in this window.</p>".to_owned()
        } else {
            format!(
                "<table><tr><th>String</th><th>Chars</th><th>Text (first {} chars)</th></tr>{rows}</table>",
                crate::query::StringMatch::PREVIEW_CHARS
            )
        };
        let more = match found.next_cursor {
            Some(next) => format!(
                "<p><a href=\"/strings?cursor={next}&max_scan={}&max_results={}{}\">Continue from string {next} →</a></p>",
                window.max_scan,
                window.max_results,
                search.query_suffix()
            ),
            None => "<p class=\"muted\">End of strings.</p>".to_owned(),
        };
        let content = format!(
            "{form}<p>{} match(es) in strings {}–{} of {}</p>{table}{more}",
            found.items.len(),
            window.cursor,
            window.cursor + found.scanned,
            found.total,
            form = search_form("/strings", &search, &STRING_MODES, None, &hidden),
        );
        Ok((StatusCode::OK, page("Strings", &content)))
    })
    .await;
    rendered(result)
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
        let Some(thread) = q.thread(serial)? else {
            return Ok(page(
                "Thread not found",
                "<p>No thread with that serial.</p>",
            ));
        };

        let mut frames_html = String::new();
        for frame in q.thread_stack(&thread)? {
            frames_html.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td></tr>",
                esc(&frame.method),
                esc(&frame.source_file),
                match frame.line {
                    crate::aux_query::LineNumber::Line(n) => n.to_string(),
                    crate::aux_query::LineNumber::Native => "native".to_owned(),
                    crate::aux_query::LineNumber::Compiled => "compiled".to_owned(),
                    crate::aux_query::LineNumber::Unknown
                    | crate::aux_query::LineNumber::NoInfo => "?".to_owned(),
                }
            ));
        }
        let frames_section = if frames_html.is_empty() {
            "<p class=\"muted\">No stack frames available.</p>".to_owned()
        } else {
            format!(
                "<table><tr><th>Method</th><th>Source</th><th>Line</th></tr>{frames_html}</table>"
            )
        };

        let object = thread.object_id.map_or_else(String::new, obj_link);
        let trace = thread
            .stack_trace_serial
            .map_or_else(String::new, |t| t.to_string());
        let content = format!(
            r#"<table>
<tr><td>Thread name</td><td>{}</td></tr>
<tr><td>Thread group</td><td>{}</td></tr>
<tr><td>Serial</td><td>{serial}</td></tr>
<tr><td>Thread object</td><td>{object}</td></tr>
<tr><td>Trace serial</td><td>{trace}</td></tr>
</table>
<h2>Stack trace</h2>
{frames_section}"#,
            esc(&thread.name),
            esc(thread.group.as_deref().unwrap_or("")),
        );
        Ok(page(&format!("Thread: {}", thread.name), &content))
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
/// resolved value.  Returns an empty string for `Value::Object` (unknown
/// type) and `Value::Null`.
fn render_java_value_banner(jv: &crate::resolved::Value) -> String {
    use crate::resolved::Value as JavaValue;

    let text = match jv {
        JavaValue::String(id, s) => {
            // Truncate very long strings so the page stays readable.
            const MAX: usize = 2000;
            let raw_link = format!(" <a href=\"/object/{id:x}/raw-string\">raw</a>");
            let chars = s.chars().count();
            if chars > MAX {
                // Cut on a character boundary: `s` may hold multi-byte text.
                let head: String = s.chars().take(MAX).collect();
                format!(
                    "\"{}\" <span class=\"muted\">… ({chars} chars total)</span>{raw_link}",
                    esc(&head),
                )
            } else {
                format!("\"{}\"  {raw_link}", esc(s))
            }
        }
        JavaValue::BoxedBoolean(_, b) => b.to_string(),
        JavaValue::BoxedByte(_, b) => format!("{b}"),
        JavaValue::BoxedShort(_, s) => format!("{s}"),
        JavaValue::BoxedCharacter(_, c) => {
            let ch = char::from_u32(u32::from(*c)).unwrap_or('?');
            format!("'{}' (U+{:04X})", esc(&ch.to_string()), c)
        }
        JavaValue::BoxedInt(_, i) => format!("{i}"),
        JavaValue::BoxedLong(_, l) => format!("{l}L"),
        JavaValue::BoxedFloat(_, f) => format!("{f}f"),
        JavaValue::BoxedDouble(_, d) => format!("{d}"),
        // Not a recognised wrapper (or a bare primitive) — no banner.
        _ => return String::new(),
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
                Some(crate::resolved::Value::String(sid, s)) => {
                    format!(
                        "{} <span class=\"muted\">\"{}\"</span> <a href=\"/object/{sid:x}/raw-string\">raw</a>",
                        obj_link(sid),
                        esc(&s)
                    )
                }
                Some(crate::resolved::Value::BoxedInt(oid, n)) => {
                    format!("{} <span class=\"muted\">= {n}</span>", obj_link(oid))
                }
                Some(crate::resolved::Value::BoxedLong(oid, n)) => {
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
fn resolve_field_type(ty: BasicType, value: &FieldValue, q: &crate::query::HeapQuery) -> String {
    if ty != BasicType::Object {
        return ty.java_name().to_owned();
    }
    let id = match value {
        FieldValue::Object(id) if *id != 0 => *id,
        _ => return "Object".to_owned(),
    };
    q.object_type_name(id)
}

// ── Helper: primitive array text ──────────────────────────────────────────────

/// `[v1, v2, …]` for the first `MAX_SHOW` elements, with a "… (n more)" tail.
fn prim_array_preview(q: &crate::query::HeapQuery, array_id: u64) -> Result<String, HprofError> {
    const MAX_SHOW: usize = 256;
    let Some(w) = q.prim_array(array_id, Page::first(MAX_SHOW))? else {
        return Ok(String::new());
    };
    let mut parts = w.elements.to_strings(true);
    if w.has_more() {
        parts.push(format!("… ({} more)", w.total - w.elements.len()));
    }
    Ok(format!("[{}]", parts.join(", ")))
}

// ── Helper: GC root type labels ───────────────────────────────────────────────

/// Display label for a GC root kind (URLs use [`GcRootType::slug`]).
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
        let q = match state.diff_query() {
            Some(q) => q,
            None => {
                return Ok(page(
                    "Object (dump 2)",
                    "<p>No second heap dump configured.</p>",
                ));
            }
        };
        let reuse_warning = match (state.query.object(object_id)?, q.object(object_id)?) {
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

// ── /diff/removed, /diff/added, /diff/common — object lists ──────────────────

/// Shared paging chrome for the three diff list pages.
#[allow(clippy::too_many_arguments)]
fn diff_list_page(
    route: &str,
    title: &str,
    heading: &str,
    columns: &str,
    rows: &str,
    total: usize,
    p: &DiffListParams,
    extra_params: &str,
) -> String {
    let offset = p.offset.min(total);
    let shown = p.limit.min(total.saturating_sub(offset));
    let (prev, next) = pager(route, offset, p.limit, shown, total, extra_params);
    let content = format!(
        "<style>.removed{{color:#cc0000}}</style>\
         <p>{total} {heading}; showing {offset}–{}</p>\
         {prev}<table>{columns}{rows}</table>{next}",
        offset + shown,
    );
    page(title, &content)
}

pub async fn diff_removed(
    State(state): State<Arc<AppState>>,
    Query(p): Query<DiffListParams>,
) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(move || -> Result<String, HprofError> {
        let Some(diff) = state.diff.as_ref().map(|d| &d.heap) else {
            return Ok(page("Diff: Removed", "<p>No diff configured.</p>"));
        };
        let class = p
            .class
            .as_deref()
            .and_then(parse_hex_id)
            .map(ClassKey::from_u64);
        let list = diff.removed(class, Page::new(p.offset, p.limit))?;

        let mut rows = String::new();
        for o in &list.items {
            rows.push_str(&format!(
                "<tr><td>{}</td><td>{}</td></tr>",
                obj_link(o.object_id),
                esc(&o.class_name),
            ));
        }
        let class_param = class
            .map(|c| format!("&class={:x}", c.to_u64()))
            .unwrap_or_default();
        let title = class.map_or_else(
            || "Removed Instances".to_owned(),
            |ck| format!("Removed: {}", diff.key_name(ck)),
        );
        Ok(diff_list_page(
            "/diff/removed",
            &title,
            "removed",
            "<tr><th>Object (dump 1)</th><th>Class</th></tr>",
            &rows,
            list.total,
            &p,
            &class_param,
        ))
    })
    .await;

    match result {
        Ok(Ok(html)) => ok(html).into_response(),
        Ok(Err(e)) => internal(e).into_response(),
        Err(e) => internal(e).into_response(),
    }
}

pub async fn diff_added(
    State(state): State<Arc<AppState>>,
    Query(p): Query<DiffListParams>,
) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(move || -> Result<String, HprofError> {
        let Some(diff) = state.diff.as_ref().map(|d| &d.heap) else {
            return Ok(page("Diff: Added", "<p>No diff configured.</p>"));
        };
        let class = p
            .class
            .as_deref()
            .and_then(parse_hex_id)
            .map(ClassKey::from_u64);
        let list = diff.added(class, Page::new(p.offset, p.limit))?;

        let mut rows = String::new();
        for o in &list.items {
            rows.push_str(&format!(
                "<tr><td><a href=\"/diff/object/{:x}\">0x{:x}</a></td><td>{}</td></tr>",
                o.object_id,
                o.object_id,
                esc(&o.class_name),
            ));
        }
        let class_param = class
            .map(|c| format!("&class={:x}", c.to_u64()))
            .unwrap_or_default();
        let title = class.map_or_else(
            || "Added Instances".to_owned(),
            |ck| format!("Added: {}", diff.key_name(ck)),
        );
        Ok(diff_list_page(
            "/diff/added",
            &title,
            "added",
            "<tr><th>Object (dump 2)</th><th>Class</th></tr>",
            &rows,
            list.total,
            &p,
            &class_param,
        ))
    })
    .await;

    match result {
        Ok(Ok(html)) => ok(html).into_response(),
        Ok(Err(e)) => internal(e).into_response(),
        Err(e) => internal(e).into_response(),
    }
}

pub async fn diff_common(
    State(state): State<Arc<AppState>>,
    Query(p): Query<DiffListParams>,
) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(move || -> Result<String, HprofError> {
        let Some(diff) = state.diff.as_ref().map(|d| &d.heap) else {
            return Ok(page("Diff: Common", "<p>No diff configured.</p>"));
        };
        let class = p.class.as_deref().and_then(parse_hex_id).map(ClassKey::from_u64);
        // changed filter: "0" = unchanged only, "1" = changed only, absent = all
        let changed: Option<bool> = match p.changed.as_deref() {
            Some("0") => Some(false),
            Some("1") => Some(true),
            _ => None,
        };
        let list = diff.common(changed, class, Page::new(p.offset, p.limit))?;

        let mut rows = String::new();
        for o in &list.items {
            let is_changed = diff.object_changed(o.object_id).unwrap_or(false);
            let badge = if is_changed {
                "<span class=\"removed\">changed</span>"
            } else {
                "<span class=\"muted\">unchanged</span>"
            };
            rows.push_str(&format!(
                "<tr><td>{}</td><td><a href=\"/diff/object/{:x}\">dump 2</a></td><td>{badge}</td><td>{}</td></tr>",
                obj_link(o.object_id),
                o.object_id,
                esc(&o.class_name),
            ));
        }

        let changed_param = p.changed.as_deref().map(|v| format!("&changed={v}")).unwrap_or_default();
        let class_param = class.map(|c| format!("&class={:x}", c.to_u64())).unwrap_or_default();
        let extra_params = format!("{changed_param}{class_param}");
        let filter_desc = match (changed, class) {
            (Some(true), None) => " (changed only)".to_owned(),
            (Some(false), None) => " (unchanged only)".to_owned(),
            (None, Some(ck)) => format!(" — {}", diff.key_name(ck)),
            (Some(true), Some(ck)) => format!(" — {} (changed)", diff.key_name(ck)),
            (Some(false), Some(ck)) => format!(" — {} (unchanged)", diff.key_name(ck)),
            (None, None) => String::new(),
        };
        Ok(diff_list_page(
            "/diff/common",
            &format!("Common Instances{filter_desc}"),
            &format!("entries{filter_desc}"),
            "<tr><th>Object (dump 1)</th><th>Dump 2</th><th>Status</th><th>Class</th></tr>",
            &rows,
            list.total,
            &p,
            &extra_params,
        ))
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
        let diff_path = match state.diff_path() {
            Some(p) => p.display().to_string(),
            None => {
                let content = r#"<p>No second heap dump configured.</p>
<p>Restart the server with <code>--diff-hprof &lt;path&gt;</code> to enable heap diff.</p>"#;
                return Ok(page("Heap Diff", content));
            }
        };

        let summary = match state.diff_summary() {
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
            let cid = entry.key.to_u64();
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

        // Served from the pre-sorted retained_by_size index: O(page).
        let ranked = q
            .retained_top(Page::new(p.offset, p.limit))
            .ok_or(HprofError::NotIndexed("retained heap"))?;
        let total_count = ranked.total;

        let mut rows = String::new();
        for (object_id, retained_bytes) in &ranked.items {
            let type_name = q.object_type_name(*object_id);
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

        let (prev, note) = pager(
            "/retained",
            p.offset,
            p.limit,
            ranked.items.len(),
            total_count,
            "",
        );

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
