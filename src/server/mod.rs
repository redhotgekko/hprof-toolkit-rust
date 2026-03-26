//! HTTP server providing a jhat-like interface and an MCP endpoint.
//!
//! Start with [`start_server`].  All heap data is served directly from
//! memory-mapped index files; no dump data is loaded into process memory.
//!
//! ## Routes
//!
//! | Method | Path                   | Description                        |
//! |--------|------------------------|------------------------------------|
//! | GET    | `/`                          | Heap summary                       |
//! | GET    | `/allClasses`                | All classes (paginated)            |
//! | GET    | `/histogram`                 | Class histogram by instance count  |
//! | GET    | `/class/{id}`                | Class detail                       |
//! | GET    | `/instances/{id}`            | Instances of a class (paginated)   |
//! | GET    | `/object/{id}`               | Object detail                      |
//! | GET    | `/object/{id}/raw-string`    | Raw string value (text/plain)      |
//! | GET    | `/object/{id}/raw-array`     | Raw primitive array values (CSV)   |
//! | GET    | `/object/{id}/root-path`     | Path from object to a GC root      |
//! | GET    | `/arrays/{type}`             | Arrays of that type, largest first |
//! | GET    | `/roots`                     | GC root type summary               |
//! | GET    | `/roots/{type}`              | GC roots of a specific type        |
//! | GET    | `/threads`                   | Thread list                        |
//! | GET    | `/thread/{serial}`           | Thread with stack trace            |
//! | POST   | `/mcp`                       | MCP JSON-RPC 2.0 endpoint          |
//!
//! IDs in paths are hex strings (with or without the `0x` prefix).

pub mod handlers;
pub mod mcp;

use crate::diff::{DiffSummary, compute_diff_summary};
use crate::diff_index::DiffIndexPaths;
use crate::heap_parser::SubRecord;
use crate::hprof::HprofError;
use crate::query::HeapQuery;
use axum::{
    Router,
    routing::{get, post},
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

// ── Synthetic ID flags ────────────────────────────────────────────────────────
//
// Histogram entries for object arrays and primitive arrays get synthetic class
// IDs to avoid collisions with real heap object IDs.  JVM heap addresses on
// 64-bit HotSpot never set the top two bits, so using bits 62/63 is safe.

/// High bit set → synthetic ID for an `ObjArrayDump` (element class in low 62 bits).
pub const SYNTHETIC_OBJ_ARRAY: u64 = 1u64 << 63;
/// Bits 62+63 set → synthetic ID for a `PrimArrayDump` (element type in low 8 bits).
pub const SYNTHETIC_PRIM_ARRAY: u64 = (1u64 << 63) | (1u64 << 62);

/// Returns `true` if `id` is a synthetic histogram key (not a real object ID).
pub fn is_synthetic(id: u64) -> bool {
    id & (1u64 << 63) != 0
}

// ── HistogramEntry ────────────────────────────────────────────────────────────

/// A single row in the class histogram.
#[derive(Debug, Clone)]
pub struct HistogramEntry {
    /// Key used in URLs.  May be a real `class_id` or a synthetic ID.
    pub class_id: u64,
    pub class_name: String,
    pub instance_count: u64,
    /// Sum of shallow sizes (instance data bytes only; header not included).
    pub shallow_bytes: u64,
}

// ── AppState ──────────────────────────────────────────────────────────────────

/// Shared state across all HTTP handlers.
pub struct AppState {
    pub query: HeapQuery,
    pub hprof_path: PathBuf,
    /// Histogram computed on first request and then cached for the lifetime of
    /// the server.  Building the histogram requires a full sequential scan of
    /// the combined index, which can be slow for large heaps.
    pub histogram_cache: Mutex<Option<Arc<Vec<HistogramEntry>>>>,
    /// Second heap dump for diff comparisons.  `None` when the server was
    /// started without `--diff-hprof`.
    pub diff_query: Option<HeapQuery>,
    /// Path to the second heap dump (for display purposes).
    pub diff_path: Option<PathBuf>,
    /// Paths to the pre-built diff index files.
    pub diff_index_paths: Option<DiffIndexPaths>,
    /// Diff summary computed on first request and cached.
    pub diff_cache: Mutex<Option<Arc<DiffSummary>>>,
}

impl AppState {
    pub fn new(query: HeapQuery, hprof_path: PathBuf) -> Self {
        Self {
            query,
            hprof_path,
            histogram_cache: Mutex::new(None),
            diff_query: None,
            diff_path: None,
            diff_index_paths: None,
            diff_cache: Mutex::new(None),
        }
    }

    /// Attach a second heap dump and its pre-built diff index paths.
    pub fn with_diff(
        mut self,
        diff_query: HeapQuery,
        diff_path: PathBuf,
        diff_index_paths: DiffIndexPaths,
    ) -> Self {
        self.diff_query = Some(diff_query);
        self.diff_path = Some(diff_path);
        self.diff_index_paths = Some(diff_index_paths);
        self
    }

    /// Return the cached histogram, computing it on first call.
    ///
    /// Must be called from a blocking context (not an async task directly).
    /// Only one histogram computation runs at a time; concurrent callers block
    /// on the mutex until the computation completes.
    pub fn histogram(&self) -> Result<Arc<Vec<HistogramEntry>>, HprofError> {
        let mut guard = self
            .histogram_cache
            .lock()
            .map_err(|_| HprofError::InvalidIndexFile)?;
        if let Some(cached) = guard.as_ref() {
            return Ok(Arc::clone(cached));
        }
        let hist = compute_histogram(&self.query)?;
        *guard = Some(Arc::clone(&hist));
        Ok(hist)
    }

    /// Return the cached diff summary, computing it on first call.
    ///
    /// Returns `None` when no second heap dump has been configured (i.e.
    /// `--diff-hprof` was not supplied at startup).
    /// Must be called from a blocking context.
    pub fn diff(&self) -> Option<Result<Arc<DiffSummary>, HprofError>> {
        let diff_query = self.diff_query.as_ref()?;
        let paths = self.diff_index_paths.as_ref()?;
        let mut guard = self
            .diff_cache
            .lock()
            .map_err(|_| HprofError::InvalidIndexFile)
            .ok()?;
        if let Some(cached) = guard.as_ref() {
            return Some(Ok(Arc::clone(cached)));
        }
        let result = compute_diff_summary(&self.query, diff_query, paths);
        match result {
            Ok(summary) => {
                let arc = Arc::new(summary);
                *guard = Some(Arc::clone(&arc));
                Some(Ok(arc))
            }
            Err(e) => Some(Err(e)),
        }
    }
}

// ── Histogram computation ─────────────────────────────────────────────────────

fn compute_histogram(query: &HeapQuery) -> Result<Arc<Vec<HistogramEntry>>, HprofError> {
    // key → (display_name, instance_count, shallow_bytes)
    let mut counts: HashMap<u64, (String, u64, u64)> = HashMap::new();

    for result in query.iter_objects() {
        match result? {
            SubRecord::InstanceDump(inst) => {
                let entry = counts.entry(inst.class_id).or_insert_with(|| {
                    let name = query
                        .class_name(inst.class_id)
                        .ok()
                        .flatten()
                        .unwrap_or_else(|| format!("0x{:x}", inst.class_id));
                    (name, 0, 0)
                });
                entry.1 += 1;
                entry.2 += inst.data.len() as u64;
            }
            SubRecord::ClassDump(cd) => {
                // Register the class entry so it appears in the histogram / all-classes
                // view, but don't count the ClassDump itself as an instance.
                counts.entry(cd.class_id).or_insert_with(|| {
                    let base = query
                        .class_name(cd.class_id)
                        .ok()
                        .flatten()
                        .unwrap_or_else(|| format!("0x{:x}", cd.class_id));
                    (format!("class {base}"), 0, 0)
                });
            }
            SubRecord::ObjArrayDump(arr) => {
                let key = SYNTHETIC_OBJ_ARRAY | arr.element_class_id;
                let id_size = u64::from(query.id_size());
                let entry = counts.entry(key).or_insert_with(|| {
                    let elem = query
                        .class_name(arr.element_class_id)
                        .ok()
                        .flatten()
                        .unwrap_or_else(|| format!("0x{:x}", arr.element_class_id));
                    (format!("{elem}[]"), 0, 0)
                });
                entry.1 += 1;
                entry.2 += u64::from(arr.num_elements) * id_size;
            }
            SubRecord::PrimArrayDump(arr) => {
                let key = SYNTHETIC_PRIM_ARRAY | u64::from(arr.element_type);
                let elem_sz = u64::from(prim_elem_size(arr.element_type));
                let entry = counts.entry(key).or_insert_with(|| {
                    let name = format!("{}[]", prim_type_name(arr.element_type));
                    (name, 0, 0)
                });
                entry.1 += 1;
                entry.2 += u64::from(arr.num_elements) * elem_sz;
            }
            _ => {} // GC roots — not counted in the histogram
        }
    }

    let mut entries: Vec<HistogramEntry> = counts
        .into_iter()
        .map(
            |(class_id, (class_name, instance_count, shallow_bytes))| HistogramEntry {
                class_id,
                class_name,
                instance_count,
                shallow_bytes,
            },
        )
        .collect();
    entries.sort_by(|a, b| {
        b.instance_count
            .cmp(&a.instance_count)
            .then_with(|| a.class_name.cmp(&b.class_name))
    });

    Ok(Arc::new(entries))
}

// ── Type helpers ──────────────────────────────────────────────────────────────

pub(crate) fn prim_type_name(element_type: u8) -> &'static str {
    match element_type {
        4 => "boolean",
        5 => "char",
        6 => "float",
        7 => "double",
        8 => "byte",
        9 => "short",
        10 => "int",
        11 => "long",
        _ => "unknown",
    }
}

pub(crate) fn prim_elem_size(element_type: u8) -> u8 {
    match element_type {
        4 | 8 => 1,
        5 | 9 => 2,
        6 | 10 => 4,
        7 | 11 => 8,
        _ => 0,
    }
}

/// Map a JVM primitive-array descriptor character to its hprof element-type code.
pub(crate) fn prim_desc_to_elem_type(desc: char) -> Option<u8> {
    match desc {
        'Z' => Some(4),
        'C' => Some(5),
        'F' => Some(6),
        'D' => Some(7),
        'B' => Some(8),
        'S' => Some(9),
        'I' => Some(10),
        'J' => Some(11),
        _ => None,
    }
}

/// Given a class name (dot-notation) and its class object ID, return the
/// histogram key that should be used when navigating to "instances of this
/// class":
///
/// * primitive array (`[B`, `[C`, …) → `SYNTHETIC_PRIM_ARRAY | element_type`
/// * object array (`[Lsome.Class;`) → `SYNTHETIC_OBJ_ARRAY | element_class_id`
///   (falls back to `class_id` if the element class is not found)
/// * regular class → `class_id` (unchanged)
pub(crate) fn instances_key_for_class(
    class_name: &str,
    class_id: u64,
    query: &crate::query::HeapQuery,
) -> u64 {
    let Some(rest) = class_name.strip_prefix('[') else {
        return class_id;
    };
    if rest.len() == 1 {
        if let Some(elem_type) = rest.chars().next().and_then(prim_desc_to_elem_type) {
            return SYNTHETIC_PRIM_ARRAY | u64::from(elem_type);
        }
    } else if let Some(elem_name) = rest.strip_prefix('L').and_then(|s| s.strip_suffix(';'))
        && let Ok(Some(elem_id)) = query.find_class_by_name(elem_name)
    {
        return SYNTHETIC_OBJ_ARRAY | elem_id;
    }
    class_id
}

/// Parse a hex object ID from a path segment (accepts `"0x…"` or plain hex).
pub(crate) fn parse_hex_id(s: &str) -> Option<u64> {
    let trimmed = s.trim_start_matches("0x").trim_start_matches("0X");
    u64::from_str_radix(trimmed, 16).ok()
}

// ── HTML helpers ──────────────────────────────────────────────────────────────

/// Escape a string for safe HTML output.
pub(crate) fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Render a clickable link to an object's detail page.
pub(crate) fn obj_link(id: u64) -> String {
    format!("<a href=\"/object/{id:x}\">0x{id:x}</a>")
}

/// Render a clickable link to a class detail page.
pub(crate) fn class_link(id: u64, name: &str) -> String {
    format!("<a href=\"/class/{id:x}\">{}</a>", esc(name))
}

/// Format a byte count as a human-readable string.
pub(crate) fn fmt_bytes(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1} GB", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1} MB", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1} KB", n as f64 / 1e3)
    } else {
        format!("{n} B")
    }
}

/// Wrap `content` in a full HTML page with navigation links.
pub(crate) fn page(title: &str, content: &str) -> String {
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>hprof: {title}</title>
<style>
body{{font-family:monospace;margin:2em;line-height:1.4}}
nav{{margin-bottom:1em;padding:0.5em;background:#f5f5f5;border:1px solid #ddd}}
nav a{{margin-right:1em}}
table{{border-collapse:collapse;max-width:100%}}
th,td{{border:1px solid #ccc;padding:3px 8px;text-align:left;white-space:nowrap}}
th{{background:#f0f0f0}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}
.muted{{color:#888}}
.warn{{background:#fff3cd;border:1px solid #ffc107;padding:0.5em 1em;margin-bottom:1em}}
a{{color:#0055cc}}
h1{{margin-top:0}}
</style>
</head>
<body>
<nav>
  <a href="/">Summary</a>
  <a href="/histogram">Histogram</a>
  <a href="/allClasses">All Classes</a>
  <a href="/roots">GC Roots</a>
  <a href="/threads">Threads</a>
  <a href="/diff">Diff</a>
</nav>
<h1>{title}</h1>
{content}
</body>
</html>"#
    )
}

// ── Server startup ────────────────────────────────────────────────────────────

/// Bind to `port` and serve the heap analysis interface.
///
/// Blocks until the server is shut down (Ctrl-C or signal).
pub async fn start_server(state: Arc<AppState>, port: u16) -> Result<(), std::io::Error> {
    let app = Router::new()
        .route("/", get(handlers::summary))
        .route("/allClasses", get(handlers::all_classes))
        .route("/histogram", get(handlers::histogram))
        .route("/class/:id", get(handlers::class_detail))
        .route("/instances/:id", get(handlers::instances_of_class))
        .route("/object/:id", get(handlers::object_detail))
        .route("/object/:id/raw-string", get(handlers::raw_string))
        .route("/object/:id/raw-array", get(handlers::raw_prim_array))
        .route("/object/:id/root-path", get(handlers::root_path_page))
        .route("/arrays/:kind", get(handlers::arrays_by_kind))
        .route("/roots", get(handlers::roots_summary))
        .route("/roots/:root_type", get(handlers::roots_by_type))
        .route("/threads", get(handlers::threads))
        .route("/thread/:serial", get(handlers::thread_detail))
        .route("/diff", get(handlers::diff_summary))
        .route("/diff/removed", get(handlers::diff_removed))
        .route("/diff/added", get(handlers::diff_added))
        .route("/diff/common", get(handlers::diff_common))
        .route("/diff/object/:id", get(handlers::diff_object_detail))
        .route("/mcp", post(mcp::handle_mcp))
        .with_state(state);

    let listener = TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    eprintln!("Listening on http://localhost:{port}/");
    axum::serve(listener, app).await
}
