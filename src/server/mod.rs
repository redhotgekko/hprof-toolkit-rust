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
//! | GET    | `/strings`                   | Search java.lang.String contents   |
//! | GET    | `/threads`                   | Thread list                        |
//! | GET    | `/thread/{serial}`           | Thread with stack trace            |
//! | POST   | `/mcp`                       | MCP JSON-RPC 2.0 endpoint          |
//!
//! IDs in paths are hex strings (with or without the `0x` prefix).
//!
//! `/histogram`, `/allClasses`, `/threads` and `/strings` take a search:
//! `q` (the text), `mode` (`contains`, `exact`, `regex` or `fuzzy`; `/strings`
//! has no `fuzzy`) and `case=1` for a case-sensitive match.  Paging links
//! keep the search.  `/strings` pages with `cursor`, `max_scan` and
//! `max_results` instead of `offset`/`limit`, because it reads string
//! contents and the number of matches is not known up front.

pub mod handlers;
pub mod mcp;

#[cfg(test)]
mod tests;

use crate::diff::{DiffSummary, HeapDiff};
use crate::hprof::HprofError;
use crate::query::HeapQuery;
use axum::{
    Router,
    routing::{get, post},
};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;

// ── AppState ──────────────────────────────────────────────────────────────────

/// A second heap dump and the differences against it.
pub struct DiffState {
    pub heap: HeapDiff,
    /// Path to the second heap dump (for display purposes).
    pub path: PathBuf,
}

/// Shared state across all HTTP handlers.
pub struct AppState {
    pub query: Arc<HeapQuery>,
    pub hprof_path: PathBuf,
    /// Set when the server was started with `--diff-hprof`.
    pub diff: Option<DiffState>,
}

impl AppState {
    pub fn new(query: Arc<HeapQuery>, hprof_path: PathBuf) -> Self {
        Self {
            query,
            hprof_path,
            diff: None,
        }
    }

    /// Attach a second heap dump and its diff.
    pub fn with_diff(mut self, heap: HeapDiff, path: PathBuf) -> Self {
        self.diff = Some(DiffState { heap, path });
        self
    }

    /// The second heap dump, when a diff is configured.
    pub fn diff_query(&self) -> Option<&HeapQuery> {
        self.diff.as_ref().map(|d| d.heap.after())
    }

    /// The path of the second heap dump, when a diff is configured.
    pub fn diff_path(&self) -> Option<&PathBuf> {
        self.diff.as_ref().map(|d| &d.path)
    }

    /// The diff summary (computed on first call and cached), or `None` when
    /// no second heap dump was configured.  Call from a blocking context.
    pub fn diff_summary(&self) -> Option<Result<&DiffSummary, HprofError>> {
        self.diff.as_ref().map(|d| d.heap.summary())
    }
}

// ── Type helpers ──────────────────────────────────────────────────────────────

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
    // Titles are built from names in the dump (classes, threads), so they are
    // untrusted text.
    let title = esc(title);
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
form.search{{margin-bottom:1em}}
nav form{{display:inline;margin-left:1em}}
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
  <a href="/strings">Strings</a>
  <a href="/retained">Retained Heap</a>
  <a href="/diff">Diff</a>
  <form method="get" action="/allClasses"><input type="text" name="q" size="20" placeholder="find class"></form>
</nav>
<h1>{title}</h1>
{content}
</body>
</html>"#
    )
}

// ── Server startup ────────────────────────────────────────────────────────────

/// The HTTP routes (HTML UI and `/mcp`) over `state`.  Separate from
/// [`start_server`] so tests can drive it without a socket.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
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
        .route("/strings", get(handlers::strings))
        .route("/thread/:serial", get(handlers::thread_detail))
        .route("/diff", get(handlers::diff_summary))
        .route("/diff/removed", get(handlers::diff_removed))
        .route("/diff/added", get(handlers::diff_added))
        .route("/diff/common", get(handlers::diff_common))
        .route("/diff/object/:id", get(handlers::diff_object_detail))
        .route("/retained", get(handlers::retained_histogram))
        .route("/mcp", post(mcp::handle_mcp))
        .with_state(state)
}

/// Bind to `port` and serve the heap analysis interface.
///
/// Blocks until the server is shut down (Ctrl-C or signal).
pub async fn start_server(state: Arc<AppState>, port: u16) -> Result<(), std::io::Error> {
    let app = router(state);
    let listener = TcpListener::bind(format!("127.0.0.1:{port}")).await?;
    eprintln!("Listening on http://localhost:{port}/");
    axum::serve(listener, app).await
}
