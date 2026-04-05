//! MCP (Model Context Protocol) endpoint — JSON-RPC 2.0 over HTTP POST.
//!
//! **Transport:** Streamable HTTP (MCP spec 2025-03-26).
//!
//! - Client sends `POST /mcp` with `Content-Type: application/json`.
//! - Server responds with `Content-Type: application/json`.
//!
//! ## Supported methods
//!
//! | Method            | Description                                    |
//! |-------------------|------------------------------------------------|
//! | `initialize`      | Handshake; returns server capabilities         |
//! | `initialized`     | No-op notification; always returns `null`      |
//! | `ping`            | Returns `{}`                                   |
//! | `tools/list`      | Returns the list of available tools            |
//! | `tools/call`      | Invoke a named tool and return its result      |
//!
//! ## Available tools
//!
//! | Tool name           | Arguments                      |
//! |---------------------|--------------------------------|
//! | `heap_summary`      | (none)                         |
//! | `class_histogram`   | `top_n?: number`               |
//! | `find_object`       | `id: string` (hex)             |
//! | `find_class_by_name`| `name: string`, `exact?: bool` |
//! | `back_references`   | `id: string` (hex)             |
//! | `gc_roots`          | `type?: string`                |
//! | `resolve_string`    | `id: string` (hex)             |
//! | `heap_diff`         | `top_n?: number`               |
//! | `retained_histogram`| `top_n?: number`               |
//! | `dominator_info`    | `id: string` (hex)             |

use super::{AppState, parse_hex_id, prim_type_name};
use crate::heap_parser::SubRecord;
use crate::hprof::HprofError;
use crate::root_index::GcRootType;
use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;

// ── JSON-RPC types ────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub method: String,
    pub params: Option<Value>,
    pub id: Option<Value>,
}

#[derive(Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
    pub id: Option<Value>,
}

#[derive(Serialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
}

impl JsonRpcResponse {
    fn ok(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            result: Some(result),
            error: None,
            id,
        }
    }

    fn err(id: Option<Value>, code: i32, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0",
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
            }),
            id,
        }
    }
}

// ── Handler ───────────────────────────────────────────────────────────────────

pub async fn handle_mcp(
    State(state): State<Arc<AppState>>,
    Json(req): Json<JsonRpcRequest>,
) -> Response {
    if req.jsonrpc != "2.0" {
        let resp = JsonRpcResponse::err(req.id, -32600, "Invalid Request: jsonrpc must be \"2.0\"");
        return (StatusCode::OK, Json(resp)).into_response();
    }

    let id = req.id;
    let params = req.params;

    let resp = match req.method.as_str() {
        "initialize" => handle_initialize(id, params),
        "initialized" | "notifications/initialized" => {
            // Notification — no response required but we return null result
            JsonRpcResponse::ok(id, Value::Null)
        }
        "ping" => JsonRpcResponse::ok(id, json!({})),
        "tools/list" => handle_tools_list(id),
        "tools/call" => {
            // tools/call needs async + blocking — dispatch to spawn_blocking
            return handle_tools_call(state, id, params).await;
        }
        other => JsonRpcResponse::err(id, -32601, format!("Method not found: {other}")),
    };

    (StatusCode::OK, Json(resp)).into_response()
}

// ── initialize ────────────────────────────────────────────────────────────────

fn handle_initialize(id: Option<Value>, _params: Option<Value>) -> JsonRpcResponse {
    JsonRpcResponse::ok(
        id,
        json!({
            "protocolVersion": "2025-03-26",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "hprof-toolkit",
                "version": "0.1.0"
            }
        }),
    )
}

// ── tools/list ────────────────────────────────────────────────────────────────

fn handle_tools_list(id: Option<Value>) -> JsonRpcResponse {
    JsonRpcResponse::ok(
        id,
        json!({
            "tools": [
                {
                    "name": "heap_summary",
                    "description": "Return a summary of the heap dump: total object count, ID size, thread count, and GC root counts.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                },
                {
                    "name": "class_histogram",
                    "description": "Return the top N classes by instance count, with shallow byte sizes.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "top_n": {
                                "type": "integer",
                                "description": "Number of classes to return (default 50, max 1000)."
                            }
                        }
                    }
                },
                {
                    "name": "find_object",
                    "description": "Find a heap object by its hexadecimal ID and return its type, class, and fields.",
                    "inputSchema": {
                        "type": "object",
                        "required": ["id"],
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Object ID as a hex string (e.g. \"0x1a2b3c\")."
                            }
                        }
                    }
                },
                {
                    "name": "find_class_by_name",
                    "description": "Find heap classes whose name contains (or equals) the given string.",
                    "inputSchema": {
                        "type": "object",
                        "required": ["name"],
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Class name to search for (substring match by default)."
                            },
                            "exact": {
                                "type": "boolean",
                                "description": "If true, require an exact match."
                            }
                        }
                    }
                },
                {
                    "name": "back_references",
                    "description": "Return the object IDs that hold a direct reference to the specified object.",
                    "inputSchema": {
                        "type": "object",
                        "required": ["id"],
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Object ID as a hex string."
                            }
                        }
                    }
                },
                {
                    "name": "gc_roots",
                    "description": "List GC roots, optionally filtered by root type. Returns up to 200 roots.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "description": "Root type: unknown | jni_global | jni_local | java_frame | native_stack | sticky_class | thread_block | monitor_used | thread_object. Omit for all types."
                            }
                        }
                    }
                },
                {
                    "name": "resolve_string",
                    "description": "Resolve a java.lang.String object to its string content.",
                    "inputSchema": {
                        "type": "object",
                        "required": ["id"],
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Object ID of the String instance as a hex string."
                            }
                        }
                    }
                },
                {
                    "name": "heap_diff",
                    "description": "Compare two heap snapshots. Returns per-class counts of added, removed, and common objects. Requires the server to have been started with --diff-hprof.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "top_n": {
                                "type": "integer",
                                "description": "Number of classes to return sorted by most changed (default 50, max 1000)."
                            }
                        }
                    }
                },
                {
                    "name": "retained_histogram",
                    "description": "Return the top N objects by retained heap size (the memory freed if that object were collected). Requires the dominator/retained index to have been built.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "top_n": {
                                "type": "integer",
                                "description": "Number of objects to return (default 50, max 1000)."
                            }
                        }
                    }
                },
                {
                    "name": "dominator_info",
                    "description": "Show retained size and immediate dominator for an object, and list the objects it directly dominates (its dominatees). Requires the dominator/retained index.",
                    "inputSchema": {
                        "type": "object",
                        "required": ["id"],
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Object ID as a hex string (e.g. \"0x1a2b3c\")."
                            }
                        }
                    }
                }
            ]
        }),
    )
}

// ── tools/call ────────────────────────────────────────────────────────────────

async fn handle_tools_call(
    state: Arc<AppState>,
    id: Option<Value>,
    params: Option<Value>,
) -> Response {
    let tool_name = match params
        .as_ref()
        .and_then(|p| p.get("name"))
        .and_then(|v| v.as_str())
        .map(str::to_owned)
    {
        Some(n) => n,
        None => {
            let resp = JsonRpcResponse::err(id, -32602, "Missing tool name in params");
            return (StatusCode::OK, Json(resp)).into_response();
        }
    };
    let args = params
        .and_then(|p| p.get("arguments").cloned())
        .unwrap_or_else(|| json!({}));

    let id_clone = id.clone();
    let result =
        tokio::task::spawn_blocking(move || dispatch_tool(&state, &tool_name, &args)).await;

    let resp = match result {
        Ok(Ok(content)) => JsonRpcResponse::ok(
            id_clone,
            json!({ "content": [{ "type": "text", "text": content }] }),
        ),
        Ok(Err(e)) => JsonRpcResponse::err(id_clone, -32603, format!("Tool error: {e}")),
        Err(e) => JsonRpcResponse::err(id_clone, -32603, format!("Internal error: {e}")),
    };

    (StatusCode::OK, Json(resp)).into_response()
}

// ── Tool dispatch (runs in spawn_blocking) ────────────────────────────────────

fn dispatch_tool(state: &AppState, tool_name: &str, args: &Value) -> Result<String, HprofError> {
    match tool_name {
        "heap_summary" => tool_heap_summary(state),
        "class_histogram" => tool_class_histogram(state, args),
        "find_object" => tool_find_object(state, args),
        "find_class_by_name" => tool_find_class_by_name(state, args),
        "back_references" => tool_back_references(state, args),
        "gc_roots" => tool_gc_roots(state, args),
        "resolve_string" => tool_resolve_string(state, args),
        "heap_diff" => tool_heap_diff(state, args),
        "retained_histogram" => tool_retained_histogram(state, args),
        "dominator_info" => tool_dominator_info(state, args),
        other => Ok(format!("Unknown tool: {other}")),
    }
}

// ── heap_summary ──────────────────────────────────────────────────────────────

fn tool_heap_summary(state: &AppState) -> Result<String, HprofError> {
    let q = &state.query;
    let object_count = q.object_count();
    let id_size = q.id_size();
    let thread_count = q.iter_threads().count();
    let ref_count = q.ref_count();

    let mut root_counts = Vec::new();
    for rt in GcRootType::ALL {
        let n = q.iter_roots(rt).count();
        if n > 0 {
            root_counts.push(format!("{}: {n}", gc_root_label(rt)));
        }
    }

    Ok(format!(
        "Heap dump: {}\nID size: {id_size} bytes\nTotal sub-records: {object_count}\nThreads: {thread_count}\nReferences indexed: {ref_count}\nGC roots:\n  {}",
        state.hprof_path.display(),
        root_counts.join("\n  ")
    ))
}

// ── class_histogram ───────────────────────────────────────────────────────────

fn tool_class_histogram(state: &AppState, args: &Value) -> Result<String, HprofError> {
    let top_n = args
        .get("top_n")
        .and_then(|v| v.as_u64())
        .unwrap_or(50)
        .min(1000) as usize;

    let hist = state.histogram()?;
    let mut out = format!(
        "{} classes total\n\n{:<12} {:>12} {:>14}  Class\n",
        hist.len(),
        "Instances",
        "Shallow",
        ""
    );
    out.push_str(&"-".repeat(70));
    out.push('\n');

    for entry in hist.iter().take(top_n) {
        out.push_str(&format!(
            "{:<12} {:>12} {:>14}  {}\n",
            entry.instance_count,
            format_bytes(entry.shallow_bytes),
            "",
            entry.class_name
        ));
    }
    Ok(out)
}

// ── find_object ───────────────────────────────────────────────────────────────

fn tool_find_object(state: &AppState, args: &Value) -> Result<String, HprofError> {
    let id_str = args
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or(HprofError::InvalidIndexFile)?;
    let object_id = parse_hex_id(id_str).ok_or(HprofError::InvalidIndexFile)?;

    let q = &state.query;
    let record = match q.find(object_id)? {
        Some(r) => r,
        None => return Ok(format!("Object 0x{object_id:x} not found.")),
    };

    let refs_to = q.refs_to(object_id);
    let root_types = q.root_types_of(object_id);

    let mut out = match &record {
        SubRecord::InstanceDump(inst) => {
            let class_name = q
                .class_name(inst.class_id)
                .ok()
                .flatten()
                .unwrap_or_else(|| format!("0x{:x}", inst.class_id));
            let fields = q.instance_fields(inst)?;
            let mut field_lines = Vec::new();
            for f in &fields {
                let val = format_field_value_text(&f.value);
                field_lines.push(format!(
                    "  {} {}: {val}",
                    field_type_name(f.field_type),
                    f.name
                ));
            }
            format!(
                "Instance: {class_name}\nID: 0x{:x}\nData size: {} bytes\nFields:\n{}",
                inst.object_id,
                inst.data.len(),
                field_lines.join("\n")
            )
        }
        SubRecord::ClassDump(cd) => {
            let name = q
                .class_name(cd.class_id)
                .ok()
                .flatten()
                .unwrap_or_else(|| format!("0x{:x}", cd.class_id));
            format!(
                "Class: {name}\nID: 0x{:x}\nInstance size: {} bytes",
                cd.class_id, cd.instance_size
            )
        }
        SubRecord::ObjArrayDump(arr) => {
            let elem = q
                .class_name(arr.element_class_id)
                .ok()
                .flatten()
                .unwrap_or_else(|| format!("0x{:x}", arr.element_class_id));
            format!(
                "Object array: {elem}[{}]\nID: 0x{:x}",
                arr.num_elements, arr.array_id
            )
        }
        SubRecord::PrimArrayDump(arr) => {
            let tn = prim_type_name(arr.element_type);
            format!(
                "Primitive array: {tn}[{}]\nID: 0x{:x}",
                arr.num_elements, arr.array_id
            )
        }
        _ => format!("GC root: 0x{object_id:x}"),
    };

    if !root_types.is_empty() {
        let labels: Vec<_> = root_types.iter().map(|rt| gc_root_label(*rt)).collect();
        out.push_str(&format!("\nGC roots: {}", labels.join(", ")));
    }
    if !refs_to.is_empty() {
        let ids: Vec<_> = refs_to.iter().map(|id| format!("0x{id:x}")).collect();
        out.push_str(&format!("\nReferenced by: {}", ids.join(", ")));
    }
    if q.has_retained_heap() {
        if let Some(retained) = q.retained_size(object_id) {
            out.push_str(&format!("\nRetained size: {}", format_bytes(retained)));
        }
        match q.dominator_of(object_id) {
            Some(0) => out.push_str("\nDominator: (GC root — virtual root)"),
            Some(dom_id) => {
                let dom_type = q
                    .object_type_name(dom_id)
                    .unwrap_or_else(|_| "?".to_owned());
                out.push_str(&format!("\nDominator: 0x{dom_id:x}  {dom_type}"));
            }
            None => {}
        }
    }
    Ok(out)
}

// ── find_class_by_name ────────────────────────────────────────────────────────

fn tool_find_class_by_name(state: &AppState, args: &Value) -> Result<String, HprofError> {
    let name = args
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or(HprofError::InvalidIndexFile)?;
    let exact = args.get("exact").and_then(|v| v.as_bool()).unwrap_or(false);

    let hist = state.histogram()?;
    let mut results = Vec::new();
    for entry in hist.iter() {
        if super::is_synthetic(entry.class_id) {
            continue;
        }
        let matches = if exact {
            entry.class_name == name
        } else {
            entry.class_name.contains(name)
        };
        if matches {
            results.push(format!(
                "0x{:x}  {}  ({} instances)",
                entry.class_id, entry.class_name, entry.instance_count
            ));
        }
    }

    if results.is_empty() {
        Ok(format!("No classes found matching \"{name}\"."))
    } else {
        Ok(format!(
            "{} class(es) matching \"{}\":\n{}",
            results.len(),
            name,
            results.join("\n")
        ))
    }
}

// ── back_references ───────────────────────────────────────────────────────────

fn tool_back_references(state: &AppState, args: &Value) -> Result<String, HprofError> {
    let id_str = args
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or(HprofError::InvalidIndexFile)?;
    let object_id = parse_hex_id(id_str).ok_or(HprofError::InvalidIndexFile)?;

    let refs_to = state.query.refs_to(object_id);
    if refs_to.is_empty() {
        return Ok(format!("No back-references to 0x{object_id:x}."));
    }

    let q = &state.query;
    let mut lines = vec![format!(
        "{} object(s) reference 0x{object_id:x}:",
        refs_to.len()
    )];
    for from_id in &refs_to {
        let type_name = q
            .object_type_name(*from_id)
            .unwrap_or_else(|_| "?".to_owned());
        lines.push(format!("  0x{from_id:x}  {type_name}"));
    }
    Ok(lines.join("\n"))
}

// ── gc_roots ──────────────────────────────────────────────────────────────────

fn tool_gc_roots(state: &AppState, args: &Value) -> Result<String, HprofError> {
    let filter = args.get("type").and_then(|v| v.as_str());
    let q = &state.query;

    let root_types: &[GcRootType] = if let Some(s) = filter {
        match parse_root_type(s) {
            Some(rt) => {
                // We need a static slice; use a match to return different static slices
                return tool_gc_roots_for_type(q, rt);
            }
            None => return Ok(format!("Unknown root type \"{s}\".")),
        }
    } else {
        &GcRootType::ALL
    };

    let mut out = String::new();
    let mut total = 0usize;
    for rt in root_types {
        let mut count = 0usize;
        for entry in q.iter_roots(*rt) {
            count += 1;
            if count <= 50 {
                let label = gc_root_label(*rt);
                let type_name = q
                    .object_type_name(entry.object_id)
                    .unwrap_or_else(|_| "?".to_owned());
                out.push_str(&format!(
                    "  [{label}] 0x{:x}  {type_name}\n",
                    entry.object_id
                ));
            }
        }
        if count > 0 {
            total += count;
        }
    }
    Ok(format!("{total} GC roots total:\n{out}"))
}

fn tool_gc_roots_for_type(
    q: &crate::query::HeapQuery,
    rt: GcRootType,
) -> Result<String, HprofError> {
    let label = gc_root_label(rt);
    let mut lines = Vec::new();
    let mut total = 0usize;
    for entry in q.iter_roots(rt) {
        total += 1;
        if total <= 200 {
            let type_name = q
                .object_type_name(entry.object_id)
                .unwrap_or_else(|_| "?".to_owned());
            lines.push(format!("  0x{:x}  {type_name}", entry.object_id));
        }
    }
    Ok(format!("{total} {label} roots:\n{}", lines.join("\n")))
}

// ── resolve_string ────────────────────────────────────────────────────────────

fn tool_resolve_string(state: &AppState, args: &Value) -> Result<String, HprofError> {
    let id_str = args
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or(HprofError::InvalidIndexFile)?;
    let object_id = parse_hex_id(id_str).ok_or(HprofError::InvalidIndexFile)?;

    match state.query.resolve_value(object_id)? {
        crate::heap_query::JavaValue::String(_, s) => {
            Ok(format!("String at 0x{object_id:x}: \"{s}\""))
        }
        other => Ok(format!("Object 0x{object_id:x} is not a String: {other:?}")),
    }
}

// ── heap_diff ─────────────────────────────────────────────────────────────────

fn tool_heap_diff(state: &AppState, args: &Value) -> Result<String, HprofError> {
    let top_n = args
        .get("top_n")
        .and_then(|v| v.as_u64())
        .unwrap_or(50)
        .min(1000) as usize;

    let summary = match state.diff() {
        None => {
            return Ok(
                "Heap diff not available. Restart the server with --diff-hprof <path>.".to_owned(),
            );
        }
        Some(r) => r?,
    };

    let diff_path = state
        .diff_path
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_default();

    let mut out = format!(
        "Heap diff: {} vs {}\n\nBefore: {} objects\nAfter:  {} objects\nAdded:  +{}\nRemoved: -{}\nCommon unchanged: {}\nCommon changed:   {}\n\n",
        state.hprof_path.display(),
        diff_path,
        summary.total_before,
        summary.total_after,
        summary.total_added,
        summary.total_removed,
        summary.total_common_unchanged,
        summary.total_common_changed,
    );

    out.push_str(&format!(
        "{} classes with changes (showing top {}):\n\n",
        summary
            .by_class
            .iter()
            .filter(|e| e.count_added > 0 || e.count_removed > 0)
            .count(),
        top_n,
    ));
    out.push_str(&format!(
        "{:<10} {:<10} {:<10} {:<12} {:<12}  Class\n",
        "Net", "Added", "Removed", "Unchanged", "Changed"
    ));
    out.push_str(&"-".repeat(80));
    out.push('\n');

    for entry in summary.by_class.iter().take(top_n) {
        let net = entry.net_change();
        let net_str = if net > 0 {
            format!("+{net}")
        } else {
            format!("{net}")
        };
        out.push_str(&format!(
            "{:<10} {:<10} {:<10} {:<12} {:<12}  {}\n",
            net_str,
            entry.count_added,
            entry.count_removed,
            entry.count_common_unchanged,
            entry.count_common_changed,
            entry.class_name,
        ));
    }

    Ok(out)
}

// ── Text formatting helpers ───────────────────────────────────────────────────

fn format_bytes(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}G", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        format!("{n}B")
    }
}

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

fn format_field_value_text(v: &crate::heap_parser::FieldValue) -> String {
    use crate::heap_parser::FieldValue;
    match v {
        FieldValue::Object(0) => "null".to_owned(),
        FieldValue::Object(id) => format!("0x{id:x}"),
        FieldValue::Bool(b) => b.to_string(),
        FieldValue::Char(c) => format!("'{}'", char::from_u32(u32::from(*c)).unwrap_or('?')),
        FieldValue::Float(f) => format!("{f}f"),
        FieldValue::Double(d) => format!("{d}"),
        FieldValue::Byte(b) => b.to_string(),
        FieldValue::Short(s) => s.to_string(),
        FieldValue::Int(i) => i.to_string(),
        FieldValue::Long(l) => format!("{l}L"),
    }
}

fn gc_root_label(rt: GcRootType) -> &'static str {
    super::handlers::gc_root_label(rt)
}

fn parse_root_type(s: &str) -> Option<GcRootType> {
    super::handlers::parse_root_type(s)
}

// ── retained_histogram ────────────────────────────────────────────────────────

fn tool_retained_histogram(state: &AppState, args: &Value) -> Result<String, HprofError> {
    let top_n = args
        .get("top_n")
        .and_then(|v| v.as_u64())
        .unwrap_or(50)
        .min(1000) as usize;

    let q = &state.query;
    if !q.has_retained_heap() {
        return Ok(
            "Retained heap index not available. Run `hprof-toolkit index` to build it.".to_owned(),
        );
    }

    let iter = q.iter_retained().ok_or(HprofError::InvalidIndexFile)?;
    let mut entries: Vec<(u64, u64)> = iter.map(|(id, ret)| (ret, id)).collect();
    entries.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    let total = entries.len();

    let mut out = format!(
        "{total} reachable objects with retained size data (showing top {top_n}):\n\n\
         {:<12} {:>14}  {:<20}  Dominator\n",
        "Object ID", "Retained", "Type"
    );
    out.push_str(&"-".repeat(80));
    out.push('\n');

    for (retained_bytes, object_id) in entries.iter().take(top_n) {
        let type_name = q
            .object_type_name(*object_id)
            .unwrap_or_else(|_| "?".to_owned());
        let dom_str = match q.dominator_of(*object_id) {
            None | Some(0) => "(root)".to_owned(),
            Some(dom_id) => format!("0x{dom_id:x}"),
        };
        out.push_str(&format!(
            "0x{:<10x} {:>14}  {:<20}  {dom_str}\n",
            object_id,
            format_bytes(*retained_bytes),
            if type_name.len() > 20 {
                &type_name[..20]
            } else {
                &type_name
            }
        ));
    }
    Ok(out)
}

// ── dominator_info ────────────────────────────────────────────────────────────

fn tool_dominator_info(state: &AppState, args: &Value) -> Result<String, HprofError> {
    let id_str = args
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or(HprofError::InvalidIndexFile)?;
    let object_id = parse_hex_id(id_str).ok_or(HprofError::InvalidIndexFile)?;

    let q = &state.query;
    if !q.has_retained_heap() {
        return Ok(
            "Retained heap index not available. Run `hprof-toolkit index` to build it.".to_owned(),
        );
    }

    let type_name = q
        .object_type_name(object_id)
        .unwrap_or_else(|_| "?".to_owned());

    let retained = q.retained_size(object_id);
    let dominator = q.dominator_of(object_id);

    if retained.is_none() && dominator.is_none() {
        return Ok(format!(
            "Object 0x{object_id:x} is not in the retained heap index (unreachable or not found)."
        ));
    }

    let retained_str = retained
        .map(format_bytes)
        .unwrap_or_else(|| "n/a".to_owned());

    let dom_str = match dominator {
        None => "n/a".to_owned(),
        Some(0) => "(GC root — virtual root)".to_owned(),
        Some(dom_id) => {
            let dom_type = q
                .object_type_name(dom_id)
                .unwrap_or_else(|_| "?".to_owned());
            format!("0x{dom_id:x}  {dom_type}")
        }
    };

    // Find objects directly dominated by object_id (dominatees).
    // We scan the full dominator index for entries whose dominator == object_id.
    let mut dominatees: Vec<(u64, u64)> = q
        .iter_retained()
        .ok_or(HprofError::InvalidIndexFile)?
        .filter_map(|(child_id, child_ret)| {
            if q.dominator_of(child_id) == Some(object_id) {
                Some((child_ret, child_id))
            } else {
                None
            }
        })
        .collect();
    dominatees.sort_unstable_by(|a, b| b.0.cmp(&a.0));

    let mut out = format!(
        "Object 0x{object_id:x}  {type_name}\n\
         Retained size: {retained_str}\n\
         Immediate dominator: {dom_str}\n\
         Directly dominates: {} object(s)\n",
        dominatees.len()
    );

    if !dominatees.is_empty() {
        out.push_str("\nTop dominatees (by retained size):\n");
        for (ret, child_id) in dominatees.iter().take(20) {
            let child_type = q
                .object_type_name(*child_id)
                .unwrap_or_else(|_| "?".to_owned());
            out.push_str(&format!(
                "  0x{child_id:<10x}  {:>14}  {child_type}\n",
                format_bytes(*ret)
            ));
        }
        if dominatees.len() > 20 {
            out.push_str(&format!("  … {} more\n", dominatees.len() - 20));
        }
    }

    Ok(out)
}
