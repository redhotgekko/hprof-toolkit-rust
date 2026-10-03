//! Model Context Protocol server.
//!
//! [`McpServer`] is a transport-agnostic JSON-RPC core: give it one decoded
//! message and it returns the reply (or `None` for a notification).  Two thin
//! transports sit on top of it: [`serve_stdio`] (newline-delimited JSON on
//! stdin/stdout, the way MCP clients launch a local server) and the HTTP
//! handler [`handle_mcp`] mounted at `/mcp`.  Neither transport contains
//! protocol logic, so tests drive the core directly with no socket and no
//! files.
//!
//! On the stdio transport stdout carries only protocol messages; every
//! diagnostic goes to stderr.
//!
//! Tool failures (bad arguments, unknown ids, a missing index) are returned
//! as results with `isError: true` and a message the model can act on.
//! JSON-RPC errors are reserved for malformed protocol traffic.

mod args;
mod tools;

use crate::server::AppState;
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};
use std::io::{BufRead, Write};
use std::sync::Arc;

use args::Args;
use tools::Tool;

/// Protocol revisions this server speaks, newest first.
const SUPPORTED_VERSIONS: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];

const INSTRUCTIONS: &str = "\
Explore a Java heap dump (.hprof) without loading it into memory.

Ids are object addresses written as \"0x…\" hex strings; pass them back exactly as shown \
(a hex string with or without 0x, or a decimal JSON number). Every list is paged: use `offset` \
and `limit`, and continue while `has_more` is true (`next_offset` is the next offset). Lists of \
classes and threads take a search (`name` with `mode`: `contains`, `exact`, `regex` or `fuzzy`), \
so narrow class_histogram with `name` rather than paging through it. Sizes are \
bytes. Shallow size is an object's own data; retained size is what would be freed if it were \
collected and needs the retained index.

Suggested workflow:
1. heap_summary — scale, roots, which indexes exist.
2. class_histogram (or retained_top) — what dominates the heap.
3. instances_of_class / find_classes — pick concrete objects.
4. object — read fields; array_contents / object_string for data.
5. path_to_gc_root, references_to, references_from — why an object is alive and who holds it.
6. threads, thread_stack — what was running. heap_diff_summary / heap_diff_objects compare two \
dumps when a second dump is configured.

Tool errors come back with isError set and say what to do next.";

const LEAK_PROMPT: &str = "\
Investigate a suspected memory leak in this heap dump.

1. Call heap_summary to see the size of the heap and whether retained sizes are available.
2. Call retained_top (or class_histogram sorted by shallow_bytes) and note the biggest objects and \
classes. Ignore JDK internals unless they hold application data; class_histogram with `name` \
(a package prefix, or a regex with `mode`) focuses on the application's own classes.
3. For the top suspects, call object, then dominated_by to see what each keeps alive.
4. For a suspect that should have been collected, call path_to_gc_root and read the chain from the \
root down: the field names show which owner is holding it. Check for weak/soft `referent` edges.
5. If two dumps are configured, call heap_diff_summary and heap_diff_objects (kind `added`) to see \
which classes grew.
6. Report: what is leaking, the reference chain that keeps it alive, and how many bytes it retains.";

/// The protocol core.  Cheap to create; holds only a handle to the shared state.
pub struct McpServer {
    state: Arc<AppState>,
    tools: Vec<Tool>,
}

impl McpServer {
    pub fn new(state: Arc<AppState>) -> Self {
        Self {
            state,
            tools: tools::all(),
        }
    }

    /// Handle one decoded JSON-RPC message (or a batch).
    ///
    /// Returns the reply to send, or `None` when nothing is to be sent: a
    /// notification, or a reply from the client to a request we never made.
    pub fn handle(&self, message: &Value) -> Option<Value> {
        if let Some(batch) = message.as_array() {
            if batch.is_empty() {
                return Some(error_reply(
                    Value::Null,
                    -32600,
                    "Invalid Request: empty batch",
                ));
            }
            let replies: Vec<Value> = batch.iter().filter_map(|m| self.handle(m)).collect();
            return (!replies.is_empty()).then_some(Value::Array(replies));
        }
        let Some(obj) = message.as_object() else {
            return Some(error_reply(
                Value::Null,
                -32600,
                "Invalid Request: expected a JSON object",
            ));
        };
        // A response to a request we never sent: ignore.
        if !obj.contains_key("method") && (obj.contains_key("result") || obj.contains_key("error"))
        {
            return None;
        }
        let id = obj.get("id").cloned();
        let Some(method) = obj.get("method").and_then(Value::as_str) else {
            return Some(error_reply(
                id.unwrap_or(Value::Null),
                -32600,
                "Invalid Request: missing method",
            ));
        };
        if obj.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
            return Some(error_reply(
                id.unwrap_or(Value::Null),
                -32600,
                "Invalid Request: jsonrpc must be \"2.0\"",
            ));
        }
        let params = obj.get("params").cloned().unwrap_or(Value::Null);

        // Notifications carry no id and never get a reply.
        let id = id?;
        let outcome = match method {
            "initialize" => Ok(self.initialize(&params)),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(self.tools_list()),
            "tools/call" => self.tools_call(&params),
            "prompts/list" => Ok(json!({"prompts": [{
                "name": "investigate_memory_leak",
                "description": "Step-by-step workflow for finding what is leaking memory in this heap dump.",
                "arguments": [],
            }]})),
            "prompts/get" => self.prompts_get(&params),
            "resources/list" => Ok(json!({"resources": [{
                "uri": "hprof://summary",
                "name": "Heap summary",
                "description": "Counts, GC roots and index availability, as heap_summary returns them.",
                "mimeType": "application/json",
            }]})),
            "resources/templates/list" => Ok(json!({"resourceTemplates": []})),
            "resources/read" => self.resources_read(&params),
            "logging/setLevel" => Ok(json!({})),
            other => Err((-32601, format!("Method not found: {other}"))),
        };
        Some(match outcome {
            Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
            Err((code, message)) => error_reply(id, code, &message),
        })
    }

    fn initialize(&self, params: &Value) -> Value {
        let requested = params.get("protocolVersion").and_then(Value::as_str);
        let version = requested
            .filter(|v| SUPPORTED_VERSIONS.contains(v))
            .unwrap_or(SUPPORTED_VERSIONS[0]);
        json!({
            "protocolVersion": version,
            "capabilities": {"tools": {}, "prompts": {}, "resources": {}},
            "serverInfo": {"name": "hprof-toolkit", "version": env!("CARGO_PKG_VERSION")},
            "instructions": self.instructions(),
        })
    }

    /// The workflow text plus what is available in this session.
    fn instructions(&self) -> String {
        let q = &self.state.query;
        format!(
            "{INSTRUCTIONS}\n\nThis session: {} — {} objects; retained sizes {}; second dump for diffs {}.",
            self.state.hprof_path.display(),
            q.histogram_totals().0,
            if q.has_retained_heap() {
                "available (retained_top, dominated_by work)"
            } else {
                "NOT built (retained_top and dominated_by will report that; run `hprof-toolkit index`)"
            },
            if self.state.diff.is_some() {
                "configured"
            } else {
                "not configured"
            },
        )
    }

    fn tools_list(&self) -> Value {
        let tools: Vec<Value> = self
            .tools
            .iter()
            .map(|t| json!({"name": t.name, "description": t.description, "inputSchema": (t.schema)()}))
            .collect();
        json!({"tools": tools})
    }

    fn tools_call(&self, params: &Value) -> Result<Value, (i32, String)> {
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .ok_or((-32602, "Invalid params: `name` is required".to_owned()))?;
        let tool = self
            .tools
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| (-32602, format!("Unknown tool: {name}")))?;
        let empty = Value::Null;
        let arguments = params.get("arguments").unwrap_or(&empty);
        Ok(match (tool.run)(&self.state, &Args::new(arguments)) {
            Ok(out) => {
                let text = format!(
                    "{}\n{}",
                    out.summary,
                    serde_json::to_string_pretty(&out.data).unwrap_or_default()
                );
                json!({
                    "content": [{"type": "text", "text": text}],
                    "structuredContent": out.data,
                    "isError": false,
                })
            }
            Err(e) => json!({
                "content": [{"type": "text", "text": e.0}],
                "isError": true,
            }),
        })
    }

    fn prompts_get(&self, params: &Value) -> Result<Value, (i32, String)> {
        match params.get("name").and_then(Value::as_str) {
            Some("investigate_memory_leak") => Ok(json!({
                "description": "Find what is leaking memory in this heap dump.",
                "messages": [{"role": "user", "content": {"type": "text", "text": LEAK_PROMPT}}],
            })),
            Some(other) => Err((-32602, format!("Unknown prompt: {other}"))),
            None => Err((-32602, "Invalid params: `name` is required".to_owned())),
        }
    }

    fn resources_read(&self, params: &Value) -> Result<Value, (i32, String)> {
        match params.get("uri").and_then(Value::as_str) {
            Some("hprof://summary") => {
                let tool = self.tools.iter().find(|t| t.name == "heap_summary");
                let out = tool
                    .and_then(|t| (t.run)(&self.state, &Args::new(&Value::Null)).ok())
                    .ok_or((-32603, "summary unavailable".to_owned()))?;
                Ok(json!({"contents": [{
                    "uri": "hprof://summary",
                    "mimeType": "application/json",
                    "text": serde_json::to_string_pretty(&out.data).unwrap_or_default(),
                }]}))
            }
            Some(other) => Err((-32002, format!("Resource not found: {other}"))),
            None => Err((-32602, "Invalid params: `uri` is required".to_owned())),
        }
    }
}

fn error_reply(id: Value, code: i32, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

// ── stdio transport ───────────────────────────────────────────────────────────

/// Serve newline-delimited JSON-RPC: read one message per line from `input`,
/// write each reply as one line to `output`.  Returns when `input` ends.
pub fn serve_stdio(
    server: &McpServer,
    input: impl BufRead,
    mut output: impl Write,
) -> std::io::Result<()> {
    for line in input.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Value>(&line) {
            Ok(message) => server.handle(&message),
            Err(e) => Some(error_reply(
                Value::Null,
                -32700,
                &format!("Parse error: {e}"),
            )),
        };
        if let Some(reply) = reply {
            serde_json::to_writer(&mut output, &reply)?;
            output.write_all(b"\n")?;
            output.flush()?;
        }
    }
    Ok(())
}

// ── HTTP transport ────────────────────────────────────────────────────────────

/// `true` when the `Origin` header is absent (not a browser) or names this
/// machine.
///
/// The MCP transport spec requires servers to validate `Origin`, so that a
/// web page (or a DNS-rebinding attack) cannot drive a local server through
/// the user's browser.  A heap dump holds credentials and personal data.
fn origin_allowed(headers: &HeaderMap) -> bool {
    let Some(origin) = headers.get(header::ORIGIN) else {
        return true;
    };
    let Ok(origin) = origin.to_str() else {
        return false;
    };
    let rest = origin.split_once("://").map_or(origin, |(_, r)| r);
    // Authority is everything before the first `/`; drop any `user@` prefix,
    // so `http://localhost@evil.example` is judged by `evil.example`.
    let authority = rest.split('/').next().unwrap_or_default();
    let authority = authority.rsplit('@').next().unwrap_or_default();
    let host = match authority.strip_prefix('[') {
        Some(v6) => v6.split(']').next().map(|h| format!("[{h}]")),
        None => authority.split(':').next().map(str::to_owned),
    };
    matches!(
        host.as_deref().map(str::to_ascii_lowercase).as_deref(),
        Some("localhost" | "127.0.0.1" | "[::1]")
    )
}

/// `POST /mcp`: one JSON-RPC message (or batch) per request.  Notifications
/// get `202 Accepted` with no body.  Requests from a non-local `Origin` are
/// refused with `403`.
///
/// Replies are always plain `application/json` (the transport's non-streaming
/// mode); there are no sessions and no server-initiated messages, so a `GET`
/// is answered `405` by the router.
pub async fn handle_mcp(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: String,
) -> Response {
    if !origin_allowed(&headers) {
        return (StatusCode::FORBIDDEN, "Origin not allowed").into_response();
    }
    let message: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            return json_response(error_reply(
                Value::Null,
                -32700,
                &format!("Parse error: {e}"),
            ));
        }
    };
    let reply = tokio::task::spawn_blocking(move || McpServer::new(state).handle(&message)).await;
    match reply {
        Ok(Some(reply)) => json_response(reply),
        Ok(None) => StatusCode::ACCEPTED.into_response(),
        Err(e) => json_response(error_reply(
            Value::Null,
            -32603,
            &format!("Internal error: {e}"),
        )),
    }
}

fn json_response(value: Value) -> Response {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        value.to_string(),
    )
        .into_response()
}

#[cfg(test)]
mod origin_tests {
    use super::*;

    fn allowed(origin: Option<&str>) -> bool {
        let mut h = HeaderMap::new();
        if let Some(o) = origin {
            h.insert(header::ORIGIN, o.parse().expect("header value"));
        }
        origin_allowed(&h)
    }

    #[test]
    fn only_local_origins_and_non_browsers_are_accepted() {
        assert!(allowed(None), "no Origin: a command line client");
        for ok in [
            "http://localhost",
            "http://localhost:7000",
            "https://LOCALHOST:8443/",
            "http://127.0.0.1:7000",
            "http://[::1]:7000",
        ] {
            assert!(allowed(Some(ok)), "{ok}");
        }
        for bad in [
            "http://evil.example",
            "https://evil.example:7000",
            "http://localhost.evil.example",
            "http://localhost@evil.example",
            "http://127.0.0.1.evil.example",
            "null",
            "",
        ] {
            assert!(!allowed(Some(bad)), "{bad}");
        }
    }
}

#[cfg(test)]
mod tests;
