//! In-process tests: a small heap is indexed into a `MemStore`, wrapped in an
//! `AppState`, and driven through the protocol core.  No socket, no files.

use super::*;
use crate::diff::HeapDiff;
use crate::heap_parser::FieldValue;
use crate::index::MemStore;
use crate::pipeline::IndexOptions;
use crate::query::HeapQuery;
use crate::test_util::{
    ClassSpec, HprofBuilder, build_in_memory, build_in_memory_with, standard_heap, std_ids::*, ty,
};
use std::io::Cursor;
use std::path::PathBuf;

fn server_for(heap: HeapQuery) -> McpServer {
    McpServer::new(Arc::new(AppState::new(
        Arc::new(heap),
        PathBuf::from("test.hprof"),
    )))
}

fn server() -> McpServer {
    server_for(build_in_memory(&standard_heap()))
}

fn request(id: u64, method: &str, params: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})
}

/// Call a tool and return the whole `result` object.
fn call(s: &McpServer, name: &str, args: Value) -> Value {
    let reply = s
        .handle(&request(
            1,
            "tools/call",
            json!({"name": name, "arguments": args}),
        ))
        .expect("a reply");
    assert!(reply.get("error").is_none(), "protocol error: {reply}");
    reply["result"].clone()
}

/// Call a tool that must succeed; return its `structuredContent`.
fn ok(s: &McpServer, name: &str, args: Value) -> Value {
    let r = call(s, name, args);
    assert_eq!(r["isError"], json!(false), "{name} failed: {r}");
    assert!(r["content"][0]["text"].is_string());
    r["structuredContent"].clone()
}

/// Call a tool that must fail as a tool error; return the message.
fn fails(s: &McpServer, name: &str, args: Value) -> String {
    let r = call(s, name, args);
    assert_eq!(r["isError"], json!(true), "{name} should fail: {r}");
    assert!(r.get("structuredContent").is_none());
    r["content"][0]["text"]
        .as_str()
        .unwrap_or_default()
        .to_owned()
}

// ── protocol ──────────────────────────────────────────────────────────────────

#[test]
fn initialize_negotiates_the_version_and_carries_guidance() {
    let s = server();
    let r = s
        .handle(&request(
            1,
            "initialize",
            json!({"protocolVersion": "2024-11-05"}),
        ))
        .unwrap();
    assert_eq!(r["result"]["protocolVersion"], "2024-11-05");
    assert_eq!(r["result"]["serverInfo"]["name"], "hprof-toolkit");
    let text = r["result"]["instructions"].as_str().unwrap();
    assert!(text.contains("heap_summary") && text.contains("has_more"));
    assert!(text.contains("test.hprof"));
    assert!(r["result"]["capabilities"]["tools"].is_object());

    // An unknown version falls back to the newest we speak.
    let r = s
        .handle(&request(
            2,
            "initialize",
            json!({"protocolVersion": "1999-01-01"}),
        ))
        .unwrap();
    assert_eq!(r["result"]["protocolVersion"], "2025-06-18");
}

#[test]
fn notifications_get_no_reply_and_ping_works() {
    let s = server();
    let note = json!({"jsonrpc": "2.0", "method": "notifications/initialized"});
    assert_eq!(s.handle(&note), None);
    assert_eq!(s.handle(&json!({"jsonrpc": "2.0", "method": "ping"})), None);
    let r = s.handle(&request(9, "ping", json!({}))).unwrap();
    assert_eq!(r["result"], json!({}));
    assert_eq!(r["id"], 9);
    // A reply from the client to something we never asked is ignored.
    assert_eq!(
        s.handle(&json!({"jsonrpc": "2.0", "id": 5, "result": {}})),
        None
    );
}

#[test]
fn malformed_traffic_gets_protocol_errors() {
    let s = server();
    let r = s.handle(&request(1, "nope/nothing", json!({}))).unwrap();
    assert_eq!(r["error"]["code"], -32601);
    let r = s
        .handle(&json!({"jsonrpc": "1.0", "id": 2, "method": "ping"}))
        .unwrap();
    assert_eq!(r["error"]["code"], -32600);
    assert_eq!(r["id"], 2);
    let r = s.handle(&json!("just a string")).unwrap();
    assert_eq!(r["error"]["code"], -32600);
    assert_eq!(r["id"], Value::Null);
    let r = s.handle(&json!([])).unwrap();
    assert_eq!(r["error"]["code"], -32600);
    let r = s.handle(&json!({"jsonrpc": "2.0", "id": 3})).unwrap();
    assert_eq!(r["error"]["code"], -32600);
    // Unknown tool and missing name are protocol errors (-32602), not tool errors.
    let r = s
        .handle(&request(4, "tools/call", json!({"name": "no_such_tool"})))
        .unwrap();
    assert_eq!(r["error"]["code"], -32602);
    let r = s.handle(&request(5, "tools/call", json!({}))).unwrap();
    assert_eq!(r["error"]["code"], -32602);
}

#[test]
fn batches_reply_only_for_requests() {
    let s = server();
    let batch = json!([
        request(1, "ping", json!({})),
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        request(2, "ping", json!({})),
    ]);
    let r = s.handle(&batch).unwrap();
    assert_eq!(r.as_array().unwrap().len(), 2);
    let only_notes = json!([{"jsonrpc": "2.0", "method": "notifications/initialized"}]);
    assert_eq!(s.handle(&only_notes), None);
}

#[test]
fn every_tool_is_listed_with_a_description_and_schema() {
    let s = server();
    let r = s.handle(&request(1, "tools/list", json!({}))).unwrap();
    let tools = r["result"]["tools"].as_array().unwrap();
    let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    let expected = [
        "heap_summary",
        "class_histogram",
        "find_classes",
        "class_info",
        "instances_of_class",
        "object",
        "object_string",
        "array_contents",
        "references_to",
        "references_from",
        "path_to_gc_root",
        "gc_roots",
        "threads",
        "thread_stack",
        "largest_arrays",
        "retained_top",
        "dominated_by",
        "search_strings",
        "heap_diff_summary",
        "heap_diff_objects",
    ];
    assert_eq!(names, expected);
    for t in tools {
        assert!(
            t["description"].as_str().unwrap().len() > 30,
            "{}",
            t["name"]
        );
        assert_eq!(t["inputSchema"]["type"], "object", "{}", t["name"]);
        let props = t["inputSchema"]["properties"].as_object().unwrap();
        for req in t["inputSchema"]["required"].as_array().unwrap() {
            assert!(
                props.contains_key(req.as_str().unwrap()),
                "{} requires unknown {req}",
                t["name"]
            );
        }
    }
    // Paged tools advertise offset and limit.
    let hist = tools
        .iter()
        .find(|t| t["name"] == "class_histogram")
        .unwrap();
    assert!(hist["inputSchema"]["properties"]["limit"].is_object());

    // Every searching tool takes the same `mode` enum and `case_sensitive`;
    // string search has no fuzzy mode.
    let searching: Vec<&Value> = tools
        .iter()
        .filter(|t| t["inputSchema"]["properties"]["mode"].is_object())
        .collect();
    let names: Vec<&str> = searching
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "class_histogram",
            "find_classes",
            "threads",
            "search_strings"
        ]
    );
    for t in &searching {
        let props = &t["inputSchema"]["properties"];
        let modes: Vec<&str> = props["mode"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m.as_str().unwrap())
            .collect();
        let expected: Vec<&str> = if t["name"] == "search_strings" {
            vec!["contains", "exact", "regex"]
        } else {
            vec!["contains", "exact", "regex", "fuzzy"]
        };
        assert_eq!(modes, expected, "{}", t["name"]);
        assert_eq!(props["case_sensitive"]["type"], "boolean", "{}", t["name"]);
        assert!(props.get("exact").is_none() && props.get("ignore_case").is_none());
    }
}

#[test]
fn prompts_and_resources_are_served() {
    let s = server();
    let r = s.handle(&request(1, "prompts/list", json!({}))).unwrap();
    assert_eq!(r["result"]["prompts"][0]["name"], "investigate_memory_leak");
    let r = s
        .handle(&request(
            2,
            "prompts/get",
            json!({"name": "investigate_memory_leak"}),
        ))
        .unwrap();
    assert!(
        r["result"]["messages"][0]["content"]["text"]
            .as_str()
            .unwrap()
            .contains("path_to_gc_root")
    );
    let r = s
        .handle(&request(3, "prompts/get", json!({"name": "x"})))
        .unwrap();
    assert_eq!(r["error"]["code"], -32602);

    let r = s.handle(&request(4, "resources/list", json!({}))).unwrap();
    assert_eq!(r["result"]["resources"][0]["uri"], "hprof://summary");
    let r = s
        .handle(&request(
            5,
            "resources/read",
            json!({"uri": "hprof://summary"}),
        ))
        .unwrap();
    let text = r["result"]["contents"][0]["text"].as_str().unwrap();
    assert!(serde_json::from_str::<Value>(text).unwrap()["id_size"] == 8);
    let r = s
        .handle(&request(
            6,
            "resources/read",
            json!({"uri": "hprof://nope"}),
        ))
        .unwrap();
    assert_eq!(r["error"]["code"], -32002);
}

#[test]
fn stdio_transport_frames_lines_and_survives_garbage() {
    let s = server();
    let input = [
        r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#,
        "",
        "this is not json",
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"threads"}}"#,
    ]
    .join("\n");
    let mut out = Vec::new();
    serve_stdio(&s, Cursor::new(input), &mut out).unwrap();
    let lines: Vec<Value> = String::from_utf8(out)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(
        lines.len(),
        3,
        "ping, parse error, threads; nothing for the notification"
    );
    assert_eq!(lines[0]["id"], 1);
    assert_eq!(lines[1]["error"]["code"], -32700);
    assert_eq!(lines[2]["result"]["structuredContent"]["total"], 1);
}

// ── argument handling and errors ──────────────────────────────────────────────

#[test]
fn bad_arguments_are_tool_errors_the_model_can_act_on() {
    let s = server();
    assert!(fails(&s, "object", json!({})).contains("Missing required argument `id`"));
    assert!(fails(&s, "object", json!({"id": "not-an-id"})).contains("object id"));
    assert!(fails(&s, "object", json!({"id": "0xdead"})).contains("No object"));
    assert!(fails(&s, "class_histogram", json!({"sort_by": "colour"})).contains("sort_by"));
    assert!(fails(&s, "class_histogram", json!({"limit": "ten"})).contains("`limit`"));
    assert!(fails(&s, "gc_roots", json!({"kind": "purple"})).contains("Unknown root kind"));
    assert!(fails(&s, "largest_arrays", json!({"kind": "wide"})).contains("Unknown array kind"));
    assert!(fails(&s, "class_info", json!({})).contains("class_id"));
    assert!(fails(&s, "class_info", json!({"class_name": "no.Such"})).contains("find_classes"));
    assert!(
        fails(&s, "object_string", json!({"id": hex_id(INTEGER_42)}))
            .contains("not a java.lang.String")
    );
    assert!(
        fails(&s, "array_contents", json!({"id": hex_id(INTEGER_42)})).contains("not an array")
    );
    assert!(fails(&s, "thread_stack", json!({"serial": 77})).contains("No thread"));
    assert!(fails(&s, "thread_stack", json!({})).contains("serial"));
    assert!(
        fails(
            &s,
            "instances_of_class",
            json!({"class_id": hex_id(INTEGER_42)})
        )
        .contains("not a class")
    );
}

fn hex_id(id: u64) -> String {
    format!("0x{id:x}")
}

#[test]
fn missing_retained_index_and_diff_are_reported_not_hidden() {
    let opts = IndexOptions {
        retained: false,
        ..IndexOptions::default()
    };
    let s = server_for(build_in_memory_with(&standard_heap(), &opts).1);
    let m = fails(&s, "retained_top", json!({}));
    assert!(
        m.contains("retained heap") && m.contains("hprof-toolkit index"),
        "{m}"
    );
    assert!(fails(&s, "dominated_by", json!({"id": "0x400"})).contains("retained heap"));
    // The object tool degrades instead of failing.
    let o = ok(&s, "object", json!({"id": hex_id(LIST)}));
    assert_eq!(o["retained_bytes"], Value::Null);
    assert!(fails(&s, "heap_diff_summary", json!({})).contains("--diff-hprof"));
    assert!(fails(&s, "heap_diff_objects", json!({"kind": "added"})).contains("--diff-hprof"));
    let sum = ok(&s, "heap_summary", json!({}));
    assert_eq!(sum["indexes"]["retained_sizes"], false);
    assert_eq!(sum["indexes"]["diff"], false);
}

// ── tools ─────────────────────────────────────────────────────────────────────

#[test]
fn heap_summary_reports_scale_roots_and_indexes() {
    let d = ok(&server(), "heap_summary", json!({}));
    assert_eq!(d["id_size"], 8);
    assert_eq!(d["objects"], 4);
    assert_eq!(d["classes"], 5);
    assert_eq!(d["references"], 2);
    assert_eq!(d["threads"], 1);
    assert_eq!(d["gc_roots"]["sticky_class"], 2);
    assert_eq!(d["gc_roots"]["java_frame"], 1);
    assert_eq!(d["indexes"]["retained_sizes"], true);
    assert_eq!(d["hprof_path"], "test.hprof");
}

#[test]
fn class_histogram_pages_and_sorts() {
    let s = server();
    let d = ok(&s, "class_histogram", json!({"limit": 3}));
    assert_eq!(d["total"], 4);
    assert_eq!(d["has_more"], true);
    assert_eq!(d["next_offset"], 3);
    assert_eq!(d["items"].as_array().unwrap().len(), 3);
    let next = ok(&s, "class_histogram", json!({"offset": 3, "limit": 3}));
    assert_eq!(next["items"].as_array().unwrap().len(), 1);
    assert_eq!(next["has_more"], false);
    assert_eq!(next["next_offset"], Value::Null);
    let names: Vec<String> = d["items"]
        .as_array()
        .unwrap()
        .iter()
        .chain(next["items"].as_array().unwrap())
        .map(|i| i["class_name"].as_str().unwrap().to_owned())
        .collect();
    assert!(names.contains(&"java.lang.Integer".to_owned()));
    assert!(names.contains(&"char[]".to_owned()));

    let by_bytes = ok(&s, "class_histogram", json!({"sort_by": "shallow_bytes"}));
    let bytes: Vec<u64> = by_bytes["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["shallow_bytes"].as_u64().unwrap())
        .collect();
    assert!(bytes.windows(2).all(|w| w[0] >= w[1]), "{bytes:?}");
    // Page limit is clamped.
    // `name` narrows the histogram; the order is still the ranking.
    let arrays = ok(
        &s,
        "class_histogram",
        json!({"name": r"\[\]$", "mode": "regex"}),
    );
    assert_eq!(arrays["total"], 1);
    assert_eq!(arrays["items"][0]["class_name"], "char[]");
    let java = ok(
        &s,
        "class_histogram",
        json!({"name": "java", "sort_by": "shallow_bytes"}),
    );
    assert_eq!(java["total"], 3);
    for item in java["items"].as_array().unwrap() {
        assert!(item["class_name"].as_str().unwrap().starts_with("java"));
    }
    let bytes: Vec<u64> = java["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["shallow_bytes"].as_u64().unwrap())
        .collect();
    assert!(bytes.windows(2).all(|w| w[0] >= w[1]));
    assert!(
        fails(
            &s,
            "class_histogram",
            json!({"name": "x", "mode": "fuzzy", "case_sensitive": "yes"})
        )
        .contains("case_sensitive")
    );

    let big = ok(&s, "class_histogram", json!({"limit": 100000}));
    assert_eq!(big["limit"], 500);
}

#[test]
fn find_classes_and_class_info() {
    let s = server();
    let d = ok(&s, "find_classes", json!({"name": "java.lang"}));
    assert_eq!(d["total"], 3);
    assert_eq!(d["items"][0]["class_name"], "java.lang.Integer");
    assert_eq!(d["items"][0]["class_id"], "0x10");
    assert_eq!(d["items"][0]["instances"], 1);
    let exact = ok(
        &s,
        "find_classes",
        json!({"name": "java.lang.Integer", "exact": true}),
    );
    assert_eq!(exact["total"], 1);
    let none = ok(&s, "find_classes", json!({"name": "nope", "exact": true}));
    assert_eq!(none["total"], 0);

    // The four modes, and case handling.
    let re = ok(
        &s,
        "find_classes",
        json!({"name": r"^java\.lang\.", "mode": "regex"}),
    );
    assert_eq!(re["total"], 3);
    let fuzzy = ok(
        &s,
        "find_classes",
        json!({"name": "jlint", "mode": "fuzzy"}),
    );
    assert_eq!(fuzzy["items"][0]["class_name"], "java.lang.Integer");
    let exact = ok(
        &s,
        "find_classes",
        json!({"name": "JAVA.LANG.INTEGER", "mode": "exact"}),
    );
    assert_eq!(exact["total"], 1, "exact is case-insensitive by default");
    let cs = ok(
        &s,
        "find_classes",
        json!({"name": "JAVA.LANG", "case_sensitive": true}),
    );
    assert_eq!(cs["total"], 0);
    assert!(
        fails(&s, "find_classes", json!({"name": "(", "mode": "regex"})).contains("invalid regex")
    );
    assert!(fails(&s, "find_classes", json!({"name": "x", "mode": "glob"})).contains("`mode`"));

    let info = ok(
        &s,
        "class_info",
        json!({"class_name": "java.util.ArrayList"}),
    );
    assert_eq!(info["class_id"], "0x50");
    assert_eq!(info["super_class"]["class_name"], "java.lang.Object");
    assert_eq!(info["instances"], 1);
    assert_eq!(info["instance_size"], 8);
    assert_eq!(info["static_fields"][0]["value"], 7);
    assert_eq!(info["instance_fields"][0]["type"], "Object");
    // By id, hex with or without prefix, and as a decimal number.
    for id in [json!("0x50"), json!("50"), json!(0x50)] {
        let by_id = ok(&s, "class_info", json!({"class_id": id}));
        assert_eq!(by_id["class_name"], "java.util.ArrayList");
    }
}

#[test]
fn instances_of_class_by_id_name_and_array_type() {
    let s = server();
    let d = ok(
        &s,
        "instances_of_class",
        json!({"class_name": "java.lang.Integer"}),
    );
    assert_eq!(d["total"], 1);
    assert_eq!(d["items"][0]["id"], "0x100");
    assert_eq!(d["class_name"], "java.lang.Integer");
    let d = ok(&s, "instances_of_class", json!({"class_id": "0x50"}));
    assert_eq!(d["items"][0]["id"], "0x400");
    let arr = ok(&s, "instances_of_class", json!({"class_name": "char[]"}));
    assert_eq!(arr["total"], 1);
    assert_eq!(arr["items"][0]["id"], "0x200");
    let empty = ok(
        &s,
        "instances_of_class",
        json!({"class_name": "java.lang.Object"}),
    );
    assert_eq!(empty["total"], 0);
    assert!(
        fails(&s, "instances_of_class", json!({"class_name": "no.Such"}))
            .contains("No class named")
    );
    assert!(fails(&s, "instances_of_class", json!({})).contains("class_id"));
}

#[test]
fn object_shows_fields_retained_size_roots_and_referrers() {
    let s = server();
    let list = ok(&s, "object", json!({"id": "0x400"}));
    assert_eq!(list["kind"], "instance");
    assert_eq!(list["class_name"], "java.util.ArrayList");
    assert_eq!(list["id"], "0x400");
    assert_eq!(list["gc_root_kinds"], json!(["java_frame"]));
    assert_eq!(list["retained_bytes"], 12);
    assert_eq!(list["fields"][0]["value"]["type"], "java.lang.Integer");
    assert_eq!(list["fields"][0]["value"]["value"], 42);
    assert_eq!(list["fields"][0]["value"]["id"], "0x100");
    assert_eq!(list["referrers"], 0);

    let int = ok(&s, "object", json!({"id": "0x100"}));
    assert_eq!(int["referrers"], 1);
    assert_eq!(int["fields"][0]["value"], 42);
    assert_eq!(int["dominator"], "0x400");

    let string = ok(&s, "object", json!({"id": "0x300"}));
    assert_eq!(string["string"]["preview"], "hi");
    assert_eq!(
        string["retained_bytes"],
        Value::Null,
        "unreachable: no retained size"
    );

    let class = ok(&s, "object", json!({"id": "0x10"}));
    assert_eq!(class["kind"], "class");
    let prim = ok(&s, "object", json!({"id": "0x200"}));
    assert_eq!(prim["kind"], "primitive_array");
    assert_eq!(prim["element_type"], "char");
    assert_eq!(prim["length"], 2);
}

#[test]
fn strings_and_arrays_are_readable() {
    let s = server();
    let d = ok(&s, "object_string", json!({"id": "0x300"}));
    assert_eq!(d["text"], "hi");
    assert_eq!(d["length_chars"], 2);
    assert_eq!(d["truncated"], false);
    let d = ok(&s, "object_string", json!({"id": "0x300", "max_chars": 1}));
    assert_eq!(d["text"], "h");
    assert_eq!(d["truncated"], true);

    let d = ok(&s, "array_contents", json!({"id": "0x200"}));
    assert_eq!(d["elements"], json!(["h", "i"]));
    assert_eq!(d["total"], 2);
    let d = ok(
        &s,
        "array_contents",
        json!({"id": "0x200", "offset": 1, "limit": 1}),
    );
    assert_eq!(d["elements"], json!(["i"]));
    assert_eq!(d["has_more"], false);
    let d = ok(&s, "array_contents", json!({"id": "0x200", "limit": 1}));
    assert_eq!(d["has_more"], true);
    assert_eq!(d["next_offset"], 1);
}

#[test]
fn object_arrays_and_ints_page_through_array_contents() {
    let bytes = HprofBuilder::new(8)
        .utf8(1, "[LElem;")
        .load_class(1, 0x10, 1)
        .class_dump(ClassSpec::new(0x10))
        .obj_array(0x50, 0x10, &[0x1, 0, 0x2])
        .int_array(0x60, &[10, 20, 30, 40])
        .build();
    let s = server_for(build_in_memory(&bytes));
    let d = ok(&s, "array_contents", json!({"id": "0x50"}));
    assert_eq!(d["total"], 3);
    assert_eq!(d["elements"][0]["id"], "0x1");
    assert_eq!(d["elements"][1]["id"], Value::Null);
    assert_eq!(d["elements"][2]["index"], 2);
    let d = ok(&s, "array_contents", json!({"id": "0x60", "offset": 2}));
    assert_eq!(d["elements"], json!([30, 40]));
    assert_eq!(d["element_type"], "int");
    let d = ok(&s, "largest_arrays", json!({"kind": "int"}));
    assert_eq!(d["items"][0]["id"], "0x60");
    assert_eq!(d["items"][0]["elements"], 4);
    assert_eq!(d["items"][0]["bytes"], 16);
    let d = ok(&s, "largest_arrays", json!({"kind": "object"}));
    assert_eq!(d["total"], 1);
}

#[test]
fn references_and_paths_are_labelled() {
    let s = server();
    let to = ok(&s, "references_to", json!({"id": "0x100"}));
    assert_eq!(to["total"], 1);
    assert_eq!(to["items"][0]["id"], "0x400");
    assert_eq!(to["items"][0]["type"], "java.util.ArrayList");
    let from = ok(&s, "references_from", json!({"id": "0x400"}));
    assert_eq!(from["items"][0]["id"], "0x100");
    assert!(from["items"][0]["via"].as_str().unwrap().starts_with('.'));

    let p = ok(&s, "path_to_gc_root", json!({"id": "0x100"}));
    assert_eq!(p["outcome"], "found");
    assert_eq!(p["root_kinds"], json!(["java_frame"]));
    assert_eq!(p["steps"][0]["id"], "0x400");
    assert_eq!(p["steps"][0]["via"], Value::Null);
    assert_eq!(p["steps"][1]["id"], "0x100");
    assert!(p["steps"][1]["via"].is_string());

    let none = ok(&s, "path_to_gc_root", json!({"id": "0x300"}));
    assert_eq!(none["outcome"], "not_reachable");
    // `max_nodes` is accepted and echoed in the search size (limits themselves are
    // covered by the graph tests).
    let bounded = ok(
        &s,
        "path_to_gc_root",
        json!({"id": "0x100", "max_nodes": 5}),
    );
    assert_eq!(bounded["outcome"], "found");
    assert!(fails(&s, "path_to_gc_root", json!({"id": "0xdead"})).contains("No object"));
}

#[test]
fn roots_threads_and_stacks() {
    let s = server();
    let counts = ok(&s, "gc_roots", json!({}));
    assert_eq!(counts["counts"]["sticky_class"], 2);
    let d = ok(&s, "gc_roots", json!({"kind": "java_frame"}));
    assert_eq!(d["total"], 1);
    assert_eq!(d["items"][0]["object_id"], "0x400");
    assert_eq!(d["items"][0]["thread_serial"], 1);
    let d = ok(&s, "gc_roots", json!({"kind": "sticky_class", "limit": 1}));
    assert_eq!(d["has_more"], true);

    let found = ok(&s, "threads", json!({"name": "mn", "mode": "fuzzy"}));
    assert_eq!(found["total"], 1);
    assert_eq!(found["items"][0]["name"], "main");
    let none = ok(&s, "threads", json!({"name": "xyz"}));
    assert_eq!(none["total"], 0);
    let t = ok(&s, "threads", json!({}));
    assert_eq!(t["items"][0]["name"], "main");
    assert_eq!(t["items"][0]["ended"], true);
    assert_eq!(t["items"][0]["object_id"], "0xabc");
    let st = ok(&s, "thread_stack", json!({"serial": 1}));
    assert_eq!(st["name"], "main");
    assert_eq!(st["frames"][0]["method"], "main");
    assert_eq!(st["frames"][0]["location"], "MyClass.java:7");
}

#[test]
fn retained_ranking_and_dominators() {
    let s = server();
    let top = ok(&s, "retained_top", json!({"limit": 2}));
    assert_eq!(top["total"], 4);
    assert_eq!(top["items"][0]["id"], "0x400");
    assert_eq!(top["items"][0]["retained_bytes"], 12);
    assert_eq!(top["has_more"], true);
    let dom = ok(&s, "dominated_by", json!({"id": "0x400"}));
    assert_eq!(dom["total"], 1);
    assert_eq!(dom["items"][0]["id"], "0x100");
    assert_eq!(dom["items"][0]["retained_bytes"], 4);
}

#[test]
fn search_strings_scans_in_budgeted_slices() {
    // 30 strings "s-00".."s-29", two of which contain "needle".
    const STRING_CLASS: u64 = 0x10;
    let mut b = HprofBuilder::new(8)
        .utf8(1, "value")
        .utf8(2, "java/lang/String")
        .load_class(1, STRING_CLASS, 2)
        .class_dump(
            ClassSpec::new(STRING_CLASS)
                .instance_size(8)
                .field(1, ty::OBJECT),
        );
    for i in 0..30u64 {
        let text = match i {
            5 => "a NEEDLE here".to_owned(),
            25 => "another needle".to_owned(),
            n => format!("s-{n:02}"),
        };
        b = b.char_array(0x1000 + i, &text).instance_values(
            0x2000 + i,
            STRING_CLASS,
            &[FieldValue::Object(0x1000 + i)],
        );
    }
    let s = server_for(build_in_memory(&b.build()));

    let first = ok(
        &s,
        "search_strings",
        json!({"text": "needle", "max_scan": 10, "case_sensitive": true}),
    );
    assert_eq!(
        first["matches"].as_array().unwrap().len(),
        0,
        "case-sensitive, and only 10 scanned"
    );
    assert_eq!(first["next_cursor"], 10);
    let ci = ok(
        &s,
        "search_strings",
        json!({"text": "needle", "max_scan": 10}),
    );
    assert_eq!(
        ci["matches"][0]["id"], "0x2005",
        "case-insensitive by default"
    );
    assert_eq!(ci["matches"][0]["text"], "a NEEDLE here");
    let old_spelling = ok(
        &s,
        "search_strings",
        json!({"text": "needle", "ignore_case": false, "max_scan": 10}),
    );
    assert!(old_spelling["matches"].as_array().unwrap().is_empty());
    let re = ok(
        &s,
        "search_strings",
        json!({"text": "^another", "mode": "regex"}),
    );
    assert_eq!(re["matches"][0]["id"], "0x2019");
    let exact = ok(
        &s,
        "search_strings",
        json!({"text": "S-07", "mode": "exact"}),
    );
    assert_eq!(exact["matches"].as_array().unwrap().len(), 1);
    assert!(fails(&s, "search_strings", json!({"text": "x", "mode": "fuzzy"})).contains("`mode`"));
    let cont = ok(
        &s,
        "search_strings",
        json!({"text": "needle", "cursor": first["next_cursor"]}),
    );
    assert_eq!(cont["matches"][0]["id"], "0x2019");
    assert_eq!(cont["next_cursor"], Value::Null, "scanned to the end");
    let capped = ok(
        &s,
        "search_strings",
        json!({"text": "s-", "max_results": 3}),
    );
    assert_eq!(capped["matches"].as_array().unwrap().len(), 3);
    assert!(
        capped["next_cursor"].is_number(),
        "stopped early, so there is more to scan"
    );
}

fn diff_server() -> McpServer {
    const CLASS: u64 = 0x10;
    let heap = |nodes: &[(u64, i32)]| {
        let mut b = HprofBuilder::new(8)
            .utf8(1, "v")
            .utf8(2, "Node")
            .load_class(1, CLASS, 2)
            .class_dump(ClassSpec::new(CLASS).instance_size(4).field(1, ty::INT));
        for &(id, v) in nodes {
            b = b.instance_values(id, CLASS, &[FieldValue::Int(v)]);
        }
        Arc::new(build_in_memory(&b.build()))
    };
    let before = heap(&[(1, 1), (2, 2), (3, 3)]);
    let after = heap(&[(2, 2), (3, 99), (4, 4)]);
    let diff =
        HeapDiff::from_store(Arc::clone(&before), Arc::clone(&after), &MemStore::new()).unwrap();
    McpServer::new(Arc::new(
        AppState::new(before, PathBuf::from("before.hprof"))
            .with_diff(diff, PathBuf::from("after.hprof")),
    ))
}

#[test]
fn diff_tools_summarise_and_list() {
    let s = diff_server();
    let sum = ok(&s, "heap_diff_summary", json!({}));
    assert_eq!(sum["totals"]["added"], 1);
    assert_eq!(sum["totals"]["removed"], 1);
    assert_eq!(sum["totals"]["changed"], 1);
    assert_eq!(sum["items"][0]["class_name"], "Node");
    assert_eq!(sum["items"][0]["added"], 1);
    assert_eq!(sum["items"][0]["net"], 0);

    let added = ok(&s, "heap_diff_objects", json!({"kind": "added"}));
    assert_eq!(added["items"][0]["id"], "0x4");
    assert_eq!(added["items"][0]["class_name"], "Node");
    let removed = ok(
        &s,
        "heap_diff_objects",
        json!({"kind": "removed", "class_name": "Node"}),
    );
    assert_eq!(removed["items"][0]["id"], "0x1");
    let changed = ok(&s, "heap_diff_objects", json!({"kind": "changed"}));
    assert_eq!(changed["items"][0]["id"], "0x3");
    let unchanged = ok(
        &s,
        "heap_diff_objects",
        json!({"kind": "unchanged", "class_name": "Node"}),
    );
    assert_eq!(unchanged["total"], 2, "the instance and the class object");
    assert!(fails(&s, "heap_diff_objects", json!({"kind": "moved"})).contains("`kind`"));
    assert!(
        fails(
            &s,
            "heap_diff_objects",
            json!({"kind": "added", "class_name": "No.Such"})
        )
        .contains("No class named")
    );
    let sum = ok(&s, "heap_summary", json!({}));
    assert_eq!(sum["indexes"]["diff"], true);
}
