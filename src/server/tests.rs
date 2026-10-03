//! HTTP smoke tests: the real router over an in-memory heap, driven with
//! `tower::ServiceExt::oneshot`.  No socket, no files.

use super::{AppState, router};
use crate::heap_parser::FieldValue;
use crate::test_util::{ClassSpec, HprofBuilder, build_in_memory, standard_heap, ty};
use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use std::path::PathBuf;
use std::sync::Arc;
use tower::ServiceExt;

fn app_for(hprof: &[u8]) -> axum::Router {
    router(Arc::new(AppState::new(
        Arc::new(build_in_memory(hprof)),
        PathBuf::from("test.hprof"),
    )))
}

async fn send(app: &axum::Router, request: Request<Body>) -> (StatusCode, String) {
    let response = app.clone().oneshot(request).await.expect("router reply");
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

async fn get(app: &axum::Router, uri: &str) -> (StatusCode, String) {
    send(app, Request::get(uri).body(Body::empty()).expect("request")).await
}

#[tokio::test]
async fn the_main_pages_render() {
    let app = app_for(&standard_heap());
    for (uri, needle) in [
        ("/", "Heap Summary"),
        ("/histogram", "java.lang.Integer"),
        ("/allClasses", "java.util.ArrayList"),
        ("/roots", "Sticky class"),
        ("/threads", "main"),
        ("/thread/1", "MyClass.java"),
        ("/retained", "Retained Heap"),
        ("/strings", "Searches the text"),
        ("/object/400", "java.util.ArrayList"),
        ("/object/400/root-path", "GC root"),
        ("/class/50", "myField"),
    ] {
        let (status, body) = get(&app, uri).await;
        assert_eq!(status, StatusCode::OK, "{uri}");
        assert!(body.contains(needle), "{uri} should mention {needle:?}");
    }
}

#[tokio::test]
async fn bad_ids_are_client_errors_and_missing_objects_say_so() {
    let app = app_for(&standard_heap());
    let (status, _) = get(&app, "/object/not-hex").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = get(&app, "/class/zz").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = get(&app, "/roots/purple").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, body) = get(&app, "/object/dead").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("No object with that ID"));
    let (status, _) = get(&app, "/object/300/raw-string").await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = get(&app, "/object/400/raw-string").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn the_mcp_endpoint_answers_json_rpc_over_post() {
    let app = app_for(&standard_heap());
    let post = |body: &str| {
        Request::post("/mcp")
            .header("content-type", "application/json")
            .body(Body::from(body.to_owned()))
            .expect("request")
    };
    let (status, body) = send(
        &app,
        post(r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"threads"}}"#),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let reply: serde_json::Value = serde_json::from_str(&body).expect("json");
    assert_eq!(reply["result"]["structuredContent"]["total"], 1);

    // A notification gets 202 and no body; garbage gets a JSON-RPC parse error.
    let (status, body) = send(
        &app,
        post(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#),
    )
    .await;
    assert_eq!((status, body.as_str()), (StatusCode::ACCEPTED, ""));
    let (_, body) = send(&app, post("not json")).await;
    assert!(body.contains("-32700"), "{body}");

    // A web page must not be able to drive the server: a foreign Origin is
    // refused, a local one is fine, and GET is not allowed.
    let from = |origin: &str| {
        Request::post("/mcp")
            .header("origin", origin)
            .body(Body::from(r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#))
            .expect("request")
    };
    let (status, _) = send(&app, from("http://evil.example")).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (status, _) = send(&app, from("http://localhost:7000")).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = get(&app, "/mcp").await;
    assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
}

/// A `java.lang.String` whose text is `a` followed by `n` two-byte characters.
fn heap_with_string(n: usize) -> Vec<u8> {
    const STRING_CLASS: u64 = 0x10;
    let text: String = std::iter::once('a')
        .chain(std::iter::repeat_n('é', n))
        .collect();
    HprofBuilder::new(8)
        .utf8(1, "value")
        .utf8(2, "java/lang/String")
        .load_class(1, STRING_CLASS, 2)
        .class_dump(
            ClassSpec::new(STRING_CLASS)
                .instance_size(8)
                .field(1, ty::OBJECT),
        )
        .char_array(0x100, &text)
        .instance_values(0x200, STRING_CLASS, &[FieldValue::Object(0x100)])
        .build()
}

/// Byte 2000 of this string falls inside a multi-byte character, which used
/// to panic the object page.
#[tokio::test]
async fn a_long_non_ascii_string_does_not_break_the_object_page() {
    let app = app_for(&heap_with_string(2500));
    let (status, body) = get(&app, "/object/200").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("2501 chars total"), "{body}");
}

/// 25 GC roots: paging by `limit` must show exactly `limit` rows each time.
#[tokio::test]
async fn root_pages_hold_exactly_limit_rows() {
    let mut b = HprofBuilder::new(8);
    for i in 0..25u64 {
        b = b.root_unknown(0x100 + i);
    }
    let app = app_for(&b.build());
    let (status, body) = get(&app, "/roots/unknown?offset=10&limit=10").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body.matches("<td><a href=\"/object/").count(), 10, "{body}");
    assert!(body.contains("/object/10a\"") && body.contains("/object/113\""));
    assert!(!body.contains("/object/109\"") && !body.contains("/object/114\""));
    assert!(body.contains("offset=20&limit=10"), "next link");
    assert!(body.contains("Total: 25"));
    let (_, last) = get(&app, "/roots/unknown?offset=20&limit=10").await;
    assert_eq!(last.matches("<td><a href=\"/object/").count(), 5);
    assert!(!last.contains("Next"));
}

/// Names come from the dump, so they must be escaped everywhere, the page
/// title included.
#[tokio::test]
async fn names_from_the_dump_are_escaped_in_titles() {
    let hprof = HprofBuilder::new(8)
        .utf8(1, "Evil<img src=x onerror=alert(1)>")
        .load_class(1, 0x10, 1)
        .class_dump(ClassSpec::new(0x10))
        .build();
    let app = app_for(&hprof);
    for uri in ["/class/10", "/instances/10", "/allClasses"] {
        let (status, body) = get(&app, uri).await;
        assert_eq!(status, StatusCode::OK, "{uri}");
        assert!(!body.contains("<img"), "{uri} leaked markup: {body}");
        assert!(body.contains("&lt;img"), "{uri}");
    }
}

#[tokio::test]
async fn class_pages_are_searchable() {
    let app = app_for(&standard_heap());

    let (status, body) = get(&app, "/allClasses?q=java.lang&mode=regex").await;
    assert_eq!(status, StatusCode::OK);
    for name in ["java.lang.Integer", "java.lang.Object", "java.lang.String"] {
        assert!(body.contains(name), "{name}");
    }
    assert!(!body.contains("java.util.ArrayList"));
    assert!(body.contains("3 classes match"), "{body}");
    // The form keeps the search, escaped.
    assert!(body.contains("value=\"java.lang\""));
    assert!(body.contains("<option value=\"regex\" selected>"));
    let (_, body) = get(&app, "/allClasses?q=%3Cb%3E").await;
    assert!(
        body.contains("value=\"&lt;b&gt;\"") && !body.contains("<b>"),
        "{body}"
    );

    // Fuzzy on the histogram only filters: the rows stay in count order.
    let (_, plain) = get(&app, "/histogram").await;
    let (status, fuzzy) = get(&app, "/histogram?q=a&mode=fuzzy").await;
    assert_eq!(status, StatusCode::OK);
    let order = |b: &str| -> Vec<usize> {
        [
            "java.lang.Integer",
            "char[]",
            "java.lang.String",
            "java.util.ArrayList",
        ]
        .iter()
        .map(|n| b.find(n).expect(n))
        .collect()
    };
    let (p, f) = (order(&plain), order(&fuzzy));
    let rank = |v: &[usize]| {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by_key(|&i| v[i]);
        idx
    };
    assert_eq!(rank(&p), rank(&f));
    assert!(fuzzy.contains("4 of 4 classes match"), "{fuzzy}");
    let (_, one) = get(&app, "/histogram?q=%5C%5B%5C%5D%24&mode=regex").await;
    assert!(
        one.contains("1 of 4 classes match") && one.contains("char[]"),
        "{one}"
    );

    // A bad pattern is a 400 that still shows the form and the reason.
    let (status, body) = get(&app, "/allClasses?q=(&mode=regex").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        body.contains("invalid regex") && body.contains("<form"),
        "{body}"
    );
    let (status, _) = get(&app, "/histogram?q=x&mode=glob").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Paging links carry the search.
    let (_, body) = get(&app, "/allClasses?q=a&limit=2").await;
    assert!(
        body.contains("offset=2&limit=2&q=a&mode=contains"),
        "{body}"
    );
    let (_, body) = get(
        &app,
        "/allClasses?q=a&limit=2&offset=2&case=1&mode=contains",
    )
    .await;
    assert!(
        body.contains("offset=0&limit=2&q=a&mode=contains&case=1"),
        "{body}"
    );
    assert!(body.contains("checked"), "{body}");
}

#[tokio::test]
async fn threads_and_strings_are_searchable() {
    let app = app_for(&standard_heap());

    let (status, body) = get(&app, "/threads?q=mn&mode=fuzzy").await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        body.contains("/thread/1") && body.contains("1 threads match"),
        "{body}"
    );
    let (_, body) = get(&app, "/threads?q=xyz").await;
    assert!(
        body.contains("No threads match") && !body.contains("/thread/1"),
        "{body}"
    );

    let (status, body) = get(&app, "/strings?q=%5Eh.%24&mode=regex").await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        body.contains("/object/300") && body.contains("1 match(es)"),
        "{body}"
    );
    assert!(body.contains("End of strings"), "{body}");
    let (_, body) = get(&app, "/strings?q=bye").await;
    assert!(body.contains("No matches"), "{body}");
    let (status, body) = get(&app, "/strings").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("<form") && !body.contains("<table"), "{body}");
    let (status, body) = get(&app, "/strings?q=x&mode=fuzzy").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body.contains("fuzzy search is not available"), "{body}");
    // The fuzzy option is not offered on this page.
    assert!(!body.contains("value=\"fuzzy\""), "{body}");
}

/// A scan that stops early continues from the cursor, keeping the search.
#[tokio::test]
async fn string_pages_continue_from_the_cursor() {
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
        b = b
            .char_array(0x1000 + i, &format!("needle {i}"))
            .instance_values(0x2000 + i, STRING_CLASS, &[FieldValue::Object(0x1000 + i)]);
    }
    let app = app_for(&b.build());
    let (_, body) = get(&app, "/strings?q=needle&max_scan=10&max_results=5").await;
    assert_eq!(body.matches("<td><a href=\"/object/").count(), 5, "{body}");
    assert!(
        body.contains("/strings?cursor=5&max_scan=10&max_results=5&q=needle&mode=contains"),
        "{body}"
    );
    let (_, body) = get(
        &app,
        "/strings?q=needle&max_scan=10&max_results=5&cursor=25",
    )
    .await;
    assert!(
        body.contains("End of strings") && body.contains("/object/201d"),
        "{body}"
    );
}
