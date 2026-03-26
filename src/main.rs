//! hprof-toolkit — unified CLI
//!
//! # Subcommands
//!
//! ```text
//! hprof-toolkit index  --hprof <path>                            Build all indexes for a heap dump
//! hprof-toolkit diff   --hprof <path> --diff-hprof <path>        Build diff indexes for two heap dumps
//! hprof-toolkit serve  --hprof <path> [--diff-hprof <path>] [--port <port>]
//!                                                                  Start the HTTP / MCP server
//! ```

use hprof_toolkit::{
    diff_index::{DiffIndexPaths, build_diff_indexes},
    pipeline::{IndexPaths, build_all_indexes},
    query::HeapQuery,
    server::{AppState, start_server},
};
use std::path::PathBuf;
use std::sync::Arc;

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let subcommand = args.get(1).map(String::as_str);

    match subcommand {
        Some("index") => {
            let hprof_path = match parse_hprof_arg(&args[2..]) {
                Ok(p) => p,
                Err(msg) => die(&msg, "index --hprof <path>"),
            };
            cmd_index(&hprof_path);
        }
        Some("diff") => {
            let (hprof_path, diff_hprof_path) = match parse_diff_args(&args[2..]) {
                Ok(pair) => pair,
                Err(msg) => die(&msg, "diff --hprof <path> --diff-hprof <path>"),
            };
            cmd_diff(&hprof_path, &diff_hprof_path);
        }
        Some("serve") => {
            let (hprof_path, diff_hprof_path, port) = match parse_serve_args(&args[2..]) {
                Ok(t) => t,
                Err(msg) => die(
                    &msg,
                    "serve --hprof <path> [--diff-hprof <path>] [--port <port>]",
                ),
            };
            cmd_serve(hprof_path, diff_hprof_path, port).await;
        }
        _ => {
            eprintln!("Usage: hprof-toolkit <subcommand> [options]");
            eprintln!();
            eprintln!("Subcommands:");
            eprintln!("  index   Build all indexes for a heap dump");
            eprintln!("  diff    Build diff indexes for two heap dumps");
            eprintln!("  serve   Start the HTTP / MCP server");
            std::process::exit(1);
        }
    }
}

fn die(msg: &str, usage: &str) -> ! {
    eprintln!("Error: {msg}");
    eprintln!("Usage: hprof-toolkit {usage}");
    std::process::exit(1);
}

// ── index ─────────────────────────────────────────────────────────────────────

fn cmd_index(hprof_path: &std::path::Path) {
    eprintln!("Building indexes for {} …", hprof_path.display());
    match build_all_indexes(hprof_path) {
        Ok(_) => eprintln!("Done."),
        Err(e) => {
            eprintln!("Failed to build indexes: {e}");
            std::process::exit(1);
        }
    }
}

// ── diff ──────────────────────────────────────────────────────────────────────

fn cmd_diff(hprof_path: &std::path::Path, diff_hprof_path: &std::path::Path) {
    eprintln!("Building indexes for {} …", hprof_path.display());
    let paths = require_indexes(hprof_path);

    eprintln!(
        "Building indexes for diff dump {} …",
        diff_hprof_path.display()
    );
    let diff_paths = require_indexes(diff_hprof_path);

    let query = open_query(hprof_path, &paths);
    let diff_query = open_query(diff_hprof_path, &diff_paths);

    let diff_index_paths = DiffIndexPaths::for_hprofs(hprof_path, diff_hprof_path);
    if diff_index_paths.all_exist() {
        eprintln!("Diff indexes: skipping (already built)");
        return;
    }

    eprintln!(
        "Building diff indexes in {} …",
        diff_index_paths.dir.display()
    );
    match build_diff_indexes(&query, &diff_query, &diff_index_paths) {
        Ok(counts) => eprintln!(
            "Diff indexes: removed={}, added={}, common={} ({} changed)",
            counts.removed, counts.added, counts.common, counts.common_changed
        ),
        Err(e) => {
            eprintln!("Failed to build diff indexes: {e}");
            std::process::exit(1);
        }
    }
}

// ── serve ─────────────────────────────────────────────────────────────────────

async fn cmd_serve(hprof_path: PathBuf, diff_hprof_path: Option<PathBuf>, port: u16) {
    eprintln!("Building indexes for {} …", hprof_path.display());
    let paths = require_indexes(&hprof_path);
    let query = open_query(&hprof_path, &paths);

    let diff_state: Option<(HeapQuery, PathBuf, DiffIndexPaths)> =
        if let Some(diff_path) = diff_hprof_path {
            eprintln!("Building indexes for diff dump {} …", diff_path.display());
            let diff_idx_paths = require_indexes(&diff_path);
            let diff_query = open_query(&diff_path, &diff_idx_paths);

            let diff_index_paths = DiffIndexPaths::for_hprofs(&hprof_path, &diff_path);
            if diff_index_paths.all_exist() {
                eprintln!("Diff indexes: skipping (already built)");
            } else {
                eprintln!(
                    "Building diff indexes in {} …",
                    diff_index_paths.dir.display()
                );
                match build_diff_indexes(&query, &diff_query, &diff_index_paths) {
                    Ok(counts) => eprintln!(
                        "Diff indexes: removed={}, added={}, common={} ({} changed)",
                        counts.removed, counts.added, counts.common, counts.common_changed
                    ),
                    Err(e) => {
                        eprintln!("Failed to build diff indexes: {e}");
                        std::process::exit(1);
                    }
                }
            }

            Some((diff_query, diff_path, diff_index_paths))
        } else {
            None
        };

    let mut app_state = AppState::new(query, hprof_path);
    if let Some((diff_query, diff_path, diff_index_paths)) = diff_state {
        app_state = app_state.with_diff(diff_query, diff_path, diff_index_paths);
    }

    if let Err(e) = start_server(Arc::new(app_state), port).await {
        eprintln!("Server error: {e}");
        std::process::exit(1);
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn require_indexes(hprof_path: &std::path::Path) -> IndexPaths {
    match build_all_indexes(hprof_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to build indexes: {e}");
            std::process::exit(1);
        }
    }
}

fn open_query(hprof_path: &std::path::Path, paths: &IndexPaths) -> HeapQuery {
    match HeapQuery::open(hprof_path, paths) {
        Ok(q) => q,
        Err(e) => {
            eprintln!("Failed to open heap index: {e}");
            std::process::exit(1);
        }
    }
}

// ── argument parsers ──────────────────────────────────────────────────────────

fn parse_hprof_arg(args: &[String]) -> Result<PathBuf, String> {
    let mut hprof_path: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--hprof" | "-f" => {
                i += 1;
                match args.get(i) {
                    Some(p) => hprof_path = Some(PathBuf::from(p)),
                    None => return Err("--hprof requires a path argument".to_owned()),
                }
            }
            arg if arg.starts_with("--hprof=") => {
                hprof_path = Some(PathBuf::from(&arg["--hprof=".len()..]));
            }
            arg => return Err(format!("Unknown argument: {arg}")),
        }
        i += 1;
    }
    let path = hprof_path.ok_or_else(|| "--hprof <path> is required".to_owned())?;
    check_exists(&path)?;
    Ok(path)
}

fn parse_diff_args(args: &[String]) -> Result<(PathBuf, PathBuf), String> {
    let mut hprof_path: Option<PathBuf> = None;
    let mut diff_hprof_path: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--hprof" | "-f" => {
                i += 1;
                match args.get(i) {
                    Some(p) => hprof_path = Some(PathBuf::from(p)),
                    None => return Err("--hprof requires a path argument".to_owned()),
                }
            }
            "--diff-hprof" => {
                i += 1;
                match args.get(i) {
                    Some(p) => diff_hprof_path = Some(PathBuf::from(p)),
                    None => return Err("--diff-hprof requires a path argument".to_owned()),
                }
            }
            arg if arg.starts_with("--hprof=") => {
                hprof_path = Some(PathBuf::from(&arg["--hprof=".len()..]));
            }
            arg if arg.starts_with("--diff-hprof=") => {
                diff_hprof_path = Some(PathBuf::from(&arg["--diff-hprof=".len()..]));
            }
            arg => return Err(format!("Unknown argument: {arg}")),
        }
        i += 1;
    }
    let path = hprof_path.ok_or_else(|| "--hprof <path> is required".to_owned())?;
    let diff_path = diff_hprof_path.ok_or_else(|| "--diff-hprof <path> is required".to_owned())?;
    check_exists(&path)?;
    check_exists(&diff_path)?;
    Ok((path, diff_path))
}

fn parse_serve_args(args: &[String]) -> Result<(PathBuf, Option<PathBuf>, u16), String> {
    let mut hprof_path: Option<PathBuf> = None;
    let mut diff_hprof_path: Option<PathBuf> = None;
    let mut port: u16 = 7000;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--hprof" | "-f" => {
                i += 1;
                match args.get(i) {
                    Some(p) => hprof_path = Some(PathBuf::from(p)),
                    None => return Err("--hprof requires a path argument".to_owned()),
                }
            }
            "--diff-hprof" => {
                i += 1;
                match args.get(i) {
                    Some(p) => diff_hprof_path = Some(PathBuf::from(p)),
                    None => return Err("--diff-hprof requires a path argument".to_owned()),
                }
            }
            "--port" | "-p" => {
                i += 1;
                match args.get(i) {
                    Some(p) => port = p.parse().map_err(|_| format!("Invalid port: {p}"))?,
                    None => return Err("--port requires a number argument".to_owned()),
                }
            }
            arg if arg.starts_with("--hprof=") => {
                hprof_path = Some(PathBuf::from(&arg["--hprof=".len()..]));
            }
            arg if arg.starts_with("--diff-hprof=") => {
                diff_hprof_path = Some(PathBuf::from(&arg["--diff-hprof=".len()..]));
            }
            arg if arg.starts_with("--port=") => {
                let s = &arg["--port=".len()..];
                port = s.parse().map_err(|_| format!("Invalid port: {s}"))?;
            }
            arg => return Err(format!("Unknown argument: {arg}")),
        }
        i += 1;
    }
    let path = hprof_path.ok_or_else(|| "--hprof <path> is required".to_owned())?;
    check_exists(&path)?;
    if let Some(ref dp) = diff_hprof_path {
        check_exists(dp)?;
    }
    Ok((path, diff_hprof_path, port))
}

fn check_exists(path: &std::path::Path) -> Result<(), String> {
    if path.exists() {
        Ok(())
    } else {
        Err(format!("File not found: {}", path.display()))
    }
}
