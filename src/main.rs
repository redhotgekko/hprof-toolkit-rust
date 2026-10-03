//! hprof-toolkit command line.
//!
//! ```text
//! hprof-toolkit index <HPROF>                       build every index (including retained sizes)
//! hprof-toolkit diff  <HPROF> --diff-hprof <HPROF>  build the diff between two dumps
//! hprof-toolkit serve <HPROF> [--diff-hprof <HPROF>] [--port N]   HTTP UI, MCP over HTTP at /mcp
//! hprof-toolkit mcp   <HPROF> [--diff-hprof <HPROF>]              MCP over stdin/stdout
//! ```
//!
//! The heap dump is a positional argument; the older `--hprof <path>` / `-f`
//! spelling still works.  Run any subcommand with `--help` for its flags.

use clap::{Args, Parser, Subcommand};
use hprof_toolkit::{
    diff::HeapDiff,
    pipeline::{IndexOptions, build_all_indexes_with},
    progress::StderrProgress,
    query::HeapQuery,
    server::{
        AppState,
        mcp::{McpServer, serve_stdio},
        start_server,
    },
};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

// ── Command line ──────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "hprof-toolkit",
    version,
    about = "Explore Java heap dumps without loading them into memory"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build every index for a heap dump, including retained sizes.
    Index {
        #[command(flatten)]
        dump: Dump,
        /// Rebuild all indexes even if a previous build finished.
        #[arg(long)]
        force: bool,
        /// Skip the dominator tree and retained sizes. They are the only step whose memory
        /// use grows with the heap.
        #[arg(long)]
        no_retained: bool,
        /// Memory budget for the dominator step, in MiB. The step is refused with a clear
        /// message when its estimate exceeds this. Default: the memory available now.
        #[arg(long, value_name = "MIB")]
        max_memory: Option<u64>,
    },
    /// Build the diff between two heap dumps (also done on demand by `serve` and `mcp`).
    Diff {
        #[command(flatten)]
        dump: Dump,
        /// The later dump, taken from the same JVM process.
        #[arg(long, value_name = "HPROF")]
        diff_hprof: PathBuf,
    },
    /// Serve the HTML UI on localhost, with MCP over HTTP at /mcp.
    Serve {
        #[command(flatten)]
        dump: Dump,
        /// A second dump to compare against.
        #[arg(long, value_name = "HPROF")]
        diff_hprof: Option<PathBuf>,
        /// Port to listen on (localhost only).
        #[arg(short, long, default_value_t = 7000)]
        port: u16,
    },
    /// Run the MCP server on stdin/stdout, for AI clients that launch it as a subprocess.
    Mcp {
        #[command(flatten)]
        dump: Dump,
        /// A second dump to compare against.
        #[arg(long, value_name = "HPROF")]
        diff_hprof: Option<PathBuf>,
    },
}

/// Which dump to open and where its indexes live.
#[derive(Args)]
struct Dump {
    /// The heap dump (.hprof).
    #[arg(value_name = "HPROF", conflicts_with = "hprof_flag")]
    hprof: Option<PathBuf>,
    /// Same as the positional argument (the original spelling).
    #[arg(short = 'f', long = "hprof", value_name = "HPROF")]
    hprof_flag: Option<PathBuf>,
    /// Directory for the index files. Default: `<dump stem>.indexes` next to the dump.
    #[arg(long, value_name = "DIR")]
    index_dir: Option<PathBuf>,
    /// Worker threads for indexing and parallel queries. Default: one per core.
    #[arg(long, value_name = "N")]
    threads: Option<usize>,
}

impl Dump {
    /// The dump path, which must exist.
    fn path(&self) -> Result<PathBuf, String> {
        let path = self
            .hprof
            .clone()
            .or_else(|| self.hprof_flag.clone())
            .ok_or("a heap dump is required: hprof-toolkit <command> <HPROF>")?;
        exists(&path)?;
        Ok(path)
    }

    fn options(&self, retained: bool) -> IndexOptions {
        IndexOptions {
            retained,
            index_dir: self.index_dir.clone(),
            ..IndexOptions::default()
        }
    }

    fn apply_threads(&self) -> Result<(), String> {
        if let Some(n) = self.threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build_global()
                .map_err(|e| format!("could not set the thread count: {e}"))?;
        }
        Ok(())
    }
}

fn exists(path: &std::path::Path) -> Result<(), String> {
    if path.exists() {
        Ok(())
    } else {
        Err(format!("file not found: {}", path.display()))
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli.command).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("Error: {msg}");
            ExitCode::FAILURE
        }
    }
}

async fn run(command: Command) -> Result<(), String> {
    match command {
        Command::Index {
            dump,
            force,
            no_retained,
            max_memory,
        } => {
            dump.apply_threads()?;
            let path = dump.path()?;
            let opts = IndexOptions {
                force,
                max_memory: max_memory.map(|mib| mib.saturating_mul(1024 * 1024)),
                ..dump.options(!no_retained)
            };
            eprintln!("Building indexes for {} …", path.display());
            let dir = build_all_indexes_with(&path, &opts, &StderrProgress)
                .map_err(|e| format!("failed to build indexes: {e}"))?;
            eprintln!("Done: {}", dir.display());
            Ok(())
        }
        Command::Diff { dump, diff_hprof } => {
            dump.apply_threads()?;
            let path = dump.path()?;
            exists(&diff_hprof)?;
            let diff = open_diff(&dump, &path, &diff_hprof)?;
            let c = diff.counts();
            eprintln!(
                "Diff indexes: removed={}, added={}, common={} ({} changed)",
                c.removed, c.added, c.common, c.common_changed
            );
            Ok(())
        }
        Command::Serve {
            dump,
            diff_hprof,
            port,
        } => {
            let state = build_state(&dump, diff_hprof)?;
            start_server(Arc::new(state), port)
                .await
                .map_err(|e| format!("server error: {e}"))
        }
        Command::Mcp { dump, diff_hprof } => {
            let state = build_state(&dump, diff_hprof)?;
            let server = McpServer::new(Arc::new(state));
            eprintln!("MCP server ready on stdin/stdout.");
            serve_stdio(&server, std::io::stdin().lock(), std::io::stdout().lock())
                .map_err(|e| format!("MCP transport error: {e}"))
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Open `path`, building the *cheap* indexes first (progress on stderr).  The
/// dominator tree and retained sizes are the only step whose memory grows
/// with the heap, so they are never started implicitly: run
/// `hprof-toolkit index` to build them (a server picks them up if they exist).
fn open_query(dump: &Dump, path: &std::path::Path) -> Result<HeapQuery, String> {
    eprintln!("Building indexes for {} …", path.display());
    HeapQuery::open_with(path, &dump.options(false), &StderrProgress)
        .map_err(|e| format!("failed to open {}: {e}", path.display()))
}

fn open_diff(
    dump: &Dump,
    path: &std::path::Path,
    diff_path: &std::path::Path,
) -> Result<HeapDiff, String> {
    let before = Arc::new(open_query(dump, path)?);
    // The second dump's indexes sit next to it; `--index-dir` applies to the first.
    let second = Dump {
        hprof: None,
        hprof_flag: None,
        index_dir: None,
        threads: None,
    };
    let after = Arc::new(open_query(&second, diff_path)?);
    eprintln!("Building diff indexes …");
    HeapDiff::open(before, after, path, diff_path)
        .map_err(|e| format!("failed to build diff indexes: {e}"))
}

/// Shared state for `serve` and `mcp`: the dump, plus the diff when a second
/// dump is given.
fn build_state(dump: &Dump, diff_hprof: Option<PathBuf>) -> Result<AppState, String> {
    dump.apply_threads()?;
    let path = dump.path()?;
    match diff_hprof {
        None => Ok(AppState::new(Arc::new(open_query(dump, &path)?), path)),
        Some(diff_path) => {
            exists(&diff_path)?;
            let diff = open_diff(dump, &path, &diff_path)?;
            // `HeapDiff` owns both queries; the state shares the first.
            let first = diff.before_arc();
            Ok(AppState::new(first, path).with_diff(diff, diff_path))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        Cli::try_parse_from(std::iter::once("hprof-toolkit").chain(args.iter().copied()))
    }

    #[test]
    fn the_command_line_definition_is_consistent() {
        Cli::command().debug_assert();
    }

    #[test]
    fn positional_and_old_flag_spellings_both_work() {
        for args in [
            ["index", "heap.hprof"].as_slice(),
            ["index", "--hprof", "heap.hprof"].as_slice(),
            ["index", "-f", "heap.hprof"].as_slice(),
            ["index", "--hprof=heap.hprof"].as_slice(),
        ] {
            let Command::Index { dump, .. } = parse(args).unwrap().command else {
                panic!("not index");
            };
            let got = dump.hprof.or(dump.hprof_flag).unwrap();
            assert_eq!(got, PathBuf::from("heap.hprof"), "{args:?}");
        }
        assert!(parse(&["index", "a.hprof", "--hprof", "b.hprof"]).is_err());
    }

    #[test]
    fn index_flags_are_parsed() {
        let cli = parse(&[
            "index",
            "h.hprof",
            "--force",
            "--no-retained",
            "--max-memory",
            "512",
            "--index-dir",
            "/idx",
            "--threads",
            "4",
        ])
        .unwrap();
        let Command::Index {
            dump,
            force,
            no_retained,
            max_memory,
        } = cli.command
        else {
            panic!("not index");
        };
        assert!(force && no_retained);
        assert_eq!(max_memory, Some(512));
        assert_eq!(dump.index_dir, Some(PathBuf::from("/idx")));
        assert_eq!(dump.threads, Some(4));
        let opts = dump.options(false);
        assert!(!opts.retained);
        assert_eq!(opts.index_dir, Some(PathBuf::from("/idx")));
    }

    #[test]
    fn serve_and_mcp_and_diff_take_their_options() {
        let Command::Serve {
            port, diff_hprof, ..
        } = parse(&["serve", "a.hprof", "--diff-hprof", "b.hprof", "-p", "8080"])
            .unwrap()
            .command
        else {
            panic!("not serve");
        };
        assert_eq!((port, diff_hprof), (8080, Some(PathBuf::from("b.hprof"))));
        let Command::Serve { port, .. } = parse(&["serve", "a.hprof"]).unwrap().command else {
            panic!("not serve");
        };
        assert_eq!(port, 7000);
        assert!(matches!(
            parse(&["mcp", "-f", "a.hprof"]).unwrap().command,
            Command::Mcp { .. }
        ));
        assert!(
            parse(&["diff", "a.hprof"]).is_err(),
            "--diff-hprof is required"
        );
        assert!(parse(&["frobnicate"]).is_err());
    }

    #[test]
    fn a_missing_dump_is_reported() {
        let Command::Index { dump, .. } = parse(&["index"]).unwrap().command else {
            panic!("not index");
        };
        assert!(dump.path().unwrap_err().contains("heap dump is required"));
        let Command::Index { dump, .. } = parse(&["index", "/no/such/file.hprof"]).unwrap().command
        else {
            panic!("not index");
        };
        assert!(dump.path().unwrap_err().contains("file not found"));
    }
}
