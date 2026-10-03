//! Progress reporting for long-running index builds.
//!
//! The library never prints.  Callers that want to see what the pipeline is
//! doing pass a [`Progress`] implementation; the CLI passes
//! [`StderrProgress`] (stdout is reserved for the MCP stdio transport), tests
//! pass [`NoProgress`].

/// Receives one message per pipeline step.
pub trait Progress: Sync {
    /// `stage` is a short fixed label ("Record index"); `detail` is what
    /// happened ("skipping", "12345 records (0.4s)").
    fn step(&self, stage: &str, detail: &str);
}

/// Discards all progress messages.
pub struct NoProgress;

impl Progress for NoProgress {
    fn step(&self, _stage: &str, _detail: &str) {}
}

/// Writes `stage: detail` lines to stderr.
pub struct StderrProgress;

impl Progress for StderrProgress {
    fn step(&self, stage: &str, detail: &str) {
        eprintln!("{stage}: {detail}");
    }
}
