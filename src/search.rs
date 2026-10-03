//! Text search: one [`Matcher`] shared by the library, the HTML UI and the
//! MCP tools.
//!
//! A [`SearchQuery`] says what to look for and how ([`SearchMode`]); a
//! [`Matcher`] is its compiled form, cheap to share across threads and to
//! apply to many candidates:
//!
//! ```
//! use hprof_toolkit::prelude::*;
//!
//! let m = Matcher::new(&SearchQuery::regex(r"^java\.util\..*Map$"))?;
//! assert!(m.is_match("java.util.HashMap"));
//! assert!(!m.is_match("java.util.ArrayList"));
//!
//! let fuzzy = Matcher::new(&SearchQuery::fuzzy("hm"))?;
//! assert!(fuzzy.score("java.util.HashMap") > fuzzy.score("com.example.SchemaManager"));
//! # Ok::<(), HprofError>(())
//! ```
//!
//! `Contains`, `Exact` and `Regex` all compile to a [`regex::Regex`], so they
//! share one code path, need no per-candidate lowercasing, and inherit the
//! regex crate's linear-time guarantee (a hostile pattern cannot hang a
//! server).  `Fuzzy` is a small in-house subsequence matcher with a score.

use crate::hprof::HprofError;
use regex::{Regex, RegexBuilder};

/// Longest pattern accepted, in bytes.
pub const MAX_PATTERN_LEN: usize = 512;
/// Upper bound on the compiled size of a regex (and of its lazy DFA cache).
const REGEX_SIZE_LIMIT: usize = 1 << 20;

/// How [`SearchQuery::text`] is interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SearchMode {
    /// Substring.  The default.
    #[default]
    Contains,
    /// The whole candidate must equal the text.
    Exact,
    /// The text is a regular expression (Rust `regex` syntax, unanchored).
    Regex,
    /// Every character of the text appears in the candidate, in order.
    /// Matches are ranked, best first.
    Fuzzy,
}

impl SearchMode {
    /// Every mode, in the order user interfaces list them.
    pub const ALL: [SearchMode; 4] = [
        SearchMode::Contains,
        SearchMode::Exact,
        SearchMode::Regex,
        SearchMode::Fuzzy,
    ];

    /// The lowercase name used in URLs and tool arguments.
    pub fn name(self) -> &'static str {
        match self {
            SearchMode::Contains => "contains",
            SearchMode::Exact => "exact",
            SearchMode::Regex => "regex",
            SearchMode::Fuzzy => "fuzzy",
        }
    }

    /// The inverse of [`Self::name`].
    pub fn from_name(name: &str) -> Option<SearchMode> {
        SearchMode::ALL.into_iter().find(|m| m.name() == name)
    }
}

/// What to search for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchQuery {
    pub text: String,
    pub mode: SearchMode,
    /// Default `false`: every mode ignores case unless asked not to.
    pub case_sensitive: bool,
}

impl SearchQuery {
    /// A query in `mode`, case-insensitive.
    pub fn new(text: impl Into<String>, mode: SearchMode) -> Self {
        Self {
            text: text.into(),
            mode,
            case_sensitive: false,
        }
    }

    /// Substring match.
    pub fn contains(text: impl Into<String>) -> Self {
        Self::new(text, SearchMode::Contains)
    }

    /// Whole-string match.
    pub fn exact(text: impl Into<String>) -> Self {
        Self::new(text, SearchMode::Exact)
    }

    /// Regular expression match.
    pub fn regex(text: impl Into<String>) -> Self {
        Self::new(text, SearchMode::Regex)
    }

    /// Ranked subsequence match.
    pub fn fuzzy(text: impl Into<String>) -> Self {
        Self::new(text, SearchMode::Fuzzy)
    }

    /// Match case exactly (the default is to ignore it).
    pub fn case_sensitive(mut self, yes: bool) -> Self {
        self.case_sensitive = yes;
        self
    }
}

/// A compiled [`SearchQuery`].  `Send + Sync`, so one matcher can serve
/// every rayon worker.
#[derive(Debug, Clone)]
pub struct Matcher {
    query: SearchQuery,
    engine: Engine,
}

#[derive(Debug, Clone)]
enum Engine {
    Regex(Regex),
    Fuzzy(FuzzyPattern),
}

impl Matcher {
    /// Compile `query`.  Fails with [`HprofError::InvalidArgument`] when the
    /// text is empty or longer than [`MAX_PATTERN_LEN`] bytes, or when a
    /// regex does not parse or would compile to something too large.
    pub fn new(query: &SearchQuery) -> Result<Matcher, HprofError> {
        if query.text.is_empty() {
            return Err(HprofError::InvalidArgument(
                "search text must not be empty".to_owned(),
            ));
        }
        if query.text.len() > MAX_PATTERN_LEN {
            return Err(HprofError::InvalidArgument(format!(
                "search text is {} bytes; at most {MAX_PATTERN_LEN} are allowed",
                query.text.len()
            )));
        }
        let engine = match query.mode {
            SearchMode::Fuzzy => {
                Engine::Fuzzy(FuzzyPattern::new(&query.text, query.case_sensitive))
            }
            mode => {
                let pattern = match mode {
                    SearchMode::Contains => regex::escape(&query.text),
                    SearchMode::Exact => format!("^(?:{})$", regex::escape(&query.text)),
                    _ => query.text.clone(),
                };
                let regex = RegexBuilder::new(&pattern)
                    .case_insensitive(!query.case_sensitive)
                    .size_limit(REGEX_SIZE_LIMIT)
                    .dfa_size_limit(REGEX_SIZE_LIMIT)
                    .build()
                    .map_err(|e| {
                        HprofError::InvalidArgument(format!("invalid regex `{}`: {e}", query.text))
                    })?;
                Engine::Regex(regex)
            }
        };
        Ok(Matcher {
            query: query.clone(),
            engine,
        })
    }

    /// The query this matcher was compiled from.
    pub fn query(&self) -> &SearchQuery {
        &self.query
    }

    /// The mode this matcher was compiled in.
    pub fn mode(&self) -> SearchMode {
        self.query.mode
    }

    /// `true` for [`SearchMode::Fuzzy`]: results should be ordered by
    /// [`Self::score`], best first.
    pub fn ranks(&self) -> bool {
        matches!(self.engine, Engine::Fuzzy(_))
    }

    /// Does `candidate` match?
    pub fn is_match(&self, candidate: &str) -> bool {
        match &self.engine {
            Engine::Regex(r) => r.is_match(candidate),
            Engine::Fuzzy(f) => f.score(candidate).is_some(),
        }
    }

    /// `None` when `candidate` does not match.  For `Fuzzy`, higher is
    /// better; for the other modes every match scores `0`.
    pub fn score(&self, candidate: &str) -> Option<u32> {
        match &self.engine {
            Engine::Regex(r) => r.is_match(candidate).then_some(0),
            Engine::Fuzzy(f) => f.score(candidate),
        }
    }
}

// ── Fuzzy matching ────────────────────────────────────────────────────────────

/// A subsequence pattern with a score that prefers runs of consecutive
/// characters and characters at word boundaries, and penalises gaps.
///
/// Matching is greedy (each pattern character takes the first candidate
/// character that fits), which is enough to rank `hm` on
/// `java.util.HashMap` above `com.example.SchemaManager`.
#[derive(Debug, Clone)]
struct FuzzyPattern {
    chars: Vec<char>,
    case_sensitive: bool,
}

const CONSECUTIVE_BONUS: i32 = 2;
const BOUNDARY_BONUS: i32 = 3;
const GAP_PENALTY: i32 = 1;

impl FuzzyPattern {
    fn new(text: &str, case_sensitive: bool) -> Self {
        let chars = text
            .chars()
            .map(|c| if case_sensitive { c } else { fold(c) })
            .collect();
        Self {
            chars,
            case_sensitive,
        }
    }

    fn score(&self, candidate: &str) -> Option<u32> {
        let mut score: i32 = 0;
        let mut next = 0; // index into self.chars
        let mut prev_matched_at: Option<usize> = None;
        let mut prev_char: Option<char> = None;
        for (i, c) in candidate.chars().enumerate() {
            if next < self.chars.len() {
                let folded = if self.case_sensitive { c } else { fold(c) };
                if folded == self.chars[next] {
                    score += 1;
                    match prev_matched_at {
                        Some(p) if p + 1 == i => score += CONSECUTIVE_BONUS,
                        Some(p) => score -= GAP_PENALTY * (i - p - 1) as i32,
                        None => {}
                    }
                    if is_boundary(prev_char, c) {
                        score += BOUNDARY_BONUS;
                    }
                    prev_matched_at = Some(i);
                    next += 1;
                }
            }
            prev_char = Some(c);
        }
        (next == self.chars.len()).then(|| score.max(0) as u32)
    }
}

/// Case-fold one character (first char of its lowercase mapping).
fn fold(c: char) -> char {
    c.to_lowercase().next().unwrap_or(c)
}

/// Is `c` the start of a word: first character, after a separator, or an
/// uppercase letter following a lowercase one (`HashMap` → `Map`)?
fn is_boundary(prev: Option<char>, c: char) -> bool {
    match prev {
        None => true,
        Some(p) => {
            matches!(p, '.' | '$' | '_' | '/' | '[' | ' ' | '-')
                || (p.is_lowercase() && c.is_uppercase())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(q: SearchQuery) -> Matcher {
        Matcher::new(&q).expect("valid query")
    }

    #[test]
    fn contains_ignores_case_unless_asked() {
        assert!(m(SearchQuery::contains("hashmap")).is_match("java.util.HashMap"));
        assert!(
            !m(SearchQuery::contains("hashmap").case_sensitive(true)).is_match("java.util.HashMap")
        );
        assert!(
            m(SearchQuery::contains("HashMap").case_sensitive(true)).is_match("java.util.HashMap")
        );
        // Metacharacters are literal in `contains`.
        assert!(m(SearchQuery::contains("[C")).is_match("[C"));
        assert!(!m(SearchQuery::contains("[C")).is_match("C"));
        assert_eq!(m(SearchQuery::contains("x")).score("x"), Some(0));
        assert_eq!(m(SearchQuery::contains("x")).score("y"), None);
    }

    #[test]
    fn exact_needs_the_whole_string() {
        let e = m(SearchQuery::exact("java.lang.String"));
        assert!(e.is_match("java.lang.String"));
        assert!(e.is_match("JAVA.LANG.STRING"));
        assert!(!e.is_match("java.lang.StringBuilder"));
        assert!(!e.is_match("java.lang.Str"));
        assert!(!m(SearchQuery::exact("[C")).is_match("[[C"));
    }

    #[test]
    fn regex_mode() {
        let r = m(SearchQuery::regex(r"^java\.util\..*Map$"));
        assert!(r.is_match("java.util.HashMap"));
        assert!(r.is_match("JAVA.UTIL.TREEMAP"));
        assert!(!r.is_match("java.util.ArrayList"));
        assert!(!m(SearchQuery::regex("^Map$").case_sensitive(true)).is_match("map"));
        assert!(m(SearchQuery::regex("(?i)^map$").case_sensitive(true)).is_match("MAP"));
        assert!(!r.ranks());
    }

    #[test]
    fn bad_patterns_are_invalid_arguments() {
        let err = Matcher::new(&SearchQuery::regex("(")).expect_err("unbalanced");
        assert!(matches!(err, HprofError::InvalidArgument(ref s) if s.contains("invalid regex")));
        for mode in SearchMode::ALL {
            let err = Matcher::new(&SearchQuery::new("", mode)).expect_err("empty");
            assert!(matches!(err, HprofError::InvalidArgument(_)));
        }
        let long = "a".repeat(MAX_PATTERN_LEN + 1);
        assert!(Matcher::new(&SearchQuery::contains(long)).is_err());
        // Too large to compile within the size limit, rejected rather than hung.
        assert!(Matcher::new(&SearchQuery::regex("(a{1000}){1000}")).is_err());
    }

    #[test]
    fn fuzzy_matches_subsequences_and_ranks_them() {
        let f = m(SearchQuery::fuzzy("hm"));
        assert!(f.ranks());
        let hashmap = f.score("java.util.HashMap").expect("matches");
        let schema = f.score("com.example.SchemaManager").expect("matches");
        assert!(hashmap > schema, "{hashmap} vs {schema}");
        assert_eq!(f.score("java.util.ArrayList"), None);
        assert!(f.is_match("HM"));

        assert!(m(SearchQuery::fuzzy("julist")).is_match("java.util.List"));
        assert!(!m(SearchQuery::fuzzy("zzz")).is_match("java.util.List"));
        // Order matters: it is a subsequence, not a bag of characters.
        assert!(!m(SearchQuery::fuzzy("ba")).is_match("ab"));
        // Case-sensitive fuzzy.
        assert!(!m(SearchQuery::fuzzy("HM").case_sensitive(true)).is_match("hashmap"));
        assert!(m(SearchQuery::fuzzy("HM").case_sensitive(true)).is_match("HashMap"));
    }

    #[test]
    fn fuzzy_prefers_consecutive_runs_and_boundaries() {
        let f = m(SearchQuery::fuzzy("map"));
        let consecutive = f.score("java.util.Map").expect("matches");
        let scattered = f.score("m.a.x.p").expect("matches");
        assert!(consecutive > scattered);
        // `Map` at a camel-case boundary beats `map` buried in a word.
        let boundary = f.score("HashMap").expect("matches");
        let buried = f.score("remapping").expect("matches");
        assert!(boundary > buried, "{boundary} vs {buried}");
    }

    #[test]
    fn mode_names_round_trip() {
        for mode in SearchMode::ALL {
            assert_eq!(SearchMode::from_name(mode.name()), Some(mode));
        }
        assert_eq!(SearchMode::from_name("glob"), None);
        assert_eq!(SearchMode::default(), SearchMode::Contains);
    }

    #[test]
    fn matcher_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Matcher>();
    }
}
