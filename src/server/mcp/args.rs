//! Tool arguments: typed access to the JSON object a client sent, with
//! errors written for the model that will read them.

use crate::hprof::HprofError;
use crate::query::Page;
use serde_json::{Map, Value};

/// Largest page any tool returns.
pub const MAX_LIMIT: usize = 500;
/// Page size when the client does not say.
pub const DEFAULT_LIMIT: usize = 50;

/// An error a tool reports to the model (`isError: true`), not a protocol error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolError(pub String);

impl ToolError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl From<HprofError> for ToolError {
    fn from(e: HprofError) -> Self {
        Self(match e {
            HprofError::InvalidArgument(m) => format!("Invalid argument: {m}"),
            HprofError::NotIndexed(what) => format!(
                "The {what} index has not been built. Run `hprof-toolkit index <hprof>` first \
                 (this tool cannot build it)."
            ),
            other => format!("Internal error reading the heap dump: {other}"),
        })
    }
}

/// The `arguments` object of a `tools/call`.
pub struct Args<'a>(Option<&'a Map<String, Value>>);

impl<'a> Args<'a> {
    pub fn new(v: &'a Value) -> Self {
        Self(v.as_object())
    }

    fn get(&self, key: &str) -> Option<&'a Value> {
        self.0.and_then(|m| m.get(key)).filter(|v| !v.is_null())
    }

    /// A required string.
    pub fn str(&self, key: &str) -> Result<&'a str, ToolError> {
        match self.get(key) {
            Some(Value::String(s)) => Ok(s),
            Some(_) => Err(ToolError::new(format!("`{key}` must be a string."))),
            None => Err(ToolError::new(format!(
                "Missing required argument `{key}`."
            ))),
        }
    }

    /// An optional string.
    pub fn opt_str(&self, key: &str) -> Result<Option<&'a str>, ToolError> {
        match self.get(key) {
            None => Ok(None),
            Some(Value::String(s)) => Ok(Some(s)),
            Some(_) => Err(ToolError::new(format!("`{key}` must be a string."))),
        }
    }

    /// A required object id.
    pub fn id(&self, key: &str) -> Result<u64, ToolError> {
        self.opt_id(key)?
            .ok_or_else(|| ToolError::new(format!("Missing required argument `{key}`.")))
    }

    /// An optional object id: a string of hex digits with an optional `0x`
    /// prefix (`"0x1a2b"`, `"1a2b"`), or a JSON number, which is decimal.
    pub fn opt_id(&self, key: &str) -> Result<Option<u64>, ToolError> {
        let bad = || {
            ToolError::new(format!(
                "`{key}` must be an object id: a hex string like \"0x1a2b3c\" (as shown in other \
                 tool results) or a decimal number."
            ))
        };
        match self.get(key) {
            None => Ok(None),
            Some(Value::Number(n)) => n.as_u64().map(Some).ok_or_else(bad),
            Some(Value::String(s)) => {
                let t = s.trim();
                let digits = t
                    .strip_prefix("0x")
                    .or_else(|| t.strip_prefix("0X"))
                    .unwrap_or(t);
                u64::from_str_radix(digits, 16).map(Some).map_err(|_| bad())
            }
            Some(_) => Err(bad()),
        }
    }

    /// An optional non-negative integer with a default and an upper bound.
    pub fn usize_or(&self, key: &str, default: usize, max: usize) -> Result<usize, ToolError> {
        match self.get(key) {
            None => Ok(default),
            Some(v) => match v.as_u64() {
                Some(n) => Ok((n as usize).min(max)),
                None => Err(ToolError::new(format!(
                    "`{key}` must be a non-negative integer."
                ))),
            },
        }
    }

    /// An optional boolean with a default.
    pub fn bool_or(&self, key: &str, default: bool) -> Result<bool, ToolError> {
        Ok(self.opt_bool(key)?.unwrap_or(default))
    }

    /// An optional boolean.
    pub fn opt_bool(&self, key: &str) -> Result<Option<bool>, ToolError> {
        match self.get(key) {
            None => Ok(None),
            Some(Value::Bool(b)) => Ok(Some(*b)),
            Some(_) => Err(ToolError::new(format!("`{key}` must be true or false."))),
        }
    }

    /// `offset` / `limit` (default [`DEFAULT_LIMIT`], at most [`MAX_LIMIT`]).
    pub fn page(&self) -> Result<Page, ToolError> {
        self.page_with(DEFAULT_LIMIT, MAX_LIMIT)
    }

    /// [`Self::page`] with a tool-specific default and maximum limit.
    pub fn page_with(&self, default_limit: usize, max_limit: usize) -> Result<Page, ToolError> {
        Ok(Page::new(
            self.usize_or("offset", 0, usize::MAX)?,
            self.usize_or("limit", default_limit, max_limit)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn ids_accept_hex_strings_and_decimal_numbers() {
        let v =
            json!({"a": "0x1F", "b": "1f", "c": 31, "d": "  0X1f ", "e": "zz", "f": -1, "g": true});
        let a = Args::new(&v);
        for k in ["a", "b", "c", "d"] {
            assert_eq!(a.id(k).unwrap(), 31, "{k}");
        }
        for k in ["e", "f", "g"] {
            assert!(a.id(k).unwrap_err().0.contains("object id"), "{k}");
        }
        assert!(a.id("missing").unwrap_err().0.contains("Missing required"));
        assert_eq!(a.opt_id("missing").unwrap(), None);
    }

    #[test]
    fn paging_defaults_and_bounds() {
        let v = json!({"offset": 7, "limit": 100000});
        assert_eq!(Args::new(&v).page().unwrap(), Page::new(7, MAX_LIMIT));
        let none = json!({});
        assert_eq!(
            Args::new(&none).page().unwrap(),
            Page::new(0, DEFAULT_LIMIT)
        );
        let bad = json!({"limit": "ten"});
        assert!(Args::new(&bad).page().is_err());
        // Arguments that are not an object behave as empty.
        let s = json!("x");
        assert_eq!(Args::new(&s).page().unwrap(), Page::new(0, DEFAULT_LIMIT));
    }

    #[test]
    fn typed_getters_report_wrong_types() {
        let v = json!({"s": "x", "n": 3, "b": true});
        let a = Args::new(&v);
        assert_eq!(a.str("s").unwrap(), "x");
        assert!(a.str("n").unwrap_err().0.contains("must be a string"));
        assert!(a.bool_or("b", false).unwrap());
        assert!(a.bool_or("s", false).is_err());
        assert_eq!(a.usize_or("n", 0, 2).unwrap(), 2);
    }

    #[test]
    fn hprof_errors_become_actionable_text() {
        let e: ToolError = HprofError::NotIndexed("retained heap").into();
        assert!(e.0.contains("retained heap") && e.0.contains("hprof-toolkit index"));
        let e: ToolError = HprofError::InvalidArgument("bad".into()).into();
        assert_eq!(e.0, "Invalid argument: bad");
    }
}
