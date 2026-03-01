//! Tokenization helpers.
//!
//! We expose a lightweight whitespace tokenizer used by the retrieval engine
//! and the training data pipeline.  The `tokenizers` crate (HuggingFace) is
//! imported here so it is actually referenced by the build — a pretrained BPE
//! vocabulary can be loaded and used by calling `BpeTokenizer::from_file`.

// Bring the tokenizers crate into scope so the dependency is exercised.
use tokenizers::Tokenizer as HfTokenizer;

// ---------------------------------------------------------------------------
// HuggingFace BPE wrapper (optional, needs a vocab file on disk)
// ---------------------------------------------------------------------------

/// Thin wrapper around the HuggingFace `Tokenizer`.
/// Load with `BpeTokenizer::from_file("tokenizer.json")`.
#[allow(dead_code)]
pub struct BpeTokenizer {
    inner: HfTokenizer,
}

impl BpeTokenizer {
    /// Load a pretrained tokenizer from a JSON file (HuggingFace format).
    #[allow(dead_code)]
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;
        Ok(Self { inner })
    }

    /// Encode text to token IDs.
    #[allow(dead_code)]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        match self.inner.encode(text, false) {
            Ok(enc) => enc.get_ids().to_vec(),
            Err(_) => vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// Built-in whitespace tokenizer (used by default — no file required)
// ---------------------------------------------------------------------------

/// Tokenize text: lowercase, strip punctuation, split on whitespace.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Return `Some(year)` if `token` looks like a 4-digit year in [2020, 2030].
pub fn is_year(token: &str) -> Option<i32> {
    if token.len() == 4 {
        if let Ok(y) = token.parse::<i32>() {
            if (2020..=2030).contains(&y) {
                return Some(y);
            }
        }
    }
    None
}

/// Match a token prefix to a calendar month name.
/// Requires at least 3 characters.  Returns the full uppercase month name.
pub fn match_month(token: &str) -> Option<String> {
    let months = [
        "january","february","march","april","may","june",
        "july","august","september","october","november","december",
    ];
    let t = token.to_lowercase();
    if t.len() < 3 { return None; }
    for m in &months {
        if m.starts_with(t.as_str()) {
            return Some(m.to_uppercase());
        }
    }
    None
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- tokenize() ---

    #[test]
    fn tokenize_basic_splits_on_whitespace() {
        let tokens = tokenize("Hello World");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn tokenize_strips_punctuation() {
        let tokens = tokenize("Higher Degrees Committee (09:00)");
        // parentheses and colon stripped, digits kept
        assert!(tokens.contains(&"higher".to_string()));
        assert!(tokens.contains(&"degrees".to_string()));
        assert!(tokens.contains(&"committee".to_string()));
    }

    #[test]
    fn tokenize_lowercases_input() {
        let tokens = tokenize("JANUARY 2026");
        assert_eq!(tokens[0], "january");
        assert_eq!(tokens[1], "2026");
    }

    #[test]
    fn tokenize_empty_string_returns_empty() {
        let tokens = tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn tokenize_ignores_extra_whitespace() {
        let tokens = tokenize("  hello   world  ");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    // --- is_year() ---

    #[test]
    fn is_year_detects_valid_years() {
        assert_eq!(is_year("2024"), Some(2024));
        assert_eq!(is_year("2025"), Some(2025));
        assert_eq!(is_year("2026"), Some(2026));
    }

    #[test]
    fn is_year_rejects_out_of_range() {
        assert_eq!(is_year("1999"), None);
        assert_eq!(is_year("2031"), None);
    }

    #[test]
    fn is_year_rejects_non_numeric() {
        assert_eq!(is_year("abcd"), None);
        assert_eq!(is_year("20ab"), None);
    }

    #[test]
    fn is_year_rejects_wrong_length() {
        assert_eq!(is_year("202"),  None);
        assert_eq!(is_year("20251"), None);
    }

    // --- match_month() ---

    #[test]
    fn match_month_full_names() {
        assert_eq!(match_month("january"),  Some("JANUARY".to_string()));
        assert_eq!(match_month("december"), Some("DECEMBER".to_string()));
        assert_eq!(match_month("june"),     Some("JUNE".to_string()));
    }

    #[test]
    fn match_month_prefix_at_least_3_chars() {
        assert_eq!(match_month("jan"), Some("JANUARY".to_string()));
        assert_eq!(match_month("feb"), Some("FEBRUARY".to_string()));
        assert_eq!(match_month("dec"), Some("DECEMBER".to_string()));
    }

    #[test]
    fn match_month_rejects_short_prefix() {
        assert_eq!(match_month("ja"), None);
        assert_eq!(match_month("j"),  None);
    }

    #[test]
    fn match_month_case_insensitive() {
        assert_eq!(match_month("JANUARY"), Some("JANUARY".to_string()));
        assert_eq!(match_month("March"),   Some("MARCH".to_string()));
    }

    #[test]
    fn match_month_rejects_non_month() {
        assert_eq!(match_month("hello"),  None);
        assert_eq!(match_month("monday"), None);
    }
}
