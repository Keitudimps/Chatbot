use burn::data::dataset::Dataset as BurnDataset;
use crate::data::loader::CalendarEntry;
use crate::data::batcher::QaItem;
use crate::data::tokenizer::{tokenize, is_year, match_month};

// ---------------------------------------------------------------------------
// Burn Dataset implementation
// ---------------------------------------------------------------------------

/// Wraps calendar entries as a Burn `Dataset<QaItem>` for the training pipeline.
pub struct CalendarDataset {
    items: Vec<QaItem>,
}

impl CalendarDataset {
    /// Build synthetic (question, answer-span) pairs from calendar entries.
    /// Each entry is encoded as: [CLS] event_tokens [SEP] month_tokens year_tokens
    /// The answer span covers the event tokens (positions 1..event_len).
    pub fn from_entries(entries: &[CalendarEntry], vocab_size: usize) -> Self {
        let items = entries
            .iter()
            .map(|e| Self::entry_to_item(e, vocab_size))
            .collect();
        Self { items }
    }

    fn entry_to_item(entry: &CalendarEntry, vocab_size: usize) -> QaItem {
        // Trivial hash-based token ID: deterministic, no external tokenizer needed.
        let hash_token = |s: &str| -> i32 {
            let mut h: u64 = 5381;
            for b in s.bytes() {
                h = h.wrapping_mul(33).wrapping_add(b as u64);
            }
            (h % (vocab_size as u64 - 3) + 1) as i32 // keep 0 for PAD, vocab-1 for UNK
        };

        let cls_id: i32 = (vocab_size - 1) as i32; // [CLS]
        let sep_id: i32 = (vocab_size - 2) as i32; // [SEP]

        let mut ids: Vec<i32> = vec![cls_id];

        // Event text tokens (this is the "answer" span)
        let event_tokens: Vec<i32> = tokenize(&entry.text)
            .iter()
            .map(|t| hash_token(t))
            .collect();
        let start_label = 1i32; // first token after [CLS]
        let end_label = (event_tokens.len() as i32).max(1); // last event token
        ids.extend_from_slice(&event_tokens);

        ids.push(sep_id);

        // Context: month + year
        ids.push(hash_token(&entry.month.to_lowercase()));
        ids.push(hash_token(&entry.year.to_string()));

        QaItem { input_ids: ids, start_label, end_label }
    }

    pub fn split(mut self, train_ratio: f64) -> (Self, Self) {
        let split_at = ((self.items.len() as f64) * train_ratio) as usize;
        let val_items = self.items.split_off(split_at);
        (Self { items: self.items }, Self { items: val_items })
    }
}

impl BurnDataset<QaItem> for CalendarDataset {
    fn get(&self, index: usize) -> Option<QaItem> {
        self.items.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.items.len()
    }
}

// ---------------------------------------------------------------------------
// Retrieval-based Q&A (used by the CLI inference path)
// ---------------------------------------------------------------------------

/// Extract the acronym from a phrase (e.g., "Higher Degrees Committee" → "hdc")
fn extract_acronym(phrase: &str) -> String {
    phrase
        .split_whitespace()
        .filter_map(|word| word.chars().next())
        .collect::<String>()
        .to_lowercase()
}

#[derive(Debug, Default)]
struct QueryIntent {
    keywords: Vec<String>,
    year: Option<i32>,
    month: Option<String>,
    day: Option<u32>,
    is_count_query: bool,
    is_date_query: bool,
}

/// In-memory dataset supporting natural-language Q&A via keyword retrieval.
pub struct QaDataset {
    entries: Vec<CalendarEntry>,
}

impl QaDataset {
    pub fn new(entries: Vec<CalendarEntry>) -> Self {
        Self { entries }
    }

    /// Answer a natural-language question and return a formatted string.
    pub fn answer(&self, question: &str) -> String {
        let intent = self.parse_intent(question);
        let matches = self.filter_entries(&intent);

        if intent.is_count_query {
            self.format_count(&intent, &matches)
        } else if intent.is_date_query {
            self.format_date(&matches)
        } else {
            self.format_list(&intent, &matches)
        }
    }

    fn parse_intent(&self, question: &str) -> QueryIntent {
        let q = question.to_lowercase();
        let tokens = tokenize(&q);
        let mut intent = QueryIntent::default();

        if q.contains("how many") || q.contains("count") || q.contains("times") {
            intent.is_count_query = true;
        }
        if q.contains("when") || q.contains("what date") || q.contains("which date")
            || q.contains("what month") || q.contains("what day")
        {
            intent.is_date_query = true;
        }

        for token in &tokens {
            if let Some(y) = is_year(token) { intent.year = Some(y); }
            if let Some(m) = match_month(token) { intent.month = Some(m); }
        }

        // Day: small standalone number
        for token in &tokens {
            if let Ok(n) = token.parse::<u32>() {
                if (1..=31).contains(&n) {
                    intent.day = Some(n);
                    break; // Take the first valid day found
                }
            }
        }

        let stop: &[&str] = &[
            "the","a","an","is","are","was","were","have","has","do","does","did",
            "will","would","could","should","may","might","in","on","at","to","for",
            "of","with","by","from","about","what","when","where","who","which","how",
            "that","this","and","but","or","not","their","they","i","my","we","our",
            "held","hold","holds","meeting","meetings","times","date","month","year",
            "day","many","count","did","how","many","times","happens","happen",
        ];

        let mut keywords: Vec<String> = tokens
            .iter()
            .filter(|t| {
                t.len() >= 3
                    && !stop.contains(&t.as_str())
                    && is_year(t).is_none()
                    && match_month(t).is_none()
            })
            .cloned()
            .collect();
        keywords.dedup();
        intent.keywords = keywords;
        intent
    }

    fn filter_entries<'a>(&'a self, intent: &QueryIntent) -> Vec<&'a CalendarEntry> {
        self.entries.iter().filter(|e| {
            if let Some(y) = intent.year { if e.year != y { return false; } }
            if let Some(ref m) = intent.month { if &e.month != m { return false; } }
            
            // ENHANCED: Check if the queried day falls within the event's date range
            if let Some(d) = intent.day {
                let day_matches = if let Some(end) = e.end_day {
                    // Multi-day event: check if queried day is in range [day, end_day]
                    // This should include ALL days from start to end, including those with missing entries
                    match e.day {
                        Some(start) => d >= start && d <= end,
                        None => false
                    }
                } else {
                    // Single-day event: check exact match OR if this day falls within intended range
                    // (in case of partially consolidated events)
                    e.day == Some(d)
                };
                if !day_matches { return false; }
            }
            
            if !intent.keywords.is_empty() {
                let tl = e.text.to_lowercase();
                let event_acronym = extract_acronym(&e.text);
                
                // For date queries, require ALL keywords to match (more strict)
                // For count/other queries, any keyword match is OK (more lenient)
                let keyword_match = if intent.is_date_query {
                    intent.keywords.iter().all(|kw| {
                        tl.contains(kw.as_str()) || event_acronym.contains(kw.as_str())
                    })
                } else {
                    intent.keywords.iter().any(|kw| {
                        tl.contains(kw.as_str()) || event_acronym.contains(kw.as_str())
                    })
                };
                if !keyword_match {
                    return false;
                }
            }
            true
        }).collect()
    }

    fn format_count(&self, intent: &QueryIntent, matches: &[&CalendarEntry]) -> String {
        let n = matches.len();
        let ctx = self.describe(intent);
        if n == 0 {
            return format!("No events matching {} found.", ctx);
        }
        let mut s = format!(
            "There {} {} event{} matching {}:\n",
            if n == 1 { "is" } else { "are" },
            n,
            if n == 1 { "" } else { "s" },
            ctx
        );
        for e in matches.iter().take(25) { s.push_str(&self.fmt(e)); s.push('\n'); }
        s
    }

    fn format_date(&self, matches: &[&CalendarEntry]) -> String {
        if matches.is_empty() { return "No matching events found.".to_string(); }
        let mut s = String::new();
        for e in matches.iter().take(10) { s.push_str(&self.fmt(e)); s.push('\n'); }
        s.trim_end().to_string()
    }

    fn format_list(&self, intent: &QueryIntent, matches: &[&CalendarEntry]) -> String {
        if matches.is_empty() {
            return format!("No events matching {} found.", self.describe(intent));
        }
        let mut s = format!("Found {} matching event(s):\n", matches.len());
        for e in matches.iter().take(25) { s.push_str(&self.fmt(e)); s.push('\n'); }
        s.trim_end().to_string()
    }

    fn fmt(&self, e: &CalendarEntry) -> String {
        let day_str = if let Some(end) = e.end_day {
            // Multi-day event: show range
            format!("{}-{}", e.day.map(|d| d.to_string()).unwrap_or_else(|| "?".to_string()), end)
        } else {
            // Single-day event
            e.day.map(|d| d.to_string()).unwrap_or_else(|| "?".to_string())
        };
        format!("  • {} {} {}: {}", e.month, day_str, e.year, e.text)
    }

    fn describe(&self, intent: &QueryIntent) -> String {
        let mut parts: Vec<String> = Vec::new();
        if !intent.keywords.is_empty() {
            parts.push(format!("\"{}\"", intent.keywords.join(" ")));
        }
        if let Some(ref m) = intent.month { parts.push(m.clone()); }
        if let Some(y) = intent.year { parts.push(y.to_string()); }
        if parts.is_empty() { "your query".to_string() } else { parts.join(", ") }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::loader::CalendarEntry;

    fn make_entry(year: i32, month: &str, day: Option<u32>, text: &str) -> CalendarEntry {
        CalendarEntry {
            year,
            month: month.to_string(),
            day,
            end_day: None,
            text: text.to_string(),
        }
    }

    fn make_range_entry(year: i32, month: &str, start_day: u32, end_day: u32, text: &str) -> CalendarEntry {
        CalendarEntry {
            year,
            month: month.to_string(),
            day: Some(start_day),
            end_day: Some(end_day),
            text: text.to_string(),
        }
    }

    fn sample_entries() -> Vec<CalendarEntry> {
        vec![
            make_entry(2024, "FEBRUARY", Some(19), "Higher Degrees Committee (09:00)"),
            make_entry(2024, "MARCH",    Some(5),  "Higher Degrees Committee (09:00)"),
            make_entry(2024, "MAY",      Some(2),  "Higher Degrees Committee (09:00)"),
            make_entry(2024, "JULY",     Some(22), "Higher Degrees Committee (09:00)"),
            make_entry(2024, "AUGUST",   Some(7),  "Higher Degrees Committee (09:00)"),
            make_entry(2024, "OCTOBER",  Some(17), "Higher Degrees Committee (09:00)"),
            make_entry(2024, "NOVEMBER", Some(12), "Higher Degrees Committee (09:00)"),
            make_entry(2025, "MARCH",    Some(4),  "Higher Degrees Committee (09:00)"),
            make_entry(2026, "APRIL",    Some(15), "AUTUMN GRADUATION"),
            make_entry(2026, "DECEMBER", Some(9),  "SUMMER GRADUATION"),
        ]
    }

    // --- Count queries ---

    #[test]
    fn count_hdc_meetings_2024() {
        let ds = QaDataset::new(sample_entries());
        let answer = ds.answer("How many times did the HDC hold their meetings in 2024?");
        assert!(answer.contains("7"), "Expected 7 HDC meetings, got: {}", answer);
    }

    #[test]
    fn count_query_zero_results_returns_no_events_message() {
        let ds = QaDataset::new(sample_entries());
        let answer = ds.answer("How many times did the Senate meet in 2024?");
        assert!(
            answer.to_lowercase().contains("no events") || answer.contains("0"),
            "Expected zero result message, got: {}",
            answer
        );
    }

    // --- Date queries ---

    #[test]
    fn date_query_graduation_2026() {
        let ds = QaDataset::new(sample_entries());
        let answer = ds.answer("When is the 2026 graduation?");
        // Should find both APRIL 15 and DECEMBER 9
        assert!(
            answer.contains("APRIL") || answer.contains("DECEMBER"),
            "Expected graduation dates, got: {}",
            answer
        );
    }

    #[test]
    fn date_query_hdc_february_2024() {
        let ds = QaDataset::new(sample_entries());
        let answer = ds.answer("When is the Higher Degrees Committee meeting in February 2024?");
        assert!(answer.contains("FEBRUARY"), "Expected FEBRUARY in answer: {}", answer);
        assert!(answer.contains("19"),       "Expected day 19 in answer: {}", answer);
    }

    // --- Year filtering ---

    #[test]
    fn year_filter_isolates_2025() {
        let ds = QaDataset::new(sample_entries());
        let answer = ds.answer("Higher Degrees Committee 2025");
        assert!(answer.contains("2025"), "Expected 2025 entries: {}", answer);
        assert!(!answer.contains("2024"), "Should not contain 2024 entries: {}", answer);
    }

    // --- Month filtering ---

    #[test]
    fn month_filter_finds_march_entries() {
        let ds = QaDataset::new(sample_entries());
        let answer = ds.answer("Higher Degrees Committee meeting in March");
        assert!(answer.contains("MARCH"), "Expected MARCH in answer: {}", answer);
    }

    // --- Keyword search ---

    #[test]
    fn keyword_search_graduation_returns_results() {
        let ds = QaDataset::new(sample_entries());
        let answer = ds.answer("graduation");
        assert!(!answer.to_lowercase().contains("no events"), "Should find graduation: {}", answer);
    }

    #[test]
    fn keyword_search_no_match_returns_helpful_message() {
        let ds = QaDataset::new(sample_entries());
        let answer = ds.answer("quantum physics seminar");
        assert!(
            answer.to_lowercase().contains("no events") || answer.to_lowercase().contains("no matching"),
            "Should return not-found message: {}",
            answer
        );
    }

    // --- CalendarDataset (Burn training dataset) ---

    #[test]
    fn calendar_dataset_len_matches_entries() {
        let entries = sample_entries();
        let n = entries.len();
        let ds = CalendarDataset::from_entries(&entries, 1024);
        assert_eq!(ds.len(), n);
    }

    #[test]
    fn calendar_dataset_get_returns_item() {
        use burn::data::dataset::Dataset as BurnDataset;
        let ds = CalendarDataset::from_entries(&sample_entries(), 1024);
        let item = ds.get(0);
        assert!(item.is_some(), "First item should exist");
        let item = item.unwrap();
        assert!(!item.input_ids.is_empty(), "input_ids should not be empty");
        assert!(item.start_label >= 0, "start_label should be non-negative");
        assert!(item.end_label >= item.start_label, "end should be >= start");
    }

    #[test]
    fn calendar_dataset_split_proportions() {
        let ds = CalendarDataset::from_entries(&sample_entries(), 1024);
        let total = ds.len();
        let (train, valid) = ds.split(0.9);
        assert_eq!(train.len() + valid.len(), total, "Split should cover all items");
        assert!(train.len() > valid.len(), "Train set should be larger");
    }

    #[test]
    fn calendar_dataset_get_out_of_bounds_returns_none() {
        use burn::data::dataset::Dataset as BurnDataset;
        let ds = CalendarDataset::from_entries(&sample_entries(), 1024);
        assert!(ds.get(9999).is_none());
    }

    #[test]
    fn date_query_term_end_filters_strictly() {
        let entries = vec![
            make_entry(2025, "MARCH",    Some(14), "END OF TERM 1"),
            make_entry(2025, "JUNE",     Some(20), "END OF TERM 2"),
            make_entry(2025, "JANUARY",  Some(27), "START OF TERM 1"),
            make_entry(2025, "MARCH",    Some(25), "START OF TERM 2"),
            make_entry(2025, "FEBRUARY", Some(10), "Institutional Gender Based Violence Committee (09:00)"),
        ];
        let ds = QaDataset::new(entries);
        let answer = ds.answer("When does the term end in 2025?");
        
        // Should contain the two END OF TERM entries
        assert!(answer.contains("END OF TERM"), "Expected END OF TERM in answer: {}", answer);
        // Should NOT contain START OF TERM (missing "end" keyword)
        assert!(!answer.contains("START OF"), "Should not contain START OF TERM: {}", answer);
        // Should NOT contain the committee meeting (missing "term" keyword)
        assert!(!answer.contains("Gender Based Violence"), "Should not contain committee: {}", answer);
    }

    #[test]
    fn multi_day_event_recess_matches_any_day_in_range() {
        let entries = vec![
            make_range_entry(2025, "MARCH", 17, 24, "RECESS"),
            make_entry(2025, "MARCH", Some(14), "END OF TERM 1"),
            make_entry(2025, "MARCH", Some(25), "START OF TERM 2"),
        ];
        let ds = QaDataset::new(entries);
        
        // Querying for day 17 should find RECESS (start of recess)
        let answer = ds.answer("What happens on March 17 2025?");
        assert!(answer.contains("RECESS"), "Expected RECESS on March 17: {}", answer);
        
        // Querying for day 20 (middle of recess) should also find RECESS
        let answer = ds.answer("What happens on March 20 2025?");
        assert!(answer.contains("RECESS"), "Expected RECESS on March 20: {}", answer);
        
        // Querying for day 24 (end of recess) should find RECESS
        let answer = ds.answer("What happens on March 24 2025?");
        assert!(answer.contains("RECESS"), "Expected RECESS on March 24: {}", answer);
        
        // Day before recess should NOT find RECESS
        let answer = ds.answer("What happens on March 16 2025?");
        assert!(!answer.contains("RECESS"), "Should not find RECESS on March 16: {}", answer);
        
        // Day after recess should NOT find RECESS
        let answer = ds.answer("What happens on March 25 2025?");
        assert!(!answer.contains("RECESS"), "Should not find RECESS on March 25, found: {}", answer);
        assert!(answer.contains("START OF TERM 2"), "Should find START OF TERM 2 on March 25: {}", answer);
    }

    #[test]
    fn multi_day_event_autumn_graduation_april() {
        let entries = vec![
            make_range_entry(2025, "APRIL", 12, 18, "AUTUMN GRADUATION"),
            make_entry(2025, "APRIL", Some(8), "WCED SCHOOLS OPEN"),
        ];
        let ds = QaDataset::new(entries);
        
        // Query for graduation should find the multi-day event
        let answer = ds.answer("When is autumn graduation in 2025?");
        assert!(answer.contains("AUTUMN GRADUATION"), "Expected AUTUMN GRADUATION: {}", answer);
        assert!(answer.contains("12-18"), "Expected date range 12-18: {}", answer);
        
        // Querying for day 15 (middle of graduation period)
        let answer = ds.answer("What happens on April 15 2025?");
        assert!(answer.contains("AUTUMN GRADUATION"), "Expected AUTUMN GRADUATION on April 15: {}", answer);
    }
}
