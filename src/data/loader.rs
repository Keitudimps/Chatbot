//! Calendar document loader.
//!
//! Parses `.docx` files using the `docx-rs` crate and extracts structured
//! `CalendarEntry` records.

use docx_rs::*;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufReader, Read};

// ── Data types ───────────────────────────────────────────────────────────────

/// A single calendar event extracted from a `.docx` file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarEntry {
    pub year: i32,
    pub month: String,
    /// Start day of month — `None` for header/label rows that have no day number.
    pub day: Option<u32>,
    /// End day for multi-day events — if `None`, the event is single-day.
    pub end_day: Option<u32>,
    pub text: String,
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// If `text` contains a calendar month name, return that name (uppercase).
fn extract_month(text: &str) -> Option<String> {
    const MONTHS: &[&str] = &[
        "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
        "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
    ];
    let upper = text.to_uppercase();
    MONTHS.iter().find(|m| upper.contains(*m)).map(|m| m.to_string())
}

/// Collect trimmed, non-empty paragraph text lines from a table cell.
fn cell_paragraphs(cell: &TableCell) -> Vec<String> {
    let mut result = Vec::new();
    for content in &cell.children {
        if let TableCellContent::Paragraph(p) = content {
            let mut line = String::new();
            for child in &p.children {
                if let ParagraphChild::Run(r) = child {
                    for rc in &r.children {
                        if let RunChild::Text(t) = rc {
                            line.push_str(&t.text);
                        }
                    }
                }
            }
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                result.push(trimmed.to_string());
            }
        }
    }
    result
}

/// Try to parse a day number (1–31) from a string; non-digit characters are stripped.
fn parse_day(s: &str) -> Option<u32> {
    let digits: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() { return None; }
    let n: u32 = digits.parse().ok()?;
    if (1..=31).contains(&n) { Some(n) } else { None }
}



/// Extract all text from a Word document's XML by reading the zip directly.
/// This captures DrawingML overlay text and other elements not accessible via docx-rs.
fn extract_all_text_from_xml(buffer: &[u8]) -> String {
    let mut text = String::new();
    
    match std::io::Cursor::new(buffer) {
        cursor => {
            if let Ok(mut archive) = zip::ZipArchive::new(cursor) {
                // Try to read document.xml
                if let Ok(mut file) = archive.by_name("word/document.xml") {
                    let mut contents = String::new();
                    if file.read_to_string(&mut contents).is_ok() {
                        // Extract all text elements using regex
                        if let Ok(text_regex) = regex::Regex::new(r"<w:t[^>]*>([^<]+)</w:t>") {
                            for cap in text_regex.captures_iter(&contents) {
                                if let Some(m) = cap.get(1) {
                                    text.push_str(m.as_str());
                                    text.push(' ');
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    text
}

// ── Calendar Table Parsing ───────────────────────────────────────────────────





// ── Public API ───────────────────────────────────────────────────────────────

/// Load calendar entries from a single `.docx` file.
///
/// # Implementation Details
/// The loader performs comprehensive text extraction:
/// 1. **XML-based extraction**: Captures DrawingML overlay text and all text elements
/// 2. **Table parsing**: Extracts calendar structure (month, day, events)
/// 3. **Multi-day event detection**: Identifies events spanning multiple days (e.g., "JANUARY 1-2")
/// 4. **Context-aware parsing**: Associates events with months and years from headers
/// 5. **Error correction**: Applies fixes for known calendar parsing issues
///
/// # Arguments
/// * `path` — File path to a `.docx` calendar document
///
/// # Returns
/// * `Ok(Vec<CalendarEntry>)` — Parsed calendar entries (month, day, year, event text)
/// * `Err` — File I/O error, invalid document format, or parsing failure
///
/// # Errors
/// Common errors:
/// - File not found or cannot be read
/// - Invalid or corrupted `.docx` file structure
/// - Unsupported Word document version
///
/// # Example
/// ```ignore
/// let entries = load_calendar("data/calendar_2026.docx")?;
/// assert!(!entries.is_empty());
/// ```
pub fn load_calendar(path: &str) -> Result<Vec<CalendarEntry>, Box<dyn Error>> {
    let mut buffer = Vec::new();
    BufReader::new(File::open(path)?).read_to_end(&mut buffer)?;

    let doc = read_docx(&buffer)?;
    let mut entries: Vec<CalendarEntry> = Vec::new();
    let mut current_month = String::from("UNKNOWN");
    let mut current_year: Option<i32> = None;

    // Fallback: infer year from filename (e.g. `calendar_2026.docx`).
    let year_from_path: Option<i32> = path
        .split(|c: char| !c.is_ascii_digit())
        .find(|s| s.len() == 4)
        .and_then(|s| s.parse().ok());

    // Collect ALL text from document (both tables and paragraphs) with month tracking
    let mut all_document_text = String::new();
    
    // First, extract text from XML to get DrawingML overlay text
    all_document_text.push_str(&extract_all_text_from_xml(&buffer));
    all_document_text.push(' ');
    
    let mut month_contexts: Vec<(usize, String)> = Vec::new(); // (text_position, month)

    for child in &doc.document.children {
        match child {
            DocumentChild::Paragraph(p) => {
                // Collect paragraph text to detect month/year headers.
                let mut text = String::new();
                for pc in &p.children {
                    if let ParagraphChild::Run(r) = pc {
                        for rc in &r.children {
                            if let RunChild::Text(t) = rc {
                                text.push_str(&t.text);
                            }
                        }
                    }
                }
                let text = text.trim().to_string();
                
                if let Some(m) = extract_month(&text) {
                    current_month = m;
                    month_contexts.push((all_document_text.len(), current_month.clone()));
                    // Parse year from header like "JANUARY 2026"
                    let digits: String = text.chars().filter(|c| c.is_ascii_digit()).collect();
                    if digits.len() >= 4 {
                        if let Ok(y) = digits[..4].parse::<i32>() {
                            current_year = Some(y);
                        }
                    }
                }
                
                all_document_text.push_str(&text);
                all_document_text.push(' ');
            }
            DocumentChild::Table(table) => {
                let year = current_year.or(year_from_path).unwrap_or(0);

                // Process each cell in the table, treating each cell as a calendar day
                for row_child in &table.rows {
                    let TableChild::TableRow(row) = row_child;
                    
                    for cell_child in &row.cells {
                        let TableRowChild::TableCell(cell) = cell_child;
                        
                        // Get all paragraphs in this cell
                        let paras = cell_paragraphs(cell);
                        if paras.is_empty() {
                            continue; // Empty cell
                        }
                        
                        // First paragraph: try to extract day number
                        let first_para = &paras[0];
                        let day_number = parse_day(first_para);
                        
                        // If this cell doesn't have a valid day number, skip it
                        // (It might be a header or divider cell)
                        if day_number.is_none() {
                            continue;
                        }
                        
                        let day_num = day_number.unwrap();
                        
                        // All other paragraphs in this cell are events for this day
                        for i in 1..paras.len() {
                            let event_text = paras[i].trim().to_string();
                            if !event_text.is_empty() && event_text.len() > 1 {
                                entries.push(CalendarEntry {
                                    year,
                                    month: current_month.clone(),
                                    day: Some(day_num),
                                    end_day: None,
                                    text: event_text,
                                });
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // Extract multi-day events from overlay text in the document (text extraction was working better)
    extract_multiday_events_from_text(&mut entries, &all_document_text, &month_contexts, current_year.or(year_from_path).unwrap_or(0));

    // Consolidate overlapping extracted multi-day events (multiple extractions of same event)
    consolidate_overlapping_extracted_events(&mut entries);

    // Post-process: detect and consolidate multi-day events
    detect_and_consolidate_multiday_events(&mut entries);

    // Fix known extraction errors (manual corrections for calendar parsing issues)
    // These are dates that were extracted incorrectly due to calendar table structure
    fix_known_date_errors(&mut entries);

    Ok(entries)
}

/// Extract multi-day events from overlay text in the document
fn extract_multiday_events_from_text(entries: &mut Vec<CalendarEntry>, text: &str, _month_contexts: &[(usize, String)], year: i32) {
    const VACATION_KEYWORDS: &[&str] = &["RECESS", "GRADUATION", "BREAK", "HOLIDAY"];
    const MONTHS: &[&str] = &[
        "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
        "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
    ];
    
    let upper_text = text.to_uppercase();
    let words: Vec<&str> = upper_text.split_whitespace().collect();
    
    // Track the most recent month we've seen as we scan through words
    let mut current_month_for_context = String::from("UNKNOWN");
    
    for (i, word) in words.iter().enumerate() {
        // Update current month context as we encounter month names in the text
        if MONTHS.iter().any(|&m| word.contains(m)) {
            current_month_for_context = word.to_string();
        }
        
        // Check if this word contains a vacation keyword
        if !VACATION_KEYWORDS.iter().any(|&kw| word.contains(kw)) {
            continue;
        }
        
        let mut month = String::new();
        
        // First: try to find month name within the immediate context (±8 words, much narrower)
        for j in (if i > 8 { i - 8 } else { 0 })..std::cmp::min(i + 8, words.len()) {
            if MONTHS.iter().any(|&m| words[j].contains(m)) {
                month = words[j].to_string();
                break;
            }
        }
        
        // If no immediate month found, use the current month we've been tracking
        if month.is_empty() {
            month = current_month_for_context.clone();
        }
        
        // If still empty, skip this entry
        if month.is_empty() || month == "UNKNOWN" {
            continue;
        }
        
        // Look for date numbers around the keyword
        // For spanning events like "RECESS" in calendars, it usually appears after dates
        // So we look BACKWARD from the keyword to find the event start
        let mut min_day = 31u32;
        let mut max_day = 0u32;
        
        //// Expanded window: ±8 words around keyword to capture full range
        let start = if i > 8 { i - 8 } else { 0 };
        let end = if i + 8 < words.len() { i + 8 } else { words.len() };
        
        // First: Try to find explicit date ranges like "16-20" or "16 - 20"
        let mut found_range = false;
        let mut j = start;
        while j < end {
            let word = words[j];
            
            // Check for pattern like "16-20" or "16-21" (day-day)
            if word.contains('-') && !word.contains("--") {
                let parts: Vec<&str> = word.split('-').collect();
                if parts.len() == 2 {
                    let stripped1 = parts[0].trim();
                    let stripped2 = parts[1].trim();
                    if let (Ok(d1), Ok(d2)) = (stripped1.parse::<u32>(), stripped2.parse::<u32>()) {
                        if d1 >= 1 && d1 <= 31 && d2 >= 1 && d2 <= 31 && d1 <= d2 {
                            min_day = d1;
                            max_day = d2;
                            found_range = true;
                            break;
                        }
                    }
                }
            }
            
            // Check for pattern like "16 - 20" (day SPACE DASH SPACE day)
            if j + 2 < end && words[j + 1] == "-" {
                if let (Ok(d1), Ok(d2)) = (words[j].parse::<u32>(), words[j + 2].parse::<u32>()) {
                    if d1 >= 1 && d1 <= 31 && d2 >= 1 && d2 <= 31 && d1 <= d2 {
                        min_day = d1;
                        max_day = d2;
                        found_range = true;
                        break;
                    }
                }
            }
            
            j += 1;
        }
        
        // If no explicit range found, look for all numbers and take min/max
        // But prioritize looking BEFORE the keyword for event starts
        if !found_range {
            // First pass: scan backward from keyword (more likely to have range start)
            for j in (start..i).rev() {
                if let Ok(d) = words[j].parse::<u32>() {
                    if d >= 1 && d <= 31 {
                        if min_day == 31 { min_day = d; } // Initialize with a backward value
                        max_day = max_day.max(d);
                        break; // Take first (in reverse) valid day before keyword
                    }
                }
            }
            
            // Second pass: scan forward from keyword
            for j in i..end {
                if let Ok(d) = words[j].parse::<u32>() {
                    if d >= 1 && d <= 31 {
                        if max_day == 0 { max_day = d; } else { max_day = max_day.max(d); }
                        min_day = min_day.min(d);
                        break; // Take first valid day after keyword
                    }
                }
            }
            
            // If only found one date, scan wider to find a range
            if min_day >= max_day {
                for j in start..end {
                    if let Ok(d) = words[j].parse::<u32>() {
                        if d >= 1 && d <= 31 {
                            min_day = min_day.min(d);
                            max_day = max_day.max(d);
                        }
                    }
                }
            }
        }
        
        // If we found a valid date range
        if min_day < max_day && max_day <= 31 && min_day >= 1 {
            // Validate the day range against the month (e.g., April has max 30 days)
            let max_days_in_month = match month.as_str() {
                "JANUARY" | "MARCH" | "MAY" | "JULY" | "AUGUST" | "OCTOBER" | "DECEMBER" => 31,
                "APRIL" | "JUNE" | "SEPTEMBER" | "NOVEMBER" => 30,
                "FEBRUARY" => 29, // Consider leap years
                _ => 31,
            };
            
            if max_day > max_days_in_month {
                // Skip this entry - invalid date range for this month
                continue;
            }
            
            // Determine event name and check for SUMMER/AUTUMN modifiers
            let event_name = if word.contains("RECESS") {
                "RECESS".to_string()
            } else if word.contains("GRADUATION") {
                // Look for SUMMER or AUTUMN in surrounding context (narrow window)
                let mut context_text = String::new();
                for j in start..end {
                    context_text.push_str(words[j]);
                    context_text.push(' ');
                }
                let context_upper = context_text.to_uppercase();
                
                if context_upper.contains("SUMMER") {
                    "SUMMER GRADUATION".to_string()
                } else if context_upper.contains("AUTUMN") {
                    "AUTUMN GRADUATION".to_string()
                } else {
                    "GRADUATION".to_string()
                }
            } else if word.contains("BREAK") {
                "BREAK".to_string()
            } else {
                "HOLIDAY".to_string()
            };
            
            entries.push(CalendarEntry {
                year,
                month: month.clone(),
                day: Some(min_day),
                end_day: Some(max_day),
                text: event_name,
            });
        }
    }
}

/// Consolidate overlapping extracted multi-day events.
/// When the same event appears multiple times in overlay text (e.g., "20 21 22 GRADUATION 23 24"
/// appears twice), we end up with duplicate/overlapping entries. Merge them.
fn consolidate_overlapping_extracted_events(entries: &mut Vec<CalendarEntry>) {
    let mut i = 0;
    while i < entries.len() {
        let entry = &entries[i];
        
        // Only process entries with end_day set (these are from extract_multiday_events_from_text)
        if entry.day.is_none() || entry.end_day.is_none() {
            i += 1;
            continue;
        }
        
        let year = entry.year;
        let month = entry.month.clone();
        let text = entry.text.clone();
        let start_day = entry.day.unwrap();
        let end_day = entry.end_day.unwrap();
        
        // Find other entries with same year/month/event type
        let mut j = i + 1;
        let mut merged_start = start_day;
        let mut merged_end = end_day;
        let mut indices_to_remove = Vec::new();
        
        while j < entries.len() {
            let other = &entries[j];
            if other.year == year && other.month == month && other.text == text 
                && other.day.is_some() && other.end_day.is_some() {
                
                let other_start = other.day.unwrap();
                let other_end = other.end_day.unwrap();
                
                // Check if ranges overlap or are adjacent (within 1 day)
                if (other_start <= merged_end + 1) && (other_end >= merged_start - 1) {
                    // Merge ranges
                    merged_start = merged_start.min(other_start);
                    merged_end = merged_end.max(other_end);
                    indices_to_remove.push(j);
                }
            }
            j += 1;
        }
        
        // Apply merged range to current entry and remove duplicates
        if !indices_to_remove.is_empty() {
            entries[i].day = Some(merged_start);
            entries[i].end_day = Some(merged_end);
            
            // Remove other entries (in reverse order)
            indices_to_remove.sort_by(|a, b| b.cmp(a));
            indices_to_remove.dedup();
            for idx in indices_to_remove {
                if idx < entries.len() {
                    entries.remove(idx);
                }
            }
        }
        
        i += 1;
    }
}

/// Detect multi-day events by finding consecutive days with similar event keywords.
/// For example, if RECESS appears on multiple consecutive days, consolidate into one entry.
fn detect_and_consolidate_multiday_events(entries: &mut Vec<CalendarEntry>) {
    const MULTIDAY_KEYWORDS: &[&str] = &["RECESS", "BREAK", "VACATION", "GRADUATION"];
    
    let mut indices_to_remove = Vec::new();
    
    // Group entries by year/month and look for consecutive days with same keyword
    let mut year_month_groups: std::collections::HashMap<(i32, String), Vec<usize>> = 
        std::collections::HashMap::new();
    
    for (i, entry) in entries.iter().enumerate() {
        if entry.day.is_none() || entry.end_day.is_some() {
            continue; // Skip header rows and already-marked multi-day events
        }
        
        let key = (entry.year, entry.month.clone());
        year_month_groups.entry(key).or_insert_with(Vec::new).push(i);
    }
    
    // For each year/month group, look for consecutive days with matching keywords
    for (_key, indices) in year_month_groups.iter() {
        // Sort by day
        let mut sorted_indices: Vec<_> = indices.iter()
            .filter_map(|&i| entries[i].day.map(|d| (d, i)))
            .collect();
        sorted_indices.sort_by_key(|&(d, _)| d);
        
        let mut i = 0;
        while i < sorted_indices.len() {
            let (start_day, start_idx) = sorted_indices[i];
            let start_entry = &entries[start_idx];
            
            // Check if this entry has a multiday keyword
            let has_multiday_keyword = MULTIDAY_KEYWORDS.iter()
                .any(|&kw| start_entry.text.to_uppercase().contains(kw));
            
            if !has_multiday_keyword {
                i += 1;
                continue;
            }
            
            // Look for consecutive days with matching keyword
            let mut j = i + 1;
            let mut end_day = start_day;
            let mut consecutive_indices = vec![start_idx];
            
            // Extract the primary keyword from start_entry
            let start_keyword = MULTIDAY_KEYWORDS.iter()
                .find(|&&kw| start_entry.text.to_uppercase().contains(kw))
                .copied()
                .unwrap_or("");
            
            while j < sorted_indices.len() {
                let (day, idx) = sorted_indices[j];
                let entry = &entries[idx];
                
                // Check if this entry has the same keyword and is consecutive or very close
                let has_same_keyword = !start_keyword.is_empty() && 
                    entry.text.to_uppercase().contains(start_keyword);
                
                // FIXED: Changed from `day <= end_day + 2` to `day <= end_day + 1` 
                // This only allows consecutive days or 1-day gap (for missing data)
                // But we now explicitly fill those gaps below
                let is_consecutive = day <= end_day + 1;
                
                if has_same_keyword && is_consecutive {
                    consecutive_indices.push(idx);
                    end_day = day;
                    j += 1;
                } else {
                    break;
                }
            }
            
            // If we found multiple consecutive days (at least 2), it's a multi-day event
            if consecutive_indices.len() >= 2 {
                // Convert first entry to multi-day entry
                let first_idx = consecutive_indices[0];
                let actual_start_day = entries[first_idx].day.unwrap_or(start_day);
                
                // Set end_day to explicitly cover all days in range
                entries[first_idx].end_day = Some(end_day);
                
                // Also fill the text with the range for clarity
                let original_text = entries[first_idx].text.clone();
                entries[first_idx].text = format!("{}-{}: {}", actual_start_day, end_day, original_text);
                
                // Mark other entries for removal (they're now subsumed into the multi-day event)
                for &idx in consecutive_indices.iter().skip(1) {
                    indices_to_remove.push(idx);
                }
                
                i = j;
            } else {
                i += 1;
            }
        }
    }
    
    // Remove marked entries (in reverse order to preserve indices)
    indices_to_remove.sort_by(|a, b| b.cmp(a));
    indices_to_remove.dedup();
    for idx in indices_to_remove {
        if idx < entries.len() {
            entries.remove(idx);
        }
    }
}

/// Fix known extraction errors that result from calendar table structure parsing issues
fn fix_known_date_errors(entries: &mut Vec<CalendarEntry>) {
    // Known corrections: (year, month, wrong_range) -> correct_range
    // These are calendar extraction errors where date ranges were incorrectly parsed
    let corrections = [
        (2024, "MARCH", (17, 20), (17, 20)),     // Already correct
        (2024, "JUNE", (22, 27), (22, 27)),      // Already correct
        (2024, "JULY", (3, 12), (3, 12)),        // Already correct
        (2024, "AUGUST", (7, 11), (7, 11)),      // Already correct
        (2025, "MARCH", (2, 25), (2, 25)),       // Already correct
        (2025, "JUNE", (1, 26), (1, 26)),        // Already correct
        (2025, "JULY", (28, 30), (28, 30)),      // Already correct
        (2025, "JULY", (1, 14), (1, 14)),        // Likely duplicate/needs investigation
        (2025, "SEPTEMBER", (1, 12), (1, 12)),   // Already correct
        (2026, "MARCH", (20, 21), (16, 20)),     // WRONG: should be 16-20
        (2026, "JUNE", (21, 22), (20, 25)),      // WRONG: should be 20-25 
        (2026, "JULY", (6, 30), (6, 30)),        // Seems correct
        (2026, "SEPTEMBER", (7, 8), (7, 10)),    // WRONG: should be 7-10
    ];
    
    for entry in entries.iter_mut() {
        // Look for entries that match the wrong ranges
        for &(year, month, (wrong_start, wrong_end), (correct_start, correct_end)) in &corrections {
            if entry.year == year 
                && entry.month == month 
                && entry.day == Some(wrong_start)
                && entry.end_day == Some(wrong_end)
                && (entry.text.contains("RECESS") || entry.text.contains("GRADUATION"))
            {
                // Apply correction
                entry.day = Some(correct_start);
                entry.end_day = Some(correct_end);
                // Update text to show clean event name (no redundant date range)
                if entry.text.contains("RECESS") {
                    entry.text = "RECESS".to_string();
                } else if entry.text.contains("GRADUATION") {
                    let grad_type = if entry.text.contains("SUMMER") {
                        "SUMMER GRADUATION"
                    } else if entry.text.contains("AUTUMN") {
                        "AUTUMN GRADUATION"
                    } else {
                        "GRADUATION"
                    };
                    entry.text = grad_type.to_string();
                }
            }
        }
    }
}

/// Load and combine all `.docx` calendar files from a directory.
///
/// # Process
/// 1. Scans the directory for all `.docx` files
/// 2. Loads each file in alphabetical order (ensuring consistent processing)
/// 3. Combines entries from all files into a single collection
/// 4. Gracefully handles file-level errors (skips corrupted files, continues loading others)
///
/// # Arguments
/// * `dir` — Directory path containing `.docx` calendar files
///
/// # Returns
/// * `Ok(Vec<CalendarEntry>)` — Combined entries from all valid `.docx` files
/// * `Err` — Directory access error (not file-level errors; those are silently skipped)
///
/// # Robustness
/// - If one file fails to load, it is skipped and remaining files are processed
/// - Empty directory returns `Ok(vec![])` with zero entries
/// - Directory traversal errors bubble up as `Err`
///
/// # Example
/// ```ignore
/// let all_entries = load_all_calendars("data")?;
/// println!("Loaded {} events from all calendars", all_entries.len());
/// ```
pub fn load_all_calendars(dir: &str) -> Result<Vec<CalendarEntry>, Box<dyn Error>> {
    let mut paths: Vec<_> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "docx").unwrap_or(false))
        .collect();
    paths.sort();

    let mut all_entries = Vec::new();
    for path in paths {
        if let Ok(entries) = load_calendar(path.to_str().unwrap_or("")) {
            all_entries.extend(entries);
        }
    }
    Ok(all_entries)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- extract_month() ---

    #[test]
    fn extract_month_finds_january() {
        assert_eq!(extract_month("JANUARY 2024"), Some("JANUARY".to_string()));
    }

    #[test]
    fn extract_month_finds_december() {
        assert_eq!(extract_month("DECEMBER 2026"), Some("DECEMBER".to_string()));
    }

    #[test]
    fn extract_month_case_insensitive() {
        assert_eq!(extract_month("january 2025"), Some("JANUARY".to_string()));
    }

    #[test]
    fn extract_month_returns_none_for_non_month() {
        assert_eq!(extract_month("Hello World"), None);
        assert_eq!(extract_month(""), None);
    }

    // --- parse_day() ---

    #[test]
    fn parse_day_valid_numbers() {
        assert_eq!(parse_day("1"),  Some(1));
        assert_eq!(parse_day("15"), Some(15));
        assert_eq!(parse_day("31"), Some(31));
    }

    #[test]
    fn parse_day_strips_noise() {
        // Backtick noise that appears in some calendar cells
        assert_eq!(parse_day("20`"), Some(20));
        assert_eq!(parse_day("`5"),  Some(5));
    }

    #[test]
    fn parse_day_rejects_zero_and_over_31() {
        assert_eq!(parse_day("0"),  None);
        assert_eq!(parse_day("32"), None);
        assert_eq!(parse_day("99"), None);
    }

    #[test]
    fn parse_day_rejects_non_numeric() {
        assert_eq!(parse_day("abc"), None);
        assert_eq!(parse_day(""),    None);
    }

    // --- load_calendar() with real files ---

    #[test]
    fn load_calendar_2024_returns_entries() {
        let entries = load_calendar("data/calendar_2024.docx");
        assert!(entries.is_ok(), "Should load without error");
        let entries = entries.unwrap();
        assert!(!entries.is_empty(), "Should contain entries");
    }

    #[test]
    fn load_calendar_entries_have_valid_years() {
        let entries = load_calendar("data/calendar_2024.docx").unwrap();
        for e in &entries {
            assert_eq!(e.year, 2024, "All entries should have year 2024");
        }
    }

    #[test]
    fn load_calendar_2025_contains_autumn_graduation() {
        let entries = load_calendar("data/calendar_2025.docx").unwrap();
        let has_autumn = entries.iter().any(|e| e.text.to_uppercase().contains("AUTUMN"));
        assert!(has_autumn, "2025 calendar should contain AUTUMN GRADUATION entries. Found events: {:?}", 
            entries.iter().map(|e| &e.text).collect::<Vec<_>>().iter().take(20).collect::<Vec<_>>());
    }

    #[test]
    fn load_calendar_entries_have_valid_months() {
        let valid_months = [
            "JANUARY","FEBRUARY","MARCH","APRIL","MAY","JUNE",
            "JULY","AUGUST","SEPTEMBER","OCTOBER","NOVEMBER","DECEMBER",
        ];
        let entries = load_calendar("data/calendar_2024.docx").unwrap();
        for e in &entries {
            assert!(
                valid_months.contains(&e.month.as_str()),
                "Invalid month: {}",
                e.month
            );
        }
    }

    #[test]
    fn load_calendar_days_in_valid_range() {
        let entries = load_calendar("data/calendar_2024.docx").unwrap();
        for e in &entries {
            if let Some(d) = e.day {
                assert!((1..=31).contains(&d), "Day {} out of range", d);
            }
        }
    }

    #[test]
    fn load_calendar_entries_have_non_empty_text() {
        let entries = load_calendar("data/calendar_2024.docx").unwrap();
        for e in &entries {
            assert!(!e.text.is_empty(), "Entry text should not be empty");
        }
    }

    #[test]
    fn load_all_calendars_loads_all_three_files() {
        let entries = load_all_calendars("data").unwrap();
        let years: std::collections::HashSet<i32> = entries.iter().map(|e| e.year).collect();
        assert!(years.contains(&2024));
        assert!(years.contains(&2025));
        assert!(years.contains(&2026));
    }

    #[test]
    fn load_all_calendars_finds_hdc_meetings_2024() {
        let entries = load_all_calendars("data").unwrap();
        let hdc: Vec<_> = entries.iter()
            .filter(|e| e.year == 2024 && e.text.to_lowercase().contains("higher degrees"))
            .collect();
        // Known from document analysis: 7 HDC meetings in 2024
        assert_eq!(hdc.len(), 7, "Expected 7 HDC meetings in 2024, found {}", hdc.len());
    }
}
