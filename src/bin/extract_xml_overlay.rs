use std::fs::File;
use std::io::Read;
use std::path::Path;
use zip::ZipArchive;
use regex::Regex;
use std::io::Cursor;

fn main() {
    println!("Starting comprehensive text extractor...");
    
    let paths = vec![
        "data/calendar_2024.docx",
        "data/calendar_2025.docx", 
        "data/calendar_2026.docx",
    ];
    
    for path in paths {
        println!("\n==================================================");
        println!("Extracting from: {}", path);
        println!("==================================================");
        if !Path::new(path).exists() {
            println!("  Not found");
            continue;
        }
        
        extract_all_text(path);
    }
}

fn extract_all_text(file_path: &str) {
    let mut file_data = Vec::new();
    let file = match File::open(file_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to open: {}", e);
            return;
        }
    };
    
    match std::io::BufReader::new(file).read_to_end(&mut file_data) {
        Ok(_) => {},
        Err(e) => {
            eprintln!("Failed to read: {}", e);
            return;
        }
    };
    
    let cursor = Cursor::new(file_data);
    let mut archive = match ZipArchive::new(cursor) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Failed to read zip: {}", e);
            return;
        }
    };
    
    let mut content = String::new();
    match archive.by_name("word/document.xml") {
        Ok(mut docxml) => {
            if let Err(e) = docxml.read_to_string(&mut content) {
                eprintln!("Failed to read document.xml: {}", e);
                return;
            }
        }
        Err(e) => {
            eprintln!("Could not read document.xml: {}", e);
            return;
        }
    }
    
    // Extract ALL text from <w:t> tags (regular Word text)
    let text_regex = match Regex::new(r"<w:t[^>]*>([^<]+)</w:t>") {
        Ok(re) => re,
        Err(_) => return,
    };
    
    let mut all_text = String::new();
    for cap in text_regex.captures_iter(&content) {
        let text = cap.get(1).map_or("", |m| m.as_str());
        all_text.push_str(text);
        all_text.push(' ');
    }
    
    println!("\n=== All extracted text chunks ===\n");
    
    // Look for patterns with vacation/graduation keywords
    let keywords = vec!["RECESS", "GRADUATION", "BREAK", "VACATION", "HOLIDAY"];
    
    // Split by words and look for context
    let words: Vec<&str> = all_text.split_whitespace().collect();
    
    for (i, word) in words.iter().enumerate() {
        let upper = word.to_uppercase();
        
        // Check if this word contains any of our keywords
        if keywords.iter().any(|&kw| upper.contains(kw)) {
            // Print surrounding context
            let start = if i > 5 { i - 5 } else { 0 };
            let end = if i + 5 < words.len() { i + 5 } else { words.len() };
            
            println!("Found near index {}:", i);
            println!("  Context: {} >>> {} <<< {}",
                words[start..i].join(" "),
                word,
                words[i+1..end].join(" ")
            );
            println!();
        }
    }
    
    println!("\n=== Looking for date range patterns ===\n");
    
    // Look for patterns like "17-24", "12-18" near RECESS/GRADUATION
    let date_pattern = Regex::new(r"\b(\d{1,2})-(\d{1,2})\b").unwrap();
    
    // Search for date ranges near keywords
    for cap in date_pattern.captures_iter(&all_text) {
        let full_match = cap.get(0).unwrap().as_str();
        let start = cap.get(0).unwrap().start();
        
        // Get context around this match (500 chars before/after)
        let context_start = if start > 500 { start - 500 } else { 0 };
        let context_end = if start + 500 < all_text.len() { start + 500 } else { all_text.len() };
        let context = &all_text[context_start..context_end];
        
        // Check if this context contains vacation keywords
        let context_upper = context.to_uppercase();
        if context_upper.contains("RECESS") || context_upper.contains("GRADUATION") || 
           context_upper.contains("BREAK") || context_upper.contains("VACATION") {
            println!("Date range: {}", full_match);
            println!("  Context snippet: ...{}...", context.chars().take(200).collect::<String>());
            println!();
        }
    }
}

