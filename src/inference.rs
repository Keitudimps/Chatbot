//! Inference engine.
//!
//! Uses the retrieval-based `QaDataset` to answer questions directly from the
//! parsed calendar data. 
//!
//! # Architecture
//! The system employs a **retrieval-augmented question answering** approach:
//! 1. **Intent Parsing**: Analyzes the question to detect query type (count, date, etc.)
//! 2. **Keyword Extraction**: Filters stop words and extracts semantic terms
//! 3. **Document Retrieval**: Filters calendar entries by temporal and semantic constraints
//! 4. **Answer Formatting**: Presents results in a human-readable format
//!
//! # Future Enhancement: Neural Inference
//! The system can be extended with neural inference using trained checkpoints:
//! - Load a trained transformer model checkpoint
//! - Tokenize question + document context
//! - Run forward pass to obtain answer spans (start/end positions)
//! - Extract answer text from predicted positions
//!
//! This hybrid approach (retrieval + optional neural ranking) provides:
//! - **Fast responses**: Retrieval-based baseline requires no GPU
//! - **Scalability**: Can handle large document collections
//! - **Interpretability**: Retrieved documents explain the answer
//! - **Optional neural ranking**: Can rerank results using trained model

use crate::data::dataset::QaDataset;
use crate::data::loader::load_all_calendars;

/// Load calendar data from `data_dir` and answer `question` using retrieval-based Q&A.
///
/// # Process
/// 1. Load calendar entries from `.docx` files in `data_dir`
/// 2. Build in-memory dataset with semantic indexing
/// 3. Parse question intent (count vs. date vs. lookup)
/// 4. Retrieve matching calendar entries
/// 5. Format answer based on query type
///
/// # Arguments
/// * `data_dir` — Directory containing `.docx` calendar files (default: "data")
/// * `question` — Natural language question (e.g., "How many HDC meetings in 2024?")
///
/// # Returns
/// Formatted answer string with matched events or "No matching events found"
///
/// # Error Handling
/// Errors in data loading are caught and formatted as friendly user messages.
/// The function never panics and always returns a valid response string.
///
/// # Example
/// ```ignore
/// let answer = infer("data", "What date is the 2026 graduation ceremony?");
/// println!("{}", answer);
/// // Output: Answer:\nDecember 9, 2026\nGraduation Planning Committee
/// ```
pub fn infer(data_dir: &str, question: &str) -> String {
    let entries = match load_all_calendars(data_dir) {
        Ok(e) => e,
        Err(err) => return format!("Failed to load calendars: {}", err),
    };
    let dataset = QaDataset::new(entries);
    dataset.answer(question)
}
