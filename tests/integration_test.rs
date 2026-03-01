//! Integration tests — exercise the full pipeline end-to-end.
//!
//! These tests use real `.docx` files from `data/` and the NdArray CPU
//! backend so they run on any machine without a GPU.

mod data {
    pub use word_doc_qa::data::loader::load_all_calendars;
    pub use word_doc_qa::data::dataset::{CalendarDataset, QaDataset};
    pub use word_doc_qa::data::batcher::{QaBatcher, QaItem, QaBatch};
    pub use word_doc_qa::data::tokenizer::tokenize;
}
mod model {
    pub use word_doc_qa::model::transformer::{QaTransformer, QaTransformerConfig};
}

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::tensor::{Int, Tensor, TensorData};

use data::{CalendarDataset, QaBatcher, QaItem, QaBatch, load_all_calendars, tokenize};
use data::QaDataset;
use model::{QaTransformer, QaTransformerConfig};

type B = NdArray;
fn device() -> NdArrayDevice { NdArrayDevice::Cpu }

// ── Data pipeline ─────────────────────────────────────────────────────────────

#[test]
fn full_pipeline_load_to_dataset() {
    let entries = load_all_calendars("data")
        .expect("Should load all calendars");
    assert!(!entries.is_empty());

    let ds = CalendarDataset::from_entries(&entries, 4096);
    assert_eq!(ds.len(), entries.len());

    // Every item should have non-empty input_ids
    for i in 0..ds.len().min(20) {
        let item = ds.get(i).expect("Item should exist");
        assert!(!item.input_ids.is_empty());
    }
}

#[test]
fn full_pipeline_load_to_batch() {
    let entries = load_all_calendars("data").unwrap();
    let ds      = CalendarDataset::from_entries(&entries, 4096);
    let (train, _valid) = ds.split(0.9);

    let batcher = QaBatcher::new(128);
    // Manually grab first 4 items and batch them
    let items: Vec<QaItem> = (0..4)
        .filter_map(|i| train.get(i))
        .collect();

    assert_eq!(items.len(), 4);
    let batch: QaBatch<B> = batcher.batch(items, &device());
    assert_eq!(batch.input_ids.dims()[0], 4);  // batch size
    assert_eq!(batch.start_labels.dims(), [4]);
}

// ── Q&A retrieval ─────────────────────────────────────────────────────────────

#[test]
fn qa_answers_hdc_count_question() {
    let entries = load_all_calendars("data").unwrap();
    let ds = QaDataset::new(entries);
    let answer = ds.answer("How many times did the HDC hold their meetings in 2024?");
    assert!(
        answer.contains("7"),
        "Expected 7 HDC meetings in 2024, got: {}", answer
    );
}

#[test]
fn qa_answers_graduation_date_question() {
    let entries = load_all_calendars("data").unwrap();
    let ds = QaDataset::new(entries);
    let answer = ds.answer("What date is the 2026 End of Year Graduation Ceremony?");
    assert!(
        answer.to_uppercase().contains("DECEMBER") || answer.contains("9"),
        "Expected December graduation date, got: {}", answer
    );
}

#[test]
fn qa_returns_not_found_for_unknown_event() {
    let entries = load_all_calendars("data").unwrap();
    let ds = QaDataset::new(entries);
    let answer = ds.answer("When is the quantum computing hackathon?");
    assert!(
        answer.to_lowercase().contains("no") || answer.to_lowercase().contains("found"),
        "Expected not-found message, got: {}", answer
    );
}

// ── Model forward pass ────────────────────────────────────────────────────────

#[test]
fn model_forward_pass_on_real_vocab_size() {
    let entries = load_all_calendars("data").unwrap();

    // Build vocab from real data
    let mut vocab: std::collections::HashSet<String> = std::collections::HashSet::new();
    for e in &entries {
        for t in tokenize(&e.text) {
            vocab.insert(t);
        }
    }
    let vocab_size = vocab.len().max(64); // at least 64

    // Use a small config to keep the test fast
    let cfg = QaTransformerConfig::new(vocab_size)
        .with_d_model(32)
        .with_n_heads(2)
        .with_d_ff(64)
        .with_n_layers(2)
        .with_max_seq_len(32)
        .with_dropout(0.0);

    let model: QaTransformer<B> = QaTransformer::new(&cfg, &device());

    let ids: Vec<i32> = (0..16i32).map(|x| x % vocab_size as i32).collect();
    let input: Tensor<B, 2, Int> = Tensor::from_data(
        TensorData::new(ids, [2, 8]),
        &device(),
    );
    let (start, end) = model.forward(input);

    assert_eq!(start.dims(), [2, 32]);
    assert_eq!(end.dims(),   [2, 32]);
}

// ── Tokenizer ─────────────────────────────────────────────────────────────────

#[test]
fn tokenizer_handles_real_calendar_text() {
    let text = "Higher Degrees Committee (09:00)\nInstitutional Forum (14:00)";
    let tokens = tokenize(text);
    assert!(tokens.contains(&"higher".to_string()));
    assert!(tokens.contains(&"degrees".to_string()));
    assert!(tokens.contains(&"committee".to_string()));
    assert!(tokens.contains(&"institutional".to_string()));
}
