//! CLI entry point for the Word-Document Q&A system.
//!
//! Commands
//! ─────────
//!   load               Parse all .docx files and print every event entry.
//!   ask  <question>    Answer a natural-language question about the calendars.
//!   train              Train the transformer model on the WGPU backend.
//!   model              Print the model architecture (CPU / NdArray backend).

mod data;
mod model;
mod train;
mod inference;

use crate::data::loader::load_all_calendars;
use crate::inference::infer;
use crate::train::{train, TrainConfig};
use std::env;

const DATA_DIR: &str = "data";

fn print_usage() {
    println!(
        r#"
Word-Document Q&A System  (Burn 0.20.1)
========================================
Usage: cargo run -- <command> [args]

Commands:
  load               Load and print all calendar entries from ./data/
  ask  <question>    Answer a question about the calendars
  train              Train the transformer model (requires WGPU / GPU)
  model              Print the model architecture summary (CPU only)

Examples:
  cargo run -- ask "What date is the 2026 End of Year Graduation Ceremony?"
  cargo run -- ask "How many times did the HDC hold their meetings in 2024?"
  cargo run -- ask "When is the Higher Degrees Committee meeting in February 2025?"
  cargo run -- load
  cargo run -- train
"#
    );
}

fn cmd_load() {
    match load_all_calendars(DATA_DIR) {
        Ok(entries) => {
            println!("Loaded {} entries.\n", entries.len());
            for e in &entries {
                let day_str = if let Some(end) = e.end_day {
                    format!("{}-{}", e.day.map(|d| d.to_string()).unwrap_or_else(|| "?".to_string()), end)
                } else {
                    e.day.map(|d| d.to_string()).unwrap_or_else(|| "?".to_string())
                };
                println!("[{} {} {}] {}", e.month, day_str, e.year, e.text);
            }
        }
        Err(err) => eprintln!("Error loading calendars: {}", err),
    }
}

fn cmd_ask(question: &str) {
    println!("Question: {}\n", question);
    let answer = infer(DATA_DIR, question);
    println!("Answer:\n{}", answer);
}

fn cmd_model() {
    // Use the CPU-only NdArray backend so this works without a GPU.
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    use crate::model::transformer::{QaTransformer, QaTransformerConfig};

    let device = NdArrayDevice::Cpu;
    let config = QaTransformerConfig::new(8_192);
    let _model: QaTransformer<NdArray> = QaTransformer::new(&config, &device);

    println!("QaTransformer Architecture");
    println!("══════════════════════════");
    println!("  vocab_size    : {}", config.vocab_size);
    println!("  d_model       : {}", config.d_model);
    println!("  n_heads       : {}", config.n_heads);
    println!("  d_ff          : {}", config.d_ff);
    println!("  n_layers      : {}", config.n_layers);
    println!("  dropout       : {}", config.dropout);
    println!("  max_seq_len   : {}", config.max_seq_len);

    // Approximate parameter count
    let d = config.d_model;
    let embed_params = config.vocab_size * d;
    let attn_params  = 4 * d * d;   // Q, K, V, O projections
    let ff_params    = 2 * d * config.d_ff;
    let ln_params    = 4 * d;        // 2 layer norms × (weight + bias)
    let head_params  = 2 * d * config.max_seq_len; // start + end heads
    let total = embed_params
        + config.n_layers * (attn_params + ff_params + ln_params)
        + head_params;
    println!("\n  Estimated parameters: {:>10}", total);
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "load"  => cmd_load(),
        "ask"   => {
            if args.len() < 3 {
                eprintln!("Provide a question, e.g.:");
                eprintln!("  cargo run -- ask \"When is graduation 2026?\"");
                return;
            }
            cmd_ask(&args[2..].join(" "));
        }
        "train" => train(&TrainConfig::default()),
        "model" => cmd_model(),
        other   => {
            eprintln!("Unknown command: '{}'", other);
            print_usage();
        }
    }
}
