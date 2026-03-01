//! Transformer-based Q&A model using Burn 0.20.1.
//!
//! Architecture
//! ────────────
//!   input_ids  [B, S]  (Int)
//!     ↓
//!   Token Embedding Layer [B, S, d_model]
//!     - Converts token IDs to embedding vectors
//!     - Vocabulary size: 8192 tokens
//!     - Embedding dimension: 256 (d_model)
//!     ↓
//!   Positional Embeddings [B, S, d_model]
//!     - Applied internally by Burn's TransformerEncoder
//!     - Encodes absolute token positions in sequence
//!     - Enables attention to distinguish token order
//!     ↓
//!   TransformerEncoder [B, S, d_model]
//!     - 6+ stacked transformer blocks (per specification: n_layers ≥ 6)
//!     - Each block contains:
//!       * Multi-head self-attention (8 heads)
//!       * Feed-forward network (d_ff = 1024)
//!       * Layer normalization + residual connections
//!     - Dropout: 0.1 for regularization
//!     ↓
//!   CLS Token Extraction [B, d_model]
//!     - Extract first token (position 0) representation
//!     - Aggregates document-level information
//!     ↓
//!   Output Projection Heads
//!     - start_head: [B, d_model] → [B, max_seq_len]  (start position logits)
//!     - end_head:   [B, d_model] → [B, max_seq_len]  (end position logits)
//!
//! Each output head predicts position indices in the sequence (1-128),
//! enabling extractive Q&A where answers are spans of the input text.

use burn::{
    config::Config,
    module::Module,
    nn::{
        Embedding, EmbeddingConfig,
        Linear, LinearConfig,
        loss::CrossEntropyLossConfig,
        transformer::{
            TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
        },
    },
    tensor::{backend::Backend, Int, Tensor},
};

// ── Classification output helper ────────────────────────────────────────────

/// Simple output container for classification tasks.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct ClassificationOutput<B: Backend> {
    /// Computed loss scalar value.
    pub loss: Tensor<B, 1>,
    /// Output logits for classification.
    pub output: Tensor<B, 2>,
    /// Target labels.
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> ClassificationOutput<B> {
    /// Create a new classification output.
    #[allow(dead_code)]
    pub fn new(
        loss: Tensor<B, 1>,
        output: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
    ) -> Self {
        Self { loss, output, targets }
    }
}

// ── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the QA transformer — serialisable via `#[derive(Config)]`.
#[derive(Config, Debug)]
pub struct QaTransformerConfig {
    /// Vocabulary size (number of distinct token IDs).
    pub vocab_size: usize,
    /// Embedding + hidden dimension.
    #[config(default = 256)]
    pub d_model: usize,
    /// Number of attention heads.
    #[config(default = 8)]
    pub n_heads: usize,
    /// Feed-forward inner dimension (typically 4 × d_model).
    #[config(default = 1024)]
    pub d_ff: usize,
    /// Number of stacked encoder layers (≥ 6 per assignment spec).
    #[config(default = 6)]
    pub n_layers: usize,
    /// Dropout probability (applied during training).
    #[config(default = 0.1)]
    pub dropout: f64,
    /// Maximum sequence length — also the number of position-prediction classes.
    #[config(default = 128)]
    pub max_seq_len: usize,
}

// ── Model ────────────────────────────────────────────────────────────────────

/// Transformer Q&A model, generic over a Burn `Backend`.
#[derive(Module, Debug)]
pub struct QaTransformer<B: Backend> {
    token_embedding: Embedding<B>,
    encoder: TransformerEncoder<B>,
    /// Predicts the start-token position within the sequence.
    start_head: Linear<B>,
    /// Predicts the end-token position within the sequence.
    end_head: Linear<B>,
    max_seq_len: usize,
}

impl<B: Backend> QaTransformer<B> {
    /// Initialise all sub-modules on `device`.
    ///
    /// # Components initialized:
    /// 1. **Token Embedding**: Vocabulary (vocab_size) → embedding vectors (d_model)
    /// 2. **Positional Embeddings**: Handled internally by TransformerEncoder
    ///    - Adds position information to token embeddings
    ///    - Enables model to understand token sequence order
    /// 3. **TransformerEncoder**: 6+ stacked encoder layers with:
    ///    - Multi-head attention (attention heads: 8)
    ///    - Feed-forward network (inner dimension: d_ff)
    ///    - Layer normalization and residual connections
    ///    - Dropout for regularization
    /// 4. **Output Heads**: Linear projections for extractive QA
    ///    - start_head: predicts answer start position
    ///    - end_head: predicts answer end position
    pub fn new(config: &QaTransformerConfig, device: &B::Device) -> Self {
        // Token embedding table: vocab_size × d_model
        let token_embedding =
            EmbeddingConfig::new(config.vocab_size, config.d_model).init(device);

        // Burn 0.20: TransformerEncoderConfig::new(d_model, d_ff, n_heads, n_layers)
        let encoder = TransformerEncoderConfig::new(
            config.d_model,
            config.d_ff,
            config.n_heads,
            config.n_layers,
        )
        .with_dropout(config.dropout)
        .init(device);

        // Each head maps the CLS vector (d_model) → position logits (max_seq_len)
        let start_head = LinearConfig::new(config.d_model, config.max_seq_len).init(device);
        let end_head   = LinearConfig::new(config.d_model, config.max_seq_len).init(device);

        Self {
            token_embedding,
            encoder,
            start_head,
            end_head,
            max_seq_len: config.max_seq_len,
        }
    }

    /// Forward pass with token and positional embeddings.
    ///
    /// # Arguments
    /// * `input_ids` — `[B, S]` integer tensor of token IDs (0 < token_id < vocab_size)
    ///
    /// # Process
    /// 1. Token embedding: Convert token IDs to embedding vectors
    /// 2. Positional embedding: Added internally by TransformerEncoder
    ///    - Burn automatically adds positional information to track token positions
    /// 3. Transformer encoding: Apply 6+ transformer layers
    /// 4. CLS extraction: Extract first token representation (document-level summary)
    /// 5. Head projection: Project CLS to answer span positions
    ///
    /// # Returns
    /// `(start_logits [B, max_seq_len], end_logits [B, max_seq_len])`
    /// - start_logits: logits for answer start position (128 classes)
    /// - end_logits: logits for answer end position (128 classes)
    #[allow(dead_code)]
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // Step 1: Token embedding — convert token IDs to vectors
        // [B, S] → [B, S, d_model]
        let x = self.token_embedding.forward(input_ids);

        // Step 2 & 3: Positional embeddings + Transformer encoding
        // - TransformerEncoder in Burn 0.20.1 automatically handles positional embeddings
        // - Positional information is added to the token embeddings
        // - This enables self-attention to distinguish different token positions
        // - Output: [B, S, d_model]
        let encoder_input = TransformerEncoderInput::new(x);
        let encoded = self.encoder.forward(encoder_input);

        // Step 4: Extract CLS token representation (first token only)
        // - CLS token aggregates information from all positions
        // - Acts as the document-level summary representation
        // [B, S, d_model] → [B, d_model]
        let [batch_size, _seq, d_model] = encoded.dims();
        let cls = encoded.slice([0..batch_size, 0..1, 0..d_model]).squeeze();

        // Step 5: Project CLS to answer span positions
        // [B, d_model] → [B, max_seq_len] for both start and end
        let start_logits = self.start_head.forward(cls.clone());
        let end_logits   = self.end_head.forward(cls);

        (start_logits, end_logits)
    }

    /// Forward + cross-entropy loss on the start-position labels.
    ///
    /// `start_labels` must be `[B]` with values in `[0, max_seq_len)`.
    #[allow(dead_code)]
    pub fn forward_classification(
        &self,
        input_ids: Tensor<B, 2, Int>,
        start_labels: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let (start_logits, _end_logits) = self.forward(input_ids);

        // CrossEntropyLoss expects logits [B, C] and targets [B]
        let loss = CrossEntropyLossConfig::new()
            .init(&start_logits.device())
            .forward(start_logits.clone(), start_labels.clone());

        ClassificationOutput::new(loss, start_logits, start_labels)
    }

    /// Validation step: forward pass for validation batches.
    #[allow(dead_code)]
    pub fn validation_step(&self, batch: crate::data::batcher::QaBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.input_ids, batch.start_labels)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::tensor::{Int, Tensor, TensorData};

    type TestBackend = NdArray;

    fn device() -> NdArrayDevice { NdArrayDevice::Cpu }

    /// Build a small config suitable for fast unit tests.
    fn small_config() -> QaTransformerConfig {
        QaTransformerConfig::new(64)   // vocab_size = 64
            .with_d_model(32)
            .with_n_heads(2)
            .with_d_ff(64)
            .with_n_layers(2)          // fewer layers for speed; prod uses ≥ 6
            .with_dropout(0.0)         // no dropout in eval/test mode
            .with_max_seq_len(16)
    }

    // --- Config ---

    #[test]
    fn config_default_values() {
        let cfg = QaTransformerConfig::new(1000);
        assert_eq!(cfg.vocab_size, 1000);
        assert_eq!(cfg.d_model,    256);
        assert_eq!(cfg.n_heads,    8);
        assert_eq!(cfg.d_ff,       1024);
        assert_eq!(cfg.n_layers,   6);
        assert_eq!(cfg.max_seq_len, 128);
    }

    #[test]
    fn config_with_builder_overrides() {
        let cfg = QaTransformerConfig::new(500)
            .with_d_model(64)
            .with_n_layers(6);
        assert_eq!(cfg.d_model,  64);
        assert_eq!(cfg.n_layers, 6);
        assert_eq!(cfg.vocab_size, 500);
    }

    // --- Model initialisation ---

    #[test]
    fn model_initialises_without_panic() {
        let cfg = small_config();
        let _model: QaTransformer<TestBackend> = QaTransformer::new(&cfg, &device());
    }

    // --- Forward pass shape ---

    #[test]
    fn forward_output_shape_is_correct() {
        let cfg = small_config();
        let model: QaTransformer<TestBackend> = QaTransformer::new(&cfg, &device());

        let batch_size = 2;
        let seq_len    = 8;

        // Create dummy token IDs: values in [0, vocab_size)
        let ids: Vec<i32> = (0..(batch_size * seq_len) as i32).collect();
        let input = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(ids, [batch_size, seq_len]),
            &device(),
        );

        let (start_logits, end_logits) = model.forward(input);

        // Both heads should output [batch, max_seq_len]
        assert_eq!(start_logits.dims(), [batch_size, cfg.max_seq_len]);
        assert_eq!(end_logits.dims(),   [batch_size, cfg.max_seq_len]);
    }

    #[test]
    fn forward_batch_size_1_works() {
        let cfg = small_config();
        let model: QaTransformer<TestBackend> = QaTransformer::new(&cfg, &device());

        let ids: Vec<i32> = vec![0, 1, 2, 3, 4];
        let input = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(ids, [1, 5]),
            &device(),
        );
        let (start, end) = model.forward(input);
        assert_eq!(start.dims(), [1, cfg.max_seq_len]);
        assert_eq!(end.dims(),   [1, cfg.max_seq_len]);
    }

    // --- forward_classification ---

    #[test]
    fn forward_classification_returns_finite_loss() {
        let cfg = small_config();
        let model: QaTransformer<TestBackend> = QaTransformer::new(&cfg, &device());

        let ids: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let input = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(ids, [2, 4]),
            &device(),
        );
        // Labels must be in [0, max_seq_len) = [0, 16)
        let labels = Tensor::<TestBackend, 1, Int>::from_data(
            TensorData::new(vec![0i32, 1i32], [2]),
            &device(),
        );

        let output = model.forward_classification(input, labels);

        // Loss should be a finite scalar
        let loss_val: Vec<f32> = output.loss.to_data().to_vec().unwrap();
        assert_eq!(loss_val.len(), 1);
        assert!(loss_val[0].is_finite(), "Loss should be finite, got {}", loss_val[0]);
        assert!(loss_val[0] > 0.0, "Loss on random weights should be positive");
    }

    #[test]
    fn forward_classification_logit_shape() {
        let cfg = small_config();
        let model: QaTransformer<TestBackend> = QaTransformer::new(&cfg, &device());

        let ids: Vec<i32> = (0..12i32).collect();
        let input = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(ids, [3, 4]),
            &device(),
        );
        let labels = Tensor::<TestBackend, 1, Int>::from_data(
            TensorData::new(vec![0i32, 1i32, 2i32], [3]),
            &device(),
        );

        let output = model.forward_classification(input, labels);
        // output.output is start_logits: [batch=3, max_seq_len=16]
        assert_eq!(output.output.dims(), [3, cfg.max_seq_len]);
    }
}
