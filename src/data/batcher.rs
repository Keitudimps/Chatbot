use burn::{
    data::dataloader::batcher::Batcher,
    tensor::{backend::Backend, Int, Tensor, TensorData},
};
use serde::{Deserialize, Serialize};

/// A single training item: token IDs for context and the answer span.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaItem {
    /// Token IDs of the (question + separator + context) sequence.
    pub input_ids: Vec<i32>,
    /// Index of the first answer token within input_ids.
    pub start_label: i32,
    /// Index of the last answer token within input_ids (inclusive).
    pub end_label: i32,
}

/// A padded batch of QA items ready for the model.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct QaBatch<B: Backend> {
    /// [batch, seq_len] — token IDs, zero-padded.
    pub input_ids: Tensor<B, 2, Int>,
    /// [batch] — start-position labels.
    pub start_labels: Tensor<B, 1, Int>,
    /// [batch] — end-position labels.
    pub end_labels: Tensor<B, 1, Int>,
}

/// Collates a list of `QaItem`s into a `QaBatch` by padding to the longest sequence.
#[allow(dead_code)]
#[derive(Clone, Default)]
pub struct QaBatcher {
    pub max_len: usize,
}

impl QaBatcher {
    /// Create a new batcher with max sequence length.
    #[allow(dead_code)]
    pub fn new(max_len: usize) -> Self {
        Self { max_len }
    }
}

impl<B: Backend> Batcher<B, QaItem, QaBatch<B>> for QaBatcher {
    fn batch(&self, items: Vec<QaItem>, device: &B::Device) -> QaBatch<B> {
        let batch_size = items.len();

        // Find the actual max length in this batch (capped by self.max_len)
        let seq_len = items
            .iter()
            .map(|it| it.input_ids.len())
            .max()
            .unwrap_or(1)
            .min(self.max_len);

        // Build flat padded token buffer
        let mut ids_flat: Vec<i32> = vec![0i32; batch_size * seq_len];
        let mut starts: Vec<i32> = Vec::with_capacity(batch_size);
        let mut ends: Vec<i32> = Vec::with_capacity(batch_size);

        for (b, item) in items.iter().enumerate() {
            let len = item.input_ids.len().min(seq_len);
            for (i, &tok) in item.input_ids[..len].iter().enumerate() {
                ids_flat[b * seq_len + i] = tok;
            }
            starts.push(item.start_label);
            ends.push(item.end_label);
        }

        let input_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(ids_flat, [batch_size, seq_len]),
            device,
        );
        let start_labels = Tensor::<B, 1, Int>::from_data(
            TensorData::new(starts, [batch_size]),
            device,
        );
        let end_labels = Tensor::<B, 1, Int>::from_data(
            TensorData::new(ends, [batch_size]),
            device,
        );

        QaBatch { input_ids, start_labels, end_labels }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;

    type TestBackend = NdArray;

    fn device() -> NdArrayDevice { NdArrayDevice::Cpu }

    fn make_item(ids: Vec<i32>, start: i32, end: i32) -> QaItem {
        QaItem { input_ids: ids, start_label: start, end_label: end }
    }

    #[test]
    fn batch_produces_correct_shape() {
        let batcher = QaBatcher::new(16);
        let items = vec![
            make_item(vec![1, 2, 3], 0, 2),
            make_item(vec![4, 5, 6, 7], 1, 3),
        ];
        let batch: QaBatch<TestBackend> = batcher.batch(items, &device());

        // [2, 4] — padded to longest sequence
        assert_eq!(batch.input_ids.dims(), [2, 4]);
        assert_eq!(batch.start_labels.dims(), [2]);
        assert_eq!(batch.end_labels.dims(), [2]);
    }

    #[test]
    fn batch_pads_shorter_sequences_with_zeros() {
        let batcher = QaBatcher::new(8);
        let items = vec![
            make_item(vec![1, 2], 0, 1),        // length 2
            make_item(vec![3, 4, 5, 6], 0, 3),  // length 4
        ];
        let batch: QaBatch<TestBackend> = batcher.batch(items, &device());

        // Shape should be [2, 4]
        assert_eq!(batch.input_ids.dims(), [2, 4]);
    }

    #[test]
    fn batch_respects_max_len_cap() {
        let batcher = QaBatcher::new(3); // cap at 3
        let items = vec![
            make_item(vec![1, 2, 3, 4, 5, 6], 0, 5), // 6 tokens, should be capped
        ];
        let batch: QaBatch<TestBackend> = batcher.batch(items, &device());

        // seq_len capped at max_len = 3
        assert_eq!(batch.input_ids.dims()[1], 3);
    }

    #[test]
    fn batch_single_item_works() {
        let batcher = QaBatcher::new(16);
        let items = vec![make_item(vec![10, 20, 30], 0, 2)];
        let batch: QaBatch<TestBackend> = batcher.batch(items, &device());
        assert_eq!(batch.input_ids.dims(), [1, 3]);
        assert_eq!(batch.start_labels.dims(), [1]);
    }

    #[test]
    fn batch_preserves_labels() {
        let batcher = QaBatcher::new(16);
        let items = vec![
            make_item(vec![1, 2, 3], 1, 2),
            make_item(vec![4, 5, 6], 0, 1),
        ];
        let batch: QaBatch<TestBackend> = batcher.batch(items, &device());

        // Convert to vec to check values
        let starts: Vec<i32> = batch.start_labels
            .to_data()
            .to_vec::<i32>()
            .unwrap();
        assert_eq!(starts, vec![1, 0]);

        let ends: Vec<i32> = batch.end_labels
            .to_data()
            .to_vec::<i32>()
            .unwrap();
        assert_eq!(ends, vec![2, 1]);
    }

    #[test]
    fn qa_item_serializes_and_deserializes() {
        let item = make_item(vec![1, 2, 3], 0, 2);
        let json = serde_json::to_string(&item).expect("serialize failed");
        let back: QaItem = serde_json::from_str(&json).expect("deserialize failed");
        assert_eq!(back.input_ids, item.input_ids);
        assert_eq!(back.start_label, item.start_label);
        assert_eq!(back.end_label, item.end_label);
    }
}
