//! Library entry point — re-exports all public modules so integration
//! tests in `tests/` can reference them as `word_doc_qa::data::...`.

pub mod data;
pub mod model;
pub mod train;
pub mod inference;
