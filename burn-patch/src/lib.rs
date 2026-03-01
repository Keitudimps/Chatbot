// Thin shim: re-export everything from the real burn crate when features are enabled.
// This patch only exists to add the "test" feature alias.

#[cfg(any(feature = "train", feature = "wgpu", feature = "autodiff", feature = "ndarray", feature = "test"))]
pub use burn_real::*;
