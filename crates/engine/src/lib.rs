//! engine: GGUF model loading and LLaMA inference on RDNA GPUs.

pub mod gguf;
pub mod hfq;
pub mod llama;
#[cfg(feature = "deltanet")]
pub mod qwen35;
pub mod tokenizer;
