//! Vireo's inference engine.
//!
//! Runs GGUF models on Vireo's own kernels: CUDA first, then Vulkan and Metal, with a
//! CPU backend as the fallback and the reference for correctness. See `PLAN.md` at the
//! root of this folder.
//!
//! Built so far:
//! - [`gguf`]: reads a GGUF file's metadata and tensor table, with the weights
//!   memory-mapped
//! - [`tensor_type`]: GGUF's tensor types and their block layouts
//! - [`tokenizer`]: text to token ids and back, from the vocabulary in a GGUF file
//! - [`quant`] and [`half`]: decoding GGUF's quantized and 16-bit numbers
//! - [`cpu`]: the CPU reference backend
//! - [`models`]: the forward pass, so far for Llama
//!
//! Next, in the order phase 1 needs them:
//! - the CUDA backend, checked against the CPU reference
//! - chat templates

pub mod cpu;
pub mod cuda;
mod dylib;
pub mod gguf;
pub mod half;
mod hash;
mod mmap;
pub mod models;
pub mod quant;
pub mod tensor_type;
pub mod tokenizer;

pub use gguf::{Gguf, GgufFile};
pub use tensor_type::TensorType;
pub use tokenizer::Tokenizer;

/// The engine's version, as shown by `vireo-cli --version`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
