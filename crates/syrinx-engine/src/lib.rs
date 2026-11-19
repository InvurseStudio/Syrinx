//! Vireo's inference engine.
//!
//! Runs GGUF models on Vireo's own kernels: CUDA first, then Vulkan and Metal, with a
//! CPU backend as the fallback and the reference for correctness. See `docs/PLAN.md` at the
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
/// The GPU backend. On macOS it's the Metal one (`src/metal`), with the CUDA backend's
/// types and methods, so everything that runs on the GPU (`models::*_cuda`, the CLI, the
/// server) is the same code on both.
#[cfg(not(target_os = "macos"))]
pub mod cuda;
#[cfg(target_os = "macos")]
#[path = "metal/mod.rs"]
pub mod cuda;
#[cfg(not(target_os = "macos"))]
mod dylib;
pub mod gguf;
pub mod grammar;
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

/// The engine's version, as shown by `syrinx-cli --version`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
