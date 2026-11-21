//! The forward pass for each model architecture.
//!
//! So far: Llama (Llama 3 and 3.x), Qwen 3.5 (to 3.8) and Gemma 4, each on the CPU
//! reference backend and on NVIDIA GPUs; Inkling 2 (and the looped Inkling 2 Mini) on the
//! CPU and on the Mac's GPU.

pub mod gemma4;
pub mod gemma4_cuda;
pub mod gemma4_vision;
mod gpu;
pub mod inkling2;
pub mod inkling2_dspark;
#[cfg(target_os = "macos")]
pub mod inkling2_cuda;
#[cfg(target_os = "macos")]
pub mod inkling2_dspark_cuda;
/// CUDA has no Inkling kernels yet: the Mac model's types, which only say so.
#[cfg(not(target_os = "macos"))]
#[path = "inkling2_unsupported.rs"]
pub mod inkling2_cuda;
#[cfg(not(target_os = "macos"))]
pub mod inkling2_dspark_cuda {
    pub use super::inkling2_cuda::DSparkCuda;
}
pub mod llama;
pub mod llama_cuda;
pub mod qwen35;
pub mod qwen35_cuda;

use std::fmt;

/// Which logits a GPU model's `forward_batch` hands back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Logits {
    /// None: the tokens only go into the context.
    None,
    /// The last token's, to pick the next token from.
    Last,
    /// Every token's, one row of the vocabulary each (for scoring a text).
    All,
}

/// How a GPU model does its matrix products.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Precision {
    /// Activations rounded to 8 bits per block of 32, as llama.cpp does: the matrix
    /// products run 4 at a time (dp4a) or on the tensor cores. Perplexity a little off
    /// the exact one (llama.cpp: 38.42 against 38.32 on Llama 3.2 1B Q4_K_M).
    #[default]
    Int8,
    /// Everything in f32 on the exact weights: the CPU reference's numbers, slower.
    Exact,
}

/// How many tokens a GPU session reads at once, unless asked for another number:
/// enough to keep the GPU busy, little enough memory to keep sessions cheap. 1,024 on both
/// backends: the matrix kernels get more blocks to spread over the GPU. On the M5 Max it
/// read Qwen's and Gemma's prompts 5–9% faster than 512; on the RTX 5060 Laptop (26 SMs)
/// ~2.5% faster on Qwen 3.5 4B and Gemma 4 E4B, whose 2,560-row matrices left the last
/// wave of 160 tiles mostly idle at 512 (Qwen 3.5 9B unchanged).
pub const DEFAULT_BATCH: usize = 1024;

/// A GPU session's contents copied to system memory, to be loaded into a session of the
/// same model and precision later (`export` and `import` on each model): its keys and
/// values for `len` positions and, for Qwen 3.5, its DeltaNet layers there. The bytes can
/// go to a file and come back; they say nothing of the model they belong to.
pub struct SavedState {
    len: usize,
    bytes: Vec<u8>,
}

impl SavedState {
    /// The number of positions saved.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// A state read back from `bytes` (as `bytes` gave them) for `len` positions. Checked
    /// against the session when it's imported.
    pub fn from_bytes(len: usize, bytes: Vec<u8>) -> Self {
        Self { len, bytes }
    }
}

use crate::GgufFile;
use crate::cpu::Matrix;
use crate::quant;

#[derive(Debug)]
pub enum Error {
    /// A tensor or setting the architecture needs isn't in the file.
    Missing(String),
    /// Something this engine can't run yet.
    Unsupported(String),
    /// A tensor has the wrong shape for the model's settings.
    Invalid(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Missing(what) => write!(f, "the model file has no {what}"),
            Error::Unsupported(what) => write!(f, "{what} isn't supported yet"),
            Error::Invalid(what) => write!(f, "the model file doesn't add up: {what}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// A setting under the architecture's prefix, which the model can't do without.
fn required(file: &GgufFile, key: &str) -> Result<usize> {
    file.arch_u64(key)
        .map(|v| v as usize)
        .ok_or_else(|| Error::Missing(format!("{} setting", key)))
}

/// A weight matrix of `rows` × `cols` (GGUF dimensions `[cols, rows]`).
fn matrix<'a>(file: &'a GgufFile, name: &str, cols: usize, rows: usize) -> Result<Matrix<'a>> {
    let t = file
        .tensor(name)
        .ok_or_else(|| Error::Missing(format!("tensor {name}")))?;
    if t.dims != [cols as u64, rows as u64] {
        return Err(Error::Invalid(format!(
            "{name} is {:?}, but the settings make it [{cols}, {rows}]",
            t.dims
        )));
    }
    let kind = t.kind.filter(|&k| quant::supports(k)).ok_or_else(|| {
        Error::Unsupported(format!(
            "{name}'s type (GGUF type {}) on the CPU",
            t.type_id
        ))
    })?;
    let data = file.tensor_data(t).expect("a known type has data");
    Ok(Matrix {
        kind,
        rows,
        cols,
        data,
    })
}

/// `count` weight matrices of `rows` × `cols` in one tensor, one after another (GGUF
/// dimensions `[cols, rows, count]`: a mixture-of-experts layer's experts), as one matrix
/// of `rows × count` rows. Expert `e` is rows `e × rows` on (`Matrix::rows_from`).
fn matrices<'a>(
    file: &'a GgufFile,
    name: &str,
    cols: usize,
    rows: usize,
    count: usize,
) -> Result<Matrix<'a>> {
    let t = file
        .tensor(name)
        .ok_or_else(|| Error::Missing(format!("tensor {name}")))?;
    if t.dims != [cols as u64, rows as u64, count as u64] {
        return Err(Error::Invalid(format!(
            "{name} is {:?}, but the settings make it [{cols}, {rows}, {count}]",
            t.dims
        )));
    }
    let kind = t.kind.filter(|&k| quant::supports(k)).ok_or_else(|| {
        Error::Unsupported(format!(
            "{name}'s type (GGUF type {}) on the CPU",
            t.type_id
        ))
    })?;
    let data = file.tensor_data(t).expect("a known type has data");
    Ok(Matrix {
        kind,
        rows: rows * count,
        cols,
        data,
    })
}

/// A 1-D tensor of `len` numbers (norm weights and the like), decoded to f32.
fn vector(file: &GgufFile, name: &str, len: usize) -> Result<Vec<f32>> {
    let m = matrix_1d(file, name, len)?;
    let mut out = vec![0.0; len];
    m.decode_row(0, &mut out);
    Ok(out)
}

fn matrix_1d<'a>(file: &'a GgufFile, name: &str, len: usize) -> Result<Matrix<'a>> {
    let t = file
        .tensor(name)
        .ok_or_else(|| Error::Missing(format!("tensor {name}")))?;
    if t.dims != [len as u64] {
        return Err(Error::Invalid(format!(
            "{name} is {:?}, but should be [{len}]",
            t.dims
        )));
    }
    let kind = t
        .kind
        .filter(|&k| quant::supports(k))
        .ok_or_else(|| Error::Unsupported(format!("{name}'s type (GGUF type {})", t.type_id)))?;
    Ok(Matrix {
        kind,
        rows: 1,
        cols: len,
        data: file.tensor_data(t).expect("known type"),
    })
}
