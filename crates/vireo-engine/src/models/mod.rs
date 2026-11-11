//! The forward pass for each model architecture.
//!
//! So far: Llama (Llama 3 and 3.x), on the CPU reference backend and on NVIDIA GPUs.

pub mod llama;
pub mod llama_cuda;

use std::fmt;

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
