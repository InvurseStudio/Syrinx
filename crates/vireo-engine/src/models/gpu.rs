//! What the models' GPU forward passes share.

use std::cell::Cell;

use super::{Error, Precision, Result};
use crate::cpu::Matrix;
use crate::cuda::{self, Buffer, CuDevicePtr, Gpu, Graph, Q8};

pub(super) fn gpu_error(e: cuda::Error) -> Error {
    Error::Unsupported(format!("Running on the GPU ({e})"))
}

/// A weight matrix in GPU memory.
pub(super) struct GpuMatrix {
    pub kind: crate::TensorType,
    pub rows: usize,
    pub cols: usize,
    pub data: Buffer,
}

impl GpuMatrix {
    pub fn upload(gpu: &Gpu, m: Matrix) -> Result<Self> {
        if !Gpu::supports(m.kind) {
            return Err(gpu_error(cuda::Error::Unsupported(m.kind)));
        }
        Ok(Self {
            kind: m.kind,
            rows: m.rows,
            cols: m.cols,
            data: gpu
                .upload_weights(m.kind, m.data, m.rows, m.cols)
                .map_err(gpu_error)?,
        })
    }

    /// `y = W x` for each of the input's rows of `x` into the rows of `y`: with 8-bit
    /// activations when the input has room for them and the weights have kernels.
    pub fn mul(&self, input: &Input, y: CuDevicePtr) -> Result<()> {
        let g = input.gpu;
        if let Some(q8) = input.q8
            && Gpu::supports_8bit(self.kind)
            && self.cols.is_multiple_of(32)
        {
            if input.rounded.get() != self.cols {
                g.quantize(input.x, q8, input.tokens, self.cols)
                    .map_err(gpu_error)?;
                input.rounded.set(self.cols);
            }
            return g
                .matmul_8bit(
                    self.kind,
                    self.data.ptr(),
                    q8,
                    y,
                    self.rows,
                    self.cols,
                    input.tokens,
                )
                .map_err(gpu_error);
        }
        g.matmul(
            self.kind,
            self.data.ptr(),
            input.x,
            y,
            self.rows,
            self.cols,
            input.tokens,
        )
        .map_err(gpu_error)
    }
}

/// The input of one or more matrix products: `tokens` rows at `x`. With room for 8-bit
/// activations (`q8`), it's rounded to 8 bits on its first use and the rounded copy
/// serves the products after it, so make a new `Input` whenever `x` changes.
pub(super) struct Input<'a> {
    gpu: &'a Gpu,
    x: CuDevicePtr,
    tokens: usize,
    q8: Option<&'a Q8>,
    /// The row length it was rounded at, or 0.
    rounded: Cell<usize>,
}

impl<'a> Input<'a> {
    pub fn new(gpu: &'a Gpu, x: CuDevicePtr, tokens: usize, q8: Option<&'a Q8>) -> Self {
        Self {
            gpu,
            x,
            tokens,
            q8,
            rounded: Cell::new(0),
        }
    }
}

/// Room for a batch's activations in 8 bits, if the model uses them: `widest` is the
/// longest row any matrix product reads.
pub(super) fn q8_room(
    gpu: &Gpu,
    precision: Precision,
    batch: usize,
    widest: usize,
) -> Result<Option<Q8>> {
    match precision {
        Precision::Exact => Ok(None),
        Precision::Int8 => gpu
            .alloc_q8(batch, widest.next_multiple_of(32))
            .map(Some)
            .map_err(gpu_error),
    }
}

/// Records the launches `queue` makes as a graph, to replay for each token.
pub(super) fn record(gpu: &Gpu, queue: impl FnOnce() -> Result<()>) -> Result<Graph> {
    gpu.start_recording().map_err(gpu_error)?;
    let queued = queue();
    let graph = gpu.finish_recording().map_err(gpu_error);
    queued?;
    graph
}

/// Logits on the GPU: room for one token's, grown when a batch's are asked for.
pub(super) struct LogitsBuffer {
    buffer: Buffer,
    rows: usize,
}

impl LogitsBuffer {
    pub fn new(gpu: &Gpu, vocab: usize) -> Result<Self> {
        Ok(Self {
            buffer: gpu.alloc(vocab * 4).map_err(gpu_error)?,
            rows: 1,
        })
    }

    /// Room for `rows` tokens' logits.
    pub fn rows(&mut self, gpu: &Gpu, rows: usize, vocab: usize) -> Result<&Buffer> {
        if rows > self.rows {
            self.buffer = gpu.alloc(rows * vocab * 4).map_err(gpu_error)?;
            self.rows = rows;
        }
        Ok(&self.buffer)
    }
}

/// The address of row `row` of a buffer with rows of `width` f32s.
pub(super) fn row(buffer: &Buffer, row: usize, width: usize) -> CuDevicePtr {
    buffer.f32_at(row * width)
}
