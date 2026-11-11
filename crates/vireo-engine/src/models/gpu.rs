//! What the models' GPU forward passes share.

use std::cell::Cell;

use super::{Error, Precision, Result, SavedState};
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

    /// The matrix rounded to Q8_0 on the way up (as llama.cpp's `quantize_row_q8_0`: a
    /// scale of max|w| / 127 per 32, in f16), so it runs on the 8-bit kernels. For
    /// unquantized weights in 8-bit mode; `None` for other kinds.
    pub fn upload_q8_0(gpu: &Gpu, m: &Matrix) -> Result<Option<Self>> {
        use crate::TensorType;
        if !matches!(m.kind, TensorType::F16 | TensorType::BF16 | TensorType::F32)
            || !m.cols.is_multiple_of(32)
        {
            return Ok(None);
        }
        let row_bytes = m.row_bytes();
        let mut out = Vec::with_capacity(m.rows * m.cols / 32 * 34);
        let mut row = vec![0f32; m.cols];
        for r in 0..m.rows {
            crate::quant::dequantize_row(m.kind, &m.data[r * row_bytes..][..row_bytes], &mut row);
            for block in row.as_chunks::<32>().0 {
                let amax = block.iter().fold(0f32, |a, v| a.max(v.abs()));
                let d = amax / 127.0;
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                out.extend_from_slice(&crate::half::f32_to_f16(d).to_le_bytes());
                out.extend(block.iter().map(|v| (v * id).round() as i8 as u8));
            }
        }
        let q = Matrix {
            kind: TensorType::Q8_0,
            rows: m.rows,
            cols: m.cols,
            data: &out,
        };
        Self::upload(gpu, q).map(Some)
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

/// Which of a batch's tokens need the last layer's output, for their logits. The last
/// layer leaves nothing for later tokens but its keys and values, so the rest of it runs
/// only for the tokens whose logits are wanted (as llama.cpp does).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Outputs {
    /// Every token (scoring a text, and one token at a time).
    All,
    /// The batch's last token, whose output ends up in row 0.
    Last,
    /// None: the last layer only writes its keys and values.
    None,
}

/// The address of row `row` of a buffer with rows of `width` f32s.
pub(super) fn row(buffer: &Buffer, row: usize, width: usize) -> CuDevicePtr {
    buffer.f32_at(row * width)
}

/// A buffer of `capacity` positions (each the same size) to save, and how many of its
/// last positions are worth keeping: `None` for all of them, `Some(w)` for a layer that
/// only ever looks `w` positions back (a sliding window), so older ones are left out.
pub(super) type Positions<'b> = (&'b Buffer, Option<usize>);

/// Which of `len` positions a buffer that keeps `window` of them saves.
fn kept(len: usize, window: Option<usize>) -> std::ops::Range<usize> {
    len.saturating_sub(window.unwrap_or(len))..len
}

/// Copies a session's first `len` positions of each of `positions` (only the last ones a
/// windowed buffer needs) and the whole of each of `whole` into one [`SavedState`].
pub(super) fn save_state(
    positions: &[Positions],
    capacity: usize,
    len: usize,
    whole: &[&Buffer],
) -> Result<SavedState> {
    let total = positions
        .iter()
        .map(|(b, w)| b.len_bytes() / capacity * kept(len, *w).len())
        .chain(whole.iter().map(|b| b.len_bytes()))
        .sum();
    let mut bytes = vec![0u8; total];
    let mut at = 0;
    for (b, w) in positions {
        let per = b.len_bytes() / capacity;
        let range = kept(len, *w);
        let n = per * range.len();
        b.read_bytes_at(per * range.start, &mut bytes[at..at + n])
            .map_err(gpu_error)?;
        at += n;
    }
    for b in whole {
        let n = b.len_bytes();
        b.read_bytes_at(0, &mut bytes[at..at + n])
            .map_err(gpu_error)?;
        at += n;
    }
    Ok(SavedState { len, bytes })
}

/// Puts the first `upto` positions of `saved` back into `positions` (and, when `whole`
/// isn't empty, its whole buffers, which hold the state at `saved.len` and nothing
/// before). Only all of it (`upto` = `saved.len`) when there's a `whole` buffer or a
/// windowed one left out positions: what's missing would be needed before then. An
/// error if `saved` came from another model, precision or layout.
pub(super) fn load_state(
    saved: &SavedState,
    upto: usize,
    positions: &[Positions],
    capacity: usize,
    whole: &[&Buffer],
) -> Result<()> {
    let expected: usize = positions
        .iter()
        .map(|(b, w)| b.len_bytes() / capacity * kept(saved.len, *w).len())
        .chain(whole.iter().map(|b| b.len_bytes()))
        .sum();
    if saved.bytes.len() != expected || upto > saved.len || saved.len > capacity {
        return Err(Error::Invalid(format!(
            "a saved state of {} bytes for {} positions doesn't fit this session",
            saved.bytes.len(),
            saved.len
        )));
    }
    let trimmed = positions.iter().any(|(_, w)| kept(saved.len, *w).start > 0);
    if (trimmed || !whole.is_empty()) && upto != saved.len {
        return Err(Error::Invalid(
            "this saved state can only be loaded whole".into(),
        ));
    }
    let mut at = 0;
    for (b, w) in positions {
        let per = b.len_bytes() / capacity;
        let range = kept(saved.len, *w);
        let n = per
            * range
                .start
                .max(range.end.min(upto))
                .saturating_sub(range.start);
        b.write_bytes_at(per * range.start, &saved.bytes[at..at + n])
            .map_err(gpu_error)?;
        at += per * range.len();
    }
    for b in whole {
        let n = b.len_bytes();
        b.write_bytes_at(0, &saved.bytes[at..at + n])
            .map_err(gpu_error)?;
        at += n;
    }
    Ok(())
}
