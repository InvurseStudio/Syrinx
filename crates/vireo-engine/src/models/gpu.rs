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
        // BF16 where the GPU has no kernel for it (CUDA; Gemma 4's Q6_K files keep
        // per_layer_model_proj so): as f32, which holds every BF16 number exactly.
        if m.kind == crate::TensorType::BF16 && !Gpu::supports(m.kind) {
            let mut wide = Vec::with_capacity(m.data.len() * 2);
            for pair in m.data.as_chunks::<2>().0 {
                wide.extend_from_slice(&((u16::from_le_bytes(*pair) as u32) << 16).to_le_bytes());
            }
            let as_f32 = Matrix {
                kind: crate::TensorType::F32,
                rows: m.rows,
                cols: m.cols,
                data: &wide,
            };
            return Self::upload(gpu, as_f32);
        }
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

    /// An f32 matrix as f16 on the way up: half the bytes for a token's pass to read, the
    /// weights off by f16's rounding. `None` for other kinds.
    pub fn upload_f16(gpu: &Gpu, m: &Matrix) -> Result<Option<Self>> {
        use crate::TensorType;
        if m.kind != TensorType::F32 {
            return Ok(None);
        }
        let out: Vec<u8> = m
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|b| crate::half::f32_to_f16(f32::from_le_bytes(*b)).to_le_bytes())
            .collect();
        let half = Matrix {
            kind: TensorType::F16,
            rows: m.rows,
            cols: m.cols,
            data: &out,
        };
        Self::upload(gpu, half).map(Some)
    }

    /// `y = W x` for each of the input's rows of `x` into the rows of `y`: with 8-bit
    /// activations when the input has room for them and the weights have kernels.
    pub fn mul(&self, input: &Input, y: CuDevicePtr) -> Result<()> {
        let g = input.gpu;
        // A few tokens (a draft's guesses): the weights read once, as for one token.
        #[cfg(target_os = "macos")]
        if input.tokens > 1
            && !(input.q8.is_some()
                && Gpu::supports_8bit(self.kind)
                && Gpu::tile_beats_few(self.kind, self.rows, self.cols, input.tokens))
            && g.matvec_multi(
                self.kind,
                self.data.ptr(),
                input.x,
                y,
                self.rows,
                self.cols,
                input.tokens,
            )
            .map_err(gpu_error)?
        {
            return Ok(());
        }
        if let Some(q8) = input.q8
            && Gpu::supports_8bit(self.kind)
            && self.cols.is_multiple_of(32)
        {
            input.round(q8, self.cols, &[self.kind])?;
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

impl GpuMatrix {
    /// `y = W (act(gate) × up)` over `tokens` rows of this matrix's columns: a
    /// feed-forward's last product, with its activation (SiLU, or tanh-GELU with `gelu`).
    /// With 8-bit activations on CUDA the activation and the rounding are one pass
    /// (`Gpu::gated_quantize`) and `gate` is left as it was; otherwise `gate` holds the
    /// activation's result, as after `swiglu` or `geglu`.
    #[allow(clippy::too_many_arguments)]
    pub fn mul_gated(
        &self,
        gpu: &Gpu,
        gate: CuDevicePtr,
        up: CuDevicePtr,
        tokens: usize,
        q8: Option<&Q8>,
        gelu: bool,
        y: CuDevicePtr,
    ) -> Result<()> {
        let n = self.cols;
        #[cfg(not(target_os = "macos"))]
        if let Some(q8) = q8
            && Gpu::supports_8bit(self.kind)
            && n.is_multiple_of(32)
        {
            lay_out(q8, &[self.kind]);
            gpu.gated_quantize(gate, up, n, q8, tokens, gelu)
                .map_err(gpu_error)?;
            return gpu
                .matmul_8bit(self.kind, self.data.ptr(), q8, y, self.rows, n, tokens)
                .map_err(gpu_error);
        }
        if gelu {
            gpu.geglu(gate, up, n, n, tokens).map_err(gpu_error)?;
        } else {
            gpu.swiglu(gate, up, n, n, tokens).map_err(gpu_error)?;
        }
        self.mul(&Input::new(gpu, gate, tokens, q8), y)
    }
}

impl GpuMatrix {
    /// A whole feed-forward for a batch, `y = self (act(gate x) × (up x))`, with the gate
    /// and up products and the activation in one kernel (`Gpu::glu_8bit`): neither
    /// product is written, and `mid` takes their rounded result for this matrix to read.
    /// False, having done nothing, where that kernel doesn't apply (the Mac, exact
    /// activations, a short batch, matrices that differ in type or shape): the caller
    /// then runs the two products and [`GpuMatrix::mul_gated`].
    #[allow(clippy::too_many_arguments)]
    pub fn mul_ffn(
        &self,
        gate: &GpuMatrix,
        up: &GpuMatrix,
        input: &Input,
        mid: Option<&Q8>,
        gelu: bool,
        y: CuDevicePtr,
    ) -> Result<bool> {
        #[cfg(not(target_os = "macos"))]
        if let (Some(q8), Some(mid)) = (input.q8, mid)
            && (gate.kind, gate.rows, gate.cols) == (up.kind, up.rows, up.cols)
            && gate.rows == self.cols
            && Gpu::supports_8bit(gate.kind)
            && Gpu::supports_8bit(self.kind)
            && gate.cols.is_multiple_of(32)
            && self.cols.is_multiple_of(32)
        {
            let g = input.gpu;
            // The middle is laid out as the input: for all three matrices.
            input.round(q8, gate.cols, &[gate.kind, self.kind])?;
            lay_out(mid, &[gate.kind, self.kind]);
            if g.glu_8bit(
                gate.kind,
                gate.data.ptr(),
                up.data.ptr(),
                q8,
                mid,
                gate.rows,
                gate.cols,
                input.tokens,
                gelu,
            )
            .map_err(gpu_error)?
            {
                g.matmul_8bit(
                    self.kind,
                    self.data.ptr(),
                    mid,
                    y,
                    self.rows,
                    self.cols,
                    input.tokens,
                )
                .map_err(gpu_error)?;
                return Ok(true);
            }
        }
        let _ = (gate, up, input, mid, gelu, y);
        Ok(false)
    }
}

#[cfg(target_os = "macos")]
impl GpuMatrix {
    /// One token's `y = act(self x) × (up x)` (SiLU, or tanh-GELU with `gelu`) in one
    /// dispatch (see [`Gpu::matvec_glu`]): false, having done nothing, when the two
    /// matrices differ in type or shape or the type has no such kernel, so the caller runs
    /// the products and the activation instead.
    pub fn mul_glu(
        &self,
        gpu: &Gpu,
        up: &GpuMatrix,
        x: CuDevicePtr,
        y: CuDevicePtr,
        gelu: bool,
    ) -> Result<bool> {
        if (up.kind, up.rows, up.cols) != (self.kind, self.rows, self.cols) {
            return Ok(false);
        }
        gpu.matvec_glu(
            self.kind,
            self.data.ptr(),
            up.data.ptr(),
            x,
            y,
            self.rows,
            self.cols,
            gelu,
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
    /// Whether it was rounded with a batch's heads (CUDA: `Q8::lay_out_for`).
    heads: Cell<bool>,
}

/// Lays a batch's 8-bit buffer out for products with weights of these kinds (CUDA: with
/// heads if any of them reads nothing else, or the model's batches always are). True for
/// heads.
fn lay_out(q8: &Q8, kinds: &[crate::TensorType]) -> bool {
    #[cfg(not(target_os = "macos"))]
    {
        // The last call decides, so the kind that needs heads goes last.
        let needs = kinds.iter().copied().find(|k| Gpu::wants_heads(*k));
        q8.lay_out_for(needs.unwrap_or(kinds[0]))
    }
    #[cfg(target_os = "macos")]
    {
        let _ = (q8, kinds);
        false
    }
}

impl Input<'_> {
    /// Rounds the input to 8 bits at rows of `cols` for products with weights of `kinds`,
    /// unless it is already rounded so.
    fn round(&self, q8: &Q8, cols: usize, kinds: &[crate::TensorType]) -> Result<()> {
        let heads = lay_out(q8, kinds);
        if self.rounded.get() != cols || self.heads.get() != heads {
            self.gpu
                .quantize(self.x, q8, self.tokens, cols)
                .map_err(gpu_error)?;
            self.rounded.set(cols);
            self.heads.set(heads);
        }
        Ok(())
    }
}

impl<'a> Input<'a> {
    pub fn new(gpu: &'a Gpu, x: CuDevicePtr, tokens: usize, q8: Option<&'a Q8>) -> Self {
        Self {
            gpu,
            x,
            tokens,
            q8,
            rounded: Cell::new(0),
            heads: Cell::new(false),
        }
    }
}

/// Room for a batch's activations in 8 bits, if the model uses them: `widest` is the
/// longest row any matrix product reads. `like` is one of the model's large matrices: a
/// model of K-quants keeps a batch's scales and sums as heads (`Q8::with_heads`).
pub(super) fn q8_room(
    gpu: &Gpu,
    precision: Precision,
    batch: usize,
    widest: usize,
    like: &GpuMatrix,
) -> Result<Option<Q8>> {
    match precision {
        Precision::Exact => Ok(None),
        Precision::Int8 => {
            let q8 = gpu
                .alloc_q8(batch, widest.next_multiple_of(32))
                .map_err(gpu_error)?;
            #[cfg(not(target_os = "macos"))]
            let q8 = if Gpu::wants_heads(like.kind) {
                q8.with_heads()
            } else {
                q8
            };
            let _ = like;
            Ok(Some(q8))
        }
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

/// The longest batch replayed from a graph recorded for its length (CUDA): a draft's
/// guesses, a tool's result, a new turn. Queuing a batch's kernels one by one costs the CPU
/// 2-5 ms that the GPU waits out (Gemma 4 E4B, one token: 16.0 ms queued, 10.8 replayed);
/// recording a length's graph costs 4-6 ms, once a session.
#[cfg(not(target_os = "macos"))]
pub(super) const GRAPH_TOKENS: usize = 64;

/// Where a batch of `t` tokens' graph is kept, by `mode` (0 to 2: which outputs it makes).
#[cfg(not(target_os = "macos"))]
pub(super) fn graph_slot(t: usize, mode: usize) -> usize {
    (t - 1) * 3 + mode
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

/// A buffer of positions (each the same size) to save; how many of its last positions are
/// worth keeping: `None` for all of them, `Some(w)` for a layer that only ever looks `w`
/// positions back (a sliding window), so older ones are left out; and its ring: 0 for a
/// buffer of `capacity` positions, each at its own row, or the number of rows of one that
/// keeps only the last ones, position p at row p mod ring.
pub(super) type Positions<'b> = (&'b Buffer, Option<usize>, usize);

/// Bytes a position takes in a buffer.
fn per_position(b: &Buffer, capacity: usize, ring: usize) -> usize {
    b.len_bytes() / if ring > 0 { ring } else { capacity.max(1) }
}

/// Positions `range` of a buffer as runs of rows: (first row, first position, count). One
/// run, or two where a ring wraps.
fn runs(range: std::ops::Range<usize>, ring: usize) -> Vec<(usize, usize, usize)> {
    if ring == 0 || range.is_empty() {
        return vec![(range.start, range.start, range.len())];
    }
    let mut out = Vec::new();
    let mut p = range.start;
    while p < range.end {
        let row = p % ring;
        let n = (ring - row).min(range.end - p);
        out.push((row, p, n));
        p += n;
    }
    out
}

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
        .map(|(b, w, r)| per_position(b, capacity, *r) * kept(len, *w).len())
        .chain(whole.iter().map(|b| b.len_bytes()))
        .sum();
    let mut bytes = vec![0u8; total];
    let mut at = 0;
    for (b, w, r) in positions {
        let per = per_position(b, capacity, *r);
        for (row, _, count) in runs(kept(len, *w), *r) {
            let n = per * count;
            b.read_bytes_at(per * row, &mut bytes[at..at + n])
                .map_err(gpu_error)?;
            at += n;
        }
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
        .map(|(b, w, r)| per_position(b, capacity, *r) * kept(saved.len, *w).len())
        .chain(whole.iter().map(|b| b.len_bytes()))
        .sum();
    if saved.bytes.len() != expected || upto > saved.len || saved.len > capacity {
        return Err(Error::Invalid(format!(
            "a saved state of {} bytes for {} positions doesn't fit this session",
            saved.bytes.len(),
            saved.len
        )));
    }
    let trimmed = positions
        .iter()
        .any(|(_, w, _)| kept(saved.len, *w).start > 0);
    if (trimmed || !whole.is_empty()) && upto != saved.len {
        return Err(Error::Invalid(
            "this saved state can only be loaded whole".into(),
        ));
    }
    let mut at = 0;
    for (b, w, r) in positions {
        let per = per_position(b, capacity, *r);
        let range = kept(saved.len, *w);
        let end = range.start.max(range.end.min(upto));
        let mut from = at;
        for (row, _, count) in runs(range.start..end, *r) {
            let n = per * count;
            b.write_bytes_at(per * row, &saved.bytes[from..from + n])
                .map_err(gpu_error)?;
            from += n;
        }
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
