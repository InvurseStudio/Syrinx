//! Inkling 2 (and the looped Inkling 2 Mini) on the GPU: the CPU reference's forward pass
//! (`inkling2.rs`, which says what each step is) on the Metal backend, for a batch of
//! tokens at once (a prompt) or one (generating).
//!
//! Inkling's own steps have kernels of their own (`ink_*` in syrinx.metal: the mix over
//! the blocks, the loop's controllers, the clips, the short convolutions, the attention
//! with its position bias and sink, the experts' pick and SiTU); the products, norms and
//! the experts' products and sum are the shared kernels, a batch's experts grouped by
//! expert. Mac only so far (the CUDA build has no `ink_` kernels).
//!
//! A batch's rows: every per-token buffer holds the batch's tokens one after another, and
//! the block states a block of `tokens` rows each (state s of token t at row s × tokens + t).
//! Sliding-window layer visits keep their keys and values in a ring of `sliding_window` +
//! the batch's rows (so a batch never overwrites a position one of its tokens still sees);
//! global ones a row a position. Each visit has its convolutions' history.

use super::gpu::{GpuMatrix, Input, gpu_error, q8_room};
use super::inkling2::{Config, Controller, Inkling2, Layer};
use super::{Error, Logits, Precision, Result};
use crate::GgufFile;
use crate::cpu::{Matrix, Pool};
use crate::cuda::{Buffer, CuDevicePtr, Gpu, InkAttention, Q8};

/// The most block states a mix reads (`ink_mix`'s INK_MAX_STATES).
const MAX_STATES: usize = 16;

/// The most tokens a session reads at once.
pub const BATCH: usize = 512;

/// From this many tokens a batch's experts are run grouped by expert (as Gemma 4's).
const MOE_GROUPED_FROM: usize = 80;

/// How far back a session can go (a checkpoint restored): sliding-window rings keep this
/// many positions past their window and a batch, so a chat can drop its last reply (its
/// thinking left out of the next prompt) and go on from where its history ended.
pub const REWIND: usize = 2048;

/// The most tokens whose logits a pass keeps on the GPU at once (`Logits::All`, held).
const HELD: usize = 64;

struct GpuController {
    down: GpuMatrix,
    gain: GpuMatrix,
    forget: GpuMatrix,
}

struct GpuLayer {
    /// The attention input's mix queries (none on layer 0), and how many.
    attn_res_query: Option<(Buffer, usize)>,
    ffn_res_query: Buffer,
    norms: Vec<Buffer>,
    q: GpuMatrix,
    k: GpuMatrix,
    v: GpuMatrix,
    o: GpuMatrix,
    gate: GpuMatrix,
    rel: GpuMatrix,
    sink: GpuMatrix,
    q_clip: Buffer,
    rel_clip: Buffer,
    rel_bias: Buffer,
    k_conv: Buffer,
    v_conv: Buffer,
    attn_post_conv: Buffer,
    ffn_post_conv: Buffer,
    k_conv_proj: Option<GpuMatrix>,
    v_conv_proj: Option<GpuMatrix>,
    ffn_norm: Buffer,
    router: GpuMatrix,
    router_bias: Buffer,
    gate_up: GpuMatrix,
    down: GpuMatrix,
    /// A looped core layer's message queries (4 rows: q, k, v, experts) and controllers.
    looped: Option<(Buffer, [GpuController; 4])>,
}

pub struct Inkling2Cuda<'a> {
    pub config: Config,
    gpu: Gpu,
    precision: Precision,
    /// The token embedding stays in the file: a token's row is decoded on the CPU.
    embedding: Matrix<'a>,
    embedding_norm: Buffer,
    output: GpuMatrix,
    output_norm: Buffer,
    output_res_query: Buffer,
    layers: Vec<GpuLayer>,
}

/// One layer visit's state.
struct Slot {
    keys: Buffer,
    values: Buffer,
    /// The cache's rows, and whether it's a ring (a sliding-window layer).
    rows: usize,
    ring: bool,
    /// Its convolutions' history, in the session's `histories`.
    history: CuDevicePtr,
}

/// A session's convolution histories at a position (`save`), to go back to
/// (`restore`): what can't be cut back, as Qwen's DeltaNet state can't. The caches are
/// left as they are; a ring holds what a restore needs while it's no more than
/// [`REWIND`] positions back.
pub struct Inkling2Checkpoint {
    len: usize,
    history: Buffer,
}

impl Inkling2Checkpoint {
    /// The position it was saved at.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Working space for a batch.
struct Scratch {
    embedded: Buffer,
    x: Buffer,
    /// The block states, a block of rows each, with room after the last for x.
    stack: Buffer,
    /// The previous pass's blocks and output (the loop's messages read them).
    bank: Buffer,
    prelude: Buffer,
    inputs: Buffer,
    normed: Buffer,
    pair: Buffer,
    low: Buffer,
    gain: Buffer,
    forget: Buffer,
    messages: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    r: Buffer,
    sink: Buffer,
    gate: Buffer,
    k_taps: Buffer,
    v_taps: Buffer,
    attention: Buffer,
    attention_parts: Buffer,
    attn_out: Buffer,
    ffn_in: Buffer,
    ffn_normed: Buffer,
    router: Buffer,
    idx: Buffer,
    share: Buffer,
    gate_up: Buffer,
    expert_out: Buffer,
    order: Buffer,
    place: Buffer,
    tiles: Buffer,
    /// The slots' inputs in their experts' order in f16, for the neural accelerators.
    sorted: Option<Buffer>,
    ffn_out: Buffer,
    /// The output's input and its norm: the last token's, or up to HELD tokens'.
    out: Buffer,
    out_normed: Buffer,
    /// Up to HELD rows of logits, `padded_vocab` apart.
    logits: Buffer,
    q8: Option<Q8>,
}

pub struct Inkling2CudaSession {
    slots: Vec<Slot>,
    /// Every slot's convolution history, one after another.
    histories: Buffer,
    scratch: Scratch,
    batch: usize,
    len: usize,
    capacity: usize,
    /// The furthest any position was read to since the session was emptied: what the
    /// rings hold from.
    furthest: usize,
    logits: Vec<f32>,
    rows: Vec<f32>,
}

impl Inkling2CudaSession {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The GPU memory a position costs in the caches that grow with the chat (the global
    /// layers'; the rings are a fixed size).
    pub fn cache_bytes_per_position(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| !s.ring)
            .map(|s| s.keys.len_bytes() * 2 / self.capacity.max(1))
            .sum()
    }
}

fn f32s(gpu: &Gpu, data: &[f32]) -> Result<Buffer> {
    gpu.upload_f32(data).map_err(gpu_error)
}

fn matrix(gpu: &Gpu, m: Matrix) -> Result<GpuMatrix> {
    GpuMatrix::upload(gpu, m)
}

fn controller(gpu: &Gpu, c: &Controller) -> Result<GpuController> {
    Ok(GpuController {
        down: matrix(gpu, c.down)?,
        gain: matrix(gpu, c.gain)?,
        forget: matrix(gpu, c.forget)?,
    })
}

fn layer(gpu: &Gpu, l: &Layer) -> Result<GpuLayer> {
    Ok(GpuLayer {
        attn_res_query: if l.attn_res_query.is_empty() {
            None
        } else {
            Some((
                f32s(gpu, &l.attn_res_query.concat())?,
                l.attn_res_query.len(),
            ))
        },
        ffn_res_query: f32s(gpu, &l.ffn_res_query)?,
        norms: l
            .norms
            .iter()
            .map(|n| f32s(gpu, n))
            .collect::<Result<_>>()?,
        q: matrix(gpu, l.q)?,
        k: matrix(gpu, l.k)?,
        v: matrix(gpu, l.v)?,
        o: matrix(gpu, l.o)?,
        gate: matrix(gpu, l.gate)?,
        rel: matrix(gpu, l.rel)?,
        sink: matrix(gpu, l.sink)?,
        q_clip: f32s(gpu, &l.q_clip)?,
        rel_clip: f32s(gpu, &l.rel_clip)?,
        rel_bias: f32s(gpu, &l.rel_bias)?,
        k_conv: f32s(gpu, &l.k_conv)?,
        v_conv: f32s(gpu, &l.v_conv)?,
        attn_post_conv: f32s(gpu, &l.attn_post_conv)?,
        ffn_post_conv: f32s(gpu, &l.ffn_post_conv)?,
        k_conv_proj: l.k_conv_proj.map(|m| matrix(gpu, m)).transpose()?,
        v_conv_proj: l.v_conv_proj.map(|m| matrix(gpu, m)).transpose()?,
        ffn_norm: f32s(gpu, &l.ffn_norm)?,
        router: matrix(gpu, l.router)?,
        router_bias: f32s(gpu, &l.router_bias)?,
        gate_up: matrix(gpu, l.gate_up)?,
        down: matrix(gpu, l.down)?,
        looped: match &l.looped {
            Some(lp) => Some((
                f32s(gpu, &lp.queries.concat())?,
                [
                    controller(gpu, &lp.controllers[0])?,
                    controller(gpu, &lp.controllers[1])?,
                    controller(gpu, &lp.controllers[2])?,
                    controller(gpu, &lp.controllers[3])?,
                ],
            )),
            None => None,
        },
    })
}

impl<'a> Inkling2Cuda<'a> {
    pub fn load(file: &'a GgufFile, gpu: Gpu) -> Result<Self> {
        // The CPU model reads and checks the file; its weights go up from there.
        let cpu = Inkling2::load(file, Pool::new(1))?;
        let c = cpu.config.clone();
        if c.layers.div_ceil(c.block) + 2 > MAX_STATES {
            return Err(Error::Unsupported(format!(
                "Inkling 2 with {} blocks on the GPU",
                c.layers.div_ceil(c.block)
            )));
        }
        if c.head_dim != 128 || c.rel_dim > 32 || c.conv > 8 || c.experts > 512 {
            return Err(Error::Unsupported(
                "Inkling 2 with these head, bias, convolution or expert sizes on the GPU".into(),
            ));
        }
        let layers = cpu
            .layers
            .iter()
            .map(|l| layer(&gpu, l))
            .collect::<Result<_>>()?;
        Ok(Self {
            embedding: cpu.embedding,
            embedding_norm: f32s(&gpu, &cpu.embedding_norm)?,
            output: matrix(&gpu, cpu.output)?,
            output_norm: f32s(&gpu, &cpu.output_norm)?,
            output_res_query: f32s(&gpu, &cpu.output_res_query)?,
            layers,
            config: c,
            gpu,
            precision: Precision::default(),
        })
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// How a batch's matrix products run: by default (Int8) on the neural accelerators
    /// where the GPU has them, the activations rounded to f16; Exact keeps f32 throughout.
    /// Sessions made after this take it.
    pub fn set_precision(&mut self, precision: Precision) {
        self.precision = precision;
    }

    pub fn precision(&self) -> Precision {
        self.precision
    }

    pub fn session(&self, capacity: usize) -> Result<Inkling2CudaSession> {
        self.session_with_batch(capacity, BATCH)
    }

    /// A session that reads up to `batch` tokens at a time (its buffers are sized by it).
    pub fn session_with_batch(
        &self,
        capacity: usize,
        batch: usize,
    ) -> Result<Inkling2CudaSession> {
        let c = &self.config;
        let g = &self.gpu;
        let batch = batch.min(capacity).max(1);
        let alloc = |floats: usize| g.alloc(floats.max(1) * 4).map_err(gpu_error);
        let kv_dim = c.kv_heads * c.head_dim;
        let channels = 2 * kv_dim + 2 * c.dim;
        let per_history = (c.conv - 1) * channels;
        let histories = alloc(c.slots() * per_history)?;
        g.zero(&histories).map_err(gpu_error)?;
        let mut slots = Vec::with_capacity(c.slots());
        for slot in 0..c.slots() {
            let ring = c.is_swa[physical(c, slot)];
            let rows = if ring {
                (c.sliding_window + batch + REWIND).min(capacity)
            } else {
                capacity
            };
            slots.push(Slot {
                keys: alloc(rows * kv_dim)?,
                values: alloc(rows * kv_dim)?,
                rows,
                ring,
                history: histories.f32_at(slot * per_history),
            });
        }
        let (dim, q_dim, b) = (c.dim, c.heads * c.head_dim, batch);
        let rank = c.looped.map_or(1, |l| l.gate_rank);
        let (used, width) = (c.experts_used, c.expert_width);
        let fast = self.precision == Precision::Int8
            && batch >= MOE_GROUPED_FROM
            && g.moe_fast(self.layers[0].gate_up.kind)
            && g.moe_fast(self.layers[0].down.kind);
        let scratch = Scratch {
            embedded: alloc(b * dim)?,
            x: alloc(b * dim)?,
            stack: alloc(MAX_STATES * b * dim)?,
            bank: alloc(MAX_STATES * b * dim)?,
            prelude: alloc(b * dim)?,
            inputs: alloc(3 * b * dim)?,
            normed: alloc(3 * b * dim)?,
            pair: alloc(b * 2 * dim)?,
            low: alloc(b * rank)?,
            gain: alloc(b * dim)?,
            forget: alloc(b * dim)?,
            messages: alloc(4 * b * dim)?,
            q: alloc(b * q_dim)?,
            k: alloc(b * kv_dim)?,
            v: alloc(b * kv_dim)?,
            r: alloc(b * c.rel_dim * c.heads)?,
            sink: alloc(b * c.heads)?,
            gate: alloc(b * q_dim)?,
            k_taps: alloc(b * c.conv * c.kv_heads)?,
            v_taps: alloc(b * c.conv * c.kv_heads)?,
            attention: alloc(b * q_dim)?,
            attention_parts: alloc(Gpu::ink_attention_scratch(c.heads, b))?,
            attn_out: alloc(b * dim)?,
            ffn_in: alloc(b * dim)?,
            ffn_normed: alloc(b * dim)?,
            router: alloc(b * c.experts)?,
            idx: alloc(b * used)?,
            share: alloc(b * used)?,
            gate_up: alloc(b * used * 2 * width)?,
            expert_out: alloc(b * used * dim)?,
            order: alloc(b * used)?,
            place: alloc(b * used)?,
            tiles: alloc(3 * Gpu::moe_tiles(b * used, c.experts))?,
            sorted: if fast {
                Some(g.alloc(b * used * dim.max(width) * 2).map_err(gpu_error)?)
            } else {
                None
            },
            ffn_out: alloc(b * dim)?,
            out: alloc(HELD.min(b) * dim)?,
            out_normed: alloc(HELD.min(b) * dim)?,
            logits: alloc(HELD.min(b) * c.padded_vocab)?,
            q8: if batch > 1 {
                q8_room(g, self.precision, batch, 2 * dim, &self.output)?
            } else {
                None
            },
        };
        Ok(Inkling2CudaSession {
            slots,
            histories,
            scratch,
            batch,
            len: 0,
            capacity,
            furthest: 0,
            logits: vec![0.0; c.vocab],
            rows: vec![0.0; batch * dim],
        })
    }

    /// Empties a session.
    pub fn reset(&self, s: &mut Inkling2CudaSession) -> Result<()> {
        self.gpu.zero(&s.histories).map_err(gpu_error)?;
        s.len = 0;
        s.furthest = 0;
        Ok(())
    }

    /// Saves where the session is (its convolution histories), into `cp` (reused).
    pub fn save(
        &self,
        s: &Inkling2CudaSession,
        cp: &mut Option<Inkling2Checkpoint>,
    ) -> Result<()> {
        let bytes = s.histories.len_bytes();
        if cp.as_ref().is_none_or(|c| c.history.len_bytes() < bytes) {
            *cp = Some(Inkling2Checkpoint {
                len: 0,
                history: self.gpu.alloc(bytes).map_err(gpu_error)?,
            });
        }
        let c = cp.as_mut().expect("made above");
        self.gpu
            .copy(c.history.ptr(), s.histories.ptr(), bytes)
            .map_err(gpu_error)?;
        c.len = s.len;
        Ok(())
    }

    /// Takes the session back to `cp`: false, changing nothing, when it can't (the
    /// checkpoint is past what the session holds, or the rings have moved on past it).
    pub fn restore(&self, s: &mut Inkling2CudaSession, cp: &Inkling2Checkpoint) -> Result<bool> {
        if !self.can_restore(s, cp.len) {
            return Ok(false);
        }
        self.gpu
            .copy(s.histories.ptr(), cp.history.ptr(), s.histories.len_bytes())
            .map_err(gpu_error)?;
        s.len = cp.len;
        Ok(true)
    }

    /// Whether the caches still hold what a session taken back to `len` needs.
    pub fn can_restore(&self, s: &Inkling2CudaSession, len: usize) -> bool {
        let window = self.config.sliding_window;
        len <= s.len
            && s.slots
                .iter()
                .filter(|slot| slot.ring && slot.rows < s.capacity)
                .all(|slot| s.furthest.saturating_sub(len) + window <= slot.rows)
    }

    /// Runs `token` at the next position; the logits if `want_logits`.
    pub fn forward<'s>(
        &self,
        s: &'s mut Inkling2CudaSession,
        token: u32,
        want_logits: bool,
    ) -> Result<Option<&'s [f32]>> {
        let logits = if want_logits { Logits::Last } else { Logits::None };
        self.forward_batch(s, &[token], logits)
    }

    /// Runs `tokens` at the next positions, in batches of up to the session's batch: the
    /// last one's logits, every one's (a row of the vocabulary each), or none.
    pub fn forward_batch<'s>(
        &self,
        s: &'s mut Inkling2CudaSession,
        tokens: &[u32],
        logits: Logits,
    ) -> Result<Option<&'s [f32]>> {
        let vocab = self.config.vocab;
        if logits == Logits::All {
            let mut all = vec![0.0; tokens.len() * vocab];
            let step = HELD.min(s.batch);
            for (i, chunk) in tokens.chunks(step).enumerate() {
                self.queue(s, chunk, true)?;
                self.queue_logits(s, chunk.len())?;
                for row in 0..chunk.len() {
                    let at = (i * step + row) * vocab;
                    self.held_logits(s, row, &mut all[at..at + vocab])?;
                }
            }
            s.logits = all;
            return Ok(Some(&s.logits));
        }
        for chunk in tokens.chunks(s.batch) {
            self.queue(s, chunk, false)?;
        }
        if logits == Logits::None || tokens.is_empty() {
            return Ok(None);
        }
        self.queue_logits(s, 1)?;
        s.logits.resize(vocab, 0.0);
        s.scratch
            .logits
            .read_f32(&mut s.logits)
            .map_err(gpu_error)?;
        Ok(Some(&s.logits))
    }

    /// Reads `tokens` (at most 64: a round of guesses) and keeps every one's logits on
    /// the GPU, for `held_logits`, until the session next runs a token.
    pub fn forward_batch_held(&self, s: &mut Inkling2CudaSession, tokens: &[u32]) -> Result<()> {
        if tokens.len() > HELD.min(s.batch) {
            return Err(Error::Invalid(format!(
                "{} tokens' logits held at once (at most {})",
                tokens.len(),
                HELD.min(s.batch)
            )));
        }
        self.queue(s, tokens, true)?;
        self.queue_logits(s, tokens.len())
    }

    /// Row `row` of the logits the last pass kept, into `out` (a row of the vocabulary).
    pub fn held_logits(&self, s: &Inkling2CudaSession, row: usize, out: &mut [f32]) -> Result<()> {
        let vocab = self.config.vocab;
        s.scratch
            .logits
            .read_f32_at(row * self.config.padded_vocab, &mut out[..vocab])
            .map_err(gpu_error)
    }

    /// Runs `token` and picks the likeliest next one on the GPU.
    pub fn forward_greedy(&self, s: &mut Inkling2CudaSession, token: u32) -> Result<u32> {
        self.queue(s, &[token], false)?;
        self.queue_logits(s, 1)?;
        self.gpu
            .argmax(s.scratch.logits.ptr(), self.config.vocab)
            .map_err(gpu_error)
    }

    /// Queues a batch (up to the session's batch size): the last token's mix of its blocks
    /// ends in `scratch.out`, or with `every` (at most HELD tokens) every token's, a row
    /// each.
    fn queue(&self, s: &mut Inkling2CudaSession, tokens: &[u32], every: bool) -> Result<()> {
        let c = &self.config;
        let g = &self.gpu;
        let (pos0, t) = (s.len, tokens.len());
        assert!(t >= 1 && t <= s.batch, "a batch of {t} tokens");
        if pos0 + t > s.capacity {
            return Err(Error::Invalid(format!(
                "the session is full ({} positions)",
                s.capacity
            )));
        }
        let dim = c.dim;
        for (i, &token) in tokens.iter().enumerate() {
            if token as usize >= c.vocab {
                return Err(Error::Invalid(format!(
                    "token {token} is past the vocabulary"
                )));
            }
            self.embedding
                .decode_row(token as usize, &mut s.rows[i * dim..(i + 1) * dim]);
        }
        let sc = &s.scratch;
        sc.embedded
            .write_f32(&s.rows[..t * dim])
            .map_err(gpu_error)?;
        g.rms_norm_rows(
            sc.embedded.ptr(),
            self.embedding_norm.ptr(),
            sc.x.ptr(),
            t,
            dim,
            c.norm_eps,
        )
        .map_err(gpu_error)?;

        let block = |buffer: &Buffer, i: usize| buffer.f32_at(i * t * dim);
        let mut blocks = 0;
        match c.looped {
            None => {
                for l in 0..c.layers {
                    self.layer(s, l, l, pos0, t, &mut blocks, false)?;
                }
            }
            Some(lp) => {
                for l in 0..lp.start {
                    self.layer(s, l, l, pos0, t, &mut blocks, false)?;
                }
                let sc = &s.scratch;
                self.copy(sc.prelude.ptr(), sc.x.ptr(), t * dim)?;
                let prelude_blocks = blocks;
                let core = lp.end - lp.start;
                let mut bank = 0;
                for pass in 0..lp.passes {
                    if pass > 0 {
                        // The previous pass's blocks after its first core block, and its
                        // output; then back to the prelude's.
                        let sc = &s.scratch;
                        let from = prelude_blocks + 1;
                        bank = blocks - from + 1;
                        self.copy(sc.bank.ptr(), block(&sc.stack, from), (bank - 1) * t * dim)?;
                        self.copy(block(&sc.bank, bank - 1), sc.x.ptr(), t * dim)?;
                        self.copy(sc.x.ptr(), sc.prelude.ptr(), t * dim)?;
                        blocks = prelude_blocks;
                    }
                    for l in lp.start..lp.end {
                        let slot = if pass == 0 {
                            l
                        } else {
                            c.layers + (pass - 1) * core + (l - lp.start)
                        };
                        if pass > 0 {
                            let (queries, _) =
                                self.layers[l].looped.as_ref().expect("a core layer");
                            let sc = &s.scratch;
                            g.ink_mix(
                                sc.bank.ptr(),
                                bank - 1,
                                block(&sc.bank, bank - 1),
                                queries.ptr(),
                                4,
                                sc.messages.ptr(),
                                dim,
                                t,
                                t,
                            )
                            .map_err(gpu_error)?;
                        }
                        self.layer(s, slot, l, pos0, t, &mut blocks, pass > 0)?;
                    }
                }
                for l in lp.end..c.layers {
                    self.layer(s, l, l, pos0, t, &mut blocks, false)?;
                }
            }
        }
        // The output mixes every block and x: for the last token (or every one).
        let sc = &s.scratch;
        let (first, rows) = if every { (0, t) } else { (t - 1, 1) };
        g.ink_mix(
            sc.stack.f32_at(first * dim),
            blocks,
            sc.x.f32_at(first * dim),
            self.output_res_query.ptr(),
            1,
            sc.out.ptr(),
            dim,
            rows,
            t,
        )
        .map_err(gpu_error)?;
        s.len += t;
        s.furthest = s.furthest.max(s.len);
        Ok(())
    }

    /// The logits of the `rows` rows of `scratch.out` (as `queue` left them).
    fn queue_logits(&self, s: &mut Inkling2CudaSession, rows: usize) -> Result<()> {
        let c = &self.config;
        let g = &self.gpu;
        let sc = &s.scratch;
        g.rms_norm_rows(
            sc.out.ptr(),
            self.output_norm.ptr(),
            sc.out_normed.ptr(),
            rows,
            c.dim,
            c.norm_eps,
        )
        .map_err(gpu_error)?;
        let q8 = if rows > 1 { sc.q8.as_ref() } else { None };
        self.output.mul(
            &Input::new(g, sc.out_normed.ptr(), rows, q8),
            sc.logits.ptr(),
        )?;
        if let Some(cap) = c.softcap {
            let n = (rows - 1) * c.padded_vocab + c.vocab;
            g.softcap(sc.logits.ptr(), cap, n).map_err(gpu_error)?;
        }
        Ok(())
    }

    fn copy(&self, to: CuDevicePtr, from: CuDevicePtr, floats: usize) -> Result<()> {
        self.gpu.copy(to, from, floats * 4).map_err(gpu_error)
    }

    /// `cur = cur − tanh(forget) cur + tanh(gain) message` for `t` tokens, gated on both
    /// (see `inkling2::control`).
    fn control(
        &self,
        sc: &Scratch,
        cur: CuDevicePtr,
        message: CuDevicePtr,
        ctrl: &GpuController,
        t: usize,
    ) -> Result<()> {
        let (g, dim) = (&self.gpu, self.config.dim);
        let q8 = sc.q8.as_ref();
        g.ink_pair_norm(cur, message, sc.pair.ptr(), dim, t)
            .map_err(gpu_error)?;
        ctrl.down
            .mul(&Input::new(g, sc.pair.ptr(), t, q8), sc.low.ptr())?;
        let low = Input::new(g, sc.low.ptr(), t, q8);
        ctrl.gain.mul(&low, sc.gain.ptr())?;
        ctrl.forget.mul(&low, sc.forget.ptr())?;
        g.ink_control(cur, message, sc.gain.ptr(), sc.forget.ptr(), cur, t * dim)
            .map_err(gpu_error)
    }

    /// One visit of physical layer `l` for the batch's `t` tokens, with its state in
    /// `slot`; with `messages` (on the loop's later passes) the controllers take the ones
    /// in `scratch.messages`.
    #[allow(clippy::too_many_arguments)]
    fn layer(
        &self,
        s: &Inkling2CudaSession,
        slot: usize,
        l: usize,
        pos0: usize,
        t: usize,
        blocks: &mut usize,
        messages: bool,
    ) -> Result<()> {
        let c = &self.config;
        let g = &self.gpu;
        let w = &self.layers[l];
        let sc = &s.scratch;
        let st = &s.slots[slot];
        let q8 = sc.q8.as_ref();
        let (dim, hd, heads) = (c.dim, c.head_dim, c.heads);
        let kv_dim = c.kv_heads * hd;
        let channels = 2 * kv_dim + 2 * dim;
        let swa = c.is_swa[l];
        let block = |buffer: &Buffer, i: usize| buffer.f32_at(i * t * dim);
        let e = gpu_error;

        // The attention's inputs, a block of rows each for q, k and v.
        match &w.attn_res_query {
            None => self.copy(sc.inputs.ptr(), sc.x.ptr(), t * dim)?,
            Some((query, n)) => {
                g.ink_mix(
                    sc.stack.ptr(),
                    *blocks,
                    sc.x.ptr(),
                    query.ptr(),
                    *n,
                    sc.inputs.ptr(),
                    dim,
                    t,
                    t,
                )
                .map_err(e)?;
            }
        }
        let distinct = !swa || messages;
        if distinct && (swa || w.attn_res_query.is_none()) {
            self.copy(block(&sc.inputs, 1), sc.inputs.ptr(), t * dim)?;
            self.copy(block(&sc.inputs, 2), sc.inputs.ptr(), t * dim)?;
        }
        if messages {
            let (_, ctrls) = w.looped.as_ref().expect("a core layer");
            for (j, ctrl) in ctrls.iter().enumerate().take(3) {
                self.control(sc, block(&sc.inputs, j), block(&sc.messages, j), ctrl, t)?;
            }
        }
        let rows = if distinct { 3 } else { 1 };
        for j in 0..rows {
            let norm = if swa { &w.norms[0] } else { &w.norms[j] };
            g.rms_norm_rows(
                block(&sc.inputs, j),
                norm.ptr(),
                block(&sc.normed, j),
                t,
                dim,
                c.norm_eps,
            )
            .map_err(e)?;
        }
        let h = |j: usize| block(&sc.normed, if distinct { j } else { 0 });

        // Projections, an input's products together (an input is rounded once for its
        // products, into the one 8-bit buffer).
        let h0 = Input::new(g, h(0), t, q8);
        w.q.mul(&h0, sc.q.ptr())?;
        w.rel.mul(&h0, sc.r.ptr())?;
        w.sink.mul(&h0, sc.sink.ptr())?;
        w.gate.mul(&h0, sc.gate.ptr())?;
        let h1_own = distinct.then(|| Input::new(g, h(1), t, q8));
        let h1 = h1_own.as_ref().unwrap_or(&h0);
        w.k.mul(h1, sc.k.ptr())?;
        if let Some(p) = &w.k_conv_proj {
            p.mul(h1, sc.k_taps.ptr())?;
        }
        let h2_own = distinct.then(|| Input::new(g, h(2), t, q8));
        let h2 = h2_own.as_ref().unwrap_or(&h0);
        w.v.mul(h2, sc.v.ptr())?;
        if let Some(p) = &w.v_conv_proj {
            p.mul(h2, sc.v_taps.ptr())?;
        }

        // Keys and values through their convolutions, clipped and normed, into the cache;
        // the queries' and biases' clips.
        let history = st.history;
        let ring = if st.ring { st.rows } else { 0 };
        g.ink_kv(
            sc.k.ptr(),
            history,
            w.k_conv.ptr(),
            w.k_conv_proj.as_ref().map(|_| sc.k_taps.ptr()),
            st.keys.ptr(),
            channels,
            c.conv,
            t,
            pos0,
            ring,
            kv_dim,
            Some(c.qk_clip),
            c.norm_eps,
        )
        .map_err(e)?;
        g.ink_kv(
            sc.v.ptr(),
            history + (kv_dim * 4) as u64,
            w.v_conv.ptr(),
            w.v_conv_proj.as_ref().map(|_| sc.v_taps.ptr()),
            st.values.ptr(),
            channels,
            c.conv,
            t,
            pos0,
            ring,
            kv_dim,
            None,
            c.norm_eps,
        )
        .map_err(e)?;
        let log_scale = (!swa).then_some((c.log_scale_alpha, c.log_scale_floor));
        g.ink_clip(
            sc.q.ptr(),
            Some(w.q_clip.ptr()),
            t * heads,
            hd,
            c.qk_clip,
            heads,
            pos0,
            log_scale,
        )
        .map_err(e)?;
        g.ink_clip(
            sc.r.ptr(),
            Some(w.rel_clip.ptr()),
            t * heads,
            c.rel_dim,
            c.rel_clip,
            heads,
            pos0,
            log_scale,
        )
        .map_err(e)?;

        g.ink_attention(&InkAttention {
            q: sc.q.ptr(),
            keys: st.keys.ptr(),
            values: st.values.ptr(),
            out: sc.attention.ptr(),
            r: sc.r.ptr(),
            bias: w.rel_bias.ptr(),
            sink: sc.sink.ptr(),
            gate: sc.gate.ptr(),
            parts: sc.attention_parts.ptr(),
            pos0,
            tokens: t,
            window: if swa { c.sliding_window } else { 0 },
            extent: if swa { c.sliding_window } else { c.rel_extent },
            ring,
            kv_dim,
            group: heads / c.kv_heads,
            head_dim: hd,
            rel_dim: c.rel_dim,
            heads,
            scale: 1.0 / (hd as f32).sqrt(),
            flash: self.precision == Precision::Int8,
        })
        .map_err(e)?;
        w.o.mul(&Input::new(g, sc.attention.ptr(), t, q8), sc.attn_out.ptr())?;
        // The first layer of a block keeps its input as a block and starts afresh with the
        // attention's output; the others add it to x (the convolution writes it there).
        let new_block = l.is_multiple_of(c.block);
        if new_block {
            self.copy(block(&sc.stack, *blocks), sc.x.ptr(), t * dim)?;
            *blocks += 1;
        }
        g.ink_conv(
            sc.attn_out.ptr(),
            history + (2 * kv_dim * 4) as u64,
            w.attn_post_conv.ptr(),
            None,
            dim,
            channels,
            c.conv,
            1,
            t,
            sc.x.ptr(),
            !new_block,
        )
        .map_err(e)?;

        // The experts, on a mix of the blocks and x.
        g.ink_mix(
            sc.stack.ptr(),
            *blocks,
            sc.x.ptr(),
            w.ffn_res_query.ptr(),
            1,
            sc.ffn_in.ptr(),
            dim,
            t,
            t,
        )
        .map_err(e)?;
        if messages {
            let (_, ctrls) = w.looped.as_ref().expect("a core layer");
            self.control(sc, sc.ffn_in.ptr(), block(&sc.messages, 3), &ctrls[3], t)?;
        }
        g.rms_norm_rows(
            sc.ffn_in.ptr(),
            w.ffn_norm.ptr(),
            sc.ffn_normed.ptr(),
            t,
            dim,
            c.norm_eps,
        )
        .map_err(e)?;
        self.experts(sc, w, t)?;
        g.ink_conv(
            sc.ffn_out.ptr(),
            history + ((2 * kv_dim + dim) * 4) as u64,
            w.ffn_post_conv.ptr(),
            None,
            dim,
            channels,
            c.conv,
            1,
            t,
            sc.x.ptr(),
            true,
        )
        .map_err(e)?;
        Ok(())
    }

    /// The experts for `t` tokens of `scratch.ffn_normed` into `scratch.ffn_out`: a slot
    /// (a token's turn with one of its experts) at a time for a few tokens, grouped by
    /// expert for a batch (each expert's matrices read once), on the neural accelerators
    /// where the session has them.
    fn experts(&self, sc: &Scratch, w: &GpuLayer, t: usize) -> Result<()> {
        let c = &self.config;
        let g = &self.gpu;
        let e = gpu_error;
        let (dim, used, width, count) = (c.dim, c.experts_used, c.expert_width, c.experts);
        let slots = t * used;
        w.router.mul(
            &Input::new(g, sc.ffn_normed.ptr(), t, None),
            sc.router.ptr(),
        )?;
        g.ink_moe_pick(
            sc.router.ptr(),
            w.router_bias.ptr(),
            sc.idx.ptr(),
            sc.share.ptr(),
            count,
            used,
            c.expert_scale,
            t,
        )
        .map_err(e)?;
        let grouped = t >= MOE_GROUPED_FROM;
        let fast = sc.sorted.as_ref().filter(|_| grouped);
        if grouped {
            g.moe_group(
                sc.idx.ptr(),
                sc.order.ptr(),
                sc.tiles.ptr(),
                sc.place.ptr(),
                slots,
                count,
                fast.is_some(),
            )
            .map_err(e)?;
        }
        let (gu, down) = (&w.gate_up, &w.down);
        if let Some(sorted) = fast {
            g.moe_gather_half(
                sc.ffn_normed.ptr(),
                sorted.ptr(),
                sc.order.ptr(),
                used,
                dim,
                dim,
                true,
                slots,
            )
            .map_err(e)?;
            g.moe_matmul_fast(
                gu.kind,
                gu.data.ptr(),
                sorted.ptr(),
                sc.gate_up.ptr(),
                2 * width,
                dim,
                sc.tiles.ptr(),
                2 * width,
                slots,
                count,
            )
            .map_err(e)?;
            g.ink_situ(
                sc.gate_up.ptr(),
                sorted.ptr(),
                width,
                slots,
                c.situ_beta,
                c.situ_linear_beta,
                width,
                true,
            )
            .map_err(e)?;
            g.moe_matmul_fast(
                down.kind,
                down.data.ptr(),
                sorted.ptr(),
                sc.expert_out.ptr(),
                dim,
                width,
                sc.tiles.ptr(),
                dim,
                slots,
                count,
            )
            .map_err(e)?;
        } else {
            if grouped {
                g.moe_matmul(
                    gu.kind,
                    gu.data.ptr(),
                    sc.ffn_normed.ptr(),
                    sc.gate_up.ptr(),
                    2 * width,
                    dim,
                    sc.order.ptr(),
                    sc.tiles.ptr(),
                    used,
                    dim,
                    false,
                    2 * width,
                    slots,
                    count,
                )
                .map_err(e)?;
            } else {
                g.moe_matvec(
                    gu.kind,
                    gu.data.ptr(),
                    sc.ffn_normed.ptr(),
                    sc.gate_up.ptr(),
                    2 * width,
                    dim,
                    sc.idx.ptr(),
                    used,
                    dim,
                    false,
                    2 * width,
                    slots,
                )
                .map_err(e)?;
            }
            // The activation into each slot's gate half, which the down product reads.
            g.ink_situ(
                sc.gate_up.ptr(),
                sc.gate_up.ptr(),
                width,
                slots,
                c.situ_beta,
                c.situ_linear_beta,
                2 * width,
                false,
            )
            .map_err(e)?;
            if grouped {
                g.moe_matmul(
                    down.kind,
                    down.data.ptr(),
                    sc.gate_up.ptr(),
                    sc.expert_out.ptr(),
                    dim,
                    width,
                    sc.order.ptr(),
                    sc.tiles.ptr(),
                    used,
                    2 * width,
                    true,
                    dim,
                    slots,
                    count,
                )
                .map_err(e)?;
            } else {
                g.moe_matvec(
                    down.kind,
                    down.data.ptr(),
                    sc.gate_up.ptr(),
                    sc.expert_out.ptr(),
                    dim,
                    width,
                    sc.idx.ptr(),
                    used,
                    2 * width,
                    true,
                    dim,
                    slots,
                )
                .map_err(e)?;
            }
        }
        g.moe_sum(
            sc.ffn_out.ptr(),
            sc.expert_out.ptr(),
            sc.share.ptr(),
            used,
            dim,
            t,
            fast.map(|_| sc.place.ptr()),
        )
        .map_err(e)
    }
}

/// The physical layer a visit (`slot`) runs.
fn physical(c: &Config, slot: usize) -> usize {
    match c.looped {
        Some(l) if slot >= c.layers => l.start + (slot - c.layers) % (l.end - l.start),
        _ => slot,
    }
}
