//! Qwen 3.5 on an NVIDIA GPU: the same steps as `qwen35.rs`, with every weight, the
//! attention layers' keys and values, and the DeltaNet layers' convolution history and
//! states in GPU memory. Prompts go through in batches and generated tokens replay a
//! recorded graph, as in `llama_cuda.rs`. A batch's DeltaNet steps still go token by
//! token (each needs the state the last one left), but inside one kernel launch.

use super::gpu::{
    GpuMatrix, Input, LogitsBuffer, Outputs, Positions, gpu_error, load_state, q8_room, record,
    row, save_state,
};
use super::qwen35::Config;
use super::{DEFAULT_BATCH, Logits, Precision, Result, SavedState, matrix, vector};
use crate::GgufFile;
use crate::cpu::Matrix;
use crate::cuda::{self, Buffer, Gpu, Graph, Q8, QkvPrep};

struct Attention {
    q: GpuMatrix,
    k: GpuMatrix,
    v: GpuMatrix,
    o: GpuMatrix,
    q_norm: Buffer,
    k_norm: Buffer,
}

struct DeltaNet {
    qkv: GpuMatrix,
    gate: GpuMatrix,
    beta: GpuMatrix,
    alpha: GpuMatrix,
    dt_bias: Buffer,
    a: Buffer,
    conv: Buffer,
    norm: Buffer,
    out: GpuMatrix,
}

enum Mixer {
    Attention(Attention),
    DeltaNet(DeltaNet),
}

struct Layer {
    attn_norm: Buffer,
    mixer: Mixer,
    ffn_norm: Buffer,
    gate: GpuMatrix,
    up: GpuMatrix,
    down: GpuMatrix,
}

pub struct Qwen35Cuda<'a> {
    pub config: Config,
    gpu: Gpu,
    precision: Precision,
    embedding: Matrix<'a>,
    output: GpuMatrix,
    output_norm: Buffer,
    layers: Vec<Layer>,
    /// The multi-token-prediction layer (`blk.N.nextn.*`, Qwen 3.8 27B), for drafting.
    mtp: Option<Mtp>,
}

/// Qwen's multi-token-prediction layer: from the model's normed hidden state after token
/// i and token i + 1, a guess at token i + 2. `fc([RMSNorm(embed(t)) × enorm, RMSNorm(h)
/// × hnorm])`, one full-attention layer of the model's kind with its own cache, then the
/// head's norm and the model's output matrix (mlx-vlm's `qwen3_5_mtp`).
struct Mtp {
    /// eh_proj split by columns: the half that takes the embedding, and the hidden state's.
    e_proj: GpuMatrix,
    h_proj: GpuMatrix,
    enorm: Buffer,
    hnorm: Buffer,
    head_norm: Buffer,
    layer: Layer,
}

enum Memory {
    Attention { keys: Buffer, values: Buffer },
    DeltaNet { history: Buffer, state: Buffer },
}

/// One conversation's state on the GPU. Use it only with the model that made it.
pub struct Qwen35CudaSession {
    memory: Vec<Memory>,
    len: usize,
    capacity: usize,
    batch: usize,
    embedding: Vec<f32>,
    pos: Buffer,
    // A row per token of the batch:
    x: Buffer,
    h: Buffer,
    mixed: Buffer,
    query_gate: Buffer,
    q: Buffer,
    k_new: Buffer,
    v_new: Buffer,
    attention: Buffer,
    qkv: Buffer,
    conv: Buffer,
    z: Buffer,
    beta: Buffer,
    alpha: Buffer,
    heads_out: Buffer,
    /// A prompt's DeltaNet chunks (Gpu::delta_scratch_floats).
    delta_scratch: Buffer,
    gate: Buffer,
    up: Buffer,
    logits_gpu: LogitsBuffer,
    logits: Vec<f32>,
    /// The batch's activations in 8 bits, unless the model is exact.
    q8: Option<Q8>,
    /// Keys and values in f16 (8-bit mode) rather than f32.
    half_cache: bool,
    graph: Option<Graph>,
    /// Graphs for batches of up to `GRAPH_TOKENS` tokens, by their number and the outputs
    /// wanted, each recorded the first time such a batch is read (CUDA).
    few_graphs: Vec<Option<Graph>>,
    /// With [`Qwen35Cuda::keep_steps`]: for each DeltaNet layer, room for its state before
    /// a batch of up to [`STEP_TOKENS`] and after each of its tokens but the last, and for
    /// its convolution's history then the batch's inputs, so the batch can be cut back
    /// ([`Qwen35Cuda::rewind`]).
    steps: Option<Vec<(Buffer, Buffer)>>,
    /// The last batch read with its steps kept: where it started and how many tokens.
    stepped: Option<(usize, usize)>,
    /// With [`Qwen35Cuda::use_mtp`]: the multi-token-prediction layer's side of the chat.
    mtp: Option<MtpSession>,
}

/// The multi-token-prediction layer's cache and inputs for one session. Its entry i
/// pairs the model's normed hidden state after token i with token i + 1 (and guesses
/// token i + 2); the hidden state after the last token read waits in `pending` for the
/// token picked after it.
struct MtpSession {
    keys: Buffer,
    values: Buffer,
    pos: Buffer,
    /// A row per token: the layer's residual stream, its normed input and output, and the
    /// two halves of its input (the tokens' embeddings, the hidden states).
    x: Buffer,
    h: Buffer,
    emb: Buffer,
    hid: Buffer,
    host: Vec<f32>,
    logits: Vec<f32>,
    pending: Buffer,
    /// `pending` as it was before the last batch, and whether it was there: a rewind to
    /// that batch's start brings it back.
    pending_before: Buffer,
    had_pending: bool,
    has_pending: bool,
    /// Entries in the cache, and how many at its end came from guesses (the layer's own
    /// output as the hidden state), to drop before the model's own hidden states replace
    /// them.
    len: usize,
    chained: usize,
}

/// A checkpoint's note of the multi-token-prediction layer's side.
struct MtpMark {
    len: usize,
    has_pending: bool,
    pending: Buffer,
}

/// The most tokens a batch keeps its DeltaNet steps for (a draft's round: the last pick
/// and its guesses), as the Metal backend's few-token kernels take.
const STEP_TOKENS: usize = 8;

impl Qwen35CudaSession {
    /// Whether short batches keep their DeltaNet steps ([`Qwen35Cuda::keep_steps`]).
    pub fn keeps_steps(&self) -> bool {
        self.steps.is_some()
    }

    /// Whether it keeps the multi-token-prediction layer's cache up ([`Qwen35Cuda::use_mtp`]).
    pub fn keeps_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The GPU memory each position takes in this session's caches (keys and values).
    pub fn cache_bytes_per_position(&self) -> usize {
        self.attention_buffers()
            .iter()
            .map(|(b, _, _)| b.len_bytes())
            .sum::<usize>()
            / self.capacity.max(1)
    }

    /// The session at checkpoint `cp` (made from it, and still valid for it: see
    /// `Qwen35Cuda::restore`) copied to system memory: the attention layers' keys and
    /// values up to there, and the DeltaNet layers as saved there.
    pub fn export(&self, cp: &Qwen35Checkpoint) -> Result<SavedState> {
        assert!(cp.len <= self.len, "a checkpoint ahead of its session");
        let whole: Vec<&Buffer> = cp.saved.iter().flat_map(|(h, s)| [h, s]).collect();
        save_state(&self.attention_buffers(), self.capacity, cp.len, &whole)
    }

    /// Loads a state `export` made from a session of the same model and precision, in
    /// place of what this session held: it goes on from the saved position. (Save a
    /// checkpoint there to come back to it.)
    pub fn import(&mut self, saved: &SavedState) -> Result<()> {
        load_state(
            saved,
            saved.len(),
            &self.attention_buffers(),
            self.capacity,
            &self.delta_buffers(),
        )?;
        self.len = saved.len();
        self.stepped = None;
        // The saved state has no multi-token-prediction cache: it starts again from here.
        if let Some(m) = self.mtp.as_mut() {
            m.len = 0;
            m.chained = 0;
            m.has_pending = false;
        }
        Ok(())
    }

    fn attention_buffers(&self) -> Vec<Positions<'_>> {
        self.memory
            .iter()
            .flat_map(|m| match m {
                Memory::Attention { keys, values } => vec![(keys, None, 0), (values, None, 0)],
                Memory::DeltaNet { .. } => vec![],
            })
            .collect()
    }

    fn delta_buffers(&self) -> Vec<&Buffer> {
        self.memory
            .iter()
            .flat_map(|m| match m {
                Memory::DeltaNet { history, state } => vec![history, state],
                Memory::Attention { .. } => vec![],
            })
            .collect()
    }
}

/// A session's DeltaNet layers (their convolution history and state) at one position,
/// saved by [`Qwen35Cuda::save`]. Unlike the attention layers' keys and values, which
/// later tokens add to but never change, a DeltaNet state is rewritten by every token,
/// so going back to an earlier position needs a copy made there.
pub struct Qwen35Checkpoint {
    len: usize,
    /// (history, state) per DeltaNet layer, in layer order.
    saved: Vec<(Buffer, Buffer)>,
    /// The multi-token-prediction layer's side, for a session that keeps it.
    mtp: Option<MtpMark>,
}

impl Qwen35Checkpoint {
    /// The position it was saved at.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<'a> Qwen35Cuda<'a> {
    pub fn load(file: &'a GgufFile, gpu: Gpu) -> Result<Self> {
        let c = Config::from_gguf(file)?;
        let dim = c.dim;
        let upload = |m: Matrix| GpuMatrix::upload(&gpu, m);
        let up = |v: Vec<f32>| gpu.upload_f32(&v).map_err(gpu_error);

        let mut layers = Vec::with_capacity(c.layers);
        for i in 0..c.layers {
            layers.push(load_layer(file, &gpu, &c, i, c.is_full_attention(i))?);
        }
        let mtp = load_mtp(file, &gpu, &c)?;
        let embedding = matrix(file, "token_embd.weight", dim, c.vocab)?;
        let output_name = if file.tensor("output.weight").is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let output = upload(matrix(file, output_name, dim, c.vocab)?)?;
        Ok(Self {
            precision: Precision::default(),
            output_norm: up(vector(file, "output_norm.weight", dim)?)?,
            config: c,
            embedding,
            output,
            layers,
            mtp,
            gpu,
        })
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// How the matrix products are done: 8-bit activations (the default) or exact f32.
    /// Sessions take it when they're made.
    pub fn set_precision(&mut self, precision: Precision) {
        self.precision = precision;
    }

    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// Saves `s`'s DeltaNet layers at its current position into `into`, reusing its GPU
    /// memory when it holds an earlier checkpoint of a session of this model.
    pub fn save(&self, s: &Qwen35CudaSession, into: &mut Option<Qwen35Checkpoint>) -> Result<()> {
        let g = &self.gpu;
        let deltas = s.memory.iter().filter_map(|m| match m {
            Memory::DeltaNet { history, state } => Some((history, state)),
            Memory::Attention { .. } => None,
        });
        if into.is_none() {
            let copy = |b: &Buffer| g.alloc(b.len_bytes()).map_err(gpu_error);
            let saved = deltas
                .clone()
                .map(|(h, st)| Ok((copy(h)?, copy(st)?)))
                .collect::<Result<Vec<_>>>()?;
            *into = Some(Qwen35Checkpoint {
                len: 0,
                saved,
                mtp: None,
            });
        }
        let cp = into.as_mut().unwrap();
        for ((history, state), (h, st)) in deltas.zip(&cp.saved) {
            g.copy(h.ptr(), history.ptr(), history.len_bytes())
                .map_err(gpu_error)?;
            g.copy(st.ptr(), state.ptr(), state.len_bytes())
                .map_err(gpu_error)?;
        }
        cp.len = s.len;
        // Its cache's entries stay as they are up to here (later ones only go after them),
        // so the count and the waiting hidden state are all there is to keep.
        cp.mtp = match (&s.mtp, cp.mtp.take()) {
            (Some(m), mark) => {
                let pending = match mark {
                    Some(mark) => mark.pending,
                    None => g.alloc(m.pending.len_bytes()).map_err(gpu_error)?,
                };
                g.copy(pending.ptr(), m.pending.ptr(), m.pending.len_bytes())
                    .map_err(gpu_error)?;
                Some(MtpMark {
                    len: m.len - m.chained,
                    has_pending: m.has_pending,
                    pending,
                })
            }
            (None, _) => None,
        };
        Ok(())
    }

    /// Takes `s` back to where `cp` was saved. Valid only while `s` still holds the same
    /// tokens up to there: it went on from the checkpoint, or went back to it, but never
    /// further back (`reset`) or to another checkpoint before it.
    pub fn restore(&self, s: &mut Qwen35CudaSession, cp: &Qwen35Checkpoint) -> Result<()> {
        assert!(cp.len <= s.len, "a checkpoint ahead of its session");
        let g = &self.gpu;
        let deltas = s.memory.iter().filter_map(|m| match m {
            Memory::DeltaNet { history, state } => Some((history, state)),
            Memory::Attention { .. } => None,
        });
        for ((history, state), (h, st)) in deltas.zip(&cp.saved) {
            g.copy(history.ptr(), h.ptr(), history.len_bytes())
                .map_err(gpu_error)?;
            g.copy(state.ptr(), st.ptr(), state.len_bytes())
                .map_err(gpu_error)?;
        }
        s.len = cp.len;
        s.stepped = None;
        if let Some(m) = s.mtp.as_mut() {
            m.chained = 0;
            match &cp.mtp {
                Some(mark) => {
                    g.copy(m.pending.ptr(), mark.pending.ptr(), m.pending.len_bytes())
                        .map_err(gpu_error)?;
                    m.len = mark.len;
                    m.has_pending = mark.has_pending;
                }
                None => {
                    m.len = 0;
                    m.has_pending = false;
                }
            }
        }
        Ok(())
    }

    /// Empties `s`, as a new session would be.
    pub fn reset(&self, s: &mut Qwen35CudaSession) -> Result<()> {
        for m in &s.memory {
            if let Memory::DeltaNet { history, state } = m {
                self.gpu.zero(history).map_err(gpu_error)?;
                self.gpu.zero(state).map_err(gpu_error)?;
            }
        }
        s.len = 0;
        s.stepped = None;
        if let Some(m) = s.mtp.as_mut() {
            m.len = 0;
            m.chained = 0;
            m.has_pending = false;
        }
        Ok(())
    }

    pub fn session(&self, capacity: usize) -> Result<Qwen35CudaSession> {
        self.session_with_batch(capacity, DEFAULT_BATCH)
    }

    /// A new, empty conversation that reads up to `batch` tokens at once.
    pub fn session_with_batch(&self, capacity: usize, batch: usize) -> Result<Qwen35CudaSession> {
        let c = &self.config;
        let g = &self.gpu;
        let batch = batch.clamp(1, capacity.max(1));
        let f32s = |n: usize| g.alloc(n * 4).map_err(gpu_error);
        // Keys and values: f16 in 8-bit mode (half the memory), f32 when exact.
        let cache_bytes = if self.precision == Precision::Int8 {
            2
        } else {
            4
        };
        let cache = |n: usize| g.alloc(n * cache_bytes).map_err(gpu_error);
        let zeros = |n: usize| -> Result<Buffer> {
            let b = f32s(n)?;
            g.zero(&b).map_err(gpu_error)?;
            Ok(b)
        };
        let conv_dim = 2 * c.k_heads * c.state + c.v_heads * c.state;
        let value_dim = c.v_heads * c.state;
        let kv_dim = c.kv_heads * c.head_dim;
        let mut memory = Vec::with_capacity(c.layers);
        for l in 0..c.layers {
            memory.push(if c.is_full_attention(l) {
                Memory::Attention {
                    keys: cache(capacity * kv_dim)?,
                    values: cache(capacity * kv_dim)?,
                }
            } else {
                Memory::DeltaNet {
                    history: zeros((c.conv_kernel - 1) * conv_dim)?,
                    state: zeros(c.v_heads * c.state * c.state)?,
                }
            });
        }
        Ok(Qwen35CudaSession {
            memory,
            len: 0,
            capacity,
            batch,
            embedding: vec![0.0; batch * c.dim],
            pos: f32s(1)?,
            x: f32s(batch * c.dim)?,
            h: f32s(batch * c.dim)?,
            mixed: f32s(batch * c.dim)?,
            query_gate: f32s(batch * 2 * c.heads * c.head_dim)?,
            q: f32s(batch * c.heads * c.head_dim)?,
            k_new: f32s(batch * kv_dim)?,
            v_new: f32s(batch * kv_dim)?,
            attention: f32s(batch * c.heads * c.head_dim)?,
            qkv: f32s(batch * conv_dim)?,
            conv: f32s(batch * conv_dim)?,
            z: f32s(batch * value_dim)?,
            beta: f32s(batch * c.v_heads)?,
            alpha: f32s(batch * c.v_heads)?,
            heads_out: f32s(batch * value_dim)?,
            delta_scratch: f32s(Gpu::delta_scratch_floats(batch, c.v_heads, c.state))?,
            gate: f32s(batch * c.ffn_dim)?,
            up: f32s(batch * c.ffn_dim)?,
            logits_gpu: LogitsBuffer::new(g, c.vocab)?,
            logits: vec![0.0; c.vocab],
            q8: q8_room(
                &self.gpu,
                self.precision,
                batch,
                c.dim
                    .max(c.heads * c.head_dim)
                    .max(value_dim)
                    .max(c.ffn_dim),
            )?,
            half_cache: self.precision == Precision::Int8,
            graph: None,
            few_graphs: Vec::new(),
            steps: None,
            stepped: None,
            mtp: None,
        })
    }

    /// Whether the file has a multi-token-prediction layer to draft with (and this backend
    /// can go back, which drafting needs).
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some() && self.can_rewind()
    }

    /// From now on `s` keeps the multi-token-prediction layer's cache up with every token
    /// it reads (one more layer a token, and the last layer run for every token of a
    /// batch), so [`Qwen35Cuda::mtp_draft`] can guess. Call it on an empty session.
    pub fn use_mtp(&self, s: &mut Qwen35CudaSession) -> Result<()> {
        assert!(
            self.has_mtp(),
            "the model has no multi-token-prediction layer"
        );
        if s.mtp.is_some() {
            return Ok(());
        }
        let c = &self.config;
        let g = &self.gpu;
        let f32s = |n: usize| g.alloc(n * 4).map_err(gpu_error);
        let cache_bytes = if s.half_cache { 2 } else { 4 };
        let kv_dim = c.kv_heads * c.head_dim;
        let cache = |n: usize| g.alloc(n * cache_bytes).map_err(gpu_error);
        s.mtp = Some(MtpSession {
            keys: cache(s.capacity * kv_dim)?,
            values: cache(s.capacity * kv_dim)?,
            pos: f32s(1)?,
            x: f32s(s.batch * c.dim)?,
            h: f32s(s.batch * c.dim)?,
            emb: f32s(s.batch * c.dim)?,
            hid: f32s(s.batch * c.dim)?,
            host: vec![0.0; s.batch * c.dim],
            logits: vec![0.0; c.vocab],
            pending: f32s(c.dim)?,
            pending_before: f32s(c.dim)?,
            had_pending: false,
            has_pending: false,
            len: s.len.saturating_sub(1),
            chained: 0,
        });
        Ok(())
    }

    /// Runs the multi-token-prediction layer for `t` entries (`m.emb` holding the tokens'
    /// embeddings, `m.hid` the hidden states), at cache positions from `*m.pos` on,
    /// leaving each entry's normed output in `m.h`. Uses `s`'s scratch buffers.
    fn queue_mtp(&self, s: &Qwen35CudaSession, m: &MtpSession, t: usize) -> Result<()> {
        let mtp = self.mtp.as_ref().expect("a model with the layer");
        let c = &self.config;
        let g = &self.gpu;
        let e = gpu_error;
        let (dim, hd, eps) = (c.dim, c.head_dim, c.norm_eps);
        let kv_dim = c.kv_heads * hd;
        g.rms_norm_rows(m.emb.ptr(), mtp.enorm.ptr(), m.emb.ptr(), t, dim, eps)
            .map_err(e)?;
        g.rms_norm_rows(m.hid.ptr(), mtp.hnorm.ptr(), m.hid.ptr(), t, dim, eps)
            .map_err(e)?;
        mtp.e_proj
            .mul(&Input::new(g, m.emb.ptr(), t, s.q8.as_ref()), m.x.ptr())?;
        mtp.h_proj
            .mul(&Input::new(g, m.hid.ptr(), t, s.q8.as_ref()), s.mixed.ptr())?;
        g.add(m.x.ptr(), s.mixed.ptr(), t * dim).map_err(e)?;
        let layer = &mtp.layer;
        let Mixer::Attention(a) = &layer.mixer else {
            unreachable!("the layer is a full-attention one");
        };
        g.residual_norm(
            m.x.ptr(),
            None,
            None,
            1.0,
            Some(&layer.attn_norm),
            m.h.ptr(),
            t,
            dim,
            eps,
        )
        .map_err(e)?;
        let h = Input::new(g, m.h.ptr(), t, s.q8.as_ref());
        a.q.mul(&h, s.query_gate.ptr())?;
        a.k.mul(&h, s.k_new.ptr())?;
        a.v.mul(&h, s.v_new.ptr())?;
        g.qkv_prep(
            &QkvPrep {
                q: s.query_gate.ptr(),
                q_stride: 2 * c.heads * hd,
                q_head_stride: 2 * hd,
                q_norm: Some(a.q_norm.ptr()),
                q_out: s.q.ptr(),
                k: s.k_new.ptr(),
                k_norm: Some(a.k_norm.ptr()),
                v: s.v_new.ptr(),
                v_norm: false,
                cache: Some((m.keys.ptr(), m.values.ptr())),
                heads: c.heads,
                kv_heads: c.kv_heads,
                head_dim: hd,
                rope_dims: c.rope_dims,
                rope_base: c.rope_base,
                rope_factors: None,
                neox: true,
                pos: m.pos.ptr(),
                eps,
                half_cache: s.half_cache,
                ring: 0,
            },
            t,
        )
        .map_err(e)?;
        g.attention(
            s.half_cache,
            s.q.ptr(),
            m.keys.ptr(),
            m.values.ptr(),
            s.attention.ptr(),
            m.pos.ptr(),
            0,
            c.heads,
            t,
            hd,
            kv_dim,
            c.heads / c.kv_heads,
            1.0 / (hd as f32).sqrt(),
        )
        .map_err(e)?;
        g.apply_gate(s.attention.ptr(), s.query_gate.ptr(), t * c.heads, hd)
            .map_err(e)?;
        a.o.mul(
            &Input::new(g, s.attention.ptr(), t, s.q8.as_ref()),
            s.mixed.ptr(),
        )?;
        g.residual_norm(
            m.x.ptr(),
            Some(s.mixed.ptr()),
            None,
            1.0,
            Some(&layer.ffn_norm),
            m.h.ptr(),
            t,
            dim,
            eps,
        )
        .map_err(e)?;
        #[cfg(target_os = "macos")]
        let fused = t == 1
            && layer
                .gate
                .mul_glu(g, &layer.up, m.h.ptr(), s.gate.ptr(), false)?;
        #[cfg(not(target_os = "macos"))]
        let fused = false;
        if !fused {
            let h = Input::new(g, m.h.ptr(), t, s.q8.as_ref());
            layer.gate.mul(&h, s.gate.ptr())?;
            layer.up.mul(&h, s.up.ptr())?;
            g.swiglu(s.gate.ptr(), s.up.ptr(), c.ffn_dim, c.ffn_dim, t)
                .map_err(e)?;
        }
        layer.down.mul(
            &Input::new(g, s.gate.ptr(), t, s.q8.as_ref()),
            s.mixed.ptr(),
        )?;
        g.residual_norm(
            m.x.ptr(),
            Some(s.mixed.ptr()),
            None,
            1.0,
            Some(&mtp.head_norm),
            m.h.ptr(),
            t,
            dim,
            eps,
        )
        .map_err(e)
    }

    /// After the model read `tokens` (their normed hidden states in `s.h`'s rows): the
    /// multi-token-prediction layer's entries for them. Guessed entries go first; then
    /// each hidden state but the last pairs with the token after it (the waiting one with
    /// the first token), and the last waits for the token picked next.
    fn mtp_pairs(&self, s: &mut Qwen35CudaSession, tokens: &[u32]) -> Result<()> {
        let Some(mut m) = s.mtp.take() else {
            return Ok(());
        };
        let r = self.mtp_pairs_with(s, &mut m, tokens);
        s.mtp = Some(m);
        r
    }

    fn mtp_pairs_with(
        &self,
        s: &Qwen35CudaSession,
        m: &mut MtpSession,
        tokens: &[u32],
    ) -> Result<()> {
        let c = &self.config;
        let g = &self.gpu;
        let e = gpu_error;
        let (dim, t) = (c.dim, tokens.len());
        m.len -= m.chained;
        m.chained = 0;
        // The pairs: (pending, tokens[0]) if one waits, then (h[i], tokens[i + 1]).
        let from = usize::from(!m.has_pending);
        let pairs = t - from;
        if pairs > 0 {
            let mut at = 0;
            if m.has_pending {
                g.copy(m.hid.ptr(), m.pending.ptr(), dim * 4).map_err(e)?;
                at = 1;
            }
            if t > 1 {
                g.copy(row(&m.hid, at, dim), s.h.ptr(), (t - 1) * dim * 4)
                    .map_err(e)?;
            }
            for (i, &token) in tokens[from..].iter().enumerate() {
                self.embedding
                    .decode_row(token as usize, &mut m.host[i * dim..(i + 1) * dim]);
            }
            m.emb.write_f32(&m.host[..pairs * dim]).map_err(e)?;
            m.pos.write_i32(&[m.len as i32]).map_err(e)?;
            self.queue_mtp(s, m, pairs)?;
            m.len += pairs;
        }
        g.copy(m.pending_before.ptr(), m.pending.ptr(), dim * 4)
            .map_err(e)?;
        m.had_pending = m.has_pending;
        g.copy(m.pending.ptr(), row(&s.h, t - 1, dim), dim * 4)
            .map_err(e)?;
        m.has_pending = true;
        Ok(())
    }

    /// Up to `k` guesses at the tokens after `first`, the token the model picked after
    /// what `s` read (not read yet): the first from the model's own hidden state, the rest
    /// each from the layer's output for the one before (greedy). None without a hidden
    /// state waiting (a session not kept up with [`Qwen35Cuda::use_mtp`], or straight after
    /// a guess).
    pub fn mtp_draft(&self, s: &mut Qwen35CudaSession, first: u32, k: usize) -> Result<Vec<u32>> {
        let Some(mut m) = s.mtp.take() else {
            return Ok(Vec::new());
        };
        let r = self.mtp_draft_with(s, &mut m, first, k);
        s.mtp = Some(m);
        r
    }

    fn mtp_draft_with(
        &self,
        s: &mut Qwen35CudaSession,
        m: &mut MtpSession,
        first: u32,
        k: usize,
    ) -> Result<Vec<u32>> {
        let c = &self.config;
        let g = &self.gpu;
        let e = gpu_error;
        let dim = c.dim;
        if !m.has_pending || k == 0 || s.len + k + 1 > s.capacity {
            return Ok(Vec::new());
        }
        m.len -= m.chained;
        m.chained = 0;
        let mut guesses = Vec::with_capacity(k);
        let mut token = first;
        for step in 0..k {
            // The hidden state: the model's (waiting) for the first, the layer's after.
            if step == 0 {
                g.copy(m.hid.ptr(), m.pending.ptr(), dim * 4).map_err(e)?;
            } else {
                g.copy(m.hid.ptr(), m.h.ptr(), dim * 4).map_err(e)?;
            }
            self.embedding
                .decode_row(token as usize, &mut m.host[..dim]);
            m.emb.write_f32(&m.host[..dim]).map_err(e)?;
            m.pos.write_i32(&[m.len as i32]).map_err(e)?;
            self.queue_mtp(s, m, 1)?;
            m.len += 1;
            if step == 0 {
                // A real entry: the model's hidden state and the token it picked.
                m.has_pending = false;
            } else {
                m.chained += 1;
            }
            let out = s.logits_gpu.rows(g, 1, c.vocab)?;
            self.output
                .mul(&Input::new(g, m.h.ptr(), 1, s.q8.as_ref()), out.ptr())?;
            out.read_f32(&mut m.logits).map_err(e)?;
            token = m
                .logits
                .iter()
                .enumerate()
                .fold((0, f32::NEG_INFINITY), |best, (i, &v)| {
                    if v > best.1 { (i, v) } else { best }
                })
                .0 as u32;
            guesses.push(token);
        }
        Ok(guesses)
    }

    /// Whether [`Qwen35Cuda::keep_steps`] and [`Qwen35Cuda::rewind`] work here: on the
    /// Mac, and on CUDA for states the one-token DeltaNet kernel holds in registers (up to
    /// 128 wide, every Qwen 3.5 so far).
    pub fn can_rewind(&self) -> bool {
        cfg!(target_os = "macos") || (self.config.state <= 128 && self.config.state.is_multiple_of(8))
    }

    /// The GPU memory [`Qwen35Cuda::keep_steps`] takes for a session (Qwen 3.5 4B: 0.3 GB,
    /// 9B: 0.4 GB).
    pub fn steps_bytes(&self) -> usize {
        let c = &self.config;
        let channels = 2 * c.k_heads * c.state + c.v_heads * c.state;
        let state = c.v_heads * c.state * c.state;
        let layers = (0..c.layers).filter(|&l| !c.is_full_attention(l)).count();
        layers * (STEP_TOKENS * state + (c.conv_kernel - 1 + STEP_TOKENS) * channels) * 4
    }

    /// From now on, batches of up to [`STEP_TOKENS`] read into `s` keep each DeltaNet
    /// layer's steps, so [`Qwen35Cuda::rewind`] can undo part of them (a draft's rejected
    /// guesses): for each layer, `STEP_TOKENS` states (Qwen 3.8 27B: 48 × 25 MB) written
    /// as the batch goes, about a millisecond a round there. Allocated once.
    pub fn keep_steps(&self, s: &mut Qwen35CudaSession) -> Result<()> {
        assert!(self.can_rewind(), "this model's DeltaNet steps can't be kept");
        if s.steps.is_some() {
            return Ok(());
        }
        // Short batches are queued with their steps from here on: recorded again.
        s.few_graphs.clear();
        let c = &self.config;
        let g = &self.gpu;
        let channels = 2 * c.k_heads * c.state + c.v_heads * c.state;
        let state = c.v_heads * c.state * c.state;
        let steps = (0..c.layers)
            .filter(|&l| !c.is_full_attention(l))
            .map(|_| {
                Ok((
                    g.alloc(STEP_TOKENS * state * 4).map_err(gpu_error)?,
                    g.alloc((c.conv_kernel - 1 + STEP_TOKENS) * channels * 4)
                        .map_err(gpu_error)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        s.steps = Some(steps);
        Ok(())
    }

    /// Takes `s` back to its first `len` tokens, when they end inside the last batch read
    /// with its steps kept (or `len` is where it is): its DeltaNet layers as they were
    /// after token `len - 1`, and the attention layers' later positions dropped. False,
    /// having done nothing, otherwise.
    pub fn rewind(&self, s: &mut Qwen35CudaSession, len: usize) -> Result<bool> {
        if len == s.len {
            return Ok(true);
        }
        let (Some((start, t)), Some(steps)) = (s.stepped, s.steps.as_ref()) else {
            return Ok(false);
        };
        if len < start || len > start + t {
            return Ok(false);
        }
        let c = &self.config;
        let g = &self.gpu;
        let m = len - start;
        let channels = 2 * c.k_heads * c.state + c.v_heads * c.state;
        let state_bytes = c.v_heads * c.state * c.state * 4;
        let history_bytes = (c.conv_kernel - 1) * channels * 4;
        let deltas = s.memory.iter().filter_map(|mem| match mem {
            Memory::DeltaNet { history, state } => Some((history, state)),
            Memory::Attention { .. } => None,
        });
        for ((history, state), (states, conv)) in deltas.zip(steps) {
            g.copy(
                state.ptr(),
                states.ptr() + (m * state_bytes) as u64,
                state_bytes,
            )
            .map_err(gpu_error)?;
            g.copy(
                history.ptr(),
                conv.ptr() + (m * channels * 4) as u64,
                history_bytes,
            )
            .map_err(gpu_error)?;
        }
        s.len = len;
        s.stepped = Some((start, m));
        if let Some(mtp) = s.mtp.as_mut() {
            // Entries for the tokens kept, and the hidden state after the last of them
            // waiting (the batch's row for it, or what waited before the batch).
            mtp.len -= mtp.chained;
            mtp.chained = 0;
            if m > 0 {
                g.copy(mtp.pending.ptr(), row(&s.h, m - 1, c.dim), c.dim * 4)
                    .map_err(gpu_error)?;
                mtp.has_pending = true;
            } else {
                g.copy(mtp.pending.ptr(), mtp.pending_before.ptr(), c.dim * 4)
                    .map_err(gpu_error)?;
                mtp.has_pending = mtp.had_pending;
            }
            // An entry for each token but the last when a hidden state waits for its pair.
            let entries = if mtp.has_pending {
                len.saturating_sub(1)
            } else {
                len
            };
            mtp.len = mtp.len.min(entries);
        }
        Ok(true)
    }

    /// Queues every layer for the `t` tokens in `s.x` (at positions `*s.pos` on),
    /// leaving the final norm of each token in `outputs` in `s.h` (`Outputs::Last`: in
    /// row 0).
    fn queue_layers(
        &self,
        s: &Qwen35CudaSession,
        t: usize,
        outputs: Outputs,
        keep_steps: bool,
    ) -> Result<()> {
        let c = &self.config;
        let g = &self.gpu;
        let e = gpu_error;
        let mut t = t;
        // The last layer's mixer runs for every token (it leaves keys and values, or the
        // DeltaNet state, for later ones); the rest of it only makes outputs, so from its
        // output projection on it runs for the last token alone, or not at all. `mixed`
        // is the mixer's output, rows of `width`.
        let last_layer = self.layers.len() - 1;
        let cut = |l: usize, t: &mut usize, mixed: &Buffer, width: usize| -> Result<bool> {
            if l != last_layer || outputs == Outputs::All {
                return Ok(true);
            }
            if outputs == Outputs::None {
                return Ok(false);
            }
            if *t > 1 {
                g.copy(mixed.ptr(), row(mixed, *t - 1, width), width * 4)
                    .map_err(e)?;
                g.copy(s.x.ptr(), row(&s.x, *t - 1, c.dim), c.dim * 4)
                    .map_err(e)?;
            }
            *t = 1;
            Ok(true)
        };
        let (hd, n) = (c.head_dim, c.state);
        let kv_dim = c.kv_heads * hd;
        let key_dim = c.k_heads * n;
        let channels = 2 * key_dim + c.v_heads * n;
        g.residual_norm(
            s.x.ptr(),
            None,
            None,
            1.0,
            Some(&self.layers[0].attn_norm),
            s.h.ptr(),
            t,
            c.dim,
            c.norm_eps,
        )
        .map_err(e)?;
        let mut delta_layer = 0;
        for (l, layer) in self.layers.iter().enumerate() {
            match (&layer.mixer, &s.memory[l]) {
                (Mixer::Attention(a), Memory::Attention { keys, values }) => {
                    let h = Input::new(g, s.h.ptr(), t, s.q8.as_ref());
                    a.q.mul(&h, s.query_gate.ptr())?;
                    a.k.mul(&h, s.k_new.ptr())?;
                    a.v.mul(&h, s.v_new.ptr())?;
                    g.qkv_prep(
                        &QkvPrep {
                            q: s.query_gate.ptr(),
                            q_stride: 2 * c.heads * hd,
                            q_head_stride: 2 * hd,
                            q_norm: Some(a.q_norm.ptr()),
                            q_out: s.q.ptr(),
                            k: s.k_new.ptr(),
                            k_norm: Some(a.k_norm.ptr()),
                            v: s.v_new.ptr(),
                            v_norm: false,
                            cache: Some((keys.ptr(), values.ptr())),
                            heads: c.heads,
                            kv_heads: c.kv_heads,
                            head_dim: hd,
                            rope_dims: c.rope_dims,
                            rope_base: c.rope_base,
                            rope_factors: None,
                            neox: true,
                            pos: s.pos.ptr(),
                            eps: c.norm_eps,
                            half_cache: s.half_cache,
                            ring: 0,
                        },
                        t,
                    )
                    .map_err(e)?;
                    g.attention(
                        s.half_cache,
                        s.q.ptr(),
                        keys.ptr(),
                        values.ptr(),
                        s.attention.ptr(),
                        s.pos.ptr(),
                        0,
                        c.heads,
                        t,
                        hd,
                        kv_dim,
                        c.heads / c.kv_heads,
                        1.0 / (hd as f32).sqrt(),
                    )
                    .map_err(e)?;
                    g.apply_gate(s.attention.ptr(), s.query_gate.ptr(), t * c.heads, hd)
                        .map_err(e)?;
                    if !cut(l, &mut t, &s.attention, c.heads * hd)? {
                        break;
                    }
                    a.o.mul(
                        &Input::new(g, s.attention.ptr(), t, s.q8.as_ref()),
                        s.mixed.ptr(),
                    )?;
                }
                (Mixer::DeltaNet(d), Memory::DeltaNet { history, state }) => {
                    let h = Input::new(g, s.h.ptr(), t, s.q8.as_ref());
                    d.qkv.mul(&h, s.qkv.ptr())?;
                    d.gate.mul(&h, s.z.ptr())?;
                    d.beta.mul(&h, s.beta.ptr())?;
                    d.alpha.mul(&h, s.alpha.ptr())?;
                    // The convolution's history, then the batch's inputs after it: any
                    // number of them kept, the last conv_kernel - 1 rows are the history.
                    let steps = match (&s.steps, keep_steps) {
                        (Some(steps), true) => {
                            let (states, conv) = &steps[delta_layer];
                            let history_bytes = (c.conv_kernel - 1) * channels * 4;
                            g.copy(conv.ptr(), history.ptr(), history_bytes)
                                .map_err(e)?;
                            g.copy(
                                conv.ptr() + history_bytes as u64,
                                s.qkv.ptr(),
                                t * channels * 4,
                            )
                            .map_err(e)?;
                            states.ptr()
                        }
                        _ => 0,
                    };
                    delta_layer += 1;
                    g.conv_silu(
                        s.qkv.ptr(),
                        history.ptr(),
                        d.conv.ptr(),
                        s.conv.ptr(),
                        channels,
                        c.conv_kernel,
                        t,
                    )
                    .map_err(e)?;
                    g.delta_net(
                        &cuda::DeltaNet {
                            conv: s.conv.ptr(),
                            channels,
                            key_dim,
                            beta_raw: s.beta.ptr(),
                            alpha_raw: s.alpha.ptr(),
                            dt_bias: d.dt_bias.ptr(),
                            a: d.a.ptr(),
                            state: state.ptr(),
                            z: s.z.ptr(),
                            norm: d.norm.ptr(),
                            out: s.heads_out.ptr(),
                            v_heads: c.v_heads,
                            n,
                            k_heads: c.k_heads,
                            eps: c.norm_eps,
                            scratch: s.delta_scratch.ptr(),
                            steps,
                        },
                        t,
                    )
                    .map_err(e)?;
                    if !cut(l, &mut t, &s.heads_out, c.v_heads * n)? {
                        break;
                    }
                    d.out.mul(
                        &Input::new(g, s.heads_out.ptr(), t, s.q8.as_ref()),
                        s.mixed.ptr(),
                    )?;
                }
                _ => unreachable!("each layer's memory matches its kind"),
            }
            g.residual_norm(
                s.x.ptr(),
                Some(s.mixed.ptr()),
                None,
                1.0,
                Some(&layer.ffn_norm),
                s.h.ptr(),
                t,
                c.dim,
                c.norm_eps,
            )
            .map_err(e)?;

            // One token on the Mac: gate, up and the activation in one dispatch.
            #[cfg(target_os = "macos")]
            let fused = t == 1
                && layer
                    .gate
                    .mul_glu(g, &layer.up, s.h.ptr(), s.gate.ptr(), false)?;
            #[cfg(not(target_os = "macos"))]
            let fused = false;
            if !fused {
                let h = Input::new(g, s.h.ptr(), t, s.q8.as_ref());
                layer.gate.mul(&h, s.gate.ptr())?;
                layer.up.mul(&h, s.up.ptr())?;
                g.swiglu(s.gate.ptr(), s.up.ptr(), c.ffn_dim, c.ffn_dim, t)
                    .map_err(e)?;
            }
            layer.down.mul(
                &Input::new(g, s.gate.ptr(), t, s.q8.as_ref()),
                s.mixed.ptr(),
            )?;
            let next = match self.layers.get(l + 1) {
                Some(next) => &next.attn_norm,
                None => &self.output_norm,
            };
            g.residual_norm(
                s.x.ptr(),
                Some(s.mixed.ptr()),
                None,
                1.0,
                Some(next),
                s.h.ptr(),
                t,
                c.dim,
                c.norm_eps,
            )
            .map_err(e)?;
        }
        Ok(())
    }

    /// Runs `token` at the session's next position, replaying the recorded layers.
    pub fn forward<'s>(
        &self,
        s: &'s mut Qwen35CudaSession,
        token: u32,
        want_logits: bool,
    ) -> Result<Option<&'s [f32]>> {
        let c = &self.config;
        assert!(
            s.len < s.capacity,
            "the session is full ({} positions)",
            s.capacity
        );
        assert!(
            (token as usize) < c.vocab,
            "token {token} is past the vocabulary"
        );
        let e = gpu_error;
        self.embedding
            .decode_row(token as usize, &mut s.embedding[..c.dim]);
        s.x.write_f32(&s.embedding[..c.dim]).map_err(e)?;
        s.pos.write_i32(&[s.len as i32]).map_err(e)?;
        if s.graph.is_none() {
            let graph = record(&self.gpu, || self.queue_layers(s, 1, Outputs::All, false))?;
            s.graph = Some(graph);
        }
        s.graph.as_ref().unwrap().launch().map_err(e)?;
        s.len += 1;
        s.stepped = None;
        if want_logits {
            let logits = s.logits_gpu.rows(&self.gpu, 1, c.vocab)?;
            self.output.mul(
                &Input::new(&self.gpu, s.h.ptr(), 1, s.q8.as_ref()),
                logits.ptr(),
            )?;
            s.logits.resize(c.vocab, 0.0);
            logits.read_f32(&mut s.logits).map_err(e)?;
        }
        self.mtp_pairs(s, &[token])?;
        Ok(want_logits.then_some(&s.logits[..]))
    }

    /// Runs `tokens` at the session's next positions, a batch at a time, and hands back
    /// the logits asked for: the last token's, or a row for every token.
    pub fn forward_batch<'s>(
        &self,
        s: &'s mut Qwen35CudaSession,
        tokens: &[u32],
        logits: Logits,
    ) -> Result<Option<&'s [f32]>> {
        let c = &self.config;
        assert!(
            s.len + tokens.len() <= s.capacity,
            "{} more tokens don't fit in the session ({} of {} positions used)",
            tokens.len(),
            s.len,
            s.capacity
        );
        let e = gpu_error;
        if logits == Logits::All {
            s.logits.resize(tokens.len() * c.vocab, 0.0);
        }
        let batch = s.batch;
        // A short batch (a draft's round) keeps its DeltaNet steps, if the session does.
        let keep = s.steps.is_some() && tokens.len() <= STEP_TOKENS.min(batch);
        s.stepped = keep.then_some((s.len, tokens.len()));
        for (b, chunk) in tokens.chunks(batch).enumerate() {
            let t = chunk.len();
            for (i, &token) in chunk.iter().enumerate() {
                assert!(
                    (token as usize) < c.vocab,
                    "token {token} is past the vocabulary"
                );
                self.embedding
                    .decode_row(token as usize, &mut s.embedding[i * c.dim..(i + 1) * c.dim]);
            }
            s.x.write_f32(&s.embedding[..t * c.dim]).map_err(e)?;
            s.pos.write_i32(&[s.len as i32]).map_err(e)?;
            let last_chunk = b * batch + t == tokens.len();
            // The multi-token-prediction layer wants every token's hidden state.
            let outputs = match logits {
                _ if s.mtp.is_some() => Outputs::All,
                Logits::All => Outputs::All,
                Logits::Last if last_chunk => Outputs::Last,
                _ => Outputs::None,
            };
            // A short batch (a draft's guesses, a tool's result) replays a graph recorded
            // for its length, as one token does: launching the kernels one by one costs
            // more than running them.
            #[cfg(not(target_os = "macos"))]
            let replayed = if t <= super::gpu::GRAPH_TOKENS {
                let mode = match outputs {
                    Outputs::All => 0,
                    Outputs::Last => 1,
                    Outputs::None => 2,
                };
                let slot = super::gpu::graph_slot(t, mode);
                if s.few_graphs.len() <= slot {
                    s.few_graphs.resize_with(slot + 1, || None);
                }
                if s.few_graphs[slot].is_none() {
                    let graph = record(&self.gpu, || self.queue_layers(s, t, outputs, keep))?;
                    s.few_graphs[slot] = Some(graph);
                }
                let graph = s.few_graphs[slot].as_ref().unwrap();
                graph.launch().map_err(e)?;
                true
            } else {
                false
            };
            #[cfg(target_os = "macos")]
            let replayed = false;
            if !replayed {
                self.queue_layers(s, t, outputs, keep)?;
            }
            s.len += t;
            // Row of the last token's hidden state: every row was kept, or only it in row 0.
            let last_row = if outputs == Outputs::All { t - 1 } else { 0 };
            match logits {
                Logits::All => {
                    let out = s
                        .logits_gpu
                        .rows(&self.gpu, batch.min(tokens.len()), c.vocab)?;
                    self.output.mul(
                        &Input::new(&self.gpu, s.h.ptr(), t, s.q8.as_ref()),
                        out.ptr(),
                    )?;
                    let start = b * batch * c.vocab;
                    out.read_f32(&mut s.logits[start..start + t * c.vocab])
                        .map_err(e)?;
                }
                Logits::Last if last_chunk => {
                    let out = s.logits_gpu.rows(&self.gpu, 1, c.vocab)?;
                    self.output.mul(
                        &Input::new(&self.gpu, row(&s.h, last_row, c.dim), 1, s.q8.as_ref()),
                        out.ptr(),
                    )?;
                    s.logits.resize(c.vocab, 0.0);
                    out.read_f32(&mut s.logits).map_err(e)?;
                }
                _ => {}
            }
            self.mtp_pairs(s, chunk)?;
        }
        Ok(match logits {
            Logits::None => None,
            _ => Some(&s.logits),
        })
    }
}

/// Layer `i` of `file`: a full-attention layer (`attention`) or a DeltaNet one.
fn load_layer(file: &GgufFile, gpu: &Gpu, c: &Config, i: usize, attention: bool) -> Result<Layer> {
    let dim = c.dim;
    let upload = |m: Matrix| GpuMatrix::upload(gpu, m);
    let up = |v: Vec<f32>| gpu.upload_f32(&v).map_err(gpu_error);
    let conv_dim = 2 * c.k_heads * c.state + c.v_heads * c.state;
    let value_dim = c.v_heads * c.state;
    let name = |part: &str| format!("blk.{i}.{part}.weight");
    let mixer = if attention {
        let (q_dim, kv_dim) = (c.heads * c.head_dim, c.kv_heads * c.head_dim);
        Mixer::Attention(Attention {
            q: upload(matrix(file, &name("attn_q"), dim, 2 * q_dim)?)?,
            k: upload(matrix(file, &name("attn_k"), dim, kv_dim)?)?,
            v: upload(matrix(file, &name("attn_v"), dim, kv_dim)?)?,
            o: upload(matrix(file, &name("attn_output"), q_dim, dim)?)?,
            q_norm: up(vector(file, &name("attn_q_norm"), c.head_dim)?)?,
            k_norm: up(vector(file, &name("attn_k_norm"), c.head_dim)?)?,
        })
    } else {
        let conv_matrix = matrix(file, &name("ssm_conv1d"), c.conv_kernel, conv_dim)?;
        let mut conv = vec![0.0; c.conv_kernel * conv_dim];
        for (ch, row) in conv.chunks_exact_mut(c.conv_kernel).enumerate() {
            conv_matrix.decode_row(ch, row);
        }
        Mixer::DeltaNet(DeltaNet {
            qkv: upload(matrix(file, &name("attn_qkv"), dim, conv_dim)?)?,
            gate: upload(matrix(file, &name("attn_gate"), dim, value_dim)?)?,
            beta: upload(matrix(file, &name("ssm_beta"), dim, c.v_heads)?)?,
            alpha: upload(matrix(file, &name("ssm_alpha"), dim, c.v_heads)?)?,
            dt_bias: up(vector(file, &format!("blk.{i}.ssm_dt.bias"), c.v_heads)?)?,
            a: up(vector(file, &format!("blk.{i}.ssm_a"), c.v_heads)?)?,
            conv: up(conv)?,
            norm: up(vector(file, &name("ssm_norm"), c.state)?)?,
            out: upload(matrix(file, &name("ssm_out"), value_dim, dim)?)?,
        })
    };
    Ok(Layer {
        attn_norm: up(vector(file, &name("attn_norm"), dim)?)?,
        mixer,
        ffn_norm: up(vector(file, &name("post_attention_norm"), dim)?)?,
        gate: upload(matrix(file, &name("ffn_gate"), dim, c.ffn_dim)?)?,
        up: upload(matrix(file, &name("ffn_up"), dim, c.ffn_dim)?)?,
        down: upload(matrix(file, &name("ffn_down"), c.ffn_dim, dim)?)?,
    })
}

/// The multi-token-prediction layer, if the file has one (it follows the model's layers).
fn load_mtp(file: &GgufFile, gpu: &Gpu, c: &Config) -> Result<Option<Mtp>> {
    let i = c.layers;
    let name = |part: &str| format!("blk.{i}.nextn.{part}.weight");
    if file.tensor(&name("eh_proj")).is_none() {
        return Ok(None);
    }
    let dim = c.dim;
    let eh = matrix(file, &name("eh_proj"), 2 * dim, dim)?;
    // Each row is [embedding's dim | hidden state's dim] numbers, whole blocks each.
    let half = eh.row_bytes() / 2;
    if !dim.is_multiple_of(eh.kind.block_len() as usize) {
        return Ok(None);
    }
    let split = |second: bool| -> Result<GpuMatrix> {
        let bytes: Vec<u8> = (0..dim)
            .flat_map(|r| {
                let row = eh.row(r);
                if second { &row[half..] } else { &row[..half] }
            })
            .copied()
            .collect();
        GpuMatrix::upload(
            gpu,
            Matrix {
                kind: eh.kind,
                rows: dim,
                cols: dim,
                data: &bytes,
            },
        )
    };
    let up = |v: Vec<f32>| gpu.upload_f32(&v).map_err(gpu_error);
    Ok(Some(Mtp {
        e_proj: split(false)?,
        h_proj: split(true)?,
        enorm: up(vector(file, &name("enorm"), dim)?)?,
        hnorm: up(vector(file, &name("hnorm"), dim)?)?,
        head_norm: up(vector(file, &name("shared_head_norm"), dim)?)?,
        layer: load_layer(file, gpu, c, i, true)?,
    }))
}
