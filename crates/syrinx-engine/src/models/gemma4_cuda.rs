//! Gemma 4 on an NVIDIA GPU: the same steps as `gemma4.rs`. Every weight is in GPU
//! memory except the two embedding tables, which are only looked up: each token's rows
//! are decoded on the CPU from the mapped file and uploaded (1536 + 8960 numbers for
//! E2B), which keeps the 1.9 GB per-layer table out of GPU memory. Prompts go through
//! in batches and generated tokens replay a recorded graph, as in `llama_cuda.rs`.

use super::gemma4::Config;
use super::gpu::{
    GpuMatrix, Input, LogitsBuffer, Positions, gpu_error, load_state, q8_room, record, row,
    save_state,
};
use super::{
    DEFAULT_BATCH, Error, Logits, Precision, Result, SavedState, matrices, matrix, vector,
};
use crate::GgufFile;
use crate::cpu::Matrix;
use crate::cuda::{Buffer, CuDevicePtr, Gpu, Graph, Q8, QkvPrep};

/// Google's draft model for Gemma 4, run on the main model's state.
#[path = "gemma4_assistant.rs"]
pub mod assistant;

struct Layer {
    attn_norm: Buffer,
    q: GpuMatrix,
    k: Option<GpuMatrix>,
    v: Option<GpuMatrix>,
    q_norm: Buffer,
    k_norm: Option<Buffer>,
    o: GpuMatrix,
    post_attention_norm: Buffer,
    ffn_norm: Buffer,
    gate: GpuMatrix,
    up: GpuMatrix,
    down: GpuMatrix,
    post_ffw_norm: Buffer,
    /// The per-layer input's gate, projection and norm (models with them).
    per_layer: Option<(GpuMatrix, GpuMatrix, Buffer)>,
    /// For 8-bit mode, where a file keeps the gate and projection unquantized (the Q6_K
    /// files: f32, 2.6 MB each on E4B, so a token read 220 MB of them at the memory's speed
    /// and a prompt took the f32 tile kernel for both): the gate rounded to Q8_0 and the
    /// projection to f16. The projection in Q8_0 too read prompts 2.5% faster again but
    /// moved the logits half as far again from exact (E4B: rms change in log p 0.100 ->
    /// 0.138, the same top token 98.1% -> 97.4%); the gate alone leaves them where they were.
    per_layer_q8: Option<(GpuMatrix, GpuMatrix)>,
    output_scale: f32,
    /// Muse Glimmer: the gate on each head's output, sigmoid(attn_gate h).
    attn_gate: Option<GpuMatrix>,
    /// The layer's experts (a mixture-of-experts model; the Mac only so far).
    experts: Option<ExpertLayer>,
}

/// A layer's experts and their router (`gemma4.rs` has the arithmetic).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
struct ExpertLayer {
    router: GpuMatrix,
    /// The file's `ffn_gate_inp.scale` over sqrt(dim): the router's input is the layer's,
    /// RMS-normed, times this.
    router_weight: Buffer,
    /// Every expert's gate and up rows (2 × width an expert), and every expert's down rows.
    gate_up: GpuMatrix,
    down: GpuMatrix,
    scale: Buffer,
    pre_norm: Buffer,
    dense_post_norm: Buffer,
    post_norm: Buffer,
}

/// From this many tokens a batch's experts are run grouped by expert (`Gpu::moe_matmul`).
/// Measured on Gemma 4 26B-A4B (M5 Max), grouped against a slot at a time: 56 tokens 254
/// against 240 ms, 103 tokens 273 against 312, 393 tokens 388 against 759.
#[cfg(target_os = "macos")]
const MOE_GROUPED_FROM: usize = 80;

/// A session's room for a batch's experts.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
struct ExpertBuffers {
    /// A token's score for each expert, then its chosen experts and their shares.
    scores: Buffer,
    idx: Buffer,
    share: Buffer,
    /// A slot's gate and up products (its activation, after), and its down product.
    gate_up: Buffer,
    out: Buffer,
    /// A token's experts' outputs added up.
    sum: Buffer,
    /// A batch's slots in order of their expert, and that order cut into tiles
    /// (`Gpu::moe_group`).
    order: Buffer,
    tiles: Buffer,
    /// Where each slot went in `order`, and (the default mode, where the GPU has neural
    /// accelerators) the slots' inputs in that order, in f16 (`Gpu::moe_gather_half`).
    place: Buffer,
    sorted: Option<Buffer>,
}

/// The per-layer inputs (E2B, E4B): their embedding, projection and the projection's norm.
struct PerLayerInputs<'a> {
    embedding: Matrix<'a>,
    model_proj: GpuMatrix,
    /// The same rounded to Q8_0, for 8-bit mode (it's F16 in Gemma's files).
    model_proj_q8: Option<GpuMatrix>,
    /// per_layer_proj_norm / sqrt(2), as llama.cpp folds it.
    proj_norm: Buffer,
}

pub struct Gemma4Cuda<'a> {
    pub config: Config,
    gpu: Gpu,
    precision: Precision,
    embedding: Matrix<'a>,
    output: GpuMatrix,
    per_layer_inputs: Option<PerLayerInputs<'a>>,
    rope_factors: Option<Buffer>,
    output_norm: Buffer,
    layers: Vec<Layer>,
    /// A draft model to guess with (`load_assistant`).
    assistant: Option<assistant::Assistant>,
}

/// One conversation's state on the GPU. Use it only with the model that made it.
pub struct Gemma4CudaSession {
    keys: Vec<Option<Buffer>>,
    values: Vec<Option<Buffer>>,
    /// Each layer's sliding window (`None` for a global layer).
    windows: Vec<Option<usize>>,
    /// Each layer's ring: on the Mac a sliding-window layer keeps only its window and a
    /// batch's new positions, position p at row p mod ring (0: every position, in order).
    rings: Vec<usize>,
    /// The furthest position written since the session was last emptied: a ring's rows
    /// before it may hold later positions than a rewind wants.
    high: usize,
    /// Rewinds can't go below this (but to 0): an imported state brought only the window.
    floor: usize,
    len: usize,
    capacity: usize,
    batch: usize,
    embedding: Vec<f32>,
    per_layer_host: Vec<f32>,
    pos: Buffer,
    /// The last token's position, for running the shared-KV layers on it alone.
    pos_last: Buffer,
    // A row per token of the batch:
    x: Buffer,
    h: Buffer,
    projected: Buffer,
    per_layer: Buffer,
    per_layer_proj: Buffer,
    q: Buffer,
    k_new: Buffer,
    v_new: Buffer,
    attention: Buffer,
    gate: Buffer,
    up: Buffer,
    small: Buffer,
    /// Room for a batch's experts (a mixture-of-experts model).
    experts: Option<ExpertBuffers>,
    logits_gpu: LogitsBuffer,
    /// Whether a batch's logits are left on the GPU (`forward_batch_held`).
    hold: bool,
    logits: Vec<f32>,
    /// The batch's activations in 8 bits, unless the model is exact.
    q8: Option<Q8>,
    /// A feed-forward's middle in 8 bits, written by the kernel that runs its first two
    /// products (`GpuMatrix::mul_ffn`; CUDA).
    q8_ffn: Option<Q8>,
    /// Keys and values in f16 (8-bit mode) rather than f32.
    half_cache: bool,
    graph: Option<Graph>,
    /// Graphs for batches of up to `GRAPH_TOKENS` tokens, by their number and the logits
    /// wanted (`graph_slot`), each recorded the first time such a batch is read (CUDA).
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    few_graphs: Vec<Option<Graph>>,
    /// The positions whose final states (after the final norm) the last read left in
    /// `h`'s rows, from row 0: `hidden_rows` of them from `hidden_from`. The draft model
    /// starts from the last position's.
    hidden_from: usize,
    hidden_rows: usize,
    /// Room for the draft model (`use_mtp`).
    assistant: Option<assistant::Room>,
}

impl Gemma4CudaSession {
    /// Whether it has room to draft with a draft model ([`Gemma4Cuda::use_mtp`]).
    pub fn keeps_mtp(&self) -> bool {
        self.assistant.is_some()
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

    /// The GPU memory each position takes in this session's caches (keys and values): a
    /// ring's rows spread over the session's positions.
    pub fn cache_bytes_per_position(&self) -> usize {
        self.cache_buffers()
            .iter()
            .map(|(b, _, _)| b.len_bytes())
            .sum::<usize>()
            / self.capacity.max(1)
    }

    /// Whether a sliding-window layer keeps only a ring of its positions here (a session
    /// longer than the ring): then [`Self::truncate`] can't always go back.
    pub fn has_rings(&self) -> bool {
        self.rings.iter().any(|&r| r > 0)
    }

    /// Forgets every position from `len` on (nothing, if the session is shorter), so the
    /// next tokens run from there: the keys and values before it stay as they are. For
    /// continuing a chat whose new prompt starts with the tokens already read. The length
    /// kept: `len`, or 0 where the sliding-window layers no longer hold the window before
    /// `len` (a ring wrote over it, or an imported state didn't bring it).
    pub fn truncate(&mut self, len: usize) -> usize {
        let len = self.len.min(len);
        self.high = self.high.max(self.len);
        // A ring's window before `len` is intact while no position written since lies a
        // whole ring past it.
        let ring_floor = self
            .rings
            .iter()
            .zip(&self.windows)
            .filter(|(r, _)| **r > 0)
            .map(|(r, w)| self.high.saturating_sub(r - w.unwrap_or(0)))
            .max()
            .unwrap_or(0);
        if len < self.len && len < self.floor.max(ring_floor) {
            self.len = 0;
            self.high = 0;
            self.floor = 0;
        } else {
            self.len = len;
        }
        self.len
    }

    /// The keys and values of the first `len` positions (no more than the session has),
    /// copied to system memory: a sliding-window layer's only for the last positions of
    /// its window, which are all that later tokens read.
    pub fn export(&self, len: usize) -> Result<SavedState> {
        assert!(
            len <= self.len,
            "{len} positions of a session of {}",
            self.len
        );
        save_state(&self.cache_buffers(), self.capacity, len, &[])
    }

    /// Loads the first `upto` positions of a state `export` made from a session of the
    /// same model and precision (of any capacity is fine as long as it fits), in place of
    /// what this session held. Only all of it once the saved positions outran the
    /// sliding window (the older ones weren't kept).
    pub fn import(&mut self, saved: &SavedState, upto: usize) -> Result<()> {
        load_state(saved, upto, &self.cache_buffers(), self.capacity, &[])?;
        self.len = upto;
        self.hidden_rows = 0;
        self.high = upto;
        // A windowed layer brought only its window: nothing before `upto` can be rewound to.
        let trimmed = self.windows.iter().flatten().any(|w| upto > *w);
        self.floor = if trimmed { upto } else { 0 };
        Ok(())
    }

    /// Each layer's keys, then values, that has its own, with the sliding window it reads
    /// (a saved state keeps only that many of its last positions).
    fn cache_buffers(&self) -> Vec<Positions<'_>> {
        let keys = self.keys.iter().zip(&self.windows).zip(&self.rings);
        let values = self.values.iter().zip(&self.windows).zip(&self.rings);
        keys.chain(values)
            .filter_map(|((b, w), r)| b.as_ref().map(|b| (b, *w, *r)))
            .collect()
    }
}

impl<'a> Gemma4Cuda<'a> {
    pub fn load(file: &'a GgufFile, gpu: Gpu) -> Result<Self> {
        let c = Config::from_gguf(file)?;
        if c.muse && !cfg!(target_os = "macos") {
            return Err(Error::Unsupported(
                "Muse Glimmer on CUDA (its attention gate's kernel is Metal's only so far)".into(),
            ));
        }
        if c.experts.is_some() && !cfg!(target_os = "macos") {
            return Err(Error::Unsupported(
                "Gemma 4's mixture-of-experts models (26B-A4B) on CUDA (the experts' kernels \
                 are Metal's only so far)"
                    .into(),
            ));
        }
        let (dim, pl) = (c.dim, c.per_layer_dim);
        let upload = |m: Matrix| GpuMatrix::upload(&gpu, m);
        let up = |v: Vec<f32>| gpu.upload_f32(&v).map_err(gpu_error);

        let mut layers = Vec::with_capacity(c.layers);
        for i in 0..c.layers {
            let name = |part: &str| format!("blk.{i}.{part}.weight");
            let hd = c.head_dim_of(i);
            let own_kv = i < c.first_shared_layer;
            // A layer without attn_v uses its keys as values (31B's global layers).
            let kv = |part: &str| -> Result<Option<GpuMatrix>> {
                if own_kv && file.tensor(&name(part)).is_some() {
                    upload(matrix(file, &name(part), dim, c.kv_heads_of(i) * hd)?).map(Some)
                } else {
                    Ok(None)
                }
            };
            layers.push(Layer {
                attn_norm: up(vector(file, &name("attn_norm"), dim)?)?,
                q: upload(matrix(file, &name("attn_q"), dim, c.heads * hd)?)?,
                k: kv("attn_k")?,
                v: kv("attn_v")?,
                q_norm: up(vector(file, &name("attn_q_norm"), hd)?)?,
                k_norm: if own_kv {
                    Some(up(vector(file, &name("attn_k_norm"), hd)?)?)
                } else {
                    None
                },
                o: upload(matrix(file, &name("attn_output"), c.heads * hd, dim)?)?,
                post_attention_norm: up(vector(file, &name("post_attention_norm"), dim)?)?,
                ffn_norm: up(vector(file, &name("ffn_norm"), dim)?)?,
                gate: upload(matrix(file, &name("ffn_gate"), dim, c.ffn_dims[i])?)?,
                up: upload(matrix(file, &name("ffn_up"), dim, c.ffn_dims[i])?)?,
                down: upload(matrix(file, &name("ffn_down"), c.ffn_dims[i], dim)?)?,
                post_ffw_norm: up(vector(file, &name("post_ffw_norm"), dim)?)?,
                per_layer: if pl > 0 {
                    Some((
                        upload(matrix(file, &name("inp_gate"), dim, pl)?)?,
                        upload(matrix(file, &name("proj"), pl, dim)?)?,
                        up(vector(file, &name("post_norm"), dim)?)?,
                    ))
                } else {
                    None
                },
                per_layer_q8: if pl > 0 && cfg!(not(target_os = "macos")) {
                    let gate =
                        GpuMatrix::upload_q8_0(&gpu, &matrix(file, &name("inp_gate"), dim, pl)?)?;
                    let proj = GpuMatrix::upload_f16(&gpu, &matrix(file, &name("proj"), pl, dim)?)?;
                    gate.zip(proj)
                } else {
                    None
                },
                output_scale: vector(file, &name("layer_output_scale"), 1).map_or(1.0, |v| v[0]),
                experts: match c.experts {
                    Some(x) => {
                        let root = (dim as f32).sqrt();
                        let gate_up =
                            matrices(file, &name("ffn_gate_up_exps"), dim, 2 * x.width, x.count)?;
                        let down = matrices(file, &name("ffn_down_exps"), x.width, dim, x.count)?;
                        #[cfg(target_os = "macos")]
                        for m in [&gate_up, &down] {
                            if !Gpu::supports_experts(m.kind) {
                                return Err(Error::Unsupported(format!(
                                    "experts stored as {:?} on the GPU (Q6_K and Q8_0 are)",
                                    m.kind
                                )));
                            }
                        }
                        Some(ExpertLayer {
                            router: upload(matrix(file, &name("ffn_gate_inp"), dim, x.count)?)?,
                            router_weight: up(vector(
                                file,
                                &format!("blk.{i}.ffn_gate_inp.scale"),
                                dim,
                            )?
                            .iter()
                            .map(|s| s / root)
                            .collect())?,
                            gate_up: upload(gate_up)?,
                            down: upload(down)?,
                            scale: up(vector(
                                file,
                                &format!("blk.{i}.ffn_down_exps.scale"),
                                x.count,
                            )?)?,
                            pre_norm: up(vector(file, &name("pre_ffw_norm_2"), dim)?)?,
                            dense_post_norm: up(vector(file, &name("post_ffw_norm_1"), dim)?)?,
                            post_norm: up(vector(file, &name("post_ffw_norm_2"), dim)?)?,
                        })
                    }
                    None => None,
                },
                attn_gate: if c.muse {
                    Some(upload(matrix(
                        file,
                        &name("attn_gate"),
                        dim,
                        c.heads * hd,
                    )?)?)
                } else {
                    None
                },
            });
            if own_kv && layers[i].k.is_none() {
                return Err(Error::Missing(format!("tensor blk.{i}.attn_k.weight")));
            }
        }
        let separate_output = file.tensor("output.weight").is_some();
        if separate_output && !c.muse {
            return Err(Error::Unsupported(
                "Gemma 4 with a separate output matrix".into(),
            ));
        }
        let embedding = matrix(file, "token_embd.weight", dim, c.vocab)?;
        let output = if separate_output {
            matrix(file, "output.weight", dim, c.vocab)?
        } else {
            embedding
        };
        // Muse's logits are scaled before the softcap: the same as scaling the final
        // norm's weight, which they're a product with.
        let output_norm: Vec<f32> = vector(file, "output_norm.weight", dim)?
            .into_iter()
            .map(|w| w * c.logit_scale)
            .collect();
        let per_layer_inputs = if pl > 0 {
            let proj_norm: Vec<f32> = vector(file, "per_layer_proj_norm.weight", pl)?
                .into_iter()
                .map(|w| w / 2f32.sqrt())
                .collect();
            let model_proj = matrix(file, "per_layer_model_proj.weight", dim, pl * c.layers)?;
            Some(PerLayerInputs {
                embedding: matrix(file, "per_layer_token_embd.weight", pl * c.layers, c.vocab)?,
                model_proj_q8: GpuMatrix::upload_q8_0(&gpu, &model_proj)?,
                model_proj: upload(model_proj)?,
                proj_norm: up(proj_norm)?,
            })
        } else {
            None
        };
        let rope_factors = if file.tensor("rope_freqs.weight").is_some() {
            Some(up(vector(file, "rope_freqs.weight", c.head_dim / 2)?)?)
        } else {
            None
        };
        Ok(Self {
            precision: Precision::default(),
            output: upload(output)?,
            embedding,
            per_layer_inputs,
            rope_factors,
            output_norm: up(output_norm)?,
            layers,
            assistant: None,
            config: c,
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

    pub fn session(&self, capacity: usize) -> Result<Gemma4CudaSession> {
        self.session_with_batch(capacity, DEFAULT_BATCH)
    }

    /// A new, empty conversation that reads up to `batch` tokens at once.
    pub fn session_with_batch(&self, capacity: usize, batch: usize) -> Result<Gemma4CudaSession> {
        let c = &self.config;
        let batch = batch.clamp(1, capacity.max(1));
        // At least one value each: a model without per-layer inputs has none to keep.
        let f32s = |n: usize| self.gpu.alloc(n.max(1) * 4).map_err(gpu_error);
        // Keys and values: f16 in 8-bit mode (half the memory), f32 when exact.
        let cache_bytes = if self.precision == Precision::Int8 {
            2
        } else {
            4
        };
        let cache_buffer = |n: usize| self.gpu.alloc(n * cache_bytes).map_err(gpu_error);
        // A sliding-window layer keeps its window and a batch's new positions (rounded to
        // the attention kernels' longest chunk, 256), not every position: for 31B at 32K
        // tokens, 1.6 GB instead of 27. On CUDA too since 2026-11-15.
        let rings: Vec<usize> = (0..c.layers)
            .map(|l| {
                let ring = (c.sliding_window + batch).next_multiple_of(256);
                if c.is_swa[l] && ring < capacity {
                    ring
                } else {
                    0
                }
            })
            .collect();
        let cache = |l: usize| -> Result<Option<Buffer>> {
            if l < c.first_shared_layer {
                let rows = if rings[l] > 0 { rings[l] } else { capacity };
                cache_buffer(rows * c.kv_heads_of(l) * c.head_dim_of(l)).map(Some)
            } else {
                Ok(None)
            }
        };
        let mut keys = Vec::with_capacity(c.layers);
        let mut values = Vec::with_capacity(c.layers);
        for l in 0..c.layers {
            keys.push(cache(l)?);
            values.push(cache(l)?);
        }
        let widest_head = c.head_dim.max(c.head_dim_swa);
        let widest_ffn = c.ffn_dims.iter().copied().max().unwrap_or(0);
        let per_layer = c.per_layer_dim * c.layers;
        Ok(Gemma4CudaSession {
            keys,
            values,
            windows: (0..c.layers)
                .map(|l| c.is_swa[l].then_some(c.sliding_window))
                .collect(),
            rings,
            high: 0,
            floor: 0,
            len: 0,
            capacity,
            batch,
            embedding: vec![0.0; batch * c.dim],
            per_layer_host: vec![0.0; batch * per_layer],
            pos: f32s(1)?,
            pos_last: f32s(1)?,
            x: f32s(batch * c.dim)?,
            h: f32s(batch * c.dim)?,
            projected: f32s(batch * c.dim)?,
            per_layer: f32s(batch * per_layer)?,
            per_layer_proj: f32s(batch * per_layer)?,
            q: f32s(batch * c.heads * widest_head)?,
            k_new: f32s(batch * c.widest_kv_heads() * widest_head)?,
            v_new: f32s(batch * c.widest_kv_heads() * widest_head)?,
            attention: f32s(batch * c.heads * widest_head)?,
            gate: f32s(batch * widest_ffn)?,
            up: f32s(batch * widest_ffn)?,
            small: f32s(batch * c.per_layer_dim)?,
            experts: match c.experts {
                Some(x) => Some(ExpertBuffers {
                    scores: f32s(batch * x.count)?,
                    idx: f32s(batch * x.used)?,
                    share: f32s(batch * x.used)?,
                    gate_up: f32s(batch * x.used * 2 * x.width)?,
                    out: f32s(batch * x.used * c.dim)?,
                    sum: f32s(batch * c.dim)?,
                    order: f32s(batch * x.used)?,
                    place: f32s(batch * x.used)?,
                    #[cfg(target_os = "macos")]
                    sorted: if self.precision == Precision::Int8
                        && batch >= MOE_GROUPED_FROM
                        && self.gpu.moe_fast(crate::TensorType::F32)
                    {
                        Some(
                            self.gpu
                                .alloc(batch * x.used * c.dim.max(x.width) * 2)
                                .map_err(gpu_error)?,
                        )
                    } else {
                        None
                    },
                    #[cfg(not(target_os = "macos"))]
                    sorted: None,
                    #[cfg(target_os = "macos")]
                    tiles: f32s(3 * Gpu::moe_tiles(batch * x.used, x.count))?,
                    #[cfg(not(target_os = "macos"))]
                    tiles: f32s(1)?,
                }),
                None => None,
            },
            logits_gpu: LogitsBuffer::new(&self.gpu, c.vocab)?,
            hold: false,
            logits: vec![0.0; c.vocab],
            q8_ffn: if cfg!(target_os = "macos") {
                None
            } else {
                q8_room(
                    &self.gpu,
                    self.precision,
                    batch,
                    widest_ffn,
                    &self.layers[0].gate,
                )?
            },
            q8: q8_room(
                &self.gpu,
                self.precision,
                batch,
                c.dim
                    .max(c.heads * widest_head)
                    .max(widest_ffn)
                    .max(c.per_layer_dim),
                &self.layers[0].gate,
            )?,
            half_cache: self.precision == Precision::Int8,
            graph: None,
            few_graphs: Vec::new(),
            hidden_from: 0,
            hidden_rows: 0,
            assistant: None,
        })
    }

    /// Writes `tokens`' inputs (the scaled embeddings and per-layer embeddings) to the
    /// GPU, for positions from `s.len` on.
    fn upload_inputs(&self, s: &mut Gemma4CudaSession, tokens: &[u32]) -> Result<()> {
        let c = &self.config;
        let (dim, pl) = (c.dim, c.per_layer_dim * c.layers);
        let embed_scale = (dim as f32).sqrt();
        let token_scale = (c.per_layer_dim as f32).sqrt() / 2f32.sqrt();
        for (i, &token) in tokens.iter().enumerate() {
            assert!(
                (token as usize) < c.vocab,
                "token {token} is past the vocabulary"
            );
            let e = &mut s.embedding[i * dim..(i + 1) * dim];
            self.embedding.decode_row(token as usize, e);
            // Muse: RMS-normed, no weight; Gemma: times sqrt(dim).
            let scale = if c.muse {
                let mean_square = e.iter().map(|v| v * v).sum::<f32>() / dim as f32;
                1.0 / (mean_square + c.norm_eps).sqrt()
            } else {
                embed_scale
            };
            e.iter_mut().for_each(|v| *v *= scale);
            if let Some(inputs) = &self.per_layer_inputs {
                let p = &mut s.per_layer_host[i * pl..(i + 1) * pl];
                inputs.embedding.decode_row(token as usize, p);
                p.iter_mut().for_each(|v| *v *= token_scale);
            }
        }
        let t = tokens.len();
        let e = gpu_error;
        s.x.write_f32(&s.embedding[..t * dim]).map_err(e)?;
        if pl > 0 {
            s.per_layer
                .write_f32(&s.per_layer_host[..t * pl])
                .map_err(e)?;
        }
        s.pos.write_i32(&[s.len as i32]).map_err(e)
    }

    /// Queues layers `layers` for the `t` tokens in `s.x` (at positions `*pos` on),
    /// leaving each token's input to the next layer (after the last, the final norm) in
    /// `s.h`. Starting at layer 0, the per-layer inputs too.
    ///
    /// With `chats`, the tokens are one each for those conversations, in their order
    /// (`forward_each`): each row's keys and values go to its own conversation's cache at
    /// its own position, and it attends over that cache; everything else, the products
    /// that read the weights, runs for the rows together in `s`'s buffers.
    fn queue_layers(
        &self,
        s: &Gemma4CudaSession,
        t: usize,
        layers: std::ops::Range<usize>,
        pos: CuDevicePtr,
        chats: &[&Gemma4CudaSession],
    ) -> Result<()> {
        // Whose cache, from which row of the batch, how many rows, at which position.
        let parts: Vec<(&Gemma4CudaSession, usize, usize, CuDevicePtr)> = if chats.is_empty() {
            vec![(s, 0, t, pos)]
        } else {
            debug_assert_eq!(chats.len(), t);
            chats
                .iter()
                .enumerate()
                .map(|(i, chat)| (*chat, i, 1, chat.pos.ptr()))
                .collect()
        };
        let c = &self.config;
        let g = &self.gpu;
        let e = gpu_error;
        let (dim, pl, eps) = (c.dim, c.per_layer_dim, c.norm_eps);
        if layers.start == 0 {
            self.queue_inputs(s, t)?;
        }
        for l in layers {
            let layer = &self.layers[l];
            let hd = c.head_dim_of(l);
            let kv_dim = c.kv_heads_of(l) * hd;
            let (base, factors) = if c.is_swa[l] {
                (c.rope_base_swa, None)
            } else {
                (c.rope_base, self.rope_factors.as_ref().map(Buffer::ptr))
            };

            let h = Input::new(g, s.h.ptr(), t, s.q8.as_ref());
            layer.q.mul(&h, s.q.ptr())?;
            let own = match (&layer.k, &s.keys[l]) {
                (Some(k), Some(_)) => {
                    k.mul(&h, s.k_new.ptr())?;
                    match &layer.v {
                        Some(v) => v.mul(&h, s.v_new.ptr())?,
                        // Keys as values, before either is normed.
                        None => g
                            .copy(s.v_new.ptr(), s.k_new.ptr(), t * kv_dim * 4)
                            .map_err(e)?,
                    }
                    true
                }
                _ => false,
            };
            let source = c.kv_source(l);
            for &(chat, at, rows, pos) in &parts {
                let q = s.q.f32_at(at * c.heads * hd);
                g.qkv_prep(
                    &QkvPrep {
                        q,
                        q_stride: c.heads * hd,
                        q_head_stride: hd,
                        q_norm: Some(layer.q_norm.ptr()),
                        q_out: q,
                        k: s.k_new.f32_at(at * kv_dim),
                        k_norm: layer.k_norm.as_ref().map(Buffer::ptr),
                        v: s.v_new.f32_at(at * kv_dim),
                        v_norm: !c.muse,
                        cache: own.then(|| {
                            (
                                chat.keys[l].as_ref().expect("its own keys").ptr(),
                                chat.values[l].as_ref().expect("and values").ptr(),
                            )
                        }),
                        heads: c.heads,
                        kv_heads: c.kv_heads_of(l),
                        head_dim: hd,
                        // Muse's global layers have no RoPE.
                        rope_dims: if c.muse && !c.is_swa[l] { 0 } else { hd },
                        rope_base: base,
                        rope_factors: factors,
                        // Muse's converter permutes its queries and keys for RoPE on pairs of
                        // neighbours (llama.cpp's "normal"), as for Llama; Gemma's rotates halves.
                        neox: !c.muse,
                        pos,
                        eps,
                        half_cache: s.half_cache,
                        ring: chat.rings[l],
                    },
                    rows,
                )
                .map_err(e)?;

                let (keys, values) = (
                    chat.keys[source]
                        .as_ref()
                        .expect("a source layer has its own keys"),
                    chat.values[source].as_ref().expect("and values"),
                );
                g.attention_ring(
                    s.half_cache,
                    q,
                    keys.ptr(),
                    values.ptr(),
                    s.attention.f32_at(at * c.heads * hd),
                    pos,
                    if c.is_swa[l] { c.sliding_window } else { 0 },
                    chat.rings[source],
                    c.heads,
                    rows,
                    hd,
                    kv_dim,
                    c.heads / c.kv_heads_of(l),
                    // Gemma's queries and keys are normed (scale 1); Muse's queries carry
                    // their scale in q_norm.
                    if c.muse {
                        1.0 / (hd as f32).sqrt()
                    } else {
                        1.0
                    },
                )
                .map_err(e)?;
            }
            if let Some(gate) = &layer.attn_gate {
                // s.h still holds the layer's normed input; s.gate is free until the FFN.
                gate.mul(&Input::new(g, s.h.ptr(), t, s.q8.as_ref()), s.gate.ptr())?;
                #[cfg(target_os = "macos")]
                g.sigmoid_mul(s.attention.ptr(), s.gate.ptr(), t * c.heads * hd)
                    .map_err(e)?;
            }
            layer.o.mul(
                &Input::new(g, s.attention.ptr(), t, s.q8.as_ref()),
                s.projected.ptr(),
            )?;
            g.residual_norm(
                s.x.ptr(),
                Some(s.projected.ptr()),
                Some(&layer.post_attention_norm),
                1.0,
                Some(&layer.ffn_norm),
                s.h.ptr(),
                t,
                dim,
                eps,
            )
            .map_err(e)?;

            let f = c.ffn_dims[l];
            // One token on the Mac: gate, up and the activation in one dispatch.
            #[cfg(target_os = "macos")]
            let fused = t == 1
                && layer
                    .gate
                    .mul_glu(g, &layer.up, s.h.ptr(), s.gate.ptr(), !c.muse)?;
            #[cfg(not(target_os = "macos"))]
            let fused = false;
            if !fused {
                let h = Input::new(g, s.h.ptr(), t, s.q8.as_ref());
                // A batch on CUDA: the two products and the activation in one kernel.
                if !layer.down.mul_ffn(
                    &layer.gate,
                    &layer.up,
                    &h,
                    s.q8_ffn.as_ref(),
                    !c.muse,
                    s.projected.ptr(),
                )? {
                    layer.gate.mul(&h, s.gate.ptr())?;
                    layer.up.mul(&h, s.up.ptr())?;
                    debug_assert_eq!(f, layer.down.cols);
                    layer.down.mul_gated(
                        g,
                        s.gate.ptr(),
                        s.up.ptr(),
                        t,
                        s.q8.as_ref(),
                        !c.muse,
                        s.projected.ptr(),
                    )?;
                }
            } else {
                layer.down.mul(
                    &Input::new(g, s.gate.ptr(), t, s.q8.as_ref()),
                    s.projected.ptr(),
                )?;
            }
            // A mixture of experts: the dense network's output, normed, plus the experts'.
            #[cfg(target_os = "macos")]
            if let (Some(m), Some(x), Some(b)) = (&layer.experts, c.experts, &s.experts) {
                let slots = t * x.used;
                // The router scores every expert from the layer's input; the highest share
                // the token.
                g.rms_norm_rows(s.x.ptr(), m.router_weight.ptr(), s.h.ptr(), t, dim, eps)
                    .map_err(e)?;
                m.router
                    .mul(&Input::new(g, s.h.ptr(), t, None), b.scores.ptr())?;
                g.moe_pick(
                    b.scores.ptr(),
                    m.scale.ptr(),
                    b.idx.ptr(),
                    b.share.ptr(),
                    x.count,
                    x.used,
                    t,
                )
                .map_err(e)?;
                // Each chosen expert: its gate and up rows on the layer's input (normed its
                // own way), the activation, then its down rows.
                g.rms_norm_rows(s.x.ptr(), m.pre_norm.ptr(), s.h.ptr(), t, dim, eps)
                    .map_err(e)?;
                // A batch: its slots grouped by expert, so each expert's matrices are read
                // once for the tokens that chose it (a token at a time read 8 × 5.3 MB a
                // token a layer, and reading was memory-bound). A few tokens: each slot's
                // own product, the weights read as for one token.
                let grouped = t >= MOE_GROUPED_FROM;
                // On the neural accelerators (the default mode): the slots' inputs and
                // products in the order of their experts, a tile's one after another.
                let fast = b
                    .sorted
                    .as_ref()
                    .filter(|_| grouped && g.moe_fast(m.gate_up.kind) && g.moe_fast(m.down.kind));
                if grouped {
                    g.moe_group(
                        b.idx.ptr(),
                        b.order.ptr(),
                        b.tiles.ptr(),
                        b.place.ptr(),
                        slots,
                        x.count,
                        fast.is_some(),
                    )
                    .map_err(e)?;
                }
                if let Some(sorted) = fast {
                    g.moe_gather_half(
                        s.h.ptr(),
                        sorted.ptr(),
                        b.order.ptr(),
                        x.used,
                        dim,
                        dim,
                        true,
                        slots,
                    )
                    .map_err(e)?;
                    g.moe_matmul_fast(
                        m.gate_up.kind,
                        m.gate_up.data.ptr(),
                        sorted.ptr(),
                        b.gate_up.ptr(),
                        2 * x.width,
                        dim,
                        b.tiles.ptr(),
                        2 * x.width,
                        slots,
                        x.count,
                    )
                    .map_err(e)?;
                } else if grouped {
                    g.moe_matmul(
                        m.gate_up.kind,
                        m.gate_up.data.ptr(),
                        s.h.ptr(),
                        b.gate_up.ptr(),
                        2 * x.width,
                        dim,
                        b.order.ptr(),
                        b.tiles.ptr(),
                        x.used,
                        dim,
                        false,
                        2 * x.width,
                        slots,
                        x.count,
                    )
                    .map_err(e)?;
                } else {
                    g.moe_matvec(
                        m.gate_up.kind,
                        m.gate_up.data.ptr(),
                        s.h.ptr(),
                        b.gate_up.ptr(),
                        2 * x.width,
                        dim,
                        b.idx.ptr(),
                        x.used,
                        dim,
                        false,
                        2 * x.width,
                        slots,
                    )
                    .map_err(e)?;
                }
                g.moe_geglu(b.gate_up.ptr(), x.width, slots).map_err(e)?;
                if let Some(sorted) = fast {
                    g.moe_gather_half(
                        b.gate_up.ptr(),
                        sorted.ptr(),
                        b.order.ptr(),
                        x.used,
                        x.width,
                        2 * x.width,
                        false,
                        slots,
                    )
                    .map_err(e)?;
                    g.moe_matmul_fast(
                        m.down.kind,
                        m.down.data.ptr(),
                        sorted.ptr(),
                        b.out.ptr(),
                        dim,
                        x.width,
                        b.tiles.ptr(),
                        dim,
                        slots,
                        x.count,
                    )
                    .map_err(e)?;
                } else if grouped {
                    g.moe_matmul(
                        m.down.kind,
                        m.down.data.ptr(),
                        b.gate_up.ptr(),
                        b.out.ptr(),
                        dim,
                        x.width,
                        b.order.ptr(),
                        b.tiles.ptr(),
                        x.used,
                        2 * x.width,
                        true,
                        dim,
                        slots,
                        x.count,
                    )
                    .map_err(e)?;
                } else {
                    g.moe_matvec(
                        m.down.kind,
                        m.down.data.ptr(),
                        b.gate_up.ptr(),
                        b.out.ptr(),
                        dim,
                        x.width,
                        b.idx.ptr(),
                        x.used,
                        2 * x.width,
                        true,
                        dim,
                        slots,
                    )
                    .map_err(e)?;
                }
                g.moe_sum(
                    b.sum.ptr(),
                    b.out.ptr(),
                    b.share.ptr(),
                    x.used,
                    dim,
                    t,
                    fast.map(|_| b.place.ptr()),
                )
                .map_err(e)?;
                g.rms_norm_rows(b.sum.ptr(), m.post_norm.ptr(), b.sum.ptr(), t, dim, eps)
                    .map_err(e)?;
                g.rms_norm_rows(
                    s.projected.ptr(),
                    m.dense_post_norm.ptr(),
                    s.projected.ptr(),
                    t,
                    dim,
                    eps,
                )
                .map_err(e)?;
                g.add(s.projected.ptr(), b.sum.ptr(), t * dim).map_err(e)?;
            }
            let next = match self.layers.get(l + 1) {
                Some(next) => &next.attn_norm,
                None => &self.output_norm,
            };
            let Some((inp_gate, proj, post_norm)) = &layer.per_layer else {
                // No per-layer input: the layer ends here.
                g.residual_norm(
                    s.x.ptr(),
                    Some(s.projected.ptr()),
                    Some(&layer.post_ffw_norm),
                    layer.output_scale,
                    Some(next),
                    s.h.ptr(),
                    t,
                    dim,
                    eps,
                )
                .map_err(e)?;
                continue;
            };
            g.residual_norm(
                s.x.ptr(),
                Some(s.projected.ptr()),
                Some(&layer.post_ffw_norm),
                1.0,
                None,
                s.h.ptr(),
                t,
                dim,
                eps,
            )
            .map_err(e)?;
            // 8-bit mode takes the rounded copies where there are any.
            let (inp_gate, proj) = match (&layer.per_layer_q8, s.q8.as_ref()) {
                (Some((gate, proj)), Some(_)) => (gate, proj),
                _ => (inp_gate, proj),
            };

            // One token on the Mac: the product and the GELU gate in one dispatch.
            #[cfg(target_os = "macos")]
            let fused = t == 1
                && g.matvec_gelu(
                    inp_gate.kind,
                    inp_gate.data.ptr(),
                    s.x.ptr(),
                    s.small.ptr(),
                    inp_gate.rows,
                    inp_gate.cols,
                    s.per_layer.f32_at(l * pl),
                )
                .map_err(e)?;
            #[cfg(not(target_os = "macos"))]
            let fused = false;
            if !fused {
                inp_gate.mul(&Input::new(g, s.x.ptr(), t, s.q8.as_ref()), s.small.ptr())?;
                g.geglu(
                    s.small.ptr(),
                    s.per_layer.f32_at(l * pl),
                    pl,
                    c.layers * pl,
                    t,
                )
                .map_err(e)?;
            }
            proj.mul(
                &Input::new(g, s.small.ptr(), t, s.q8.as_ref()),
                s.projected.ptr(),
            )?;
            g.residual_norm(
                s.x.ptr(),
                Some(s.projected.ptr()),
                Some(post_norm),
                layer.output_scale,
                Some(next),
                s.h.ptr(),
                t,
                dim,
                eps,
            )
            .map_err(e)?;
        }
        Ok(())
    }

    /// Queues the per-layer inputs for the `t` tokens in `s.x`, and layer 0's input norm.
    fn queue_inputs(&self, s: &Gemma4CudaSession, t: usize) -> Result<()> {
        let c = &self.config;
        let g = &self.gpu;
        let e = gpu_error;
        let (dim, pl, eps) = (c.dim, c.per_layer_dim, c.norm_eps);
        if let Some(inputs) = &self.per_layer_inputs {
            let proj = match (&inputs.model_proj_q8, self.precision) {
                (Some(q8), Precision::Int8) => q8,
                _ => &inputs.model_proj,
            };
            proj.mul(
                &Input::new(g, s.x.ptr(), t, s.q8.as_ref()),
                s.per_layer_proj.ptr(),
            )?;
            g.rms_norm_rows(
                s.per_layer_proj.ptr(),
                inputs.proj_norm.ptr(),
                s.per_layer_proj.ptr(),
                t * c.layers,
                pl,
                eps * dim as f32,
            )
            .map_err(e)?;
            g.add(s.per_layer.ptr(), s.per_layer_proj.ptr(), t * c.layers * pl)
                .map_err(e)?;
        }
        g.residual_norm(
            s.x.ptr(),
            None,
            None,
            1.0,
            Some(&self.layers[0].attn_norm),
            s.h.ptr(),
            t,
            dim,
            eps,
        )
        .map_err(e)
    }

    /// The logits of the `rows` tokens in `s.h` from row `first` on, into `out`.
    fn logits_into(
        &self,
        s: &mut Gemma4CudaSession,
        first: usize,
        rows: usize,
        out_at: usize,
    ) -> Result<()> {
        let c = &self.config;
        let e = gpu_error;
        let out = s.logits_gpu.rows(&self.gpu, rows, c.vocab)?;
        self.output.mul(
            &Input::new(&self.gpu, row(&s.h, first, c.dim), rows, s.q8.as_ref()),
            out.ptr(),
        )?;
        if let Some(cap) = c.softcap {
            self.gpu
                .softcap(out.ptr(), cap, rows * c.vocab)
                .map_err(e)?;
        }
        if s.hold {
            return Ok(());
        }
        out.read_f32(&mut s.logits[out_at..out_at + rows * c.vocab])
            .map_err(e)
    }

    /// Runs `token` at the session's next position, replaying the recorded layers.
    pub fn forward<'s>(
        &self,
        s: &'s mut Gemma4CudaSession,
        token: u32,
        want_logits: bool,
    ) -> Result<Option<&'s [f32]>> {
        assert!(
            s.len < s.capacity,
            "the session is full ({} positions)",
            s.capacity
        );
        self.upload_inputs(s, &[token])?;
        if s.graph.is_none() {
            let all = 0..self.config.layers;
            let graph = record(&self.gpu, || self.queue_layers(s, 1, all, s.pos.ptr(), &[]))?;
            s.graph = Some(graph);
        }
        s.graph.as_ref().unwrap().launch().map_err(gpu_error)?;
        (s.hidden_from, s.hidden_rows) = (s.len, 1);
        s.len += 1;
        if !want_logits {
            return Ok(None);
        }
        s.logits.resize(self.config.vocab, 0.0);
        self.logits_into(s, 0, 1, 0)?;
        Ok(Some(&s.logits))
    }

    /// As [`Self::forward_batch`] for every token's logits, but they stay on the GPU:
    /// [`Self::held_logits`] fetches a row, until the session next runs a token. For a
    /// draft's guesses, checked in one pass: only the rows of the guesses taken are ever
    /// read back (a megabyte each for a 262,144-token vocabulary). `tokens` must fit one
    /// batch.
    pub fn forward_batch_held(&self, s: &mut Gemma4CudaSession, tokens: &[u32]) -> Result<()> {
        assert!(
            tokens.len() <= s.batch,
            "{} tokens held, in batches of {}",
            tokens.len(),
            s.batch
        );
        s.hold = true;
        let done = self.forward_batch(s, tokens, Logits::All).map(|_| ());
        s.hold = false;
        done
    }

    /// Row `row` of the logits [`Self::forward_batch_held`] left on the GPU, into `out`.
    pub fn held_logits(&self, s: &Gemma4CudaSession, row: usize, out: &mut [f32]) -> Result<()> {
        assert_eq!(
            out.len(),
            self.config.vocab,
            "a row of logits is the vocabulary long"
        );
        s.logits_gpu.read_row(row, out)
    }

    /// Runs `tokens` at the session's next positions, a batch at a time, and hands back
    /// the logits asked for: the last token's, or a row for every token.
    pub fn forward_batch<'s>(
        &self,
        s: &'s mut Gemma4CudaSession,
        tokens: &[u32],
        logits: Logits,
    ) -> Result<Option<&'s [f32]>> {
        let vocab = self.config.vocab;
        assert!(
            s.len + tokens.len() <= s.capacity,
            "{} more tokens don't fit in the session ({} of {} positions used)",
            tokens.len(),
            s.len,
            s.capacity
        );
        s.logits.resize(
            if logits == Logits::All && !s.hold {
                tokens.len() * vocab
            } else {
                vocab
            },
            0.0,
        );
        let batch = s.batch;
        for (b, chunk) in tokens.chunks(batch).enumerate() {
            let t = chunk.len();
            let last_chunk = b * batch + t == tokens.len();
            self.upload_inputs(s, chunk)?;
            // Where the last token is, for its pass through the shared layers.
            s.pos_last
                .write_i32(&[(s.len + t - 1) as i32])
                .map_err(gpu_error)?;
            let wanted = match logits {
                Logits::Last if !last_chunk => Logits::None,
                wanted => wanted,
            };
            // A short batch (a draft's guesses, a tool's result) replays a graph recorded
            // for its length, as one token does: launching the kernels one by one costs
            // more than running them.
            #[cfg(not(target_os = "macos"))]
            let replayed = if t <= super::gpu::GRAPH_TOKENS {
                let mode = match wanted {
                    Logits::All => 0,
                    Logits::Last => 1,
                    Logits::None => 2,
                };
                let slot = super::gpu::graph_slot(t, mode);
                if s.few_graphs.len() <= slot {
                    s.few_graphs.resize_with(slot + 1, || None);
                }
                if s.few_graphs[slot].is_none() {
                    let graph = record(&self.gpu, || self.queue_batch(s, t, wanted))?;
                    s.few_graphs[slot] = Some(graph);
                }
                let graph = s.few_graphs[slot].as_ref().unwrap();
                graph.launch().map_err(gpu_error)?;
                true
            } else {
                false
            };
            #[cfg(target_os = "macos")]
            let replayed = false;
            if !replayed {
                self.queue_batch(s, t, wanted)?;
            }
            (s.hidden_from, s.hidden_rows) = match wanted {
                // Every layer ran for every token.
                Logits::All => (s.len, t),
                // The last token's in row 0.
                Logits::Last => (s.len + t - 1, 1),
                Logits::None => (s.len, self.rows_without_logits(t)),
            };
            s.len += t;
            match wanted {
                Logits::All => self.logits_into(s, 0, t, b * batch * vocab)?,
                Logits::Last => self.logits_into(s, 0, 1, 0)?,
                Logits::None => {}
            }
        }
        Ok(match logits {
            Logits::None => None,
            _ => Some(&s.logits),
        })
    }

    /// One token for each of several conversations in one pass: `tokens[i]` at `chats[i]`'s
    /// next position, attending over that conversation's own keys and values. The weights
    /// are read once for all of them, which is most of what a token costs, so several
    /// people's replies written together cost little more than one's. `work` lends its
    /// batch buffers (any session of this model with a batch that long; what it holds
    /// itself is untouched), and keeps every row's logits on the GPU:
    /// [`Self::held_logits`]`(work, i, ..)` is `chats[i]`'s. The conversations must be of
    /// one precision, which sessions of one model are.
    pub fn forward_each(
        &self,
        work: &mut Gemma4CudaSession,
        chats: &mut [&mut Gemma4CudaSession],
        tokens: &[u32],
    ) -> Result<()> {
        let t = tokens.len();
        assert_eq!(t, chats.len(), "a token for each conversation");
        assert!(
            t >= 1 && t <= work.batch,
            "{t} conversations, in batches of {}",
            work.batch
        );
        self.upload_inputs(work, tokens)?;
        for chat in chats.iter() {
            assert!(
                chat.len < chat.capacity,
                "a session is full ({} positions)",
                chat.capacity
            );
            chat.pos.write_i32(&[chat.len as i32]).map_err(gpu_error)?;
        }
        {
            let chats: Vec<&Gemma4CudaSession> = chats.iter().map(|c| &**c).collect();
            self.queue_layers(work, t, 0..self.config.layers, work.pos.ptr(), &chats)?;
        }
        // Each conversation's final state to its own row 0, where its draft model looks
        // for the last position's (as after `forward`).
        let dim = self.config.dim;
        for (i, chat) in chats.iter_mut().enumerate() {
            self.gpu
                .copy(chat.h.ptr(), row(&work.h, i, dim), dim * 4)
                .map_err(gpu_error)?;
            (chat.hidden_from, chat.hidden_rows) = (chat.len, 1);
            chat.len += 1;
        }
        work.hidden_rows = 0;
        work.hold = true;
        let done = self.logits_into(work, 0, t, 0);
        work.hold = false;
        done
    }

    /// How many of a batch's `t` rows of `h` hold final states after a read for no
    /// logits: all of them, unless some layers share keys and values (they didn't run).
    fn rows_without_logits(&self, t: usize) -> usize {
        if self.config.first_shared_layer >= self.config.layers {
            t
        } else {
            0
        }
    }

    /// Queues the layers for the `t` tokens in `s.x`, for the logits `wanted`: every
    /// token's (their final norms in `s.h`'s rows), the last token's (in row 0), or none.
    fn queue_batch(&self, s: &Gemma4CudaSession, t: usize, wanted: Logits) -> Result<()> {
        let c = &self.config;
        let (all, shared) = (0..c.layers, c.first_shared_layer..c.layers);
        if wanted == Logits::All {
            return self.queue_layers(s, t, all, s.pos.ptr(), &[]);
        }
        // The layers that share an earlier layer's keys and values leave nothing for
        // later tokens: a token's pass through them only makes its own logits. So
        // they run for the one token whose logits are wanted, and for no other.
        self.queue_layers(s, t, 0..c.first_shared_layer, s.pos.ptr(), &[])?;
        if wanted == Logits::Last {
            if t > 1 {
                // The last token's state to row 0, where a one-token pass reads it.
                let g = &self.gpu;
                let (dim, pl) = (c.dim, c.per_layer_dim * c.layers);
                let e = gpu_error;
                g.copy(s.x.ptr(), row(&s.x, t - 1, dim), dim * 4)
                    .map_err(e)?;
                g.copy(s.h.ptr(), row(&s.h, t - 1, dim), dim * 4)
                    .map_err(e)?;
                if pl > 0 {
                    g.copy(s.per_layer.ptr(), row(&s.per_layer, t - 1, pl), pl * 4)
                        .map_err(e)?;
                }
            }
            self.queue_layers(s, 1, shared, s.pos_last.ptr(), &[])?;
        }
        Ok(())
    }
}
