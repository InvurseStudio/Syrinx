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
}

impl Qwen35CudaSession {
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
        self.attention_buffers().iter().map(|(b, _)| b.len_bytes()).sum::<usize>() / self.capacity.max(1)
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
        Ok(())
    }

    fn attention_buffers(&self) -> Vec<Positions<'_>> {
        self.memory
            .iter()
            .flat_map(|m| match m {
                Memory::Attention { keys, values } => vec![(keys, None), (values, None)],
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
        let conv_dim = 2 * c.k_heads * c.state + c.v_heads * c.state;
        let value_dim = c.v_heads * c.state;

        let mut layers = Vec::with_capacity(c.layers);
        for i in 0..c.layers {
            let name = |part: &str| format!("blk.{i}.{part}.weight");
            let mixer = if c.is_full_attention(i) {
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
            layers.push(Layer {
                attn_norm: up(vector(file, &name("attn_norm"), dim)?)?,
                mixer,
                ffn_norm: up(vector(file, &name("post_attention_norm"), dim)?)?,
                gate: upload(matrix(file, &name("ffn_gate"), dim, c.ffn_dim)?)?,
                up: upload(matrix(file, &name("ffn_up"), dim, c.ffn_dim)?)?,
                down: upload(matrix(file, &name("ffn_down"), c.ffn_dim, dim)?)?,
            });
        }
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
            *into = Some(Qwen35Checkpoint { len: 0, saved });
        }
        let cp = into.as_mut().unwrap();
        for ((history, state), (h, st)) in deltas.zip(&cp.saved) {
            g.copy(h.ptr(), history.ptr(), history.len_bytes())
                .map_err(gpu_error)?;
            g.copy(st.ptr(), state.ptr(), state.len_bytes())
                .map_err(gpu_error)?;
        }
        cp.len = s.len;
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
        })
    }

    /// Queues every layer for the `t` tokens in `s.x` (at positions `*s.pos` on),
    /// leaving the final norm of each token in `outputs` in `s.h` (`Outputs::Last`: in
    /// row 0).
    fn queue_layers(&self, s: &Qwen35CudaSession, t: usize, outputs: Outputs) -> Result<()> {
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

            let h = Input::new(g, s.h.ptr(), t, s.q8.as_ref());
            layer.gate.mul(&h, s.gate.ptr())?;
            layer.up.mul(&h, s.up.ptr())?;
            g.swiglu(s.gate.ptr(), s.up.ptr(), c.ffn_dim, c.ffn_dim, t)
                .map_err(e)?;
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
            let graph = record(&self.gpu, || self.queue_layers(s, 1, Outputs::All))?;
            s.graph = Some(graph);
        }
        s.graph.as_ref().unwrap().launch().map_err(e)?;
        s.len += 1;
        if !want_logits {
            return Ok(None);
        }
        let logits = s.logits_gpu.rows(&self.gpu, 1, c.vocab)?;
        self.output.mul(
            &Input::new(&self.gpu, s.h.ptr(), 1, s.q8.as_ref()),
            logits.ptr(),
        )?;
        s.logits.resize(c.vocab, 0.0);
        logits.read_f32(&mut s.logits).map_err(e)?;
        Ok(Some(&s.logits))
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
            let outputs = match logits {
                Logits::All => Outputs::All,
                Logits::Last if last_chunk => Outputs::Last,
                _ => Outputs::None,
            };
            self.queue_layers(s, t, outputs)?;
            s.len += t;
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
                        &Input::new(&self.gpu, s.h.ptr(), 1, s.q8.as_ref()),
                        out.ptr(),
                    )?;
                    s.logits.resize(c.vocab, 0.0);
                    out.read_f32(&mut s.logits).map_err(e)?;
                }
                _ => {}
            }
        }
        Ok(match logits {
            Logits::None => None,
            _ => Some(&s.logits),
        })
    }
}
