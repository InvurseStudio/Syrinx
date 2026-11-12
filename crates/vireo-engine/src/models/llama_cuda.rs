//! Llama's forward pass on an NVIDIA GPU: the same steps as `llama.rs`, with every
//! weight and the whole key/value cache in GPU memory.
//!
//! A prompt goes through in batches ([`LlamaCuda::forward_batch`]): each weight is read
//! once per batch, not once per token. Generating ([`LlamaCuda::forward`]) replays one
//! token's launches, recorded once per session as a CUDA graph; only the token's
//! embedding and position go up, and the logits come back.

use super::gpu::{
    GpuMatrix, Input, LogitsBuffer, Outputs, Positions, gpu_error, load_state, q8_room, record,
    row, save_state,
};
use super::llama::Config;
use super::{DEFAULT_BATCH, Logits, Precision, Result, SavedState, matrix, vector};
use crate::GgufFile;
use crate::cpu::Matrix;
use crate::cuda::{Buffer, Gpu, Graph, Q8, QkvPrep};

struct Layer {
    attn_norm: Buffer,
    q: GpuMatrix,
    k: GpuMatrix,
    v: GpuMatrix,
    o: GpuMatrix,
    ffn_norm: Buffer,
    gate: GpuMatrix,
    up: GpuMatrix,
    down: GpuMatrix,
}

/// A Llama model on the GPU. The embedding table stays in the (mapped) file: a token's
/// row is decoded on the CPU and uploaded.
pub struct LlamaCuda<'a> {
    pub config: Config,
    gpu: Gpu,
    precision: Precision,
    embedding: Matrix<'a>,
    output: GpuMatrix,
    output_norm: Buffer,
    rope_factors: Option<Buffer>,
    layers: Vec<Layer>,
}

/// One conversation's state on the GPU. Use it only with the model that made it.
pub struct CudaSession {
    keys: Vec<Buffer>,
    values: Vec<Buffer>,
    len: usize,
    capacity: usize,
    batch: usize,
    embedding: Vec<f32>,
    /// The position of the first token being run, read by the kernels.
    pos: Buffer,
    /// The position of the batch's last token (for the last layer's `Outputs::Last`).
    pos_last: Buffer,
    // A row per token of the batch:
    x: Buffer,
    h: Buffer,
    q: Buffer,
    k_new: Buffer,
    v_new: Buffer,
    attention: Buffer,
    projected: Buffer,
    gate: Buffer,
    up: Buffer,
    logits_gpu: LogitsBuffer,
    logits: Vec<f32>,
    /// The batch's activations in 8 bits, unless the model is exact.
    q8: Option<Q8>,
    /// Keys and values in f16 (8-bit mode) rather than f32.
    half_cache: bool,
    /// One token's layers, recorded on first use.
    graph: Option<Graph>,
}

impl CudaSession {
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
        self.cache_buffers()
            .iter()
            .map(|(b, _)| b.len_bytes())
            .sum::<usize>()
            / self.capacity.max(1)
    }

    /// Forgets every position from `len` on (nothing, if the session is shorter), so the
    /// next tokens run from there: the keys and values before it stay as they are. For
    /// continuing a chat whose new prompt starts with the tokens already read.
    pub fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }

    /// The keys and values of every position so far, copied to system memory.
    pub fn export(&self) -> Result<SavedState> {
        save_state(&self.cache_buffers(), self.capacity, self.len, &[])
    }

    /// Loads the first `upto` positions of a state `export` made from a session of the
    /// same model and precision (of any capacity is fine as long as `upto` fits), in
    /// place of what this session held.
    pub fn import(&mut self, saved: &SavedState, upto: usize) -> Result<()> {
        load_state(saved, upto, &self.cache_buffers(), self.capacity, &[])?;
        self.len = upto;
        Ok(())
    }

    fn cache_buffers(&self) -> Vec<Positions<'_>> {
        self.keys
            .iter()
            .chain(&self.values)
            .map(|b| (b, None))
            .collect()
    }
}

impl<'a> LlamaCuda<'a> {
    /// Loads a Llama model from `file` onto `gpu`.
    pub fn load(file: &'a GgufFile, gpu: Gpu) -> Result<Self> {
        let c = Config::from_gguf(file)?;
        let (dim, kv_dim, q_dim) = (c.dim, c.kv_heads * c.head_dim, c.heads * c.head_dim);
        let upload = |m: Matrix| GpuMatrix::upload(&gpu, m);
        let upload_vector = |v: Vec<f32>| gpu.upload_f32(&v).map_err(gpu_error);

        let mut layers = Vec::with_capacity(c.layers);
        for i in 0..c.layers {
            let name = |part: &str| format!("blk.{i}.{part}.weight");
            layers.push(Layer {
                attn_norm: upload_vector(vector(file, &name("attn_norm"), dim)?)?,
                q: upload(matrix(file, &name("attn_q"), dim, q_dim)?)?,
                k: upload(matrix(file, &name("attn_k"), dim, kv_dim)?)?,
                v: upload(matrix(file, &name("attn_v"), dim, kv_dim)?)?,
                o: upload(matrix(file, &name("attn_output"), q_dim, dim)?)?,
                ffn_norm: upload_vector(vector(file, &name("ffn_norm"), dim)?)?,
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
        let rope_factors = if file.tensor("rope_freqs.weight").is_some() {
            Some(upload_vector(vector(
                file,
                "rope_freqs.weight",
                c.head_dim / 2,
            )?)?)
        } else {
            None
        };
        Ok(Self {
            precision: Precision::default(),
            output_norm: upload_vector(vector(file, "output_norm.weight", dim)?)?,
            config: c,
            embedding,
            output,
            rope_factors,
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

    /// A new, empty conversation that can hold `capacity` positions.
    pub fn session(&self, capacity: usize) -> Result<CudaSession> {
        self.session_with_batch(capacity, DEFAULT_BATCH)
    }

    /// A new, empty conversation that reads up to `batch` tokens at once.
    pub fn session_with_batch(&self, capacity: usize, batch: usize) -> Result<CudaSession> {
        let c = &self.config;
        let batch = batch.clamp(1, capacity.max(1));
        let kv_dim = c.kv_heads * c.head_dim;
        let f32s = |n: usize| self.gpu.alloc(n * 4).map_err(gpu_error);
        // Keys and values: f16 in 8-bit mode (half the memory), f32 when exact.
        let cache_bytes = if self.precision == Precision::Int8 {
            2
        } else {
            4
        };
        let cache = |n: usize| self.gpu.alloc(n * cache_bytes).map_err(gpu_error);
        let mut keys = Vec::with_capacity(c.layers);
        let mut values = Vec::with_capacity(c.layers);
        for _ in 0..c.layers {
            keys.push(cache(capacity * kv_dim)?);
            values.push(cache(capacity * kv_dim)?);
        }
        Ok(CudaSession {
            keys,
            values,
            len: 0,
            capacity,
            batch,
            embedding: vec![0.0; batch * c.dim],
            pos: f32s(1)?,
            pos_last: f32s(1)?,
            x: f32s(batch * c.dim)?,
            h: f32s(batch * c.dim)?,
            q: f32s(batch * c.heads * c.head_dim)?,
            k_new: f32s(batch * kv_dim)?,
            v_new: f32s(batch * kv_dim)?,
            attention: f32s(batch * c.heads * c.head_dim)?,
            projected: f32s(batch * c.dim)?,
            gate: f32s(batch * c.ffn_dim)?,
            up: f32s(batch * c.ffn_dim)?,
            logits_gpu: LogitsBuffer::new(&self.gpu, c.vocab)?,
            logits: vec![0.0; c.vocab],
            q8: q8_room(
                &self.gpu,
                self.precision,
                batch,
                c.dim.max(c.heads * c.head_dim).max(c.ffn_dim),
            )?,
            half_cache: self.precision == Precision::Int8,
            graph: None,
        })
    }

    /// Queues every layer for the `tokens` tokens in `s.x` (at positions `*s.pos` on),
    /// leaving the final norm of each token in `outputs` in `s.h` (`Outputs::Last`: in
    /// row 0; that needs `*s.pos_last` set to the last token's position).
    fn queue_layers(&self, s: &CudaSession, tokens: usize, outputs: Outputs) -> Result<()> {
        let c = &self.config;
        let g = &self.gpu;
        let e = gpu_error;
        let (hd, kv_dim) = (c.head_dim, c.kv_heads * c.head_dim);
        let mut t = tokens;
        let mut pos = s.pos.ptr();
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
            let h = Input::new(g, s.h.ptr(), t, s.q8.as_ref());
            layer.q.mul(&h, s.q.ptr())?;
            layer.k.mul(&h, s.k_new.ptr())?;
            layer.v.mul(&h, s.v_new.ptr())?;
            g.qkv_prep(
                &QkvPrep {
                    q: s.q.ptr(),
                    q_stride: c.heads * hd,
                    q_head_stride: hd,
                    q_norm: None,
                    q_out: s.q.ptr(),
                    k: s.k_new.ptr(),
                    k_norm: None,
                    v: s.v_new.ptr(),
                    v_norm: false,
                    cache: Some((s.keys[l].ptr(), s.values[l].ptr())),
                    heads: c.heads,
                    kv_heads: c.kv_heads,
                    head_dim: hd,
                    rope_dims: hd,
                    rope_base: c.rope_base,
                    rope_factors: self.rope_factors.as_ref().map(Buffer::ptr),
                    neox: false,
                    pos: s.pos.ptr(),
                    eps: c.norm_eps,
                    half_cache: s.half_cache,
                },
                t,
            )
            .map_err(e)?;
            if l + 1 == self.layers.len() && outputs != Outputs::All {
                // The last layer's keys and values are in the cache; the rest of it only
                // makes outputs, so it runs for the last token alone, or not at all.
                if outputs == Outputs::None {
                    break;
                }
                if t > 1 {
                    let q_dim = c.heads * hd;
                    g.copy(s.q.ptr(), row(&s.q, t - 1, q_dim), q_dim * 4)
                        .map_err(e)?;
                    g.copy(s.x.ptr(), row(&s.x, t - 1, c.dim), c.dim * 4)
                        .map_err(e)?;
                }
                t = 1;
                pos = s.pos_last.ptr();
            }
            g.attention(
                s.half_cache,
                s.q.ptr(),
                s.keys[l].ptr(),
                s.values[l].ptr(),
                s.attention.ptr(),
                pos,
                0,
                c.heads,
                t,
                hd,
                kv_dim,
                c.heads / c.kv_heads,
                1.0 / (hd as f32).sqrt(),
            )
            .map_err(e)?;
            layer.o.mul(
                &Input::new(g, s.attention.ptr(), t, s.q8.as_ref()),
                s.projected.ptr(),
            )?;
            g.residual_norm(
                s.x.ptr(),
                Some(s.projected.ptr()),
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
                s.projected.ptr(),
            )?;
            let next = match self.layers.get(l + 1) {
                Some(next) => &next.attn_norm,
                None => &self.output_norm,
            };
            g.residual_norm(
                s.x.ptr(),
                Some(s.projected.ptr()),
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

    /// Runs `token` at the session's next position, as `Llama::forward` does.
    pub fn forward<'s>(
        &self,
        s: &'s mut CudaSession,
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
        s: &'s mut CudaSession,
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
                Logits::Last if last_chunk => {
                    s.pos_last.write_i32(&[(s.len + t - 1) as i32]).map_err(e)?;
                    Outputs::Last
                }
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

    /// Waits for the GPU and reports any error a kernel hit.
    pub fn synchronize(&self) -> Result<()> {
        self.gpu.synchronize().map_err(gpu_error)
    }
}
