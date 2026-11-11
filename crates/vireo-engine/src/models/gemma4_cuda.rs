//! Gemma 4 on an NVIDIA GPU: the same steps as `gemma4.rs`. Every weight is in GPU
//! memory except the two embedding tables, which are only looked up: each token's rows
//! are decoded on the CPU from the mapped file and uploaded (1536 + 8960 numbers for
//! E2B), which keeps the 1.9 GB per-layer table out of GPU memory. Prompts go through
//! in batches and generated tokens replay a recorded graph, as in `llama_cuda.rs`.

use super::gemma4::Config;
use super::gpu::{GpuMatrix, Input, LogitsBuffer, gpu_error, q8_room, record, row};
use super::{DEFAULT_BATCH, Error, Logits, Precision, Result, matrix, vector};
use crate::GgufFile;
use crate::cpu::Matrix;
use crate::cuda::{Buffer, Gpu, Graph, Q8, QkvPrep};

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
    inp_gate: GpuMatrix,
    proj: GpuMatrix,
    post_norm: Buffer,
    output_scale: f32,
}

pub struct Gemma4Cuda<'a> {
    pub config: Config,
    gpu: Gpu,
    precision: Precision,
    embedding: Matrix<'a>,
    per_layer_embedding: Matrix<'a>,
    output: GpuMatrix,
    per_layer_model_proj: GpuMatrix,
    /// per_layer_proj_norm / sqrt(2), as llama.cpp folds it.
    per_layer_proj_norm: Buffer,
    rope_factors: Option<Buffer>,
    output_norm: Buffer,
    layers: Vec<Layer>,
}

/// One conversation's state on the GPU. Use it only with the model that made it.
pub struct Gemma4CudaSession {
    keys: Vec<Option<Buffer>>,
    values: Vec<Option<Buffer>>,
    len: usize,
    capacity: usize,
    batch: usize,
    embedding: Vec<f32>,
    per_layer_host: Vec<f32>,
    pos: Buffer,
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
    logits_gpu: LogitsBuffer,
    logits: Vec<f32>,
    /// The batch's activations in 8 bits, unless the model is exact.
    q8: Option<Q8>,
    /// Keys and values in f16 (8-bit mode) rather than f32.
    half_cache: bool,
    graph: Option<Graph>,
}

impl Gemma4CudaSession {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<'a> Gemma4Cuda<'a> {
    pub fn load(file: &'a GgufFile, gpu: Gpu) -> Result<Self> {
        let c = Config::from_gguf(file)?;
        let (dim, pl) = (c.dim, c.per_layer_dim);
        let upload = |m: Matrix| GpuMatrix::upload(&gpu, m);
        let up = |v: Vec<f32>| gpu.upload_f32(&v).map_err(gpu_error);

        let mut layers = Vec::with_capacity(c.layers);
        for i in 0..c.layers {
            let name = |part: &str| format!("blk.{i}.{part}.weight");
            let hd = c.head_dim_of(i);
            let own_kv = i < c.first_shared_layer;
            if own_kv && file.tensor(&name("attn_v")).is_none() {
                return Err(Error::Unsupported(format!(
                    "Layer {i} without attn_v on the GPU"
                )));
            }
            let kv = |part: &str| -> Result<Option<GpuMatrix>> {
                if own_kv {
                    upload(matrix(file, &name(part), dim, c.kv_heads * hd)?).map(Some)
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
                inp_gate: upload(matrix(file, &name("inp_gate"), dim, pl)?)?,
                proj: upload(matrix(file, &name("proj"), pl, dim)?)?,
                post_norm: up(vector(file, &name("post_norm"), dim)?)?,
                output_scale: vector(file, &name("layer_output_scale"), 1).map_or(1.0, |v| v[0]),
            });
        }
        if file.tensor("output.weight").is_some() {
            return Err(Error::Unsupported(
                "Gemma 4 with a separate output matrix".into(),
            ));
        }
        let embedding = matrix(file, "token_embd.weight", dim, c.vocab)?;
        let proj_norm: Vec<f32> = vector(file, "per_layer_proj_norm.weight", pl)?
            .into_iter()
            .map(|w| w / 2f32.sqrt())
            .collect();
        let rope_factors = if file.tensor("rope_freqs.weight").is_some() {
            Some(up(vector(file, "rope_freqs.weight", c.head_dim / 2)?)?)
        } else {
            None
        };
        Ok(Self {
            precision: Precision::default(),
            output: upload(embedding)?,
            embedding,
            per_layer_embedding: matrix(
                file,
                "per_layer_token_embd.weight",
                pl * c.layers,
                c.vocab,
            )?,
            per_layer_model_proj: upload(matrix(
                file,
                "per_layer_model_proj.weight",
                dim,
                pl * c.layers,
            )?)?,
            per_layer_proj_norm: up(proj_norm)?,
            rope_factors,
            output_norm: up(vector(file, "output_norm.weight", dim)?)?,
            layers,
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
        let f32s = |n: usize| self.gpu.alloc(n * 4).map_err(gpu_error);
        // Keys and values: f16 in 8-bit mode (half the memory), f32 when exact.
        let cache_bytes = if self.precision == Precision::Int8 {
            2
        } else {
            4
        };
        let cache_buffer = |n: usize| self.gpu.alloc(n * cache_bytes).map_err(gpu_error);
        let cache = |l: usize| -> Result<Option<Buffer>> {
            if l < c.first_shared_layer {
                cache_buffer(capacity * c.kv_heads * c.head_dim_of(l)).map(Some)
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
            len: 0,
            capacity,
            batch,
            embedding: vec![0.0; batch * c.dim],
            per_layer_host: vec![0.0; batch * per_layer],
            pos: f32s(1)?,
            x: f32s(batch * c.dim)?,
            h: f32s(batch * c.dim)?,
            projected: f32s(batch * c.dim)?,
            per_layer: f32s(batch * per_layer)?,
            per_layer_proj: f32s(batch * per_layer)?,
            q: f32s(batch * c.heads * widest_head)?,
            k_new: f32s(batch * c.kv_heads * widest_head)?,
            v_new: f32s(batch * c.kv_heads * widest_head)?,
            attention: f32s(batch * c.heads * widest_head)?,
            gate: f32s(batch * widest_ffn)?,
            up: f32s(batch * widest_ffn)?,
            small: f32s(batch * c.per_layer_dim)?,
            logits_gpu: LogitsBuffer::new(&self.gpu, c.vocab)?,
            logits: vec![0.0; c.vocab],
            q8: q8_room(
                &self.gpu,
                self.precision,
                batch,
                c.dim
                    .max(c.heads * widest_head)
                    .max(widest_ffn)
                    .max(c.per_layer_dim),
            )?,
            half_cache: self.precision == Precision::Int8,
            graph: None,
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
            e.iter_mut().for_each(|v| *v *= embed_scale);
            let p = &mut s.per_layer_host[i * pl..(i + 1) * pl];
            self.per_layer_embedding.decode_row(token as usize, p);
            p.iter_mut().for_each(|v| *v *= token_scale);
        }
        let t = tokens.len();
        let e = gpu_error;
        s.x.write_f32(&s.embedding[..t * dim]).map_err(e)?;
        s.per_layer
            .write_f32(&s.per_layer_host[..t * pl])
            .map_err(e)?;
        s.pos.write_i32(&[s.len as i32]).map_err(e)
    }

    /// Queues the per-layer inputs and every layer for the `t` tokens in `s.x` (at
    /// positions `*s.pos` on), leaving the final norm of each in `s.h`.
    fn queue_layers(&self, s: &Gemma4CudaSession, t: usize) -> Result<()> {
        let c = &self.config;
        let g = &self.gpu;
        let e = gpu_error;
        let (dim, pl, eps) = (c.dim, c.per_layer_dim, c.norm_eps);

        self.per_layer_model_proj.mul(
            &Input::new(g, s.x.ptr(), t, s.q8.as_ref()),
            s.per_layer_proj.ptr(),
        )?;
        g.rms_norm_rows(
            s.per_layer_proj.ptr(),
            self.per_layer_proj_norm.ptr(),
            s.per_layer_proj.ptr(),
            t * c.layers,
            pl,
            eps * dim as f32,
        )
        .map_err(e)?;
        g.add(s.per_layer.ptr(), s.per_layer_proj.ptr(), t * c.layers * pl)
            .map_err(e)?;

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
        .map_err(e)?;
        for (l, layer) in self.layers.iter().enumerate() {
            let hd = c.head_dim_of(l);
            let kv_dim = c.kv_heads * hd;
            let (base, factors) = if c.is_swa[l] {
                (c.rope_base_swa, None)
            } else {
                (c.rope_base, self.rope_factors.as_ref().map(Buffer::ptr))
            };

            let h = Input::new(g, s.h.ptr(), t, s.q8.as_ref());
            layer.q.mul(&h, s.q.ptr())?;
            let own = match (&layer.k, &layer.v, &s.keys[l], &s.values[l]) {
                (Some(k), Some(v), Some(keys), Some(values)) => {
                    k.mul(&h, s.k_new.ptr())?;
                    v.mul(&h, s.v_new.ptr())?;
                    Some((keys.ptr(), values.ptr()))
                }
                _ => None,
            };
            g.qkv_prep(
                &QkvPrep {
                    q: s.q.ptr(),
                    q_stride: c.heads * hd,
                    q_head_stride: hd,
                    q_norm: Some(layer.q_norm.ptr()),
                    q_out: s.q.ptr(),
                    k: s.k_new.ptr(),
                    k_norm: layer.k_norm.as_ref().map(Buffer::ptr),
                    v: s.v_new.ptr(),
                    v_norm: true,
                    cache: own,
                    heads: c.heads,
                    kv_heads: c.kv_heads,
                    head_dim: hd,
                    rope_dims: hd,
                    rope_base: base,
                    rope_factors: factors,
                    neox: true,
                    pos: s.pos.ptr(),
                    eps,
                    half_cache: s.half_cache,
                },
                t,
            )
            .map_err(e)?;

            let source = c.kv_source(l);
            let (keys, values) = (
                s.keys[source]
                    .as_ref()
                    .expect("a source layer has its own keys"),
                s.values[source].as_ref().expect("and values"),
            );
            g.attention(
                s.half_cache,
                s.q.ptr(),
                keys.ptr(),
                values.ptr(),
                s.attention.ptr(),
                s.pos.ptr(),
                if c.is_swa[l] { c.sliding_window } else { 0 },
                c.heads,
                t,
                hd,
                kv_dim,
                c.heads / c.kv_heads,
                1.0,
            )
            .map_err(e)?;
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
            let h = Input::new(g, s.h.ptr(), t, s.q8.as_ref());
            layer.gate.mul(&h, s.gate.ptr())?;
            layer.up.mul(&h, s.up.ptr())?;
            g.geglu(s.gate.ptr(), s.up.ptr(), f, f, t).map_err(e)?;
            layer.down.mul(
                &Input::new(g, s.gate.ptr(), t, s.q8.as_ref()),
                s.projected.ptr(),
            )?;
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

            layer
                .inp_gate
                .mul(&Input::new(g, s.x.ptr(), t, s.q8.as_ref()), s.small.ptr())?;
            g.geglu(
                s.small.ptr(),
                s.per_layer.f32_at(l * pl),
                pl,
                c.layers * pl,
                t,
            )
            .map_err(e)?;
            layer.proj.mul(
                &Input::new(g, s.small.ptr(), t, s.q8.as_ref()),
                s.projected.ptr(),
            )?;
            let next = match self.layers.get(l + 1) {
                Some(next) => &next.attn_norm,
                None => &self.output_norm,
            };
            g.residual_norm(
                s.x.ptr(),
                Some(s.projected.ptr()),
                Some(&layer.post_norm),
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
            let graph = record(&self.gpu, || self.queue_layers(s, 1))?;
            s.graph = Some(graph);
        }
        s.graph.as_ref().unwrap().launch().map_err(gpu_error)?;
        s.len += 1;
        if !want_logits {
            return Ok(None);
        }
        s.logits.resize(self.config.vocab, 0.0);
        self.logits_into(s, 0, 1, 0)?;
        Ok(Some(&s.logits))
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
            if logits == Logits::All {
                tokens.len() * vocab
            } else {
                vocab
            },
            0.0,
        );
        let batch = s.batch;
        for (b, chunk) in tokens.chunks(batch).enumerate() {
            let t = chunk.len();
            self.upload_inputs(s, chunk)?;
            self.queue_layers(s, t)?;
            s.len += t;
            match logits {
                Logits::All => self.logits_into(s, 0, t, b * batch * vocab)?,
                Logits::Last if b * batch + t == tokens.len() => {
                    self.logits_into(s, t - 1, 1, 0)?
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
