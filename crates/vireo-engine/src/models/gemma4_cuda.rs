//! Gemma 4 on an NVIDIA GPU: the same steps as `gemma4.rs`. Every weight is in GPU
//! memory except the two embedding tables, which are only looked up: each token's rows
//! are decoded on the CPU from the mapped file and uploaded (1536 + 8960 numbers for
//! E2B), which keeps the 1.9 GB per-layer table out of GPU memory.

use super::gemma4::Config;
use super::{Error, Result, matrix, vector};
use crate::GgufFile;
use crate::cpu::Matrix;
use crate::cuda::{self, Buffer, Gpu};

struct GpuMatrix {
    kind: crate::TensorType,
    rows: usize,
    cols: usize,
    data: Buffer,
}

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

pub struct Gemma4CudaSession {
    keys: Vec<Option<Buffer>>,
    values: Vec<Option<Buffer>>,
    len: usize,
    capacity: usize,
    embedding: Vec<f32>,
    per_layer_host: Vec<f32>,
    x: Buffer,
    h: Buffer,
    projected: Buffer,
    per_layer: Buffer,
    per_layer_proj: Buffer,
    q: Buffer,
    attention: Buffer,
    scores: Buffer,
    gate: Buffer,
    up: Buffer,
    small: Buffer,
    logits_gpu: Buffer,
    logits: Vec<f32>,
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

fn gpu_error(e: cuda::Error) -> Error {
    Error::Unsupported(format!("Running on the GPU ({e})"))
}

impl<'a> Gemma4Cuda<'a> {
    pub fn load(file: &'a GgufFile, gpu: Gpu) -> Result<Self> {
        let c = Config::from_gguf(file)?;
        let (dim, pl) = (c.dim, c.per_layer_dim);
        let upload = |m: Matrix| -> Result<GpuMatrix> {
            if !Gpu::supports(m.kind) {
                return Err(gpu_error(cuda::Error::Unsupported(m.kind)));
            }
            Ok(GpuMatrix {
                kind: m.kind,
                rows: m.rows,
                cols: m.cols,
                data: gpu.upload(m.data).map_err(gpu_error)?,
            })
        };
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

    pub fn session(&self, capacity: usize) -> Result<Gemma4CudaSession> {
        let c = &self.config;
        let f32s = |n: usize| self.gpu.alloc(n * 4).map_err(gpu_error);
        let cache = |l: usize| -> Result<Option<Buffer>> {
            if l < c.first_shared_layer {
                f32s(capacity * c.kv_heads * c.head_dim_of(l)).map(Some)
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
        Ok(Gemma4CudaSession {
            keys,
            values,
            len: 0,
            capacity,
            embedding: vec![0.0; c.dim],
            per_layer_host: vec![0.0; c.per_layer_dim * c.layers],
            x: f32s(c.dim)?,
            h: f32s(c.dim)?,
            projected: f32s(c.dim)?,
            per_layer: f32s(c.per_layer_dim * c.layers)?,
            per_layer_proj: f32s(c.per_layer_dim * c.layers)?,
            q: f32s(c.heads * widest_head)?,
            attention: f32s(c.heads * widest_head)?,
            scores: f32s(c.heads * capacity)?,
            gate: f32s(widest_ffn)?,
            up: f32s(widest_ffn)?,
            small: f32s(c.per_layer_dim)?,
            logits_gpu: f32s(c.vocab)?,
            logits: vec![0.0; c.vocab],
        })
    }

    pub fn forward<'s>(
        &self,
        s: &'s mut Gemma4CudaSession,
        token: u32,
        want_logits: bool,
    ) -> Result<Option<&'s [f32]>> {
        let c = &self.config;
        let pos = s.len;
        assert!(
            pos < s.capacity,
            "the session is full ({} positions)",
            s.capacity
        );
        assert!(
            (token as usize) < c.vocab,
            "token {token} is past the vocabulary"
        );
        let g = &self.gpu;
        let e = gpu_error;
        let mv = |m: &GpuMatrix, x: u64, y: u64| {
            g.matvec(m.kind, m.data.ptr(), x, y, m.rows, m.cols)
                .map_err(e)
        };
        let (dim, pl, eps) = (c.dim, c.per_layer_dim, c.norm_eps);

        // Inputs, as in gemma4.rs.
        self.embedding.decode_row(token as usize, &mut s.embedding);
        let embed_scale = (dim as f32).sqrt();
        s.embedding.iter_mut().for_each(|v| *v *= embed_scale);
        s.x.write_f32(&s.embedding).map_err(e)?;
        self.per_layer_embedding
            .decode_row(token as usize, &mut s.per_layer_host);
        let token_scale = (pl as f32).sqrt() / 2f32.sqrt();
        s.per_layer_host.iter_mut().for_each(|v| *v *= token_scale);
        s.per_layer.write_f32(&s.per_layer_host).map_err(e)?;
        mv(
            &self.per_layer_model_proj,
            s.x.ptr(),
            s.per_layer_proj.ptr(),
        )?;
        g.rms_norm_rows(
            s.per_layer_proj.ptr(),
            self.per_layer_proj_norm.ptr(),
            s.per_layer_proj.ptr(),
            c.layers,
            pl,
            eps * dim as f32,
        )
        .map_err(e)?;
        g.add(s.per_layer.ptr(), s.per_layer_proj.ptr(), c.layers * pl)
            .map_err(e)?;

        for (l, layer) in self.layers.iter().enumerate() {
            let hd = c.head_dim_of(l);
            let kv_dim = c.kv_heads * hd;
            let (base, factors) = if c.is_swa[l] {
                (c.rope_base_swa, 0)
            } else {
                (
                    c.rope_base,
                    self.rope_factors.as_ref().map_or(0, Buffer::ptr),
                )
            };

            g.rms_norm(s.x.ptr(), layer.attn_norm.ptr(), s.h.ptr(), dim, eps)
                .map_err(e)?;
            mv(&layer.q, s.h.ptr(), s.q.ptr())?;
            g.rms_norm_rows(s.q.ptr(), layer.q_norm.ptr(), s.q.ptr(), c.heads, hd, eps)
                .map_err(e)?;
            g.rope(s.q.ptr(), c.heads, hd, hd, pos, base, factors, true)
                .map_err(e)?;
            if let (Some(k), Some(v), Some(k_norm), Some(keys), Some(values)) =
                (&layer.k, &layer.v, &layer.k_norm, &s.keys[l], &s.values[l])
            {
                let (k_slot, v_slot) = (keys.f32_at(pos * kv_dim), values.f32_at(pos * kv_dim));
                mv(k, s.h.ptr(), k_slot)?;
                mv(v, s.h.ptr(), v_slot)?;
                g.rms_norm_rows(k_slot, k_norm.ptr(), k_slot, c.kv_heads, hd, eps)
                    .map_err(e)?;
                g.rms_norm_rows(v_slot, 0, v_slot, c.kv_heads, hd, eps)
                    .map_err(e)?;
                g.rope(k_slot, c.kv_heads, hd, hd, pos, base, factors, true)
                    .map_err(e)?;
            }

            let source = c.kv_source(l);
            let (keys, values) = (
                s.keys[source]
                    .as_ref()
                    .expect("a source layer has its own keys"),
                s.values[source].as_ref().expect("and values"),
            );
            let first = if c.is_swa[l] {
                (pos + 1).saturating_sub(c.sliding_window)
            } else {
                0
            };
            g.attention(
                s.q.ptr(),
                keys.ptr(),
                values.ptr(),
                s.attention.ptr(),
                s.scores.ptr(),
                c.heads,
                first,
                pos + 1,
                hd,
                kv_dim,
                c.heads / c.kv_heads,
                s.capacity,
                1.0,
            )
            .map_err(e)?;
            mv(&layer.o, s.attention.ptr(), s.projected.ptr())?;
            g.rms_norm(
                s.projected.ptr(),
                layer.post_attention_norm.ptr(),
                s.h.ptr(),
                dim,
                eps,
            )
            .map_err(e)?;
            g.add(s.x.ptr(), s.h.ptr(), dim).map_err(e)?;

            let f = c.ffn_dims[l];
            g.rms_norm(s.x.ptr(), layer.ffn_norm.ptr(), s.h.ptr(), dim, eps)
                .map_err(e)?;
            mv(&layer.gate, s.h.ptr(), s.gate.ptr())?;
            mv(&layer.up, s.h.ptr(), s.up.ptr())?;
            g.geglu(s.gate.ptr(), s.up.ptr(), f).map_err(e)?;
            mv(&layer.down, s.gate.ptr(), s.projected.ptr())?;
            g.rms_norm(
                s.projected.ptr(),
                layer.post_ffw_norm.ptr(),
                s.h.ptr(),
                dim,
                eps,
            )
            .map_err(e)?;
            g.add(s.x.ptr(), s.h.ptr(), dim).map_err(e)?;

            mv(&layer.inp_gate, s.x.ptr(), s.small.ptr())?;
            g.geglu(s.small.ptr(), s.per_layer.f32_at(l * pl), pl)
                .map_err(e)?;
            mv(&layer.proj, s.small.ptr(), s.projected.ptr())?;
            g.rms_norm(
                s.projected.ptr(),
                layer.post_norm.ptr(),
                s.h.ptr(),
                dim,
                eps,
            )
            .map_err(e)?;
            g.add(s.x.ptr(), s.h.ptr(), dim).map_err(e)?;
            if layer.output_scale != 1.0 {
                g.scale(s.x.ptr(), layer.output_scale, dim).map_err(e)?;
            }
        }
        s.len += 1;

        if !want_logits {
            return Ok(None);
        }
        g.rms_norm(s.x.ptr(), self.output_norm.ptr(), s.h.ptr(), dim, eps)
            .map_err(e)?;
        mv(&self.output, s.h.ptr(), s.logits_gpu.ptr())?;
        if let Some(cap) = c.softcap {
            g.softcap(s.logits_gpu.ptr(), cap, c.vocab).map_err(e)?;
        }
        s.logits_gpu.read_f32(&mut s.logits).map_err(e)?;
        Ok(Some(&s.logits))
    }
}
