//! Llama's forward pass on an NVIDIA GPU: the same steps as `llama.rs`, with every
//! weight and the whole key/value cache in GPU memory. Only the new token's embedding
//! goes up (8 KB) and the logits come back down each step.
//!
//! The first, straightforward version: one kernel launch per step of each layer, with
//! no fusing and no CUDA graphs yet.

use super::llama::Config;
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
    embedding: Matrix<'a>,
    output: GpuMatrix,
    output_norm: Buffer,
    rope_factors: Option<Buffer>,
    layers: Vec<Layer>,
}

/// One conversation's state on the GPU.
pub struct CudaSession {
    keys: Vec<Buffer>,
    values: Vec<Buffer>,
    len: usize,
    capacity: usize,
    embedding: Vec<f32>,
    x: Buffer,
    h: Buffer,
    q: Buffer,
    attention: Buffer,
    projected: Buffer,
    gate: Buffer,
    up: Buffer,
    scores: Buffer,
    logits_gpu: Buffer,
    logits: Vec<f32>,
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
}

fn gpu_error(e: cuda::Error) -> Error {
    Error::Unsupported(format!("Running on the GPU ({e})"))
}

impl<'a> LlamaCuda<'a> {
    /// Loads a Llama model from `file` onto `gpu`.
    pub fn load(file: &'a GgufFile, gpu: Gpu) -> Result<Self> {
        let c = Config::from_gguf(file)?;
        let (dim, kv_dim, q_dim) = (c.dim, c.kv_heads * c.head_dim, c.heads * c.head_dim);
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

    /// A new, empty conversation that can hold `capacity` positions.
    pub fn session(&self, capacity: usize) -> Result<CudaSession> {
        let c = &self.config;
        let kv_dim = c.kv_heads * c.head_dim;
        let f32s = |n: usize| self.gpu.alloc(n * 4).map_err(gpu_error);
        let mut keys = Vec::with_capacity(c.layers);
        let mut values = Vec::with_capacity(c.layers);
        for _ in 0..c.layers {
            keys.push(f32s(capacity * kv_dim)?);
            values.push(f32s(capacity * kv_dim)?);
        }
        Ok(CudaSession {
            keys,
            values,
            len: 0,
            capacity,
            embedding: vec![0.0; c.dim],
            x: f32s(c.dim)?,
            h: f32s(c.dim)?,
            q: f32s(c.heads * c.head_dim)?,
            attention: f32s(c.heads * c.head_dim)?,
            projected: f32s(c.dim)?,
            gate: f32s(c.ffn_dim)?,
            up: f32s(c.ffn_dim)?,
            scores: f32s(c.heads * capacity)?,
            logits_gpu: f32s(c.vocab)?,
            logits: vec![0.0; c.vocab],
        })
    }

    /// Runs `token` at the session's next position, as `Llama::forward` does.
    pub fn forward<'s>(
        &self,
        s: &'s mut CudaSession,
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
        let (hd, kv_dim) = (c.head_dim, c.kv_heads * c.head_dim);
        let g = &self.gpu;
        let e = gpu_error;
        let mv = |m: &GpuMatrix, x: u64, y: u64| {
            g.matvec(m.kind, m.data.ptr(), x, y, m.rows, m.cols)
                .map_err(e)
        };
        let factors = self.rope_factors.as_ref().map_or(0, Buffer::ptr);

        self.embedding.decode_row(token as usize, &mut s.embedding);
        s.x.write_f32(&s.embedding).map_err(e)?;
        for (l, layer) in self.layers.iter().enumerate() {
            // The new key and value go straight into the cache at this position.
            let (k_slot, v_slot) = (
                s.keys[l].f32_at(pos * kv_dim),
                s.values[l].f32_at(pos * kv_dim),
            );
            g.rms_norm(
                s.x.ptr(),
                layer.attn_norm.ptr(),
                s.h.ptr(),
                c.dim,
                c.norm_eps,
            )
            .map_err(e)?;
            mv(&layer.q, s.h.ptr(), s.q.ptr())?;
            mv(&layer.k, s.h.ptr(), k_slot)?;
            mv(&layer.v, s.h.ptr(), v_slot)?;
            g.rope_norm(s.q.ptr(), c.heads, hd, pos, c.rope_base, factors)
                .map_err(e)?;
            g.rope_norm(k_slot, c.kv_heads, hd, pos, c.rope_base, factors)
                .map_err(e)?;
            g.attention(
                s.q.ptr(),
                s.keys[l].ptr(),
                s.values[l].ptr(),
                s.attention.ptr(),
                s.scores.ptr(),
                c.heads,
                0,
                pos + 1,
                hd,
                kv_dim,
                c.heads / c.kv_heads,
                s.capacity,
                1.0 / (hd as f32).sqrt(),
            )
            .map_err(e)?;
            mv(&layer.o, s.attention.ptr(), s.projected.ptr())?;
            g.add(s.x.ptr(), s.projected.ptr(), c.dim).map_err(e)?;

            g.rms_norm(
                s.x.ptr(),
                layer.ffn_norm.ptr(),
                s.h.ptr(),
                c.dim,
                c.norm_eps,
            )
            .map_err(e)?;
            mv(&layer.gate, s.h.ptr(), s.gate.ptr())?;
            mv(&layer.up, s.h.ptr(), s.up.ptr())?;
            g.swiglu(s.gate.ptr(), s.up.ptr(), c.ffn_dim).map_err(e)?;
            mv(&layer.down, s.gate.ptr(), s.projected.ptr())?;
            g.add(s.x.ptr(), s.projected.ptr(), c.dim).map_err(e)?;
        }
        s.len += 1;

        if !want_logits {
            return Ok(None);
        }
        g.rms_norm(
            s.x.ptr(),
            self.output_norm.ptr(),
            s.h.ptr(),
            c.dim,
            c.norm_eps,
        )
        .map_err(e)?;
        mv(&self.output, s.h.ptr(), s.logits_gpu.ptr())?;
        s.logits_gpu.read_f32(&mut s.logits).map_err(e)?;
        Ok(Some(&s.logits))
    }

    /// Waits for the GPU and reports any error a kernel hit.
    pub fn synchronize(&self) -> Result<()> {
        self.gpu.synchronize().map_err(gpu_error)
    }
}
