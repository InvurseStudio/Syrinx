//! Qwen 3.5 on an NVIDIA GPU: the same steps as `qwen35.rs`, with every weight, the
//! attention layers' keys and values, and the DeltaNet layers' convolution history and
//! states in GPU memory.

use super::qwen35::Config;
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
    embedding: Matrix<'a>,
    output: GpuMatrix,
    output_norm: Buffer,
    layers: Vec<Layer>,
}

enum Memory {
    Attention { keys: Buffer, values: Buffer },
    DeltaNet { history: Buffer, state: Buffer },
}

pub struct Qwen35CudaSession {
    memory: Vec<Memory>,
    len: usize,
    capacity: usize,
    embedding: Vec<f32>,
    x: Buffer,
    h: Buffer,
    mixed: Buffer,
    query_gate: Buffer,
    q: Buffer,
    attention: Buffer,
    scores: Buffer,
    qkv: Buffer,
    conv: Buffer,
    z: Buffer,
    beta: Buffer,
    alpha: Buffer,
    heads_out: Buffer,
    gate: Buffer,
    up: Buffer,
    logits_gpu: Buffer,
    logits: Vec<f32>,
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
}

fn gpu_error(e: cuda::Error) -> Error {
    Error::Unsupported(format!("Running on the GPU ({e})"))
}

impl<'a> Qwen35Cuda<'a> {
    pub fn load(file: &'a GgufFile, gpu: Gpu) -> Result<Self> {
        let c = Config::from_gguf(file)?;
        let dim = c.dim;
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

    pub fn session(&self, capacity: usize) -> Result<Qwen35CudaSession> {
        let c = &self.config;
        let g = &self.gpu;
        let f32s = |n: usize| g.alloc(n * 4).map_err(gpu_error);
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
                    keys: f32s(capacity * kv_dim)?,
                    values: f32s(capacity * kv_dim)?,
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
            embedding: vec![0.0; c.dim],
            x: f32s(c.dim)?,
            h: f32s(c.dim)?,
            mixed: f32s(c.dim)?,
            query_gate: f32s(2 * c.heads * c.head_dim)?,
            q: f32s(c.heads * c.head_dim)?,
            attention: f32s(c.heads * c.head_dim)?,
            scores: f32s(c.heads * capacity)?,
            qkv: f32s(conv_dim)?,
            conv: f32s(conv_dim)?,
            z: f32s(value_dim)?,
            beta: f32s(c.v_heads)?,
            alpha: f32s(c.v_heads)?,
            heads_out: f32s(value_dim)?,
            gate: f32s(c.ffn_dim)?,
            up: f32s(c.ffn_dim)?,
            logits_gpu: f32s(c.vocab)?,
            logits: vec![0.0; c.vocab],
        })
    }

    pub fn forward<'s>(
        &self,
        s: &'s mut Qwen35CudaSession,
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
        let (hd, n) = (c.head_dim, c.state);
        let kv_dim = c.kv_heads * hd;
        let key_dim = c.k_heads * n;

        self.embedding.decode_row(token as usize, &mut s.embedding);
        s.x.write_f32(&s.embedding).map_err(e)?;
        for (l, layer) in self.layers.iter().enumerate() {
            g.rms_norm(
                s.x.ptr(),
                layer.attn_norm.ptr(),
                s.h.ptr(),
                c.dim,
                c.norm_eps,
            )
            .map_err(e)?;
            match (&layer.mixer, &s.memory[l]) {
                (Mixer::Attention(a), Memory::Attention { keys, values }) => {
                    let (k_slot, v_slot) = (keys.f32_at(pos * kv_dim), values.f32_at(pos * kv_dim));
                    mv(&a.q, s.h.ptr(), s.query_gate.ptr())?;
                    mv(&a.k, s.h.ptr(), k_slot)?;
                    mv(&a.v, s.h.ptr(), v_slot)?;
                    g.split_query(s.query_gate.ptr(), s.q.ptr(), c.heads, hd)
                        .map_err(e)?;
                    g.rms_norm_rows(
                        s.q.ptr(),
                        a.q_norm.ptr(),
                        s.q.ptr(),
                        c.heads,
                        hd,
                        c.norm_eps,
                    )
                    .map_err(e)?;
                    g.rms_norm_rows(k_slot, a.k_norm.ptr(), k_slot, c.kv_heads, hd, c.norm_eps)
                        .map_err(e)?;
                    g.rope(
                        s.q.ptr(),
                        c.heads,
                        hd,
                        c.rope_dims,
                        pos,
                        c.rope_base,
                        0,
                        true,
                    )
                    .map_err(e)?;
                    g.rope(
                        k_slot,
                        c.kv_heads,
                        hd,
                        c.rope_dims,
                        pos,
                        c.rope_base,
                        0,
                        true,
                    )
                    .map_err(e)?;
                    g.attention(
                        s.q.ptr(),
                        keys.ptr(),
                        values.ptr(),
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
                    g.apply_gate(s.attention.ptr(), s.query_gate.ptr(), c.heads, hd)
                        .map_err(e)?;
                    mv(&a.o, s.attention.ptr(), s.mixed.ptr())?;
                }
                (Mixer::DeltaNet(d), Memory::DeltaNet { history, state }) => {
                    let channels = 2 * key_dim + c.v_heads * n;
                    mv(&d.qkv, s.h.ptr(), s.qkv.ptr())?;
                    mv(&d.gate, s.h.ptr(), s.z.ptr())?;
                    mv(&d.beta, s.h.ptr(), s.beta.ptr())?;
                    mv(&d.alpha, s.h.ptr(), s.alpha.ptr())?;
                    g.conv_silu(
                        s.qkv.ptr(),
                        history.ptr(),
                        d.conv.ptr(),
                        s.conv.ptr(),
                        channels,
                        c.conv_kernel,
                    )
                    .map_err(e)?;
                    // q and k sit next to each other: 2 × k_heads rows of n to normalize.
                    g.l2_norm_rows(s.conv.ptr(), 2 * c.k_heads, n, c.norm_eps)
                        .map_err(e)?;
                    g.delta_net(
                        s.conv.ptr(),
                        s.conv.f32_at(key_dim),
                        s.conv.f32_at(2 * key_dim),
                        s.beta.ptr(),
                        s.alpha.ptr(),
                        d.dt_bias.ptr(),
                        d.a.ptr(),
                        state.ptr(),
                        s.z.ptr(),
                        d.norm.ptr(),
                        s.heads_out.ptr(),
                        c.v_heads,
                        n,
                        c.k_heads,
                        c.norm_eps,
                    )
                    .map_err(e)?;
                    mv(&d.out, s.heads_out.ptr(), s.mixed.ptr())?;
                }
                _ => unreachable!("each layer's memory matches its kind"),
            }
            g.add(s.x.ptr(), s.mixed.ptr(), c.dim).map_err(e)?;

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
            mv(&layer.down, s.gate.ptr(), s.mixed.ptr())?;
            g.add(s.x.ptr(), s.mixed.ptr(), c.dim).map_err(e)?;
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
}
