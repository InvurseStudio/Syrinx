//! Qwen 3.5 (and 3.6–3.8, the same `qwen35` architecture): the forward pass, on the CPU
//! reference backend.
//!
//! A hybrid. Every `full_attention_interval`-th layer (every 4th) is gated attention;
//! the others are Gated DeltaNet, a linear-attention layer that keeps a fixed-size state
//! per head instead of a cache that grows with the conversation. Every layer then has the
//! same SwiGLU feed-forward:
//!
//! ```text
//! h = RMSNorm(x)                       attn_norm
//! x = x + (full attention or DeltaNet)(h)
//! x = x + FFN(RMSNorm(x))              post_attention_norm, then SwiGLU
//! ```
//!
//! Gated attention (layers 3, 7, 11, …):
//!
//! ```text
//! [q | gate] per head = Wq h           each head's query, then its gate
//! q, k = RMSNorm per head              attn_q_norm, attn_k_norm
//! rotate q and k                       the first 64 of each head's 256 numbers, in halves
//! a = attention(q, k, v) × sigmoid(gate)
//! out = Wo a
//! ```
//!
//! Gated DeltaNet (the other layers), for one token:
//!
//! ```text
//! qkv = Wqkv h, z = Wgate h
//! qkv = SiLU(causal conv1d over the last 4 tokens' qkv)    ssm_conv1d, per channel
//! q, k, v = split(qkv); q, k = q / |q|, k / |k|            per head
//! beta = sigmoid(Wbeta h); g = softplus(Walpha h + dt_bias) × ssm_a   per head (g ≤ 0)
//! S = exp(g) S                                             decay the head's 128×128 state
//! S = S + k (beta (v − Sᵀk))ᵀ                              the delta rule
//! o = Sᵀq / sqrt(128)
//! out = Wout (RMSNorm(o) × ssm_norm × SiLU(z))             per head
//! ```
//!
//! This follows llama.cpp's `qwen35.cpp` graph and its CPU `gated_delta_net` and
//! `ssm_conv` kernels, so the numbers are the same.

use super::{Error, Result, matrix, required, vector};
use crate::GgufFile;
use crate::cpu::{self, Matrix, Pool};
use crate::quant;

/// A Qwen 3.5 model's sizes, from the file's `qwen35.*` settings.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub layers: usize,
    pub dim: usize,
    pub ffn_dim: usize,
    pub vocab: usize,
    pub norm_eps: f32,
    /// Every this-many-th layer is full attention (the last of each group).
    pub full_attention_interval: usize,
    // Full attention.
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// How many of each head's numbers RoPE rotates.
    pub rope_dims: usize,
    pub rope_base: f32,
    // Gated DeltaNet.
    pub conv_kernel: usize,
    /// The size of each head's key and value, and so of its state (state × state).
    pub state: usize,
    pub k_heads: usize,
    pub v_heads: usize,
}

impl Config {
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        if file.architecture() != Some("qwen35") {
            return Err(Error::Unsupported(format!(
                "The {:?} architecture with the Qwen 3.5 forward pass",
                file.architecture().unwrap_or("unnamed")
            )));
        }
        let dim = required(file, "embedding_length")?;
        let heads = required(file, "attention.head_count")?;
        let head_dim = file
            .arch_u64("attention.key_length")
            .map_or(dim / heads, |v| v as usize);
        let vocab = file
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .ok_or_else(|| Error::Missing("vocabulary".into()))?;
        let c = Self {
            layers: required(file, "block_count")?,
            dim,
            ffn_dim: required(file, "feed_forward_length")?,
            vocab,
            norm_eps: file
                .arch_f64("attention.layer_norm_rms_epsilon")
                .unwrap_or(1e-6) as f32,
            full_attention_interval: file
                .arch_u64("full_attention_interval")
                .map_or(4, |v| v as usize),
            heads,
            kv_heads: required(file, "attention.head_count_kv")?,
            head_dim,
            rope_dims: file
                .arch_u64("rope.dimension_count")
                .map_or(head_dim, |v| v as usize),
            rope_base: file.arch_f64("rope.freq_base").unwrap_or(10_000_000.0) as f32,
            conv_kernel: required(file, "ssm.conv_kernel")?,
            state: required(file, "ssm.state_size")?,
            k_heads: required(file, "ssm.group_count")?,
            v_heads: required(file, "ssm.time_step_rank")?,
        };
        let inner = required(file, "ssm.inner_size")?;
        if inner != c.state * c.v_heads
            || !c.v_heads.is_multiple_of(c.k_heads)
            || c.rope_dims > c.head_dim
        {
            return Err(Error::Invalid(format!(
                "the DeltaNet sizes don't fit together (inner {inner}, state {}, {} key heads, {} value heads)",
                c.state, c.k_heads, c.v_heads
            )));
        }
        Ok(c)
    }

    pub fn is_full_attention(&self, layer: usize) -> bool {
        (layer + 1).is_multiple_of(self.full_attention_interval)
    }

    fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// The width of the DeltaNet's q, k and v together: what the convolution runs over.
    fn conv_dim(&self) -> usize {
        2 * self.k_heads * self.state + self.v_heads * self.state
    }
}

struct Attention<'a> {
    q: Matrix<'a>,
    k: Matrix<'a>,
    v: Matrix<'a>,
    o: Matrix<'a>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
}

struct DeltaNet<'a> {
    qkv: Matrix<'a>,
    gate: Matrix<'a>,
    beta: Matrix<'a>,
    alpha: Matrix<'a>,
    dt_bias: Vec<f32>,
    a: Vec<f32>,
    /// `conv_kernel` weights per channel, oldest token first.
    conv: Vec<f32>,
    norm: Vec<f32>,
    out: Matrix<'a>,
}

enum Mixer<'a> {
    Attention(Attention<'a>),
    DeltaNet(DeltaNet<'a>),
}

struct Layer<'a> {
    attn_norm: Vec<f32>,
    mixer: Mixer<'a>,
    ffn_norm: Vec<f32>,
    gate: Matrix<'a>,
    up: Matrix<'a>,
    down: Matrix<'a>,
}

/// A Qwen 3.5 model whose weights stay in the (memory-mapped) file.
pub struct Qwen35<'a> {
    pub config: Config,
    embedding: Matrix<'a>,
    output: Matrix<'a>,
    output_norm: Vec<f32>,
    layers: Vec<Layer<'a>>,
    pool: Pool,
}

/// What each layer remembers between tokens.
enum Memory {
    /// Keys and values of every position so far.
    Attention { keys: Vec<f32>, values: Vec<f32> },
    /// The last `conv_kernel − 1` tokens' qkv (oldest first), and each value head's
    /// state, stored transposed as llama.cpp does: `state[h][j × S + i]` is S[i][j].
    DeltaNet { history: Vec<f32>, state: Vec<f32> },
}

/// One conversation's state, and working space for the next token.
pub struct Session {
    memory: Vec<Memory>,
    len: usize,
    capacity: usize,
    x: Vec<f32>,
    h: Vec<f32>,
    mixed: Vec<f32>,
    // Attention.
    q_gate: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attention: Vec<f32>,
    scores: Vec<f32>,
    // DeltaNet.
    qkv: Vec<f32>,
    conv: Vec<f32>,
    z: Vec<f32>,
    beta: Vec<f32>,
    alpha: Vec<f32>,
    delta: Vec<f32>,
    heads_out: Vec<f32>,
    // Feed-forward.
    gate: Vec<f32>,
    up: Vec<f32>,
    logits: Vec<f32>,
}

impl Session {
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

impl<'a> Qwen35<'a> {
    pub fn load(file: &'a GgufFile, pool: Pool) -> Result<Self> {
        let c = Config::from_gguf(file)?;
        let dim = c.dim;
        let mut layers = Vec::with_capacity(c.layers);
        for i in 0..c.layers {
            let name = |part: &str| format!("blk.{i}.{part}.weight");
            let mixer = if c.is_full_attention(i) {
                let q_dim = c.heads * c.head_dim;
                Mixer::Attention(Attention {
                    q: matrix(file, &name("attn_q"), dim, 2 * q_dim)?,
                    k: matrix(file, &name("attn_k"), dim, c.kv_dim())?,
                    v: matrix(file, &name("attn_v"), dim, c.kv_dim())?,
                    o: matrix(file, &name("attn_output"), q_dim, dim)?,
                    q_norm: vector(file, &name("attn_q_norm"), c.head_dim)?,
                    k_norm: vector(file, &name("attn_k_norm"), c.head_dim)?,
                })
            } else {
                let value_dim = c.v_heads * c.state;
                let conv_matrix = matrix(file, &name("ssm_conv1d"), c.conv_kernel, c.conv_dim())?;
                let mut conv = vec![0.0; c.conv_kernel * c.conv_dim()];
                for (ch, row) in conv.chunks_exact_mut(c.conv_kernel).enumerate() {
                    conv_matrix.decode_row(ch, row);
                }
                Mixer::DeltaNet(DeltaNet {
                    qkv: matrix(file, &name("attn_qkv"), dim, c.conv_dim())?,
                    gate: matrix(file, &name("attn_gate"), dim, value_dim)?,
                    beta: matrix(file, &name("ssm_beta"), dim, c.v_heads)?,
                    alpha: matrix(file, &name("ssm_alpha"), dim, c.v_heads)?,
                    dt_bias: vector(file, &format!("blk.{i}.ssm_dt.bias"), c.v_heads)?,
                    a: vector(file, &format!("blk.{i}.ssm_a"), c.v_heads)?,
                    conv,
                    norm: vector(file, &name("ssm_norm"), c.state)?,
                    out: matrix(file, &name("ssm_out"), value_dim, dim)?,
                })
            };
            layers.push(Layer {
                attn_norm: vector(file, &name("attn_norm"), dim)?,
                mixer,
                ffn_norm: vector(file, &name("post_attention_norm"), dim)?,
                gate: matrix(file, &name("ffn_gate"), dim, c.ffn_dim)?,
                up: matrix(file, &name("ffn_up"), dim, c.ffn_dim)?,
                down: matrix(file, &name("ffn_down"), c.ffn_dim, dim)?,
            });
        }
        let embedding = matrix(file, "token_embd.weight", dim, c.vocab)?;
        let output = if file.tensor("output.weight").is_some() {
            matrix(file, "output.weight", dim, c.vocab)?
        } else {
            embedding
        };
        Ok(Self {
            output_norm: vector(file, "output_norm.weight", dim)?,
            config: c,
            embedding,
            output,
            layers,
            pool,
        })
    }

    pub fn threads(&self) -> usize {
        self.pool.threads()
    }

    /// A new, empty conversation that can hold `capacity` positions. Only the attention
    /// layers' memory grows with `capacity`; the DeltaNet layers' is fixed.
    pub fn session(&self, capacity: usize) -> Session {
        let c = &self.config;
        let memory = (0..c.layers)
            .map(|l| {
                if c.is_full_attention(l) {
                    Memory::Attention {
                        keys: vec![0.0; capacity * c.kv_dim()],
                        values: vec![0.0; capacity * c.kv_dim()],
                    }
                } else {
                    Memory::DeltaNet {
                        history: vec![0.0; (c.conv_kernel - 1) * c.conv_dim()],
                        state: vec![0.0; c.v_heads * c.state * c.state],
                    }
                }
            })
            .collect();
        let value_dim = c.v_heads * c.state;
        Session {
            memory,
            len: 0,
            capacity,
            x: vec![0.0; c.dim],
            h: vec![0.0; c.dim],
            mixed: vec![0.0; c.dim],
            q_gate: vec![0.0; 2 * c.heads * c.head_dim],
            q: vec![0.0; c.heads * c.head_dim],
            k: vec![0.0; c.kv_dim()],
            v: vec![0.0; c.kv_dim()],
            attention: vec![0.0; c.heads * c.head_dim],
            scores: vec![0.0; capacity],
            qkv: vec![0.0; c.conv_dim()],
            conv: vec![0.0; c.conv_dim()],
            z: vec![0.0; value_dim],
            beta: vec![0.0; c.v_heads],
            alpha: vec![0.0; c.v_heads],
            delta: vec![0.0; c.state],
            heads_out: vec![0.0; value_dim],
            gate: vec![0.0; c.ffn_dim],
            up: vec![0.0; c.ffn_dim],
            logits: vec![0.0; c.vocab],
        }
    }

    /// Runs `token` at the session's next position; see `Llama::forward`.
    pub fn forward<'s>(
        &self,
        s: &'s mut Session,
        token: u32,
        want_logits: bool,
    ) -> Option<&'s [f32]> {
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
        let pool = &self.pool;

        self.embedding.decode_row(token as usize, &mut s.x);
        for (l, layer) in self.layers.iter().enumerate() {
            cpu::rms_norm(&s.x, &layer.attn_norm, c.norm_eps, &mut s.h);
            match &layer.mixer {
                Mixer::Attention(a) => self.attention(a, l, pos, s),
                Mixer::DeltaNet(d) => self.delta_net(d, l, s),
            }
            add(&mut s.x, &s.mixed);

            cpu::rms_norm(&s.x, &layer.ffn_norm, c.norm_eps, &mut s.h);
            cpu::matvec(pool, &layer.gate, &s.h, &mut s.gate);
            cpu::matvec(pool, &layer.up, &s.h, &mut s.up);
            cpu::swiglu(&mut s.gate, &s.up);
            cpu::matvec(pool, &layer.down, &s.gate, &mut s.mixed);
            add(&mut s.x, &s.mixed);
        }
        s.len += 1;

        if !want_logits {
            return None;
        }
        cpu::rms_norm(&s.x, &self.output_norm, c.norm_eps, &mut s.h);
        cpu::matvec(pool, &self.output, &s.h, &mut s.logits);
        Some(&s.logits)
    }

    /// Gated attention: reads `s.h`, writes `s.mixed`.
    fn attention(&self, a: &Attention, l: usize, pos: usize, s: &mut Session) {
        let c = &self.config;
        let (hd, kv_dim) = (c.head_dim, c.kv_dim());
        let pool = &self.pool;
        cpu::matvec(pool, &a.q, &s.h, &mut s.q_gate);
        cpu::matvec(pool, &a.k, &s.h, &mut s.k);
        cpu::matvec(pool, &a.v, &s.h, &mut s.v);

        // Each head's query is followed by its gate: [q0 | gate0 | q1 | gate1 | …].
        let mut normed = vec![0.0; hd];
        for head in 0..c.heads {
            cpu::rms_norm(
                &s.q_gate[head * 2 * hd..head * 2 * hd + hd],
                &a.q_norm,
                c.norm_eps,
                &mut normed,
            );
            s.q[head * hd..(head + 1) * hd].copy_from_slice(&normed);
        }
        for head in 0..c.kv_heads {
            let k = &mut s.k[head * hd..(head + 1) * hd];
            cpu::rms_norm(k, &a.k_norm, c.norm_eps, &mut normed);
            k.copy_from_slice(&normed);
        }
        rope_halves(&mut s.q, hd, c.rope_dims, pos, c.rope_base);
        rope_halves(&mut s.k, hd, c.rope_dims, pos, c.rope_base);

        let Memory::Attention { keys, values } = &mut s.memory[l] else {
            unreachable!("layer {l} is an attention layer")
        };
        keys[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&s.k);
        values[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&s.v);

        let group = c.heads / c.kv_heads;
        let scale = 1.0 / (hd as f32).sqrt();
        for head in 0..c.heads {
            let q = &s.q[head * hd..(head + 1) * hd];
            let kv = (head / group) * hd;
            let scores = &mut s.scores[..=pos];
            for (t, score) in scores.iter_mut().enumerate() {
                *score = quant::dot(q, &keys[t * kv_dim + kv..t * kv_dim + kv + hd]) * scale;
            }
            cpu::softmax(scores);
            let out = &mut s.attention[head * hd..(head + 1) * hd];
            out.fill(0.0);
            for (t, &p) in scores.iter().enumerate() {
                for (o, &v) in out
                    .iter_mut()
                    .zip(&values[t * kv_dim + kv..t * kv_dim + kv + hd])
                {
                    *o += p * v;
                }
            }
            // The output gate.
            let gate = &s.q_gate[head * 2 * hd + hd..(head + 1) * 2 * hd];
            for (o, &g) in out.iter_mut().zip(gate) {
                *o *= sigmoid(g);
            }
        }
        cpu::matvec(pool, &a.o, &s.attention, &mut s.mixed);
    }

    /// Gated DeltaNet for one token: reads `s.h`, writes `s.mixed`.
    fn delta_net(&self, d: &DeltaNet, l: usize, s: &mut Session) {
        let c = &self.config;
        let pool = &self.pool;
        let (n, kernel) = (c.state, c.conv_kernel);
        let channels = c.conv_dim();
        cpu::matvec(pool, &d.qkv, &s.h, &mut s.qkv);
        cpu::matvec(pool, &d.gate, &s.h, &mut s.z);
        cpu::matvec(pool, &d.beta, &s.h, &mut s.beta);
        cpu::matvec(pool, &d.alpha, &s.h, &mut s.alpha);
        let Memory::DeltaNet { history, state } = &mut s.memory[l] else {
            unreachable!("layer {l} is a DeltaNet layer")
        };

        // The causal convolution: each channel's last `kernel` values, oldest first, times
        // its weights; then SiLU. The history moves along by one token.
        for ch in 0..channels {
            let w = &d.conv[ch * kernel..(ch + 1) * kernel];
            let mut sum = 0.0;
            for (i, &weight) in w[..kernel - 1].iter().enumerate() {
                sum += history[i * channels + ch] * weight;
            }
            sum += s.qkv[ch] * w[kernel - 1];
            s.conv[ch] = silu(sum);
        }
        history.copy_within(channels.., 0);
        history[(kernel - 2) * channels..].copy_from_slice(&s.qkv);

        // q, k and v, with q and k scaled to unit length per head.
        let key_dim = c.k_heads * n;
        let (q, rest) = s.conv.split_at_mut(key_dim);
        let (k, v) = rest.split_at_mut(key_dim);
        for head in q.chunks_exact_mut(n).chain(k.chunks_exact_mut(n)) {
            let norm = head.iter().map(|x| x * x).sum::<f32>().sqrt();
            let inv = 1.0 / norm.max(c.norm_eps);
            head.iter_mut().for_each(|x| *x *= inv);
        }

        let scale = 1.0 / (n as f32).sqrt();
        for h in 0..c.v_heads {
            let kh = h % c.k_heads; // Qwen 3.5 tiles key heads across value heads
            let (qh, kh_vec) = (&q[kh * n..(kh + 1) * n], &k[kh * n..(kh + 1) * n]);
            let vh = &v[h * n..(h + 1) * n];
            let beta = sigmoid(s.beta[h]);
            let g = softplus(s.alpha[h] + d.dt_bias[h]) * d.a[h];
            let st = &mut state[h * n * n..(h + 1) * n * n];

            let decay = g.exp();
            st.iter_mut().for_each(|x| *x *= decay);
            for j in 0..n {
                let column = &st[j * n..(j + 1) * n];
                s.delta[j] = (vh[j] - quant::dot(column, kh_vec)) * beta;
            }
            for j in 0..n {
                let dj = s.delta[j];
                for (x, &ki) in st[j * n..(j + 1) * n].iter_mut().zip(kh_vec) {
                    *x += ki * dj;
                }
            }
            let out = &mut s.heads_out[h * n..(h + 1) * n];
            for (j, o) in out.iter_mut().enumerate() {
                *o = quant::dot(&st[j * n..(j + 1) * n], qh) * scale;
            }

            // Gated RMSNorm: RMSNorm(o) × ssm_norm × SiLU(z).
            let mut normed = vec![0.0; n];
            cpu::rms_norm(out, &d.norm, c.norm_eps, &mut normed);
            for ((o, &x), &z) in out.iter_mut().zip(&normed).zip(&s.z[h * n..(h + 1) * n]) {
                *o = x * silu(z);
            }
        }
        cpu::matvec(pool, &d.out, &s.heads_out, &mut s.mixed);
    }
}

/// RoPE on the first `rope_dims` numbers of each head, paired in halves: number j turns
/// with number j + rope_dims/2 (ggml's NEOX / IMROPE layout; for text, all of Qwen 3.5's
/// position sections hold the same position, so the sections change nothing).
fn rope_halves(x: &mut [f32], head_dim: usize, rope_dims: usize, pos: usize, base: f32) {
    let half = rope_dims / 2;
    for head in x.chunks_exact_mut(head_dim) {
        for j in 0..half {
            let theta = pos as f32 * base.powf(-2.0 * j as f32 / rope_dims as f32);
            let (sin, cos) = theta.sin_cos();
            let (a, b) = (head[j], head[j + half]);
            head[j] = a * cos - b * sin;
            head[j + half] = a * sin + b * cos;
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// log(1 + eˣ), with ggml's cut-off where it equals x to f32 precision.
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

fn add(x: &mut [f32], y: &[f32]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += b;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_pairs_halves_and_leaves_the_rest() {
        // head_dim 8, rope_dims 4: numbers 0..4 rotate as (0, 2) and (1, 3); 4..8 stay.
        let mut x = [1.0, 0.0, 0.0, 1.0, 5.0, 6.0, 7.0, 8.0];
        rope_halves(&mut x, 8, 4, 1, 10000.0);
        // Pair (0, 2) turns by 1 radian; pair (1, 3) by 10000^(-1/2) = 0.01.
        assert!((x[0] - 1f32.cos()).abs() < 1e-6 && (x[2] - 1f32.sin()).abs() < 1e-6);
        assert!((x[1] + 0.01f32.sin()).abs() < 1e-6 && (x[3] - 0.01f32.cos()).abs() < 1e-6);
        assert_eq!(&x[4..], &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn softplus_matches_its_definition() {
        for x in [-30.0f32, -1.0, 0.0, 1.0, 19.0, 25.0] {
            let expected = (1.0 + (x as f64).exp()).ln() as f32;
            assert!(
                (softplus(x) - expected).abs() <= expected.abs() * 1e-6 + 1e-12,
                "{x}"
            );
        }
    }
}
