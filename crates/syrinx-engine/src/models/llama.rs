//! Llama (3 and 3.x): the forward pass, on the CPU reference backend.
//!
//! Each of the model's layers, for one new token at position `pos`:
//!
//! ```text
//! h = RMSNorm(x)                                   attn_norm
//! q, k, v = Wq h, Wk h, Wv h                       grouped-query: fewer k/v heads than q heads
//! rotate q and k by pos (RoPE, adjacent pairs)     scaled by rope_freqs in Llama 3.1+
//! attention over every position so far             k and v are kept in the session's cache
//! x = x + Wo · attention
//! h = RMSNorm(x)                                   ffn_norm
//! x = x + Wdown (SiLU(Wgate h) × Wup h)
//! ```
//!
//! then the logits are `Woutput · RMSNorm(x)`, where Llama 3.2's output matrix is its
//! token embedding matrix (the file has no separate `output.weight`).
//!
//! GGUF files store Llama's q and k weights permuted (llama.cpp's converter does it), so
//! that rotating adjacent pairs here equals the rotation of halves in the original model.

use super::{Error, Result, matrix, required, vector};
use crate::GgufFile;
use crate::cpu::{self, Matrix, Pool};
use crate::quant;

/// A Llama model's sizes, from the file's `llama.*` settings.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub layers: usize,
    /// The width of the model: each token's vector.
    pub dim: usize,
    pub ffn_dim: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub norm_eps: f32,
    pub rope_base: f32,
    /// The longest context the model was trained for.
    pub context: usize,
}

impl Config {
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        if file.architecture() != Some("llama") {
            return Err(Error::Unsupported(format!(
                "The {:?} architecture with the Llama forward pass",
                file.architecture().unwrap_or("unnamed")
            )));
        }
        let dim = required(file, "embedding_length")?;
        let heads = required(file, "attention.head_count")?;
        let head_dim = file
            .arch_u64("attention.key_length")
            .map_or(dim / heads.max(1), |v| v as usize);
        let rope_dims = file
            .arch_u64("rope.dimension_count")
            .map_or(head_dim, |v| v as usize);
        if rope_dims != head_dim {
            return Err(Error::Unsupported(
                "Rotating only part of each head (partial RoPE)".into(),
            ));
        }
        if let Some(scaling) = file.get_str("llama.rope.scaling.type")
            && scaling != "none"
        {
            return Err(Error::Unsupported(format!(
                "RoPE scaling of type {scaling:?}"
            )));
        }
        let vocab = file
            .arch_u64("vocab_size")
            .map(|v| v as usize)
            .or_else(|| {
                file.get("tokenizer.ggml.tokens")?
                    .as_array()
                    .map(|a| a.len())
            })
            .ok_or_else(|| Error::Missing("vocabulary size".into()))?;
        Ok(Self {
            layers: required(file, "block_count")?,
            dim,
            ffn_dim: required(file, "feed_forward_length")?,
            heads,
            kv_heads: file
                .arch_u64("attention.head_count_kv")
                .map_or(heads, |v| v as usize),
            head_dim,
            vocab,
            norm_eps: file
                .arch_f64("attention.layer_norm_rms_epsilon")
                .unwrap_or(1e-5) as f32,
            rope_base: file.arch_f64("rope.freq_base").unwrap_or(10000.0) as f32,
            context: file.arch_u64("context_length").unwrap_or(4096) as usize,
        })
    }

    fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }
}

struct Layer<'a> {
    attn_norm: Vec<f32>,
    q: Matrix<'a>,
    k: Matrix<'a>,
    v: Matrix<'a>,
    o: Matrix<'a>,
    ffn_norm: Vec<f32>,
    gate: Matrix<'a>,
    up: Matrix<'a>,
    down: Matrix<'a>,
}

/// A Llama model whose weights stay in the (memory-mapped) file.
pub struct Llama<'a> {
    pub config: Config,
    embedding: Matrix<'a>,
    output: Matrix<'a>,
    output_norm: Vec<f32>,
    rope_factors: Option<Vec<f32>>,
    layers: Vec<Layer<'a>>,
    pool: Pool,
}

/// One conversation's state: the keys and values of every position so far, and
/// working space for the next token.
pub struct Session {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    len: usize,
    capacity: usize,
    x: Vec<f32>,
    h: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attention: Vec<f32>,
    projected: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    scores: Vec<f32>,
    logits: Vec<f32>,
}

impl Session {
    /// How many positions it holds so far.
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

impl<'a> Llama<'a> {
    /// Loads a Llama model from `file`, using `pool` for the matrix products.
    pub fn load(file: &'a GgufFile, pool: Pool) -> Result<Self> {
        let c = Config::from_gguf(file)?;
        let (dim, kv_dim) = (c.dim, c.kv_dim());
        let q_dim = c.heads * c.head_dim;
        let layers = (0..c.layers)
            .map(|i| {
                let name = |part: &str| format!("blk.{i}.{part}.weight");
                Ok(Layer {
                    attn_norm: vector(file, &name("attn_norm"), dim)?,
                    q: matrix(file, &name("attn_q"), dim, q_dim)?,
                    k: matrix(file, &name("attn_k"), dim, kv_dim)?,
                    v: matrix(file, &name("attn_v"), dim, kv_dim)?,
                    o: matrix(file, &name("attn_output"), q_dim, dim)?,
                    ffn_norm: vector(file, &name("ffn_norm"), dim)?,
                    gate: matrix(file, &name("ffn_gate"), dim, c.ffn_dim)?,
                    up: matrix(file, &name("ffn_up"), dim, c.ffn_dim)?,
                    down: matrix(file, &name("ffn_down"), c.ffn_dim, dim)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let embedding = matrix(file, "token_embd.weight", dim, c.vocab)?;
        let output = if file.tensor("output.weight").is_some() {
            matrix(file, "output.weight", dim, c.vocab)?
        } else {
            embedding // tied: the embedding matrix is the output matrix too
        };
        let rope_factors = if file.tensor("rope_freqs.weight").is_some() {
            Some(vector(file, "rope_freqs.weight", c.head_dim / 2)?)
        } else {
            None
        };
        Ok(Self {
            output_norm: vector(file, "output_norm.weight", dim)?,
            config: c,
            embedding,
            output,
            rope_factors,
            layers,
            pool,
        })
    }

    /// A new, empty conversation that can hold `capacity` positions.
    pub fn session(&self, capacity: usize) -> Session {
        let c = &self.config;
        let cache = || {
            (0..c.layers)
                .map(|_| vec![0.0; capacity * c.kv_dim()])
                .collect()
        };
        Session {
            keys: cache(),
            values: cache(),
            len: 0,
            capacity,
            x: vec![0.0; c.dim],
            h: vec![0.0; c.dim],
            q: vec![0.0; c.heads * c.head_dim],
            k: vec![0.0; c.kv_dim()],
            v: vec![0.0; c.kv_dim()],
            attention: vec![0.0; c.heads * c.head_dim],
            projected: vec![0.0; c.dim],
            gate: vec![0.0; c.ffn_dim],
            up: vec![0.0; c.ffn_dim],
            scores: vec![0.0; capacity],
            logits: vec![0.0; c.vocab],
        }
    }

    pub fn threads(&self) -> usize {
        self.pool.threads()
    }

    /// Runs `token` through the model at the session's next position. With
    /// `want_logits`, returns the scores for the token after it (unnormalized, one per
    /// vocabulary entry); without, skips the output matrix, which a prompt's tokens
    /// before the last don't need.
    ///
    /// # Panics
    /// If the session is full or `token` is past the end of the vocabulary.
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
        let (hd, kv_dim) = (c.head_dim, c.kv_dim());
        let group = c.heads / c.kv_heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let pool = &self.pool;
        let factors = self.rope_factors.as_deref();

        self.embedding.decode_row(token as usize, &mut s.x);
        for (l, layer) in self.layers.iter().enumerate() {
            cpu::rms_norm(&s.x, &layer.attn_norm, c.norm_eps, &mut s.h);
            cpu::matvec(pool, &layer.q, &s.h, &mut s.q);
            cpu::matvec(pool, &layer.k, &s.h, &mut s.k);
            cpu::matvec(pool, &layer.v, &s.h, &mut s.v);
            cpu::rope_normal(&mut s.q, hd, pos, c.rope_base, factors);
            cpu::rope_normal(&mut s.k, hd, pos, c.rope_base, factors);
            s.keys[l][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&s.k);
            s.values[l][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&s.v);

            let (keys, values) = (&s.keys[l], &s.values[l]);
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
                    let v = &values[t * kv_dim + kv..t * kv_dim + kv + hd];
                    for (o, &x) in out.iter_mut().zip(v) {
                        *o += p * x;
                    }
                }
            }
            cpu::matvec(pool, &layer.o, &s.attention, &mut s.projected);
            add(&mut s.x, &s.projected);

            cpu::rms_norm(&s.x, &layer.ffn_norm, c.norm_eps, &mut s.h);
            cpu::matvec(pool, &layer.gate, &s.h, &mut s.gate);
            cpu::matvec(pool, &layer.up, &s.h, &mut s.up);
            cpu::swiglu(&mut s.gate, &s.up);
            cpu::matvec(pool, &layer.down, &s.gate, &mut s.projected);
            add(&mut s.x, &s.projected);
        }
        s.len += 1;

        if !want_logits {
            return None;
        }
        cpu::rms_norm(&s.x, &self.output_norm, c.norm_eps, &mut s.h);
        cpu::matvec(pool, &self.output, &s.h, &mut s.logits);
        Some(&s.logits)
    }
}

fn add(x: &mut [f32], y: &[f32]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += b;
    }
}
