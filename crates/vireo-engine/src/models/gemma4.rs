//! Gemma 4 (E2B, E4B, 31B, and the mixture-of-experts 26B-A4B): the forward pass, on the
//! CPU reference backend. Follows llama.cpp's `gemma4.cpp` graph.
//!
//! Each token's input:
//!
//! ```text
//! x   = token_embd[t] × sqrt(dim)
//! PLI = per_layer_token_embd[t] × sqrt(256)/sqrt(2)                 one 256-wide input
//!     + RMSNorm(per_layer_model_proj · x) × per_layer_proj_norm/sqrt(2)   per layer
//! ```
//!
//! Each layer (sliding-window layers with 256-wide heads; every 5th global, 512-wide):
//!
//! ```text
//! h = RMSNorm(x) × attn_norm
//! q = RoPE(RMSNorm(Wq h) × q_norm)                per head
//! k = RoPE(RMSNorm(Wk h) × k_norm), v = RMSNorm(Wv h)   (v has no weight)
//!     or, in the last `shared_kv_layers` layers, the keys and values of the last
//!     earlier layer of the same kind
//! a = x + RMSNorm(Wo · attention(q, k, v)) × post_attention_norm   (scale 1: q, k are normed)
//! f = a + RMSNorm(down(GELU(gate a') × up a')) × post_ffw_norm     a' = RMSNorm(a) × ffn_norm
//! x = (f + RMSNorm(proj · (GELU(inp_gate f) × PLI[layer])) × post_norm) × layer_output_scale
//! ```
//!
//! A mixture-of-experts layer (26B-A4B: 128 experts, 8 used, beside a dense network)
//! replaces the `down(...)` in `f` with two parts, each normed, added:
//!
//! ```text
//! dense   = RMSNorm(down(GELU(gate a') × up a')) × post_ffw_norm_1
//! scores  = ffn_gate_inp · (RMSNorm(a) × ffn_gate_inp.scale / sqrt(dim))      one per expert
//! shares  = softmax over the `used` highest scores
//! experts = RMSNorm(Σ share_e × scale_e × down_e(GELU(gate_e a'') × up_e a'')) × post_ffw_norm_2
//!           a'' = RMSNorm(a) × pre_ffw_norm_2;  gate_e, up_e: the two halves of expert e's
//!           rows in ffn_gate_up_exps;  scale_e: ffn_down_exps.scale
//! f = a + RMSNorm(dense + experts) × post_ffw_norm
//! ```
//!
//! and the logits are `30 × tanh(token_embd · (RMSNorm(x) × output_norm) / 30)`.

use super::{Error, Result, matrices, matrix, required, vector};
use crate::GgufFile;
use crate::cpu::{self, Matrix, Pool};
use crate::gguf::Value;
use crate::half::round_to_f16;
use crate::quant;

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub layers: usize,
    pub dim: usize,
    /// Each layer's feed-forward width.
    pub ffn_dims: Vec<usize>,
    pub vocab: usize,
    pub norm_eps: f32,
    pub heads: usize,
    /// Each layer's key-value heads: one count for the small models, a count per layer
    /// for the large (31B has 16 on sliding-window layers and 4 on global ones).
    pub kv_heads: Vec<usize>,
    /// Head size of global layers, and of sliding-window layers.
    pub head_dim: usize,
    pub head_dim_swa: usize,
    pub rope_base: f32,
    pub rope_base_swa: f32,
    pub sliding_window: usize,
    /// Which layers use the sliding window.
    pub is_swa: Vec<bool>,
    /// Layers from here on reuse an earlier layer's keys and values.
    pub first_shared_layer: usize,
    /// The width of each layer's per-layer input (E2B, E4B); 0 for models without them
    /// (31B).
    pub per_layer_dim: usize,
    pub softcap: Option<f32>,
    /// Muse Glimmer (`muse-glimmer`), Gemma 4's layer with other choices (see
    /// `gemma4_cuda.rs`): the embeddings RMS-normed rather than scaled, values not normed,
    /// attention scaled by 1/sqrt(head size), no RoPE on global layers, SiLU, a sigmoid
    /// gate (`attn_gate`) on each head's output, and logits times `logit_scale`. GPU
    /// (Metal) only.
    pub muse: bool,
    /// The logits' multiplier before the softcap (Muse: `logit_scale`; Gemma 1).
    pub logit_scale: f32,
    /// A mixture-of-experts model's experts (26B-A4B); `None` for the dense models.
    pub experts: Option<Experts>,
}

/// A mixture-of-experts model's settings: every layer has `count` small feed-forward
/// networks `width` wide beside its dense one, and each token goes through the `used`
/// of them its router scores highest.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Experts {
    pub count: usize,
    pub used: usize,
    pub width: usize,
}

impl Config {
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let arch = file.architecture().unwrap_or("");
        let muse = arch == "muse-glimmer";
        if arch != "gemma4" && !muse {
            return Err(Error::Unsupported(format!(
                "The {:?} architecture with the Gemma 4 forward pass",
                file.architecture().unwrap_or("unnamed")
            )));
        }
        let layers = required(file, "block_count")?;
        let list = |key: &str| -> Option<Vec<Value>> {
            file.get(&format!("{arch}.{key}"))?
                .as_array()
                .map(<[_]>::to_vec)
        };
        let ffn_dims = match list("feed_forward_length") {
            Some(items) => items
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect(),
            None => vec![required(file, "feed_forward_length")?; layers],
        };
        let is_swa: Vec<bool> = list("attention.sliding_window_pattern")
            .ok_or_else(|| {
                Error::Missing(format!("{arch}.attention.sliding_window_pattern setting"))
            })?
            .iter()
            .map(|v| v.as_bool().unwrap_or(false))
            .collect();
        let kv_heads = match list("attention.head_count_kv") {
            Some(items) => items
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect(),
            None => vec![required(file, "attention.head_count_kv")?; layers],
        };
        if ffn_dims.len() != layers || is_swa.len() != layers || kv_heads.len() != layers {
            return Err(Error::Invalid(
                "per-layer settings don't have one entry per layer".into(),
            ));
        }
        let experts = if file.tensor("blk.0.ffn_gate_inp.weight").is_some() {
            let e = Experts {
                count: required(file, "expert_count")?,
                used: required(file, "expert_used_count")?,
                width: required(file, "expert_feed_forward_length")?,
            };
            if e.used == 0 || e.used > e.count || e.width == 0 {
                return Err(Error::Invalid(
                    "the mixture-of-experts settings don't fit each other".into(),
                ));
            }
            Some(e)
        } else {
            None
        };
        let shared = file.arch_u64("attention.shared_kv_layers").unwrap_or(0) as usize;
        let vocab = file
            .get("tokenizer.ggml.tokens")
            .and_then(Value::as_array)
            .map(<[_]>::len)
            .ok_or_else(|| Error::Missing("vocabulary".into()))?;
        let head_dim = required(file, "attention.key_length")?;
        let rope_base = file.arch_f64("rope.freq_base").unwrap_or(1_000_000.0) as f32;
        let c = Self {
            layers,
            dim: required(file, "embedding_length")?,
            ffn_dims,
            vocab,
            norm_eps: file
                .arch_f64("attention.layer_norm_rms_epsilon")
                .unwrap_or(1e-6) as f32,
            heads: required(file, "attention.head_count")?,
            kv_heads,
            head_dim,
            // Muse's heads and RoPE base are the same on both kinds of layer.
            head_dim_swa: match file.arch_u64("attention.key_length_swa") {
                Some(v) => v as usize,
                None if muse => head_dim,
                None => required(file, "attention.key_length_swa")?,
            },
            rope_base,
            rope_base_swa: file
                .arch_f64("rope.freq_base_swa")
                .map_or(if muse { rope_base } else { 10_000.0 }, |v| v as f32),
            sliding_window: required(file, "attention.sliding_window")?,
            is_swa,
            first_shared_layer: layers - shared,
            per_layer_dim: file
                .arch_u64("embedding_length_per_layer_input")
                .unwrap_or(0) as usize,
            softcap: file.arch_f64("final_logit_softcapping").map(|v| v as f32),
            muse,
            logit_scale: file.arch_f64("logit_scale").map_or(1.0, |v| v as f32),
            experts,
        };
        if c.kv_heads
            .iter()
            .any(|&k| k == 0 || !c.heads.is_multiple_of(k))
        {
            return Err(Error::Invalid(
                "a layer's key-value heads don't divide its query heads".into(),
            ));
        }
        if c.first_shared_layer < 2 {
            return Err(Error::Invalid(
                "too many layers share keys and values".into(),
            ));
        }
        Ok(c)
    }

    pub fn kv_heads_of(&self, layer: usize) -> usize {
        self.kv_heads[layer]
    }

    /// The most key-value heads of any layer, for buffers every layer uses.
    pub fn widest_kv_heads(&self) -> usize {
        self.kv_heads.iter().copied().max().unwrap_or(0)
    }

    pub fn head_dim_of(&self, layer: usize) -> usize {
        if self.is_swa[layer] {
            self.head_dim_swa
        } else {
            self.head_dim
        }
    }

    /// The layer whose keys and values `layer` uses: itself, or for a shared layer the
    /// last layer with its own of the same kind (llama.cpp's `kv_source`).
    pub fn kv_source(&self, layer: usize) -> usize {
        if layer < self.first_shared_layer {
            layer
        } else {
            self.first_shared_layer - if self.is_swa[layer] { 2 } else { 1 }
        }
    }
}

struct Layer<'a> {
    attn_norm: Vec<f32>,
    q: Matrix<'a>,
    /// The layer's own keys and values, if it has them.
    k: Option<Matrix<'a>>,
    v: Option<Matrix<'a>>,
    q_norm: Vec<f32>,
    k_norm: Option<Vec<f32>>,
    o: Matrix<'a>,
    post_attention_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    gate: Matrix<'a>,
    up: Matrix<'a>,
    down: Matrix<'a>,
    post_ffw_norm: Vec<f32>,
    /// The per-layer input's gate, projection and norm (models with them).
    per_layer: Option<(Matrix<'a>, Matrix<'a>, Vec<f32>)>,
    /// The layer's experts (a mixture-of-experts model).
    experts: Option<ExpertLayer<'a>>,
    output_scale: f32,
}

/// A layer's experts and the router that picks among them.
struct ExpertLayer<'a> {
    /// The router: a score per expert from the layer's input, RMS-normed and times
    /// `router_weight` (the file's `ffn_gate_inp.scale` over sqrt(dim)).
    router: Matrix<'a>,
    router_weight: Vec<f32>,
    /// Every expert's gate and up matrices, an expert's `2 × width` rows after the last's
    /// (the gate's rows, then the up's), and every expert's down matrix likewise.
    gate_up: Matrix<'a>,
    down: Matrix<'a>,
    /// Each expert's multiplier on its share (`ffn_down_exps.scale`).
    scale: Vec<f32>,
    /// The norm on the experts' input, on the dense network's output and on the experts'.
    pre_norm: Vec<f32>,
    dense_post_norm: Vec<f32>,
    post_norm: Vec<f32>,
}

/// The per-layer inputs' embedding, projection and its norm (E2B, E4B).
struct PerLayerInputs<'a> {
    embedding: Matrix<'a>,
    model_proj: Matrix<'a>,
    proj_norm: Vec<f32>,
}

pub struct Gemma4<'a> {
    pub config: Config,
    embedding: Matrix<'a>,
    per_layer_inputs: Option<PerLayerInputs<'a>>,
    rope_factors: Option<Vec<f32>>,
    output_norm: Vec<f32>,
    layers: Vec<Layer<'a>>,
    pool: Pool,
    /// Whether GELU rounds its input and output to f16, like a lookup table over f16
    /// inputs. Off: exact GELU matched llama.cpp's probabilities more closely (see
    /// PLAN.md); kept to look into the remaining 0.005% difference in perplexity.
    pub gelu_f16: bool,
}

pub struct Session {
    /// Keys and values of each layer that has its own (empty for shared layers).
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    len: usize,
    capacity: usize,
    x: Vec<f32>,
    h: Vec<f32>,
    per_layer: Vec<f32>,
    per_layer_proj: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attention: Vec<f32>,
    projected: Vec<f32>,
    scores: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    small: Vec<f32>,
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

impl<'a> Gemma4<'a> {
    pub fn load(file: &'a GgufFile, pool: Pool) -> Result<Self> {
        let c = Config::from_gguf(file)?;
        if c.muse {
            return Err(Error::Unsupported(
                "Muse Glimmer on the CPU (it runs on the GPU only)".into(),
            ));
        }
        let (dim, pl) = (c.dim, c.per_layer_dim);
        let mut layers = Vec::with_capacity(c.layers);
        for i in 0..c.layers {
            let name = |part: &str| format!("blk.{i}.{part}.weight");
            let hd = c.head_dim_of(i);
            let own_kv = i < c.first_shared_layer;
            let optional = |part: &str, rows: usize| -> Result<Option<Matrix<'a>>> {
                if own_kv && file.tensor(&name(part)).is_some() {
                    matrix(file, &name(part), dim, rows).map(Some)
                } else {
                    Ok(None)
                }
            };
            let scale = vector(file, &name("layer_output_scale"), 1).map_or(1.0, |v| v[0]);
            layers.push(Layer {
                attn_norm: vector(file, &name("attn_norm"), dim)?,
                q: matrix(file, &name("attn_q"), dim, c.heads * hd)?,
                k: optional("attn_k", c.kv_heads_of(i) * hd)?,
                v: optional("attn_v", c.kv_heads_of(i) * hd)?,
                q_norm: vector(file, &name("attn_q_norm"), hd)?,
                k_norm: if own_kv {
                    Some(vector(file, &name("attn_k_norm"), hd)?)
                } else {
                    None
                },
                o: matrix(file, &name("attn_output"), c.heads * hd, dim)?,
                post_attention_norm: vector(file, &name("post_attention_norm"), dim)?,
                ffn_norm: vector(file, &name("ffn_norm"), dim)?,
                gate: matrix(file, &name("ffn_gate"), dim, c.ffn_dims[i])?,
                up: matrix(file, &name("ffn_up"), dim, c.ffn_dims[i])?,
                down: matrix(file, &name("ffn_down"), c.ffn_dims[i], dim)?,
                post_ffw_norm: vector(file, &name("post_ffw_norm"), dim)?,
                experts: match c.experts {
                    Some(e) => {
                        let root = (dim as f32).sqrt();
                        Some(ExpertLayer {
                            router: matrix(file, &name("ffn_gate_inp"), dim, e.count)?,
                            router_weight: vector(file, &format!("blk.{i}.ffn_gate_inp.scale"), dim)?
                                .iter()
                                .map(|s| s / root)
                                .collect(),
                            gate_up: matrices(
                                file,
                                &name("ffn_gate_up_exps"),
                                dim,
                                2 * e.width,
                                e.count,
                            )?,
                            down: matrices(file, &name("ffn_down_exps"), e.width, dim, e.count)?,
                            scale: vector(file, &format!("blk.{i}.ffn_down_exps.scale"), e.count)?,
                            pre_norm: vector(file, &name("pre_ffw_norm_2"), dim)?,
                            dense_post_norm: vector(file, &name("post_ffw_norm_1"), dim)?,
                            post_norm: vector(file, &name("post_ffw_norm_2"), dim)?,
                        })
                    }
                    None => None,
                },
                per_layer: if pl > 0 {
                    Some((
                        matrix(file, &name("inp_gate"), dim, pl)?,
                        matrix(file, &name("proj"), pl, dim)?,
                        vector(file, &name("post_norm"), dim)?,
                    ))
                } else {
                    None
                },
                output_scale: scale,
            });
            if own_kv && layers[i].k.is_none() {
                return Err(Error::Missing(format!("tensor blk.{i}.attn_k.weight")));
            }
        }
        if file.tensor("output.weight").is_some() {
            return Err(Error::Unsupported(
                "Gemma 4 with a separate output matrix".into(),
            ));
        }
        let rope_factors = if file.tensor("rope_freqs.weight").is_some() {
            Some(vector(file, "rope_freqs.weight", c.head_dim / 2)?)
        } else {
            None
        };
        Ok(Self {
            embedding: matrix(file, "token_embd.weight", dim, c.vocab)?,
            per_layer_inputs: if pl > 0 {
                Some(PerLayerInputs {
                    embedding: matrix(file, "per_layer_token_embd.weight", pl * c.layers, c.vocab)?,
                    model_proj: matrix(file, "per_layer_model_proj.weight", dim, pl * c.layers)?,
                    proj_norm: vector(file, "per_layer_proj_norm.weight", pl)?,
                })
            } else {
                None
            },
            rope_factors,
            output_norm: vector(file, "output_norm.weight", dim)?,
            layers,
            config: c,
            pool,
            gelu_f16: false,
        })
    }

    pub fn threads(&self) -> usize {
        self.pool.threads()
    }

    pub fn session(&self, capacity: usize) -> Session {
        let c = &self.config;
        let cache = |l: usize| {
            if l < c.first_shared_layer {
                vec![0.0; capacity * c.kv_heads_of(l) * c.head_dim_of(l)]
            } else {
                Vec::new()
            }
        };
        let widest_head = c.head_dim.max(c.head_dim_swa);
        let widest_ffn = c.ffn_dims.iter().copied().max().unwrap_or(0);
        Session {
            keys: (0..c.layers).map(cache).collect(),
            values: (0..c.layers).map(cache).collect(),
            len: 0,
            capacity,
            x: vec![0.0; c.dim],
            h: vec![0.0; c.dim],
            per_layer: vec![0.0; c.per_layer_dim * c.layers],
            per_layer_proj: vec![0.0; c.per_layer_dim * c.layers],
            q: vec![0.0; c.heads * widest_head],
            k: vec![0.0; c.widest_kv_heads() * widest_head],
            v: vec![0.0; c.widest_kv_heads() * widest_head],
            attention: vec![0.0; c.heads * widest_head],
            projected: vec![0.0; c.dim],
            scores: vec![0.0; capacity],
            gate: vec![0.0; widest_ffn],
            up: vec![0.0; widest_ffn],
            small: vec![0.0; c.per_layer_dim],
            logits: vec![0.0; c.vocab],
        }
    }

    fn gelu(&self, x: f32) -> f32 {
        if !self.gelu_f16 {
            return gelu_tanh(x);
        }
        // As a table over f16 inputs of f16 results would: exact outside ±10.
        if x <= -10.0 {
            0.0
        } else if x >= 10.0 {
            x
        } else {
            round_to_f16(gelu_tanh(round_to_f16(x)))
        }
    }

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
        let (dim, pl) = (c.dim, c.per_layer_dim);
        let eps = c.norm_eps;

        // Inputs: the scaled embedding, and each layer's per-layer input.
        self.embedding.decode_row(token as usize, &mut s.x);
        let embed_scale = (dim as f32).sqrt();
        s.x.iter_mut().for_each(|v| *v *= embed_scale);
        if let Some(inputs) = &self.per_layer_inputs {
            inputs
                .embedding
                .decode_row(token as usize, &mut s.per_layer);
            let token_scale = (pl as f32).sqrt() / 2f32.sqrt();
            s.per_layer.iter_mut().for_each(|v| *v *= token_scale);
            cpu::matvec(pool, &inputs.model_proj, &s.x, &mut s.per_layer_proj);
            let proj_norm: Vec<f32> = inputs.proj_norm.iter().map(|w| w / 2f32.sqrt()).collect();
            for l in 0..c.layers {
                let range = l * pl..(l + 1) * pl;
                // RMSNorm of (W x) / sqrt(dim) with eps, written as llama.cpp does: eps × dim.
                cpu::rms_norm(
                    &s.per_layer_proj[range.clone()],
                    &proj_norm,
                    eps * dim as f32,
                    &mut s.small,
                );
                for (p, v) in s.per_layer[range].iter_mut().zip(&s.small) {
                    *p += v;
                }
            }
        }

        for (l, layer) in self.layers.iter().enumerate() {
            let hd = c.head_dim_of(l);
            let (q_dim, kv_dim) = (c.heads * hd, c.kv_heads_of(l) * hd);
            let (base, factors) = if c.is_swa[l] {
                (c.rope_base_swa, None)
            } else {
                (c.rope_base, self.rope_factors.as_deref())
            };

            cpu::rms_norm(&s.x, &layer.attn_norm, eps, &mut s.h);
            cpu::matvec(pool, &layer.q, &s.h, &mut s.q[..q_dim]);
            norm_heads(&mut s.q[..q_dim], hd, Some(&layer.q_norm), eps);
            rope_neox(&mut s.q[..q_dim], hd, pos, base, factors);
            if let (Some(k), Some(k_norm)) = (&layer.k, &layer.k_norm) {
                cpu::matvec(pool, k, &s.h, &mut s.k[..kv_dim]);
                match &layer.v {
                    Some(v) => cpu::matvec(pool, v, &s.h, &mut s.v[..kv_dim]),
                    None => s.v[..kv_dim].copy_from_slice(&s.k[..kv_dim]),
                }
                norm_heads(&mut s.k[..kv_dim], hd, Some(k_norm), eps);
                norm_heads(&mut s.v[..kv_dim], hd, None, eps);
                rope_neox(&mut s.k[..kv_dim], hd, pos, base, factors);
                s.keys[l][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&s.k[..kv_dim]);
                s.values[l][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&s.v[..kv_dim]);
            }

            let source = c.kv_source(l);
            let (keys, values) = (&s.keys[source], &s.values[source]);
            let first = if c.is_swa[l] {
                (pos + 1).saturating_sub(c.sliding_window)
            } else {
                0
            };
            let group = c.heads / c.kv_heads_of(l);
            for head in 0..c.heads {
                let q = &s.q[head * hd..(head + 1) * hd];
                let kv = (head / group) * hd;
                let scores = &mut s.scores[first..=pos];
                for (i, score) in scores.iter_mut().enumerate() {
                    let t = first + i;
                    *score = quant::dot(q, &keys[t * kv_dim + kv..t * kv_dim + kv + hd]);
                }
                cpu::softmax(scores);
                let out = &mut s.attention[head * hd..(head + 1) * hd];
                out.fill(0.0);
                for (i, &p) in scores.iter().enumerate() {
                    let t = first + i;
                    for (o, &v) in out
                        .iter_mut()
                        .zip(&values[t * kv_dim + kv..t * kv_dim + kv + hd])
                    {
                        *o += p * v;
                    }
                }
            }
            cpu::matvec(pool, &layer.o, &s.attention[..q_dim], &mut s.projected);
            cpu::rms_norm(&s.projected, &layer.post_attention_norm, eps, &mut s.h);
            add(&mut s.x, &s.h); // x is now attn_out

            let f = c.ffn_dims[l];
            cpu::rms_norm(&s.x, &layer.ffn_norm, eps, &mut s.h);
            cpu::matvec(pool, &layer.gate, &s.h, &mut s.gate[..f]);
            cpu::matvec(pool, &layer.up, &s.h, &mut s.up[..f]);
            for (g, &u) in s.gate[..f].iter_mut().zip(&s.up[..f]) {
                *g = self.gelu(*g) * u;
            }
            cpu::matvec(pool, &layer.down, &s.gate[..f], &mut s.projected);
            if let (Some(m), Some(e)) = (&layer.experts, c.experts) {
                // The dense network's output, normed, plus the experts': the router
                // scores every expert from the layer's input, the `used` highest share
                // the token by a softmax over their scores (times each one's scale).
                let mut dense = vec![0.0; c.dim];
                cpu::rms_norm(&s.projected, &m.dense_post_norm, eps, &mut dense);
                cpu::rms_norm(&s.x, &m.router_weight, eps, &mut s.h);
                let mut scores = vec![0.0; e.count];
                cpu::matvec(pool, &m.router, &s.h, &mut scores);
                let mut picked: Vec<usize> = (0..e.count).collect();
                picked.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
                picked.truncate(e.used);
                let mut shares: Vec<f32> = picked.iter().map(|&i| scores[i]).collect();
                cpu::softmax(&mut shares);
                cpu::rms_norm(&s.x, &m.pre_norm, eps, &mut s.h);
                let w = e.width;
                let mut gate_up = vec![0.0; 2 * w];
                let mut out = vec![0.0; c.dim];
                let mut sum = vec![0.0; c.dim];
                for (&i, &share) in picked.iter().zip(&shares) {
                    cpu::matvec(pool, &m.gate_up.rows_from(i * 2 * w, 2 * w), &s.h, &mut gate_up);
                    let act: Vec<f32> = (0..w)
                        .map(|j| self.gelu(gate_up[j]) * gate_up[w + j])
                        .collect();
                    cpu::matvec(pool, &m.down.rows_from(i * c.dim, c.dim), &act, &mut out);
                    let share = share * m.scale[i];
                    for (s, &o) in sum.iter_mut().zip(&out) {
                        *s += share * o;
                    }
                }
                cpu::rms_norm(&sum, &m.post_norm, eps, &mut s.projected);
                add(&mut s.projected, &dense);
            }
            cpu::rms_norm(&s.projected, &layer.post_ffw_norm, eps, &mut s.h);
            add(&mut s.x, &s.h);

            // The per-layer input.
            if let Some((inp_gate, proj, post_norm)) = &layer.per_layer {
                cpu::matvec(pool, inp_gate, &s.x, &mut s.small);
                for (g, &p) in s.small.iter_mut().zip(&s.per_layer[l * pl..(l + 1) * pl]) {
                    *g = self.gelu(*g) * p;
                }
                cpu::matvec(pool, proj, &s.small, &mut s.projected);
                cpu::rms_norm(&s.projected, post_norm, eps, &mut s.h);
                add(&mut s.x, &s.h);
            }
            if layer.output_scale != 1.0 {
                s.x.iter_mut().for_each(|v| *v *= layer.output_scale);
            }
        }
        s.len += 1;

        if !want_logits {
            return None;
        }
        cpu::rms_norm(&s.x, &self.output_norm, eps, &mut s.h);
        cpu::matvec(pool, &self.embedding, &s.h, &mut s.logits);
        if let Some(cap) = c.softcap {
            s.logits
                .iter_mut()
                .for_each(|v| *v = cap * (*v / cap).tanh());
        }
        Some(&s.logits)
    }
}

/// RMSNorm of each head in place, times `weight` if given.
fn norm_heads(x: &mut [f32], head_dim: usize, weight: Option<&[f32]>, eps: f32) {
    let ones;
    let w = match weight {
        Some(w) => w,
        None => {
            ones = vec![1.0; head_dim];
            &ones
        }
    };
    let mut out = vec![0.0; head_dim];
    for head in x.chunks_exact_mut(head_dim) {
        cpu::rms_norm(head, w, eps, &mut out);
        head.copy_from_slice(&out);
    }
}

/// RoPE over whole heads, paired in halves (ggml's NEOX), with optional per-pair factors.
fn rope_neox(x: &mut [f32], head_dim: usize, pos: usize, base: f32, factors: Option<&[f32]>) {
    let half = head_dim / 2;
    for head in x.chunks_exact_mut(head_dim) {
        for j in 0..half {
            let mut theta = pos as f32 * base.powf(-2.0 * j as f32 / head_dim as f32);
            if let Some(f) = factors {
                theta /= f[j];
            }
            let (sin, cos) = theta.sin_cos();
            let (a, b) = (head[j], head[j + half]);
            head[j] = a * cos - b * sin;
            head[j + half] = a * sin + b * cos;
        }
    }
}

/// GELU, tanh approximation (ggml's `ggml_gelu`).
fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + 0.044_715 * x * x)).tanh())
}

fn add(x: &mut [f32], y: &[f32]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += b;
    }
}
