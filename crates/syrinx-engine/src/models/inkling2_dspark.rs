//! The DSpark drafter of Inkling 2 Mini (`general.architecture = "dflash"`, Unsloth's
//! `dspark-Inkling-2-Mini-*.gguf`): a block drafter that guesses up to 7 tokens after the
//! last one Inkling chose, for Inkling to check in one batch. The CPU reference, after
//! llama.cpp PR #32218 (`graph_inkling2` in `src/models/dspark.cpp`; a copy in
//! `reference/inkling2-llama/pr32218.diff`).
//!
//! **Features.** For every token Inkling reads, four rows of `dim`: the residual stream
//! going into layer visits `target_layers[0..3]` (5, 23 and 41 of the 48) and the final
//! normed hidden state (`target_layers[3]` = 48). They're mixed per draft layer `l` by
//! `softmax(fuse[l])`, RMS-normed by `dflare_norm[l]`, and become that layer's keys
//! (`attn_k_ctx`, a per-head RMS norm, RoPE at the token's position) and values
//! (`attn_v_ctx`) in the drafter's own cache. The fourth mix (`fuse[layers]`, normed by
//! `dspark_norm`) is the recurrent head's context.
//!
//! **A draft.** At position `p` (the anchor: the last token Inkling chose, not read yet),
//! a block of 8 rows: the anchor's embedding normed as Inkling's (`token_embd_norm`), then
//! `mask_embd` seven times, at positions `p..p + 8`. Three layers of pre-norm attention
//! (q and k each RMS-normed per head, NeoX RoPE, base 1e6, scale 1/√128, no causal mask:
//! every row sees the whole block and the cached positions before `p`, the last 1024 on
//! sliding-window layers) and a SwiGLU feed-forward, then `output_norm`.
//!
//! **The recurrent head**, for slots 1, 2, … in turn: `joint = joint_proj · [state,
//! markov_w1[prev], hidden_slot, context] + b`, thirds g, c, z; `state = tanh(c) +
//! sigmoid(g) (state − tanh(c))`; the slot's logits are `softcap(output · (hidden_slot +
//! hidden_out · tanh(z) + b), 30)` with Inkling's output matrix (llama.cpp adds the two
//! products; one product of the sum is the same arithmetic, and a draft only changes
//! speed). The guess is the first maximum; `prev` starts as the anchor.

use super::inkling2::Inkling2;
use super::{Error, Result, matrix, vector};
use crate::GgufFile;
use crate::cpu::{self, Matrix, Pool};

/// The drafter's settings, from the file.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub layers: usize,
    pub dim: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub block: usize,
    /// Inkling's layer visits the features come from (the last is the final hidden state).
    pub target_layers: Vec<usize>,
    pub rank: usize,
    pub window: usize,
    pub is_swa: Vec<bool>,
    pub rope_base: f32,
    pub norm_eps: f32,
    pub softcap: f32,
    pub vocab: usize,
}

impl Config {
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        if file.architecture() != Some("dflash") || file.tensor("dspark_hidden_out.weight").is_none() {
            return Err(Error::Unsupported(
                "this isn't an Inkling 2 DSpark drafter".into(),
            ));
        }
        let int = |k: &str| super::required(file, k);
        let layers = int("block_count")?;
        let target_layers = file
            .get("dflash.target_layers")
            .and_then(|v| v.as_array())
            .and_then(|a| a.iter().map(|v| v.as_u64().map(|n| n as usize)).collect::<Option<Vec<_>>>())
            .ok_or_else(|| Error::Missing("dflash.target_layers".into()))?;
        let is_swa = file
            .get("dflash.attention.sliding_window_pattern")
            .and_then(|v| v.as_array())
            .and_then(|a| a.iter().map(|v| v.as_bool()).collect::<Option<Vec<_>>>())
            .ok_or_else(|| Error::Missing("dflash.attention.sliding_window_pattern".into()))?;
        let f = |k: &str| {
            file.arch_f64(k)
                .map(|v| v as f32)
                .ok_or_else(|| Error::Missing(format!("dflash.{k}")))
        };
        let c = Config {
            layers,
            dim: int("embedding_length")?,
            heads: int("attention.head_count")?,
            kv_heads: int("attention.head_count_kv")?,
            head_dim: int("attention.key_length")?,
            ffn: int("feed_forward_length")?,
            block: int("block_size")?,
            target_layers,
            rank: int("dspark.markov_rank")?,
            window: int("attention.sliding_window")?,
            is_swa,
            rope_base: f("rope.freq_base")?,
            norm_eps: f("attention.layer_norm_rms_epsilon")?,
            softcap: f("final_logit_softcapping")?,
            vocab: int("vocab_size_unpadded")?,
        };
        if c.is_swa.len() != c.layers
            || c.target_layers.len() != 4
            || !c.heads.is_multiple_of(c.kv_heads)
            || !c.head_dim.is_multiple_of(2)
            || c.block < 2
        {
            return Err(Error::Invalid(format!("unexpected drafter settings {c:?}")));
        }
        Ok(c)
    }

    fn q_dim(&self) -> usize {
        self.heads * self.head_dim
    }

    fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// The features of one token, in numbers.
    pub fn feature_len(&self) -> usize {
        self.target_layers.len() * self.dim
    }
}

struct Layer<'a> {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    dflare_norm: Vec<f32>,
    q: Matrix<'a>,
    k: Matrix<'a>,
    v: Matrix<'a>,
    o: Matrix<'a>,
    k_ctx: Matrix<'a>,
    v_ctx: Matrix<'a>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    k_ctx_norm: Vec<f32>,
    gate: Matrix<'a>,
    up: Matrix<'a>,
    down: Matrix<'a>,
}

pub struct DSpark<'a> {
    pub config: Config,
    layers: Vec<Layer<'a>>,
    /// `[layers + 1][targets]` mixing logits.
    fuse: Vec<f32>,
    mask: Vec<f32>,
    output_norm: Vec<f32>,
    head_norm: Vec<f32>,
    markov: Matrix<'a>,
    joint: Matrix<'a>,
    joint_bias: Vec<f32>,
    hidden_out: Matrix<'a>,
    hidden_out_bias: Vec<f32>,
    pool: Pool,
}

/// What the drafter keeps for a conversation: each layer's keys and values a position,
/// and each position's recurrent-head context.
pub struct DraftState {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    context: Vec<f32>,
    /// Positions with features so far (positions 0..len).
    len: usize,
    capacity: usize,
}

impl DraftState {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Forgets positions from `len` on (Inkling took back tokens it read).
    pub fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }
}

impl<'a> DSpark<'a> {
    pub fn load(file: &'a GgufFile, pool: Pool) -> Result<Self> {
        let c = Config::from_gguf(file)?;
        let (dim, q, kv) = (c.dim, c.q_dim(), c.kv_dim());
        let name = |i: usize, n: &str| format!("blk.{i}.{n}.weight");
        let layers = (0..c.layers)
            .map(|i| {
                Ok(Layer {
                    attn_norm: vector(file, &name(i, "attn_norm"), dim)?,
                    ffn_norm: vector(file, &name(i, "ffn_norm"), dim)?,
                    dflare_norm: vector(file, &name(i, "dflare_norm"), dim)?,
                    q: matrix(file, &name(i, "attn_q"), dim, q)?,
                    k: matrix(file, &name(i, "attn_k"), dim, kv)?,
                    v: matrix(file, &name(i, "attn_v"), dim, kv)?,
                    o: matrix(file, &name(i, "attn_output"), q, dim)?,
                    k_ctx: matrix(file, &name(i, "attn_k_ctx"), dim, kv)?,
                    v_ctx: matrix(file, &name(i, "attn_v_ctx"), dim, kv)?,
                    q_norm: vector(file, &name(i, "attn_q_norm"), c.head_dim)?,
                    k_norm: vector(file, &name(i, "attn_k_norm"), c.head_dim)?,
                    k_ctx_norm: vector(file, &name(i, "attn_k_ctx_norm"), c.head_dim)?,
                    gate: matrix(file, &name(i, "ffn_gate"), dim, c.ffn)?,
                    up: matrix(file, &name(i, "ffn_up"), dim, c.ffn)?,
                    down: matrix(file, &name(i, "ffn_down"), c.ffn, dim)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let targets = c.target_layers.len();
        let fuse_m = matrix(file, "dspark_fuse.weight", targets, c.layers + 1)?;
        let mut fuse = vec![0.0; targets * (c.layers + 1)];
        for (r, row) in fuse.chunks_exact_mut(targets).enumerate() {
            fuse_m.decode_row(r, row);
        }
        let markov = {
            let t = file
                .tensor("markov_w1.weight")
                .ok_or_else(|| Error::Missing("tensor markov_w1.weight".into()))?;
            let rows = t.dims.get(1).copied().unwrap_or(0) as usize;
            matrix(file, "markov_w1.weight", c.rank, rows)?
        };
        let joint_in = 2 * c.rank + 2 * dim;
        Ok(DSpark {
            layers,
            fuse,
            mask: vector(file, "mask_embd.weight", dim)?,
            output_norm: vector(file, "output_norm.weight", dim)?,
            head_norm: vector(file, "dspark_norm.weight", dim)?,
            markov,
            joint: matrix(file, "joint_proj.weight", joint_in, 3 * c.rank)?,
            joint_bias: vector(file, "joint_proj.bias", 3 * c.rank)?,
            hidden_out: matrix(file, "dspark_hidden_out.weight", c.rank, dim)?,
            hidden_out_bias: vector(file, "dspark_hidden_out.bias", dim)?,
            config: c,
            pool,
        })
    }

    /// Checks that the drafter was made for `target` (its features' layer visits and size).
    pub fn check_target(&self, target: &Inkling2) -> Result<()> {
        let (c, t) = (&self.config, &target.config);
        let visits = t.slots();
        if t.dim != c.dim
            || c.target_layers.last() != Some(&visits)
            || c.target_layers.windows(2).any(|w| w[0] >= w[1])
            || t.vocab < c.vocab
        {
            return Err(Error::Invalid(format!(
                "the drafter's features ({:?}, {} wide) don't fit this Inkling ({visits} visits, {} wide)",
                c.target_layers, c.dim, t.dim
            )));
        }
        Ok(())
    }

    pub fn state(&self, capacity: usize) -> DraftState {
        let c = &self.config;
        DraftState {
            keys: (0..c.layers).map(|_| vec![0.0; capacity * c.kv_dim()]).collect(),
            values: (0..c.layers).map(|_| vec![0.0; capacity * c.kv_dim()]).collect(),
            context: vec![0.0; capacity * c.dim],
            len: 0,
            capacity,
        }
    }

    /// `softmax(fuse[row])`-weighted sum of a token's feature rows, RMS-normed by `norm`.
    fn mix(&self, features: &[f32], row: usize, norm: &[f32]) -> Vec<f32> {
        let c = &self.config;
        let n = c.target_layers.len();
        let mut w = self.fuse[row * n..(row + 1) * n].to_vec();
        cpu::softmax(&mut w);
        let mut x = vec![0.0; c.dim];
        for (k, wk) in w.iter().enumerate() {
            for (xi, f) in x.iter_mut().zip(&features[k * c.dim..(k + 1) * c.dim]) {
                *xi += wk * f;
            }
        }
        let mut out = vec![0.0; c.dim];
        cpu::rms_norm(&x, norm, c.norm_eps, &mut out);
        out
    }

    /// The features (`Config::feature_len` numbers) of the token Inkling read at position
    /// `s.len()`, into the drafter's cache.
    pub fn add(&self, s: &mut DraftState, features: &[f32]) {
        let c = &self.config;
        let pos = s.len;
        assert!(pos < s.capacity, "the drafter's cache is full");
        assert_eq!(features.len(), c.feature_len());
        let kv = c.kv_dim();
        for (l, layer) in self.layers.iter().enumerate() {
            let context = self.mix(features, l, &layer.dflare_norm);
            let k = &mut s.keys[l][pos * kv..(pos + 1) * kv];
            cpu::matvec(&self.pool, &layer.k_ctx, &context, k);
            head_norm(k, c.head_dim, &layer.k_ctx_norm, c.norm_eps);
            rope_neox(k, c.head_dim, pos, c.rope_base);
            let v = &mut s.values[l][pos * kv..(pos + 1) * kv];
            cpu::matvec(&self.pool, &layer.v_ctx, &context, v);
        }
        let context = self.mix(features, c.layers, &self.head_norm);
        s.context[pos * c.dim..(pos + 1) * c.dim].copy_from_slice(&context);
        s.len += 1;
    }

    /// Up to `count` (at most `block − 1`) guesses of the tokens after `anchor`, the token
    /// Inkling chose for position `s.len()` (which it hasn't read yet).
    pub fn draft(&self, s: &DraftState, target: &Inkling2, anchor: u32, count: usize) -> Vec<u32> {
        let c = &self.config;
        let (dim, hd) = (c.dim, c.head_dim);
        let (q_dim, kv) = (c.q_dim(), c.kv_dim());
        let rows = c.block;
        let count = count.min(rows - 1);
        let pos = s.len;
        assert!(pos > 0, "the drafter needs the features of the token before the anchor");

        // The block: the anchor's embedding (normed as Inkling's), then the mask rows.
        let mut x = vec![0.0; rows * dim];
        {
            let mut raw = vec![0.0; dim];
            target.embedding.decode_row(anchor as usize, &mut raw);
            cpu::rms_norm(&raw, &target.embedding_norm, target.config.norm_eps, &mut x[..dim]);
        }
        for r in 1..rows {
            x[r * dim..(r + 1) * dim].copy_from_slice(&self.mask);
        }

        let group = c.heads / c.kv_heads;
        let scale = 1.0 / (hd as f32).sqrt();
        for (l, layer) in self.layers.iter().enumerate() {
            let first = if c.is_swa[l] { pos.saturating_sub(c.window) } else { 0 };
            let cached = pos - first;
            let mut q = vec![0.0; rows * q_dim];
            let mut k = vec![0.0; rows * kv];
            let mut v = vec![0.0; rows * kv];
            for r in 0..rows {
                let mut h = vec![0.0; dim];
                cpu::rms_norm(&x[r * dim..(r + 1) * dim], &layer.attn_norm, c.norm_eps, &mut h);
                let (qr, kr, vr) = (
                    &mut q[r * q_dim..(r + 1) * q_dim],
                    &mut k[r * kv..(r + 1) * kv],
                    &mut v[r * kv..(r + 1) * kv],
                );
                cpu::matvec(&self.pool, &layer.q, &h, qr);
                cpu::matvec(&self.pool, &layer.k, &h, kr);
                cpu::matvec(&self.pool, &layer.v, &h, vr);
                head_norm(qr, hd, &layer.q_norm, c.norm_eps);
                head_norm(kr, hd, &layer.k_norm, c.norm_eps);
                rope_neox(qr, hd, pos + r, c.rope_base);
                rope_neox(kr, hd, pos + r, c.rope_base);
            }
            let (ck, cv) = (&s.keys[l], &s.values[l]);
            let key = |j: usize| -> &[f32] {
                if j < cached {
                    let p = first + j;
                    &ck[p * kv..(p + 1) * kv]
                } else {
                    let r = j - cached;
                    &k[r * kv..(r + 1) * kv]
                }
            };
            let value = |j: usize| -> &[f32] {
                if j < cached {
                    let p = first + j;
                    &cv[p * kv..(p + 1) * kv]
                } else {
                    let r = j - cached;
                    &v[r * kv..(r + 1) * kv]
                }
            };
            let n = cached + rows;
            let mut attn = vec![0.0; rows * q_dim];
            let mut scores = vec![0.0; n];
            for r in 0..rows {
                for head in 0..c.heads {
                    let g = head / group;
                    let qh = &q[r * q_dim + head * hd..r * q_dim + (head + 1) * hd];
                    for (j, sj) in scores.iter_mut().enumerate() {
                        let kh = &key(j)[g * hd..(g + 1) * hd];
                        *sj = qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() * scale;
                    }
                    cpu::softmax(&mut scores);
                    let out = &mut attn[r * q_dim + head * hd..r * q_dim + (head + 1) * hd];
                    for (j, p) in scores.iter().enumerate() {
                        let vh = &value(j)[g * hd..(g + 1) * hd];
                        for (o, vv) in out.iter_mut().zip(vh) {
                            *o += p * vv;
                        }
                    }
                }
            }
            for r in 0..rows {
                let xr = &mut x[r * dim..(r + 1) * dim];
                let mut o = vec![0.0; dim];
                cpu::matvec(&self.pool, &layer.o, &attn[r * q_dim..(r + 1) * q_dim], &mut o);
                xr.iter_mut().zip(&o).for_each(|(a, b)| *a += b);
                let mut h = vec![0.0; dim];
                cpu::rms_norm(xr, &layer.ffn_norm, c.norm_eps, &mut h);
                let mut gate = vec![0.0; c.ffn];
                let mut up = vec![0.0; c.ffn];
                cpu::matvec(&self.pool, &layer.gate, &h, &mut gate);
                cpu::matvec(&self.pool, &layer.up, &h, &mut up);
                cpu::swiglu(&mut gate, &up);
                cpu::matvec(&self.pool, &layer.down, &gate, &mut o);
                xr.iter_mut().zip(&o).for_each(|(a, b)| *a += b);
            }
        }

        // The recurrent head, a slot at a time.
        let rank = c.rank;
        let context = &s.context[(pos - 1) * dim..pos * dim];
        let mut state = vec![0.0; rank];
        let mut prev = anchor;
        let mut guesses = Vec::with_capacity(count);
        let padded = target.output.rows;
        let mut logits = vec![0.0; padded];
        for slot in 1..=count {
            let mut hidden = vec![0.0; dim];
            cpu::rms_norm(&x[slot * dim..(slot + 1) * dim], &self.output_norm, c.norm_eps, &mut hidden);
            let mut joint_in = Vec::with_capacity(2 * rank + 2 * dim);
            joint_in.extend_from_slice(&state);
            let mut w1 = vec![0.0; rank];
            self.markov.decode_row(prev as usize, &mut w1);
            joint_in.extend_from_slice(&w1);
            joint_in.extend_from_slice(&hidden);
            joint_in.extend_from_slice(context);
            let mut j = vec![0.0; 3 * rank];
            cpu::matvec(&self.pool, &self.joint, &joint_in, &mut j);
            j.iter_mut().zip(&self.joint_bias).for_each(|(a, b)| *a += b);
            let mut latent = vec![0.0; rank];
            for i in 0..rank {
                let gate = 1.0 / (1.0 + (-j[i]).exp());
                let cand = j[rank + i].tanh();
                state[i] = cand + gate * (state[i] - cand);
                latent[i] = j[2 * rank + i].tanh();
            }
            let mut corr = vec![0.0; dim];
            cpu::matvec(&self.pool, &self.hidden_out, &latent, &mut corr);
            for i in 0..dim {
                hidden[i] += corr[i] + self.hidden_out_bias[i];
            }
            cpu::matvec(&self.pool, &target.output, &hidden, &mut logits);
            // The first maximum (softcap is monotonic, so it doesn't change the choice).
            let mut best = 0usize;
            for (i, &v) in logits[..c.vocab].iter().enumerate() {
                if v > logits[best] {
                    best = i;
                }
            }
            prev = best as u32;
            guesses.push(prev);
        }
        guesses
    }
}

/// RMS norm of each `head_dim` slice of `x` by `weight`, in place.
fn head_norm(x: &mut [f32], head_dim: usize, weight: &[f32], eps: f32) {
    for h in x.chunks_exact_mut(head_dim) {
        let copy = h.to_vec();
        cpu::rms_norm(&copy, weight, eps, h);
    }
}

/// NeoX RoPE (pairs i, i + head_dim / 2) over whole heads.
fn rope_neox(x: &mut [f32], head_dim: usize, pos: usize, base: f32) {
    let half = head_dim / 2;
    for head in x.chunks_exact_mut(head_dim) {
        for j in 0..half {
            let theta = pos as f32 * base.powf(-2.0 * j as f32 / head_dim as f32);
            let (sin, cos) = theta.sin_cos();
            let (a, b) = (head[j], head[j + half]);
            head[j] = a * cos - b * sin;
            head[j + half] = a * sin + b * cos;
        }
    }
}
