//! Inkling 2 (Thinking Machines; `inkling2`, and the looped `inkling2_loop` of Inkling 2
//! Mini): the forward pass, on the CPU reference backend. Follows llama.cpp's
//! `src/models/inkling2.cpp` and `inkling2-loop.cpp` (ggml-org/llama.cpp PR #32217, the
//! attention-mask fix #32429 and the looped model #32694; copies in `reference/inkling2-llama`).
//!
//! There is no RoPE: a learned bias on each query–key distance carries the positions.
//! Layers come in blocks of `attn_res.block_size` (6). Instead of one residual stream,
//! each block keeps its input, and the layers read a learned mix of the block inputs so far
//! ("attention over depth", [`mix`]):
//!
//! ```text
//! x0 = RMSNorm(token_embd[t]) × token_embd_norm
//! per layer (`blocks`: the inputs of every block so far):
//!   in    = mix(blocks, x, attn_res_query)       one mix (sliding window) or 3 (global: q, k, v)
//!   h     = RMSNorm(in) × attn_norm               global layers: a norm each for q, k and v
//!   q     = clip(Wq h × q_clip, 22.6)             per head: scaled to an L2 norm of at most 22.6
//!   k     = clip(conv(Wk h), 22.6),  v = RMSNorm(conv(Wv h))     per head, no weight
//!   r     = clip(Wr h × rel_clip, 8)              16 numbers per head
//!   global layers: q, r × (1 + 0.1 × ln(max(1, (pos + 1)/8192)))
//!   score = q·k/sqrt(128) + r·rel_bias[distance] − sink(h)       and a sink logit of 0
//!   a     = conv(Wo (softmax(score) · v × sigmoid(W_gate h)))
//!   first layer of a block: blocks += x; x = a.  Others: x = x + a.
//!   f     = conv(experts(RMSNorm(mix(blocks, x, ffn_res_query)) × ffn_norm))
//!   x     = x + f
//! out = softcap(output · RMSNorm(mix(blocks, x, output_res_query)) × output_norm, 30)
//! ```
//!
//! `conv` is a causal convolution over the last 4 tokens with a residual, scaled by
//! 1/sqrt(5) (global layers' keys and values add 4 taps per head computed from the input,
//! 1/sqrt(9)). Each layer keeps the last 3 inputs of its 4 convolutions.
//!
//! The experts: a sigmoid score per expert, the 8 highest after a learned bias picked,
//! their scores normalized to sum 1 and times 8; each expert is
//! `down(4 tanh(g/4) sigmoid(g) × 25 tanh(u/25))`, `g`, `u` the two halves of `gate_up`.
//!
//! The looped model (`loop.start`..`loop.end`, `loop.count` passes) runs the core layers
//! again from the prelude's output, with the weights shared. From the second pass on, each
//! core layer also reads messages from the previous pass: a mix of its blocks after the
//! first core block and its output, with 4 learned queries (q, k, v and the experts' input),
//! gated into those inputs by a small controller ([`control`]). Every visit of a layer
//! keeps its own keys, values and convolution history (`Slot`s: the 24 physical layers,
//! then the core's later passes).

use super::{Error, Result, matrices, matrix, vector};
use crate::GgufFile;
use crate::cpu::{self, Matrix, Pool};
use crate::gguf::Value;
use crate::quant;

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// Physical layers (each core layer counted once).
    pub layers: usize,
    pub dim: usize,
    /// Tokens the tokenizer knows; the embedding and output have `padded_vocab` rows.
    pub vocab: usize,
    pub padded_vocab: usize,
    pub norm_eps: f32,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub is_swa: Vec<bool>,
    pub sliding_window: usize,
    /// Layers per residual block.
    pub block: usize,
    /// Taps of the short convolutions (4).
    pub conv: usize,
    /// The relative-position bias: numbers per head, and distances with their own bias on
    /// global layers (sliding-window layers: `sliding_window`).
    pub rel_dim: usize,
    pub rel_extent: usize,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub log_scale_floor: f32,
    pub log_scale_alpha: f32,
    pub experts: usize,
    pub experts_used: usize,
    pub expert_width: usize,
    pub expert_scale: f32,
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    pub softcap: Option<f32>,
    /// The looped core, if any.
    pub looped: Option<Loop>,
}

/// The looped core: physical layers `start..end` run `passes` times.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Loop {
    pub start: usize,
    pub end: usize,
    pub passes: usize,
    pub gate_rank: usize,
}

impl Config {
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let arch = file.architecture().unwrap_or("");
        if arch != "inkling2" && arch != "inkling2_loop" {
            return Err(Error::Unsupported(format!(
                "The {arch:?} architecture with the Inkling 2 forward pass"
            )));
        }
        let key = |k: &str| format!("{arch}.{k}");
        let num = |k: &str| -> Result<f64> {
            file.get(&key(k))
                .and_then(|v| match v.as_array() {
                    // Per-layer arrays: Inkling's are the same on every layer.
                    Some(items) => items.first().and_then(Value::as_f64),
                    None => v.as_f64(),
                })
                .ok_or_else(|| Error::Missing(format!("{arch}.{k} setting")))
        };
        let int = |k: &str| num(k).map(|v| v as usize);
        let same_everywhere = |k: &str| -> Result<()> {
            if let Some(items) = file.get(&key(k)).and_then(Value::as_array)
                && items.windows(2).any(|w| w[0].as_f64() != w[1].as_f64())
            {
                return Err(Error::Unsupported(format!(
                    "Inkling 2 with {k} differing between layers"
                )));
            }
            Ok(())
        };
        same_everywhere("attention.head_count")?;
        same_everywhere("attention.head_count_kv")?;
        same_everywhere("expert_feed_forward_length")?;

        let layers = int("block_count")?;
        let is_swa: Vec<bool> = file
            .get(&key("attention.sliding_window_pattern"))
            .and_then(Value::as_array)
            .ok_or_else(|| Error::Missing(format!("{arch}.attention.sliding_window_pattern")))?
            .iter()
            .map(|v| v.as_bool().unwrap_or(false))
            .collect();
        if is_swa.len() != layers {
            return Err(Error::Invalid(format!(
                "{} sliding-window flags for {layers} layers",
                is_swa.len()
            )));
        }
        let head_dim = int("attention.key_length")?;
        if int("attention.value_length")? != head_dim
            || file
                .arch_u64("attention.key_length_swa")
                .is_some_and(|v| v as usize != head_dim)
        {
            return Err(Error::Unsupported(
                "Inkling 2 with heads of different sizes".into(),
            ));
        }
        let vocab = file
            .get("tokenizer.ggml.tokens")
            .and_then(Value::as_array)
            .map(<[_]>::len)
            .ok_or_else(|| Error::Missing("vocabulary".into()))?;
        let looped = if arch == "inkling2_loop" {
            let varying = file
                .get(&key("loop.varying"))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if varying {
                return Err(Error::Unsupported(
                    "a looped model with a varying pass count".into(),
                ));
            }
            Some(Loop {
                start: int("loop.start")?,
                end: int("loop.end")?,
                passes: int("loop.count")?,
                gate_rank: int("loop.gate_rank")?,
            })
        } else {
            None
        };
        let c = Self {
            layers,
            dim: int("embedding_length")?,
            vocab,
            padded_vocab: int("vocab_size")?,
            norm_eps: num("attention.layer_norm_rms_epsilon")? as f32,
            heads: int("attention.head_count")?,
            kv_heads: int("attention.head_count_kv")?,
            head_dim,
            is_swa,
            sliding_window: int("attention.sliding_window")?,
            block: int("attn_res.block_size")?,
            conv: int("shortconv.l_cache")?,
            rel_dim: int("attention.relative_length")?,
            rel_extent: int("attention.relative_extent")?,
            qk_clip: num("attention.qk_clip")? as f32,
            rel_clip: num("attention.relative_clip")? as f32,
            log_scale_floor: num("attention.log_scale_floor")? as f32,
            log_scale_alpha: num("attention.log_scale_alpha")? as f32,
            experts: int("expert_count")?,
            experts_used: int("expert_used_count")?,
            expert_width: int("expert_feed_forward_length")?,
            expert_scale: file.arch_f64("expert_weights_scale").unwrap_or(1.0) as f32,
            situ_beta: num("activation.situ_beta")? as f32,
            situ_linear_beta: num("activation.situ_linear_beta")? as f32,
            softcap: file
                .arch_f64("final_logit_softcapping")
                .filter(|&v| v != 0.0)
                .map(|v| v as f32),
            looped,
        };
        if let Some(unpadded) = file.arch_u64("vocab_size_unpadded")
            && unpadded as usize != c.vocab
        {
            return Err(Error::Invalid(format!(
                "the tokenizer has {} tokens, the model {unpadded}",
                c.vocab
            )));
        }
        if c.vocab > c.padded_vocab
            || c.kv_heads == 0
            || !c.heads.is_multiple_of(c.kv_heads)
            || c.block == 0
            || c.conv < 2
            || c.experts_used == 0
            || c.experts_used > c.experts
            || c.rel_extent == 0
            || c.sliding_window == 0
        {
            return Err(Error::Invalid(
                "the attention, convolution or expert settings don't fit each other".into(),
            ));
        }
        if let Some(l) = c.looped
            && (l.start == 0
                || l.start >= l.end
                || l.end > layers
                || !l.start.is_multiple_of(c.block)
                || !l.end.is_multiple_of(c.block)
                || l.passes < 2
                || l.gate_rank == 0)
        {
            return Err(Error::Invalid(format!(
                "the loop {l:?} doesn't fit the layers"
            )));
        }
        Ok(c)
    }

    /// Layer visits per token: each layer once, the core `passes` times.
    pub fn slots(&self) -> usize {
        match self.looped {
            Some(l) => self.layers + (l.passes - 1) * (l.end - l.start),
            None => self.layers,
        }
    }

    fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// The channels a layer's convolutions keep history for: keys, values, the attention's
    /// output and the experts'.
    fn conv_channels(&self) -> usize {
        2 * self.kv_dim() + 2 * self.dim
    }
}

pub(super) struct Layer<'a> {
    /// The mix queries of the attention input: none on layer 0, one on a sliding-window
    /// layer, one each for q, k and v on a global layer.
    pub(super) attn_res_query: Vec<Vec<f32>>,
    pub(super) ffn_res_query: Vec<f32>,
    /// The input norm (sliding-window layers), or q's, k's and v's (global layers).
    pub(super) norms: Vec<Vec<f32>>,
    pub(super) q: Matrix<'a>,
    pub(super) k: Matrix<'a>,
    pub(super) v: Matrix<'a>,
    pub(super) o: Matrix<'a>,
    pub(super) gate: Matrix<'a>,
    pub(super) rel: Matrix<'a>,
    pub(super) sink: Matrix<'a>,
    /// Per head: q's scale before the clip (head_dim each), r's (rel_dim each).
    pub(super) q_clip: Vec<f32>,
    pub(super) rel_clip: Vec<f32>,
    /// The bias per distance, rel_dim numbers each, with a row of zeros after the last.
    pub(super) rel_bias: Vec<f32>,
    /// Each channel's taps, oldest input first.
    pub(super) k_conv: Vec<f32>,
    pub(super) v_conv: Vec<f32>,
    pub(super) attn_post_conv: Vec<f32>,
    pub(super) ffn_post_conv: Vec<f32>,
    /// Global layers: the input-dependent taps of the key and value convolutions.
    pub(super) k_conv_proj: Option<Matrix<'a>>,
    pub(super) v_conv_proj: Option<Matrix<'a>>,
    pub(super) ffn_norm: Vec<f32>,
    pub(super) router: Matrix<'a>,
    pub(super) router_bias: Vec<f32>,
    pub(super) gate_up: Matrix<'a>,
    pub(super) down: Matrix<'a>,
    /// A core layer of a looped model: its message queries and controllers.
    pub(super) looped: Option<LoopLayer<'a>>,
}

/// A looped core layer's message queries (q, k, v, experts' input) and the controller of
/// each, which gates its message into the input.
pub(super) struct LoopLayer<'a> {
    pub(super) queries: Vec<Vec<f32>>,
    pub(super) controllers: [Controller<'a>; 4],
}

pub(super) struct Controller<'a> {
    pub(super) down: Matrix<'a>,
    pub(super) gain: Matrix<'a>,
    pub(super) forget: Matrix<'a>,
}

pub struct Inkling2<'a> {
    pub config: Config,
    pub(super) embedding: Matrix<'a>,
    pub(super) embedding_norm: Vec<f32>,
    pub(super) output: Matrix<'a>,
    pub(super) output_norm: Vec<f32>,
    pub(super) output_res_query: Vec<f32>,
    pub(super) layers: Vec<Layer<'a>>,
    pub(super) pool: Pool,
}

/// One visit of a layer's state in a conversation.
struct Slot {
    keys: Vec<f32>,
    values: Vec<f32>,
    /// The last `conv − 1` inputs of the 4 convolutions, oldest first:
    /// `[conv − 1][conv_channels]`, channels as in `Config::conv_channels`.
    history: Vec<f32>,
}

pub struct Session {
    slots: Vec<Slot>,
    len: usize,
    capacity: usize,
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

/// A tensor of `rows` vectors of `len` numbers, decoded (GGUF `[len]` for one row, or
/// `[len, rows]`).
fn rows_f32(file: &GgufFile, name: &str, len: usize, rows: usize) -> Result<Vec<Vec<f32>>> {
    let t = file
        .tensor(name)
        .ok_or_else(|| Error::Missing(format!("tensor {name}")))?;
    if rows == 1 && t.dims == [len as u64] {
        return Ok(vec![vector(file, name, len)?]);
    }
    let m = matrix(file, name, len, rows)?;
    Ok((0..rows)
        .map(|r| {
            let mut out = vec![0.0; len];
            m.decode_row(r, &mut out);
            out
        })
        .collect())
}

fn flat_f32(file: &GgufFile, name: &str, len: usize, rows: usize) -> Result<Vec<f32>> {
    Ok(rows_f32(file, name, len, rows)?.concat())
}

impl<'a> Inkling2<'a> {
    pub fn load(file: &'a GgufFile, pool: Pool) -> Result<Self> {
        let c = Config::from_gguf(file)?;
        let (dim, hd, conv) = (c.dim, c.head_dim, c.conv);
        let (q_dim, kv_dim) = (c.heads * hd, c.kv_dim());
        let mut layers = Vec::with_capacity(c.layers);
        for i in 0..c.layers {
            let name = |part: &str| format!("blk.{i}.{part}.weight");
            let swa = c.is_swa[i];
            let extent = if swa { c.sliding_window } else { c.rel_extent };
            let mut rel_bias = flat_f32(file, &name("attn_rel_bias"), c.rel_dim, extent)?;
            rel_bias.extend(std::iter::repeat_n(0.0, c.rel_dim));
            let looped = match c.looped {
                Some(l) if (l.start..l.end).contains(&i) => {
                    let controller = |kind: &str| -> Result<Controller<'a>> {
                        Ok(Controller {
                            down: matrix(
                                file,
                                &name(&format!("loop_{kind}_ctrl_down")),
                                2 * dim,
                                l.gate_rank,
                            )?,
                            gain: matrix(
                                file,
                                &name(&format!("loop_{kind}_ctrl_gain")),
                                l.gate_rank,
                                dim,
                            )?,
                            forget: matrix(
                                file,
                                &name(&format!("loop_{kind}_ctrl_forget")),
                                l.gate_rank,
                                dim,
                            )?,
                        })
                    };
                    Some(LoopLayer {
                        queries: rows_f32(file, &name("loop_msg_query"), dim, 4)?,
                        controllers: [
                            controller("q")?,
                            controller("k")?,
                            controller("v")?,
                            controller("ffn")?,
                        ],
                    })
                }
                _ => None,
            };
            layers.push(Layer {
                attn_res_query: if i == 0 {
                    Vec::new()
                } else {
                    rows_f32(file, &name("attn_res_query"), dim, if swa { 1 } else { 3 })?
                },
                ffn_res_query: vector(file, &name("ffn_res_query"), dim)?,
                norms: if swa {
                    vec![vector(file, &name("attn_norm"), dim)?]
                } else {
                    vec![
                        vector(file, &name("attn_q_inp_norm"), dim)?,
                        vector(file, &name("attn_k_inp_norm"), dim)?,
                        vector(file, &name("attn_v_inp_norm"), dim)?,
                    ]
                },
                q: matrix(file, &name("attn_q"), dim, q_dim)?,
                k: matrix(file, &name("attn_k"), dim, kv_dim)?,
                v: matrix(file, &name("attn_v"), dim, kv_dim)?,
                o: matrix(file, &name("attn_output"), q_dim, dim)?,
                gate: matrix(file, &name("attn_gate"), dim, q_dim)?,
                rel: matrix(file, &name("attn_rel"), dim, c.rel_dim * c.heads)?,
                sink: matrix(file, &name("attn_sink_proj"), dim, c.heads)?,
                q_clip: flat_f32(file, &name("attn_q_clip"), hd, c.heads)?,
                rel_clip: flat_f32(file, &name("attn_rel_clip"), c.rel_dim, c.heads)?,
                rel_bias,
                k_conv: flat_f32(file, &name("attn_k_conv"), conv, kv_dim)?,
                v_conv: flat_f32(file, &name("attn_v_conv"), conv, kv_dim)?,
                attn_post_conv: flat_f32(file, &name("attn_post_conv"), conv, dim)?,
                ffn_post_conv: flat_f32(file, &name("ffn_post_conv"), conv, dim)?,
                k_conv_proj: if swa {
                    None
                } else {
                    Some(matrix(
                        file,
                        &name("attn_k_conv_proj"),
                        dim,
                        conv * c.kv_heads,
                    )?)
                },
                v_conv_proj: if swa {
                    None
                } else {
                    Some(matrix(
                        file,
                        &name("attn_v_conv_proj"),
                        dim,
                        conv * c.kv_heads,
                    )?)
                },
                ffn_norm: vector(file, &name("ffn_norm"), dim)?,
                router: matrix(file, &name("ffn_gate_inp"), dim, c.experts)?,
                router_bias: if file.tensor(&format!("blk.{i}.exp_probs_b.bias")).is_some() {
                    vector(file, &format!("blk.{i}.exp_probs_b.bias"), c.experts)?
                } else {
                    vec![0.0; c.experts]
                },
                gate_up: matrices(
                    file,
                    &name("ffn_gate_up_exps"),
                    dim,
                    2 * c.expert_width,
                    c.experts,
                )?,
                down: matrices(file, &name("ffn_down_exps"), c.expert_width, dim, c.experts)?,
                looped,
            });
        }
        Ok(Self {
            embedding: matrix(file, "token_embd.weight", dim, c.padded_vocab)?,
            embedding_norm: vector(file, "token_embd_norm.weight", dim)?,
            output: matrix(file, "output.weight", dim, c.padded_vocab)?,
            output_norm: vector(file, "output_norm.weight", dim)?,
            output_res_query: vector(file, "output_res_query.weight", dim)?,
            layers,
            config: c,
            pool,
        })
    }

    pub fn threads(&self) -> usize {
        self.pool.threads()
    }

    /// A new conversation that can hold `capacity` positions. Every layer visit keeps
    /// every position's keys and values (sliding-window layers too, for now).
    pub fn session(&self, capacity: usize) -> Session {
        let c = &self.config;
        Session {
            slots: (0..c.slots())
                .map(|_| Slot {
                    keys: vec![0.0; capacity * c.kv_dim()],
                    values: vec![0.0; capacity * c.kv_dim()],
                    history: vec![0.0; (c.conv - 1) * c.conv_channels()],
                })
                .collect(),
            len: 0,
            capacity,
            logits: vec![0.0; c.vocab],
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
        let dim = c.dim;

        let mut x = vec![0.0; dim];
        self.embedding.decode_row(token as usize, &mut x);
        let raw = x.clone();
        cpu::rms_norm(&raw, &self.embedding_norm, c.norm_eps, &mut x);

        // Global layers scale q and r up for positions past `log_scale_floor`.
        let tau = 1.0 + c.log_scale_alpha * ((pos as f32 + 1.0) / c.log_scale_floor).max(1.0).ln();

        let mut blocks: Vec<Vec<f32>> = Vec::new();
        match c.looped {
            None => {
                for l in 0..c.layers {
                    self.layer(s, l, l, pos, tau, &mut x, &mut blocks, None);
                }
            }
            Some(lp) => {
                for l in 0..lp.start {
                    self.layer(s, l, l, pos, tau, &mut x, &mut blocks, None);
                }
                // Every pass starts from the prelude's output.
                let (prelude, prelude_blocks) = (x.clone(), blocks.clone());
                let core = lp.end - lp.start;
                for pass in 0..lp.passes {
                    // The previous pass's blocks after its first core block, and its output.
                    let mut bank: Vec<Vec<f32>> = Vec::new();
                    if pass > 0 {
                        bank = blocks[prelude_blocks.len() + 1..].to_vec();
                        bank.push(std::mem::replace(&mut x, prelude.clone()));
                        blocks = prelude_blocks.clone();
                    }
                    for l in lp.start..lp.end {
                        let slot = if pass == 0 {
                            l
                        } else {
                            c.layers + (pass - 1) * core + (l - lp.start)
                        };
                        let messages = (pass > 0).then(|| {
                            let queries = &self.layers[l]
                                .looped
                                .as_ref()
                                .expect("a core layer")
                                .queries;
                            mix(&bank, queries)
                        });
                        self.layer(
                            s,
                            slot,
                            l,
                            pos,
                            tau,
                            &mut x,
                            &mut blocks,
                            messages.as_deref(),
                        );
                    }
                }
                for l in lp.end..c.layers {
                    self.layer(s, l, l, pos, tau, &mut x, &mut blocks, None);
                }
            }
        }
        s.len += 1;

        if !want_logits {
            return None;
        }
        blocks.push(x);
        let mixed = mix(&blocks, std::slice::from_ref(&self.output_res_query)).remove(0);
        let mut h = vec![0.0; dim];
        cpu::rms_norm(&mixed, &self.output_norm, c.norm_eps, &mut h);
        dump("result_norm", usize::MAX, &h);
        let mut logits = vec![0.0; c.padded_vocab];
        cpu::matvec(&self.pool, &self.output, &h, &mut logits);
        logits.truncate(c.vocab);
        if let Some(cap) = c.softcap {
            logits.iter_mut().for_each(|v| *v = cap * (*v / cap).tanh());
        }
        s.logits.copy_from_slice(&logits);
        Some(&s.logits)
    }

    /// One visit of physical layer `l`, keeping its state in `slot`. `messages` (q, k, v
    /// and the experts' input) on the looped core's later passes.
    #[allow(clippy::too_many_arguments)]
    fn layer(
        &self,
        s: &mut Session,
        slot: usize,
        l: usize,
        pos: usize,
        tau: f32,
        x: &mut Vec<f32>,
        blocks: &mut Vec<Vec<f32>>,
        messages: Option<&[Vec<f32>]>,
    ) {
        let c = &self.config;
        let layer = &self.layers[l];
        let pool = &self.pool;
        let (dim, hd, heads) = (c.dim, c.head_dim, c.heads);
        let (q_dim, kv_dim) = (heads * hd, c.kv_dim());
        let swa = c.is_swa[l];
        let eps = c.norm_eps;

        // The attention's inputs: q, k and v the same on a sliding-window layer, unless
        // messages make them differ.
        let mut inputs: Vec<Vec<f32>> = if layer.attn_res_query.is_empty() {
            vec![x.clone(); 3]
        } else {
            let mut states = blocks.clone();
            states.push(x.clone());
            let mixed = mix(&states, &layer.attn_res_query);
            if swa {
                vec![mixed[0].clone(); 3]
            } else {
                mixed
            }
        };
        let looped = messages.map(|m| (m, layer.looped.as_ref().expect("a core layer")));
        if let Some((m, lp)) = looped {
            for j in 0..3 {
                inputs[j] = control(pool, &inputs[j], &m[j], &lp.controllers[j]);
            }
        }
        let mut h = vec![vec![0.0; dim]; 3];
        for j in 0..3 {
            let norm = if swa {
                &layer.norms[0]
            } else {
                &layer.norms[j]
            };
            cpu::rms_norm(&inputs[j], norm, eps, &mut h[j]);
        }
        dump("norm", slot, &h[0]);

        // Projections, and the convolutions over the last tokens' keys and values.
        let history = &mut s.slots[slot].history;
        let channels = c.conv_channels();
        let mut q = vec![0.0; q_dim];
        let mut k = vec![0.0; kv_dim];
        let mut v = vec![0.0; kv_dim];
        let mut r = vec![0.0; c.rel_dim * heads];
        let mut sink = vec![0.0; heads];
        let mut gate = vec![0.0; q_dim];
        cpu::matvec(pool, &layer.q, &h[0], &mut q);
        cpu::matvec(pool, &layer.k, &h[1], &mut k);
        cpu::matvec(pool, &layer.v, &h[2], &mut v);
        cpu::matvec(pool, &layer.rel, &h[0], &mut r);
        cpu::matvec(pool, &layer.sink, &h[0], &mut sink);
        cpu::matvec(pool, &layer.gate, &h[0], &mut gate);
        let dynamic = |proj: &Option<Matrix<'_>>, input: &[f32]| {
            proj.as_ref().map(|p| {
                let mut taps = vec![0.0; c.conv * c.kv_heads];
                cpu::matvec(pool, p, input, &mut taps);
                taps
            })
        };
        let k_taps = dynamic(&layer.k_conv_proj, &h[1]);
        let v_taps = dynamic(&layer.v_conv_proj, &h[2]);
        convolve(
            history,
            channels,
            0,
            &mut k,
            &layer.k_conv,
            k_taps.as_deref(),
            c.conv,
        );
        convolve(
            history,
            channels,
            kv_dim,
            &mut v,
            &layer.v_conv,
            v_taps.as_deref(),
            c.conv,
        );

        for (head, qh) in q.chunks_exact_mut(hd).enumerate() {
            for (a, w) in qh.iter_mut().zip(&layer.q_clip[head * hd..(head + 1) * hd]) {
                *a *= w;
            }
            clip(qh, c.qk_clip);
        }
        k.chunks_exact_mut(hd).for_each(|kh| clip(kh, c.qk_clip));
        let ones = vec![1.0; hd];
        let mut normed = vec![0.0; hd];
        for vh in v.chunks_exact_mut(hd) {
            cpu::rms_norm(vh, &ones, eps, &mut normed);
            vh.copy_from_slice(&normed);
        }
        for (head, rh) in r.chunks_exact_mut(c.rel_dim).enumerate() {
            for (a, w) in rh
                .iter_mut()
                .zip(&layer.rel_clip[head * c.rel_dim..(head + 1) * c.rel_dim])
            {
                *a *= w;
            }
            clip(rh, c.rel_clip);
        }
        if !swa {
            q.iter_mut().for_each(|a| *a *= tau);
            r.iter_mut().for_each(|a| *a *= tau);
        }

        dump("Qcur", slot, &q);
        dump("Kcur", slot, &k);
        dump("Vcur", slot, &v);
        dump("Rcur", slot, &r);
        let slot_state = &mut s.slots[slot];
        slot_state.keys[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&k);
        slot_state.values[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&v);

        // Attention: the bias by distance (the last row, zeros, for distances past the
        // table) less the head's sink, and a sink logit of 0 beside the keys.
        let (first, extent) = if swa {
            ((pos + 1).saturating_sub(c.sliding_window), c.sliding_window)
        } else {
            (0, c.rel_extent)
        };
        let group = heads / c.kv_heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut attention = vec![0.0; q_dim];
        let mut scores = vec![0.0; pos + 2 - first];
        for head in 0..heads {
            let qh = &q[head * hd..(head + 1) * hd];
            let rh = &r[head * c.rel_dim..(head + 1) * c.rel_dim];
            let kv = (head / group) * hd;
            for (i, score) in scores[..pos + 1 - first].iter_mut().enumerate() {
                let t = first + i;
                let distance = (pos - t).min(extent);
                let bias = quant::dot(
                    rh,
                    &layer.rel_bias[distance * c.rel_dim..(distance + 1) * c.rel_dim],
                );
                let key = &slot_state.keys[t * kv_dim + kv..t * kv_dim + kv + hd];
                // Only q·k is scaled: the bias and sink are logits as they are (checked
                // against llama.cpp's attention weights).
                *score = quant::dot(qh, key) * scale + bias - sink[head];
            }
            *scores.last_mut().expect("the sink") = 0.0;
            cpu::softmax(&mut scores);
            let out = &mut attention[head * hd..(head + 1) * hd];
            for (i, &p) in scores[..pos + 1 - first].iter().enumerate() {
                let t = first + i;
                for (o, &v) in out
                    .iter_mut()
                    .zip(&slot_state.values[t * kv_dim + kv..t * kv_dim + kv + hd])
                {
                    *o += p * v;
                }
            }
        }
        dump("kqv_out", slot, &attention);
        for (a, g) in attention.iter_mut().zip(&gate) {
            *a *= sigmoid(*g);
        }
        let mut attn_out = vec![0.0; dim];
        cpu::matvec(pool, &layer.o, &attention, &mut attn_out);
        let history = &mut s.slots[slot].history;
        convolve(
            history,
            channels,
            2 * kv_dim,
            &mut attn_out,
            &layer.attn_post_conv,
            None,
            c.conv,
        );

        dump("attn_out", slot, &attn_out);
        // The first layer of a block keeps its input as a block and starts afresh.
        if l.is_multiple_of(c.block) {
            blocks.push(std::mem::replace(x, attn_out));
        } else {
            add(x, &attn_out);
        }

        dump("ffn_inp", slot, x);
        // The experts, on a mix of the blocks and x.
        let mut states = blocks.clone();
        states.push(x.clone());
        let mut input = mix(&states, std::slice::from_ref(&layer.ffn_res_query)).remove(0);
        if let Some((m, lp)) = looped {
            input = control(pool, &input, &m[3], &lp.controllers[3]);
        }
        let mut hf = vec![0.0; dim];
        cpu::rms_norm(&input, &layer.ffn_norm, eps, &mut hf);
        dump("norm", slot, &hf);
        let mut ffn_out = self.experts(layer, &hf);
        dump("ffn_moe_out", slot, &ffn_out);
        convolve(
            history,
            channels,
            2 * kv_dim + dim,
            &mut ffn_out,
            &layer.ffn_post_conv,
            None,
            c.conv,
        );
        dump("ffn_out", slot, &ffn_out);
        add(x, &ffn_out);
        dump("l_out", slot, x);
    }

    /// The experts' output for a normed input: llama.cpp's `build_moe_ffn` with sigmoid
    /// gating, a selection bias, normalized weights times `expert_scale`, and SiTU.
    fn experts(&self, layer: &Layer, h: &[f32]) -> Vec<f32> {
        let c = &self.config;
        let pool = &self.pool;
        let mut probs = vec![0.0; c.experts];
        cpu::matvec(pool, &layer.router, h, &mut probs);
        probs.iter_mut().for_each(|p| *p = sigmoid(*p));
        let mut picked: Vec<usize> = (0..c.experts).collect();
        let pick = |i: usize| probs[i] + layer.router_bias[i];
        picked.sort_by(|&a, &b| pick(b).total_cmp(&pick(a)).then(a.cmp(&b)));
        picked.truncate(c.experts_used);
        let total: f32 = picked.iter().map(|&i| probs[i]).sum::<f32>() + 1e-20;

        let w = c.expert_width;
        let (beta, linear_beta) = (c.situ_beta, c.situ_linear_beta);
        let mut gate_up = vec![0.0; 2 * w];
        let mut act = vec![0.0; w];
        let mut out = vec![0.0; c.dim];
        let mut sum = vec![0.0; c.dim];
        for &e in &picked {
            cpu::matvec(
                pool,
                &layer.gate_up.rows_from(e * 2 * w, 2 * w),
                h,
                &mut gate_up,
            );
            for j in 0..w {
                let (g, u) = (gate_up[j], gate_up[w + j]);
                act[j] =
                    beta * (g / beta).tanh() * sigmoid(g) * linear_beta * (u / linear_beta).tanh();
            }
            cpu::matvec(
                pool,
                &layer.down.rows_from(e * c.dim, c.dim),
                &act,
                &mut out,
            );
            let share = probs[e] / total * c.expert_scale;
            for (s, &o) in sum.iter_mut().zip(&out) {
                *s += share * o;
            }
        }
        sum
    }
}

/// Debugging: with `SYRINX_DUMP` set, prints a value's sum, sum of magnitudes and first
/// numbers under llama.cpp's name for it (`name-layer`), to compare with its eval callback.
fn dump(name: &str, slot: usize, x: &[f32]) {
    if std::env::var_os("SYRINX_DUMP").is_none() {
        return;
    }
    let label = if slot == usize::MAX {
        name.to_string()
    } else {
        format!("{name}-{slot}")
    };
    let sum: f64 = x.iter().map(|&v| v as f64).sum();
    let abs: f64 = x.iter().map(|&v| (v as f64).abs()).sum();
    let at = |i: usize| x.get(i).copied().unwrap_or(0.0);
    eprintln!(
        "{label:<28} n={:<8} sum={sum:.6} abs={abs:.6}  {:.5} {:.5} {:.5} {:.5}",
        x.len(),
        at(0),
        at(1),
        at(2),
        at(3)
    );
}

/// Attention over depth: each query's softmax-weighted mix of the RMS-normed `states`
/// (no weight), scores scaled by 1/sqrt(dim).
fn mix(states: &[Vec<f32>], queries: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let dim = states[0].len();
    let ones = vec![1.0; dim];
    let normed: Vec<Vec<f32>> = states
        .iter()
        .map(|s| {
            let mut out = vec![0.0; dim];
            cpu::rms_norm(s, &ones, 1e-6, &mut out);
            out
        })
        .collect();
    let scale = 1.0 / (dim as f32).sqrt();
    queries
        .iter()
        .map(|query| {
            let mut p: Vec<f32> = normed
                .iter()
                .map(|s| quant::dot(s, query) * scale)
                .collect();
            cpu::softmax(&mut p);
            let mut out = vec![0.0; dim];
            for (s, &w) in normed.iter().zip(&p) {
                for (o, &v) in out.iter_mut().zip(s) {
                    *o += w * v;
                }
            }
            out
        })
        .collect()
}

/// A looped layer's controller: `(1 − forget) × cur + gain × message`, with `gain` and
/// `forget` tanh of a rank-`gate_rank` map of `[RMSNorm(cur); RMSNorm(message)]`.
fn control(pool: &Pool, cur: &[f32], message: &[f32], ctrl: &Controller) -> Vec<f32> {
    let dim = cur.len();
    let ones = vec![1.0; dim];
    let mut both = vec![0.0; 2 * dim];
    cpu::rms_norm(cur, &ones, 1e-6, &mut both[..dim]);
    cpu::rms_norm(message, &ones, 1e-6, &mut both[dim..]);
    let mut low = vec![0.0; ctrl.down.rows];
    cpu::matvec(pool, &ctrl.down, &both, &mut low);
    let mut gain = vec![0.0; dim];
    let mut forget = vec![0.0; dim];
    cpu::matvec(pool, &ctrl.gain, &low, &mut gain);
    cpu::matvec(pool, &ctrl.forget, &low, &mut forget);
    cur.iter()
        .zip(message)
        .zip(gain.iter().zip(&forget))
        .map(|((&c, &m), (&g, &f))| c - f.tanh() * c + g.tanh() * m)
        .collect()
}

/// A causal convolution of `x` (this token's `x.len()` channels) over the last
/// `conv − 1` inputs kept in `history` at channel `offset`, with a residual:
/// `(x + Σ_k w[k] x_{t−(conv−1)+k} + Σ_lag taps[lag] x_{t−lag}) / sqrt(n × conv + 1)`, `n`
/// 2 with input-dependent `taps` (`conv` per head, one head per `x.len() / heads`
/// channels) and 1 without. Then this token's input joins the history.
fn convolve(
    history: &mut [f32],
    channels: usize,
    offset: usize,
    x: &mut [f32],
    weight: &[f32],
    taps: Option<&[f32]>,
    conv: usize,
) {
    let n = x.len();
    let head_width = taps.map(|t| n / (t.len() / conv));
    let terms = if taps.is_some() { 2 * conv } else { conv };
    let scale = 1.0 / ((terms + 1) as f32).sqrt();
    let mut window = vec![0.0; conv];
    for ch in 0..n {
        // The channel's inputs, oldest first: conv − 1 from the history, then x.
        for (k, w) in window[..conv - 1].iter_mut().enumerate() {
            *w = history[k * channels + offset + ch];
        }
        window[conv - 1] = x[ch];
        let mut sum = x[ch];
        for k in 0..conv {
            sum += weight[ch * conv + k] * window[k];
        }
        if let (Some(t), Some(width)) = (taps, head_width) {
            let head = ch / width;
            for lag in 0..conv {
                sum += t[head * conv + lag] * window[conv - 1 - lag];
            }
        }
        // Shift the history by one, keeping this raw input as the newest.
        for k in 0..conv - 1 {
            history[k * channels + offset + ch] = window[k + 1];
        }
        x[ch] = sum * scale;
    }
}

/// Scales `x` down to an L2 norm of at most `threshold` (llama.cpp: norm + 1e-6).
fn clip(x: &mut [f32], threshold: f32) {
    let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-6;
    let factor = (threshold / norm).clamp(0.0, 1.0);
    x.iter_mut().for_each(|v| *v *= factor);
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
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
    fn convolution_keeps_history_and_scales() {
        // One channel, 3 taps of history: weights (1, 2, 3, 4), oldest first.
        let (conv, channels) = (4, 1);
        let mut history = vec![0.0; (conv - 1) * channels];
        let weight = [1.0, 2.0, 3.0, 4.0];
        let mut outs = Vec::new();
        for value in [1.0, 2.0, 3.0] {
            let mut x = [value];
            convolve(&mut history, channels, 0, &mut x, &weight, None, conv);
            outs.push(x[0]);
        }
        let s = 1.0 / 5f32.sqrt();
        // t0: 1 + 4×1; t1: 2 + 3×1 + 4×2; t2: 3 + 2×1 + 3×2 + 4×3.
        let want = [5.0 * s, 13.0 * s, 23.0 * s];
        for (o, w) in outs.iter().zip(want) {
            assert!((o - w).abs() < 1e-5, "{outs:?} vs {want:?}");
        }
        assert_eq!(history, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn mix_of_one_state_is_that_state_normed() {
        let state = vec![3.0, -3.0, 3.0, -3.0];
        let out = mix(&[state], &[vec![0.5; 4], vec![-1.0; 4]]);
        for o in out {
            assert!(
                o.iter()
                    .zip([1.0, -1.0, 1.0, -1.0])
                    .all(|(a, b)| (a - b).abs() < 1e-4)
            );
        }
    }

    #[test]
    fn clip_only_shrinks() {
        let mut x = [3.0, 4.0];
        clip(&mut x, 10.0);
        assert_eq!(x, [3.0, 4.0]);
        clip(&mut x, 1.0);
        assert!((x[0] - 0.6).abs() < 1e-5 && (x[1] - 0.8).abs() < 1e-5);
    }
}
