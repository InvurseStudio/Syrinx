//! Google's draft model for Gemma 4 (`Gemma4AssistantForCausalLM`, e.g.
//! `gemma-4-31B-it-assistant`): four small layers that guess the next tokens from the
//! main model's own state, for it to check in one pass. Its weights come as safetensors
//! (bf16), read here directly and rounded to Q8_0 on the way up.
//!
//! What it reads, as Hugging Face's `Gemma4AssistantForCausalLM` and mlx-vlm's
//! `Gemma4AssistantDraftModel` do:
//! - Its input for a guess is the main model's (scaled) embedding of the token just
//!   picked, then a hidden state: for the first guess, the main model's last state after
//!   its final norm (`s.h`'s row for the last position read); for each guess after, its
//!   own output projected back to the main model's width (`post_projection`).
//!   `pre_projection` takes the pair down to the draft's width.
//! - It has no keys or values of its own. Each of its layers reads the main model's: the
//!   last sliding-window layer's for a sliding layer, the last global layer's for a global
//!   one, with its queries at the last position read (held for every guess of a round).
//! - Its layers are Gemma 4's (norms used as they are, tanh-GELU, a scalar on each
//!   layer's output), and its logits are its tied embedding times its final norm's output.
//! - The small models' (E2B, E4B: `use_ordered_embeddings`) have a centroid head instead
//!   of the whole product (`Gemma4AssistantMaskedEmbedder`): the vocabulary is in 2,048
//!   clusters of 128 tokens, the state scores the clusters' centroids, and only the 32
//!   best clusters' tokens get logits. That's 4,096 rows of 256 for a guess, not 262,144:
//!   done here on the CPU from the state alone, so a guess reads 1 KB back, not 1 MB.

use std::path::{Path, PathBuf};

use super::{
    Error, Gemma4Cuda, Gemma4CudaSession, GpuMatrix, Input, LogitsBuffer, Precision, Result,
    gpu_error, q8_room, row,
};
#[cfg(not(target_os = "macos"))]
use super::record;
use crate::TensorType;
use crate::cpu::Matrix;
use crate::cuda::{Buffer, Graph, Q8, QkvPrep};
use crate::mmap::Mmap;

struct Layer {
    attn_norm: Buffer,
    q: GpuMatrix,
    q_norm: Buffer,
    o: GpuMatrix,
    post_attention_norm: Buffer,
    ffn_norm: Buffer,
    gate: GpuMatrix,
    up: GpuMatrix,
    down: GpuMatrix,
    post_ffw_norm: Buffer,
    /// `layer_scalar`: the layer's output times this.
    scale: f32,
    /// A sliding-window layer (head size the main model's sliding one) or a global one.
    swa: bool,
    head_dim: usize,
}

/// A loaded draft model, for the main model it was made for.
pub struct Assistant {
    /// Its folder's or file's name.
    pub name: String,
    dim: usize,
    ffn: usize,
    heads: usize,
    layers: Vec<Layer>,
    pre: GpuMatrix,
    post: GpuMatrix,
    /// The tied embedding, for the logits of a model without a centroid head.
    head: Option<GpuMatrix>,
    /// The centroid head of a model with one (`use_ordered_embeddings`).
    centroids: Option<Centroids>,
    norm: Buffer,
    /// The main model's layers whose keys and values its sliding and global layers read.
    swa_source: usize,
    full_source: usize,
}

/// A centroid head: the vocabulary in clusters, each with a centroid the state scores,
/// and logits only for the tokens of the best few clusters. Held in the PC's memory.
struct Centroids {
    /// `count` centroids, a row of `dim` each.
    weights: Vec<f32>,
    count: usize,
    dim: usize,
    /// Tokens a cluster.
    per: usize,
    /// How many clusters' tokens get logits (`centroid_intermediate_top_k`).
    top: usize,
    /// Each cluster's tokens, cluster after cluster (`token_ordering`).
    order: Vec<u32>,
    /// The tied embedding's rows (bf16) in that order, so a cluster's are side by side.
    rows: Vec<u16>,
}

/// `a · b`, in eight running sums (so the compiler can do eight at once).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sums = [0f32; 8];
    let (a8, b8) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f32 = a8
        .remainder()
        .iter()
        .zip(b8.remainder())
        .map(|(x, y)| x * y)
        .sum();
    for (x, y) in a8.zip(b8) {
        for i in 0..8 {
            sums[i] += x[i] * y[i];
        }
    }
    sums.iter().sum::<f32>() + tail
}

/// `a · b` for a row of bf16.
fn dot_bf16(a: &[u16], b: &[f32]) -> f32 {
    let wide = |v: u16| f32::from_bits(u32::from(v) << 16);
    let mut sums = [0f32; 8];
    let (a8, b8) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f32 = a8
        .remainder()
        .iter()
        .zip(b8.remainder())
        .map(|(x, y)| wide(*x) * y)
        .sum();
    for (x, y) in a8.zip(b8) {
        for i in 0..8 {
            sums[i] += wide(x[i]) * y[i];
        }
    }
    sums.iter().sum::<f32>() + tail
}

impl Centroids {
    /// The most likely token for the final state `h`, the best of the `top` best
    /// clusters' tokens, and its probability among them. `scores` and `logits` are scratch.
    fn pick(&self, h: &[f32], scores: &mut Vec<(f32, u32)>, logits: &mut Vec<f32>) -> (u32, f32) {
        scores.clear();
        scores.extend(
            self.weights
                .chunks_exact(self.dim)
                .enumerate()
                .map(|(c, w)| (dot(w, h), c as u32)),
        );
        let top = self.top.min(self.count);
        if top < self.count {
            scores.select_nth_unstable_by(top - 1, |a, b| b.0.total_cmp(&a.0));
        }
        let mut best = (f32::NEG_INFINITY, 0u32);
        let cluster = self.per * self.dim;
        logits.clear();
        for &(_, c) in &scores[..top] {
            let c = c as usize;
            let rows = &self.rows[c * cluster..(c + 1) * cluster];
            for (j, row) in rows.chunks_exact(self.dim).enumerate() {
                let v = dot_bf16(row, h);
                logits.push(v);
                let token = self.order[c * self.per + j];
                // Ties to the lower token, as an argmax over the whole vocabulary's.
                if v > best.0 || (v == best.0 && token < best.1) {
                    best = (v, token);
                }
            }
        }
        (best.1, share_of_best(logits, best.0))
    }
}

/// The probability of the token with the logit `best`, the highest of `logits`. Tokens
/// far below it are left out of the sum: 262,144 of them at e^-20 add 0.0005.
fn share_of_best(logits: &[f32], best: f32) -> f32 {
    let floor = best - 20.0;
    let sum: f32 = logits
        .iter()
        .filter(|&&v| v > floor)
        .map(|&v| (v - best).exp())
        .sum();
    if sum > 0.0 { 1.0 / sum } else { 0.0 }
}

/// `centroid_intermediate_top_k` from the `config.json` beside the weights; Google's 32
/// without one.
fn centroid_top(weights: &Path) -> usize {
    let key = "\"centroid_intermediate_top_k\"";
    weights
        .parent()
        .and_then(|d| std::fs::read_to_string(d.join("config.json")).ok())
        .and_then(|text| {
            let rest = &text[text.find(key)? + key.len()..];
            let digits: String = rest
                .trim_start_matches([':', ' ', '\n', '\t'])
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();
            digits.parse().ok()
        })
        .filter(|&n| n > 0)
        .unwrap_or(32)
}

/// A session's room for drafting (`Gemma4Cuda::use_mtp`).
pub(super) struct Room {
    /// The embedding and the hidden state, side by side.
    cat: Buffer,
    x: Buffer,
    h: Buffer,
    q: Buffer,
    attention: Buffer,
    projected: Buffer,
    gate: Buffer,
    up: Buffer,
    /// The draft's state projected back to the main model's width: the next guess's.
    out: Buffer,
    pos: Buffer,
    logits: LogitsBuffer,
    host: Vec<f32>,
    logits_host: Vec<f32>,
    /// The draft's final state read back, and scratch, for a centroid head.
    state_host: Vec<f32>,
    scores: Vec<(f32, u32)>,
    candidates: Vec<f32>,
    q8: Option<Q8>,
    /// One guess's layers, recorded the first time (CUDA).
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    graph: Option<Graph>,
}

/// Where the draft model for the Gemma 4 file at `model` is, if it's anywhere Vireo keeps
/// them: beside the file (a folder or file named like it with "assistant"), or in the
/// Mac app's draft models folder (`DraftModels/<name>-assistant-bf16`, the name the file's
/// up to its quantization, e.g. `gemma-4-31B-it`).
pub fn find(model: &Path) -> Option<PathBuf> {
    let stem = model.file_stem()?.to_string_lossy().into_owned();
    // "gemma-4-31B-it-Q6_K" -> "gemma-4-31B-it"
    let quant = |tail: &str| {
        tail.starts_with(['Q', 'q'])
            || tail.starts_with("IQ")
            || ["F16", "BF16", "F32", "f16", "bf16", "f32"].contains(&tail)
    };
    let base = match stem.rfind(['-', '.']) {
        Some(i) if quant(&stem[i + 1..]) => stem[..i].to_string(),
        _ => stem.clone(),
    };
    // Google's own files name the quantization mid-name: "gemma-4-E2B_q4_0-it" is
    // "gemma-4-E2B-it" too.
    let plain: Vec<&str> = base
        .split('-')
        .map(|part| match part.split_once('_') {
            Some((name, tail)) if quant(tail) => name,
            _ => part,
        })
        .collect();
    let mut bases = vec![base.clone()];
    if plain.join("-") != base {
        bases.push(plain.join("-"));
    }
    let mut places: Vec<PathBuf> = Vec::new();
    if let Some(dir) = model.parent() {
        for base in &bases {
            places.push(dir.join(format!("{base}-assistant-bf16")));
            places.push(dir.join(format!("{base}-assistant")));
            places.push(dir.join(format!("{base}-assistant.safetensors")));
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let drafts = Path::new(&home).join("Library/Application Support/InvurseChat/DraftModels");
        places.push(drafts.join(format!("{base}-assistant-bf16")));
        places.push(drafts.join(format!("{base}-assistant")));
    }
    places.into_iter().find(|p| weights_file(p).is_some())
}

/// The safetensors file at `path`: the file itself, or a folder's `model.safetensors`.
fn weights_file(path: &Path) -> Option<PathBuf> {
    if path.is_dir() {
        let f = path.join("model.safetensors");
        f.is_file().then_some(f)
    } else {
        (path.is_file() && path.extension().is_some_and(|e| e == "safetensors"))
            .then(|| path.to_path_buf())
    }
}

/// A tensor in a safetensors header: its name, type, shape, and where its bytes start and
/// end after the header.
type Entry = (String, String, Vec<usize>, usize, usize);

/// A safetensors file's tensors: name, type, shape and bytes.
struct Tensors {
    map: Mmap,
    header: usize,
    entries: Vec<Entry>,
}

impl Tensors {
    fn open(path: &Path) -> Result<Self> {
        let map = unsafe { Mmap::open(path) }
            .map_err(|e| Error::Unsupported(format!("reading {}: {e}", path.display())))?;
        let bad = || Error::Unsupported(format!("{} isn't a safetensors file", path.display()));
        let len = u64::from_le_bytes(map.get(..8).ok_or_else(bad)?.try_into().unwrap()) as usize;
        let text = std::str::from_utf8(map.get(8..8 + len).ok_or_else(bad)?).map_err(|_| bad())?;
        let entries = parse_header(text).ok_or_else(bad)?;
        Ok(Self {
            map,
            header: 8 + len,
            entries,
        })
    }

    fn get(&self, name: &str) -> Result<(&str, &[usize], &[u8])> {
        let (_, kind, shape, start, end) = self
            .entries
            .iter()
            .find(|e| e.0 == name)
            .ok_or_else(|| Error::Missing(format!("tensor {name} in the draft model")))?;
        let bytes = self
            .map
            .get(self.header + start..self.header + end)
            .ok_or_else(|| Error::Unsupported(format!("tensor {name} runs past the file")))?;
        Ok((kind.as_str(), shape.as_slice(), bytes))
    }

    /// A bf16 matrix (`out × in`, rows the outputs, as GGUF's), rounded to Q8_0 on the GPU.
    fn matrix(
        &self,
        gpu: &crate::cuda::Gpu,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<GpuMatrix> {
        let (kind, shape, data) = self.get(name)?;
        if kind != "BF16" || shape != [rows, cols] {
            return Err(Error::Unsupported(format!(
                "draft tensor {name}: {kind} {shape:?}, not BF16 [{rows}, {cols}]"
            )));
        }
        let m = Matrix {
            kind: TensorType::BF16,
            rows,
            cols,
            data,
        };
        GpuMatrix::upload_q8_0(gpu, &m)?
            .ok_or_else(|| Error::Unsupported(format!("draft tensor {name} as Q8_0")))
    }

    /// A bf16 vector as f32.
    fn vector(&self, name: &str, len: usize) -> Result<Vec<f32>> {
        let (kind, shape, data) = self.get(name)?;
        if kind != "BF16" || shape.iter().product::<usize>() != len {
            return Err(Error::Unsupported(format!(
                "draft tensor {name}: {kind} {shape:?}, not BF16 of {len}"
            )));
        }
        Ok(data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
            .collect())
    }

    /// A vector of `len` token ids (I64 or I32), each below `len`.
    fn indices(&self, name: &str, len: usize) -> Result<Vec<u32>> {
        let (kind, shape, data) = self.get(name)?;
        let width = match kind {
            "I64" => 8,
            "I32" => 4,
            _ => 0,
        };
        if width == 0 || shape != [len] || data.len() != len * width {
            return Err(Error::Unsupported(format!(
                "draft tensor {name}: {kind} {shape:?}, not {len} whole numbers"
            )));
        }
        let ids: Vec<u32> = data
            .chunks_exact(width)
            .map(|b| u32::from_le_bytes(b[..4].try_into().unwrap()))
            .collect();
        // Every token once: a cluster's logits then stand for the vocabulary's.
        let mut seen = vec![false; len];
        for &id in &ids {
            match seen.get_mut(id as usize) {
                Some(s) if !*s => *s = true,
                _ => {
                    return Err(Error::Unsupported(format!(
                        "draft tensor {name} isn't an ordering of {len} tokens"
                    )));
                }
            }
        }
        Ok(ids)
    }

    fn shape(&self, name: &str) -> Option<&[usize]> {
        self.entries
            .iter()
            .find(|e| e.0 == name)
            .map(|e| e.2.as_slice())
    }
}

/// A safetensors header: `{"name": {"dtype": "BF16", "shape": [..], "data_offsets": [a, b]}, ..}`
/// (and `__metadata__`, skipped). Just enough JSON for that.
fn parse_header(text: &str) -> Option<Vec<Entry>> {
    #[derive(Debug)]
    enum V {
        S(String),
        N(f64),
        A(Vec<V>),
        O(Vec<(String, V)>),
        Other,
    }
    fn ws(b: &[u8], i: &mut usize) {
        while *i < b.len() && b[*i].is_ascii_whitespace() {
            *i += 1;
        }
    }
    fn string(b: &[u8], i: &mut usize) -> Option<String> {
        if b.get(*i) != Some(&b'"') {
            return None;
        }
        *i += 1;
        let mut out = Vec::new();
        while *i < b.len() {
            match b[*i] {
                b'"' => {
                    *i += 1;
                    return String::from_utf8(out).ok();
                }
                b'\\' => {
                    out.push(*b.get(*i + 1)?);
                    *i += 2;
                }
                c => {
                    out.push(c);
                    *i += 1;
                }
            }
        }
        None
    }
    fn value(b: &[u8], i: &mut usize) -> Option<V> {
        ws(b, i);
        match *b.get(*i)? {
            b'"' => string(b, i).map(V::S),
            b'[' => {
                *i += 1;
                let mut items = Vec::new();
                loop {
                    ws(b, i);
                    if b.get(*i) == Some(&b']') {
                        *i += 1;
                        return Some(V::A(items));
                    }
                    items.push(value(b, i)?);
                    ws(b, i);
                    if b.get(*i) == Some(&b',') {
                        *i += 1;
                    }
                }
            }
            b'{' => {
                *i += 1;
                let mut fields = Vec::new();
                loop {
                    ws(b, i);
                    if b.get(*i) == Some(&b'}') {
                        *i += 1;
                        return Some(V::O(fields));
                    }
                    let key = string(b, i)?;
                    ws(b, i);
                    if b.get(*i) != Some(&b':') {
                        return None;
                    }
                    *i += 1;
                    fields.push((key, value(b, i)?));
                    ws(b, i);
                    if b.get(*i) == Some(&b',') {
                        *i += 1;
                    }
                }
            }
            c if c == b'-' || c.is_ascii_digit() => {
                let start = *i;
                while *i < b.len()
                    && matches!(b[*i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
                {
                    *i += 1;
                }
                std::str::from_utf8(&b[start..*i])
                    .ok()?
                    .parse()
                    .ok()
                    .map(V::N)
            }
            _ => {
                while *i < b.len() && b[*i].is_ascii_alphabetic() {
                    *i += 1;
                }
                Some(V::Other)
            }
        }
    }
    let b = text.as_bytes();
    let mut i = 0;
    let V::O(fields) = value(b, &mut i)? else {
        return None;
    };
    let nums = |v: &V| -> Option<Vec<usize>> {
        match v {
            V::A(items) => items
                .iter()
                .map(|x| match x {
                    V::N(n) if *n >= 0.0 => Some(*n as usize),
                    _ => None,
                })
                .collect(),
            _ => None,
        }
    };
    let mut out = Vec::new();
    for (name, v) in fields {
        if name == "__metadata__" {
            continue;
        }
        let V::O(parts) = v else { return None };
        let field = |k: &str| parts.iter().find(|(n, _)| n == k).map(|(_, v)| v);
        let Some(V::S(kind)) = field("dtype") else {
            return None;
        };
        let shape = nums(field("shape")?)?;
        let offsets = nums(field("data_offsets")?)?;
        if offsets.len() != 2 || offsets[1] < offsets[0] {
            return None;
        }
        out.push((name, kind.clone(), shape, offsets[0], offsets[1]));
    }
    Some(out)
}

impl Gemma4Cuda<'_> {
    /// Loads the draft model at `path` (a folder with `model.safetensors`, or the file) for
    /// this model to draft with (`use_mtp`, `mtp_draft`). Its shapes must fit this model:
    /// its input twice this model's width, its heads this model's.
    pub fn load_assistant(&mut self, path: &Path) -> Result<()> {
        let c = &self.config;
        if c.muse {
            return Err(Error::Unsupported(
                "a Gemma 4 draft model with Muse Glimmer".into(),
            ));
        }
        let file = weights_file(path)
            .ok_or_else(|| Error::Missing(format!("a draft model at {}", path.display())))?;
        let t = Tensors::open(&file)?;
        let gpu = &self.gpu;
        let up = |v: Vec<f32>| gpu.upload_f32(&v).map_err(gpu_error);
        let pre_shape = t
            .shape("pre_projection.weight")
            .ok_or_else(|| {
                Error::Missing("tensor pre_projection.weight in the draft model".into())
            })?
            .to_vec();
        let dim = pre_shape[0];
        if pre_shape != [dim, 2 * c.dim] {
            return Err(Error::Unsupported(format!(
                "a draft model for a model {} wide, not {}",
                pre_shape.get(1).copied().unwrap_or(0) / 2,
                c.dim
            )));
        }
        let ffn = t
            .shape("model.layers.0.mlp.gate_proj.weight")
            .map(|s| s[0])
            .ok_or_else(|| Error::Missing("the draft model's layers".into()))?;
        let mut layers = Vec::new();
        let mut heads = 0;
        while let Some(qn) = t.shape(&format!(
            "model.layers.{}.self_attn.q_norm.weight",
            layers.len()
        )) {
            let i = layers.len();
            let name = |part: &str| format!("model.layers.{i}.{part}.weight");
            let head_dim = qn[0];
            let swa = if head_dim == c.head_dim_swa {
                true
            } else if head_dim == c.head_dim {
                false
            } else {
                return Err(Error::Unsupported(format!(
                    "draft layer {i}'s head size {head_dim} (the model's are {} and {})",
                    c.head_dim_swa, c.head_dim
                )));
            };
            let q_rows = t.shape(&name("self_attn.q_proj")).map_or(0, |s| s[0]);
            // Its own count of query heads (the small models' have 4 to the model's 8);
            // they must share the model's key-value heads evenly (checked below).
            let layer_heads = q_rows / head_dim;
            if layer_heads == 0 || (heads != 0 && layer_heads != heads) {
                return Err(Error::Unsupported(format!(
                    "draft layer {i}'s {layer_heads} heads"
                )));
            }
            heads = layer_heads;
            let scalar = t.vector(&format!("model.layers.{i}.layer_scalar"), 1)?[0];
            layers.push(Layer {
                attn_norm: up(t.vector(&name("input_layernorm"), dim)?)?,
                q: t.matrix(gpu, &name("self_attn.q_proj"), heads * head_dim, dim)?,
                q_norm: up(t.vector(&name("self_attn.q_norm"), head_dim)?)?,
                o: t.matrix(gpu, &name("self_attn.o_proj"), dim, heads * head_dim)?,
                post_attention_norm: up(t.vector(&name("post_attention_layernorm"), dim)?)?,
                ffn_norm: up(t.vector(&name("pre_feedforward_layernorm"), dim)?)?,
                gate: t.matrix(gpu, &name("mlp.gate_proj"), ffn, dim)?,
                up: t.matrix(gpu, &name("mlp.up_proj"), ffn, dim)?,
                down: t.matrix(gpu, &name("mlp.down_proj"), dim, ffn)?,
                post_ffw_norm: up(t.vector(&name("post_feedforward_layernorm"), dim)?)?,
                scale: scalar,
                swa,
                head_dim,
            });
        }
        if layers.is_empty() {
            return Err(Error::Missing("the draft model's layers".into()));
        }
        // The main model's last layer of each kind, through its key/value source.
        let last = |swa: bool| {
            (0..c.layers)
                .rev()
                .find(|&l| c.is_swa[l] == swa)
                .map(|l| c.kv_source(l))
        };
        let (Some(swa_source), Some(full_source)) = (last(true), last(false)) else {
            return Err(Error::Unsupported(
                "a draft model for a model without both kinds of layer".into(),
            ));
        };
        for source in [swa_source, full_source] {
            let kv = c.kv_heads_of(source);
            if !heads.is_multiple_of(kv) {
                return Err(Error::Unsupported(format!(
                    "a draft model of {heads} heads for a model with {kv} key-value heads"
                )));
            }
        }
        // A centroid head (E2B, E4B) keeps the tied embedding in the PC's memory, by
        // cluster; without one it's a matrix on the GPU.
        let centroids = match t.shape("masked_embedding.centroids.weight") {
            Some(shape) => {
                let count = shape[0];
                if shape != [count, dim] || count == 0 || !c.vocab.is_multiple_of(count) {
                    return Err(Error::Unsupported(format!(
                        "the draft model's centroids {shape:?} for a vocabulary of {}",
                        c.vocab
                    )));
                }
                let order = t.indices("masked_embedding.token_ordering", c.vocab)?;
                let (_, _, table) = t.get("model.embed_tokens.weight")?;
                if table.len() != c.vocab * dim * 2 {
                    return Err(Error::Unsupported(
                        "the draft model's embedding isn't BF16 [vocabulary, width]".into(),
                    ));
                }
                let mut rows = Vec::with_capacity(c.vocab * dim);
                for &token in &order {
                    let at = token as usize * dim * 2;
                    rows.extend(
                        table[at..at + dim * 2]
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|b| u16::from_le_bytes(*b)),
                    );
                }
                Some(Centroids {
                    weights: t.vector("masked_embedding.centroids.weight", count * dim)?,
                    count,
                    dim,
                    per: c.vocab / count,
                    top: centroid_top(&file),
                    order,
                    rows,
                })
            }
            None => None,
        };
        let assistant = Assistant {
            name: path
                .file_name()
                .map_or_else(String::new, |n| n.to_string_lossy().into_owned()),
            pre: t.matrix(gpu, "pre_projection.weight", dim, 2 * c.dim)?,
            post: t.matrix(gpu, "post_projection.weight", c.dim, dim)?,
            head: match centroids {
                Some(_) => None,
                None => Some(t.matrix(gpu, "model.embed_tokens.weight", c.vocab, dim)?),
            },
            centroids,
            norm: up(t.vector("model.norm.weight", dim)?)?,
            dim,
            ffn,
            heads,
            layers,
            swa_source,
            full_source,
        };
        self.assistant = Some(assistant);
        Ok(())
    }

    /// The draft model loaded (`load_assistant`), if any.
    pub fn assistant(&self) -> Option<&Assistant> {
        self.assistant.as_ref()
    }

    /// Whether there's a draft model to guess with.
    pub fn has_mtp(&self) -> bool {
        self.assistant.is_some()
    }

    /// Gives `s` room to draft with the draft model ([`Gemma4Cuda::mtp_draft`]).
    pub fn use_mtp(&self, s: &mut Gemma4CudaSession) -> Result<()> {
        let a = self
            .assistant
            .as_ref()
            .expect("load_assistant before use_mtp");
        if s.assistant.is_some() {
            return Ok(());
        }
        let c = &self.config;
        let g = &self.gpu;
        let f32s = |n: usize| g.alloc(n.max(1) * 4).map_err(gpu_error);
        let widest_q = a.heads * a.layers.iter().map(|l| l.head_dim).max().unwrap_or(0);
        s.assistant = Some(Room {
            cat: f32s(2 * c.dim)?,
            x: f32s(a.dim)?,
            h: f32s(a.dim)?,
            q: f32s(widest_q)?,
            attention: f32s(widest_q)?,
            projected: f32s(a.dim)?,
            gate: f32s(a.ffn)?,
            up: f32s(a.ffn)?,
            out: f32s(c.dim)?,
            pos: f32s(1)?,
            logits: LogitsBuffer::new(g, if a.head.is_some() { c.vocab } else { 1 })?,
            host: vec![0.0; c.dim],
            logits_host: vec![0.0; if a.head.is_some() { c.vocab } else { 0 }],
            state_host: vec![0.0; a.dim],
            scores: Vec::new(),
            candidates: Vec::new(),
            graph: None,
            q8: q8_room(
                g,
                if s.q8.is_some() {
                    Precision::Int8
                } else {
                    Precision::Exact
                },
                1,
                (2 * c.dim).max(widest_q).max(a.ffn),
                &a.pre,
            )?,
        });
        Ok(())
    }

    /// Up to `k` guesses at the tokens after `first`, the token the model picked after the
    /// last position `s` read (not read yet), each the draft's most likely (greedy). None
    /// without the main model's state for that position: a session without room
    /// (`use_mtp`), or one whose last read left it elsewhere.
    pub fn mtp_draft(&self, s: &mut Gemma4CudaSession, first: u32, k: usize) -> Result<Vec<u32>> {
        let guesses = self.mtp_draft_sure(s, first, k, 0.0)?;
        Ok(guesses.into_iter().map(|(token, _)| token).collect())
    }

    /// [`Gemma4Cuda::mtp_draft`], each guess with the draft's probability for it, stopping
    /// before the first guess it gives less than `floor`: a guess the draft isn't sure of
    /// is seldom the model's pick, and checking it costs a good part of a token.
    pub fn mtp_draft_sure(
        &self,
        s: &mut Gemma4CudaSession,
        first: u32,
        k: usize,
        floor: f32,
    ) -> Result<Vec<(u32, f32)>> {
        let (Some(a), Some(room)) = (self.assistant.as_ref(), s.assistant.take()) else {
            return Ok(Vec::new());
        };
        let mut room = room;
        let r = self.draft_with(a, s, &mut room, first, k, floor);
        s.assistant = Some(room);
        r
    }

    fn draft_with(
        &self,
        a: &Assistant,
        s: &Gemma4CudaSession,
        room: &mut Room,
        first: u32,
        k: usize,
        floor: f32,
    ) -> Result<Vec<(u32, f32)>> {
        let c = &self.config;
        let g = &self.gpu;
        let e = gpu_error;
        let Some(last) = s.len.checked_sub(1) else {
            return Ok(Vec::new());
        };
        // The main model's final state for the last position read: a row of `s.h` while
        // the last batch read left it there.
        if k == 0 || last < s.hidden_from || last >= s.hidden_from + s.hidden_rows {
            return Ok(Vec::new());
        }
        let hidden_row = last - s.hidden_from;
        let dim = c.dim;
        room.pos.write_i32(&[last as i32]).map_err(e)?;
        let mut guesses = Vec::with_capacity(k);
        let mut token = first;
        for step in 0..k {
            assert!(
                (token as usize) < c.vocab,
                "token {token} is past the vocabulary"
            );
            self.embedding.decode_row(token as usize, &mut room.host);
            let scale = (dim as f32).sqrt();
            room.host.iter_mut().for_each(|v| *v *= scale);
            room.cat.write_f32(&room.host).map_err(e)?;
            let hidden = if step == 0 {
                row(&s.h, hidden_row, dim)
            } else {
                room.out.ptr()
            };
            g.copy(room.cat.f32_at(dim), hidden, dim * 4).map_err(e)?;
            // The layers: one launch for them all on CUDA, a graph recorded the first time
            // (queuing a guess's ~40 kernels one by one cost 0.65 of its 0.9 ms on Gemma 4
            // E2B; the GPU's work is 0.1-0.3 ms).
            #[cfg(not(target_os = "macos"))]
            {
                if room.graph.is_none() {
                    room.graph = Some(record(g, || self.queue_guess(a, s, room))?);
                }
                room.graph.as_ref().unwrap().launch().map_err(e)?;
            }
            #[cfg(target_os = "macos")]
            self.queue_guess(a, s, room)?;
            // room.h: the final norm's output, for its logits. room.out: its state for the
            // next guess.
            let q8 = room.q8.as_ref();
            let h = Input::new(g, room.h.ptr(), 1, q8);
            let sure;
            (token, sure) = match (&a.centroids, &a.head) {
                (Some(centroids), _) => {
                    room.h.read_f32(&mut room.state_host).map_err(e)?;
                    centroids.pick(&room.state_host, &mut room.scores, &mut room.candidates)
                }
                (None, Some(head)) => {
                    let out = room.logits.rows(g, 1, c.vocab)?;
                    head.mul(&h, out.ptr())?;
                    out.read_f32(&mut room.logits_host).map_err(e)?;
                    let (best, logit) = room.logits_host.iter().enumerate().fold(
                        (0, f32::NEG_INFINITY),
                        |best, (i, &v)| {
                            if v > best.1 { (i, v) } else { best }
                        },
                    );
                    (best as u32, share_of_best(&room.logits_host, logit))
                }
                (None, None) => unreachable!("a draft model has a head"),
            };
            if sure < floor {
                break;
            }
            guesses.push((token, sure));
        }
        Ok(guesses)
    }
    /// Queues one guess's layers: from the embedding and the hidden state side by side in
    /// `room.cat` to the final norm's output in `room.h`, and that projected back to the
    /// main model's width in `room.out`, for the next guess.
    fn queue_guess(&self, a: &Assistant, s: &Gemma4CudaSession, room: &Room) -> Result<()> {
        let c = &self.config;
        let g = &self.gpu;
        let e = gpu_error;
        let (adim, eps) = (a.dim, c.norm_eps);
        let q8 = room.q8.as_ref();
        a.pre
            .mul(&Input::new(g, room.cat.ptr(), 1, q8), room.x.ptr())?;
        g.residual_norm(
            room.x.ptr(),
            None,
            None,
            1.0,
            Some(&a.layers[0].attn_norm),
            room.h.ptr(),
            1,
            adim,
            eps,
        )
        .map_err(e)?;
        for (i, layer) in a.layers.iter().enumerate() {
            let hd = layer.head_dim;
            let source = if layer.swa {
                a.swa_source
            } else {
                a.full_source
            };
            let kv_heads = c.kv_heads_of(source);
            layer
                .q
                .mul(&Input::new(g, room.h.ptr(), 1, q8), room.q.ptr())?;
            g.qkv_prep(
                &QkvPrep {
                    q: room.q.ptr(),
                    q_stride: a.heads * hd,
                    q_head_stride: hd,
                    q_norm: Some(layer.q_norm.ptr()),
                    q_out: room.q.ptr(),
                    // No keys or values of its own: queries only.
                    k: room.q.ptr(),
                    k_norm: None,
                    v: room.q.ptr(),
                    v_norm: false,
                    cache: None,
                    heads: a.heads,
                    kv_heads,
                    head_dim: hd,
                    rope_dims: hd,
                    rope_base: if layer.swa {
                        c.rope_base_swa
                    } else {
                        c.rope_base
                    },
                    rope_factors: if layer.swa {
                        None
                    } else {
                        self.rope_factors.as_ref().map(Buffer::ptr)
                    },
                    neox: true,
                    pos: room.pos.ptr(),
                    eps,
                    half_cache: s.half_cache,
                    ring: 0,
                },
                1,
            )
            .map_err(e)?;
            let (keys, values) = (
                s.keys[source]
                    .as_ref()
                    .expect("a source layer has its own keys"),
                s.values[source].as_ref().expect("and values"),
            );
            g.attention_ring(
                s.half_cache,
                room.q.ptr(),
                keys.ptr(),
                values.ptr(),
                room.attention.ptr(),
                room.pos.ptr(),
                if layer.swa { c.sliding_window } else { 0 },
                s.rings[source],
                a.heads,
                1,
                hd,
                kv_heads * hd,
                a.heads / kv_heads,
                1.0,
            )
            .map_err(e)?;
            layer.o.mul(
                &Input::new(g, room.attention.ptr(), 1, q8),
                room.projected.ptr(),
            )?;
            g.residual_norm(
                room.x.ptr(),
                Some(room.projected.ptr()),
                Some(&layer.post_attention_norm),
                1.0,
                Some(&layer.ffn_norm),
                room.h.ptr(),
                1,
                adim,
                eps,
            )
            .map_err(e)?;
            #[cfg(target_os = "macos")]
            let fused =
                layer
                    .gate
                    .mul_glu(g, &layer.up, room.h.ptr(), room.gate.ptr(), true)?;
            #[cfg(not(target_os = "macos"))]
            let fused = false;
            if !fused {
                let h = Input::new(g, room.h.ptr(), 1, q8);
                layer.gate.mul(&h, room.gate.ptr())?;
                layer.up.mul(&h, room.up.ptr())?;
                g.geglu(room.gate.ptr(), room.up.ptr(), a.ffn, a.ffn, 1)
                    .map_err(e)?;
            }
            layer
                .down
                .mul(&Input::new(g, room.gate.ptr(), 1, q8), room.projected.ptr())?;
            let next = match a.layers.get(i + 1) {
                Some(next) => &next.attn_norm,
                None => &a.norm,
            };
            g.residual_norm(
                room.x.ptr(),
                Some(room.projected.ptr()),
                Some(&layer.post_ffw_norm),
                layer.scale,
                Some(next),
                room.h.ptr(),
                1,
                adim,
                eps,
            )
            .map_err(e)?;
        }
        let h = Input::new(g, room.h.ptr(), 1, q8);
        a.post.mul(&h, room.out.ptr())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_a_safetensors_header() {
        let header = r#"{"__metadata__":{"format":"pt"},"a.weight":{"dtype":"BF16","shape":[4, 2],"data_offsets":[0,16]},"b":{"dtype":"F32","shape":[],"data_offsets":[16,20]}}"#;
        let t = parse_header(header).unwrap();
        assert_eq!(t.len(), 2);
        assert_eq!(t[0], ("a.weight".into(), "BF16".into(), vec![4, 2], 0, 16));
        assert_eq!(t[1].2, Vec::<usize>::new());
        assert!(parse_header("[1, 2]").is_none());
    }

    #[test]
    fn a_centroid_head_picks_the_best_token_of_the_best_clusters() {
        // Four clusters of two tokens, 8 wide; cluster c's centroid and its tokens' rows
        // point along axis c (the second token half as far), and the tokens are numbered
        // against the clusters' order.
        let (count, per, dim) = (4usize, 2usize, 8usize);
        let bf16 = |v: f32| (v.to_bits() >> 16) as u16;
        let mut weights = vec![0.0; count * dim];
        let mut rows = vec![0u16; count * per * dim];
        for c in 0..count {
            weights[c * dim + c] = 1.0;
            rows[(c * per) * dim + c] = bf16(2.0);
            rows[(c * per + 1) * dim + c] = bf16(1.0);
        }
        let head = Centroids {
            weights,
            count,
            dim,
            per,
            top: 2,
            order: vec![7, 6, 5, 4, 3, 2, 1, 0],
            rows,
        };
        let (mut scores, mut logits) = (Vec::new(), Vec::new());
        // Mostly along axis 2, some along 0: cluster 2's first token, 7 - 2 × 2 = 3.
        let mut h = vec![0.0; dim];
        (h[2], h[0], h[3]) = (3.0, 1.0, -5.0);
        let (token, sure) = head.pick(&h, &mut scores, &mut logits);
        assert_eq!(token, 3);
        // Logits only for the two best clusters' four tokens: 6, 3, 2 and 1.
        assert_eq!(logits.len(), 4);
        let sum: f32 = [6.0f32, 3.0, 2.0, 1.0].iter().map(|v| (v - 6.0).exp()).sum();
        assert!((sure - 1.0 / sum).abs() < 1e-6, "{sure}");
        // A state the clusters can't tell apart: the lowest token of a tie.
        let (token, sure) = head.pick(&vec![0.0; dim], &mut scores, &mut logits);
        assert_eq!(sure, 0.25);
        assert!(head.order.contains(&token));
        assert_eq!(share_of_best(&[5.0, -100.0], 5.0), 1.0);
    }

    #[test]
    fn finds_the_draft_model_by_the_models_name() {
        let dir = std::env::temp_dir().join(format!("vireo-assistant-{}", std::process::id()));
        let drafts = dir.join("gemma-4-31B-it-assistant-bf16");
        std::fs::create_dir_all(&drafts).unwrap();
        std::fs::write(drafts.join("model.safetensors"), b"x").unwrap();
        let model = dir.join("gemma-4-31B-it-Q6_K.gguf");
        assert_eq!(find(&model), Some(drafts.clone()));
        // Google's file names, with the quantization mid-name.
        let small = dir.join("gemma-4-E2B-it-assistant-bf16");
        std::fs::create_dir_all(&small).unwrap();
        std::fs::write(small.join("model.safetensors"), b"x").unwrap();
        assert_eq!(find(&dir.join("gemma-4-E2B_q4_0-it.gguf")), Some(small));
        assert_eq!(
            find(&dir.join("other-model-Q4_K_M.gguf")).filter(|p| p.starts_with(&dir)),
            None
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
