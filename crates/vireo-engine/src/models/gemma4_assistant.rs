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
//!   layer's output), and its logits are its tied embedding times its final norm's output
//!   (the 31B's has no centroid head: `use_ordered_embeddings` false).

use std::path::{Path, PathBuf};

use super::{
    Error, Gemma4Cuda, Gemma4CudaSession, GpuMatrix, Input, LogitsBuffer, Precision, Result,
    gpu_error, q8_room, row,
};
use crate::TensorType;
use crate::cpu::Matrix;
use crate::cuda::{Buffer, Q8, QkvPrep};
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
    head: GpuMatrix,
    norm: Buffer,
    /// The main model's layers whose keys and values its sliding and global layers read.
    swa_source: usize,
    full_source: usize,
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
    q8: Option<Q8>,
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
    let mut places: Vec<PathBuf> = Vec::new();
    if let Some(dir) = model.parent() {
        places.push(dir.join(format!("{base}-assistant-bf16")));
        places.push(dir.join(format!("{base}-assistant")));
        places.push(dir.join(format!("{base}-assistant.safetensors")));
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
            heads = q_rows / head_dim;
            if heads != c.heads {
                return Err(Error::Unsupported(format!(
                    "draft layer {i}'s {heads} heads (the model has {})",
                    c.heads
                )));
            }
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
        let assistant = Assistant {
            name: path
                .file_name()
                .map_or_else(String::new, |n| n.to_string_lossy().into_owned()),
            pre: t.matrix(gpu, "pre_projection.weight", dim, 2 * c.dim)?,
            post: t.matrix(gpu, "post_projection.weight", c.dim, dim)?,
            head: t.matrix(gpu, "model.embed_tokens.weight", c.vocab, dim)?,
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
            logits: LogitsBuffer::new(g, c.vocab)?,
            host: vec![0.0; c.dim],
            logits_host: vec![0.0; c.vocab],
            q8: q8_room(
                g,
                if s.q8.is_some() {
                    Precision::Int8
                } else {
                    Precision::Exact
                },
                1,
                (2 * c.dim).max(widest_q).max(a.ffn),
            )?,
        });
        Ok(())
    }

    /// Up to `k` guesses at the tokens after `first`, the token the model picked after the
    /// last position `s` read (not read yet), each the draft's most likely (greedy). None
    /// without the main model's state for that position: a session without room
    /// (`use_mtp`), or one whose last read left it elsewhere.
    pub fn mtp_draft(&self, s: &mut Gemma4CudaSession, first: u32, k: usize) -> Result<Vec<u32>> {
        let (Some(a), Some(room)) = (self.assistant.as_ref(), s.assistant.take()) else {
            return Ok(Vec::new());
        };
        let mut room = room;
        let r = self.draft_with(a, s, &mut room, first, k);
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
    ) -> Result<Vec<u32>> {
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
        let (dim, adim, eps) = (c.dim, a.dim, c.norm_eps);
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
            // room.h: the final norm's output. Its state for the next guess, and its logits.
            let h = Input::new(g, room.h.ptr(), 1, q8);
            a.post.mul(&h, room.out.ptr())?;
            let out = room.logits.rows(g, 1, c.vocab)?;
            a.head.mul(&h, out.ptr())?;
            out.read_f32(&mut room.logits_host).map_err(e)?;
            token = room
                .logits_host
                .iter()
                .enumerate()
                .fold((0, f32::NEG_INFINITY), |best, (i, &v)| {
                    if v > best.1 { (i, v) } else { best }
                })
                .0 as u32;
            guesses.push(token);
        }
        Ok(guesses)
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
    fn finds_the_draft_model_by_the_models_name() {
        let dir = std::env::temp_dir().join(format!("vireo-assistant-{}", std::process::id()));
        let drafts = dir.join("gemma-4-31B-it-assistant-bf16");
        std::fs::create_dir_all(&drafts).unwrap();
        std::fs::write(drafts.join("model.safetensors"), b"x").unwrap();
        let model = dir.join("gemma-4-31B-it-Q6_K.gguf");
        assert_eq!(find(&model), Some(drafts.clone()));
        assert_eq!(
            find(&dir.join("other-model-Q4_K_M.gguf")).filter(|p| p.starts_with(&dir)),
            None
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
