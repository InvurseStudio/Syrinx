//! Gemma 4's vision encoder (the `mmproj` file, projector type `gemma4v`), on the CPU
//! reference backend. Follows transformers' `Gemma4VisionModel` + `embed_vision`.
//!
//! An image, resized so its sides are multiples of 48 (`patch × pool`) and it makes at
//! most `soft_tokens × 9` patches, is cut into 16×16 patches, row after row:
//!
//! ```text
//! x   = patch_embd · 2(pixels − 0.5) + position_embd[0][px] + position_embd[1][py]
//! ```
//!
//! Each of the 27 layers (no causal mask: every patch sees every other):
//!
//! ```text
//! h = RMSNorm(x) × ln1
//! q = RoPE2(RMSNorm(Wq h) × q_norm), k = RoPE2(RMSNorm(Wk h) × k_norm), v = RMSNorm(Wv h)
//! x = x + RMSNorm(Wo · attention(q, k, v)) × attn_post_norm        (scale 1)
//! x = x + RMSNorm(down(GELU(gate h') × up h')) × ffn_post_norm     h' = RMSNorm(x) × ln2
//! ```
//!
//! RoPE2 turns the first half of each 72-wide head by the patch's column and the second
//! half by its row, each half "neox" style (pairs i, i + 18), base 100. Then each 3×3
//! block of patches is averaged into one soft token, and
//!
//! ```text
//! s   = (mean × sqrt(dim) − std_bias) × std_scale
//! out = input_projection · RMSNorm(s)       (no weight; one row of the text model's width)
//! ```
//!
//! The text model takes `out` in place of an image token's embedding, unscaled.

use super::{Error, Result, vector};
use crate::GgufFile;
use crate::cpu::{self, Matrix, Pool};
use crate::quant;

/// The pooling kernel (not in the file: transformers' default, which every Gemma 4 uses).
pub const POOL: usize = 3;
/// The vision RoPE's base (`rope_theta`, not in the file).
const ROPE_BASE: f32 = 100.0;
/// The soft-token budgets Gemma 4 was trained with; 280 is its default.
pub const SOFT_TOKEN_BUDGETS: [usize; 5] = [70, 140, 280, 560, 1120];

#[derive(Clone, Debug, PartialEq)]
pub struct VisionConfig {
    pub dim: usize,
    pub ffn_dim: usize,
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub patch: usize,
    pub norm_eps: f32,
    /// The text model's width: each soft token's length.
    pub out_dim: usize,
    pub positions: usize,
}

impl VisionConfig {
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let g: &crate::gguf::Gguf = file;
        if g.get_str("clip.vision.projector_type") != Some("gemma4v") {
            return Err(Error::Unsupported(format!(
                "vision projector {:?} (Gemma 4's is gemma4v)",
                g.get_str("clip.vision.projector_type")
            )));
        }
        let req = |k: &str| {
            g.get_u64(k)
                .map(|v| v as usize)
                .ok_or_else(|| Error::Missing(format!("setting {k}")))
        };
        let dim = req("clip.vision.embedding_length")?;
        let heads = req("clip.vision.attention.head_count")?;
        let positions = file
            .tensor("v.position_embd.weight")
            .and_then(|t| t.dims.get(1).copied())
            .ok_or_else(|| Error::Missing("tensor v.position_embd.weight".into()))?
            as usize;
        Ok(Self {
            dim,
            ffn_dim: req("clip.vision.feed_forward_length")?,
            layers: req("clip.vision.block_count")?,
            heads,
            head_dim: dim / heads,
            patch: req("clip.vision.patch_size")?,
            norm_eps: g
                .get_f64("clip.vision.attention.layer_norm_epsilon")
                .unwrap_or(1e-6) as f32,
            out_dim: req("clip.vision.projection_dim")?,
            positions,
        })
    }

    /// The size an image of `height × width` is resized to for a budget of `soft_tokens`:
    /// the largest with the same shape whose sides are multiples of `patch × POOL` and
    /// that makes at most `soft_tokens × POOL²` patches (transformers'
    /// `get_aspect_ratio_preserving_size`). Returns (height, width).
    pub fn fit(&self, height: usize, width: usize, soft_tokens: usize) -> (usize, usize) {
        let max_patches = soft_tokens * POOL * POOL;
        let target_px = (max_patches * self.patch * self.patch) as f64;
        let factor = (target_px / (height * width) as f64).sqrt();
        let side = self.patch * POOL;
        let mut h = (factor * height as f64 / side as f64).floor() as usize * side;
        let mut w = (factor * width as f64 / side as f64).floor() as usize * side;
        let longest = (max_patches / (POOL * POOL)) * side;
        if h == 0 && w == 0 {
            return (side, side);
        }
        if h == 0 {
            h = side;
            w = ((width / height.max(1)) * side).clamp(side, longest);
        } else if w == 0 {
            w = side;
            h = ((height / width.max(1)) * side).clamp(side, longest);
        }
        (h, w)
    }
}

struct Layer<'a> {
    ln1: Vec<f32>,
    q: Matrix<'a>,
    k: Matrix<'a>,
    v: Matrix<'a>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    o: Matrix<'a>,
    attn_post_norm: Vec<f32>,
    ln2: Vec<f32>,
    gate: Matrix<'a>,
    up: Matrix<'a>,
    down: Matrix<'a>,
    ffn_post_norm: Vec<f32>,
}

pub struct Gemma4Vision<'a> {
    pub config: VisionConfig,
    /// `dim` rows of `3 × patch²` numbers, each row's columns in (channel, y, x) order.
    patch_embd: Matrix<'a>,
    /// The column table then the row table, `positions × dim` each.
    position_embd: Vec<f32>,
    layers: Vec<Layer<'a>>,
    std_bias: Vec<f32>,
    std_scale: Vec<f32>,
    projection: Matrix<'a>,
    pool: Pool,
}

/// A tensor of any shape as a matrix of `rows` rows of `cols` numbers.
fn flat<'a>(file: &'a GgufFile, name: &str, cols: usize, rows: usize) -> Result<Matrix<'a>> {
    let t = file
        .tensor(name)
        .ok_or_else(|| Error::Missing(format!("tensor {name}")))?;
    if t.elements() != (cols * rows) as u64 {
        return Err(Error::Invalid(format!(
            "{name} is {:?}, but should hold {rows} × {cols}",
            t.dims
        )));
    }
    let kind = t
        .kind
        .filter(|&k| quant::supports(k))
        .ok_or_else(|| Error::Unsupported(format!("{name}'s type (GGUF type {})", t.type_id)))?;
    Ok(Matrix {
        kind,
        rows,
        cols,
        data: file.tensor_data(t).expect("a known type has data"),
    })
}

impl<'a> Gemma4Vision<'a> {
    pub fn load(file: &'a GgufFile, pool: Pool) -> Result<Self> {
        let c = VisionConfig::from_gguf(file)?;
        let (dim, ffn) = (c.dim, c.ffn_dim);
        let mut layers = Vec::with_capacity(c.layers);
        for i in 0..c.layers {
            let name = |part: &str| format!("v.blk.{i}.{part}.weight");
            let m = |part: &str, cols, rows| flat(file, &name(part), cols, rows);
            layers.push(Layer {
                ln1: vector(file, &name("ln1"), dim)?,
                q: m("attn_q", dim, dim)?,
                k: m("attn_k", dim, dim)?,
                v: m("attn_v", dim, dim)?,
                q_norm: vector(file, &name("attn_q_norm"), c.head_dim)?,
                k_norm: vector(file, &name("attn_k_norm"), c.head_dim)?,
                o: m("attn_out", dim, dim)?,
                attn_post_norm: vector(file, &name("attn_post_norm"), dim)?,
                ln2: vector(file, &name("ln2"), dim)?,
                gate: m("ffn_gate", dim, ffn)?,
                up: m("ffn_up", dim, ffn)?,
                down: m("ffn_down", ffn, dim)?,
                ffn_post_norm: vector(file, &name("ffn_post_norm"), dim)?,
            });
        }
        let table = flat(file, "v.position_embd.weight", dim, 2 * c.positions)?;
        let mut position_embd = vec![0.0; 2 * c.positions * dim];
        for (r, row) in position_embd.chunks_exact_mut(dim).enumerate() {
            table.decode_row(r, row);
        }
        Ok(Self {
            patch_embd: flat(file, "v.patch_embd.weight", 3 * c.patch * c.patch, dim)?,
            position_embd,
            layers,
            std_bias: vector(file, "v.std_bias", dim)?,
            std_scale: vector(file, "v.std_scale", dim)?,
            projection: flat(file, "mm.input_projection.weight", dim, c.out_dim)?,
            config: c,
            pool,
        })
    }

    /// The soft tokens an image of `height × width` (already sized by `fit`) makes.
    pub fn soft_tokens(&self, height: usize, width: usize) -> usize {
        let side = self.config.patch * POOL;
        (height / side) * (width / side)
    }

    /// Encodes an image: `rgb` is `height × width` pixels, row after row, three numbers
    /// each in 0..=1. Both sides must be multiples of `patch × POOL` (see `fit`). Returns
    /// `soft_tokens(height, width)` rows of `out_dim` numbers, in reading order.
    pub fn encode(&self, rgb: &[f32], height: usize, width: usize) -> Result<Vec<f32>> {
        let c = &self.config;
        let features = self.features(rgb, height, width)?;
        let ones = vec![1.0f32; c.dim];
        let mut out = vec![0.0f32; features.len() / c.dim * c.out_dim];
        let mut s = vec![0.0f32; c.dim];
        for (f, o) in features
            .chunks_exact(c.dim)
            .zip(out.chunks_exact_mut(c.out_dim))
        {
            s.copy_from_slice(f);
            norm_in_place(&mut s, &ones, c.norm_eps);
            cpu::matvec(&self.pool, &self.projection, &s, o);
        }
        Ok(out)
    }

    /// The soft tokens before the projection: pooled, scaled and standardized, `dim`
    /// numbers each (transformers' `Gemma4VisionModel` output, to check against).
    pub fn features(&self, rgb: &[f32], height: usize, width: usize) -> Result<Vec<f32>> {
        let c = &self.config;
        let (p, dim) = (c.patch, c.dim);
        let side = p * POOL;
        if height == 0 || width == 0 || !height.is_multiple_of(side) || !width.is_multiple_of(side)
        {
            return Err(Error::Invalid(format!(
                "an image of {width}×{height}: both sides must be multiples of {side}"
            )));
        }
        if rgb.len() != height * width * 3 {
            return Err(Error::Invalid(format!(
                "{} numbers for a {width}×{height} RGB image",
                rgb.len()
            )));
        }
        let (ph, pw) = (height / p, width / p);
        if ph.max(pw) > c.positions {
            return Err(Error::Invalid(format!(
                "an image {pw}×{ph} patches is too large"
            )));
        }
        let n = ph * pw;

        // Patches, embedded, plus their column and row embeddings.
        let mut x = vec![0.0f32; n * dim];
        let mut pixels = vec![0.0f32; 3 * p * p];
        for py in 0..ph {
            for px in 0..pw {
                for ch in 0..3 {
                    for y in 0..p {
                        for xx in 0..p {
                            let v = rgb[((py * p + y) * width + px * p + xx) * 3 + ch];
                            pixels[ch * p * p + y * p + xx] = 2.0 * (v - 0.5);
                        }
                    }
                }
                let i = py * pw + px;
                let row = &mut x[i * dim..(i + 1) * dim];
                cpu::matvec(&self.pool, &self.patch_embd, &pixels, row);
                let col_e = &self.position_embd[px * dim..(px + 1) * dim];
                let row_e =
                    &self.position_embd[(c.positions + py) * dim..(c.positions + py + 1) * dim];
                for ((v, a), b) in row.iter_mut().zip(col_e).zip(row_e) {
                    *v += a + b;
                }
            }
        }

        let ones = vec![1.0f32; dim.max(c.head_dim)];
        let rope = Rope::new(c.head_dim);
        let mut h = vec![0.0f32; dim];
        let mut q = vec![0.0f32; n * dim];
        let mut k = vec![0.0f32; n * dim];
        let mut v = vec![0.0f32; n * dim];
        let mut attn = vec![0.0f32; n * dim];
        let mut t = vec![0.0f32; dim];
        let mut gate = vec![0.0f32; c.ffn_dim];
        let mut up = vec![0.0f32; c.ffn_dim];
        let hd = c.head_dim;
        for layer in &self.layers {
            for i in 0..n {
                let xi = &x[i * dim..(i + 1) * dim];
                cpu::rms_norm(xi, &layer.ln1, c.norm_eps, &mut h);
                let (qi, ki, vi) = (
                    &mut q[i * dim..(i + 1) * dim],
                    &mut k[i * dim..(i + 1) * dim],
                    &mut v[i * dim..(i + 1) * dim],
                );
                cpu::matvec(&self.pool, &layer.q, &h, qi);
                cpu::matvec(&self.pool, &layer.k, &h, ki);
                cpu::matvec(&self.pool, &layer.v, &h, vi);
                let (px, py) = (i % pw, i / pw);
                for head in 0..c.heads {
                    let r = head * hd..(head + 1) * hd;
                    norm_in_place(&mut qi[r.clone()], &layer.q_norm, c.norm_eps);
                    norm_in_place(&mut ki[r.clone()], &layer.k_norm, c.norm_eps);
                    norm_in_place(&mut vi[r.clone()], &ones[..hd], c.norm_eps);
                    rope.apply(&mut qi[r.clone()], px, py);
                    rope.apply(&mut ki[r], px, py);
                }
            }
            attention(&self.pool, &q, &k, &v, &mut attn, n, c.heads, hd);
            for i in 0..n {
                cpu::matvec(&self.pool, &layer.o, &attn[i * dim..(i + 1) * dim], &mut t);
                norm_in_place(&mut t, &layer.attn_post_norm, c.norm_eps);
                let xi = &mut x[i * dim..(i + 1) * dim];
                for (a, b) in xi.iter_mut().zip(&t) {
                    *a += b;
                }
                cpu::rms_norm(xi, &layer.ln2, c.norm_eps, &mut h);
                cpu::matvec(&self.pool, &layer.gate, &h, &mut gate);
                cpu::matvec(&self.pool, &layer.up, &h, &mut up);
                for (g, u) in gate.iter_mut().zip(&up) {
                    *g = gelu_tanh(*g) * u;
                }
                cpu::matvec(&self.pool, &layer.down, &gate, &mut t);
                norm_in_place(&mut t, &layer.ffn_post_norm, c.norm_eps);
                for (a, b) in xi.iter_mut().zip(&t) {
                    *a += b;
                }
            }
        }

        // Each POOL × POOL block of patches averaged into one soft token, scaled and
        // standardized.
        let (th, tw) = (ph / POOL, pw / POOL);
        let mut out = vec![0.0f32; th * tw * dim];
        let root = (dim as f32).sqrt();
        let share = 1.0 / (POOL * POOL) as f32;
        for ty in 0..th {
            for tx in 0..tw {
                let j = ty * tw + tx;
                let s = &mut out[j * dim..(j + 1) * dim];
                for dy in 0..POOL {
                    for dx in 0..POOL {
                        let i = (ty * POOL + dy) * pw + tx * POOL + dx;
                        for (a, b) in s.iter_mut().zip(&x[i * dim..(i + 1) * dim]) {
                            *a += b;
                        }
                    }
                }
                for ((a, bias), scale) in s.iter_mut().zip(&self.std_bias).zip(&self.std_scale) {
                    *a = (*a * share * root - bias) * scale;
                }
            }
        }
        Ok(out)
    }
}

/// The 2D rotary embedding: angles for each half-head's pairs.
struct Rope {
    /// `base^(−2i / (head_dim / 2))` for each of the head_dim / 4 pairs in a half.
    inv_freq: Vec<f32>,
}

impl Rope {
    fn new(head_dim: usize) -> Self {
        let half = head_dim / 2;
        Self {
            inv_freq: (0..half / 2)
                .map(|i| 1.0 / ROPE_BASE.powf((2 * i) as f32 / half as f32))
                .collect(),
        }
    }

    /// Turns one head: its first half by column `px`, its second half by row `py`.
    fn apply(&self, head: &mut [f32], px: usize, py: usize) {
        let half = head.len() / 2;
        let pairs = half / 2;
        for (part, pos) in [(0, px), (1, py)] {
            let x = &mut head[part * half..(part + 1) * half];
            for i in 0..pairs {
                let (sin, cos) = (pos as f32 * self.inv_freq[i]).sin_cos();
                let (a, b) = (x[i], x[i + pairs]);
                x[i] = a * cos - b * sin;
                x[i + pairs] = b * cos + a * sin;
            }
        }
    }
}

/// Every patch attends to every patch, head by head (scale 1: q and k are normed).
#[allow(clippy::too_many_arguments)]
fn attention(
    pool: &Pool,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    n: usize,
    heads: usize,
    hd: usize,
) {
    let dim = heads * hd;
    struct Out(*mut f32);
    unsafe impl Sync for Out {}
    let o = Out(out.as_mut_ptr());
    pool.run(n * heads, &|task| {
        let o = &o;
        let (i, head) = (task / heads, task % heads);
        let qi = &q[i * dim + head * hd..i * dim + (head + 1) * hd];
        let mut scores: Vec<f32> = (0..n)
            .map(|j| {
                let kj = &k[j * dim + head * hd..j * dim + (head + 1) * hd];
                qi.iter().zip(kj).map(|(a, b)| a * b).sum()
            })
            .collect();
        cpu::softmax(&mut scores);
        let mut acc = vec![0.0f32; hd];
        for (j, w) in scores.iter().enumerate() {
            let vj = &v[j * dim + head * hd..j * dim + (head + 1) * hd];
            for (a, b) in acc.iter_mut().zip(vj) {
                *a += w * b;
            }
        }
        // Safety: each task writes only its own patch's head.
        let dst = unsafe { std::slice::from_raw_parts_mut(o.0.add(i * dim + head * hd), hd) };
        dst.copy_from_slice(&acc);
    });
}

fn norm_in_place(x: &mut [f32], weight: &[f32], eps: f32) {
    let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean_square + eps).sqrt();
    for (v, w) in x.iter_mut().zip(weight) {
        *v *= scale * w;
    }
}

fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + 0.044_715 * x * x)).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> VisionConfig {
        VisionConfig {
            dim: 1152,
            ffn_dim: 4304,
            layers: 27,
            heads: 16,
            head_dim: 72,
            patch: 16,
            norm_eps: 1e-6,
            out_dim: 2816,
            positions: 10240,
        }
    }

    #[test]
    fn fit_matches_transformers() {
        let c = config();
        // Values from transformers' get_aspect_ratio_preserving_size.
        assert_eq!(c.fit(768, 1024, 280), (672, 912));
        assert_eq!(c.fit(1000, 1000, 280), (768, 768));
        assert_eq!(c.fit(224, 224, 70), (384, 384));
        assert_eq!(c.fit(10, 4000, 280), (48, 13440));
        assert_eq!(c.fit(3024, 4032, 560), (960, 1296));
    }

    #[test]
    fn rope_turns_column_then_row() {
        let r = Rope::new(8);
        let mut head = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        r.apply(&mut head, 0, 0);
        assert_eq!(head, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        r.apply(&mut head, 1, 2);
        // Pair (0, 2) turned by 1 radian; pair (4, 6) by 2.
        assert!((head[0] - 1f32.cos()).abs() < 1e-6 && (head[2] - 1f32.sin()).abs() < 1e-6);
        assert!((head[4] - 2f32.cos()).abs() < 1e-6 && (head[6] - 2f32.sin()).abs() < 1e-6);
    }

    /// The test image `tools/vision_reference.py` makes too.
    pub(super) fn test_image(h: usize, w: usize) -> Vec<f32> {
        let mut img = vec![0.0f32; h * w * 3];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let (x, y, c) = (x as f64, y as f64, c as f64);
                    img[((y as usize) * w + x as usize) * 3 + c as usize] = (0.5
                        + 0.5 * (x * 0.05 + c * 1.3).sin() * (y * 0.07 - c * 0.4).cos())
                        as f32;
                }
            }
        }
        img
    }

    /// The 26B's encoder against mlx-vlm's in f32 (`tools/vision_reference.py`, written
    /// where `VIREO_VISION_REF` says). Skipped without the files.
    #[test]
    fn gemma4_26b_features_match_mlx_vlm() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let path = root.join("models/gemma-4-26B-A4B-it-GGUF/mmproj-F16.gguf");
        let Some(reference) = std::env::var_os("VIREO_VISION_REF").filter(|_| path.exists()) else {
            eprintln!("skipped: needs the 26B mmproj and VIREO_VISION_REF");
            return;
        };
        let bytes = std::fs::read(reference).unwrap();
        let want: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let file = GgufFile::open(&path).unwrap();
        let v = Gemma4Vision::load(&file, Pool::with_all_cores()).unwrap();
        let (h, w) = (144, 192);
        let got = v.features(&test_image(h, w), h, w).unwrap();
        assert_eq!(got.len(), want.len());
        let (mut worst, mut sum) = (0.0f32, 0.0f32);
        for (a, b) in got.iter().zip(&want) {
            worst = worst.max((a - b).abs());
            sum += b.abs();
        }
        let mean = sum / want.len() as f32;
        eprintln!("largest difference {worst}, mean |x| {mean}");
        for t in 0..3 {
            eprintln!(
                "token {t}: got {:?} want {:?}",
                &got[t * 1152..t * 1152 + 4],
                &want[t * 1152..t * 1152 + 4]
            );
        }
        assert!(worst < 0.02 * mean.max(1.0), "largest difference {worst}");
        let out = v.encode(&test_image(h, w), h, w).unwrap();
        assert_eq!(out.len(), 12 * v.config.out_dim);
        assert!(out.iter().all(|x| x.is_finite()));
    }
}
