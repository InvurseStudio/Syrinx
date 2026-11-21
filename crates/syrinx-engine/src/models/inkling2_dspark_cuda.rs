//! Inkling 2's DSpark drafter on the GPU (Mac): the same model as [`super::inkling2_dspark`]
//! (see there), its matrices rounded to Q8_0 on the way up. It reads the features
//! [`Inkling2Cuda`] keeps ([`Inkling2Cuda::keep_features`]) and borrows Inkling's
//! embedding, embedding norm and output matrix.
//!
//! The block's attention has no causal mask: every row sees the cached positions and the
//! whole block. The attention kernels are causal by position, so each row runs on its own
//! as if it were the block's last (and a sliding window 7 wider, to start where the
//! anchor's would).

use super::gpu::{GpuMatrix, Input, gpu_error};
use super::inkling2_cuda::Inkling2Cuda;
use super::inkling2_dspark::Config;
use super::{Error, Result, matrix, vector};
use crate::GgufFile;
use crate::cpu::Matrix;
use crate::cuda::{Buffer, Gpu, QkvPrep};

struct Layer {
    attn_norm: Buffer,
    ffn_norm: Buffer,
    dflare_norm: Buffer,
    q: GpuMatrix,
    k: GpuMatrix,
    v: GpuMatrix,
    o: GpuMatrix,
    k_ctx: GpuMatrix,
    v_ctx: GpuMatrix,
    q_norm: Buffer,
    k_norm: Buffer,
    k_ctx_norm: Buffer,
    gate: GpuMatrix,
    up: GpuMatrix,
    down: GpuMatrix,
    /// `softmax(fuse[layer])`.
    fuse: [f32; 4],
}

/// The most positions a draft layer sees for its block's attention to run as one call
/// (past that, a call a row is faster).
const ONE_CALL_UP_TO: usize = 4096;

pub struct DSparkCuda<'a> {
    pub config: Config,
    layers: Vec<Layer>,
    head_fuse: [f32; 4],
    mask: Vec<f32>,
    output_norm: Buffer,
    head_norm: Buffer,
    /// The recurrent head's token table stays in the file (a row a slot).
    markov: Matrix<'a>,
    joint: GpuMatrix,
    joint_bias: Buffer,
    hidden_out: GpuMatrix,
    hidden_out_bias: Buffer,
    /// A shorter vocabulary to guess from (Inkling's output rows of those tokens), and
    /// the tokens: guessing reads 1/6 of the output matrix for 32K tokens. `None`: all.
    vocab: Option<(GpuMatrix, Vec<u32>)>,
}

/// The drafter's cache for a conversation, and its working space.
pub struct DSparkCudaState {
    /// Per layer, f16 keys and values a position (with room for a block past the end).
    keys: Vec<Buffer>,
    values: Vec<Buffer>,
    /// The recurrent head's context, a row a position.
    context: Buffer,
    len: usize,
    capacity: usize,
    batch: usize,
    /// The caches hold f16.
    half: bool,
    // Working space.
    pos_add: Buffer,
    pos_block: Buffer,
    pos_mask: Buffer,
    mixed: Buffer,
    k_new: Buffer,
    v_new: Buffer,
    q_dummy: Buffer,
    x: Buffer,
    h: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    /// The queries and the attention's output in [kv][row][g] order.
    qp: Buffer,
    attn_p: Buffer,
    proj: Buffer,
    gate: Buffer,
    up: Buffer,
    hidden: Buffer,
    joint_in: Buffer,
    j: Buffer,
    latent: Buffer,
    corr: Buffer,
    logits: Buffer,
}

impl DSparkCudaState {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Forgets positions from `len` on.
    pub fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }
}

fn softmax4(w: &[f32]) -> [f32; 4] {
    let mut v = [w[0], w[1], w[2], w[3]];
    crate::cpu::softmax(&mut v);
    v
}

impl<'a> DSparkCuda<'a> {
    /// Loads the drafter onto Inkling's GPU (`gpu`, [`Inkling2Cuda::gpu`]).
    pub fn load(file: &'a GgufFile, gpu: &Gpu) -> Result<Self> {
        let c = Config::from_gguf(file)?;
        let (dim, q, kv) = (c.dim, c.heads * c.head_dim, c.kv_heads * c.head_dim);
        let g = gpu;
        let up = |name: &str, cols: usize, rows: usize| -> Result<GpuMatrix> {
            let m = matrix(file, name, cols, rows)?;
            // Unsloth's Q8_0 drafter goes up as it is; BF16 and F16 are rounded to it.
            if m.kind == crate::TensorType::Q8_0 {
                return GpuMatrix::upload(g, m);
            }
            GpuMatrix::upload_q8_0(g, &m)?
                .ok_or_else(|| Error::Unsupported(format!("{name} to Q8_0")))
        };
        let vec_up = |name: &str, len: usize| -> Result<Buffer> {
            let v = vector(file, name, len)?;
            let b = g.alloc(len * 4).map_err(gpu_error)?;
            b.write_f32(&v).map_err(gpu_error)?;
            Ok(b)
        };
        let targets = c.target_layers.len();
        let fuse_m = matrix(file, "dspark_fuse.weight", targets, c.layers + 1)?;
        let fuse_row = |r: usize| {
            let mut w = vec![0.0; targets];
            fuse_m.decode_row(r, &mut w);
            softmax4(&w)
        };
        let name = |i: usize, n: &str| format!("blk.{i}.{n}.weight");
        let layers = (0..c.layers)
            .map(|i| {
                Ok(Layer {
                    attn_norm: vec_up(&name(i, "attn_norm"), dim)?,
                    ffn_norm: vec_up(&name(i, "ffn_norm"), dim)?,
                    dflare_norm: vec_up(&name(i, "dflare_norm"), dim)?,
                    q: up(&name(i, "attn_q"), dim, q)?,
                    k: up(&name(i, "attn_k"), dim, kv)?,
                    v: up(&name(i, "attn_v"), dim, kv)?,
                    o: up(&name(i, "attn_output"), q, dim)?,
                    k_ctx: up(&name(i, "attn_k_ctx"), dim, kv)?,
                    v_ctx: up(&name(i, "attn_v_ctx"), dim, kv)?,
                    q_norm: vec_up(&name(i, "attn_q_norm"), c.head_dim)?,
                    k_norm: vec_up(&name(i, "attn_k_norm"), c.head_dim)?,
                    k_ctx_norm: vec_up(&name(i, "attn_k_ctx_norm"), c.head_dim)?,
                    gate: up(&name(i, "ffn_gate"), dim, c.ffn)?,
                    up: up(&name(i, "ffn_up"), dim, c.ffn)?,
                    down: up(&name(i, "ffn_down"), c.ffn, dim)?,
                    fuse: fuse_row(i),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let markov = {
            let rows = file
                .tensor("markov_w1.weight")
                .and_then(|t| t.dims.get(1).copied())
                .ok_or_else(|| Error::Missing("tensor markov_w1.weight".into()))?
                as usize;
            matrix(file, "markov_w1.weight", c.rank, rows)?
        };
        let joint_in = 2 * c.rank + 2 * dim;
        Ok(DSparkCuda {
            layers,
            head_fuse: fuse_row(c.layers),
            mask: vector(file, "mask_embd.weight", dim)?,
            output_norm: vec_up("output_norm.weight", dim)?,
            head_norm: vec_up("dspark_norm.weight", dim)?,
            markov,
            joint: up("joint_proj.weight", joint_in, 3 * c.rank)?,
            joint_bias: vec_up("joint_proj.bias", 3 * c.rank)?,
            hidden_out: up("dspark_hidden_out.weight", c.rank, dim)?,
            hidden_out_bias: vec_up("dspark_hidden_out.bias", dim)?,
            vocab: None,
            config: c,
        })
    }

    /// Guesses only `tokens` from now on (the likeliest-to-be-written ones, say), with
    /// those rows of Inkling's output matrix (from Inkling's file, `target_file`) on the
    /// GPU. A guess outside them is never made, which costs only kept guesses.
    pub fn set_vocab(&mut self, gpu: &Gpu, target_file: &GgufFile, tokens: &[u32]) -> Result<()> {
        let t = target_file
            .tensor("output.weight")
            .or_else(|| target_file.tensor("token_embd.weight"))
            .ok_or_else(|| Error::Missing("Inkling's output matrix".into()))?;
        let (cols, rows) = (t.dims[0] as usize, t.dims[1] as usize);
        let name = if target_file.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
        let m = matrix(target_file, name, cols, rows)?;
        let mut data = Vec::with_capacity(tokens.len() * m.row_bytes());
        for &tok in tokens {
            if (tok as usize) >= rows {
                return Err(Error::Invalid(format!("token {tok} is past the vocabulary")));
            }
            data.extend_from_slice(m.row(tok as usize));
        }
        let sub = Matrix { kind: m.kind, rows: tokens.len(), cols, data: &data };
        self.vocab = Some((GpuMatrix::upload(gpu, sub)?, tokens.to_vec()));
        Ok(())
    }

    /// Checks that the drafter was made for `target`.
    pub fn check_target(&self, target: &Inkling2Cuda) -> Result<()> {
        let (c, t) = (&self.config, &target.config);
        let visits = t.slots();
        if t.dim != c.dim || c.target_layers.last() != Some(&visits) || t.vocab < c.vocab {
            return Err(Error::Invalid(format!(
                "the drafter's features ({:?}, {} wide) don't fit this Inkling ({visits} visits, {} wide)",
                c.target_layers, c.dim, t.dim
            )));
        }
        Ok(())
    }

    /// The layer visits whose inputs the drafter reads (for `keep_features`).
    pub fn target_visits(&self) -> &[usize] {
        let t = &self.config.target_layers;
        &t[..t.len() - 1]
    }

    /// A cache for `capacity` positions, taking features `batch` tokens at a time.
    pub fn state(&self, target: &Inkling2Cuda, capacity: usize, batch: usize) -> Result<DSparkCudaState> {
        let c = &self.config;
        let g = target.gpu();
        let (dim, kv, q) = (c.dim, c.kv_heads * c.head_dim, c.heads * c.head_dim);
        let rows = c.block;
        let f = |n: usize| g.alloc(n.max(1) * 4).map_err(gpu_error);
        // f16 caches (half the reads of f32: drafting at 32K took 7.6 ms, not 10.7).
        let half = true;
        let kv_bytes = (capacity + rows) * kv * if half { 2 } else { 4 };
        let cache = || g.alloc(kv_bytes).map_err(gpu_error);
        Ok(DSparkCudaState {
            keys: (0..c.layers).map(|_| cache()).collect::<Result<_>>()?,
            values: (0..c.layers).map(|_| cache()).collect::<Result<_>>()?,
            half,
            context: f(capacity * dim)?,
            len: 0,
            capacity,
            batch,
            pos_add: f(1)?,
            pos_block: f(1)?,
            pos_mask: f(1)?,
            mixed: f(batch * dim)?,
            k_new: f(batch * kv)?,
            v_new: f(batch * kv)?,
            q_dummy: f(batch * kv)?,
            x: f(rows * dim)?,
            h: f(rows * dim)?,
            q: f(rows * q)?,
            k: f(rows * kv)?,
            v: f(rows * kv)?,
            attn: f(rows * q)?,
            qp: f(rows * q)?,
            attn_p: f(rows * q)?,
            proj: f(rows * dim)?,
            gate: f(rows * c.ffn)?,
            up: f(rows * c.ffn)?,
            hidden: f(rows * dim)?,
            joint_in: f(2 * c.rank + 2 * dim)?,
            j: f(3 * c.rank)?,
            latent: f(c.rank)?,
            corr: f(dim)?,
            logits: f(1)?,
        })
    }

    /// The cache's first `len` positions, appended to `out` (each layer's keys and
    /// values, then the recurrent head's context).
    pub fn export(&self, s: &DSparkCudaState, len: usize, out: &mut Vec<u8>) -> Result<()> {
        let c = &self.config;
        assert!(len <= s.len, "{len} positions of a drafter's cache of {}", s.len);
        let kv = c.kv_heads * c.head_dim * if s.half { 2 } else { 4 };
        let mut part = |b: &Buffer, n: usize| -> Result<()> {
            let at = out.len();
            out.resize(at + n, 0);
            b.read_bytes_at(0, &mut out[at..]).map_err(gpu_error)
        };
        for l in 0..c.layers {
            part(&s.keys[l], len * kv)?;
            part(&s.values[l], len * kv)?;
        }
        part(&s.context, len * c.dim * 4)
    }

    /// Puts back what `export` saved of `len` positions: an error, changing nothing, when
    /// it doesn't fit this cache (another drafter's, or too long).
    pub fn import(&self, s: &mut DSparkCudaState, len: usize, bytes: &[u8]) -> Result<()> {
        let c = &self.config;
        let kv = c.kv_heads * c.head_dim * if s.half { 2 } else { 4 };
        let ctx = len * c.dim * 4;
        if len > s.capacity || bytes.len() != c.layers * 2 * len * kv + ctx {
            return Err(Error::Invalid("a drafter's cache from another drafter".into()));
        }
        let mut at = 0;
        let mut put = |b: &Buffer, n: usize| -> Result<()> {
            b.write_bytes_at(0, &bytes[at..at + n]).map_err(gpu_error)?;
            at += n;
            Ok(())
        };
        for l in 0..c.layers {
            put(&s.keys[l], len * kv)?;
            put(&s.values[l], len * kv)?;
        }
        put(&s.context, ctx)?;
        s.len = len;
        Ok(())
    }

    /// The features of the first `rows` of the tokens Inkling just read (at positions
    /// `s.len()` on), as [`Inkling2Cuda::features`] gives them (a batch of `batch_rows`),
    /// into the cache.
    pub fn add(
        &self,
        target: &Inkling2Cuda,
        s: &mut DSparkCudaState,
        features: (crate::cuda::CuDevicePtr, usize),
        rows: usize,
    ) -> Result<()> {
        self.add_from(target, s, features, 0, rows)
    }

    /// As [`add`](Self::add), for the batch's rows `from..from + rows`.
    pub fn add_from(
        &self,
        target: &Inkling2Cuda,
        s: &mut DSparkCudaState,
        features: (crate::cuda::CuDevicePtr, usize),
        from: usize,
        rows: usize,
    ) -> Result<()> {
        let c = &self.config;
        let g = target.gpu();
        let e = gpu_error;
        let (ptr, batch_rows) = features;
        assert!(from + rows <= batch_rows, "{rows} from {from} of a batch of {batch_rows}");
        let ptr = ptr + (from * c.dim * 4) as u64;
        if rows == 0 {
            return Ok(());
        }
        if rows > s.batch || s.len + rows > s.capacity {
            return Err(Error::Invalid(format!(
                "{rows} drafter positions at {} (batch {}, capacity {})",
                s.len, s.batch, s.capacity
            )));
        }
        let (dim, hd, kv) = (c.dim, c.head_dim, c.kv_heads * c.head_dim);
        let block = batch_rows * dim;
        // The positions buffer is the host's to write: nothing queued may still read it.
        g.synchronize().map_err(e)?;
        s.pos_add.write_i32(&[s.len as i32]).map_err(e)?;
        for (l, layer) in self.layers.iter().enumerate() {
            g.dspark_mix(ptr, block, layer.fuse, layer.dflare_norm.ptr(), s.mixed.ptr(), rows, dim, c.norm_eps)
                .map_err(e)?;
            let x = Input::new(g, s.mixed.ptr(), rows, None);
            layer.k_ctx.mul(&x, s.k_new.ptr())?;
            layer.v_ctx.mul(&x, s.v_new.ptr())?;
            g.qkv_prep(
                &QkvPrep {
                    q: s.k_new.ptr(),
                    q_stride: kv,
                    q_head_stride: hd,
                    q_norm: Some(layer.k_ctx_norm.ptr()),
                    q_out: s.q_dummy.ptr(),
                    k: s.k_new.ptr(),
                    k_norm: Some(layer.k_ctx_norm.ptr()),
                    v: s.v_new.ptr(),
                    v_norm: false,
                    cache: Some((s.keys[l].ptr(), s.values[l].ptr())),
                    heads: c.kv_heads,
                    kv_heads: c.kv_heads,
                    head_dim: hd,
                    rope_dims: hd,
                    rope_base: c.rope_base,
                    rope_factors: None,
                    neox: true,
                    pos: s.pos_add.ptr(),
                    eps: c.norm_eps,
                    half_cache: s.half,
                    ring: 0,
                },
                rows,
            )
            .map_err(e)?;
        }
        g.dspark_mix(
            ptr,
            block,
            self.head_fuse,
            self.head_norm.ptr(),
            s.context.f32_at(s.len * dim),
            rows,
            dim,
            c.norm_eps,
        )
        .map_err(e)?;
        s.len += rows;
        Ok(())
    }

    /// Up to `count` guesses of the tokens after `anchor` (Inkling's choice for position
    /// `s.len()`, not read yet).
    pub fn draft(
        &self,
        s: &mut DSparkCudaState,
        target: &Inkling2Cuda,
        anchor: u32,
        count: usize,
    ) -> Result<Vec<u32>> {
        let c = &self.config;
        let g = target.gpu();
        let e = gpu_error;
        let (dim, hd) = (c.dim, c.head_dim);
        let (q_dim, kv) = (c.heads * hd, c.kv_heads * hd);
        let rows = c.block;
        let count = count.min(rows - 1);
        let pos = s.len;
        if pos == 0 || count == 0 {
            return Ok(Vec::new());
        }
        g.synchronize().map_err(e)?;
        // The block: the anchor's embedding (normed below), then the mask rows.
        let mut x = vec![0.0f32; rows * dim];
        target.embedding.decode_row(anchor as usize, &mut x[..dim]);
        for r in 1..rows {
            x[r * dim..(r + 1) * dim].copy_from_slice(&self.mask);
        }
        s.x.write_f32(&x).map_err(e)?;
        s.pos_block.write_i32(&[pos as i32]).map_err(e)?;
        s.pos_mask.write_i32(&[(pos + rows - 1) as i32]).map_err(e)?;
        let rank = c.rank;
        // The recurrent head's input: state 0, the anchor's row, (hidden), the context
        // of the position before the anchor.
        let mut head_start = vec![0.0f32; 2 * rank];
        self.markov.decode_row(anchor as usize, &mut head_start[rank..]);
        s.joint_in.write_f32_at(0, &head_start).map_err(e)?;
        g.rms_norm_rows(s.x.ptr(), target.embedding_norm.ptr(), s.x.ptr(), 1, dim, target.config.norm_eps)
            .map_err(e)?;

        let scale = 1.0 / (hd as f32).sqrt();
        for (l, layer) in self.layers.iter().enumerate() {
            g.rms_norm_rows(s.x.ptr(), layer.attn_norm.ptr(), s.h.ptr(), rows, dim, c.norm_eps)
                .map_err(e)?;
            let h = Input::new(g, s.h.ptr(), rows, None);
            layer.q.mul(&h, s.q.ptr())?;
            layer.k.mul(&h, s.k.ptr())?;
            layer.v.mul(&h, s.v.ptr())?;
            g.qkv_prep(
                &QkvPrep {
                    q: s.q.ptr(),
                    q_stride: q_dim,
                    q_head_stride: hd,
                    q_norm: Some(layer.q_norm.ptr()),
                    q_out: s.q.ptr(),
                    k: s.k.ptr(),
                    k_norm: Some(layer.k_norm.ptr()),
                    v: s.v.ptr(),
                    v_norm: false,
                    cache: Some((s.keys[l].ptr(), s.values[l].ptr())),
                    heads: c.heads,
                    kv_heads: c.kv_heads,
                    head_dim: hd,
                    rope_dims: hd,
                    rope_base: c.rope_base,
                    rope_factors: None,
                    neox: true,
                    pos: s.pos_block.ptr(),
                    eps: c.norm_eps,
                    half_cache: s.half,
                    ring: 0,
                },
                rows,
            )
            .map_err(e)?;
            // Each row sees the whole block: every row's heads attend as the block's last
            // position. Short contexts: all of a key/value head's rows in one call; long
            // ones: a call a row (faster there).
            let window = if c.is_swa[l] { c.window + rows - 1 } else { 0 };
            let group = c.heads / c.kv_heads;
            let seen = if window > 0 { window.min(pos + rows) } else { pos + rows };
            if seen <= ONE_CALL_UP_TO {
                g.dspark_permute(s.q.ptr(), s.qp.ptr(), rows, c.kv_heads, group, hd, false)
                    .map_err(e)?;
                g.attention(
                    s.half,
                    s.qp.ptr(),
                    s.keys[l].ptr(),
                    s.values[l].ptr(),
                    s.attn_p.ptr(),
                    s.pos_mask.ptr(),
                    window,
                    rows * c.heads,
                    1,
                    hd,
                    kv,
                    rows * group,
                    scale,
                )
                .map_err(e)?;
                g.dspark_permute(s.attn_p.ptr(), s.attn.ptr(), rows, c.kv_heads, group, hd, true)
                    .map_err(e)?;
            } else {
                for r in 0..rows {
                    g.attention(
                        s.half,
                        s.q.f32_at(r * q_dim),
                        s.keys[l].ptr(),
                        s.values[l].ptr(),
                        s.attn.f32_at(r * q_dim),
                        s.pos_mask.ptr(),
                        window,
                        c.heads,
                        1,
                        hd,
                        kv,
                        group,
                        scale,
                    )
                    .map_err(e)?;
                }
            }
            layer.o.mul(&Input::new(g, s.attn.ptr(), rows, None), s.proj.ptr())?;
            g.add(s.x.ptr(), s.proj.ptr(), rows * dim).map_err(e)?;
            g.rms_norm_rows(s.x.ptr(), layer.ffn_norm.ptr(), s.h.ptr(), rows, dim, c.norm_eps)
                .map_err(e)?;
            let h = Input::new(g, s.h.ptr(), rows, None);
            layer.gate.mul(&h, s.gate.ptr())?;
            layer.up.mul(&h, s.up.ptr())?;
            g.swiglu(s.gate.ptr(), s.up.ptr(), c.ffn, c.ffn, rows)
                .map_err(e)?;
            layer.down.mul(&Input::new(g, s.gate.ptr(), rows, None), s.proj.ptr())?;
            g.add(s.x.ptr(), s.proj.ptr(), rows * dim).map_err(e)?;
        }
        g.rms_norm_rows(s.x.ptr(), self.output_norm.ptr(), s.hidden.ptr(), rows, dim, c.norm_eps)
            .map_err(e)?;

        // The recurrent head, a slot at a time.
        let ctx_at = 2 * rank + dim;
        g.copy(
            s.joint_in.f32_at(ctx_at),
            s.context.f32_at((pos - 1) * dim),
            dim * 4,
        )
        .map_err(e)?;
        let padded = target.output.rows;
        if s.logits.len_bytes() < padded * 4 {
            s.logits = g.alloc(padded * 4).map_err(e)?;
        }
        let mut guesses = Vec::with_capacity(count);
        let mut row = vec![0.0f32; rank];
        for slot in 1..=count {
            if slot > 1 {
                self.markov.decode_row(*guesses.last().unwrap() as usize, &mut row);
                s.joint_in.write_f32_at(rank, &row).map_err(e)?;
            }
            g.copy(s.joint_in.f32_at(2 * rank), s.hidden.f32_at(slot * dim), dim * 4)
                .map_err(e)?;
            self.joint
                .mul(&Input::new(g, s.joint_in.ptr(), 1, None), s.j.ptr())?;
            g.add(s.j.ptr(), self.joint_bias.ptr(), 3 * rank).map_err(e)?;
            g.dspark_head_act(s.j.ptr(), s.joint_in.ptr(), s.latent.ptr(), rank)
                .map_err(e)?;
            self.hidden_out
                .mul(&Input::new(g, s.latent.ptr(), 1, None), s.corr.ptr())?;
            g.add(s.corr.ptr(), self.hidden_out_bias.ptr(), dim).map_err(e)?;
            g.add(s.corr.ptr(), s.hidden.f32_at(slot * dim), dim).map_err(e)?;
            match &self.vocab {
                Some((head, tokens)) => {
                    head.mul(&Input::new(g, s.corr.ptr(), 1, None), s.logits.ptr())?;
                    let i = g.argmax(s.logits.ptr(), tokens.len()).map_err(e)? as usize;
                    guesses.push(tokens[i]);
                }
                None => {
                    target
                        .output
                        .mul(&Input::new(g, s.corr.ptr(), 1, None), s.logits.ptr())?;
                    guesses.push(g.argmax(s.logits.ptr(), c.vocab).map_err(e)?);
                }
            }
        }
        Ok(guesses)
    }
}
