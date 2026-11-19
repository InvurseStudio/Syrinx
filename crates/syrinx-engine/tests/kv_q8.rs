//! Keys and values kept in Q8_0 (`VIREO_KV_Q8=1`): what `qkv_prep` writes,
//! and attention over it, one token at a time and in a batch. In its own file, since the
//! setting is read once a process. Skipped (with a note) on a computer without a GPU.

use syrinx_engine::cuda::{Gpu, QkvPrep};
use syrinx_engine::half::f16_to_f32;

#[test]
fn attention_over_a_q8_cache_matches_f64() {
    // Before anything asks: the setting is read once.
    unsafe { std::env::set_var("VIREO_KV_Q8", "1") };
    let gpu = match Gpu::open() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("skipped: {e}");
            return;
        }
    };
    // Heads that aren't whole blocks of 32, or under 128, stay in f16.
    assert_eq!(Gpu::kv_head_bytes(true, 144), 288);
    assert_eq!(Gpu::kv_head_bytes(true, 64), 128);
    // (head size, query heads, key/value heads, sliding window, scale, first position,
    // tokens): batches (the cache put back in f16 for the prompt kernels), and a few tokens
    // deep in a context (the one-token kernels, reading the 8 bits themselves).
    for (hd, heads, kv_heads, window, scale, p0, tokens) in [
        (128, 8, 2, 0, 0.088f32, 30, 100),
        (256, 8, 1, 40, 1.0, 30, 100),
        (512, 8, 1, 0, 1.0, 30, 100),
        (128, 8, 8, 0, 0.088, 900, 24),
        (256, 16, 4, 0, 0.0625, 2000, 7),
        (512, 8, 2, 0, 1.0, 2000, 3),
        (128, 8, 2, 40, 0.088, 300, 4),
    ] {
        let len = p0 + tokens;
        let kv_dim = kv_heads * hd;
        let head_bytes = Gpu::kv_head_bytes(true, hd);
        assert_eq!(head_bytes, hd * 17 / 16, "the cache isn't in Q8_0");
        let wave = |i: usize, k: usize| (((i * k) % 1009) as f32 / 504.5 - 1.0) * 0.6;
        let q: Vec<f32> = (0..tokens * heads * hd).map(|i| wave(i, 7919)).collect();
        let k: Vec<f32> = (0..len * kv_dim).map(|i| wave(i, 6007)).collect();
        let v: Vec<f32> = (0..len * kv_dim).map(|i| wave(i, 3001)).collect();
        let (k_gpu, v_gpu) = (gpu.upload_f32(&k).unwrap(), gpu.upload_f32(&v).unwrap());
        let keys = gpu.alloc(len * kv_heads * head_bytes).unwrap();
        let values = gpu.alloc(len * kv_heads * head_bytes).unwrap();
        let pos = gpu.alloc(4).unwrap();

        // The cache as qkv_prep writes it (no norm, no rotation: the numbers as given), 8
        // positions a call.
        const STEP: usize = 8;
        let spare_q = gpu.upload_f32(&vec![0f32; STEP * heads * hd]).unwrap();
        let spare_out = gpu.alloc(STEP * heads * hd * 4).unwrap();
        for at in (0..len).step_by(STEP) {
            let n = STEP.min(len - at);
            pos.write_i32(&[at as i32]).unwrap();
            gpu.qkv_prep(
                &QkvPrep {
                    q: spare_q.ptr(),
                    q_stride: heads * hd,
                    q_head_stride: hd,
                    q_norm: None,
                    q_out: spare_out.ptr(),
                    k: k_gpu.f32_at(at * kv_dim),
                    k_norm: None,
                    v: v_gpu.f32_at(at * kv_dim),
                    v_norm: false,
                    cache: Some((keys.ptr(), values.ptr())),
                    heads,
                    kv_heads,
                    head_dim: hd,
                    rope_dims: 0,
                    rope_base: 10000.0,
                    rope_factors: None,
                    neox: false,
                    pos: pos.ptr(),
                    eps: 1e-6,
                    half_cache: true,
                    ring: 0,
                },
                n,
            )
            .unwrap();
        }

        // What it holds, and how far that is from what was given: half a step of each
        // block of 32's scale (its largest over 127), and the scale's own rounding to f16.
        let held = |cache: &syrinx_engine::cuda::Buffer, given: &[f32]| -> Vec<f32> {
            let mut bytes = vec![0u8; len * kv_heads * head_bytes];
            cache.read_bytes_at(0, &mut bytes).unwrap();
            let mut out = vec![0f32; len * kv_dim];
            for head in 0..len * kv_heads {
                let b = &bytes[head * head_bytes..][..head_bytes];
                for block in 0..hd / 32 {
                    let scale =
                        f16_to_f32(u16::from_le_bytes([b[hd + 2 * block], b[hd + 2 * block + 1]]));
                    let at = head * hd + block * 32;
                    let top = given[at..at + 32].iter().fold(0f32, |m, x| m.max(x.abs()));
                    for i in 0..32 {
                        let got = b[block * 32 + i] as i8 as f32 * scale;
                        let off = (got - given[at + i]).abs();
                        assert!(
                            off <= top / 127.0 * 0.51 + top * 1e-3,
                            "head size {hd}: {got} for {} (block's largest {top})",
                            given[at + i]
                        );
                        out[at + i] = got;
                    }
                }
            }
            out
        };
        let (k8, v8) = (held(&keys, &k), held(&values, &v));

        // The exact answer over what the cache holds, in f64.
        let group = heads / kv_heads;
        let mut expected = vec![0f64; tokens * heads * hd];
        for t in 0..tokens {
            let at = p0 + t;
            let first = if window > 0 {
                (at + 1).saturating_sub(window)
            } else {
                0
            };
            for h in 0..heads {
                let kv = (h / group) * hd;
                let qh = &q[(t * heads + h) * hd..][..hd];
                let scores: Vec<f64> = (first..=at)
                    .map(|j| {
                        (0..hd)
                            .map(|d| qh[d] as f64 * k8[j * kv_dim + kv + d] as f64)
                            .sum::<f64>()
                            * scale as f64
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f64 = weights.iter().sum();
                for d in 0..hd {
                    expected[(t * heads + h) * hd + d] = (first..=at)
                        .zip(&weights)
                        .map(|(j, w)| w * v8[j * kv_dim + kv + d] as f64)
                        .sum::<f64>()
                        / total;
                }
            }
        }
        let largest = expected.iter().fold(0f64, |m, v| m.max(v.abs()));
        let worst = |got: &[f32]| {
            got.iter()
                .zip(&expected)
                .map(|(g, e)| (*g as f64 - e).abs())
                .fold(0f64, f64::max)
        };
        let q_gpu = gpu.upload_f32(&q).unwrap();
        let out = gpu.alloc(tokens * heads * hd * 4).unwrap();
        let run = |t0: usize, n: usize| {
            pos.write_i32(&[(p0 + t0) as i32]).unwrap();
            gpu.attention(
                true,
                q_gpu.f32_at(t0 * heads * hd),
                keys.ptr(),
                values.ptr(),
                out.f32_at(t0 * heads * hd),
                pos.ptr(),
                window,
                heads,
                n,
                hd,
                kv_dim,
                group,
                scale,
            )
            .unwrap();
        };
        let mut got = vec![0f32; tokens * heads * hd];
        run(0, tokens);
        out.read_f32(&mut got).unwrap();
        let batch = worst(&got);
        // A batch on the tensor cores leaves its queries in f16 where they lay: as given
        // again for the one-token kernel.
        q_gpu.write_f32(&q).unwrap();
        for t in 0..tokens {
            run(t, 1);
        }
        out.read_f32(&mut got).unwrap();
        let single = worst(&got);
        eprintln!(
            "head size {hd}, window {window}, {tokens} tokens: batch off by {batch:.1e}, one at a time {single:.1e} (largest {largest:.2})"
        );
        assert!(single <= largest * 1e-4, "one at a time: {single}");
        assert!(batch <= largest.max(0.05) * 2e-3, "batch: {batch}");
    }
}
