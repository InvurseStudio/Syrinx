//! How long a prompt batch's attention takes deep in a context on CUDA, for the small
//! models' layer shapes, with a sum of the output to tell one build's numbers from
//! another's. Not a check: run with
//! `cargo test --release --test prompt_attention_speed_cuda -- --ignored --nocapture`
//! (`VIREO_ATT_TOKENS`, the batch's size, default 1024).
#![cfg(not(target_os = "macos"))]

use std::time::Instant;

use syrinx_engine::cuda::Gpu;
use syrinx_engine::half::f32_to_f16;

fn noise(n: usize, seed: u64, size: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * size
        })
        .collect()
}

fn halves(x: &[f32]) -> Vec<u8> {
    x.iter()
        .flat_map(|&v| f32_to_f16(v).to_le_bytes())
        .collect()
}

#[test]
#[ignore]
fn a_batch_of_attention_deep_in_a_context() {
    let gpu = Gpu::open().unwrap();
    // (name, head size, query heads, key/value heads, sliding window)
    let shapes = [
        ("Gemma 4 E2B global", 512usize, 8usize, 1usize, 0usize),
        ("Gemma 4 E4B global", 512, 8, 2, 0),
        ("Gemma 4 sliding", 256, 8, 2, 512),
        ("Qwen 3.5 4B/9B", 256, 16, 4, 0),
    ];
    let tokens: usize = std::env::var("VIREO_ATT_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024);
    for (name, hd, heads, kv_heads, window) in shapes {
        for len in [2_304usize, 8_000, 30_000] {
            if len < tokens {
                continue;
            }
            let kv_dim = kv_heads * hd;
            let cache = (len + 256) * kv_dim * 2;
            let keys = gpu.alloc(cache).unwrap();
            let values = gpu.alloc(cache).unwrap();
            gpu.zero(&keys).unwrap();
            gpu.zero(&values).unwrap();
            let row = len.min(2048) * kv_dim;
            let (k, v) = (halves(&noise(row, 1, 0.2)), halves(&noise(row, 2, 1.0)));
            let mut at = 0;
            while at < len * kv_dim * 2 {
                let n = k.len().min(len * kv_dim * 2 - at);
                keys.write_bytes_at(at, &k[..n]).unwrap();
                values.write_bytes_at(at, &v[..n]).unwrap();
                at += n;
            }
            let floats = tokens * heads * hd;
            // The kernel turns the queries to f16 where they lie: a fresh copy each run.
            let q0 = gpu.upload_f32(&noise(floats, 3, 1.0)).unwrap();
            let q = gpu.alloc(floats * 4).unwrap();
            let out = gpu.alloc(floats * 4).unwrap();
            let pos = gpu.alloc(4).unwrap();
            pos.write_i32(&[(len - tokens) as i32]).unwrap();
            let run = || {
                gpu.copy(q.ptr(), q0.ptr(), floats * 4).unwrap();
                gpu.attention(
                    true,
                    q.ptr(),
                    keys.ptr(),
                    values.ptr(),
                    out.ptr(),
                    pos.ptr(),
                    window,
                    heads,
                    tokens,
                    hd,
                    kv_dim,
                    heads / kv_heads,
                    0.06,
                )
                .unwrap()
            };
            run();
            gpu.synchronize().unwrap();
            let mut got = vec![0f32; floats];
            out.read_f32(&mut got).unwrap();
            let sum: f64 = got.iter().map(|&v| f64::from(v.abs())).sum();
            let n = if tokens < 512 { 40 } else { 6 };
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let t = Instant::now();
                for _ in 0..n {
                    run();
                }
                gpu.synchronize().unwrap();
                best = best.min(t.elapsed().as_secs_f64() * 1e3 / n as f64);
            }
            let seen = if window > 0 {
                window.min(len - tokens / 2)
            } else {
                len - tokens / 2
            };
            let tflops = 4.0 * (tokens * heads * seen * hd) as f64 / (best / 1e3) / 1e12;
            eprintln!(
                "{name:20} {len:6} positions: {best:7.3} ms ({tflops:4.1} TF)  sum {sum:.6e}"
            );
        }
    }
}
