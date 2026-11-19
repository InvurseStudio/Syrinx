//! How long a prompt batch's attention takes deep in a context, for the big models'
//! full-attention layer shapes, the grouped tensor kernel (attention_tcg) against the
//! one-head one (attention_tc), and how far their outputs are apart. Not a check: run with
//! `cargo test --release --test prompt_attention_speed -- --ignored --nocapture`.
//! `VIREO_ATT_TOKENS` sets the batch (1,024 by default), `VIREO_ATT_LENGTHS` the positions
//! (comma-separated, e.g. `32768,65536,131072,262144` for long contexts). A long run heats
//! the Mac and slows the later lengths: compare two builds alternately, cool between.
#![cfg(target_os = "macos")]

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
        ("Gemma 4 31B global", 512usize, 32usize, 4usize, 0usize),
        ("Qwen 3.8 27B", 256, 24, 4, 0),
        ("Muse 30B global", 128, 32, 2, 0),
        ("Gemma 4 31B sliding", 256, 32, 16, 1024),
        ("Muse 30B sliding", 128, 32, 2, 2048),
        ("Gemma 4 E4B sliding", 256, 8, 2, 512),
        ("Qwen 3.5 4B", 256, 16, 4, 0),
    ];
    let tokens: usize = std::env::var("VIREO_ATT_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024);
    let settings = ["off", "on"];
    // 1 to 8 tokens: the one-token kernels against attention_tcg.
    let few = tokens <= 8;
    if few {
        gpu.group_prompt_attention(true);
    }
    for (name, hd, heads, kv_heads, window) in shapes {
        let lengths: Vec<usize> = std::env::var("VIREO_ATT_LENGTHS")
            .ok()
            .map(|v| v.split(',').filter_map(|n| n.trim().parse().ok()).collect())
            .unwrap_or_else(|| vec![if few { 200 } else { 1_100 }, 2_304, 8_000, 30_000]);
        for len in lengths {
            let kv_dim = kv_heads * hd;
            // Room for whole chunks of 128 positions past the end.
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
            let q = gpu.upload_f32(&noise(tokens * heads * hd, 3, 1.0)).unwrap();
            let out = gpu.alloc(tokens * heads * hd * 4).unwrap();
            let pos = gpu.alloc(4).unwrap();
            pos.write_i32(&[(len - tokens) as i32]).unwrap();
            let run = || {
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
            let mut first: Option<Vec<f32>> = None;
            let mut line = format!("{name:20} {len:6} positions:");
            for setting in settings {
                if few {
                    gpu.group_few_attention(setting == "on");
                } else {
                    gpu.group_prompt_attention(setting == "on");
                }
                for _ in 0..if few { 50 } else { 1 } {
                    run();
                }
                gpu.synchronize().unwrap();
                let mut got = vec![0f32; tokens * heads * hd];
                out.read_f32(&mut got).unwrap();
                let n = if tokens <= 8 {
                    300
                } else if tokens < 512 {
                    40
                } else {
                    6
                };
                let t = Instant::now();
                for _ in 0..n {
                    run();
                }
                gpu.synchronize().unwrap();
                let ms = t.elapsed().as_secs_f64() * 1e3 / n as f64;
                let off = match &first {
                    Some(f) => f
                        .iter()
                        .zip(&got)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f32::max),
                    None => {
                        first = Some(got);
                        0.0
                    }
                };
                // Two products of tokens × heads × positions × head size.
                let seen = if window > 0 {
                    window.min(len - tokens / 2)
                } else {
                    len - tokens / 2
                };
                let tflops = 4.0 * (tokens * heads * seen * hd) as f64 / (ms / 1e3) / 1e12;
                line += &format!("  {setting}: {ms:6.2} ms ({tflops:4.1} TF, off {off:.1e})");
            }
            eprintln!("{line}");
        }
    }
}
