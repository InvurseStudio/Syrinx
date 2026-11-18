//! How long one generated token's attention takes deep in a context, for the big models'
//! layer shapes. Not a check: run with
//! `cargo test --release --test decode_attention_speed -- --ignored --nocapture`.

use std::time::Instant;

use syrinx_engine::cuda::Gpu;

#[test]
#[ignore]
fn one_token_attention_deep_in_a_context() {
    let gpu = Gpu::open().unwrap();
    // (name, head size, query heads, key/value heads, sliding window)
    let shapes = [
        ("Qwen 3.8 27B", 256, 24, 4, 0),
        ("Gemma 4 31B global", 512, 32, 4, 0),
        ("Gemma 4 31B sliding", 256, 32, 16, 1024),
        ("Muse 30B global", 128, 32, 2, 0),
        ("Muse 30B sliding", 128, 32, 2, 2048),
    ];
    for (name, hd, heads, kv_heads, window) in shapes {
        for len in [64usize, 2_300, 8_000, 30_000] {
            let kv_dim = kv_heads * hd;
            let keys = gpu.alloc(len * kv_dim * 2).unwrap();
            let values = gpu.alloc(len * kv_dim * 2).unwrap();
            gpu.zero(&keys).unwrap();
            gpu.zero(&values).unwrap();
            let q = gpu.upload_f32(&vec![0.01; heads * hd]).unwrap();
            let out = gpu.alloc(heads * hd * 4).unwrap();
            let pos = gpu.alloc(4).unwrap();
            pos.write_i32(&[len as i32 - 1]).unwrap();
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
                    1,
                    hd,
                    kv_dim,
                    heads / kv_heads,
                    0.06,
                )
                .unwrap()
            };
            for _ in 0..20 {
                run();
            }
            gpu.synchronize().unwrap();
            let n = 200;
            let t = Instant::now();
            for _ in 0..n {
                run();
            }
            gpu.synchronize().unwrap();
            let us = t.elapsed().as_secs_f64() * 1e6 / n as f64;
            let read = if window > 0 { len.min(window) } else { len };
            let gb = (2 * read * kv_dim * 2) as f64 / 1e9;
            eprintln!(
                "{name:20} {len:6} positions: {us:7.1} µs ({:.0} GB/s of cache read once)",
                gb / (us / 1e6)
            );
        }
    }
}
