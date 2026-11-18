//! A large model's matrix products on its real weights, for tuning the Mac's kernels:
//! prompt batches on the neural accelerators (`matmul_8bit`, against an F16 matrix of the
//! same shape, which needs no decoding) and a few tokens' products (`matvec_multi`).
//! Ignored by default (the model isn't in the repo); run with
//! `VIREO_BIG_MODEL=<model.gguf> cargo test --release -p syrinx-engine --test big_matmul -- --ignored --nocapture`,
//! and `VIREO_BIG_TOKENS` (default 1024) and `VIREO_BIG_NAMES` (weights, comma-separated)
//! to choose. Each dispatch's own time is better read with VIREO_METAL_PROFILE.

// The Metal backend's own kernels and drafting (`matvec_multi`, `use_mtp`): Mac only.
#![cfg(target_os = "macos")]

use std::time::Instant;

use syrinx_engine::GgufFile;
use syrinx_engine::cuda::Gpu;
use syrinx_engine::half::f32_to_f16;
use syrinx_engine::tensor_type::TensorType;

const NAMES: &str =
    "blk.0.ffn_up.weight,blk.0.ffn_down.weight,blk.0.attn_q.weight,blk.0.attn_k.weight";

fn setup() -> Option<(GgufFile, Gpu, String)> {
    let Ok(path) = std::env::var("VIREO_BIG_MODEL") else {
        eprintln!("skipped: VIREO_BIG_MODEL isn't set");
        return None;
    };
    let names = std::env::var("VIREO_BIG_NAMES").unwrap_or_else(|_| NAMES.into());
    Some((GgufFile::open(path).unwrap(), Gpu::open().unwrap(), names))
}

/// Seconds a run, once the GPU is at full clocks (it idles at a fraction of them).
fn time(gpu: &Gpu, mut run: impl FnMut()) -> f64 {
    let warm = Instant::now();
    while warm.elapsed().as_millis() < 400 {
        run();
        gpu.synchronize().unwrap();
    }
    let runs = 10;
    let started = Instant::now();
    for _ in 0..runs {
        run();
    }
    gpu.synchronize().unwrap();
    started.elapsed().as_secs_f64() / runs as f64
}

fn activations(n: usize) -> Vec<f32> {
    (0..n)
        .map(|c| ((c * 7919) % 1000) as f32 / 250.0 - 2.0)
        .collect()
}

#[test]
#[ignore]
fn big_matmul_speed() {
    let Some((file, gpu, names)) = setup() else {
        return;
    };
    let tokens: usize = std::env::var("VIREO_BIG_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024);
    // Read against 100 tokens too (the 128-token tile): a short batch's tile must give
    // its rows the same sums, to f32 rounding.
    let most = tokens.max(100);
    for name in names.split(',') {
        let t = file.tensor(name).unwrap();
        let (kind, cols, rows) = (t.kind.unwrap(), t.dims[0] as usize, t.dims[1] as usize);
        let w = gpu
            .upload_weights(kind, file.tensor_data(t).unwrap(), rows, cols)
            .unwrap();
        let x_gpu = gpu.upload_f32(&activations(cols * most)).unwrap();
        let q8 = gpu.alloc_q8(most, cols).unwrap();
        gpu.quantize(x_gpu.ptr(), &q8, most, cols).unwrap();
        let y_gpu = gpu.alloc(rows * most * 4).unwrap();
        let product = |n: usize| {
            // The rounded copy is laid out for its batch's tokens.
            gpu.quantize(x_gpu.ptr(), &q8, n, cols).unwrap();
            gpu.matmul_8bit(kind, w.ptr(), &q8, y_gpu.ptr(), rows, cols, n)
                .unwrap();
            let mut out = vec![0.0f32; rows * n];
            y_gpu.read_f32(&mut out).unwrap();
            out
        };
        let (mine, wide) = (product(tokens), product(100));
        let n = rows * tokens.min(100);
        let worst = mine[..n]
            .iter()
            .zip(&wide[..n])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let scale = wide[..n].iter().fold(0f32, |m, v| m.max(v.abs()));
        // The same sums, but for f32 rounding where one of the two is split along the
        // columns (batches of up to 128 tokens) and the other isn't.
        // The 8-bit kernel (long batches of Q6_K and Q8_0) rounds both sides: its sums
        // are near the f16 kernel's, and how near is printed.
        let whole = Gpu::whole_number_products(kind, cols, tokens);
        assert!(
            worst <= scale * if whole { 3e-2 } else { 1e-5 },
            "{name}: {tokens} tokens' tile differs by {worst} of {scale}"
        );
        if whole {
            let (mut err, mut all) = (0f64, 0f64);
            for (a, b) in mine[..n].iter().zip(&wide[..n]) {
                err += ((a - b) as f64).powi(2);
                all += (*b as f64).powi(2);
            }
            eprintln!(
                "{name:24} 8-bit against f16: {:.3}% of the sums' size (rms), worst {:.3}%",
                (err / all).sqrt() * 100.0,
                worst / scale * 100.0
            );
        }

        gpu.quantize(x_gpu.ptr(), &q8, tokens, cols).unwrap();
        let s = time(&gpu, || {
            gpu.matmul_8bit(kind, w.ptr(), &q8, y_gpu.ptr(), rows, cols, tokens)
                .unwrap()
        });
        let f16: Vec<u8> = (0..rows * cols)
            .flat_map(|i| f32_to_f16(((i * 31) % 97) as f32 / 970.0 - 0.05).to_le_bytes())
            .collect();
        let wf = gpu
            .upload_weights(TensorType::F16, &f16, rows, cols)
            .unwrap();
        let sf = time(&gpu, || {
            gpu.matmul_8bit(
                TensorType::F16,
                wf.ptr(),
                &q8,
                y_gpu.ptr(),
                rows,
                cols,
                tokens,
            )
            .unwrap()
        });
        let flops = 2.0 * (rows * cols * tokens) as f64;
        eprintln!(
            "{name:24} {:>5} {rows:>6}x{cols:<6} x{tokens}: {:>7.2} ms ({:>5.1} TFLOPS); F16 {:>7.2} ms ({:>5.1} TFLOPS)",
            kind.name(),
            s * 1e3,
            flops / s / 1e12,
            sf * 1e3,
            flops / sf / 1e12
        );
    }
}

#[test]
#[ignore]
fn big_few_speed() {
    let Some((file, gpu, names)) = setup() else {
        return;
    };
    for name in names.split(',') {
        let t = file.tensor(name).unwrap();
        let (kind, cols, rows) = (t.kind.unwrap(), t.dims[0] as usize, t.dims[1] as usize);
        let w = gpu
            .upload_weights(kind, file.tensor_data(t).unwrap(), rows, cols)
            .unwrap();
        let x_gpu = gpu.upload_f32(&activations(cols * 8)).unwrap();
        let y_gpu = gpu.alloc(rows * 8 * 4).unwrap();
        // One token's result, which every count's first row must match.
        gpu.matvec(kind, w.ptr(), x_gpu.ptr(), y_gpu.ptr(), rows, cols)
            .unwrap();
        let mut one = vec![0.0f32; rows];
        y_gpu.read_f32(&mut one).unwrap();
        let mut line = format!("{name:24} {:>5} {rows:>6}x{cols:<6}", kind.name());
        for tokens in 1..=8 {
            let s = time(&gpu, || {
                if tokens == 1 {
                    gpu.matvec(kind, w.ptr(), x_gpu.ptr(), y_gpu.ptr(), rows, cols)
                        .unwrap();
                } else {
                    assert!(
                        gpu.matvec_multi(
                            kind,
                            w.ptr(),
                            x_gpu.ptr(),
                            y_gpu.ptr(),
                            rows,
                            cols,
                            tokens
                        )
                        .unwrap()
                    );
                }
            });
            let mut got = vec![0.0f32; rows];
            y_gpu.read_f32(&mut got).unwrap();
            let worst = one
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                worst < 1e-3,
                "{name} at {tokens} tokens: worst difference {worst}"
            );
            line += &format!("  {tokens}: {:>6.1} µs", s * 1e6);
        }
        eprintln!("{line}");
    }
}

/// A few tokens through the few-token kernels (`matvec_multi`) and through the short
/// batches' tiles (`matmul_8bit`), µs: where `Gpu::tile_beats_few` should switch.
#[test]
#[ignore]
fn big_few_or_tile() {
    let Some((file, gpu, names)) = setup() else {
        return;
    };
    for name in names.split(',') {
        let t = file.tensor(name).unwrap();
        let (kind, cols, rows) = (t.kind.unwrap(), t.dims[0] as usize, t.dims[1] as usize);
        let w = gpu
            .upload_weights(kind, file.tensor_data(t).unwrap(), rows, cols)
            .unwrap();
        let x_gpu = gpu.upload_f32(&activations(cols * 32)).unwrap();
        let q8 = gpu.alloc_q8(32, cols).unwrap();
        gpu.quantize(x_gpu.ptr(), &q8, 32, cols).unwrap();
        let y_gpu = gpu.alloc(rows * 32 * 4).unwrap();
        let mut line = format!("{name:24} {:>5} {rows:>6}x{cols:<6}", kind.name());
        for tokens in [2usize, 3, 4, 5, 6, 8, 12, 16, 24, 32] {
            let few = (tokens <= 8).then(|| {
                time(&gpu, || {
                    gpu.matvec_multi(kind, w.ptr(), x_gpu.ptr(), y_gpu.ptr(), rows, cols, tokens)
                        .unwrap();
                })
            });
            let mut want = vec![0.0f32; rows * tokens];
            y_gpu.read_f32(&mut want).unwrap();
            let tile = time(&gpu, || {
                gpu.matmul_8bit(kind, w.ptr(), &q8, y_gpu.ptr(), rows, cols, tokens)
                    .unwrap();
            });
            if few.is_some() {
                let mut got = vec![0.0f32; rows * tokens];
                y_gpu.read_f32(&mut got).unwrap();
                let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
                let worst = want
                    .iter()
                    .zip(&got)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    worst <= scale * 0.02,
                    "{name} at {tokens}: {worst} of {scale}"
                );
            }
            line += &match few {
                Some(f) => format!("  {tokens}: {:.0}/{:.0}", f * 1e6, tile * 1e6),
                None => format!("  {tokens}: {:.0}", tile * 1e6),
            };
        }
        eprintln!("{line}");
    }
}

/// A large model's weights through every GPU path a batch size takes (one token, a few,
/// a short and a long tile), each token's row against the CPU's exact product: the
/// default mode's f16 activations keep it within ~1% of the row's largest value.
#[test]
#[ignore]
fn big_matches_the_cpu() {
    let Some((file, gpu, names)) = setup() else {
        return;
    };
    let pool = syrinx_engine::cpu::Pool::with_all_cores();
    for name in names.split(',') {
        let t = file.tensor(name).unwrap();
        let (kind, cols, rows) = (t.kind.unwrap(), t.dims[0] as usize, t.dims[1] as usize);
        let data = file.tensor_data(t).unwrap();
        let m = syrinx_engine::cpu::Matrix {
            kind,
            rows,
            cols,
            data,
        };
        let w = gpu.upload_weights(kind, data, rows, cols).unwrap();
        let most = 300;
        let x = activations(cols * most);
        let x_gpu = gpu.upload_f32(&x).unwrap();
        let q8 = gpu.alloc_q8(most, cols).unwrap();
        let y_gpu = gpu.alloc(rows * most * 4).unwrap();
        // The CPU's rows for the first two tokens (each count's are checked against them).
        let expected: Vec<Vec<f32>> = (0..2)
            .map(|t| {
                let mut y = vec![0.0f32; rows];
                syrinx_engine::cpu::matvec(&pool, &m, &x[t * cols..(t + 1) * cols], &mut y);
                y
            })
            .collect();
        let mut line = format!("{name:28} {:>5} {rows:>6}x{cols:<6}", kind.name());
        for tokens in [1usize, 3, 20, 100, 300] {
            if tokens == 1 {
                gpu.matvec(kind, w.ptr(), x_gpu.ptr(), y_gpu.ptr(), rows, cols)
                    .unwrap();
            } else if tokens <= 8 {
                assert!(
                    gpu.matvec_multi(kind, w.ptr(), x_gpu.ptr(), y_gpu.ptr(), rows, cols, tokens)
                        .unwrap()
                );
            } else {
                gpu.quantize(x_gpu.ptr(), &q8, tokens, cols).unwrap();
                gpu.matmul_8bit(kind, w.ptr(), &q8, y_gpu.ptr(), rows, cols, tokens)
                    .unwrap();
            }
            let mut got = vec![0.0f32; rows * tokens];
            y_gpu.read_f32(&mut got).unwrap();
            let mut worst = 0f32;
            for (t, want) in expected.iter().enumerate().take(tokens) {
                let scale = want.iter().fold(0f32, |a, v| a.max(v.abs()));
                let diff = want
                    .iter()
                    .zip(&got[t * rows..(t + 1) * rows])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                worst = worst.max(diff / scale);
            }
            line += &format!("  {tokens}: {worst:.1e}");
            assert!(
                worst < 0.01,
                "{name} at {tokens} tokens: {worst} of the largest value"
            );
        }
        eprintln!("{line}");
    }
}
