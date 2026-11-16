//! How a prompt batch's 8-bit product scales with its rows on CUDA, for tuning: a matrix's
//! first `rows` rows multiplied by a batch, for each of a list of row counts. A block of
//! the kernel takes 128 rows and one tile of tokens, one block an SM, so the time steps up
//! where the blocks stop dividing evenly among the SMs.
//! Ignored by default; run with
//! `VIREO_WAVES_MODEL=<model.gguf> cargo test --release -p vireo-engine --test tile_waves -- --ignored --nocapture`,
//! and `VIREO_WAVES_NAMES` (weights, comma-separated), `VIREO_WAVES_TOKENS` (default
//! 1024) and `VIREO_WAVES_ROWS` (row counts, comma-separated; default the whole matrix).

#![cfg(not(target_os = "macos"))]

use std::time::Instant;

use vireo_engine::GgufFile;
use vireo_engine::cuda::Gpu;

/// Seconds a run, once the GPU is at full clocks (it idles at a fraction of them).
fn time(gpu: &Gpu, mut run: impl FnMut()) -> f64 {
    let warm = Instant::now();
    while warm.elapsed().as_millis() < 400 {
        run();
        gpu.synchronize().unwrap();
    }
    let mut best = f64::MAX;
    for _ in 0..5 {
        let runs = 10;
        let started = Instant::now();
        for _ in 0..runs {
            run();
        }
        gpu.synchronize().unwrap();
        best = best.min(started.elapsed().as_secs_f64() / runs as f64);
    }
    best
}

#[test]
#[ignore]
fn rows_and_waves() {
    let Ok(path) = std::env::var("VIREO_WAVES_MODEL") else {
        eprintln!("skipped: VIREO_WAVES_MODEL isn't set");
        return;
    };
    let names = std::env::var("VIREO_WAVES_NAMES")
        .unwrap_or_else(|_| "blk.0.ffn_up.weight,blk.0.ffn_down.weight".into());
    let tokens: usize = std::env::var("VIREO_WAVES_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024);
    let file = GgufFile::open(path).unwrap();
    let gpu = Gpu::open().unwrap();
    for name in names.split(',') {
        let t = file.tensor(name).unwrap();
        let (kind, cols, all) = (t.kind.unwrap(), t.dims[0] as usize, t.dims[1] as usize);
        let data = file.tensor_data(t).unwrap();
        let row_bytes = data.len() / all;
        let x: Vec<f32> = (0..cols * tokens)
            .map(|c| ((c * 7919) % 1000) as f32 / 250.0 - 2.0)
            .collect();
        let x_gpu = gpu.upload_f32(&x).unwrap();
        let q8 = gpu.alloc_q8(tokens, cols).unwrap();
        gpu.quantize(x_gpu.ptr(), &q8, tokens, cols).unwrap();
        let y_gpu = gpu.alloc(all * tokens * 4).unwrap();
        let counts: Vec<usize> = match std::env::var("VIREO_WAVES_ROWS") {
            Ok(list) => list.split(',').filter_map(|v| v.parse().ok()).collect(),
            Err(_) => vec![all],
        };
        println!("{name}: {kind:?} {all} x {cols}, {tokens} tokens");
        for rows in counts.into_iter().filter(|&r| r <= all) {
            let w = gpu
                .upload_weights(kind, &data[..rows * row_bytes], rows, cols)
                .unwrap();
            let seconds = time(&gpu, || {
                gpu.matmul_8bit(kind, w.ptr(), &q8, y_gpu.ptr(), rows, cols, tokens)
                    .unwrap()
            });
            let tops = 2.0 * (rows * cols * tokens) as f64 / seconds / 1e12;
            println!(
                "  {rows:>6} rows ({:>3} tiles of 128): {:>8.3} ms, {:>6.3} ns a row and token, {tops:.1} TOPS",
                rows.div_ceil(128),
                seconds * 1e3,
                seconds * 1e9 / (rows * tokens) as f64,
            );
        }
    }
}
