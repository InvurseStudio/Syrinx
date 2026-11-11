//! The CUDA kernels against the CPU reference, on real model weights. Skipped (with a
//! note) on a computer without an NVIDIA GPU or without the test model.

use std::path::PathBuf;
use std::time::Instant;

use vireo_engine::cpu::{self, Matrix, Pool};
use vireo_engine::cuda::Gpu;
use vireo_engine::models::llama_cuda::LlamaCuda;
use vireo_engine::{GgufFile, Tokenizer};

fn open_gpu() -> Option<Gpu> {
    match Gpu::open() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("skipped: {e}");
            None
        }
    }
}

fn model(name: &str) -> Option<GgufFile> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models")
        .join(name);
    if !path.exists() {
        eprintln!("skipped: {name} isn't in VireoPrototype/models");
        return None;
    }
    Some(GgufFile::open(path).unwrap())
}

fn matrix<'a>(file: &'a GgufFile, name: &str) -> Matrix<'a> {
    let t = file.tensor(name).unwrap();
    Matrix {
        kind: t.kind.unwrap(),
        cols: t.dims[0] as usize,
        rows: t.dims[1] as usize,
        data: file.tensor_data(t).unwrap(),
    }
}

#[test]
fn matvec_kernels_match_the_cpu_on_llama_weights() {
    let Some(gpu) = open_gpu() else { return };
    let Some(file) = model("Llama-3.2-1B-Instruct-Q4_K_M.gguf") else {
        return;
    };
    let i = &gpu.info;
    eprintln!(
        "GPU: {}, compute {}.{}, {} SMs, {:.1} GB, driver CUDA {}.{}",
        i.name,
        i.compute_capability.0,
        i.compute_capability.1,
        i.multiprocessors,
        i.memory_bytes as f64 / 1e9,
        i.driver_version / 1000,
        i.driver_version % 1000 / 10
    );
    let pool = Pool::with_all_cores();

    for name in [
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_down.weight",
        "blk.0.attn_k.weight",
        "token_embd.weight",
    ] {
        let w = matrix(&file, name);
        // A vector with both signs and a spread of sizes, like real activations.
        let x: Vec<f32> = (0..w.cols)
            .map(|c| ((c * 7919) % 1000) as f32 / 250.0 - 2.0)
            .collect();
        let mut expected = vec![0.0; w.rows];
        cpu::matvec(&pool, &w, &x, &mut expected);

        let weights = gpu.upload(w.data).unwrap();
        let x_gpu = gpu.upload_f32(&x).unwrap();
        let y_gpu = gpu.alloc(w.rows * 4).unwrap();
        gpu.matvec(
            w.kind,
            weights.ptr(),
            x_gpu.ptr(),
            y_gpu.ptr(),
            w.rows,
            w.cols,
        )
        .unwrap();
        let mut got = vec![0.0; w.rows];
        y_gpu.read_f32(&mut got).unwrap();

        // The same arithmetic, summed in a different order: equal to f32 rounding.
        let scale = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = expected
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            worst <= scale * 1e-5,
            "{name}: worst difference {worst}, largest value {scale}"
        );

        // Speed: generation reads every weight once per token, so bytes per second is
        // what matters.
        let runs = 200;
        gpu.synchronize().unwrap();
        let started = Instant::now();
        for _ in 0..runs {
            gpu.matvec(
                w.kind,
                weights.ptr(),
                x_gpu.ptr(),
                y_gpu.ptr(),
                w.rows,
                w.cols,
            )
            .unwrap();
        }
        gpu.synchronize().unwrap();
        let gpu_time = started.elapsed() / runs;
        let started = Instant::now();
        for _ in 0..10 {
            cpu::matvec(&pool, &w, &x, &mut expected);
        }
        let cpu_time = started.elapsed() / 10;
        eprintln!(
            "{name:24} {:>5} {:>6}×{:<5} worst diff {:.1e} of {:.1}  GPU {:>7.1?} ({:>5.0} GB/s)  CPU {:>7.1?}",
            w.kind.name(),
            w.rows,
            w.cols,
            worst,
            scale,
            gpu_time,
            w.data.len() as f64 / gpu_time.as_secs_f64() / 1e9,
            cpu_time
        );
    }
}

/// The whole forward pass on the GPU gives llama.cpp's exact numbers (see
/// `llama_3_2_1b_instruct_forward_pass_matches_llama_cpp` in real_models.rs for where they
/// come from), and the same answer.
#[test]
fn llama_forward_pass_on_the_gpu_matches_llama_cpp() {
    let Some(gpu) = open_gpu() else { return };
    let Some(file) = model("Llama-3.2-1B-Instruct-Q4_K_M.gguf") else {
        return;
    };
    let tokenizer = Tokenizer::from_gguf(&file).unwrap();
    let model = LlamaCuda::load(&file, gpu).unwrap();
    let prompt = "<|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of France? \
                  Answer in one sentence.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    let ids = tokenizer.encode(prompt, true, true);
    let mut session = model.session(64).unwrap();
    let mut logits = Vec::new();
    for (i, &id) in ids.iter().enumerate() {
        if let Some(l) = model.forward(&mut session, id, i + 1 == ids.len()).unwrap() {
            logits = l.to_vec();
        }
    }
    cpu::softmax(&mut logits);
    for (id, percent) in [
        (791, 95.7887),
        (60704, 3.6702),
        (49, 0.2236),
        (50100, 0.0463),
    ] {
        let ours = logits[id] as f64 * 100.0;
        assert!(
            (ours - percent).abs() < 0.001,
            "token {id}: {ours:.4}% vs llama.cpp's {percent}%"
        );
    }

    let mut next = 791u32;
    let mut answer = vec![next];
    while !tokenizer.is_end_of_generation(next) && answer.len() < 20 {
        let logits = model.forward(&mut session, next, true).unwrap().unwrap();
        next = (0..logits.len())
            .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
            .unwrap() as u32;
        answer.push(next);
    }
    assert_eq!(
        tokenizer.decode(&answer, false),
        "The capital of France is Paris."
    );
}
