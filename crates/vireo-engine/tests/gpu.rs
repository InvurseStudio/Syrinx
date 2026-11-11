//! The CUDA kernels against the CPU reference, on real model weights. Skipped (with a
//! note) on a computer without an NVIDIA GPU or without the test model.

use std::path::PathBuf;
use std::time::Instant;

use vireo_engine::cpu::{self, Matrix, Pool};
use vireo_engine::cuda::Gpu;
use vireo_engine::models::gemma4_cuda::Gemma4Cuda;
use vireo_engine::models::llama_cuda::LlamaCuda;
use vireo_engine::models::qwen35_cuda::Qwen35Cuda;
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

/// Each named weight of `file`, multiplied on the GPU and on the CPU: they must agree to
/// f32 rounding. Prints each one's speed.
fn check_matvec(gpu: &Gpu, file: &GgufFile, names: &[&str]) {
    let pool = Pool::with_all_cores();
    for &name in names {
        let w = matrix(file, name);
        // A vector with both signs and a spread of sizes, like real activations.
        let x: Vec<f32> = (0..w.cols)
            .map(|c| ((c * 7919) % 1000) as f32 / 250.0 - 2.0)
            .collect();
        let mut expected = vec![0.0; w.rows];
        cpu::matvec(&pool, &w, &x, &mut expected);

        let weights = gpu.upload(w.data).unwrap();
        let x_gpu = gpu.upload_f32(&x).unwrap();
        let y_gpu = gpu.alloc(w.rows * 4).unwrap();
        let run = || {
            gpu.matvec(
                w.kind,
                weights.ptr(),
                x_gpu.ptr(),
                y_gpu.ptr(),
                w.rows,
                w.cols,
            )
            .unwrap()
        };
        run();
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
            run();
        }
        gpu.synchronize().unwrap();
        let gpu_time = started.elapsed() / runs;
        eprintln!(
            "{name:28} {:>5} {:>6}x{:<5} worst diff {:.1e} of {:>5.1}  GPU {:>7.1?} ({:>4.0} GB/s)",
            w.kind.name(),
            w.rows,
            w.cols,
            worst,
            scale,
            gpu_time,
            w.data.len() as f64 / gpu_time.as_secs_f64() / 1e9,
        );
    }
}

#[test]
fn matvec_kernels_match_the_cpu_on_real_weights() {
    let Some(gpu) = open_gpu() else { return };
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
    // Q4_K and Q6_K.
    if let Some(file) = model("Llama-3.2-1B-Instruct-Q4_K_M.gguf") {
        check_matvec(
            &gpu,
            &file,
            &[
                "blk.0.ffn_gate.weight",
                "blk.0.ffn_down.weight",
                "blk.0.attn_k.weight",
                "token_embd.weight",
            ],
        );
    }
    // Q5_K and Q8_0.
    if let Some(file) = model("Qwen3.5-0.8B-Q4_K_M.gguf") {
        check_matvec(
            &gpu,
            &file,
            &[
                "blk.0.attn_qkv.weight",
                "blk.0.ssm_out.weight",
                "blk.0.ssm_alpha.weight",
                "blk.3.attn_q.weight",
            ],
        );
    }
    // Q4_0 and F16.
    if let Some(file) = model("gemma-4-E2B_q4_0-it.gguf") {
        check_matvec(
            &gpu,
            &file,
            &[
                "blk.0.ffn_gate.weight",
                "blk.20.ffn_down.weight",
                "blk.4.attn_q.weight",
                "per_layer_model_proj.weight",
            ],
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

/// Runs `prompt` through a GPU model and greedy-generates up to 20 tokens after `first`:
/// returns the prompt's next-token probabilities and the answer.
fn probabilities_and_answer(
    forward: &mut dyn FnMut(u32, bool) -> Option<Vec<f32>>,
    ids: &[u32],
    first: u32,
    tokenizer: &Tokenizer,
) -> (Vec<f32>, String) {
    let mut logits = Vec::new();
    for (i, &id) in ids.iter().enumerate() {
        if let Some(l) = forward(id, i + 1 == ids.len()) {
            logits = l;
        }
    }
    cpu::softmax(&mut logits);
    let mut next = first;
    let mut answer = vec![next];
    while !tokenizer.is_end_of_generation(next) && answer.len() < 20 {
        let l = forward(next, true).unwrap();
        next = (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap() as u32;
        answer.push(next);
    }
    (logits, tokenizer.decode(&answer, false))
}

/// Qwen 3.5 on the GPU against llama.cpp's numbers (see
/// `qwen35_0_8b_forward_pass_matches_llama_cpp` in real_models.rs).
#[test]
fn qwen35_forward_pass_on_the_gpu_matches_llama_cpp() {
    let Some(gpu) = open_gpu() else { return };
    let Some(file) = model("Qwen3.5-0.8B-Q4_K_M.gguf") else {
        return;
    };
    let tokenizer = Tokenizer::from_gguf(&file).unwrap();
    let model = Qwen35Cuda::load(&file, gpu).unwrap();
    let prompt = "<|im_start|>user\nWhat is the capital of France? Answer in one sentence.<|im_end|>\n\
                  <|im_start|>assistant\n<think>\n\n</think>\n\n";
    let ids = tokenizer.encode(prompt, true, true);
    let mut session = model.session(64).unwrap();
    let mut forward = |id, want| {
        model
            .forward(&mut session, id, want)
            .unwrap()
            .map(<[f32]>::to_vec)
    };
    let (probs, answer) = probabilities_and_answer(&mut forward, &ids, 760, &tokenizer);
    for (id, percent) in [
        (760, 56.6133),
        (57590, 35.0073),
        (47358, 0.8957),
        (38261, 0.6041),
    ] {
        let ours = probs[id] as f64 * 100.0;
        assert!(
            (ours - percent).abs() < 0.002,
            "token {id}: {ours:.4}% vs llama.cpp's {percent}%"
        );
    }
    assert_eq!(answer, "The capital of France is Paris.");
}

/// Gemma 4 on the GPU against llama.cpp's numbers (see
/// `gemma4_e2b_forward_pass_matches_llama_cpp` in real_models.rs).
#[test]
fn gemma4_forward_pass_on_the_gpu_matches_llama_cpp() {
    let Some(gpu) = open_gpu() else { return };
    let Some(file) = model("gemma-4-E2B_q4_0-it.gguf") else {
        return;
    };
    let tokenizer = Tokenizer::from_gguf(&file).unwrap();
    let model = Gemma4Cuda::load(&file, gpu).unwrap();
    let prompt = "<|turn>user\nWhat is the capital of France? Answer in one sentence.<turn|>\n<|turn>model\n";
    let ids = tokenizer.encode(prompt, true, true);
    let mut session = model.session(64).unwrap();
    let mut forward = |id, want| {
        model
            .forward(&mut session, id, want)
            .unwrap()
            .map(<[f32]>::to_vec)
    };
    let (probs, answer) = probabilities_and_answer(&mut forward, &ids, 818, &tokenizer);
    for (id, percent) in [
        (818, 99.7068),
        (50429, 0.2201),
        (1018, 0.0387),
        (31756, 0.0049),
    ] {
        let ours = probs[id] as f64 * 100.0;
        assert!(
            (ours - percent).abs() < 0.001,
            "token {id}: {ours:.4}% vs llama.cpp's {percent}%"
        );
    }
    assert_eq!(answer, "The capital of France is Paris.");
}
