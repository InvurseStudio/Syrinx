//! The CUDA kernels against the CPU reference, on real model weights. Skipped (with a
//! note) on a computer without an NVIDIA GPU or without the test model.

use std::path::PathBuf;
use std::time::Instant;

use vireo_engine::cpu::{self, Matrix, Pool};
use vireo_engine::cuda::Gpu;
use vireo_engine::models::gemma4_cuda::Gemma4Cuda;
use vireo_engine::models::llama_cuda::LlamaCuda;
use vireo_engine::models::qwen35_cuda::Qwen35Cuda;
use vireo_engine::models::{Logits, Precision};
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

        let weights = gpu.upload_weights(w.kind, w.data, w.rows, w.cols).unwrap();
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
        let warm = Instant::now();
        while warm.elapsed().as_millis() < 200 {
            run();
            gpu.synchronize().unwrap();
        }
        let runs = 200;
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

/// Each named weight times a batch of 100 vectors (not a whole number of the kernels'
/// tiles) on the GPU, against the CPU one vector at a time. Prints the speed.
fn check_matmul(gpu: &Gpu, file: &GgufFile, names: &[&str]) {
    let pool = Pool::with_all_cores();
    let tokens = 100;
    for &name in names {
        let w = matrix(file, name);
        let x: Vec<f32> = (0..w.cols * tokens)
            .map(|c| ((c * 7919) % 1000) as f32 / 250.0 - 2.0)
            .collect();
        let mut expected = vec![0.0; w.rows * tokens];
        for (xt, yt) in x.chunks(w.cols).zip(expected.chunks_mut(w.rows)) {
            cpu::matvec(&pool, &w, xt, yt);
        }
        let weights = gpu.upload_weights(w.kind, w.data, w.rows, w.cols).unwrap();
        let x_gpu = gpu.upload_f32(&x).unwrap();
        let y_gpu = gpu.alloc(w.rows * tokens * 4).unwrap();
        let run = || {
            gpu.matmul(
                w.kind,
                weights.ptr(),
                x_gpu.ptr(),
                y_gpu.ptr(),
                w.rows,
                w.cols,
                tokens,
            )
            .unwrap()
        };
        run();
        let mut got = vec![0.0; w.rows * tokens];
        y_gpu.read_f32(&mut got).unwrap();
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
        // Bring the GPU up to full clocks first: it idles at a fraction of them.
        let warm = Instant::now();
        while warm.elapsed().as_millis() < 300 {
            run();
            gpu.synchronize().unwrap();
        }
        let runs = 20;
        let started = Instant::now();
        for _ in 0..runs {
            run();
        }
        gpu.synchronize().unwrap();
        let time = started.elapsed() / runs;
        let flops = 2.0 * (w.rows * w.cols * tokens) as f64;
        eprintln!(
            "{name:28} {:>5} {:>6}x{:<5} x{tokens} worst diff {:.1e} of {:>5.1}  GPU {:>7.1?} ({:>5.2} TFLOPS)",
            w.kind.name(),
            w.rows,
            w.cols,
            worst,
            scale,
            time,
            flops / time.as_secs_f64() / 1e12,
        );
    }
}

#[test]
fn matmul_kernels_match_the_cpu_on_real_weights() {
    let Some(gpu) = open_gpu() else { return };
    if let Some(file) = model("Llama-3.2-1B-Instruct-Q4_K_M.gguf") {
        check_matmul(
            &gpu,
            &file,
            &[
                "blk.0.ffn_gate.weight",
                "blk.0.ffn_down.weight",
                "blk.0.attn_k.weight",
            ],
        );
    }
    if let Some(file) = model("Qwen3.5-0.8B-Q4_K_M.gguf") {
        check_matmul(
            &gpu,
            &file,
            &["blk.0.attn_qkv.weight", "blk.0.ssm_alpha.weight"],
        );
    }
    if let Some(file) = model("gemma-4-E2B_q4_0-it.gguf") {
        check_matmul(
            &gpu,
            &file,
            &["blk.0.ffn_gate.weight", "per_layer_model_proj.weight"],
        );
    }
    // The same Llama weights decoded to f32 first: the arithmetic alone, without decoding.
    if let Some(file) = model("Llama-3.2-1B-Instruct-from-Q4_K_M-F32.gguf") {
        check_matmul(&gpu, &file, &["blk.0.ffn_gate.weight"]);
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
                // Q6_K in short rows: the output projection, the largest matrix a token reads.
                "token_embd.weight",
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
    let mut model = LlamaCuda::load(&file, gpu).unwrap();
    model.set_precision(Precision::Exact);
    let prompt = "<|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of France? \
                  Answer in one sentence.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    let ids = tokenizer.encode(prompt, true, true);
    let mut session = model.session(64).unwrap();
    let mut logits = model
        .forward_batch(&mut session, &ids, Logits::Last)
        .unwrap()
        .unwrap()
        .to_vec();
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

/// Reads the prompt `ids` into a GPU model in one batch (`read`) and greedy-generates up
/// to 20 tokens after `first` one at a time (`forward`): returns the prompt's next-token
/// probabilities and the answer.
fn probabilities_and_answer(
    read: &mut dyn FnMut(&[u32]) -> Vec<f32>,
    forward: &mut dyn FnMut(u32, bool) -> Option<Vec<f32>>,
    ids: &[u32],
    first: u32,
    tokenizer: &Tokenizer,
) -> (Vec<f32>, String) {
    let mut logits = read(ids);
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
    let mut model = Qwen35Cuda::load(&file, gpu).unwrap();
    model.set_precision(Precision::Exact);
    let prompt = "<|im_start|>user\nWhat is the capital of France? Answer in one sentence.<|im_end|>\n\
                  <|im_start|>assistant\n<think>\n\n</think>\n\n";
    let ids = tokenizer.encode(prompt, true, true);
    let session = std::cell::RefCell::new(model.session(64).unwrap());
    let mut read = |ids: &[u32]| {
        model
            .forward_batch(&mut session.borrow_mut(), ids, Logits::Last)
            .unwrap()
            .unwrap()
            .to_vec()
    };
    let mut forward = |id, want| {
        model
            .forward(&mut session.borrow_mut(), id, want)
            .unwrap()
            .map(<[f32]>::to_vec)
    };
    let (probs, answer) = probabilities_and_answer(&mut read, &mut forward, &ids, 760, &tokenizer);
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
    let mut model = Gemma4Cuda::load(&file, gpu).unwrap();
    model.set_precision(Precision::Exact);
    let prompt = "<|turn>user\nWhat is the capital of France? Answer in one sentence.<turn|>\n<|turn>model\n";
    let ids = tokenizer.encode(prompt, true, true);
    let session = std::cell::RefCell::new(model.session(64).unwrap());
    let mut read = |ids: &[u32]| {
        model
            .forward_batch(&mut session.borrow_mut(), ids, Logits::Last)
            .unwrap()
            .unwrap()
            .to_vec()
    };
    let mut forward = |id, want| {
        model
            .forward(&mut session.borrow_mut(), id, want)
            .unwrap()
            .map(<[f32]>::to_vec)
    };
    let (probs, answer) = probabilities_and_answer(&mut read, &mut forward, &ids, 818, &tokenizer);
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

/// A text long enough to cross several of the attention kernels' chunks (over 128
/// positions), and Gemma's sliding window when `long`.
fn long_text(tokenizer: &Tokenizer, long: bool) -> Vec<u32> {
    let paragraph = "The engine reads a prompt in batches: every weight is decoded once for many \
                     tokens, and attention looks at a block of keys for a block of queries. When \
                     it writes a reply, it replays one recorded graph per token. ";
    let text = paragraph.repeat(if long { 12 } else { 6 });
    tokenizer.encode(&text, true, false)
}

/// Every token's logits from reading `ids` in batches (of 7, so batches end mid-tile) and
/// from running them one at a time nearly always pick the same next token, and differ by
/// at most `worst` and on average `mean` of their range. Exact, they differ only by f32
/// rounding (sums in another order). In 8-bit mode the two take different paths on
/// purpose (a batch's attention runs on the tensor cores, rounding queries and softmax
/// weights to f16; one token's doesn't), and small differences tip activations across
/// 8-bit steps, so single logits drift much further: on Gemma, over 35 layers and a logit
/// softcap, up to a quarter of the range on one token in 554 (0.6% on average), with
/// perplexity within 0.1% of exact. That's the two paths' own roundings, not one path
/// going wrong: measured against exact on the same Gemma text, batches were off by 0.14
/// on average and token-by-token by 0.16, each picking the exact next token 552 and 551
/// times in 554. So 8-bit mode is held to its average and its picks; how close it stays
/// to exact is checked by `eight_bit_activations_stay_close_to_exact` and perplexity.
fn check_batches_match_tokens(
    name: &str,
    worst_tolerance: f32,
    mean_tolerance: f32,
    ids: &[u32],
    batched: &mut dyn FnMut(&[u32]) -> Vec<f32>,
    one_by_one: &mut dyn FnMut(u32) -> Vec<f32>,
) {
    let all = batched(ids);
    let vocab = all.len() / ids.len();
    let (mut worst, mut largest, mut same_pick, mut total) = (0f32, 0f32, 0, 0f64);
    for (i, &id) in ids.iter().enumerate() {
        let single = one_by_one(id);
        let row = &all[i * vocab..(i + 1) * vocab];
        for (a, b) in row.iter().zip(&single) {
            worst = worst.max((a - b).abs());
            total += f64::from((a - b).abs());
            largest = largest.max(b.abs());
        }
        let pick = |l: &[f32]| (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap();
        same_pick += usize::from(pick(row) == pick(&single));
    }
    let mean = (total / all.len() as f64) as f32;
    eprintln!(
        "{name}: {} tokens, logit difference worst {worst:.2e}, mean {mean:.1e} of {largest:.1}, same next token {same_pick}/{}",
        ids.len(),
        ids.len()
    );
    assert!(
        worst <= largest * worst_tolerance,
        "{name}: worst logit difference {worst} of {largest}"
    );
    assert!(
        mean <= largest * mean_tolerance,
        "{name}: mean logit difference {mean} of {largest}"
    );
    assert!(same_pick * 100 >= ids.len() * 99, "{name}: picks differ");
}

#[test]
fn reading_in_batches_matches_one_token_at_a_time() {
    for (precision, worst, mean) in [(Precision::Exact, 1e-3, 1e-4), (Precision::Int8, 1.0, 0.01)] {
        let label = |model: &str| format!("{model}, {precision:?}");
        if let Some(file) = model("Llama-3.2-1B-Instruct-Q4_K_M.gguf") {
            let Some(gpu) = open_gpu() else { return };
            let tokenizer = Tokenizer::from_gguf(&file).unwrap();
            let ids = long_text(&tokenizer, false);
            let mut m = LlamaCuda::load(&file, gpu).unwrap();
            m.set_precision(precision);
            let mut a = m.session_with_batch(ids.len(), 7).unwrap();
            let mut b = m.session(ids.len()).unwrap();
            check_batches_match_tokens(
                &label("Llama"),
                worst,
                mean,
                &ids,
                &mut |ids| {
                    m.forward_batch(&mut a, ids, Logits::All)
                        .unwrap()
                        .unwrap()
                        .to_vec()
                },
                &mut |id| m.forward(&mut b, id, true).unwrap().unwrap().to_vec(),
            );
        }
        if let Some(file) = model("Qwen3.5-0.8B-Q4_K_M.gguf") {
            let Some(gpu) = open_gpu() else { return };
            let tokenizer = Tokenizer::from_gguf(&file).unwrap();
            let ids = long_text(&tokenizer, false);
            let mut m = Qwen35Cuda::load(&file, gpu).unwrap();
            m.set_precision(precision);
            // Batches of 7 stay inside one of the DeltaNet's 64-token chunks; batches of
            // 200 cross three and end in a part of one (then a shorter batch follows).
            assert!(ids.len() > 200, "{} tokens", ids.len());
            for batch in [7, 200] {
                let mut a = m.session_with_batch(ids.len(), batch).unwrap();
                let mut b = m.session(ids.len()).unwrap();
                check_batches_match_tokens(
                    &label(&format!("Qwen 3.5, batches of {batch}")),
                    worst,
                    mean,
                    &ids,
                    &mut |ids| {
                        m.forward_batch(&mut a, ids, Logits::All)
                            .unwrap()
                            .unwrap()
                            .to_vec()
                    },
                    &mut |id| m.forward(&mut b, id, true).unwrap().unwrap().to_vec(),
                );
            }
        }
        if let Some(file) = model("gemma-4-E2B_q4_0-it.gguf") {
            let Some(gpu) = open_gpu() else { return };
            let tokenizer = Tokenizer::from_gguf(&file).unwrap();
            let ids = long_text(&tokenizer, true);
            let mut m = Gemma4Cuda::load(&file, gpu).unwrap();
            m.set_precision(precision);
            assert!(
                ids.len() > m.config.sliding_window,
                "the text should cross the sliding window ({} tokens)",
                ids.len()
            );
            let mut a = m.session_with_batch(ids.len(), 7).unwrap();
            let mut b = m.session(ids.len()).unwrap();
            check_batches_match_tokens(
                &label("Gemma 4"),
                worst,
                mean,
                &ids,
                &mut |ids| {
                    m.forward_batch(&mut a, ids, Logits::All)
                        .unwrap()
                        .unwrap()
                        .to_vec()
                },
                &mut |id| m.forward(&mut b, id, true).unwrap().unwrap().to_vec(),
            );
        }
    }
}

/// The France prompt's next-token probabilities (read in one batch) and greedy answer
/// (generated a token at a time) with the given precision.
fn france(file: &GgufFile, prompt: &str, precision: Precision) -> (Vec<f32>, String) {
    let tokenizer = Tokenizer::from_gguf(file).unwrap();
    let ids = tokenizer.encode(prompt, true, true);
    let gpu = open_gpu().unwrap();
    let pick = |l: &[f32]| (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap() as u32;
    let mut probs: Vec<f32>;
    let mut answer = Vec::new();
    macro_rules! run {
        ($model:expr) => {{
            let mut m = $model;
            m.set_precision(precision);
            let mut s = m.session(64).unwrap();
            probs = m
                .forward_batch(&mut s, &ids, Logits::Last)
                .unwrap()
                .unwrap()
                .to_vec();
            let mut next = pick(&probs);
            answer.push(next);
            while !tokenizer.is_end_of_generation(next) && answer.len() < 20 {
                next = pick(m.forward(&mut s, next, true).unwrap().unwrap());
                answer.push(next);
            }
        }};
    }
    match file.architecture() {
        Some("qwen35") => run!(Qwen35Cuda::load(file, gpu).unwrap()),
        Some("gemma4") => run!(Gemma4Cuda::load(file, gpu).unwrap()),
        _ => run!(LlamaCuda::load(file, gpu).unwrap()),
    }
    cpu::softmax(&mut probs);
    (probs, tokenizer.decode(&answer, false))
}

/// With 8-bit activations (the default), the France prompt still gets its answer and the
/// same likeliest first token as exact arithmetic, with probabilities a few points apart
/// at most, as llama.cpp's own are from exact ones.
#[test]
fn eight_bit_activations_stay_close_to_exact() {
    let prompts = [
        (
            "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            "<|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of France? \
             Answer in one sentence.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        ),
        (
            "Qwen3.5-0.8B-Q4_K_M.gguf",
            "<|im_start|>user\nWhat is the capital of France? Answer in one sentence.<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n",
        ),
        (
            "gemma-4-E2B_q4_0-it.gguf",
            "<|turn>user\nWhat is the capital of France? Answer in one sentence.<turn|>\n<|turn>model\n",
        ),
    ];
    for (name, prompt) in prompts {
        let Some(file) = model(name) else { continue };
        let (exact, exact_answer) = france(&file, prompt, Precision::Exact);
        let (fast, fast_answer) = france(&file, prompt, Precision::Int8);
        let worst = exact
            .iter()
            .zip(&fast)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let top = |p: &[f32]| (0..p.len()).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap();
        eprintln!(
            "{name}: likeliest first token {:.2}% exact, {:.2}% 8-bit; biggest probability change {:.2} points; answer {fast_answer:?}",
            exact[top(&exact)] * 100.0,
            fast[top(&exact)] * 100.0,
            worst * 100.0
        );
        assert_eq!(
            top(&exact),
            top(&fast),
            "{name}: a different likeliest token"
        );
        assert!(worst < 0.05, "{name}: probabilities {worst} apart");
        assert_eq!(exact_answer, fast_answer, "{name}: a different answer");
        assert_eq!(fast_answer, "The capital of France is Paris.");
    }
}

/// With activations rounded to 8 bits, the tensor-core kernel (a batch) and the dp4a
/// kernel (one row of x at a time) sum the same whole numbers, so they agree to f32
/// rounding; both are within the rounding of x of the exact CPU result. Prints speeds.
fn check_8bit(gpu: &Gpu, file: &GgufFile, names: &[&str]) {
    let pool = Pool::with_all_cores();
    let tokens = 100;
    for &name in names {
        let w = matrix(file, name);
        if !Gpu::supports_8bit(w.kind) {
            // The Metal backend has no 8-bit kernels.
            eprintln!("skipped: {name} ({}) has no 8-bit kernels on this GPU", w.kind);
            continue;
        }
        let x: Vec<f32> = (0..w.cols * tokens)
            .map(|c| ((c * 7919) % 1000) as f32 / 250.0 - 2.0)
            .collect();
        let mut exact = vec![0.0; w.rows * tokens];
        for (xt, yt) in x.chunks(w.cols).zip(exact.chunks_mut(w.rows)) {
            cpu::matvec(&pool, &w, xt, yt);
        }
        let weights = gpu.upload_weights(w.kind, w.data, w.rows, w.cols).unwrap();
        let x_gpu = gpu.upload_f32(&x).unwrap();
        let q8 = gpu.alloc_q8(tokens, w.cols).unwrap();
        let one = gpu.alloc_q8(1, w.cols).unwrap();
        let y_batch = gpu.alloc(w.rows * tokens * 4).unwrap();
        let y_one = gpu.alloc(w.rows * 4).unwrap();
        gpu.quantize(x_gpu.ptr(), &q8, tokens, w.cols).unwrap();
        let batch = || {
            gpu.matmul_8bit(
                w.kind,
                weights.ptr(),
                &q8,
                y_batch.ptr(),
                w.rows,
                w.cols,
                tokens,
            )
            .unwrap()
        };
        batch();
        let mut got = vec![0.0; w.rows * tokens];
        y_batch.read_f32(&mut got).unwrap();
        let mut by_row = vec![0.0; w.rows * tokens];
        for t in 0..tokens {
            gpu.quantize(x_gpu.f32_at(t * w.cols), &one, 1, w.cols)
                .unwrap();
            gpu.matmul_8bit(w.kind, weights.ptr(), &one, y_one.ptr(), w.rows, w.cols, 1)
                .unwrap();
            y_one
                .read_f32(&mut by_row[t * w.rows..(t + 1) * w.rows])
                .unwrap();
        }
        let scale = exact.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max)
        };
        let (kernels, rounding) = (worst(&got, &by_row), worst(&got, &exact));
        // On CUDA both kernels round x to the same 8 bits, so they agree but for the order
        // of sums. On Metal a batch rounds the weights to f16 on the neural accelerators
        // and one token is f32, so they're as far apart as f16 rounding takes them.
        let agree = if cfg!(target_os = "macos") { 2e-3 } else { 2e-5 };
        assert!(
            kernels <= scale * agree,
            "{name}: tensor cores and dp4a differ by {kernels} (largest value {scale})"
        );
        assert!(
            rounding <= scale * 0.02,
            "{name}: 8-bit result off the exact one by {rounding} (largest value {scale})"
        );

        let warm = Instant::now();
        while warm.elapsed().as_millis() < 300 {
            batch();
            gpu.synchronize().unwrap();
        }
        let runs = 20;
        let started = Instant::now();
        for _ in 0..runs {
            batch();
        }
        gpu.synchronize().unwrap();
        let batch_time = started.elapsed() / runs;
        let single = || {
            gpu.matmul_8bit(w.kind, weights.ptr(), &one, y_one.ptr(), w.rows, w.cols, 1)
                .unwrap()
        };
        let runs = 200;
        let started = Instant::now();
        for _ in 0..runs {
            single();
        }
        gpu.synchronize().unwrap();
        let single_time = started.elapsed() / runs;
        eprintln!(
            "{name:28} {:>5} {:>6}x{:<5} kernels differ {:.1e}, off exact {:.1e} of {:>5.1}  x{tokens}: {:>7.1?} ({:>5.1} TOPS)  x1: {:>7.1?} ({:>4.0} GB/s)",
            w.kind.name(),
            w.rows,
            w.cols,
            kernels,
            rounding,
            scale,
            batch_time,
            2.0 * (w.rows * w.cols * tokens) as f64 / batch_time.as_secs_f64() / 1e12,
            single_time,
            w.data.len() as f64 / single_time.as_secs_f64() / 1e9,
        );
    }
}

#[test]
fn eight_bit_kernels_agree_and_stay_close_to_exact() {
    let Some(gpu) = open_gpu() else { return };
    if let Some(file) = model("Llama-3.2-1B-Instruct-Q4_K_M.gguf") {
        check_8bit(
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
    if let Some(file) = model("Qwen3.5-0.8B-Q4_K_M.gguf") {
        check_8bit(
            &gpu,
            &file,
            &["blk.0.attn_qkv.weight", "blk.0.ssm_alpha.weight"],
        );
    }
    if let Some(file) = model("gemma-4-E2B_q4_0-it.gguf") {
        check_8bit(
            &gpu,
            &file,
            &["blk.0.ffn_gate.weight", "blk.20.ffn_down.weight"],
        );
    }
}

/// Attention over an f16 cache (8-bit mode), for each head size: the tensor-core kernel
/// (a batch of tokens) and the per-token kernel against f64 arithmetic on the same f16
/// keys and values. The tensor cores also round the queries and the softmax weights to
/// f16, so they're allowed a little more.
#[test]
fn attention_over_an_f16_cache_matches_f64() {
    use vireo_engine::half::{f16_to_f32, f32_to_f16};
    let Some(gpu) = open_gpu() else { return };
    // (head size, query heads, key/value heads, sliding window, scale)
    for (hd, heads, kv_heads, window, scale) in [
        (64, 8, 2, 0, 0.125),
        (128, 8, 2, 0, 0.088),
        (256, 8, 2, 0, 0.0625),
        (256, 8, 1, 40, 1.0),
        (512, 8, 1, 0, 1.0),
    ] {
        let (p0, tokens) = (30, 100);
        let len = p0 + tokens;
        let kv_dim = kv_heads * hd;
        let wave = |i: usize, k: usize| (((i * k) % 1009) as f32 / 504.5 - 1.0) * 0.6;
        let q: Vec<f32> = (0..tokens * heads * hd).map(|i| wave(i, 7919)).collect();
        let k16: Vec<u16> = (0..len * kv_dim)
            .map(|i| f32_to_f16(wave(i, 6007)))
            .collect();
        let v16: Vec<u16> = (0..len * kv_dim)
            .map(|i| f32_to_f16(wave(i, 3001)))
            .collect();
        let bytes = |h: &[u16]| h.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
        let (keys, values) = (
            gpu.upload(&bytes(&k16)).unwrap(),
            gpu.upload(&bytes(&v16)).unwrap(),
        );
        let q_gpu = gpu.upload_f32(&q).unwrap();
        let out = gpu.alloc(tokens * heads * hd * 4).unwrap();
        let pos = gpu.alloc(4).unwrap();

        // The exact answer, in f64.
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
                            .map(|d| qh[d] as f64 * f16_to_f32(k16[j * kv_dim + kv + d]) as f64)
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
                        .map(|(j, w)| w * f16_to_f32(v16[j * kv_dim + kv + d]) as f64)
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
        for t in 0..tokens {
            run(t, 1);
        }
        out.read_f32(&mut got).unwrap();
        let single = worst(&got);
        eprintln!(
            "head size {hd}, window {window}: batch off by {batch:.1e}, one at a time {single:.1e} (largest {largest:.2})"
        );
        assert!(single <= largest * 1e-5, "one at a time: {single}");
        assert!(batch <= largest * 2e-3, "batch: {batch}");
    }
}

/// How far `got`'s logits are from `want`'s, as a share of `want`'s largest, and whether
/// both pick the same next token.
fn compare_logits(name: &str, got: &[f32], want: &[f32], tolerance: f32) {
    let worst = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let largest = want.iter().fold(0f32, |m, x| m.max(x.abs()));
    let pick = |l: &[f32]| (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap();
    eprintln!("{name}: worst logit difference {worst:.2e} of {largest:.1}");
    assert!(worst <= largest * tolerance, "{name}: {worst} of {largest}");
    assert_eq!(pick(got), pick(want), "{name}: a different next token");
}

/// A chat's next turn read on top of the last one (the session cut back to the tokens
/// both share, or Qwen's DeltaNet taken back to a checkpoint) gives the logits a new
/// session reading the whole prompt gives. The last turn's own tail (a reply the next
/// prompt writes differently) is thrown away first, as the server does.
#[test]
fn continuing_a_session_matches_reading_from_the_start() {
    for (precision, tolerance) in [(Precision::Exact, 1e-4), (Precision::Int8, 0.05)] {
        let label = |model: &str| format!("{model}, {precision:?}");
        // The shared start, the last turn's tail (thrown away) and the new turn.
        let split = |ids: &[u32]| (ids.len() / 2, ids.len() * 3 / 4);
        if let Some(file) = model("Llama-3.2-1B-Instruct-Q4_K_M.gguf") {
            let Some(gpu) = open_gpu() else { return };
            let tokenizer = Tokenizer::from_gguf(&file).unwrap();
            let ids = long_text(&tokenizer, false);
            let (shared, tail) = split(&ids);
            let mut m = LlamaCuda::load(&file, gpu).unwrap();
            m.set_precision(precision);
            let mut s = m.session(ids.len() + 64).unwrap();
            m.forward_batch(&mut s, &ids[..tail], Logits::Last).unwrap();
            for &id in &ids[..8] {
                m.forward(&mut s, id, true).unwrap(); // a reply, thrown away
            }
            s.truncate(shared);
            let got = m
                .forward_batch(&mut s, &ids[shared..], Logits::Last)
                .unwrap()
                .unwrap()
                .to_vec();
            let mut fresh = m.session(ids.len()).unwrap();
            let want = m
                .forward_batch(&mut fresh, &ids, Logits::Last)
                .unwrap()
                .unwrap()
                .to_vec();
            compare_logits(&label("Llama"), &got, &want, tolerance);
        }
        if let Some(file) = model("Qwen3.5-0.8B-Q4_K_M.gguf") {
            let Some(gpu) = open_gpu() else { return };
            let tokenizer = Tokenizer::from_gguf(&file).unwrap();
            let ids = long_text(&tokenizer, false);
            let (shared, tail) = split(&ids);
            let mut m = Qwen35Cuda::load(&file, gpu).unwrap();
            m.set_precision(precision);
            let mut s = m.session(ids.len() + 64).unwrap();
            let mut cp = None;
            m.forward_batch(&mut s, &ids[..shared], Logits::None)
                .unwrap();
            m.save(&s, &mut cp).unwrap();
            m.forward_batch(&mut s, &ids[shared..tail], Logits::Last)
                .unwrap();
            for &id in &ids[..8] {
                m.forward(&mut s, id, true).unwrap();
            }
            m.restore(&mut s, cp.as_ref().unwrap()).unwrap();
            let got = m
                .forward_batch(&mut s, &ids[shared..], Logits::Last)
                .unwrap()
                .unwrap()
                .to_vec();
            let mut fresh = m.session(ids.len()).unwrap();
            let want = m
                .forward_batch(&mut fresh, &ids, Logits::Last)
                .unwrap()
                .unwrap()
                .to_vec();
            compare_logits(&label("Qwen 3.5"), &got, &want, tolerance);
            // And from empty again, after a reset.
            m.reset(&mut s).unwrap();
            let again = m
                .forward_batch(&mut s, &ids, Logits::Last)
                .unwrap()
                .unwrap()
                .to_vec();
            compare_logits(&label("Qwen 3.5 after a reset"), &again, &want, 1e-6);
        }
        if let Some(file) = model("gemma-4-E2B_q4_0-it.gguf") {
            let Some(gpu) = open_gpu() else { return };
            let tokenizer = Tokenizer::from_gguf(&file).unwrap();
            let ids = long_text(&tokenizer, true);
            let (shared, tail) = split(&ids);
            let mut m = Gemma4Cuda::load(&file, gpu).unwrap();
            m.set_precision(precision);
            let mut s = m.session(ids.len() + 64).unwrap();
            m.forward_batch(&mut s, &ids[..tail], Logits::Last).unwrap();
            for &id in &ids[..8] {
                m.forward(&mut s, id, true).unwrap();
            }
            s.truncate(shared);
            let got = m
                .forward_batch(&mut s, &ids[shared..], Logits::Last)
                .unwrap()
                .unwrap()
                .to_vec();
            let mut fresh = m.session(ids.len()).unwrap();
            let want = m
                .forward_batch(&mut fresh, &ids, Logits::Last)
                .unwrap()
                .unwrap()
                .to_vec();
            compare_logits(&label("Gemma 4"), &got, &want, tolerance);
        }
    }
}

/// A session's state exported to system memory and imported into a new session (of
/// another capacity) goes on exactly as the session it came from does: the same logits
/// to the bit. Llama and Gemma can import part of a state (a shared start); Qwen's
/// imports whole, from a checkpoint.
#[test]
fn an_exported_session_goes_on_as_the_original() {
    let same = |name: &str, a: &[f32], b: &[f32]| {
        assert!(a == b, "{name}: an imported session gives other logits");
        eprintln!("{name}: the same logits");
    };
    for precision in [Precision::Exact, Precision::Int8] {
        let label = |model: &str, what: &str| format!("{model}, {precision:?}, {what}");
        if let Some(file) = model("Llama-3.2-1B-Instruct-Q4_K_M.gguf") {
            let Some(gpu) = open_gpu() else { return };
            let tokenizer = Tokenizer::from_gguf(&file).unwrap();
            let ids = long_text(&tokenizer, false);
            let (part, half) = (ids.len() / 3, ids.len() / 2);
            let mut m = LlamaCuda::load(&file, gpu).unwrap();
            m.set_precision(precision);
            let mut a = m.session(ids.len()).unwrap();
            m.forward_batch(&mut a, &ids[..half], Logits::None).unwrap();
            let saved = a.export().unwrap();
            assert_eq!(saved.len(), half);
            let mut b = m.session(ids.len() + 100).unwrap();
            b.import(&saved, half).unwrap();
            let la = m
                .forward_batch(&mut a, &ids[half..], Logits::Last)
                .unwrap()
                .unwrap()
                .to_vec();
            let lb = m
                .forward_batch(&mut b, &ids[half..], Logits::Last)
                .unwrap()
                .unwrap()
                .to_vec();
            same(&label("Llama", "whole"), &la, &lb);
            // Part of it, against the original cut back as far.
            a.truncate(part);
            b.import(&saved, part).unwrap();
            let la = m
                .forward_batch(&mut a, &ids[part..], Logits::Last)
                .unwrap()
                .unwrap()
                .to_vec();
            let lb = m
                .forward_batch(&mut b, &ids[part..], Logits::Last)
                .unwrap()
                .unwrap()
                .to_vec();
            same(&label("Llama", "part"), &la, &lb);
        }
        if let Some(file) = model("Qwen3.5-0.8B-Q4_K_M.gguf") {
            let Some(gpu) = open_gpu() else { return };
            let tokenizer = Tokenizer::from_gguf(&file).unwrap();
            let ids = long_text(&tokenizer, false);
            let half = ids.len() / 2;
            let mut m = Qwen35Cuda::load(&file, gpu).unwrap();
            m.set_precision(precision);
            let mut a = m.session(ids.len()).unwrap();
            let mut cp = None;
            m.forward_batch(&mut a, &ids[..half], Logits::None).unwrap();
            m.save(&a, &mut cp).unwrap();
            // The original goes on past the checkpoint before it's exported.
            m.forward_batch(&mut a, &ids[half..half + 20], Logits::None)
                .unwrap();
            let saved = a.export(cp.as_ref().unwrap()).unwrap();
            let mut b = m.session(ids.len() + 100).unwrap();
            b.import(&saved).unwrap();
            m.restore(&mut a, cp.as_ref().unwrap()).unwrap();
            let la = m
                .forward_batch(&mut a, &ids[half..], Logits::Last)
                .unwrap()
                .unwrap()
                .to_vec();
            let lb = m
                .forward_batch(&mut b, &ids[half..], Logits::Last)
                .unwrap()
                .unwrap()
                .to_vec();
            same(&label("Qwen 3.5", "from a checkpoint"), &la, &lb);
            // A state from another model's session doesn't load.
            assert!(
                b.import(&vireo_engine::models::SavedState::from_bytes(
                    half,
                    vec![0; 10]
                ))
                .is_err()
            );
        }
        if let Some(file) = model("gemma-4-E2B_q4_0-it.gguf") {
            let Some(gpu) = open_gpu() else { return };
            let tokenizer = Tokenizer::from_gguf(&file).unwrap();
            let ids = long_text(&tokenizer, true).repeat(2);
            let most = ids.len() * 3 / 4;
            let mut m = Gemma4Cuda::load(&file, gpu).unwrap();
            m.set_precision(precision);
            let window = m.config.sliding_window;
            assert!(most > window, "the text should outrun the sliding window");
            let mut a = m.session(ids.len()).unwrap();
            m.forward_batch(&mut a, &ids[..most], Logits::None).unwrap();
            // Past the window, its layers' older positions are left out.
            let saved = a.export(most).unwrap();
            let full_size = {
                let early = a.export(window).unwrap();
                early.bytes().len() / window * most
            };
            eprintln!(
                "Gemma 4, {precision:?}: {most} positions saved in {} bytes, {full_size} with every position",
                saved.bytes().len()
            );
            assert!(saved.bytes().len() < full_size);
            let mut b = m.session(ids.len() + 100).unwrap();
            b.import(&saved, most).unwrap();
            let la = m
                .forward_batch(&mut a, &ids[most..], Logits::Last)
                .unwrap()
                .unwrap()
                .to_vec();
            let lb = m
                .forward_batch(&mut b, &ids[most..], Logits::Last)
                .unwrap()
                .unwrap()
                .to_vec();
            same(&label("Gemma 4", "past its window"), &la, &lb);
            // Only whole, then: what an earlier start needs wasn't kept.
            assert!(b.import(&saved, most / 2).is_err());
            // Within the window every position is there, so part of it loads.
            let part = window / 2;
            let early = a.export(window).unwrap();
            a.truncate(part);
            b.import(&early, part).unwrap();
            let la = m
                .forward_batch(&mut a, &ids[part..], Logits::Last)
                .unwrap()
                .unwrap()
                .to_vec();
            let lb = m
                .forward_batch(&mut b, &ids[part..], Logits::Last)
                .unwrap()
                .unwrap()
                .to_vec();
            same(&label("Gemma 4", "part, within its window"), &la, &lb);
        }
    }
}

/// A prompt's Gated DeltaNet in chunks (the WY form) gives what reading it token by token
/// does, for prompts shorter than a chunk, of whole chunks and in between: on random
/// inputs shaped as Qwen 3.5 0.8B's (16 value heads of 128, 16 key heads), from a state
/// that isn't zero.
#[test]
fn delta_net_in_chunks_matches_token_by_token() {
    use vireo_engine::cuda::DeltaNet;
    let Some(gpu) = open_gpu() else { return };
    let (v_heads, k_heads, n) = (16, 16, 128);
    let key_dim = k_heads * n;
    let channels = 2 * key_dim + v_heads * n;
    let mut seed = 12345u64;
    let mut random = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    };
    for tokens in [7, 32, 33, 100] {
        let conv: Vec<f32> = (0..tokens * channels).map(|_| random()).collect();
        let beta: Vec<f32> = (0..tokens * v_heads).map(|_| random()).collect();
        let alpha: Vec<f32> = (0..tokens * v_heads).map(|_| random()).collect();
        let dt: Vec<f32> = (0..v_heads).map(|_| random()).collect();
        let a: Vec<f32> = (0..v_heads).map(|_| -random().abs() - 0.1).collect();
        let z: Vec<f32> = (0..tokens * v_heads * n).map(|_| random()).collect();
        let norm: Vec<f32> = (0..n).map(|_| 1.0 + 0.1 * random()).collect();
        let state0: Vec<f32> = (0..v_heads * n * n).map(|_| 0.1 * random()).collect();
        let run = |chunked: bool| -> (Vec<f32>, Vec<f32>) {
            let up = |v: &[f32]| gpu.upload_f32(v).unwrap();
            let (c, b, al, d, aa, zz, nn, st) = (
                up(&conv),
                up(&beta),
                up(&alpha),
                up(&dt),
                up(&a),
                up(&z),
                up(&norm),
                up(&state0),
            );
            let out = gpu.alloc(tokens * v_heads * n * 4).unwrap();
            let scratch = gpu
                .alloc(Gpu::delta_scratch_floats(tokens, v_heads, n) * 4)
                .unwrap();
            gpu.delta_net(
                &DeltaNet {
                    conv: c.ptr(),
                    channels,
                    key_dim,
                    beta_raw: b.ptr(),
                    alpha_raw: al.ptr(),
                    dt_bias: d.ptr(),
                    a: aa.ptr(),
                    state: st.ptr(),
                    z: zz.ptr(),
                    norm: nn.ptr(),
                    out: out.ptr(),
                    v_heads,
                    n,
                    k_heads,
                    eps: 1e-6,
                    scratch: if chunked { scratch.ptr() } else { 0 },
                },
                tokens,
            )
            .unwrap();
            let mut o = vec![0.0; tokens * v_heads * n];
            out.read_f32(&mut o).unwrap();
            let mut s = vec![0.0; v_heads * n * n];
            st.read_f32(&mut s).unwrap();
            (o, s)
        };
        let (o1, s1) = run(false);
        let (o2, s2) = run(true);
        let worst = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let scale = |a: &[f32]| a.iter().fold(0f32, |m, v| m.max(v.abs()));
        let (wo, ws) = (worst(&o1, &o2), worst(&s1, &s2));
        eprintln!(
            "{tokens} tokens: output differs by {wo:.2e} (largest {:.2}), state by {ws:.2e} (largest {:.2})",
            scale(&o1),
            scale(&s1)
        );
        assert!(wo <= 1e-4 * scale(&o1).max(1.0), "{tokens} tokens: outputs differ by {wo}");
        assert!(ws <= 1e-4 * scale(&s1).max(1.0), "{tokens} tokens: states differ by {ws}");
    }
}
