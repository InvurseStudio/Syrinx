//! Qwen's multi-token-prediction layer as a draft: how often its guesses are the tokens the
//! model goes on to pick (greedy), on a large model with the layer (Qwen 3.8 27B).
//! Ignored by default; run with
//! `VIREO_MTP_MODEL=<model.gguf> cargo test --release -p vireo-engine --test mtp -- --ignored --nocapture`.

// The Metal backend's own kernels and drafting (`matvec_multi`, `use_mtp`): Mac only.
#![cfg(target_os = "macos")]

use std::time::Instant;

use vireo_engine::cuda::Gpu;
use vireo_engine::models::Logits;
use vireo_engine::models::qwen35_cuda::Qwen35Cuda;
use vireo_engine::{GgufFile, Tokenizer};

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .fold(
            (0, f32::NEG_INFINITY),
            |b, (i, &x)| if x > b.1 { (i, x) } else { b },
        )
        .0 as u32
}

#[test]
#[ignore]
fn mtp_guesses_what_the_model_picks() {
    let Ok(path) = std::env::var("VIREO_MTP_MODEL") else {
        eprintln!("skipped: VIREO_MTP_MODEL isn't set");
        return;
    };
    let file = GgufFile::open(path).unwrap();
    let tokenizer = Tokenizer::from_gguf(&file).unwrap();
    let m = Qwen35Cuda::load(&file, Gpu::open().unwrap()).unwrap();
    assert!(m.has_mtp(), "the model has no multi-token-prediction layer");
    let prompts = [
        "Write a short story about a lighthouse keeper who finds a message in a bottle.",
        "Explain how a hash map works, with a small example in Python.",
        "List five things to check before a long road trip, one line each.",
    ];
    let k = 3;
    let (mut steps, mut first, mut runs) = (0usize, 0usize, vec![0usize; k + 1]);
    let mut drafting = 0f64;
    for prompt in prompts {
        let text = format!(
            "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        let ids = tokenizer.encode(&text, true, true);
        let mut s = m.session(4096).unwrap();
        m.use_mtp(&mut s).unwrap();
        let mut pick = argmax(
            m.forward_batch(&mut s, &ids, Logits::Last)
                .unwrap()
                .unwrap(),
        );
        let mut written = vec![pick];
        let mut guesses = Vec::new();
        for _ in 0..120 {
            let started = Instant::now();
            guesses.push(m.mtp_draft(&mut s, pick, k).unwrap());
            drafting += started.elapsed().as_secs_f64();
            pick = argmax(m.forward(&mut s, pick, true).unwrap().unwrap());
            written.push(pick);
        }
        // Guesses made after written[i] are for written[i + 1..].
        for (i, g) in guesses.iter().enumerate() {
            let truth = &written[i + 1..];
            if truth.len() < k {
                break;
            }
            steps += 1;
            first += usize::from(g[0] == truth[0]);
            let run = g.iter().zip(truth).take_while(|(a, b)| a == b).count();
            runs[run] += 1;
        }
        eprintln!(
            "{prompt:.40}… {:?}",
            tokenizer.decode(&written[..40], false)
        );
    }
    let taken: usize = runs.iter().enumerate().map(|(n, c)| n * c).sum();
    eprintln!(
        "{steps} rounds of {k}: first guess right {:.0}%, runs of 0..={k} right {:?}, {:.2} guesses taken a round; drafting {:.1} ms a round",
        100.0 * first as f64 / steps as f64,
        runs,
        taken as f64 / steps as f64,
        drafting * 1e3 / steps as f64
    );
    assert!(
        first * 2 > steps,
        "the first guess should be right most of the time"
    );
}
