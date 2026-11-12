//! What a few tokens read at once cost a large model deep in a chat: a draft's guesses
//! checked in one pass, a short tool result, a new turn. Gemma 4 (and Muse Glimmer, which
//! runs on its code) and Qwen 3.5. Ignored by default; run with
//! `VIREO_CHECK_MODEL=<model.gguf> VIREO_CHECK_TEXT=<file> cargo test --release -p vireo-engine --test check_speed -- --ignored --nocapture`
//! (`VIREO_CHECK_DEPTHS`, tokens read first, default "2000,8000,30000").

// The Metal backend's few-token kernels: Mac only.
#![cfg(target_os = "macos")]

use std::time::Instant;

use vireo_engine::cuda::Gpu;
use vireo_engine::models::Logits;
use vireo_engine::models::gemma4_cuda::Gemma4Cuda;
use vireo_engine::models::qwen35_cuda::Qwen35Cuda;
use vireo_engine::{GgufFile, Tokenizer};

const COUNTS: [usize; 10] = [1, 2, 3, 4, 5, 6, 8, 12, 16, 32];

#[test]
#[ignore]
fn a_few_tokens_deep_in_a_chat() {
    let (Ok(path), Ok(text)) = (
        std::env::var("VIREO_CHECK_MODEL"),
        std::env::var("VIREO_CHECK_TEXT"),
    ) else {
        eprintln!("skipped: VIREO_CHECK_MODEL and VIREO_CHECK_TEXT aren't set");
        return;
    };
    let depths: Vec<usize> = std::env::var("VIREO_CHECK_DEPTHS")
        .unwrap_or_else(|_| "2000,8000,30000".into())
        .split(',')
        .map(|d| d.parse().unwrap())
        .collect();
    let file = GgufFile::open(&path).unwrap();
    let tokenizer = Tokenizer::from_gguf(&file).unwrap();
    let ids = tokenizer.encode(&std::fs::read_to_string(text).unwrap(), true, false);
    let most = *depths.iter().max().unwrap();
    assert!(ids.len() >= most + 64, "the text has {} tokens", ids.len());
    let gpu = Gpu::open().unwrap();
    let qwen = file.architecture().is_some_and(|a| a.starts_with("qwen"));
    let report = |depth: usize, ms: &[f64]| {
        let cells: Vec<String> = COUNTS
            .iter()
            .zip(ms)
            .map(|(n, t)| format!("{n}: {t:.1}"))
            .collect();
        eprintln!("at {depth:>6} tokens, ms a batch of  {}", cells.join("  "));
    };
    if qwen {
        let m = Qwen35Cuda::load(&file, gpu).unwrap();
        let mut s = m.session(most + 128).unwrap();
        m.keep_steps(&mut s).unwrap();
        let mut read = 0;
        for &depth in &depths {
            m.forward_batch(&mut s, &ids[read..depth], Logits::Last)
                .unwrap();
            read = depth;
            let mut ms = Vec::new();
            for n in COUNTS {
                // Qwen goes back over up to 8 tokens; longer batches are read on.
                let (runs, back) = if n <= 8 { (6, true) } else { (1, false) };
                let mut best = f64::MAX;
                for _ in 0..runs {
                    let t = Instant::now();
                    m.forward_batch(&mut s, &ids[read..read + n], Logits::Last)
                        .unwrap();
                    m.gpu().synchronize().unwrap();
                    best = best.min(t.elapsed().as_secs_f64() * 1e3);
                    if back {
                        assert!(m.rewind(&mut s, read).unwrap());
                    } else {
                        read += n;
                    }
                }
                ms.push(best);
            }
            report(depth, &ms);
        }
    } else {
        let m = Gemma4Cuda::load(&file, gpu).unwrap();
        let mut s = m.session(most + 128).unwrap();
        let mut read = 0;
        for &depth in &depths {
            m.forward_batch(&mut s, &ids[read..depth], Logits::Last)
                .unwrap();
            read = depth;
            let mut ms = Vec::new();
            for n in COUNTS {
                let mut best = f64::MAX;
                for _ in 0..6 {
                    let t = Instant::now();
                    m.forward_batch(&mut s, &ids[read..read + n], Logits::Last)
                        .unwrap();
                    m.gpu().synchronize().unwrap();
                    best = best.min(t.elapsed().as_secs_f64() * 1e3);
                    assert_eq!(s.truncate(read), read);
                }
                ms.push(best);
            }
            report(depth, &ms);
        }
    }
}
