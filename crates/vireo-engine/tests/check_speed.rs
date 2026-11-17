//! What a few tokens read at once cost a large model deep in a chat: a draft's guesses
//! checked in one pass, a short tool result, a new turn. Gemma 4 (and Muse Glimmer, which
//! runs on its code) and Qwen 3.5. Ignored by default; run with
//! `VIREO_CHECK_MODEL=<model.gguf> VIREO_CHECK_TEXT=<file> cargo test --release -p vireo-engine --test check_speed -- --ignored --nocapture`
//! (`VIREO_CHECK_DEPTHS`, tokens read first, default "2000,8000,30000"; `VIREO_CHECK_COUNTS`,
//! the batch sizes timed, and nothing else, say "4" with VIREO_METAL_PROFILE to see one
//! size's kernels).

use std::time::Instant;

use vireo_engine::cuda::Gpu;
use vireo_engine::models::Logits;
use vireo_engine::models::gemma4_cuda::Gemma4Cuda;
use vireo_engine::models::qwen35_cuda::Qwen35Cuda;
use vireo_engine::{GgufFile, Tokenizer};

const ALL_COUNTS: [usize; 13] = [1, 2, 3, 4, 5, 6, 8, 12, 16, 32, 64, 128, 256];

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
    let only: Option<Vec<usize>> = std::env::var("VIREO_CHECK_COUNTS")
        .ok()
        .map(|c| c.split(',').filter_map(|n| n.parse().ok()).collect());
    let counts: Vec<usize> = only.clone().unwrap_or_else(|| ALL_COUNTS.to_vec());
    let counts = &counts[..];
    let depths: Vec<usize> = std::env::var("VIREO_CHECK_DEPTHS")
        .unwrap_or_else(|_| "2000,8000,30000".into())
        .split(',')
        .map(|d| d.parse().unwrap())
        .collect();
    let file = GgufFile::open(&path).unwrap();
    let tokenizer = Tokenizer::from_gguf(&file).unwrap();
    let ids = tokenizer.encode(&std::fs::read_to_string(text).unwrap(), true, false);
    let most = *depths.iter().max().unwrap();
    assert!(ids.len() >= most + 256, "the text has {} tokens", ids.len());
    let gpu = Gpu::open().unwrap();
    let qwen = file.architecture().is_some_and(|a| a.starts_with("qwen"));
    let report = |depth: usize, ms: &[f64]| {
        let cells: Vec<String> = counts
            .iter()
            .zip(ms)
            .map(|(n, t)| format!("{n}: {t:.1}"))
            .collect();
        eprintln!("at {depth:>6} tokens, ms a batch of  {}", cells.join("  "));
    };
    if qwen {
        let m = Qwen35Cuda::load(&file, gpu).unwrap();
        let mut s = m.session(most + 512).unwrap();
        // Short batches keep their DeltaNet steps, as the server's do when it drafts. Going
        // back over a timed batch is through a checkpoint made before it (restored outside
        // the timing), whatever its length.
        if m.can_rewind() {
            m.keep_steps(&mut s).unwrap();
        }
        let mut checkpoint = None;
        let mut read = 0;
        for &depth in &depths {
            m.forward_batch(&mut s, &ids[read..depth], Logits::Last)
                .unwrap();
            read = depth;
            m.save(&s, &mut checkpoint).unwrap();
            m.gpu().synchronize().unwrap();
            let go_back = |s: &mut vireo_engine::models::qwen35_cuda::Qwen35CudaSession| {
                m.restore(s, checkpoint.as_ref().unwrap()).unwrap();
                m.gpu().synchronize().unwrap();
            };
            if only.is_none() {
                // One token as it's generated (the recorded graph), and how much of a short
                // batch's time is the CPU queuing it (no logits read, so nothing waits).
                let mut best = f64::MAX;
                for _ in 0..6 {
                    let t = Instant::now();
                    m.forward(&mut s, ids[read], true).unwrap();
                    best = best.min(t.elapsed().as_secs_f64() * 1e3);
                    go_back(&mut s);
                }
                let mut line = format!("at {depth:>6} tokens, one generated {best:.1} ms; queued / done:");
                for n in [2usize, 8, 64] {
                    let (mut q, mut d) = (f64::MAX, f64::MAX);
                    for _ in 0..6 {
                        let t = Instant::now();
                        m.forward_batch(&mut s, &ids[read..read + n], Logits::None)
                            .unwrap();
                        q = q.min(t.elapsed().as_secs_f64() * 1e3);
                        m.gpu().synchronize().unwrap();
                        d = d.min(t.elapsed().as_secs_f64() * 1e3);
                        go_back(&mut s);
                    }
                    line += &format!("  {n}: {q:.1} / {d:.1}");
                }
                eprintln!("{line}");
            }
            let mut ms = Vec::new();
            for &n in counts {
                let mut best = f64::MAX;
                for _ in 0..6 {
                    let t = Instant::now();
                    m.forward_batch(&mut s, &ids[read..read + n], Logits::Last)
                        .unwrap();
                    m.gpu().synchronize().unwrap();
                    best = best.min(t.elapsed().as_secs_f64() * 1e3);
                    go_back(&mut s);
                }
                ms.push(best);
            }
            report(depth, &ms);
        }
    } else {
        let m = Gemma4Cuda::load(&file, gpu).unwrap();
        let mut s = m.session(most + 512).unwrap();
        let mut read = 0;
        for &depth in &depths {
            m.forward_batch(&mut s, &ids[read..depth], Logits::Last)
                .unwrap();
            read = depth;
            if only.is_none() {
                let mut best = f64::MAX;
                for _ in 0..6 {
                    let t = Instant::now();
                    m.forward(&mut s, ids[read], true).unwrap();
                    best = best.min(t.elapsed().as_secs_f64() * 1e3);
                    assert_eq!(s.truncate(read), read);
                }
                let mut line = format!("at {depth:>6} tokens, one generated {best:.1} ms; queued / done:");
                for n in [2usize, 8, 64] {
                    let (mut q, mut d) = (f64::MAX, f64::MAX);
                    for _ in 0..6 {
                        let t = Instant::now();
                        m.forward_batch(&mut s, &ids[read..read + n], Logits::None)
                            .unwrap();
                        q = q.min(t.elapsed().as_secs_f64() * 1e3);
                        m.gpu().synchronize().unwrap();
                        d = d.min(t.elapsed().as_secs_f64() * 1e3);
                        assert_eq!(s.truncate(read), read);
                    }
                    line += &format!("  {n}: {q:.1} / {d:.1}");
                }
                eprintln!("{line}");
            }
            let mut ms = Vec::new();
            for &n in counts {
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
