//! Google's draft model for Gemma 4 (`gemma-4-31B-it-assistant`, and the E2B's and E4B's
//! with their centroid heads) as a draft: how often
//! its guesses are the tokens the model goes on to pick (greedy), as `tests/mtp.rs` does
//! for Qwen's layer. A wrong reading of the main model's state or keys shows as guesses
//! that are almost never right. Ignored by default; run with
//! `VIREO_GEMMA_MODEL=<gemma-4-31B-it.gguf> [VIREO_GEMMA_ASSISTANT=<folder>] cargo test --release -p vireo-engine --test gemma_assistant -- --ignored --nocapture`
//! (the draft model is found beside the model or in the app's DraftModels folder when
//! VIREO_GEMMA_ASSISTANT isn't set).

use std::path::Path;
use std::time::Instant;

use vireo_engine::cuda::Gpu;
use vireo_engine::models::Logits;
use vireo_engine::models::gemma4_cuda::{Gemma4Cuda, assistant};
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
fn the_draft_model_guesses_what_gemma_picks() {
    let Ok(path) = std::env::var("VIREO_GEMMA_MODEL") else {
        eprintln!("skipped: VIREO_GEMMA_MODEL isn't set");
        return;
    };
    let draft = std::env::var_os("VIREO_GEMMA_ASSISTANT")
        .map(Into::into)
        .or_else(|| assistant::find(Path::new(&path)))
        .expect("no draft model found: set VIREO_GEMMA_ASSISTANT");
    let file = GgufFile::open(&path).unwrap();
    let tokenizer = Tokenizer::from_gguf(&file).unwrap();
    let mut m = Gemma4Cuda::load(&file, Gpu::open().unwrap()).unwrap();
    let started = Instant::now();
    m.load_assistant(&draft).unwrap();
    eprintln!("draft model loaded in {:.1?}", started.elapsed());
    let prompts = [
        "Write a short story about a lighthouse keeper who finds a message in a bottle.",
        "Explain how a hash map works, with a small example in Python.",
        "List five things to check before a long road trip, one line each.",
    ];
    let k = 3;
    let (mut steps, mut first, mut runs) = (0usize, 0usize, vec![0usize; k + 1]);
    let mut drafting = 0f64;
    // By how sure the draft is of a guess (tenths), for the first, second and third guess:
    // guesses, those the model's greedy pick, and the model's own probability for them
    // (what a sampler at temperature 1 would take).
    let mut sure = vec![[(0usize, 0usize, 0f64); 10]; k];
    for prompt in prompts {
        let text =
            format!("<|turn>user\n{prompt}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>");
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
        let mut scored = Vec::new();
        // The model's probability, after each token written, for any token.
        let mut chances: Vec<Vec<(u32, f32)>> = Vec::new();
        for _ in 0..120 {
            let started = Instant::now();
            let g = m.mtp_draft_sure(&mut s, pick, k, 0.0).unwrap();
            drafting += started.elapsed().as_secs_f64();
            assert_eq!(g.len(), k, "a guess for each of {k} tokens");
            guesses.push(g.iter().map(|g| g.0).collect::<Vec<u32>>());
            scored.push(g);
            let logits = m.forward(&mut s, pick, true).unwrap().unwrap();
            pick = argmax(logits);
            let top = logits[pick as usize];
            let sum: f32 = logits.iter().map(|v| (v - top).exp()).sum();
            chances.push(
                logits
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| **v > top - 12.0)
                    .map(|(i, v)| (i as u32, (v - top).exp() / sum))
                    .collect(),
            );
            written.push(pick);
        }
        for (i, g) in scored.iter().enumerate() {
            // Guess j is for the token written at i + 1 + j; it's only read after the
            // guesses before it were right.
            for (j, &(token, p)) in g.iter().enumerate() {
                if i + j >= chances.len() || g[..j].iter().zip(&written[i + 1..]).any(|(a, b)| a.0 != *b) {
                    break;
                }
                let cell = &mut sure[j][((p * 10.0) as usize).min(9)];
                cell.0 += 1;
                cell.1 += usize::from(token == written[i + 1 + j]);
                cell.2 += chances[i + j]
                    .iter()
                    .find(|c| c.0 == token)
                    .map_or(0.0, |c| f64::from(c.1));
            }
        }
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
    for (j, cells) in sure.iter().enumerate() {
        let line: Vec<String> = cells
            .iter()
            .enumerate()
            .filter(|(_, c)| c.0 > 0)
            .map(|(b, c)| {
                format!(
                    "{:.1}+: {} ({:.0}% right, {:.0}% sampled)",
                    b as f64 / 10.0,
                    c.0,
                    100.0 * c.1 as f64 / c.0 as f64,
                    100.0 * c.2 / c.0 as f64
                )
            })
            .collect();
        eprintln!("guess {} by how sure the draft is: {}", j + 1, line.join("  "));
    }
    assert!(
        first * 2 > steps,
        "the first guess should be right most of the time"
    );
}

/// Writing with the draft model as the server does (guesses read with the last pick in one
/// pass, the session taken back past a wrong one) against one token at a time: the same
/// text, greedy, and how much faster. Ignored, as above.
#[test]
#[ignore]
fn drafting_writes_the_same_text_faster() {
    let Ok(path) = std::env::var("VIREO_GEMMA_MODEL") else {
        eprintln!("skipped: VIREO_GEMMA_MODEL isn't set");
        return;
    };
    let draft = std::env::var_os("VIREO_GEMMA_ASSISTANT")
        .map(Into::into)
        .or_else(|| assistant::find(Path::new(&path)))
        .expect("no draft model found: set VIREO_GEMMA_ASSISTANT");
    let file = GgufFile::open(&path).unwrap();
    let tokenizer = Tokenizer::from_gguf(&file).unwrap();
    let mut m = Gemma4Cuda::load(&file, Gpu::open().unwrap()).unwrap();
    m.load_assistant(&draft).unwrap();
    let n = 200;
    // Guesses a round at most, and how sure the draft must be of each (0: any guess).
    let ways: Vec<(usize, f32)> = match std::env::var("VIREO_DRAFT_WAYS") {
        Ok(v) => v
            .split(',')
            .filter_map(|w| {
                let (k, floor) = w.split_once('@')?;
                Some((k.parse().ok()?, floor.parse().ok()?))
            })
            .collect(),
        Err(_) => vec![(2, 0.0), (3, 0.0), (4, 0.0), (3, 0.7), (5, 0.7), (7, 0.7)],
    };
    for (k, floor) in ways {
        let mut plain_s = 0f64;
        let mut drafted_s = 0f64;
        let (mut rounds, mut taken, mut guessed) = (0usize, 0usize, 0usize);
        for prompt in [
            "Write a short story about a lighthouse keeper who finds a message in a bottle.",
            "Explain how a hash map works, with a small example in Python.",
        ] {
            let text = format!(
                "<|turn>user\n{prompt}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
            );
            let ids = tokenizer.encode(&text, true, true);
            // One token at a time.
            let mut s = m.session(4096).unwrap();
            let mut pick = argmax(
                m.forward_batch(&mut s, &ids, Logits::Last)
                    .unwrap()
                    .unwrap(),
            );
            let started = Instant::now();
            let mut plain = vec![pick];
            while plain.len() < n {
                pick = argmax(m.forward(&mut s, pick, true).unwrap().unwrap());
                plain.push(pick);
            }
            plain_s += started.elapsed().as_secs_f64();
            // Drafted.
            let mut s = m.session(4096).unwrap();
            m.use_mtp(&mut s).unwrap();
            let mut pick = argmax(
                m.forward_batch(&mut s, &ids, Logits::Last)
                    .unwrap()
                    .unwrap(),
            );
            let started = Instant::now();
            let mut out = vec![pick];
            while out.len() < n {
                let guesses: Vec<u32> = m
                    .mtp_draft_sure(&mut s, pick, k, floor)
                    .unwrap()
                    .into_iter()
                    .map(|g| g.0)
                    .collect();
                guessed += guesses.len();
                let before = s.len();
                let mut batch = vec![pick];
                batch.extend(&guesses);
                let vocab = m.config.vocab;
                let rows = m
                    .forward_batch(&mut s, &batch, Logits::All)
                    .unwrap()
                    .unwrap()
                    .to_vec();
                rounds += 1;
                // Row i's pick follows batch[i]; a guess is kept while it's that pick.
                let mut kept = 0;
                loop {
                    pick = argmax(&rows[kept * vocab..(kept + 1) * vocab]);
                    out.push(pick);
                    if kept < guesses.len() && guesses[kept] == pick {
                        kept += 1;
                    } else {
                        break;
                    }
                }
                taken += kept;
                // Back to the last token kept (the pick after it is read next round).
                assert_eq!(s.truncate(before + 1 + kept), before + 1 + kept);
            }
            drafted_s += started.elapsed().as_secs_f64();
            out.truncate(n);
            assert_eq!(
                tokenizer.decode(&out, false),
                tokenizer.decode(&plain, false),
                "drafting changed the text (k = {k}, floor {floor})"
            );
        }
        eprintln!(
            "up to {k} guesses, sure to {floor}: one at a time {:.1} tokens/s, drafted {:.1} tokens/s ({:.2}x), {:.2} guessed and {:.2} taken a round",
            2.0 * n as f64 / plain_s,
            2.0 * n as f64 / drafted_s,
            plain_s / drafted_s,
            guessed as f64 / rounds as f64,
            taken as f64 / rounds as f64
        );
    }
}

/// What a round costs: a check of the last pick and n guesses (every token's logits), a
/// read of as many for the last token's only, and the draft's guesses. Ignored, as above.
#[test]
#[ignore]
fn what_a_round_costs() {
    let Ok(path) = std::env::var("VIREO_GEMMA_MODEL") else {
        eprintln!("skipped: VIREO_GEMMA_MODEL isn't set");
        return;
    };
    let draft = std::env::var_os("VIREO_GEMMA_ASSISTANT")
        .map(Into::into)
        .or_else(|| assistant::find(Path::new(&path)))
        .expect("no draft model found: set VIREO_GEMMA_ASSISTANT");
    let file = GgufFile::open(&path).unwrap();
    let tokenizer = Tokenizer::from_gguf(&file).unwrap();
    let mut m = Gemma4Cuda::load(&file, Gpu::open().unwrap()).unwrap();
    m.load_assistant(&draft).unwrap();
    let text = "<|turn>user\nExplain how a hash map works, with a small example in Python.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>A hash map stores pairs of keys and values so that a value can be found from its key in about the same time however many pairs there are.";
    let ids = tokenizer.encode(text, true, true);
    let (start, tail) = ids.split_at(ids.len() - 8);
    let mut s = m.session(4096).unwrap();
    m.use_mtp(&mut s).unwrap();
    m.forward_batch(&mut s, start, Logits::Last).unwrap();
    let at = s.len();
    let best = |f: &mut dyn FnMut()| {
        let mut best = f64::MAX;
        for _ in 0..12 {
            let t = Instant::now();
            f();
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    };
    let one = best(&mut || {
        m.forward(&mut s, tail[0], true).unwrap();
        s.truncate(at);
    });
    eprintln!("one token written: {one:.2} ms");
    for n in 1..=5 {
        let all = best(&mut || {
            m.forward_batch(&mut s, &tail[..n], Logits::All).unwrap();
            s.truncate(at);
        });
        let last = best(&mut || {
            m.forward_batch(&mut s, &tail[..n], Logits::Last).unwrap();
            s.truncate(at);
        });
        eprintln!("{n} tokens: every token's logits {all:.2} ms, the last one's {last:.2} ms");
    }
    m.forward(&mut s, tail[0], true).unwrap();
    for k in 1..=4 {
        let d = best(&mut || {
            assert_eq!(m.mtp_draft(&mut s, tail[1], k).unwrap().len(), k);
        });
        eprintln!("{k} guesses drafted: {d:.2} ms");
    }
}
