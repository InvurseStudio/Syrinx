//! Drafting (speculative decoding): a small model of the same family and vocabulary
//! (Gemma 4 E2B for 31B) guesses the next few tokens, the large model reads them in one
//! batch, and the reply takes each guess for as long as the token the large model's own
//! sampler picks is the guess. A batch of a few tokens costs the large model little more
//! than one token (it reads its weights once either way), so each accepted guess is a
//! token nearly for free. Every token is still the large model's own pick, from its own
//! logits, at any temperature: a guess only saves the pass that would have made them.

use std::time::Instant;

use vireo_engine::cuda::Gpu;
use vireo_engine::models::gemma4_cuda::{Gemma4Cuda, Gemma4CudaSession};
use vireo_engine::models::{Logits, Precision};
use vireo_engine::{GgufFile, Tokenizer};

/// Guesses per round when the command line doesn't say. Gemma 4 31B with E2B on the M5
/// Max, a 234-token answer: 1 guess 17.2 tokens/s, 2 22.5, 3 19.1, against 19.9 without
/// (a round's pass costs 61 ms for 2 tokens, 69 for 3 and 105 for 4, against 54 for 1).
pub const DEFAULT_GUESSES: usize = 2;

pub struct Drafter {
    model: Gemma4Cuda<'static>,
    session: Gemma4CudaSession,
    /// The tokens the session holds, and the logits after the last of them.
    tokens: Vec<u32>,
    logits: Vec<f32>,
    /// Tokens guessed a round.
    pub guesses: usize,
    pub name: String,
    /// This reply's guesses, those taken, and the time spent guessing.
    pub guessed: usize,
    pub taken: usize,
    pub seconds: f64,
}

impl Drafter {
    /// Loads the draft model at `path` for a large model with `tokenizer`, with room for
    /// `context` tokens: only a Gemma 4 with the same vocabulary, token for token.
    pub fn load(
        path: &str,
        tokenizer: &Tokenizer,
        precision: Precision,
        context: usize,
        guesses: usize,
    ) -> Result<Self, String> {
        let file: &'static GgufFile =
            Box::leak(Box::new(GgufFile::open(path).map_err(|e| {
                format!("Couldn't open the draft model {path}: {e}")
            })?));
        if file.architecture() != Some("gemma4") {
            return Err("The draft model must be a Gemma 4, as the model is.".into());
        }
        let theirs = Tokenizer::from_gguf(file).map_err(|e| e.to_string())?;
        let (mut a, mut b) = (Vec::new(), Vec::new());
        let same = theirs.vocab_size() == tokenizer.vocab_size()
            && (0..tokenizer.vocab_size() as u32).all(|id| {
                a.clear();
                b.clear();
                tokenizer.token_bytes(id, true, &mut a);
                theirs.token_bytes(id, true, &mut b);
                a == b
            });
        if !same {
            return Err("The draft model's vocabulary isn't the model's.".into());
        }
        let gpu = Gpu::open().map_err(|e| format!("No GPU for the draft model: {e}"))?;
        let mut model = Gemma4Cuda::load(file, gpu).map_err(|e| e.to_string())?;
        model.set_precision(precision);
        let session = model
            .session(context + guesses + 1)
            .map_err(|e| e.to_string())?;
        let name = std::path::Path::new(path)
            .file_stem()
            .map_or_else(|| path.to_string(), |s| s.to_string_lossy().into_owned());
        Ok(Self {
            model,
            session,
            tokens: Vec::new(),
            logits: Vec::new(),
            guesses: guesses.max(1),
            name,
            guessed: 0,
            taken: 0,
            seconds: 0.0,
        })
    }

    /// Up to `n` guesses at the tokens after `tokens` (the large model's, from the start
    /// of the chat), its likeliest each.
    pub fn guess(&mut self, tokens: &[u32], n: usize) -> Result<Vec<u32>, String> {
        let started = Instant::now();
        let e = |e: vireo_engine::models::Error| e.to_string();
        self.sync(tokens)?;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let next = argmax(&self.logits);
            out.push(next);
            if i + 1 < n {
                let logits = self
                    .model
                    .forward(&mut self.session, next, true)
                    .map_err(e)?
                    .ok_or("no logits")?;
                self.logits.copy_from_slice(logits);
                self.tokens.push(next);
            }
        }
        self.guessed += out.len();
        self.seconds += started.elapsed().as_secs_f64();
        Ok(out)
    }

    /// Reads what's new in `tokens` (the large model's), from where the two part: the
    /// prompt, while the large model reads it, so the first guesses don't wait for it.
    pub fn sync(&mut self, tokens: &[u32]) -> Result<(), String> {
        let e = |e: vireo_engine::models::Error| e.to_string();
        // In step with the large model: back to where the two part, then what's new.
        let shared = self
            .tokens
            .iter()
            .zip(tokens)
            .take_while(|(a, b)| a == b)
            .count();
        if shared < tokens.len() || self.tokens.len() != tokens.len() || self.logits.is_empty() {
            // At least the last token read again, for its logits.
            let back = shared.min(tokens.len().saturating_sub(1));
            let kept = self.session.truncate(back);
            self.tokens.truncate(kept);
            let logits = self
                .model
                .forward_batch(&mut self.session, &tokens[kept..], Logits::Last)
                .map_err(e)?
                .ok_or("no logits")?;
            self.logits = logits.to_vec();
            self.tokens.extend_from_slice(&tokens[kept..]);
        }
        Ok(())
    }

    /// Forgets this reply's counts.
    pub fn start_reply(&mut self) {
        self.guessed = 0;
        self.taken = 0;
        self.seconds = 0.0;
    }
}

fn argmax(logits: &[f32]) -> u32 {
    (0..logits.len())
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
        .unwrap_or(0) as u32
}

/// Guessing from the chat itself ("prompt lookup"), with no draft model: where the chat's
/// last few tokens appeared before, the tokens that followed them there. Replies copy a
/// lot from their chat (paths, names, a tool result's lines, the question's words), and
/// checking a few guesses costs the model about what one token does, so each guess
/// taken is a token nearly for free; one not taken costs little. As with a draft model,
/// every token is still the model's own pick.
pub struct Lookup {
    /// The most tokens guessed a round.
    pub max: usize,
    /// This reply's guesses and those taken.
    pub guessed: usize,
    pub taken: usize,
}

/// Guesses a round from the chat when the command line doesn't say (`--lookup`). Gemma 4
/// E2B on the M5 Max, 200-400-token replies: rewriting a document as a list 217 → 342-358
/// tokens/s, returning an edited code file 217 → 445, a summary 217 → 212-216, a new story
/// 219 → 216-217 (E4B: 125 → 165-196, 125 → 209-269, 125 → 128-131, 125 → 122-124).
pub const DEFAULT_LOOKUP: usize = 3;

/// The longest and shortest runs of the chat's last tokens looked for, longest first.
const LONGEST_RUN: usize = 4;
const SHORTEST_RUN: usize = 2;

impl Lookup {
    pub fn new(max: usize) -> Self {
        Self {
            max,
            guessed: 0,
            taken: 0,
        }
    }

    /// Up to `n` guesses at the tokens after `tokens`: what followed the latest earlier
    /// place the last 4 tokens (or 3, or 2) appeared, at most 3 (2, 1) of them. None when
    /// they haven't.
    pub fn guess(&mut self, tokens: &[u32], n: usize) -> Vec<u32> {
        let out = continuation(tokens, n);
        self.guessed += out.len();
        out
    }
}

fn continuation(tokens: &[u32], n: usize) -> Vec<u32> {
    let len = tokens.len();
    for run in (SHORTEST_RUN..=LONGEST_RUN).rev() {
        if len <= run {
            continue;
        }
        let tail = &tokens[len - run..];
        // The latest earlier run with at least one token after it.
        if let Some(start) = (0..len - run).rev().find(|&s| &tokens[s..s + run] == tail) {
            // Fewer guesses on a shorter match: a run of 2 is often chance (in new prose a
            // guess was taken 2% of the time), and each guess checked costs a little.
            let n = n.min(1 + run - SHORTEST_RUN);
            let from = start + run;
            return tokens[from..len.min(from + n)].to_vec();
        }
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guesses_what_followed_the_latest_match() {
        // ... 7 8 9 10 11 ... 7 8 9 → 10 11 (the run of 3 found, its latest place).
        let tokens = [1, 7, 8, 9, 5, 6, 2, 7, 8, 9, 10, 11, 12, 3, 7, 8, 9];
        assert_eq!(continuation(&tokens, 2), vec![10, 11]);
        // A run of 4: up to 3 guesses, and no further than what's there.
        let tokens = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4];
        assert_eq!(continuation(&tokens, 8), vec![5, 6, 1]);
        // A run of 2: one guess.
        let tokens = [4, 5, 6, 4, 5];
        assert_eq!(continuation(&tokens, 5), vec![6]);
        // A single token's match isn't enough.
        assert!(continuation(&[1, 2, 3, 9, 3], 3).is_empty());
        assert!(continuation(&[], 3).is_empty());
    }
}
