//! Picking each next token: repetition penalty, temperature, top-k, min-p and top-p, as
//! the app's settings name them.

use crate::json::Value;

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub temperature: f32,
    pub top_p: f32,
    /// 0 for no limit.
    pub top_k: usize,
    pub min_p: f32,
    pub repetition_penalty: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
        }
    }
}

impl Settings {
    /// The request's `generation` settings over these (the app sends only what the
    /// person changed; nothing means the model's own).
    pub fn with(mut self, generation: Option<&Value>) -> Self {
        let Some(g) = generation else { return self };
        if let Some(v) = g.num("temperature") {
            self.temperature = v as f32;
        }
        if let Some(v) = g.num("top_p") {
            self.top_p = v as f32;
        }
        if let Some(v) = g.num("top_k") {
            self.top_k = v.max(0.0) as usize;
        }
        if let Some(v) = g.num("min_p") {
            self.min_p = v as f32;
        }
        if let Some(v) = g.num("repetition_penalty") {
            self.repetition_penalty = v as f32;
        }
        self
    }
}

/// Without top-k, only this many likeliest tokens are kept (the rest of a 262,144-token
/// vocabulary carries next to nothing after top-p, and sorting it every token is slow).
const MOST_CANDIDATES: usize = 1024;
/// Tokens the repetition penalty looks back over.
const PENALTY_WINDOW: usize = 64;

pub struct Sampler {
    settings: Settings,
    state: u64,
    candidates: Vec<(u32, f32)>,
}

impl Sampler {
    pub fn new(settings: Settings, seed: u64) -> Self {
        Self {
            settings,
            state: seed | 1,
            candidates: Vec::new(),
        }
    }

    fn random(&mut self) -> f32 {
        // xorshift64*
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let x = self.state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (x >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn pick(&mut self, logits: &[f32], recent: &[u32]) -> u32 {
        let s = self.settings;
        // The recent tokens' penalized logits (only these differ from `logits`).
        let mut penalized: Vec<(u32, f32)> = Vec::new();
        if s.repetition_penalty != 1.0 {
            let start = recent.len().saturating_sub(PENALTY_WINDOW);
            for &id in &recent[start..] {
                let l = logits[id as usize];
                let v = if l > 0.0 {
                    l / s.repetition_penalty
                } else {
                    l * s.repetition_penalty
                };
                if !penalized.iter().any(|&(p, _)| p == id) {
                    penalized.push((id, v));
                }
            }
        }
        let keep = if s.temperature <= 0.0 {
            1
        } else if s.top_k > 0 {
            s.top_k
        } else {
            MOST_CANDIDATES
        }
        .min(logits.len());
        // The likeliest `keep` by the raw logits, plus as many more as there are penalized
        // tokens (each may drop out); then the penalized ones swapped for their new values.
        top_n(logits, keep + penalized.len(), &mut self.candidates);
        if !penalized.is_empty() {
            self.candidates
                .retain(|c| !penalized.iter().any(|&(p, _)| p == c.0));
            self.candidates
                .extend(penalized.iter().filter(|p| p.1 > f32::NEG_INFINITY));
        }
        if self.candidates.is_empty() {
            return 0;
        }
        if keep < self.candidates.len() {
            self.candidates
                .select_nth_unstable_by(keep - 1, |a, b| b.1.total_cmp(&a.1));
            self.candidates.truncate(keep);
        }
        self.candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        if s.temperature <= 0.0 {
            return self.candidates[0].0;
        }
        let top = self.candidates[0].1;
        let mut total = 0.0f32;
        for c in &mut self.candidates {
            c.1 = ((c.1 - top) / s.temperature).exp();
            total += c.1;
        }
        // min-p: drop what's under min_p of the likeliest; top-p: keep the likeliest up to p.
        let floor = s.min_p * self.candidates[0].1;
        let mut kept = 0;
        let mut mass = 0.0f32;
        for c in &self.candidates {
            if kept > 0 && (c.1 < floor || mass >= s.top_p * total) {
                break;
            }
            mass += c.1;
            kept += 1;
        }
        let mut r = self.random() * mass;
        for c in &self.candidates[..kept] {
            r -= c.1;
            if r <= 0.0 {
                return c.0;
            }
        }
        self.candidates[kept - 1].0
    }
}

/// The `n` largest of `values` (by value, in no order) into `out`, as (index, value), in
/// one pass: only values over the smallest kept so far are taken, and the list is cut back
/// to `n` whenever it doubles. Infinitely small and NaN values are never taken.
fn top_n(values: &[f32], n: usize, out: &mut Vec<(u32, f32)>) {
    out.clear();
    if n == 0 {
        return;
    }
    let cap = (2 * n).max(n + 256);
    let mut floor = f32::NEG_INFINITY;
    for (chunk_at, chunk) in values.chunks(64).enumerate() {
        // Most chunks hold nothing over the floor: a branch-free check first.
        if !chunk.iter().fold(false, |any, &v| any | (v > floor)) {
            continue;
        }
        for (i, &v) in chunk.iter().enumerate() {
            if v > floor {
                out.push(((chunk_at * 64 + i) as u32, v));
                if out.len() >= cap {
                    out.select_nth_unstable_by(n - 1, |a, b| b.1.total_cmp(&a.1));
                    out.truncate(n);
                    floor = out[n - 1].1;
                }
            }
        }
    }
    if out.len() > n {
        out.select_nth_unstable_by(n - 1, |a, b| b.1.total_cmp(&a.1));
        out.truncate(n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_at_zero_temperature_and_spread_otherwise() {
        let logits = [0.0, 5.0, 4.9, -3.0];
        let mut greedy = Sampler::new(
            Settings {
                temperature: 0.0,
                ..Settings::default()
            },
            7,
        );
        assert_eq!(greedy.pick(&logits, &[]), 1);
        let mut s = Sampler::new(Settings::default(), 7);
        let picks: Vec<u32> = (0..200).map(|_| s.pick(&logits, &[])).collect();
        assert!(picks.contains(&1) && picks.contains(&2));
        // top-k 1 is greedy too.
        let mut k1 = Sampler::new(
            Settings {
                top_k: 1,
                ..Settings::default()
            },
            7,
        );
        assert!((0..50).all(|_| k1.pick(&logits, &[]) == 1));
    }

    #[test]
    fn repetition_penalty_pushes_repeats_down() {
        let logits = [0.0, 5.0, 4.9];
        let mut s = Sampler::new(
            Settings {
                temperature: 0.0,
                repetition_penalty: 1.5,
                ..Settings::default()
            },
            1,
        );
        assert_eq!(s.pick(&logits, &[1]), 2);
    }
}

#[cfg(test)]
mod top_n_tests {
    use super::*;

    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % 1_000_000) as f32 / 50_000.0 - 10.0
            })
            .collect()
    }

    #[test]
    fn top_n_matches_a_full_sort() {
        for (len, n, seed) in [
            (10, 3, 1),
            (1000, 1, 2),
            (5000, 64, 3),
            (262_144, 1024, 4),
            (300, 300, 5),
        ] {
            let mut values = noise(len, seed);
            values[len / 2] = f32::NEG_INFINITY;
            let mut got = Vec::new();
            top_n(&values, n, &mut got);
            let mut want: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
            want.sort_by(|a, b| b.total_cmp(a));
            want.truncate(n);
            let mut got_values: Vec<f32> = got.iter().map(|c| c.1).collect();
            got_values.sort_by(|a, b| b.total_cmp(a));
            assert_eq!(got_values, want, "len {len}, n {n}");
            assert!(got.iter().all(|&(i, v)| values[i as usize] == v));
        }
    }

    #[test]
    fn penalized_greedy_matches_a_full_scan() {
        let logits = noise(50_000, 9);
        let mut order: Vec<u32> = (0..logits.len() as u32).collect();
        order.sort_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
        // The 40 likeliest were just said, and are pushed down.
        let recent: Vec<u32> = order[..40].to_vec();
        for penalty in [1.5f32, 0.7] {
            let adjusted = |id: u32| {
                let l = logits[id as usize];
                if !recent.contains(&id) {
                    l
                } else if l > 0.0 {
                    l / penalty
                } else {
                    l * penalty
                }
            };
            let want = (0..logits.len() as u32)
                .max_by(|&a, &b| adjusted(a).total_cmp(&adjusted(b)))
                .unwrap();
            let mut s = Sampler::new(
                Settings {
                    temperature: 0.0,
                    repetition_penalty: penalty,
                    ..Settings::default()
                },
                1,
            );
            assert_eq!(s.pick(&logits, &recent), want, "penalty {penalty}");
        }
    }
}

#[cfg(test)]
mod timing {
    use super::*;

    /// Time a pick over a 262,144-token vocabulary. Prints; run with --ignored.
    #[test]
    #[ignore]
    fn time_pick() {
        let mut state = 12345u64;
        let logits: Vec<f32> = (0..262_144)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % 10_000) as f32 / 500.0 - 10.0
            })
            .collect();
        let recent: Vec<u32> = (0..200).map(|i| i * 37).collect();
        for (label, settings) in [
            (
                "greedy",
                Settings {
                    temperature: 0.0,
                    ..Settings::default()
                },
            ),
            (
                "gemma",
                Settings {
                    temperature: 1.0,
                    top_p: 0.95,
                    top_k: 64,
                    ..Settings::default()
                },
            ),
            (
                "qwen",
                Settings {
                    temperature: 0.6,
                    top_p: 0.95,
                    top_k: 20,
                    ..Settings::default()
                },
            ),
            ("no top-k", Settings::default()),
            (
                "gemma + penalty 1.1",
                Settings {
                    temperature: 1.0,
                    top_p: 0.95,
                    top_k: 64,
                    repetition_penalty: 1.1,
                    ..Settings::default()
                },
            ),
        ] {
            let mut s = Sampler::new(settings, 7);
            s.pick(&logits, &recent);
            let started = std::time::Instant::now();
            for _ in 0..100 {
                std::hint::black_box(s.pick(&logits, &recent));
            }
            eprintln!(
                "{label}: {:.0} µs a pick",
                started.elapsed().as_secs_f64() * 1e4
            );
        }
    }
}
