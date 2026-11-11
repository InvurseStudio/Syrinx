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
        let logit = |id: u32| {
            penalized
                .iter()
                .find(|&&(p, _)| p == id)
                .map_or(logits[id as usize], |&(_, v)| v)
        };
        if s.temperature <= 0.0 {
            return (0..logits.len() as u32)
                .max_by(|&a, &b| logit(a).total_cmp(&logit(b)))
                .unwrap_or(0);
        }
        let keep = if s.top_k > 0 {
            s.top_k
        } else {
            MOST_CANDIDATES
        }
        .min(logits.len());
        self.candidates.clear();
        self.candidates
            .extend((0..logits.len() as u32).map(|id| (id, logit(id))));
        if keep < self.candidates.len() {
            self.candidates
                .select_nth_unstable_by(keep - 1, |a, b| b.1.total_cmp(&a.1));
            self.candidates.truncate(keep);
        }
        self.candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
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
