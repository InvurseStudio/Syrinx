//! Byte-pair merging: turns one word's byte tokens into the vocabulary's tokens.
//!
//! The vocabulary's merge list is ranked: the earlier a merge, the more common the pair
//! was in training. Starting from one token per byte, the pair of neighbours with the
//! best-ranked merge is joined, again and again, until no neighbouring pair has a merge.
//! When two pairs have the same rank, the one further left goes first, as llama.cpp
//! does.
//!
//! The symbols are a linked list and the candidate pairs a heap, so a word of n bytes
//! takes O(n log n), not O(n²): this matters for long runs of letters or punctuation.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::hash::FastHashMap;

/// For each mergeable pair of token ids: the merge's rank and the token it makes.
pub(super) type Merges = FastHashMap<(u32, u32), Merge>;

#[derive(Clone, Copy, Debug)]
pub(super) struct Merge {
    pub rank: u32,
    pub token: u32,
}

const NONE: usize = usize::MAX;

/// Merges `tokens` (one word's tokens, usually one per byte) in place.
pub(super) fn merge(tokens: &mut Vec<u32>, merges: &Merges) {
    let n = tokens.len();
    if n < 2 {
        return;
    }
    let mut prev: Vec<usize> = (0..n).map(|i| i.wrapping_sub(1)).collect();
    prev[0] = NONE;
    let mut next: Vec<usize> = (1..=n).collect();
    next[n - 1] = NONE;
    let mut alive = vec![true; n];

    // (rank, left symbol, left token, right token): popped lowest rank first, then
    // leftmost. An entry is stale once either symbol has changed.
    let mut heap = BinaryHeap::new();
    let consider = |heap: &mut BinaryHeap<_>, left: usize, l: u32, r: u32| {
        if let Some(m) = merges.get(&(l, r)) {
            heap.push(Reverse((m.rank, left, l, r)));
        }
    };
    for i in 0..n - 1 {
        consider(&mut heap, i, tokens[i], tokens[i + 1]);
    }

    while let Some(Reverse((_, left, l, r))) = heap.pop() {
        let right = next[left];
        if !alive[left] || tokens[left] != l || right == NONE || tokens[right] != r {
            continue;
        }
        tokens[left] = merges[&(l, r)].token;
        alive[right] = false;
        next[left] = next[right];
        if next[right] != NONE {
            prev[next[right]] = left;
        }
        if prev[left] != NONE {
            let p = prev[left];
            consider(&mut heap, p, tokens[p], tokens[left]);
        }
        if next[left] != NONE {
            let q = next[left];
            consider(&mut heap, left, tokens[left], tokens[q]);
        }
    }

    let mut i = 0;
    let mut out = 0;
    while i != NONE {
        tokens[out] = tokens[i];
        out += 1;
        i = next[i];
    }
    tokens.truncate(out);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tokens: 0 'a', 1 'b', 2 'c', 3 "ab", 4 "bc", 5 "abc", 6 "aa", 7 "aaaa".
    fn merges(list: &[(u32, u32, u32)]) -> Merges {
        list.iter()
            .enumerate()
            .map(|(rank, &(l, r, token))| {
                (
                    (l, r),
                    Merge {
                        rank: rank as u32,
                        token,
                    },
                )
            })
            .collect()
    }

    fn run(tokens: &[u32], m: &Merges) -> Vec<u32> {
        let mut t = tokens.to_vec();
        merge(&mut t, m);
        t
    }

    #[test]
    fn best_rank_goes_first() {
        // "bc" ranks above "ab", so a·b·c becomes a·bc, and "abc" from "ab"+"c" never fits.
        let m = merges(&[(1, 2, 4), (0, 1, 3), (3, 2, 5)]);
        assert_eq!(run(&[0, 1, 2], &m), [0, 4]);
        // With "ab" first, a·b·c becomes ab·c and then abc.
        let m = merges(&[(0, 1, 3), (1, 2, 4), (3, 2, 5)]);
        assert_eq!(run(&[0, 1, 2], &m), [5]);
    }

    #[test]
    fn ties_go_left_first() {
        // a·a·a: both pairs are "aa"; the left one merges, leaving aa·a.
        let m = merges(&[(0, 0, 6)]);
        assert_eq!(run(&[0, 0, 0], &m), [6, 0]);
        // a·a·a·a → aa·aa → aaaa.
        let m = merges(&[(0, 0, 6), (6, 6, 7)]);
        assert_eq!(run(&[0, 0, 0, 0], &m), [7]);
    }

    #[test]
    fn short_and_unmergeable_words() {
        let m = merges(&[(0, 1, 3)]);
        assert_eq!(run(&[], &m), Vec::<u32>::new());
        assert_eq!(run(&[2], &m), [2]);
        assert_eq!(run(&[2, 2, 1, 0], &m), [2, 2, 1, 0]);
    }

    #[test]
    fn long_words_are_fast() {
        let m = merges(&[(0, 0, 6), (6, 6, 7)]);
        let mut t = vec![0; 200_000];
        merge(&mut t, &m);
        assert_eq!(t, vec![7; 50_000]);
    }
}
