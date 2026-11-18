//! A fast hash for the engine's lookup tables (the tokenizer's vocabulary and merges).
//!
//! The same multiply-and-rotate hash Firefox and rustc use. It isn't resistant to
//! deliberately chosen keys, which doesn't matter here: the keys come from model files,
//! and lookups happen hundreds of thousands of times per prompt, where std's SipHash
//! is several times slower.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

#[derive(Clone, Copy, Default)]
pub struct FastHasher {
    hash: u64,
}

impl FastHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FastHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let (words, rest) = bytes.as_chunks::<8>();
        for word in words {
            self.add(u64::from_le_bytes(*word));
        }
        let mut tail = [0u8; 8];
        tail[..rest.len()].copy_from_slice(rest);
        if !rest.is_empty() {
            self.add(u64::from_le_bytes(tail));
        }
    }

    #[inline]
    fn write_u8(&mut self, n: u8) {
        self.add(n.into());
    }

    #[inline]
    fn write_u32(&mut self, n: u32) {
        self.add(n.into());
    }

    #[inline]
    fn write_u64(&mut self, n: u64) {
        self.add(n);
    }

    #[inline]
    fn write_usize(&mut self, n: usize) {
        self.add(n as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

pub type FastHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{BuildHasher, Hash};

    fn hash_of(value: impl Hash) -> u64 {
        BuildHasherDefault::<FastHasher>::default().hash_one(value)
    }

    #[test]
    fn different_keys_hash_differently() {
        assert_ne!(hash_of("hello"), hash_of("hellp"));
        assert_ne!(hash_of("ab"), hash_of("ba"));
        assert_ne!(hash_of((1u32, 2u32)), hash_of((2u32, 1u32)));
        assert_eq!(hash_of("same"), hash_of("same"));
    }

    #[test]
    fn works_as_a_map() {
        let mut map: FastHashMap<String, u32> = FastHashMap::default();
        for i in 0..10_000u32 {
            map.insert(format!("token {i}"), i);
        }
        assert_eq!(map.len(), 10_000);
        assert_eq!(map.get("token 1234"), Some(&1234));
    }
}
