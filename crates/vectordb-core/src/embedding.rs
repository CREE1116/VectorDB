//! Embedding generation for vector representations.
//! Provides a fast, deterministic local feature hasher (Zero-dependency, instant startup)
//! and pluggable traits for remote/neural embedding providers.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::simd::normalize_in_place;

pub trait EmbeddingProvider: Send + Sync {
    fn dim(&self) -> usize;
    fn embed(&self, text: &str) -> Vec<f32>;
    fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}

/// Fast, deterministic subword and token n-gram feature hasher.
/// Produces dense, normalized vector embeddings without requiring heavy neural model downloads.
#[derive(Debug, Clone)]
pub struct FastFeatureEmbedder {
    dim: usize,
}

impl FastFeatureEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    #[inline]
    fn hash_feature(feat: &str, seed: u64) -> (usize, f32) {
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        feat.hash(&mut hasher);
        let h = hasher.finish();

        let index = (h >> 1) as usize;
        let sign = if (h & 1) == 0 { 1.0f32 } else { -1.0f32 };
        (index, sign)
    }

    /// Tokenize text by splitting on punctuation, whitespace, and camelCase boundaries.
    fn tokenize(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current_token = String::new();

        let chars: Vec<char> = text.chars().collect();
        for i in 0..chars.len() {
            let c = chars[i];
            if c.is_alphanumeric() || c == '_' {
                // Check camelCase split
                if c.is_uppercase() && !current_token.is_empty() {
                    let prev = chars[i - 1];
                    let next_is_lower = (i + 1 < chars.len()) && chars[i + 1].is_lowercase();
                    if prev.is_lowercase() || next_is_lower {
                        tokens.push(current_token.to_lowercase());
                        current_token = String::new();
                    }
                }
                current_token.push(c);
            } else {
                if !current_token.is_empty() {
                    tokens.push(current_token.to_lowercase());
                    current_token = String::new();
                }
            }
        }
        if !current_token.is_empty() {
            tokens.push(current_token.to_lowercase());
        }

        tokens
    }
}

impl Default for FastFeatureEmbedder {
    fn default() -> Self {
        Self::new(256)
    }
}

impl EmbeddingProvider for FastFeatureEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let mut vec = vec![0.0f32; self.dim];
        let tokens = Self::tokenize(text);
        if tokens.is_empty() {
            return vec;
        }

        let mut term_counts: HashMap<String, f32> = HashMap::new();

        // 1. Unigrams
        for token in &tokens {
            *term_counts.entry(token.clone()).or_insert(0.0) += 1.0;

            // Character 3-grams for subword similarity (e.g. "vector" -> "vec", "ect", "cto", "tor")
            let chars: Vec<char> = token.chars().collect();
            if chars.len() >= 3 {
                for window in chars.windows(3) {
                    let trigram: String = window.iter().collect();
                    *term_counts.entry(format!("c3_{}", trigram)).or_insert(0.0) += 0.5;
                }
            }
        }

        // 2. Bigrams (word-level)
        for window in tokens.windows(2) {
            let bigram = format!("{}_{}", window[0], window[1]);
            *term_counts.entry(bigram).or_insert(0.0) += 1.5;
        }

        // 3. Project into dense vector using signed feature hashing
        for (term, count) in term_counts {
            // sublinear tf scaling
            let weight = 1.0 + count.ln();
            let (idx, sign) = Self::hash_feature(&term, 0x9e3779b97f4a7c15);
            let target_dim = idx % self.dim;
            vec[target_dim] += sign * weight;
        }

        // 4. L2 Normalize using SIMD
        normalize_in_place(&mut vec);

        vec
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::cosine_similarity;

    #[test]
    fn test_fast_embedder_similarity() {
        let embedder = FastFeatureEmbedder::new(256);

        let v1 = embedder.embed("fn calculate_cosine_distance(a: &[f32], b: &[f32]) -> f32");
        let v2 =
            embedder.embed("pub fn cosine_distance(vector_a: &[f32], vector_b: &[f32]) -> f32");
        let v3 = embedder.embed("struct UserAccount { username: String, email: String }");

        let sim_similar = cosine_similarity(&v1, &v2);
        let sim_different = cosine_similarity(&v1, &v3);

        assert!(
            sim_similar > sim_different,
            "Expected similar code {sim_similar} > different code {sim_different}"
        );
    }
}
