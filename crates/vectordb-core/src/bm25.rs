//! Fast In-Memory BM25 (Best Matching 25) Inverted Index for Exact Keyword Search.
//! Combined with HNSW Vector Search via Reciprocal Rank Fusion (RRF).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::ChunkId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bm25Index {
    /// Inverted index: term -> Vec<(chunk_id, term_frequency)>
    inverted_index: HashMap<String, Vec<(ChunkId, f32)>>,
    /// Document lengths (total tokens per chunk)
    doc_lengths: HashMap<ChunkId, f32>,
    /// Average document length across all chunks
    avg_doc_length: f32,
    /// Total number of indexed chunks
    total_docs: usize,
    /// BM25 hyperparameter k1 (term saturation, default: 1.2)
    k1: f32,
    /// BM25 hyperparameter b (document length penalty, default: 0.75)
    b: f32,
}

impl Default for Bm25Index {
    fn default() -> Self {
        Self::new(1.2, 0.75)
    }
}

impl Bm25Index {
    pub fn new(k1: f32, b: f32) -> Self {
        Self {
            inverted_index: HashMap::new(),
            doc_lengths: HashMap::new(),
            avg_doc_length: 0.0,
            total_docs: 0,
            k1,
            b,
        }
    }

    /// Tokenize text into normalized terms for keyword indexing.
    /// Extracts both full identifier tokens and snake_case / camelCase subwords.
    pub fn tokenize(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut cur = String::new();

        for ch in text.chars() {
            if ch.is_alphanumeric() || ch == '_' {
                cur.push(ch);
            } else {
                if !cur.is_empty() {
                    Self::extract_subtokens(&cur, &mut tokens);
                    cur.clear();
                }
            }
        }
        if !cur.is_empty() {
            Self::extract_subtokens(&cur, &mut tokens);
        }
        tokens
    }

    fn extract_subtokens(word: &str, out: &mut Vec<String>) {
        let lower = word.to_lowercase();
        out.push(lower);

        // Split snake_case
        if word.contains('_') {
            for part in word.split('_') {
                if !part.is_empty() {
                    let p = part.to_lowercase();
                    if !out.contains(&p) {
                        out.push(p);
                    }
                }
            }
        }

        // Split camelCase / PascalCase
        let chars: Vec<char> = word.chars().collect();
        let mut part = String::new();
        for i in 0..chars.len() {
            let c = chars[i];
            if c == '_' {
                if !part.is_empty() {
                    let p = part.to_lowercase();
                    if !out.contains(&p) {
                        out.push(p);
                    }
                    part.clear();
                }
                continue;
            }
            if c.is_uppercase() && !part.is_empty() {
                let prev_lower = chars[i - 1].is_lowercase();
                let next_lower = (i + 1 < chars.len()) && chars[i + 1].is_lowercase();
                if prev_lower || next_lower {
                    let p = part.to_lowercase();
                    if !out.contains(&p) {
                        out.push(p);
                    }
                    part.clear();
                }
            }
            part.push(c);
        }
        if !part.is_empty() {
            let p = part.to_lowercase();
            if !out.contains(&p) {
                out.push(p);
            }
        }
    }

    /// Index a document/chunk.
    pub fn add_document(&mut self, id: ChunkId, text: &str) {
        self.remove_document(&id);
        let tokens = Self::tokenize(text);
        let doc_len = tokens.len() as f32;
        if doc_len == 0.0 {
            return;
        }

        let mut term_freqs: HashMap<String, f32> = HashMap::new();
        for t in tokens {
            *term_freqs.entry(t).or_insert(0.0) += 1.0;
        }

        for (term, tf) in term_freqs {
            self.inverted_index
                .entry(term)
                .or_default()
                .push((id.clone(), tf));
        }

        let previous_total_tokens = self.avg_doc_length * self.total_docs as f32;
        self.doc_lengths.insert(id, doc_len);
        self.total_docs += 1;
        self.avg_doc_length = (previous_total_tokens + doc_len) / self.total_docs as f32;
    }

    /// Remove a document by ID (for incremental updates)
    pub fn remove_document(&mut self, id: &str) {
        if let Some(doc_len) = self.doc_lengths.remove(id) {
            let previous_total_tokens = self.avg_doc_length * self.total_docs as f32;
            self.total_docs = self.total_docs.saturating_sub(1);
            for postings in self.inverted_index.values_mut() {
                postings.retain(|(doc_id, _)| doc_id != id);
            }
            if self.total_docs > 0 {
                self.avg_doc_length =
                    (previous_total_tokens - doc_len).max(0.0) / self.total_docs as f32;
            } else {
                self.avg_doc_length = 0.0;
            }
        }
    }

    /// Search the BM25 index with a query string.
    /// Returns sorted (ChunkId, BM25 Score) descending.
    pub fn search(&self, query: &str, limit: usize) -> Vec<(ChunkId, f32)> {
        let query_tokens = Self::tokenize(query);
        if query_tokens.is_empty() || self.total_docs == 0 {
            return Vec::new();
        }

        let mut scores: HashMap<ChunkId, f32> = HashMap::new();
        let n = self.total_docs as f32;
        let avg_dl = self.avg_doc_length.max(1.0);

        for term in query_tokens {
            if let Some(postings) = self.inverted_index.get(&term) {
                let df = postings.len() as f32;
                // Robertson-Spärck Jones IDF
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

                for (doc_id, tf) in postings {
                    let doc_len = *self.doc_lengths.get(doc_id).unwrap_or(&avg_dl);
                    let denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_dl));
                    let term_score = idf * (tf * (self.k1 + 1.0)) / denom;

                    *scores.entry(doc_id.clone()).or_insert(0.0) += term_score;
                }
            }
        }

        let mut results: Vec<(ChunkId, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }
}

/// Combine Dense HNSW results and Sparse BM25 results using Reciprocal Rank Fusion (RRF).
/// RRF score = sum( 1.0 / (k_rrf + rank_i) )
pub fn reciprocal_rank_fusion(
    dense_ranked_ids: &[ChunkId],
    sparse_ranked_ids: &[ChunkId],
    k_rrf: f32,
) -> Vec<(ChunkId, f32)> {
    reciprocal_rank_fusion_many(&[dense_ranked_ids, sparse_ranked_ids], k_rrf)
}

/// Fuse independent ranked lists without comparing their raw similarity scores.
pub fn reciprocal_rank_fusion_many(ranked_lists: &[&[ChunkId]], k_rrf: f32) -> Vec<(ChunkId, f32)> {
    let mut rrf_scores: HashMap<ChunkId, (f32, usize)> = HashMap::new();
    for (list_index, ranked_ids) in ranked_lists.iter().enumerate() {
        for (rank, id) in ranked_ids.iter().enumerate() {
            let score = 1.0 / (k_rrf + rank as f32 + 1.0);
            let entry = rrf_scores.entry(id.clone()).or_insert((0.0, list_index));
            entry.0 += score;
            entry.1 = entry.1.min(list_index);
        }
    }

    let mut combined: Vec<_> = rrf_scores.into_iter().collect();
    combined.sort_by(|a, b| {
        b.1 .0
            .total_cmp(&a.1 .0)
            .then_with(|| a.1 .1.cmp(&b.1 .1))
            .then_with(|| a.0.cmp(&b.0))
    });
    combined
        .into_iter()
        .map(|(id, (score, _))| (id, score))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_exact_matching() {
        let mut index = Bm25Index::default();
        index.add_document(
            "doc1".into(),
            "pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32",
        );
        index.add_document(
            "doc2".into(),
            "error[E0502]: cannot borrow *self as mutable",
        );
        index.add_document(
            "doc3".into(),
            "struct UserAccount { username: String, email: String }",
        );

        let res = index.search("E0502", 5);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, "doc2");

        let res_sim = index.search("cosine similarity", 5);
        assert_eq!(res_sim[0].0, "doc1");
    }

    #[test]
    fn test_rrf_fusion() {
        let dense = vec!["doc1".to_string(), "doc2".to_string(), "doc3".to_string()];
        let sparse = vec!["doc2".to_string(), "doc1".to_string(), "doc4".to_string()];

        let combined = reciprocal_rank_fusion(&dense, &sparse, 60.0);
        // doc1 and doc2 are in both, so they should be top
        assert!(combined[0].0 == "doc1" || combined[0].0 == "doc2");
    }
}
