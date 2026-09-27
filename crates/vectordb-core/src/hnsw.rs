//! Hierarchical Navigable Small World (HNSW) graph index.
//! Optimized for high-throughput, low-latency approximate nearest neighbor (ANN) search.

use parking_lot::RwLock;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use crate::simd::{compute_distance, DistanceMetric};
use crate::types::{SearchResult, VectorId};

#[derive(Debug, Clone, Copy)]
struct DistNode {
    dist: f32,
    id: VectorId,
}

impl PartialEq for DistNode {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for DistNode {}

impl PartialOrd for DistNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.id.cmp(&other.id))
    }
}

#[derive(Debug, Clone, Copy)]
struct MinDistNode {
    dist: f32,
    id: VectorId,
}

impl PartialEq for MinDistNode {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for MinDistNode {}

impl PartialOrd for MinDistNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinDistNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .dist
            .total_cmp(&self.dist)
            .then_with(|| self.id.cmp(&other.id))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    pub m: usize,
    pub m0: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub metric: DistanceMetric,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 64,
            ef_search: 40,
            metric: DistanceMetric::Cosine,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HnswNode {
    id: VectorId,
    level: usize,
    /// neighbors[layer] = Vec<VectorId>
    neighbors: Vec<Vec<VectorId>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HnswIndex {
    config: HnswConfig,
    dim: usize,
    ml: f64,
    entry_point: Option<VectorId>,
    max_level: usize,
    nodes: Vec<Option<HnswNode>>,
    vectors: Vec<Option<Vec<f32>>>,
    count: usize,
    #[serde(default)]
    seed_state: Option<u64>,
}

pub struct HnswIndexThreadSafe {
    inner: RwLock<HnswIndex>,
}

impl HnswIndexThreadSafe {
    pub fn new(dim: usize, config: HnswConfig) -> Self {
        Self {
            inner: RwLock::new(HnswIndex::new(dim, config)),
        }
    }

    pub fn insert(&self, id: VectorId, vector: Vec<f32>) {
        self.inner.write().insert(id, vector);
    }

    /// Rebuild after removals; HNSW cannot safely leave deleted nodes in search paths.
    pub fn rebuild(&self, vectors: &[(VectorId, Vec<f32>)]) {
        let mut inner = self.inner.write();
        let dim = inner.dim;
        let config = inner.config.clone();
        *inner = HnswIndex::new(dim, config);
        for (id, vector) in vectors {
            inner.insert(*id, vector.clone());
        }
    }

    pub fn search(&self, query: &[f32], k: usize, ef_search: Option<usize>) -> Vec<SearchResult> {
        self.inner.read().search(query, k, ef_search)
    }

    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.read().is_empty()
    }

    pub fn get_vector(&self, id: VectorId) -> Option<Vec<f32>> {
        self.inner.read().get_vector(id).cloned()
    }

    pub fn contains_vector(&self, id: VectorId) -> bool {
        self.inner.read().get_vector(id).is_some()
    }

    pub fn snapshot_bytes(&self) -> anyhow::Result<Vec<u8>> {
        Ok(bincode::serialize(&*self.inner.read())?)
    }

    pub fn from_snapshot_bytes(bytes: &[u8], expected_dim: usize) -> anyhow::Result<Self> {
        let index: HnswIndex = bincode::deserialize(bytes)?;
        anyhow::ensure!(
            index.dim == expected_dim,
            "HNSW snapshot dimension mismatch"
        );
        anyhow::ensure!(
            index.count <= index.nodes.len() && index.count <= index.vectors.len(),
            "invalid HNSW snapshot count"
        );
        Ok(Self {
            inner: RwLock::new(index),
        })
    }
}

impl HnswIndex {
    pub fn new(dim: usize, config: HnswConfig) -> Self {
        let ml = 1.0 / (config.m as f64).ln().max(1e-6);
        Self {
            config,
            dim,
            ml,
            entry_point: None,
            max_level: 0,
            nodes: Vec::new(),
            vectors: Vec::new(),
            count: 0,
            seed_state: None,
        }
    }

    /// Deterministic graph construction for benchmarks and reproducible tests.
    pub fn new_seeded(dim: usize, config: HnswConfig, seed: u64) -> Self {
        let mut index = Self::new(dim, config);
        index.seed_state = Some(seed.max(1));
        index
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn get_vector(&self, id: VectorId) -> Option<&Vec<f32>> {
        let idx = id as usize;
        if idx < self.vectors.len() {
            self.vectors[idx].as_ref()
        } else {
            None
        }
    }

    fn sample_level(&mut self) -> usize {
        let r: f64 = if let Some(state) = &mut self.seed_state {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            (*state as f64 / u64::MAX as f64).max(1e-9)
        } else {
            rand::thread_rng().gen::<f64>().max(1e-9)
        };
        (-r.ln() * self.ml).floor() as usize
    }

    #[inline]
    fn dist(&self, v1: &[f32], id2: VectorId) -> f32 {
        let v2 = self.vectors[id2 as usize]
            .as_ref()
            .expect("Vector must exist");
        compute_distance(v1, v2, self.config.metric)
    }

    #[inline]
    pub fn dist_between(&self, id1: VectorId, id2: VectorId) -> f32 {
        let v1 = self.vectors[id1 as usize]
            .as_ref()
            .expect("Vector 1 must exist");
        let v2 = self.vectors[id2 as usize]
            .as_ref()
            .expect("Vector 2 must exist");
        compute_distance(v1, v2, self.config.metric)
    }

    pub fn insert(&mut self, id: VectorId, vector: Vec<f32>) {
        assert_eq!(vector.len(), self.dim, "Vector dimension mismatch");

        let idx = id as usize;
        if idx >= self.nodes.len() {
            self.nodes.resize_with(idx + 1, || None);
            self.vectors.resize_with(idx + 1, || None);
        }

        let node_level = self.sample_level();

        let mut node = HnswNode {
            id,
            level: node_level,
            neighbors: vec![Vec::new(); node_level + 1],
        };

        self.vectors[idx] = Some(vector.clone());

        if self.entry_point.is_none() {
            self.entry_point = Some(id);
            self.max_level = node_level;
            self.nodes[idx] = Some(node);
            self.count += 1;
            return;
        }

        let mut curr_ep = self.entry_point.unwrap();
        let mut curr_dist = self.dist(&vector, curr_ep);
        let top_level = self.max_level;

        // 1. Traverse down to node_level using greedy search (ef = 1)
        for lc in (node_level + 1..=top_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                if let Some(Some(ep_node)) = self.nodes.get(curr_ep as usize) {
                    if lc < ep_node.neighbors.len() {
                        for &neighbor in &ep_node.neighbors[lc] {
                            let d = self.dist(&vector, neighbor);
                            if d < curr_dist {
                                curr_dist = d;
                                curr_ep = neighbor;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        // 2. Insert into layers from min(top_level, node_level) down to 0
        let mut ep_candidates = vec![curr_ep];
        for lc in (0..=std::cmp::min(top_level, node_level)).rev() {
            let m_max = if lc == 0 {
                self.config.m0
            } else {
                self.config.m
            };
            let candidates = self.search_layer_internal(
                &vector,
                &ep_candidates,
                self.config.ef_construction,
                lc,
            );

            // Select neighbors
            let selected_neighbors = self.select_neighbors(&vector, &candidates, m_max);
            node.neighbors[lc] = selected_neighbors.clone();

            // Connect symmetrically
            for &neighbor in &selected_neighbors {
                let neighbor_idx = neighbor as usize;
                let should_prune =
                    if let Some(Some(ref mut n_node)) = self.nodes.get_mut(neighbor_idx) {
                        if lc < n_node.neighbors.len() {
                            n_node.neighbors[lc].push(id);
                            n_node.neighbors[lc].len() > m_max
                        } else {
                            false
                        }
                    } else {
                        false
                    };

                if should_prune {
                    let n_vec = self.vectors[neighbor_idx].as_ref().unwrap().clone();
                    let current_neighbors =
                        self.nodes[neighbor_idx].as_ref().unwrap().neighbors[lc].clone();
                    let pruned = self.select_neighbors_by_ids(&n_vec, &current_neighbors, m_max);
                    self.nodes[neighbor_idx].as_mut().unwrap().neighbors[lc] = pruned;
                }
            }

            ep_candidates = candidates.into_iter().map(|item| item.id).collect();
        }

        if node_level > self.max_level {
            self.max_level = node_level;
            self.entry_point = Some(id);
        }

        self.nodes[idx] = Some(node);
        self.count += 1;
    }

    fn search_layer_internal(
        &self,
        query: &[f32],
        entry_points: &[VectorId],
        ef: usize,
        level: usize,
    ) -> Vec<DistNode> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new(); // MinDistNode: closest at top
        let mut w = BinaryHeap::new(); // DistNode: furthest at top

        for &ep in entry_points {
            let d = self.dist(query, ep);
            visited.insert(ep);
            candidates.push(MinDistNode { dist: d, id: ep });
            w.push(DistNode { dist: d, id: ep });
        }

        while let Some(closest_candidate) = candidates.pop() {
            let furthest_w = w.peek().cloned().unwrap();
            if closest_candidate.dist > furthest_w.dist {
                break;
            }

            let cand_node = match self.nodes.get(closest_candidate.id as usize) {
                Some(Some(node)) => node,
                _ => continue,
            };

            if level < cand_node.neighbors.len() {
                for &neighbor in &cand_node.neighbors[level] {
                    if visited.insert(neighbor) {
                        let furthest_w = w.peek().unwrap();
                        let d = self.dist(query, neighbor);

                        if d < furthest_w.dist || w.len() < ef {
                            candidates.push(MinDistNode {
                                dist: d,
                                id: neighbor,
                            });
                            w.push(DistNode {
                                dist: d,
                                id: neighbor,
                            });
                            if w.len() > ef {
                                w.pop();
                            }
                        }
                    }
                }
            }
        }

        w.into_vec()
    }

    fn select_neighbors(
        &self,
        _query: &[f32],
        candidates: &[DistNode],
        m_max: usize,
    ) -> Vec<VectorId> {
        let mut sorted = candidates.to_vec();
        sorted.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        sorted.into_iter().take(m_max).map(|item| item.id).collect()
    }

    fn select_neighbors_by_ids(
        &self,
        origin: &[f32],
        neighbor_ids: &[VectorId],
        m_max: usize,
    ) -> Vec<VectorId> {
        let mut scored: Vec<DistNode> = neighbor_ids
            .iter()
            .map(|&id| DistNode {
                dist: self.dist(origin, id),
                id,
            })
            .collect();

        scored.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        scored.into_iter().take(m_max).map(|item| item.id).collect()
    }

    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search_opt: Option<usize>,
    ) -> Vec<SearchResult> {
        if self.count == 0 || self.entry_point.is_none() {
            return Vec::new();
        }

        let ef = ef_search_opt.unwrap_or(self.config.ef_search).max(k);
        let mut curr_ep = self.entry_point.unwrap();
        let mut curr_dist = self.dist(query, curr_ep);

        // 1. Greedy search from max_level down to 1
        for lc in (1..=self.max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                if let Some(Some(ep_node)) = self.nodes.get(curr_ep as usize) {
                    if lc < ep_node.neighbors.len() {
                        for &neighbor in &ep_node.neighbors[lc] {
                            let d = self.dist(query, neighbor);
                            if d < curr_dist {
                                curr_dist = d;
                                curr_ep = neighbor;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        // 2. Beam search at layer 0
        let candidates = self.search_layer_internal(query, &[curr_ep], ef, 0);

        let mut results: Vec<SearchResult> = candidates
            .into_iter()
            .map(|item| {
                let score = match self.config.metric {
                    DistanceMetric::Cosine => 1.0 - item.dist,
                    DistanceMetric::DotProduct => -item.dist,
                    DistanceMetric::Euclidean => 1.0 / (1.0 + item.dist),
                };
                SearchResult::new(item.id, item.dist, score)
            })
            .collect();

        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        results.truncate(k);
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_insert_and_search() {
        let dim = 64;
        let config = HnswConfig {
            m: 8,
            m0: 16,
            ef_construction: 32,
            ef_search: 20,
            metric: DistanceMetric::Cosine,
        };

        let mut hnsw = HnswIndex::new(dim, config);

        // Insert 50 vectors
        for i in 0..50 {
            let mut v = vec![0.0f32; dim];
            v[i] = 1.0;
            v[(i + 1) % dim] = 0.5;
            crate::simd::normalize_in_place(&mut v);
            hnsw.insert(i as u64, v);
        }

        assert_eq!(hnsw.len(), 50);

        // Search for vector 5
        let target = hnsw.get_vector(5).unwrap().clone();
        let results = hnsw.search(&target, 3, None);

        assert!(!results.is_empty());
        assert_eq!(results[0].id, 5);
        assert!(results[0].distance < 1e-4);
    }
}
