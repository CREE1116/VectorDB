use serde::{Deserialize, Serialize};

pub type VectorId = u64;
pub type NodeId = String;
pub type ChunkId = String;

/// Vector coordinates are comparable only within the same feature space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeatureSpace {
    Text,
    Code,
    BinaryFingerprint,
    Metadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Feature {
    pub vector_id: VectorId,
    pub space: FeatureSpace,
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: VectorId,
    pub distance: f32,
    pub score: f32,
}

impl SearchResult {
    pub fn new(id: VectorId, distance: f32, score: f32) -> Self {
        Self {
            id,
            distance,
            score,
        }
    }
}
