pub mod bm25;
pub mod embedding;
pub mod engine;
pub mod graph;
pub mod hnsw;
pub mod simd;
pub mod store;
pub mod types;

pub use bm25::{reciprocal_rank_fusion, Bm25Index};

pub use embedding::{EmbeddingProvider, FastFeatureEmbedder};
pub use engine::{SearchHit, SearchMode, SearchResponse, VectorDBEngine};
pub use graph::{Edge, EdgeKind, ImpactAnalysis, KnowledgeGraph, Node, NodeKind, Subgraph};
pub use hnsw::{HnswConfig, HnswIndex, HnswIndexThreadSafe};
pub use simd::{
    cosine_distance, cosine_similarity, dot_product, euclidean_distance_sq, DistanceMetric,
};
pub use store::{Chunk, ChunkStore};
pub use types::{ChunkId, Feature, FeatureSpace, NodeId, SearchResult, VectorId};
