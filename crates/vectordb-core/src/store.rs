//! In-memory and persistent chunk storage.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::{ChunkId, NodeId, VectorId};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: ChunkId,
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
    pub summary: Option<String>,
    pub node_id: Option<NodeId>,
    pub vector_id: VectorId,
    pub tokens_approx: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChunkStore {
    chunks: HashMap<ChunkId, Chunk>,
    vector_to_chunk: HashMap<VectorId, ChunkId>,
}

impl ChunkStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, chunk: Chunk) {
        if let Some(old) = self.chunks.get(&chunk.id) {
            self.vector_to_chunk.remove(&old.vector_id);
        }
        self.vector_to_chunk
            .insert(chunk.vector_id, chunk.id.clone());
        self.chunks.insert(chunk.id.clone(), chunk);
    }

    pub fn get(&self, id: &str) -> Option<&Chunk> {
        self.chunks.get(id)
    }

    pub fn get_by_vector(&self, vector_id: VectorId) -> Option<&Chunk> {
        self.vector_to_chunk
            .get(&vector_id)
            .and_then(|id| self.chunks.get(id))
    }

    pub fn remove(&mut self, id: &str) -> Option<Chunk> {
        if let Some(chunk) = self.chunks.remove(id) {
            self.vector_to_chunk.remove(&chunk.vector_id);
            Some(chunk)
        } else {
            None
        }
    }

    pub fn remove_by_file(&mut self, file_path: &str) -> Vec<Chunk> {
        let ids: Vec<ChunkId> = self
            .chunks
            .values()
            .filter(|c| c.file_path == file_path)
            .map(|c| c.id.clone())
            .collect();

        let mut removed = Vec::new();
        for id in ids {
            if let Some(c) = self.remove(&id) {
                removed.push(c);
            }
        }
        removed
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    pub fn all_chunks(&self) -> impl Iterator<Item = &Chunk> {
        self.chunks.values()
    }
}
