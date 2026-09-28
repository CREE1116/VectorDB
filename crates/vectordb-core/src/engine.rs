//! Main Engine uniting HNSW Vector Search, Knowledge Graph, and Chunk Storage.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::embedding::{EmbeddingProvider, FastFeatureEmbedder};
use crate::graph::{Edge, EdgeKind, ImpactAnalysis, KnowledgeGraph, Node, NodeKind, Subgraph};
use crate::hnsw::{HnswConfig, HnswIndexThreadSafe};
use crate::store::{Chunk, ChunkStore};
use crate::types::{Feature, FeatureSpace, NodeId, VectorId};

const DUMP_MAGIC_V2: &[u8; 4] = b"VDB2";
const DUMP_MAGIC_V3: &[u8; 4] = b"VDB3";
static SAVE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub chunk: Chunk,
    pub score: f32,
    pub distance: f32,
    pub node: Option<Node>,
    pub related_nodes: Vec<Node>,
}

#[cfg(test)]
mod feature_space_tests {
    use super::*;
    use std::collections::BTreeMap;

    fn chunk(path: &str, content: &str) -> Chunk {
        Chunk {
            id: format!("chunk:{path}"),
            file_path: path.into(),
            start_line: 1,
            end_line: 1,
            content: content.into(),
            summary: None,
            node_id: None,
            vector_id: 0,
            tokens_approx: 1,
        }
    }

    #[test]
    fn removes_legacy_binary_on_reload() {
        let db = VectorDBEngine::new(256);
        db.add_chunk_and_node(chunk("notes.md", "alpha topic"), None, None);
        db.add_chunk_and_node(chunk("src/main.rs", "fn beta() {}"), None, None);
        db.add_chunk_and_node(chunk("a.bin", "Binary File A"), None, Some(vec![1.0; 256]));
        db.add_chunk_and_node(chunk("b.bin", "Binary File B"), None, Some(vec![1.0; 256]));
        db.add_chunk_and_node(chunk("song.mp3", "Audio Track: song"), None, None);
        db.add_chunk_and_node(
            chunk("scan.pdf", "PDF Document: scan.pdf\nSize: 42 bytes"),
            None,
            None,
        );

        assert_eq!(db.text_hnsw.len(), 3);
        assert_eq!(db.code_hnsw.len(), 1);
        assert_eq!(db.binary_hnsw.len(), 2);

        let dir = std::env::temp_dir().join(format!(
            "vectordb-feature-test-{}-{}",
            std::process::id(),
            db.allocate_id()
        ));
        db.save_to_dir(&dir).unwrap();
        let reloaded = VectorDBEngine::load_from_dir(&dir, 256).unwrap();
        assert_eq!(reloaded.text_hnsw.len(), 1);
        assert_eq!(
            db.code_hnsw.snapshot_bytes().unwrap(),
            reloaded.code_hnsw.snapshot_bytes().unwrap()
        );
        assert_eq!(reloaded.binary_hnsw.len(), 0);
        assert_eq!(reloaded.store.read().all_chunks().count(), 2);
        assert!(reloaded
            .search("Audio Track PDF Document", 10, 0)
            .hits
            .iter()
            .all(|hit| matches!(hit.chunk.file_path.as_str(), "notes.md" | "src/main.rs")));
        assert!(reloaded
            .search("alpha", 10, 0)
            .hits
            .iter()
            .all(|h| !h.chunk.file_path.ends_with(".bin")));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn loads_v2_and_upgrades_on_next_save() {
        let db = VectorDBEngine::new(256);
        db.add_chunk_and_node(chunk("notes.md", "legacy snapshot"), None, None);
        let dir = std::env::temp_dir().join(format!("vectordb-v2-upgrade-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dump = EngineDumpV2 {
            graph: db.graph.read().clone(),
            store: db.store.read().clone(),
            bm25: db.bm25.read().clone(),
            next_id: *db.next_id.read(),
            features: db.features.read().values().cloned().collect(),
        };
        let mut file = File::create(dir.join("metadata.bin")).unwrap();
        file.write_all(DUMP_MAGIC_V2).unwrap();
        bincode::serialize_into(&mut file, &dump).unwrap();
        drop(file);
        let upgraded = VectorDBEngine::load_from_dir(&dir, 256).unwrap();
        assert_eq!(
            upgraded.search("legacy", 1, 0).hits[0].chunk.file_path,
            "notes.md"
        );
        upgraded.save_to_dir(&dir).unwrap();
        assert_eq!(
            &std::fs::read(dir.join("metadata.bin")).unwrap()[..4],
            DUMP_MAGIC_V3
        );
        assert_eq!(
            VectorDBEngine::load_from_dir(&dir, 256).unwrap().stats(),
            upgraded.stats()
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn batch_removal_keeps_unaffected_space_and_live_results() {
        let db = VectorDBEngine::new(256);
        for (path, content) in [
            ("a.md", "alpha document"),
            ("b.md", "beta document"),
            ("c.md", "gamma document"),
            ("src/lib.rs", "fn durable_code() {}"),
        ] {
            add_file(&db, path, content);
        }
        let code_before = db.code_hnsw.snapshot_bytes().unwrap();
        db.remove_files_batch(&["a.md", "b.md", "a.md"]);
        assert_eq!(db.text_hnsw.len(), 1);
        assert_eq!(db.code_hnsw.snapshot_bytes().unwrap(), code_before);
        assert_eq!(db.search("gamma", 1, 0).hits[0].chunk.file_path, "c.md");
        assert!(db
            .store
            .read()
            .all_chunks()
            .all(|chunk| chunk.file_path != "a.md" && chunk.file_path != "b.md"));
    }

    fn add_file(db: &VectorDBEngine, path: &str, content: &str) {
        let node = Node {
            id: format!("file:{path}"),
            label: path.into(),
            kind: NodeKind::File,
            file_path: path.into(),
            start_line: 1,
            end_line: 1,
            signature: None,
            docstring: None,
            chunk_id: None,
            vector_id: None,
            metadata: HashMap::new(),
        };
        db.add_chunk_and_node(chunk(path, content), Some(node), None);
    }

    fn assert_matches_fresh(db: &VectorDBEngine, files: &BTreeMap<String, String>) {
        let fresh = VectorDBEngine::new(256);
        for (path, content) in files {
            add_file(&fresh, path, content);
        }
        let actual: BTreeMap<_, _> = db
            .store
            .read()
            .all_chunks()
            .map(|c| (c.file_path.clone(), c.content.clone()))
            .collect();
        assert_eq!(&actual, files);
        assert_eq!(db.stats(), fresh.stats());
        assert_eq!(db.features.read().len(), fresh.features.read().len());
        assert_eq!(db.text_hnsw.len(), fresh.text_hnsw.len());
        assert_eq!(db.code_hnsw.len(), fresh.code_hnsw.len());
        let nodes: BTreeMap<_, _> = db
            .get_all_nodes()
            .into_iter()
            .map(|n| (n.id, n.file_path))
            .collect();
        let fresh_nodes: BTreeMap<_, _> = fresh
            .get_all_nodes()
            .into_iter()
            .map(|n| (n.id, n.file_path))
            .collect();
        assert_eq!(nodes, fresh_nodes);
        for (path, content) in files {
            let vector_id = db
                .store
                .read()
                .get(&format!("chunk:{path}"))
                .unwrap()
                .vector_id;
            let fresh_id = fresh
                .store
                .read()
                .get(&format!("chunk:{path}"))
                .unwrap()
                .vector_id;
            assert_eq!(
                db.features.read()[&vector_id].vector,
                fresh.features.read()[&fresh_id].vector
            );
            let marker = content.split_whitespace().next().unwrap();
            assert_eq!(db.search(marker, 1, 0).hits[0].chunk.file_path, *path);
            assert_eq!(fresh.search(marker, 1, 0).hits[0].chunk.file_path, *path);
        }
    }

    fn mutation_equivalence(steps: usize) {
        let db = VectorDBEngine::new(256);
        let mut files = BTreeMap::new();
        let paths = ["a.md", "b.md", "c.md", "nested/d.md", "e.md", "f.md"];
        let mut state = 0x3a7b_82c9_u64;
        for step in 0..steps {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let source = paths[(state as usize) % paths.len()];
            let target = paths[((state >> 16) as usize) % paths.len()];
            match state % 5 {
                0 | 1 => {
                    db.remove_file(source);
                    let content = format!("marker{step} content for {source}");
                    add_file(&db, source, &content);
                    files.insert(source.into(), content);
                }
                2 => {
                    db.remove_file(source);
                    files.remove(source);
                }
                3 if source != target => {
                    if files.remove(source).is_some() {
                        db.remove_file(source);
                        db.remove_file(target);
                        let content = format!("marker{step} moved content");
                        add_file(&db, target, &content);
                        files.insert(target.into(), content);
                    }
                }
                4 if source != target && files.contains_key(source) => {
                    db.remove_file(target);
                    let content = format!("marker{step} copied content");
                    add_file(&db, target, &content);
                    files.insert(target.into(), content);
                }
                _ => {}
            }
            if step % 100 == 99 {
                assert_matches_fresh(&db, &files);
            }
            if step % 1_000 == 999 {
                let dir = std::env::temp_dir().join(format!(
                    "vectordb-mutation-test-{}-{step}",
                    std::process::id()
                ));
                db.save_to_dir(&dir).unwrap();
                let reloaded = VectorDBEngine::load_from_dir(&dir, 256).unwrap();
                assert_matches_fresh(&reloaded, &files);
                std::fs::remove_dir_all(dir).unwrap();
            }
        }
    }

    #[test]
    fn incremental_matches_fresh_after_mutations() {
        mutation_equivalence(1_000);
    }

    #[test]
    #[ignore = "long running stress check"]
    fn incremental_matches_fresh_after_ten_thousand_mutations() {
        mutation_equivalence(10_000);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub query: String,
    pub hits: Vec<SearchHit>,
    pub total_nodes: usize,
    pub total_vectors: usize,
    pub subgraph: Option<Subgraph>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    Bm25,
    Dense,
    Hybrid,
}

use crate::bm25::Bm25Index;

#[derive(Serialize, Deserialize)]
struct EngineDump {
    graph: KnowledgeGraph,
    store: ChunkStore,
    bm25: Bm25Index,
    next_id: u64,
}

#[derive(Serialize, Deserialize)]
struct EngineDumpV2 {
    graph: KnowledgeGraph,
    store: ChunkStore,
    bm25: Bm25Index,
    next_id: u64,
    features: Vec<Feature>,
}

#[derive(Serialize, Deserialize)]
struct EngineDumpV3 {
    graph: KnowledgeGraph,
    store: ChunkStore,
    bm25: Bm25Index,
    next_id: u64,
    features: Vec<Feature>,
    text_hnsw: Vec<u8>,
    code_hnsw: Vec<u8>,
    binary_hnsw: Vec<u8>,
}

pub struct VectorDBEngine {
    embedder: Arc<dyn EmbeddingProvider>,
    text_hnsw: Arc<HnswIndexThreadSafe>,
    code_hnsw: Arc<HnswIndexThreadSafe>,
    binary_hnsw: Arc<HnswIndexThreadSafe>,
    features: Arc<RwLock<HashMap<VectorId, Feature>>>,
    graph: Arc<RwLock<KnowledgeGraph>>,
    store: Arc<RwLock<ChunkStore>>,
    bm25: Arc<RwLock<Bm25Index>>,
    next_id: Arc<RwLock<u64>>,
}

impl VectorDBEngine {
    fn feature_space(chunk: &Chunk, node: Option<&Node>, override_vector: bool) -> FeatureSpace {
        if override_vector {
            return FeatureSpace::BinaryFingerprint;
        }
        if matches!(node.map(|n| &n.kind), Some(NodeKind::AudioTrack)) {
            return FeatureSpace::Metadata;
        }
        if matches!(
            Path::new(&chunk.file_path)
                .extension()
                .and_then(|e| e.to_str()),
            Some("rs" | "py" | "js" | "ts" | "jsx" | "tsx" | "go" | "c" | "cpp" | "h" | "hpp")
        ) {
            FeatureSpace::Code
        } else {
            FeatureSpace::Text
        }
    }

    pub fn new(dim: usize) -> Self {
        let embedder = Arc::new(FastFeatureEmbedder::new(dim));
        let text_hnsw = Arc::new(HnswIndexThreadSafe::new(dim, HnswConfig::default()));
        let code_hnsw = Arc::new(HnswIndexThreadSafe::new(dim, HnswConfig::default()));
        let binary_hnsw = Arc::new(HnswIndexThreadSafe::new(256, HnswConfig::default()));
        let graph = Arc::new(RwLock::new(KnowledgeGraph::new()));
        let store = Arc::new(RwLock::new(ChunkStore::new()));
        let bm25 = Arc::new(RwLock::new(Bm25Index::default()));
        let next_id = Arc::new(RwLock::new(1));

        Self {
            embedder,
            text_hnsw,
            code_hnsw,
            binary_hnsw,
            features: Arc::new(RwLock::new(HashMap::new())),
            graph,
            store,
            bm25,
            next_id,
        }
    }

    pub fn with_embedder(embedder: Arc<dyn EmbeddingProvider>) -> Self {
        let dim = embedder.dim();
        let text_hnsw = Arc::new(HnswIndexThreadSafe::new(dim, HnswConfig::default()));
        let code_hnsw = Arc::new(HnswIndexThreadSafe::new(dim, HnswConfig::default()));
        let binary_hnsw = Arc::new(HnswIndexThreadSafe::new(256, HnswConfig::default()));
        let graph = Arc::new(RwLock::new(KnowledgeGraph::new()));
        let store = Arc::new(RwLock::new(ChunkStore::new()));
        let bm25 = Arc::new(RwLock::new(Bm25Index::default()));
        let next_id = Arc::new(RwLock::new(1));

        Self {
            embedder,
            text_hnsw,
            code_hnsw,
            binary_hnsw,
            features: Arc::new(RwLock::new(HashMap::new())),
            graph,
            store,
            bm25,
            next_id,
        }
    }

    pub fn allocate_id(&self) -> VectorId {
        let mut id = self.next_id.write();
        let current = *id;
        *id += 1;
        current
    }

    /// Add a parsed code/text chunk and associated graph node.
    pub fn add_chunk_and_node(
        &self,
        mut chunk: Chunk,
        mut node: Option<Node>,
        embedding_override: Option<Vec<f32>>,
    ) -> VectorId {
        let vid = self.allocate_id();
        chunk.vector_id = vid;

        let space = Self::feature_space(&chunk, node.as_ref(), embedding_override.is_some());
        let vec = match embedding_override {
            Some(v) => v,
            None => {
                let text_to_embed = if let Some(ref n) = node {
                    format!(
                        "{} {}\n{}",
                        n.label,
                        n.signature.as_deref().unwrap_or(""),
                        chunk.content
                    )
                } else {
                    chunk.content.clone()
                };
                self.embedder.embed(&text_to_embed)
            }
        };

        match space {
            FeatureSpace::Text => self.text_hnsw.insert(vid, vec.clone()),
            FeatureSpace::Code => self.code_hnsw.insert(vid, vec.clone()),
            FeatureSpace::BinaryFingerprint => self.binary_hnsw.insert(vid, vec.clone()),
            FeatureSpace::Metadata => {}
        }
        if space != FeatureSpace::Metadata {
            self.features.write().insert(
                vid,
                Feature {
                    vector_id: vid,
                    space,
                    vector: vec,
                },
            );
        }

        if let Some(ref mut n) = node {
            n.vector_id = Some(vid);
            n.chunk_id = Some(chunk.id.clone());
            chunk.node_id = Some(n.id.clone());
            self.graph.write().add_node(n.clone());
        }

        self.bm25
            .write()
            .add_document(chunk.id.clone(), &chunk.content);
        self.store.write().insert(chunk);
        vid
    }

    pub fn add_graph_node(&self, node: Node) {
        self.graph.write().add_node(node);
    }

    pub fn add_graph_edge(
        &self,
        source: NodeId,
        target: NodeId,
        kind: EdgeKind,
        weight: f32,
    ) -> bool {
        self.graph.write().add_edge(source, target, kind, weight)
    }

    /// Remove all chunks, BM25 postings, and graph nodes for a given file path.
    pub fn remove_file(&self, file_path: &str) {
        self.remove_files_batch(&[file_path]);
    }

    /// Drop non-text entries carried over from snapshots written before text-only indexing.
    fn purge_legacy_non_text(&self) {
        let store = self.store.read();
        let graph = self.graph.read();
        let features = self.features.read();
        let paths: std::collections::HashSet<String> = store
            .all_chunks()
            .filter(|chunk| {
                let extension = Path::new(&chunk.file_path)
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                matches!(
                    extension.as_str(),
                    "mp3"
                        | "m4a"
                        | "aac"
                        | "ogg"
                        | "wav"
                        | "flac"
                        | "png"
                        | "jpg"
                        | "jpeg"
                        | "gif"
                        | "webp"
                        | "svg"
                        | "zip"
                        | "tar"
                        | "gz"
                        | "7z"
                        | "rar"
                        | "bin"
                        | "dat"
                        | "exe"
                        | "dll"
                        | "so"
                        | "dylib"
                        | "wasm"
                        | "docx"
                        | "pptx"
                        | "xlsx"
                ) || matches!(
                    graph.get_node_by_chunk(&chunk.id).map(|node| &node.kind),
                    Some(NodeKind::AudioTrack | NodeKind::Image | NodeKind::Binary)
                ) || matches!(
                    features.get(&chunk.vector_id).map(|feature| feature.space),
                    Some(FeatureSpace::BinaryFingerprint | FeatureSpace::Metadata)
                ) || chunk.content.starts_with("PDF Document: ")
            })
            .map(|chunk| chunk.file_path.clone())
            .collect();
        drop(features);
        drop(graph);
        drop(store);
        if !paths.is_empty() {
            self.remove_files_batch(&paths.iter().map(String::as_str).collect::<Vec<_>>());
        }
    }

    /// Remove several files while rebuilding each affected vector space only once.
    pub fn remove_files_batch(&self, file_paths: &[&str]) {
        let mut affected = std::collections::HashSet::new();
        let unique_paths: std::collections::HashSet<_> = file_paths.iter().copied().collect();
        let removed_chunks = {
            let mut store = self.store.write();
            store.remove_by_files(&unique_paths)
        };
        {
            let mut bm25 = self.bm25.write();
            for chunk in &removed_chunks {
                bm25.remove_document(&chunk.id);
            }
        }
        {
            let mut features = self.features.write();
            for chunk in &removed_chunks {
                if let Some(feature) = features.remove(&chunk.vector_id) {
                    affected.insert(feature.space);
                }
            }
        }
        for file_path in unique_paths {
            self.graph.write().remove_file_nodes(file_path);
        }
        if !affected.is_empty() {
            self.rebuild_vector_indices(&affected);
        }
    }

    fn rebuild_vector_indices(&self, affected: &std::collections::HashSet<FeatureSpace>) {
        let features = self.features.read();
        let vectors_for = |space| {
            let mut vectors: Vec<_> = features
                .values()
                .filter(|feature| feature.space == space)
                .map(|feature| (feature.vector_id, feature.vector.clone()))
                .collect();
            vectors.sort_by_key(|(id, _)| *id);
            vectors
        };
        if affected.contains(&FeatureSpace::Text) {
            self.text_hnsw.rebuild(&vectors_for(FeatureSpace::Text));
        }
        if affected.contains(&FeatureSpace::Code) {
            self.code_hnsw.rebuild(&vectors_for(FeatureSpace::Code));
        }
        if affected.contains(&FeatureSpace::BinaryFingerprint) {
            self.binary_hnsw
                .rebuild(&vectors_for(FeatureSpace::BinaryFingerprint));
        }
    }

    /// Hybrid search: Combines Dense HNSW Vector similarity with Sparse BM25 via Reciprocal Rank Fusion (RRF)
    /// and expands context with Knowledge Graph.
    pub fn search(&self, query: &str, limit: usize, expand_graph_hops: usize) -> SearchResponse {
        self.search_with_mode(query, limit, expand_graph_hops, SearchMode::Hybrid)
    }

    pub fn search_with_mode(
        &self,
        query: &str,
        limit: usize,
        expand_graph_hops: usize,
        mode: SearchMode,
    ) -> SearchResponse {
        let (text_hits, code_hits) = if mode == SearchMode::Bm25 {
            (Vec::new(), Vec::new())
        } else {
            let query_vec = self.embedder.embed(query);
            (
                self.text_hnsw
                    .search(&query_vec, limit.saturating_mul(2), None),
                self.code_hnsw
                    .search(&query_vec, limit.saturating_mul(2), None),
            )
        };

        let store_guard = self.store.read();
        let graph_guard = self.graph.read();
        let bm25_guard = self.bm25.read();

        // 1. Dense candidate chunk IDs
        let mut dense_scores = std::collections::HashMap::new();
        let feature_guard = self.features.read();
        let to_ids = |raw_hits: &[crate::types::SearchResult],
                      scores: &mut HashMap<String, (f32, f32)>| {
            raw_hits
                .iter()
                .filter_map(|hit| {
                    if !feature_guard.contains_key(&hit.id) {
                        return None;
                    }
                    store_guard.get_by_vector(hit.id).map(|chunk| {
                        scores.insert(chunk.id.clone(), (hit.score, hit.distance));
                        chunk.id.clone()
                    })
                })
                .collect::<Vec<_>>()
        };
        let text_chunk_ids = to_ids(&text_hits, &mut dense_scores);
        let code_chunk_ids = to_ids(&code_hits, &mut dense_scores);

        // 2. Sparse BM25 candidate chunk IDs
        let bm25_hits = if mode == SearchMode::Dense {
            Vec::new()
        } else {
            bm25_guard.search(query, limit.saturating_mul(2))
        };
        let sparse_chunk_ids: Vec<String> = bm25_hits.iter().map(|(id, _)| id.clone()).collect();

        // 3. Reciprocal Rank Fusion (RRF)
        let fused = crate::bm25::reciprocal_rank_fusion_many(
            &[&sparse_chunk_ids, &text_chunk_ids, &code_chunk_ids],
            60.0,
        );

        let mut hits = Vec::new();
        let mut hit_node_ids = Vec::new();

        for (chunk_id, rrf_score) in fused.into_iter().take(limit) {
            if let Some(chunk) = store_guard.get(&chunk_id) {
                let node = graph_guard.get_node_by_chunk(&chunk_id).cloned();
                let related_nodes = if expand_graph_hops > 0 {
                    if let Some(ref n) = node {
                        hit_node_ids.push(n.id.clone());
                        graph_guard
                            .get_context_expansion_nodes(&n.id, 5)
                            .into_iter()
                            .cloned()
                            .collect()
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                };

                let (raw_score, distance) = dense_scores
                    .get(&chunk_id)
                    .copied()
                    .unwrap_or((rrf_score * 50.0, 1.0 - rrf_score.min(1.0)));

                hits.push(SearchHit {
                    chunk: chunk.clone(),
                    score: raw_score.max(rrf_score * 50.0),
                    distance,
                    node,
                    related_nodes,
                });
            }
        }

        let subgraph = if expand_graph_hops > 0 && !hit_node_ids.is_empty() {
            Some(graph_guard.k_hop_subgraph(&hit_node_ids, expand_graph_hops))
        } else {
            None
        };

        SearchResponse {
            query: query.to_string(),
            hits,
            total_nodes: graph_guard.node_count(),
            total_vectors: self.features.read().len(),
            subgraph,
        }
    }

    /// Package the search results into a concise, token-efficient Markdown prompt ready for LLM Agents!
    pub fn package_for_llm(&self, response: &SearchResponse) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "# VectorDB Retrieval Context for: \"{}\"\n\n",
            response.query
        ));

        if response.hits.is_empty() {
            out.push_str("No relevant code chunks found.\n");
            return out;
        }

        for (idx, hit) in response.hits.iter().enumerate() {
            out.push_str(&format!(
                "## [{}] File: `{}` (Lines {}-{}, Score: {:.3})\n",
                idx + 1,
                hit.chunk.file_path,
                hit.chunk.start_line,
                hit.chunk.end_line,
                hit.score
            ));

            if let Some(ref node) = hit.node {
                out.push_str(&format!("- Symbol: `{}` ({:?})\n", node.label, node.kind));
                if let Some(ref sig) = node.signature {
                    out.push_str(&format!("- Signature: `{}`\n", sig));
                }
            }

            if !hit.related_nodes.is_empty() {
                out.push_str("- Connected Symbols:\n");
                for rel in &hit.related_nodes {
                    out.push_str(&format!(
                        "  * `{}` in `{}` ({:?})\n",
                        rel.label, rel.file_path, rel.kind
                    ));
                }
            }

            // Determine language for code block syntax highlighting
            let lang = match Path::new(&hit.chunk.file_path)
                .extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or("")
            {
                "rs" => "rust",
                "py" => "python",
                "js" => "javascript",
                "ts" => "typescript",
                "go" => "go",
                "md" => "markdown",
                _ => "",
            };

            out.push_str(&format!(
                "\n```{lang}\n{}\n```\n\n",
                hit.chunk.content.trim()
            ));
        }

        out
    }

    pub fn get_subgraph(&self, seed_ids: &[NodeId], hops: usize) -> Subgraph {
        self.graph.read().k_hop_subgraph(seed_ids, hops)
    }

    pub fn get_all_nodes(&self) -> Vec<Node> {
        self.graph.read().all_nodes().cloned().collect()
    }

    pub fn indexed_file_paths(&self) -> Vec<String> {
        let mut paths: Vec<_> = self
            .store
            .read()
            .all_chunks()
            .map(|chunk| chunk.file_path.clone())
            .collect();
        paths.extend(
            self.graph
                .read()
                .all_nodes()
                .map(|node| node.file_path.clone()),
        );
        paths.sort();
        paths.dedup();
        paths
    }

    pub fn get_all_edges(&self) -> Vec<Edge> {
        self.graph.read().all_edges().into_iter().cloned().collect()
    }

    pub fn get_chunk(&self, id: &str) -> Option<Chunk> {
        self.store.read().get(id).cloned()
    }

    pub fn stats(&self) -> (usize, usize, usize) {
        let total_chunks = self.store.read().len();
        let total_nodes = self.graph.read().node_count();
        let total_edges = self.graph.read().edge_count();
        (total_chunks, total_nodes, total_edges)
    }

    pub fn find_symbols(&self, query: &str) -> Vec<Node> {
        self.graph
            .read()
            .find_symbols(query)
            .into_iter()
            .cloned()
            .collect()
    }

    pub fn compute_impact(&self, target_node_id: &str, max_depth: usize) -> Option<ImpactAnalysis> {
        self.graph.read().compute_impact(target_node_id, max_depth)
    }

    /// Save database state to disk.
    pub fn save_to_dir(&self, dir_path: &Path) -> anyhow::Result<()> {
        std::fs::create_dir_all(dir_path)?;
        let dump_path = dir_path.join("metadata.bin");
        let sequence = SAVE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temp_path = dir_path.join(format!(
            "metadata.bin.{}.{}.tmp",
            std::process::id(),
            sequence
        ));

        let dump = EngineDumpV3 {
            graph: self.graph.read().clone(),
            store: self.store.read().clone(),
            bm25: self.bm25.read().clone(),
            next_id: *self.next_id.read(),
            features: self.features.read().values().cloned().collect(),
            text_hnsw: self.text_hnsw.snapshot_bytes()?,
            code_hnsw: self.code_hnsw.snapshot_bytes()?,
            binary_hnsw: self.binary_hnsw.snapshot_bytes()?,
        };

        let result = (|| -> anyhow::Result<()> {
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp_path)?;
            let mut writer = BufWriter::new(file);
            writer.write_all(DUMP_MAGIC_V3)?;
            bincode::serialize_into(&mut writer, &dump)?;
            writer.flush()?;
            writer.into_inner()?.sync_all()?;
            std::fs::rename(&temp_path, &dump_path)?;
            #[cfg(unix)]
            File::open(dir_path)?.sync_all()?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&temp_path);
        }
        result
    }

    /// Load database state from disk.
    pub fn load_from_dir(dir_path: &Path, dim: usize) -> anyhow::Result<Self> {
        let dump_path = dir_path.join("metadata.bin");
        if !dump_path.exists() {
            return Ok(Self::new(dim));
        }

        let file = File::open(&dump_path)?;
        let mut reader = BufReader::new(file);
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        let (mut graph, store, bm25, next_id, features, index_bytes) = if &magic == DUMP_MAGIC_V3 {
            let dump: EngineDumpV3 = bincode::deserialize_from(reader)?;
            (
                dump.graph,
                dump.store,
                dump.bm25,
                dump.next_id,
                dump.features,
                Some((dump.text_hnsw, dump.code_hnsw, dump.binary_hnsw)),
            )
        } else if &magic == DUMP_MAGIC_V2 {
            let dump: EngineDumpV2 = bincode::deserialize_from(reader)?;
            (
                dump.graph,
                dump.store,
                dump.bm25,
                dump.next_id,
                dump.features,
                None,
            )
        } else {
            // v0.1 did not persist custom vectors. Recover text/code vectors;
            // binary files remain searchable by BM25 until reindexed.
            let file = File::open(&dump_path)?;
            let dump: EngineDump = bincode::deserialize_from(BufReader::new(file))?;
            let embedder = FastFeatureEmbedder::new(dim);
            let features = dump
                .store
                .all_chunks()
                .filter_map(|chunk| {
                    let node = dump.graph.get_node_by_vector(chunk.vector_id);
                    let space = Self::feature_space(chunk, node, false);
                    if matches!(
                        node.map(|n| &n.kind),
                        Some(NodeKind::Binary | NodeKind::Image)
                    ) || space == FeatureSpace::Metadata
                    {
                        return None;
                    }
                    let text = if let Some(n) = node {
                        format!(
                            "{} {}\n{}",
                            n.label,
                            n.signature.as_deref().unwrap_or(""),
                            chunk.content
                        )
                    } else {
                        chunk.content.clone()
                    };
                    Some(Feature {
                        vector_id: chunk.vector_id,
                        space,
                        vector: embedder.embed(&text),
                    })
                })
                .collect();
            (
                dump.graph,
                dump.store,
                dump.bm25,
                dump.next_id,
                features,
                None,
            )
        };

        graph.prune_dangling_edges();
        let mut engine = Self::new(dim);
        let has_index_snapshot = index_bytes.is_some();
        if let Some((text, code, binary)) = index_bytes {
            engine.text_hnsw = Arc::new(HnswIndexThreadSafe::from_snapshot_bytes(&text, dim)?);
            engine.code_hnsw = Arc::new(HnswIndexThreadSafe::from_snapshot_bytes(&code, dim)?);
            engine.binary_hnsw = Arc::new(HnswIndexThreadSafe::from_snapshot_bytes(&binary, 256)?);
        }
        for feature in features {
            match feature.space {
                FeatureSpace::Text if feature.vector.len() == dim => {
                    if !has_index_snapshot {
                        engine
                            .text_hnsw
                            .insert(feature.vector_id, feature.vector.clone());
                    } else {
                        anyhow::ensure!(
                            engine.text_hnsw.contains_vector(feature.vector_id),
                            "text HNSW missing feature"
                        );
                    }
                }
                FeatureSpace::Code if feature.vector.len() == dim => {
                    if !has_index_snapshot {
                        engine
                            .code_hnsw
                            .insert(feature.vector_id, feature.vector.clone());
                    } else {
                        anyhow::ensure!(
                            engine.code_hnsw.contains_vector(feature.vector_id),
                            "code HNSW missing feature"
                        );
                    }
                }
                FeatureSpace::BinaryFingerprint if feature.vector.len() == 256 => {
                    if !has_index_snapshot {
                        engine
                            .binary_hnsw
                            .insert(feature.vector_id, feature.vector.clone());
                    } else {
                        anyhow::ensure!(
                            engine.binary_hnsw.contains_vector(feature.vector_id),
                            "binary HNSW missing feature"
                        );
                    }
                }
                FeatureSpace::Metadata => continue,
                _ => anyhow::bail!("saved feature dimension does not match its space"),
            }
            engine.features.write().insert(feature.vector_id, feature);
        }
        let features_guard = engine.features.read();
        for (space, count) in [
            (FeatureSpace::Text, engine.text_hnsw.len()),
            (FeatureSpace::Code, engine.code_hnsw.len()),
            (FeatureSpace::BinaryFingerprint, engine.binary_hnsw.len()),
        ] {
            anyhow::ensure!(
                features_guard
                    .values()
                    .filter(|feature| feature.space == space)
                    .count()
                    == count,
                "HNSW feature count mismatch"
            );
        }
        drop(features_guard);
        *engine.graph.write() = graph;
        *engine.store.write() = store;
        *engine.bm25.write() = bm25;
        *engine.next_id.write() = next_id;
        engine.purge_legacy_non_text();
        Ok(engine)
    }
}
