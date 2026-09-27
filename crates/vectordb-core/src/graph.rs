//! Graph Engine for Code & Knowledge structures.
//! Tracks files, chunks, AST symbols (functions, structs, imports) and their relationships.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

use crate::types::{ChunkId, NodeId, VectorId};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeKind {
    File,
    Module,
    Function,
    Method,
    Class,
    Struct,
    Trait,
    Interface,
    DocSection,
    Document,
    AudioTrack,
    Image,
    Binary,
    Chunk,
    Concept,
}

impl std::fmt::Display for NodeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeKind {
    Contains,
    Calls,
    Imports,
    Defines,
    Implements,
    References,
    SimilarTo,
    NextChunk,
    CreatedBy,
    BelongsTo,
    TaggedWith,
    CoChangedWith,
    /// Call-shaped text match without scope or import resolution.
    CallsCandidate,
}

impl std::fmt::Display for EdgeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub label: String,
    pub kind: NodeKind,
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub signature: Option<String>,
    pub docstring: Option<String>,
    pub chunk_id: Option<ChunkId>,
    pub vector_id: Option<VectorId>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub source: NodeId,
    pub target: NodeId,
    pub kind: EdgeKind,
    pub weight: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Subgraph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KnowledgeGraph {
    nodes: HashMap<NodeId, Node>,
    outgoing: HashMap<NodeId, Vec<Edge>>,
    incoming: HashMap<NodeId, Vec<Edge>>,
    chunk_to_node: HashMap<ChunkId, NodeId>,
    vector_to_node: HashMap<VectorId, NodeId>,
}

impl KnowledgeGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.outgoing.values().map(|e| e.len()).sum()
    }

    pub fn add_node(&mut self, node: Node) {
        if let Some(ref chunk_id) = node.chunk_id {
            self.chunk_to_node.insert(chunk_id.clone(), node.id.clone());
        }
        if let Some(vector_id) = node.vector_id {
            self.vector_to_node.insert(vector_id, node.id.clone());
        }
        self.nodes.insert(node.id.clone(), node);
    }

    /// Remove all nodes and associated edges for a given file path.
    pub fn remove_file_nodes(&mut self, file_path: &str) -> Vec<NodeId> {
        let to_remove: Vec<NodeId> = self
            .nodes
            .values()
            .filter(|n| n.file_path == file_path)
            .map(|n| n.id.clone())
            .collect();

        let remove_set: HashSet<NodeId> = to_remove.iter().cloned().collect();

        for id in &to_remove {
            if let Some(node) = self.nodes.remove(id) {
                if let Some(ref chunk_id) = node.chunk_id {
                    self.chunk_to_node.remove(chunk_id);
                }
                if let Some(vector_id) = node.vector_id {
                    self.vector_to_node.remove(&vector_id);
                }
            }
            self.outgoing.remove(id);
            self.incoming.remove(id);
        }

        // Clean up remaining dangling edge references
        for edges in self.outgoing.values_mut() {
            edges.retain(|e| !remove_set.contains(&e.target));
        }
        for edges in self.incoming.values_mut() {
            edges.retain(|e| !remove_set.contains(&e.source));
        }

        to_remove
    }

    pub fn add_edge(
        &mut self,
        source: NodeId,
        target: NodeId,
        kind: EdgeKind,
        weight: f32,
    ) -> bool {
        if !self.nodes.contains_key(&source) || !self.nodes.contains_key(&target) {
            return false;
        }
        if self
            .outgoing
            .get(&source)
            .is_some_and(|edges| edges.iter().any(|e| e.target == target && e.kind == kind))
        {
            return false;
        }
        let edge = Edge {
            source: source.clone(),
            target: target.clone(),
            kind,
            weight,
        };
        self.outgoing
            .entry(source.clone())
            .or_default()
            .push(edge.clone());
        self.incoming.entry(target).or_default().push(edge);
        true
    }

    /// Remove legacy edges whose endpoints are absent and rebuild the reverse index.
    pub fn prune_dangling_edges(&mut self) {
        self.outgoing
            .retain(|source, _| self.nodes.contains_key(source));
        for edges in self.outgoing.values_mut() {
            edges.retain(|edge| self.nodes.contains_key(&edge.target));
        }
        self.incoming.clear();
        for edges in self.outgoing.values() {
            for edge in edges {
                self.incoming
                    .entry(edge.target.clone())
                    .or_default()
                    .push(edge.clone());
            }
        }
    }

    pub fn get_node(&self, id: &str) -> Option<&Node> {
        self.nodes.get(id)
    }

    pub fn get_node_by_chunk(&self, chunk_id: &str) -> Option<&Node> {
        self.chunk_to_node
            .get(chunk_id)
            .and_then(|id| self.nodes.get(id))
    }

    pub fn get_node_by_vector(&self, vector_id: VectorId) -> Option<&Node> {
        self.vector_to_node
            .get(&vector_id)
            .and_then(|id| self.nodes.get(id))
    }

    pub fn get_outgoing(&self, id: &str) -> &[Edge] {
        self.outgoing.get(id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn get_incoming(&self, id: &str) -> &[Edge] {
        self.incoming.get(id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn all_nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    pub fn all_edges(&self) -> Vec<&Edge> {
        self.outgoing
            .values()
            .flat_map(|edges| edges.iter())
            .collect()
    }

    /// Extract a k-hop subgraph around the given seed nodes.
    /// Explores both outgoing and incoming edges.
    pub fn k_hop_subgraph(&self, seed_node_ids: &[NodeId], max_hops: usize) -> Subgraph {
        let mut visited_nodes: HashSet<NodeId> = HashSet::new();
        let mut collected_edges: Vec<Edge> = Vec::new();
        let mut edge_keys: HashSet<(NodeId, NodeId, String)> = HashSet::new();

        let mut queue: VecDeque<(NodeId, usize)> = VecDeque::new();

        for id in seed_node_ids {
            if self.nodes.contains_key(id) {
                visited_nodes.insert(id.clone());
                queue.push_back((id.clone(), 0));
            }
        }

        while let Some((curr_id, hop)) = queue.pop_front() {
            if hop >= max_hops {
                continue;
            }

            // Outgoing
            if let Some(edges) = self.outgoing.get(&curr_id) {
                for edge in edges {
                    let key = (
                        edge.source.clone(),
                        edge.target.clone(),
                        edge.kind.to_string(),
                    );
                    if edge_keys.insert(key) {
                        collected_edges.push(edge.clone());
                    }
                    if visited_nodes.insert(edge.target.clone()) {
                        queue.push_back((edge.target.clone(), hop + 1));
                    }
                }
            }

            // Incoming
            if let Some(edges) = self.incoming.get(&curr_id) {
                for edge in edges {
                    let key = (
                        edge.source.clone(),
                        edge.target.clone(),
                        edge.kind.to_string(),
                    );
                    if edge_keys.insert(key) {
                        collected_edges.push(edge.clone());
                    }
                    if visited_nodes.insert(edge.source.clone()) {
                        queue.push_back((edge.source.clone(), hop + 1));
                    }
                }
            }
        }

        let nodes: Vec<Node> = visited_nodes
            .into_iter()
            .filter_map(|id| self.nodes.get(&id).cloned())
            .collect();

        Subgraph {
            nodes,
            edges: collected_edges,
        }
    }

    /// Find related nodes for context expansion (e.g. called functions, imports, definitions)
    pub fn get_context_expansion_nodes(&self, node_id: &str, max_nodes: usize) -> Vec<&Node> {
        let mut result = Vec::new();
        let mut seen = HashSet::new();
        seen.insert(node_id.to_string());

        // Prioritize outgoing Calls and References
        for edge in self.get_outgoing(node_id) {
            if (edge.kind == EdgeKind::Calls
                || edge.kind == EdgeKind::References
                || edge.kind == EdgeKind::Defines)
                && seen.insert(edge.target.clone())
            {
                if let Some(n) = self.get_node(&edge.target) {
                    result.push(n);
                    if result.len() >= max_nodes {
                        return result;
                    }
                }
            }
        }

        // Also parent file/container
        for edge in self.get_incoming(node_id) {
            if edge.kind == EdgeKind::Contains && seen.insert(edge.source.clone()) {
                if let Some(n) = self.get_node(&edge.source) {
                    result.push(n);
                    if result.len() >= max_nodes {
                        return result;
                    }
                }
            }
        }

        result
    }

    /// Search symbols by exact or partial name match.
    pub fn find_symbols(&self, query: &str) -> Vec<&Node> {
        let q_lower = query.to_lowercase();
        let mut results = Vec::new();
        for node in self.nodes.values() {
            if node.label.to_lowercase().contains(&q_lower) {
                results.push(node);
            }
        }
        // Prioritize exact match or start match
        results.sort_by(|a, b| {
            let a_exact = a.label.eq_ignore_ascii_case(query);
            let b_exact = b.label.eq_ignore_ascii_case(query);
            b_exact.cmp(&a_exact)
        });
        results
    }

    /// Compute blast radius and affected files if a symbol is modified.
    pub fn compute_impact(&self, target_node_id: &str, max_depth: usize) -> Option<ImpactAnalysis> {
        let target_node = self.get_node(target_node_id)?.clone();

        let mut direct_callers = Vec::new();
        let mut indirect_callers = Vec::new();
        let mut affected_files_set = HashSet::new();
        affected_files_set.insert(target_node.file_path.clone());

        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        visited.insert(target_node_id.to_string());

        // Queue direct callers
        if let Some(edges) = self.incoming.get(target_node_id) {
            for edge in edges {
                if (edge.kind == EdgeKind::Calls
                    || edge.kind == EdgeKind::References
                    || edge.kind == EdgeKind::Implements)
                    && visited.insert(edge.source.clone())
                {
                    if let Some(caller) = self.get_node(&edge.source) {
                        direct_callers.push(caller.clone());
                        affected_files_set.insert(caller.file_path.clone());
                        queue.push_back((edge.source.clone(), 1));
                    }
                }
            }
        }

        // Trace indirect callers
        while let Some((curr_id, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }
            if let Some(edges) = self.incoming.get(&curr_id) {
                for edge in edges {
                    if (edge.kind == EdgeKind::Calls
                        || edge.kind == EdgeKind::References
                        || edge.kind == EdgeKind::Implements)
                        && visited.insert(edge.source.clone())
                    {
                        if let Some(caller) = self.get_node(&edge.source) {
                            indirect_callers.push(caller.clone());
                            affected_files_set.insert(caller.file_path.clone());
                            queue.push_back((edge.source.clone(), depth + 1));
                        }
                    }
                }
            }
        }

        let mut affected_files: Vec<String> = affected_files_set.into_iter().collect();
        affected_files.sort();

        Some(ImpactAnalysis {
            target: target_node,
            direct_callers,
            indirect_callers,
            affected_files,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAnalysis {
    pub target: Node,
    pub direct_callers: Vec<Node>,
    pub indirect_callers: Vec<Node>,
    pub affected_files: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_construction_and_subgraph() {
        let mut graph = KnowledgeGraph::new();

        let n1 = Node {
            id: "file:main.rs".into(),
            label: "main.rs".into(),
            kind: NodeKind::File,
            file_path: "src/main.rs".into(),
            start_line: 1,
            end_line: 100,
            signature: None,
            docstring: None,
            chunk_id: None,
            vector_id: None,
            metadata: HashMap::new(),
        };

        let n2 = Node {
            id: "fn:main.rs:run".into(),
            label: "run()".into(),
            kind: NodeKind::Function,
            file_path: "src/main.rs".into(),
            start_line: 10,
            end_line: 30,
            signature: Some("pub fn run()".into()),
            docstring: Some("Runs the vector engine".into()),
            chunk_id: Some("chunk-1".into()),
            vector_id: Some(1),
            metadata: HashMap::new(),
        };

        let n3 = Node {
            id: "fn:util.rs:calc".into(),
            label: "calc()".into(),
            kind: NodeKind::Function,
            file_path: "src/util.rs".into(),
            start_line: 5,
            end_line: 15,
            signature: Some("pub fn calc()".into()),
            docstring: None,
            chunk_id: Some("chunk-2".into()),
            vector_id: Some(2),
            metadata: HashMap::new(),
        };

        graph.add_node(n1);
        graph.add_node(n2);
        graph.add_node(n3);

        graph.add_edge(
            "file:main.rs".into(),
            "fn:main.rs:run".into(),
            EdgeKind::Contains,
            1.0,
        );
        graph.add_edge(
            "fn:main.rs:run".into(),
            "fn:util.rs:calc".into(),
            EdgeKind::Calls,
            1.0,
        );

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);

        let sub = graph.k_hop_subgraph(&["fn:main.rs:run".into()], 1);
        assert_eq!(sub.nodes.len(), 3); // run, file:main.rs (incoming), calc (outgoing)
        assert_eq!(sub.edges.len(), 2);
    }
}
