//! VectorDB HTTP server with Axum and embedded Web GUI.

use axum::{
    extract::{Path as AxumPath, Query, State},
    http::StatusCode,
    response::{Html, IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

use vectordb_core::{SearchResponse, Subgraph, VectorDBEngine};
use vectordb_parser::CodeParser;

const UI_HTML: &str = include_str!("ui.html");

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<VectorDBEngine>,
    pub db_dir: PathBuf,
}

#[derive(Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub limit: Option<usize>,
    pub expand_graph: Option<usize>,
}

#[derive(Deserialize)]
pub struct IndexRequest {
    pub path: String,
}

#[derive(Serialize)]
pub struct StatusResponse {
    pub total_chunks: usize,
    pub total_vectors: usize,
    pub total_nodes: usize,
    pub total_edges: usize,
    pub status: String,
}

#[derive(Serialize)]
pub struct IndexResponse {
    pub files_indexed: usize,
    pub chunks_indexed: usize,
    pub nodes_indexed: usize,
    pub edges_indexed: usize,
}

#[derive(Deserialize)]
pub struct LlmQuery {
    pub q: String,
    pub limit: Option<usize>,
}

pub struct VectorDBServer {
    engine: Arc<VectorDBEngine>,
    db_dir: PathBuf,
    port: u16,
}

impl VectorDBServer {
    pub fn new(engine: Arc<VectorDBEngine>, db_dir: PathBuf, port: u16) -> Self {
        Self {
            engine,
            db_dir,
            port,
        }
    }

    pub fn router(state: AppState) -> Router {
        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any);

        Router::new()
            .route("/", get(serve_ui))
            .route("/api/status", get(get_status))
            .route("/api/search", post(search))
            .route("/api/graph", get(get_graph))
            .route("/api/chunk/:id", get(get_chunk))
            .route("/api/index", post(index_directory))
            .route("/api/llm-context", get(get_llm_context))
            .layer(cors)
            .with_state(state)
    }

    pub async fn run(&self) -> anyhow::Result<()> {
        let state = AppState {
            engine: self.engine.clone(),
            db_dir: self.db_dir.clone(),
        };

        let app = Self::router(state);
        let addr = SocketAddr::from(([127, 0, 0, 1], self.port));
        log::info!("VectorDB Web GUI & REST API listening on http://{}", addr);

        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;
        Ok(())
    }
}

async fn serve_ui() -> Html<&'static str> {
    Html(UI_HTML)
}

async fn get_status(State(state): State<AppState>) -> Json<StatusResponse> {
    let (chunks, nodes, edges) = state.engine.stats();
    Json(StatusResponse {
        total_chunks: chunks,
        total_vectors: chunks,
        total_nodes: nodes,
        total_edges: edges,
        status: "healthy".into(),
    })
}

async fn search(
    State(state): State<AppState>,
    Json(payload): Json<SearchRequest>,
) -> Json<SearchResponse> {
    let limit = payload.limit.unwrap_or(10);
    let hops = payload.expand_graph.unwrap_or(1);
    let res = state.engine.search(&payload.query, limit, hops);
    Json(res)
}

async fn get_graph(State(state): State<AppState>) -> Json<Subgraph> {
    let nodes = state.engine.get_all_nodes();
    let edges = state.engine.get_all_edges();
    Json(Subgraph { nodes, edges })
}

async fn get_chunk(
    AxumPath(chunk_id): AxumPath<String>,
    State(state): State<AppState>,
) -> Response {
    if let Some(chunk) = state.engine.get_chunk(&chunk_id) {
        Json(chunk).into_response()
    } else {
        (StatusCode::NOT_FOUND, "Chunk not found").into_response()
    }
}

async fn index_directory(
    State(state): State<AppState>,
    Json(payload): Json<IndexRequest>,
) -> Result<Json<IndexResponse>, (StatusCode, String)> {
    let base_path = Path::new(&payload.path);
    if !base_path.is_dir() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("Index path is not a directory: {}", payload.path),
        ));
    }

    let parser = CodeParser::new();
    let parsed_files = parser.scan_directory(base_path).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to parse directory: {}", e),
        )
    })?;
    let files_count = parsed_files.len();

    let mut chunks_count = 0;
    let mut nodes_count = 0;
    let mut edges_count = 0;
    let mut pending_edges = Vec::new();

    let existing_paths = state.engine.indexed_file_paths();
    state.engine.remove_files_batch(
        &existing_paths
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
    );

    for file in parsed_files {
        // Collect node by chunk_id
        let mut node_by_chunk: std::collections::HashMap<String, vectordb_core::Node> =
            std::collections::HashMap::new();
        for node in file.nodes {
            if let Some(ref cid) = node.chunk_id {
                node_by_chunk.insert(cid.clone(), node);
            } else {
                state.engine.add_graph_node(node);
                nodes_count += 1;
            }
        }

        for chunk in file.chunks {
            let node = node_by_chunk.remove(&chunk.id);
            if node.is_some() {
                nodes_count += 1;
            }
            state.engine.add_chunk_and_node(chunk, node, None);
            chunks_count += 1;
        }

        pending_edges.extend(file.edges);
    }

    for edge in pending_edges {
        edges_count += state
            .engine
            .add_graph_edge(edge.source, edge.target, edge.kind, edge.weight)
            as usize;
    }

    state.engine.save_to_dir(&state.db_dir).map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to save index: {error}"),
        )
    })?;

    Ok(Json(IndexResponse {
        files_indexed: files_count,
        chunks_indexed: chunks_count,
        nodes_indexed: nodes_count,
        edges_indexed: edges_count,
    }))
}

async fn get_llm_context(Query(query): Query<LlmQuery>, State(state): State<AppState>) -> String {
    let limit = query.limit.unwrap_or(5);
    let search_res = state.engine.search(&query.q, limit, 1);
    state.engine.package_for_llm(&search_res)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn api_reindex_persists_text_only_snapshot() {
        let root = std::env::temp_dir().join(format!(
            "vectordb-api-reindex-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let corpus = root.join("corpus");
        let db_dir = root.join("db");
        std::fs::create_dir_all(&corpus).unwrap();
        let note = corpus.join("notes.txt");
        std::fs::write(&note, "persistent API evidence").unwrap();
        let state = AppState {
            engine: Arc::new(VectorDBEngine::new(256)),
            db_dir: db_dir.clone(),
        };
        let payload = || IndexRequest {
            path: corpus.to_string_lossy().into_owned(),
        };
        let first = index_directory(State(state.clone()), Json(payload()))
            .await
            .unwrap();
        assert_eq!(first.files_indexed, 1);
        assert_eq!(
            VectorDBEngine::load_from_dir(&db_dir, 256)
                .unwrap()
                .indexed_file_paths(),
            vec!["notes.txt"]
        );

        std::fs::write(&note, b"\x89PNG\r\n\x1a\n\0binary").unwrap();
        let second = index_directory(State(state), Json(payload()))
            .await
            .unwrap();
        assert_eq!(second.files_indexed, 0);
        assert!(VectorDBEngine::load_from_dir(&db_dir, 256)
            .unwrap()
            .indexed_file_paths()
            .is_empty());
        std::fs::remove_dir_all(root).unwrap();
    }
}
