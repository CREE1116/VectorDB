//! VectorDB CLI: Hybrid Vector & Code Graph Engine.
//! LLM Agent-friendly and human-friendly interface.

use clap::{Parser, Subcommand, ValueEnum};
use colored::*;
use indicatif::{ProgressBar, ProgressStyle};
use notify::{EventKind, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use vectordb_core::VectorDBEngine;
use vectordb_parser::CodeParser;
use vectordb_server::VectorDBServer;

mod ann_benchmark;
mod benchmark;
mod cold_benchmark;

const DEFAULT_DB_DIR: &str = ".vectordb";
const EMBED_DIM: usize = 256;

#[derive(Parser)]
#[command(name = "vectordb")]
#[command(about = "Fast, SIMD-accelerated Hybrid Vector & Code Graph Engine", long_about = None)]
#[command(version)]
struct Cli {
    #[arg(short, long, global = true, default_value = DEFAULT_DB_DIR)]
    db_dir: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, Copy, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
    Llm,
}

#[derive(Subcommand)]
enum Commands {
    /// Recursively scan and index source code and documents into Vector & Graph DB
    Index {
        /// Directory path to index
        #[arg(default_value = ".")]
        path: PathBuf,
    },

    /// Search the VectorDB using hybrid vector similarity and graph expansion
    Search {
        /// Natural language or code search query
        query: String,

        /// Number of results to return
        #[arg(short, long, default_value_t = 5)]
        limit: usize,

        /// Graph hop expansion depth (0 = pure vector search, 1+ = include called functions/types)
        #[arg(short, long, default_value_t = 1)]
        expand_graph: usize,

        /// Output format (text: pretty human-readable, json: structured for agents, llm: prompt context)
        #[arg(short, long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
    },

    /// Find indexed files with a similar binary byte fingerprint
    SimilarBinary {
        /// Indexed file path, relative to the indexed directory
        file: String,
        #[arg(short, long, default_value_t = 5)]
        limit: usize,
    },

    /// Evaluate BM25, dense, and hybrid retrieval against labeled JSON queries
    Benchmark { fixture: PathBuf },

    /// Measure HNSW recall and latency against an exact SIMD scan
    AnnBenchmark {
        #[arg(long, default_value_t = 1_000)]
        vectors: usize,
        #[arg(long, default_value_t = 256)]
        dimension: usize,
        #[arg(long, default_value_t = 100)]
        queries: usize,
        #[arg(long, default_value_t = 10)]
        k: usize,
        #[arg(long, default_value_t = 64)]
        ef_search: usize,
        #[arg(long, default_value_t = 16)]
        m: usize,
        #[arg(long, default_value_t = 64)]
        ef_construction: usize,
    },

    /// Measure snapshot build, save, load, and first-query time
    ColdBenchmark {
        #[arg(long, default_value_t = 1_000)]
        vectors: usize,
        #[arg(long, default_value_t = 5)]
        iterations: usize,
    },

    /// Fetch and display a specific code chunk by ID
    Chunk {
        /// Chunk ID (e.g. chunk:src/main.rs:10:45)
        id: String,
    },

    /// Inspect a symbol node and its relationships in the knowledge graph
    Graph {
        /// Node ID (e.g. symbol:crates/vectordb-core/src/hnsw.rs:HnswIndex)
        node_id: String,

        /// Traversal hops
        #[arg(short, long, default_value_t = 1)]
        hops: usize,
    },

    /// Launch the interactive Web GUI and REST API server
    Serve {
        /// Port to bind the server
        #[arg(short, long, default_value_t = 8080)]
        port: u16,

        /// Automatically open browser
        #[arg(long, default_value_t = false)]
        open: bool,
    },

    /// Locate and inspect a symbol (function, struct, class) across the codebase
    Find {
        /// Symbol name or partial pattern (e.g. compute_distance, HnswIndex)
        query: String,
    },

    /// Analyze blast radius and affected files if a symbol is modified (impact analysis)
    Impact {
        /// Target symbol name or node ID
        symbol: String,

        /// Maximum call depth
        #[arg(short, long, default_value_t = 2)]
        depth: usize,
    },

    /// Generate an all-in-one refactoring/modification context package for LLMs
    Context {
        /// Modification task description or symbol name
        task: String,
    },

    /// Watch workspace files and incrementally re-index on change in ~5ms
    Watch {
        /// Directory path to watch
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Debounce interval in milliseconds
        #[arg(short, long, default_value_t = 300)]
        debounce_ms: u64,
    },

    /// Analyze git co-change coupling to discover files historically edited together
    CoChange {
        /// File path to analyze (e.g. src/bm25.rs)
        file: String,

        /// Maximum commits to inspect
        #[arg(short, long, default_value_t = 200)]
        commits: usize,
    },

    /// Display VectorDB statistics
    Status,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Index { path } => handle_index(&cli.db_dir, &path),
        Commands::Search {
            query,
            limit,
            expand_graph,
            format,
        } => handle_search(&cli.db_dir, &query, limit, expand_graph, format),
        Commands::SimilarBinary { file, limit } => {
            let engine = load_or_create_engine(&cli.db_dir)?;
            let hits = engine.search_similar_binary(&file, limit);
            if hits.is_empty() {
                println!("No indexed binary matches for {file}. Reindex if this database predates feature spaces.");
            }
            for hit in hits {
                println!("{:.4}\t{}", hit.score, hit.chunk.file_path);
            }
            Ok(())
        }
        Commands::Benchmark { fixture } => benchmark::run(&fixture),
        Commands::AnnBenchmark {
            vectors,
            dimension,
            queries,
            k,
            ef_search,
            m,
            ef_construction,
        } => ann_benchmark::run(
            vectors,
            dimension,
            queries,
            k,
            ef_search,
            m,
            ef_construction,
        ),
        Commands::ColdBenchmark {
            vectors,
            iterations,
        } => cold_benchmark::run(vectors, iterations),
        Commands::Chunk { id } => handle_chunk(&cli.db_dir, &id),
        Commands::Graph { node_id, hops } => handle_graph(&cli.db_dir, &node_id, hops),
        Commands::Serve { port, open } => handle_serve(&cli.db_dir, port, open).await,
        Commands::Find { query } => handle_find(&cli.db_dir, &query),
        Commands::Impact { symbol, depth } => handle_impact(&cli.db_dir, &symbol, depth),
        Commands::Context { task } => handle_context(&cli.db_dir, &task),
        Commands::Watch { path, debounce_ms } => {
            handle_watch(&cli.db_dir, &path, debounce_ms).await
        }
        Commands::CoChange { file, commits } => handle_co_change(&file, commits),
        Commands::Status => handle_status(&cli.db_dir),
    }
}

fn load_or_create_engine(db_dir: &Path) -> anyhow::Result<VectorDBEngine> {
    if db_dir.exists() {
        VectorDBEngine::load_from_dir(db_dir, EMBED_DIM)
    } else {
        Ok(VectorDBEngine::new(EMBED_DIM))
    }
}

fn handle_index(db_dir: &Path, target_dir: &Path) -> anyhow::Result<()> {
    println!("{}", "🚀 Scanning and parsing codebase...".bold().cyan());

    let parser = CodeParser::new();
    let parsed_files = parser.scan_directory(target_dir)?;

    if parsed_files.is_empty() {
        println!(
            "{}",
            "⚠️  No supported source files found in target directory.".yellow()
        );
        return Ok(());
    }

    let pb = ProgressBar::new(parsed_files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} files ({msg})")?
            .progress_chars("#>-"),
    );

    let engine = load_or_create_engine(db_dir)?;
    engine.remove_files_batch(
        &parsed_files
            .iter()
            .map(|file| file.file_path.as_str())
            .collect::<Vec<_>>(),
    );

    let mut total_chunks = 0;
    let mut total_nodes = 0;
    let mut total_edges = 0;
    let mut pending_edges = Vec::new();

    for file in parsed_files {
        pb.set_message(file.file_path.clone());

        let mut node_by_chunk = std::collections::HashMap::new();
        for node in file.nodes {
            if let Some(ref cid) = node.chunk_id {
                node_by_chunk.insert(cid.clone(), node);
            } else {
                engine.add_graph_node(node);
                total_nodes += 1;
            }
        }

        for chunk in file.chunks {
            let node = node_by_chunk.remove(&chunk.id);
            if node.is_some() {
                total_nodes += 1;
            }
            engine.add_chunk_and_node(chunk, node, file.custom_vector.clone());
            total_chunks += 1;
        }

        pending_edges.extend(file.edges);

        pb.inc(1);
    }

    pb.finish_with_message("Done");

    for edge in pending_edges {
        if engine.add_graph_edge(edge.source, edge.target, edge.kind, edge.weight) {
            total_edges += 1;
        }
    }

    // Extract Git Co-Change coupling if in a git repository
    if let Ok(commits) = vectordb_parser::GitCoChangeAnalyzer::extract_history(target_dir, 300) {
        let git_edges = vectordb_parser::GitCoChangeAnalyzer::build_co_change_edges(&commits, 1);
        if !git_edges.is_empty() {
            println!(
                "{}",
                format!(
                    "🔗 Integrating {} Git Co-change evolutionary coupling edges...",
                    git_edges.len()
                )
                .cyan()
            );
            for (file_a, file_b, jaccard) in git_edges {
                let node_a = format!("file:{}", file_a);
                let node_b = format!("file:{}", file_b);
                total_edges += engine.add_graph_edge(
                    node_a.clone(),
                    node_b.clone(),
                    vectordb_core::EdgeKind::CoChangedWith,
                    jaccard,
                ) as usize;
                total_edges += engine.add_graph_edge(
                    node_b,
                    node_a,
                    vectordb_core::EdgeKind::CoChangedWith,
                    jaccard,
                ) as usize;
            }
        }
    }

    println!("{}", "💾 Persisting database snapshot to disk...".dimmed());
    engine.save_to_dir(db_dir)?;

    println!("{}", "✅ Indexing successfully completed!".bold().green());
    println!(
        "   📁 DB Directory:    {}",
        db_dir.display().to_string().cyan()
    );
    println!("   🧩 Chunks indexed:  {}", total_chunks.to_string().bold());
    println!("   🌐 Graph Nodes:     {}", total_nodes.to_string().bold());
    println!("   🔗 Graph Edges:     {}", total_edges.to_string().bold());

    Ok(())
}

fn handle_search(
    db_dir: &Path,
    query: &str,
    limit: usize,
    expand_graph: usize,
    format: OutputFormat,
) -> anyhow::Result<()> {
    let engine = load_or_create_engine(db_dir)?;
    let response = engine.search(query, limit, expand_graph);

    match format {
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&response)?;
            println!("{}", json);
        }
        OutputFormat::Llm => {
            let context = engine.package_for_llm(&response);
            print!("{}", context);
        }
        OutputFormat::Text => {
            println!(
                "{}",
                "═══════════════════════════════════════════════════════════════".dimmed()
            );
            println!(
                "{} \"{}\" (Found {} hits)",
                "🔍 Hybrid Search Results for:".bold().cyan(),
                query.yellow(),
                response.hits.len()
            );
            println!(
                "{}",
                "═══════════════════════════════════════════════════════════════".dimmed()
            );

            if response.hits.is_empty() {
                println!(
                    "{}",
                    "No relevant code snippets or documentation found.".dimmed()
                );
                return Ok(());
            }

            for (i, hit) in response.hits.iter().enumerate() {
                let score_pct = (hit.score * 100.0).clamp(0.0, 100.0);
                println!(
                    "\n{} [{}] {} {} (Lines {}-{})",
                    "●".cyan(),
                    i + 1,
                    hit.chunk.file_path.bold().white(),
                    format!("[{:.1}% match]", score_pct).green(),
                    hit.chunk.start_line,
                    hit.chunk.end_line
                );

                if let Some(ref node) = hit.node {
                    println!(
                        "  {} {} ({:?})",
                        "Symbol:".dimmed(),
                        node.label.bold().yellow(),
                        node.kind
                    );
                    if let Some(ref sig) = node.signature {
                        println!("  {} {}", "Signature:".dimmed(), sig.cyan());
                    }
                }

                if !hit.related_nodes.is_empty() {
                    let connected: Vec<String> = hit
                        .related_nodes
                        .iter()
                        .map(|n| format!("{}({:?})", n.label, n.kind))
                        .collect();
                    println!(
                        "  {} {}",
                        "Connected:".dimmed(),
                        connected.join(", ").blue()
                    );
                }

                println!("  {}", "─".repeat(50).dimmed());
                for line in hit.chunk.content.lines().take(6) {
                    println!("  │ {}", line.dimmed());
                }
                if hit.chunk.content.lines().count() > 6 {
                    println!("  │ {}", "...".dimmed());
                }
            }
            println!();
        }
    }

    Ok(())
}

fn handle_chunk(db_dir: &Path, id: &str) -> anyhow::Result<()> {
    let engine = load_or_create_engine(db_dir)?;
    if let Some(chunk) = engine.get_chunk(id) {
        println!(
            "{}",
            format!(
                "File: {} (Lines {}-{})",
                chunk.file_path, chunk.start_line, chunk.end_line
            )
            .bold()
            .cyan()
        );
        println!("{}", "─".repeat(60).dimmed());
        println!("{}", chunk.content);
    } else {
        println!("{}", format!("Chunk not found: {}", id).red());
    }
    Ok(())
}

fn handle_graph(db_dir: &Path, node_id: &str, hops: usize) -> anyhow::Result<()> {
    let engine = load_or_create_engine(db_dir)?;
    let subgraph = engine.get_subgraph(&[node_id.to_string()], hops);

    println!(
        "{}",
        format!("🌐 Subgraph for '{}' (hops = {})", node_id, hops)
            .bold()
            .cyan()
    );
    println!("Nodes ({}):", subgraph.nodes.len());
    for n in &subgraph.nodes {
        println!(
            "  - [{:?}] {} ({})",
            n.kind,
            n.label.bold(),
            n.file_path.dimmed()
        );
    }

    println!("\nEdges ({}):", subgraph.edges.len());
    for e in &subgraph.edges {
        println!(
            "  {} --[{}]--> {}",
            e.source.dimmed(),
            e.kind.to_string().yellow(),
            e.target.dimmed()
        );
    }

    Ok(())
}

async fn handle_serve(db_dir: &Path, port: u16, open: bool) -> anyhow::Result<()> {
    let engine = Arc::new(load_or_create_engine(db_dir)?);
    let (chunks, nodes, edges) = engine.stats();

    println!(
        "{}",
        "===============================================================".cyan()
    );
    println!(
        "{}",
        "   ⚡ VectorDB Interactive Web GUI & REST API Server ⚡"
            .bold()
            .green()
    );
    println!(
        "{}",
        "===============================================================".cyan()
    );
    println!(
        "   🌐 Web GUI URL:    {}",
        format!("http://localhost:{}", port)
            .bold()
            .underline()
            .blue()
    );
    println!("   📊 Vectors/Chunks: {}", chunks.to_string().bold());
    println!("   🧬 Graph Nodes:    {}", nodes.to_string().bold());
    println!("   🔗 Graph Edges:    {}", edges.to_string().bold());
    println!(
        "{}",
        "---------------------------------------------------------------".dimmed()
    );
    println!("Press Ctrl+C to stop the server\n");

    if open {
        let _ = open_browser(&format!("http://localhost:{}", port));
    }

    let server = VectorDBServer::new(engine, port);
    server.run().await?;
    Ok(())
}

fn handle_find(db_dir: &Path, query: &str) -> anyhow::Result<()> {
    let engine = load_or_create_engine(db_dir)?;
    let matches = engine.find_symbols(query);

    println!(
        "{}",
        "═══════════════════════════════════════════════════════════════".dimmed()
    );
    println!(
        "{} \"{}\" (Found {} symbols)",
        "🔍 Symbols matching:".bold().cyan(),
        query.yellow(),
        matches.len()
    );
    println!(
        "{}",
        "═══════════════════════════════════════════════════════════════".dimmed()
    );

    if matches.is_empty() {
        println!("{}", "No matching symbols found.".dimmed());
        return Ok(());
    }

    for (i, node) in matches.iter().enumerate() {
        println!(
            "\n{} [{}] {} {} ({:?})",
            "●".cyan(),
            i + 1,
            node.label.bold().yellow(),
            format!("{}:{}", node.file_path, node.start_line)
                .white()
                .dimmed(),
            node.kind
        );
        if let Some(ref sig) = node.signature {
            println!("  {} {}", "Signature:".dimmed(), sig.cyan());
        }
        if let Some(ref doc) = node.docstring {
            println!("  {} {}", "Docstring:".dimmed(), doc.dimmed());
        }
        println!("  {} {}", "Node ID:  ".dimmed(), node.id.blue());
    }
    println!();
    Ok(())
}

fn handle_impact(db_dir: &Path, symbol: &str, depth: usize) -> anyhow::Result<()> {
    let engine = load_or_create_engine(db_dir)?;
    let target_node = if let Some(n) = engine.get_all_nodes().iter().find(|n| n.id == symbol) {
        n.clone()
    } else {
        let matches = engine.find_symbols(symbol);
        if matches.is_empty() {
            println!("{}", format!("No symbol found matching '{}'", symbol).red());
            return Ok(());
        }
        matches[0].clone()
    };

    let impact = match engine.compute_impact(&target_node.id, depth) {
        Some(imp) => imp,
        None => {
            println!(
                "{}",
                format!("Could not analyze impact for '{}'", target_node.id).red()
            );
            return Ok(());
        }
    };

    println!(
        "{}",
        "═══════════════════════════════════════════════════════════════".dimmed()
    );
    println!(
        "{} {} ({:?})",
        "💥 Blast Radius / Impact Analysis for:".bold().red(),
        impact.target.label.bold().yellow(),
        impact.target.kind
    );
    println!(
        "   {} {}:{}",
        "Defined in:".dimmed(),
        impact.target.file_path.cyan(),
        impact.target.start_line
    );
    println!(
        "{}",
        "═══════════════════════════════════════════════════════════════".dimmed()
    );

    println!(
        "\n{} ({})",
        "🎯 Direct Callers (Depth 1):".bold().white(),
        impact.direct_callers.len()
    );
    if impact.direct_callers.is_empty() {
        println!("  (No direct callers recorded in graph)");
    } else {
        for c in &impact.direct_callers {
            println!(
                "  • {} in {}:{} ({:?})",
                c.label.yellow(),
                c.file_path.dimmed(),
                c.start_line,
                c.kind
            );
        }
    }

    println!(
        "\n{} ({})",
        "🌊 Indirect Callers (Depth 2+):".bold().white(),
        impact.indirect_callers.len()
    );
    if impact.indirect_callers.is_empty() {
        println!("  (No transitive callers)");
    } else {
        for c in &impact.indirect_callers {
            println!(
                "  • {} in {}:{} ({:?})",
                c.label.yellow(),
                c.file_path.dimmed(),
                c.start_line,
                c.kind
            );
        }
    }

    println!(
        "\n{} ({})",
        "📁 Affected Files to Review/Test:".bold().green(),
        impact.affected_files.len()
    );
    for f in &impact.affected_files {
        println!("  ✓ {}", f.cyan());
    }

    // Check Git Co-change history
    if let Ok(commits) = vectordb_parser::GitCoChangeAnalyzer::extract_history(Path::new("."), 200)
    {
        let co_change =
            vectordb_parser::GitCoChangeAnalyzer::analyze_file(&commits, &impact.target.file_path);
        if !co_change.candidates.is_empty() {
            println!(
                "\n{} ({})",
                "🔗 Git Co-Change Coupling (Historically edited together):"
                    .bold()
                    .magenta(),
                co_change.candidates.len().min(5)
            );
            for cand in co_change.candidates.iter().take(5) {
                println!(
                    "  • {} (co-changed in {} commits, {:.0}% confidence)",
                    cand.file_path.cyan(),
                    cand.co_change_count,
                    cand.confidence * 100.0
                );
            }
        }
    }
    println!();

    Ok(())
}

fn handle_context(db_dir: &Path, task: &str) -> anyhow::Result<()> {
    let engine = load_or_create_engine(db_dir)?;
    let search_res = engine.search(task, 4, 1);

    println!("# Refactoring & Modification Context for Task\n");
    println!("**Task Description**: `{}`\n", task);

    if search_res.hits.is_empty() {
        println!("No relevant code symbols found for task.");
        return Ok(());
    }

    println!("## Key Target Symbols & Files\n");
    for hit in &search_res.hits {
        let label = hit
            .node
            .as_ref()
            .map(|n| n.label.as_str())
            .unwrap_or("chunk");
        println!(
            "- `{}` in `{}:{}` (Relevance: {:.1}%)",
            label,
            hit.chunk.file_path,
            hit.chunk.start_line,
            hit.score * 100.0
        );
        if let Some(ref n) = hit.node {
            if let Some(impact) = engine.compute_impact(&n.id, 2) {
                if !impact.affected_files.is_empty() && impact.affected_files.len() > 1 {
                    println!(
                        "  * Potential Blast Radius: {} affected files ({})",
                        impact.affected_files.len(),
                        impact.affected_files.join(", ")
                    );
                }
            }
        }
    }

    println!("\n---\n");
    print!("{}", engine.package_for_llm(&search_res));

    Ok(())
}

fn handle_status(db_dir: &Path) -> anyhow::Result<()> {
    if !db_dir.exists() {
        println!(
            "{}",
            "No VectorDB found in current directory. Run 'vectordb index' first.".yellow()
        );
        return Ok(());
    }

    let engine = load_or_create_engine(db_dir)?;
    let (chunks, nodes, edges) = engine.stats();

    println!("{}", "📊 VectorDB Status:".bold().cyan());
    println!("   📁 Path:           {}", db_dir.display());
    println!("   🧩 Total Chunks:   {}", chunks.to_string().bold());
    println!("   🌐 Graph Nodes:    {}", nodes.to_string().bold());
    println!("   🔗 Graph Edges:    {}", edges.to_string().bold());
    println!("   ⚡ SIMD Engine:    ARM NEON / AVX2 Accelerated");
    println!("   🧠 Vector Dim:     {} dimensions", EMBED_DIM);
    Ok(())
}

fn open_browser(url: &str) -> std::io::Result<std::process::ExitStatus> {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open").arg(url).status()
    }
    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open").arg(url).status()
    }
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("cmd")
            .args(["/C", "start", url])
            .status()
    }
}

fn simple_time() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let sec = now % 60;
    let min = (now / 60) % 60;
    let hour = (now / 3600) % 24;
    format!("{:02}:{:02}:{:02}", hour, min, sec)
}

fn handle_co_change(target_file: &str, max_commits: usize) -> anyhow::Result<()> {
    let repo_root = Path::new(".");
    let commits = vectordb_parser::GitCoChangeAnalyzer::extract_history(repo_root, max_commits)?;
    if commits.is_empty() {
        println!(
            "{}",
            "⚠️  No git commit history found or not a git repository.".yellow()
        );
        return Ok(());
    }

    let report = vectordb_parser::GitCoChangeAnalyzer::analyze_file(&commits, target_file);

    println!(
        "{}",
        "═══════════════════════════════════════════════════════════════".dimmed()
    );
    println!(
        "{} {}",
        "🔗 Git Co-Change History Analysis for:".bold().cyan(),
        report.target_file.yellow()
    );
    println!(
        "   {} {} commits (Analyzed last {} commits)",
        "Found in:".dimmed(),
        report.target_commit_count.to_string().bold(),
        report.total_commits_analyzed
    );
    println!(
        "{}",
        "═══════════════════════════════════════════════════════════════".dimmed()
    );

    if report.candidates.is_empty() {
        println!(
            "{}",
            "No co-changed files detected for this file in recent commits.".dimmed()
        );
        return Ok(());
    }

    for (i, cand) in report.candidates.iter().enumerate() {
        println!(
            "\n{} [{}] {} (Co-changed in {} commits, {:.1}% confidence)",
            "●".cyan(),
            i + 1,
            cand.file_path.bold().white(),
            cand.co_change_count,
            cand.confidence * 100.0
        );
        if !cand.recent_commits.is_empty() {
            for c in &cand.recent_commits {
                println!("     ↳ {}", c.dimmed());
            }
        }
    }
    println!();

    Ok(())
}

type FileStamp = (std::time::SystemTime, u64);

fn watch_path_allowed(root: &Path, path: &Path) -> bool {
    let Ok(relative) = path.strip_prefix(root) else {
        return false;
    };
    !relative.components().any(|part| {
        let name = part.as_os_str().to_string_lossy();
        name.starts_with('.') || matches!(name.as_ref(), "target" | "node_modules")
    })
}

fn scan_watch_path(
    root: &Path,
    path: &Path,
    parser: &CodeParser,
) -> std::collections::HashMap<PathBuf, FileStamp> {
    let mut files = std::collections::HashMap::new();
    if !watch_path_allowed(root, path) {
        return files;
    }
    for entry in walkdir::WalkDir::new(path)
        .into_iter()
        .filter_entry(|entry| watch_path_allowed(root, entry.path()))
        .filter_map(Result::ok)
    {
        let file = entry.path();
        if file.is_file() && parser.is_supported(file) {
            if let Ok(meta) = file.metadata() {
                files.insert(
                    file.to_path_buf(),
                    (
                        meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH),
                        meta.len(),
                    ),
                );
            }
        }
    }
    files
}

async fn handle_watch(db_dir: &Path, target_dir: &Path, debounce_ms: u64) -> anyhow::Result<()> {
    let root = target_dir.canonicalize()?;
    println!(
        "👁️  Watching {} ({} ms debounce). Press Ctrl+C to stop.",
        root.display(),
        debounce_ms
    );
    let parser = CodeParser::new();
    let engine = load_or_create_engine(db_dir)?;
    let mut file_snapshots = scan_watch_path(&root, &root, &parser);
    println!("👀 Tracking {} files", file_snapshots.len());

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let mut watcher = notify::recommended_watcher(move |event| {
        let _ = tx.send(event);
    })?;
    watcher.watch(&root, RecursiveMode::Recursive)?;

    while let Some(first) = rx.recv().await {
        // Group bursts from editors, file moves, and build tools into one update.
        let mut events = vec![first];
        tokio::time::sleep(std::time::Duration::from_millis(debounce_ms)).await;
        while let Ok(event) = rx.try_recv() {
            events.push(event);
        }

        let mut rescan = false;
        let mut paths = std::collections::HashSet::new();
        let mut direct_files = std::collections::HashSet::new();
        for event in events {
            match event {
                Ok(event) if !matches!(event.kind, EventKind::Access(_)) => {
                    if event.need_rescan() || event.paths.is_empty() {
                        rescan = true;
                    }
                    for path in event.paths {
                        let absolute = if path.is_absolute() {
                            path
                        } else {
                            root.join(path)
                        };
                        if watch_path_allowed(&root, &absolute) {
                            if absolute.is_file() {
                                direct_files.insert(absolute.clone());
                            }
                            paths.insert(absolute);
                        }
                    }
                }
                Err(error) => {
                    eprintln!("Watcher event error: {error}; rescanning");
                    rescan = true;
                }
                _ => {}
            }
        }
        if rescan {
            paths.clear();
            paths.insert(root.clone());
        }
        if paths.is_empty() {
            continue;
        }

        let mut observed = std::collections::HashMap::new();
        let mut candidates = std::collections::HashSet::new();
        for path in &paths {
            for old in file_snapshots.keys().filter(|old| old.starts_with(path)) {
                candidates.insert(old.clone());
            }
            let found = scan_watch_path(&root, path, &parser);
            candidates.extend(found.keys().cloned());
            observed.extend(found);
        }
        let mut parsed = Vec::new();
        let mut deleted = Vec::new();
        for path in candidates {
            if let Some(&stamp) = observed.get(&path) {
                if direct_files.contains(&path) || file_snapshots.get(&path) != Some(&stamp) {
                    let relative = path.strip_prefix(&root)?.to_string_lossy().to_string();
                    match parser.parse_file(&path, &relative) {
                        Ok(file) => parsed.push((path, stamp, file)),
                        Err(error) => eprintln!("Could not parse {relative}: {error}"),
                    }
                }
            } else if file_snapshots.contains_key(&path) {
                deleted.push(path);
            }
        }
        if parsed.is_empty() && deleted.is_empty() {
            continue;
        }
        let removal_paths: Vec<String> = parsed
            .iter()
            .map(|(_, _, file)| file.file_path.clone())
            .chain(deleted.iter().map(|path| {
                path.strip_prefix(&root)
                    .unwrap_or(path)
                    .to_string_lossy()
                    .to_string()
            }))
            .collect();
        engine.remove_files_batch(&removal_paths.iter().map(String::as_str).collect::<Vec<_>>());

        for (path, stamp, file) in parsed {
            let relative = file.file_path.clone();
            let mut node_by_chunk = std::collections::HashMap::new();
            for node in file.nodes {
                if let Some(ref id) = node.chunk_id {
                    node_by_chunk.insert(id.clone(), node);
                } else {
                    engine.add_graph_node(node);
                }
            }
            let chunks = file.chunks.len();
            for chunk in file.chunks {
                let node = node_by_chunk.remove(&chunk.id);
                engine.add_chunk_and_node(chunk, node, file.custom_vector.clone());
            }
            for edge in file.edges {
                engine.add_graph_edge(edge.source, edge.target, edge.kind, edge.weight);
            }
            file_snapshots.insert(path, stamp);
            println!(
                "⚡ [{}] Re-indexed: {} ({} chunks)",
                simple_time(),
                relative,
                chunks
            );
        }
        for path in deleted {
            file_snapshots.remove(&path);
            println!("🗑️  [{}] Removed: {}", simple_time(), path.display());
        }
        if let Err(error) = engine.save_to_dir(db_dir) {
            eprintln!("Failed to save database: {error}");
        }
    }
    anyhow::bail!("filesystem watcher stopped")
}
