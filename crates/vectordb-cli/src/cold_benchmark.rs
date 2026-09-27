//! Measures in-process snapshot load and the first query after loading.

use serde::Serialize;
use std::time::{Duration, Instant};
use vectordb_core::{Chunk, VectorDBEngine};

#[derive(Serialize)]
struct Report {
    vectors: usize,
    iterations: usize,
    snapshot_bytes: u64,
    build_ms: f64,
    save_ms: f64,
    load_p50_ms: f64,
    load_p95_ms: f64,
    query_p50_ms: f64,
    query_p95_ms: f64,
    total_p50_ms: f64,
    total_p95_ms: f64,
    cli_e2e_p50_ms: f64,
    cli_e2e_p95_ms: f64,
}

fn percentile(samples: &mut [Duration], fraction: f64) -> f64 {
    samples.sort_unstable();
    let index = ((samples.len() - 1) as f64 * fraction).ceil() as usize;
    samples[index].as_secs_f64() * 1_000.0
}

pub fn run(count: usize, iterations: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        count > 0 && iterations > 0,
        "vectors and iterations must be positive"
    );
    let dir = std::env::temp_dir().join(format!(
        "vectordb-cold-bench-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    let result = (|| -> anyhow::Result<Report> {
        let engine = VectorDBEngine::new(256);
        let started = Instant::now();
        for id in 0..count {
            let content = format!("marker{id} vector search document content");
            engine.add_chunk_and_node(
                Chunk {
                    id: format!("chunk:{id}"),
                    file_path: format!("docs/{id}.md"),
                    start_line: 1,
                    end_line: 1,
                    tokens_approx: 5,
                    content,
                    summary: None,
                    node_id: None,
                    vector_id: 0,
                },
                None,
                None,
            );
        }
        let build_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let started = Instant::now();
        engine.save_to_dir(&dir)?;
        let save_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let snapshot_bytes = std::fs::metadata(dir.join("metadata.bin"))?.len();
        drop(engine);

        let mut loads = Vec::with_capacity(iterations);
        let mut queries = Vec::with_capacity(iterations);
        let mut totals = Vec::with_capacity(iterations);
        let mut cli_totals = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let started = Instant::now();
            let loaded = VectorDBEngine::load_from_dir(&dir, 256)?;
            let load_time = started.elapsed();
            let query_started = Instant::now();
            let response = loaded.search("marker0", 5, 0);
            let query_time = query_started.elapsed();
            anyhow::ensure!(!response.hits.is_empty(), "cold query returned no results");
            loads.push(load_time);
            queries.push(query_time);
            totals.push(started.elapsed());
        }
        let executable = std::env::current_exe()?;
        for _ in 0..iterations {
            let started = Instant::now();
            let status = std::process::Command::new(&executable)
                .arg("--db-dir")
                .arg(&dir)
                .args([
                    "search",
                    "marker0",
                    "--limit",
                    "5",
                    "--expand-graph",
                    "0",
                    "--format",
                    "json",
                ])
                .stdout(std::process::Stdio::null())
                .status()?;
            anyhow::ensure!(status.success(), "cold CLI search failed");
            cli_totals.push(started.elapsed());
        }
        Ok(Report {
            vectors: count,
            iterations,
            snapshot_bytes,
            build_ms,
            save_ms,
            load_p50_ms: percentile(&mut loads.clone(), 0.5),
            load_p95_ms: percentile(&mut loads, 0.95),
            query_p50_ms: percentile(&mut queries.clone(), 0.5),
            query_p95_ms: percentile(&mut queries, 0.95),
            total_p50_ms: percentile(&mut totals.clone(), 0.5),
            total_p95_ms: percentile(&mut totals, 0.95),
            cli_e2e_p50_ms: percentile(&mut cli_totals.clone(), 0.5),
            cli_e2e_p95_ms: percentile(&mut cli_totals, 0.95),
        })
    })();
    let _ = std::fs::remove_dir_all(&dir);
    println!("{}", serde_json::to_string_pretty(&result?)?);
    Ok(())
}
