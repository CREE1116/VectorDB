//! HNSW versus exact SIMD scan at one corpus size.

use serde::Serialize;
use std::time::{Duration, Instant};
use vectordb_core::{cosine_distance, DistanceMetric, HnswConfig, HnswIndex};

#[derive(Serialize)]
struct Report {
    seed: u64,
    vectors: usize,
    dimension: usize,
    queries: usize,
    k: usize,
    ef_search: usize,
    m: usize,
    ef_construction: usize,
    build_ms: f64,
    recall_at_k: f64,
    exact_p50_us: f64,
    exact_p95_us: f64,
    hnsw_p50_us: f64,
    hnsw_p95_us: f64,
}

fn next_u32(state: &mut u64) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 32) as u32
}

fn normalized_vector(dim: usize, state: &mut u64) -> Vec<f32> {
    let mut vector: Vec<f32> = (0..dim)
        .map(|_| next_u32(state) as f32 / u32::MAX as f32 - 0.5)
        .collect();
    vectordb_core::simd::normalize_in_place(&mut vector);
    vector
}

fn percentile(samples: &mut [Duration], fraction: f64) -> f64 {
    samples.sort_unstable();
    let index = ((samples.len() - 1) as f64 * fraction).ceil() as usize;
    samples[index].as_secs_f64() * 1_000_000.0
}

pub fn run(
    count: usize,
    dim: usize,
    query_count: usize,
    k: usize,
    ef_search: usize,
    m: usize,
    ef_construction: usize,
) -> anyhow::Result<()> {
    const SEED: u64 = 0x6fa2_74b1_19cd_8e37;
    if count == 0
        || dim == 0
        || query_count == 0
        || k == 0
        || k > count
        || ef_search == 0
        || m < 2
        || ef_construction == 0
    {
        anyhow::bail!(
            "vectors, dimension, queries, k, and ef-search must be positive; k <= vectors"
        );
    }
    let mut state = SEED;
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|_| normalized_vector(dim, &mut state))
        .collect();
    let mut queries = Vec::with_capacity(query_count);
    for i in 0..query_count {
        let mut q = vectors[(i * count / query_count).min(count - 1)].clone();
        let noise = normalized_vector(dim, &mut state);
        for (value, delta) in q.iter_mut().zip(noise) {
            *value += delta * 0.05;
        }
        vectordb_core::simd::normalize_in_place(&mut q);
        queries.push(q);
    }
    let config = HnswConfig {
        m,
        m0: m.saturating_mul(2),
        ef_construction,
        ef_search,
        metric: DistanceMetric::Cosine,
    };
    let mut index = HnswIndex::new_seeded(dim, config, SEED ^ 0x9387_24ad_61bf_2019);
    let build_start = Instant::now();
    for (id, vector) in vectors.iter().enumerate() {
        index.insert(id as u64, vector.clone());
    }
    let build_ms = build_start.elapsed().as_secs_f64() * 1_000.0;

    let mut exact_times = Vec::with_capacity(query_count);
    let mut ann_times = Vec::with_capacity(query_count);
    let mut matched = 0usize;
    for query in &queries {
        let started = Instant::now();
        let mut exact: Vec<_> = vectors
            .iter()
            .enumerate()
            .map(|(id, vector)| (id as u64, cosine_distance(query, vector)))
            .collect();
        let by_distance =
            |a: &(u64, f32), b: &(u64, f32)| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0));
        if k < exact.len() {
            exact.select_nth_unstable_by(k - 1, by_distance);
        }
        exact.truncate(k);
        exact.sort_by(by_distance);
        exact_times.push(started.elapsed());

        let started = Instant::now();
        let found = index.search(query, k, Some(ef_search));
        ann_times.push(started.elapsed());
        let gold: std::collections::HashSet<_> = exact.iter().map(|(id, _)| *id).collect();
        matched += found.iter().filter(|hit| gold.contains(&hit.id)).count();
        std::hint::black_box(found);
    }
    let report = Report {
        seed: SEED,
        vectors: count,
        dimension: dim,
        queries: query_count,
        k,
        ef_search,
        m,
        ef_construction,
        build_ms,
        recall_at_k: matched as f64 / (query_count * k) as f64,
        exact_p50_us: percentile(&mut exact_times.clone(), 0.5),
        exact_p95_us: percentile(&mut exact_times, 0.95),
        hnsw_p50_us: percentile(&mut ann_times.clone(), 0.5),
        hnsw_p95_us: percentile(&mut ann_times, 0.95),
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
