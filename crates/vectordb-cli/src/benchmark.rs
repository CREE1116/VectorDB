//! Small, labeled retrieval evaluation. No model calls or external datasets.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use vectordb_core::{Chunk, SearchMode, VectorDBEngine};

#[derive(Deserialize)]
struct Fixture {
    documents: Vec<Document>,
    queries: Vec<Query>,
}

#[derive(Deserialize)]
struct Document {
    id: String,
    path: String,
    content: String,
    #[serde(default)]
    fingerprint_byte: Option<u8>,
}

#[derive(Deserialize)]
struct Query {
    id: String,
    kind: String,
    text: String,
    relevance: HashMap<String, u8>,
}

#[derive(Serialize)]
struct ResultRow {
    mode: &'static str,
    kind: String,
    queries: usize,
    recall_at_5: f64,
    recall_at_10: f64,
    mrr: f64,
    ndcg_at_10: f64,
    hit_at_1: f64,
}

#[derive(Default)]
struct Totals {
    count: usize,
    recall_5: f64,
    recall_10: f64,
    mrr: f64,
    ndcg_10: f64,
    hit_1: f64,
}

fn score(ids: &[String], relevant: &HashMap<String, u8>) -> (f64, f64, f64, f64, f64) {
    let relevant_count = relevant.values().filter(|&&grade| grade > 0).count();
    if relevant_count == 0 {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let recall = |k| {
        ids.iter()
            .take(k)
            .filter(|id| relevant.get(*id).copied().unwrap_or(0) > 0)
            .count() as f64
            / relevant_count as f64
    };
    let first = ids
        .iter()
        .position(|id| relevant.get(id).copied().unwrap_or(0) > 0);
    let mrr = first.map_or(0.0, |rank| 1.0 / (rank + 1) as f64);
    let dcg = |grades: &[u8]| {
        grades
            .iter()
            .take(10)
            .enumerate()
            .map(|(rank, &grade)| ((1u32 << grade.min(20)) - 1) as f64 / (rank as f64 + 2.0).log2())
            .sum::<f64>()
    };
    let actual: Vec<u8> = ids
        .iter()
        .take(10)
        .map(|id| relevant.get(id).copied().unwrap_or(0))
        .collect();
    let mut ideal: Vec<u8> = relevant.values().copied().collect();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let ideal_dcg = dcg(&ideal);
    (
        recall(5),
        recall(10),
        mrr,
        if ideal_dcg > 0.0 {
            dcg(&actual) / ideal_dcg
        } else {
            0.0
        },
        f64::from(first == Some(0)),
    )
}

pub fn run(path: &Path) -> anyhow::Result<()> {
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(path)?)?;
    if fixture.documents.is_empty() || fixture.queries.is_empty() {
        anyhow::bail!("benchmark fixture needs documents and queries");
    }
    let engine = VectorDBEngine::new(256);
    let mut seen = std::collections::HashSet::new();
    for doc in &fixture.documents {
        if !seen.insert(&doc.id) {
            anyhow::bail!("duplicate document id: {}", doc.id);
        }
        let vector = doc.fingerprint_byte.map(|byte| {
            let mut histogram = vec![0.0; 256];
            histogram[byte as usize] = 1.0;
            histogram
        });
        engine.add_chunk_and_node(
            Chunk {
                id: doc.id.clone(),
                file_path: doc.path.clone(),
                start_line: 1,
                end_line: 1,
                content: doc.content.clone(),
                summary: None,
                node_id: None,
                vector_id: 0,
                tokens_approx: doc.content.split_whitespace().count(),
            },
            None,
            vector,
        );
    }
    for query in &fixture.queries {
        if query.relevance.is_empty() || query.relevance.keys().any(|id| !seen.contains(id)) {
            anyhow::bail!("query {} has missing or unknown relevance labels", query.id);
        }
    }

    let mut rows = Vec::new();
    for (name, mode) in [
        ("bm25", SearchMode::Bm25),
        ("dense", SearchMode::Dense),
        ("hybrid", SearchMode::Hybrid),
    ] {
        let mut totals: HashMap<&str, Totals> = HashMap::new();
        for query in &fixture.queries {
            let response = engine.search_with_mode(&query.text, 10, 0, mode);
            let ids: Vec<_> = response.hits.into_iter().map(|hit| hit.chunk.id).collect();
            let (r5, r10, mrr, ndcg, hit1) = score(&ids, &query.relevance);
            let total = totals.entry(&query.kind).or_default();
            total.count += 1;
            total.recall_5 += r5;
            total.recall_10 += r10;
            total.mrr += mrr;
            total.ndcg_10 += ndcg;
            total.hit_1 += hit1;
        }
        for (kind, total) in totals {
            let n = total.count as f64;
            rows.push(ResultRow {
                mode: name,
                kind: kind.into(),
                queries: total.count,
                recall_at_5: total.recall_5 / n,
                recall_at_10: total.recall_10 / n,
                mrr: total.mrr / n,
                ndcg_at_10: total.ndcg_10 / n,
                hit_at_1: total.hit_1 / n,
            });
        }
    }
    rows.sort_by(|a, b| a.kind.cmp(&b.kind).then_with(|| a.mode.cmp(b.mode)));
    println!("{}", serde_json::to_string_pretty(&rows)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graded_metrics_are_normalized() {
        let relevant = HashMap::from([("a".into(), 3), ("b".into(), 1)]);
        let (r5, r10, mrr, ndcg, hit1) = score(&["a".into(), "b".into()], &relevant);
        assert_eq!((r5, r10, mrr, hit1), (1.0, 1.0, 1.0, 1.0));
        assert!((ndcg - 1.0).abs() < 1e-12);
        let (_, _, mrr, _, hit1) = score(&["noise".into(), "a".into()], &relevant);
        assert_eq!((mrr, hit1), (0.5, 0.0));
    }
}
