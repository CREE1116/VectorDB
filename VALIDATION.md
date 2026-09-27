# Validation status

This file records what the current tests actually prove. It is not a release gate.

| Area | Current evidence | Remaining gap |
|---|---|---|
| Feature spaces | Unit test separates text, code, and binary fingerprint indices; save/load preserves binary vectors. A CLI fixture exercises `similar-binary` and natural-language search. | Mixed-corpus Recall@K/MRR/nDCG comparison against the old unified index. |
| Incremental consistency | Deterministic 10,000-operation synthetic test covers create/edit, delete, rename/move, copy, and periodic save/load on six text paths. Batch removal test checks unaffected HNSW spaces. A live watcher smoke test covers file create/edit/rename/delete and query visibility. | Event latency distribution, Git checkout/branch switch, directory rename, large corpora, and cross-file graph equivalence. HNSW removal still rebuilds its affected space. |
| Graph integrity | New edges require existing endpoints. Chunk edges are normalized to node IDs; a small ambiguous-name fixture confirms call text becomes `CallsCandidate`, not `Calls`, and contains no dangling stored edges. Legacy dangling edges are pruned on load. | Ground-truth edge Precision/Recall/F1 across language constructs. Candidate calls still require compiler/LSP resolution before they can count as confirmed calls or impact edges. |
| Retrieval quality | `benchmark` evaluates BM25, dense, and hybrid on 12 labeled queries across 14 synthetic code, document, and binary items. It reports Recall@5/10, MRR, nDCG@10, and Hit@1 by query type. | Larger independent queries, genuine semantic embeddings, mixed-corpus comparison with the previous unified index, and agent task outcomes. |
| ANN crossover | `ann-benchmark` measures exact SIMD scan and HNSW on the same normalized vectors, including Recall@K and p50/p95 search latency. The first 1k/10k/100k runs are in `REPORT.md`. | Repeated runs, more distributions/dimensions, build/index size curves, and parameter tuning at 100k+. |
| Snapshot writing | Saving now writes and syncs a temporary file before an atomic rename to `metadata.bin`; failed writes leave the previous snapshot in place. | Kill-9 fault injection, concurrent writer consistency, and platform-specific crash recovery verification. |
| Cold start | VDB3 saves HNSW topology. VDB2 loading and upgrade are tested. `cold-benchmark` measures build/save/load/query and full CLI first search at 1k/10k/100k vectors. | Warm filesystem cache only, one host, synthetic corpus, five runs per size. No 1M result or old-format apples-to-apples baseline. |
| Watcher | Native filesystem events replace periodic whole-tree polling. Initial and rescan-event traversal remain. A live smoke test passed on macOS. | Idle CPU/wakeups, burst event loss, 10k+ file trees, and save/write amplification are unmeasured. |

Run the checks with:

```bash
cargo test
cargo test incremental_matches_fresh_after_ten_thousand_mutations -- --ignored
cargo run --release -p vectordb-cli -- benchmark benchmarks/retrieval-smoke.json
cargo run --release -p vectordb-cli -- ann-benchmark --vectors 10000 --queries 50 --ef-search 256
cargo run --release -p vectordb-cli -- cold-benchmark --vectors 100000 --iterations 5
```
