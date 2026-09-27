# VectorDB

로컬 파일과 코드에서 필요한 청크를 찾기 위한 Rust 실험 프로젝트입니다. BM25, HNSW, 코드 구조 그래프를 결합하고 CLI와 웹 API를 제공합니다. 현재 버전은 검색 품질과 일관성을 검증하는 단계이며, 제품 수준 성능은 [REPORT.md](REPORT.md)의 측정 범위 안에서만 해석해야 합니다.

## 현재 동작

- 코드·문서·오디오 메타데이터·바이너리 파일을 청크와 그래프 노드로 인덱싱합니다. PDF 추출과 언어 파서는 제한적인 자체 구현이므로 포맷별 정확도를 별도로 확인해야 합니다.
- 자연어 검색은 BM25와 텍스트/코드별 HNSW 결과를 순위 기반 RRF로 결합합니다. 기본 벡터 생성기는 토큰·문자 n-gram 해시이며 신경망 의미 임베딩이 아닙니다.
- 바이너리 바이트 fingerprint는 별도 인덱스에 저장하고, 인덱싱된 파일끼리만 유사도를 비교합니다. 같은 256차원이라는 이유로 자연어 벡터와 cosine 점수를 섞지 않습니다.
- 코드의 `Contains`, `Defines`, `NextChunk`와 Git 이력의 `CoChangedWith` 관계를 기록합니다. 이름 패턴으로 찾은 호출은 `CallsCandidate`로 표시하며, 확정된 호출로 사용하지 않습니다.
- 파일 감시로 변경 파일을 다시 인덱싱할 수 있습니다. 삭제가 일어나면 HNSW를 재구축하므로 이전의 밀리초 단위 증분 성능 주장은 현재 코드에 적용되지 않습니다.

## 실행

```bash
cargo build --release
./target/release/vectordb index .
./target/release/vectordb search "cosine similarity" --limit 5
./target/release/vectordb search "HnswIndex" --format json
./target/release/vectordb search "index updates" --format llm
./target/release/vectordb similar-binary sample_files/firmware.bin --limit 5
./target/release/vectordb find HnswIndex
./target/release/vectordb graph 'symbol:src/main.rs:main' --hops 1
./target/release/vectordb impact HnswIndex --depth 2
./target/release/vectordb co-change crates/vectordb-core/src/bm25.rs
./target/release/vectordb watch . --debounce-ms 200
./target/release/vectordb serve --port 8080
```

인덱스는 기본적으로 `.vectordb/`에 저장됩니다. `--db-dir`로 다른 위치를 지정할 수 있습니다. `similar-binary`의 입력 경로는 인덱싱 대상 디렉터리 기준 상대 경로입니다. 이전 DB는 바이너리 fingerprint를 보존하지 않았으므로 이 기능을 사용하려면 다시 인덱싱해야 합니다.

## 검증 명령

```bash
cargo test
cargo test incremental_matches_fresh_after_ten_thousand_mutations -- --ignored
cargo run --release -p vectordb-cli -- benchmark benchmarks/retrieval-smoke.json
cargo run --release -p vectordb-cli -- ann-benchmark --vectors 10000 --queries 50 --ef-search 256
```

`benchmark`는 라벨된 JSON fixture에서 BM25, dense, hybrid의 Recall@5/10, MRR, nDCG@10, Hit@1을 출력합니다. 포함된 fixture는 기능 점검용이며 규모가 작습니다. `ann-benchmark`는 같은 벡터에 대해 HNSW와 전수 SIMD 검색을 비교합니다. 반복 측정과 더 다양한 분포가 필요한 결과는 [REPORT.md](REPORT.md)에 구분해 적었습니다.

## 프로젝트 구조

| 경로 | 역할 |
|---|---|
| `crates/vectordb-core` | BM25, 벡터 인덱스, 그래프, 저장, 검색 |
| `crates/vectordb-parser` | 코드·파일 파싱, Git 이력 관계 |
| `crates/vectordb-cli` | CLI, 벤치마크 명령, 파일 감시 |
| `crates/vectordb-server` | REST API와 웹 화면 |

현재 확인된 범위와 미검증 항목은 [VALIDATION.md](VALIDATION.md)에 정리했습니다.
