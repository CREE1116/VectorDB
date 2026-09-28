# Semantic Unit Graph: design direction

This document distinguishes the proposed model from the current implementation. The current CLI still stores `Chunk`, `Node`, `Edge`, and `Feature` separately. `Unit` is a design target, not a shipped API.

## One indexing principle

Every indexed object must expose **readable text** and a location back to its source. Parsers turn that text into addressable semantic units. Search finds units; relationships provide optional context. Code functions, document sections, paragraphs, configuration keys, and text table cells can participate under the same rule. Images, audio, opaque binaries, and metadata invented from their filenames are outside the index. A PDF is eligible only when actual text is extracted. CSV and other structured text require their own quality checks before claiming row or column units.

```text
text source → parser → units → text/code features → retrieval
                         ↘ typed relationships → optional expansion
```

The source file remains a physical container node for ownership, updates, and citations. It is not the default search result or the assumed endpoint of a semantic relation. A result should identify the smallest useful unit and its source span.

## Proposed unit and relation contracts

```text
Unit {
  id, kind, source_path, span, text, parent_id?, feature_refs[], metadata
}

Relation {
  source_unit_id, target_unit_id, kind, evidence,
  origin, confidence?, extractor_version?
}
```

`id` must survive a stable source where practical, but edits and moves can change spans. A relation therefore also needs source evidence so it can be invalidated or rebuilt when the underlying text changes. `confidence` is a model score, not a probability of truth unless calibrated.

| Origin | Examples | Treatment |
|---|---|---|
| Structural | file contains section; class contains method | Deterministic parser output, checked for valid endpoints |
| Explicit | Markdown link; import; cited identifier | Text or syntax evidence with a source span; resolution may remain a candidate |
| Derived | shared topic; text similarity; Git co-change | Ranked candidate for navigation, with method/version and score |
| Provenance | generated-from or exported-from when recorded by a trusted tool | Recorded event with origin and timestamp; no inference from matching names |

These are origins, not four independent graph databases. They share unit IDs and typed edges. A query can filter by edge kind and origin. Impact analysis must use only relation types that actually support an impact claim; shared-topic or co-change edges do not.

## Topic nodes

Topic modeling should operate on substantive units such as sections or paragraphs, not only whole files. A topic is an intermediate node connecting units across files and code. `Unit → Topic` edges are derived candidates with a model version and strength. A topic match is evidence of similar subject matter, not a citation, implementation link, or factual dependency. The first prototype should build these links offline and compare search/navigation quality against the current retrieval baseline. Topic IDs can change when the model is retrained, so their IDs must be scoped to a model version.

## Retrieval contract

1. Retrieve text-backed units using BM25 and compatible text/code features.
2. Return source path, span, unit type, score, and evidence needed to inspect the current file.
3. Expand through selected relations only when the task asks for context or the ranking experiment shows a benefit.
4. Keep `graph`/explanation output available for inspection; do not silently turn an inferred edge into a confirmed answer.

## Current state and next proof

Today, Markdown sections and code symbols are represented as graph nodes with containment and some structural edges. Local Markdown anchor links connect distinct section units as `LinksTo`, with source lines recorded on the source node. Cross-file call-shaped matches are `CallsCandidate`; Git co-change is recorded separately. Cross-document links, document concepts, citations, paragraph-level identity, and topic nodes are not yet extracted. The present feature embedder is lexical hashing, so topic modeling with meaningful semantic embeddings requires a separate experiment.

The next evaluation needs labeled links between units in the same document and across documents, plus queries where graph expansion might help. Measure edge precision by relation kind, retrieval Recall@K/MRR/nDCG, mutation invalidation, index size, and cold query latency. Add a derived relation to the default path only when it improves a stated task without burying exact evidence.
