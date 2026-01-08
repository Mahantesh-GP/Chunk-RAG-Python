# Coding Task Plan and Estimates

This plan outlines the engineering (coding) work for the RAG chunking evaluation project. Tasks are grouped into phases with clear descriptions and hour estimates for manager tracking.

- Base scope total: 114 hours
- Optional Azure integration scope: +16 hours
- Potential grand total: 130 hours

---

## Phase 1 — Core Retrieval & Features (24h)
- [ ] BM25 lexical retriever — 4h
  - Description: Implement a BM25-based retriever and integrate as a pluggable backend alongside the current lexical method. Expose `top_k` and BM25 parameters.
  - Affects: [rag_eval/evaluator.py](rag_eval/evaluator.py), new module under `rag_eval/retrievers`.
- [ ] Hybrid retriever (BM25 + vector) — 6h
  - Description: Add rank fusion (e.g., Reciprocal Rank Fusion) combining BM25 and embeddings-based similarity with tunable weights.
  - Affects: new `rag_eval/retrievers/hybrid.py` and evaluator wiring.
- [ ] Embedding provider abstraction — 8h
  - Description: Create `Embedder` interface and implement a local baseline (e.g., Sentence-Transformers). Keep `MockEmbedder` as default.
  - Affects: [rag_eval/strategies.py](rag_eval/strategies.py) (refactor), new `rag_eval/embeddings/`.
- [ ] Reranking (optional cross-encoder) — 6h
  - Description: Optional rerank step over top-N candidates with a light cross-encoder or a heuristic proxy.
  - Affects: new `rag_eval/rerankers/` and evaluator hook.

Section total: 24h

---

## Phase 2 — Chunking Strategy Improvements (18h)
- [ ] Parameterize strategies via config — 3h
  - Description: Centralize parameters (sizes, overlaps, thresholds) in a config object/JSON.
  - Affects: [rag_eval/strategies.py](rag_eval/strategies.py), [rag_eval/cli.py](rag_eval/cli.py).
- [ ] Sliding window with dynamic overlap — 4h
  - Description: Add token-budget aware overlap that adapts to sentence boundaries.
  - Affects: `FixedSizeChunker` implementation.
- [ ] Headings/Markdown-aware splitter — 4h
  - Description: Improve structure chunker to respect headings, lists, and code fences using a small parser.
  - Affects: `StructureChunker`.
- [ ] Semantic segmentation refinement — 4h
  - Description: Use sentence embeddings with similarity scanning to create coherent semantic chunks.
  - Affects: `SemanticChunker`.
- [ ] Multigranular weighting — 3h
  - Description: Score per granularity and combine with weights for final ranking.
  - Affects: `MultigranularChunker`.

Section total: 18h

---

## Phase 3 — Evaluation Framework & Metrics (20h)
- [ ] Add recall@k, precision@k, MRR, nDCG — 6h
  - Description: Extend metrics beyond Jaccard/coverage, compute per-query and averages.
  - Affects: [rag_eval/evaluator.py](rag_eval/evaluator.py), results schema.
- [ ] LLM/QAG-based faithfulness (offline-friendly) — 8h
  - Description: Add a question-answer generation or entailment-style scorer with fallback heuristic; keep provider-agnostic.
  - Affects: evaluator metrics and optional provider stubs.
- [ ] Per-query breakdown + confusion analysis — 3h
  - Description: Export per-query detail tables and simple confusion indicators for diagnostics.
  - Affects: `evaluation_results.json` structure and [show_results.py](show_results.py).
- [ ] Experiment runner & seed control — 3h
  - Description: Deterministic runs, seed management, and experiment manifests for reproducibility.
  - Affects: [rag_eval/cli.py](rag_eval/cli.py), new `experiments/`.

Section total: 20h

---

## Phase 4 — Data Handling & PDF Quality (12h)
- [ ] Layout-preserving PDF extraction — 6h
  - Description: Improve extraction to retain headings, paragraphs, and code blocks where possible.
  - Affects: PDF path in [rag_eval/cli.py](rag_eval/cli.py).
- [ ] Normalization pipeline — 3h
  - Description: Unicode normalization, whitespace cleanup, optional lowercasing/stopword handling behind flags.
  - Affects: new `rag_eval/preprocess.py`.
- [ ] Multipage chunk boundary constraints — 3h
  - Description: Avoid splitting in the middle of figures/tables; keep chunks page-aware when desired.
  - Affects: structure/semantic chunkers.

Section total: 12h

---

## Phase 5 — CLI & Results UX (10h)
- [ ] Extended CLI flags and presets — 4h
  - Description: Select strategies, params, retrievers, and output naming/tagging from CLI.
  - Affects: [rag_eval/cli.py](rag_eval/cli.py).
- [ ] Progress bars and timing summary — 2h
  - Description: Add rich/tqdm progress with per-phase timing and counts.
  - Affects: CLI orchestration.
- [ ] HTML/Lightweight viewer — 4h
  - Description: Generate a simple HTML report from `evaluation_results.json` with charts/tables.
  - Affects: [show_results.py](show_results.py) or new `tools/report.py`.

Section total: 10h

---

## Phase 6 — Performance & Reliability (16h)
- [ ] Profiling and optimizations — 4h
  - Description: Use cProfile to identify hotspots; optimize tokenization and set ops.
  - Affects: evaluator and chunkers.
- [ ] Parallel execution (queries/strategies) — 6h
  - Description: Multiprocessing or joblib to parallelize evaluations with safe logging/artifacts.
  - Affects: CLI runner and evaluator.
- [ ] Chunk/retrieval caching — 3h
  - Description: Cache chunk outputs and retrieval scores keyed by content hash + params.
  - Affects: new `rag_eval/cache.py`.
- [ ] Run metadata capture — 3h
  - Description: Attach run id, params, versions, and timing to outputs for auditability.
  - Affects: result schema and CLI.

Section total: 16h

---

## Phase 7 — Code Quality & Cleanup (14h)
- [ ] Full type hints + mypy — 4h
  - Description: Add annotations and static checks; enforce in CI.
  - Affects: entire `rag_eval/` package.
- [ ] Docstrings and module docs — 3h
  - Description: Consistent docstrings and minimal developer docs for public APIs.
  - Affects: strategies, evaluator, retrievers.
- [ ] Error handling & custom exceptions — 3h
  - Description: Standardize error types, messages, and user-facing guidance.
  - Affects: CLI and library boundaries.
- [ ] Modularization and interfaces — 4h
  - Description: Introduce clear interfaces for chunkers, retrievers, embedders, and scorers.
  - Affects: package structure.

Section total: 14h

---

## Optional — Azure Integration (+16h)
- [ ] Azure OpenAI embeddings provider — 8h
  - Description: Implement `AzureOpenAIEmbedder` behind the embedder interface with `.env`-driven config.
  - Affects: `rag_eval/embeddings/azure_openai.py`.
- [ ] Azure Cognitive Search retriever — 8h
  - Description: Adapter for ACS (indexing + query) and fallback to local modes via flags.
  - Affects: `rag_eval/retrievers/azure_search.py`, CLI flags, result metadata.

Section total: 16h (optional)

---

## Rollup Summary
- Phase 1: 24h
- Phase 2: 18h
- Phase 3: 20h
- Phase 4: 12h
- Phase 5: 10h
- Phase 6: 16h
- Phase 7: 14h
- Optional Azure: +16h

Base scope total: 114h
Optional Azure scope: +16h
Potential grand total: 130h

---

## Notes & Tracking
- Convert items into GitHub Issues with labels: `phase:<n>`, `type:feature`, `type:quality`, `priority`.
- Attach affected files/modules to issues (see per-task references above).
- Link runs and artifacts (JSON/HTML) back to the issue or PR for traceability.
