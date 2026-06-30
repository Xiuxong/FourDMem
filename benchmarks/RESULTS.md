# FourDMem Benchmark Results

> **Date**: 2026-06-26 | **Version**: v4.1 (Agent-Driven)
> **Hardware**: Intel i7-9750H, Windows 11, CPU-only
> **Embedding**: BAAI/bge-small-zh-v1.5 (dim=512, CPU)

---

## Synthetic Recall Benchmark

**Setup**: 20 facts ingested, 18 questions. Deterministic seed=42.

| Metric | Value |
|:---|---:|
| Accuracy | **100.0%** |
| Recall@1 | **77.8%** (14/18) |
| Recall@3 | **100.0%** (18/18) |
| Recall@5 | **100.0%** (18/18) |
| Latency P50 | 8.4 ms |
| Latency P99 | 12.3 ms |
| Ingest Time | 0.35 s |

```bash
python -m benchmarks.synthetic_recall --db-path data/benchmarks/synthetic_recall.db --seed 42
```

---

## Ablation Study

**Setup**: Same 20 facts + 18 questions. Each component isolated.

| Component | Acc | R@1 | R@3 | R@5 | P50 |
|:---|---:|---:|---:|---:|---:|
| text_only (FTS5, no graph, no RRF) | 100.0% | 77.8% | 77.8% | 77.8% | 9.5ms |
| **vector_only (HNSW)** | **100.0%** | **83.3%** | **94.4%** | **100.0%** | 0.0ms |
| fulltext_vector (no graph, no RRF) | 100.0% | 77.8% | 77.8% | 77.8% | 1.0ms |
| rrf_fusion (no RIF-U) | 100.0% | 77.8% | 94.4% | 100.0% | 0.0ms |
| full_pipeline (all enabled) | 100.0% | 77.8% | 94.4% | 100.0% | 0.0ms |

**Key findings**:
- **Vector (HNSW) wins**: 83.3% R@1, +5.5pp over text-only; 94.4% R@3, +16.6pp
- **RRF fusion boosts recall**: R@3 improves from 77.8% → 94.4% when fusing multiple paths
- **RIF-U neutral** on this dataset: rrf_fusion = full_pipeline
- **Vector-only is strongest single path**: better than text-only by every metric

```bash
python -m benchmarks.ablation
```

---

## LoCoMo Benchmark (Synthetic)

**Setup**: 3 conversations × 10 sessions, 312 turns, 312 L1 facts, 54 QA pairs.
Embedding judge (threshold=0.45).

| Metric | Value |
|:---|---:|
| Accuracy | **74.1%** |
| Recall@1 | **51.8%** |
| Recall@3 | **61.1%** |
| Recall@5 | **63.0%** |
| Latency P50 | 4.0 ms |
| Latency P99 | 13.0 ms |

| Question Type | Accuracy |
|:---|---:|
| single-session | 83.3% (25/30) |
| multi-session | 100.0% (9/9) |
| temporal | 66.7% (6/9) |
| adversarial | 0.0% (0/6) |

**Key findings**:
- Single-session recall strong (83.3%)
- Multi-session recall perfect (100%) — cross-session knowledge preserved
- Temporal reasoning moderate (66.7%)
- Adversarial resilience absent (0.0%) — needs dedicated adversarial filter
- R@5=63%: room for reranking improvement

```bash
python -m benchmarks.run_all --output results --seed 42 --conversations 3 --skip-synthetic --judge
```

---

## Component Status

| Component | Status | Notes |
|:---|---:|:---|
| L0 Storage (SQLite+FTS5) | ✅ | 158 evidence rows |
| L1 Graph (petgraph) | ✅ | 10 nodes, edges via l0_refs |
| L2 Scenarios | ✅ | Auto-aggregation from graph |
| Vector Index (usearch HNSW) | ✅ | dim=512 |
| Fulltext Index (Tantivy BM25) | ✅ | |
| RRF Fusion (3-way) | ✅ | |
| RIF-U Scoring | ✅ | |
| Token Budget (L3/L2/L1/L0) | ✅ | |
| Subjective Time (Active Ticks) | ✅ | |
| Version Tree | ✅ | |
| SSGM Governance | ✅ | |
| Signal Bus (6 types) | ✅ | With persistence |
| Evolution Scheduler | ✅ | Background daemon |
| Health Monitor | ✅ | wake_up probes |
| Integrity Checker | ✅ | Cross-layer validation |

---

## Completed (v4.1)

- [x] P0: Fix 13 pyo3 signatures (eliminates all Python→Rust parameter bugs)
- [x] P0: Add QueryRequest disable_* flags (full ablation support)
- [x] P1: Delete timem_consolidator.py (328 lines, superseded by Agent-driven pipeline)
- [x] P1: Delete strange_loop_guard.py (471 lines, Observer now pushes signals)
- [x] P1: Add TODO to system3.py (concept valid, needs signal-driven rewrite)
- [x] Fix L0 workspace_id=None search bug (was filtering all results)
- [x] Fix Tantivy doc_id content resolution (bare IDs not looked up)
- [x] LoCoMo synthetic benchmark: 74.1% Acc, 51.8% R@1
- [x] Ablation study: 5 configurations, vector-only = strongest single path

## Next Steps

- [ ] Run LoCoMo with real dataset (10 conversations, ~1GB)
- [ ] Add cross-encoder reranking for R@1 improvement
- [ ] GPU-accelerated embedding (10x faster)
- [ ] Adversarial query filter
- [ ] Compare against mem0, MemPalace baselines
