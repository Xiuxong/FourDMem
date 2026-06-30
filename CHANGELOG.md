# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-30

### Added

#### Core Architecture (L0-L4)
- **L0**: SQLite + FTS5 append-only evidence store with jieba CJK tokenization
- **L1**: petgraph knowledge graph with HNSW vector index, semantic deduplication, version tree
- **L2**: Markdown scenario blocks with dual-channel index (FAISS + TraceMem)
- **L3**: YAML persona with Thompson sampling weighted rule injection
- **L4**: Observer node, paradigm shift engine, myelination tracker, topological monitor, genetic sandbox

#### Retrieval Pipeline
- 4-way recall: Tantivy BM25 + SQLite FTS5 + Graph label + Vector ANN
- RRF (Reciprocal Rank Fusion) multi-path fusion
- RIF-U scoring (Recency, Importance, Frequency, Utility)
- Token budget allocation: L3(20%) + L2(40%) + L1(30%) + L0(10%)
- MetaRouter confidence-based drill-down

#### MCP Integration
- 14 MCP tools: search_memory, submit_feedback, extract_deep, wake_up, save_memory, reflect, abandon_branch, re_enable_branch, checkpoint_memory, load_memory, get_entity_context, memory_health, write_scenario, check_cognition_signals
- Auto-archive: every interaction automatically captured to L0
- Salience detection: high-value conversations trigger immediate L0→L1 extraction

#### Cognitive Evolution Engine
- StrangeLoopGuard: proposal→verify→commit→rollback safety gate
- SignalBus: 6 signal types with cooldown and persistence
- DreamPruner: Ebbinghaus decay with immunity for high-utility memories
- GapReflection: cold-start context recovery after dormancy

#### Testing & Benchmarking
- 143 Rust unit tests, 98 Python integration tests
- Recall benchmark: R@1=100%, R@2=100% (full pipeline, 18q×20f)
- CI: GitHub Actions for Linux (full) + Windows (Rust core)

### Fixed
- Token budget redistribution: empty layer quota cascades downward only
- StrangeLoopGuard: floor check before delta, same-tick rate limiting
- Dedup tests: mock embedder for vector_search path
- extract_deep: count "linked" status in addition to "added"/"merged"
- FTS5/Tantivy: jieba tokenization on both index and query sides
- FTS5/Tantivy: English stop word filtering to prevent BM25 dilution
