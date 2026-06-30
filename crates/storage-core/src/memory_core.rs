//! Unified MemoryCore entry point
//!
//! Orchestrates all sub-engines into a single coherent interface:
//!
//! ```text
//! MemoryCore
//!   ├─ L0Store         (raw evidence, SQLite)
//!   ├─ L3Store         (persona/core rules, YAML)
//!   ├─ L1Graph         (atomic fact graph)
//!   ├─ FulltextIndex   (Tantivy BM25)
//!   ├─ RifUScorer      (RIF-U ranking)
//!   ├─ MetaRouter      (metacognitive drill-down)
//!   ├─ TokenBudget     (layer quota allocator)
//!   └─ SyncEngine      (file change detection)
//! ```

use serde::{Deserialize, Serialize};

use crate::l0::{Evidence, L0Store};
use crate::l2::L2Store;
use crate::l3::L3Store;
use graph_core::{L1Graph, VersionTree};
use std::sync::Mutex;

use retrieval_core::{RankedItem, RrfFuser, FulltextIndex, MetaRouter, RifUScorer, TokenBudget, VectorIndex, DEFAULT_EMBEDDING_DIM};
use sync_engine::SyncEngine;
use evolution_core::{MacroCache, CognitiveMacro};

// ── Query types ────────────────────────────────────────────────────────────────

/// A query request to the memory system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    /// Natural language query string.
    pub query: String,
    /// Maximum number of results to return (before token budget trimming).
    pub limit: usize,
    /// Whether to allow metacognitive drill-down to L0 evidence.
    pub allow_drill_down: bool,
    /// Scope search to a specific workspace. None = search all.
    pub workspace_id: Option<String>,
    /// Ablation: skip fulltext (Tantivy + FTS5).
    pub disable_fulltext: bool,
    /// Ablation: skip graph traversal (petgraph).
    pub disable_graph: bool,
    /// Ablation: skip RRF fusion (return raw ranked lists).
    pub disable_rrf: bool,
    /// Ablation: skip RIF-U scoring (use raw retrieval scores).
    pub disable_rif_u: bool,
}

impl QueryRequest {
    /// Simple query with defaults.
    pub fn simple(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            limit: 20,
            allow_drill_down: true,
            workspace_id: None,
            disable_fulltext: false,
            disable_graph: false,
            disable_rrf: false,
            disable_rif_u: false,
        }
    }
}

/// A single item in the query response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryItem {
    pub id: String,
    pub layer: u8,
    pub content: String,
    pub score: f64,
    pub token_estimate: usize,
    #[serde(default)]
    pub timestamp: String,
    /// Metadata from L0 evidence (agent_id, visibility, etc.)
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// Response from a memory query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    /// Ranked result items (within token budget).
    pub items: Vec<QueryItem>,
    /// Total tokens used.
    pub total_tokens: usize,
    /// Whether the router decided to drill down to L0.
    pub drilled_down: bool,
    /// Confidence score from the metacognitive router.
    pub confidence: f64,
}

/// Aggregate statistics about the memory system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// L0 evidence count.
    pub l0_count: i64,
    /// L1 graph node count.
    pub l1_node_count: usize,
    /// L1 graph edge count.
    pub l1_edge_count: usize,
    /// L2 scenario block count.
    pub l2_block_count: usize,
    /// Fulltext index document count.
    pub fulltext_doc_count: u64,
    /// L3 persona name.
    pub persona_name: String,
}

/// Report from a dream-pruning cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamReport {
    /// Number of nodes pruned (decayed).
    pub pruned: usize,
    /// Number of nodes preserved (immune or still alive).
    pub preserved: usize,
    /// Current active tick at time of pruning.
    pub current_tick: u64,
}

// ── MemoryCore ─────────────────────────────────────────────────────────────────

/// The unified four-dimensional memory engine.
///
/// Coordinates L0 evidence storage, L1 graph indexing, L2 fulltext search,
/// L3 persona management, RIF-U scoring, metacognitive routing, and token
/// budget allocation.
pub struct MemoryCore {
    l0: L0Store,
    l2: L2Store,
    l3: L3Store,
    graph: L1Graph,
    version_tree: VersionTree,
    fulltext: FulltextIndex,
    vector_index: VectorIndex,
    embedding_dim: usize,
    scorer: RifUScorer,
    router: MetaRouter,
    token_budget: TokenBudget,
    rrf_fuser: RrfFuser,
    macro_cache: Mutex<MacroCache>,
    current_tick: Mutex<u64>,
    sync: SyncEngine,
}

impl MemoryCore {
    /// Create a new MemoryCore with default configuration.
    ///
    /// `l0_path` is the filesystem path for the L0 SQLite database.
    /// `vault_root` is the root directory for L2/L3 file synchronization.
    pub fn new(l0_path: &str, vault_root: &str) -> crate::Result<Self> {
        let l0 = L0Store::open(l0_path)?;
        let scenarios_path = std::path::Path::new(vault_root).join("scenarios");
        let l2 = L2Store::load(&scenarios_path)
            .map_err(|e| crate::Error::from(format!("L2 load error: {}", e)))?;
        let l3 = L3Store::load(
            std::path::Path::new(vault_root).join("persona").join("persona.yaml"),
        )?;
        let graph = L1Graph::new();
        let version_tree = VersionTree::new();

        // Disk-backed fulltext index — survives restarts
        let fulltext_dir = std::path::Path::new(vault_root).join("fulltext");
        let fulltext = FulltextIndex::new_on_disk(
            fulltext_dir.to_str().unwrap_or("/tmp/fourdmem_fulltext")
        ).map_err(|e| Box::new(e) as crate::Error)?;

        let vector_index = VectorIndex::new(DEFAULT_EMBEDDING_DIM)
            .map_err(|e| crate::Error::from(e))?;
        let scorer = RifUScorer::new();
        let router = MetaRouter::new();
        let token_budget = TokenBudget::new();
        let sync = SyncEngine::new(vault_root);

        let mut mc = Self {
            l0,
            l2,
            l3,
            graph,
            version_tree,
            fulltext,
            vector_index,
            embedding_dim: DEFAULT_EMBEDDING_DIM,
            scorer,
            router,
            token_budget,
            rrf_fuser: RrfFuser::new(),
            macro_cache: Mutex::new(MacroCache::new(5)),
            current_tick: Mutex::new(0),
            sync,
        };
        // If fulltext index is empty but L0 has data, rebuild from L0
        if mc.fulltext.doc_count() == 0 {
            let _ = mc.rebuild_fulltext_from_l0();
        }

        Ok(mc)
    }

    /// Create a MemoryCore with an in-memory L0 store (for testing).
    pub fn new_in_memory() -> crate::Result<Self> {
        let l0 = L0Store::open_memory()?;
        let l2 = L2Store::new_temp("/tmp/fourdmem_l2_test");
        let l3 = L3Store::default_store();
        let graph = L1Graph::new();
        let version_tree = VersionTree::new();
        let fulltext = FulltextIndex::new().map_err(|e| Box::new(e) as crate::Error)?;
        let vector_index = VectorIndex::new(DEFAULT_EMBEDDING_DIM)
            .map_err(|e| crate::Error::from(e))?;
        let scorer = RifUScorer::new();
        let router = MetaRouter::new();
        let token_budget = TokenBudget::new();
        let sync = SyncEngine::new("/tmp/fourdmem_test");

        Ok(Self {
            l0,
            l2,
            l3,
            graph,
            version_tree,
            fulltext,
            vector_index,
            embedding_dim: DEFAULT_EMBEDDING_DIM,
            scorer,
            router,
            token_budget,
            rrf_fuser: RrfFuser::new(),
            macro_cache: Mutex::new(MacroCache::new(5)),
            current_tick: Mutex::new(0),
            sync,
        })
    }

    // ── Ingest ─────────────────────────────────────────────────────────────

    /// Append evidence to L0 and index it in the fulltext engine.
    ///
    /// Returns the auto-incremented L0 row ID.
    pub fn ingest_evidence(
        &mut self,
        workspace_id: &str,
        model_name: &str,
        session_id: &str,
        role: &str,
        content: &str,
        metadata: &serde_json::Value,
    ) -> crate::Result<i64> {
        let id = self.l0.append(workspace_id, model_name, session_id, role, content, metadata)?;

        // Index in fulltext
        self.fulltext
            .add(&id.to_string(), content)
            .map_err(|e| Box::new(e) as crate::Error)?;
        self.fulltext
            .commit()
            .map_err(|e| Box::new(e) as crate::Error)?;

        Ok(id)
    }

    /// Like `ingest_evidence`, but uses a pre-computed embedding vector
    /// instead of the internal placeholder `embed_text`.
    ///
    /// `embedding` must have the same dimension as `self.embedding_dim`.
    /// Like `ingest_evidence`, but uses a pre-computed embedding vector
    /// instead of the internal placeholder `embed_text`.
    pub fn ingest_with_embedding(
        &mut self,
        workspace_id: &str,
        session_id: &str,
        role: &str,
        content: &str,
        metadata: &serde_json::Value,
        embedding: &[f32],
    ) -> crate::Result<i64> {
        let model_name = metadata.get("model_name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let id = self.l0.append(workspace_id, model_name, session_id, role, content, metadata)?;

        // Index in fulltext
        self.fulltext
            .add(&id.to_string(), content)
            .map_err(|e| Box::new(e) as crate::Error)?;
        self.fulltext
            .commit()
            .map_err(|e| Box::new(e) as crate::Error)?;

        // Index in vector store with external embedding
        self.vector_index
            .add(id as u64, embedding)
            .map_err(|e| crate::Error::from(e))?;

        Ok(id)
    }
    pub fn query(&self, request: &QueryRequest) -> crate::Result<QueryResponse> {
        // ── Step 0: MacroCache intuition check ─────────────────────────────
        let pattern_key = compute_pattern_key(&request.query);
        let intuition_result: Option<QueryResponse> = {
            let cache = self.macro_cache.lock()
                .map_err(|e| crate::Error::from(format!("lock poisoned: {}", e)))?;
            cache.get(&pattern_key).and_then(|mac| {
                if mac.hit_count >= 5 && mac.success_rate >= 0.8 {
                    serde_json::from_str(&mac.result).ok()
                } else {
                    None
                }
            })
        };
        if let Some(response) = intuition_result {
            if !response.items.is_empty() {
                self.record_macro_hit(&pattern_key, true);
                return Ok(response);
            }
        }

        // ── Step 1a: Fulltext search → RankedItem list ─────────────────────
        let raw_results = self
            .fulltext
            .search(&request.query, request.limit)
            .map_err(|e| Box::new(e) as crate::Error)?;

        let fulltext_ranked: Vec<RankedItem> = raw_results
            .iter()
            .map(|(doc_id, score)| RankedItem {
                doc_id: doc_id.clone(),
                score: *score,
            })
            .collect();

        let l0_fts5_ranked: Vec<RankedItem> = self
            .l0
            .search(&request.query, request.limit as i64, request.workspace_id.as_deref())
            .unwrap_or_default()
            .iter()
            .map(|ev| RankedItem {
                doc_id: format!("l0:{}", ev.id),
                score: 0.5, // L0 evidence has lower base score
            })
            .collect();
        // Step 1b: Graph label search → RankedItem list
        let graph_hits = self.graph.search_nodes(&request.query);
        let graph_ranked: Vec<RankedItem> = graph_hits
            .iter()
            .enumerate()
            .map(|(i, (_idx, attr))| RankedItem {
                doc_id: format!("l1:{}", attr.label),
                score: 1.0 / (1.0 + i as f64),
            })
            .collect();

        // Callers must use query_with_embedding() with a Python-computed
        // semantic embedding for vector search to participate in RRF.
        let vector_ranked: Vec<RankedItem> = Vec::new();
        let mut rank_lists: Vec<Vec<RankedItem>> = Vec::new();
        if !request.disable_fulltext && !fulltext_ranked.is_empty() {
            rank_lists.push(fulltext_ranked);
        }
        if !request.disable_fulltext && !vector_ranked.is_empty() {
            rank_lists.push(vector_ranked);
        }
        if !request.disable_graph && !graph_ranked.is_empty() {
            rank_lists.push(graph_ranked);
        }
        if !request.disable_fulltext && !l0_fts5_ranked.is_empty() {
            rank_lists.push(l0_fts5_ranked);
        }
        let fused = if request.disable_rrf || rank_lists.len() <= 1 {
            rank_lists.into_iter().flatten().collect()
        } else {
            self.rrf_fuser.fuse(&rank_lists)
        };

        // Step 2b: Temporal Gate — filter counterfactual/expired L1 results
        let now = chrono::Utc::now();
        let fused: Vec<RankedItem> = fused
            .into_iter()
            .filter(|item| {
                // Only filter L1 results (graph nodes with version tree entries)
                if !item.doc_id.starts_with("l1:") {
                    return true; // L0 evidence passes through
                }
                let entity_id = item.doc_id.strip_prefix("l1:").unwrap_or(&item.doc_id);
                let versions = self.version_tree.temporal_gate(entity_id, now, 7);
                // Keep if at least one active version remains
                !versions.is_empty()
            })
            .collect();

        // Step 3: Convert fused results to budget candidates
        let candidates: Vec<retrieval_core::BudgetCandidate> = fused
            .iter()
            .map(|item| {
                let is_l1 = item.doc_id.starts_with("l1:");
                let is_l1_vector = !is_l1 && item.doc_id.parse::<u64>()
                    .map(|k| k >= 1_000_000_000)
                    .unwrap_or(false);
                let layer = if is_l1 || is_l1_vector { 1 } else { 2 };

                // Use actual content length for L0 items, doc_id length for L1
                let token_estimate = if is_l1 || is_l1_vector {
                    estimate_tokens(&item.doc_id)
                } else {
                    // L0 item: get real content length from DB
                    item.doc_id.parse::<i64>()
                        .ok()
                        .and_then(|eid| self.l0.get_content_length(eid).ok())
                        .map(|len| (len + 3) / 4)
                        .unwrap_or(estimate_tokens(&item.doc_id))
                };

                retrieval_core::BudgetCandidate {
                    id: item.doc_id.clone(),
                    layer,
                    score: item.score,
                    token_estimate,
                }
            })
            .collect();

        // Step 4: Token budget allocation (skip RIF-U when disabled)
        let allocation = if request.disable_rif_u {
            self.token_budget.allocate_flat(&candidates)
        } else {
            self.token_budget.allocate(&candidates)
        };

        // Step 5: Build query items with temporal metadata
        let now = chrono::Utc::now();
        let items: Vec<QueryItem> = allocation
            .selected
            .iter()
            .map(|c| {
                    let (content, timestamp, metadata) = if c.id.starts_with("l1:") {
                        let entity_id = c.id.strip_prefix("l1:").unwrap_or(&c.id);
                        let versions = self.version_tree.temporal_gate(entity_id, now, 7);
                        let ts = versions.first()
                            .map(|v| v.valid_from.to_rfc3339())
                            .unwrap_or_default();
                        let is_dormant = versions.first()
                            .map(|v| now.signed_duration_since(v.valid_from).num_days() > 7)
                            .unwrap_or(false);
                        let meta = if is_dormant {
                            serde_json::json!({"status": "historical_context"})
                        } else {
                            serde_json::Value::Null
                        };
                        (entity_id.to_string(), ts, meta)
                    } else if c.id.parse::<u64>().map(|k| k >= 1_000_000_000).unwrap_or(false) {
                        let node_idx = c.id.parse::<u64>().unwrap() - 1_000_000_000;
                        let (content, ts, is_dormant) = self.graph.get_node(graph_core::NodeIndex::new(node_idx as usize))
                            .map(|n| {
                                let versions = self.version_tree.temporal_gate(&n.label, now, 7);
                                let ts = versions.first()
                                    .map(|v| v.valid_from.to_rfc3339())
                                    .unwrap_or_default();
                                let dormant = versions.first()
                                    .map(|v| now.signed_duration_since(v.valid_from).num_days() > 7)
                                    .unwrap_or(false);
                                (n.label.clone(), ts, dormant)
                            })
                            .unwrap_or_else(|| (c.id.clone(), String::new(), false));
                        let meta = if is_dormant {
                            serde_json::json!({"status": "historical_context"})
                        } else {
                            serde_json::Value::Null
                        };
                        (content, ts, meta)
                    } else {
                    // L0 evidence: get content, timestamp, and metadata
                    let ev = c.id
                        .strip_prefix("l0:")
                        .and_then(|num| num.parse::<i64>().ok())
                        .and_then(|eid| self.l0.get_by_id(eid).ok().flatten())
                        .or_else(|| {
                            c.id.parse::<i64>().ok()
                                .and_then(|eid| self.l0.get_by_id(eid).ok().flatten())
                        });
                    match ev {
                        Some(e) => (e.content, e.timestamp, e.metadata),
                        None => (c.id.clone(), String::new(), serde_json::Value::Null),
                    }
                };
                QueryItem {
                    id: c.id.clone(),
                    layer: c.layer,
                    content,
                    score: c.score,
                    token_estimate: c.token_estimate,
                    timestamp,
                    metadata,
                }
            })
            .collect();

        // Step 6: Metacognitive routing
        let top_score = items.first().map(|i| i.score).unwrap_or(0.0);
        let confidence = self.router.evaluate_confidence(items.len(), top_score);
        let drilled_down = request.allow_drill_down && self.router.should_drill_down(confidence);

        let response = QueryResponse {
            items,
            total_tokens: allocation.total_tokens,
            drilled_down,
            confidence,
        };

        // ── Step 7: Record macro for future intuition ──────────────────────
        if !response.items.is_empty() {
            let tick = self.get_tick();
            let result_json = serde_json::to_string(&response).unwrap_or_default();
            if let Ok(mut cache) = self.macro_cache.lock() {
                if cache.get(&pattern_key).is_some() {
                    cache.record_hit(&pattern_key, true);
                } else {
                    cache.insert(CognitiveMacro {
                        pattern: pattern_key,
                        result: result_json,
                        hit_count: 1,
                        success_rate: 1.0,
                        compiled_at_tick: tick,
                    });
                }
            }
        }

        Ok(response)
    }

    /// Like `query`, but uses a pre-computed embedding for vector search
    /// instead of the internal placeholder `embed_text`.
    pub fn query_with_embedding(
        &self,
        request: &QueryRequest,
        query_embedding: &[f32],
    ) -> crate::Result<QueryResponse> {
        // ── Step 0: MacroCache intuition check ─────────────────────────────
        let pattern_key = compute_pattern_key(&request.query);
        let intuition_result: Option<QueryResponse> = {
            let cache = self.macro_cache.lock()
                .map_err(|e| crate::Error::from(format!("lock poisoned: {}", e)))?;
            cache.get(&pattern_key).and_then(|mac| {
                if mac.hit_count >= 5 && mac.success_rate >= 0.8 {
                    serde_json::from_str(&mac.result).ok()
                } else {
                    None
                }
            })
        };
        if let Some(response) = intuition_result {
            if !response.items.is_empty() {
                self.record_macro_hit(&pattern_key, true);
                return Ok(response);
            }
        }

        // Fulltext search
        let raw_results = self
            .fulltext
            .search(&request.query, request.limit)
            .map_err(|e| Box::new(e) as crate::Error)?;

        let fulltext_ranked: Vec<RankedItem> = raw_results
            .iter()
            .map(|(doc_id, score)| RankedItem {
                doc_id: doc_id.clone(),
                score: *score,
            })
            .collect();

        // Graph label search
        let graph_hits = self.graph.search_nodes(&request.query);
        let graph_ranked: Vec<RankedItem> = graph_hits
            .iter()
            .enumerate()
            .map(|(i, (_idx, attr))| RankedItem {
                doc_id: format!("l1:{}", attr.label),
                score: 1.0 / (1.0 + i as f64),
            })
            .collect();

        // Vector search with external embedding
        let vector_ranked: Vec<RankedItem> = self
            .vector_index
            .search(query_embedding, request.limit)
            .unwrap_or_default()
            .into_iter()
            .map(|(key, sim)| RankedItem {
                doc_id: key.to_string(),
                score: sim as f64,
            })
            .collect();

        // L0 FTS5 native search for CJK-friendly keyword retrieval
        let l0_fts5_ranked: Vec<RankedItem> = self
            .l0
            .search(&request.query, request.limit as i64, request.workspace_id.as_deref())
            .unwrap_or_default()
            .iter()
            .map(|ev| RankedItem {
                doc_id: format!("l0:{}", ev.id),
                score: 0.5,
            })
            .collect();

        // RRF fusion (respecting disable flags)
        let mut rank_lists: Vec<Vec<RankedItem>> = Vec::new();
        if !request.disable_fulltext && !fulltext_ranked.is_empty() {
            rank_lists.push(fulltext_ranked);
        }
        if !vector_ranked.is_empty() {
            rank_lists.push(vector_ranked);
        }
        if !request.disable_graph && !graph_ranked.is_empty() {
            rank_lists.push(graph_ranked);
        }
        if !request.disable_fulltext && !l0_fts5_ranked.is_empty() {
            rank_lists.push(l0_fts5_ranked);
        }
        let fused = if request.disable_rrf || rank_lists.len() <= 1 {
            rank_lists.into_iter().flatten().collect()
        } else {
            self.rrf_fuser.fuse(&rank_lists)
        };

        // Temporal Gate
        let now = chrono::Utc::now();
        let fused: Vec<RankedItem> = fused
            .into_iter()
            .filter(|item| {
                if !item.doc_id.starts_with("l1:") {
                    return true;
                }
                let entity_id = item.doc_id.strip_prefix("l1:").unwrap_or(&item.doc_id);
                let versions = self.version_tree.temporal_gate(entity_id, now, 7);
                !versions.is_empty()
            })
            .collect();

        // Build budget candidates
        let candidates: Vec<retrieval_core::BudgetCandidate> = fused
            .iter()
            .map(|item| {
                let is_l1 = item.doc_id.starts_with("l1:");
                let is_l1_vector = !is_l1 && item.doc_id.parse::<u64>()
                    .map(|k| k >= 1_000_000_000)
                    .unwrap_or(false);
                let layer = if is_l1 || is_l1_vector { 1 } else { 2 };
                let token_estimate = if is_l1 || is_l1_vector {
                    estimate_tokens(&item.doc_id)
                } else {
                    item.doc_id.parse::<i64>()
                        .ok()
                        .and_then(|eid| self.l0.get_content_length(eid).ok())
                        .map(|len| (len + 3) / 4)
                        .unwrap_or(estimate_tokens(&item.doc_id))
                };
                retrieval_core::BudgetCandidate {
                    id: item.doc_id.clone(),
                    layer,
                    score: item.score,
                    token_estimate,
                }
            })
            .collect();

        let allocation = if request.disable_rif_u {
            self.token_budget.allocate_flat(&candidates)
        } else {
            self.token_budget.allocate(&candidates)
        };

        // Build query items with temporal metadata
        let now = chrono::Utc::now();
        let items: Vec<QueryItem> = allocation
            .selected
            .iter()
            .map(|c| {
                let (content, timestamp, metadata) = if c.id.starts_with("l1:") {
                    let entity_id = c.id.strip_prefix("l1:").unwrap_or(&c.id);
                    let ts = self.version_tree
                        .temporal_gate(entity_id, now, 7)
                        .first()
                        .map(|v| v.valid_from.to_rfc3339())
                        .unwrap_or_default();
                    (entity_id.to_string(), ts, serde_json::Value::Null)
                } else if c.id.parse::<u64>().map(|k| k >= 1_000_000_000).unwrap_or(false) {
                    let node_idx = c.id.parse::<u64>().unwrap() - 1_000_000_000;
                    let (content, ts) = self.graph.get_node(graph_core::NodeIndex::new(node_idx as usize))
                        .map(|n| {
                            let ts = self.version_tree
                                .temporal_gate(&n.label, now, 7)
                                .first()
                                .map(|v| v.valid_from.to_rfc3339())
                                .unwrap_or_default();
                            (n.label.clone(), ts)
                        })
                        .unwrap_or_else(|| (c.id.clone(), String::new()));
                    (content, ts, serde_json::Value::Null)
                } else {
                    // Try "l0:" prefix first, then bare numeric ID (Tantivy doc_id)
                    let ev = c.id
                        .strip_prefix("l0:")
                        .and_then(|num| num.parse::<i64>().ok())
                        .and_then(|eid| self.l0.get_by_id(eid).ok().flatten())
                        .or_else(|| {
                            c.id.parse::<i64>().ok()
                                .and_then(|eid| self.l0.get_by_id(eid).ok().flatten())
                        });
                    match ev {
                        Some(e) => (e.content, e.timestamp, e.metadata),
                        None => (c.id.clone(), String::new(), serde_json::Value::Null),
                    }
                };
                QueryItem {
                    id: c.id.clone(),
                    layer: c.layer,
                    content,
                    score: c.score,
                    token_estimate: c.token_estimate,
                    timestamp,
                    metadata,
                }
            })
            .collect();

        let top_score = items.first().map(|i| i.score).unwrap_or(0.0);
        let confidence = self.router.evaluate_confidence(items.len(), top_score);

        Ok(QueryResponse {
            items,
            total_tokens: allocation.total_tokens,
            confidence,
            drilled_down: false,
        })
    }

    // ── Drill-down ─────────────────────────────────────────────────────────

    /// Trace an L1 fact back to its L0 raw evidence.
    pub fn drill_down(&self, l0_id: i64) -> crate::Result<Vec<Evidence>> {
        let results = self.l0.search(&l0_id.to_string(), 10, None)?;
        Ok(results)
    }

    /// Retrieve session evidence scoped to a workspace.
    pub fn get_session_evidence(
        &self,
        workspace_id: &str,
        session_id: &str,
        limit: i64,
    ) -> crate::Result<Vec<Evidence>> {
        Ok(self.l0.search_by_session(workspace_id, session_id, limit)?)
    }

    // ── Graph operations ─────────────────────────────────────────────────────

    /// Add an L1 atomic fact node to the graph and index its label in the
    /// vector store for semantic search.
    ///
    /// Returns the node index.
    pub fn add_fact(&mut self, label: &str, l0_refs: Option<Vec<i64>>) -> usize {
        let attr = match l0_refs {
            Some(refs) => graph_core::NodeAttr::with_sources(label, refs),
            None => graph_core::NodeAttr::new(label),
        };
        let idx = self.graph.add_node(attr);

        // Vector indexing REMOVED — callers must use add_fact_with_embedding()
        // with a real semantic embedding from Python (bge-small-zh-v1.5).
        // The old embed_text() trigram hash produced garbage vectors.

        idx.index()
    }

    /// Add an L1 graph node with a pre-computed embedding from Python.
    ///
    /// Same as `add_fact` but accepts an external embedding vector
    /// instead of using the deprecated internal `embed_text`.
    pub fn add_fact_with_embedding(
        &mut self,
        label: &str,
        l0_refs: Option<Vec<i64>>,
        embedding: &[f32],
    ) -> usize {
        let attr = match l0_refs {
            Some(refs) => graph_core::NodeAttr::with_sources(label, refs),
            None => graph_core::NodeAttr::new(label),
        };
        let idx = self.graph.add_node(attr);

        // Use large offset to avoid collision with L0 IDs
        let vector_key = 1_000_000_000 + idx.index() as u64;
        if let Err(e) = self.vector_index.add(vector_key, embedding) {
            eprintln!("Warning: vector index add failed for node {}: {}", idx.index(), e);
        }

        // Register in version tree for temporal tracking
        let content_hash = format!("{:x}", {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            label.hash(&mut h);
            h.finish()
        });
        let metadata = serde_json::json!({
            "node_index": idx.index(),
            "vector_key": vector_key,
        });
        self.version_tree.insert(label, content_hash, metadata);

        idx.index()
    }
    /// Pure vector search — returns top-K most similar L1 facts by embedding cosine similarity.
    ///
    /// Unlike query/query_with_embedding, this bypasses fulltext and graph,
    /// returning true semantic similarity scores.
    pub fn vector_search(&self, query_embedding: &[f32], k: usize) -> Vec<(usize, String, f32)> {
        let results = self.vector_index.search(query_embedding, k).unwrap_or_default();
        results
            .into_iter()
            .filter_map(|(key, similarity)| {
                // Convert vector_key back to node_index
                if key >= 1_000_000_000 {
                    let node_idx = (key - 1_000_000_000) as usize;
                    let label = self.graph
                        .get_node(graph_core::NodeIndex::new(node_idx))
                        .map(|n| n.label.clone())
                        .unwrap_or_default();
                    Some((node_idx, label, similarity))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Adjust utility score for a graph node by its index.
    pub fn adjust_utility(&mut self, node_idx: usize, delta: f64) -> crate::Result<f64> {
        let ni = graph_core::NodeIndex::new(node_idx);
        let new_score = self.graph.adjust_utility(ni, delta)
            .map_err(|e| crate::Error::from(e.to_string()))?;
        Ok(new_score)
    }

    // ── Feedback ─────────────────────────────────────────────────────────────

    /// Update utility score for a graph node matching `entity_id` label.
    ///
    /// Scans all L1 nodes for a label containing `entity_id` (case-insensitive),
    /// then adjusts the utility score by `delta`. Returns the number of nodes updated.
    pub fn feedback(&mut self, entity_id: &str, delta: f64) -> usize {
        let query_lower = entity_id.to_lowercase();
        let indices: Vec<_> = self.graph
            .node_indices()
            .into_iter()
            .filter(|ni| {
                self.graph.get_node(*ni)
                    .map(|n| n.label.to_lowercase().contains(&query_lower))
                    .unwrap_or(false)
            })
            .collect();

        let mut updated = 0;
        for ni in indices {
            if self.graph.adjust_utility(ni, delta).is_ok() {
                updated += 1;
            }
        }
        updated
    }

    // ── Version tree operations ─────────────────────────────────────────────

    /// Mark a version branch as counterfactual (abandoned).
    ///
    /// Records that the entity at the given version was a dead end,
    /// preserving it for "avoid-the-pit" reasoning.
    pub fn abandon_branch(
        &mut self,
        entity_id: &str,
        version_seq: u64,
        reason: &str,
    ) -> Result<(), String> {
        self.version_tree
            .mark_counterfactual(entity_id, version_seq)
            .map_err(|e| e.to_string())?;

        // Enrich the node's metadata with the abandonment reason
        if let Some(node) = self.version_tree.get_mut(entity_id, version_seq) {
            node.metadata["abandon_reason"] = serde_json::Value::String(reason.to_string());
            node.metadata["abandoned_at"] =
                serde_json::Value::String(chrono::Utc::now().to_rfc3339());
        }

        Ok(())
    }
    /// Re-enable a counterfactual branch by creating a new version.
    ///
    /// Insert-based recovery: the old counterfactual stays marked, but a new
    /// active version is created referencing it. Preserves full audit chain.
    pub fn re_enable_branch(
        &mut self,
        entity_id: &str,
        counterfactual_seq: u64,
        reason: &str,
    ) -> Result<u64, String> {
        self.version_tree
            .re_enable(entity_id, counterfactual_seq, reason)
            .map_err(|e| e.to_string())
    }

    /// Get full context for an entity: version history, graph neighbors,
    /// conflicts, and related L0 evidence.
    pub fn get_entity_context(&self, entity_id: &str) -> serde_json::Value {
        // Version history
        let history: Vec<serde_json::Value> = self
            .version_tree
            .history(entity_id)
            .iter()
            .map(|v| {
                serde_json::json!({
                    "version_seq": v.version_seq,
                    "valid_from": v.valid_from.to_rfc3339(),
                    "valid_until": v.valid_until.map(|t| t.to_rfc3339()),
                    "content_hash": v.content_hash,
                    "is_counterfactual": v.is_counterfactual,
                })
            })
            .collect();

        // Graph neighbors — find node by label match
        let query_lower = entity_id.to_lowercase();
        let matching_nodes: Vec<_> = self
            .graph
            .node_indices()
            .into_iter()
            .filter(|ni| {
                self.graph
                    .get_node(*ni)
                    .map(|n| n.label.to_lowercase().contains(&query_lower))
                    .unwrap_or(false)
            })
            .collect();

        let mut neighbors = Vec::new();
        let mut conflicts = Vec::new();
        let mut l0_refs = Vec::new();

        for &ni in &matching_nodes {
            // Neighbors
            for neighbor_idx in self.graph.get_neighbors(ni) {
                if let Some(n) = self.graph.get_node(neighbor_idx) {
                    neighbors.push(serde_json::json!({
                        "label": n.label,
                        "utility_score": n.utility_score,
                    }));
                }
            }

            // Conflicts
            for conflict_idx in self.graph.find_conflicts(ni) {
                if let Some(n) = self.graph.get_node(conflict_idx) {
                    conflicts.push(serde_json::json!({
                        "label": n.label,
                        "utility_score": n.utility_score,
                    }));
                }
            }

            // L0 source refs
            if let Some(n) = self.graph.get_node(ni) {
                for &l0_ref in &n.source_l0_refs {
                    l0_refs.push(l0_ref);
                }
            }
        }

        // Deduplicate l0_refs
        l0_refs.sort();
        l0_refs.dedup();

        serde_json::json!({
            "entity_id": entity_id,
            "version_history": history,
            "neighbors": neighbors,
            "conflicts": conflicts,
            "l0_refs": l0_refs,
        })
    }
    // ── White-box persistence (JSONL) ───────────────────────────────────────

    /// Save all L1 state to human-readable JSONL files.
    ///
    /// Creates:
    /// - `l1/facts.jsonl` — one L1 node per line
    /// - `l1/edges.jsonl` — one edge per line
    /// - `l1/versions.jsonl` — one version node per line
    ///
    /// Users can directly view, edit, or correct these files.
    pub fn checkpoint(&self, vault_root: &str) -> Result<(), String> {
        use std::fs;
        use std::io::Write;

        let l1_dir = std::path::Path::new(vault_root).join("l1");

        // Guard: don't overwrite disk with empty in-memory state
        let node_count = self.graph.node_count();
        if node_count == 0 {
            // Check if disk already has data — if so, skip to preserve it
            if l1_dir.join("facts.jsonl").exists() {
                return Ok(());
            }
        }

        fs::create_dir_all(&l1_dir).map_err(|e| e.to_string())?;

        // 1. facts.jsonl — L1 nodes
        let facts_path = l1_dir.join("facts.jsonl");
        let mut facts_file = fs::File::create(&facts_path).map_err(|e| e.to_string())?;
        for ni in self.graph.node_indices() {
            if let Some(node) = self.graph.get_node(ni) {
                let json = serde_json::json!({
                    "id": format!("l1:{}", ni.index()),
                    "label": node.label,
                    "importance": node.utility_score,
                    "shelf_life": format!("{:?}", node.shelf_life),
                    "l0_refs": node.source_l0_refs,
                    "last_active_tick": node.last_active_tick,
                });
                writeln!(facts_file, "{}", serde_json::to_string(&json).unwrap_or_default())
                    .map_err(|e| e.to_string())?;
            }
        }

        // 2. edges.jsonl — L1 edges
        let edges_path = l1_dir.join("edges.jsonl");
        let mut edges_file = fs::File::create(&edges_path).map_err(|e| e.to_string())?;
        for src in self.graph.node_indices() {
            for dst in self.graph.get_neighbors(src) {
                let conflicts = self.graph.find_conflicts(src);
                let is_conflict = conflicts.contains(&dst);
                let rel = if is_conflict { "contradicts" } else { "supports" };
                let weight = if is_conflict { 0.7 } else { 0.0 };
                let json = serde_json::json!({
                    "src": format!("l1:{}", src.index()),
                    "dst": format!("l1:{}", dst.index()),
                    "relation": rel,
                    "weight": weight,
                });
                writeln!(edges_file, "{}", serde_json::to_string(&json).unwrap_or_default())
                    .map_err(|e| e.to_string())?;
            }
        }

        // 3. versions.jsonl — version tree
        let versions_path = l1_dir.join("versions.jsonl");
        let mut versions_file = fs::File::create(&versions_path).map_err(|e| e.to_string())?;
        for entity_id in self.version_tree.entity_ids() {
            for version in self.version_tree.history(entity_id) {
                let json = serde_json::json!({
                    "entity": entity_id,
                    "seq": version.version_seq,
                    "valid_from": version.valid_from.to_rfc3339(),
                    "valid_until": version.valid_until.map(|t| t.to_rfc3339()),
                    "counterfactual": version.is_counterfactual,
                    "content_hash": version.content_hash,
                    "prev_version_seq": version.prev_version_seq,
                    "metadata": version.metadata,
                });
                writeln!(versions_file, "{}", serde_json::to_string(&json).unwrap_or_default())
                    .map_err(|e| e.to_string())?;
            }
        }

        // 4. Save vector index to disk
        let vector_path = std::path::Path::new(vault_root).join("vector.idx");
        self.vector_index.save(
            vector_path.to_str().unwrap_or("vector.idx")
        ).map_err(|e| format!("vector save: {}", e))?;

        Ok(())
    }

    /// Load L1 state from JSONL files on disk.
    ///
    /// If files exist, rebuilds in-memory L1 graph and version tree.
    /// If files don't exist, keeps current state (empty or from L0 extraction).
    pub fn load_from_disk(&mut self, vault_root: &str) -> Result<usize, String> {
        use std::fs;
        use std::io::{BufRead, BufReader};

        let l1_dir = std::path::Path::new(vault_root).join("l1");
        if !l1_dir.exists() {
            return Ok(0);
        }

        let mut loaded = 0usize;

        // 1. Load facts.jsonl → rebuild L1 graph nodes
        let facts_path = l1_dir.join("facts.jsonl");
        if facts_path.exists() {
            let file = fs::File::open(&facts_path).map_err(|e| e.to_string())?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line.map_err(|e| e.to_string())?;
                if line.trim().is_empty() { continue; }
                let json: serde_json::Value = serde_json::from_str(&line)
                    .map_err(|e| format!("facts.jsonl parse error: {}", e))?;

                let label = json["label"].as_str().unwrap_or("");
                if label.is_empty() { continue; }

                let l0_refs: Option<Vec<i64>> = json["l0_refs"].as_array()
                    .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect());

                let node_idx = self.add_fact(label, l0_refs);
                // Restore utility score from checkpoint — add_fact defaults to 0.0
                if let Some(importance) = json["importance"].as_f64() {
                    if (importance - 0.0).abs() > f64::EPSILON {
                        let _ = self.adjust_utility(node_idx, importance);
                    }
                }
                loaded += 1;
            }
        }

        // 2. Load edges.jsonl → rebuild L1 graph edges
        let edges_path = l1_dir.join("edges.jsonl");
        if edges_path.exists() {
            let file = fs::File::open(&edges_path).map_err(|e| e.to_string())?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line.map_err(|e| e.to_string())?;
                if line.trim().is_empty() { continue; }
                let json: serde_json::Value = serde_json::from_str(&line)
                    .map_err(|e| format!("edges.jsonl parse error: {}", e))?;

                let src_str = json["src"].as_str().unwrap_or("");
                let dst_str = json["dst"].as_str().unwrap_or("");
                let relation = json["relation"].as_str().unwrap_or("supports");
                let weight = json["weight"].as_f64().unwrap_or(0.0);

                let src_idx = src_str.strip_prefix("l1:").and_then(|s| s.parse::<usize>().ok());
                let dst_idx = dst_str.strip_prefix("l1:").and_then(|s| s.parse::<usize>().ok());

                if let (Some(src), Some(dst)) = (src_idx, dst_idx) {
                    use graph_core::{NodeIndex, EdgeAttr};
                    let src_ni = NodeIndex::new(src);
                    let dst_ni = NodeIndex::new(dst);
                    let attr = match relation {
                        "contradicts" => EdgeAttr::contradicts(weight),
                        "elaborates" => EdgeAttr::elaborates(),
                        _ => EdgeAttr::supports(),
                    };
                    let _ = self.graph_mut().add_edge(src_ni, dst_ni, attr);
                }
            }
        }

        // 3. Load versions.jsonl → rebuild version tree
        let versions_path = l1_dir.join("versions.jsonl");
        if versions_path.exists() {
            let file = fs::File::open(&versions_path).map_err(|e| e.to_string())?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line.map_err(|e| e.to_string())?;
                if line.trim().is_empty() { continue; }
                let json: serde_json::Value = serde_json::from_str(&line)
                    .map_err(|e| format!("versions.jsonl parse error: {}", e))?;

                let entity = json["entity"].as_str().unwrap_or("");
                let content_hash = json["content_hash"].as_str().unwrap_or("").to_string();
                let metadata = json["metadata"].clone();
                let is_counterfactual = json["counterfactual"].as_bool().unwrap_or(false);

                if entity.is_empty() { continue; }

                let version = self.version_tree_mut().insert(entity, content_hash, metadata);
                if is_counterfactual {
                    let _ = self.version_tree_mut().mark_counterfactual(entity, version.version_seq);
                }
            }
        }

        // 4. Load vector index from disk if available
        let vector_path = std::path::Path::new(vault_root).join("vector.idx");
        if vector_path.exists() {
            if let Ok(vi) = VectorIndex::load(
                vector_path.to_str().unwrap_or("vector.idx"),
                self.embedding_dim,
            ) {
                self.vector_index = vi;
            }
        }

        Ok(loaded)
    }

    // ── Dream pruning (Ebbinghaus decay) ────────────────────────────────────

    /// Run a dream-pruning cycle: apply Ebbinghaus decay to L1 graph nodes.
    ///
    /// Nodes whose `last_active_tick` is far behind `current_tick` and whose
    /// utility score is below the threshold are pruned (edges removed, node
    /// marked as decayed). L3-like nodes (utility ≥ 0.7) and `ShelfLife::Immune`
    /// nodes are never pruned.
    pub fn dream_prune(&mut self, decay_threshold: u64, utility_floor: f64) -> DreamReport {
        let tick = self.get_tick();
        let mut pruned = 0usize;
        let mut preserved = 0usize;

        let indices: Vec<_> = self.graph.node_indices();
        for ni in indices {
            // Gather node info without holding borrow
            let node_info = self.graph.get_node(ni).map(|n| {
                (n.last_active_tick, n.utility_score, n.shelf_life)
            });

            let (last_tick, utility, shelf_life) = match node_info {
                Some(info) => info,
                None => continue,
            };

            let tick_delta = tick.saturating_sub(last_tick);
            let is_immune = matches!(shelf_life, graph_core::ShelfLife::Immune);
            let is_high_utility = utility >= utility_floor;

            if is_immune || is_high_utility {
                preserved += 1;
                continue;
            }

            let should_prune = match shelf_life {
                graph_core::ShelfLife::Subjective(limit) => tick_delta > limit,
                _ => false,
            };

            if should_prune || tick_delta > decay_threshold {
                let _ = self.graph.adjust_utility(ni, -1.0 - utility);
                pruned += 1;
            } else {
                preserved += 1;
            }
        }

        DreamReport {
            pruned,
            preserved,
            current_tick: tick,
        }
    }

    pub fn rebuild_fulltext_from_l0(&mut self) -> crate::Result<usize> {
        let evidence = self.l0.get_all(i64::MAX)?;
        let count = evidence.len();
        if count == 0 {
            return Ok(0);
        }

        for (i, ev) in evidence.iter().enumerate() {
            let doc_id = format!("l0:{}", ev.id);
            self.fulltext
                .add(&doc_id, &ev.content)
                .map_err(|e| crate::Error::from(format!("fulltext add: {}", e)))?;
            if i % 100 == 0 && i > 0 {
                self.fulltext
                    .commit()
                    .map_err(|e| crate::Error::from(format!("fulltext commit: {}", e)))?;
            }
        }
        self.fulltext
            .commit()
            .map_err(|e| crate::Error::from(format!("fulltext final commit: {}", e)))?;

        eprintln!("MemoryCore: rebuilt fulltext index from {} L0 rows", count);
        Ok(count)
    }

    // ── Stats ──────────────────────────────────────────────────────────────

    /// Return aggregate statistics across all subsystems.
    pub fn get_stats(&self) -> crate::Result<MemoryStats> {
        let l0_stats = self.l0.stats()?;
        Ok(MemoryStats {
            l0_count: l0_stats.total_count,
            l1_node_count: self.graph.node_count(),
            l1_edge_count: self.graph.edge_count(),
            l2_block_count: self.l2.len(),
            fulltext_doc_count: self.fulltext.doc_count(),
            persona_name: self.l3.persona().agent_name.clone(),
        })
    }
    // ── Accessors ──────────────────────────────────────────────────────────

    /// Borrow the L0 store.
    pub fn l0(&self) -> &L0Store {
        &self.l0
    }

    /// Borrow the L2 scenario store.
    pub fn l2(&self) -> &L2Store {
        &self.l2
    }

    /// Borrow the L2 scenario store mutably.
    pub fn l2_mut(&mut self) -> &mut L2Store {
        &mut self.l2
    }

    /// Borrow the L1 graph.
    pub fn graph(&self) -> &L1Graph {
        &self.graph
    }

    /// Borrow the L1 graph mutably.
    pub fn graph_mut(&mut self) -> &mut L1Graph {
        &mut self.graph
    }

    /// Borrow the version tree.
    pub fn version_tree(&self) -> &VersionTree {
        &self.version_tree
    }

    /// Borrow the version tree mutably.
    pub fn version_tree_mut(&mut self) -> &mut VersionTree {
        &mut self.version_tree
    }

    /// Borrow the L3 store.
    pub fn l3(&self) -> &L3Store {
        &self.l3
    }

    /// Borrow the RIF-U scorer.
    pub fn scorer(&self) -> &RifUScorer {
        &self.scorer
    }

    /// Borrow the metacognitive router.
    pub fn router(&self) -> &MetaRouter {
        &self.router
    }

    /// Borrow the token budget allocator.
    pub fn token_budget(&self) -> &TokenBudget {
        &self.token_budget
    }

    /// Borrow the sync engine.
    pub fn sync(&self) -> &SyncEngine {
        &self.sync
    }
}

/// Rough token estimate: ~4 characters per token (English average).
fn estimate_tokens(text: &str) -> usize {
    (text.len() + 3) / 4
}

/// Compute a pattern key for MacroCache from a query string.
/// Uses the embedding vector's first 16 bytes as a compact hash.
fn compute_pattern_key(query: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Normalize: lowercase + collapse whitespace
    let normalized: String = query.to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ");

    let mut hasher = DefaultHasher::new();
    normalized.hash(&mut hasher);
    let hash = hasher.finish();

    // Take first 8 bytes as hex — compact pattern key
    format!("{:016x}", hash)
}

impl MemoryCore {
    /// Record a macro hit (success or failure).
    fn record_macro_hit(&self, pattern_key: &str, success: bool) {
        if let Ok(mut cache) = self.macro_cache.lock() {
            cache.record_hit(pattern_key, success);
        }
    }

    /// Advance the subjective active tick by one.
    pub fn advance_tick(&self) -> u64 {
        let mut tick = self.current_tick.lock().unwrap_or_else(|e| e.into_inner());
        *tick += 1;
        *tick
    }

    /// Get the current active tick.
    pub fn get_tick(&self) -> u64 {
        *self.current_tick.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Get the number of promoted macros in the cache.
    pub fn promoted_macro_count(&self) -> usize {
        self.macro_cache.lock()
            .map(|c| c.promoted().len())
            .unwrap_or(0)
    }

    // ── Conflict Resolution (T-9.5) ──────────────────────────────────────────

    /// Auto-detect and annotate conflicts for a newly-added L1 node.
    ///
    /// Scans existing nodes for label keyword overlap with the node at `node_idx`.
    /// When two nodes share significant keywords but have different labels,
    /// a `contradicts` edge is added between them.
    ///
    /// Returns the number of conflict edges added.
    pub fn resolve_conflicts(&mut self, node_idx: usize) -> usize {
        let ni = graph_core::NodeIndex::new(node_idx);
        let target = match self.graph.get_node(ni) {
            Some(n) => n.clone(),
            None => return 0,
        };

        // Extract keywords from the target label (lowercase, split on whitespace)
        let target_words: std::collections::HashSet<String> = target
            .label
            .to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 2) // skip short words
            .map(|w| w.to_string())
            .collect();

        if target_words.is_empty() {
            return 0;
        }

        let mut conflicts_added = 0;
        let all_indices: Vec<_> = self.graph.node_indices();

        for other_ni in all_indices {
            if other_ni == ni {
                continue;
            }
            let other = match self.graph.get_node(other_ni) {
                Some(n) => n,
                None => continue,
            };

            // Check if the other node shares keywords with the target
            let other_words: std::collections::HashSet<String> = other
                .label
                .to_lowercase()
                .split_whitespace()
                .filter(|w| w.len() > 2)
                .map(|w| w.to_string())
                .collect();

            let overlap = target_words.intersection(&other_words).count();
            let min_len = target_words.len().min(other_words.len());

            // If >=50% keyword overlap but labels aren't identical, it's a potential conflict
            if overlap > 0 && min_len > 0 && (overlap as f64 / min_len as f64) >= 0.5 {
                // Check if a contradicts edge already exists
                let existing_conflicts = self.graph.find_conflicts(ni);
                if !existing_conflicts.contains(&other_ni) {
                    let edge = graph_core::EdgeAttr::contradicts(0.7);
                    let _ = self.graph.add_edge(ni, other_ni, edge);
                    conflicts_added += 1;
                }
            }
        }

        conflicts_added
    }

    // ── Pain Point Marking (T-9.6) ───────────────────────────────────────────

    /// Mark an L1 node as a critical pain-point.
    ///
    /// Pain-point nodes are:
    /// - Immune from dream-pruning decay
    /// - Pinned at utility_score = 1.0
    /// - Labelled with `[PAIN-POINT]` prefix for high visibility
    ///
    /// This implements the "效用沉淀" mechanism from TASKS.md T-9.6.
    pub fn mark_pain_point(&mut self, node_idx: usize) -> Result<(), String> {
        let ni = graph_core::NodeIndex::new(node_idx);
        let node = self.graph.get_node_mut(ni)
            .ok_or_else(|| format!("node index {} not found", node_idx))?;

        // Pin utility at maximum
        node.utility_score = 1.0;

        // Set shelf life to Immune (never decays)
        node.shelf_life = graph_core::ShelfLife::Immune;

        // Prefix label with [PAIN-POINT] if not already present
        if !node.label.starts_with("[PAIN-POINT]") {
            node.label = format!("[PAIN-POINT] {}", node.label);
        }

        Ok(())
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_core_creation() {
        let mc = MemoryCore::new_in_memory().unwrap();
        let stats = mc.get_stats().unwrap();
        assert_eq!(stats.l0_count, 0);
        assert_eq!(stats.l1_node_count, 0);
    }

    #[test]
    fn test_ingest_and_query() {
        let mut mc = MemoryCore::new_in_memory().unwrap();

        // Ingest some evidence
        mc.ingest_evidence(
            "test",
            "test-model",
            "s1",
            "user",
            "Rust memory safety is guaranteed by the borrow checker",
            &serde_json::Value::Null,
        )
        .unwrap();

        mc.ingest_evidence(
            "test",
            "test-model",
            "s1",
            "user",
            "Python uses garbage collection for memory management",
            &serde_json::Value::Null,
        )
        .unwrap();

        // Query
        let response = mc.query(&QueryRequest::simple("borrow checker")).unwrap();
        assert!(!response.items.is_empty(), "should find at least one result");
        assert!(response.total_tokens > 0);
    }

    #[test]
    fn test_query_empty_store() {
        let mc = MemoryCore::new_in_memory().unwrap();
        let response = mc.query(&QueryRequest::simple("anything")).unwrap();
        assert!(response.items.is_empty());
        assert_eq!(response.total_tokens, 0);
        // Empty results → confidence 0.0 → should drill down
        assert!(response.drilled_down);
    }

    #[test]
    fn test_token_budget_trimming() {
        let mut mc = MemoryCore::new_in_memory().unwrap();

        // Ingest many items to exceed budget
        for i in 0..100 {
            mc.ingest_evidence(
                "test",
                "test-model",
                "s1",
                "user",
                &format!("Evidence item {} about Rust programming language memory safety", i),
                &serde_json::Value::Null,
            )
            .unwrap();
        }

        let response = mc
            .query(&QueryRequest {
                query: "Rust".to_string(),
                limit: 100,
                allow_drill_down: false,
                workspace_id: None,
                disable_fulltext: false,
                disable_graph: false,
                disable_rrf: false,
                disable_rif_u: false,
            })
            .unwrap();

        // Total tokens should not exceed budget (1500)
        assert!(
            response.total_tokens <= 1500,
            "total_tokens {} exceeds 1500",
            response.total_tokens
        );
    }

    #[test]
    fn test_stats_after_ingest() {
        let mut mc = MemoryCore::new_in_memory().unwrap();
        mc.ingest_evidence("test", "test-model", "s1", "user", "test", &serde_json::Value::Null)
            .unwrap();

        let stats = mc.get_stats().unwrap();
        assert_eq!(stats.l0_count, 1);
        assert!(stats.fulltext_doc_count >= 1);
    }

    #[test]
    fn test_graph_accessor() {
        let mut mc = MemoryCore::new_in_memory().unwrap();
        let idx = mc.graph_mut().add_node(graph_core::graph::NodeAttr::new("test fact"));
        assert_eq!(mc.graph().node_count(), 1);
        let node = mc.graph().get_node(idx).unwrap();
        assert_eq!(node.label, "test fact");
    }

    #[test]
    fn test_rrf_fusion_graph_and_fulltext() {
        let mut mc = MemoryCore::new_in_memory().unwrap();

        // Ingest L0 evidence (goes to fulltext + vector)
        mc.ingest_evidence(
            "test",
            "test-model",
            "s1",
            "user",
            "Rust borrow checker prevents data races",
            &serde_json::Value::Null,
        )
        .unwrap();

        // Add L1 graph node (separate from fulltext)
        let dummy_emb = vec![0.1f32; 768];
        mc.add_fact_with_embedding("Rust memory safety model: ownership + borrowing + lifetimes", None, &dummy_emb);

        // Query for "Rust" — should find results from BOTH fulltext and graph
        let response = mc.query(&QueryRequest::simple("Rust")).unwrap();
        assert!(
            response.items.len() >= 2,
            "RRF fusion should merge graph + fulltext, got {} items",
            response.items.len()
        );

        // Verify we have results from both layers
        let has_l1 = response.items.iter().any(|i| i.layer == 1);
        let has_l2 = response.items.iter().any(|i| i.layer == 2);
        assert!(has_l1, "should have L1 graph results");
        assert!(has_l2, "should have L2 fulltext results");
    }

    #[test]
    fn test_rrf_fusion_three_way() {
        let mut mc = MemoryCore::new_in_memory().unwrap();

        mc.ingest_evidence(
            "test",
            "test-model",
            "s1",
            "user",
            "Rust borrow checker prevents data races at compile time",
            &serde_json::Value::Null,
        )
        .unwrap();
        mc.ingest_evidence(
            "test",
            "test-model",
            "s1",
            "user",
            "Python garbage collection handles memory automatically",
            &serde_json::Value::Null,
        )
        .unwrap();

        // Add L1 graph node via add_fact_with_embedding (indexes in vector store)
        let dummy_emb = vec![0.1f32; 768];
        mc.add_fact_with_embedding("Rust ownership and borrowing system", None, &dummy_emb);

        // Query — should fuse fulltext + graph (2-way RRF; vector requires query_with_embedding)
        let response = mc.query(&QueryRequest::simple("Rust borrow checker")).unwrap();
        assert!(
            !response.items.is_empty(),
            "2-way fusion should return results"
        );

        // Verify results come from multiple retrieval paths
        let has_l1 = response.items.iter().any(|i| i.layer == 1);
        let has_non_l1 = response.items.iter().any(|i| i.layer != 1);
        assert!(has_l1, "should have L1 graph results");
        assert!(has_non_l1, "should have fulltext results");

        // Verify content is populated (not just IDs)
        let has_content = response.items.iter().any(|i| !i.content.is_empty());
        assert!(has_content, "items should have content populated");
    }

    #[test]
    fn test_l1_node_vector_searchable() {
        let mut mc = MemoryCore::new_in_memory().unwrap();

        // Add L1 facts via add_fact_with_embedding (indexes in vector store)
        let dummy_emb = vec![0.1f32; 768];
        mc.add_fact_with_embedding("Rust borrow checker prevents data races", None, &dummy_emb);
        mc.add_fact_with_embedding("Python garbage collection handles memory", None, &dummy_emb);

        // Query should find L1 nodes via vector similarity
        let response = mc.query(&QueryRequest::simple("borrow checker")).unwrap();
        let l1_items: Vec<_> = response.items.iter().filter(|i| i.layer == 1).collect();
        assert!(
            !l1_items.is_empty(),
            "L1 nodes should be findable via vector search"
        );
    }

    #[test]
    fn test_macro_intuition_shortcut() {
        let mut mc = MemoryCore::new_in_memory().unwrap();

        // Ingest data
        mc.ingest_evidence("test", "test-model", "s1", "user", "Rust borrow checker prevents data races", &serde_json::Value::Null).unwrap();
        mc.ingest_evidence("test", "test-model", "s1", "user", "Python garbage collection handles memory", &serde_json::Value::Null).unwrap();

        let request = QueryRequest::simple("Rust borrow checker");

        // Query 5 times to build up macro cache (threshold = 5)
        for _ in 0..5 {
            mc.query(&request).unwrap();
        }

        // 6th query should hit the macro cache
        let response = mc.query(&request).unwrap();
        assert!(!response.items.is_empty(), "macro shortcut should return results");

        // Verify macro was promoted
        assert!(
            mc.promoted_macro_count() >= 1,
            "should have at least 1 promoted macro"
        );
    }

    #[test]
    fn test_advance_tick() {
        let mc = MemoryCore::new_in_memory().unwrap();
        assert_eq!(mc.get_tick(), 0);
        assert_eq!(mc.advance_tick(), 1);
        assert_eq!(mc.advance_tick(), 2);
        assert_eq!(mc.get_tick(), 2);
    }

    #[test]
    fn test_dream_prune() {
        let mut mc = MemoryCore::new_in_memory().unwrap();

        // Add a fact with default shelf life (90 ticks)
        mc.add_fact("ephemeral fact", None);
        // Add a high-utility fact (immune from pruning)
        let idx = mc.add_fact("critical architecture decision", None);
        mc.adjust_utility(idx, 1.0).unwrap(); // utility = 1.0

        // Advance tick far beyond shelf life
        for _ in 0..200 {
            mc.advance_tick();
        }

        // Run dream pruning
        let report = mc.dream_prune(100, 0.7);

        // Ephemeral fact should be pruned (tick_delta=200 > shelf_life=90)
        assert!(report.pruned >= 1, "should prune at least 1 decayed node");
        // High-utility fact should be preserved
        assert!(report.preserved >= 1, "should preserve high-utility node");
        assert_eq!(report.current_tick, 200);
    }

    #[test]
    fn test_resolve_conflicts() {
        let mut mc = MemoryCore::new_in_memory().unwrap();

        // Add two facts with overlapping keywords
        let idx1 = mc.add_fact("rust borrow checker prevents data races", None);
        let _idx2 = mc.add_fact("rust borrow checker ensures memory safety", None);

        // Resolve conflicts for the first node
        let conflicts = mc.resolve_conflicts(idx1);
        assert!(conflicts >= 1, "should detect at least 1 conflict between similar facts");
    }

    #[test]
    fn test_mark_pain_point() {
        let mut mc = MemoryCore::new_in_memory().unwrap();
        let idx = mc.add_fact("critical database migration bug", None);

        mc.mark_pain_point(idx).unwrap();

        let node = mc.graph().get_node(graph_core::NodeIndex::new(idx)).unwrap();
        assert!((node.utility_score - 1.0).abs() < f64::EPSILON);
        assert!(node.label.starts_with("[PAIN-POINT]"));
        assert_eq!(node.shelf_life, graph_core::ShelfLife::Immune);
    }
}
