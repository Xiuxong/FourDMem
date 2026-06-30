//! py-bindings: PyO3 wrapper exposing FourDMem to Python
//!
//! Uses `MemoryCore` as the unified entry point — all retrieval goes through
//! the RRF 3-way fusion pipeline (fulltext + graph + vector).

use std::sync::Mutex;

use pyo3::prelude::*;

use storage_core::MemoryCore;
use graph_core::TdaAnalyzer;
use evolution_core::{CognitiveDna, MacroCache};

// ── Python module ─────────────────────────────────────────────────────────────

#[pymodule]
fn fourdmem(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FourDMemEngine>()?;
    Ok(())
}

// ── Core engine ───────────────────────────────────────────────────────────────

/// The FourDMem engine exposed to Python.
///
/// Delegates to `MemoryCore` for all retrieval operations, ensuring Python
/// uses the same RRF 3-way fusion pipeline as the Rust tests.
#[pyclass]
struct FourDMemEngine {
    core: Mutex<MemoryCore>,
    dna: Mutex<CognitiveDna>,
    macros: Mutex<MacroCache>,
    current_tick: Mutex<u64>,
}

#[pymethods]
impl FourDMemEngine {
    /// Create a new engine instance.
    ///
    /// `db_path` is the path to the L0 SQLite database.
    /// Pass `:memory:` for an ephemeral in-memory database.
    #[new]
    fn new(db_path: &str) -> PyResult<Self> {
        let core = if db_path == ":memory:" {
            MemoryCore::new_in_memory()
        } else {
            // Use the db_path directory as vault_root for simplicity
            let vault_root = std::path::Path::new(db_path)
                .parent()
                .unwrap_or(std::path::Path::new("."))
                .to_string_lossy()
                .to_string();
            MemoryCore::new(db_path, &vault_root)
        }
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            core: Mutex::new(core),
            dna: Mutex::new(CognitiveDna::new()),
            macros: Mutex::new(MacroCache::new(20)),
            current_tick: Mutex::new(0),
        })
    }

    /// Search memory with a four-dimensional query.
    ///
    /// Goes through the full RRF pipeline: fulltext + graph + vector → fusion →
    /// token budget → metacognitive routing.
    #[pyo3(signature = (query, limit=None, workspace_id=None, disable_fulltext=false, disable_graph=false, disable_rrf=false, disable_rif_u=false))]
    fn query(&self, query: &str, limit: Option<usize>, workspace_id: Option<&str>, disable_fulltext: bool, disable_graph: bool, disable_rrf: bool, disable_rif_u: bool) -> PyResult<String> {
        let mut core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        let request = storage_core::QueryRequest {
            query: query.to_string(),
            limit: limit.unwrap_or(10),
            allow_drill_down: true,
            workspace_id: workspace_id.map(|s| s.to_string()),
            disable_fulltext,
            disable_graph,
            disable_rrf,
            disable_rif_u,
        };

        let response = core.query(&request).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })?;

        let tick = *self.current_tick.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        let result = serde_json::json!({
            "query": query,
            "results": response.items.iter().map(|item| {
                let mut obj = serde_json::json!({
                    "id": item.id,
                    "layer": item.layer,
                    "score": item.score,
                    "content": item.content,
                    "token_estimate": item.token_estimate,
                });
                if !item.timestamp.is_empty() {
                    obj["timestamp"] = serde_json::Value::String(item.timestamp.clone());
                }
                if !item.metadata.is_null() {
                    obj["metadata"] = item.metadata.clone();
                }
                obj
            }).collect::<Vec<_>>(),
            "total_tokens": response.total_tokens,
            "confidence": response.confidence,
            "drilled_down": response.drilled_down,
            "current_tick": tick,
        });

        serde_json::to_string_pretty(&result).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    /// Save evidence to L0 storage, index in fulltext + vector.
    ///
    /// Returns the new evidence ID as a JSON string.
    #[pyo3(signature = (session_id, role, content, metadata=None, workspace_id=None))]
    fn save(&self, session_id: &str, role: &str, content: &str, metadata: Option<&str>, workspace_id: Option<&str>) -> PyResult<String> {
        let ws = workspace_id.unwrap_or("default");
        let meta: serde_json::Value = metadata
            .and_then(|m| serde_json::from_str(m).ok())
            .unwrap_or(serde_json::Value::Null);

        let mut core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        let model_name = meta.get("model_name").and_then(|v| v.as_str()).unwrap_or("unknown");
        let id = core.ingest_evidence(ws, model_name, session_id, role, content, &meta).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })?;

        let response = serde_json::json!({
            "status": "saved",
            "evidence_id": id,
            "workspace_id": ws,
            "session_id": session_id,
            "role": role,
            "content_length": content.len(),
        });

        serde_json::to_string_pretty(&response).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    /// Like `query`, but uses a pre-computed embedding from Python
    /// instead of the internal placeholder embedding.
    ///
    /// `embedding` should be a list of floats (e.g. from sentence-transformers).
    #[pyo3(signature = (query, embedding, limit=None, workspace_id=None, disable_fulltext=false, disable_graph=false, disable_rrf=false, disable_rif_u=false))]
    fn query_with_embedding(&self, query: &str, embedding: Vec<f32>, limit: Option<usize>, workspace_id: Option<&str>, disable_fulltext: bool, disable_graph: bool, disable_rrf: bool, disable_rif_u: bool) -> PyResult<String> {
        let core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        let request = storage_core::QueryRequest {
            query: query.to_string(),
            limit: limit.unwrap_or(10),
            allow_drill_down: true,
            workspace_id: workspace_id.map(|s| s.to_string()),
            disable_fulltext,
            disable_graph,
            disable_rrf,
            disable_rif_u,
        };

        let response = core.query_with_embedding(&request, &embedding).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })?;

        let tick = *self.current_tick.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        let result = serde_json::json!({
            "query": query,
            "results": response.items.iter().map(|item| {
                let mut obj = serde_json::json!({
                    "id": item.id,
                    "layer": item.layer,
                    "score": item.score,
                    "content": item.content,
                    "token_estimate": item.token_estimate,
                });
                if !item.timestamp.is_empty() {
                    obj["timestamp"] = serde_json::Value::String(item.timestamp.clone());
                }
                if !item.metadata.is_null() {
                    obj["metadata"] = item.metadata.clone();
                }
                obj
            }).collect::<Vec<_>>(),
            "total_tokens": response.total_tokens,
            "confidence": response.confidence,
            "drilled_down": response.drilled_down,
            "current_tick": tick,
        });

        serde_json::to_string_pretty(&result).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }
    /// Like `save`, but uses a pre-computed embedding from Python.
    #[pyo3(signature = (session_id, role, content, embedding, metadata=None, workspace_id=None))]
    fn save_with_embedding(&self, session_id: &str, role: &str, content: &str, embedding: Vec<f32>, metadata: Option<&str>, workspace_id: Option<&str>) -> PyResult<String> {
        let ws = workspace_id.unwrap_or("default");
        let meta: serde_json::Value = metadata
            .and_then(|m| serde_json::from_str(m).ok())
            .unwrap_or(serde_json::Value::Null);

        let mut core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        let id = core.ingest_with_embedding(ws, session_id, role, content, &meta, &embedding).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })?;

        let response = serde_json::json!({
            "status": "saved",
            "evidence_id": id,
            "workspace_id": ws,
            "session_id": session_id,
            "role": role,
            "content_length": content.len(),
        });

        serde_json::to_string_pretty(&response).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    /// Drill down from a result to its L0 raw evidence.
    fn drill_down(&self, entity_id: &str) -> PyResult<String> {
        let core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        // Try to parse as L0 ID
        if let Ok(l0_id) = entity_id.parse::<i64>() {
            let evidence = core.drill_down(l0_id).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
            })?;

            let response = serde_json::json!({
                "entity_id": entity_id,
                "evidence": evidence.iter().map(|ev| {
                    serde_json::json!({
                        "id": ev.id,
                        "session_id": ev.session_id,
                        "timestamp": ev.timestamp,
                        "role": ev.role,
                        "content": ev.content,
                    })
                }).collect::<Vec<_>>(),
                "total_evidence": evidence.len(),
            });

            serde_json::to_string_pretty(&response).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
            })
        } else {
            Ok(serde_json::json!({
                "entity_id": entity_id,
                "evidence": [],
                "message": "Entity ID is not a numeric L0 reference"
            }).to_string())
        }
    }

    /// Submit external feedback to update utility scores.
    ///
    /// Scans L1 graph nodes for matching labels and adjusts their utility.
    fn feedback(&self, entity_id: &str, score: f64) -> PyResult<String> {
        let mut core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        let updated = core.feedback(entity_id, score);

        let response = serde_json::json!({
            "status": "feedback_applied",
            "entity_id": entity_id,
            "score": score,
            "nodes_updated": updated,
        });

        serde_json::to_string_pretty(&response).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    /// Add an L1 graph node (atomic fact) and index in vector store.
    #[pyo3(signature = (label, l0_refs=None))]
    fn add_fact(&self, label: &str, l0_refs: Option<Vec<i64>>) -> PyResult<String> {
        let mut core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        let node_idx = core.add_fact(label, l0_refs);

        let response = serde_json::json!({
            "status": "fact_added",
            "label": label,
            "node_index": node_idx,
        });

        serde_json::to_string_pretty(&response).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    /// Add an L1 graph node with a pre-computed embedding from Python.
    /// Use this instead of add_fact() to ensure the vector index gets
    /// real semantic embeddings (bge-small-zh-v1.5) instead of trigram hashes.
    #[pyo3(signature = (label, l0_refs, embedding))]
    fn add_fact_with_embedding(&self, label: &str, l0_refs: Option<Vec<i64>>, embedding: Vec<f32>) -> PyResult<String> {
        let mut core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        let node_idx = core.add_fact_with_embedding(label, l0_refs, &embedding);

        let response = serde_json::json!({
            "status": "fact_added",
            "label": label,
            "node_index": node_idx,
        });

        serde_json::to_string_pretty(&response).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    /// Detect and auto-resolve conflicts for a newly-added L1 node.
    fn resolve_conflicts(&self, node_idx: usize) -> PyResult<usize> {
        let mut core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;
        Ok(core.resolve_conflicts(node_idx))
    }

    /// Mark an L1 node as a critical pain-point (immune from decay).
    fn mark_pain_point(&self, node_idx: usize) -> PyResult<()> {
        let mut core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;
        core.mark_pain_point(node_idx)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    /// Trigger cognitive reflection on a topic.
    #[pyo3(signature = (topic, results_count=None, top_score=None))]
    fn reflect(&self, topic: &str, results_count: Option<usize>, top_score: Option<f64>) -> PyResult<String> {
        let core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        let count = results_count.unwrap_or(0);
        let score = top_score.unwrap_or(0.0);

        let confidence = core.router().evaluate_confidence(count, score);
        let should_drill = core.router().should_drill_down(confidence);

        let response = serde_json::json!({
            "topic": topic,
            "confidence": confidence,
            "should_drill_down": should_drill,
            "recommendation": if should_drill {
                "Confidence is low. Drill down to L1/L0 for better evidence."
            } else {
                "Confidence is sufficient. Current results are adequate."
            },
        });

        serde_json::to_string_pretty(&response).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    /// Cold-start wake-up: assess current memory state.
    fn wake_up(&self) -> PyResult<String> {
        let core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        let stats = core.get_stats().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })?;

        let tick = *self.current_tick.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        let response = serde_json::json!({
            "status": "awake",
            "current_tick": tick,
            "memory_stats": {
                "l0_evidence": stats.l0_count,
                "l1_nodes": stats.l1_node_count,
                "l1_edges": stats.l1_edge_count,
                "fulltext_docs": stats.fulltext_doc_count,
                "persona": stats.persona_name,
            },
            "message": "Wake-up complete. Memory state loaded.",
        });

        serde_json::to_string_pretty(&response).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    /// Trigger cognitive evolution cycle.
    fn evolve(&self) -> PyResult<String> {
        let dna = self.dna.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        let mutated = dna.mutate(0.1);

        let response = serde_json::json!({
            "current_dna": {
                "rif_weights": [dna.rif_weights.0, dna.rif_weights.1, dna.rif_weights.2, dna.rif_weights.3],
                "confidence_threshold": dna.confidence_threshold,
                "macro_compilation_threshold": dna.macro_compilation_threshold,
            },
            "proposed_mutation": {
                "rif_weights": [mutated.rif_weights.0, mutated.rif_weights.1, mutated.rif_weights.2, mutated.rif_weights.3],
                "confidence_threshold": mutated.confidence_threshold,
            },
            "message": "Mutation proposed. Use sandbox to evaluate before hot-swap.",
        });

        serde_json::to_string_pretty(&response).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    /// Hot-swap the cognitive DNA — replace current parameters with new ones.
    ///
    /// Used by the Observer node (strange loop) to self-modify system
    /// parameters when it detects inefficiency.
    fn hot_swap_dna(&self, new_dna_json: &str) -> PyResult<String> {
        let new_dna: CognitiveDna = serde_json::from_str(new_dna_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(
                format!("invalid DNA JSON: {}", e)
            ))?;

        let mut dna = self.dna.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        let old = serde_json::json!({
            "rif_weights": [dna.rif_weights.0, dna.rif_weights.1, dna.rif_weights.2, dna.rif_weights.3],
            "confidence_threshold": dna.confidence_threshold,
        });

        *dna = new_dna.clone();

        let response = serde_json::json!({
            "status": "dna_swapped",
            "old_dna": old,
            "new_dna": {
                "rif_weights": [new_dna.rif_weights.0, new_dna.rif_weights.1, new_dna.rif_weights.2, new_dna.rif_weights.3],
                "confidence_threshold": new_dna.confidence_threshold,
                "failure_rate_threshold": new_dna.failure_rate_threshold,
            },
        });

        serde_json::to_string_pretty(&response).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    /// Advance the subjective active tick.
    fn advance_tick(&self) -> PyResult<u64> {
        let mut tick = self.current_tick.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;
        *tick += 1;

        // Also advance MemoryCore's internal tick
        let core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;
        core.advance_tick();

        Ok(*tick)
    }

    /// Get the current active tick.
    fn get_tick(&self) -> PyResult<u64> {
        let tick = self.current_tick.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;
        Ok(*tick)
    }

    /// Run dream pruning: apply Ebbinghaus decay to L1 graph nodes.
    ///
    /// Nodes with low utility and old last_active_tick are pruned.
    /// High-utility (≥0.7) and Immune nodes survive.
    #[pyo3(signature = (decay_threshold=None, utility_floor=None))]
    fn dream_prune(&self, decay_threshold: Option<u64>, utility_floor: Option<f64>) -> PyResult<String> {
        let mut core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        let report = core.dream_prune(
            decay_threshold.unwrap_or(90),
            utility_floor.unwrap_or(0.7),
        );

        serde_json::to_string_pretty(&serde_json::json!({
            "pruned": report.pruned,
            "preserved": report.preserved,
            "current_tick": report.current_tick,
        })).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Mark a branch as counterfactual (abandon a decision path).
    ///
    /// Records that the entity at the given version was abandoned,
    /// preserving the reasoning as a "what not to do" lesson.
    fn abandon_branch(&self, entity_id: &str, version_seq: u64, reason: &str) -> PyResult<String> {
        let mut core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        core.abandon_branch(entity_id, version_seq, reason).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e)
        })?;

        let response = serde_json::json!({
            "status": "branch_abandoned",
            "entity_id": entity_id,
            "version_seq": version_seq,
            "reason": reason,
            "is_counterfactual": true,
        });

        serde_json::to_string_pretty(&response).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    /// Mark a branch as counterfactual with structured failure conditions.
    ///
    /// `conditions` is a JSON string describing the failure context,
    /// enabling future re-evaluation when conditions change.
    #[pyo3(signature = (entity_id, version_seq, reason, conditions = "{}"))]
    fn abandon_branch_with_conditions(&self, entity_id: &str, version_seq: u64, reason: &str, conditions: &str) -> PyResult<String> {
        let mut core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        let cond_value: Option<serde_json::Value> = serde_json::from_str(conditions).ok();
        core.abandon_branch_with_conditions(entity_id, version_seq, reason, cond_value.clone()).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e)
        })?;

        let response = serde_json::json!({
            "status": "branch_abandoned",
            "entity_id": entity_id,
            "version_seq": version_seq,
            "reason": reason,
            "conditions": cond_value,
            "is_counterfactual": true,
        });

        serde_json::to_string_pretty(&response).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }
    /// Re-enable a counterfactual branch by creating a new version.
    ///
    /// Insert-based recovery: old counterfactual stays, new active version created.
    fn re_enable_branch(&self, entity_id: &str, counterfactual_seq: u64, reason: &str) -> PyResult<String> {
        let mut core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        let new_seq = core.re_enable_branch(entity_id, counterfactual_seq, reason)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        let response = serde_json::json!({
            "status": "branch_re_enabled",
            "entity_id": entity_id,
            "counterfactual_seq": counterfactual_seq,
            "new_version_seq": new_seq,
            "reason": reason,
        });

        serde_json::to_string_pretty(&response).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    /// Get full context for an entity: version history, graph neighbors,
    /// conflicts, and related L0 evidence.
    fn get_entity_context(&self, entity_id: &str) -> PyResult<String> {
        let core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        let context = core.get_entity_context(entity_id);

        serde_json::to_string_pretty(&context).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }
    /// Save L1 state to human-readable JSONL files (white-box persistence).
    ///
    /// Creates facts.jsonl, edges.jsonl, versions.jsonl in vault_root/l1/.
    fn checkpoint(&self, vault_root: &str) -> PyResult<String> {
        let core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;
        core.checkpoint(vault_root).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e)
        })?;
        Ok(serde_json::json!({
            "status": "checkpoint_complete",
            "vault_root": vault_root,
        }).to_string())
    }

    /// Load L1 state from JSONL files on disk.
    ///
    /// Returns number of facts loaded.
    fn load_from_disk(&mut self, vault_root: &str) -> PyResult<usize> {
        let mut core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;
        core.load_from_disk(vault_root).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e)
        })
    }
    #[pyo3(signature = (session_id, limit=None, workspace_id=None))]
    fn get_session_evidence(&self, session_id: &str, limit: Option<i64>, workspace_id: Option<&str>) -> PyResult<String> {
        let ws = workspace_id.unwrap_or("default");
        let core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        let evidence = core.get_session_evidence(ws, session_id, limit.unwrap_or(100))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let response = serde_json::json!({
            "session_id": session_id,
            "evidence_count": evidence.len(),
            "evidence": evidence.iter().map(|ev| {
                serde_json::json!({
                    "id": ev.id,
                    "workspace_id": ev.workspace_id,
                    "session_id": ev.session_id,
                    "timestamp": ev.timestamp,
                    "role": ev.role,
                    "content": ev.content,
                })
            }).collect::<Vec<_>>(),
        });
        serde_json::to_string_pretty(&response).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    // ── T-3.4: Point-in-time snapshot query ──────────────────────────────

    /// Query the state of an entity at a specific timestamp.
    ///
    /// Args:
    ///     entity_id: The entity to query (e.g. graph node label).
    ///     timestamp: RFC3339 timestamp (e.g. "2025-03-15T00:00:00Z").
    ///
    /// Returns:
    ///     JSON with the version active at that timestamp, or null.
    fn query_at(&self, entity_id: &str, timestamp: &str) -> PyResult<String> {
        let core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        let ts = chrono::DateTime::parse_from_rfc3339(timestamp)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid timestamp: {}", e)))?
            .with_timezone(&chrono::Utc);

        let version = core.version_tree().query_at(entity_id, ts);

        let response = match version {
            Some(v) => serde_json::json!({
                "found": true,
                "entity_id": v.entity_id,
                "version_seq": v.version_seq,
                "valid_from": v.valid_from.to_rfc3339(),
                "valid_until": v.valid_until.map(|u| u.to_rfc3339()),
                "content_hash": v.content_hash,
                "is_counterfactual": v.is_counterfactual,
                "metadata": v.metadata,
            }),
            None => serde_json::json!({
                "found": false,
                "entity_id": entity_id,
                "timestamp": timestamp,
            }),
        };

        serde_json::to_string_pretty(&response).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    // ── T-3.5: Version diff & change traceability ────────────────────────

    /// Compare two versions of an entity and return a diff report.
    ///
    /// Args:
    ///     entity_id: The entity to diff.
    ///     seq_a: First version sequence number.
    ///     seq_b: Second version sequence number.
    ///
    /// Returns:
    ///     JSON with diff details including content change, time delta,
    ///     and metadata differences.
    fn get_diff(&self, entity_id: &str, seq_a: u64, seq_b: u64) -> PyResult<String> {
        let core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        match core.version_tree().get_diff(entity_id, seq_a, seq_b) {
            Some(diff) => {
                let response = serde_json::json!({
                    "found": true,
                    "entity_id": diff.entity_id,
                    "version_a": {
                        "seq": diff.version_a.version_seq,
                        "valid_from": diff.version_a.valid_from.to_rfc3339(),
                        "content_hash": diff.version_a.content_hash,
                    },
                    "version_b": {
                        "seq": diff.version_b.version_seq,
                        "valid_from": diff.version_b.valid_from.to_rfc3339(),
                        "content_hash": diff.version_b.content_hash,
                    },
                    "content_changed": diff.content_changed,
                    "time_delta_seconds": diff.time_delta_seconds,
                    "metadata_diff": diff.metadata_diff,
                });
                serde_json::to_string_pretty(&response).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
                })
            }
            None => {
                let response = serde_json::json!({
                    "found": false,
                    "entity_id": entity_id,
                    "seq_a": seq_a,
                    "seq_b": seq_b,
                });
                serde_json::to_string_pretty(&response).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
                })
            }
        }
    }

    // ── T-3.7: Version compression ───────────────────────────────────────

    /// Compress old versions of an entity.
    ///
    /// Versions older than `active_tick_threshold` sequence numbers behind
    /// the current head are merged into a single summary version.
    ///
    /// Args:
    ///     entity_id: The entity to compress.
    ///     threshold: Number of versions to keep uncompressed (default 3).
    ///
    /// Returns:
    ///     JSON with compression result.
    #[pyo3(signature = (entity_id, threshold=None))]
    fn compress_versions(&self, entity_id: &str, threshold: Option<u64>) -> PyResult<String> {
        let mut core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        let thresh = threshold.unwrap_or(3);
        match core.version_tree_mut().compress(entity_id, thresh) {
            Ok(removed) => {
                let response = serde_json::json!({
                    "status": "compressed",
                    "entity_id": entity_id,
                    "versions_removed": removed,
                    "threshold": thresh,
                });
                serde_json::to_string_pretty(&response).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
                })
            }
            Err(e) => {
                let response = serde_json::json!({
                    "status": "error",
                    "entity_id": entity_id,
                    "error": e.to_string(),
                });
                serde_json::to_string_pretty(&response).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
                })
            }
        }
    }

    // ── Graph access for NetworkX (T-7.2) ───────────────────────────────────

    /// Get the number of nodes in the L1 graph.
    fn graph_node_count(&self) -> PyResult<usize> {
        let core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;
        Ok(core.graph().node_count())
    }

    /// Get all node indices in the L1 graph as a list of integers.
    fn graph_node_indices(&self) -> PyResult<Vec<usize>> {
        let core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;
        Ok(core.graph().node_indices().iter().map(|ni| ni.index()).collect())
    }

    /// Get a node's attributes as JSON by its index.
    fn graph_get_node(&self, idx: usize) -> PyResult<String> {
        let core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;
        let ni = graph_core::NodeIndex::new(idx);
        let node = core.graph().get_node(ni)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(format!("node {} not found", idx)))?;

        let response = serde_json::json!({
            "label": node.label,
            "layer": node.layer,
            "utility_score": node.utility_score,
            "last_active_tick": node.last_active_tick,
            "shelf_life": format!("{:?}", node.shelf_life),
            "is_counterfactual": false,
            "embedding_dim": node.embedding.len(),
        });
        serde_json::to_string(&response).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    /// Get all edges as a list of [src, dst, relation_type, conflict_weight].
    fn graph_get_edges(&self) -> PyResult<Vec<(usize, usize, String, f64)>> {
        let core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;
        let g = core.graph();
        let mut edges = Vec::new();
        for idx in g.node_indices() {
            for neighbor in g.get_neighbors(idx) {
                // We don't have direct edge access, so infer from find_conflicts
                let conflicts = g.find_conflicts(idx);
                let is_conflict = conflicts.contains(&neighbor);
                let weight = if is_conflict { 0.7 } else { 0.0 };
                let rel = if is_conflict { "contradicts" } else { "supports" };
                edges.push((idx.index(), neighbor.index(), rel.to_string(), weight));
            }
        }
        Ok(edges)
    }
    /// Add a directed edge between two L1 graph nodes.
    ///
    /// relation_type: "supports", "contradicts", or "elaborates"
    /// conflict_weight: 0.0 for supports/elaborates, 0.0-1.0 for contradicts
    #[pyo3(signature = (src, dst, relation_type, conflict_weight=None))]
    fn graph_add_edge(&self, src: usize, dst: usize, relation_type: &str, conflict_weight: Option<f64>) -> PyResult<bool> {
        let mut core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;
        let src_ni = graph_core::NodeIndex::new(src);
        let dst_ni = graph_core::NodeIndex::new(dst);
        let weight = conflict_weight.unwrap_or(0.0);
        let attr = match relation_type {
            "contradicts" => graph_core::EdgeAttr::contradicts(weight),
            "elaborates" => graph_core::EdgeAttr::elaborates(),
            _ => graph_core::EdgeAttr::supports(),
        };
        match core.graph_mut().add_edge(src_ni, dst_ni, attr) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
    /// Pure vector search — returns top-K most similar L1 facts by embedding cosine similarity.
    ///
    /// Returns list of [node_index, label, similarity] tuples.
    /// Unlike query_with_embedding, this returns true cosine similarity, not RRF score.
    #[pyo3(signature = (query_embedding, k=10))]
    fn vector_search(&self, query_embedding: Vec<f32>, k: Option<usize>) -> PyResult<Vec<(usize, String, f32)>> {
        let core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;
        Ok(core.vector_search(&query_embedding, k.unwrap_or(10)))
    }

    // ── TDA Topology Analysis ──────────────────────────────────────────────

    /// Compute topological metrics for the L1 graph.
    ///
    /// Returns Betti numbers (β₀, β₁), density, clustering coefficient, etc.
    fn compute_topology(&self) -> PyResult<String> {
        let core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        let analyzer = TdaAnalyzer::new();
        let metrics = analyzer.compute_metrics(core.graph());

        serde_json::to_string_pretty(&serde_json::json!({
            "node_count": metrics.node_count,
            "edge_count": metrics.edge_count,
            "betti_0": metrics.betti_0,
            "betti_1": metrics.betti_1,
            "density": metrics.density,
            "avg_degree": metrics.avg_degree,
            "isolated_ratio": metrics.isolated_ratio,
            "clustering_coefficient": metrics.clustering_coeff,
        })).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Check if the L1 graph is approaching a topological phase transition.
    ///
    /// Returns signal details when β₁ (cycle count) exceeds the critical
    /// threshold (default 45 per RIF-SQCE.md), or density/isolation signals.
    #[pyo3(signature = (betti_threshold=None))]
    fn check_phase_transition(&self, betti_threshold: Option<u32>) -> PyResult<String> {
        let core = self.core.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        let analyzer = match betti_threshold {
            Some(t) => TdaAnalyzer::with_thresholds(t, 0.15, 0.3),
            None => TdaAnalyzer::new(),
        };
        let signal = analyzer.check_phase_transition(core.graph());

        serde_json::to_string_pretty(&signal).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    // ── Evolution API ─────────────────────────────────────────────────────

    /// Mutate the current cognitive DNA and return the proposed mutation.
    #[pyo3(signature = (max_delta=None))]
    fn mutate_dna(&self, max_delta: Option<f64>) -> PyResult<String> {
        let dna = self.dna.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        let mutated = dna.mutate(max_delta.unwrap_or(0.1));

        serde_json::to_string_pretty(&serde_json::json!({
            "original": {
                "rif_weights": [dna.rif_weights.0, dna.rif_weights.1, dna.rif_weights.2, dna.rif_weights.3],
                "confidence_threshold": dna.confidence_threshold,
            },
            "mutated": {
                "rif_weights": [mutated.rif_weights.0, mutated.rif_weights.1, mutated.rif_weights.2, mutated.rif_weights.3],
                "confidence_threshold": mutated.confidence_threshold,
                "betti_threshold": mutated.betti_threshold,
                "failure_rate_threshold": mutated.failure_rate_threshold,
            },
        })).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Perform crossover between current DNA and a provided DNA strand.
    #[pyo3(signature = (other_dna_json, crossover_point=None))]
    fn crossover_dna(&self, other_dna_json: &str, crossover_point: Option<usize>) -> PyResult<String> {
        let dna = self.dna.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        let other: CognitiveDna = serde_json::from_str(other_dna_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid DNA JSON: {}", e)))?;

        let child = dna.crossover(&other, crossover_point.unwrap_or(2));

        serde_json::to_string_pretty(&serde_json::json!({
            "parent_a": {
                "rif_weights": [dna.rif_weights.0, dna.rif_weights.1, dna.rif_weights.2, dna.rif_weights.3],
            },
            "parent_b": {
                "rif_weights": [other.rif_weights.0, other.rif_weights.1, other.rif_weights.2, other.rif_weights.3],
            },
            "child": {
                "rif_weights": [child.rif_weights.0, child.rif_weights.1, child.rif_weights.2, child.rif_weights.3],
                "confidence_threshold": child.confidence_threshold,
            },
        })).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Get cognitive macro cache statistics.
    fn get_macro_stats(&self) -> PyResult<String> {
        let cache = self.macros.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {}", e))
        })?;

        let promoted = cache.promoted();
        let total = cache.len();

        serde_json::to_string_pretty(&serde_json::json!({
            "total_macros": total,
            "promoted_macros": promoted.len(),
            "macros": promoted.iter().map(|m| {
                serde_json::json!({
                    "pattern": m.pattern,
                    "hit_count": m.hit_count,
                    "success_rate": m.success_rate,
                    "compiled_at_tick": m.compiled_at_tick,
                })
            }).collect::<Vec<_>>(),
        })).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}
