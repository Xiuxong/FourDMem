"""FourDMem MCP Server — MCP tool definitions.

All 14 MCP tools for Agent interaction with the memory system.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from mcp_server.state import (
    _session_id, _interaction_count, _turns_since_log,
    _auto_archive_buffer, _model_name, _PROJECT_ROOT, _WORKSPACE_DIR,
    _has_waken_up, _db_path, _workspace_id,
    get_engine, _get, _get_embedder, _get_observer, _get_myelination,
    _get_paradigm_engine, _get_aggregator, _get_promoter,
    _get_topology, _get_auto_plugin_generator, _load_l3_rules, _save_l3_rule,
    _get_turn_classifier, _get_salience, _get_dream_pruner,
)
from mcp_server.lifecycle import _auto_archive, _post_interaction


# ── C2 Thompson sampling helper ───────────────────────────────────────────────

def thompson_sample(alpha: float, beta: float) -> float:
    """Draw a sample from Beta(alpha, beta) via gamma approximation."""
    import random
    if alpha <= 0 or beta <= 0:
        return 0.5
    x = random.gammavariate(alpha, 1.0)
    y = random.gammavariate(beta, 1.0)
    if x + y == 0:
        return 0.5
    return x / (x + y)


# ── search_memory ─────────────────────────────────────────────────────────────

def search_memory(query: str, limit: int = 10, filter_by_turn_type: str = "") -> str:
    """Search the agent's four-dimensional memory. Auto-wakes on first call.

    Args:
        query: Natural-language search query.
        limit: Maximum results to return (default 10).
        filter_by_turn_type: Optional turn type filter.
            When set, over-fetches 3x limit then filters down.
    """
    import mcp_server.state as state
    engine = get_engine()

    # Auto-wake on first search of session
    wake_context = None
    if not state._has_waken_up:
        state._has_waken_up = True
        try:
            wake_context = json.loads(wake_up())
        except Exception:
            pass

    # P0 auto-capture: archive the search query as a user turn
    _auto_archive("user", query, {"source": "search_memory_auto", "turn_type": "user_query", "turn": state._interaction_count})

    plugin_gen = _get_auto_plugin_generator()
    if plugin_gen is not None:
        try:
            query = plugin_gen.preprocess_query(query)
        except Exception:
            pass

    # Over-fetch when filtering by turn_type
    fetch_limit = limit * 3 if filter_by_turn_type else limit

    embedder = _get_embedder()
    embedding = None
    embedding_en = None
    dual_search = os.environ.get("FOURDMEM_DUAL_SEARCH", "") == "1"

    if embedder is not None:
        try:
            if dual_search:
                dual = embedder.embed_dual(query)
                embedding = dual.get('zh')
                embedding_en = dual.get('en')
            else:
                embedding = embedder.embed(query)
            if not embedding or all(x == 0.0 for x in embedding):
                embedding = None
        except Exception:
            embedding = None

    if embedding is not None:
        try:
            result = engine.query_with_embedding(query, embedding, limit=fetch_limit)
        except Exception:
            result = engine.query(query, limit=fetch_limit)
    else:
        result = engine.query(query, limit=fetch_limit)

    # Dual-search: merge en embedding results via simple score addition
    if dual_search and embedding_en is not None and any(x != 0.0 for x in embedding_en):
        try:
            en_result = engine.query_with_embedding(query, embedding_en, limit=fetch_limit)
            en_data = json.loads(en_result) if isinstance(en_result, str) else en_result
            en_results = en_data.get("results", [])
            # Merge: for overlapping IDs, add scores; for unique, append
            result_data = json.loads(result) if isinstance(result, str) else result
            existing_ids = {r.get('id'): i for i, r in enumerate(result_data.get('results', []))}
            for er in en_results:
                eid = er.get('id')
                if eid in existing_ids:
                    result_data['results'][existing_ids[eid]]['score'] += er.get('score', 0)
                else:
                    result_data['results'].append(er)
            result_data['results'].sort(key=lambda r: r.get('score', 0), reverse=True)
            result_data['results'] = result_data['results'][:fetch_limit]
            result = json.dumps(result_data, ensure_ascii=False)
        except Exception:
            pass

    # Post-processing
    try:
        result_data = json.loads(result) if isinstance(result, str) else result
        results_list = result_data.get("results", [])

        from cognition.ssgm_governor import get_ssgm_governor, AccessRequest, negation_weight
        governor = get_ssgm_governor()

        filtered = []
        for r in results_list:
            r_meta = r.get("metadata") or {}
            if isinstance(r_meta, str):
                try:
                    r_meta = json.loads(r_meta)
                except Exception:
                    r_meta = {}

            # SSGM dynamic access control
            access = governor.access_control(AccessRequest(
                memory_id=r.get("id", ""),
                workspace_id=_workspace_id,
                agent_id=_model_name,
                memory_visibility=r_meta.get("visibility", "shared"),
                memory_workspace_id=r_meta.get("workspace_id", ""),
            ))
            if not access.access_granted:
                continue

            # SSGM temporal decay
            decay = governor.temporal_decay_model({
                "last_active_tick": r.get("last_active_tick", 0),
                "utility_score": r_meta.get("utility", 0.0),
                "shelf_life": r_meta.get("shelf_life", ""),
            }, state._interaction_count if hasattr(state, '_interaction_count') else 0)
            r["score"] = r.get("score", 0.0) * decay

            # RIF-U turn_type weighting
            turn_type = r_meta.get("turn_type", "")
            _TURN_TYPE_WEIGHTS = {
                "final_answer": 1.20,
                "plan": 1.15,
                "tool_result": 1.00,
                "user_query": 1.00,
                "reasoning": 0.85,
                "tool_call": 0.70,
                "system_prompt": 0.30,
            }
            if turn_type and turn_type in _TURN_TYPE_WEIGHTS:
                r["score"] = r.get("score", 0.0) * _TURN_TYPE_WEIGHTS[turn_type]

            # L0 noise reduction: apply noise_weight penalty if annotated
            if r_meta.get("noise_weight"):
                r["score"] *= float(r_meta["noise_weight"])

            if r_meta.get("tool") and not r_meta.get("source"):
                r["score"] = r.get("score", 0.0) * 0.3
            filtered.append(r)

            # A3: Adversarial negation filter
            r["score"] = r.get("score", 0.0) * negation_weight(r.get("content", ""))

        # Apply turn_type filter
        if filter_by_turn_type:
            def _get_tt(meta):
                if isinstance(meta, dict):
                    return meta.get("turn_type", "")
                if isinstance(meta, str):
                    try:
                        return json.loads(meta).get("turn_type", "")
                    except Exception:
                        return ""
                return ""
            filtered = [r for r in filtered if _get_tt(r.get("metadata")) == filter_by_turn_type]
            filtered = filtered[:limit]

        filtered.sort(key=lambda r: (r.get("score", 0.0), r.get("timestamp", "")), reverse=True)

        # L2 Dual-Channel Search
        try:
            from cognition.l2_indexer import get_l2_index
            l2 = get_l2_index()
            l2_results = l2.search(query, top_k=5)
            for l2r in l2_results.get("results", []):
                filtered.append({
                    "id": l2r.get("scenario_id") or l2r.get("schema_id", ""),
                    "layer": 2,
                    "score": l2r["score"] * 0.8,
                    "content": (l2r.get("lesson") or l2r.get("trigger_event", "") or "")[:500],
                    "metadata": {"channel": l2r.get("channel", "semantic"), "source": "l2_dual_index"},
                    "token_estimate": len(l2r.get("lesson", "")) // 4,
                })
            filtered.sort(key=lambda r: (r.get("score", 0.0), r.get("timestamp", "")), reverse=True)
        except Exception:
            pass

        l3_rules = _load_l3_rules()
        warnings = []
        if l3_rules:
            for rule in l3_rules:
                # C2 Thompson sampling for rule weighting
                alpha = rule.get("alpha", None)
                beta = rule.get("beta", None)
                weight = None
                if alpha is not None and beta is not None:
                    weight = thompson_sample(alpha, beta)
                    # C3 guardrails: cap weight at max_weight if configured
                    guardrails = rule.get("guardrails", {})
                    max_weight = guardrails.get("max_weight", None)
                    if max_weight is not None and weight > max_weight:
                        weight = max_weight
                # Counterfactual conflict detection
                if rule.get("type") == "counterfactual":
                    entity = rule.get("entity_id", "").lower()
                    for r in filtered:
                        if entity and entity in r.get("content", "").lower():
                            warnings.append(f"Counterfactual conflict: {rule['entity_id']}")
                            break
                # Apply Thompson-sampled weight to matching results
                if weight is not None:
                    entity = rule.get("entity_id", "").lower()
                    for r in filtered:
                        if entity and entity in r.get("content", "").lower():
                            r["score"] = r.get("score", 0.0) * weight
        result_data["results"] = filtered
        if warnings:
            result_data["l4_warnings"] = list(set(warnings))
        if wake_context:
            result_data["wake_context"] = wake_context
        # E2 extraction hint: flag when results are thin
        confidence = result_data.get("confidence", 1.0)
        if confidence < 0.3 or len(filtered) < 3:
            result_data["extraction_needed"] = True
        else:
            result_data["extraction_needed"] = False
        result = json.dumps(result_data, ensure_ascii=False)
    except Exception:
        pass

    # Background post-processing
    def _bg_postprocess():
        try:
            detector = _get_salience()
            if detector is not None:
                detector.check(query)
            result_data = json.loads(result) if isinstance(result, str) else result
            results_list = result_data.get("results", [])
            observer = _get_observer()
            if observer is not None:
                observer.observe(engine, {
                    "query": query,
                    "confidence": result_data.get("confidence", 1.0),
                    "results": results_list,
                }, project_root=_PROJECT_ROOT)
            myelin = _get_myelination()
            if myelin is not None:
                myelin.record_query(query, len(results_list), result_data.get("confidence", 0.0))
            _post_interaction()
        except Exception:
            pass

    from concurrent.futures import ThreadPoolExecutor
    _bg = ThreadPoolExecutor(max_workers=1)
    _bg.submit(_bg_postprocess)
    return result




# ── submit_feedback ──────────────────────────────────────────────────────────

def submit_feedback(entity_id: str, score: float) -> str:
    """Submit feedback on a memory's usefulness."""
    engine = get_engine()
    engine.feedback(entity_id, score)
    # D2 last_active_tick: bump tick for matching nodes
    try:
        engine.bump_tick(entity_id)
    except Exception:
        pass
    # C1 Bayesian stats: update matching L3 rules in persona.yaml
    try:
        import yaml
        import mcp_server.state as state
        persona_path = os.path.join(state._PROJECT_ROOT, "data", "vault", "persona", "persona.yaml")
        persona = {}
        if os.path.exists(persona_path):
            with open(persona_path, "r", encoding="utf-8") as f:
                persona = yaml.safe_load(f) or {}
        rules = persona.get("rules", [])
        updated = False
        for rule in rules:
            if rule.get("entity_id") == entity_id:
                wins = rule.get("wins", 0)
                losses = rule.get("losses", 0)
                if score > 0:
                    wins += 1
                else:
                    losses += 1
                rule["wins"] = wins
                rule["losses"] = losses
                rule["alpha"] = 1 + wins
                rule["beta"] = 1 + losses
                updated = True
        if updated:
            persona["rules"] = rules
            os.makedirs(os.path.dirname(persona_path), exist_ok=True)
            with open(persona_path, "w", encoding="utf-8") as f:
                yaml.dump(persona, f, default_flow_style=False, allow_unicode=True)
            # Refresh cached rules
            state._L3_RULES = rules
    except Exception:
        pass
    _auto_archive("tool", f"feedback: {entity_id} += {score}", {"tool": "submit_feedback"})

    if score < -0.5:
        paradigm = _get_paradigm_engine()
        if paradigm is not None:
            try:
                domain = _infer_domain(entity_id)
                paradigm.record_outcome(domain, success=False)
                is_crisis, rate = paradigm.check_crisis(domain)
                if is_crisis:
                    _auto_archive("system", json.dumps({
                        "event": "cognitive_crisis",
                        "domain": domain,
                        "failure_rate": round(rate, 3),
                    }), {"event": "paradigm_shift_alert"})
            except Exception:
                pass
    elif score > 0.5:
        _save_l3_rule({
            "type": "preference",
            "entity_id": entity_id,
            "score": score,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        paradigm = _get_paradigm_engine()
        if paradigm is not None:
            try:
                domain = _infer_domain(entity_id)
                paradigm.record_outcome(domain, success=True)
            except Exception:
                pass


    _post_interaction()


def _infer_domain(entity_id: str) -> str:
    lower = entity_id.lower()
    for domain, keywords in [
        ("rust", ["rust", "cargo", "crate", "borrow", "lifetime"]),
        ("python", ["python", "pip", "venv", "django", "flask"]),
        ("architecture", ["architect", "design", "pattern", "microservice"]),
        ("frontend", ["react", "vue", "css", "html", "dom"]),
        ("database", ["sql", "postgres", "mysql", "redis", "query"]),
        ("devops", ["docker", "k8s", "ci", "deploy", "nginx"]),
    ]:
        if any(kw in lower for kw in keywords):
            return domain
    return "general"


# ── wake_up ──────────────────────────────────────────────────────────────────

def wake_up() -> str:
    """Cold-start wake-up: assess current memory state and recover context."""
    import mcp_server.state as state
    engine = get_engine()

    raw_stats = engine.wake_up()
    try:
        stats = json.loads(raw_stats) if isinstance(raw_stats, str) else raw_stats
    except (json.JSONDecodeError, TypeError):
        stats = {"raw": str(raw_stats)}

    try:
        current_tick = engine.get_tick()
    except Exception:
        current_tick = 0

    # Load L1 from disk BEFORE computing gap/snapshot
    if not state._l1_loaded:
        try:
            vault_root = os.path.join(_PROJECT_ROOT, "data", "vault")
            engine.load_from_disk(vault_root)
            state._l1_loaded = True
            # Refetch stats — engine.wake_up() was called before load
            raw_stats = engine.wake_up()
            try:
                stats = json.loads(raw_stats) if isinstance(raw_stats, str) else raw_stats
            except (json.JSONDecodeError, TypeError):
                pass
        except Exception:
            pass

    recent_facts = []
    try:
        raw_results = engine.query("*", 5)
        results_data = json.loads(raw_results) if isinstance(raw_results, str) else raw_results
        for item in results_data.get("results", []):
            recent_facts.append({
                "id": item.get("id"),
                "content": item.get("content", "")[:200],
                "score": item.get("score", 0),
                "layer": item.get("layer", "?"),
            })
    except Exception:
        pass

    # Restore persisted cognitive signals
    signals_restored = 0
    try:
        from cognition.signals import get_signal_bus
        signals_path = os.path.join(_PROJECT_ROOT, "data", "cognition", "signals.json")
        bus = get_signal_bus()
        signals_restored = bus.load(signals_path)
    except Exception:
        pass

    gap = None
    try:
        from cognition.gap_reflection import compute_gap, save_snapshot
        gap = compute_gap(_PROJECT_ROOT, engine)
        save_snapshot(_PROJECT_ROOT, engine)
    except Exception:
        pass

    mem_stats = stats.get("memory_stats", {})
    response = {
        "status": "awake",
        "current_tick": current_tick,
        "memory_stats": {
            "fulltext_docs": mem_stats.get("fulltext_docs", 0),
            "l0_evidence": mem_stats.get("l0_evidence", 0),
            "l1_edges": mem_stats.get("l1_edges", 0),
            "l1_nodes": mem_stats.get("l1_nodes", 0),
            "persona": mem_stats.get("persona", "FourDMem Agent"),
        },
        "signals_restored": signals_restored,
        "recent_facts": recent_facts,
        "session_id": state._session_id,
        "interaction_count": state._interaction_count,
        "db_path": state._db_path,
        "message": "Wake-up complete. Use recent_facts to orient your response.",
    }
    if gap:
        response["gap_reflection"] = gap
    try:
        from evolution.strange_loop import restore_cognitive_state
        response["restored_cognitive_state"] = restore_cognitive_state()
    except Exception:
        pass
    return json.dumps(response, indent=2, ensure_ascii=False)


# ── log_turn ─────────────────────────────────────────────────────────────────

def log_turn(user_message: str, assistant_response: str, turn_type: str = "") -> str:
    """Archive a conversation turn to memory. OPTIONAL.

    Args:
        user_message: The user's message in this turn.
        assistant_response: The assistant's full response.
        turn_type: Optional hint (reasoning, plan, tool_call, tool_result, final_answer).
                   If empty, auto-classified by TurnClassifier.
    """
    import mcp_server.state as state
    try:
        engine = get_engine()
        from cognition.embed_utils import ingest_safely

        assistant_meta = {"auto": True, "turn": state._interaction_count}
        if turn_type:
            assistant_meta["turn_type"] = turn_type
        else:
            try:
                clf = _get_turn_classifier()
                if clf is not None:
                    classified = clf.classify(assistant_response, role="assistant")
                    if classified and classified != "unknown":
                        assistant_meta["turn_type"] = classified if isinstance(classified, str) else classified.value
            except Exception:
                pass

        user_meta = {"auto": True, "turn": state._interaction_count, "turn_type": "user_query"}

        ingest_safely(engine, state._session_id, "user", user_message[:2000],
                      json.dumps(user_meta), state._workspace_id)
        ingest_safely(engine, state._session_id, "assistant", assistant_response[:2000],
                      json.dumps(assistant_meta), state._workspace_id)
        detector = _get_salience()
        if detector is not None:
            detector.check(f"{user_message[:2000]}\n{assistant_response[:2000]}")

        try:
            from cognition.l2_indexer import get_l2_index
            l2 = get_l2_index()
            l2.extract_schema_from_turn(
                user_message=user_message[:2000],
                assistant_response=assistant_response[:2000],
                workspace_id=state._workspace_id,
                session_id=state._session_id,
                tick=state._interaction_count,
            )
        except Exception:
            pass
    except Exception:
        pass
    state._turns_since_log = 0
    state._interaction_count += 1
    return json.dumps({"status": "archived", "turn": state._interaction_count, "unarchived_turns": state._turns_since_log}, indent=2)


# ── checkpoint_turn ──────────────────────────────────────────────────────────

def checkpoint_turn(agent_response: str = "") -> str:
    """Auto-capture: flush buffer and archive this turn's response. Called by hooks."""
    import mcp_server.state as state

    flushed = len(state._auto_archive_buffer)

    if agent_response:
        try:
            engine = get_engine()
            from cognition.embed_utils import ingest_safely
            checkpoint_meta = {"source": "checkpoint_turn", "turn": state._interaction_count}
            try:
                clf = _get_turn_classifier()
                if clf is not None:
                    classified = clf.classify(agent_response, role="assistant")
                    if classified and classified != "unknown":
                        checkpoint_meta["turn_type"] = classified if isinstance(classified, str) else classified.value
            except Exception:
                pass
            ingest_safely(engine, state._session_id, "assistant", agent_response[:2000],
                          json.dumps(checkpoint_meta), state._workspace_id)
        except Exception:
            pass

    agent_extracted = any(
        item.get("metadata", {}).get("tool") == "extract_deep"
        for item in state._auto_archive_buffer
    )

    extraction_suggested = False
    if not agent_extracted and len(state._auto_archive_buffer) >= 2:
        def _bg_extract():
            try:
                messages = [
                    item for item in state._auto_archive_buffer
                    if item.get("role") in ("user", "assistant")
                    and not item.get("metadata", {}).get("tool")
                ]
                if len(messages) < 2:
                    return
                detector = _get_salience()
                if detector is not None:
                    combined = " ".join(m["content"][:100] for m in messages[-4:])
                    detector.check(combined)
                    if not detector.should_extract():
                        return
                # Push extraction_suggested signal with buffered summary for Agent context
                try:
                    from mcp_server.lifecycle import _get_buffered_summary
                    summary = _get_buffered_summary()
                except Exception:
                    summary = ""

                try:
                    from cognition.signals import get_signal_bus
                    bus = get_signal_bus()
                    bus.push(
                        "extraction_suggested",
                        priority=1,  # Low priority — Agent checks when idle
                        payload={
                            "message_count": len(messages),
                            "buffered_summary": summary[:500],
                            "instruction": (
                                "Conversation facts pending extraction. "
                                "Call extract_deep with your LLM to extract 1-3 "
                                "atomic facts. See buffered_summary for context."
                            ),
                        },
                    )
                except ImportError:
                    pass
            except Exception:
                pass

        from concurrent.futures import ThreadPoolExecutor
        _bg = ThreadPoolExecutor(max_workers=1)
        _bg.submit(_bg_extract)
        extraction_suggested = True

    state._auto_archive_buffer.clear()
    state._turns_since_log = 0
    state._interaction_count += 1
    _post_interaction()

    return json.dumps({
        "status": "checkpoint_complete",
        "interactions_flushed": flushed,
        "turn": state._interaction_count,
        "agent_extracted": agent_extracted,
        "extraction_suggested": extraction_suggested,
    }, indent=2)


# ── save_memory ──────────────────────────────────────────────────────────────

def save_memory(content: str, role: str = "user", metadata: str = "{}") -> str:
    """Explicitly save a fact or decision to long-term memory."""
    import hashlib
    import mcp_server.state as state
    engine = get_engine()
    from cognition.embed_utils import ingest_safely

    # Content dedup: skip if same content already written this session
    content_hash = hashlib.md5(content[:500].encode()).hexdigest()
    if content_hash in state._l0_content_hashes:
        return json.dumps({"status": "deduplicated", "reason": "duplicate content"}, ensure_ascii=False)
    state._l0_content_hashes.add(content_hash)

    result = ingest_safely(engine, _session_id, role, content, metadata)
    _post_interaction()
    return result


# ── reflect ──────────────────────────────────────────────────────────────────

def reflect(query: str) -> str:
    """Evaluate retrieval confidence for a topic."""
    engine = get_engine()
    result = engine.query(query, 5)
    data = json.loads(result) if isinstance(result, str) else result
    results = data.get("results", [])
    confidence = 0.0
    if results:
        scores = [r.get("score", 0) for r in results[:3]]
        confidence = sum(scores) / len(scores) if scores else 0.0
    return json.dumps({
        "confidence": round(confidence, 4),
        "result_count": len(results),
        "needs_drill_down": confidence < 0.3,
    }, indent=2)


# ── abandon_branch ───────────────────────────────────────────────────────────

def abandon_branch(entity_id: str, reason: str = "", conditions: str = "{}") -> str:
    """Mark a decision branch as abandoned (counterfactual).

    Args:
        entity_id: The entity to mark as counterfactual.
        reason: Why this branch was abandoned.
        conditions: JSON string describing failure conditions for future re-evaluation.
                    Example: '{"tool":"redis","approach":"no_eviction","env":{"memory":"1GB"}}'
    """
    engine = get_engine()
    cond_value = json.loads(conditions) if conditions and conditions != "{}" else None
    if cond_value:
        engine.abandon_branch_with_conditions(entity_id, 0, reason, json.dumps(cond_value))
    else:
        engine.abandon_branch(entity_id, reason)
    _auto_archive("tool", f"abandon_branch: {entity_id}", {"tool": "abandon_branch", "conditions": cond_value})
    _post_interaction()
    return json.dumps({"status": "abandoned", "entity_id": entity_id, "reason": reason, "conditions": cond_value}, indent=2)


# ── re_enable_branch ─────────────────────────────────────────────────────────

def re_enable_branch(entity_id: str, counterfactual_seq: int, reason: str = "") -> str:
    """Re-enable a counterfactual branch."""
    engine = get_engine()
    engine.re_enable_branch(entity_id, counterfactual_seq, reason)
    _post_interaction()
    return json.dumps({"status": "re_enabled", "entity_id": entity_id}, indent=2)


# ── checkpoint_memory ─────────────────────────────────────────────────────────

def checkpoint_memory() -> str:
    """Save L1 state to human-readable JSONL files."""
    engine = get_engine()
    vault_root = os.path.join(_PROJECT_ROOT, "data", "vault")
    result = engine.checkpoint(vault_root)
    _auto_archive("tool", "checkpoint_memory", {"tool": "checkpoint_memory"})
    _post_interaction()
    return result


def load_memory() -> str:
    """Load L1 state from JSONL files on disk."""
    import mcp_server.state as state
    engine = get_engine()
    vault_root = os.path.join(_PROJECT_ROOT, "data", "vault")
    loaded = engine.load_from_disk(vault_root)
    state._l1_loaded = True
    _post_interaction()
    return json.dumps({"status": "loaded", "facts_loaded": loaded})



# ── get_entity_context ───────────────────────────────────────────────────────

def get_entity_context(entity_id: str) -> str:
    """Get full context for a memory entity."""
    engine = get_engine()
    result = engine.get_entity_context(entity_id)
    _post_interaction()
    return result


# ── memory_health ────────────────────────────────────────────────────────────

def memory_health() -> str:
    """Check memory system health."""
    engine = get_engine()
    stats = json.loads(engine.wake_up())
    ms = stats.get("memory_stats", {})
    return json.dumps({
        "status": "healthy",
        "tick": engine.get_tick(),
        "memory_stats": ms,
        "session": _session_id,
    }, indent=2)


# ── write_scenario ───────────────────────────────────────────────────────────

def write_scenario(title: str, content: str, conditions: str = "") -> str:
    """Write an L2 scenario block to the vault."""
    import re
    safe_name = re.sub(r'[^\w\-]', '_', title)[:64]
    scenario_dir = os.path.join(_PROJECT_ROOT, "data", "vault", "scenarios")
    os.makedirs(scenario_dir, exist_ok=True)
    path = os.path.join(scenario_dir, f"{safe_name}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        if conditions:
            f.write(f"## Conditions\n{conditions}\n\n")
        f.write(content)
    _post_interaction()
    return json.dumps({"status": "written", "path": path}, indent=2)


# ── extract_deep ─────────────────────────────────────────────────────────────

def extract_deep(facts: str) -> str:
    """Agent-driven deep fact extraction with automatic L0 provenance.

    The Agent extracts atomic facts using its LLM and submits them as JSON.
    FourDMem auto-injects l0_refs from the current session's recent L0
    evidence, enabling graph edge creation and L1→L2 aggregation.
    """
    import mcp_server.state as state
    engine = get_engine()
    try:
        facts_list = json.loads(facts) if isinstance(facts, str) else facts
    except Exception:
        facts_list = [{"content": facts}]

    # Auto-inject l0_refs from current session's recent L0 evidence
    l0_refs = _get_recent_l0_refs(engine, state._session_id, state._workspace_id)

    dedup = _get("dedup", "cognition.dedup", "SemanticDeduplicator")
    added = 0
    node_indices: list[int] = []
    new_labels: dict[int, str] = {}  # node_index → label for cross-session matching
    for f in facts_list:
        content = f.get("label") or f.get("content") or ""
        if not content or len(content.strip()) < 5:
            continue
        try:
            raw = dedup.add_fact_with_dedup(engine, content)
            result = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(result, dict) and result.get("status") in ("added", "merged", "linked"):
                added += 1
                ni = result.get("node_index")
                if ni is not None:
                    node_indices.append(ni)
                    new_labels[ni] = content
                importance = f.get("importance", 0.5)
                if isinstance(f, dict) and importance != 0.0:
                    try:
                        engine.feedback(content, importance)
                    except Exception:
                        pass
        except Exception:
            pass

    # Create L1 edges: pairwise connect facts from same extract_deep batch
    edge_count = 0
    if len(node_indices) >= 2:
        for i in range(len(node_indices)):
            for j in range(i + 1, len(node_indices)):
                try:
                    ok = engine.graph_add_edge(node_indices[i], node_indices[j], "related_to", 0.5)
                    if ok: edge_count += 1
                except Exception:
                    pass

    # A2: Cross-session auto-edges — link new facts to existing L1 nodes with label overlap
    MAX_CROSS_EDGES = 50
    if node_indices and new_labels:
        try:
            all_indices = engine.graph_node_indices()
            cross_edges = 0
            for ni in node_indices:
                if cross_edges >= MAX_CROSS_EDGES:
                    break
                new_label = new_labels.get(ni, "")
                if not new_label:
                    continue
                new_lower = new_label.lower()
                for existing_idx in all_indices:
                    if cross_edges >= MAX_CROSS_EDGES:
                        break
                    if existing_idx == ni:
                        continue  # skip self
                    try:
                        node_json = engine.graph_get_node(existing_idx)
                        node_data = json.loads(node_json) if isinstance(node_json, str) else node_json
                        existing_label = node_data.get("label") or node_data.get("content") or ""
                        if not existing_label:
                            continue
                        # Case-insensitive substring overlap
                        existing_lower = existing_label.lower()
                        if new_lower in existing_lower or existing_lower in new_lower:
                            ok = engine.graph_add_edge(ni, existing_idx, "related_to", 0.3)
                            if ok:
                                edge_count += 1
                                cross_edges += 1
                    except Exception:
                        pass
        except Exception:
            pass

    _post_interaction()
    return json.dumps({
        "status": "extracted",
        "facts_submitted": len(facts_list),
        "facts_added": added,
        "edges_created": edge_count,
        "l0_refs_attached": len(l0_refs),
    }, indent=2)


def _get_recent_l0_refs(engine: Any, session_id: str, workspace_id: str) -> list[int]:
    """Get recent L0 evidence IDs from the current session for provenance linking."""
    try:
        raw = engine.get_session_evidence(session_id, 10, workspace_id)
        data = json.loads(raw) if isinstance(raw, str) else raw
        evidence = data.get("evidence", [])
        return [e["id"] for e in evidence if e.get("id")]
    except Exception:
        return []




# ── synthesize_l2 ────────────────────────────────────────────────────────────

def synthesize_l2(title: str, fact_ids: str, synthesis: str, conditions: str = "") -> str:
    """Agent-driven L2 scenario synthesis."""
    ids = json.loads(fact_ids) if isinstance(fact_ids, str) else fact_ids
    engine = get_engine()
    try:
        engine.synthesize_l2(title, ids, synthesis, conditions)
    except Exception:
        pass
    _post_interaction()
    return json.dumps({"status": "synthesized", "title": title}, indent=2)


# ── reflect_and_synthesize ───────────────────────────────────────────────────

def reflect_and_synthesize(domain: str, thesis: str, antithesis: str, synthesis: str) -> str:
    """Agent-driven dialectical synthesis."""
    engine = get_engine()
    try:
        engine.reflect_and_synthesize(domain, thesis, antithesis, synthesis)
    except Exception:
        pass
    _post_interaction()
    return json.dumps({"status": "synthesized", "domain": domain}, indent=2)


# ── evolution_status ─────────────────────────────────────────────────────────

def evolution_status() -> str:
    """Check the status of the cognitive evolution engine."""
    obs = _get_observer()
    ps = _get_paradigm_engine()
    dp = _get_dream_pruner()
    mye = _get_myelination()
    return json.dumps({
        "observer": obs.status() if obs else "N/A",
        "paradigm_shift": ps.status() if ps else "N/A",
        "dream_pruner": dp.status() if dp else "N/A",
        "myelination": mye.status() if mye else "N/A",
    }, indent=2)

# ── check_cognition_signals ──────────────────────────────────────────────────

def check_cognition_signals() -> str:
    """Poll the cognitive signal bus for pending Agent-driven tasks.

    Returns any signals that FourDMem's monitors have detected:
    - macro_candidate: high-frequency query pattern ready for compilation
    - paradigm_crisis: domain failure rate exceeding threshold
    - phase_transition: graph topology at critical complexity
    - extraction_suggested: new conversation content waiting for extract_deep
    - promotion_proposed: L2 scenario eligible for L3 promotion (Agent approval needed)

    The Agent reviews signals and decides whether to act on each.
    Call extract_deep, reflect_and_synthesize, or synthesize_l2 to respond.
    """
    try:
        from cognition.signals import get_signal_bus
        bus = get_signal_bus()
        signals = bus.poll(limit=10)
        return json.dumps({
            "pending_count": len(signals),
            "signals": signals,
        }, indent=2)
    except ImportError:
        return json.dumps({"pending_count": 0, "signals": []}, indent=2)




def _get_analogy_engine():
    from mcp_server.state import _get as _get_factory
    return _get_factory("analogy", "evolution.analogy_engine", "AnalogyEngine")


# ── reload_modules ────────────────────────────────────────────────────────────

def reload_modules() -> str:
    """Hot-reload: flush buffers and restart server via process exit."""
    import sys
    try:
        import mcp_server.state as state_mod
        engine = state_mod._engine
        if engine is not None:
            engine.advance_tick()
    except Exception:
        pass
    sys.exit(0)

# ── rebuild_l1 ───────────────────────────────────────────────────────────────

def rebuild_l1() -> str:
    """Rebuild L1 facts from L0 evidence with proper l0_refs and importance."""
    import sqlite3
    import mcp_server.state as state
    engine = get_engine()

    db_path = state._db_path
    if not os.path.exists(db_path):
        return json.dumps({"status": "error", "reason": "L0 database not found"}, indent=2)

    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    # Only use workspace with real conversation data
    rows = db.execute(
        "SELECT id, role, content FROM evidence "
        "WHERE workspace_id = ? AND LENGTH(content) > 20 "
        "ORDER BY id", (state._workspace_id,)
    ).fetchall()
    db.close()

    evidence_list = [{"id": r["id"], "role": r["role"], "content": r["content"]} for r in rows]

    # Prepare evidence batches for Agent-driven extraction
    evidence_batches = []
    for i in range(0, len(evidence_list), 30):
        batch = evidence_list[i:i + 30]
        batch_preview = []
        for ev in batch[:10]:
            batch_preview.append({
                "id": ev["id"],
                "role": ev["role"],
                "preview": ev["content"][:200],
            })
        evidence_batches.append({
            "batch_index": i // 30,
            "evidence_count": len(batch),
            "preview": batch_preview,
        })

    return json.dumps({
        "status": "ready_for_extraction",
        "l0_scanned": len(evidence_list),
        "batches": len(evidence_batches),
        "evidence_batches": evidence_batches,
        "instruction": (
            "Evidence gathered from L0. Use extract_deep with your LLM "
            "to extract atomic facts from the evidence. "
            "Format per fact: {'label': '...', 'importance': 0.8, 'tags': ['tag1']}"
        ),
    }, indent=2)


# ── interact ──────────────────────────────────────────────────────────────────

def interact(query: str) -> str:
    """Shortcut: search + auto-extract if confidence is low.

    Calls search_memory(query) and if results have low confidence or
    extraction_needed flag, auto-calls extract_deep with a default prompt.
    Returns combined search + optional extraction result.
    """
    search_result_str = search_memory(query)
    try:
        search_data = json.loads(search_result_str)
    except Exception:
        search_data = {"results": []}

    results = search_data.get("results", [])
    confidence = search_data.get("confidence", 1.0)
    extraction_needed = search_data.get("extraction_needed", False)

    if extraction_needed or confidence < 0.3 or len(results) < 3:
        try:
            extract_result_str = extract_deep(
                f"Auto-extract from query: {query}\n\nContext: memory search returned "
                f"{len(results)} results with confidence {confidence:.2f}. "
                f"Extract key facts relevant to this query."
            )
            extract_data = json.loads(extract_result_str)
            search_data["auto_extraction"] = extract_data
        except Exception:
            search_data["auto_extraction"] = {"status": "extraction_skipped"}

    return json.dumps(search_data, ensure_ascii=False)
