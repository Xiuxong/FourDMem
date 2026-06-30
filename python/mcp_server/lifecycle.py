"""FourDMem MCP Server — Lifecycle management.

Auto-archive, L0→L1 extraction, L1→L2 aggregation, DreamPruner, flush.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from mcp_server.state import (
    _session_id, _interaction_count, _turns_since_log,
    _auto_archive_buffer, _last_extraction_time,
    _AUTO_CAPTURE_INTERVAL, _EXTRACTION_COOLDOWN,
    _model_name, _agent_visibility, _WORKSPACE_DIR, _PROJECT_ROOT,
    get_engine, _get_salience, _get_aggregator, _get_observer,
    _get_dream_pruner,
)
from cognition.ssgm_governor import get_ssgm_governor
from cognition.embed_utils import ingest_safely


# ── Idempotency ──────────────────────────────────────────────────────────────
_last_post_interaction_tick: int = -1
_EXTRACTION_SIGNAL_BATCH: int = 3  # push extraction_suggested only every N turns


def _get_buffered_summary() -> str:
    """Return a short summary of recent buffered interactions for signal payload."""
    import mcp_server.state as state
    from mcp_server.state import _auto_archive_buffer
    if not _auto_archive_buffer:
        return ""
    recent = _auto_archive_buffer[-5:]
    parts = []
    for item in recent:
        content = (item.get("content") or "")[:120]
        role = item.get("role", "?")
        parts.append(f"[{role}] {content}")
    return " | ".join(parts)


# ── Auto-archive: every interaction → L0 ─────────────────────────────────────

def _auto_archive(role: str, content: str, metadata: dict = None):
    """Silently archive an interaction to L0 with semantic embedding."""
    import hashlib
    import mcp_server.state as state

    meta = metadata or {}
    meta.setdefault("model_name", _model_name)
    meta.setdefault("visibility", _agent_visibility)

    # Inject turn_type classification (harness signals + ML fallback)
    if "turn_type" not in meta:
        try:
            from mcp_server.state import _get_turn_classifier
            clf = _get_turn_classifier()
            if clf is not None:
                turn_type = clf.classify(content, role=role)
                if turn_type and turn_type != "unknown":
                    meta["turn_type"] = turn_type if isinstance(turn_type, str) else turn_type.value
        except Exception:
            pass
    # Content dedup: skip if same content already written this session
    content_hash = hashlib.md5(content[:500].encode()).hexdigest()
    if content_hash in state._l0_content_hashes:
        return
    state._l0_content_hashes.add(content_hash)
    # L0 conservative noise reduction: mark low-signal interactions
    if meta.get("source") == "search_memory_auto" or meta.get("tool") in ("tool_call", "checkpoint_turn"):
        meta["noise_weight"] = 0.3
    # Cap hash set at 2000 entries
    if len(state._l0_content_hashes) > 2000:
        state._l0_content_hashes = set(list(state._l0_content_hashes)[1000:])

    # Buffer for checkpoint (cap at 100)
    _auto_archive_buffer.append({
        "role": role,
        "content": content[:2000],
        "metadata": meta,
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    if len(_auto_archive_buffer) > 100:
        _auto_archive_buffer[:] = _auto_archive_buffer[-50:]

    try:
        engine = get_engine()
        meta_str = json.dumps(meta)
        ingest_safely(engine, _session_id, role, content, meta_str, state._workspace_id)

        # SSGM governance: log the write for audit
        governor = get_ssgm_governor()
        governor._modification_log.append({
            "action": "ingest",
            "workspace_id": state._workspace_id,
            "session_id": _session_id,
            "role": role,
            "content_hash": content_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        try:
            from loguru import logger as _log
            _log.warning(f'_auto_archive ingest failed: {e}')
        except Exception:
            import sys
            print(f'[FourDMem] _auto_archive ingest failed: {e}', file=sys.stderr)

    detector = _get_salience()
    if detector is not None:
        try:
            detector.check(content)
        except Exception:
            pass


def _run_extraction(engine: Any):
    """Push extraction_suggested signal if Agent hasn't extracted recently.

    No longer runs rule-based extraction. The Agent calls extract_deep
    with LLM-extracted facts when it's ready.
    """
    try:
        from cognition.signals import get_signal_bus
        bus = get_signal_bus()
        bus.push(
            "extraction_suggested",
            priority=3,
            payload={
                "message": "New conversation content available for extraction.",
                "instruction": "Call extract_deep with LLM-extracted atomic facts.",
            },
        )
    except ImportError:
        pass


def _write_l2_scenario(scenario: dict):
    """Write an auto-aggregated L2 scenario to the vault."""
    try:
        scenarios_dir = os.path.join(_WORKSPACE_DIR, "scenarios")
        os.makedirs(scenarios_dir, exist_ok=True)

        title = scenario.get("title", "Untitled Scenario")
        slug = title.lower().replace(" ", "_").replace("/", "_")[:50]
        filepath = os.path.join(scenarios_dir, f"{slug}.md")

        frontmatter = {
            "title": title,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "conditions": scenario.get("conditions", []),
            "tags": scenario.get("tags", []),
            "l1_refs": scenario.get("l1_refs", []),
            "auto_aggregated": True,
            "source": "auto_aggregator",
        }

        content = scenario.get("content", "")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("---\n")
            f.write(json.dumps(frontmatter, indent=2, ensure_ascii=False))
            f.write("\n---\n\n")
            f.write(content)
            f.write("\n")

        logger.info(f"Auto-aggregated L2 scenario written: {filepath}")
    except Exception as e:
        logger.warning(f"Failed to write L2 scenario: {e}")


def _post_interaction():
    """Called after every tool: auto-tick + salience + signal persistence + L1→L2 aggregation.

    Idempotent: skips if already called for the current interaction count.
    """
    import mcp_server.state as state
    global _last_post_interaction_tick

    # Idempotency: skip if this interaction count was already processed
    current_count = state._interaction_count
    if current_count <= _last_post_interaction_tick:
        return

    state._interaction_count += 1
    state._turns_since_log += 1
    _last_post_interaction_tick = state._interaction_count

    try:
        engine = get_engine()
        engine.advance_tick()
    except Exception:
        pass

    # ── Signal persistence ────────────────────────────────────────────────
    _persist_signals()

    # ── L1→L2 aggregation: every 10 interactions, if edges exist ──────────
    if state._interaction_count % 10 == 0:
        try:
            engine = get_engine()
            from cognition.extractor import _aggregate_connected_components
            l2_count = _aggregate_connected_components(engine)
            if l2_count >= 3:
                try:
                    from cognition.signals import get_signal_bus
                    bus = get_signal_bus()
                    bus.push(
                        "l2_aggregation_notable",
                        priority=2,
                        payload={
                            "scenarios_aggregated": l2_count,
                            "action_hint": (
                                f"{l2_count} L1 fact clusters auto-aggregated to L2 scenarios. "
                                "Consider synthesize_l2 for deeper synthesis."
                            ),
                        },
                    )
                except Exception:
                    pass
                logger.info(f"L1→L2: auto-aggregated {l2_count} scenarios")
            elif l2_count > 0:
                logger.info(f"L1→L2: auto-aggregated {l2_count} scenarios")
        except Exception:
            pass

    # ── L1 auto-checkpoint: every 10 interactions, if graph has data ─────
    if state._interaction_count % 10 == 0:
        try:
            engine = get_engine()
            if engine.graph_node_count() > 0:  # guard: don't overwrite with empty
                vault_root = os.path.join(state._PROJECT_ROOT, "data", "vault")
                engine.checkpoint(vault_root)
        except Exception:
            pass

    # ── Cross-layer integrity check: every 50 interactions ───────────────
    if state._interaction_count % 50 == 0:
        try:
            from daemon.integrity_checker import check_vault_integrity
            vault_root = os.path.join(state._PROJECT_ROOT, "data", "vault")
            report = check_vault_integrity(vault_root)
            issues = report.get("issues", [])
            if issues:
                critical = [i for i in issues if i.get("severity") == "critical"]
                if critical:
                    from cognition.signals import get_signal_bus
                    bus = get_signal_bus()
                    bus.push("observer_alert", priority=9, payload={
                        "issue": "integrity_violation",
                        "critical_count": len(critical),
                        "total_issues": len(issues),
                    })
                    logger.warning(f"Integrity: {len(critical)} critical issues, {len(issues)} total")
                else:
                    logger.info(f"Integrity: {len(issues)} non-critical issues")
        except Exception:
            pass

    # ── Salience-based extraction suggestion ──────────────────────────────
    detector = _get_salience()
    if detector is not None and detector.should_extract():
        now = time.time()
        if now - state._last_extraction_time > _EXTRACTION_COOLDOWN:
            detector.get_pending_content()
            try:
                _run_extraction(get_engine())
                state._last_extraction_time = now
            except Exception:
                pass
        return

    if state._interaction_count % _AUTO_CAPTURE_INTERVAL == 0:
        now = time.time()
        if now - state._last_extraction_time > _EXTRACTION_COOLDOWN:
            try:
                _run_extraction(get_engine())
                state._last_extraction_time = now
            except Exception:
                pass


def _persist_signals():
    """Persist cognitive signal bus to disk for cold-start recovery."""
    try:
        from cognition.signals import get_signal_bus
        import mcp_server.state as state
        bus = get_signal_bus()
        signals_path = os.path.join(state._PROJECT_ROOT, "data", "cognition", "signals.json")
        bus.save(signals_path)
    except Exception:
        pass


# ── DreamPruner background thread ────────────────────────────────────────────

_dream_thread: Any = None
_DREAM_INTERVAL_TICKS: int = 100


def _dream_pruner_loop():
    """Background loop that runs dream pruning every N ticks."""
    while True:
        try:
            time.sleep(30)
            engine = get_engine()
            if engine is not None:
                try:
                    tick = engine.get_tick()
                    if tick > 0 and tick % _DREAM_INTERVAL_TICKS == 0:
                        pruner = _get_dream_pruner()
                        if pruner is not None:
                            report = pruner.run_cycle(engine)
                            if report.get("pruned", 0) > 0:
                                _auto_archive("system", json.dumps(report), {
                                    "event": "dream_pruning",
                                })
                except Exception:
                    pass
        except Exception:
            pass




def _start_evolution_scheduler():
    """Start the background evolution scheduler threads."""
    global _dream_thread
    try:
        import threading
        _dream_thread = threading.Thread(target=_dream_pruner_loop, daemon=True)
        _dream_thread.start()
        print("Evolution scheduler started (DreamPruner background thread)", file=sys.stderr)
    except Exception as e:
        print(f"WARNING: Failed to start evolution scheduler: {e}", file=sys.stderr)


def _flush_on_exit():
    """Flush pending L0 data when MCP server exits."""
    try:
        observer = _get_observer()
        engine = get_engine()
        if observer is not None and engine is not None:
            observer.persist_cognitive_state(engine, _PROJECT_ROOT)
    except Exception:
        pass

    try:
        engine = get_engine()
        if engine is not None:
            # NOTE: Do NOT call _auto_archive here — tantivy writes during
            # Python interpreter shutdown corrupt the segment metadata,
            # leaving stale references in meta.json that crash the next
            # startup.  The session-end event is non-critical; skip it.
            print(f"[FourDMem] Session {_session_id} ended (flush skipped to protect fulltext index).", file=sys.stderr)
    except Exception:
        pass
