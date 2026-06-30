"""L0→L1 Agent-Driven Fact Storage Pipeline.

Accepts facts extracted by the Agent's LLM (via extract_deep MCP tool),
handles deduplication, storage to L1 graph, importance scoring, and
L1→L2 graph aggregation.

Key principle: FourDMem stores and indexes facts; the Agent extracts them.
No rule-based extraction — all cognitive work is Agent-side.

Architecture:
    Agent (LLM) → extract_deep(facts=JSON) → FourDMem stores in L1
    Agent (LLM) → synthesize_l2(title, facts, synthesis) → FourDMem stores in L2
"""

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from cognition.embed_utils import add_fact_safely


# ── Fact storage pipeline ──────────────────────────────────────────────────────

class FactExtractor:
    """Storage pipeline for Agent-extracted facts.

    Accepts pre-extracted facts (with label, importance, tags, l0_refs)
    from the Agent and handles deduplication + L1 graph insertion + edge creation.
    Does NOT perform extraction itself — the Agent's LLM does that.
    """

    def __init__(self):
        pass

    def ingest_facts(
        self,
        engine: Any,
        facts: list[dict[str, Any]],
        embedder: Any = None,
    ) -> dict:
        """Ingest Agent-extracted facts into L1 graph with dedup.

        Args:
            engine: FourDMemEngine instance.
            facts: List of dicts with 'label', 'importance' (0-1), 'tags', 'l0_refs'.
            embedder: Optional embedder for semantic dedup.

        Returns:
            Dict with status, counts, and details.
        """
        if not facts:
            return {"status": "skipped", "reason": "no_facts", "facts_stored": 0}

        try:
            from cognition.dedup import SemanticDeduplicator
            deduplicator = SemanticDeduplicator()
        except ImportError:
            deduplicator = None

        PENDING_THRESHOLD = 0.3
        stored: list[str] = []
        dedup_results: list[dict] = []
        _facts_with_refs: list[dict] = []

        for fact in facts[:20]:
            label = fact.get("label") or fact.get("content") or ""
            if not label or len(label.strip()) < 8:
                continue

            importance = float(fact.get("importance", 0.5))
            l0_refs = fact.get("l0_refs") or fact.get("source_l0_refs") or []
            tags = fact.get("tags", [])

            if importance < PENDING_THRESHOLD:
                continue

            try:
                if deduplicator is not None:
                    result = deduplicator.add_fact_with_dedup(
                        engine, label, l0_refs if l0_refs else None,
                        embedder=embedder,
                    )
                    dedup_results.append(result)
                    status = result.get("status", "")
                    if status in ("added", "linked"):
                        stored.append(label)
                        try:
                            engine.feedback(label, importance)
                        except Exception:
                            pass
                    elif status == "merged":
                        stored.append(f"[merged] {label}")
                        try:
                            engine.feedback(label, importance * 0.5)
                        except Exception:
                            pass
                    _facts_with_refs.append({"label": label, "l0_refs": l0_refs})
                else:
                    add_fact_safely(engine, label, l0_refs if l0_refs else None)
                    stored.append(label)
                    try:
                        engine.feedback(label, importance)
                    except Exception:
                        pass
                    _facts_with_refs.append({"label": label, "l0_refs": l0_refs})
            except Exception as e:
                logger.warning(f"Failed to ingest fact '{label[:60]}': {e}")

        # Create L1 edges: facts sharing L0 references
        if len(stored) >= 2 and deduplicator is not None:
            try:
                _create_l0_ref_edges(engine, _facts_with_refs, deduplicator)
            except Exception:
                pass

        # L1→L2 graph aggregation
        l2_count = 0
        try:
            l2_count = _aggregate_connected_components(engine)
        except Exception:
            pass

        return {
            "status": "ingested",
            "facts_submitted": len(facts),
            "facts_stored": len(stored),
            "facts": stored,
            "l2_scenarios": l2_count,
            "dedup_summary": {
                "added": sum(1 for r in dedup_results if r.get("status") == "added"),
                "merged": sum(1 for r in dedup_results if r.get("status") == "merged"),
                "linked": sum(1 for r in dedup_results if r.get("status") == "linked"),
            },
        }

    def get_session_evidence_for_agent(
        self, engine: Any, session_id: str, limit: int = 50
    ) -> dict:
        """Fetch raw L0 evidence from a session for the Agent to process.

        The Agent calls this to get raw evidence, then uses its LLM to
        extract atomic facts, then calls extract_deep to store them.

        Args:
            engine: FourDMemEngine instance.
            session_id: Session to fetch evidence from.
            limit: Max evidence messages to return.

        Returns:
            Dict with status and evidence list (id, role, content).
        """
        try:
            raw = engine.get_session_evidence(session_id, limit)
            data = json.loads(raw) if isinstance(raw, str) else raw
            evidence_list = data.get("evidence", [])
        except Exception:
            try:
                raw = engine.query("*", limit)
                data = json.loads(raw) if isinstance(raw, str) else raw
                results = data.get("results", [])
                evidence_list = []
                for item in results:
                    if item.get("layer") == "L0" or "content" in item:
                        evidence_list.append({
                            "id": item.get("id"),
                            "content": item.get("content", ""),
                            "role": item.get("role", "unknown"),
                        })
            except Exception as e:
                return {"status": "error", "error": str(e), "evidence": []}

        if len(evidence_list) < 3:
            return {"status": "skipped", "reason": "too_few_messages", "evidence": evidence_list}

        # Build transcript for the Agent
        transcript_lines = []
        for ev in evidence_list:
            role = ev.get("role", "unknown")
            content = ev.get("content", "")
            transcript_lines.append(f"[{role}] {content}")

        return {
            "status": "ready",
            "evidence_count": len(evidence_list),
            "evidence": evidence_list,
            "transcript": "\n".join(transcript_lines),
            "instruction": (
                "Extract 1-3 atomic facts from this conversation. "
                "Each fact should be a single sentence capturing a decision, "
                "preference, technical choice, or important fact. "
                "Assign importance 0.5-1.0 and relevant tags. "
                "Return as JSON array: "
                '[{"label":"...", "importance":0.8, "tags":["tag1"]}]'
            ),
        }


# ── Graph helpers ──────────────────────────────────────────────────────────────

def _create_l0_ref_edges(engine: Any, facts: list[dict], deduplicator: Any):
    """Create 'related_to' edges between facts sharing L0 references."""
    fact_by_ref: dict[int, list[dict]] = {}
    for f in facts:
        for ref_id in f.get("l0_refs", []):
            if ref_id not in fact_by_ref:
                fact_by_ref[ref_id] = []
            fact_by_ref[ref_id].append(f)

    edge_count = 0
    for _ref_id, group in fact_by_ref.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                label_a = group[i].get("label", "")
                label_b = group[j].get("label", "")
                if label_a and label_b:
                    try:
                        engine.add_edge(label_a, label_b, "related_to", 0.5)
                        edge_count += 1
                    except Exception:
                        pass

    if edge_count > 0:
        logger.info(f"L1 edges created: {edge_count} related_to edges")


def _aggregate_connected_components(engine: Any) -> int:
    """L1→L2 aggregation via graph connected components.

    Reads edges from edges.jsonl on disk (cross-process safe).
    Finds connected components with >=3 nodes, creates L2 scenarios.
    """
    import json as _json

    edge_paths = [
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "vault", "l1", "edges.jsonl"),
    ]
    edge_file = None
    for p in edge_paths:
        if os.path.exists(os.path.normpath(p)):
            edge_file = os.path.normpath(p)
            break
    if not edge_file:
        return 0

    adj: dict[str, list[str]] = {}
    all_nodes: set[str] = set()
    try:
        with open(edge_file) as f:
            for line in f:
                e = _json.loads(line.strip())
                src, dst = e.get("src", ""), e.get("dst", "")
                if src and dst:
                    adj.setdefault(src, []).append(dst)
                    adj.setdefault(dst, []).append(src)
                    all_nodes.add(src)
                    all_nodes.add(dst)
    except Exception:
        return 0

    visited: set[str] = set()
    components: list[list[str]] = []
    for node in all_nodes:
        if node in visited:
            continue
        comp: list[str] = []
        queue = [node]
        visited.add(node)
        while queue:
            u = queue.pop(0)
            comp.append(u)
            for v in adj.get(u, []):
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        components.append(comp)

    fact_labels: dict[str, str] = {}
    fact_file = edge_file.replace("edges.jsonl", "facts.jsonl")
    if os.path.exists(fact_file):
        try:
            with open(fact_file) as f:
                for line in f:
                    fact = _json.loads(line.strip())
                    fid = fact.get("id", "")
                    label = fact.get("label", "")
                    if fid and label:
                        fact_labels[fid] = label
        except Exception:
            pass

    l2_count = 0
    for comp in components:
        if len(comp) < 3:
            continue
        labels = [fact_labels.get(nid, nid) for nid in comp if nid in fact_labels]
        if len(labels) < 2:
            continue

        safe_name = f"l2_auto_comp_{comp[0].replace(':','_').replace('.','_')}"
        scenario_content = "\n".join(f"- {l}" for l in labels[:30])
        sc_dir = os.path.normpath(os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "vault", "scenarios"
        ))
        try:
            os.makedirs(sc_dir, exist_ok=True)
            filepath = os.path.join(sc_dir, f"{safe_name}.md")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("---\n")
                f.write(_json.dumps({
                    "title": f"Auto-Aggregated Scenario ({len(comp)} facts, {len(labels)} labeled)",
                    "created": datetime.now(timezone.utc).isoformat(),
                    "l1_refs": comp[:20],
                    "source": "graph_component",
                }, indent=2))
                f.write("\n---\n\n")
                f.write(scenario_content)
                f.write("\n")
            l2_count += 1
        except Exception:
            pass

    return l2_count


def auto_capture_hook(engine: Any, session_id: str, _save_count: int) -> dict | None:
    """Hook for L0→L1 extraction check.

    No longer triggers rule-based extraction. Returns None — the Agent
    calls extract_deep explicitly when it wants to extract facts.
    The signal bus (cognition/signals.py) pushes extraction_suggested
    if the Agent hasn't extracted recently.
    """
    return None
