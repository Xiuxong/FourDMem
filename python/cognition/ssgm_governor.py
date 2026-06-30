"""SSGM (Stability & Safety-Governed Memory) Framework

Based on: "Governing Evolving Memory in LLM Agents: Risks, Mechanisms,
and the Stability and Safety Governed Memory (SSGM) Framework"
Lam et al., March 2026 (arXiv:2603.11768)

Core principles:
1. Decouple memory evolution from execution — modifications don't take
   effect until verified and committed.
2. Consistency verification — new memories must not contradict existing facts.
3. Temporal decay modeling — memory weights adjust with subjective time.
4. Dynamic access control — different agents see different memory views.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class GovernanceDecision:
    """Result of a pre-consolidation check."""
    action: str               # "allow" | "flag_conflict" | "reject" | "quarantine"
    contradicts: list[str] = field(default_factory=list)  # conflicting memory IDs
    suggested_resolution: Optional[str] = None
    retention_score: float = 1.0
    access_granted: bool = True
    reason: str = ""


@dataclass
class AccessRequest:
    """A request to read a memory item."""
    memory_id: str
    workspace_id: str
    agent_id: str
    memory_visibility: str  # "shared" | "workspace" | "private"
    memory_workspace_id: str


# ── Immutable constraints ─────────────────────────────────────────────────────

IMMUTABLE_CONSTRAINTS: dict[str, Any] = {
    "max_weight_delta": 0.1,           # Max single-intervention weight change
    "min_rif_weight": 0.05,            # Any RIF-U dimension floor
    "rollback_enabled": True,          # Always keep old parameter snapshots
    "intervention_cooldown_ticks": 5,  # Cooldown between interventions
    "forbidden_targets": [             # Targets that can NEVER be modified
        "delete_evidence",
        "drop_index",
        "truncate_table",
        "purge_l0",
        "modify_schema",
    ],
    "constitutional_rules": [          # Cannot be changed by Paradigm Shift
        "never_delete_raw_evidence",
        "always_preserve_provenance_chain",
        "maintain_version_history",
        "no_user_data_exfiltration",
    ],
}

# ── SSGM Governor ──────────────────────────────────────────────────────────────

class SSGMGovernor:
    """Memory governance: enforces safety, consistency, and access control.

    Integrates at three points in the memory lifecycle:
    1. **Pre-consolidation** — before new facts are written to L1
    2. **Pre-evolution** — before Strange Loop / Paradigm Shift modifies rules
    3. **Pre-access** — before search_memory returns results to an agent
    """

    def __init__(self, vault_root: str = ""):
        self.vault_root = vault_root
        self._modification_log: list[dict] = []
        self._rollback_snapshots: dict[str, dict] = {}
        self._intervention_ticks: dict[str, int] = {}
        self._last_intervention_tick: int = 0

    # ── Pre-consolidation check ────────────────────────────────────────────

    def pre_consolidation_check(
        self,
        new_content: str,
        existing_facts: list[dict],
        current_tick: int,
    ) -> GovernanceDecision:
        """SSGM Step 1: verify new memory doesn't contradict existing facts.

        Args:
            new_content: the candidate fact content
            existing_facts: list of {id, content, embedding, ...} from L1
            current_tick: current subjective time tick

        Returns:
            GovernanceDecision with action and details
        """
        # Semantic contradiction detection via embedding similarity
        contradictions = []
        new_emb = self._get_embedding(new_content)

        for fact in existing_facts:
            existing_emb = fact.get("embedding")
            if not existing_emb:
                continue
            # High cosine similarity + opposite sentiment → potential contradiction
            sim = self._cosine_sim(new_emb, existing_emb)
            if sim > 0.85:
                # Check if it's a contradiction or just a restatement
                if self._is_contradictory(new_content, fact.get("content", "")):
                    contradictions.append(fact["id"])

        if contradictions:
            resolution = self._propose_resolution(new_content, [
                f for f in existing_facts if f["id"] in contradictions
            ])
            return GovernanceDecision(
                action="flag_conflict",
                contradicts=contradictions,
                suggested_resolution=resolution,
                reason=f"Contradicts {len(contradictions)} existing fact(s)",
            )

        return GovernanceDecision(action="allow", reason="No conflicts detected")

    # ── Temporal decay model ───────────────────────────────────────────────

    def temporal_decay_model(
        self,
        memory: dict,
        current_tick: int,
    ) -> float:
        """SSGM Step 2: compute retention score based on subjective time.

        Uses Ebbinghaus-inspired decay curve modified for Active Tick space.
        High-utility memories receive decay immunity.

        Args:
            memory: {last_active_tick, utility_score, half_life_ticks, ...}
            current_tick: current subjective time

        Returns:
            retention score in [0.0, 1.0]
        """
        last_tick = memory.get("last_active_tick", 0)
        half_life = memory.get("half_life_ticks", 90)  # default: 90 ticks
        utility = memory.get("utility_score", 0.0)

        ticks_since = max(0, current_tick - last_tick)

        # Ebbinghaus decay: retention = e^(-ticks / half_life)
        if half_life <= 0:
            retention = 1.0
        else:
            import math
            retention = math.exp(-ticks_since / half_life)

        # Utility-based immunity: high utility memories resist decay
        if utility > 0.8:
            retention = max(retention, 0.9)
        elif utility > 0.5:
            retention = max(retention, 0.7)

        # Immune shelf-life: never decay
        shelf_life = memory.get("shelf_life", "")
        if shelf_life == "immune":
            retention = 1.0

        return min(1.0, max(0.0, retention))

    # ── Dynamic access control ─────────────────────────────────────────────

    def access_control(self, request: AccessRequest) -> GovernanceDecision:
        """SSGM Step 3: determine if an agent can access a memory item.

        Visibility levels:
          - "shared"   → all agents in all workspaces
          - "workspace" → same workspace agents only
          - "private"  → creating agent only
        """
        if request.memory_visibility == "shared":
            return GovernanceDecision(action="allow", access_granted=True,
                                      reason="shared visibility")

        if request.memory_visibility == "workspace":
            if request.workspace_id == request.memory_workspace_id:
                return GovernanceDecision(action="allow", access_granted=True,
                                          reason="same workspace")
            return GovernanceDecision(action="reject", access_granted=False,
                                      reason=f"workspace mismatch: {request.workspace_id} != {request.memory_workspace_id}")

        if request.memory_visibility == "private":
            if request.agent_id == request.memory_id.split(":")[0]:
                return GovernanceDecision(action="allow", access_granted=True,
                                          reason="private owner")
            return GovernanceDecision(action="reject", access_granted=False,
                                      reason="private memory")

        return GovernanceDecision(action="allow", access_granted=True,
                                  reason="default allow")

    # ── Evolution governance ──────────────────────────────────────────────

    def pre_evolution_check(
        self,
        target: str,
        proposed_value: Any,
        current_value: Any,
        current_tick: int,
    ) -> GovernanceDecision:
        """Govern any self-modification attempt (Strange Loop / Paradigm Shift).

        Args:
            target: what is being modified (e.g. "rif_weights.recency")
            proposed_value: new value
            current_value: current value
            current_tick: current subjective time

        Returns:
            GovernanceDecision
        """
        # 1. Immutable check
        if target in IMMUTABLE_CONSTRAINTS["forbidden_targets"]:
            return GovernanceDecision(action="reject",
                                      reason=f"target '{target}' is immutable")

        # 2. Constitutional rule check
        for rule in IMMUTABLE_CONSTRAINTS["constitutional_rules"]:
            if self._violates_constitutional_rule(target, proposed_value, rule):
                return GovernanceDecision(action="reject",
                                          reason=f"violates constitutional rule: {rule}")

        # 3. Delta check (weight-type targets)
        if isinstance(current_value, (int, float)) and isinstance(proposed_value, (int, float)):
            max_delta = IMMUTABLE_CONSTRAINTS["max_weight_delta"]
            current_abs = abs(current_value) if abs(current_value) > 0 else 1.0
            delta = abs(proposed_value - current_value) / current_abs
            if delta > max_delta:
                return GovernanceDecision(action="reject",
                                          reason=f"delta {delta:.3f} exceeds max {max_delta}")

        # 4. Floor check
        if isinstance(proposed_value, (int, float)):
            if proposed_value < IMMUTABLE_CONSTRAINTS["min_rif_weight"]:
                return GovernanceDecision(action="reject",
                                          reason=f"value {proposed_value} below floor {IMMUTABLE_CONSTRAINTS['min_rif_weight']}")

        # 5. Cooldown check
        cooldown = IMMUTABLE_CONSTRAINTS["intervention_cooldown_ticks"]
        last = self._last_intervention_tick
        if current_tick - last < cooldown and last > 0:
            return GovernanceDecision(action="reject",
                                      reason=f"cooldown: {current_tick - last} < {cooldown} ticks since last intervention")

        # 6. Create rollback snapshot
        if IMMUTABLE_CONSTRAINTS["rollback_enabled"]:
            snap_id = hashlib.md5(f"{target}:{current_tick}".encode()).hexdigest()[:12]
            self._rollback_snapshots[snap_id] = {
                "target": target,
                "previous_value": current_value,
                "tick": current_tick,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        self._last_intervention_tick = current_tick

        return GovernanceDecision(
            action="allow",
            reason=f"approved (snapshot: {snap_id if IMMUTABLE_CONSTRAINTS['rollback_enabled'] else 'none'})",
        )

    def rollback(self, snapshot_id: str) -> Optional[dict]:
        """Restore a previous parameter state."""
        if snapshot_id not in self._rollback_snapshots:
            return None
        snap = self._rollback_snapshots.pop(snapshot_id)
        self._modification_log.append({
            "action": "rollback",
            "snapshot_id": snapshot_id,
            "restored_target": snap["target"],
            "restored_value": snap["previous_value"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return snap

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_embedding(self, text: str) -> list[float]:
        """Get semantic embedding for a piece of text."""
        try:
            from cognition.embedder import get_embedder
            emb = get_embedder()
            return emb.embed(text)
        except Exception:
            return [0.0] * 768

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _is_contradictory(new_text: str, existing_text: str) -> bool:
        """Simple heuristic for contradiction detection.
        
        A more robust implementation would use an NLI model (e.g. deberta-v3).
        For now, look for negation pattern mismatches.
        """
        negation_words = {"not", "never", "no", "cannot", "shouldn't", "don't", "doesn't"}
        new_lower = new_text.lower()
        exist_lower = existing_text.lower()

        new_has_neg = any(w in new_lower.split() for w in negation_words)
        exist_has_neg = any(w in exist_lower.split() for w in negation_words)

        # High similarity + opposite negation → likely contradiction
        if new_has_neg != exist_has_neg:
            return True

        return False

    def _propose_resolution(
        self, new_content: str, conflicting_facts: list[dict]
    ) -> str:
        """Suggest how to resolve a contradiction."""
        if len(conflicting_facts) == 1:
            return (
                f"New fact contradicts existing fact '{conflicting_facts[0]['id']}'. "
                f"Consider: (1) marking existing fact as outdated, "
                f"(2) creating a version branch, or (3) adding counterfactual context."
            )
        return (
            f"New fact contradicts {len(conflicting_facts)} existing facts. "
            f"Recommend creating a new version branch with explicit counterfactual "
            f"marking to preserve both old and new knowledge."
        )

    def _violates_constitutional_rule(
        self, target: str, proposed_value: Any, rule: str
    ) -> bool:
        """Check if a proposed modification violates a constitutional rule."""
        # Target-level matching
        if rule == "never_delete_raw_evidence" and "delete" in target.lower():
            return True
        if rule == "always_preserve_provenance_chain" and "origin" in target.lower():
            if proposed_value is None:
                return True
        if rule == "maintain_version_history" and "version" in target.lower():
            return True
        return False

    def get_modification_log(self, limit: int = 20) -> list[dict]:
        """Return recent governance decisions for audit."""
        return self._modification_log[-limit:]

    def get_stats(self) -> dict:
        """Return governance statistics."""
        total = len(self._modification_log)
        allowed = sum(1 for m in self._modification_log if m.get("action") in ("allow", "approved"))
        rejected = total - allowed
        return {
            "total_decisions": total,
            "allowed": allowed,
            "rejected": rejected,
            "rejection_rate": rejected / total if total > 0 else 0.0,
            "active_snapshots": len(self._rollback_snapshots),
            "last_intervention_tick": self._last_intervention_tick,
        }


# ── Negation weight ─────────────────────────────────────────────────────────

def negation_weight(text: str) -> float:
    """Detect negation patterns and return a penalty weight.

    Chinese patterns: 不是, 不要, 避免, 切勿
    English patterns: revert, rollback, don't, never, avoid, should not

    Returns 0.3 if any negation pattern is found, 1.0 otherwise.
    """
    if not text:
        return 1.0

    text_lower = text.lower()

    # Chinese negation patterns
    cn_patterns = ["不是", "不要", "避免", "切勿"]
    for pat in cn_patterns:
        if pat in text:
            return 0.3

    # English negation patterns
    en_patterns = ["revert", "rollback", "don't", "never", "avoid", "should not"]
    for pat in en_patterns:
        if pat in text_lower:
            return 0.3

    return 1.0


# ── Singleton factory ─────────────────────────────────────────────────────────

_governor: Optional[SSGMGovernor] = None


def get_ssgm_governor(vault_root: str = "") -> SSGMGovernor:
    """Lazy singleton for SSGMGovernor."""
    global _governor
    if _governor is None:
        _governor = SSGMGovernor(vault_root=vault_root)
    return _governor
