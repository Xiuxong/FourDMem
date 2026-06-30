# TODO(v4.2): Rewrite complexity estimation to use signal bus metrics
# instead of keyword heuristics. Then wire decide_action() into
# search_memory as a routing decision (macro_shortcut/standard_rrf/deep_reasoning).
# NarrativeIdentity tracking should be wired into _post_interaction.
# See TASKS.md Phase 2 for signal-driven cognition design.

"""System 3 Meta-Layer — Persistent agent identity and self-improvement.

Based on: "Sophia: A Persistent Agent Framework of Artificial Life"
(Sun et al., Dec 2025, arXiv:2512.18202)

Sophia introduces a third stratum (System 3) above System 1 (perception)
and System 2 (deliberation), responsible for:
  1. Narrative identity — maintaining a coherent autobiographical thread
  2. Process-supervised optimization — deciding when to think vs intuit
  3. User and self modeling — understanding both user preferences and own capabilities
  4. Hybrid reward — combining intrinsic (curiosity) and extrinsic (task success) signals

Mapped to FourDMem: L4 Observer → System 3 Meta-Layer upgrade
"""

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ── Narrative identity ────────────────────────────────────────────────────────

@dataclass
class NarrativeIdentity:
    """Continuous autobiographical thread (Sophia System 3)."""
    total_interactions: int = 0
    total_tasks: int = 0
    successful_tasks: int = 0
    total_errors: int = 0
    dominant_moods: list[str] = field(default_factory=list)
    key_milestones: list[dict] = field(default_factory=list)
    self_capability_belief: dict = field(default_factory=lambda: {
        "code_generation": 0.7,
        "debugging": 0.65,
        "architecture_design": 0.6,
        "refactoring": 0.55,
        "testing": 0.7,
    })
    user_preferences: dict = field(default_factory=dict)
    last_updated_tick: int = 0


@dataclass
class System3Decision:
    """Output of System 3 meta-cognitive decision."""
    action: str             # "macro_shortcut" | "standard_rrf" | "deep_reasoning"
    confidence: float
    reasoning_depth: int    # 0=macro, 1=standard, 2=deep multi-path
    expected_latency_ms: int
    rationale: str


class System3MetaLayer:
    """Sophia-inspired System 3 for persistent meta-cognition.

    Extends L4 Observer with:
      - Narrative identity maintenance
      - Task complexity estimation
      - Process-supervised decision making
      - Hybrid reward tracking
    """

    def __init__(self, vault_root: str = ""):
        self.vault_root = vault_root
        self.identity = NarrativeIdentity()

        # Decision thresholds (calibrated from Sophia paper)
        self.complexity_thresholds = {
            "simple": 0.25,     # below → macro shortcut
            "moderate": 0.55,   # below → standard RRF, above → deep reasoning
        }

        # Hybrid reward
        self.intrinsic_reward: float = 0.5   # curiosity/learning drive
        self.extrinsic_reward: float = 0.5   # task success feedback
        self.reward_history: list[float] = []

    def record_interaction(
        self,
        outcome: str,
        task_type: str,
        complexity: float,
        tick: int,
    ):
        """Update narrative identity after each interaction."""
        self.identity.total_interactions += 1
        self.identity.total_tasks += 1

        if outcome == "success":
            self.identity.successful_tasks += 1

            # Update self-capability belief (Bayesian-style update)
            cap = self.identity.self_capability_belief.get(task_type, 0.5)
            # Simple moving average toward 1.0 on success
            self.identity.self_capability_belief[task_type] = cap * 0.9 + 1.0 * 0.1

        elif outcome == "failure":
            self.identity.total_errors += 1

            cap = self.identity.self_capability_belief.get(task_type, 0.5)
            # Downgrade toward 0.0 on failure
            self.identity.self_capability_belief[task_type] = cap * 0.9 + 0.0 * 0.1

        # Record milestone if significant
        if self.identity.total_interactions % 100 == 0:
            self.identity.key_milestones.append({
                "tick": tick,
                "total_interactions": self.identity.total_interactions,
                "success_rate": (
                    self.identity.successful_tasks / self.identity.total_tasks
                    if self.identity.total_tasks > 0 else 0
                ),
                "capability_belief": dict(self.identity.self_capability_belief),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        self.identity.last_updated_tick = tick

    # ── Task complexity estimation ───────────────────────────────────────

    def estimate_complexity(self, query: str) -> float:
        """Estimate task complexity from query features.

        Returns complexity score in [0, 1].
        """
        score = 0.0

        # Length heuristic
        query_len = len(query)
        score += min(query_len / 1000, 0.2)

        # Multi-step indicators
        multi_step_indicators = [
            "然后", "接着", "首先", "之后", "最后",
            "first", "then", "next", "finally",
            "and also", "additionally",
        ]
        for kw in multi_step_indicators:
            if kw in query.lower():
                score += 0.1
                break

        # Architecture/design indicators → higher complexity
        arch_indicators = [
            "architecture", "design", "refactor", "migrate",
            "架构", "重构", "设计", "迁移", "方案",
        ]
        for kw in arch_indicators:
            if kw in query.lower():
                score += 0.15
                break

        # Debug/error indicators → moderate complexity
        debug_indicators = [
            "error", "bug", "fix", "debug", "panic",
            "报错", "修复", "调试",
        ]
        for kw in debug_indicators:
            if kw in query.lower():
                score += 0.08
                break

        # Code generation indicators → moderate-high
        code_indicators = ["implement", "write", "create", "generate", "实现", "写"]
        for kw in code_indicators:
            if kw in query.lower():
                score += 0.05
                break

        return min(score, 1.0)

    # ── Process-supervised decision ──────────────────────────────────────

    def decide_action(self, query: str) -> System3Decision:
        """Sophia System 3: decide which cognitive path to take.

        Three paths:
          1. macro_shortcut — cached intuition, ~10ms latency
          2. standard_rrf    — full RRF pipeline, ~100ms latency
          3. deep_reasoning  — multi-path exploration, ~500ms+ latency
        """
        complexity = self.estimate_complexity(query)

        # Check if we're capable enough in this domain
        task_type = self._infer_task_type(query)
        capability = self.identity.self_capability_belief.get(task_type, 0.5)

        if complexity < self.complexity_thresholds["simple"] and capability > 0.6:
            return System3Decision(
                action="macro_shortcut",
                confidence=capability,
                reasoning_depth=0,
                expected_latency_ms=10,
                rationale=f"Low complexity ({complexity:.2f}) + high capability ({capability:.2f})",
            )

        elif complexity < self.complexity_thresholds["moderate"]:
            return System3Decision(
                action="standard_rrf",
                confidence=0.7,
                reasoning_depth=1,
                expected_latency_ms=100,
                rationale=f"Moderate complexity ({complexity:.2f}), standard RRF",
            )

        else:
            return System3Decision(
                action="deep_reasoning",
                confidence=capability * 0.8,
                reasoning_depth=2,
                expected_latency_ms=500,
                rationale=f"High complexity ({complexity:.2f}), deep reasoning needed",
            )

    @staticmethod
    def _infer_task_type(query: str) -> str:
        """Infer task type from query."""
        q = query.lower()
        if any(w in q for w in ["debug", "error", "bug", "fix", "报错", "修复"]):
            return "debugging"
        if any(w in q for w in ["implement", "write", "code", "generate", "实现"]):
            return "code_generation"
        if any(w in q for w in ["test", "测试"]):
            return "testing"
        if any(w in q for w in ["refactor", "重构", "clean", "migrate"]):
            return "refactoring"
        if any(w in q for w in ["architecture", "design", "架构", "设计"]):
            return "architecture_design"
        return "code_generation"

    # ── Hybrid reward ────────────────────────────────────────────────────

    def update_reward(self, outcome: str, novelty: float = 0.5):
        """Update hybrid reward (intrinsic + extrinsic).

        Sophia uses dual reward:
          - Extrinsic: 1.0 for success, -1.0 for failure
          - Intrinsic: curiosity bonus based on query novelty
        """
        extrinsic = 1.0 if outcome == "success" else -1.0
        # Blend: weighted sum
        total_reward = 0.6 * extrinsic + 0.4 * novelty
        self.reward_history.append(total_reward)

        # Update running averages
        self.extrinsic_reward = (
            self.extrinsic_reward * 0.95 + extrinsic * 0.05
        )
        self.intrinsic_reward = (
            self.intrinsic_reward * 0.90 + novelty * 0.10
        )

        # Cap at [-1, 1]
        self.extrinsic_reward = max(-1.0, min(1.0, self.extrinsic_reward))
        self.intrinsic_reward = max(0.0, min(1.0, self.intrinsic_reward))

    # ── Self-model evaluation ────────────────────────────────────────────

    def evaluate_self_confidence(self, task_type: str) -> float:
        """Get current self-confidence in a task domain."""
        return self.identity.self_capability_belief.get(task_type, 0.5)

    def get_success_rate(self) -> float:
        """Compute overall success rate."""
        if self.identity.total_tasks == 0:
            return 0.5
        return self.identity.successful_tasks / self.identity.total_tasks

    # ── Stats ────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return System 3 statistics."""
        return {
            "narrative_identity": {
                "total_interactions": self.identity.total_interactions,
                "total_tasks": self.identity.total_tasks,
                "successful_tasks": self.identity.successful_tasks,
                "success_rate": self.get_success_rate(),
                "total_errors": self.identity.total_errors,
                "milestones": len(self.identity.key_milestones),
            },
            "capability_belief": dict(self.identity.self_capability_belief),
            "reward": {
                "extrinsic": round(self.extrinsic_reward, 3),
                "intrinsic": round(self.intrinsic_reward, 3),
                "history_len": len(self.reward_history),
            },
            "last_updated_tick": self.identity.last_updated_tick,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_system3: Optional[System3MetaLayer] = None


def get_system3(vault_root: str = "") -> System3MetaLayer:
    global _system3
    if _system3 is None:
        _system3 = System3MetaLayer(vault_root=vault_root)
    return _system3
