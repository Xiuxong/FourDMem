"""L2→L3 Auto-Promoter — Automatic Core Rule Promotion.

Tracks L2 scenario usage and promotes frequently-referenced scenarios
to L3 core rules when they meet the criteria:
- Referenced ≥ 20 times in search_memory
- Span ≥ 50 active ticks
- EMA score ≥ 5.0
- Requires explicit approval (submit_feedback with high score)

Implements T-9.3: 引用 ≥ 20 次且跨度 ≥ 50 个 active_ticks 且 EMA ≥ 5.0，提议晋升 L3.
"""

import json
import os
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class AutoPromoter:
    """Tracks L2 scenario usage and promotes to L3.

    Args:
        min_references: Minimum search hits before promotion eligible.
        min_tick_span: Minimum active tick span before promotion eligible.
        project_root: Project root directory for file paths.
    """

    def __init__(
        self,
        min_references: int = 20,
        min_tick_span: int = 50,
        project_root: str | None = None,
    ):
        self.min_references = min_references
        self.min_tick_span = min_tick_span
        self.project_root = project_root or os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        # Track L2 scenario references
        self._reference_counts: dict[str, dict] = {}
        self._load_state()

    def record_reference(self, scenario_id: str, tick: int = 0) -> dict | None:
        """Record that an L2 scenario was referenced in a search result.

        Args:
            scenario_id: The L2 scenario identifier (filename or slug).
            tick: Current active tick.

        Returns:
            Promotion proposal if criteria met, None otherwise.
        """
        if scenario_id not in self._reference_counts:
            self._reference_counts[scenario_id] = {
                "count": 0,
                "first_tick": tick,
                "last_tick": tick,
                "ema_score": 0.0,
                "approved": False,
            }
        entry = self._reference_counts[scenario_id]

        # EMA: decay based on time since last reference, then boost
        tick_delta = tick - entry.get("last_tick", tick)
        decay = 0.95 ** max(tick_delta, 1)
        entry["ema_score"] = entry.get("ema_score", 0.0) * decay + 1.0

        entry["count"] += 1
        entry["last_tick"] = tick
        # Check promotion criteria (count + tick_span)
        tick_span = entry["last_tick"] - entry["first_tick"]
        if (
            entry["count"] >= self.min_references
            and tick_span >= self.min_tick_span
            and entry["ema_score"] >= 5.0
            and not entry["approved"]
        ):
            return self._propose_promotion(scenario_id, entry)

        return None

    def approve_promotion(self, scenario_id: str) -> dict:
        """Approve a pending promotion and execute L3 writeback.

        Args:
            scenario_id: The scenario to promote.

        Returns:
            Promotion result dict.
        """
        if scenario_id not in self._reference_counts:
            return {"status": "not_found", "scenario_id": scenario_id}

        entry = self._reference_counts[scenario_id]
        entry["approved"] = True
        self._save_state()

        # Execute promotion
        result = self._promote_to_l3(scenario_id)

        logger.info(f"L2→L3 promotion approved and executed: {scenario_id}")
        return result

    def get_pending_proposals(self) -> list[dict]:
        """Get all pending promotion proposals.

        Returns:
            List of scenarios eligible for promotion but not yet approved.
        """
        proposals = []
        for scenario_id, entry in self._reference_counts.items():
            tick_span = entry["last_tick"] - entry["first_tick"]
            if (
                entry["count"] >= self.min_references
                and tick_span >= self.min_tick_span
                and not entry["approved"]
            ):
                proposals.append({
                    "scenario_id": scenario_id,
                    "reference_count": entry["count"],
                    "tick_span": tick_span,
                    "ema_score": round(entry.get("ema_score", 0.0), 2),
                    "first_tick": entry["first_tick"],
                    "last_tick": entry["last_tick"],
                })

        return proposals

    def get_stats(self) -> dict:
        """Get promoter statistics."""
        total = len(self._reference_counts)
        approved = sum(1 for e in self._reference_counts.values() if e["approved"])
        pending = len(self.get_pending_proposals())

        return {
            "tracked_scenarios": total,
            "approved": approved,
            "pending_proposals": pending,
            "min_references": self.min_references,
            "min_tick_span": self.min_tick_span,
        }

    def _propose_promotion(self, scenario_id: str, entry: dict) -> dict:
        proposal = {
            "event": "promotion_proposed",
            "scenario_id": scenario_id,
            "reference_count": entry["count"],
            "tick_span": entry["last_tick"] - entry["first_tick"],
            "ema_score": round(entry.get("ema_score", 0.0), 2),
            "message": (
                f"L2 scenario '{scenario_id}' has been referenced {entry['count']} times "
                f"over {entry['last_tick'] - entry['first_tick']} ticks "
                f"(EMA={entry.get('ema_score', 0.0):.1f}). "
                f"Eligible for L3 promotion. Call approve_promotion() to execute."
            ),
        }

        # Push cognitive signal for Agent review
        try:
            from cognition.signals import get_signal_bus
            bus = get_signal_bus()
            bus.push(
                "promotion_proposed",
                priority=5,
                payload={
                    "scenario_id": scenario_id,
                    "reference_count": entry["count"],
                    "tick_span": entry["last_tick"] - entry["first_tick"],
                    "ema_score": round(entry.get("ema_score", 0.0), 2),
                    "action_hint": (
                        f"Scenario '{scenario_id}' eligible for L3 promotion "
                        f"({entry['count']} refs over {entry['last_tick'] - entry['first_tick']} ticks). "
                        f"Approve with approve_promotion('{scenario_id}') or ignore."
                    ),
                },
            )
        except Exception:
            pass

        logger.info(f"Promotion proposed: {scenario_id} (refs={entry['count']}, ema={entry.get('ema_score', 0.0):.1f})")
        return proposal

    def _promote_to_l3(self, scenario_id: str) -> dict:
        """Execute the L2→L3 promotion.

        Reads the L2 scenario and adds its key rules to L3 persona.
        """
        try:
            # Read L2 scenario
            scenarios_dir = os.path.join(self.project_root, "data", "vault", "scenarios")
            scenario_path = os.path.join(scenarios_dir, f"{scenario_id}.md")

            if not os.path.exists(scenario_path):
                # Try with .md extension
                for f in os.listdir(scenarios_dir):
                    if f.startswith(scenario_id):
                        scenario_path = os.path.join(scenarios_dir, f)
                        break

            if not os.path.exists(scenario_path):
                return {"status": "scenario_not_found", "scenario_id": scenario_id}

            with open(scenario_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract frontmatter
            frontmatter = {}
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    try:
                        import yaml
                        frontmatter = yaml.safe_load(parts[1]) or {}
                    except Exception:
                        pass

            # Read L3 persona
            persona_path = os.path.join(
                self.project_root, "data", "vault", "persona", "persona.yaml"
            )
            persona = {}
            if os.path.exists(persona_path):
                try:
                    import yaml
                    with open(persona_path, "r", encoding="utf-8") as f:
                        persona = yaml.safe_load(f) or {}
                except Exception:
                    pass

            # Add promoted rule to L3
            if "promoted_rules" not in persona:
                persona["promoted_rules"] = []

            persona["promoted_rules"].append({
                "source_scenario": scenario_id,
                "title": frontmatter.get("title", scenario_id),
                "tags": frontmatter.get("tags", []),
                "reference_count": self._reference_counts[scenario_id]["count"],
                "promoted_at_tick": self._reference_counts[scenario_id]["last_tick"],
                "guardrails": {
                    "max_weight": 0.5,
                    "forbidden_patterns": [],
                    "rollback_enabled": True,
                },
            })

            # Write back
            os.makedirs(os.path.dirname(persona_path), exist_ok=True)
            import yaml
            with open(persona_path, "w", encoding="utf-8") as f:
                yaml.dump(persona, f, default_flow_style=False, allow_unicode=True)

            self._save_state()

            return {
                "status": "promoted",
                "scenario_id": scenario_id,
                "persona_path": persona_path,
            }

        except Exception as e:
            logger.error(f"L3 promotion failed: {e}")
            return {"status": "error", "error": str(e)}

    def _state_path(self) -> str:
        """Path to persistence file."""
        return os.path.join(self.project_root, "data", "vault", ".promotion_state.json")

    def _save_state(self) -> None:
        """Save reference counts to disk."""
        try:
            os.makedirs(os.path.dirname(self._state_path()), exist_ok=True)
            with open(self._state_path(), "w") as f:
                json.dump(self._reference_counts, f, indent=2)
        except Exception:
            pass

    def _load_state(self) -> None:
        """Load reference counts from disk."""
        try:
            if os.path.exists(self._state_path()):
                with open(self._state_path()) as f:
                    self._reference_counts = json.load(f)
        except Exception:
            self._reference_counts = {}
