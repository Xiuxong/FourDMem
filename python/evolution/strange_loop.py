"""Strange Loop — Observer Node (L4 Meta-Cognition)

The Observer monitors the retrieval system's own behavior and
modifies its parameters when it detects inefficiency.

Implements T-10.6: the system can "notice" it's stuck and modify
its own RIF-U weights, confidence threshold, or prompt templates.

Detection patterns:
- Confidence crisis: N consecutive queries with confidence < 0.3
- Layer starvation: one layer consistently empty in results
- Macro stagnation: promoted macros have declining success rate
- Model change: agent_id changed → persist cognitive state, reset counters
"""

import json
import os
import time
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class ObserverNode:
    """L4 meta-cognitive observer that monitors and adjusts the retrieval system.

    Manages the cognitive state lifecycle: persists subsystem state to
    data/cognition/*.json on model-change or explicit flush, restores
    on wake_up, and detects when the underlying model/agent has swapped.

    Usage:
        observer = ObserverNode()
        action = observer.observe(engine, recent_queries)
        if action:
            print(f"Observer intervened: {action}")
    """

    def __init__(
        self,
        confidence_crisis_threshold: float = 0.3,
        crisis_streak_length: int = 5,
    ):
        self.confidence_crisis_threshold = confidence_crisis_threshold
        self.crisis_streak_length = crisis_streak_length
        self.query_history: list[dict] = []
        # Model-change tracking: last known model name
        self._last_model_name: str | None = None

    # ── Core observe loop ────────────────────────────────────────────────

    def observe(
        self,
        engine: Any,
        query_result: dict | None = None,
        project_root: str | None = None,
    ) -> dict | None:
        """Observe a query result and decide if intervention is needed.

        First checks for model change; if detected, persists all cognitive
        state and resets counters before proceeding with normal observation.

        Args:
            engine: FourDMemEngine instance.
            query_result: The latest query result dict.
            project_root: Optional repo root for persistence. If provided,
                model-change detection and persistence are enabled.

        Returns:
            Action dict if intervention triggered, None otherwise.
        """
        # Step 0: detect model change (if project_root available)
        if project_root:
            change = self._detect_model_change(engine)
            if change:
                self._handle_model_change(engine, change, project_root)

        if query_result:
            self.query_history.append(query_result)

        # Keep only recent history
        if len(self.query_history) > 100:
            self.query_history = self.query_history[-50:]

        # Check for confidence crisis
        if self._detect_confidence_crisis():
            return self._intervene_confidence(engine)

        # Check for layer starvation
        if self._detect_layer_starvation():
            return self._intervene_layer_balance(engine)

        return None

    # ── Detection methods ────────────────────────────────────────────────

    def _detect_confidence_crisis(self) -> bool:
        """Detect if the last N queries all had low confidence."""
        if len(self.query_history) < self.crisis_streak_length:
            return False

        recent = self.query_history[-self.crisis_streak_length:]
        return all(
            q.get("confidence", 1.0) < self.confidence_crisis_threshold
            for q in recent
        )

    def _detect_layer_starvation(self) -> bool:
        """Detect if one layer is consistently empty in results."""
        if len(self.query_history) < 10:
            return False

        recent = self.query_history[-10:]
        layer_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        for q in recent:
            for item in q.get("results", []):
                layer = item.get("layer", 2)
                layer_counts[layer] = layer_counts.get(layer, 0) + 1

        # If any layer has 0 results across 10 queries, it's starving
        total_results = sum(layer_counts.values())
        if total_results == 0:
            return False

        for layer, count in layer_counts.items():
            if count == 0 and layer in [1, 2]:  # L1 and L2 should have results
                return True

        return False

    def _detect_model_change(self, engine: Any) -> dict | None:
        """Compare current model name with last known.

        Returns a change dict if the model has swapped since last
        observation, None if unchanged.

        The model_name is derived from the engine's session metadata or
        a deterministic hash of the engine's configuration.
        """
        try:
            current_id = self._derive_model_name(engine)
        except Exception:
            return None

        if self._last_model_name is None:
            # First observation — record baseline, no change
            self._last_model_name = current_id
            return None

        if current_id != self._last_model_name:
            change = {
                "event": "model_change",
                "old_model_name": self._last_model_name,
                "new_model_name": current_id,
                "timestamp": time.time(),
            }
            logger.warning(
                f"Observer: model change detected "
                f"{self._last_model_name!r} → {current_id!r}"
            )
            self._last_model_name = current_id
            return change

        return None

    @staticmethod
    def _derive_model_name(engine: Any) -> str:
        """Derive a stable identity string from the engine instance.

        Tries engine attributes in order of specificity; falls back to
        a repr-based hash. This ID changes when the model or config swaps.
        """
        # Prefer an explicit agent_id attribute if the Rust bindings expose one
        for attr in ("agent_id", "model_name", "session_model"):
            val = getattr(engine, attr, None)
            if val is not None:
                return str(val)

        # Fallback: hash the engine's repr (changes on config swap)
        import hashlib
        return hashlib.md5(repr(engine).encode()).hexdigest()[:16]

    # ── Model-change handler ─────────────────────────────────────────────

    def _handle_model_change(
        self,
        engine: Any,
        change: dict,
        project_root: str,
    ) -> None:
        """Persist all cognitive state, reset counters, log event.

        Called when a model/agent change is detected. Ensures the new agent
        starts with a clean slate while preserving the old agent's learned
        state on disk.
        """
        logger.info("Observer: handling model change — persisting cognitive state")

        # 1. Persist current state to disk
        self.persist_cognitive_state(engine, project_root)

        # 2. Reset observer's own counters
        self.query_history = []

        # 3. Write the model-change event to disk
        self._write_model_change_event(change, project_root)

        logger.info("Observer: model-change handling complete")

    def _write_model_change_event(self, change: dict, project_root: str) -> None:
        """Write model-change event to data/cognition/model_changes.json."""
        cognition_dir = os.path.join(project_root, "data", "cognition")
        events_path = os.path.join(cognition_dir, "model_changes.json")

        try:
            os.makedirs(cognition_dir, exist_ok=True)
            events = []
            if os.path.exists(events_path):
                with open(events_path, "r", encoding="utf-8") as f:
                    events = json.load(f)

            events.append(change)

            with open(events_path, "w", encoding="utf-8") as f:
                json.dump(events, f, indent=2, ensure_ascii=False)

            logger.debug(f"Observer: model-change event written to {events_path}")
        except Exception as e:
            logger.warning(f"Observer: failed to write model-change event: {e}")

    # ── Intervention methods ─────────────────────────────────────────────

    def _intervene_confidence(self, engine: Any) -> dict:
        """Detected confidence crisis — push signal to SignalBus for Agent review.

        Instead of auto-adjusting thresholds, the Observer pushes an advisory
        signal. The Agent decides whether to adjust retrieval parameters.
        """
        logger.warning(
            f"Observer: confidence crisis detected "
            f"({self.crisis_streak_length} consecutive low-confidence queries)"
        )

        recent_confidences = [
            q.get("confidence", 1.0) for q in self.query_history[-self.crisis_streak_length:]
        ]
        avg_conf = sum(recent_confidences) / len(recent_confidences) if recent_confidences else 0.0

        try:
            from cognition.signals import get_signal_bus
            bus = get_signal_bus()
            bus.push(
                "observer_alert",
                priority=7,
                payload={
                    "issue": "confidence_crisis",
                    "avg_confidence": round(avg_conf, 3),
                    "streak_length": self.crisis_streak_length,
                    "recommendation": (
                        "Multiple consecutive queries returned low confidence. "
                        "Consider: (1) checking if indices are populated, "
                        "(2) lowering the confidence_drill_down_threshold, "
                        "(3) rebuilding L1 from L0 evidence."
                    ),
                },
            )
        except ImportError:
            pass

        self.query_history = []
        return {"action": "signal_pushed", "reason": "confidence_crisis",
                "avg_confidence": round(avg_conf, 3)}

    def _intervene_layer_balance(self, engine: Any) -> dict:
        """Detected layer starvation — push signal to SignalBus for Agent review."""
        logger.warning("Observer: layer starvation detected")

        try:
            from cognition.signals import get_signal_bus
            bus = get_signal_bus()
            bus.push(
                "observer_alert",
                priority=6,
                payload={
                    "issue": "layer_starvation",
                    "recommendation": (
                        "One or more layers returned no results across recent queries. "
                        "Check if graph/fulltext indices are populated. "
                        "Consider running rebuild_l1 to restore indices."
                    ),
                },
            )
        except ImportError:
            pass

        self.query_history = []
        return {"action": "signal_pushed", "reason": "layer_starvation"}

    # ── Cognitive state persistence ──────────────────────────────────────

    def persist_cognitive_state(self, engine: Any, project_root: str) -> None:
        """Serialize all cognitive state to data/cognition/*.json.

        Writes four files:
        - macros.json — Rust MacroCache promoted macros
        - domain_reliability.json — ParadigmShift failure_counts
        - query_patterns.json — MyelinationTracker patterns
        - observer_snapshot.json — this observer's query_history summary
        """
        cognition_dir = os.path.join(project_root, "data", "cognition")
        try:
            os.makedirs(cognition_dir, exist_ok=True)
        except OSError as e:
            logger.warning(f"Observer: cannot create cognition dir: {e}")
            return

        # 1. Macro stats from Rust engine
        self._persist_macros(engine, cognition_dir)

        # 2. ParadigmShift domain reliability
        self._persist_domain_reliability(cognition_dir)

        # 3. Myelination query patterns
        self._persist_query_patterns(cognition_dir)

        # 4. Observer snapshot
        self._persist_observer_snapshot(cognition_dir)

        logger.info(f"Observer: cognitive state persisted to {cognition_dir}")

    def _persist_macros(self, engine: Any, cognition_dir: str) -> None:
        """Persist Rust MacroCache stats to macros.json."""
        path = os.path.join(cognition_dir, "macros.json")
        try:
            raw = engine.get_macro_stats()
            stats = json.loads(raw) if isinstance(raw, str) else raw

            # Reset counters for persistence (new agent re-validates)
            for macro in stats.get("macros", []):
                macro["hit_count"] = 0
                macro["success_rate"] = 0.5

            stats["persisted_at"] = time.time()
            stats["persist_note"] = (
                "Counters reset to 0/0.5 for new-agent re-validation"
            )

            with open(path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

            logger.debug(
                f"Observer: persisted {stats.get('promoted_macros', 0)} macros"
            )
        except Exception as e:
            logger.warning(f"Observer: failed to persist macros: {e}")

    def _persist_domain_reliability(self, cognition_dir: str) -> None:
        """Persist ParadigmShift failure_counts to domain_reliability.json."""
        path = os.path.join(cognition_dir, "domain_reliability.json")
        try:
            paradigm = self._get_paradigm_engine()
            if paradigm is None:
                return

            data = {
                "failure_counts": paradigm.failure_counts,
                "failure_threshold": paradigm.failure_threshold,
                "persisted_at": time.time(),
            }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug(
                f"Observer: persisted {len(paradigm.failure_counts)} domain records"
            )
        except Exception as e:
            logger.warning(f"Observer: failed to persist domain reliability: {e}")

    def _persist_query_patterns(self, cognition_dir: str) -> None:
        """Persist MyelinationTracker patterns to query_patterns.json."""
        path = os.path.join(cognition_dir, "query_patterns.json")
        try:
            myelination = self._get_myelination()
            if myelination is None:
                return

            # Serialize the internal _patterns dict
            patterns_data = {}
            for key, entry in myelination._patterns.items():
                patterns_data[key] = {
                    "query_template": entry.get("representative_query", ""),
                    "hit_count": entry.get("hits", 0),
                    "success_count": entry.get("successes", 0),
                    "last_confidence": entry.get("last_confidence", 0.0),
                }

            data = {
                "patterns": patterns_data,
                "compilation_threshold": myelination.compilation_threshold,
                "success_rate_required": myelination.success_rate_required,
                "persisted_at": time.time(),
            }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug(
                f"Observer: persisted {len(patterns_data)} query patterns"
            )
        except Exception as e:
            logger.warning(f"Observer: failed to persist query patterns: {e}")

    def _persist_observer_snapshot(self, cognition_dir: str) -> None:
        """Persist observer's own state to observer_snapshot.json."""
        path = os.path.join(cognition_dir, "observer_snapshot.json")
        try:
            snapshot = {
                "summary": self.get_observation_summary(),
                "query_history_length": len(self.query_history),
                "confidence_crisis_threshold": self.confidence_crisis_threshold,
                "crisis_streak_length": self.crisis_streak_length,
                "last_model_name": self._last_model_name,
                "persisted_at": time.time(),
            }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)

            logger.debug("Observer: snapshot persisted")
        except Exception as e:
            logger.warning(f"Observer: failed to persist snapshot: {e}")

    # ── Cognitive state restoration ──────────────────────────────────────

    def restore_cognitive_state(self, engine: Any, project_root: str) -> dict:
        """Load cognitive state from data/cognition/*.json and inject into singletons.

        Reads persisted state files and restores subsystem counters.
        All counters are reset (hit_count→0, success_rate→0.5) so the new
        agent re-validates from a known baseline.

        Args:
            engine: FourDMemEngine instance.
            project_root: Repo root where data/cognition/ lives.

        Returns:
            Summary dict of what was restored.
        """
        cognition_dir = os.path.join(project_root, "data", "cognition")
        summary: dict[str, Any] = {
            "macros_restored": 0,
            "domains_restored": 0,
            "patterns_restored": 0,
            "observer_restored": False,
            "errors": [],
        }

        if not os.path.isdir(cognition_dir):
            logger.info("Observer: no cognition dir found — fresh start")
            return summary

        # 1. Restore macros into Rust engine
        summary["macros_restored"] = self._restore_macros(engine, cognition_dir, summary)

        # 2. Restore domain reliability into ParadigmShift
        summary["domains_restored"] = self._restore_domain_reliability(cognition_dir, summary)

        # 3. Restore query patterns into MyelinationTracker
        summary["patterns_restored"] = self._restore_query_patterns(cognition_dir, summary)

        # 4. Restore observer snapshot
        summary["observer_restored"] = self._restore_observer_snapshot(cognition_dir, summary)

        logger.info(
            f"Observer: restore complete — "
            f"{summary['macros_restored']} macros, "
            f"{summary['domains_restored']} domains, "
            f"{summary['patterns_restored']} patterns, "
            f"observer={'yes' if summary['observer_restored'] else 'no'}"
        )

        return summary

    def _restore_macros(self, engine: Any, cognition_dir: str, summary: dict) -> int:
        """Restore macro stats from macros.json into the Rust engine.

        Since the Rust MacroCache doesn't expose an insert_macro() method,
        we store the macro definitions for reference. The Rust engine will
        re-compile macros organically through MyelinationTracker.
        """
        path = os.path.join(cognition_dir, "macros.json")
        if not os.path.exists(path):
            return 0

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            macros = data.get("macros", [])
            if not macros:
                return 0

            # Store as reference — actual re-insertion happens via
            # MyelinationTracker pattern re-compilation
            self._restored_macros = macros
            logger.info(
                f"Observer: loaded {len(macros)} macro definitions "
                f"(will re-compile organically)"
            )
            return len(macros)
        except Exception as e:
            summary["errors"].append(f"macros: {e}")
            logger.warning(f"Observer: failed to restore macros: {e}")
            return 0

    def _restore_domain_reliability(self, cognition_dir: str, summary: dict) -> int:
        """Restore ParadigmShift failure_counts from domain_reliability.json.

        Restores domain records with counters reset: successes and failures
        are set to 0 so the new agent re-evaluates each domain from scratch.
        """
        path = os.path.join(cognition_dir, "domain_reliability.json")
        if not os.path.exists(path):
            return 0

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            paradigm = self._get_paradigm_engine()
            if paradigm is None:
                return 0

            persisted = data.get("failure_counts", {})
            restored = 0

            for domain, _counts in persisted.items():
                # Reset counters for new agent re-validation
                paradigm.failure_counts[domain] = {
                    "successes": 0,
                    "failures": 0,
                }
                restored += 1

            if restored:
                logger.info(
                    f"Observer: restored {restored} domain records "
                    f"(counters reset to 0)"
                )
            return restored
        except Exception as e:
            summary["errors"].append(f"domain_reliability: {e}")
            logger.warning(f"Observer: failed to restore domain reliability: {e}")
            return 0

    def _restore_query_patterns(self, cognition_dir: str, summary: dict) -> int:
        """Restore MyelinationTracker patterns from query_patterns.json.

        Uses MyelinationTracker.load_patterns() which merges into existing
        state and resets counters (hit_count=0, success_rate=0.5).
        """
        path = os.path.join(cognition_dir, "query_patterns.json")
        if not os.path.exists(path):
            return 0

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            myelination = self._get_myelination()
            if myelination is None:
                return 0

            patterns = data.get("patterns", {})
            if not patterns:
                return 0

            # load_patterns() merges and resets counters automatically
            loaded = myelination.load_patterns(patterns)
            logger.info(
                f"Observer: restored {loaded} query patterns into MyelinationTracker"
            )
            return loaded
        except Exception as e:
            summary["errors"].append(f"query_patterns: {e}")
            logger.warning(f"Observer: failed to restore query patterns: {e}")
            return 0

    def _restore_observer_snapshot(self, cognition_dir: str, summary: dict) -> bool:
        """Restore observer context from observer_snapshot.json.

        Restores configuration but keeps query_history empty (new agent
        must build its own history from scratch).
        """
        path = os.path.join(cognition_dir, "observer_snapshot.json")
        if not os.path.exists(path):
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                snapshot = json.load(f)

            # Restore configuration thresholds (may have been tuned)
            if "confidence_crisis_threshold" in snapshot:
                self.confidence_crisis_threshold = snapshot["confidence_crisis_threshold"]
            if "crisis_streak_length" in snapshot:
                self.crisis_streak_length = snapshot["crisis_streak_length"]

            # Restore last_model_name for continuity
            if "last_model_name" in snapshot:
                self._last_model_name = snapshot["last_model_name"]

            # Do NOT restore query_history — new agent builds fresh history
            logger.info(
                f"Observer: snapshot restored "
                f"(threshold={self.confidence_crisis_threshold}, "
                f"streak={self.crisis_streak_length})"
            )
            return True
        except Exception as e:
            summary["errors"].append(f"observer_snapshot: {e}")
            logger.warning(f"Observer: failed to restore snapshot: {e}")
            return False

    # ── Singleton accessors (lazy imports) ───────────────────────────────

    @staticmethod
    def _get_paradigm_engine() -> Any:
        """Get ParadigmShiftEngine singleton via lazy import."""
        try:
            from evolution.paradigm_shift import ParadigmShiftEngine
            # The server.py module maintains its own singleton;
            # if we can access it, prefer that. Otherwise create local.
            try:
                import importlib
                server = importlib.import_module("mcp_server.server")
                getter = getattr(server, "_get_paradigm_engine", None)
                if getter:
                    return getter()
            except (ImportError, AttributeError, SystemExit):
                pass
            return ParadigmShiftEngine()
        except (ImportError, SystemExit):
            return None

    @staticmethod
    def _get_myelination() -> Any:
        """Get MyelinationTracker singleton via lazy import."""
        try:
            from evolution.myelination import MyelinationTracker
            try:
                import importlib
                server = importlib.import_module("mcp_server.server")
                getter = getattr(server, "_get_myelination", None)
                if getter:
                    return getter()
            except (ImportError, AttributeError, SystemExit):
                pass
            return MyelinationTracker()
        except (ImportError, SystemExit):
            return None

    # ── Observation summary ──────────────────────────────────────────────

    def get_observation_summary(self) -> dict:
        """Get a summary of the observer's current state."""
        if not self.query_history:
            return {"status": "no_data", "queries_observed": 0}

        confidences = [q.get("confidence", 0) for q in self.query_history]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        return {
            "status": "monitoring",
            "queries_observed": len(self.query_history),
            "avg_confidence": round(avg_confidence, 3),
            "min_confidence": round(min(confidences), 3) if confidences else 0,
            "max_confidence": round(max(confidences), 3) if confidences else 0,
            "last_model_name": self._last_model_name,
        }
