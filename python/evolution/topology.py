"""Topological Monitor — Phase transition detection via TDA.

Monitors L1 graph topology using Betti numbers computed by the Rust
TDA engine (graph-core/tda.rs). When the graph reaches critical
complexity, signals a "phase transition" that may trigger cross-domain
analogy search.

Implements T-10.2: 引入拓扑数据分析 (持续同调)。

Betti numbers:
- β₀ (Betti-0): Connected components — measures graph fragmentation
- β₁ (Betti-1): Independent cycles — measures structural complexity

Phase transition signals:
- β₁ > 45: Critical cycle complexity (per RIF-SQCE.md)
- Density > 0.15: High interconnection
- Isolated ratio > 0.3: Poor connectivity
"""

import json
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class TopologicalMonitor:
    """Monitors L1 graph topology for phase transitions via Rust TDA.

    Args:
        betti_1_threshold: Betti-1 critical threshold (default: 45).
        density_threshold: Density threshold for high-density signal.
        isolated_ratio_threshold: Isolation threshold for poor connectivity.
    """

    def __init__(
        self,
        betti_1_threshold: int = 45,
        density_threshold: float = 0.15,
        isolated_ratio_threshold: float = 0.3,
    ):
        self.betti_1_threshold = betti_1_threshold
        self.density_threshold = density_threshold
        self.isolated_ratio_threshold = isolated_ratio_threshold
        self._history: list[dict] = []

    def compute_metrics(self, engine: Any) -> dict:
        """Compute topological metrics using Rust TDA engine.

        Args:
            engine: FourDMemEngine instance (PyO3 bindings).

        Returns:
            Dict with betti_0, betti_1, density, clustering_coefficient, etc.
        """
        try:
            raw = engine.compute_topology()
            metrics = json.loads(raw) if isinstance(raw, str) else raw
            metrics["status"] = "computed"
            return metrics
        except AttributeError:
            # Fallback: engine doesn't have compute_topology yet
            return self._fallback_metrics(engine)
        except Exception as e:
            return {"status": f"error: {e}"}

    def check_phase_transition(self, engine: Any) -> dict | None:
        """Check if the graph is approaching a phase transition.

        Uses Rust TDA to compute Betti numbers and detect critical thresholds.

        Args:
            engine: FourDMemEngine instance.

        Returns:
            Phase transition signal dict if thresholds crossed, None otherwise.
        """
        try:
            raw = engine.check_phase_transition(self.betti_1_threshold)
            signal = json.loads(raw) if isinstance(raw, str) else raw

            # Record in history
            if "metrics" in signal:
                self._history.append(signal["metrics"])
                if len(self._history) > 20:
                    self._history = self._history[-10:]

            if signal.get("triggered", False):
                logger.warning(
                    f"Phase transition detected! Signals: {signal.get('signals', [])}"
                )
                return {
                    "event": "phase_transition_signal",
                    "metrics": signal.get("metrics", {}),
                    "signals": signal.get("signals", []),
                }

            return None

        except AttributeError:
            # Fallback for engines without TDA
            return self._check_phase_fallback(engine)
        except Exception as e:
            logger.error(f"Phase transition check failed: {e}")
            return None

    def get_status(self) -> dict:
        """Get current monitoring status."""
        if not self._history:
            return {"status": "no_data", "history_length": 0}

        latest = self._history[-1]
        return {
            "status": "monitoring",
            "history_length": len(self._history),
            "latest_metrics": latest,
            "betti_1_threshold": self.betti_1_threshold,
            "density_threshold": self.density_threshold,
        }

    def _fallback_metrics(self, engine: Any) -> dict:
        """Fallback metrics when Rust TDA is unavailable."""
        metrics = {
            "node_count": 0,
            "edge_count": 0,
            "betti_0": 0,
            "betti_1": 0,
            "density": 0.0,
            "avg_degree": 0.0,
            "isolated_ratio": 0.0,
            "clustering_coefficient": 0.0,
            "status": "fallback",
        }

        try:
            raw = engine.wake_up()
            stats = json.loads(raw) if isinstance(raw, str) else raw
            mem_stats = stats.get("memory_stats", stats)

            node_count = mem_stats.get("l1_nodes", mem_stats.get("l1_node_count", 0))
            edge_count = mem_stats.get("l1_edges", mem_stats.get("l1_edge_count", 0))

            metrics["node_count"] = node_count
            metrics["edge_count"] = edge_count

            if node_count > 1:
                max_edges = node_count * (node_count - 1) / 2
                metrics["density"] = round(edge_count / max_edges, 4) if max_edges > 0 else 0
                metrics["avg_degree"] = round(2 * edge_count / node_count, 2)
                # Cyclomatic approximation: β₁ = E - V + 1 (assume single component)
                metrics["betti_0"] = 1
                metrics["betti_1"] = max(0, edge_count - node_count + 1)

        except Exception as e:
            metrics["status"] = f"error: {e}"

        return metrics

    def _check_phase_fallback(self, engine: Any) -> dict | None:
        """Fallback phase check when Rust TDA is unavailable."""
        metrics = self._fallback_metrics(engine)
        signals = []

        if metrics.get("betti_1", 0) >= self.betti_1_threshold:
            signals.append(f"betti_1_critical: {metrics['betti_1']} >= {self.betti_1_threshold}")

        if metrics.get("density", 0) > self.density_threshold:
            signals.append(f"high_density: {metrics['density']:.4f} > {self.density_threshold}")

        if signals:
            return {
                "event": "phase_transition_signal",
                "metrics": metrics,
                "signals": signals,
            }

        return None
