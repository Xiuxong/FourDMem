"""Dream Pruner — Ebbinghaus Decay Scheduler

Periodically calls MemoryCore.dream_prune() to apply Ebbinghaus forgetting
curve decay to L1 graph nodes. Memories that haven't been accessed in a
long time and have low utility are pruned.

L2/L3 memories and high-utility (≥0.7) memories are immune to pruning.
"""

import json
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class DreamPruner:
    """Periodic dream-pruning scheduler for the L1 graph.

    Usage:
        pruner = DreamPruner(decay_threshold=90, utility_floor=0.7)
        report = pruner.run_cycle(engine)
    """

    def __init__(
        self,
        decay_threshold: int = 90,
        utility_floor: float = 0.7,
    ):
        """Initialize the dream pruner.

        Args:
            decay_threshold: Tick delta beyond which nodes are pruned
                regardless of shelf life.
            utility_floor: Minimum utility score for immunity from pruning.
        """
        self.decay_threshold = decay_threshold
        self.utility_floor = utility_floor

    def run_cycle(self, engine: Any) -> dict:
        """Run one dream-pruning cycle.

        Args:
            engine: FourDMemEngine instance (py-bindings).

        Returns:
            Dream report dict with pruned/preserved counts.
        """
        try:
            result = engine.dream_prune(self.decay_threshold, self.utility_floor)
            if isinstance(result, str):
                report = json.loads(result)
            else:
                report = result

            logger.info(
                f"Dream cycle: pruned={report.get('pruned', 0)}, "
                f"preserved={report.get('preserved', 0)}, "
                f"tick={report.get('current_tick', 0)}"
            )

            return report

        except Exception as e:
            logger.error(f"Dream cycle failed: {e}")
            return {"pruned": 0, "preserved": 0, "error": str(e)}

    def run_periodic(self, engine: Any, interval_ticks: int = 10) -> None:
        """Advance tick and run dream pruning every `interval_ticks`.

        This is a simple synchronous scheduler. For production, use
        apscheduler or asyncio.

        Args:
            engine: FourDMemEngine instance.
            interval_ticks: Run dream pruning every N ticks.
        """
        tick = engine.advance_tick()
        if tick % interval_ticks == 0:
            self.run_cycle(engine)
