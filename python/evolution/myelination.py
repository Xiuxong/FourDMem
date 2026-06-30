"""Myelination Tracker — Cognitive Macro Compilation (Python side).

Tracks query patterns and their retrieval success rates. When a pattern
is queried frequently (>= threshold) with high success rate (>= 0.9),
it's "compiled" into a cognitive macro — a fast-path shortcut.

This is the Python-side orchestrator for the Rust MacroCache.
The Rust side (evolution-core/macro_cache.rs) stores the actual macros;
this module tracks patterns at the MCP layer and triggers compilation.
"""

import json
import hashlib
from typing import Any


class MyelinationTracker:
    """Tracks query patterns for cognitive macro compilation."""

    def __init__(self, compilation_threshold: int = 10, success_rate_required: float = 0.9):
        self.compilation_threshold = compilation_threshold
        self.success_rate_required = success_rate_required
        # pattern_key -> {"hits": int, "successes": int, "results_cache": list}
        self._patterns: dict[str, dict] = {}

    def _pattern_key(self, query: str) -> str:
        """Compute a normalized pattern key from a query."""
        normalized = query.lower().strip()
        # Use first 100 chars + hash for grouping similar queries
        key_input = normalized[:100]
        return hashlib.md5(key_input.encode()).hexdigest()[:16]

    def record_query(self, query: str, result_count: int, confidence: float) -> dict | None:
        """Record a query and its retrieval result.

        Args:
            query: The search query.
            result_count: Number of results returned.
            confidence: Confidence score from the retrieval.

        Returns:
            Compilation event if a macro was compiled, None otherwise.
        """
        key = self._pattern_key(query)
        success = result_count > 0 and confidence > 0.3

        if key not in self._patterns:
            self._patterns[key] = {
                "hits": 0,
                "successes": 0,
                "representative_query": query[:200],
                "last_confidence": 0.0,
            }

        p = self._patterns[key]
        p["hits"] += 1
        if success:
            p["successes"] += 1
        p["last_confidence"] = confidence

        # Check for macro compilation
        if p["hits"] >= self.compilation_threshold:
            success_rate = p["successes"] / p["hits"]
            if success_rate >= self.success_rate_required:
                return {
                    "event": "macro_compiled",
                    "pattern_key": key,
                    "representative_query": p["representative_query"],
                    "hits": p["hits"],
                    "success_rate": round(success_rate, 3),
                    "message": f"Pattern compiled to cognitive macro after {p['hits']} hits ({success_rate:.0%} success)",
                }

        return None

    def get_stats(self) -> dict:
        """Get myelination statistics."""
        total_patterns = len(self._patterns)
        compiling = sum(
            1 for p in self._patterns.values()
            if p["hits"] >= self.compilation_threshold
            and p["successes"] / max(p["hits"], 1) >= self.success_rate_required
        )
        approaching = sum(
            1 for p in self._patterns.values()
            if p["hits"] >= self.compilation_threshold // 2
        )

        return {
            "total_patterns_tracked": total_patterns,
            "compiled_macros": compiling,
            "approaching_compilation": approaching,
            "threshold": self.compilation_threshold,
            "success_rate_required": self.success_rate_required,
        }

    def load_patterns(self, patterns: dict[str, dict]) -> int:
        """Load persisted query patterns into the tracker.

        Merges into existing state — does NOT overwrite patterns already
        known to this agent.  Loaded patterns get their counters reset
        (hit_count=0, success_rate=0.5) so the new agent re-validates them.

        Args:
            patterns: mapping of pattern_hash → {
                "query_template": str,
                "hit_count": int,        (ignored on load — reset to 0)
                "success_rate": float,   (ignored on load — reset to 0.5)
                ...                      (extra fields preserved)
            }

        Returns:
            Number of patterns loaded (skipped ones are already local).
        """
        loaded = 0
        for key, entry in patterns.items():
            if key in self._patterns:
                continue  # don't overwrite live state

            self._patterns[key] = {
                "hits": 0,
                "successes": 0,
                "representative_query": entry.get("query_template", "")[:200],
                "last_confidence": 0.5,
            }
            loaded += 1

        return loaded

    def reset(self):
        """Reset all tracking data."""
        self._patterns.clear()
