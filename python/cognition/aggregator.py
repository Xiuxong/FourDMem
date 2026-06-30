"""L1→L2 Auto-Aggregator — Weighted Scenario Generation.

When L1 facts under a topic accumulate weighted score >= threshold,
automatically generates L2 scenario blocks from the cluster.

Weight system:
- Conversation facts (decisions, architecture): weight 2.0
- Tool facts (errors, fixes, successes): weight 1.0
- Default: weight 1.0

Threshold: 5 weighted points (e.g., 3 conversation facts OR 5 tool facts
OR any mix summing to >= 5).

Flow:
1. Group L1 facts by topic/keyword similarity
2. When a group reaches weighted threshold, synthesize L2 scenario
3. Write to data/vault/scenarios/ with proper frontmatter
4. Record source_l1_refs for provenance chain
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


# Source weights for aggregation
SOURCE_WEIGHTS = {
    "conversation": 2.0,  # Decisions, architecture, preferences
    "tool": 1.0,          # Errors, fixes, successes
    "default": 1.0,
}


class AutoAggregator:
    """Automatically aggregates L1 facts into L2 scenario blocks.

    Args:
        aggregation_threshold: Weighted score threshold to trigger aggregation.
    """

    def __init__(
        self,
        aggregation_threshold: float = 5.0,
    ):
        self.aggregation_threshold = aggregation_threshold
        self._cluster_cache: dict[str, list[dict]] = {}

    def record_fact(self, fact: dict, source: str = "default") -> dict | None:
        """Record an L1 fact and check if aggregation should trigger.

        Args:
            fact: L1 fact dict with at least 'label', optional 'tags'.
            source: Fact source — "conversation", "tool", or "default".

        Returns:
            Aggregation result if triggered, None otherwise.
        """
        topic = self._extract_topic(fact.get("label", ""))
        if not topic:
            return None

        if topic not in self._cluster_cache:
            self._cluster_cache[topic] = []

        fact["_source"] = source
        self._cluster_cache[topic].append(fact)

        # Weighted aggregation check
        weighted_score = sum(
            SOURCE_WEIGHTS.get(f.get("_source", "default"), 1.0)
            for f in self._cluster_cache[topic]
        )

        if weighted_score >= self.aggregation_threshold:
            result = self._aggregate_cluster(topic, self._cluster_cache[topic])
            self._cluster_cache[topic] = []  # Reset after aggregation
            return result

        return None

    def force_aggregate(self, engine: Any, topic: str) -> dict:
        """Force aggregation for a specific topic by querying L1 facts.

        Args:
            engine: FourDMemEngine instance.
            topic: Topic to aggregate.

        Returns:
            Aggregation result or error.
        """
        try:
            raw = engine.query(topic, 20)
            data = json.loads(raw) if isinstance(raw, str) else raw
            l1_facts = [r for r in data.get("results", []) if r.get("layer") in ("L1", 1, "1")]

            if len(l1_facts) < 2:
                return {"status": "insufficient_facts", "count": len(l1_facts)}

            return self._aggregate_cluster(topic, l1_facts)

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _extract_topic(self, label: str) -> str:
        """Extract a topic keyword from a fact label.

        Priority:
        1. English/technical terms (SQLite, Rust, FourDMem, cargo, etc.)
        2. Chinese noun phrases (skip common verbs/adverbs)
        """
        cleaned = label.strip()
        if not cleaned:
            return ""

        # Remove common English prefixes
        for prefix in ["the ", "a ", "an ", "this ", "that "]:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):]

        # Priority 1: Extract English/technical terms
        english_terms = re.findall(r'[A-Za-z][A-Za-z0-9_\-\.]{1,}', cleaned)
        _skip_words = {"is", "are", "was", "were", "has", "have", "had", "the", "and", "or", "but", "not", "for", "with", "from", "this", "that"}
        english_terms = [t for t in english_terms if t.lower() not in _skip_words]
        if english_terms:
            return english_terms[0].lower()

        # Priority 2: Chinese noun phrases (skip verbs/adverbs)
        _skip_chinese = {"是", "有", "在", "使用", "用", "采用", "基于", "负责", "处理",
                         "管理", "控制", "实现", "支持", "包含", "提供", "导致", "造成",
                         "需要", "必须", "可以", "能够", "会", "应该", "决定", "选择",
                         "记住", "注意", "确保", "修复", "错误", "优于", "劣于"}
        chinese_phrases = re.findall(r'[\u4e00-\u9fff]+', cleaned)
        for phrase in chinese_phrases:
            if phrase not in _skip_chinese and len(phrase) >= 2:
                return phrase[:4]

        # Fallback: first 3 words
        words = cleaned.split()[:3]
        if not words:
            return ""
        return " ".join(words).lower()

    def _aggregate_cluster(self, topic: str, facts: list[dict]) -> dict:
        """Aggregate a cluster of L1 facts into an L2 scenario.

        Args:
            topic: Topic identifier.
            facts: List of L1 fact dicts.

        Returns:
            Aggregation result with scenario content.
        """
        fact_texts = [
            f"- {f.get('label', f.get('content', str(f)))}" for f in facts
        ]
        facts_str = "\n".join(fact_texts)

        scenario_data = self._rule_based_aggregate(topic, facts)

        # Build L2 frontmatter
        l1_refs = [
            str(f.get("id", f.get("node_index", "")))
            for f in facts
            if f.get("id") or f.get("node_index")
        ]

        result = {
            "status": "aggregated",
            "topic": topic,
            "fact_count": len(facts),
            "title": scenario_data.get("title", f"Scenario: {topic}"),
            "content": scenario_data.get("content", facts_str),
            "conditions": scenario_data.get("conditions", [topic]),
            "tags": scenario_data.get("tags", ["auto_aggregated"]),
            "l1_refs": l1_refs,
            "key_insight": scenario_data.get("key_insight", ""),
        }

        logger.info(
            f"Aggregated {len(facts)} L1 facts into L2 scenario: {result['title']}"
        )

        return result

    def _rule_based_aggregate(self, topic: str, facts: list[dict]) -> dict:
        """Rule-based aggregation with source-aware synthesis.

        Separates conversation facts (decisions) from tool facts (experiences)
        and generates structured scenario content.
        """
        labels = [f.get("label", f.get("content", "")) for f in facts]

        # Separate by source
        conversation_facts = [f for f in facts if f.get("_source") == "conversation"]
        tool_facts = [f for f in facts if f.get("_source") == "tool"]
        other_facts = [f for f in facts if f.get("_source") not in ("conversation", "tool")]

        content_lines = [f"## {topic}\n"]

        if conversation_facts:
            content_lines.append("### 决策与知识")
            for f in conversation_facts:
                content_lines.append(f"- {f.get('label', f.get('content', ''))}")
            content_lines.append("")

        if tool_facts:
            content_lines.append("### 工具经验")
            for f in tool_facts:
                content_lines.append(f"- {f.get('label', f.get('content', ''))}")
            content_lines.append("")

        if other_facts:
            content_lines.append("### 相关事实")
            for f in other_facts:
                content_lines.append(f"- {f.get('label', f.get('content', ''))}")
            content_lines.append("")

        # Key insight
        insight_parts = []
        if conversation_facts:
            insight_parts.append(f"{len(conversation_facts)} 条决策知识")
        if tool_facts:
            insight_parts.append(f"{len(tool_facts)} 条工具经验")
        key_insight = f"{topic}：包含 {' + '.join(insight_parts)}"

        # Extract common tags
        all_tags = set()
        for f in facts:
            for tag in f.get("tags", []):
                all_tags.add(tag)
        if not all_tags:
            all_tags = {topic.replace(" ", "_")}

        return {
            "title": f"{topic} — 知识场景",
            "content": "\n".join(content_lines),
            "conditions": [topic],
            "tags": list(all_tags)[:5],
            "key_insight": key_insight,
        }

    def get_pending_clusters(self) -> dict:
        """Get info about clusters that haven't reached threshold yet.

        Returns:
            Dict of topic -> {"count": int, "weighted_score": float, "threshold": float, "remaining": float}.
        """
        result = {}
        for topic, facts in self._cluster_cache.items():
            if not facts:
                continue
            weighted = sum(
                SOURCE_WEIGHTS.get(f.get("_source", "default"), 1.0)
                for f in facts
            )
            result[topic] = {
                "count": len(facts),
                "weighted_score": round(weighted, 1),
                "threshold": self.aggregation_threshold,
                "remaining": round(max(0, self.aggregation_threshold - weighted), 1),
            }
        return result

    def get_stats(self) -> dict:
        """Get aggregator statistics."""
        total_facts = sum(len(f) for f in self._cluster_cache.values())
        return {
            "active_clusters": len(self._cluster_cache),
            "total_pending_facts": total_facts,
            "aggregation_threshold": self.aggregation_threshold,
        }
