"""Cross-Domain Analogy Engine — Emergent Insights.

When the topological monitor detects a phase transition (high Betti-1),
this engine extracts high-density subgraph themes and uses rule-based
analysis to find structural isomorphisms between unrelated knowledge clusters.

Implements T-10.3: cross-domain structural analogy detection.

Flow:
1. TopologicalMonitor signals phase transition
2. Extract top N subgraph clusters by density
3. Rule-based analysis compares pairs for structural analogies
4. Generate cross-domain L2 scenario blocks
"""

import json
from typing import Any


try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class AnalogyEngine:
    """Detects cross-domain analogies between knowledge clusters.

    Args:
        min_cluster_size: Minimum facts per cluster to consider.
    """

    def __init__(self, min_cluster_size: int = 3):
        self.min_cluster_size = min_cluster_size

    def find_analogies(
        self,
        engine: Any,
        phase_signal: dict | None = None,
    ) -> list[dict]:
        """Find cross-domain analogies when phase transition is detected.

        Args:
            engine: FourDMemEngine instance.
            phase_signal: Signal from TopologicalMonitor (optional).

        Returns:
            List of analogy dicts.
        """
        # Extract clusters from the graph
        clusters = self._extract_clusters(engine)

        if len(clusters) < 2:
            return []

        analogies = []

        # Compare pairs of clusters from different domains
        cluster_items = list(clusters.items())
        for i in range(len(cluster_items)):
            for j in range(i + 1, len(cluster_items)):
                domain_a, facts_a = cluster_items[i]
                domain_b, facts_b = cluster_items[j]

                if domain_a == domain_b:
                    continue

                analogy = self._analyze_pair(domain_a, facts_a, domain_b, facts_b)
                if analogy and analogy.get("confidence", 0) > 0.5:
                    analogies.append(analogy)

        return analogies

    def _extract_clusters(self, engine: Any) -> dict[str, list[str]]:
        """Extract topic clusters from L1 graph.

        Returns:
            Dict of topic -> list of fact labels.
        """
        clusters: dict[str, list[str]] = {}

        try:
            # Query for diverse topics
            topics = [
                "architecture", "design pattern", "algorithm",
                "database", "frontend", "backend", "testing",
                "security", "performance", "api",
            ]

            for topic in topics:
                raw = engine.query(topic, 10)
                data = json.loads(raw) if isinstance(raw, str) else raw
                results = data.get("results", [])

                l1_facts = [
                    r.get("content", r.get("label", ""))
                    for r in results
                    if r.get("layer") == "L1"
                ]

                if len(l1_facts) >= self.min_cluster_size:
                    clusters[topic] = l1_facts

        except Exception as e:
            logger.error(f"Cluster extraction failed: {e}")

        return clusters

    def _analyze_pair(
        self,
        domain_a: str,
        facts_a: list[str],
        domain_b: str,
        facts_b: list[str],
    ) -> dict | None:
        """Return a cognition task for the Agent to analyze structural analogies.

        The Agent uses its LLM to find deep structural isomorphisms between
        the two clusters, then calls synthesize_l2() to store the result.
        """
        return {
            "status": "cognition_task",
            "type": "cross_domain_analogy",
            "domain_a": domain_a,
            "domain_b": domain_b,
            "facts_a": facts_a[:10],
            "facts_b": facts_b[:10],
            "instruction": (
                f"Find structural analogies between '{domain_a}' and '{domain_b}' "
                "knowledge clusters. Look for isomorphic patterns, not just shared "
                "keywords. If a meaningful analogy exists, call "
                "synthesize_l2(title, fact_ids, synthesis) to create an L2 scenario."
            ),
        }

    def generate_scenario(self, analogy: dict) -> dict:
        """Generate an L2 scenario block from an analogy.

        Args:
            analogy: Analogy dict from find_analogies().

        Returns:
            Scenario dict ready for L2 write.
        """
        title = analogy.get("analogy_title", "Cross-Domain Analogy")

        content = f"""# Cross-Domain Analogy: {title}

## Domains
- **{analogy.get('domain_a', 'A')}**
- **{analogy.get('domain_b', 'B')}**

## Structural Similarity
{analogy.get('structural_similarity', 'N/A')}

## Conceptual Mappings
"""
        for mapping in analogy.get("mappings", []):
            content += f"- **{mapping.get('from_a', '?')}** in {analogy.get('domain_a', 'A')} "
            content += f"≈ **{mapping.get('to_b', '?')}** in {analogy.get('domain_b', 'B')}: "
            content += f"{mapping.get('why', '')}\n"

        content += f"\n## Key Insight\n{analogy.get('insight', 'N/A')}\n"

        return {
            "title": f"Analogy: {title}",
            "content": content,
            "conditions": [analogy.get("domain_a", ""), analogy.get("domain_b", "")],
            "tags": ["cross_domain_analogy", analogy.get("domain_a", ""), analogy.get("domain_b", "")],
            "confidence": analogy.get("confidence", 0.0),
        }
