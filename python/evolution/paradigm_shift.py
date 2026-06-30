"""Paradigm Shift Engine — Dialectical Synthesis

Monitors assimilation failure rates per domain. When failures exceed
the threshold, triggers a cognitive crisis:
1. Freezes L1 writes in that domain
2. Pulls old L3 rules (thesis) + new L0 evidence (antithesis)
3. Dialectical debate → generates new L3 rules (synthesis)
4. Records the paradigm shift in L2

Implements T-10.4 (failure monitoring) and T-10.5 (dialectical paradigm shift).
"""

from typing import Any


try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class ParadigmShiftEngine:
    """Monitors assimilation failure rates and triggers paradigm shifts.

    When a domain's failure rate exceeds the threshold, the engine
    initiates a dialectical debate between old rules (thesis) and
    new evidence (antithesis) to produce updated rules (synthesis).
    """

    def __init__(self, failure_threshold: float = 0.3):
        self.failure_threshold = failure_threshold
        self.failure_counts: dict[str, dict[str, int]] = {}

    def record_outcome(self, domain: str, success: bool) -> None:
        """Record an execution outcome for a domain.

        Args:
            domain: The knowledge domain (e.g., "rust", "python", "architecture").
            success: Whether the memory-guided action succeeded.
        """
        if domain not in self.failure_counts:
            self.failure_counts[domain] = {"successes": 0, "failures": 0}

        if success:
            self.failure_counts[domain]["successes"] += 1
        else:
            self.failure_counts[domain]["failures"] += 1

    def check_crisis(self, domain: str) -> tuple[bool, float]:
        """Check if a domain is in cognitive crisis.

        Returns:
            (is_crisis, failure_rate) tuple.
        """
        counts = self.failure_counts.get(domain, {"successes": 0, "failures": 0})
        total = counts["successes"] + counts["failures"]
        if total < 3:  # Need minimum data
            return False, 0.0

        rate = counts["failures"] / total
        return rate > self.failure_threshold, rate

    def run_dialectic(
        self,
        engine: Any,
        domain: str,
        old_rule: str,
        new_evidence: list[str],
    ) -> dict:
        """Run a dialectical paradigm shift.

        Args:
            engine: FourDMemEngine instance.
            domain: The domain in crisis.
            old_rule: The current L3 rule (thesis).
            new_evidence: Recent L0 evidence contradicting the rule (antithesis).

        Returns:
            Dict with synthesis, reasoning, and confidence.
        """
        # Return structured task prompt for the Agent's dialectical synthesis.
        # The Agent calls reflect_and_synthesize(domain, thesis, antithesis, synthesis)
        # to submit the actual synthesis.
        evidence_text = "\n".join(f"- {ev}" for ev in new_evidence)
        return {
            "status": "cognition_task",
            "type": "dialectic_synthesis",
            "domain": domain,
            "thesis": old_rule,
            "antithesis": evidence_text,
            "instruction": (
                "A paradigm crisis has been detected in this domain. "
                "Old rule (thesis) no longer matches evidence (antithesis). "
                "Use your LLM reasoning to synthesize a new rule (synthesis) "
                "that resolves the contradiction. Then call "
                "reflect_and_synthesize(domain, thesis, antithesis, synthesis) "
                "to store the result."
            ),
        }

    def monitor_and_shift(
        self,
        engine: Any,
        domain: str,
        old_rule: str,
        new_evidence: list[str],
        writeback_l3: bool = True,
    ) -> dict | None:
        """Check for crisis and trigger paradigm shift if needed.

        Args:
            engine: FourDMemEngine instance.
            domain: The domain in crisis.
            old_rule: The current L3 rule (thesis).
            new_evidence: Recent L0 evidence contradicting the rule.
            writeback_l3: Whether to auto-write synthesis to L3.

        Returns:
            Paradigm shift result if triggered, None otherwise.
        """
        is_crisis, rate = self.check_crisis(domain)
        if not is_crisis:
            return None

        logger.warning(
            f"Cognitive crisis in '{domain}': failure rate={rate:.1%} "
            f"(threshold={self.failure_threshold:.1%})"
        )

        result = self.run_dialectic(engine, domain, old_rule, new_evidence)

        # Reset counters after shift
        self.failure_counts[domain] = {"successes": 0, "failures": 0}

        # Writeback synthesis to L3 if confidence is high enough
        if writeback_l3 and result.get("confidence", 0) > 0.5:
            self._writeback_l3(result, domain)
            self._record_to_l2(result, domain)

        return result

    def _writeback_l3(self, result: dict, domain: str) -> bool:
        """Write paradigm shift synthesis back to L3 persona.

        Updates the persona YAML with the new synthesis rule.
        """
        try:
            import os
            import yaml
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            persona_path = os.path.join(project_root, "data", "vault", "persona", "persona.yaml")

            # Load existing persona
            persona = {}
            if os.path.exists(persona_path):
                with open(persona_path, "r", encoding="utf-8") as f:
                    persona = yaml.safe_load(f) or {}

            # Add paradigm shift to rules
            if "paradigm_shifts" not in persona:
                persona["paradigm_shifts"] = []

            shift_entry = {
                "domain": domain,
                "synthesis": result.get("synthesis", ""),
                "reasoning": result.get("reasoning", ""),
                "confidence": result.get("confidence", 0.0),
                "replaces": result.get("old_rule", ""),
            }
            persona["paradigm_shifts"].append(shift_entry)

            # Update core rules if synthesis is confident
            if result.get("confidence", 0) > 0.7:
                if "rules" not in persona:
                    persona["rules"] = []
                persona["rules"].append({
                    "domain": domain,
                    "rule": result.get("synthesis", ""),
                    "source": "paradigm_shift",
                })

            # Write back
            os.makedirs(os.path.dirname(persona_path), exist_ok=True)
            with open(persona_path, "w", encoding="utf-8") as f:
                yaml.dump(persona, f, default_flow_style=False, allow_unicode=True)

            logger.info(f"Paradigm shift written to L3: {domain}")
            return True

        except Exception as e:
            logger.error(f"L3 writeback failed: {e}")
            return False

    def _record_to_l2(self, result: dict, domain: str) -> bool:
        """Record paradigm shift event as L2 scenario."""
        try:
            import os
            from datetime import datetime, timezone

            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            scenarios_dir = os.path.join(project_root, "data", "vault", "scenarios")
            os.makedirs(scenarios_dir, exist_ok=True)

            slug = f"paradigm_shift_{domain}".replace(" ", "_").lower()
            filepath = os.path.join(scenarios_dir, f"{slug}.md")

            content = f"""# Paradigm Shift: {domain}

## Old Rule (Thesis)
{result.get('old_rule', 'N/A')}

## Synthesis (New Rule)
{result.get('synthesis', 'N/A')}

## Reasoning
{result.get('reasoning', 'N/A')}

## Confidence
{result.get('confidence', 0.0):.0%}
"""

            frontmatter = {
                "title": f"Paradigm Shift: {domain}",
                "type": "paradigm_shift",
                "domain": domain,
                "confidence": result.get("confidence", 0.0),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "tags": ["paradigm_shift", domain],
            }

            with open(filepath, "w", encoding="utf-8") as f:
                f.write("---\n")
                import json
                f.write(json.dumps(frontmatter, indent=2, ensure_ascii=False))
                f.write("\n---\n\n")
                f.write(content)

            logger.info(f"Paradigm shift recorded to L2: {filepath}")
            return True

        except Exception as e:
            logger.error(f"L2 recording failed: {e}")
            return False
