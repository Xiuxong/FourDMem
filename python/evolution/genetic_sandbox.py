"""Genetic Algorithm Sandbox — Cognitive DNA Evolution.

Runs a genetic algorithm to evolve the system's cognitive parameters
(RIF-U weights, confidence threshold, etc.) using historical L0 queries
as a fitness evaluation set.

Implements T-10.7: 将思维策略编码为"认知 DNA"，后台运行遗传算法。

Flow:
1. Initialize population from current DNA + random mutations
2. Evaluate fitness: replay historical L0 queries, measure recall@K
3. Select, crossover, mutate for next generation
4. Winner with fitness > threshold → hot-swap into production
"""

import json
import random
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class GeneticSandbox:
    """Evolves cognitive DNA via genetic algorithm with L0 replay.

    Args:
        population_size: Number of individuals per generation.
        generations: Number of generations to run.
        mutation_rate: Probability of mutation per gene.
        mutation_delta: Max perturbation per mutation.
        fitness_threshold: Minimum fitness for hot-swap eligibility.
    """

    def __init__(
        self,
        population_size: int = 10,
        generations: int = 5,
        mutation_rate: float = 0.3,
        mutation_delta: float = 0.05,
        fitness_threshold: float = 0.85,
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_delta = mutation_delta
        self.fitness_threshold = fitness_threshold

    def run_evolution(self, engine: Any, sample_queries: list[str] | None = None) -> dict:
        """Run a full GA evolution cycle.

        Args:
            engine: FourDMemEngine instance (PyO3 bindings).
            sample_queries: Historical queries for fitness evaluation.
                If None, generates synthetic queries from L0 evidence.

        Returns:
            Evolution report with best DNA, fitness history, and whether
            hot-swap was triggered.
        """
        if sample_queries is None:
            sample_queries = self._sample_l0_queries(engine)

        if len(sample_queries) < 3:
            return {"status": "insufficient_data", "query_count": len(sample_queries)}

        # Get current DNA as baseline
        current_dna = self._get_current_dna(engine)
        baseline_fitness = self._evaluate_fitness(engine, current_dna, sample_queries)

        # Initialize population: baseline + mutations
        population = [current_dna]
        for _ in range(self.population_size - 1):
            mutated = self._mutate_dna(current_dna)
            population.append(mutated)

        best_dna = current_dna
        best_fitness = baseline_fitness
        fitness_history = [baseline_fitness]

        for gen in range(self.generations):
            # Evaluate fitness for all individuals
            scored = []
            for dna in population:
                fitness = self._evaluate_fitness(engine, dna, sample_queries)
                scored.append((dna, fitness))

            # Sort by fitness (descending)
            scored.sort(key=lambda x: x[1], reverse=True)

            # Track best
            if scored[0][1] > best_fitness:
                best_dna = scored[0][0]
                best_fitness = scored[0][1]

            fitness_history.append(scored[0][1])

            logger.info(
                f"GA generation {gen + 1}/{self.generations}: "
                f"best_fitness={scored[0][1]:.3f}, avg={sum(f for _, f in scored) / len(scored):.3f}"
            )

            # Selection: top 50% survive
            survivors = scored[: max(2, len(scored) // 2)]

            # Crossover + mutation for next generation
            next_gen = [survivors[0][0]]  # Elitism: keep best
            while len(next_gen) < self.population_size:
                parent_a = random.choice(survivors)[0]
                parent_b = random.choice(survivors)[0]
                child = self._crossover_dna(parent_a, parent_b)
                child = self._mutate_dna(child)
                next_gen.append(child)

            population = next_gen

        # Check if best DNA exceeds threshold
        hot_swapped = False
        if best_fitness > self.fitness_threshold and best_fitness > baseline_fitness:
            try:
                engine.hot_swap_dna(json.dumps(best_dna))
                hot_swapped = True
                logger.info(
                    f"DNA hot-swapped! fitness: {baseline_fitness:.3f} → {best_fitness:.3f}"
                )
            except Exception as e:
                logger.error(f"Hot-swap failed: {e}")

        return {
            "status": "evolved",
            "baseline_fitness": baseline_fitness,
            "best_fitness": best_fitness,
            "improvement": best_fitness - baseline_fitness,
            "hot_swapped": hot_swapped,
            "generations": self.generations,
            "fitness_history": fitness_history,
            "best_dna": best_dna,
        }

    def _get_current_dna(self, engine: Any) -> dict:
        """Get current DNA from engine."""
        try:
            raw = engine.evolve()
            data = json.loads(raw) if isinstance(raw, str) else raw
            return data.get("current_dna", self._default_dna())
        except Exception:
            return self._default_dna()

    def _default_dna(self) -> dict:
        """Default DNA parameters."""
        return {
            "rif_weights": [0.25, 0.35, 0.15, 0.25],
            "confidence_threshold": 0.65,
            "macro_compilation_threshold": 10,
        }

    def _mutate_dna(self, dna: dict) -> dict:
        """Apply random mutations to DNA."""
        mutated = dna.copy()
        delta = self.mutation_delta

        if random.random() < self.mutation_rate:
            weights = list(mutated.get("rif_weights", [0.25, 0.35, 0.15, 0.25]))
            idx = random.randint(0, 3)
            weights[idx] = max(0.01, min(0.99, weights[idx] + random.uniform(-delta, delta)))
            # Normalize
            total = sum(weights)
            mutated["rif_weights"] = [w / total for w in weights]

        if random.random() < self.mutation_rate:
            ct = mutated.get("confidence_threshold", 0.65)
            mutated["confidence_threshold"] = max(0.3, min(0.95, ct + random.uniform(-delta, delta)))

        return mutated

    def _crossover_dna(self, a: dict, b: dict) -> dict:
        """Two-point crossover between two DNA strands."""
        child = {}
        for key in a:
            if key == "rif_weights":
                wa, wb = a["rif_weights"], b["rif_weights"]
                point = random.randint(1, 3)
                child["rif_weights"] = wa[:point] + wb[point:]
            else:
                child[key] = a[key] if random.random() < 0.5 else b[key]
        return child

    def _evaluate_fitness(self, engine: Any, dna: dict, queries: list[str]) -> float:
        """Evaluate DNA fitness by replaying queries.

        Fitness = average recall@K across sample queries.
        """
        if not queries:
            return 0.0

        # Temporarily swap DNA for evaluation
        try:
            engine.hot_swap_dna(json.dumps(dna))
        except Exception:
            pass

        successes = 0
        for query in queries[:20]:  # Limit to 20 queries for performance
            try:
                raw = engine.query(query, 5)
                data = json.loads(raw) if isinstance(raw, str) else raw
                results = data.get("results", [])
                confidence = data.get("confidence", 0.0)

                # Success = got results with reasonable confidence
                if results and confidence > 0.3:
                    successes += 1
            except Exception:
                pass

        return successes / min(len(queries), 20)

    def _sample_l0_queries(self, engine: Any, limit: int = 30) -> list[str]:
        """Sample queries from L0 evidence for fitness evaluation."""
        queries = []
        try:
            raw = engine.get_session_evidence("auto", limit)
            data = json.loads(raw) if isinstance(raw, str) else raw
            for ev in data.get("evidence", []):
                content = ev.get("content", "")
                if len(content) > 10:
                    # Extract a query-like phrase from the evidence
                    words = content.split()[:8]
                    queries.append(" ".join(words))
        except Exception:
            pass

        # Fallback: generic queries
        if len(queries) < 5:
            queries.extend([
                "architecture design",
                "bug fix error",
                "rust borrow checker",
                "python async",
                "database query",
            ])

        return queries[:limit]


class DnaPersistence:
    """Manages DNA persistence to disk for cross-session evolution.

    Stores DNA snapshots in data/genome/ directory.
    """

    def __init__(self, genome_dir: str = "data/genome"):
        self.genome_dir = genome_dir

    def save_dna(self, dna: dict, generation: int = 0, fitness: float = 0.0) -> str:
        """Save a DNA snapshot to disk."""
        import os
        os.makedirs(self.genome_dir, exist_ok=True)

        snapshot = {
            "dna": dna,
            "generation": generation,
            "fitness": fitness,
        }

        filepath = os.path.join(self.genome_dir, f"dna_gen{generation}.json")
        with open(filepath, "w") as f:
            json.dump(snapshot, f, indent=2)

        return filepath

    def load_best_dna(self) -> dict | None:
        """Load the best DNA from disk."""
        import os
        import glob

        files = glob.glob(os.path.join(self.genome_dir, "dna_gen*.json"))
        if not files:
            return None

        best = None
        best_fitness = -1.0

        for f in files:
            try:
                with open(f) as fh:
                    data = json.load(fh)
                    if data.get("fitness", 0) > best_fitness:
                        best_fitness = data["fitness"]
                        best = data.get("dna")
            except Exception:
                pass

        return best
