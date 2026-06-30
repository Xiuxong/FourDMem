"""Synthetic recall benchmark for FourDMem.

Injects known facts into the memory system, then queries to measure recall.
This is the simplest benchmark — a prerequisite before running LoCoMo/LongMemEval.

Usage:
    python -m benchmarks.synthetic_recall [--db-path PATH] [--seed 42]
"""

import json
import os
import sys
import time
import random
from typing import Any

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Synthetic dataset ─────────────────────────────────────────────────────────

FACTS = [
    {"label": "Alice works at Google as a senior engineer", "tags": ["alice", "work"], "importance": 0.9},
    {"label": "Bob prefers dark mode and vim keybindings", "tags": ["bob", "preferences"], "importance": 0.7},
    {"label": "The project uses PostgreSQL for ACID compliance", "tags": ["architecture", "database"], "importance": 0.95},
    {"label": "We decided to migrate from REST to GraphQL in Q3", "tags": ["architecture", "api"], "importance": 0.85},
    {"label": "The deployment pipeline runs on GitHub Actions", "tags": ["devops", "ci"], "importance": 0.8},
    {"label": "Carol is the team lead for the frontend squad", "tags": ["carol", "team"], "importance": 0.75},
    {"label": "The auth system uses JWT tokens with 15-minute expiry", "tags": ["auth", "security"], "importance": 0.9},
    {"label": "We had a major outage on 2025-03-15 due to Redis memory overflow", "tags": ["incident", "redis"], "importance": 1.0},
    {"label": "The new API version is v3 and it breaks backward compatibility", "tags": ["api", "versioning"], "importance": 0.85},
    {"label": "David recommends using Rust for performance-critical paths", "tags": ["david", "architecture"], "importance": 0.8},
    {"label": "The test suite takes 45 minutes and needs parallelization", "tags": ["testing", "performance"], "importance": 0.7},
    {"label": "We use Tailwind CSS for all new components", "tags": ["frontend", "css"], "importance": 0.75},
    {"label": "The database migration script is in scripts/migrate_v2.sql", "tags": ["database", "migration"], "importance": 0.8},
    {"label": "Eve joined the team on 2025-06-01 as a junior developer", "tags": ["eve", "team"], "importance": 0.6},
    {"label": "The monitoring stack uses Prometheus and Grafana", "tags": ["devops", "monitoring"], "importance": 0.8},
    {"label": "We agreed to use conventional commits for all repositories", "tags": ["process", "git"], "importance": 0.7},
    {"label": "The caching layer reduces API latency by 60 percent", "tags": ["performance", "caching"], "importance": 0.85},
    {"label": "Frank is responsible for the data pipeline", "tags": ["frank", "team"], "importance": 0.7},
    {"label": "The CI pipeline deploys to staging automatically on merge to main", "tags": ["devops", "deployment"], "importance": 0.8},
    {"label": "We switched from MongoDB to PostgreSQL in January 2025", "tags": ["database", "migration"], "importance": 0.9},
]

QUESTIONS = [
    {"query": "Where does Alice work?", "expected_keywords": ["google", "senior engineer"], "difficulty": "easy"},
    {"query": "What are Bob's preferences?", "expected_keywords": ["dark mode", "vim"], "difficulty": "easy"},
    {"query": "Why did we choose PostgreSQL?", "expected_keywords": ["acid", "compliance"], "difficulty": "medium"},
    {"query": "When did we migrate to GraphQL?", "expected_keywords": ["q3"], "difficulty": "medium"},
    {"query": "What caused the outage in March?", "expected_keywords": ["redis", "memory overflow"], "difficulty": "hard"},
    {"query": "Who is the frontend team lead?", "expected_keywords": ["carol"], "difficulty": "easy"},
    {"query": "What auth mechanism do we use?", "expected_keywords": ["jwt", "15-minute"], "difficulty": "medium"},
    {"query": "What is the new API version?", "expected_keywords": ["v3", "backward compatibility"], "difficulty": "easy"},
    {"query": "Who recommends Rust?", "expected_keywords": ["david"], "difficulty": "easy"},
    {"query": "How long does the test suite take?", "expected_keywords": ["45 minutes"], "difficulty": "medium"},
    {"query": "What CSS framework do we use?", "expected_keywords": ["tailwind"], "difficulty": "easy"},
    {"query": "Where is the migration script?", "expected_keywords": ["scripts/migrate_v2.sql"], "difficulty": "hard"},
    {"query": "When did Eve join?", "expected_keywords": ["2025-06-01", "junior"], "difficulty": "medium"},
    {"query": "What monitoring tools do we use?", "expected_keywords": ["prometheus", "grafana"], "difficulty": "easy"},
    {"query": "How much does caching reduce latency?", "expected_keywords": ["60 percent"], "difficulty": "medium"},
    {"query": "Who handles the data pipeline?", "expected_keywords": ["frank"], "difficulty": "easy"},
    {"query": "What happens when we merge to main?", "expected_keywords": ["staging", "deploy"], "difficulty": "hard"},
    {"query": "What database did we use before PostgreSQL?", "expected_keywords": ["mongodb"], "difficulty": "medium"},
]


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(engine: Any, embedder: Any = None) -> dict:
    """Run the synthetic recall benchmark.

    Args:
        engine: FourDMem engine instance
        embedder: Optional embedder for vector search

    Returns:
        Dict with benchmark results
    """
    results = {
        "benchmark": "synthetic_recall",
        "total_facts": len(FACTS),
        "total_questions": len(QUESTIONS),
        "correct": 0,
        "recall_at_1": 0,
        "recall_at_3": 0,
        "recall_at_5": 0,
        "latencies": [],
        "details": [],
    }

    # Phase 1: Ingest facts
    print(f"[1/2] Ingesting {len(FACTS)} facts...")
    ingest_start = time.time()
    for fact in FACTS:
        meta = json.dumps({"tags": fact["tags"], "importance": fact["importance"]})
        try:
            from cognition.embed_utils import ingest_safely
            ingest_safely(engine, "benchmark-session", "user", fact["label"], meta)
        except Exception as e:
            print(f"  Warning: failed to ingest '{fact['label'][:40]}': {e}")
    ingest_time = time.time() - ingest_start
    print(f"  Ingested in {ingest_time:.2f}s")

    # Phase 2: Query and measure recall
    print(f"[2/2] Querying {len(QUESTIONS)} questions...")
    for q in QUESTIONS:
        query_start = time.time()

        # Compute embedding if available
        embedding = None
        if embedder is not None:
            try:
                embedding = embedder.embed(q["query"])
                if not embedding or all(x == 0.0 for x in embedding):
                    embedding = None
            except Exception:
                pass

        # Search
        try:
            if embedding is not None:
                raw = engine.query_with_embedding(q["query"], embedding, 5)
            else:
                raw = engine.query(q["query"], 5)
            data = json.loads(raw) if isinstance(raw, str) else raw
        except Exception as e:
            data = {"results": [], "error": str(e)}

        latency = (time.time() - query_start) * 1000  # ms
        results["latencies"].append(latency)

        # Check recall
        top_results = data.get("results", [])
        found_at = -1
        for rank, r in enumerate(top_results):
            content = r.get("content", "").lower()
            if any(kw.lower() in content for kw in q["expected_keywords"]):
                found_at = rank
                break

        hit = found_at >= 0
        if hit:
            results["correct"] += 1
            if found_at < 1:
                results["recall_at_1"] += 1
            if found_at < 3:
                results["recall_at_3"] += 1
            if found_at < 5:
                results["recall_at_5"] += 1

        results["details"].append({
            "query": q["query"],
            "difficulty": q["difficulty"],
            "expected_keywords": q["expected_keywords"],
            "found": hit,
            "rank": found_at,
            "latency_ms": round(latency, 2),
            "top_result": top_results[0].get("content", "")[:100] if top_results else "",
        })

    # Compute final metrics
    n = len(QUESTIONS)
    results["recall_at_1"] = round(results["recall_at_1"] / n, 4)
    results["recall_at_3"] = round(results["recall_at_3"] / n, 4)
    results["recall_at_5"] = round(results["recall_at_5"] / n, 4)
    results["accuracy"] = round(results["correct"] / n, 4)

    latencies = sorted(results["latencies"])
    results["latency_p50_ms"] = round(latencies[len(latencies) // 2], 2) if latencies else 0
    results["latency_p99_ms"] = round(latencies[int(len(latencies) * 0.99)], 2) if latencies else 0
    results["ingest_time_s"] = round(ingest_time, 2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Synthetic Recall Benchmark Results")
    print(f"{'='*60}")
    print(f"  Facts ingested:  {len(FACTS)}")
    print(f"  Questions:       {n}")
    print(f"  Accuracy:        {results['accuracy']:.1%}")
    print(f"  Recall@1:        {results['recall_at_1']:.1%}")
    print(f"  Recall@3:        {results['recall_at_3']:.1%}")
    print(f"  Recall@5:        {results['recall_at_5']:.1%}")
    print(f"  Latency P50:     {results['latency_p50_ms']}ms")
    print(f"  Latency P99:     {results['latency_p99_ms']}ms")
    print(f"  Ingest time:     {results['ingest_time_s']}s")
    print(f"{'='*60}")

    return results


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="FourDMem Synthetic Recall Benchmark")
    parser.add_argument("--db-path", default=None, help="Path to SQLite database")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    random.seed(args.seed)

    # Initialize engine
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
    try:
        import fourdmem
    except ImportError:
        print("ERROR: fourdmem Rust bindings not available. Run 'maturin develop' first.")
        sys.exit(1)

    db_path = args.db_path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "benchmarks", "synthetic_recall.db"
    )
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    engine = fourdmem.FourDMemEngine(db_path)

    # Try to get embedder
    embedder = None
    try:
        from cognition.embedder import get_embedder
        embedder = get_embedder()
        if embedder:
            embedder.warmup()
    except Exception:
        print("Warning: No embedder available, using text-only search")

    results = run_benchmark(engine, embedder)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")

    return results


if __name__ == "__main__":
    main()
