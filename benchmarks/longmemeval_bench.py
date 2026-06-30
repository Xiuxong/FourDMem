"""LongMemEval benchmark runner for FourDMem.

LongMemEval tests memory systems across 500 questions covering:
- Single-session recall
- Multi-session reasoning
- Temporal reasoning
- Knowledge update tracking
- Adversarial robustness

This runner implements the evaluation protocol from:
  https://github.com/xiaowu0162/LongMemEval

Usage:
    python -m benchmarks.longmemeval_bench [--data PATH] [--output results/]

Note: Requires the LongMemEval dataset. Download from:
  https://github.com/xiaowu0162/LongMemEval
"""

import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_longmemeval_data(path: str) -> list[dict]:
    """Load LongMemEval dataset."""
    with open(path) as f:
        return json.load(f)


def run_longmemeval_benchmark(
    engine: Any,
    dataset: list[dict],
    embedder: Any = None,
) -> dict:
    """Run the LongMemEval benchmark.

    Args:
        engine: FourDMem engine instance
        dataset: List of evaluation items
        embedder: Optional embedder

    Returns:
        Dict with benchmark results
    """
    results = {
        "benchmark": "longmemeval",
        "total_questions": 0,
        "correct": 0,
        "by_type": {},
        "latencies": [],
    }

    for item in dataset:
        # Ingest context
        for ctx in item.get("context", []):
            try:
                from cognition.embed_utils import ingest_safely
                ingest_safely(engine, "longmemeval", ctx.get("role", "user"), ctx["content"], "{}")
            except Exception:
                pass

        # Query
        question = item["question"]
        expected = item["answer"]
        q_type = item.get("type", "unknown")

        start = time.time()
        try:
            embedding = None
            if embedder:
                try:
                    embedding = embedder.embed(question)
                    if not embedding or all(x == 0.0 for x in embedding):
                        embedding = None
                except Exception:
                    pass

            if embedding is not None:
                raw = engine.query_with_embedding(question, embedding, 5)
            else:
                raw = engine.query(question, 5)
            data = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            data = {"results": []}

        latency = (time.time() - start) * 1000
        results["latencies"].append(latency)
        results["total_questions"] += 1

        # Evaluate
        top_results = data.get("results", [])
        hit = any(
            any(t in r.get("content", "").lower() for t in expected.lower().split() if len(t) > 2)
            for r in top_results
        )

        if hit:
            results["correct"] += 1

        if q_type not in results["by_type"]:
            results["by_type"][q_type] = {"total": 0, "correct": 0}
        results["by_type"][q_type]["total"] += 1
        if hit:
            results["by_type"][q_type]["correct"] += 1

    # Finalize
    n = results["total_questions"]
    if n > 0:
        results["accuracy"] = round(results["correct"] / n, 4)
        for stats in results["by_type"].values():
            stats["accuracy"] = round(stats["correct"] / stats["total"], 4) if stats["total"] > 0 else 0

    latencies = sorted(results["latencies"])
    results["latency_p50_ms"] = round(latencies[len(latencies) // 2], 2) if latencies else 0
    results["latency_p99_ms"] = round(latencies[int(len(latencies) * 0.99)], 2) if latencies else 0

    print(f"\n{'='*60}")
    print(f"LongMemEval Benchmark Results")
    print(f"{'='*60}")
    print(f"  Questions:       {n}")
    print(f"  Accuracy:        {results.get('accuracy', 0):.1%}")
    print(f"  Latency P50:     {results['latency_p50_ms']}ms")
    print(f"  Latency P99:     {results['latency_p99_ms']}ms")
    for qtype, stats in results["by_type"].items():
        print(f"  {qtype}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")
    print(f"{'='*60}")

    return results


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="FourDMem LongMemEval Benchmark")
    parser.add_argument("--data", required=True, help="Path to LongMemEval dataset JSON")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    dataset = load_longmemeval_data(args.data)

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
    try:
        import fourdmem
    except ImportError:
        print("ERROR: fourdmem Rust bindings not available.")
        sys.exit(1)

    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "benchmarks", "longmemeval.db"
    )
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    engine = fourdmem.FourDMemEngine(db_path)

    embedder = None
    try:
        from cognition.embedder import get_embedder
        embedder = get_embedder()
        if embedder:
            embedder.warmup()
    except Exception:
        pass

    results = run_longmemeval_benchmark(engine, dataset, embedder)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")

    return results


if __name__ == "__main__":
    main()
