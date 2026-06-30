"""Run all FourDMem benchmarks and generate a report.

Usage:
    python -m benchmarks.run_all                           # keyword eval, synthetic data
    python -m benchmarks.run_all --judge                   # LLM-as-Judge
    python -m benchmarks.run_all --conversations 5         # more data
    python -m benchmarks.run_all --data benchmarks/data/locomo10.json  # real data
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run all FourDMem benchmarks")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--conversations", type=int, default=3, help="Synthetic conversations")
    parser.add_argument("--data", default=None, help="Real LoCoMo dataset path")
    parser.add_argument("--judge", action="store_true", help="Use LLM-as-Judge")
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="Judge model")
    parser.add_argument("--skip-synthetic", action="store_true", help="Skip synthetic recall")
    parser.add_argument("--skip-locomo", action="store_true", help="Skip LoCoMo benchmark")
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    all_results = {
        "timestamp": timestamp,
        "seed": args.seed,
        "judge": args.judge,
        "benchmarks": {},
    }

    # Initialize engine
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
    try:
        import fourdmem
    except ImportError:
        print("ERROR: fourdmem Rust bindings not available. Run 'maturin develop' first.")
        sys.exit(1)

    # Get embedder
    embedder = None
    try:
        from cognition.embedder import get_embedder
        embedder = get_embedder()
        if embedder:
            embedder.warmup()
    except Exception:
        print("Warning: No embedder available")

    # Get judge (embedding-based, zero API calls)
    judge = None
    if args.judge:
        try:
            from benchmarks.llm_judge import EmbeddingJudge
            judge = EmbeddingJudge(threshold=0.45)
            print(f"EmbeddingJudge: threshold={judge.threshold}")
        except Exception as e:
            print(f"Warning: EmbeddingJudge unavailable: {e}, falling back to keyword matching")

    # ── Synthetic Recall ──────────────────────────────────────────────────────
    if not args.skip_synthetic:
        print("\n" + "=" * 60)
        print("Running Synthetic Recall Benchmark")
        print("=" * 60)

        db_path = os.path.join(args.output, "synthetic_recall.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        engine = fourdmem.FourDMemEngine(db_path)

        from benchmarks.synthetic_recall import run_benchmark
        synthetic_results = run_benchmark(engine, embedder)
        all_results["benchmarks"]["synthetic_recall"] = synthetic_results

    # ── LoCoMo ────────────────────────────────────────────────────────────────
    if not args.skip_locomo:
        print("\n" + "=" * 60)
        print("Running LoCoMo Benchmark")
        print("=" * 60)

        db_path = os.path.join(args.output, "locomo.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        engine = fourdmem.FourDMemEngine(db_path)

        from benchmarks.locomo_bench import run_locomo_benchmark

        if args.data:
            from benchmarks.locomo_bench import load_locomo_data
            dataset = load_locomo_data(args.data)
        else:
            from benchmarks.locomo_dataset import generate_locomo_dataset
            dataset = generate_locomo_dataset(seed=args.seed, n_conversations=args.conversations)

        locomo_results = run_locomo_benchmark(engine, dataset, embedder, judge)
        all_results["benchmarks"]["locomo"] = locomo_results

    # ── Save report ───────────────────────────────────────────────────────────
    report_path = os.path.join(args.output, f"benchmark_report_{timestamp}.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Full report: {report_path}")
    print(f"{'='*60}")

    return all_results


if __name__ == "__main__":
    main()
