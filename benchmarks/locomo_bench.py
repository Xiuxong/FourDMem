"""LoCoMo benchmark runner for FourDMem with LLM-as-Judge evaluation.

Supports:
- Real LoCoMo dataset (--data path/to/locomo10.json)
- Synthetic LoCoMo-format dataset (--generate, default)
- LLM-as-Judge evaluation (--judge, uses gpt-4o-mini)
- Keyword matching fallback (no API key needed)

Usage:
    python -m benchmarks.locomo_bench --generate --judge
    python -m benchmarks.locomo_bench --data benchmarks/data/locomo10.json --judge
"""

import json
import os
import sys
import time
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_locomo_data(path: str) -> list[dict]:
    """Load LoCoMo dataset from JSON file."""
    with open(path) as f:
        return json.load(f)


# ── Evaluation methods ────────────────────────────────────────────────────────

def evaluate_keyword(predicted: str, expected: str, question_type: str) -> dict:
    """Keyword-based evaluation (fallback, no API needed)."""
    if question_type == "adversarial":
        # For adversarial questions, "correct" means the system returned low-confidence/no results
        return {"score": 0.0, "method": "keyword", "type": question_type}

    expected_lower = expected.lower()
    predicted_lower = predicted.lower()
    key_terms = [t.strip() for t in expected_lower.replace(",", " ").split() if len(t.strip()) > 2]
    matched = sum(1 for t in key_terms if t in predicted_lower)
    score = matched / len(key_terms) if key_terms else 0
    return {"score": min(score, 1.0), "method": "keyword", "type": question_type}


def evaluate_with_judge(
    question: str, expected: str, retrieved: str, question_type: str, judge: Any
) -> dict:
    """Embedding-based evaluation (local model, zero API calls)."""
    if question_type == "adversarial":
        result = judge.evaluate_adversarial(question, expected, retrieved)
    else:
        result = judge.evaluate(question, expected, retrieved)
    return {
        "score": 1.0 if result["correct"] else 0.0,
        "method": result.get("method", "embedding"),
        "type": question_type,
        "similarity": result.get("score", 0),
        "latency_ms": result.get("latency_ms", 0),
    }


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_locomo_benchmark(
    engine: Any,
    dataset: list[dict],
    embedder: Any = None,
    judge: Any = None,
) -> dict:
    """Run the LoCoMo benchmark.

    Args:
        engine: FourDMem engine instance
        dataset: List of conversations in LoCoMo format
        embedder: Optional embedder for vector search
        judge: Optional LLMJudge instance (if None, uses keyword matching)

    Returns:
        Dict with benchmark results
    """
    eval_method = "llm_judge" if judge else "keyword"
    results = {
        "benchmark": "locomo",
        "eval_method": eval_method,
        "conversations": len(dataset),
        "total_questions": 0,
        "total_correct": 0,
        "by_type": {},
        "recall_at_1": 0,
        "recall_at_3": 0,
        "recall_at_5": 0,
        "latencies": [],
        "details": [],
    }

    for conv_idx, conv in enumerate(dataset):
        conv_id = conv.get("conversation_id", f"conv_{conv_idx}")
        print(f"\n[{conv_idx+1}/{len(dataset)}] Processing: {conv_id}")

        # Phase 1: Ingest all sessions
        ingest_start = time.time()
        turn_count = 0
        for session in conv["sessions"]:
            sid = f"locomo-{conv_id}-s{session['session_id']}"
            for turn in session["dialogue"]:
                role = turn["role"]
                content = turn["content"]
                meta = json.dumps({
                    "conversation_id": conv_id,
                    "session_id": session["session_id"],
                    "auto": True,
                })
                try:
                    from cognition.embed_utils import ingest_safely
                    ingest_safely(engine, sid, role, content, meta)
                    turn_count += 1
                except Exception as e:
                    print(f"  Warning: ingest failed: {e}")
        ingest_time = (time.time() - ingest_start) * 1000

        # Phase 1b: L0→L1 fact ingestion (dialogue turns as pre-extracted facts)
        extract_start = time.time()
        total_facts = 0
        try:
            from cognition.extractor import FactExtractor
            from cognition.aggregator import AutoAggregator
            extractor = FactExtractor()
            aggregator = AutoAggregator(aggregation_threshold=3)
            # Use get_session_evidence_for_agent then ingest as facts
            for session in conv["sessions"]:
                sid = f"locomo-{conv_id}-s{session['session_id']}"
                evidence_result = extractor.get_session_evidence_for_agent(engine, sid, limit=50)
                evidence = evidence_result.get("evidence", [])
                # Convert evidence items to fact dicts
                fact_dicts = []
                for ev in evidence:
                    content = ev.get("content", "")
                    if len(content) >= 8:
                        fact_dicts.append({
                            "label": content[:300],
                            "importance": 0.5,
                            "tags": ["locomo", f"session_{session['session_id']}"],
                            "l0_refs": [ev.get("id")] if ev.get("id") else [],
                        })
                if fact_dicts:
                    result = extractor.ingest_facts(engine, fact_dicts)
                    stored = result.get("facts_stored", 0)
                    total_facts += stored
            extract_time = (time.time() - extract_start) * 1000
            print(f"  Ingested {turn_count} turns in {ingest_time:.0f}ms, "
                  f"indexed {total_facts} L1 facts in {extract_time:.0f}ms")
        except Exception as e:
            print(f"  Ingested {turn_count} turns in {ingest_time:.0f}ms "
                  f"(L1 indexing failed: {e})")

        # Phase 2: Query and evaluate
        qa_list = conv.get("qa", [])
        print(f"  Evaluating {len(qa_list)} questions...")

        for qa_idx, qa in enumerate(qa_list):
            question = qa["question"]
            expected = qa["answer"]
            q_type = qa.get("type", "unknown")

            query_start = time.time()

            # Compute embedding
            embedding = None
            if embedder:
                try:
                    embedding = embedder.embed(question)
                    if not embedding or all(x == 0.0 for x in embedding):
                        embedding = None
                except Exception:
                    pass

            # Search (top-5)
            try:
                if embedding is not None:
                    raw = engine.query_with_embedding(question, embedding, 5)
                else:
                    raw = engine.query(question, 5)
                data = json.loads(raw) if isinstance(raw, str) else raw
            except Exception as e:
                data = {"results": [], "error": str(e)}

            latency = (time.time() - query_start) * 1000
            results["latencies"].append(latency)

            top_results = data.get("results", [])

            # Join top-K results into single context for judge
            retrieved_context = "\n---\n".join(
                r.get("content", "") for r in top_results[:5]
            )

            # Evaluate
            if judge:
                eval_result = evaluate_with_judge(question, expected, retrieved_context, q_type, judge)
            else:
                eval_result = evaluate_keyword(retrieved_context, expected, q_type)

            hit = eval_result["score"] >= 0.5
            results["total_questions"] += 1
            if hit:
                results["total_correct"] += 1

            # Recall@K: check if expected keywords appear in top-K
            for rank, r in enumerate(top_results):
                content = r.get("content", "").lower()
                expected_terms = [t for t in expected.lower().split() if len(t) > 2]
                if any(t in content for t in expected_terms):
                    if rank < 1:
                        results["recall_at_1"] += 1
                    if rank < 3:
                        results["recall_at_3"] += 1
                    if rank < 5:
                        results["recall_at_5"] += 1
                    break

            # Track by type
            if q_type not in results["by_type"]:
                results["by_type"][q_type] = {"total": 0, "correct": 0}
            results["by_type"][q_type]["total"] += 1
            if hit:
                results["by_type"][q_type]["correct"] += 1

            results["details"].append({
                "conversation_id": conv_id,
                "question": question,
                "expected": expected,
                "type": q_type,
                "score": round(eval_result["score"], 3),
                "hit": hit,
                "latency_ms": round(latency, 2),
                "eval_method": eval_method,
                "reasoning": eval_result.get("reasoning", ""),
                "top_result": top_results[0].get("content", "")[:120] if top_results else "",
            })

        # Progress
        correct_so_far = results["total_correct"]
        total_so_far = results["total_questions"]
        print(f"  Progress: {correct_so_far}/{total_so_far} ({correct_so_far/total_so_far:.0%})")

    # ── Compute final metrics ─────────────────────────────────────────────────
    n = results["total_questions"]
    if n > 0:
        results["accuracy"] = round(results["total_correct"] / n, 4)
        results["recall_at_1"] = round(results["recall_at_1"] / n, 4)
        results["recall_at_3"] = round(results["recall_at_3"] / n, 4)
        results["recall_at_5"] = round(results["recall_at_5"] / n, 4)

        for qtype, stats in results["by_type"].items():
            stats["accuracy"] = round(stats["correct"] / stats["total"], 4) if stats["total"] > 0 else 0

    latencies = sorted(results["latencies"])
    results["latency_p50_ms"] = round(latencies[len(latencies) // 2], 2) if latencies else 0
    results["latency_p99_ms"] = round(latencies[int(len(latencies) * 0.99)], 2) if latencies else 0

    if judge:
        results["judge_stats"] = judge.get_stats()

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"LoCoMo Benchmark Results (eval: {eval_method})")
    print(f"{'='*60}")
    print(f"  Conversations:   {results['conversations']}")
    print(f"  Questions:       {n}")
    print(f"  Accuracy:        {results.get('accuracy', 0):.1%}")
    print(f"  Recall@1:        {results['recall_at_1']:.1%}")
    print(f"  Recall@3:        {results['recall_at_3']:.1%}")
    print(f"  Recall@5:        {results['recall_at_5']:.1%}")
    print(f"  Latency P50:     {results['latency_p50_ms']}ms")
    print(f"  Latency P99:     {results['latency_p99_ms']}ms")
    print(f"\n  By question type:")
    for qtype, stats in sorted(results["by_type"].items()):
        print(f"    {qtype:20s}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")
    if judge:
        js = results["judge_stats"]
        print(f"\n  Judge: {js.get('method', 'embedding')}, {js['total_calls']} calls, avg {js['avg_latency_ms']}ms/call")
    print(f"{'='*60}")

    return results


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="FourDMem LoCoMo Benchmark")
    parser.add_argument("--data", default=None, help="Path to LoCoMo dataset JSON")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic dataset")
    parser.add_argument("--conversations", type=int, default=3, help="Number of synthetic conversations")
    parser.add_argument("--judge", action="store_true", help="Use LLM-as-Judge (requires API key)")
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="Judge model")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    # Load or generate data
    if args.data:
        print(f"Loading dataset from {args.data}")
        dataset = load_locomo_data(args.data)
    else:
        from benchmarks.locomo_dataset import generate_locomo_dataset
        print(f"Generating synthetic dataset ({args.conversations} conversations)")
        dataset = generate_locomo_dataset(seed=args.seed, n_conversations=args.conversations)

    # Initialize engine
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
    try:
        import fourdmem
    except ImportError:
        print("ERROR: fourdmem Rust bindings not available. Run 'maturin develop' first.")
        sys.exit(1)

    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "benchmarks", "locomo.db"
    )
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    engine = fourdmem.FourDMemEngine(db_path)

    # Get embedder
    embedder = None
    try:
        from cognition.embedder import get_embedder
        embedder = get_embedder()
        if embedder:
            embedder.warmup()
    except Exception:
        print("Warning: No embedder available, using text-only search")

    # Get judge (embedding-based, zero API calls)
    judge = None
    if args.judge:
        try:
            from benchmarks.llm_judge import EmbeddingJudge
            judge = EmbeddingJudge(threshold=0.45)
            print(f"Using EmbeddingJudge (threshold={judge.threshold})")
        except Exception as e:
            print(f"Warning: Could not initialize EmbeddingJudge: {e}")
            print("Falling back to keyword matching")

    results = run_locomo_benchmark(engine, dataset, embedder, judge)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults written to {args.output}")

    return results


if __name__ == "__main__":
    main()
