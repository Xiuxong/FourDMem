"""Single-project multi-session benchmark for FourDMem.

Simulates the real use case: one project, multiple agents/sessions.
All conversations are about the SAME project (not different people).
Tests whether FourDMem can correctly recall project knowledge across sessions.

This is the correct benchmark for FourDMem's architecture.
"""

import json
import os
import sys
import time
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "python"))
sys.path.insert(0, _ROOT)


def generate_single_project_dataset(seed: int = 42) -> dict:
    """Generate a single-project dataset with multiple sessions.

    All sessions are about the same project (FourDMem itself).
    Different "agents" work on different parts of the project.
    """
    import random
    random.seed(seed)

    sessions = [
        # Session 1: Architecture decisions (Agent A)
        {
            "session_id": 1,
            "agent_id": "agent-a",
            "dialogue": [
                {"role": "user", "content": "We need to decide on the storage layer for FourDMem."},
                {"role": "assistant", "content": "What are the options you're considering?"},
                {"role": "user", "content": "SQLite for L0, JSON graph for L1, Markdown for L2, YAML for L3. This is the white-box principle."},
                {"role": "assistant", "content": "That's a solid layered approach. SQLite with FTS5 gives you fulltext search on L0."},
                {"role": "user", "content": "For the embedding model, we're using BAAI/bge-small-zh-v1.5, 512 dimensions."},
                {"role": "assistant", "content": "Good choice for Chinese+English bilingual support. 512 dims is compact enough for CPU inference."},
                {"role": "user", "content": "The retrieval uses RRF fusion across fulltext, graph, and vector search."},
                {"role": "assistant", "content": "Three-way RRF fusion is robust. Each path catches different aspects of relevance."},
            ]
        },
        # Session 2: Bug fixing (Agent B)
        {
            "session_id": 2,
            "agent_id": "agent-b",
            "dialogue": [
                {"role": "user", "content": "Found a bug in the extraction pipeline. The FactExtractor returns empty facts for short sessions."},
                {"role": "assistant", "content": "What's the threshold?"},
                {"role": "user", "content": "It skips sessions with less than 3 messages. We should lower it to 2 or handle single-message sessions differently."},
                {"role": "assistant", "content": "Makes sense. Single-message sessions might be important user instructions."},
                {"role": "user", "content": "Also, the deduplication is too aggressive. It merges facts that are similar but distinct."},
                {"role": "assistant", "content": "What's the similarity threshold?"},
                {"role": "user", "content": "0.85 cosine similarity. We should lower it to 0.9 to be more conservative."},
                {"role": "assistant", "content": "Agreed. 0.9 is safer for distinct facts that happen to share terminology."},
            ]
        },
        # Session 3: Performance optimization (Agent C)
        {
            "session_id": 3,
            "agent_id": "agent-c",
            "dialogue": [
                {"role": "user", "content": "The query latency is too high. P99 is 50ms, we need it under 10ms."},
                {"role": "assistant", "content": "What's the breakdown?"},
                {"role": "user", "content": "Embedding computation takes 30ms, fulltext search 5ms, graph search 3ms, RRF fusion 2ms."},
                {"role": "assistant", "content": "Embedding is the bottleneck. Consider caching embeddings for repeated queries."},
                {"role": "user", "content": "We added an LRU cache for embeddings. Cache hit rate is 60%, so average embedding time dropped to 12ms."},
                {"role": "assistant", "content": "Good. With cache hits, total latency should be around 22ms on average."},
                {"role": "user", "content": "Also added SIMD acceleration for cosine similarity in the vector index. That cut vector search from 3ms to 0.5ms."},
                {"role": "assistant", "content": "Excellent. So now P99 should be around 17ms with cold cache, 7ms with warm cache."},
            ]
        },
        # Session 4: Feature discussion (Agent A again)
        {
            "session_id": 4,
            "agent_id": "agent-a",
            "dialogue": [
                {"role": "user", "content": "We need to add the Version Tree feature for temporal queries."},
                {"role": "assistant", "content": "What's the design?"},
                {"role": "user", "content": "Each entity maintains a version chain. Updates create new versions, old versions get valid_until timestamp."},
                {"role": "assistant", "content": "That's a classic event sourcing pattern. Enables point-in-time queries."},
                {"role": "user", "content": "The temporal gate filters out counterfactual and expired versions before returning search results."},
                {"role": "assistant", "content": "Smart. Counterfactual marking lets you 'abandon' approaches without losing the history."},
                {"role": "user", "content": "We also need subjective time — active ticks instead of physical time. This prevents dormancy from wiping memories."},
                {"role": "assistant", "content": "Active ticks solve the 'vacation problem' — Agent doesn't lose memories just because it wasn't used for a week."},
            ]
        },
    ]

    # QA pairs about the project (single-session and cross-session)
    qa = [
        # Single-session (Session 1)
        {"question": "What storage format is used for L0?", "answer": "SQLite with FTS5",
         "type": "single-session", "evidence_session_ids": [1]},
        {"question": "What embedding model does FourDMem use?", "answer": "BAAI/bge-small-zh-v1.5, 512 dimensions",
         "type": "single-session", "evidence_session_ids": [1]},
        {"question": "How many search paths are fused in retrieval?", "answer": "Three: fulltext, graph, and vector",
         "type": "single-session", "evidence_session_ids": [1]},

        # Single-session (Session 2)
        {"question": "What was the bug in the extraction pipeline?", "answer": "FactExtractor returns empty facts for short sessions with less than 3 messages",
         "type": "single-session", "evidence_session_ids": [2]},
        {"question": "What should the deduplication similarity threshold be?", "answer": "0.9 cosine similarity",
         "type": "single-session", "evidence_session_ids": [2]},

        # Single-session (Session 3)
        {"question": "What is the query latency bottleneck?", "answer": "Embedding computation takes 30ms",
         "type": "single-session", "evidence_session_ids": [3]},
        {"question": "What optimization was applied to vector search?", "answer": "SIMD acceleration for cosine similarity, reduced from 3ms to 0.5ms",
         "type": "single-session", "evidence_session_ids": [3]},

        # Cross-session (Session 1 + 3)
        {"question": "What is the retrieval architecture and what is its latency?", "answer": "RRF fusion of fulltext, graph, and vector search. P99 is around 17ms with cold cache",
         "type": "multi-session", "evidence_session_ids": [1, 3]},

        # Cross-session (Session 1 + 4)
        {"question": "What storage layers does FourDMem use and what temporal feature was added?", "answer": "SQLite L0, JSON L1, Markdown L2, YAML L3. Version Tree was added for temporal queries",
         "type": "multi-session", "evidence_session_ids": [1, 4]},

        # Cross-session (Session 2 + 3)
        {"question": "What bugs were found and what performance optimizations were made?", "answer": "FactExtractor empty facts bug, dedup threshold too aggressive. Added embedding cache and SIMD acceleration",
         "type": "multi-session", "evidence_session_ids": [2, 3]},

        # Temporal
        {"question": "What was done first: the bug fixes or the performance optimization?", "answer": "Bug fixes (session 2) came before performance optimization (session 3)",
         "type": "temporal", "evidence_session_ids": [2, 3]},

        # Adversarial
        {"question": "What database does FourDMem use for production deployment?", "answer": "This was not discussed — FourDMem uses SQLite for L0, not a production database",
         "type": "adversarial", "evidence_session_ids": []},
    ]

    return {
        "conversation_id": "fourdmem-project",
        "sessions": sessions,
        "qa": qa,
    }


def run_single_project_benchmark(
    engine: Any,
    dataset: dict,
    embedder: Any = None,
) -> dict:
    """Run the single-project benchmark."""
    results = {
        "benchmark": "single_project",
        "total_questions": 0,
        "total_correct": 0,
        "by_type": {},
        "latencies": [],
        "details": [],
    }

    conv_id = dataset["conversation_id"]

    # Ingest all sessions
    print(f"Ingesting {len(dataset['sessions'])} sessions...")
    for session in dataset["sessions"]:
        sid = f"{conv_id}-s{session['session_id']}"
        for turn in session["dialogue"]:
            meta = json.dumps({
                "conversation_id": conv_id,
                "session_id": session["session_id"],
                "agent_id": session.get("agent_id", "unknown"),
                "visibility": "shared",
            })
            try:
                from cognition.embed_utils import ingest_safely
                ingest_safely(engine, sid, turn["role"], turn["content"], meta)
            except Exception as e:
                print(f"  Warning: {e}")

    # Extract L1 facts
    print("Extracting L1 facts...")
    from cognition.extractor import FactExtractor
    extractor = FactExtractor()
    total_facts = 0
    for session in dataset["sessions"]:
        sid = f"{conv_id}-s{session['session_id']}"
        result = extractor.get_session_evidence_for_agent(engine, sid, limit=50)
        total_facts += result.get("facts_extracted", 0)
    print(f"  Extracted {total_facts} L1 facts")

    # Evaluate each question
    print(f"\nEvaluating {len(dataset['qa'])} questions...")
    for qa in dataset["qa"]:
        q = qa["question"]
        expected = qa["answer"]
        qtype = qa.get("type", "unknown")

        # Search
        emb = embedder.embed(q) if embedder else None
        t0 = time.time()
        try:
            if emb and not all(x == 0 for x in emb):
                raw = engine.query_with_embedding(q, emb, 5)
            else:
                raw = engine.query(q, 5)
            data = json.loads(raw) if isinstance(raw, str) else raw
            top = data.get("results", [])
        except Exception:
            try:
                raw = engine.query(q.replace("?", ""), 5)
                data = json.loads(raw) if isinstance(raw, str) else raw
                top = data.get("results", [])
            except Exception:
                top = []
        latency = (time.time() - t0) * 1000
        results["latencies"].append(latency)

        context = "\n---\n".join(r.get("content", "") for r in top[:5])

        # Keyword evaluation (quick, no LLM needed)
        expected_lower = expected.lower()
        context_lower = context.lower()
        keywords = [t for t in expected_lower.split() if len(t) > 3]
        matched = sum(1 for k in keywords if k in context_lower)
        hit = matched >= len(keywords) * 0.4 if keywords else False

        results["total_questions"] += 1
        if hit:
            results["total_correct"] += 1

        if qtype not in results["by_type"]:
            results["by_type"][qtype] = {"total": 0, "correct": 0}
        results["by_type"][qtype]["total"] += 1
        if hit:
            results["by_type"][qtype]["correct"] += 1

        results["details"].append({
            "question": q[:80],
            "expected": expected[:80],
            "type": qtype,
            "hit": hit,
            "latency_ms": round(latency, 1),
            "top_result": top[0].get("content", "")[:100] if top else "",
        })

    # Summary
    n = results["total_questions"]
    c = results["total_correct"]
    latencies = sorted(results["latencies"])

    print(f"\n{'='*60}")
    print(f"FourDMem — Single Project Benchmark")
    print(f"{'='*60}")
    print(f"  Project: {conv_id}")
    print(f"  Sessions: {len(dataset['sessions'])}")
    print(f"  Questions: {n}")
    print(f"  Accuracy: {c}/{n} = {c/n:.1%}")
    print(f"  Latency P50: {latencies[len(latencies)//2]:.1f}ms")
    print(f"\n  By type:")
    for t, s in sorted(results["by_type"].items()):
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0
        print(f"    {t:20s}: {acc:.1%} ({s['correct']}/{s['total']})")
    print(f"{'='*60}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    import fourdmem
    from cognition.embedder import get_embedder

    db_path = os.path.join(args.output, "single_project.db")
    os.makedirs(args.output, exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)
    engine = fourdmem.FourDMemEngine(db_path)
    embedder = get_embedder()
    embedder.warmup()

    dataset = generate_single_project_dataset(seed=args.seed)
    results = run_single_project_benchmark(engine, dataset, embedder)

    out_path = os.path.join(args.output, "single_project_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
