"""Rigorous recall benchmark using embedding similarity for evaluation.

Replaces keyword substring matching with cosine similarity between
query embedding and result content embedding — unbiased semantic evaluation.

Usage: python -m benchmarks.recall_rigorous
"""
import json, os, sys, tempfile, time, gc, math
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "python"))
from benchmarks.synthetic_recall import FACTS, QUESTIONS


def cosine(a, b):
    """Cosine similarity between two vectors."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def run_rigorous_recall() -> dict:
    import fourdmem
    from cognition.embed_utils import ingest_safely

    embedder = None
    try:
        from cognition.embedder import get_embedder
        embedder = get_embedder()
        embedder.warmup()
    except Exception:
        pass

    TOP_K = 10
    SIM_THRESHOLD = 0.65  # min cosine similarity to count as relevant

    results = {
        "benchmark": "recall_rigorous",
        "eval_method": f"cosine_similarity(threshold={SIM_THRESHOLD})",
        "total_facts": len(FACTS),
        "total_questions": len(QUESTIONS),
        "top_k": TOP_K,
        "components": {},
        "per_question": [],
    }

    def engine_for(name):
        import shutil
        db = os.path.join(tempfile.gettempdir(), f"rig_{name}.db")
        for p in [db, db + "-shm", db + "-wal"]:
            if os.path.exists(p):
                os.remove(p)
        # Also clean Tantivy index directory
        tantivy_dir = os.path.join(os.path.dirname(db), "fulltext")
        if os.path.exists(tantivy_dir):
            shutil.rmtree(tantivy_dir, ignore_errors=True)
        return fourdmem.FourDMemEngine(db)

    def test(name, label, **flags):
        print(f"  [{name}] {label}...", end=" ", flush=True)
        eng = engine_for(name)

        # Ingest facts
        fact_labels = []
        fact_embeddings = []
        for f in FACTS:
            meta = json.dumps({"tags": f["tags"], "importance": f["importance"]})
            try:
                ingest_safely(eng, "rig", "user", f["label"], meta)
            except Exception:
                pass
            fact_labels.append(f["label"].lower())
            if embedder:
                fact_embeddings.append(embedder.embed(f["label"]))
            else:
                fact_embeddings.append([0.0] * 512)

        # Pre-compute question embeddings + expected fact index
        q_embeddings = []
        q_expected_idx = []
        for q in QUESTIONS:
            if embedder:
                q_embeddings.append(embedder.embed(q["query"]))
            else:
                q_embeddings.append([0.0] * 512)
            # Find which fact is the ground-truth answer (cosine match)
            qe = q_embeddings[-1]
            best_idx = -1
            best_sim = -1
            for i, fe in enumerate(fact_embeddings):
                sim = cosine(qe, fe)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i
            q_expected_idx.append(best_idx)

        correct = 0
        r_at = {k: 0 for k in range(1, 11)}  # r1..r10
        lats = []
        per_q = []
        n = len(QUESTIONS)

        for qi, q in enumerate(QUESTIONS):
            t0 = time.time()
            try:
                vec = q_embeddings[qi]
                raw = eng.query_with_embedding(q["query"], vec, TOP_K, **flags)
                data = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                data = {"results": []}
            lats.append((time.time() - t0) * 1000)

            result_items = data.get("results", [])
            found_at = -1
            best_result_sim = 0.0

            for rank, r in enumerate(result_items):
                content = r.get("content", "")
                if embedder:
                    content_emb = embedder.embed(content)
                    sim = cosine(vec, content_emb)
                else:
                    sim = 1.0 if content.lower() == fact_labels[q_expected_idx[qi]] else 0.0

                if rank == 0:
                    best_result_sim = sim

                if sim >= SIM_THRESHOLD and found_at < 0:
                    found_at = rank
                    break

            if found_at >= 0:
                correct += 1
                for k in range(1, 11):
                    if found_at < k:
                        r_at[k] += 1

            per_q.append(
                {
                    "query": q["query"][:60],
                    "difficulty": q["difficulty"],
                    "found_at": found_at,
                    "top1_sim": round(best_result_sim, 4),
                    "result_count": len(result_items),
                }
            )

        lats.sort()
        return {
            "accuracy": round(correct / n, 4),
            **{f"R@{k}": round(r_at[k] / n, 4) for k in range(1, 11)},
            "P50_ms": round(lats[len(lats) // 2], 2) if lats else 0,
            "P99_ms": round(lats[min(int(len(lats) * 0.99), len(lats) - 1)], 2) if lats else 0,
            "per_question": per_q,
        }

    configs = [
        ("text_only", "Text-only (FTS5)", dict(disable_graph=True, disable_rrf=True)),
        ("fulltext_vector", "Fulltext+Vector", dict(disable_graph=True, disable_rrf=True)),
        ("rrf_fusion", "RRF Fusion (no RIF-U)", dict(disable_rif_u=True)),
        ("full_pipeline", "Full Pipeline", {}),
    ]
    if embedder:
        configs.insert(1, ("vector_only", "Vector-only (HNSW)", dict(disable_fulltext=True, disable_graph=True, disable_rrf=True)))

    for name, label, flags in configs:
        comp = test(name, label, **flags)
        results["components"][name] = comp
        print(f"R@1={comp['R@1']:.1%} R@5={comp['R@5']:.1%} R@10={comp['R@10']:.1%}")

        # Report per-question failures for full pipeline
        if name == "full_pipeline":
            results["per_question"] = comp["per_question"]

    return results


def print_table(results):
    comps = results["components"]
    header = f"{'Config':<22} {'Acc':>5} " + " ".join(f"R@{k}" for k in range(1, 11)) + f" {'P50':>8}"
    sep = "-" * len(header)
    print()
    print(f"Rigorous Recall ({results['total_questions']}q × {results['total_facts']}f, eval={results['eval_method']}, top_k={results['top_k']})")
    print(sep)
    print(header)
    print(sep)
    for name in comps:
        c = comps[name]
        r_vals = " ".join(f"{c.get(f'R@{k}', 0):>4.0%}" for k in range(1, 11))
        print(f"{name:<22} {c['accuracy']:>5.1%} {r_vals} {c['P50_ms']:>7.1f}ms")
    print(sep)

    # Show failures
    pq = results.get("per_question", [])
    failures = [q for q in pq if q["found_at"] < 0]
    if failures:
        print(f"\nMissed ({len(failures)}/{len(pq)}):")
        for f in failures:
            print(f"  [{f['difficulty']}] {f['query']} (top1_sim={f['top1_sim']})")


if __name__ == "__main__":
    r = run_rigorous_recall()
    print_table(r)
    os.makedirs(os.path.join(_ROOT, "benchmarks", "results"), exist_ok=True)
    out = os.path.join(_ROOT, "benchmarks", "results", "recall_rigorous.json")
    json.dump(r, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")
