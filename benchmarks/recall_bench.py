"""Recall benchmark: R@1, R@5, R@10 across all ablation configs.
Usage: python -m benchmarks.recall_bench
"""
import json, os, sys, tempfile, time, gc
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "python"))
from benchmarks.synthetic_recall import FACTS, QUESTIONS


def run_recall() -> dict:
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
    results = {
        "benchmark": "recall",
        "total_facts": len(FACTS),
        "total_questions": len(QUESTIONS),
        "top_k": TOP_K,
        "components": {},
    }

    def engine_for(name):
        db = os.path.join(tempfile.gettempdir(), f"rec_{name}.db")
        if os.path.exists(db):
            os.remove(db)
        return fourdmem.FourDMemEngine(db)

    def test(name, label, **flags):
        print(f"  [{name}] {label}...", end=" ", flush=True)
        eng = engine_for(name)
        try:
            for f in FACTS:
                meta = json.dumps({"tags": f["tags"], "importance": f["importance"]})
                try:
                    ingest_safely(eng, "rec", "user", f["label"], meta)
                except Exception:
                    pass

            correct = 0
            r1 = r3 = r5 = r10 = 0
            lats = []
            n = len(QUESTIONS)

            for q in QUESTIONS:
                t0 = time.time()
                try:
                    vec = embedder.embed(q["query"]) if embedder else [0.0] * 512
                    raw = eng.query_with_embedding(q["query"], vec, TOP_K, **flags)
                    data = json.loads(raw) if isinstance(raw, str) else raw
                except Exception:
                    data = {"results": []}
                lats.append((time.time() - t0) * 1000)

                found = False
                for rank, r in enumerate(data.get("results", [])):
                    content = r.get("content", "").lower()
                    if any(kw.lower() in content for kw in q["expected_keywords"]):
                        if not found:
                            correct += 1
                            found = True
                        if rank == 0:
                            r1 += 1
                        if rank < 3:
                            r3 += 1
                        if rank < 5:
                            r5 += 1
                        if rank < 10:
                            r10 += 1
                        break

            lats.sort()
            return {
                "accuracy": round(correct / n, 4),
                "R@1": round(r1 / n, 4),
                "R@3": round(r3 / n, 4),
                "R@5": round(r5 / n, 4),
                "R@10": round(r10 / n, 4),
                "P50_ms": round(lats[len(lats) // 2], 2) if lats else 0,
                "P99_ms": round(lats[min(int(len(lats) * 0.99), len(lats) - 1)], 2) if lats else 0,
            }
        finally:
            del eng
            gc.collect()

    configs = [
        ("text_only", "Text-only (FTS5)", dict(disable_graph=True, disable_rrf=True)),
        ("fulltext_vector", "Fulltext+Vector", dict(disable_graph=True, disable_rrf=True)),
        ("rrf_fusion", "RRF Fusion (no RIF-U)", dict(disable_rif_u=True)),
        ("full_pipeline", "Full Pipeline", {}),
    ]

    if embedder:
        configs.insert(1, ("vector_only", "Vector-only (HNSW)", dict(disable_fulltext=True, disable_graph=True, disable_rrf=True)))

    for name, label, flags in configs:
        results["components"][name] = test(name, label, **flags)
        comp = results["components"][name]
        print(f"R@1={comp['R@1']:.1%} R@5={comp['R@5']:.1%} R@10={comp['R@10']:.1%}")

    return results


def print_table(results):
    comps = results["components"]
    header = f"{'Config':<22} {'Acc':>6} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} {'P50':>8} {'P99':>8}"
    sep = "-" * len(header)
    print()
    print(f"Recall Benchmark ({results['total_questions']} questions, {results['total_facts']} facts, top_k={results['top_k']})")
    print(sep)
    print(header)
    print(sep)
    for name in comps:
        c = comps[name]
        print(
            f"{name:<22} {c['accuracy']:>6.1%} {c['R@1']:>6.1%} {c['R@3']:>6.1%} {c['R@5']:>6.1%} {c['R@10']:>6.1%} {c['P50_ms']:>7.1f}ms {c['P99_ms']:>7.1f}ms"
        )
    print(sep)


if __name__ == "__main__":
    r = run_recall()
    print_table(r)
    out = os.path.join(_ROOT, "benchmarks", "results", "recall_bench.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(r, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")
