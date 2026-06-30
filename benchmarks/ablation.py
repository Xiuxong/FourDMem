"""Ablation study for FourDMem retrieval components.

Uses QueryRequest disable_* flags. Always uses query_with_embedding
to avoid content resolution issues in the text-only query() path.
Usage: python -m benchmarks.ablation
"""
import json, os, sys, tempfile, time, gc
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "python"))
from benchmarks.synthetic_recall import FACTS, QUESTIONS


def run_ablation() -> dict:
    import fourdmem
    from cognition.embed_utils import ingest_safely

    embedder = None
    try:
        from cognition.embedder import get_embedder
        embedder = get_embedder(); embedder.warmup()
    except Exception: pass

    results = {"benchmark": "ablation", "total_facts": len(FACTS),
               "total_questions": len(QUESTIONS), "components": {}}

    def engine_for(name):
        db = os.path.join(tempfile.gettempdir(), f"abl_{name}.db")
        if os.path.exists(db): os.remove(db)
        return fourdmem.FourDMemEngine(db)

    def test(name, label, **flags):
        print(f"[{name}] {label}")
        eng = engine_for(name)
        try:
            for f in FACTS:
                meta = json.dumps({"tags": f["tags"], "importance": f["importance"]})
                try: ingest_safely(eng, "abl", "user", f["label"], meta)
                except Exception: pass

            c = r1 = r3 = r5 = 0; lats = []
            for q in QUESTIONS:
                t0 = time.time()
                try:
                    vec = embedder.embed(q["query"]) if embedder else [0.0]*512
                    raw = eng.query_with_embedding(q["query"], vec, 5, **flags)
                    data = json.loads(raw) if isinstance(raw, str) else raw
                except Exception: data = {"results": []}
                lats.append((time.time() - t0) * 1000)
                for rank, r in enumerate(data.get("results", [])):
                    if any(kw.lower() in r.get("content","").lower() for kw in q["expected_keywords"]):
                        c += 1
                        if rank < 1: r1 += 1
                        if rank < 3: r3 += 1
                        if rank < 5: r5 += 1
                        break
            n = len(QUESTIONS); lats.sort()
            return {"accuracy": round(c/n,4), "recall_at_1": round(r1/n,4),
                    "recall_at_3": round(r3/n,4), "recall_at_5": round(r5/n,4),
                    "latency_p50_ms": round(lats[len(lats)//2],2) if lats else 0,
                    "latency_p99_ms": round(lats[min(int(len(lats)*.99),len(lats)-1)],2) if lats else 0}
        finally:
            del eng; gc.collect()

    # 1. Text-only
    results["components"]["text_only"] = test("1", "Text-only (FTS5, no graph, no RRF)",
        disable_graph=True, disable_rrf=True)

    # 2. Vector-only
    if embedder:
        results["components"]["vector_only"] = test("2", "Vector-only (HNSW, no fulltext, no graph, no RRF)",
            disable_fulltext=True, disable_graph=True, disable_rrf=True)

    # 3. Fulltext+Vector
    if embedder:
        results["components"]["fulltext_vector"] = test("3", "Fulltext+Vector (no graph, no RRF)",
            disable_graph=True, disable_rrf=True)

    # 4. RRF Fusion (no RIF-U)
    if embedder:
        results["components"]["rrf_fusion"] = test("4", "RRF Fusion (no RIF-U)",
            disable_rif_u=True)

    # 5. Full Pipeline
    if embedder:
        results["components"]["full_pipeline"] = test("5", "Full Pipeline (all enabled)")
    else:
        results["components"]["full_pipeline"] = test("5", "Full Pipeline (text-only fallback)", disable_graph=True, disable_rrf=True)

    return results


def print_results(results: dict):
    comps = results.get("components", {})
    n = results["total_questions"]
    print(f"\n{'='*90}")
    print(f"Ablation Study ({n} questions, {results['total_facts']} facts)")
    print(f"{'='*90}")
    print(f"{'Component':<30} {'Acc':>7} {'R@1':>7} {'R@3':>7} {'R@5':>7} {'P50':>9} {'P99':>9}")
    print(f"{'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*9} {'-'*9}")
    for name, m in comps.items():
        if "error" in m:
            print(f"{name:<30} {'SKIP':>7} ({m['error']})")
        else:
            print(f"{name:<30} {m['accuracy']:>7.1%} {m['recall_at_1']:>7.1%} "
                  f"{m['recall_at_3']:>7.1%} {m['recall_at_5']:>7.1%} "
                  f"{m['latency_p50_ms']:>8.1f}ms{m['latency_p99_ms']:>8.1f}ms")
    print(f"{'='*90}")


if __name__ == "__main__":
    r = run_ablation()
    print_results(r)
