"""Failure memory recall benchmark.

Tests that the system can:
1. Store failure experiences with structured conditions
2. Recall failure memories when queried about past failures
3. Detect condition changes via keyword matching
4. Distinguish between "failed approach" and "successful approach"

Usage: python -m benchmarks.recall_failure
"""
import json, os, sys, tempfile, shutil, math

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "python"))


# ── Test data ─────────────────────────────────────────────────────────────

NORMAL_FACTS = [
    {"label": "The project uses PostgreSQL for ACID compliance", "importance": 0.95},
    {"label": "We use Redis for caching with LRU eviction policy", "importance": 0.9},
    {"label": "The deployment pipeline runs on GitHub Actions", "importance": 0.8},
    {"label": "The auth system uses JWT tokens with 15-minute expiry", "importance": 0.9},
    {"label": "We switched from MongoDB to PostgreSQL in January 2025", "importance": 0.9},
    {"label": "The monitoring stack uses Prometheus and Grafana", "importance": 0.8},
    {"label": "David recommends using Rust for performance-critical paths", "importance": 0.8},
    {"label": "The caching layer reduces API latency by 60 percent", "importance": 0.85},
]

FAILURE_FACTS = [
    {
        "label": "Redis OOM crash: used Redis without eviction policy, memory overflow caused outage on 2025-03-15",
        "conditions": {"tool": "redis", "approach": "no_eviction", "env": {"memory": "1GB"}},
        "keywords": ["eviction", "redis", "OOM", "memory overflow"],
    },
    {
        "label": "MongoDB schema drift: no schema enforcement led to inconsistent documents and data pipeline failures",
        "conditions": {"tool": "mongodb", "approach": "schemaless", "env": {"team_size": "5+"}},
        "keywords": ["schema", "mongodb", "schemaless", "drift"],
    },
    {
        "label": "GraphQL N+1 query problem: naive resolver caused 100+ DB queries per request, latency spiked to 5s",
        "conditions": {"tool": "graphql", "approach": "naive_resolver", "env": {"data_loader": "none"}},
        "keywords": ["resolver", "N+1", "graphql", "data loader"],
    },
    {
        "label": "JWT without refresh tokens: 15-minute expiry caused frequent re-authentication, users logged out mid-task",
        "conditions": {"tool": "jwt", "approach": "no_refresh_token", "env": {"expiry": "15min"}},
        "keywords": ["refresh token", "jwt", "expiry", "re-authentication"],
    },
]

FAILURE_QUESTIONS = [
    {"query": "What happened with Redis in March 2025?", "keywords": ["OOM", "memory overflow", "eviction"], "difficulty": "medium"},
    {"query": "Why did we stop using MongoDB?", "keywords": ["schema drift", "schemaless", "inconsistent"], "difficulty": "medium"},
    {"query": "What was the problem with our GraphQL implementation?", "keywords": ["N+1", "resolver", "latency"], "difficulty": "hard"},
    {"query": "What issue did we have with JWT authentication?", "keywords": ["refresh token", "re-authentication", "expiry"], "difficulty": "medium"},
    {"query": "What caching approach should we avoid?", "keywords": ["eviction", "OOM", "memory"], "difficulty": "hard"},
]

CONDITION_CHANGES = [
    {"description": "Redis eviction policy configured", "content": "Configured Redis with LRU eviction policy and increased memory to 4GB", "should_trigger": True},
    {"description": "MongoDB schema validation added", "content": "Added MongoDB schema validation with JSON Schema enforcement", "should_trigger": True},
    {"description": "Unrelated Redis discussion", "content": "Redis is a popular in-memory data structure store", "should_trigger": False},
    {"description": "GraphQL data loader implemented", "content": "Implemented DataLoader for GraphQL resolvers to batch database queries", "should_trigger": True},
]


def make_engine(name):
    db = os.path.join(tempfile.gettempdir(), f"fail_{name}.db")
    for p in [db, db + "-shm", db + "-wal"]:
        if os.path.exists(p):
            os.remove(p)
    td = os.path.join(os.path.dirname(db), "fulltext")
    if os.path.exists(td):
        shutil.rmtree(td, ignore_errors=True)
    return __import__("fourdmem").FourDMemEngine(db)


def run_benchmark() -> dict:
    results = {
        "benchmark": "recall_failure",
        "total_normal": len(NORMAL_FACTS),
        "total_failure": len(FAILURE_FACTS),
        "total_questions": len(FAILURE_QUESTIONS),
    }

    # ── Test 1: Failure recall ────────────────────────────────────────────
    print("\n[Test 1] Failure memory recall...")
    eng = make_engine("recall")

    # Store normal facts
    for f in NORMAL_FACTS:
        eng.save("bench", "user", f["label"], json.dumps({"importance": f["importance"]}))

    # Store failure facts with [FAILURE] prefix and conditions in metadata
    for f in FAILURE_FACTS:
        meta = {"importance": 1.0, "type": "failure_experience", "conditions": f["conditions"]}
        eng.save("bench", "user", f"[FAILURE] {f['label']}", json.dumps(meta))

    # Query failure questions
    failure_recall = []
    for q in FAILURE_QUESTIONS:
        raw = eng.query(q["query"], 10)
        data = json.loads(raw) if isinstance(raw, str) else raw

        found = False
        found_at = -1
        for rank, r in enumerate(data.get("results", [])):
            content = r.get("content", "").lower()
            if any(kw.lower() in content for kw in q["keywords"]):
                found = True
                found_at = rank
                break

        failure_recall.append({
            "query": q["query"],
            "found": found,
            "found_at": found_at,
            "difficulty": q["difficulty"],
        })
        status = f"R@{found_at + 1}" if found else "MISS"
        print(f"  [{status}] {q['query']}")

    # ── Test 2: Condition change detection ────────────────────────────────
    print("\n[Test 2] Condition change detection...")
    condition_results = []
    for cc in CONDITION_CHANGES:
        # Check how many failure keywords match
        matched = []
        for f in FAILURE_FACTS:
            for kw in f["keywords"]:
                if kw.lower() in cc["content"].lower():
                    matched.append(kw)

        would_trigger = len(matched) >= 2
        correct = would_trigger == cc["should_trigger"]
        condition_results.append({
            "description": cc["description"],
            "would_trigger": would_trigger,
            "should_trigger": cc["should_trigger"],
            "correct": correct,
            "matched": matched,
        })
        status = "OK" if correct else "FAIL"
        print(f"  [{status}] {cc['description']} → trigger={would_trigger}, matched={matched}")

    # ── Test 3: Failure vs Success distinction ────────────────────────────
    print("\n[Test 3] Failure vs Success distinction...")
    eng2 = make_engine("distinct")

    # Store failure and success for same tool
    eng2.save("bench", "user",
        "[FAILURE] Redis OOM crash: used Redis without eviction policy, memory overflow",
        json.dumps({"importance": 1.0, "type": "failure"}))
    eng2.save("bench", "user",
        "Redis with LRU eviction works great: reduced API latency by 60 percent",
        json.dumps({"importance": 0.9}))

    raw = eng2.query("Redis caching approach", 5)
    data = json.loads(raw) if isinstance(raw, str) else raw
    results_list = data.get("results", [])

    has_failure = any("OOM" in r.get("content", "") or "crash" in r.get("content", "") for r in results_list)
    has_success = any("latency" in r.get("content", "") or "works great" in r.get("content", "") for r in results_list)

    distinction_ok = has_failure and has_success
    print(f"  [{'OK' if distinction_ok else 'FAIL'}] Both recalled: failure={has_failure}, success={has_success}")

    # ── Compile results ──────────────────────────────────────────────────
    failure_found = sum(1 for r in failure_recall if r["found"])
    condition_correct = sum(1 for r in condition_results if r["correct"])

    results["test1_failure_recall"] = {
        "rate": round(failure_found / len(failure_recall), 4),
        "per_question": failure_recall,
    }
    results["test2_condition_detection"] = {
        "accuracy": round(condition_correct / len(condition_results), 4),
        "per_scenario": condition_results,
    }
    results["test3_distinction"] = {
        "passed": distinction_ok,
        "has_failure": has_failure,
        "has_success": has_success,
    }

    return results


def print_summary(results):
    print("\n" + "=" * 70)
    print("Failure Memory Recall Benchmark")
    print("=" * 70)

    t1 = results["test1_failure_recall"]
    print(f"\n[Test 1] Failure Recall Rate: {t1['rate']:.1%}")
    for q in t1["per_question"]:
        status = f"R@{q['found_at']+1}" if q["found"] else "MISS"
        print(f"  [{status}] {q['query']}")

    t2 = results["test2_condition_detection"]
    print(f"\n[Test 2] Condition Detection: {t2['accuracy']:.1%}")
    for s in t2["per_scenario"]:
        status = "OK" if s["correct"] else "FAIL"
        print(f"  [{status}] {s['description']}")

    t3 = results["test3_distinction"]
    print(f"\n[Test 3] Distinction: {'PASS' if t3['passed'] else 'FAIL'}")
    print(f"  Failure={t3['has_failure']}, Success={t3['has_success']}")
    print("=" * 70)


if __name__ == "__main__":
    r = run_benchmark()
    print_summary(r)
    os.makedirs(os.path.join(_ROOT, "benchmarks", "results"), exist_ok=True)
    out = os.path.join(_ROOT, "benchmarks", "results", "recall_failure.json")
    json.dump(r, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")
