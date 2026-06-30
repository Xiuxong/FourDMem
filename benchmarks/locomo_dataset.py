"""LoCoMo-format synthetic dataset generator.

Generates a realistic multi-session conversation dataset that mimics
the structure and difficulty of the real LoCoMo benchmark.

When the real LoCoMo dataset is available, replace this with:
    python -m benchmarks.locomo_bench --data benchmarks/data/locomo10.json
"""

import json
import random
from typing import Optional


def generate_locomo_dataset(seed: int = 42, n_conversations: int = 3) -> list[dict]:
    """Generate LoCoMo-format dataset with realistic multi-session conversations.

    Each conversation has:
    - 10 sessions with 15-25 turns each
    - QA pairs covering: single-session, multi-session, temporal, adversarial
    - Realistic coding/work scenarios
    """
    random.seed(seed)

    conversations = []

    # ── Conversation templates ────────────────────────────────────────────────

    people = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry"]
    companies = ["Acme Corp", "TechFlow", "DataSphere", "CloudNine", "NeuralPath"]
    tech_stacks = [
        {"lang": "Python", "db": "PostgreSQL", "cache": "Redis", "ci": "GitHub Actions"},
        {"lang": "TypeScript", "db": "MongoDB", "cache": "Memcached", "ci": "GitLab CI"},
        {"lang": "Rust", "db": "SQLite", "cache": "Redis", "ci": "Jenkins"},
        {"lang": "Go", "db": "CockroachDB", "cache": "Redis", "ci": "CircleCI"},
    ]

    for conv_idx in range(n_conversations):
        person = people[conv_idx % len(people)]
        company = companies[conv_idx % len(companies)]
        tech = tech_stacks[conv_idx % len(tech_stacks)]
        sessions = []
        facts_learned = []  # Track facts for QA generation

        # ── Session 1: Onboarding ─────────────────────────────────────────────
        s1_dialogue = [
            {"role": "user", "content": f"Hi! I just started working at {company} as a {random.choice(['senior', 'junior', 'staff'])} engineer."},
            {"role": "assistant", "content": f"Welcome to {company}! That's exciting. What team are you on?"},
            {"role": "user", "content": f"I'm on the platform team. My manager is {people[(conv_idx+1) % len(people)]}, who's been here for {random.randint(2,8)} years."},
            {"role": "assistant", "content": f"Great to have an experienced manager. What's your tech stack looking like?"},
            {"role": "user", "content": f"We use {tech['lang']} for the backend, {tech['db']} for the database, and {tech['cache']} for caching. The CI/CD runs on {tech['ci']}."},
            {"role": "assistant", "content": f"Solid stack! {tech['lang']} + {tech['db']} is a proven combination."},
            {"role": "user", "content": f"My first task is to refactor the authentication module. It currently uses session-based auth but we want to migrate to JWT."},
            {"role": "assistant", "content": "JWT migration is a common pattern. Make sure to handle token refresh and revocation properly."},
            {"role": "user", "content": "The codebase is about 150K lines. The auth module is in src/auth/ and it's about 3000 lines."},
            {"role": "assistant", "content": "3000 lines for an auth module is manageable. Focus on the interface boundaries first."},
        ]
        facts_learned.extend([
            f"{person} works at {company}",
            f"{person} is on the platform team",
            f"Manager is {people[(conv_idx+1) % len(people)]}, {random.randint(2,8)} years at company",
            f"Tech stack: {tech['lang']}, {tech['db']}, {tech['cache']}, {tech['ci']}",
            f"First task: refactor auth from session-based to JWT",
            f"Codebase is 150K lines, auth module is 3000 lines in src/auth/",
        ])
        sessions.append({"session_id": 1, "dialogue": s1_dialogue})

        # ── Session 2: Debugging incident ─────────────────────────────────────
        s2_dialogue = [
            {"role": "user", "content": "We had a production incident yesterday. The API response time spiked to 5 seconds."},
            {"role": "assistant", "content": "That's serious. Did you identify the root cause?"},
            {"role": "user", "content": f"Turns out it was a missing index on the {tech['db']} users table. A full table scan was triggered by a new query in the search endpoint."},
            {"role": "assistant", "content": "Classic issue. Missing indexes are one of the most common performance problems."},
            {"role": "user", "content": "We fixed it by adding a composite index on (email, status). Response time dropped back to 50ms."},
            {"role": "assistant", "content": "Great recovery. 5s to 50ms is a 100x improvement."},
            {"role": "user", "content": f"{people[(conv_idx+1) % len(people)]} wants us to add query performance monitoring going forward."},
            {"role": "assistant", "content": "Proactive monitoring is the right approach. Consider pg_stat_statements for PostgreSQL."},
            {"role": "user", "content": "We're also adding a slow query log threshold of 200ms."},
            {"role": "assistant", "content": "200ms is a reasonable threshold for API queries."},
        ]
        facts_learned.extend([
            "Production incident: API response time spiked to 5 seconds",
            f"Root cause: missing index on {tech['db']} users table",
            "Fix: composite index on (email, status), response time dropped to 50ms",
            "Adding query performance monitoring and slow query log (200ms threshold)",
        ])
        sessions.append({"session_id": 2, "dialogue": s2_dialogue})

        # ── Session 3: Architecture decision ──────────────────────────────────
        s3_dialogue = [
            {"role": "user", "content": "We're debating whether to use microservices or keep the monolith for the new payment feature."},
            {"role": "assistant", "content": "What's driving the discussion?"},
            {"role": "user", "content": "The payment module needs to be isolated for PCI compliance. But the team is only 5 people."},
            {"role": "assistant", "content": "With 5 people, a full microservices setup might be overkill. Consider a modular monolith with clear boundaries."},
            {"role": "user", "content": f"We decided to go with a modular monolith. The payment module will be a separate package with its own database schema."},
            {"role": "assistant", "content": "Good call. Modular monolith gives you isolation without the operational overhead."},
            {"role": "user", "content": "We're using the Strangler Fig pattern to gradually extract it if needed later."},
            {"role": "assistant", "content": "Smart. The Strangler Fig pattern is perfect for this kind of gradual migration."},
            {"role": "user", "content": "The payment integration uses Stripe. We need to handle webhooks for payment status updates."},
            {"role": "assistant", "content": "Stripe webhooks are reliable. Make sure to verify the webhook signature and handle idempotency."},
        ]
        facts_learned.extend([
            "Architecture: modular monolith (not microservices) for payment module",
            "Reason: team too small (5 people) for microservices, PCI compliance needs isolation",
            "Payment module: separate package with own database schema",
            "Using Strangler Fig pattern for future extraction",
            "Payment integration: Stripe with webhook handling",
        ])
        sessions.append({"session_id": 3, "dialogue": s3_dialogue})

        # ── Session 4: Testing strategy ───────────────────────────────────────
        s4_dialogue = [
            {"role": "user", "content": "Our test suite takes 45 minutes to run. It's becoming a bottleneck."},
            {"role": "assistant", "content": "45 minutes is too long. What's the breakdown?"},
            {"role": "user", "content": "30 minutes is integration tests hitting the real database. 10 minutes is E2E tests. 5 minutes is unit tests."},
            {"role": "assistant", "content": "The database-bound tests are the bottleneck. Consider parallelizing them or using testcontainers."},
            {"role": "user", "content": "We tried parallelizing but hit race conditions. The tests share a test database and create conflicting data."},
            {"role": "assistant", "content": "Use database transactions that rollback after each test, or give each test its own schema."},
            {"role": "user", "content": "We're going with per-test schema isolation. Each test gets its own schema that's created and dropped."},
            {"role": "assistant", "content": "That's the cleanest approach. It eliminates race conditions completely."},
            {"role": "user", "content": f"We also added a flaky test detector. If a test fails twice in a row but passes on retry, it's flagged for investigation."},
            {"role": "assistant", "content": "Flaky test detection is valuable. It helps maintain test suite reliability."},
        ]
        facts_learned.extend([
            "Test suite takes 45 minutes (30min integration, 10min E2E, 5min unit)",
            "Integration tests bottleneck: hitting real database",
            "Fix: per-test schema isolation (create/drop schema per test)",
            "Added flaky test detector: fails twice + passes on retry = flagged",
        ])
        sessions.append({"session_id": 4, "dialogue": s4_dialogue})

        # ── Session 5: Performance optimization ───────────────────────────────
        s5_dialogue = [
            {"role": "user", "content": "We need to optimize the data export feature. It currently takes 20 minutes for 1M records."},
            {"role": "assistant", "content": "What's the current approach?"},
            {"role": "user", "content": "It loads all records into memory, transforms them, then writes to CSV. We're hitting OOM errors on large exports."},
            {"role": "assistant", "content": "Classic problem. Switch to streaming: process records in batches and write incrementally."},
            {"role": "user", "content": "We implemented batch processing with a 10K record batch size. Export time dropped to 3 minutes and memory usage is constant."},
            {"role": "assistant", "content": "Excellent. 20 minutes to 3 minutes with constant memory is a great improvement."},
            {"role": "user", "content": "We also added progress tracking. The user can see a progress bar and estimated time remaining."},
            {"role": "assistant", "content": "Progress tracking is a great UX improvement for long-running operations."},
            {"role": "user", "content": "Next we're looking at adding async export. The user starts the export, gets a notification when it's ready."},
            {"role": "assistant", "content": "Async export is the right pattern for large datasets. Use a job queue like Celery or Bull."},
        ]
        facts_learned.extend([
            "Data export: 1M records took 20 minutes, now 3 minutes with batch processing",
            "Batch size: 10K records, constant memory usage",
            "Added progress bar and estimated time remaining",
            "Planning async export with job queue (Celery/Bull)",
        ])
        sessions.append({"session_id": 5, "dialogue": s5_dialogue})

        # ── Session 6: Security review ────────────────────────────────────────
        s6_dialogue = [
            {"role": "user", "content": "We had a security audit. They found three issues."},
            {"role": "assistant", "content": "What were the findings?"},
            {"role": "user", "content": "First, SQL injection in the search endpoint. Second, missing CSRF protection. Third, overly permissive CORS policy."},
            {"role": "assistant", "content": "All three are common but serious. SQL injection is the highest priority."},
            {"role": "user", "content": "We fixed the SQL injection by using parameterized queries. The CSRF fix added token validation. CORS was restricted to our domain only."},
            {"role": "assistant", "content": "Good response. Parameterized queries eliminate SQL injection entirely."},
            {"role": "user", "content": "We also added rate limiting: 100 requests per minute per user for API endpoints, 10 per minute for auth endpoints."},
            {"role": "assistant", "content": "Rate limiting is essential. The stricter auth endpoint limit is smart — it slows brute force attacks."},
            {"role": "user", "content": f"{people[(conv_idx+2) % len(people)]} suggested adding a WAF. We're evaluating Cloudflare vs AWS WAF."},
            {"role": "assistant", "content": "Both are solid. Cloudflare is easier to set up, AWS WAF integrates better with the AWS ecosystem."},
        ]
        facts_learned.extend([
            "Security audit found: SQL injection, missing CSRF, overly permissive CORS",
            "Fixes: parameterized queries, CSRF token validation, restricted CORS",
            "Rate limiting: 100/min per user (API), 10/min (auth)",
            "Evaluating WAF: Cloudflare vs AWS WAF",
        ])
        sessions.append({"session_id": 6, "dialogue": s6_dialogue})

        # ── Session 7: New feature ────────────────────────────────────────────
        s7_dialogue = [
            {"role": "user", "content": "We're building a real-time notification system. The requirements are: WebSocket-based, supports 10K concurrent connections."},
            {"role": "assistant", "content": "10K WebSocket connections is achievable. What framework are you considering?"},
            {"role": "user", "content": f"We're using {tech['lang']} with Socket.IO for the WebSocket layer. The notification queue uses RabbitMQ."},
            {"role": "assistant", "content": "Socket.IO + RabbitMQ is a solid combination for real-time notifications."},
            {"role": "user", "content": "We need to handle reconnection gracefully. If a user disconnects, they should receive missed notifications on reconnect."},
            {"role": "assistant", "content": "Store notifications in a per-user queue. On reconnect, flush the queue before streaming new ones."},
            {"role": "user", "content": "We implemented a notification store with a 7-day TTL. Notifications are persisted in {tech['db']} and cached in {tech['cache']}."},
            {"role": "assistant", "content": "7-day TTL is reasonable. The {tech['db']} + {tech['cache']} combo gives you durability and speed."},
            {"role": "user", "content": "Load testing showed we can handle 15K concurrent connections with 99th percentile latency under 100ms."},
            {"role": "assistant", "content": "15K connections with p99 under 100ms is excellent performance."},
        ]
        facts_learned.extend([
            "Real-time notification system: WebSocket-based, 10K concurrent connections target",
            f"Stack: {tech['lang']} + Socket.IO + RabbitMQ",
            "Notification store: 7-day TTL, persisted in DB, cached in Redis",
            "Load test: 15K concurrent connections, p99 < 100ms",
        ])
        sessions.append({"session_id": 7, "dialogue": s7_dialogue})

        # ── Session 8: Deployment pipeline ────────────────────────────────────
        s8_dialogue = [
            {"role": "user", "content": "We're revamping the deployment pipeline. Currently it takes 30 minutes from merge to production."},
            {"role": "assistant", "content": "30 minutes is long. What's the breakdown?"},
            {"role": "user", "content": "Build: 5 min. Tests: 15 min. Staging deploy: 5 min. Smoke tests: 5 min."},
            {"role": "assistant", "content": "Tests are the bottleneck. Can you parallelize or cache dependencies?"},
            {"role": "user", "content": "We parallelized the test suite into 4 workers. Test time dropped from 15 to 4 minutes."},
            {"role": "assistant", "content": "Nice. 4x parallelization with near-linear speedup."},
            {"role": "user", "content": "We also added Docker layer caching. Build time dropped from 5 to 2 minutes."},
            {"role": "assistant", "content": "Layer caching is a big win for Docker builds."},
            {"role": "user", "content": "Total pipeline time is now 16 minutes. We're targeting under 10 minutes."},
            {"role": "assistant", "content": "16 minutes is good. Consider running smoke tests in parallel with staging deploy."},
            {"role": "user", "content": f"We also added automatic rollback. If error rate exceeds 5% within 5 minutes of deploy, it auto-reverts."},
            {"role": "assistant", "content": "Automatic rollback based on error rate is a great safety net."},
        ]
        facts_learned.extend([
            "Deployment pipeline: 30 min → 16 min",
            "Breakdown: build 2min, tests 4min, staging 5min, smoke 5min",
            "Optimizations: 4x test parallelization, Docker layer caching",
            "Target: under 10 minutes",
            "Auto-rollback: if error rate > 5% within 5 min of deploy",
        ])
        sessions.append({"session_id": 8, "dialogue": s8_dialogue})

        # ── Session 9: Team process ───────────────────────────────────────────
        s9_dialogue = [
            {"role": "user", "content": "We're switching from 2-week sprints to Kanban."},
            {"role": "assistant", "content": "What prompted the change?"},
            {"role": "user", "content": "Too many interrupts. We'd plan a sprint and then get 3 urgent production issues that blow up the plan."},
            {"role": "assistant", "content": "Kanban is better for high-interrupt environments. It gives you more flexibility."},
            {"role": "user", "content": "We set WIP limits: max 3 items per developer, max 8 items in the review column."},
            {"role": "assistant", "content": "WIP limits are the key to Kanban. They prevent context switching and improve flow."},
            {"role": "user", "content": f"We also introduced code review SLAs. PRs must be reviewed within 4 hours. {people[(conv_idx+3) % len(people)]} is the review champion."},
            {"role": "assistant", "content": "4-hour SLA is aggressive but achievable with WIP limits. Review champions help maintain velocity."},
            {"role": "user", "content": "Our lead time dropped from 5 days to 2 days after switching to Kanban."},
            {"role": "assistant", "content": "5 days to 2 days is a significant improvement. The WIP limits are working."},
        ]
        facts_learned.extend([
            "Switched from 2-week sprints to Kanban (too many interrupts)",
            "WIP limits: max 3 per developer, max 8 in review",
            "Code review SLA: 4 hours",
            f"Review champion: {people[(conv_idx+3) % len(people)]}",
            "Lead time: 5 days → 2 days",
        ])
        sessions.append({"session_id": 9, "dialogue": s9_dialogue})

        # ── Session 10: Future plans ──────────────────────────────────────────
        s10_dialogue = [
            {"role": "user", "content": "We're planning the Q3 roadmap. Three big items."},
            {"role": "assistant", "content": "What's on the roadmap?"},
            {"role": "user", "content": "First, migrate from REST to GraphQL for the mobile API. Second, add multi-tenancy support. Third, implement the audit log."},
            {"role": "assistant", "content": "All three are significant. What's the priority order?"},
            {"role": "user", "content": "Multi-tenancy is top priority — it's blocking our enterprise sales. GraphQL is second. Audit log is third."},
            {"role": "assistant", "content": "Multi-tenancy first makes sense if it's blocking revenue."},
            {"role": "user", "content": "For multi-tenancy, we're going with a shared database, separate schema approach."},
            {"role": "assistant", "content": "Shared DB separate schema is a good middle ground between shared everything and separate databases."},
            {"role": "user", "content": "The GraphQL migration will use Apollo Server. We'll keep the REST API for backward compatibility."},
            {"role": "assistant", "content": "Running both GraphQL and REST in parallel is the safe migration path."},
            {"role": "user", "content": f"{people[(conv_idx+1) % len(people)]} wants the audit log to be immutable. We're considering an append-only table with cryptographic chaining."},
            {"role": "assistant", "content": "Cryptographic chaining gives you tamper-evidence. It's the gold standard for audit logs."},
        ]
        facts_learned.extend([
            "Q3 roadmap: GraphQL migration, multi-tenancy, audit log",
            "Priority: multi-tenancy (blocking enterprise sales) > GraphQL > audit log",
            "Multi-tenancy: shared database, separate schema approach",
            "GraphQL: Apollo Server, keep REST for backward compatibility",
            "Audit log: immutable, append-only with cryptographic chaining",
        ])
        sessions.append({"session_id": 10, "dialogue": s10_dialogue})

        # ── Generate QA pairs ─────────────────────────────────────────────────
        qa = _generate_qa_pairs(facts_learned, person, company, tech, conv_idx)

        conversations.append({
            "conversation_id": f"synth_{conv_idx:03d}",
            "sessions": sessions,
            "qa": qa,
        })

    return conversations


def _generate_qa_pairs(facts: list, person: str, company: str, tech: dict, idx: int) -> list[dict]:
    """Generate QA pairs from learned facts, covering all question types."""
    qa = []

    # Single-session questions (easy)
    qa.append({
        "question": f"Where does {person} work?",
        "answer": company,
        "type": "single-session",
        "evidence_session_ids": [1],
    })
    qa.append({
        "question": f"What programming language does {person}'s team use?",
        "answer": tech["lang"],
        "type": "single-session",
        "evidence_session_ids": [1],
    })
    qa.append({
        "question": "What was the root cause of the production incident?",
        "answer": f"Missing index on the {tech['db']} users table",
        "type": "single-session",
        "evidence_session_ids": [2],
    })
    qa.append({
        "question": "How was the production incident fixed?",
        "answer": "Added a composite index on (email, status)",
        "type": "single-session",
        "evidence_session_ids": [2],
    })
    qa.append({
        "question": "What architecture decision was made for the payment module?",
        "answer": "Modular monolith with separate package and own database schema",
        "type": "single-session",
        "evidence_session_ids": [3],
    })
    qa.append({
        "question": "How long does the test suite take to run?",
        "answer": "45 minutes (30 min integration, 10 min E2E, 5 min unit)",
        "type": "single-session",
        "evidence_session_ids": [4],
    })
    qa.append({
        "question": "What was the data export time before and after optimization?",
        "answer": "20 minutes before, 3 minutes after batch processing optimization",
        "type": "single-session",
        "evidence_session_ids": [5],
    })
    qa.append({
        "question": "What security issues were found in the audit?",
        "answer": "SQL injection, missing CSRF protection, overly permissive CORS policy",
        "type": "single-session",
        "evidence_session_ids": [6],
    })
    qa.append({
        "question": "What is the rate limit for API endpoints?",
        "answer": "100 requests per minute per user",
        "type": "single-session",
        "evidence_session_ids": [6],
    })
    qa.append({
        "question": "How many concurrent WebSocket connections can the notification system handle?",
        "answer": "15K concurrent connections with p99 under 100ms",
        "type": "single-session",
        "evidence_session_ids": [7],
    })

    # Multi-session questions (medium)
    qa.append({
        "question": f"What database does {person}'s team use and what index was added after the incident?",
        "answer": f"{tech['db']}, composite index on (email, status)",
        "type": "multi-session",
        "evidence_session_ids": [1, 2],
    })
    qa.append({
        "question": "What is the deployment pipeline time and what optimizations were applied?",
        "answer": "16 minutes total, optimized with 4x test parallelization and Docker layer caching",
        "type": "multi-session",
        "evidence_session_ids": [4, 8],
    })
    qa.append({
        "question": "What testing strategy was adopted and how does it affect the deployment pipeline?",
        "answer": "Per-test schema isolation, parallelized to 4 workers, test time dropped from 15 to 4 minutes",
        "type": "multi-session",
        "evidence_session_ids": [4, 8],
    })

    # Temporal questions (hard)
    qa.append({
        "question": "What happened first: the security audit or the architecture decision for payments?",
        "answer": "The architecture decision came first (session 3), then the security audit (session 6)",
        "type": "temporal",
        "evidence_session_ids": [3, 6],
    })
    qa.append({
        "question": "In what order were the performance issues addressed across the project?",
        "answer": "First API response time (index fix), then data export (batch processing), then deployment pipeline (parallelization)",
        "type": "temporal",
        "evidence_session_ids": [2, 5, 8],
    })
    qa.append({
        "question": "What process changes were made and in what order?",
        "answer": "First added monitoring after incident, then switched to Kanban, then added code review SLAs",
        "type": "temporal",
        "evidence_session_ids": [2, 9],
    })

    # Adversarial questions (should not be answerable)
    qa.append({
        "question": "What is the company's annual revenue?",
        "answer": "This information was not discussed",
        "type": "adversarial",
        "evidence_session_ids": [],
    })
    qa.append({
        "question": "What programming language does the frontend use?",
        "answer": "The frontend language was not mentioned",
        "type": "adversarial",
        "evidence_session_ids": [],
    })

    return qa


def save_dataset(conversations: list[dict], path: str):
    """Save dataset to JSON file."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(conversations)} conversations to {path}")


def main():
    """Generate and save dataset."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmarks/data/locomo_synthetic.json")
    parser.add_argument("--conversations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    conversations = generate_locomo_dataset(seed=args.seed, n_conversations=args.conversations)
    save_dataset(conversations, args.output)

    # Stats
    total_qa = sum(len(c["qa"]) for c in conversations)
    total_turns = sum(sum(len(s["dialogue"]) for s in c["sessions"]) for c in conversations)
    print(f"Total QA pairs: {total_qa}")
    print(f"Total dialogue turns: {total_turns}")
    print(f"Question types: {set(q['type'] for c in conversations for q in c['qa'])}")


if __name__ == "__main__":
    main()
