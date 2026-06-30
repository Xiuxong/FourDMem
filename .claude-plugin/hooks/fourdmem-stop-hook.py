#!/usr/bin/env python3
"""FourDMem Stop Hook — auto-archive agent response at end of each turn.

Reads agent response from stdin, resolves project root by walking up from
this script's location, archives response to L0, extracts L1 facts, checkpoints.
"""
import sys, os, json, base64

# ── Resolve project root ────────────────────────────────────────────────────
script = os.path.abspath(__file__)
d = os.path.dirname(script)  # hooks/
d = os.path.dirname(d)        # .claude-plugin/
root = os.path.dirname(d)     # project root
if not os.path.exists(os.path.join(root, 'Cargo.toml')):
    # Walk up as fallback
    d = os.path.dirname(script)
    while d and d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, 'Cargo.toml')):
            root = d; break
        d = os.path.dirname(d)

sys.path.insert(0, os.path.join(root, 'python'))
db_path = os.path.join(root, 'data', 'vault', 'evidence.db')

if not os.path.exists(db_path):
    sys.stderr.write(f'FourDMem stop-hook: db not found at {db_path}\n')
    sys.exit(1)

# ── Read stdin ──────────────────────────────────────────────────────────────
resp_bytes = sys.stdin.buffer.read()
if not resp_bytes:
    sys.exit(0)
resp = resp_bytes.decode('utf-8', errors='replace').strip()
if len(resp) < 20:
    sys.exit(0)

# ── Archive + Extract + Checkpoint ──────────────────────────────────────────
try:
    import fourdmem
    engine = fourdmem.FourDMemEngine(db_path)
    engine.advance_tick()

    # Archive agent response to L0
    from cognition.embed_utils import ingest_safely
    import uuid
    sid = f'auto-{uuid.uuid4().hex[:8]}'
    meta = json.dumps({'source': 'stop_hook', 'turn_type': 'final_answer'})
    ingest_safely(engine, sid, 'assistant', resp[:3000], meta, 'python')

    # L0→L1 extraction
    import sqlite3
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        'SELECT id, role, content FROM evidence ORDER BY id DESC LIMIT 20'
    ).fetchall()
    conn.close()

    if len(rows) >= 2:
        from cognition.extractor import FactExtractor
        from cognition.dedup import SemanticDeduplicator
        facts = FactExtractor().extract_from_evidence(
            [{'id': r[0], 'role': r[1], 'content': r[2]} for r in rows]
        )
        if facts:
            dedup = SemanticDeduplicator()
            n = 0
            for f in facts[:8]:
                if f.get('label', '') and len(f['label'].strip()) >= 8:
                    r = dedup.add_fact_with_dedup(engine, f['label'], f.get('l0_refs', []))
                    if r.get('status') in ('added', 'merged'):
                        n += 1

    if engine.graph_node_count() > 0:
        engine.checkpoint(os.path.dirname(db_path))
    sys.stderr.write('FourDMem stop-hook OK\n')
except Exception as e:
    sys.stderr.write(f'FourDMem stop-hook error: {e}\n')
    import traceback; traceback.print_exc(file=sys.stderr)
    sys.exit(1)
