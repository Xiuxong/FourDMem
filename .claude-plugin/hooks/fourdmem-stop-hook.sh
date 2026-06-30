#!/bin/bash
# FourDMem Stop Hook — auto-archive agent response at end of each turn.
# Receives agent response via stdin.
# Path resolution: delegates everything to Python to avoid MSYS/Cygwin issues.
set -euo pipefail

# Resolve project root (Windows-aware)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if command -v cygpath &>/dev/null; then
    SCRIPT_DIR=$(cygpath -w "$SCRIPT_DIR")
fi
ROOT=""
if [ -f "$SCRIPT_DIR/../../Cargo.toml" ]; then
    ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
    command -v cygpath &>/dev/null && ROOT=$(cygpath -w "$ROOT")
fi

if [ -z "$PYTHON" ]; then
    echo "FourDMem: no Python found" >&2
    exit 1
fi

# Read stdin (agent response) — pass as base64 to avoid shell escaping
AGENT_RESPONSE_B64=""
if [ ! -t 0 ]; then
    AGENT_RESPONSE_B64=$(cat | "$PYTHON" -c "import sys,base64; print(base64.b64encode(sys.stdin.buffer.read()).decode())")
fi

if [ -z "$AGENT_RESPONSE_B64" ]; then
    exit 0

# Delegate all work to Python
"$PYTHON" -c "
import sys, os, json, base64

root = r'$ROOT'
sys.path.insert(0, os.path.join(root, 'python'))
db_path = os.path.join(root, 'data', 'vault', 'evidence.db')

if not os.path.exists(db_path):
    sys.stderr.write(f'FourDMem: db not found at {db_path}\n')
    sys.exit(1)

resp = base64.b64decode('$AGENT_RESPONSE_B64'.encode()).decode('utf-8', errors='replace')
if len(resp.strip()) < 20:
    sys.exit(0)
db_path = os.path.join(root, 'data', 'vault', 'evidence.db')
if not os.path.exists(db_path):
    sys.stderr.write(f'FourDMem: db not found at {db_path}\n')
    sys.exit(1)

# Decode response
resp = base64.b64decode('$AGENT_RESPONSE_B64'.encode()).decode('utf-8', errors='replace')
if len(resp.strip()) < 20:
    sys.exit(0)

try:
    import fourdmem
    engine = fourdmem.FourDMemEngine(db_path)
    engine.advance_tick()

    # Archive agent response
    from cognition.embed_utils import ingest_safely
    import uuid
    sid = f'auto-{uuid.uuid4().hex[:8]}'
    meta = json.dumps({'source': 'stop_hook', 'turn_type': 'final_answer'})
    ingest_safely(engine, sid, 'assistant', resp[:3000], meta, 'python')
    sys.stderr.write(f'FourDMem stop-hook: archived {len(resp)} chars\n')

    # L0->L1 extraction from recent 20 evidence
    import sqlite3
    conn = sqlite3.connect(db_path)
    rows = conn.execute('SELECT id, role, content FROM evidence ORDER BY id DESC LIMIT 20').fetchall()
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
                if f.get('label','') and len(f['label'].strip()) >= 8:
                    r = dedup.add_fact_with_dedup(engine, f['label'], f.get('l0_refs',[]))
                    if r.get('status') in ('added','merged'): n += 1
            if n:
                sys.stderr.write(f'FourDMem stop-hook: {n} L1 facts\n')

    engine.checkpoint(os.path.dirname(db_path))
    sys.stderr.write('FourDMem stop-hook OK\n')
except Exception as e:
    sys.stderr.write(f'FourDMem hook error: {e}\n')
    import traceback; traceback.print_exc(file=sys.stderr)
" 2>&1 || true
