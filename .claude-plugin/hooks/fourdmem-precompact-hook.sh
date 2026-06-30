#!/bin/bash
# FourDMem PreCompact Hook — flush before context compaction
# Called by Claude Code before the context window is compacted.
# Ensures all buffered interactions are archived before context is lost.

set -euo pipefail

find_fourdmem_root() {
  local dir="$PWD"
  while [[ "$dir" != "/" ]]; do
    if [[ -f "$dir/Cargo.toml" ]] && [[ -d "$dir/python/mcp_server" ]]; then
      echo "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  return 1
}

FOURDMEM_ROOT="$(find_fourdmem_root)" || {
  echo "FourDMem: project root not found, skipping precompact flush" >&2
  exit 0
}

cd "$FOURDMEM_ROOT"
python -c "
import sys, json
sys.path.insert(0, 'python')
try:
    import fourdmem
    import os
    workspace = os.environ.get('FOURDMEM_WORKSPACE', os.path.basename(os.getcwd()).replace(' ', '_').lower())
    db_path = os.path.join('data', 'workspaces', workspace, 'evidence.db')
    if not os.path.exists(db_path):
        sys.exit(0)
    engine = fourdmem.FourDMemEngine(db_path)
    engine.advance_tick()

    # Auto-extract before context is lost
    from cognition.extractor import FactExtractor
    from cognition.salience import SalienceDetector
    import sqlite3

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT id, role, content FROM evidence ORDER BY id DESC LIMIT 20')
    rows = c.fetchall()
    conn.close()

    evidence_list = [{'id': r[0], 'role': r[1], 'content': r[2]} for r in rows]
    if len(evidence_list) >= 2:
        detector = SalienceDetector(threshold=1.5)
        combined = ' '.join(e['content'][:100] for e in evidence_list[:4])
        detector.check(combined)
        if detector.should_extract():
            extractor = FactExtractor()
            facts = extractor.extract_from_evidence(evidence_list)
            if facts:
                from cognition.dedup import SemanticDeduplicator
                dedup = SemanticDeduplicator()
                added = 0
                for fact in facts[:5]:
                    label = fact.get('label', '')
                    if not label or len(label.strip()) < 5:
                        continue
                    try:
                        dedup.add_fact_with_dedup(engine, label, fact.get('l0_refs'))
                        added += 1
                    except Exception:
                        pass
                if added > 0:
                    print(f'FourDMem precompact: auto-extracted {added} facts', file=sys.stderr)

    stats = json.loads(engine.wake_up())
    print(f'FourDMem precompact: flushed, tick={engine.get_tick()}', file=sys.stderr)
except Exception as e:
    print(f'FourDMem precompact skipped: {e}', file=sys.stderr)
" 2>&1 || true
