"""Merge Mnemopi bank memories into FourDMem vault + deduplicate.

Steps:
1. Read Mnemopi working_memory + facts from bank db
2. Insert into vault evidence.db as L0 evidence rows
3. Deduplicate by content hash within vault evidence
4. Tag test data (session-test, auto-xxx) as low priority
"""

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone

VAULT_DB = os.path.join(os.path.dirname(__file__), "..", "data", "vault", "evidence.db")
BANK_DB = os.path.join(
    os.path.dirname(__file__), "..",
    "data", "vault", "banks", "FourDMem-2jo73lz5tve59", "mnemopi.db",
)


def merge_mnemopi_to_vault(bank_path: str, vault_path: str) -> int:
    """Read Mnemopi memories and insert into vault evidence table."""
    bank = sqlite3.connect(bank_path)
    vault = sqlite3.connect(vault_path)

    # Read working_memory
    try:
        wm_rows = bank.execute(
            "SELECT content, source, timestamp, session_id, importance, metadata_json, created_at FROM working_memory ORDER BY id"
        ).fetchall()
        print(f"  Working memory: {len(wm_rows)} rows")
    except Exception:
        wm_rows = []

    # Read facts
    try:
        fact_rows = bank.execute(
            "SELECT object as content, created_at FROM facts ORDER BY fact_id"
        ).fetchall()
        print(f"  Facts: {len(fact_rows)} rows")
    except Exception:
        fact_rows = []

    inserted = 0
    ws_id = "fourdmem"  # workspace tag for merged data

    for row in wm_rows:
        content, source, ts, session_id, importance, meta_json, created = row
        content = content[:2000] if content else ""
        if len(content) < 30:
            continue

        # Build metadata
        try:
            meta = json.loads(meta_json) if meta_json else {}
        except Exception:
            meta = {}

        # Check if this content already exists in vault
        content_hash = hashlib.md5(content[:500].encode()).hexdigest()
        existing = vault.execute(
            "SELECT COUNT(*) FROM evidence WHERE content LIKE ?",
            (content[:100] + "%",),
        ).fetchone()[0]
        if existing > 0:
            continue  # Skip duplicate

        vault.execute(
            """INSERT INTO evidence (workspace_id, session_id, role, content, metadata, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                ws_id,
                f"mnemopi-{session_id[:12] if session_id else 'unknown'}",
                "agent",
                content,
                json.dumps({"merged_from": "mnemopi", "source": source, "importance": importance, **meta}),
                ts or created or datetime.now(timezone.utc).isoformat(),
            ),
        )
        inserted += 1

    # Insert facts as tool evidence
    for row in fact_rows:
        content, created = row
        if not content or len(content) < 20:
            continue
        content = content[:2000]

        content_hash = hashlib.md5(content[:500].encode()).hexdigest()
        existing = vault.execute(
            "SELECT COUNT(*) FROM evidence WHERE content LIKE ?",
            (content[:100] + "%",),
        ).fetchone()[0]
        if existing > 0:
            continue

        vault.execute(
            """INSERT INTO evidence (workspace_id, session_id, role, content, metadata, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                ws_id,
                "mnemopi-facts",
                "tool",
                content,
                json.dumps({"merged_from": "mnemopi", "type": "fact"}),
                created or datetime.now(timezone.utc).isoformat(),
            ),
        )
        inserted += 1

    vault.commit()
    bank.close()
    vault.close()
    return inserted


def dedup_vault(vault_path: str) -> tuple:
    """Deduplicate vault evidence by content similarity.

    Strategy:
    - Group by content prefix (first 200 chars)
    - Keep the one with highest metadata richness
    - Mark duplicates as superseded
    """
    vault = sqlite3.connect(vault_path)

    # Find potential duplicates
    dupes = vault.execute("""
        SELECT SUBSTR(content, 1, 200) as prefix, COUNT(*) as cnt, GROUP_CONCAT(id)
        FROM evidence
        GROUP BY prefix
        HAVING cnt > 1
    """).fetchall()

    removed = 0
    kept = 0
    for prefix, cnt, id_list in dupes:
        ids = [int(x) for x in id_list.split(",")]
        # Find the best row to keep (longest content, richest metadata)
        best = None
        best_score = -1
        for eid in ids:
            row = vault.execute(
                "SELECT id, LENGTH(content) as clen, LENGTH(metadata) as mlen FROM evidence WHERE id=?",
                (eid,),
            ).fetchone()
            if row:
                score = row[1] + row[2] * 2  # Content len + 2x metadata len
                if score > best_score:
                    best_score = score
                    best = row[0]

        # Delete others
        for eid in ids:
            if eid != best:
                vault.execute("DELETE FROM evidence WHERE id=?", (eid,))
                removed += 1
        kept += 1

    vault.commit()

    # Also clean FTS5
    vault.execute("INSERT INTO evidence_fts(evidence_fts) VALUES('rebuild')")

    total = vault.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
    vault.close()
    return removed, kept, total


def tag_test_data(vault_path: str) -> int:
    """Tag session-test and auto-xxx rows as test data."""
    vault = sqlite3.connect(vault_path)
    tagged = 0

    # Tag test sessions
    for pattern in ["session-test%", "auto-%"]:
        rows = vault.execute(
            "SELECT id, metadata FROM evidence WHERE session_id LIKE ? AND metadata NOT LIKE '%\"test\": true%'",
            (pattern,),
        ).fetchall()
        for row_id, meta_str in rows:
            try:
                meta = json.loads(meta_str) if meta_str else {}
            except Exception:
                meta = {}
            meta["test"] = True
            vault.execute("UPDATE evidence SET metadata=? WHERE id=?", (json.dumps(meta), row_id))
            tagged += 1

    vault.commit()
    vault.close()
    return tagged


if __name__ == "__main__":
    vault = os.path.abspath(VAULT_DB)
    bank = os.path.abspath(BANK_DB)

    print("=== FourDMem Memory Consolidation ===\n")

    # 1. Merge Mnemopi → Vault
    print(f"1. Merging Mnemopi bank → vault...")
    if os.path.exists(bank):
        merged = merge_mnemopi_to_vault(bank, vault)
        print(f"   ✅ Merged {merged} new rows")
    else:
        print(f"   ⚠️  Bank not found at: {bank}")

    # 2. Deduplicate vault
    print(f"\n2. Deduplicating vault by content...")
    removed, kept, total = dedup_vault(vault)
    print(f"   🗑️  Removed {removed} duplicate rows")
    print(f"   📋 Kept {kept} groups ({total} total rows)")

    # 3. Tag test data
    print(f"\n3. Tagging test/session-test data...")
    tagged = tag_test_data(vault)
    print(f"   🏷️  Tagged {tagged} test data rows")

    print(f"\n=== Done ===")
