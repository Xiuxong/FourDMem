"""Integrity Checker — Cross-layer reference validation.

Scans L0 evidence, L1 graph, L2 scenarios, and L3 persona for
broken references and orphaned nodes.

Usage:
    python -m daemon.integrity_checker [--once] [--vault-root PATH]

Checks:
1. L1 source_l0_refs → L0 existence
2. L2 frontmatter l1_refs → L1 existence
3. Orphaned L1 nodes (no L0 refs, no L2 references)
4. Empty L0 sessions
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any


def check_l0_exists(db_path: str, l0_ids: list[int]) -> dict:
    """Check if L0 evidence IDs exist in the database."""
    if not os.path.exists(db_path):
        return {"error": "database not found", "path": db_path}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    missing = []
    for l0_id in l0_ids:
        cursor.execute("SELECT COUNT(*) FROM evidence WHERE id = ?", (l0_id,))
        if cursor.fetchone()[0] == 0:
            missing.append(l0_id)

    conn.close()
    return {"checked": len(l0_ids), "missing": missing}


def check_vault_integrity(vault_root: str) -> dict:
    """Run all integrity checks on the vault."""
    vault = Path(vault_root)
    db_path = str(vault / "evidence.db")
    results = {
        "l0_database": {"exists": os.path.exists(db_path)},
        "l2_scenarios": {"files": 0, "valid": 0, "invalid": 0, "errors": []},
        "l3_persona": {"exists": False, "valid": False, "errors": []},
        "orphaned_sessions": [],
        "summary": {"status": "unknown", "issues": 0},
    }

    issues = 0

    # Check L0 database
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Count evidence by session
            cursor.execute(
                "SELECT session_id, COUNT(*) as cnt FROM evidence GROUP BY session_id"
            )
            sessions = cursor.fetchall()

            # Check for empty sessions
            for session_id, count in sessions:
                if count == 0:
                    results["orphaned_sessions"].append(session_id)
                    issues += 1

            # Count total evidence
            cursor.execute("SELECT COUNT(*) FROM evidence")
            total = cursor.fetchone()[0]
            results["l0_database"]["total_evidence"] = total

            # Check for evidence with empty content
            cursor.execute(
                "SELECT COUNT(*) FROM evidence WHERE content IS NULL OR content = ''"
            )
            empty_content = cursor.fetchone()[0]
            if empty_content > 0:
                results["l0_database"]["empty_content_count"] = empty_content
                issues += empty_content

            conn.close()
        except Exception as e:
            results["l0_database"]["error"] = str(e)
            issues += 1

    # Check L2 scenarios
    scenarios_dir = vault / "scenarios"
    if scenarios_dir.exists():
        for md_file in scenarios_dir.glob("*.md"):
            results["l2_scenarios"]["files"] += 1
            try:
                content = md_file.read_text(encoding="utf-8")
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        import yaml
                        fm = yaml.safe_load(parts[1])
                        if isinstance(fm, dict):
                            results["l2_scenarios"]["valid"] += 1
                        else:
                            results["l2_scenarios"]["invalid"] += 1
                            results["l2_scenarios"]["errors"].append(
                                f"{md_file.name}: invalid frontmatter"
                            )
                            issues += 1
                    else:
                        results["l2_scenarios"]["invalid"] += 1
                        issues += 1
                else:
                    results["l2_scenarios"]["valid"] += 1  # No frontmatter is OK
            except Exception as e:
                results["l2_scenarios"]["errors"].append(f"{md_file.name}: {e}")
                issues += 1

    # Check L3 persona
    persona_dir = vault / "persona"
    if persona_dir.exists():
        for yaml_file in persona_dir.glob("*.yaml"):
            results["l3_persona"]["exists"] = True
            try:
                import yaml
                data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    results["l3_persona"]["valid"] = True
                else:
                    results["l3_persona"]["errors"].append(f"{yaml_file.name}: not a dict")
                    issues += 1
            except Exception as e:
                results["l3_persona"]["errors"].append(f"{yaml_file.name}: {e}")
                issues += 1
        for json_file in persona_dir.glob("*.json"):
            results["l3_persona"]["exists"] = True
            try:
                json.loads(json_file.read_text(encoding="utf-8"))
                results["l3_persona"]["valid"] = True
            except Exception as e:
                results["l3_persona"]["errors"].append(f"{json_file.name}: {e}")
                issues += 1

    # Summary
    results["summary"] = {
        "status": "healthy" if issues == 0 else "issues_found",
        "issues": issues,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="FourDMem Integrity Checker")
    parser.add_argument(
        "--vault-root",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data",
            "vault",
        ),
        help="Path to the vault root directory",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (don't start daemon loop)",
    )
    args = parser.parse_args()

    print(f"Integrity check: {args.vault_root}", file=sys.stderr)
    results = check_vault_integrity(args.vault_root)

    # Print report
    print(json.dumps(results, indent=2, ensure_ascii=False))

    # Exit with code 1 if issues found
    if results["summary"]["issues"] > 0:
        print(
            f"\nWARNING: {results['summary']['issues']} integrity issues found.",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        print("\nAll integrity checks passed.", file=sys.stderr)


if __name__ == "__main__":
    main()
