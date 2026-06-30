"""Migrate vault evidence.db to new workspace-aware schema.

Run once after upgrading to workspace_id architecture.
- Adds workspace_id='default' to existing rows (via Rust ALTER TABLE)
- Creates workspaces table
- Inserts default workspace entry
- Optionally removes old workspace evidence.db files
"""

import sqlite3
import os
import shutil

VAULT_DB = os.path.join(os.path.dirname(__file__), "..", "data", "vault", "evidence.db")
WORKSPACES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "workspaces")


def migrate_vault(db_path: str):
    """Apply schema migration to vault evidence.db."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Check if workspace_id column exists
    cur.execute("PRAGMA table_info(evidence)")
    columns = [row[1] for row in cur.fetchall()]

    if "workspace_id" not in columns:
        print("Adding workspace_id column to evidence table...")
        cur.execute("ALTER TABLE evidence ADD COLUMN workspace_id TEXT NOT NULL DEFAULT 'default'")
        conn.commit()
        print("  ✓ workspace_id column added")
    else:
        print("  ✓ workspace_id column already exists")

    # Create workspaces table if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS workspaces (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            config      TEXT DEFAULT '{}'
        )
    """)
    conn.commit()
    print("  ✓ workspaces table ready")

    # Insert default workspace if not exists
    cur.execute("SELECT COUNT(*) FROM workspaces")
    if cur.fetchone()[0] == 0:
        cur.execute(
            "INSERT INTO workspaces (id, name) VALUES (?, ?)",
            ("default", "Default Workspace"),
        )
        conn.commit()
        print("  ✓ default workspace created")

    # Create workspace index if not exists
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_evidence_workspace
        ON evidence(workspace_id, timestamp)
    """)
    conn.commit()
    print("  ✓ workspace index ready")

    # Summary
    cur.execute("SELECT COUNT(*) FROM evidence")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM evidence WHERE workspace_id='default'")
    migrated = cur.fetchone()[0]
    print(f"\nMigration complete: {migrated}/{total} rows tagged as workspace 'default'")

    conn.close()


def cleanup_old_workspaces(workspaces_dir: str):
    """Remove old workspace evidence.db files (data now in vault)."""
    if not os.path.exists(workspaces_dir):
        print(f"No workspaces directory at {workspaces_dir}")
        return

    for entry in os.listdir(workspaces_dir):
        entry_path = os.path.join(workspaces_dir, entry)
        if os.path.isdir(entry_path):
            db_file = os.path.join(entry_path, "evidence.db")
            if os.path.exists(db_file):
                size = os.path.getsize(db_file)
                print(f"  Removing {db_file} ({size} bytes)")
                os.remove(db_file)
                # Remove -shm and -wal too
                for suffix in ["-shm", "-wal"]:
                    f = db_file + suffix
                    if os.path.exists(f):
                        os.remove(f)

    print("  ✓ Old workspace evidence.db files removed")


if __name__ == "__main__":
    print("=== FourDMem Vault Migration ===\n")

    vault = os.path.abspath(VAULT_DB)
    if os.path.exists(vault):
        print(f"Migrating: {vault}")
        migrate_vault(vault)
    else:
        print(f"Vault DB not found at {vault}, skipping migration")

    print(f"\nCleaning up old workspace DBs...")
    cleanup_old_workspaces(os.path.abspath(WORKSPACES_DIR))

    print("\nDone.")
