//! L0 raw evidence storage (SQLite + FTS5, append-only)
//!
//! This is the "truth substrate" of the four-dimensional memory system.
//! All evidence is append-only — no UPDATE or DELETE is ever issued.

use std::path::Path;
use std::sync::Mutex;

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Pre-tokenize CJK text with jieba for FTS5 BM25 matching.
/// Filters English stop words to prevent BM25 dilution.
fn tokenize_cjk(text: &str) -> String {
    static STOP_WORDS: &[&str] = &[
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "just",
        "because", "but", "and", "or", "if", "while", "what", "which", "who",
        "whom", "this", "that", "these", "those", "i", "me", "my", "we", "our",
        "you", "your", "he", "him", "his", "she", "her", "it", "its", "they",
        "them", "their", "about", "up", "down", "get", "got", "also",
    ];
    let jieba = jieba_rs::Jieba::new();
    let tokens: Vec<&str> = jieba.cut(text, false)
        .iter()
        .map(|t| t.word)
        .filter(|w| w.len() > 1 && !STOP_WORDS.contains(&w.to_lowercase().as_str()))
        .collect();
    tokens.join(" ")
}

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum L0Error {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

// ── Data types ────────────────────────────────────────────────────────────────

/// A single row of raw evidence, as returned by queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub id: i64,
    pub workspace_id: String,
    pub model_name: String,
    pub session_id: String,
    pub timestamp: String,
    pub role: String,
    pub content: String,
    pub metadata: serde_json::Value,
}

/// Aggregate statistics about the L0 store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L0Stats {
    pub total_count: i64,
    pub earliest: Option<String>,
    pub latest: Option<String>,
}

// ── Store ─────────────────────────────────────────────────────────────────────

/// Append-only SQLite + FTS5 evidence store.
///
/// Thread-safe: interior `Connection` is behind `std::sync::Mutex`.
pub struct L0Store {
    conn: Mutex<Connection>,
}

// SAFETY: Mutex<Connection> provides thread-safety; Connection itself is Send but
// not Sync, so we must not impl Sync without the Mutex.
unsafe impl Send for L0Store {}
unsafe impl Sync for L0Store {}

impl L0Store {
    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /// Open (or create) an L0 evidence database at `path`.
    ///
    /// Creates the `evidence` table, `evidence_fts` virtual table,
    /// and the session+time index if they do not already exist.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, L0Error> {
        let conn = Connection::open(path)?;

        // Enable WAL mode for better concurrent read performance.
        // PRAGMA journal_mode returns a result set, so use query.
        let _: String = conn.query_row("PRAGMA journal_mode=WAL;", [], |row| row.get(0))?;
        // Main evidence table — append-only.
        // workspace_id scopes evidence to a project for isolation.
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS evidence (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_id TEXT   NOT NULL DEFAULT 'default',
                model_name  TEXT    NOT NULL DEFAULT 'unknown',
                session_id  TEXT    NOT NULL,
                timestamp   TEXT    NOT NULL DEFAULT (datetime('now')),
                role        TEXT    NOT NULL,
                content     TEXT    NOT NULL,
                metadata    TEXT    DEFAULT '{}'
            );",
        )?;

        // Migrate: add workspace_id column to existing tables missing it.
        let has_ws_col: bool = conn
            .prepare("SELECT COUNT(*) FROM pragma_table_info('evidence') WHERE name='workspace_id'")?
            .query_row([], |row| row.get::<_, i64>(0))?
            > 0;
        if !has_ws_col {
            conn.execute_batch(
                "ALTER TABLE evidence ADD COLUMN workspace_id TEXT NOT NULL DEFAULT 'default';",
            )?;
        }

        // Migrate: add model_name column to existing tables missing it.
        let has_mn_col: bool = conn
            .prepare("SELECT COUNT(*) FROM pragma_table_info('evidence') WHERE name='model_name'")?
            .query_row([], |row| row.get::<_, i64>(0))?
            > 0;
        if !has_mn_col {
            conn.execute_batch(
                "ALTER TABLE evidence ADD COLUMN model_name TEXT NOT NULL DEFAULT 'unknown';",
            )?;
        }

        // Index for model-scoped queries.
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_evidence_model
             ON evidence(model_name, timestamp);",
        )?;

        // FTS5 full-text index on content.
        conn.execute_batch(
            "CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts
             USING fts5(content, content_rowid='id', content='evidence');",
        )?;

        // Composite index for time-range queries scoped to a session.
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_evidence_session_time
             ON evidence(session_id, timestamp);",
        )?;

        // Index for workspace-scoped queries.
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_evidence_workspace
             ON evidence(workspace_id, timestamp);",
        )?;
        // Workspace metadata table.
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS workspaces (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                config      TEXT DEFAULT '{}'
            );",
        )?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Convenience: open an in-memory database (for testing).
    pub fn open_memory() -> Result<Self, L0Error> {
        Self::open(":memory:")
    }

    // ── Writes ────────────────────────────────────────────────────────────────

    /// Append a single piece of evidence.
    ///
    pub fn append(
        &self,
        workspace_id: &str,
        model_name: &str,
        session_id: &str,
        role: &str,
        content: &str,
        metadata: &serde_json::Value,
    ) -> Result<i64, L0Error> {
        let conn = self.conn.lock().expect("l0: mutex poisoned");
        let meta_str = serde_json::to_string(metadata)?;

        conn.execute(
            "INSERT INTO evidence (workspace_id, model_name, session_id, role, content, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![workspace_id, model_name, session_id, role, content, meta_str],
        )?;

        let rowid = conn.last_insert_rowid();

        conn.execute(
            "INSERT INTO evidence_fts(rowid, content) VALUES (?1, ?2)",
            params![rowid, tokenize_cjk(content)],
        )?;

        Ok(rowid)
    }

    // ── Reads ─────────────────────────────────────────────────────────────────

    /// Full-text search over all evidence.
    ///
    /// Results are ranked by BM25 relevance (most relevant first) and
    /// limited to `limit` rows. If `workspace_id` is Some, results are scoped.
    pub fn search(
        &self,
        query: &str,
        limit: i64,
        workspace_id: Option<&str>,
    ) -> Result<Vec<Evidence>, L0Error> {
        let conn = self.conn.lock().expect("l0: mutex poisoned");
        let tokenized_query = tokenize_cjk(query);
        let ws = workspace_id.unwrap_or("");
        let sql = if ws.is_empty() {
            "SELECT e.id, e.workspace_id, e.model_name, e.session_id, e.timestamp, e.role, e.content, e.metadata
             FROM evidence e
             INNER JOIN evidence_fts f ON f.rowid = e.id
             WHERE evidence_fts MATCH ?1
             ORDER BY bm25(evidence_fts)
             LIMIT ?2"
        } else {
            "SELECT e.id, e.workspace_id, e.model_name, e.session_id, e.timestamp, e.role, e.content, e.metadata
             FROM evidence e
             INNER JOIN evidence_fts f ON f.rowid = e.id
             WHERE evidence_fts MATCH ?1 AND e.workspace_id = ?3
             ORDER BY bm25(evidence_fts)
             LIMIT ?2"
        };
        let mut stmt = conn.prepare(sql)?;

        let rows: Vec<Evidence> = if ws.is_empty() {
            stmt.query_map(params![tokenized_query, limit], |row| {
                Ok(Evidence {
                    id: row.get(0)?,
                    workspace_id: row.get(1)?,
                    model_name: row.get(2)?,
                    session_id: row.get(3)?,
                    timestamp: row.get(4)?,
                    role: row.get(5)?,
                    content: row.get(6)?,
                    metadata: serde_json::from_str(&row.get::<_, String>(7)?)
                        .unwrap_or(serde_json::Value::Null),
                })
            })?.collect::<Result<Vec<_>, _>>().map_err(L0Error::from)?
        } else {
            stmt.query_map(params![tokenized_query, limit, ws], |row| {
                Ok(Evidence {
                    id: row.get(0)?,
                    workspace_id: row.get(1)?,
                    model_name: row.get(2)?,
                    session_id: row.get(3)?,
                    timestamp: row.get(4)?,
                    role: row.get(5)?,
                    content: row.get(6)?,
                    metadata: serde_json::from_str(&row.get::<_, String>(7)?)
                        .unwrap_or(serde_json::Value::Null),
                })
            })?.collect::<Result<Vec<_>, _>>().map_err(L0Error::from)?
        };

        Ok(rows)
    }

    /// Look up a single evidence row by its primary key.
    ///
    /// Returns `None` if no row with the given ID exists.
    pub fn get_by_id(&self, id: i64) -> Result<Option<Evidence>, L0Error> {
        let conn = self.conn.lock().expect("l0: mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, workspace_id, model_name, session_id, timestamp, role, content, metadata
             FROM evidence WHERE id = ?1",
        )?;

        let mut rows = stmt.query_map(params![id], |row| {
            Ok(Evidence {
                id: row.get(0)?,
                workspace_id: row.get(1)?,
                    model_name: row.get(2)?,
                session_id: row.get(3)?,
                timestamp: row.get(4)?,
                role: row.get(5)?,
                content: row.get(6)?,
                metadata: serde_json::from_str(&row.get::<_, String>(7)?).unwrap_or(serde_json::Value::Null),
            })        })?;

        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }
    /// Get the content length for a single evidence row by its primary key.
    ///
    /// Returns 0 if no row with the given ID exists.
    pub fn get_content_length(&self, id: i64) -> Result<usize, L0Error> {
        let conn = self.conn.lock().expect("l0: mutex poisoned");
        let len: usize = conn.query_row(
            "SELECT COALESCE(length(content), 0) FROM evidence WHERE id = ?1",
            params![id],
            |row| row.get(0),
        )?;
        Ok(len)
    }

    /// Time-range query scoped to a session.
    ///
    /// `from` and `to` are ISO-8601 datetime strings (e.g. `"2025-01-01T00:00:00"`).
    /// Results are ordered by timestamp descending (newest first).
    pub fn search_by_time(
        &self,
        workspace_id: &str,
        session_id: &str,
        from: &str,
        to: &str,
        limit: i64,
    ) -> Result<Vec<Evidence>, L0Error> {
        let conn = self.conn.lock().expect("l0: mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, workspace_id, model_name, session_id, timestamp, role, content, metadata
             FROM evidence
             WHERE workspace_id = ?1 AND session_id = ?2
               AND timestamp BETWEEN ?3 AND ?4
             ORDER BY timestamp DESC
             LIMIT ?5",
        )?;

        let rows = stmt.query_map(params![workspace_id, session_id, from, to, limit], |row| {
            Ok(Evidence {
                id: row.get(0)?,
                workspace_id: row.get(1)?,
                    model_name: row.get(2)?,
                session_id: row.get(3)?,
                timestamp: row.get(4)?,
                role: row.get(5)?,
                content: row.get(6)?,
                metadata: serde_json::from_str(&row.get::<_, String>(7)?).unwrap_or(serde_json::Value::Null),
            })        })?;

        rows.collect::<Result<Vec<_>, _>>().map_err(L0Error::from)
    }

    /// Retrieve all evidence for a given session, ordered by timestamp ascending.
    pub fn search_by_session(
        &self,
        workspace_id: &str,
        session_id: &str,
        limit: i64,
    ) -> Result<Vec<Evidence>, L0Error> {
        let conn = self.conn.lock().expect("l0: mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, workspace_id, model_name, session_id, timestamp, role, content, metadata
             FROM evidence
             WHERE workspace_id = ?1 AND session_id = ?2
             ORDER BY timestamp ASC
             LIMIT ?3",
        )?;

        let rows = stmt.query_map(params![workspace_id, session_id, limit], |row| {
            Ok(Evidence {
                id: row.get(0)?,
                workspace_id: row.get(1)?,
                    model_name: row.get(2)?,
                session_id: row.get(3)?,
                timestamp: row.get(4)?,
                role: row.get(5)?,
                content: row.get(6)?,
                metadata: serde_json::from_str(&row.get::<_, String>(7)?).unwrap_or(serde_json::Value::Null),
            })        })?;

        rows.collect::<Result<Vec<_>, _>>().map_err(L0Error::from)
    }

    /// Retrieve all evidence rows (for index rebuild), up to `limit`.
    pub fn get_all(&self, limit: i64) -> Result<Vec<Evidence>, L0Error> {
        let conn = self.conn.lock().expect("l0: mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, workspace_id, model_name, session_id, timestamp, role, content, metadata
             FROM evidence ORDER BY id LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit], |row| {
            Ok(Evidence {
                id: row.get(0)?,
                workspace_id: row.get(1)?,
                    model_name: row.get(2)?,
                session_id: row.get(3)?,
                timestamp: row.get(4)?,
                role: row.get(5)?,
                content: row.get(6)?,
                metadata: serde_json::from_str(&row.get::<_, String>(7)?)
                    .unwrap_or(serde_json::Value::Null),
            })        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(L0Error::from)
    }

    /// Check if an evidence row with the given `id` exists.
    pub fn exists(&self, id: i64) -> Result<bool, L0Error> {
        let conn = self.conn.lock().expect("l0: mutex poisoned");
        let mut stmt = conn.prepare("SELECT 1 FROM evidence WHERE id = ?1 LIMIT 1")?;
        let mut rows = stmt.query_map(params![id], |_| Ok(()))?;
        Ok(rows.next().is_some())
    }

    /// Return aggregate statistics: total row count and earliest/latest timestamps.
    pub fn stats(&self) -> Result<L0Stats, L0Error> {
        let conn = self.conn.lock().expect("l0: mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT COUNT(*) AS total,
                    MIN(timestamp) AS earliest,
                    MAX(timestamp) AS latest
             FROM evidence",
        )?;

        let row = stmt.query_row([], |row| {
            Ok(L0Stats {
                total_count: row.get(0)?,
                earliest: row.get(1)?,
                latest: row.get(2)?,
            })
        })?;

        Ok(row)
    }

    // ── Workspace management ──────────────────────────────────────────────────

    /// Create or update a workspace.
    pub fn upsert_workspace(
        &self,
        id: &str,
        name: &str,
        config: Option<&str>,
    ) -> Result<(), L0Error> {
        let conn = self.conn.lock().expect("l0: mutex poisoned");
        conn.execute(
            "INSERT INTO workspaces (id, name, config) VALUES (?1, ?2, ?3)
             ON CONFLICT(id) DO UPDATE SET name=?2, config=?3",
            params![id, name, config.unwrap_or("{}")],
        )?;
        Ok(())
    }

    /// List all workspaces.
    pub fn list_workspaces(&self) -> Result<Vec<(String, String, String)>, L0Error> {
        let conn = self.conn.lock().expect("l0: mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, name, created_at FROM workspaces ORDER BY created_at DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(L0Error::from)
    }

    /// Delete a workspace and all its evidence.
    pub fn delete_workspace(&self, workspace_id: &str) -> Result<usize, L0Error> {
        let conn = self.conn.lock().expect("l0: mutex poisoned");
        let deleted = conn.execute(
            "DELETE FROM evidence WHERE workspace_id = ?1",
            params![workspace_id],
        )?;
        conn.execute(
            "DELETE FROM workspaces WHERE id = ?1",
            params![workspace_id],
        )?;
        Ok(deleted)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_and_stats_empty() {
        let store = L0Store::open_memory().expect("open_memory failed");
        let stats = store.stats().expect("stats failed");
        assert_eq!(stats.total_count, 0);
        assert!(stats.earliest.is_none());
        assert!(stats.latest.is_none());
    }

    #[test]
    fn test_append_and_search() {
        let store = L0Store::open_memory().expect("open_memory failed");

        let id = store
            .append("test", "test-model", "s1", "user", "Hello world", &serde_json::Value::Null)
            .expect("append failed");

        let id2 = store
            .append("test", "test-model", "s1", "assistant", "Hi there, how can I help?", &serde_json::Value::Null)
            .expect("append failed");
        assert!(id2 > id);

        let results = store.search("hello", 10, None).expect("search failed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "Hello world");
        assert_eq!(results[0].role, "user");
    }

    #[test]
    fn test_search_by_time() {
        let store = L0Store::open_memory().expect("open_memory failed");

        // Insert evidence with explicit timestamps
        let conn = store.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO evidence (workspace_id, model_name, session_id, timestamp, role, content)
             VALUES ('test', 'test-model', 's1', '2025-06-01T10:00:00', 'user', 'first message')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO evidence (workspace_id, model_name, session_id, timestamp, role, content)
             VALUES ('test', 'test-model', 's1', '2025-06-01T12:00:00', 'user', 'second message')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO evidence (workspace_id, model_name, session_id, timestamp, role, content)
             VALUES ('test', 'test-model', 's2', '2025-06-01T11:00:00', 'user', 'other session')",
            [],
        )
        .unwrap();
        drop(conn);

        let results = store
            .search_by_time("test", "s1", "2025-06-01T09:00:00", "2025-06-01T13:00:00", 10)
            .expect("search_by_time failed");

        assert_eq!(results.len(), 2);
        // Descending order: second message first
        assert_eq!(results[0].content, "second message");
        assert_eq!(results[1].content, "first message");

        // s2 should not appear
        let results_s2 = store
            .search_by_time("test", "s2", "2025-06-01T09:00:00", "2025-06-01T13:00:00", 10)
            .expect("search_by_time failed");
        assert_eq!(results_s2.len(), 1);
        assert_eq!(results_s2[0].content, "other session");
    }

    #[test]
    fn test_append_with_metadata() {
        let store = L0Store::open_memory().expect("open_memory failed");

        let meta = serde_json::json!({"source": "cli", "priority": 1});
        let id = store
            .append("test", "test-model", "s1", "tool", "tool output", &meta)
            .expect("append failed");

        let results = store.search("tool output", 10, None).expect("search failed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].metadata["source"], "cli");
        assert_eq!(results[0].metadata["priority"], 1);
        let _ = id;
    }
}
