//! Blake3 incremental sync engine
//!
//! Tracks file hashes to detect changes without re-reading full content.
//! Used by the sync daemon to efficiently identify which L2/L3 files
//! need re-parsing after a bulk file-system event.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use thiserror::Error;

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum SyncError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

// ── Sync engine ───────────────────────────────────────────────────────────────

/// Tracks file content hashes for incremental change detection.
pub struct SyncEngine {
    root: PathBuf,
    /// Map from relative file path to its last-known Blake3 hex hash.
    hashes: HashMap<PathBuf, String>,
}

impl SyncEngine {
    /// Create a new sync engine rooted at `root`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            hashes: HashMap::new(),
        }
    }

    /// Compute the Blake3 hex hash of a file's contents.
    pub fn hash_file(&self, path: impl AsRef<Path>) -> Result<String, SyncError> {
        let data = std::fs::read(path)?;
        Ok(blake3::hash(&data).to_hex().to_string())
    }

    /// Check whether a file has changed since the last recorded hash.
    ///
    /// Returns `true` if the file is new, modified, or unreadable.
    /// If `update` is true, the stored hash is updated to the current value.
    pub fn needs_update(&mut self, rel_path: &Path, update: bool) -> bool {
        let abs_path = self.root.join(rel_path);

        let current_hash = match self.hash_file(&abs_path) {
            Ok(h) => h,
            Err(_) => return true, // can't read → treat as changed
        };

        let changed = match self.hashes.get(rel_path) {
            Some(stored) => stored != &current_hash,
            None => true, // new file
        };

        if changed && update {
            self.hashes.insert(rel_path.to_path_buf(), current_hash);
        }

        changed
    }

    /// Record a hash for a file (useful for initial seeding).
    pub fn set_hash(&mut self, rel_path: PathBuf, hash: String) {
        self.hashes.insert(rel_path, hash);
    }

    /// Number of tracked files.
    pub fn tracked_count(&self) -> usize {
        self.hashes.len()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_file_and_needs_update() {
        let dir = std::env::temp_dir().join("sync_engine_test");
        let _ = std::fs::create_dir_all(&dir);
        let file_path = dir.join("test.txt");
        std::fs::write(&file_path, "hello world").unwrap();

        let mut engine = SyncEngine::new(&dir);

        // First check: new file → needs update
        assert!(engine.needs_update(Path::new("test.txt"), true));

        // Second check: same content → no update needed
        assert!(!engine.needs_update(Path::new("test.txt"), false));

        // Modify the file
        std::fs::write(&file_path, "hello world!").unwrap();
        assert!(engine.needs_update(Path::new("test.txt"), true));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
