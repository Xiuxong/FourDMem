//! L2 scenario / episode storage (Markdown with YAML frontmatter)
//!
//! L2 holds **scenarios** — consolidated knowledge fragments that aggregate
//! multiple L1 graph facts into human-readable, conditionally-activatable
//! documents. Each scenario is a Markdown file with YAML frontmatter,
//! parsed/written via `sync_engine::ScenarioBlock`.
//!
//! ## Storage layout
//!
//! Scenarios are stored as `data/vault/scenarios/*.md` files. The [`L2Store`]
//! maintains an in-memory index keyed by scenario ID for O(1) lookup, and
//! provides methods for searching by condition keywords or L1 references.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use sync_engine::{L2Parser, ScenarioBlock};
use thiserror::Error;

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum L2Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("scenario '{id}' not found")]
    NotFound { id: String },

    #[error("YAML serialization error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("parse error: {0}")]
    Parse(String),
}

// ── L2Store ───────────────────────────────────────────────────────────────────

/// In-memory index of L2 scenario blocks, backed by `.md` files on disk.
pub struct L2Store {
    /// Root directory for scenario files (e.g. `data/vault/scenarios/`).
    root: PathBuf,

    /// In-memory scenario index by ID.
    scenarios: HashMap<String, ScenarioBlock>,
}

impl L2Store {
    /// Create a new L2Store rooted at the given directory.
    ///
    /// Loads all existing `.md` files from `root` into memory.
    pub fn load(root: impl AsRef<Path>) -> Result<Self, L2Error> {
        let root = root.as_ref().to_path_buf();
        let mut scenarios = HashMap::new();

        if root.exists() {
            for entry in std::fs::read_dir(&root)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().map(|e| e == "md").unwrap_or(false) {
                    match L2Parser::parse_file(&path) {
                        Ok(block) => {
                            let id = if block.id.is_empty() {
                                path.file_stem()
                                    .and_then(|s| s.to_str())
                                    .unwrap_or("unknown")
                                    .to_string()
                            } else {
                                block.id.clone()
                            };
                            scenarios.insert(id, block);
                        }
                        Err(e) => {
                            eprintln!("L2Store: skipping {:?}: {}", path, e);
                        }
                    }
                }
            }
        }

        Ok(Self { root, scenarios })
    }

    /// Create an L2Store with a temporary in-memory root (for testing).
    pub fn new_temp(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            scenarios: HashMap::new(),
        }
    }

    /// Number of loaded scenarios.
    pub fn len(&self) -> usize {
        self.scenarios.len()
    }

    /// Whether the store contains zero scenarios.
    pub fn is_empty(&self) -> bool {
        self.scenarios.is_empty()
    }

    // ── Reads ─────────────────────────────────────────────────────────────────

    /// Get a scenario by ID.
    pub fn get(&self, id: &str) -> Option<&ScenarioBlock> {
        self.scenarios.get(id)
    }

    /// List all scenario IDs.
    pub fn ids(&self) -> Vec<&String> {
        self.scenarios.keys().collect()
    }

    /// Search scenarios whose `conditions` contain any of the given keywords
    /// (case-insensitive substring match).
    pub fn search_by_conditions(&self, keywords: &[&str]) -> Vec<&ScenarioBlock> {
        let lower_keywords: Vec<String> = keywords.iter().map(|k| k.to_lowercase()).collect();
        self.scenarios
            .values()
            .filter(|block| {
                block.conditions.iter().any(|cond| {
                    let lower_cond = cond.to_lowercase();
                    lower_keywords.iter().any(|kw| lower_cond.contains(kw.as_str()))
                })
            })
            .collect()
    }

    /// Search scenarios that reference a specific L1 graph node index.
    pub fn search_by_l1_ref(&self, l1_ref: usize) -> Vec<&ScenarioBlock> {
        self.scenarios
            .values()
            .filter(|block| block.source_l1_refs.contains(&l1_ref))
            .collect()
    }

    // ── Writes ────────────────────────────────────────────────────────────────

    /// Save a scenario to disk and update the in-memory index.
    ///
    /// If the scenario already exists (by ID), it is overwritten.
    pub fn save(&mut self, block: ScenarioBlock) -> Result<(), L2Error> {
        let id = if block.id.is_empty() {
            // Generate an ID from the title or a timestamp
            if !block.title.is_empty() {
                slugify(&block.title)
            } else {
                format!("scenario-{}", chrono::Utc::now().timestamp_millis())
            }
        } else {
            block.id.clone()
        };

        let mut block = block;
        block.id = id.clone();

        // Ensure root directory exists
        std::fs::create_dir_all(&self.root)?;

        // Write to disk
        let path = self.root.join(format!("{}.md", id));
        let content = format_scenario_md(&block);
        std::fs::write(&path, content)?;

        // Update in-memory index
        self.scenarios.insert(id, block);

        Ok(())
    }

    /// Remove a scenario by ID, deleting its file from disk.
    pub fn remove(&mut self, id: &str) -> Result<(), L2Error> {
        let path = self.root.join(format!("{}.md", id));
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        self.scenarios.remove(id);
        Ok(())
    }

    /// Reload all scenarios from disk, replacing the in-memory index.
    pub fn reload(&mut self) -> Result<(), L2Error> {
        self.scenarios.clear();
        if self.root.exists() {
            for entry in std::fs::read_dir(&self.root)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().map(|e| e == "md").unwrap_or(false) {
                    match L2Parser::parse_file(&path) {
                        Ok(block) => {
                            let id = if block.id.is_empty() {
                                path.file_stem()
                                    .and_then(|s| s.to_str())
                                    .unwrap_or("unknown")
                                    .to_string()
                            } else {
                                block.id.clone()
                            };
                            self.scenarios.insert(id, block);
                        }
                        Err(e) => {
                            eprintln!("L2Store: skipping {:?}: {}", path, e);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Format a [`ScenarioBlock`] as a Markdown string with YAML frontmatter.
fn format_scenario_md(block: &ScenarioBlock) -> String {
    let mut out = String::new();

    // YAML frontmatter
    out.push_str("---\n");
    if let Ok(yaml) = serde_yaml::to_string(block) {
        out.push_str(&yaml);
    }
    out.push_str("---\n\n");

    // Body
    if !block.body.is_empty() {
        out.push_str(&block.body);
        out.push('\n');
    }

    out
}

/// Convert a title string to a URL/file-safe slug.
fn slugify(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_block(id: &str, title: &str) -> ScenarioBlock {
        ScenarioBlock {
            id: id.into(),
            title: title.into(),
            body: String::new(),
            conditions: Vec::new(),
            valence: 0.0,
            active_ticks: 0,
            source_l1_refs: Vec::new(),
            tags: Vec::new(),
            created: String::new(),
            updated: String::new(),
        }
    }

    #[test]
    fn test_l2_store_save_and_get() {
        let tmp = TempDir::new().unwrap();
        let mut store = L2Store::load(tmp.path()).unwrap();
        assert!(store.is_empty());

        let block = ScenarioBlock {
            id: "test-scenario".into(),
            title: "Test Scenario".into(),
            body: "# Test\n\nThis is a test.".into(),
            conditions: vec!["when using async".into()],
            valence: 0.5,
            active_ticks: 10,
            source_l1_refs: vec![1, 2],
            tags: vec!["test".into()],
            created: "2025-01-01T00:00:00Z".into(),
            updated: "2025-01-01T00:00:00Z".into(),
        };

        store.save(block).unwrap();
        assert_eq!(store.len(), 1);

        let retrieved = store.get("test-scenario").unwrap();
        assert_eq!(retrieved.title, "Test Scenario");
        assert_eq!(retrieved.conditions, vec!["when using async"]);
    }

    #[test]
    fn test_l2_store_search_by_conditions() {
        let tmp = TempDir::new().unwrap();
        let mut store = L2Store::load(tmp.path()).unwrap();

        let mut a = make_block("a", "A");
        a.conditions = vec!["when using async functions".into()];
        store.save(a).unwrap();

        let mut b = make_block("b", "B");
        b.conditions = vec!["when compiling with --release".into()];
        store.save(b).unwrap();

        let results = store.search_by_conditions(&["async"]);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_l2_store_search_by_l1_ref() {
        let tmp = TempDir::new().unwrap();
        let mut store = L2Store::load(tmp.path()).unwrap();

        let mut x = make_block("x", "X");
        x.source_l1_refs = vec![10, 20];
        store.save(x).unwrap();

        let mut y = make_block("y", "Y");
        y.source_l1_refs = vec![30];
        store.save(y).unwrap();

        let results = store.search_by_l1_ref(10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "x");

        let results = store.search_by_l1_ref(30);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "y");

        let results = store.search_by_l1_ref(99);
        assert!(results.is_empty());
    }

    #[test]
    fn test_l2_store_reload() {
        let tmp = TempDir::new().unwrap();
        let mut store = L2Store::load(tmp.path()).unwrap();

        let mut block = make_block("persist", "Persist");
        block.body = "Should survive reload".into();
        store.save(block).unwrap();

        // Create a new store from the same root — should pick up existing files
        let store2 = L2Store::load(tmp.path()).unwrap();
        assert_eq!(store2.len(), 1);
        assert_eq!(store2.get("persist").unwrap().body, "Should survive reload");
    }

    #[test]
    fn test_slugify() {
        assert_eq!(slugify("Hello World!"), "hello-world");
        assert_eq!(slugify("  multiple   spaces  "), "multiple-spaces");
        assert_eq!(slugify("special!@#$chars"), "special-chars");
    }
}
