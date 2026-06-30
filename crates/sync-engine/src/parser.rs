//! L2 scenario block parser: Markdown + YAML Frontmatter
//!
//! Parses `.md` files with YAML frontmatter into [`ScenarioBlock`]s,
//! and writes them back with frontmatter intact. Gracefully degrades
//! when frontmatter is missing or malformed.

use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum ParserError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("UTF-8 error: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
}

// ── Scenario Block ────────────────────────────────────────────────────────────

/// A single L2 scenario block — a Markdown document with structured metadata
/// encoded in YAML frontmatter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioBlock {
    /// Unique identifier (typically the filename without extension).
    #[serde(default)]
    pub id: String,

    /// Human-readable title (first `# ` heading in the body).
    #[serde(default)]
    pub title: String,

    /// Markdown body (everything after the frontmatter).
    #[serde(default)]
    pub body: String,

    /// Conditional activation rules (e.g. `"when using ? operator in async fn"`).
    #[serde(default)]
    pub conditions: Vec<String>,

    /// Emotional / utility valence in `[-1.0, 1.0]`.
    /// Positive = rewarding knowledge; negative = painful lesson.
    #[serde(default)]
    pub valence: f64,

    /// Subjective time counter (active ticks).
    #[serde(default)]
    pub active_ticks: u64,

    /// References back to L1 graph node indices.
    #[serde(default)]
    pub source_l1_refs: Vec<usize>,

    /// Free-form tags for categorization.
    #[serde(default)]
    pub tags: Vec<String>,

    /// ISO-8601 creation timestamp.
    #[serde(default)]
    pub created: String,

    /// ISO-8601 last-updated timestamp.
    #[serde(default)]
    pub updated: String,
}

impl ScenarioBlock {
    /// Create a minimal block with defaults (used when frontmatter is missing).
    pub fn from_body(body: impl Into<String>) -> Self {
        let body = body.into();
        let title = extract_title(&body).unwrap_or_default();
        Self {
            id: String::new(),
            title,
            body,
            conditions: Vec::new(),
            valence: 0.0,
            active_ticks: 0,
            source_l1_refs: Vec::new(),
            tags: Vec::new(),
            created: String::new(),
            updated: String::new(),
        }
    }

    /// Serialize the block to a Markdown string with YAML frontmatter.
    pub fn to_markdown(&self) -> String {
        let mut out = String::from("---\n");

        // Build a serde_yaml::Value map so we control key order
        let mut map = serde_yaml::Mapping::new();

        if !self.id.is_empty() {
            map.insert(
                serde_yaml::Value::String("id".into()),
                serde_yaml::Value::String(self.id.clone()),
            );
        }
        if !self.conditions.is_empty() {
            let cond: Vec<serde_yaml::Value> = self
                .conditions
                .iter()
                .map(|s| serde_yaml::Value::String(s.clone()))
                .collect();
            map.insert(
                serde_yaml::Value::String("conditions".into()),
                serde_yaml::Value::Sequence(cond),
            );
        }
        map.insert(
            serde_yaml::Value::String("valence".into()),
            serde_yaml::Value::from(self.valence),
        );
        map.insert(
            serde_yaml::Value::String("active_ticks".into()),
            serde_yaml::Value::from(self.active_ticks),
        );
        if !self.source_l1_refs.is_empty() {
            let refs: Vec<serde_yaml::Value> = self
                .source_l1_refs
                .iter()
                .map(|n| serde_yaml::Value::from(*n as i64))
                .collect();
            map.insert(
                serde_yaml::Value::String("source_l1_refs".into()),
                serde_yaml::Value::Sequence(refs),
            );
        }
        if !self.tags.is_empty() {
            let tags: Vec<serde_yaml::Value> = self
                .tags
                .iter()
                .map(|s| serde_yaml::Value::String(s.clone()))
                .collect();
            map.insert(
                serde_yaml::Value::String("tags".into()),
                serde_yaml::Value::Sequence(tags),
            );
        }
        if !self.created.is_empty() {
            map.insert(
                serde_yaml::Value::String("created".into()),
                serde_yaml::Value::String(self.created.clone()),
            );
        }
        if !self.updated.is_empty() {
            map.insert(
                serde_yaml::Value::String("updated".into()),
                serde_yaml::Value::String(self.updated.clone()),
            );
        }

        let yaml = serde_yaml::to_string(&serde_yaml::Value::Mapping(map))
            .unwrap_or_default();
        out.push_str(&yaml);
        out.push_str("---\n\n");
        out.push_str(&self.body);

        // Ensure trailing newline
        if !out.ends_with('\n') {
            out.push('\n');
        }
        out
    }
}

// ── Parser ────────────────────────────────────────────────────────────────────

/// Stateless parser for L2 scenario block Markdown files.
pub struct L2Parser;

impl L2Parser {
    /// Parse a single `.md` file into a [`ScenarioBlock`].
    ///
    /// If the file has no frontmatter or the frontmatter is malformed,
    /// the entire file content is treated as the body and metadata fields
    /// get safe defaults.
    pub fn parse_file(path: impl AsRef<Path>) -> Result<ScenarioBlock, ParserError> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path)?;

        let mut block = Self::parse_content(&raw);

        // Derive `id` from filename if not set in frontmatter
        if block.id.is_empty() {
            block.id = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
        }

        Ok(block)
    }

    /// Parse raw Markdown content (with optional frontmatter) into a
    /// [`ScenarioBlock`].
    pub fn parse_content(raw: &str) -> ScenarioBlock {
        // Try to split frontmatter
        if let Some((fm, body)) = split_frontmatter(raw) {
            // Try parsing YAML frontmatter
            match serde_yaml::from_str::<ScenarioBlock>(fm) {
                Ok(mut block) => {
                    block.body = body.trim().to_string();
                    // Re-extract title from body if not in frontmatter
                    if block.title.is_empty() {
                        block.title = extract_title(body).unwrap_or_default();
                    }
                    return block;
                }
                Err(_) => {
                    // Malformed frontmatter — fall through to default
                }
            }
        }

        // No frontmatter or parse failure — treat entire content as body
        ScenarioBlock::from_body(raw)
    }

    /// Parse all `.md` files in a directory (non-recursive).
    ///
    /// Uses `tokio::fs` for async I/O; parsing itself is CPU-bound but fast.
    pub async fn parse_dir(dir: impl AsRef<Path>) -> Result<Vec<ScenarioBlock>, ParserError> {
        let dir = dir.as_ref();
        let mut blocks = Vec::new();
        let mut read_dir = tokio::fs::read_dir(dir).await?;

        while let Some(entry) = read_dir.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("md") {
                // Read async, parse sync (parsing is fast)
                let raw = tokio::fs::read_to_string(&path).await?;
                let mut block = Self::parse_content(&raw);
                if block.id.is_empty() {
                    block.id = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                }
                blocks.push(block);
            }
        }

        Ok(blocks)
    }

    /// Write a [`ScenarioBlock`] to a file as Markdown with YAML frontmatter.
    pub fn write(
        block: &ScenarioBlock,
        path: impl AsRef<Path>,
    ) -> Result<(), ParserError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, block.to_markdown())?;
        Ok(())
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Split a Markdown document into (frontmatter, body) if frontmatter exists.
///
/// Frontmatter is delimited by `---` on its own line at the very start of the
/// file and the next `---` on its own line.
fn split_frontmatter(raw: &str) -> Option<(&str, &str)> {
    let trimmed = raw.trim_start();
    if !trimmed.starts_with("---") {
        return None;
    }

    // Find the closing ---
    let after_opening = &trimmed[3..];
    let close_pos = after_opening.find("\n---")?;
    let fm = &after_opening[..close_pos].trim();
    let body_start = close_pos + 4; // skip "\n---"
    let body = after_opening[body_start..].trim_start_matches('\n');

    Some((fm, body))
}

/// Extract the first `# ` heading from Markdown text as the title.
fn extract_title(body: &str) -> Option<String> {
    for line in body.lines() {
        let trimmed = line.trim();
        if let Some(title) = trimmed.strip_prefix("# ") {
            return Some(title.trim().to_string());
        }
    }
    None
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_content_with_frontmatter() {
        let raw = r#"---
id: rust-error-handling
conditions:
  - "when using ? operator in async fn"
valence: 0.8
active_ticks: 42
source_l1_refs: [3, 7, 15]
tags: ["rust", "error-handling", "async"]
created: "2025-06-01T10:00:00"
updated: "2025-06-15T14:30:00"
---

# Rust Error Handling Patterns

Content here...
"#;

        let block = L2Parser::parse_content(raw);
        assert_eq!(block.id, "rust-error-handling");
        assert_eq!(block.title, "Rust Error Handling Patterns");
        assert_eq!(block.valence, 0.8);
        assert_eq!(block.active_ticks, 42);
        assert_eq!(block.source_l1_refs, vec![3, 7, 15]);
        assert_eq!(block.tags, vec!["rust", "error-handling", "async"]);
        assert!(block.body.contains("Content here..."));
        assert_eq!(block.conditions, vec!["when using ? operator in async fn"]);
    }

    #[test]
    fn test_parse_content_no_frontmatter() {
        let raw = "# Just a Title\n\nSome content here.\n";
        let block = L2Parser::parse_content(raw);
        assert_eq!(block.title, "Just a Title");
        assert!(block.body.contains("# Just a Title"));
        assert!(block.body.contains("Some content here."));
        assert!(block.id.is_empty());
        assert_eq!(block.valence, 0.0);
    }

    #[test]
    fn test_parse_content_malformed_frontmatter() {
        let raw = "---\n{{{{not yaml\n---\n\nBody text\n";
        let block = L2Parser::parse_content(raw);
        // Should degrade gracefully — entire content treated as body
        assert!(block.body.contains("Body text") || block.body.contains("not yaml"));
    }

    #[test]
    fn test_parse_content_empty() {
        let block = L2Parser::parse_content("");
        assert!(block.body.is_empty());
        assert!(block.title.is_empty());
    }

    #[test]
    fn test_to_markdown_roundtrip() {
        let raw = r#"---
id: test-block
conditions:
  - "always"
valence: 0.5
active_ticks: 10
source_l1_refs: [1, 2]
tags: ["test"]
created: "2025-01-01"
updated: "2025-06-01"
---

# Test Block

Hello world.
"#;

        let block = L2Parser::parse_content(raw);
        let md = block.to_markdown();
        let reparsed = L2Parser::parse_content(&md);

        assert_eq!(reparsed.id, block.id);
        assert_eq!(reparsed.valence, block.valence);
        assert_eq!(reparsed.active_ticks, block.active_ticks);
        assert_eq!(reparsed.source_l1_refs, block.source_l1_refs);
        assert_eq!(reparsed.tags, block.tags);
        assert_eq!(reparsed.conditions, block.conditions);
        assert!(reparsed.body.contains("Hello world."));
    }

    #[test]
    fn test_write_and_parse_file() {
        let dir = std::env::temp_dir().join("l2_parser_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test-scenario.md");

        let block = ScenarioBlock {
            id: "test-scenario".to_string(),
            title: "Test Scenario".to_string(),
            body: "# Test Scenario\n\nContent.".to_string(),
            conditions: vec!["when X".to_string()],
            valence: -0.3,
            active_ticks: 7,
            source_l1_refs: vec![5],
            tags: vec!["test".to_string()],
            created: "2025-06-01".to_string(),
            updated: "2025-06-15".to_string(),
        };

        L2Parser::write(&block, &path).expect("write failed");

        let loaded = L2Parser::parse_file(&path).expect("parse failed");
        assert_eq!(loaded.id, "test-scenario");
        assert_eq!(loaded.valence, -0.3);
        assert_eq!(loaded.source_l1_refs, vec![5]);
        assert!(loaded.body.contains("Content."));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_parse_file_derives_id_from_filename() {
        let dir = std::env::temp_dir().join("l2_id_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("my-cool-scenario.md");

        std::fs::write(&path, "# Title\n\nBody.\n").unwrap();

        let block = L2Parser::parse_file(&path).expect("parse failed");
        assert_eq!(block.id, "my-cool-scenario");
        assert_eq!(block.title, "Title");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_split_frontmatter() {
        let raw = "---\nfoo: bar\n---\n\nBody";
        let (fm, body) = split_frontmatter(raw).unwrap();
        assert_eq!(fm, "foo: bar");
        assert_eq!(body, "Body");
    }

    #[test]
    fn test_split_frontmatter_none() {
        let raw = "# No frontmatter\n\nJust content.";
        assert!(split_frontmatter(raw).is_none());
    }

    #[test]
    fn test_extract_title() {
        assert_eq!(
            extract_title("# My Title\n\nContent"),
            Some("My Title".to_string())
        );
        assert_eq!(extract_title("No title here"), None);
        assert_eq!(
            extract_title("## Subtitle\n# Real Title"),
            Some("Real Title".to_string())
        );
    }
}
