//! L3 persona / core-rules storage (YAML / JSON)
//!
//! L3 holds the agent's stable identity: core rules, preferences, constraints,
//! and a system-prompt fragment that is injected verbatim. It is the top-most
//! layer of the four-dimensional memory stack and almost never changes —
//! except when the *paradigm-shift engine* (Epic 10) decides to rewrite it.
//!
//! ## Degradation strategy
//!
//! If the backing file is missing or malformed, [`L3Store::load`] falls back
//! to a safe default persona rather than panicking. This keeps the rest of the
//! system operational even when L3 is corrupted.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum L3Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("YAML deserialization error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("JSON deserialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("validation error: {0}")]
    Validation(String),
}

// ── Persona ───────────────────────────────────────────────────────────────────

/// The agent's stable identity, stored as a single YAML or JSON document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Persona {
    /// Human-readable agent name (e.g. "FourDMem Assistant").
    pub agent_name: String,

    /// Schema version (e.g. "4.0.0").
    pub version: String,

    /// Core behavioural rules — short imperative sentences.
    pub core_rules: Vec<String>,

    /// Key-value preference map (free-form JSON values).
    #[serde(default)]
    pub preferences: HashMap<String, Value>,

    /// Hard constraints that must never be violated.
    #[serde(default)]
    pub constraints: Vec<String>,

    /// A ready-to-inject fragment for the LLM system prompt.
    #[serde(default)]
    pub system_prompt_fragment: String,

    /// ISO-8601 timestamp of the last modification.
    #[serde(default)]
    pub last_modified: String,
}

impl Persona {
    /// Construct a minimal, always-valid default persona.
    pub fn default_persona() -> Self {
        Self {
            agent_name: "FourDMem Agent".to_string(),
            version: "4.0.0".to_string(),
            core_rules: vec![
                "Be helpful and precise.".to_string(),
                "Preserve context across sessions.".to_string(),
                "Never fabricate information.".to_string(),
            ],
            preferences: HashMap::new(),
            constraints: Vec::new(),
            system_prompt_fragment: String::new(),
            last_modified: String::new(),
        }
    }

    /// Validate structural invariants.
    ///
    /// Checks:
    /// - `agent_name` is non-empty
    /// - `version` is non-empty
    /// - `core_rules` is non-empty
    pub fn validate(&self) -> Result<(), L3Error> {
        if self.agent_name.trim().is_empty() {
            return Err(L3Error::Validation(
                "agent_name must not be empty".to_string(),
            ));
        }
        if self.version.trim().is_empty() {
            return Err(L3Error::Validation(
                "version must not be empty".to_string(),
            ));
        }
        if self.core_rules.is_empty() {
            return Err(L3Error::Validation(
                "core_rules must contain at least one rule".to_string(),
            ));
        }
        Ok(())
    }

    /// Render a System-Prompt injection fragment.
    ///
    /// Combines the explicit `system_prompt_fragment` (if any) with the
    /// core rules and hard constraints into a single text block that the
    /// caller can splice into the LLM system message.
    pub fn to_system_prompt(&self) -> String {
        let mut parts = Vec::new();

        // Header
        parts.push(format!(
            "# Agent: {} (v{})\n",
            self.agent_name, self.version
        ));

        // Explicit fragment
        if !self.system_prompt_fragment.trim().is_empty() {
            parts.push(self.system_prompt_fragment.clone());
        }

        // Core rules
        if !self.core_rules.is_empty() {
            parts.push("## Core Rules".to_string());
            for (i, rule) in self.core_rules.iter().enumerate() {
                parts.push(format!("{}. {}", i + 1, rule));
            }
        }

        // Hard constraints
        if !self.constraints.is_empty() {
            parts.push("\n## Hard Constraints (must not violate)".to_string());
            for c in &self.constraints {
                parts.push(format!("- {}", c));
            }
        }

        parts.join("\n")
    }
}

impl Default for Persona {
    fn default() -> Self {
        Self::default_persona()
    }
}

// ── L3 Store ──────────────────────────────────────────────────────────────────

/// Wrapper around a [`Persona`] that knows how to load / save itself.
pub struct L3Store {
    persona: Persona,
}

impl L3Store {
    /// Load a persona from a YAML (`.yaml` / `.yml`) or JSON (`.json`) file.
    ///
    /// **Degradation**: if the file does not exist or cannot be parsed, a safe
    /// default persona is returned instead of an error.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, L3Error> {
        let path = path.as_ref();
        let persona = match std::fs::read_to_string(path) {
            Ok(content) => {
                let ext = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_lowercase();

                let parsed: std::result::Result<Persona, Box<dyn std::error::Error + Send + Sync>> =
                    match ext.as_str() {
                        "yaml" | "yml" => serde_yaml::from_str::<Persona>(&content)
                            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>),
                        "json" => serde_json::from_str::<Persona>(&content)
                            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>),
                        _ => {
                            // Try YAML first, then JSON
                            serde_yaml::from_str::<Persona>(&content)
                                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
                                .or_else(|_| {
                                    serde_json::from_str::<Persona>(&content)
                                        .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
                                })
                        }
                    };

                parsed.unwrap_or_else(|_| {
                    eprintln!(
                        "L3Store: failed to parse {}, using default persona",
                        path.display()
                    );
                    Persona::default_persona()
                })
            }
            Err(_) => {
                eprintln!(
                    "L3Store: file {} not found, using default persona",
                    path.display()
                );
                Persona::default_persona()
            }
        };

        Ok(Self { persona })
    }

    /// Create an L3Store with the built-in default persona (no file needed).
    pub fn default_store() -> Self {
        Self {
            persona: Persona::default_persona(),
        }
    }

    /// Return a read-only reference to the current persona.
    pub fn persona(&self) -> &Persona {
        &self.persona
    }

    /// Render the persona as a System-Prompt injection string.
    pub fn to_system_prompt(&self) -> String {
        self.persona.to_system_prompt()
    }

    /// Validate the current persona.
    pub fn validate(&self) -> Result<(), L3Error> {
        self.persona.validate()
    }

    /// Save the persona to a file. Default output is YAML.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), L3Error> {
        let path = path.as_ref();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let content = match ext.as_str() {
            "json" => serde_json::to_string_pretty(&self.persona)?,
            _ => serde_yaml::to_string(&self.persona)?,
        };

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(path, content)?;
        Ok(())
    }

    /// Get a mutable reference to the persona for in-place edits.
    pub fn persona_mut(&mut self) -> &mut Persona {
        &mut self.persona
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_default_persona_is_valid() {
        let p = Persona::default_persona();
        p.validate().expect("default persona should be valid");
    }

    #[test]
    fn test_persona_validation_empty_name() {
        let mut p = Persona::default_persona();
        p.agent_name = String::new();
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_persona_validation_empty_rules() {
        let mut p = Persona::default_persona();
        p.core_rules.clear();
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_to_system_prompt() {
        let mut p = Persona::default_persona();
        p.system_prompt_fragment = "Always think step by step.".to_string();
        p.constraints = vec!["Never expose API keys.".to_string()];

        let prompt = p.to_system_prompt();
        assert!(prompt.contains("FourDMem Agent"));
        assert!(prompt.contains("Always think step by step."));
        assert!(prompt.contains("Core Rules"));
        assert!(prompt.contains("Never expose API keys."));
        assert!(prompt.contains("Hard Constraints"));
    }

    #[test]
    fn test_load_yaml_file() {
        let dir = std::env::temp_dir().join("l3_test_yaml");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("persona.yaml");

        let yaml = r#"
agent_name: TestAgent
version: "1.0"
core_rules:
  - Rule A
  - Rule B
preferences:
  theme: dark
constraints:
  - Do no harm
system_prompt_fragment: "You are TestAgent."
last_modified: "2025-06-15T00:00:00Z"
"#;
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(yaml.as_bytes()).unwrap();

        let store = L3Store::load(&path).expect("load should succeed");
        assert_eq!(store.persona().agent_name, "TestAgent");
        assert_eq!(store.persona().core_rules.len(), 2);
        assert_eq!(store.persona().constraints, vec!["Do no harm"]);
        assert_eq!(
            store.persona().preferences["theme"],
            serde_json::json!("dark")
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_json_file() {
        let dir = std::env::temp_dir().join("l3_test_json");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("persona.json");

        let json = r#"{
  "agent_name": "JsonAgent",
  "version": "2.0",
  "core_rules": ["Be nice"]
}"#;
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(json.as_bytes()).unwrap();

        let store = L3Store::load(&path).expect("load should succeed");
        assert_eq!(store.persona().agent_name, "JsonAgent");
        assert_eq!(store.persona().version, "2.0");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_missing_file_returns_default() {
        let store = L3Store::load("/nonexistent/path/persona.yaml")
            .expect("should degrade gracefully");
        assert_eq!(store.persona().agent_name, "FourDMem Agent");
        store.validate().expect("default should be valid");
    }

    #[test]
    fn test_load_malformed_file_returns_default() {
        let dir = std::env::temp_dir().join("l3_test_malformed");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("persona.yaml");

        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"{{{{not valid yaml").unwrap();

        let store = L3Store::load(&path).expect("should degrade gracefully");
        assert_eq!(store.persona().agent_name, "FourDMem Agent");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_and_reload_yaml() {
        let dir = std::env::temp_dir().join("l3_test_roundtrip");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("persona.yaml");

        let mut store = L3Store::default_store();
        store.persona_mut().agent_name = "RoundTrip".to_string();
        store.save(&path).expect("save should succeed");

        let loaded = L3Store::load(&path).expect("reload should succeed");
        assert_eq!(loaded.persona().agent_name, "RoundTrip");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_json_format() {
        let dir = std::env::temp_dir().join("l3_test_save_json");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("persona.json");

        let store = L3Store::default_store();
        store.save(&path).expect("save should succeed");

        let content = std::fs::read_to_string(&path).unwrap();
        // Should be valid JSON
        let _: serde_json::Value = serde_json::from_str(&content).expect("invalid JSON");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
