//! Sandbox configuration for genetic-algorithm mutations
//!
//! Before a mutated [`CognitiveDna`] (from `genome.rs`) is hot-swapped
//! into production, it must be evaluated in a sandbox. This module holds
//! the sandbox configuration and result types.

use serde::{Deserialize, Serialize};

/// Configuration for the sandbox execution environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    /// Maximum wall-clock time for a single evaluation run (milliseconds).
    pub timeout_ms: u64,

    /// Memory ceiling for the sandbox process (megabytes).
    pub memory_limit_mb: u64,

    /// Minimum fitness score a mutated DNA must achieve before it can be
    /// hot-swapped into the production configuration.
    pub min_fitness_for_hotswap: f64,

    /// Whether the "auto-plugin" mechanism (Epic 10, T-10.8) is enabled.
    /// When enabled, the sandbox may generate and test new Python plugins.
    pub enable_auto_plugin: bool,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            timeout_ms: 5_000,
            memory_limit_mb: 256,
            min_fitness_for_hotswap: 0.85,
            enable_auto_plugin: false,
        }
    }
}

/// The outcome of a sandbox evaluation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxResult {
    /// Whether the evaluation passed all safety checks.
    pub passed: bool,
    /// Computed fitness score in [0.0, 1.0].
    pub fitness: f64,
    /// Captured stdout / diagnostic output from the run.
    pub output: String,
}

impl SandboxConfig {
    /// Create a config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate that a plugin's source code passes basic safety checks.
    ///
    /// Currently a placeholder — in production this would use AST analysis
    /// to reject dangerous imports (e.g. `os.system`, `subprocess`).
    pub fn validate_plugin(&self, code: &str) -> Result<(), String> {
        // Basic denylist check
        let dangerous = ["os.system", "subprocess", "eval(", "exec("];
        for pattern in &dangerous {
            if code.contains(pattern) {
                return Err(format!("blocked: code contains '{}'", pattern));
            }
        }
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = SandboxConfig::new();
        assert_eq!(cfg.timeout_ms, 5_000);
        assert_eq!(cfg.memory_limit_mb, 256);
        assert!((cfg.min_fitness_for_hotswap - 0.85).abs() < f64::EPSILON);
        assert!(!cfg.enable_auto_plugin);
    }

    #[test]
    fn test_validate_plugin_clean() {
        let cfg = SandboxConfig::new();
        assert!(cfg.validate_plugin("def search(): return 42").is_ok());
    }

    #[test]
    fn test_validate_plugin_blocks_dangerous() {
        let cfg = SandboxConfig::new();
        assert!(cfg.validate_plugin("import os; os.system('rm -rf /')").is_err());
        assert!(cfg.validate_plugin("eval('malicious')").is_err());
    }
}
