//! sync-engine: Async IO & automatic synchronization
//!
//! - Async batch file parser (L2/L3 frontmatter)
//! - File watcher with 500ms debounce
//! - Incremental sync with Blake3 hash verification

pub mod parser;
pub mod watcher;
pub mod sync;

// Re-export L2 parser public API at crate root.
pub use parser::{L2Parser, ParserError, ScenarioBlock};

// Re-export sync engine.
pub use sync::{SyncEngine, SyncError};

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;
