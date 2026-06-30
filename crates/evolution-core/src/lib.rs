//! evolution-core: Cognitive Evolution Engine (V4.0)
//!
//! - Cognitive DNA genome storage
//! - Sandbox scheduling for genetic algorithm mutations
//! - Cognitive macro compilation cache (intuitive responses)
//! - Paradigm shift detection & dialectical synthesis

pub mod genome;
pub mod macro_cache;
pub mod sandbox;

// Re-export public API at crate root.
pub use genome::CognitiveDna;
pub use macro_cache::{CognitiveMacro, MacroCache};
pub use sandbox::{SandboxConfig, SandboxResult};

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;
