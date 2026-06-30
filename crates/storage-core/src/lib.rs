//! storage-core: L0-L3 layered storage engine
//!
//! - L0: SQLite + FTS5 raw evidence store (append-only)
//! - L3: YAML/JSON persona & core rules
//! - Cross-layer reference integrity

pub mod l0;
pub mod l2;
pub mod l3;
pub mod integrity;
pub mod memory_core;

// Re-export L0 public API at crate root for convenience.
pub use l0::{Evidence, L0Error, L0Stats, L0Store};

// Re-export L2 public API at crate root for convenience.
pub use l2::{L2Error, L2Store};

// Re-export L3 public API at crate root for convenience.
pub use l3::{L3Error, L3Store, Persona};

// Re-export integrity checker.
pub use integrity::{check_integrity, IntegrityReport, OrphanRef};

// Re-export MemoryCore.
pub use memory_core::{MemoryCore, MemoryStats, DreamReport, QueryItem, QueryRequest, QueryResponse};

/// Re-export common types
pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;
