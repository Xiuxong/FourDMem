//! retrieval-core: Four-dimensional retrieval & reranking engine
//!
//! - HNSW vector index (usearch) or flat brute-force index (pure Rust)
//! - Text embedding via trigram hashing
//! - Tantivy full-text search
//! - RRF (Reciprocal Rank Fusion) reranker
//! - RIF-U scoring model (subjective-time based)
//! - Token budget layer quota allocator
//! - Metacognitive retrieval router

pub mod vector_index;
pub mod embedding;
pub mod fulltext;
pub mod rrf;
pub mod rif_u;
pub mod router;
pub mod token_budget;

// Re-export public API at crate root.
pub use embedding::{embed_text, cosine_sim, cross_domain_analogies, DEFAULT_EMBEDDING_DIM};
pub use fulltext::FulltextIndex;
pub use rif_u::{RifUScorer, RifUWeights};
pub use rrf::{RankedItem, RrfFuser};
pub use router::MetaRouter;
pub use token_budget::{BudgetAllocation, BudgetCandidate, TokenBudget};
pub use vector_index::VectorIndex;

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;
