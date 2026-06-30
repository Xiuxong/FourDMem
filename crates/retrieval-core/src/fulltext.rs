//! Tantivy full-text search index
//!
//! Wraps an in-memory Tantivy index for L2/L0 content search.
//! Supports incremental `add` + `commit` and BM25-ranked `search`.

use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy};

/// In-memory full-text index backed by Tantivy.
pub struct FulltextIndex {
    index: Index,
    #[allow(dead_code)]
    schema: Schema,
    text_field: Field,
    id_field: Field,
    writer: IndexWriter,
    reader: IndexReader,
}

impl FulltextIndex {
    /// Create a new in-memory full-text index (for testing/compat).
    pub fn new() -> Result<Self, tantivy::TantivyError> {
        Self::new_in_ram()
    }

    /// Create an in-memory full-text index.
    pub fn new_in_ram() -> Result<Self, tantivy::TantivyError> {
        let (schema, id_field, text_field) = Self::build_schema();
        let index = Index::create_in_ram(schema.clone());
        let writer = index.writer(50_000_000)?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;
        Ok(Self { index, schema, text_field, id_field, writer, reader })
    }

    /// Create or open a disk-backed full-text index at `dir_path`.
    /// If the directory exists and contains an index, it is opened.
    /// Otherwise, a new index is created.
    pub fn new_on_disk(dir_path: &str) -> Result<Self, tantivy::TantivyError> {
        let dir = std::path::Path::new(dir_path);
        let (schema, id_field, text_field) = Self::build_schema();
        let is_new = !dir.join("meta.json").exists();
        let index = if is_new {
            std::fs::create_dir_all(dir_path).unwrap_or(());
            Index::create_in_dir(dir_path, schema.clone())?
        } else {
            // Open existing index; if it lacks jieba tokenizer, rebuild
            Index::open_in_dir(dir_path).unwrap_or_else(|_| {
                // Corrupt/old schema — wipe and recreate
                let _ = std::fs::remove_dir_all(dir_path);
                std::fs::create_dir_all(dir_path).unwrap_or(());
                Index::create_in_dir(dir_path, schema.clone()).expect("create fulltext index")
            })
        };
        let writer = index.writer(50_000_000)?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;
        Ok(Self { index, schema, text_field, id_field, writer, reader })
    }

    fn build_schema() -> (Schema, Field, Field) {
        let mut schema_builder = Schema::builder();
        let id_field = schema_builder.add_text_field("doc_id", STRING | STORED);
        let text_field = schema_builder.add_text_field("text", TEXT | STORED);
        (schema_builder.build(), id_field, text_field)
    }

    /// Pre-tokenize Chinese text with jieba for better BM25 matching.
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
    pub fn add(&mut self, doc_id: &str, text: &str) -> tantivy::Result<()> {
        let tokenized = Self::tokenize_cjk(text);
        self.writer.add_document(doc!(
            self.id_field => doc_id,
            self.text_field => tokenized,
        ))?;
        Ok(())
    }

    /// Commit all pending additions. Makes them searchable.
    pub fn commit(&mut self) -> tantivy::Result<()> {
        self.writer.commit()?;
        self.reader.reload()?;
        Ok(())
    }

    /// Search the index. Returns `(doc_id, score)` pairs ranked by BM25.
    pub fn search(&self, query_str: &str, limit: usize) -> tantivy::Result<Vec<(String, f64)>> {
        let searcher = self.reader.searcher();
        let query_parser =
            QueryParser::for_index(&self.index, vec![self.text_field]);
        let tokenized_query = Self::tokenize_cjk(query_str);
        let query = query_parser.parse_query(&tokenized_query)?;
        let top_docs = searcher.search(&query, &TopDocs::with_limit(limit))?;

        let mut results = Vec::new();
        for (score, doc_address) in top_docs {
            let doc = searcher.doc::<tantivy::TantivyDocument>(doc_address)?;
            let doc_id = doc
                .get_first(self.id_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            results.push((doc_id, score as f64));
        }

        Ok(results)
    }

    /// Number of committed documents.
    pub fn doc_count(&self) -> u64 {
        self.reader.searcher().num_docs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_search_roundtrip() {
        let mut idx = FulltextIndex::new().expect("new failed");
        idx.add("doc-1", "Rust memory safety is guaranteed by the borrow checker")
            .unwrap();
        idx.add("doc-2", "Python garbage collection handles memory automatically")
            .unwrap();
        idx.add("doc-3", "The borrow checker prevents data races at compile time")
            .unwrap();
        idx.commit().unwrap();

        let results = idx.search("borrow checker", 10).unwrap();
        assert!(!results.is_empty());
        // Both doc-1 and doc-3 mention "borrow checker"
        let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains(&"doc-1") || ids.contains(&"doc-3"));
    }
}
