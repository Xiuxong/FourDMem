"""Embedder — Dual-model semantic embedding with auto language detection.

Models:
- CJK text → BAAI/bge-small-zh-v1.5 (512-dim native)
- English text → BAAI/bge-small-en-v1.5 (384-dim, zero-padded to 512)

Both fully local — no cloud API calls.

Usage:
    from cognition.embedder import get_embedder
    emb = get_embedder()
    vector = emb.embed("Rust borrow checker prevents data races")  # → en model
    vector = emb.embed("Rust 的所有权系统防止数据竞争")              # → zh model
"""


import os
import re
import sys
import threading
from typing import Any

# ── Configuration ─────────────────────────────────────────────────────────────

if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

ZH_MODEL = os.environ.get("FOURDMEM_EMBED_MODEL_ZH", "BAAI/bge-small-zh-v1.5")
EN_MODEL = os.environ.get("FOURDMEM_EMBED_MODEL_EN", "BAAI/bge-base-en-v1.5")
DEFAULT_MODEL = os.environ.get("FOURDMEM_EMBED_MODEL", EN_MODEL)
OUTPUT_DIM = int(os.environ.get("FOURDMEM_EMBED_DIM", "768"))
ZH_DIM = 512
EN_DIM = 768


# ── Language detection ──────────────────────────────────────────────────────

# Unicode ranges for CJK characters
_CJK_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3000-\u303f\uff00-\uffef]')


def _has_cjk(text: str) -> bool:
    """True if text contains any CJK character."""
    return bool(_CJK_RE.search(text))


# ── Singleton ─────────────────────────────────────────────────────────────────

_embedder_instance: Any = None
_lock = threading.Lock()


def get_embedder() -> "Embedder":
    """Get or create the global Embedder singleton (thread-safe)."""
    global _embedder_instance
    if _embedder_instance is None:
        with _lock:
            if _embedder_instance is None:
                _embedder_instance = Embedder()
    return _embedder_instance


def _pad_to_dim(vector: list[float], target_dim: int) -> list[float]:
    """Zero-pad a vector to target dimension. Cosine similarity preserved."""
    if len(vector) >= target_dim:
        return vector[:target_dim]
    return vector + [0.0] * (target_dim - len(vector))


class Embedder:
    """Dual-model embedder with auto language detection.

    Features:
    - CJK text → bge-small-zh-v1.5 (512-dim, native)
    - English → bge-small-en-v1.5 (384-dim, zero-padded to 512)
    - LRU cache per model (256 entries each)
    - Lazy loading — first call to each language triggers model download
    """

    def __init__(
        self,
        zh_model: str = ZH_MODEL,
        en_model: str = EN_MODEL,
        output_dim: int = OUTPUT_DIM,
        cache_size: int = 256,
    ):
        self.zh_model_name = zh_model
        self.en_model_name = en_model
        self.output_dim = output_dim

        # Per-language models (lazy loaded)
        self._zh_model: Any = None
        self._en_model: Any = None
        self._zh_loaded = False
        self._en_loaded = False
        self._device = "cpu"

        # LRU caches
        self._cache: dict[str, list[float]] = {}
        self._cache_order: list[str] = []
        self._cache_size = cache_size
        self._cache_hits = 0
        self._cache_misses = 0

    def _cache_key(self, text: str) -> str:
        import hashlib
        return hashlib.md5(text.strip().encode()).hexdigest()

    def _cache_get(self, text: str) -> list[float] | None:
        key = self._cache_key(text)
        if key in self._cache:
            self._cache_hits += 1
            if key in self._cache_order:
                self._cache_order.remove(key)
            self._cache_order.append(key)
            return self._cache[key]
        self._cache_misses += 1
        return None

    def _cache_put(self, text: str, vector: list[float]):
        key = self._cache_key(text)
        self._cache[key] = vector
        self._cache_order.append(key)
        while len(self._cache) > self._cache_size:
            oldest = self._cache_order.pop(0)
            self._cache.pop(oldest, None)

    def _load_model(self, model_name: str, native_dim: int) -> Any:
        """Load a sentence-transformers model. Returns (model, loaded)"""
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            if torch.cuda.is_available():
                self._device = "cuda"

            print(
                f"[Embedder] Loading {model_name} ({native_dim}d) on {self._device}...",
                file=sys.stderr,
            )
            model = SentenceTransformer(model_name, device=self._device)
            print(
                f"[Embedder] {model_name} loaded. dim={native_dim}, device={self._device}",
                file=sys.stderr,
            )
            return model
        except Exception as e:
            print(f"[Embedder] Failed to load {model_name}: {e}", file=sys.stderr)
            return None

    def warmup(self):
        """Pre-load zh model and warm up. en model loads lazily on first use."""
        if not self._zh_loaded:
            self._zh_model = self._load_model(self.zh_model_name, ZH_DIM)
            self._zh_loaded = self._zh_model is not None
        if self._zh_loaded:
            self._zh_model.encode("预热", normalize_embeddings=True)
            print("[Embedder] Warmup complete (zh model only, en will load lazily).", file=sys.stderr)


    def _embed_zh(self, text: str) -> list[float]:
        """Embed using zh model. Used as fallback when en model unavailable."""
        if not self._zh_loaded:
            self._zh_model = self._load_model(self.zh_model_name, ZH_DIM)
            self._zh_loaded = self._zh_model is not None
        if self._zh_loaded and self._zh_model is not None:
            try:
                raw = self._zh_model.encode(text, normalize_embeddings=True)
                v = raw.tolist() if hasattr(raw, 'tolist') else list(raw)
            except Exception:
                v = [0.0] * ZH_DIM
        else:
            v = [0.0] * ZH_DIM
        return _pad_to_dim(v, self.output_dim)

    def embed(self, text: str) -> list[float]:
        """Embed text into OUTPUT_DIM vector with auto language detection."""
        # Cache check
        cached = self._cache_get(text)
        if cached is not None:
            return cached

        vector: list[float]

        if _has_cjk(text):
            # Chinese: use zh model (native 512 → passthrough)
            if not self._zh_loaded:
                self._zh_model = self._load_model(self.zh_model_name, ZH_DIM)
                self._zh_loaded = self._zh_model is not None
            if self._zh_loaded and self._zh_model is not None:
                try:
                    raw = self._zh_model.encode(text, normalize_embeddings=True)
                    vector = raw.tolist() if hasattr(raw, 'tolist') else list(raw)
                except Exception:
                    vector = [0.0] * ZH_DIM
            else:
                vector = [0.0] * ZH_DIM
            # Ensure output dim
            vector = _pad_to_dim(vector, self.output_dim)
        else:
            # English: try en model, fall back to zh if unavailable
            if not self._en_loaded:
                self._en_model = self._load_model(self.en_model_name, EN_DIM)
                self._en_loaded = self._en_model is not None
            if self._en_loaded and self._en_model is not None:
                try:
                    raw = self._en_model.encode(text, normalize_embeddings=True)
                    vector = raw.tolist() if hasattr(raw, 'tolist') else list(raw)
                except Exception:
                    vector = [0.0] * EN_DIM
                vector = _pad_to_dim(vector, self.output_dim)
            else:
                # Fallback: use zh model for English text
                vector = self._embed_zh(text)
        # ── Cache + return ──
        self._cache_put(text, vector)
        return vector

    def embed_dual(self, text: str) -> dict[str, list[float]]:
        """Embed with both zh and en models. Returns {'zh': vec, 'en': vec}.

        Used for dual-language retrieval: query with both embeddings,
        merge results via RRF for cross-language recall.
        """
        result: dict[str, list[float]] = {}

        # zh embedding
        result['zh'] = self._embed_zh(text)

        # en embedding
        if not self._en_loaded:
            self._en_model = self._load_model(self.en_model_name, EN_DIM)
            self._en_loaded = self._en_model is not None
        if self._en_loaded and self._en_model is not None:
            try:
                raw = self._en_model.encode(text, normalize_embeddings=True)
                v = raw.tolist() if hasattr(raw, 'tolist') else list(raw)
                result['en'] = _pad_to_dim(v, self.output_dim)
            except Exception:
                result['en'] = [0.0] * self.output_dim
        else:
            result['en'] = self._embed_zh(text)  # fallback

        return result

    def get_cache_stats(self) -> dict:
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache),
            "max_size": self._cache_size,
            "zh_loaded": self._zh_loaded,
            "en_loaded": self._en_loaded,
        }

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Grouped by language for efficiency."""
        results: list[list[float]] = []

        # Group by language
        zh_indices = []
        en_indices = []
        for i, t in enumerate(texts):
            if _has_cjk(t):
                zh_indices.append((i, t))
            else:
                en_indices.append((i, t))

        # Placeholder
        results = [[0.0] * self.output_dim for _ in texts]

        # Batch zh
        if zh_indices and self._zh_loaded and self._zh_model:
            try:
                zh_texts = [t for _, t in zh_indices]
                raw = self._zh_model.encode(zh_texts, normalize_embeddings=True)
                for j, (orig_i, _) in enumerate(zh_indices):
                    vec = raw[j].tolist() if hasattr(raw[j], 'tolist') else list(raw[j])
                    results[orig_i] = _pad_to_dim(vec, self.output_dim)
            except Exception:
                pass

        # Batch en
        if en_indices and self._en_loaded and self._en_model:
            try:
                en_texts = [t for _, t in en_indices]
                raw = self._en_model.encode(en_texts, normalize_embeddings=True)
                for j, (orig_i, _) in enumerate(en_indices):
                    vec = raw[j].tolist() if hasattr(raw[j], 'tolist') else list(raw[j])
                    results[orig_i] = _pad_to_dim(vec, self.output_dim)
            except Exception:
                pass

        return results

    def get_status(self) -> dict:
        return {
            "zh_model": self.zh_model_name,
            "en_model": self.en_model_name,
            "output_dim": self.output_dim,
            "zh_loaded": self._zh_loaded,
            "en_loaded": self._en_loaded,
            "device": self._device,
            "cache": self.get_cache_stats(),
        }
