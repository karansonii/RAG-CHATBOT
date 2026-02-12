# ============================================================
# 1. src/embedder.py - UPDATED FOR CONSISTENCY
# ============================================================
import json
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.utils import get_env_var
import numpy as np
VECTOR_DB_PATH = get_env_var("VECTOR_DB_PATH", "./data/vectorstore")
INDEX_DIR = Path(VECTOR_DB_PATH)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = get_env_var("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
# Global variables
_model, _index, _metas = None, None, None
EMBED_DIM = None
# FIX: Add seed for reproducibility
RANDOM_SEED = int(get_env_var("RANDOM_SEED", "42"))
np.random.seed(RANDOM_SEED)
def get_model():
    """Load and cache embedding model with consistent settings."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
        # FIX: Set model to eval mode for consistency
        _model.eval()
    return _model
def get_embed_dim():
    """Get embedding dimension dynamically from the model."""
    global EMBED_DIM
    if EMBED_DIM is None:
        model = get_model()
        sample_embedding = model.encode(["test"], convert_to_numpy=True)
        EMBED_DIM = sample_embedding.shape[1]
    return EMBED_DIM
def create_or_load_index(chunks, rebuild: bool = False):
    """
    Create or load FAISS index with CONSISTENT embedding.
   
    FIX: Ensures reproducible embeddings each time
    """
    global _index, _metas
    index_file = INDEX_DIR / "faiss.index"
    meta_file = INDEX_DIR / "metas.json"
    embed_dim = get_embed_dim()
    # Load existing index if possible
    if not rebuild and index_file.exists() and meta_file.exists():
        _index = faiss.read_index(str(index_file))
        _metas = json.loads(meta_file.read_text(encoding="utf-8"))
        if _index.d != embed_dim:
            print(f"⚠️ FAISS index dimension mismatch ({_index.d} != {embed_dim}), rebuilding...")
            rebuild = True
        else:
            print(f"✅ Loaded existing index with {_index.ntotal} vectors")
            return _index, _metas
    if not chunks:
        raise ValueError("No document chunks found to build index.")
    # Create embeddings with CONSISTENT settings
    model = get_model()
    texts = [c["text"] for c in chunks]
   
    # FIX: Use consistent parameters
    batch_size = int(get_env_var("EMBEDDING_BATCH_SIZE", "32"))
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=batch_size,
        normalize_embeddings=False
    )
    # Normalize for cosine similarity
    normalize_embeddings = get_env_var("NORMALIZE_EMBEDDINGS", "true", config_type=bool)
    if normalize_embeddings:
        faiss.normalize_L2(embeddings)
    # Create FAISS index with consistent metric
    index = faiss.IndexFlatIP(embed_dim)
    index.add(embeddings)
    # Save index and metadata
    faiss.write_index(index, str(index_file))
    meta_file.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    _index, _metas = index, chunks
    print(f"✅ Created new index with {len(chunks)} chunks")
    return _index, _metas
def embed_query(query: str):
    """Embed a single query with consistent parameters."""
    model = get_model()
    v = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=False
    )
    faiss.normalize_L2(v)
    return v
__all__ = ['create_or_load_index', 'embed_query', 'get_embed_dim']
