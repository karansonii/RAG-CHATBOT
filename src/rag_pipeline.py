"""
RAG Pipeline - Enhanced with better visual support
"""
import os
os.environ["HF_HOME"] = r"C:\Users\karan.soni\Documents\Ai Chat Bot\ai-chat-bot\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"C:\Users\karan.soni\Documents\Ai Chat Bot\ai-chat-bot\hf_cache"
import numpy as np
import os
import re
import hashlib
import json
import shutil
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.translator import detect_language, translate_text
import google.generativeai as genai
from src.data_loader import read_ppt_text
from src.system_prompt import SystemPrompts, PromptValidator
from src.utils import get_env_var
from rapidfuzz import fuzz
from collections import Counter
import os
# Must come BEFORE any sentence_transformers import!
# Load configuration
GEMINI_MODEL = get_env_var("GEMINI_MODEL_NAME", "gemini-2.5-flash")
RAG_TOP_K = int(get_env_var("RAG_TOP_K", "10"))
RAG_THRESHOLD = float(get_env_var("RAG_SIMILARITY_THRESHOLD", "0.0"))
MAX_TOKENS = int(get_env_var("MAX_RESPONSE_TOKENS", "4096"))
# Initialize embedding model
# EMBED_MODEL_NAME = get_env_var("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
# embed_model = SentenceTransformer(EMBED_MODEL_NAME)
# ---- RERANKER (Cross-Encoder) ----
embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
custom_cache = os.environ["TRANSFORMERS_CACHE"]
# Clear corrupted cache dirs for this model (both default and custom) to fix Windows lock issue
default_cache_root = os.path.expanduser("~/.cache/huggingface/hub")
custom_cache_root = os.path.join(custom_cache, "hub")
for cache_root in [default_cache_root, custom_cache_root]:
    model_cache_dir = os.path.join(cache_root, f"models--{RERANK_MODEL_NAME.replace('/', '--')}")
    if os.path.exists(model_cache_dir):
        print(f"Clearing cache for {RERANK_MODEL_NAME} in {cache_root} to resolve lock issue...")
        shutil.rmtree(model_cache_dir)
# Load with env vars handling custom cache
cross_encoder = CrossEncoder(RERANK_MODEL_NAME)
# Initialize prompt handler
prompts = SystemPrompts()
validator = PromptValidator()
# ==================== ANSWER CACHE FOR CONSISTENCY ====================
ANSWER_CACHE = {}
CACHE_MAX_SIZE = 1000
def get_query_hash(query: str, lang: str) -> str:
    """Generate unique hash for query to cache answers."""
    normalized = query.lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = re.sub(r'[?!.,;:]', '', normalized)
    cache_key = f"{lang}:{normalized}"
    return hashlib.md5(cache_key.encode()).hexdigest()
def cache_answer(query_hash: str, answer: str):
    """Store answer in cache."""
    global ANSWER_CACHE
    if len(ANSWER_CACHE) >= CACHE_MAX_SIZE:
        keys_to_remove = list(ANSWER_CACHE.keys())[:int(CACHE_MAX_SIZE * 0.2)]
        for key in keys_to_remove:
            del ANSWER_CACHE[key]
    ANSWER_CACHE[query_hash] = answer
    print(f"✓ Cached answer for query hash: {query_hash[:8]}...")
def get_cached_answer(query_hash: str):
    """Retrieve cached answer if available."""
    answer = ANSWER_CACHE.get(query_hash)
    if answer:
        print(f"✓ Using cached answer for query hash: {query_hash[:8]}...")
    return answer
# ==================== EMBEDDING & SEARCH ====================
def embed_text(text: str) -> np.ndarray:
    """Generate embedding for given text."""
    return embed_model.encode([text.lower()])[0]
from rapidfuzz import fuzz
from numpy import dot
from numpy.linalg import norm
def safe_rerank_chunks(query, chunks, top_k=5):
    """
    Heavy reranker using cross-encoder for better accuracy.
    """
    if not chunks:
        return chunks
    scored_chunks = []
  
    # Prepare query-passage pairs
    pairs = [[query, c['text']] for c in chunks]
  
    # Score with cross-encoder
    scores = cross_encoder.predict(pairs)
  
    for c, score in zip(chunks, scores):
        c['rerank_score'] = float(score)
        scored_chunks.append(c)
  
    # Sort chunks descending by score
    scored_chunks = sorted(scored_chunks, key=lambda x: x['rerank_score'], reverse=True)
  
    return scored_chunks[:top_k]
def search_index(query, index, metas, top_k=None):
    """Search FAISS index and return top-k chunks."""
    if index is None or metas is None:
        return []
    if top_k is None:
        top_k = RAG_TOP_K
    query = query.lower().strip()
    q_embed = np.array([embed_text(query)]).astype("float32")
  
    import faiss
    faiss.normalize_L2(q_embed)
  
    scores, idxs = index.search(q_embed, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        meta = metas[idx]
        results.append({
            "text": meta.get("text", ""),
            "metadata": meta.get("metadata", meta),
            "score": float(score)
        })
    return results
# ==================== GEMINI API CALL ====================
def call_gemini(prompt, api_key, model_name=None, temperature=0.3):
    """Call Gemini API with consistent settings."""
    validator.validate_prompt(prompt)
  
    if model_name is None:
        model_name = GEMINI_MODEL
  
    genai.configure(api_key=api_key)
  
    generation_config = {
        "temperature": temperature,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": MAX_TOKENS,
    }
  
    model = genai.GenerativeModel(model_name, generation_config=generation_config)
  
    try:
        response = model.generate_content(prompt)
        if response and response.candidates:
            return response.candidates[0].content.parts[0].text.strip()
    except Exception as e:
        fallback_model = get_env_var("GEMINI_FALLBACK_MODEL", None)
        if fallback_model and fallback_model != model_name:
            try:
                model = genai.GenerativeModel(fallback_model, generation_config=generation_config)
                response = model.generate_content(prompt)
                if response and response.candidates:
                    return response.candidates[0].content.parts[0].text.strip()
            except:
                pass
        print(f"⚠️ Gemini API error: {e}")
  
    return ""
# ==================== ANSWER CLEANING (FIXED!) ====================
def clean_answer_sources(text: str) -> str:
    """
    🔥 FIXED: Remove citations but PRESERVE newlines and formatting!
  
    Args:
        text: Answer text
    """
    if not text:
        return text
  
    # Remove [Document X]: patterns
    text = re.sub(r'\[Document \d+\]:\s*', '', text, flags=re.IGNORECASE)
  
    # Remove (Document X) patterns
    text = re.sub(r'\s*\(Document \d+\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\(Slide \d+\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\(slide \d+\)', '', text, flags=re.IGNORECASE)
  
    # Remove Document X: patterns at start of sentences
    text = re.sub(r'Document \d+:\s*', '', text, flags=re.IGNORECASE)
  
    # Remove ppt_X patterns
    text = re.sub(r'\s*\(ppt_\d+\)', '', text, flags=re.IGNORECASE)
  
    # Remove "from Document X" patterns
    text = re.sub(r'\s*from Document \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\(from slide \d+\)', '', text, flags=re.IGNORECASE)
  
    # Remove references
    text = re.sub(r'as mentioned in Document \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'according to Document \d+', '', text, flags=re.IGNORECASE)
  
    # 🔥 CRITICAL FIX: Clean spaces but PRESERVE newlines!
    # Replace multiple spaces with single space (but keep newlines)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Clean multiple spaces within each line
        cleaned_line = re.sub(r'[ \t]+', ' ', line)
        cleaned_lines.append(cleaned_line.strip())
  
    # Join lines back together
    text = '\n'.join(cleaned_lines)
  
    # Remove excessive blank lines (more than 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
  
    return text.strip()
# ==================== QUERY EXPANSION ====================
def expand_query_with_synonyms(query: str) -> str:
    """Expand query with synonyms for better matching."""
    expansions = {
        'acronym': ['acronym', 'abbreviation', 'short form', 'initialism'],
        'diagram': ['diagram', 'visual', 'chart', 'flowchart', 'graph', 'illustration'],
        'process': ['process', 'procedure', 'workflow', 'steps', 'flow'],
        'tender': ['tender', 'bid', 'procurement', 'rfp', 'rfq'],
        'committee': ['committee', 'board', 'panel', 'group', 'team'],
        'opening': ['opening', 'launch', 'start', 'initiation'],
        'preparation': ['preparation', 'planning', 'setup', 'ready'],
        'evaluation': ['evaluation', 'assessment', 'review', 'analysis'],
        'approval': ['approval', 'authorization', 'permission', 'clearance'],
    }
  
    query_lower = query.lower()
    expanded_terms = [query]
  
    for key, synonyms in expansions.items():
        if key in query_lower:
            expanded_terms.extend(synonyms[:3])
  
    return ' '.join(expanded_terms)
# ==================== MAIN RAG FUNCTION ====================
def rag_answer(query, index, metas, api_key, model_name=None, threshold=None,
               top_k=None, ppt_path=None, use_cache=True):
    """
    RAG pipeline with proper formatting preservation.
  
    Args:
        query: User's question
        index: FAISS index
        metas: Metadata for chunks
        api_key: Gemini API key
        model_name: Optional model override
        threshold: Similarity threshold
        top_k: Number of chunks to retrieve
        ppt_path: Optional PPT file path for OCR extraction
        use_cache: Whether to use answer caching (default: True)
    """
    # Use configured values if not provided
    if model_name is None:
        model_name = GEMINI_MODEL
    if threshold is None:
        threshold = RAG_THRESHOLD
    if top_k is None:
        top_k = RAG_TOP_K
    # Sanitize user input
    query = validator.sanitize_user_input(query)
    # Detect language
    user_lang = detect_language(query)
  
    # 🔥 DEBUG: Log the query and language
    print(f"🔍 Query: {query}")
    print(f"🌐 Detected language: {user_lang}")
    # Check cache FIRST for consistency
    query_hash = get_query_hash(query, user_lang)
    if use_cache:
        cached = get_cached_answer(query_hash)
        if cached:
            return cached
    # Translate query → English for retrieval if needed
    query_en = query if user_lang == "en" else translate_text(query, target_lang="en").lower().strip()
    # If PPT path provided, extract text using OCR and append to metas
    if ppt_path:
        ppt_text_dict = read_ppt_text(ppt_path)
        ppt_chunks = [
            {"text": text, "metadata": {"source": f"ppt_{slide}"}}
            for slide, text in ppt_text_dict.items()
        ]
        if metas is None:
            metas = ppt_chunks
        else:
            metas = list(metas) + ppt_chunks
    # ========== RETRIEVAL ==========
    expanded_query = expand_query_with_synonyms(query_en)
    retrieved_chunks = search_index(expanded_query, index, metas, top_k=top_k * 2)
    # ====== APPLY RERANKER ======
    if retrieved_chunks:
        print("🔍 Applying Reranker for better relevance...")
        retrieved_chunks = safe_rerank_chunks(query_en, retrieved_chunks, top_k=top_k)
  
    if not retrieved_chunks:
        retrieved_chunks = search_index(query_en, index, metas, top_k=top_k * 2)
  
    if not retrieved_chunks and metas:
        print("⚠️ No semantic matches found, using fallback chunks")
        retrieved_chunks = [
            {"text": meta.get("text", ""),
             "metadata": meta.get("metadata", {}),
             "score": 0.0}
            for meta in metas[:top_k]
        ]
    # Apply threshold
    if threshold > 0:
        filtered_chunks = [item for item in retrieved_chunks if item.get("score", 0) >= threshold]
    else:
        filtered_chunks = retrieved_chunks
    # Fallback if no chunks
    if not filtered_chunks:
        if metas:
            print("⚠️ Using all available chunks as last resort")
            filtered_chunks = [
                {"text": meta.get("text", ""),
                 "metadata": meta.get("metadata", {}),
                 "score": 0.0}
                for meta in metas[:min(top_k * 2, len(metas))]
            ]
  
    if not filtered_chunks:
        fallback_msg = "I don't have any documents loaded to answer your question."
        return fallback_msg if user_lang == "en" else translate_text(fallback_msg, target_lang=user_lang)
    # Limit to top_k
    filtered_chunks = filtered_chunks[:top_k]
    # Build context
    context = "\n\n".join([chunk['text'] for chunk in filtered_chunks])
    # Log stats
    score_list = [f"{c.get('score', 0):.3f}" for c in filtered_chunks[:3]]
    print(f"✓ Retrieved {len(filtered_chunks)} chunks (scores: {score_list})")
    # ========== BUILD PROMPT ==========
    base_prompt = prompts.get_rag_base_prompt(context, query_en, False)
    base_prompt += prompts.get_multilingual_instruction(user_lang)
    base_prompt += prompts.get_token_limit_prompt(MAX_TOKENS)
  
    # 🔥 DEBUG: Show prompt snippet
    print(f"📝 Prompt length: {len(base_prompt)} chars")
    # Get answer from Gemini
    answer_en = call_gemini(base_prompt, api_key, model_name, temperature=0.3)
  
    if not answer_en:
        error_msg = "I encountered an error generating the response."
        return error_msg if user_lang == "en" else translate_text(error_msg, target_lang=user_lang)
    # CRITICAL: Clean citations but PRESERVE formatting
    answer_en = clean_answer_sources(answer_en)
    # Translate if needed
    answer = answer_en if user_lang == "en" else translate_text(answer_en, target_lang=user_lang)
    # Clean again after translation
    answer = clean_answer_sources(answer)
    # Cache the answer
    if use_cache:
        cache_answer(query_hash, answer)
    return answer
# ==================== BATCH PROCESSING ====================
def batch_rag_answer(queries: list, index, metas, api_key, model_name=None,
                     threshold=None, top_k=None) -> list:
    """Process multiple queries at once."""
    answers = []
    for query in queries:
        answer = rag_answer(query, index, metas, api_key, model_name, threshold, top_k)
        answers.append(answer)
    return answers
# ==================== CACHE MANAGEMENT ====================
def clear_answer_cache():
    """Clear the answer cache."""
    global ANSWER_CACHE
    ANSWER_CACHE = {}
    print("✓ Answer cache cleared")
def get_cache_stats():
    """Get cache statistics."""
    return {
        "size": len(ANSWER_CACHE),
        "max_size": CACHE_MAX_SIZE,
        "usage_percent": (len(ANSWER_CACHE) / CACHE_MAX_SIZE) * 100
    }
# ==================== EXPORT ====================
__all__ = [
    'rag_answer',
    'batch_rag_answer',
    'search_index',
    'call_gemini',
    'embed_text',
    'clean_answer_sources',
    'clear_answer_cache',
    'get_cache_stats',
    'expand_query_with_synonyms'
]
