# src/utils.py
import os
from dotenv import load_dotenv
from rapidfuzz import process, fuzz
from pathlib import Path
import logging
# Load environment variables
load_dotenv()
# Setup logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
class ConfigManager:
    """Centralized configuration manager for environment variables."""
   
    # Cache for loaded config
    _config_cache = {}
   
    @classmethod
    def get(cls, key: str, default=None, required=False, config_type=str):
        """
        Get configuration value with type casting and caching.
       
        Args:
            key: Environment variable name
            default: Default value if not found
            required: Raise error if not found and no default
            config_type: Type to cast the value to (str, int, float, bool)
        """
        # Check cache first
        if key in cls._config_cache:
            return cls._config_cache[key]
       
        value = os.getenv(key, default)
       
        if required and value is None:
            raise EnvironmentError(f"Missing required environment variable: {key}")
       
        if value is None:
            return None
       
        # Type casting
        try:
            if config_type == bool:
                typed_value = value.lower() in ('true', '1', 'yes', 'on')
            elif config_type == int:
                typed_value = int(value)
            elif config_type == float:
                typed_value = float(value)
            elif config_type == list:
                typed_value = [item.strip() for item in value.split(',')]
            else:
                typed_value = str(value)
           
            # Cache the value
            cls._config_cache[key] = typed_value
            return typed_value
           
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to cast {key}={value} to {config_type}: {e}")
            return default
   
    @classmethod
    def get_path(cls, key: str, default=None, create_if_missing=True):
        """
        Get path configuration and optionally create directory.
       
        Args:
            key: Environment variable name
            default: Default path
            create_if_missing: Create directory if it doesn't exist
        """
        path_str = cls.get(key, default)
        if not path_str:
            return None
       
        path = Path(path_str)
       
        if create_if_missing and not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")
       
        return path
   
    @classmethod
    def get_api_key(cls, service: str):
        """
        Get API key for a service.
       
        Args:
            service: Service name (e.g., 'gemini', 'huggingface')
        """
        key_map = {
            'gemini': 'GEMINI_API_KEY',
            'huggingface': 'HUGGINGFACE_API_KEY',
            'openai': 'OPENAI_API_KEY'
        }
       
        env_key = key_map.get(service.lower())
        if not env_key:
            raise ValueError(f"Unknown service: {service}")
       
        api_key = cls.get(env_key, required=True)
        return api_key
   
    @classmethod
    def get_model_config(cls):
        """Get AI model configuration."""
        return {
            'primary_model': cls.get('GEMINI_MODEL_NAME', 'gemini-2.5-flash'),
            'fallback_model': cls.get('GEMINI_FALLBACK_MODEL', 'gemini-1.5-flash'),
            'embed_model': cls.get('EMBED_MODEL', 'sentence-transformers/all-mpnet-base-v2'),
            'max_tokens': cls.get('MAX_RESPONSE_TOKENS', 4096, config_type=int),
            'timeout': cls.get('API_TIMEOUT', 30, config_type=int)
        }
   
    @classmethod
    def get_rag_config(cls):
        """Get RAG configuration."""
        return {
            'top_k': cls.get('RAG_TOP_K', 5, config_type=int),
            'threshold': cls.get('RAG_SIMILARITY_THRESHOLD', 0.0, config_type=float),
            'chunk_size': cls.get('CHUNK_SIZE', 200, config_type=int),
            'overlap': cls.get('CHUNK_OVERLAP', 40, config_type=int),
            'ppt_chunk_size': cls.get('PPT_CHUNK_SIZE', 800, config_type=int),
            'ppt_overlap': cls.get('PPT_CHUNK_OVERLAP', 100, config_type=int)
        }
   
    @classmethod
    def get_language_config(cls):
        """Get language configuration."""
        return {
            'default': cls.get('DEFAULT_LANGUAGE', 'en'),
            'supported': cls.get('SUPPORTED_LANGUAGES', ['en', 'sw'], config_type=list),
            'auto_detect': cls.get('AUTO_LANGUAGE_DETECTION', True, config_type=bool)
        }
   
    @classmethod
    def get_voice_config(cls):
        """Get voice configuration."""
        return {
            'tts_lang': cls.get('TTS_DEFAULT_LANG', 'en'),
            'sample_rate': cls.get('AUDIO_SAMPLE_RATE', 16000, config_type=int),
            'chunk_size': cls.get('AUDIO_CHUNK_SIZE', 1024, config_type=int),
            'fuzzy_threshold': cls.get('FUZZY_MATCH_THRESHOLD', 80, config_type=int)
        }
   
    @classmethod
    def is_debug_mode(cls):
        """Check if debug mode is enabled."""
        return cls.get('DEBUG_MODE', False, config_type=bool)
   
    @classmethod
    def clear_cache(cls):
        """Clear configuration cache."""
        cls._config_cache.clear()
        logger.info("Configuration cache cleared")
# Backward compatibility function
def get_env_var(name: str, default=None, required=False):
    """
    Legacy function for backward compatibility.
    Use ConfigManager.get() for new code.
    """
    return ConfigManager.get(name, default, required)
# Fuzzy matching utility
def fuzzy_match_text(text: str, possibilities=None, threshold: int = None) -> str:
    """
    Fuzzy match text to a list of possibilities.
   
    Args:
        text: Input text
        possibilities: List of valid phrases/words
        threshold: Minimum score (0-100) to accept a match
    """
    if not text:
        return text
   
    # Use configured threshold if not provided
    if threshold is None:
        threshold = ConfigManager.get('FUZZY_MATCH_THRESHOLD', 80, config_type=int)
   
    # Import domain keywords if not provided (avoid circular import)
    if not possibilities:
        possibilities = []
   
    if not possibilities:
        return text
    # Normalize to lowercase for comparison
    text_norm = text.strip().lower()
    possibilities_norm = [p.lower() for p in possibilities]
    best_match, score, _ = process.extractOne(
        text_norm,
        possibilities_norm,
        scorer=fuzz.token_sort_ratio
    )
    if score >= threshold:
        # Return original case version
        for p in possibilities:
            if p.lower() == best_match:
                return p
    return text
# Path utilities
def ensure_directory(path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.
   
    Args:
        path: Directory path
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
def get_data_folder():
    """Get data folder path."""
    return ConfigManager.get_path('DATA_FOLDER', './data/')
def get_vector_db_path():
    """Get vector database path."""
    return ConfigManager.get_path('VECTOR_DB_PATH', './data/vectorstore')
# Validation utilities
def validate_api_keys():
    """Validate that all required API keys are present."""
    required_keys = ['GEMINI_API_KEY']
    missing = []
   
    for key in required_keys:
        if not os.getenv(key):
            missing.append(key)
   
    if missing:
        raise EnvironmentError(
            f"Missing required API keys: {', '.join(missing)}\n"
            f"Please set them in your .env file"
        )
   
    logger.info("✅ All required API keys validated")
# Export
__all__ = [
    'ConfigManager',
    'get_env_var',
    'fuzzy_match_text',
    'ensure_directory',
    'get_data_folder',
    'get_vector_db_path',
    'validate_api_keys'
]
