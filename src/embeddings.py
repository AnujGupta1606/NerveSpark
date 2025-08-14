from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional
import logging
import os
import pickle
import time
import hashlib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecipeEmbeddingGenerator:
    """
    Generates embeddings for recipe text using sentence transformers.
    Supports both local and OpenAI embedding models with rate limiting and caching.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_openai: bool = False):
        self.model_name = model_name
        self.use_openai = use_openai
        self.model = None
        self.embedding_cache = {}
        self.cache_file = f"embedding_cache_{model_name.replace('/', '_')}.pkl"
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum 1 second between requests
        
        # Load cached embeddings
        self._load_cache()
        
        if use_openai:
            import openai
            self.openai = openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            self._load_sentence_transformer()
    
    def _load_cache(self):
        """Load embedding cache from file."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
        except Exception as e:
            logger.warning(f"Could not load cache: {e}")
            self.embedding_cache = {}
    
    def _save_cache(self):
        """Save embedding cache to file."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")
    
    def _rate_limit(self):
        """Implement rate limiting to avoid 429 errors."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _load_sentence_transformer(self):
        """Load the sentence transformer model with retry logic."""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Loading sentence transformer model: {self.model_name} (attempt {attempt + 1})")
                
                # Rate limit the request
                self._rate_limit()
                
                # Set offline mode if we've failed before
                if attempt > 0:
                    os.environ['TRANSFORMERS_OFFLINE'] = '1'
                    os.environ['HF_HUB_OFFLINE'] = '1'
                
                # Try to load with specific device handling
                self.model = SentenceTransformer(
                    self.model_name, 
                    device='cpu',
                    cache_folder='./model_cache'  # Local cache folder
                )
                logger.info("Model loaded successfully!")
                return
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("All attempts failed, using fallback...")
                    self._create_fallback_model()
    
    def _create_fallback_model(self):
        """Create a simple fallback embedding model."""
        logger.warning("Creating fallback embedding model...")
        
        class FallbackEmbedder:
            def encode(self, texts, **kwargs):
                # Simple word-based embedding using hash
                if isinstance(texts, str):
                    texts = [texts]
                
                embeddings = []
                for text in texts:
                    # Create a simple 384-dimensional embedding
                    words = text.lower().split()
                    embedding = np.zeros(384)
                    
                    for i, word in enumerate(words[:20]):  # Use first 20 words
                        hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
                        for j in range(19):  # Fill embedding dimensions
                            embedding[i * 19 + j] = (hash_val >> (j * 2)) & 0x3F
                    
                    # Normalize
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    
                    embeddings.append(embedding)
                
                return np.array(embeddings)
        
        self.model = FallbackEmbedder()
        logger.info("Fallback embedding model created")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text with caching and rate limiting."""
        if not text or text.strip() == "":
            return np.zeros(384)  # Default embedding size for MiniLM
        
        # Create cache key
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        # Check cache first
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # If model failed to load, use fallback
        if self.model is None:
            logger.warning("Model not loaded, using fallback embedding")
            return self._create_fallback_embedding(text)
        
        try:
            # Rate limit for API calls
            if not self.use_openai:
                self._rate_limit()
            
            if self.use_openai:
                embedding = self._get_openai_embedding(text)
            else:
                # Handle both single text and batch processing
                if hasattr(self.model, 'encode'):
                    embedding = self.model.encode([text], convert_to_numpy=True)[0]
                else:
                    # Fallback model
                    embedding = self.model.encode([text])[0]
            
            # Cache the embedding
            self.embedding_cache[cache_key] = embedding
            
            # Periodically save cache
            if len(self.embedding_cache) % 10 == 0:
                self._save_cache()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return fallback embedding instead of zeros
            return self._create_fallback_embedding(text)
    
    def _create_fallback_embedding(self, text: str) -> np.ndarray:
        """Create a simple fallback embedding."""
        # Generate a consistent embedding based on text hash
        text_hash = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        np.random.seed(text_hash % 2**32)
        embedding = np.random.randn(384).astype(np.float32)
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    
    def _get_openai_embedding(self, text: str) -> np.ndarray:
        """Get embedding using OpenAI API."""
        try:
            response = self.openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return np.array(response['data'][0]['embedding'])
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            return np.zeros(1536)  # OpenAI embedding size
    
    def generate_batch_embeddings(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts."""
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        embeddings = []
        
        if self.use_openai:
            # Process one by one for OpenAI (rate limiting)
            for text in texts:
                embeddings.append(self.generate_embedding(text))
        else:
            # Process in batches for sentence transformers
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                try:
                    batch_embeddings = self.model.encode(
                        batch_texts, 
                        convert_to_numpy=True,
                        show_progress_bar=True
                    )
                    embeddings.extend(batch_embeddings)
                    
                    # Cache the embeddings
                    for text, embedding in zip(batch_texts, batch_embeddings):
                        self.embedding_cache[text] = embedding
                        
                except Exception as e:
                    logger.error(f"Error in batch processing: {e}")
                    # Fallback to individual processing
                    for text in batch_texts:
                        embeddings.append(self.generate_embedding(text))
        
        logger.info("Batch embedding generation completed!")
        return embeddings
    
    def generate_recipe_embeddings(self, recipe_chunks: Dict[str, List[str]]) -> Dict[str, List[np.ndarray]]:
        """
        Generate embeddings for different types of recipe chunks.
        
        Args:
            recipe_chunks: Dictionary with chunk types as keys and list of texts as values
            
        Returns:
            Dictionary with chunk types as keys and list of embeddings as values
        """
        chunk_embeddings = {}
        
        for chunk_type, texts in recipe_chunks.items():
            logger.info(f"Processing {chunk_type} chunks...")
            chunk_embeddings[chunk_type] = self.generate_batch_embeddings(texts)
        
        return chunk_embeddings
    
    def create_composite_embedding(self, embeddings_dict: Dict[str, np.ndarray], weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Create a composite embedding from multiple chunk embeddings.
        
        Args:
            embeddings_dict: Dictionary of chunk type to embedding
            weights: Optional weights for different chunk types
        """
        if weights is None:
            weights = {
                'basic_info': 0.2,
                'ingredients': 0.4,
                'instructions': 0.2,
                'nutrition': 0.2
            }
        
        composite_embedding = np.zeros_like(list(embeddings_dict.values())[0])
        total_weight = 0
        
        for chunk_type, embedding in embeddings_dict.items():
            weight = weights.get(chunk_type, 0.25)  # Default equal weight
            composite_embedding += weight * embedding
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            composite_embedding /= total_weight
        
        return composite_embedding
    
    def save_embeddings_cache(self, filepath: str):
        """Save embedding cache to disk."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.info(f"Embedding cache saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def load_embeddings_cache(self, filepath: str):
        """Load embedding cache from disk."""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Embedding cache loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
    
    def get_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def find_most_similar(self, query_embedding: np.ndarray, candidate_embeddings: List[np.ndarray], top_k: int = 5) -> List[tuple]:
        """
        Find the most similar embeddings to a query embedding.
        
        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = []
        
        for i, candidate_embedding in enumerate(candidate_embeddings):
            similarity = self.get_similarity(query_embedding, candidate_embedding)
            similarities.append((i, similarity))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_embedding_stats(self) -> Dict[str, any]:
        """Get statistics about generated embeddings."""
        if not self.embedding_cache:
            return {"cache_size": 0}
        
        embeddings = list(self.embedding_cache.values())
        embedding_dims = [emb.shape[0] for emb in embeddings]
        
        return {
            "cache_size": len(self.embedding_cache),
            "embedding_dimension": embeddings[0].shape[0] if embeddings else 0,
            "total_embeddings": len(embeddings),
            "avg_embedding_norm": np.mean([np.linalg.norm(emb) for emb in embeddings]),
            "model_name": self.model_name
        }
