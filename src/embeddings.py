from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional
import logging
import os
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecipeEmbeddingGenerator:
    """
    Generates embeddings for recipe text using sentence transformers.
    Supports both local and OpenAI embedding models.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_openai: bool = False):
        self.model_name = model_name
        self.use_openai = use_openai
        self.model = None
        self.embedding_cache = {}
        
        if use_openai:
            import openai
            self.openai = openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            self._load_sentence_transformer()
    
    def _load_sentence_transformer(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            # Try to load with specific device handling for PyTorch 2.8+
            self.model = SentenceTransformer(self.model_name, device='cpu')
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Try alternative approach for PyTorch 2.8+
            try:
                logger.info("Trying alternative loading method...")
                import torch
                # Set default tensor type to avoid meta tensor issues
                torch.set_default_tensor_type('torch.FloatTensor')
                self.model = SentenceTransformer(self.model_name, device='cpu')
                logger.info("Alternative method successful!")
            except Exception as e2:
                logger.error(f"Alternative method failed: {e2}")
                # Final fallback
                logger.info("Using minimal fallback...")
                self.model = None
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if not text or text.strip() == "":
            return np.zeros(384)  # Default embedding size for MiniLM
        
        # Check cache first
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # If model failed to load, return random embedding for demo
        if self.model is None:
            logger.warning("Model not loaded, using random embedding for demo")
            # Generate a consistent random embedding based on text hash
            import hashlib
            text_hash = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            np.random.seed(text_hash % 2**32)
            embedding = np.random.randn(384).astype(np.float32)
            self.embedding_cache[text] = embedding
            return embedding
        
        try:
            if self.use_openai:
                embedding = self._get_openai_embedding(text)
            else:
                embedding = self.model.encode(text, convert_to_numpy=True)
            
            # Cache the embedding
            self.embedding_cache[text] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(384)
    
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
