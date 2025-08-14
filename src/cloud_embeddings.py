"""
Cloud-optimized embedding system that doesn't require HuggingFace downloads.
Uses pre-computed embeddings and simple similarity calculations.
"""

import numpy as np
import hashlib
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class CloudEmbeddingGenerator:
    """
    Simple embedding generator for cloud deployment.
    No external dependencies, uses hash-based embeddings.
    """
    
    def __init__(self):
        self.embedding_dim = 384
        self.vocabulary = self._create_vocabulary()
        
    def _create_vocabulary(self):
        """Create a simple vocabulary for cooking terms."""
        return {
            # Proteins
            'chicken': [1.0, 0.8, 0.2, 0.0],
            'beef': [1.0, 0.9, 0.1, 0.0],
            'fish': [1.0, 0.6, 0.0, 0.8],
            'salmon': [1.0, 0.6, 0.0, 0.9],
            'turkey': [1.0, 0.7, 0.3, 0.0],
            'pork': [1.0, 0.85, 0.15, 0.0],
            'eggs': [0.9, 0.8, 0.2, 0.1],
            'tofu': [0.8, 0.2, 0.0, 0.9],
            'beans': [0.7, 0.0, 0.8, 0.9],
            'lentils': [0.7, 0.0, 0.9, 0.9],
            
            # Vegetables
            'broccoli': [0.1, 0.0, 0.9, 1.0],
            'spinach': [0.1, 0.0, 0.8, 1.0],
            'lettuce': [0.1, 0.0, 0.7, 1.0],
            'tomato': [0.2, 0.0, 0.6, 0.9],
            'onion': [0.15, 0.0, 0.5, 0.8],
            'garlic': [0.2, 0.0, 0.3, 0.7],
            'carrot': [0.2, 0.0, 0.7, 0.8],
            'potato': [0.3, 0.0, 0.8, 0.6],
            'cucumber': [0.1, 0.0, 0.5, 0.9],
            
            # Grains
            'rice': [0.4, 0.0, 0.9, 0.3],
            'pasta': [0.5, 0.0, 0.8, 0.2],
            'bread': [0.6, 0.0, 0.7, 0.1],
            'quinoa': [0.5, 0.0, 0.8, 0.7],
            'oats': [0.4, 0.0, 0.8, 0.6],
            
            # Cooking methods
            'grilled': [0.8, 0.9, 0.0, 0.0],
            'baked': [0.7, 0.8, 0.0, 0.0],
            'fried': [0.9, 0.5, 0.0, 0.0],
            'steamed': [0.3, 0.2, 0.0, 0.9],
            'roasted': [0.8, 0.9, 0.0, 0.0],
            'boiled': [0.2, 0.1, 0.0, 0.8],
            
            # Diet types
            'vegetarian': [0.0, 0.0, 0.9, 1.0],
            'vegan': [0.0, 0.0, 0.8, 1.0],
            'keto': [0.9, 1.0, 0.1, 0.0],
            'paleo': [0.8, 0.8, 0.3, 0.2],
            'healthy': [0.3, 0.2, 0.8, 0.9],
            'protein': [1.0, 0.8, 0.0, 0.0],
            'low-carb': [0.8, 0.6, 0.2, 0.0],
            
            # Meal types
            'breakfast': [0.6, 0.3, 0.5, 0.4],
            'lunch': [0.7, 0.5, 0.6, 0.5],
            'dinner': [0.8, 0.7, 0.4, 0.3],
            'snack': [0.4, 0.2, 0.3, 0.6],
            'salad': [0.2, 0.0, 0.8, 1.0],
            'soup': [0.3, 0.1, 0.6, 0.8],
            'curry': [0.6, 0.4, 0.5, 0.3],
            'stir-fry': [0.7, 0.6, 0.6, 0.4],
        }
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate a simple embedding for text."""
        if not text:
            return np.zeros(self.embedding_dim)
        
        text = text.lower().strip()
        words = text.split()
        
        # Start with base embedding
        embedding = np.zeros(self.embedding_dim)
        
        # Use vocabulary if available
        vocab_score = np.zeros(4)  # [protein, fat, carb, health]
        vocab_matches = 0
        
        for word in words:
            if word in self.vocabulary:
                vocab_score += np.array(self.vocabulary[word])
                vocab_matches += 1
        
        if vocab_matches > 0:
            vocab_score /= vocab_matches
        
        # Fill embedding with vocabulary-based features
        embedding[:4] = vocab_score
        
        # Use hash-based features for the rest
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hex to numbers
        for i in range(4, min(self.embedding_dim, len(text_hash))):
            hex_char = text_hash[i - 4]
            embedding[i] = int(hex_char, 16) / 15.0 - 0.5
        
        # Add word-based features
        for i, word in enumerate(words[:20]):
            word_hash = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
            start_idx = 4 + (i * 18)
            end_idx = min(start_idx + 18, self.embedding_dim)
            
            for j in range(start_idx, end_idx):
                embedding[j] += ((word_hash >> (j - start_idx)) & 1) * 0.1
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)
    
    def calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings."""
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def search_similar(self, query_embedding: np.ndarray, recipe_embeddings: List[np.ndarray], n_results: int = 5) -> List[int]:
        """Find most similar recipes to query."""
        similarities = []
        
        for i, recipe_emb in enumerate(recipe_embeddings):
            similarity = self.calculate_similarity(query_embedding, recipe_emb)
            similarities.append((similarity, i))
        
        # Sort by similarity (descending)
        similarities.sort(reverse=True)
        
        # Return indices of top matches
        return [idx for _, idx in similarities[:n_results]]
