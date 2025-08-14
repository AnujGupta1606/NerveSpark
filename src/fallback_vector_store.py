"""
Fallback vector store for cloud deployment when ChromaDB fails.
Uses simple similarity search with numpy/pandas.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
import logging
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FallbackVectorStore:
    """
    Simple fallback vector store using numpy/pandas when ChromaDB fails.
    Provides basic similarity search functionality.
    """
    
    def __init__(self, collection_name: str = "recipes"):
        self.collection_name = collection_name
        self.recipes = []
        self.embeddings = []
        self.metadata = []
        logger.info("Initialized fallback vector store")
    
    def add_recipe(self, recipe_id: str, recipe_data: Dict[str, Any], embedding: np.ndarray):
        """Add a recipe with its embedding to the store."""
        try:
            self.recipes.append(recipe_id)
            self.embeddings.append(embedding)
            self.metadata.append(recipe_data)
            logger.info(f"Added recipe: {recipe_id}")
        except Exception as e:
            logger.error(f"Error adding recipe {recipe_id}: {e}")
    
    def search_similar_recipes(self, query_embedding: np.ndarray, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar recipes using cosine similarity."""
        try:
            if not self.embeddings:
                logger.warning("No recipes in fallback vector store")
                return []
            
            # Convert to numpy array for similarity calculation
            embeddings_matrix = np.array(self.embeddings)
            query_embedding = query_embedding.reshape(1, -1)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_embedding, embeddings_matrix)[0]
            
            # Get top N results
            top_indices = np.argsort(similarities)[::-1][:n_results]
            
            results = []
            for idx in top_indices:
                result = {
                    'id': self.recipes[idx],
                    'metadata': self.metadata[idx],
                    'distance': 1 - similarities[idx],  # Convert similarity to distance
                    'similarity': similarities[idx]
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} similar recipes")
            return results
            
        except Exception as e:
            logger.error(f"Error searching recipes: {e}")
            return []
    
    def get_recipe_by_id(self, recipe_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific recipe by ID."""
        try:
            if recipe_id in self.recipes:
                idx = self.recipes.index(recipe_id)
                return self.metadata[idx]
        except Exception as e:
            logger.error(f"Error getting recipe {recipe_id}: {e}")
        return None
    
    def count(self) -> int:
        """Get the number of recipes in the store."""
        return len(self.recipes)
    
    def list_recipes(self) -> List[str]:
        """List all recipe IDs."""
        return self.recipes.copy()
    
    def clear(self):
        """Clear all data from the store."""
        self.recipes.clear()
        self.embeddings.clear()
        self.metadata.clear()
        logger.info("Cleared fallback vector store")
    
    def save_to_file(self, filepath: str):
        """Save the store to a file."""
        try:
            data = {
                'recipes': self.recipes,
                'embeddings': [emb.tolist() for emb in self.embeddings],
                'metadata': self.metadata
            }
            with open(filepath, 'w') as f:
                json.dump(data, f)
            logger.info(f"Saved fallback store to {filepath}")
        except Exception as e:
            logger.error(f"Error saving store: {e}")
    
    def load_from_file(self, filepath: str):
        """Load the store from a file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.recipes = data['recipes']
            self.embeddings = [np.array(emb) for emb in data['embeddings']]
            self.metadata = data['metadata']
            logger.info(f"Loaded fallback store from {filepath}")
        except Exception as e:
            logger.error(f"Error loading store: {e}")
