#!/usr/bin/env python3

from src.vector_store import RecipeVectorStore
import numpy as np

def test_vector_store_robustness():
    """Test vector store with fallback scenarios."""
    print("=== Testing Vector Store Robustness ===")
    
    # Create vector store
    vector_store = RecipeVectorStore()
    
    # Test search with dummy embedding
    dummy_embedding = np.random.rand(384)  # Typical sentence transformer size
    
    print(f"Using ChromaDB: {vector_store.using_chromadb}")
    print(f"Collection exists: {vector_store.collection is not None}")
    print(f"Fallback store exists: {vector_store.fallback_store is not None}")
    
    # Test search_similar_recipes
    try:
        results = vector_store.search_similar_recipes(dummy_embedding, n_results=3)
        print(f"search_similar_recipes returned: {len(results)} results")
        for i, result in enumerate(results[:2]):
            name = result.get('metadata', {}).get('name', 'Unknown')
            print(f"  {i+1}. {name}")
    except Exception as e:
        print(f"search_similar_recipes failed: {e}")
    
    # Test search_recipes
    try:
        results = vector_store.search_recipes(dummy_embedding.tolist(), n_results=3)
        print(f"search_recipes returned: {results.get('total_results', 0)} results")
        for i, recipe in enumerate(results.get('recipes', [])[:2]):
            name = recipe.get('metadata', {}).get('name', 'Unknown')
            print(f"  {i+1}. {name}")
    except Exception as e:
        print(f"search_recipes failed: {e}")
    
    print("=== Test completed ===")

if __name__ == "__main__":
    test_vector_store_robustness()
