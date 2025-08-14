#!/usr/bin/env python3

from src.rag_system import NutritionalRAGSystem
from models.user_profile import UserProfile

def test_rag_search():
    """Test the RAG system search functionality."""
    print("=== Testing NerveSpark RAG Search ===")
    
    # Create a basic user profile
    profile = UserProfile(user_id='test_user')
    rag = NutritionalRAGSystem()

    print('Testing with basic profile...')
    result = rag.process_user_query(
        query='chicken recipes',
        user_profile=profile.to_dict(),
        max_results=3
    )

    print(f'Results found: {len(result.get("recommended_recipes", []))}')
    
    # Print recipe details
    for i, recipe in enumerate(result.get('recommended_recipes', [])[:3]):
        name = recipe.get('metadata', {}).get('name', recipe.get('recipe_name', 'Unknown'))
        print(f'{i+1}. {name}')
        
    # Check other result components
    print(f'\nRecommendations: {len(result.get("recommendations", []))}')
    print(f'Health insights: {len(result.get("health_insights", []))}')
    print(f'AI Summary: {"Yes" if result.get("ai_summary") else "No"}')
    
    # Debug: Print keys in result
    print(f'\nResult keys: {list(result.keys())}')
    print(f'Recommended recipes count: {len(result.get("recommended_recipes", []))}')
    print(f'Total found: {result.get("total_found", 0)}')
    print(f'Safe recipes count: {result.get("safe_recipes_count", 0)}')
    
    return result

if __name__ == "__main__":
    test_rag_search()
