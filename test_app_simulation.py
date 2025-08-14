#!/usr/bin/env python3
"""
Quick test to check if the app's search is working correctly.
This simulates what the Streamlit app does.
"""

from src.rag_system import NutritionalRAGSystem
from models.user_profile import UserProfile
import logging

logging.basicConfig(level=logging.INFO)

def simulate_app_search():
    """Simulate the exact search process the app uses."""
    print("=== Simulating Streamlit App Search ===")
    
    # Initialize RAG system (like app does)
    rag_system = NutritionalRAGSystem()
    
    # Create default profile (like app does when no profile exists)
    default_profile = UserProfile(
        user_id="demo_user",
        name="Demo User",
        age=30,
        gender="Other",
        weight=70.0,
        height=170.0,
        activity_level="moderate",
        health_conditions=[],
        dietary_restrictions=[],
        allergies=[],
        nutrition_goals={
            "daily_calories": 2000,
            "protein_grams": 50,
            "carb_grams": 250,
            "fat_grams": 67
        }
    )
    
    # Test search for "chicken recipes" (like user did)
    query = "chicken recipes"
    print(f"Searching for: '{query}'")
    
    try:
        results = rag_system.process_user_query(
            query=query,
            user_profile=default_profile.to_dict(),
            max_results=5
        )
        
        print(f"✅ Search successful!")
        print(f"📊 Found {len(results.get('recommended_recipes', []))} recommendations")
        print(f"📈 Total found: {results.get('total_found', 0)}")
        print(f"✅ Safe recipes: {results.get('safe_recipes_count', 0)}")
        
        # Show first few recipes
        for i, recipe in enumerate(results.get('recommended_recipes', [])[:3], 1):
            name = recipe.get('metadata', {}).get('name', 'Unknown')
            score = recipe.get('health_assessment', {}).get('overall_score', 0)
            print(f"  {i}. {name} (Score: {score:.2f})")
        
        # Check AI summary
        if results.get('ai_summary'):
            print(f"🤖 AI Summary: {results['ai_summary'][:100]}...")
        
        return True, results
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return False, None

if __name__ == "__main__":
    success, results = simulate_app_search()
    if success:
        print("\n🎉 The search functionality is working correctly!")
        print("The issue might be in the Streamlit UI display logic.")
    else:
        print("\n💥 There's an issue with the search system itself.")
