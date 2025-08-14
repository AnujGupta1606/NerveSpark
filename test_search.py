from src.rag_system import NutritionalRAGSystem

# Initialize RAG system
rag = NutritionalRAGSystem()
print('RAG system initialized successfully')

# Test search
query = 'chicken'
print(f'Searching for: {query}')

try:
    # Use the correct method name
    user_profile = {
        'age': 30,
        'dietary_restrictions': [],
        'health_conditions': [],
        'health_goals': []
    }
    
    result = rag.process_user_query(query, user_profile, max_results=3)
    print(f'Response type: {type(result)}')
    print(f'Response keys: {result.keys() if isinstance(result, dict) else "Not a dict"}')
    
    if isinstance(result, dict) and 'recommended_recipes' in result:
        recipes = result['recommended_recipes']
        print(f'Found {len(recipes)} recipes:')
        for i, recipe in enumerate(recipes, 1):
            print(f'{i}. {recipe.get("metadata", {}).get("name", "Unknown")}')
            print(f'   Cuisine: {recipe.get("metadata", {}).get("cuisine", "Unknown")}')
            print(f'   Calories: {recipe.get("metadata", {}).get("calories", "Unknown")}')
            print()
    else:
        print(f'Unexpected result format: {result}')
        
except Exception as e:
    print(f'Search error: {e}')
    import traceback
    traceback.print_exc()

# Test different queries
queries = ['pasta', 'salad', 'breakfast', 'vegetarian']
for q in queries:
    print(f'\n--- Testing query: {q} ---')
    try:
        user_profile = {'dietary_restrictions': [], 'health_conditions': []}
        result = rag.process_user_query(q, user_profile, max_results=2)
        if isinstance(result, dict) and 'recommended_recipes' in result:
            recipes = result['recommended_recipes']
            print(f'Found {len(recipes)} results')
            for recipe in recipes:
                print(f'- {recipe.get("metadata", {}).get("name", "Unknown")}')
        else:
            print(f'No recipes found or unexpected format')
    except Exception as e:
        print(f'Error: {e}')
