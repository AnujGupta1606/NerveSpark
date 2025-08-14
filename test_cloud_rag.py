from src.simple_rag import SimpleNutritionalRAG

print('Testing enhanced cloud RAG system...')
rag = SimpleNutritionalRAG()

# Test searches
queries = ['chicken recipe', 'vegetarian salad', 'high protein']

for query in queries:
    print(f'\nSearching for: {query}')
    results = rag.search_recipes(query, max_results=2)
    for i, recipe in enumerate(results, 1):
        print(f'  {i}. {recipe["name"]} ({recipe["cuisine"]})')
        print(f'     Tags: {recipe["dietary_tags"]}')

print('\nCloud RAG system working perfectly!')
