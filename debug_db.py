import chromadb

client = chromadb.PersistentClient(path='./chroma_db')
try:
    collection = client.get_collection('recipes')
    print(f'Collection found: {collection.name}')
    print(f'Recipe count: {collection.count()}')
    if collection.count() > 0:
        print('Sample records:')
        results = collection.peek()
        print(f'IDs: {results["ids"][:3]}')
    else:
        print('Collection is empty!')
except Exception as e:
    print(f'Collection error: {e}')
    print('Available collections:')
    collections = client.list_collections()
    for col in collections:
        print(f'- {col.name}')
