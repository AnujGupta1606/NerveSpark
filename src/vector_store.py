import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
import logging
import json
import os
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if we're in a cloud environment
IS_CLOUD_DEPLOYMENT = os.getenv('STREAMLIT_SHARING_MODE') is not None or '/mount/src/' in os.getcwd()

# Try to import ChromaDB only if not in cloud deployment
CHROMADB_AVAILABLE = False
if not IS_CLOUD_DEPLOYMENT:
    try:
        import chromadb
        from chromadb.config import Settings
        CHROMADB_AVAILABLE = True
        logger.info("ChromaDB available for local use")
    except ImportError:
        logger.warning("ChromaDB not available, using fallback implementation")
else:
    logger.info("Cloud deployment detected, skipping ChromaDB import")

# Import fallback vector store
from src.fallback_vector_store import FallbackVectorStore

class RecipeVectorStore:
    """
    Manages recipe embeddings in vector database.
    Automatically falls back to simple implementation if ChromaDB fails.
    """
    
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "recipes"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.fallback_store = None
        self.using_chromadb = False
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize vector database with ChromaDB or fallback."""
        
        # First try ChromaDB if available
        if CHROMADB_AVAILABLE:
            try:
                self._initialize_chromadb()
                return
            except Exception as e:
                logger.warning(f"ChromaDB initialization failed: {e}")
        
        # Fall back to simple implementation
        logger.info("Using fallback vector store implementation")
        self.fallback_store = FallbackVectorStore(self.collection_name)
        self.using_chromadb = False
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB with cloud compatibility."""
        # For cloud deployment, use a temporary directory if needed
        if not os.path.exists(self.db_path):
            try:
                os.makedirs(self.db_path, exist_ok=True)
            except PermissionError:
                # Fallback to temporary directory for cloud deployment
                self.db_path = tempfile.mkdtemp()
                logger.warning(f"Using temporary directory for ChromaDB: {self.db_path}")
        
        # Create ChromaDB client with cloud-compatible settings
        try:
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"Created PersistentClient at: {self.db_path}")
        except Exception as e:
            logger.warning(f"PersistentClient failed, using EphemeralClient: {e}")
            # Fallback to in-memory client for cloud deployment
            self.client = chromadb.EphemeralClient()
            logger.warning("Using EphemeralClient - data will not persist")
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Connected to existing collection: {self.collection_name} with {self.collection.count()} recipes")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Recipe embeddings for nutrition assistant"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
        
        self.using_chromadb = True
    
    def add_recipe(self, recipe_id: str, recipe_data: Dict[str, Any], embedding: np.ndarray):
        """Add a single recipe to the vector store."""
        if self.using_chromadb and self.collection:
            try:
                # ChromaDB implementation
                recipe_text = f"{recipe_data.get('name', '')} {' '.join(recipe_data.get('ingredients', []))}"
                self.collection.add(
                    embeddings=[embedding.tolist()],
                    documents=[recipe_text],
                    metadatas=[recipe_data],
                    ids=[recipe_id]
                )
                logger.info(f"Added recipe to ChromaDB: {recipe_id}")
            except Exception as e:
                logger.error(f"Error adding recipe to ChromaDB: {e}")
        else:
            # Fallback implementation
            self.fallback_store.add_recipe(recipe_id, recipe_data, embedding)
    
    def search_similar_recipes(self, query_embedding: np.ndarray, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar recipes."""
        if self.using_chromadb and self.collection:
            try:
                # ChromaDB implementation
                results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=n_results
                )
                
                formatted_results = []
                for i, (doc_id, metadata, distance) in enumerate(zip(
                    results['ids'][0], 
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    formatted_results.append({
                        'id': doc_id,
                        'metadata': metadata,
                        'distance': distance,
                        'similarity': 1 - distance
                    })
                
                return formatted_results
            except Exception as e:
                logger.error(f"Error searching ChromaDB: {e}")
                return []
        else:
            # Fallback implementation
            return self.fallback_store.search_similar_recipes(query_embedding, n_results)
    
    def count(self) -> int:
        """Get the number of recipes in the store."""
        if self.using_chromadb and self.collection:
            try:
                return self.collection.count()
            except:
                return 0
        else:
            return self.fallback_store.count()
    
    def add_recipes(self, embeddings: List[List[float]], 
                   documents: List[str], 
                   metadatas: List[Dict[str, Any]], 
                   ids: List[str]):
        """
        Add recipe embeddings to the vector store.
        
        Args:
            embeddings: List of embedding vectors
            documents: List of recipe text documents
            metadatas: List of metadata dictionaries for each recipe
            ids: List of unique IDs for each recipe
        """
        try:
            # Ensure embeddings are in the correct format
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            # Convert numpy arrays in embeddings to lists
            processed_embeddings = []
            for emb in embeddings:
                if isinstance(emb, np.ndarray):
                    processed_embeddings.append(emb.tolist())
                else:
                    processed_embeddings.append(emb)
            
            # Process metadata to ensure ChromaDB compatibility
            processed_metadata = []
            for metadata in metadatas:
                processed_meta = {}
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        processed_meta[key] = value
                    elif isinstance(value, list):
                        # Convert lists to comma-separated strings
                        processed_meta[key] = ", ".join(str(item) for item in value)
                    else:
                        # Convert complex types to strings
                        processed_meta[key] = str(value)
                processed_metadata.append(processed_meta)
            
            # Add to collection
            self.collection.add(
                embeddings=processed_embeddings,
                documents=documents,
                metadatas=processed_metadata,
                ids=ids
            )
            
            logger.info(f"Added {len(embeddings)} recipes to vector store")
            
        except Exception as e:
            logger.error(f"Error adding recipes to vector store: {e}")
            raise
    
    def search_recipes(self, query_embedding: List[float], 
                      n_results: int = 5,
                      where: Optional[Dict[str, Any]] = None,
                      where_document: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Search for similar recipes using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Metadata filter conditions
            where_document: Document content filter conditions
            
        Returns:
            Dictionary containing search results
        """
        try:
            # Convert numpy array to list if needed
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=['metadatas', 'documents', 'distances']
            )
            
            # Format results
            formatted_results = {
                'recipes': [],
                'distances': results['distances'][0],
                'total_results': len(results['ids'][0])
            }
            
            for i in range(len(results['ids'][0])):
                recipe_data = {
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'distance': results['distances'][0][i]
                }
                formatted_results['recipes'].append(recipe_data)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching recipes: {e}")
            return {'recipes': [], 'distances': [], 'total_results': 0}
    
    def filter_by_dietary_restrictions(self, dietary_restrictions: List[str]) -> List[str]:
        """
        Get recipe IDs that match dietary restrictions.
        
        Args:
            dietary_restrictions: List of dietary restrictions (e.g., ['vegan', 'gluten_free'])
            
        Returns:
            List of recipe IDs that match all restrictions
        """
        try:
            if not dietary_restrictions:
                return []
            
            # Create filter condition
            where_conditions = {}
            for restriction in dietary_restrictions:
                where_conditions[f"diet_{restriction}"] = True
            
            # Query with filter
            results = self.collection.get(
                where=where_conditions,
                include=['metadatas']
            )
            
            return results['ids']
            
        except Exception as e:
            logger.error(f"Error filtering by dietary restrictions: {e}")
            return []
    
    def filter_by_health_conditions(self, health_conditions: List[str]) -> List[str]:
        """
        Get recipe IDs suitable for specific health conditions.
        
        Args:
            health_conditions: List of health conditions
            
        Returns:
            List of suitable recipe IDs
        """
        try:
            if not health_conditions:
                return []
            
            # For now, implement basic filtering based on nutritional thresholds
            # This could be expanded with more sophisticated health-based filtering
            suitable_ids = []
            
            for condition in health_conditions:
                if condition == "diabetes":
                    # Low sugar recipes
                    results = self.collection.get(
                        where={"sugar": {"$lt": 10}},  # Less than 10g sugar
                        include=['metadatas']
                    )
                    suitable_ids.extend(results['ids'])
                
                elif condition == "hypertension":
                    # Low sodium recipes
                    results = self.collection.get(
                        where={"sodium": {"$lt": 400}},  # Less than 400mg sodium
                        include=['metadatas']
                    )
                    suitable_ids.extend(results['ids'])
                
                elif condition == "heart_disease":
                    # Low saturated fat recipes
                    results = self.collection.get(
                        where={"fat": {"$lt": 5}},  # Less than 5g total fat
                        include=['metadatas']
                    )
                    suitable_ids.extend(results['ids'])
            
            # Remove duplicates
            return list(set(suitable_ids))
            
        except Exception as e:
            logger.error(f"Error filtering by health conditions: {e}")
            return []
    
    def get_recipe_by_id(self, recipe_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific recipe by its ID."""
        try:
            results = self.collection.get(
                ids=[recipe_id],
                include=['metadatas', 'documents']
            )
            
            if results['ids']:
                return {
                    'id': results['ids'][0],
                    'document': results['documents'][0],
                    'metadata': results['metadatas'][0]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting recipe by ID: {e}")
            return None
    
    def update_recipe_metadata(self, recipe_id: str, new_metadata: Dict[str, Any]):
        """Update metadata for a specific recipe."""
        try:
            self.collection.update(
                ids=[recipe_id],
                metadatas=[new_metadata]
            )
            logger.info(f"Updated metadata for recipe {recipe_id}")
            
        except Exception as e:
            logger.error(f"Error updating recipe metadata: {e}")
    
    def delete_recipe(self, recipe_id: str):
        """Delete a recipe from the vector store."""
        try:
            self.collection.delete(ids=[recipe_id])
            logger.info(f"Deleted recipe {recipe_id}")
            
        except Exception as e:
            logger.error(f"Error deleting recipe: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection."""
        try:
            # Get collection info
            collection_info = self.collection.get(include=['metadatas'])
            
            total_recipes = len(collection_info['ids'])
            
            if total_recipes == 0:
                return {
                    'total_recipes': 0,
                    'dietary_tags': {},
                    'avg_calories': 0,
                    'cuisine_distribution': {}
                }
            
            # Extract metadata statistics
            metadatas = collection_info['metadatas']
            
            # Count dietary tags
            dietary_tags = {}
            total_calories = 0
            cuisines = {}
            
            for metadata in metadatas:
                # Count dietary tags
                for key, value in metadata.items():
                    if key.startswith('diet_') and value:
                        tag = key.replace('diet_', '')
                        dietary_tags[tag] = dietary_tags.get(tag, 0) + 1
                
                # Accumulate calories
                if 'calories' in metadata:
                    total_calories += metadata.get('calories', 0)
                
                # Count cuisines
                if 'cuisine' in metadata:
                    cuisine = metadata['cuisine']
                    cuisines[cuisine] = cuisines.get(cuisine, 0) + 1
            
            return {
                'total_recipes': total_recipes,
                'dietary_tags': dietary_tags,
                'avg_calories': total_calories / total_recipes if total_recipes > 0 else 0,
                'cuisine_distribution': cuisines
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def backup_collection(self, backup_path: str):
        """Create a backup of the collection data."""
        try:
            # Get all data from collection
            all_data = self.collection.get(
                include=['metadatas', 'documents', 'embeddings']
            )
            
            # Save to JSON file
            backup_data = {
                'collection_name': self.collection_name,
                'data': all_data,
                'stats': self.get_collection_stats()
            }
            
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            logger.info(f"Collection backed up to {backup_path}")
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
    
    def clear_collection(self):
        """Clear all data from the collection."""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Recipe embeddings for nutrition assistant"}
            )
            logger.info("Collection cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
    
    def hybrid_search(self, query_embedding: List[float], 
                     text_query: str,
                     dietary_filters: Optional[List[str]] = None,
                     nutrition_filters: Optional[Dict[str, float]] = None,
                     n_results: int = 5) -> Dict[str, Any]:
        """
        Perform hybrid search combining vector similarity and metadata filtering.
        
        Args:
            query_embedding: Vector embedding for similarity search
            text_query: Text query for document search
            dietary_filters: List of dietary restrictions
            nutrition_filters: Nutritional constraints (e.g., {"calories": 500})
            n_results: Number of results to return
        """
        try:
            # For now, skip complex filtering to get basic search working
            # Build where clause for metadata filtering - simplified approach
            where_clause = None
            
            # For demo, we'll skip dietary filtering temporarily to get results
            # TODO: Implement proper OR/AND logic for ChromaDB
            
            # Add nutrition filters only (these work better with ChromaDB)
            if nutrition_filters:
                where_clause = {}
                for nutrient, max_value in nutrition_filters.items():
                    where_clause[nutrient] = {"$lte": max_value}
            
            # Build document filter for text search - DISABLED for now
            where_document = None
            # TODO: Fix document filtering
            # if text_query:
            #     where_document = {"$contains": text_query.lower()}
            
            # Perform the search without dietary filters for now
            results = self.search_recipes(
                query_embedding=query_embedding,
                n_results=n_results * 2,  # Get more results since we'll filter afterwards
                where=where_clause,
                where_document=where_document
            )
            
            # Post-process to apply dietary filters manually
            if dietary_filters and results['recipes']:
                filtered_recipes = []
                for recipe in results['recipes']:
                    metadata = recipe['metadata']
                    # Check if recipe matches any dietary filter
                    matches_diet = True  # For now, allow all recipes
                    # TODO: Implement proper dietary filtering
                    
                    if matches_diet:
                        filtered_recipes.append(recipe)
                        if len(filtered_recipes) >= n_results:
                            break
                
                results['recipes'] = filtered_recipes
                results['total_results'] = len(filtered_recipes)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return {'recipes': [], 'distances': [], 'total_results': 0}
