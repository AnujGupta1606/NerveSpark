#!/usr/bin/env python3
"""
Cloud Data Setup Script for NerveSpark
This script creates the necessary data files for Streamlit Cloud deployment
"""

import os
import json
import pandas as pd
import chromadb
from src.embeddings import RecipeEmbeddingGenerator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_cloud_data():
    """Setup data files for cloud deployment"""
    try:
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Sample recipes data for cloud
        recipes_data = [
            {
                "name": "Grilled Chicken Breast",
                "ingredients": "chicken breast, olive oil, salt, pepper, garlic powder",
                "instructions": "1. Season chicken with salt, pepper, and garlic powder\n2. Heat olive oil in pan\n3. Cook chicken 6-7 minutes per side\n4. Rest before serving",
                "cooking_time": "20 minutes",
                "servings": 2,
                "calories": 165,
                "protein": 31,
                "carbs": 0,
                "fat": 3.6,
                "fiber": 0,
                "category": "protein",
                "health_tags": "high-protein,low-carb,gluten-free"
            },
            {
                "name": "Quinoa Salad Bowl",
                "ingredients": "quinoa, cucumber, tomatoes, red onion, lemon juice, olive oil, feta cheese",
                "instructions": "1. Cook quinoa according to package directions\n2. Dice vegetables\n3. Mix quinoa with vegetables\n4. Add lemon juice and olive oil\n5. Top with feta",
                "cooking_time": "25 minutes",
                "servings": 4,
                "calories": 220,
                "protein": 8,
                "carbs": 35,
                "fat": 6,
                "fiber": 4,
                "category": "salad",
                "health_tags": "vegetarian,high-fiber,gluten-free"
            },
            {
                "name": "Salmon with Vegetables",
                "ingredients": "salmon fillet, broccoli, carrots, olive oil, lemon, herbs",
                "instructions": "1. Preheat oven to 400°F\n2. Place salmon and vegetables on baking sheet\n3. Drizzle with olive oil and lemon\n4. Bake for 15-20 minutes",
                "cooking_time": "25 minutes",
                "servings": 2,
                "calories": 280,
                "protein": 25,
                "carbs": 8,
                "fat": 18,
                "fiber": 3,
                "category": "seafood",
                "health_tags": "high-protein,omega-3,low-carb"
            },
            {
                "name": "Vegetable Stir Fry",
                "ingredients": "mixed vegetables, soy sauce, garlic, ginger, sesame oil, brown rice",
                "instructions": "1. Heat oil in wok\n2. Add garlic and ginger\n3. Stir fry vegetables\n4. Add soy sauce\n5. Serve over brown rice",
                "cooking_time": "15 minutes",
                "servings": 3,
                "calories": 190,
                "protein": 6,
                "carbs": 32,
                "fat": 5,
                "fiber": 6,
                "category": "vegetarian",
                "health_tags": "vegetarian,high-fiber,low-fat"
            },
            {
                "name": "Greek Yogurt Parfait",
                "ingredients": "greek yogurt, berries, granola, honey",
                "instructions": "1. Layer yogurt in glass\n2. Add berries\n3. Sprinkle granola\n4. Drizzle with honey",
                "cooking_time": "5 minutes",
                "servings": 1,
                "calories": 180,
                "protein": 15,
                "carbs": 25,
                "fat": 4,
                "fiber": 3,
                "category": "breakfast",
                "health_tags": "high-protein,probiotic,breakfast"
            }
        ]
        
        # Create recipes.csv
        recipes_df = pd.DataFrame(recipes_data)
        recipes_df.to_csv("data/recipes.csv", index=False)
        logger.info("Created recipes.csv with sample data")
        
        # Create nutrition database
        nutrition_data = [
            {"food": "chicken breast", "calories_per_100g": 165, "protein": 31, "carbs": 0, "fat": 3.6, "fiber": 0},
            {"food": "quinoa", "calories_per_100g": 368, "protein": 14, "carbs": 64, "fat": 6, "fiber": 7},
            {"food": "salmon", "calories_per_100g": 208, "protein": 25, "carbs": 0, "fat": 12, "fiber": 0},
            {"food": "broccoli", "calories_per_100g": 34, "protein": 3, "carbs": 7, "fat": 0.4, "fiber": 3},
            {"food": "greek yogurt", "calories_per_100g": 97, "protein": 9, "carbs": 4, "fat": 5, "fiber": 0}
        ]
        
        nutrition_df = pd.DataFrame(nutrition_data)
        nutrition_df.to_csv("data/nutrition_db.csv", index=False)
        logger.info("Created nutrition_db.csv")
        
        # Create substitutions.json
        substitutions = {
            "dairy_free": {
                "milk": ["almond milk", "oat milk", "coconut milk"],
                "cheese": ["nutritional yeast", "cashew cheese", "vegan cheese"],
                "yogurt": ["coconut yogurt", "almond yogurt"]
            },
            "gluten_free": {
                "wheat flour": ["almond flour", "rice flour", "coconut flour"],
                "bread": ["gluten-free bread", "rice cakes"],
                "pasta": ["rice pasta", "quinoa pasta", "zucchini noodles"]
            },
            "low_carb": {
                "rice": ["cauliflower rice", "broccoli rice"],
                "pasta": ["zucchini noodles", "spaghetti squash"],
                "potatoes": ["turnips", "radishes", "cauliflower"]
            }
        }
        
        with open("data/substitutions.json", "w") as f:
            json.dump(substitutions, f, indent=2)
        logger.info("Created substitutions.json")
        
        # Setup ChromaDB with recipe data
        try:
            client = chromadb.PersistentClient(path="./chroma_db")
            
            # Delete existing collection if it exists
            try:
                client.delete_collection("recipes")
                logger.info("Deleted existing recipes collection")
            except:
                pass
            
            # Create new collection
            collection = client.create_collection(
                name="recipes",
                metadata={"description": "Recipe embeddings for NerveSpark"}
            )
            
            # Generate embeddings
            embedding_gen = RecipeEmbeddingGenerator()
            
            # Add recipes to vector store
            for i, recipe in enumerate(recipes_data):
                # Create searchable text
                searchable_text = f"{recipe['name']} {recipe['ingredients']} {recipe['category']} {recipe['health_tags']}"
                
                # Generate embedding
                embedding = embedding_gen.generate_embedding(searchable_text)
                
                # Add to collection
                collection.add(
                    embeddings=[embedding],
                    documents=[searchable_text],
                    metadatas=[recipe],
                    ids=[f"recipe_{i}"]
                )
            
            logger.info(f"Added {len(recipes_data)} recipes to ChromaDB")
            logger.info(f"Collection count: {collection.count()}")
            
        except Exception as e:
            logger.error(f"ChromaDB setup failed: {e}")
            logger.info("App will use fallback data storage")
        
        logger.info("✅ Cloud data setup completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    setup_cloud_data()
