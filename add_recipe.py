#!/usr/bin/env python3
"""
Recipe Addition Tool for NerveSpark
Use this script to easily add new recipes to the database.
"""

import pandas as pd
import os
from src.data_processor import RecipeDataProcessor
from src.embeddings import RecipeEmbeddingGenerator
from src.vector_store import RecipeVectorStore

def add_new_recipe():
    """Interactive tool to add a new recipe."""
    
    print("=== Add New Recipe to NerveSpark ===")
    
    # Get recipe details from user
    recipe = {}
    
    recipe['name'] = input("Recipe Name: ")
    recipe['cuisine'] = input("Cuisine (e.g., italian, mexican, asian): ")
    recipe['ingredients'] = input("Ingredients (comma-separated): ")
    recipe['instructions'] = input("Instructions: ")
    recipe['prep_time'] = input("Prep Time (e.g., 15 min): ")
    recipe['cook_time'] = input("Cook Time (e.g., 20 min): ")
    recipe['servings'] = int(input("Number of Servings: "))
    recipe['calories'] = int(input("Calories per serving: "))
    recipe['protein'] = int(input("Protein (grams): "))
    recipe['carbs'] = int(input("Carbs (grams): "))
    recipe['fat'] = int(input("Fat (grams): "))
    recipe['fiber'] = int(input("Fiber (grams): "))
    recipe['sugar'] = int(input("Sugar (grams): "))
    recipe['sodium'] = int(input("Sodium (mg): "))
    recipe['description'] = input("Description: ")
    
    return recipe

def add_recipe_to_csv(recipe):
    """Add recipe to CSV file."""
    
    # Load existing CSV
    csv_path = 'data/recipes.csv'
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Create new CSV with headers
        df = pd.DataFrame()
    
    # Create a single-row DataFrame for the new recipe
    new_recipe_df = pd.DataFrame([recipe])
    
    # Process the new recipe using the existing processor
    processor = RecipeDataProcessor()
    processed_df = processor.process_recipe_dataset(new_recipe_df)
    
    # Get the processed recipe as a dictionary
    processed_recipe = processed_df.iloc[0].to_dict()
    
    # Add to existing dataframe
    df = pd.concat([df, processed_df], ignore_index=True)
    
    # Save back to CSV
    df.to_csv(csv_path, index=False)
    print(f"✅ Recipe '{recipe['name']}' added to CSV!")
    
    return processed_recipe

def add_recipe_to_vector_store(recipe):
    """Add recipe to vector database."""
    
    try:
        # Generate embedding
        embedding_gen = RecipeEmbeddingGenerator()
        vector_store = RecipeVectorStore()
        
        # Create document text
        doc_text = f"Recipe: {recipe['name']}\nCuisine: {recipe['cuisine']}\nIngredients: {recipe['ingredients']}\nInstructions: {recipe['instructions']}\nDescription: {recipe['description']}"
        
        # Generate embedding
        embedding = embedding_gen.generate_embedding(doc_text)
        
        # Create unique ID
        recipe_id = f"recipe_{len(vector_store.collection.get()['ids'])}"
        
        # Add to vector store
        vector_store.add_recipes(
            embeddings=[embedding],
            documents=[doc_text],
            metadatas=[recipe],
            ids=[recipe_id]
        )
        
        print(f"✅ Recipe '{recipe['name']}' added to vector database!")
        
    except Exception as e:
        print(f"❌ Error adding to vector store: {e}")

def main():
    """Main function to add a recipe."""
    
    print("This tool will help you add a new recipe to NerveSpark.")
    print("Make sure you're in the NerveSpark project directory.\n")
    
    # Get recipe from user
    recipe = add_new_recipe()
    
    print(f"\n=== Adding Recipe: {recipe['name']} ===")
    
    # Add to CSV
    processed_recipe = add_recipe_to_csv(recipe)
    
    # Add to vector store
    add_recipe_to_vector_store(processed_recipe)
    
    print(f"\n🎉 Recipe '{recipe['name']}' successfully added!")
    print("You can now search for it in the NerveSpark app.")
    
    # Ask if user wants to add another
    add_another = input("\nAdd another recipe? (y/n): ")
    if add_another.lower() == 'y':
        main()

if __name__ == "__main__":
    main()
