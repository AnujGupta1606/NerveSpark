"""
Data setup script for NerveSpark nutrition assistant.
Downloads sample recipe data and initializes the vector database.
"""

import pandas as pd
import numpy as np
import json
import os
import logging
from typing import Dict, List, Any
import requests
from io import StringIO

# Import our modules
from src.data_processor import RecipeDataProcessor
from src.embeddings import RecipeEmbeddingGenerator
from src.vector_store import RecipeVectorStore
from utils.helpers import generate_recipe_tags, categorize_cuisine, estimate_difficulty_level
from models.user_profile import UserProfileManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSetupManager:
    """Manages the setup and initialization of recipe data for NerveSpark."""
    
    def __init__(self):
        self.data_processor = RecipeDataProcessor()
        self.embedding_generator = RecipeEmbeddingGenerator()
        self.vector_store = RecipeVectorStore()
        self.profile_manager = UserProfileManager()
        
    def create_sample_recipe_data(self) -> pd.DataFrame:
        """Create sample recipe data for demonstration."""
        sample_recipes = [
            {
                'name': 'Mediterranean Quinoa Salad',
                'cuisine': 'mediterranean',
                'ingredients': 'quinoa, cucumber, tomatoes, red onion, feta cheese, olive oil, lemon juice, oregano, salt, pepper',
                'instructions': 'Cook quinoa according to package instructions. Let cool. Dice cucumber, tomatoes, and red onion. Crumble feta cheese. Mix all ingredients with olive oil, lemon juice, oregano, salt, and pepper.',
                'prep_time': '15 min',
                'cook_time': '15 min',
                'servings': 4,
                'calories': 320,
                'protein': 12,
                'carbs': 45,
                'fat': 12,
                'fiber': 6,
                'sugar': 8,
                'sodium': 450,
                'description': 'A fresh and healthy Mediterranean salad perfect for lunch or as a side dish.'
            },
            {
                'name': 'Grilled Chicken with Steamed Broccoli',
                'cuisine': 'american',
                'ingredients': 'chicken breast, broccoli, olive oil, garlic, lemon, salt, pepper, herbs',
                'instructions': 'Season chicken with salt, pepper, and herbs. Grill for 6-8 minutes per side. Steam broccoli until tender. Serve with lemon and olive oil drizzle.',
                'prep_time': '10 min',
                'cook_time': '20 min',
                'servings': 2,
                'calories': 280,
                'protein': 35,
                'carbs': 8,
                'fat': 12,
                'fiber': 4,
                'sugar': 3,
                'sodium': 320,
                'description': 'A simple, high-protein meal perfect for maintaining a healthy diet.'
            },
            {
                'name': 'Vegetarian Black Bean Tacos',
                'cuisine': 'mexican',
                'ingredients': 'black beans, corn tortillas, avocado, lime, cilantro, red onion, tomato, lettuce, cumin, chili powder',
                'instructions': 'Heat black beans with cumin and chili powder. Warm tortillas. Dice avocado, tomato, and onion. Assemble tacos with beans, vegetables, and cilantro. Serve with lime.',
                'prep_time': '15 min',
                'cook_time': '10 min',
                'servings': 3,
                'calories': 245,
                'protein': 10,
                'carbs': 40,
                'fat': 8,
                'fiber': 12,
                'sugar': 5,
                'sodium': 380,
                'description': 'Delicious plant-based tacos packed with protein and fiber.'
            },
            {
                'name': 'Asian Stir-Fry with Tofu',
                'cuisine': 'asian',
                'ingredients': 'firm tofu, mixed vegetables, soy sauce, ginger, garlic, sesame oil, brown rice, green onions',
                'instructions': 'Press and cube tofu. Stir-fry tofu until golden. Add vegetables, garlic, and ginger. Add soy sauce and sesame oil. Serve over brown rice with green onions.',
                'prep_time': '15 min',
                'cook_time': '15 min',
                'servings': 2,
                'calories': 310,
                'protein': 18,
                'carbs': 35,
                'fat': 14,
                'fiber': 6,
                'sugar': 8,
                'sodium': 600,
                'description': 'A flavorful and nutritious Asian-inspired dish with plant-based protein.'
            },
            {
                'name': 'Heart-Healthy Salmon with Sweet Potato',
                'cuisine': 'american',
                'ingredients': 'salmon fillet, sweet potato, spinach, olive oil, lemon, dill, garlic, salt, pepper',
                'instructions': 'Roast sweet potato cubes with olive oil. Season salmon with dill, salt, and pepper. Bake salmon for 12-15 minutes. Sauté spinach with garlic. Serve together with lemon.',
                'prep_time': '10 min',
                'cook_time': '25 min',
                'servings': 2,
                'calories': 420,
                'protein': 32,
                'carbs': 25,
                'fat': 22,
                'fiber': 5,
                'sugar': 8,
                'sodium': 280,
                'description': 'Omega-3 rich salmon with nutritious sweet potato, perfect for heart health.'
            },
            {
                'name': 'Low-Carb Zucchini Noodles with Pesto',
                'cuisine': 'italian',
                'ingredients': 'zucchini, basil pesto, cherry tomatoes, pine nuts, parmesan cheese, olive oil',
                'instructions': 'Spiralize zucchini into noodles. Sauté lightly in olive oil. Toss with pesto. Top with halved cherry tomatoes, pine nuts, and parmesan cheese.',
                'prep_time': '10 min',
                'cook_time': '5 min',
                'servings': 2,
                'calories': 180,
                'protein': 8,
                'carbs': 12,
                'fat': 14,
                'fiber': 4,
                'sugar': 8,
                'sodium': 320,
                'description': 'A light, low-carb alternative to pasta that doesn\'t compromise on flavor.'
            },
            {
                'name': 'Diabetes-Friendly Lentil Soup',
                'cuisine': 'mediterranean',
                'ingredients': 'red lentils, vegetable broth, onion, carrots, celery, garlic, turmeric, cumin, spinach, lemon juice',
                'instructions': 'Sauté onion, carrots, and celery. Add garlic and spices. Add lentils and broth. Simmer until lentils are tender. Stir in spinach and lemon juice.',
                'prep_time': '15 min',
                'cook_time': '30 min',
                'servings': 4,
                'calories': 220,
                'protein': 14,
                'carbs': 35,
                'fat': 3,
                'fiber': 15,
                'sugar': 6,
                'sodium': 480,
                'description': 'A fiber-rich, low-glycemic soup perfect for managing blood sugar levels.'
            },
            {
                'name': 'High-Protein Greek Yogurt Bowl',
                'cuisine': 'mediterranean',
                'ingredients': 'Greek yogurt, berries, almonds, honey, chia seeds, granola',
                'instructions': 'Layer Greek yogurt in a bowl. Top with fresh berries, chopped almonds, chia seeds, and a small amount of granola. Drizzle with honey.',
                'prep_time': '5 min',
                'cook_time': '0 min',
                'servings': 1,
                'calories': 350,
                'protein': 25,
                'carbs': 35,
                'fat': 15,
                'fiber': 8,
                'sugar': 22,
                'sodium': 100,
                'description': 'A protein-packed breakfast or snack that supports muscle health and satiety.'
            },
            {
                'name': 'Low-Sodium Herb-Crusted Chicken',
                'cuisine': 'american',
                'ingredients': 'chicken breast, fresh herbs, lemon zest, garlic, olive oil, black pepper, paprika',
                'instructions': 'Mix herbs, lemon zest, garlic, and olive oil. Coat chicken with herb mixture. Bake at 375°F for 20-25 minutes until cooked through.',
                'prep_time': '10 min',
                'cook_time': '25 min',
                'servings': 3,
                'calories': 240,
                'protein': 30,
                'carbs': 2,
                'fat': 12,
                'fiber': 1,
                'sugar': 1,
                'sodium': 150,
                'description': 'Flavorful chicken with minimal sodium, perfect for heart-healthy diets.'
            },
            {
                'name': 'Vegan Buddha Bowl',
                'cuisine': 'asian',
                'ingredients': 'quinoa, chickpeas, kale, sweet potato, avocado, tahini, lemon juice, maple syrup, sesame seeds',
                'instructions': 'Roast sweet potato and chickpeas. Cook quinoa. Massage kale with lemon. Make tahini dressing with lemon and maple syrup. Assemble bowl and top with sesame seeds.',
                'prep_time': '20 min',
                'cook_time': '30 min',
                'servings': 2,
                'calories': 480,
                'protein': 18,
                'carbs': 65,
                'fat': 18,
                'fiber': 14,
                'sugar': 12,
                'sodium': 290,
                'description': 'A complete plant-based meal with all essential nutrients and vibrant flavors.'
            }
        ]
        
        return pd.DataFrame(sample_recipes)
    
    def download_recipe_dataset(self) -> pd.DataFrame:
        """Download recipe dataset from a public source (fallback to sample data)."""
        try:
            # Try to download from a public recipe dataset
            # This is a placeholder - in practice, you might use Kaggle datasets or other sources
            logger.info("Attempting to download recipe dataset...")
            
            # For now, we'll use our sample data
            # In a real implementation, you could download from:
            # - Kaggle recipe datasets
            # - Recipe APIs like Spoonacular
            # - Open recipe databases
            
            sample_data = self.create_sample_recipe_data()
            logger.info(f"Using sample dataset with {len(sample_data)} recipes")
            return sample_data
            
        except Exception as e:
            logger.warning(f"Could not download external dataset: {e}")
            logger.info("Using built-in sample data")
            return self.create_sample_recipe_data()
    
    def process_and_enhance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and enhance the recipe dataset."""
        logger.info("Processing and enhancing recipe data...")
        
        # Use our data processor
        processed_df = self.data_processor.process_recipe_dataset(df)
        
        # Add additional enhancements
        processed_df['recipe_tags'] = processed_df.apply(
            lambda row: generate_recipe_tags(row.to_dict()), axis=1
        )
        
        # Ensure cuisine is categorized
        for idx, row in processed_df.iterrows():
            if pd.isna(row.get('cuisine')) or row.get('cuisine') == 'unknown':
                ingredients_list = row.get('ingredients_list', [])
                cuisine = categorize_cuisine(
                    row.get('name', ''), 
                    ingredients_list, 
                    row.get('instructions', '')
                )
                processed_df.at[idx, 'cuisine'] = cuisine
        
        # Add diet compatibility flags
        for idx, row in processed_df.iterrows():
            diet_tags = row.get('diet_tags', [])
            
            # Add individual diet flags for easier filtering
            processed_df.at[idx, 'diet_vegan'] = 'vegan' in diet_tags
            processed_df.at[idx, 'diet_vegetarian'] = 'vegetarian' in diet_tags
            processed_df.at[idx, 'diet_gluten_free'] = 'gluten_free' in diet_tags
            processed_df.at[idx, 'diet_dairy_free'] = 'dairy_free' in diet_tags
            processed_df.at[idx, 'diet_low_carb'] = 'low_carb' in diet_tags
            processed_df.at[idx, 'diet_high_protein'] = 'high_protein' in diet_tags
        
        logger.info("Data processing completed!")
        return processed_df
    
    def create_embeddings_and_store(self, df: pd.DataFrame):
        """Create embeddings for recipes and store in vector database."""
        logger.info("Creating embeddings and storing in vector database...")
        
        # Clear existing collection
        self.vector_store.clear_collection()
        
        recipes_data = []
        embeddings = []
        documents = []
        metadatas = []
        ids = []
        
        for idx, row in df.iterrows():
            # Create different chunks for each recipe
            chunks = self.data_processor.chunk_recipe_data(row)
            
            # Create a composite document for embedding
            composite_doc = f"{chunks['basic_info']}\n\n{chunks['ingredients']}\n\n{chunks['nutrition']}"
            
            # Generate embedding
            embedding = self.embedding_generator.generate_embedding(composite_doc)
            
            # Prepare metadata
            metadata = {
                'name': row.get('name', ''),
                'cuisine': row.get('cuisine', ''),
                'calories': float(row.get('calories', 0)),
                'protein': float(row.get('protein', 0)),
                'carbs': float(row.get('carbs', 0)),
                'fat': float(row.get('fat', 0)),
                'fiber': float(row.get('fiber', 0)),
                'sugar': float(row.get('sugar', 0)),
                'sodium': float(row.get('sodium', 0)),
                'servings': int(row.get('servings', 1)),
                'prep_time': str(row.get('prep_time', '')),
                'cook_time': str(row.get('cook_time', '')),
                'difficulty': row.get('difficulty', 1),
                'diet_vegan': bool(row.get('diet_vegan', False)),
                'diet_vegetarian': bool(row.get('diet_vegetarian', False)),
                'diet_gluten_free': bool(row.get('diet_gluten_free', False)),
                'diet_dairy_free': bool(row.get('diet_dairy_free', False)),
                'diet_low_carb': bool(row.get('diet_low_carb', False)),
                'diet_high_protein': bool(row.get('diet_high_protein', False)),
                'recipe_tags': row.get('recipe_tags', []),
                'ingredients': row.get('ingredients', ''),
                'instructions': row.get('instructions', ''),
                'description': row.get('description', '')
            }
            
            recipes_data.append(row.to_dict())
            embeddings.append(embedding.tolist())
            documents.append(composite_doc)
            metadatas.append(metadata)
            ids.append(f"recipe_{idx}")
        
        # Add to vector store
        self.vector_store.add_recipes(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Successfully stored {len(embeddings)} recipes in vector database!")
    
    def save_processed_data(self, df: pd.DataFrame):
        """Save processed data to CSV files."""
        logger.info("Saving processed data...")
        
        # Save main recipe data
        df.to_csv('data/recipes.csv', index=False)
        
        # Create nutrition database
        nutrition_columns = ['name', 'calories', 'protein', 'carbs', 'fat', 'fiber', 'sugar', 'sodium']
        nutrition_df = df[nutrition_columns].copy()
        nutrition_df.to_csv('data/nutrition_db.csv', index=False)
        
        logger.info("Data saved successfully!")
    
    def setup_sample_user_profiles(self):
        """Create sample user profiles for testing."""
        logger.info("Setting up sample user profiles...")
        
        # Sample profile 1: Health-conscious vegetarian
        profile1 = self.profile_manager.create_sample_profile("health_conscious_vegetarian")
        
        # Sample profile 2: Diabetic user
        from models.user_profile import UserProfile
        profile2 = UserProfile(
            user_id="diabetic_user",
            name="John Smith",
            age=55,
            gender="male",
            height_cm=175,
            weight_kg=80,
            dietary_restrictions=["low_carb"],
            health_conditions=["diabetes"],
            weight_goal="lose",
            cooking_skill_level="beginner"
        )
        self.profile_manager.save_profile(profile2)
        
        # Sample profile 3: Heart-healthy focused
        profile3 = UserProfile(
            user_id="heart_healthy_user",
            name="Mary Johnson",
            age=62,
            gender="female",
            height_cm=160,
            weight_kg=70,
            health_conditions=["heart_disease", "hypertension"],
            dietary_restrictions=["low_sodium"],
            allergies=["shellfish"],
            cooking_skill_level="advanced"
        )
        self.profile_manager.save_profile(profile3)
        
        logger.info("Sample user profiles created!")
    
    def validate_setup(self) -> bool:
        """Validate that the setup was successful."""
        logger.info("Validating setup...")
        
        try:
            # Check vector store
            stats = self.vector_store.get_collection_stats()
            if stats.get('total_recipes', 0) == 0:
                logger.error("No recipes found in vector store")
                return False
            
            # Check data files
            if not os.path.exists('data/recipes.csv'):
                logger.error("Recipe CSV file not found")
                return False
            
            # Check user profiles
            profiles = self.profile_manager.list_profiles()
            if len(profiles) == 0:
                logger.warning("No user profiles found")
            
            logger.info(f"Setup validation successful! Found {stats.get('total_recipes', 0)} recipes")
            return True
            
        except Exception as e:
            logger.error(f"Setup validation failed: {e}")
            return False
    
    def run_full_setup(self):
        """Run the complete data setup process."""
        logger.info("Starting NerveSpark data setup...")
        
        try:
            # Step 1: Download or create recipe data
            df = self.download_recipe_dataset()
            
            # Step 2: Process and enhance the data
            processed_df = self.process_and_enhance_data(df)
            
            # Step 3: Create embeddings and store in vector database
            self.create_embeddings_and_store(processed_df)
            
            # Step 4: Save processed data
            self.save_processed_data(processed_df)
            
            # Step 5: Setup sample user profiles
            self.setup_sample_user_profiles()
            
            # Step 6: Validate setup
            if self.validate_setup():
                logger.info("✅ NerveSpark setup completed successfully!")
                print("\n🎉 Setup Complete! You can now run: streamlit run app.py")
            else:
                logger.error("❌ Setup validation failed")
                
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise

def main():
    """Main function to run the setup."""
    setup_manager = DataSetupManager()
    setup_manager.run_full_setup()

if __name__ == "__main__":
    main()
