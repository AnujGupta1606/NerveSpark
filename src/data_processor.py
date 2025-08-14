import pandas as pd
import numpy as np
import re
import json
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecipeDataProcessor:
    """
    Processes and cleans recipe data for the RAG system.
    Handles nutritional information standardization and recipe chunking.
    """
    
    def __init__(self):
        self.unit_conversions = {
            # Weight conversions to grams
            'kg': 1000, 'g': 1, 'mg': 0.001, 'oz': 28.35, 'lb': 453.59,
            'pound': 453.59, 'ounce': 28.35,
            # Volume conversions to ml
            'l': 1000, 'ml': 1, 'cup': 240, 'tbsp': 15, 'tsp': 5,
            'tablespoon': 15, 'teaspoon': 5, 'pint': 473, 'quart': 946,
            'gallon': 3785, 'fl oz': 29.57, 'fluid ounce': 29.57
        }
        
    def clean_ingredient_text(self, ingredient: str) -> str:
        """Clean and normalize ingredient text."""
        if pd.isna(ingredient):
            return ""
        
        # Convert to lowercase
        ingredient = ingredient.lower().strip()
        
        # Remove extra whitespace
        ingredient = re.sub(r'\s+', ' ', ingredient)
        
        # Remove quantity information (numbers and units)
        ingredient = re.sub(r'\d+\.?\d*\s*(cups?|tbsp|tsp|oz|lbs?|g|kg|ml|l)\s*', '', ingredient)
        
        # Remove common cooking terms
        cooking_terms = ['chopped', 'diced', 'minced', 'sliced', 'grated', 'fresh', 'dried', 'ground']
        for term in cooking_terms:
            ingredient = ingredient.replace(term, '').strip()
        
        # Remove extra commas and parentheses content
        ingredient = re.sub(r'\([^)]*\)', '', ingredient)
        ingredient = ingredient.replace(',', '').strip()
        
        return ingredient
    
    def extract_nutritional_value(self, nutrition_text: str, nutrient: str) -> float:
        """Extract specific nutrient value from nutrition text."""
        if pd.isna(nutrition_text):
            return 0.0
        
        nutrition_text = str(nutrition_text).lower()
        
        # Common patterns for nutrients
        patterns = {
            'calories': r'(\d+\.?\d*)\s*cal',
            'protein': r'(\d+\.?\d*)\s*g.*protein',
            'fat': r'(\d+\.?\d*)\s*g.*fat',
            'carbs': r'(\d+\.?\d*)\s*g.*carb',
            'fiber': r'(\d+\.?\d*)\s*g.*fiber',
            'sugar': r'(\d+\.?\d*)\s*g.*sugar',
            'sodium': r'(\d+\.?\d*)\s*mg.*sodium'
        }
        
        pattern = patterns.get(nutrient.lower())
        if pattern:
            match = re.search(pattern, nutrition_text)
            if match:
                return float(match.group(1))
        
        return 0.0
    
    def standardize_nutrition_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize nutritional information across recipes."""
        logger.info("Standardizing nutrition data...")
        
        # Define expected nutritional columns
        nutrition_columns = ['calories', 'protein', 'fat', 'carbs', 'fiber', 'sugar', 'sodium']
        
        # If nutrition data is in a single column, extract individual nutrients
        if 'nutrition' in df.columns and not any(col in df.columns for col in nutrition_columns):
            for nutrient in nutrition_columns:
                df[nutrient] = df['nutrition'].apply(lambda x: self.extract_nutritional_value(x, nutrient))
        
        # Fill missing nutritional values with 0
        for col in nutrition_columns:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        # Calculate derived metrics
        df['protein_calories'] = df['protein'] * 4  # 4 cal per gram
        df['carb_calories'] = df['carbs'] * 4
        df['fat_calories'] = df['fat'] * 9  # 9 cal per gram
        
        # Calculate macronutrient ratios
        total_macro_calories = df['protein_calories'] + df['carb_calories'] + df['fat_calories']
        df['protein_ratio'] = np.where(total_macro_calories > 0, df['protein_calories'] / total_macro_calories, 0)
        df['carb_ratio'] = np.where(total_macro_calories > 0, df['carb_calories'] / total_macro_calories, 0)
        df['fat_ratio'] = np.where(total_macro_calories > 0, df['fat_calories'] / total_macro_calories, 0)
        
        return df
    
    def clean_recipe_text(self, text: str) -> str:
        """Clean recipe instructions and descriptions."""
        if pd.isna(text):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', str(text))
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        
        return text.strip()
    
    def process_ingredients_list(self, ingredients: str) -> List[str]:
        """Process ingredients string into a clean list."""
        if pd.isna(ingredients):
            return []
        
        # Handle different separators
        if isinstance(ingredients, str):
            # Try common separators
            separators = ['\n', ';', ',', '|']
            for sep in separators:
                if sep in ingredients:
                    ingredient_list = ingredients.split(sep)
                    break
            else:
                ingredient_list = [ingredients]
        else:
            ingredient_list = [str(ingredients)]
        
        # Clean each ingredient
        cleaned_ingredients = []
        for ingredient in ingredient_list:
            cleaned = self.clean_ingredient_text(ingredient)
            if cleaned and len(cleaned) > 2:  # Filter out very short strings
                cleaned_ingredients.append(cleaned)
        
        return cleaned_ingredients
    
    def chunk_recipe_data(self, recipe_row: pd.Series) -> Dict[str, str]:
        """Create different chunks of recipe data for embedding."""
        chunks = {}
        
        # Basic info chunk
        basic_info = f"Recipe: {recipe_row.get('name', '')}\n"
        basic_info += f"Cuisine: {recipe_row.get('cuisine', 'Unknown')}\n"
        basic_info += f"Prep Time: {recipe_row.get('prep_time', 'Unknown')}\n"
        basic_info += f"Cook Time: {recipe_row.get('cook_time', 'Unknown')}\n"
        basic_info += f"Servings: {recipe_row.get('servings', 'Unknown')}"
        chunks['basic_info'] = basic_info
        
        # Ingredients chunk
        ingredients = recipe_row.get('ingredients', '')
        if isinstance(ingredients, list):
            ingredients_text = "Ingredients:\n" + "\n".join([f"- {ing}" for ing in ingredients])
        else:
            ingredients_list = self.process_ingredients_list(ingredients)
            ingredients_text = "Ingredients:\n" + "\n".join([f"- {ing}" for ing in ingredients_list])
        chunks['ingredients'] = ingredients_text
        
        # Instructions chunk
        instructions = self.clean_recipe_text(recipe_row.get('instructions', ''))
        chunks['instructions'] = f"Instructions:\n{instructions}"
        
        # Nutrition chunk
        nutrition_info = f"Nutritional Information (per serving):\n"
        nutrition_info += f"Calories: {recipe_row.get('calories', 0)}\n"
        nutrition_info += f"Protein: {recipe_row.get('protein', 0)}g\n"
        nutrition_info += f"Carbohydrates: {recipe_row.get('carbs', 0)}g\n"
        nutrition_info += f"Fat: {recipe_row.get('fat', 0)}g\n"
        nutrition_info += f"Fiber: {recipe_row.get('fiber', 0)}g\n"
        nutrition_info += f"Sugar: {recipe_row.get('sugar', 0)}g\n"
        nutrition_info += f"Sodium: {recipe_row.get('sodium', 0)}mg"
        chunks['nutrition'] = nutrition_info
        
        return chunks
    
    def process_recipe_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main method to process the entire recipe dataset."""
        logger.info(f"Processing {len(df)} recipes...")
        
        # Clean text columns
        text_columns = ['name', 'description', 'instructions', 'cuisine']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(self.clean_recipe_text)
        
        # Process ingredients
        if 'ingredients' in df.columns:
            df['ingredients_list'] = df['ingredients'].apply(self.process_ingredients_list)
            df['ingredient_count'] = df['ingredients_list'].apply(len)
        
        # Standardize nutrition data
        df = self.standardize_nutrition_data(df)
        
        # Add difficulty rating based on ingredients and cook time
        df['difficulty'] = self.calculate_difficulty_rating(df)
        
        # Add diet tags
        df['diet_tags'] = df.apply(self.identify_diet_tags, axis=1)
        
        logger.info("Recipe processing completed!")
        return df
    
    def calculate_difficulty_rating(self, df: pd.DataFrame) -> pd.Series:
        """Calculate recipe difficulty based on various factors."""
        difficulty_scores = []
        
        for _, row in df.iterrows():
            score = 1  # Base easy score
            
            # Add complexity based on ingredient count
            ingredient_count = row.get('ingredient_count', 0)
            if ingredient_count > 15:
                score += 2
            elif ingredient_count > 10:
                score += 1
            
            # Add complexity based on cook time
            cook_time = str(row.get('cook_time', '')).lower()
            if 'hour' in cook_time or any(str(i) in cook_time for i in range(60, 300)):
                score += 2
            elif any(str(i) in cook_time for i in range(30, 60)):
                score += 1
            
            # Add complexity based on instructions length
            instructions = str(row.get('instructions', ''))
            if len(instructions) > 500:
                score += 1
            
            # Cap at 5 (very hard)
            difficulty_scores.append(min(score, 5))
        
        return pd.Series(difficulty_scores)
    
    def identify_diet_tags(self, row: pd.Series) -> List[str]:
        """Identify dietary tags for a recipe based on ingredients."""
        tags = []
        ingredients_text = str(row.get('ingredients', '')).lower()
        
        # Check for vegan (no animal products)
        animal_products = ['meat', 'chicken', 'beef', 'pork', 'fish', 'salmon', 'tuna', 
                          'milk', 'cheese', 'butter', 'cream', 'egg', 'honey', 'gelatin']
        if not any(product in ingredients_text for product in animal_products):
            tags.append('vegan')
        
        # Check for vegetarian (no meat/fish)
        meat_products = ['meat', 'chicken', 'beef', 'pork', 'fish', 'salmon', 'tuna', 'turkey']
        if not any(product in ingredients_text for product in meat_products):
            tags.append('vegetarian')
        
        # Check for gluten-free
        gluten_ingredients = ['wheat', 'flour', 'bread', 'pasta', 'barley', 'rye', 'oats']
        if not any(ingredient in ingredients_text for ingredient in gluten_ingredients):
            tags.append('gluten_free')
        
        # Check for dairy-free
        dairy_ingredients = ['milk', 'cheese', 'butter', 'cream', 'yogurt']
        if not any(ingredient in ingredients_text for ingredient in dairy_ingredients):
            tags.append('dairy_free')
        
        # Check for low carb (less than 20g carbs)
        if row.get('carbs', 0) < 20:
            tags.append('low_carb')
        
        # Check for high protein (more than 20g protein)
        if row.get('protein', 0) > 20:
            tags.append('high_protein')
        
        return tags
    
    def validate_recipe_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate processed recipe data and return quality metrics."""
        validation_results = {
            'total_recipes': len(df),
            'missing_names': df['name'].isna().sum(),
            'missing_ingredients': df['ingredients'].isna().sum(),
            'missing_instructions': df['instructions'].isna().sum(),
            'recipes_with_nutrition': (df['calories'] > 0).sum(),
            'avg_ingredients_per_recipe': df.get('ingredient_count', pd.Series([0])).mean(),
            'cuisine_diversity': df['cuisine'].nunique() if 'cuisine' in df.columns else 0
        }
        
        logger.info(f"Validation Results: {validation_results}")
        return validation_results
