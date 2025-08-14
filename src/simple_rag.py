"""
Simplified RAG system for cloud deployment.
Uses minimal dependencies and sample data.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleNutritionalRAG:
    """
    Simplified RAG system for cloud deployment.
    Provides basic nutrition recommendations using sample data.
    """
    
    def __init__(self):
        self.sample_recipes = self._load_sample_recipes()
        self.sample_nutrition = self._load_nutrition_data()
        logger.info("Initialized simple RAG system for cloud deployment")
    
    def _load_sample_recipes(self):
        """Load sample recipes for demonstration."""
        return [
            {
                "name": "Grilled Chicken Salad",
                "ingredients": ["chicken breast", "lettuce", "tomatoes", "cucumber", "olive oil"],
                "instructions": "Grill chicken, chop vegetables, mix with olive oil",
                "cuisine": "Mediterranean",
                "dietary_tags": ["gluten-free", "low-carb"],
                "calories": 350,
                "protein": 35,
                "carbs": 10,
                "fat": 15,
                "cooking_time": "25 minutes",
                "difficulty": "easy"
            },
            {
                "name": "Vegetarian Pasta",
                "ingredients": ["pasta", "tomatoes", "basil", "garlic", "olive oil"],
                "instructions": "Cook pasta, make tomato sauce with herbs",
                "cuisine": "Italian",
                "dietary_tags": ["vegetarian"],
                "calories": 450,
                "protein": 12,
                "carbs": 65,
                "fat": 18,
                "cooking_time": "20 minutes",
                "difficulty": "easy"
            },
            {
                "name": "Quinoa Buddha Bowl",
                "ingredients": ["quinoa", "chickpeas", "spinach", "avocado", "tahini"],
                "instructions": "Cook quinoa, roast chickpeas, assemble bowl",
                "cuisine": "Healthy",
                "dietary_tags": ["vegan", "gluten-free"],
                "calories": 380,
                "protein": 18,
                "carbs": 45,
                "fat": 16,
                "cooking_time": "30 minutes",
                "difficulty": "medium"
            },
            {
                "name": "Salmon with Vegetables",
                "ingredients": ["salmon fillet", "broccoli", "carrots", "lemon", "herbs"],
                "instructions": "Bake salmon with vegetables and lemon",
                "cuisine": "Healthy",
                "dietary_tags": ["low-carb", "high-protein"],
                "calories": 320,
                "protein": 28,
                "carbs": 12,
                "fat": 20,
                "cooking_time": "25 minutes",
                "difficulty": "medium"
            },
            {
                "name": "Greek Yogurt Parfait",
                "ingredients": ["greek yogurt", "berries", "granola", "honey"],
                "instructions": "Layer yogurt with berries and granola",
                "cuisine": "Healthy",
                "dietary_tags": ["vegetarian", "high-protein"],
                "calories": 250,
                "protein": 15,
                "carbs": 30,
                "fat": 8,
                "cooking_time": "5 minutes",
                "difficulty": "easy"
            }
        ]
    
    def _load_nutrition_data(self):
        """Load nutrition data for ingredients."""
        return {
            "chicken breast": {"calories": 165, "protein": 31, "carbs": 0, "fat": 3.6},
            "lettuce": {"calories": 15, "protein": 1.4, "carbs": 2.9, "fat": 0.2},
            "tomatoes": {"calories": 18, "protein": 0.9, "carbs": 3.9, "fat": 0.2},
            "pasta": {"calories": 131, "protein": 5, "carbs": 25, "fat": 1.1},
            "quinoa": {"calories": 120, "protein": 4.4, "carbs": 22, "fat": 1.9},
            "salmon fillet": {"calories": 208, "protein": 25, "carbs": 0, "fat": 12},
            "greek yogurt": {"calories": 59, "protein": 10, "carbs": 3.6, "fat": 0.4}
        }
    
    def search_recipes(self, query: str, dietary_restrictions: List[str] = None, max_results: int = 3):
        """
        Simple text-based recipe search.
        """
        query_lower = query.lower()
        dietary_restrictions = dietary_restrictions or []
        
        # Simple keyword matching
        matching_recipes = []
        for recipe in self.sample_recipes:
            # Check if query matches recipe name or ingredients
            if (query_lower in recipe['name'].lower() or 
                any(query_lower in ingredient.lower() for ingredient in recipe['ingredients'])):
                
                # Check dietary restrictions
                if dietary_restrictions:
                    recipe_tags = recipe.get('dietary_tags', [])
                    if any(restriction in recipe_tags for restriction in dietary_restrictions):
                        matching_recipes.append(recipe)
                else:
                    matching_recipes.append(recipe)
        
        # If no matches, return all recipes
        if not matching_recipes:
            matching_recipes = self.sample_recipes
        
        return matching_recipes[:max_results]
    
    def get_nutrition_info(self, ingredients: List[str]):
        """Get nutritional information for ingredients."""
        nutrition_info = {}
        for ingredient in ingredients:
            if ingredient in self.sample_nutrition:
                nutrition_info[ingredient] = self.sample_nutrition[ingredient]
        return nutrition_info
    
    def generate_recommendation(self, query: str, user_profile: Dict[str, Any] = None):
        """
        Generate nutrition recommendation based on query and user profile.
        """
        user_profile = user_profile or {}
        
        # Extract dietary restrictions
        dietary_restrictions = user_profile.get('dietary_restrictions', [])
        health_conditions = user_profile.get('health_conditions', [])
        
        # Search for recipes
        recipes = self.search_recipes(query, dietary_restrictions)
        
        # Generate response
        response = {
            'query': query,
            'recipes': recipes,
            'recommendations': self._generate_health_recommendations(recipes, health_conditions),
            'nutrition_summary': self._calculate_nutrition_summary(recipes)
        }
        
        return response
    
    def _generate_health_recommendations(self, recipes: List[Dict], health_conditions: List[str]):
        """Generate health-based recommendations."""
        recommendations = []
        
        for condition in health_conditions:
            if condition == 'diabetes':
                recommendations.append("Choose low-carb options like the Grilled Chicken Salad")
            elif condition == 'hypertension':
                recommendations.append("Opt for low-sodium recipes with fresh ingredients")
            elif condition == 'heart_disease':
                recommendations.append("Focus on lean proteins and omega-3 rich foods like salmon")
        
        if not recommendations:
            recommendations.append("All recipes provide balanced nutrition for general health")
        
        return recommendations
    
    def _calculate_nutrition_summary(self, recipes: List[Dict]):
        """Calculate average nutrition across recipes."""
        if not recipes:
            return {}
        
        total_calories = sum(recipe.get('calories', 0) for recipe in recipes)
        total_protein = sum(recipe.get('protein', 0) for recipe in recipes)
        total_carbs = sum(recipe.get('carbs', 0) for recipe in recipes)
        total_fat = sum(recipe.get('fat', 0) for recipe in recipes)
        
        count = len(recipes)
        
        return {
            'avg_calories': round(total_calories / count),
            'avg_protein': round(total_protein / count, 1),
            'avg_carbs': round(total_carbs / count, 1),
            'avg_fat': round(total_fat / count, 1)
        }
