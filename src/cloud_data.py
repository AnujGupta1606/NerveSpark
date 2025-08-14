"""
Cloud-compatible data initialization for NerveSpark
This module handles data setup for cloud deployments where file system access may be limited.
"""

import streamlit as st
import pandas as pd
import os
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

@st.cache_data
def load_sample_recipes():
    """Load sample recipes for cloud deployment."""
    sample_recipes = [
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
    
    return pd.DataFrame(sample_recipes)

@st.cache_data
def load_nutrition_data():
    """Load nutrition data for cloud deployment."""
    nutrition_data = [
        {"ingredient": "chicken breast", "calories_per_100g": 165, "protein": 31, "carbs": 0, "fat": 3.6},
        {"ingredient": "lettuce", "calories_per_100g": 15, "protein": 1.4, "carbs": 2.9, "fat": 0.2},
        {"ingredient": "tomatoes", "calories_per_100g": 18, "protein": 0.9, "carbs": 3.9, "fat": 0.2},
        {"ingredient": "pasta", "calories_per_100g": 131, "protein": 5, "carbs": 25, "fat": 1.1},
        {"ingredient": "quinoa", "calories_per_100g": 120, "protein": 4.4, "carbs": 22, "fat": 1.9},
        {"ingredient": "salmon fillet", "calories_per_100g": 208, "protein": 25, "carbs": 0, "fat": 12},
        {"ingredient": "greek yogurt", "calories_per_100g": 59, "protein": 10, "carbs": 3.6, "fat": 0.4}
    ]
    
    return pd.DataFrame(nutrition_data)

def initialize_cloud_data():
    """Initialize data for cloud deployment."""
    try:
        # Load sample data
        recipes_df = load_sample_recipes()
        nutrition_df = load_nutrition_data()
        
        logger.info(f"Loaded {len(recipes_df)} sample recipes for cloud deployment")
        logger.info(f"Loaded {len(nutrition_df)} nutrition entries")
        
        return recipes_df, nutrition_df
        
    except Exception as e:
        logger.error(f"Error initializing cloud data: {e}")
        # Return minimal fallback data
        return pd.DataFrame(), pd.DataFrame()
