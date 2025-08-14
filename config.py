import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Model Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer model
OPENAI_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 150
TEMPERATURE = 0.7

# Vector Database Configuration
VECTOR_DB_PATH = "./chroma_db"
COLLECTION_NAME = "recipes"
TOP_K_RESULTS = 5

# Data Paths
DATA_DIR = "./data"
RECIPES_CSV = os.path.join(DATA_DIR, "recipes.csv")
NUTRITION_DB = os.path.join(DATA_DIR, "nutrition_db.csv")
SUBSTITUTIONS_JSON = os.path.join(DATA_DIR, "substitutions.json")

# User Database
USER_DB_PATH = "./user_profiles.db"

# Health Condition Guidelines (per day)
HEALTH_CONDITIONS = {
    "diabetes": {
        "max_sugar": 25,  # grams
        "max_carbs": 45,  # grams per meal
        "preferred_fiber": 25,  # grams
        "avoid_ingredients": ["white sugar", "corn syrup", "white flour"]
    },
    "hypertension": {
        "max_sodium": 1500,  # mg
        "min_potassium": 3500,  # mg
        "avoid_ingredients": ["salt", "soy sauce", "pickles", "processed meat"]
    },
    "heart_disease": {
        "max_saturated_fat": 13,  # grams
        "max_cholesterol": 200,  # mg
        "min_omega3": 1,  # grams
        "avoid_ingredients": ["butter", "lard", "palm oil", "trans fat"]
    },
    "kidney_disease": {
        "max_protein": 50,  # grams per day
        "max_phosphorus": 800,  # mg
        "max_potassium": 2000,  # mg
        "avoid_ingredients": ["processed meat", "dairy", "nuts", "whole grains"]
    }
}

# Dietary Restrictions
DIETARY_RESTRICTIONS = {
    "vegan": {
        "avoid_ingredients": ["meat", "poultry", "fish", "dairy", "eggs", "honey", "gelatin"]
    },
    "vegetarian": {
        "avoid_ingredients": ["meat", "poultry", "fish", "gelatin"]
    },
    "gluten_free": {
        "avoid_ingredients": ["wheat", "barley", "rye", "oats", "bread", "pasta", "flour"]
    },
    "dairy_free": {
        "avoid_ingredients": ["milk", "cheese", "butter", "cream", "yogurt", "whey", "casein"]
    },
    "nut_free": {
        "avoid_ingredients": ["almonds", "peanuts", "walnuts", "cashews", "pecans", "hazelnuts", "pine nuts"]
    },
    "low_carb": {
        "max_carbs_per_serving": 20,  # grams
        "avoid_ingredients": ["bread", "pasta", "rice", "potatoes", "sugar"]
    },
    "keto": {
        "max_carbs_per_serving": 5,  # grams
        "min_fat_percentage": 70,
        "avoid_ingredients": ["grains", "legumes", "fruits", "sugar", "starchy vegetables"]
    }
}

# Common Allergies
ALLERGIES = {
    "nuts": ["almonds", "peanuts", "tree nuts", "walnuts", "cashews", "pecans"],
    "shellfish": ["shrimp", "crab", "lobster", "oysters", "mussels", "clams"],
    "fish": ["salmon", "tuna", "cod", "bass", "trout"],
    "eggs": ["eggs", "egg whites", "egg yolks", "mayonnaise"],
    "soy": ["soy sauce", "tofu", "tempeh", "soy milk", "edamame"],
    "dairy": ["milk", "cheese", "butter", "cream", "yogurt"]
}

# Nutritional Goals (daily recommended values)
DAILY_NUTRITION_GOALS = {
    "calories": {"male": 2500, "female": 2000},
    "protein": {"male": 56, "female": 46},  # grams
    "carbs": {"male": 325, "female": 325},  # grams
    "fat": {"male": 78, "female": 65},  # grams
    "fiber": {"male": 38, "female": 25},  # grams
    "sodium": {"max": 2300},  # mg
    "sugar": {"max": 50}  # grams
}

# App Configuration
APP_TITLE = "NerveSpark - Intelligent Nutrition Assistant"
APP_ICON = "🍽️"
SIDEBAR_TITLE = "Your Health Profile"
