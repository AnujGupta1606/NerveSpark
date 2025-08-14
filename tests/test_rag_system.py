"""
Basic tests for the NerveSpark RAG system.
Run with: python -m pytest tests/
"""

import unittest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestRAGSystem(unittest.TestCase):
    """Test cases for the RAG system components."""
    
    def setUp(self):
        """Set up test fixtures."""
        pass
    
    def test_user_profile_creation(self):
        """Test user profile creation."""
        try:
            from models.user_profile import UserProfile
            
            profile = UserProfile(
                user_id="test_user",
                name="Test User",
                age=30,
                gender="female",
                dietary_restrictions=["vegetarian"],
                allergies=["nuts"]
            )
            
            self.assertEqual(profile.user_id, "test_user")
            self.assertEqual(profile.name, "Test User")
            self.assertTrue(profile.has_dietary_restriction("vegetarian"))
            self.assertTrue(profile.has_allergy("nuts"))
            self.assertFalse(profile.has_allergy("dairy"))
            
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
    
    def test_nutrition_calculator(self):
        """Test nutrition calculations."""
        try:
            from utils.nutrition_calc import NutritionCalculator
            
            calc = NutritionCalculator()
            
            # Test macronutrient ratios
            ratios = calc.calculate_macronutrient_ratios(400, 20, 40, 15)
            
            self.assertAlmostEqual(ratios['protein_ratio'], 20.0, places=1)  # 80 cal / 400 cal = 20%
            self.assertAlmostEqual(ratios['carb_ratio'], 40.0, places=1)    # 160 cal / 400 cal = 40%
            self.assertAlmostEqual(ratios['fat_ratio'], 33.8, places=1)     # 135 cal / 400 cal = 33.8%
            
            # Test daily value percentage
            dv_percent = calc.calculate_daily_value_percentage('sodium', 1150)
            self.assertEqual(dv_percent, 50.0)  # 1150mg is 50% of 2300mg DV
            
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
    
    def test_recipe_data_processing(self):
        """Test recipe data processing."""
        try:
            from src.data_processor import RecipeDataProcessor
            import pandas as pd
            
            processor = RecipeDataProcessor()
            
            # Test ingredient cleaning
            cleaned = processor.clean_ingredient_text("2 cups chopped fresh onions")
            self.assertIn("onion", cleaned.lower())
            self.assertNotIn("chopped", cleaned.lower())
            
            # Test nutrition extraction
            nutrition_text = "350 calories, 15g protein, 45g carbs, 12g fat"
            calories = processor.extract_nutritional_value(nutrition_text, "calories")
            protein = processor.extract_nutritional_value(nutrition_text, "protein")
            
            self.assertEqual(calories, 350.0)
            self.assertEqual(protein, 15.0)
            
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
    
    def test_substitution_engine(self):
        """Test ingredient substitution logic."""
        try:
            from src.substitution import IngredientSubstitutionEngine
            
            engine = IngredientSubstitutionEngine()
            
            # Test finding substitutions for dairy-free user
            substitutions = engine.find_substitutions(
                "milk",
                dietary_restrictions=["dairy_free"],
                allergies=[],
                health_conditions=[]
            )
            
            self.assertGreater(len(substitutions), 0)
            
            # Check that substitutions don't contain dairy
            for sub in substitutions:
                substitute_name = sub['substitute'].lower()
                self.assertNotIn('milk', substitute_name)
                self.assertNotIn('dairy', substitute_name)
            
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
    
    def test_health_logic_engine(self):
        """Test health condition logic."""
        try:
            from src.health_logic import HealthLogicEngine
            
            engine = HealthLogicEngine()
            
            # Test recipe safety check
            recipe_data = {
                'ingredients': 'chicken, peanuts, salt',
                'ingredients_list': ['chicken', 'peanuts', 'salt'],
                'sodium': 500,
                'sugar': 5
            }
            
            user_allergies = ['nuts']
            user_restrictions = []
            
            is_safe, warnings = engine.check_recipe_safety(
                recipe_data, user_allergies, user_restrictions
            )
            
            self.assertFalse(is_safe)  # Should be unsafe due to peanuts
            self.assertGreater(len(warnings), 0)
            
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
    
    def test_helpers_functions(self):
        """Test utility helper functions."""
        try:
            from utils.helpers import (
                parse_cooking_time, format_cooking_time, 
                normalize_ingredient_name, categorize_cuisine
            )
            
            # Test cooking time parsing
            time_minutes = parse_cooking_time("1 hr 30 min")
            self.assertEqual(time_minutes, 90)
            
            time_minutes = parse_cooking_time("45 minutes")
            self.assertEqual(time_minutes, 45)
            
            # Test time formatting
            formatted = format_cooking_time(90)
            self.assertIn("1 hr 30 min", formatted)
            
            # Test ingredient normalization
            normalized = normalize_ingredient_name("2 cups fresh chopped onions")
            self.assertEqual(normalized.lower(), "onions")
            
            # Test cuisine categorization
            cuisine = categorize_cuisine(
                "Spaghetti Carbonara",
                ["pasta", "parmesan", "eggs", "bacon"],
                "Cook pasta and mix with cheese"
            )
            self.assertEqual(cuisine, "italian")
            
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_sample_data_creation(self):
        """Test that sample data can be created."""
        try:
            from setup_data import DataSetupManager
            
            setup_manager = DataSetupManager()
            df = setup_manager.create_sample_recipe_data()
            
            self.assertGreater(len(df), 0)
            self.assertIn('name', df.columns)
            self.assertIn('ingredients', df.columns)
            self.assertIn('calories', df.columns)
            
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
    
    def test_config_loading(self):
        """Test that configuration loads properly."""
        try:
            import config
            
            self.assertIsNotNone(config.APP_TITLE)
            self.assertIsNotNone(config.HEALTH_CONDITIONS)
            self.assertIsNotNone(config.DIETARY_RESTRICTIONS)
            
            # Test that health conditions have required fields
            diabetes_config = config.HEALTH_CONDITIONS.get('diabetes', {})
            self.assertIn('max_sugar', diabetes_config)
            self.assertIn('avoid_ingredients', diabetes_config)
            
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
