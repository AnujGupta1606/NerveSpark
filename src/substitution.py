from typing import Dict, List, Any, Optional, Tuple
import json
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IngredientSubstitutionEngine:
    """
    Handles intelligent ingredient substitutions based on dietary restrictions,
    allergies, and health conditions while preserving recipe quality.
    """
    
    def __init__(self, substitutions_file: str = None):
        self.substitutions_db = self._load_substitutions_db(substitutions_file)
        self.measurement_conversions = self._initialize_measurement_conversions()
    
    def _load_substitutions_db(self, filepath: str = None) -> Dict[str, Any]:
        """Load ingredient substitution database."""
        default_substitutions = {
            "dairy_substitutions": {
                "milk": [
                    {"substitute": "almond milk", "ratio": "1:1", "notes": "Lower calories, nutty flavor"},
                    {"substitute": "oat milk", "ratio": "1:1", "notes": "Creamy texture, slightly sweet"},
                    {"substitute": "coconut milk", "ratio": "1:1", "notes": "Rich, tropical flavor"},
                    {"substitute": "soy milk", "ratio": "1:1", "notes": "High protein, neutral taste"}
                ],
                "butter": [
                    {"substitute": "coconut oil", "ratio": "3:4", "notes": "Solid at room temp, mild coconut flavor"},
                    {"substitute": "olive oil", "ratio": "3:4", "notes": "Liquid, fruity flavor"},
                    {"substitute": "avocado", "ratio": "1:2", "notes": "Adds fiber, creamy texture"},
                    {"substitute": "applesauce", "ratio": "1:2", "notes": "Lower fat, adds sweetness"}
                ],
                "cheese": [
                    {"substitute": "nutritional yeast", "ratio": "1:4", "notes": "Nutty, cheesy flavor"},
                    {"substitute": "cashew cream", "ratio": "1:1", "notes": "Creamy, mild flavor"},
                    {"substitute": "tofu ricotta", "ratio": "1:1", "notes": "High protein, crumbly texture"}
                ],
                "cream": [
                    {"substitute": "coconut cream", "ratio": "1:1", "notes": "Rich, slightly sweet"},
                    {"substitute": "cashew cream", "ratio": "1:1", "notes": "Neutral, very creamy"},
                    {"substitute": "silken tofu blended", "ratio": "1:1", "notes": "High protein, smooth"}
                ],
                "yogurt": [
                    {"substitute": "coconut yogurt", "ratio": "1:1", "notes": "Probiotic, creamy"},
                    {"substitute": "almond yogurt", "ratio": "1:1", "notes": "Lower calories, mild flavor"},
                    {"substitute": "cashew yogurt", "ratio": "1:1", "notes": "Rich, tangy"}
                ]
            },
            "gluten_substitutions": {
                "wheat flour": [
                    {"substitute": "almond flour", "ratio": "1:1", "notes": "Nutty, dense, high protein"},
                    {"substitute": "rice flour", "ratio": "1:1", "notes": "Light, neutral flavor"},
                    {"substitute": "oat flour", "ratio": "1:1", "notes": "Mild, slightly sweet"},
                    {"substitute": "coconut flour", "ratio": "1:4", "notes": "Very absorbent, add extra liquid"}
                ],
                "bread crumbs": [
                    {"substitute": "crushed nuts", "ratio": "1:1", "notes": "Adds healthy fats, crunchy"},
                    {"substitute": "ground flax", "ratio": "1:1", "notes": "High omega-3, nutty flavor"},
                    {"substitute": "crushed rice cakes", "ratio": "1:1", "notes": "Light, crispy texture"}
                ],
                "pasta": [
                    {"substitute": "zucchini noodles", "ratio": "1:1", "notes": "Very low carb, fresh taste"},
                    {"substitute": "shirataki noodles", "ratio": "1:1", "notes": "Almost zero calories"},
                    {"substitute": "lentil pasta", "ratio": "1:1", "notes": "High protein, earthy flavor"},
                    {"substitute": "chickpea pasta", "ratio": "1:1", "notes": "High protein and fiber"}
                ]
            },
            "meat_substitutions": {
                "ground beef": [
                    {"substitute": "lentils", "ratio": "1:1", "notes": "High protein, fiber, earthy taste"},
                    {"substitute": "mushrooms", "ratio": "1:1", "notes": "Umami flavor, meaty texture"},
                    {"substitute": "black beans", "ratio": "1:1", "notes": "High protein, mild flavor"},
                    {"substitute": "tempeh", "ratio": "1:1", "notes": "Fermented, nutty, high protein"}
                ],
                "chicken": [
                    {"substitute": "tofu", "ratio": "1:1", "notes": "Absorbs flavors well, high protein"},
                    {"substitute": "cauliflower", "ratio": "1:1", "notes": "Mild flavor, versatile texture"},
                    {"substitute": "jackfruit", "ratio": "1:1", "notes": "Stringy texture, mild taste"},
                    {"substitute": "seitan", "ratio": "1:1", "notes": "Very high protein, chewy texture"}
                ],
                "bacon": [
                    {"substitute": "smoky tempeh", "ratio": "1:1", "notes": "Smoky flavor, less fat"},
                    {"substitute": "coconut bacon", "ratio": "1:1", "notes": "Crispy, smoky seasoning"},
                    {"substitute": "mushroom bacon", "ratio": "1:1", "notes": "Umami, crispy when cooked"}
                ]
            },
            "sugar_substitutions": {
                "white sugar": [
                    {"substitute": "stevia", "ratio": "1:24", "notes": "Zero calories, very sweet"},
                    {"substitute": "monk fruit", "ratio": "1:3", "notes": "Zero calories, no aftertaste"},
                    {"substitute": "erythritol", "ratio": "1:1", "notes": "Low calories, cooling effect"},
                    {"substitute": "dates", "ratio": "1:1", "notes": "Natural, adds fiber, rich flavor"}
                ],
                "brown sugar": [
                    {"substitute": "coconut sugar", "ratio": "1:1", "notes": "Lower glycemic, caramel notes"},
                    {"substitute": "maple syrup", "ratio": "3:4", "notes": "Natural, reduce other liquids"},
                    {"substitute": "honey", "ratio": "3:4", "notes": "Natural, floral notes, reduce liquids"}
                ]
            },
            "egg_substitutions": {
                "eggs": [
                    {"substitute": "flax egg", "ratio": "1:1", "notes": "1 tbsp ground flax + 3 tbsp water"},
                    {"substitute": "chia egg", "ratio": "1:1", "notes": "1 tbsp chia + 3 tbsp water"},
                    {"substitute": "applesauce", "ratio": "1:4", "notes": "1/4 cup per egg, adds moisture"},
                    {"substitute": "banana", "ratio": "1:2", "notes": "1/2 banana per egg, adds sweetness"},
                    {"substitute": "aquafaba", "ratio": "1:3", "notes": "3 tbsp chickpea liquid per egg"}
                ]
            },
            "oil_substitutions": {
                "vegetable oil": [
                    {"substitute": "avocado oil", "ratio": "1:1", "notes": "High smoke point, neutral flavor"},
                    {"substitute": "coconut oil", "ratio": "1:1", "notes": "Solid at room temp, mild coconut"},
                    {"substitute": "applesauce", "ratio": "1:2", "notes": "Lower fat, adds moisture"},
                    {"substitute": "mashed banana", "ratio": "1:2", "notes": "Adds sweetness, very moist"}
                ]
            }
        }
        
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    loaded_subs = json.load(f)
                    # Merge with defaults
                    for category, subs in loaded_subs.items():
                        if category in default_substitutions:
                            default_substitutions[category].update(subs)
                        else:
                            default_substitutions[category] = subs
            except Exception as e:
                logger.warning(f"Could not load substitutions file: {e}. Using defaults.")
        
        return default_substitutions
    
    def _initialize_measurement_conversions(self) -> Dict[str, float]:
        """Initialize measurement conversion ratios."""
        return {
            # Volume conversions (to cups)
            'tbsp': 1/16, 'tablespoon': 1/16, 'tsp': 1/48, 'teaspoon': 1/48,
            'ml': 1/240, 'l': 1000/240, 'fl oz': 1/8, 'pint': 2, 'quart': 4,
            # Weight conversions (to grams)
            'oz': 28.35, 'lb': 453.59, 'kg': 1000, 'g': 1
        }
    
    def find_substitutions(self, ingredient: str, 
                          dietary_restrictions: List[str] = None,
                          allergies: List[str] = None,
                          health_conditions: List[str] = None) -> List[Dict[str, Any]]:
        """
        Find suitable substitutions for an ingredient based on user restrictions.
        
        Args:
            ingredient: The ingredient to substitute
            dietary_restrictions: List of dietary restrictions
            allergies: List of allergies
            health_conditions: List of health conditions
            
        Returns:
            List of substitution options with ratios and notes
        """
        ingredient_lower = ingredient.lower().strip()
        substitutions = []
        
        # Check each substitution category
        for category, substitution_dict in self.substitutions_db.items():
            for base_ingredient, sub_options in substitution_dict.items():
                if base_ingredient.lower() in ingredient_lower or ingredient_lower in base_ingredient.lower():
                    # Filter substitutions based on restrictions
                    filtered_subs = self._filter_substitutions_by_restrictions(
                        sub_options, dietary_restrictions, allergies, health_conditions
                    )
                    if filtered_subs:
                        substitutions.extend([{
                            'original_ingredient': ingredient,
                            'category': category,
                            **sub
                        } for sub in filtered_subs])
        
        # Remove duplicates and sort by preference
        substitutions = self._deduplicate_and_rank_substitutions(substitutions)
        
        return substitutions
    
    def _filter_substitutions_by_restrictions(self, substitutions: List[Dict[str, Any]],
                                            dietary_restrictions: List[str] = None,
                                            allergies: List[str] = None,
                                            health_conditions: List[str] = None) -> List[Dict[str, Any]]:
        """Filter substitutions based on user restrictions."""
        if not (dietary_restrictions or allergies or health_conditions):
            return substitutions
        
        filtered = []
        
        for sub in substitutions:
            substitute_name = sub['substitute'].lower()
            is_suitable = True
            
            # Check allergies
            if allergies:
                for allergy in allergies:
                    allergy_ingredients = self._get_allergy_ingredients(allergy)
                    if any(allergen in substitute_name for allergen in allergy_ingredients):
                        is_suitable = False
                        break
            
            # Check dietary restrictions
            if dietary_restrictions and is_suitable:
                for restriction in dietary_restrictions:
                    restricted_ingredients = self._get_restricted_ingredients(restriction)
                    if any(restricted in substitute_name for restricted in restricted_ingredients):
                        is_suitable = False
                        break
            
            # Check health condition compatibility
            if health_conditions and is_suitable:
                if not self._is_suitable_for_health_conditions(substitute_name, health_conditions):
                    is_suitable = False
            
            if is_suitable:
                filtered.append(sub)
        
        return filtered
    
    def _get_allergy_ingredients(self, allergy: str) -> List[str]:
        """Get list of ingredients to avoid for a specific allergy."""
        allergy_map = {
            'nuts': ['almond', 'peanut', 'walnut', 'cashew', 'pecan', 'hazelnut', 'pistachio'],
            'dairy': ['milk', 'cheese', 'butter', 'cream', 'yogurt', 'whey', 'casein'],
            'eggs': ['egg', 'albumin'],
            'soy': ['soy', 'tofu', 'tempeh', 'miso'],
            'shellfish': ['shrimp', 'crab', 'lobster'],
            'fish': ['salmon', 'tuna', 'cod', 'fish']
        }
        return allergy_map.get(allergy.lower(), [])
    
    def _get_restricted_ingredients(self, restriction: str) -> List[str]:
        """Get list of ingredients to avoid for a dietary restriction."""
        restriction_map = {
            'vegan': ['meat', 'chicken', 'beef', 'pork', 'fish', 'milk', 'cheese', 'butter', 'egg', 'honey'],
            'vegetarian': ['meat', 'chicken', 'beef', 'pork', 'fish', 'bacon'],
            'gluten_free': ['wheat', 'barley', 'rye', 'flour', 'bread'],
            'dairy_free': ['milk', 'cheese', 'butter', 'cream', 'yogurt'],
            'keto': ['sugar', 'honey', 'maple syrup', 'flour', 'rice', 'potato'],
            'low_carb': ['sugar', 'flour', 'rice', 'pasta', 'bread']
        }
        return restriction_map.get(restriction.lower(), [])
    
    def _is_suitable_for_health_conditions(self, ingredient: str, conditions: List[str]) -> bool:
        """Check if an ingredient is suitable for specific health conditions."""
        for condition in conditions:
            if condition.lower() == 'diabetes':
                # Avoid high-sugar substitutes
                if any(sweet in ingredient for sweet in ['honey', 'maple syrup', 'agave', 'sugar']):
                    return False
            elif condition.lower() == 'hypertension':
                # Avoid high-sodium substitutes
                if any(salty in ingredient for salty in ['salt', 'soy sauce', 'miso']):
                    return False
            elif condition.lower() == 'heart_disease':
                # Prefer lower saturated fat options
                if any(fat in ingredient for fat in ['coconut oil', 'palm oil']):
                    return False
        return True
    
    def _deduplicate_and_rank_substitutions(self, substitutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicates and rank substitutions by preference."""
        # Remove duplicates based on substitute name
        seen = set()
        unique_subs = []
        
        for sub in substitutions:
            substitute_name = sub['substitute'].lower()
            if substitute_name not in seen:
                seen.add(substitute_name)
                unique_subs.append(sub)
        
        # Rank by healthiness and commonality
        health_scores = {
            'almond': 0.9, 'oat': 0.85, 'coconut': 0.8, 'avocado': 0.95,
            'lentils': 0.9, 'beans': 0.85, 'tofu': 0.8, 'tempeh': 0.85,
            'stevia': 0.9, 'monk fruit': 0.85, 'dates': 0.8,
            'flax': 0.9, 'chia': 0.85, 'applesauce': 0.7
        }
        
        for sub in unique_subs:
            substitute_name = sub['substitute'].lower()
            sub['health_score'] = max([score for ingredient, score in health_scores.items() 
                                     if ingredient in substitute_name] + [0.5])
        
        # Sort by health score (descending)
        unique_subs.sort(key=lambda x: x.get('health_score', 0.5), reverse=True)
        
        return unique_subs
    
    def apply_substitutions_to_recipe(self, recipe_data: Dict[str, Any],
                                    user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply substitutions to a recipe based on user profile.
        
        Args:
            recipe_data: Original recipe data
            user_profile: User's dietary restrictions, allergies, etc.
            
        Returns:
            Modified recipe with substitutions applied
        """
        dietary_restrictions = user_profile.get('dietary_restrictions', [])
        allergies = user_profile.get('allergies', [])
        health_conditions = user_profile.get('health_conditions', [])
        
        modified_recipe = recipe_data.copy()
        substitutions_made = []
        ingredients_list = recipe_data.get('ingredients_list', [])
        
        if not ingredients_list:
            # Try to parse ingredients from string
            ingredients_text = recipe_data.get('ingredients', '')
            ingredients_list = self._parse_ingredients_string(ingredients_text)
        
        modified_ingredients = []
        
        for ingredient in ingredients_list:
            # Find substitutions for this ingredient
            possible_subs = self.find_substitutions(
                ingredient, dietary_restrictions, allergies, health_conditions
            )
            
            if possible_subs:
                # Use the best substitution
                best_sub = possible_subs[0]
                
                # Apply the substitution
                modified_ingredient = self._apply_single_substitution(ingredient, best_sub)
                modified_ingredients.append(modified_ingredient)
                
                substitutions_made.append({
                    'original': ingredient,
                    'substitute': best_sub['substitute'],
                    'ratio': best_sub['ratio'],
                    'notes': best_sub['notes']
                })
            else:
                modified_ingredients.append(ingredient)
        
        # Update recipe with modified ingredients
        modified_recipe['ingredients_list'] = modified_ingredients
        modified_recipe['ingredients'] = '\n'.join(modified_ingredients)
        modified_recipe['substitutions_made'] = substitutions_made
        
        # Recalculate nutrition if significant substitutions were made
        if substitutions_made:
            modified_recipe = self._estimate_nutritional_changes(modified_recipe, substitutions_made)
        
        return modified_recipe
    
    def _parse_ingredients_string(self, ingredients_text: str) -> List[str]:
        """Parse ingredients string into a list."""
        if not ingredients_text:
            return []
        
        # Split by common separators
        separators = ['\n', ';', ',']
        ingredients = [ingredients_text]
        
        for sep in separators:
            if sep in ingredients_text:
                ingredients = ingredients_text.split(sep)
                break
        
        # Clean and filter ingredients
        cleaned_ingredients = []
        for ingredient in ingredients:
            ingredient = ingredient.strip()
            if ingredient and len(ingredient) > 2:
                cleaned_ingredients.append(ingredient)
        
        return cleaned_ingredients
    
    def _apply_single_substitution(self, original_ingredient: str, substitution: Dict[str, Any]) -> str:
        """Apply a single substitution to an ingredient."""
        substitute_name = substitution['substitute']
        ratio = substitution['ratio']
        
        # Try to extract quantity from original ingredient
        quantity_match = re.search(r'(\d+\.?\d*)\s*(\w+)?', original_ingredient)
        
        if quantity_match:
            original_quantity = float(quantity_match.group(1))
            unit = quantity_match.group(2) if quantity_match.group(2) else ''
            
            # Calculate new quantity based on ratio
            try:
                ratio_parts = ratio.split(':')
                if len(ratio_parts) == 2:
                    ratio_value = float(ratio_parts[1]) / float(ratio_parts[0])
                    new_quantity = original_quantity * ratio_value
                    
                    # Format new ingredient
                    new_ingredient = f"{new_quantity:.1f} {unit} {substitute_name}".strip()
                    return new_ingredient
            except:
                pass
        
        # If quantity extraction fails, just replace the ingredient name
        return re.sub(r'\b\w+\b', substitute_name, original_ingredient, count=1)
    
    def _estimate_nutritional_changes(self, recipe_data: Dict[str, Any], 
                                    substitutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate nutritional changes based on substitutions."""
        # This is a simplified estimation - could be enhanced with a nutrition database
        
        # General nutritional impact of common substitutions
        nutrition_impacts = {
            'almond milk': {'calories': -0.4, 'fat': -0.6, 'protein': -0.3, 'carbs': 0.1},
            'coconut oil': {'calories': 0, 'fat': 0, 'carbs': 0, 'protein': 0},
            'stevia': {'calories': -0.95, 'carbs': -0.95, 'sugar': -0.95},
            'lentils': {'calories': -0.2, 'protein': 1.5, 'fiber': 2.0, 'fat': -0.8},
            'tofu': {'calories': -0.3, 'protein': 0.8, 'fat': -0.4},
            'applesauce': {'calories': -0.6, 'fat': -0.9, 'sugar': 0.2}
        }
        
        modified_recipe = recipe_data.copy()
        
        for sub in substitutions:
            substitute_name = sub['substitute'].lower()
            
            # Find matching nutrition impact
            for ingredient, impacts in nutrition_impacts.items():
                if ingredient in substitute_name:
                    # Apply nutritional changes
                    for nutrient, impact in impacts.items():
                        current_value = modified_recipe.get(nutrient, 0)
                        if impact > 1:  # Multiplier for increases
                            modified_recipe[nutrient] = current_value * impact
                        else:  # Percentage change for decreases
                            modified_recipe[nutrient] = max(0, current_value * (1 + impact))
                    break
        
        return modified_recipe
    
    def get_substitution_suggestions(self, recipe_data: Dict[str, Any],
                                   user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get substitution suggestions without applying them to the recipe.
        
        Returns:
            Dictionary with suggested substitutions and their benefits
        """
        dietary_restrictions = user_profile.get('dietary_restrictions', [])
        allergies = user_profile.get('allergies', [])
        health_conditions = user_profile.get('health_conditions', [])
        
        ingredients_list = recipe_data.get('ingredients_list', [])
        if not ingredients_list:
            ingredients_text = recipe_data.get('ingredients', '')
            ingredients_list = self._parse_ingredients_string(ingredients_text)
        
        suggestions = {
            'substitutions': [],
            'benefits': [],
            'nutritional_impact': 'neutral'
        }
        
        for ingredient in ingredients_list:
            possible_subs = self.find_substitutions(
                ingredient, dietary_restrictions, allergies, health_conditions
            )
            
            if possible_subs:
                # Get top 3 substitutions
                top_subs = possible_subs[:3]
                suggestion = {
                    'original_ingredient': ingredient,
                    'alternatives': top_subs,
                    'reason': self._get_substitution_reason(dietary_restrictions, allergies, health_conditions)
                }
                suggestions['substitutions'].append(suggestion)
        
        # Calculate overall benefits
        if suggestions['substitutions']:
            suggestions['benefits'] = self._calculate_substitution_benefits(suggestions['substitutions'])
            suggestions['nutritional_impact'] = self._assess_overall_nutritional_impact(suggestions['substitutions'])
        
        return suggestions
    
    def _get_substitution_reason(self, dietary_restrictions: List[str], 
                               allergies: List[str], 
                               health_conditions: List[str]) -> str:
        """Generate reason for substitution based on user profile."""
        reasons = []
        
        if allergies:
            reasons.append(f"allergy-friendly ({', '.join(allergies)})")
        if dietary_restrictions:
            reasons.append(f"dietary preference ({', '.join(dietary_restrictions)})")
        if health_conditions:
            reasons.append(f"health condition ({', '.join(health_conditions)})")
        
        return '; '.join(reasons) if reasons else "healthier alternative"
    
    def _calculate_substitution_benefits(self, substitutions: List[Dict[str, Any]]) -> List[str]:
        """Calculate potential benefits of substitutions."""
        benefits = set()
        
        for sub_group in substitutions:
            for alternative in sub_group['alternatives']:
                notes = alternative.get('notes', '').lower()
                
                if 'lower calories' in notes or 'low calorie' in notes:
                    benefits.add("Reduced calorie content")
                if 'high protein' in notes or 'protein' in notes:
                    benefits.add("Increased protein content")
                if 'fiber' in notes:
                    benefits.add("Added fiber")
                if 'omega' in notes:
                    benefits.add("Heart-healthy fats")
                if 'probiotic' in notes:
                    benefits.add("Digestive health support")
        
        return list(benefits)
    
    def _assess_overall_nutritional_impact(self, substitutions: List[Dict[str, Any]]) -> str:
        """Assess overall nutritional impact of substitutions."""
        positive_indicators = ['high protein', 'fiber', 'lower calories', 'omega', 'probiotic']
        negative_indicators = ['higher calories', 'more fat']
        
        positive_count = 0
        negative_count = 0
        
        for sub_group in substitutions:
            for alternative in sub_group['alternatives']:
                notes = alternative.get('notes', '').lower()
                
                for indicator in positive_indicators:
                    if indicator in notes:
                        positive_count += 1
                
                for indicator in negative_indicators:
                    if indicator in notes:
                        negative_count += 1
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
