from typing import List, Dict, Any, Optional, Tuple
import logging
from config import HEALTH_CONDITIONS, DIETARY_RESTRICTIONS, ALLERGIES, DAILY_NUTRITION_GOALS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthLogicEngine:
    """
    Manages health condition logic, dietary restrictions, and nutritional recommendations.
    Provides personalized filtering and scoring for recipes based on user health profile.
    """
    
    def __init__(self):
        self.health_conditions = HEALTH_CONDITIONS
        self.dietary_restrictions = DIETARY_RESTRICTIONS
        self.allergies = ALLERGIES
        self.nutrition_goals = DAILY_NUTRITION_GOALS
    
    def check_recipe_safety(self, recipe_data: Dict[str, Any], 
                           user_allergies: List[str],
                           user_restrictions: List[str]) -> Tuple[bool, List[str]]:
        """
        Check if a recipe is safe for a user given their allergies and restrictions.
        
        Args:
            recipe_data: Recipe information including ingredients
            user_allergies: List of user's allergies
            user_restrictions: List of dietary restrictions
            
        Returns:
            Tuple of (is_safe, list_of_warnings)
        """
        warnings = []
        is_safe = True
        
        ingredients_text = str(recipe_data.get('ingredients', '')).lower()
        ingredients_list = recipe_data.get('ingredients_list', [])
        
        # Check allergies
        for allergy in user_allergies:
            if allergy in self.allergies:
                allergen_ingredients = self.allergies[allergy]
                for allergen in allergen_ingredients:
                    if allergen.lower() in ingredients_text:
                        warnings.append(f"⚠️ Contains {allergen} (allergy: {allergy})")
                        is_safe = False
        
        # Check dietary restrictions
        for restriction in user_restrictions:
            if restriction in self.dietary_restrictions:
                restricted_ingredients = self.dietary_restrictions[restriction].get('avoid_ingredients', [])
                
                for restricted_ingredient in restricted_ingredients:
                    if restricted_ingredient.lower() in ingredients_text:
                        warnings.append(f"🚫 Contains {restricted_ingredient} (restriction: {restriction})")
                        is_safe = False
                
                # Check nutritional limits for restrictions like low_carb, keto
                if restriction == 'low_carb':
                    max_carbs = self.dietary_restrictions[restriction].get('max_carbs_per_serving', 20)
                    recipe_carbs = recipe_data.get('carbs', 0)
                    if recipe_carbs > max_carbs:
                        warnings.append(f"🚫 Too high in carbs ({recipe_carbs}g > {max_carbs}g limit)")
                        is_safe = False
                
                elif restriction == 'keto':
                    max_carbs = self.dietary_restrictions[restriction].get('max_carbs_per_serving', 5)
                    min_fat_percent = self.dietary_restrictions[restriction].get('min_fat_percentage', 70)
                    
                    recipe_carbs = recipe_data.get('carbs', 0)
                    recipe_calories = recipe_data.get('calories', 1)  # Avoid division by zero
                    recipe_fat = recipe_data.get('fat', 0)
                    fat_percentage = (recipe_fat * 9 / recipe_calories) * 100 if recipe_calories > 0 else 0
                    
                    if recipe_carbs > max_carbs:
                        warnings.append(f"🚫 Too high in carbs for keto ({recipe_carbs}g > {max_carbs}g)")
                        is_safe = False
                    
                    if fat_percentage < min_fat_percent:
                        warnings.append(f"⚠️ Low fat percentage for keto ({fat_percentage:.1f}% < {min_fat_percent}%)")
        
        return is_safe, warnings
    
    def score_recipe_for_health_conditions(self, recipe_data: Dict[str, Any], 
                                         health_conditions: List[str]) -> Tuple[float, List[str]]:
        """
        Score a recipe's suitability for specific health conditions.
        
        Args:
            recipe_data: Recipe nutritional information
            health_conditions: List of user's health conditions
            
        Returns:
            Tuple of (health_score, list_of_recommendations)
        """
        health_score = 1.0  # Start with perfect score
        recommendations = []
        
        for condition in health_conditions:
            if condition not in self.health_conditions:
                continue
            
            condition_guidelines = self.health_conditions[condition]
            condition_score = 1.0
            
            # Check condition-specific nutritional guidelines
            if condition == 'diabetes':
                # Check sugar content
                recipe_sugar = recipe_data.get('sugar', 0)
                max_sugar = condition_guidelines.get('max_sugar', 25)
                
                if recipe_sugar > max_sugar:
                    penalty = min(0.5, (recipe_sugar - max_sugar) / max_sugar)
                    condition_score -= penalty
                    recommendations.append(f"🍯 High sugar content ({recipe_sugar}g). Consider reducing sweeteners.")
                
                # Check carbs
                recipe_carbs = recipe_data.get('carbs', 0)
                max_carbs = condition_guidelines.get('max_carbs', 45)
                
                if recipe_carbs > max_carbs:
                    penalty = min(0.3, (recipe_carbs - max_carbs) / max_carbs)
                    condition_score -= penalty
                    recommendations.append(f"🍞 High carb content ({recipe_carbs}g). Consider portion control.")
                
                # Bonus for high fiber
                recipe_fiber = recipe_data.get('fiber', 0)
                if recipe_fiber > 5:
                    condition_score += 0.1
                    recommendations.append(f"✅ Good fiber content ({recipe_fiber}g) - helps with blood sugar control.")
            
            elif condition == 'hypertension':
                # Check sodium content
                recipe_sodium = recipe_data.get('sodium', 0)
                max_sodium = condition_guidelines.get('max_sodium', 1500)
                
                if recipe_sodium > max_sodium / 3:  # Per meal limit
                    penalty = min(0.6, (recipe_sodium - max_sodium/3) / (max_sodium/3))
                    condition_score -= penalty
                    recommendations.append(f"🧂 High sodium content ({recipe_sodium}mg). Consider reducing salt.")
                
                # Bonus for potassium
                recipe_potassium = recipe_data.get('potassium', 0)
                if recipe_potassium > 400:  # Good source
                    condition_score += 0.1
                    recommendations.append(f"✅ Good potassium source ({recipe_potassium}mg) - helps lower blood pressure.")
            
            elif condition == 'heart_disease':
                # Check saturated fat
                recipe_sat_fat = recipe_data.get('saturated_fat', recipe_data.get('fat', 0) * 0.3)  # Estimate if not available
                max_sat_fat = condition_guidelines.get('max_saturated_fat', 13) / 3  # Per meal
                
                if recipe_sat_fat > max_sat_fat:
                    penalty = min(0.5, (recipe_sat_fat - max_sat_fat) / max_sat_fat)
                    condition_score -= penalty
                    recommendations.append(f"🥩 High saturated fat ({recipe_sat_fat:.1f}g). Consider leaner options.")
                
                # Check cholesterol
                recipe_cholesterol = recipe_data.get('cholesterol', 0)
                max_cholesterol = condition_guidelines.get('max_cholesterol', 200) / 3
                
                if recipe_cholesterol > max_cholesterol:
                    penalty = min(0.3, (recipe_cholesterol - max_cholesterol) / max_cholesterol)
                    condition_score -= penalty
                    recommendations.append(f"🥚 High cholesterol ({recipe_cholesterol}mg). Consider alternatives.")
                
                # Bonus for omega-3 rich ingredients
                ingredients_text = str(recipe_data.get('ingredients', '')).lower()
                omega3_sources = ['salmon', 'mackerel', 'sardines', 'walnuts', 'chia seeds', 'flax']
                if any(source in ingredients_text for source in omega3_sources):
                    condition_score += 0.15
                    recommendations.append("✅ Contains omega-3 rich ingredients - good for heart health.")
            
            elif condition == 'kidney_disease':
                # Check protein content
                recipe_protein = recipe_data.get('protein', 0)
                max_protein = condition_guidelines.get('max_protein', 50) / 3  # Per meal
                
                if recipe_protein > max_protein:
                    penalty = min(0.4, (recipe_protein - max_protein) / max_protein)
                    condition_score -= penalty
                    recommendations.append(f"🥩 High protein content ({recipe_protein}g). Consider reducing portion size.")
                
                # Check phosphorus and potassium (if available)
                recipe_phosphorus = recipe_data.get('phosphorus', 0)
                recipe_potassium = recipe_data.get('potassium', 0)
                
                if recipe_phosphorus > 200:  # Per meal estimate
                    condition_score -= 0.2
                    recommendations.append(f"⚠️ May be high in phosphorus. Consult with healthcare provider.")
                
                if recipe_potassium > 600:  # Per meal estimate
                    condition_score -= 0.2
                    recommendations.append(f"⚠️ May be high in potassium. Monitor intake carefully.")
            
            # Update overall health score
            health_score *= max(0.1, condition_score)  # Ensure score doesn't go below 0.1
        
        return health_score, recommendations
    
    def calculate_nutrition_goal_match(self, recipe_data: Dict[str, Any], 
                                     user_goals: Dict[str, float],
                                     gender: str = "female") -> Tuple[float, List[str]]:
        """
        Calculate how well a recipe matches user's nutritional goals.
        
        Args:
            recipe_data: Recipe nutritional information
            user_goals: User's daily nutritional goals
            gender: User's gender for default goals
            
        Returns:
            Tuple of (goal_match_score, list_of_insights)
        """
        insights = []
        goal_scores = []
        
        # Get default goals if not provided
        default_goals = self.nutrition_goals
        
        # Check each nutritional component
        nutrition_components = ['calories', 'protein', 'carbs', 'fat', 'fiber']
        
        for component in nutrition_components:
            recipe_value = recipe_data.get(component, 0)
            
            if component in user_goals:
                target = user_goals[component]
            elif component in default_goals and gender in default_goals[component]:
                target = default_goals[component][gender]
            elif component in default_goals:
                target = default_goals[component].get('max', default_goals[component].get('female', 0))
            else:
                continue
            
            # Calculate what percentage of daily goal this recipe provides
            daily_percentage = (recipe_value / target) * 100 if target > 0 else 0
            
            # Score based on reasonable meal portions (aim for 25-35% of daily intake per meal)
            ideal_percentage = 30  # 30% of daily intake per meal
            
            if 20 <= daily_percentage <= 40:
                score = 1.0  # Perfect range
            elif 15 <= daily_percentage < 20 or 40 < daily_percentage <= 50:
                score = 0.8  # Good range
            elif 10 <= daily_percentage < 15 or 50 < daily_percentage <= 60:
                score = 0.6  # Acceptable range
            else:
                score = 0.3  # Outside ideal range
            
            goal_scores.append(score)
            
            # Generate insights
            if daily_percentage > 50:
                insights.append(f"⚠️ High {component} content ({daily_percentage:.1f}% of daily goal)")
            elif daily_percentage < 15:
                insights.append(f"ℹ️ Low {component} content ({daily_percentage:.1f}% of daily goal)")
            else:
                insights.append(f"✅ Good {component} balance ({daily_percentage:.1f}% of daily goal)")
        
        # Calculate overall goal match score
        overall_score = sum(goal_scores) / len(goal_scores) if goal_scores else 0.5
        
        return overall_score, insights
    
    def get_recipe_recommendations(self, recipe_data: Dict[str, Any],
                                 user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive recommendations for a recipe based on user profile.
        
        Args:
            recipe_data: Recipe information
            user_profile: User's health profile, allergies, restrictions, goals
            
        Returns:
            Dictionary with safety, health score, recommendations, and insights
        """
        user_allergies = user_profile.get('allergies', [])
        user_restrictions = user_profile.get('dietary_restrictions', [])
        user_conditions = user_profile.get('health_conditions', [])
        user_goals = user_profile.get('nutrition_goals', {})
        user_gender = user_profile.get('gender', 'female')
        
        # Check safety
        is_safe, safety_warnings = self.check_recipe_safety(
            recipe_data, user_allergies, user_restrictions
        )
        
        # Score for health conditions
        health_score, health_recommendations = self.score_recipe_for_health_conditions(
            recipe_data, user_conditions
        )
        
        # Check nutrition goal match
        goal_score, goal_insights = self.calculate_nutrition_goal_match(
            recipe_data, user_goals, user_gender
        )
        
        # Calculate overall recommendation score
        if not is_safe:
            overall_score = 0.0  # Unsafe recipes get zero score
        else:
            overall_score = (health_score * 0.6 + goal_score * 0.4)
        
        return {
            'is_safe': is_safe,
            'overall_score': overall_score,
            'health_score': health_score,
            'goal_match_score': goal_score,
            'safety_warnings': safety_warnings,
            'health_recommendations': health_recommendations,
            'nutrition_insights': goal_insights,
            'recommendation_level': self._get_recommendation_level(overall_score)
        }
    
    def _get_recommendation_level(self, score: float) -> str:
        """Get recommendation level based on score."""
        if score >= 0.8:
            return "Highly Recommended 🌟"
        elif score >= 0.6:
            return "Recommended ✅"
        elif score >= 0.4:
            return "Acceptable with Caution ⚠️"
        elif score > 0:
            return "Not Recommended ❌"
        else:
            return "Unsafe - Avoid 🚫"
    
    def suggest_meal_combinations(self, recipes: List[Dict[str, Any]], 
                                user_profile: Dict[str, Any],
                                target_calories: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Suggest combinations of recipes to meet daily nutritional goals.
        
        Args:
            recipes: List of available recipes
            user_profile: User's profile and goals
            target_calories: Target calories for the meal combination
            
        Returns:
            List of meal combination suggestions
        """
        combinations = []
        
        # This is a simplified version - could be expanded with more sophisticated algorithms
        safe_recipes = []
        for recipe in recipes:
            recommendations = self.get_recipe_recommendations(recipe, user_profile)
            if recommendations['is_safe'] and recommendations['overall_score'] > 0.5:
                safe_recipes.append((recipe, recommendations))
        
        # Sort by overall score
        safe_recipes.sort(key=lambda x: x[1]['overall_score'], reverse=True)
        
        # Take top recipes for combination
        if len(safe_recipes) >= 3:
            combination = {
                'recipes': [recipe[0] for recipe in safe_recipes[:3]],
                'total_calories': sum(recipe[0].get('calories', 0) for recipe in safe_recipes[:3]),
                'combined_score': sum(rec[1]['overall_score'] for rec in safe_recipes[:3]) / 3,
                'meal_type': 'Balanced Daily Meal Plan'
            }
            combinations.append(combination)
        
        return combinations
