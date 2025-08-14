from typing import Dict, List, Any, Optional, Tuple
import math

class NutritionCalculator:
    """
    Utility class for nutritional calculations and conversions.
    """
    
    # Calories per gram for macronutrients
    CALORIES_PER_GRAM = {
        'protein': 4,
        'carbs': 4,
        'fat': 9,
        'alcohol': 7
    }
    
    # Daily Value (DV) reference values for adults
    DAILY_VALUES = {
        'calories': 2000,
        'fat': 65,  # grams
        'saturated_fat': 20,  # grams
        'cholesterol': 300,  # mg
        'sodium': 2300,  # mg
        'carbs': 300,  # grams
        'fiber': 25,  # grams
        'sugar': 50,  # grams
        'protein': 50,  # grams
        'vitamin_a': 900,  # mcg
        'vitamin_c': 90,  # mg
        'calcium': 1000,  # mg
        'iron': 18,  # mg
        'potassium': 3500  # mg
    }
    
    @staticmethod
    def calculate_macronutrient_ratios(calories: float, protein: float, 
                                     carbs: float, fat: float) -> Dict[str, float]:
        """Calculate macronutrient ratios as percentages of total calories."""
        if calories <= 0:
            return {'protein_ratio': 0, 'carb_ratio': 0, 'fat_ratio': 0}
        
        protein_calories = protein * NutritionCalculator.CALORIES_PER_GRAM['protein']
        carb_calories = carbs * NutritionCalculator.CALORIES_PER_GRAM['carbs']
        fat_calories = fat * NutritionCalculator.CALORIES_PER_GRAM['fat']
        
        return {
            'protein_ratio': round((protein_calories / calories) * 100, 1),
            'carb_ratio': round((carb_calories / calories) * 100, 1),
            'fat_ratio': round((fat_calories / calories) * 100, 1)
        }
    
    @staticmethod
    def calculate_daily_value_percentage(nutrient: str, amount: float) -> float:
        """Calculate percentage of Daily Value for a nutrient."""
        if nutrient not in NutritionCalculator.DAILY_VALUES:
            return 0.0
        
        daily_value = NutritionCalculator.DAILY_VALUES[nutrient]
        return round((amount / daily_value) * 100, 1)
    
    @staticmethod
    def estimate_glycemic_load(carbs: float, glycemic_index: int = 55) -> float:
        """
        Estimate glycemic load.
        GL = (GI × carbs) / 100
        """
        return round((glycemic_index * carbs) / 100, 1)
    
    @staticmethod
    def calculate_net_carbs(total_carbs: float, fiber: float, sugar_alcohols: float = 0) -> float:
        """Calculate net carbs (total carbs - fiber - sugar alcohols)."""
        return max(0, total_carbs - fiber - sugar_alcohols)
    
    @staticmethod
    def calculate_protein_percentage_of_calories(protein: float, calories: float) -> float:
        """Calculate what percentage of calories come from protein."""
        if calories <= 0:
            return 0.0
        
        protein_calories = protein * NutritionCalculator.CALORIES_PER_GRAM['protein']
        return round((protein_calories / calories) * 100, 1)
    
    @staticmethod
    def estimate_satiety_score(protein: float, fiber: float, fat: float, 
                             calories: float, volume_factor: float = 1.0) -> float:
        """
        Estimate satiety score based on macronutrients.
        Higher scores indicate more filling foods.
        Score range: 0-100
        """
        if calories <= 0:
            return 0.0
        
        # Protein and fiber contribute most to satiety
        protein_score = min(30, (protein / calories) * 1000 * 3)  # Protein factor
        fiber_score = min(25, fiber * 5)  # Fiber factor
        fat_score = min(20, (fat / calories) * 1000 * 2)  # Fat factor (moderate)
        volume_score = min(25, volume_factor * 25)  # Volume/water content
        
        total_score = protein_score + fiber_score + fat_score + volume_score
        return round(min(100, total_score), 1)
    
    @staticmethod
    def calculate_nutrient_density(nutrients: Dict[str, float], calories: float) -> Dict[str, float]:
        """
        Calculate nutrient density (nutrients per 100 calories).
        Higher values indicate more nutrient-dense foods.
        """
        if calories <= 0:
            return {nutrient: 0 for nutrient in nutrients}
        
        density_scores = {}
        for nutrient, amount in nutrients.items():
            # Normalize to per 100 calories
            density_scores[nutrient] = round((amount / calories) * 100, 2)
        
        return density_scores
    
    @staticmethod
    def assess_dietary_quality(nutrition_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Assess overall dietary quality based on nutritional composition.
        Returns scores and recommendations.
        """
        calories = nutrition_data.get('calories', 0)
        protein = nutrition_data.get('protein', 0)
        carbs = nutrition_data.get('carbs', 0)
        fat = nutrition_data.get('fat', 0)
        fiber = nutrition_data.get('fiber', 0)
        sodium = nutrition_data.get('sodium', 0)
        sugar = nutrition_data.get('sugar', 0)
        
        if calories <= 0:
            return {'overall_score': 0, 'category': 'Unknown', 'recommendations': []}
        
        scores = {}
        recommendations = []
        
        # Macronutrient balance score (0-25 points)
        ratios = NutritionCalculator.calculate_macronutrient_ratios(calories, protein, carbs, fat)
        
        # Ideal ranges: Protein 10-35%, Carbs 45-65%, Fat 20-35%
        protein_score = 25
        if ratios['protein_ratio'] < 10:
            protein_score = ratios['protein_ratio'] * 2.5
            recommendations.append("Consider adding more protein-rich foods")
        elif ratios['protein_ratio'] > 35:
            protein_score = 25 - (ratios['protein_ratio'] - 35)
            recommendations.append("Consider moderating protein intake")
        
        carb_score = 25
        if ratios['carb_ratio'] < 45:
            carb_score = ratios['carb_ratio'] * 0.56
            recommendations.append("Consider adding healthy carbohydrates")
        elif ratios['carb_ratio'] > 65:
            carb_score = 25 - (ratios['carb_ratio'] - 65) * 0.5
            recommendations.append("Consider reducing refined carbohydrates")
        
        fat_score = 25
        if ratios['fat_ratio'] < 20:
            fat_score = ratios['fat_ratio'] * 1.25
            recommendations.append("Consider adding healthy fats")
        elif ratios['fat_ratio'] > 35:
            fat_score = 25 - (ratios['fat_ratio'] - 35) * 0.5
            recommendations.append("Consider reducing saturated fats")
        
        scores['macronutrient_balance'] = round((protein_score + carb_score + fat_score) / 3, 1)
        
        # Fiber score (0-25 points)
        fiber_per_1000_cal = (fiber / calories) * 1000
        if fiber_per_1000_cal >= 12.5:  # Excellent
            scores['fiber'] = 25
        elif fiber_per_1000_cal >= 10:  # Good
            scores['fiber'] = 20
        elif fiber_per_1000_cal >= 7.5:  # Fair
            scores['fiber'] = 15
        else:  # Poor
            scores['fiber'] = fiber_per_1000_cal * 2
            recommendations.append("Increase fiber intake with fruits, vegetables, and whole grains")
        
        # Sodium score (0-25 points)
        sodium_per_1000_cal = (sodium / calories) * 1000
        if sodium_per_1000_cal <= 1000:  # Excellent
            scores['sodium'] = 25
        elif sodium_per_1000_cal <= 1500:  # Good
            scores['sodium'] = 20
        elif sodium_per_1000_cal <= 2000:  # Fair
            scores['sodium'] = 15
        else:  # High
            scores['sodium'] = max(0, 25 - (sodium_per_1000_cal - 2000) * 0.01)
            recommendations.append("Reduce sodium intake by limiting processed foods")
        
        # Sugar score (0-25 points)
        sugar_percentage = (sugar / calories) * 100 * 4  # Convert to % of calories
        if sugar_percentage <= 10:  # Excellent
            scores['sugar'] = 25
        elif sugar_percentage <= 15:  # Good
            scores['sugar'] = 20
        elif sugar_percentage <= 20:  # Fair
            scores['sugar'] = 15
        else:  # High
            scores['sugar'] = max(0, 25 - (sugar_percentage - 20) * 0.5)
            recommendations.append("Reduce added sugar intake")
        
        # Calculate overall score
        overall_score = sum(scores.values())
        
        # Determine category
        if overall_score >= 90:
            category = "Excellent"
        elif overall_score >= 75:
            category = "Good"
        elif overall_score >= 60:
            category = "Fair"
        elif overall_score >= 45:
            category = "Poor"
        else:
            category = "Very Poor"
        
        return {
            'overall_score': round(overall_score, 1),
            'category': category,
            'individual_scores': scores,
            'recommendations': recommendations,
            'macronutrient_ratios': ratios
        }
    
    @staticmethod
    def calculate_meal_timing_calories(total_daily_calories: int, meal_type: str) -> int:
        """
        Calculate recommended calories for different meal types.
        Based on typical meal distribution patterns.
        """
        distributions = {
            'breakfast': 0.25,  # 25% of daily calories
            'lunch': 0.30,      # 30% of daily calories
            'dinner': 0.30,     # 30% of daily calories
            'snack': 0.075,     # 7.5% of daily calories
            'morning_snack': 0.075,
            'afternoon_snack': 0.075,
            'evening_snack': 0.05
        }
        
        percentage = distributions.get(meal_type.lower(), 0.25)
        return int(total_daily_calories * percentage)
    
    @staticmethod
    def compare_nutritional_profiles(food1: Dict[str, float], 
                                   food2: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare nutritional profiles of two foods.
        Returns detailed comparison with better/worse indicators.
        """
        comparison = {}
        nutrients = set(food1.keys()) | set(food2.keys())
        
        # Nutrients where higher is better
        higher_is_better = {'protein', 'fiber', 'vitamin_a', 'vitamin_c', 'calcium', 
                           'iron', 'potassium', 'omega_3'}
        
        # Nutrients where lower is better
        lower_is_better = {'sodium', 'sugar', 'saturated_fat', 'trans_fat', 'cholesterol'}
        
        for nutrient in nutrients:
            value1 = food1.get(nutrient, 0)
            value2 = food2.get(nutrient, 0)
            
            if value1 == value2:
                status = "equal"
                difference = 0
            elif nutrient in higher_is_better:
                if value1 > value2:
                    status = "food1_better"
                    difference = value1 - value2
                else:
                    status = "food2_better"
                    difference = value2 - value1
            elif nutrient in lower_is_better:
                if value1 < value2:
                    status = "food1_better"
                    difference = value2 - value1
                else:
                    status = "food2_better"
                    difference = value1 - value2
            else:
                # For neutral nutrients (like calories), just show difference
                status = "neutral"
                difference = abs(value1 - value2)
            
            comparison[nutrient] = {
                'food1_value': value1,
                'food2_value': value2,
                'difference': round(difference, 2),
                'status': status,
                'percentage_difference': round(
                    (difference / max(value1, value2, 1)) * 100, 1
                ) if max(value1, value2) > 0 else 0
            }
        
        return comparison
    
    @staticmethod
    def estimate_cooking_nutrition_changes(raw_nutrition: Dict[str, float], 
                                         cooking_method: str) -> Dict[str, float]:
        """
        Estimate nutritional changes due to cooking methods.
        Returns adjusted nutrition values.
        """
        # Cooking method effects (retention percentages)
        cooking_effects = {
            'raw': {
                'vitamin_c': 1.0, 'vitamin_a': 1.0, 'folate': 1.0,
                'calories': 1.0, 'protein': 1.0, 'fat': 1.0, 'carbs': 1.0
            },
            'steaming': {
                'vitamin_c': 0.85, 'vitamin_a': 0.95, 'folate': 0.80,
                'calories': 1.0, 'protein': 1.0, 'fat': 1.0, 'carbs': 1.0
            },
            'boiling': {
                'vitamin_c': 0.70, 'vitamin_a': 0.90, 'folate': 0.65,
                'calories': 1.0, 'protein': 1.0, 'fat': 1.0, 'carbs': 1.0
            },
            'roasting': {
                'vitamin_c': 0.75, 'vitamin_a': 0.85, 'folate': 0.75,
                'calories': 1.05, 'protein': 1.0, 'fat': 1.1, 'carbs': 1.0
            },
            'frying': {
                'vitamin_c': 0.60, 'vitamin_a': 0.80, 'folate': 0.70,
                'calories': 1.3, 'protein': 1.0, 'fat': 1.5, 'carbs': 1.0
            },
            'grilling': {
                'vitamin_c': 0.70, 'vitamin_a': 0.85, 'folate': 0.75,
                'calories': 1.0, 'protein': 1.0, 'fat': 0.9, 'carbs': 1.0
            }
        }
        
        method_effects = cooking_effects.get(cooking_method.lower(), cooking_effects['raw'])
        adjusted_nutrition = {}
        
        for nutrient, value in raw_nutrition.items():
            retention_factor = method_effects.get(nutrient, 1.0)
            adjusted_nutrition[nutrient] = round(value * retention_factor, 2)
        
        return adjusted_nutrition
