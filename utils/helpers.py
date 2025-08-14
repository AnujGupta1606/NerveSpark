from typing import List, Dict, Any, Optional, Tuple
import re
import unicodedata
import string

def clean_text(text: str) -> str:
    """Clean and normalize text data."""
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', str(text))
    
    # Normalize unicode
    text = unicodedata.normalize('NFKD', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_numbers_from_text(text: str) -> List[float]:
    """Extract numbers from text string."""
    if not text:
        return []
    
    # Pattern to match numbers (including decimals)
    pattern = r'\d+\.?\d*'
    matches = re.findall(pattern, str(text))
    
    return [float(match) for match in matches]

def parse_cooking_time(time_str: str) -> int:
    """Parse cooking time string and return minutes."""
    if not time_str:
        return 0
    
    time_str = str(time_str).lower()
    total_minutes = 0
    
    # Extract hours
    hours_match = re.search(r'(\d+)\s*(?:hr|hour|hours)', time_str)
    if hours_match:
        total_minutes += int(hours_match.group(1)) * 60
    
    # Extract minutes
    minutes_match = re.search(r'(\d+)\s*(?:min|minute|minutes)', time_str)
    if minutes_match:
        total_minutes += int(minutes_match.group(1))
    
    # If no specific unit found, assume the first number is minutes
    if total_minutes == 0:
        numbers = extract_numbers_from_text(time_str)
        if numbers:
            total_minutes = int(numbers[0])
    
    return total_minutes

def format_cooking_time(minutes: int) -> str:
    """Format minutes into human-readable cooking time."""
    if minutes <= 0:
        return "Unknown"
    
    if minutes < 60:
        return f"{minutes} min"
    else:
        hours = minutes // 60
        remaining_minutes = minutes % 60
        if remaining_minutes == 0:
            return f"{hours} hr"
        else:
            return f"{hours} hr {remaining_minutes} min"

def extract_serving_size(servings_str: str) -> int:
    """Extract serving size from string."""
    if not servings_str:
        return 1
    
    numbers = extract_numbers_from_text(str(servings_str))
    return int(numbers[0]) if numbers else 1

def normalize_ingredient_name(ingredient: str) -> str:
    """Normalize ingredient name for better matching."""
    if not ingredient:
        return ""
    
    ingredient = clean_text(ingredient)
    ingredient = ingredient.lower()
    
    # Remove common descriptors
    descriptors = [
        'fresh', 'dried', 'ground', 'chopped', 'diced', 'minced', 'sliced',
        'whole', 'half', 'quarter', 'organic', 'raw', 'cooked', 'frozen',
        'canned', 'bottled', 'extra virgin', 'virgin', 'unsalted', 'salted'
    ]
    
    for descriptor in descriptors:
        ingredient = ingredient.replace(descriptor, '').strip()
    
    # Remove measurements
    ingredient = re.sub(r'\d+\.?\d*\s*(cups?|tbsp|tsp|oz|lbs?|g|kg|ml|l)\s*', '', ingredient)
    
    # Remove extra whitespace and punctuation
    ingredient = re.sub(r'[,\(\)]', '', ingredient)
    ingredient = re.sub(r'\s+', ' ', ingredient)
    
    return ingredient.strip()

def categorize_cuisine(recipe_name: str, ingredients: List[str], 
                      instructions: str = "") -> str:
    """Automatically categorize cuisine type based on recipe content."""
    content = f"{recipe_name} {' '.join(ingredients)} {instructions}".lower()
    
    cuisine_keywords = {
        'italian': ['pasta', 'pizza', 'risotto', 'parmesan', 'mozzarella', 'basil', 
                   'oregano', 'tomato sauce', 'olive oil', 'marinara'],
        'mexican': ['salsa', 'tortilla', 'cilantro', 'jalapeño', 'cumin', 'chili',
                   'avocado', 'lime', 'beans', 'cheese', 'pepper'],
        'asian': ['soy sauce', 'ginger', 'garlic', 'sesame', 'rice', 'noodles',
                 'stir fry', 'tofu', 'miso', 'wasabi'],
        'chinese': ['soy sauce', 'ginger', 'scallions', 'sesame oil', 'rice wine',
                   'stir fry', 'wok', 'hoisin', 'five spice'],
        'indian': ['curry', 'turmeric', 'cumin', 'coriander', 'garam masala',
                  'cardamom', 'cinnamon', 'ghee', 'naan', 'basmati'],
        'mediterranean': ['olive oil', 'feta', 'olives', 'lemon', 'herbs',
                         'tomatoes', 'chickpeas', 'tahini', 'hummus'],
        'american': ['barbecue', 'burger', 'fries', 'mac and cheese', 'cornbread',
                    'pulled pork', 'coleslaw', 'ranch'],
        'french': ['butter', 'cream', 'wine', 'herbs de provence', 'brie',
                  'croissant', 'baguette', 'roux', 'confit'],
        'thai': ['coconut milk', 'fish sauce', 'lime leaves', 'lemongrass',
                'thai chili', 'pad thai', 'curry paste', 'basil'],
        'japanese': ['miso', 'sake', 'mirin', 'nori', 'wasabi', 'sushi',
                    'tempura', 'dashi', 'soba', 'udon']
    }
    
    cuisine_scores = {}
    for cuisine, keywords in cuisine_keywords.items():
        score = sum(1 for keyword in keywords if keyword in content)
        if score > 0:
            cuisine_scores[cuisine] = score
    
    if cuisine_scores:
        return max(cuisine_scores, key=cuisine_scores.get)
    
    return "unknown"

def estimate_difficulty_level(ingredient_count: int, cook_time: int, 
                            instructions: str = "") -> str:
    """Estimate recipe difficulty level."""
    difficulty_score = 0
    
    # Ingredient count factor
    if ingredient_count > 15:
        difficulty_score += 3
    elif ingredient_count > 10:
        difficulty_score += 2
    elif ingredient_count > 5:
        difficulty_score += 1
    
    # Cooking time factor
    if cook_time > 180:  # 3 hours
        difficulty_score += 3
    elif cook_time > 60:  # 1 hour
        difficulty_score += 2
    elif cook_time > 30:  # 30 minutes
        difficulty_score += 1
    
    # Instructions complexity factor
    if instructions:
        instruction_text = instructions.lower()
        complex_techniques = [
            'julienne', 'brunoise', 'chiffonade', 'sous vide', 'confit',
            'emulsify', 'temper', 'fold', 'reduce', 'deglaze', 'flambé',
            'braise', 'poach', 'clarify', 'caramelize'
        ]
        
        complexity_count = sum(1 for technique in complex_techniques 
                             if technique in instruction_text)
        difficulty_score += min(complexity_count, 3)
        
        # Length of instructions
        if len(instructions) > 1000:
            difficulty_score += 2
        elif len(instructions) > 500:
            difficulty_score += 1
    
    # Determine difficulty level
    if difficulty_score <= 2:
        return "easy"
    elif difficulty_score <= 5:
        return "medium"
    elif difficulty_score <= 8:
        return "hard"
    else:
        return "expert"

def create_recipe_summary(recipe_data: Dict[str, Any]) -> str:
    """Create a concise recipe summary."""
    name = recipe_data.get('name', 'Unknown Recipe')
    calories = recipe_data.get('calories', 0)
    cook_time = recipe_data.get('cook_time', 0)
    servings = recipe_data.get('servings', 1)
    cuisine = recipe_data.get('cuisine', 'Unknown')
    
    cook_time_str = format_cooking_time(parse_cooking_time(str(cook_time)))
    
    summary_parts = [
        f"'{name}'",
        f"{calories} calories per serving" if calories > 0 else "Calories unknown",
        f"Serves {servings}",
        f"{cook_time_str} cook time",
        f"{cuisine.title()} cuisine"
    ]
    
    return " | ".join(summary_parts)

def validate_recipe_data(recipe_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate recipe data completeness and quality."""
    errors = []
    warnings = []
    
    # Required fields
    required_fields = ['name']
    for field in required_fields:
        if not recipe_data.get(field):
            errors.append(f"Missing required field: {field}")
    
    # Recommended fields
    recommended_fields = ['ingredients', 'instructions', 'calories']
    for field in recommended_fields:
        if not recipe_data.get(field):
            warnings.append(f"Missing recommended field: {field}")
    
    # Data validation
    if recipe_data.get('calories', 0) < 0:
        errors.append("Calories cannot be negative")
    
    if recipe_data.get('servings', 1) <= 0:
        errors.append("Servings must be greater than 0")
    
    # Nutrition validation
    nutrition_fields = ['protein', 'carbs', 'fat', 'fiber']
    for field in nutrition_fields:
        value = recipe_data.get(field, 0)
        if value < 0:
            errors.append(f"{field.title()} cannot be negative")
    
    # Check if macronutrients add up reasonably
    calories = recipe_data.get('calories', 0)
    if calories > 0:
        calculated_calories = (
            recipe_data.get('protein', 0) * 4 +
            recipe_data.get('carbs', 0) * 4 +
            recipe_data.get('fat', 0) * 9
        )
        
        if calculated_calories > 0:
            difference_percent = abs(calories - calculated_calories) / calories * 100
            if difference_percent > 30:  # More than 30% difference
                warnings.append(f"Macronutrients don't match calorie count (difference: {difference_percent:.1f}%)")
    
    is_valid = len(errors) == 0
    return is_valid, errors + warnings

def generate_recipe_tags(recipe_data: Dict[str, Any]) -> List[str]:
    """Generate appropriate tags for a recipe."""
    tags = []
    
    # Time-based tags
    cook_time = parse_cooking_time(str(recipe_data.get('cook_time', '')))
    if cook_time <= 15:
        tags.append('quick')
    elif cook_time <= 30:
        tags.append('fast')
    elif cook_time >= 120:
        tags.append('slow-cook')
    
    # Calorie-based tags
    calories = recipe_data.get('calories', 0)
    if calories <= 200:
        tags.append('low-calorie')
    elif calories <= 400:
        tags.append('moderate-calorie')
    elif calories >= 600:
        tags.append('high-calorie')
    
    # Protein-based tags
    protein = recipe_data.get('protein', 0)
    if protein >= 20:
        tags.append('high-protein')
    elif protein >= 10:
        tags.append('good-protein')
    
    # Fiber-based tags
    fiber = recipe_data.get('fiber', 0)
    if fiber >= 10:
        tags.append('high-fiber')
    elif fiber >= 5:
        tags.append('good-fiber')
    
    # Fat-based tags
    fat = recipe_data.get('fat', 0)
    if fat <= 5:
        tags.append('low-fat')
    elif fat >= 15:
        tags.append('high-fat')
    
    # Sodium-based tags
    sodium = recipe_data.get('sodium', 0)
    if sodium <= 200:
        tags.append('low-sodium')
    elif sodium >= 800:
        tags.append('high-sodium')
    
    # Difficulty tags
    ingredient_count = len(recipe_data.get('ingredients_list', []))
    difficulty = estimate_difficulty_level(
        ingredient_count, cook_time, recipe_data.get('instructions', '')
    )
    tags.append(difficulty)
    
    # Meal type tags (basic heuristics)
    name = recipe_data.get('name', '').lower()
    ingredients = ' '.join(recipe_data.get('ingredients_list', [])).lower()
    
    if any(word in name for word in ['pancake', 'cereal', 'oatmeal', 'toast', 'egg']):
        tags.append('breakfast')
    elif any(word in name for word in ['salad', 'sandwich', 'soup']):
        tags.append('lunch')
    elif any(word in name for word in ['pasta', 'steak', 'roast', 'casserole']):
        tags.append('dinner')
    elif any(word in name for word in ['cookie', 'cake', 'muffin', 'bar']):
        tags.append('dessert')
    
    # Cooking method tags
    instructions = recipe_data.get('instructions', '').lower()
    if 'bake' in instructions or 'oven' in instructions:
        tags.append('baked')
    if 'grill' in instructions:
        tags.append('grilled')
    if 'fry' in instructions:
        tags.append('fried')
    if 'boil' in instructions or 'simmer' in instructions:
        tags.append('boiled')
    if 'steam' in instructions:
        tags.append('steamed')
    
    return list(set(tags))  # Remove duplicates

def format_ingredient_list(ingredients: List[str]) -> str:
    """Format ingredient list for display."""
    if not ingredients:
        return "No ingredients listed"
    
    formatted_ingredients = []
    for ingredient in ingredients:
        # Capitalize first letter of each ingredient
        formatted_ingredient = ingredient.strip().capitalize()
        if formatted_ingredient:
            formatted_ingredients.append(f"• {formatted_ingredient}")
    
    return '\n'.join(formatted_ingredients)

def calculate_recipe_cost_estimate(ingredients: List[str]) -> Dict[str, Any]:
    """Estimate recipe cost based on ingredients (very basic estimation)."""
    # This is a simplified cost estimation
    # In a real application, you'd integrate with a pricing API
    
    ingredient_costs = {
        # Proteins (per serving estimate)
        'chicken': 2.50, 'beef': 4.00, 'pork': 3.00, 'fish': 3.50,
        'salmon': 5.00, 'shrimp': 4.50, 'eggs': 0.50, 'tofu': 1.00,
        
        # Grains/Starches
        'rice': 0.30, 'pasta': 0.40, 'bread': 0.50, 'potato': 0.40,
        'quinoa': 1.20, 'oats': 0.25,
        
        # Vegetables
        'onion': 0.20, 'garlic': 0.10, 'tomato': 0.50, 'carrot': 0.30,
        'broccoli': 0.60, 'spinach': 0.80, 'bell pepper': 0.70,
        
        # Dairy
        'milk': 0.30, 'cheese': 1.00, 'butter': 0.40, 'cream': 0.60,
        'yogurt': 0.50,
        
        # Others
        'oil': 0.20, 'salt': 0.05, 'pepper': 0.10, 'herbs': 0.30,
        'spices': 0.25
    }
    
    total_cost = 0
    identified_ingredients = 0
    
    for ingredient in ingredients:
        ingredient_lower = ingredient.lower()
        for key, cost in ingredient_costs.items():
            if key in ingredient_lower:
                total_cost += cost
                identified_ingredients += 1
                break
        else:
            # Default cost for unidentified ingredients
            total_cost += 0.50
    
    cost_category = "unknown"
    if total_cost <= 3:
        cost_category = "budget"
    elif total_cost <= 6:
        cost_category = "moderate"
    elif total_cost <= 10:
        cost_category = "expensive"
    else:
        cost_category = "luxury"
    
    return {
        'estimated_cost': round(total_cost, 2),
        'cost_category': cost_category,
        'confidence': min(100, (identified_ingredients / len(ingredients)) * 100) if ingredients else 0
    }
