# Example Recipes Template
# Copy this format to add multiple recipes at once

SAMPLE_RECIPES = [
    {
        'name': 'Spaghetti Carbonara',
        'cuisine': 'italian',
        'ingredients': 'spaghetti, eggs, pancetta, parmesan cheese, black pepper, olive oil',
        'instructions': 'Cook spaghetti. Fry pancetta until crispy. Beat eggs with parmesan. Toss hot pasta with pancetta, then eggs. Season with pepper.',
        'prep_time': '10 min',
        'cook_time': '15 min',
        'servings': 4,
        'calories': 420,
        'protein': 18,
        'carbs': 45,
        'fat': 18,
        'fiber': 2,
        'sugar': 3,
        'sodium': 680,
        'description': 'Classic Italian pasta dish with creamy egg and cheese sauce.'
    },
    {
        'name': 'Thai Green Curry',
        'cuisine': 'thai',
        'ingredients': 'green curry paste, coconut milk, chicken, thai basil, bell peppers, bamboo shoots, fish sauce, palm sugar',
        'instructions': 'Fry curry paste. Add coconut milk gradually. Add chicken and vegetables. Simmer until cooked. Add basil and seasonings.',
        'prep_time': '15 min',
        'cook_time': '20 min',
        'servings': 3,
        'calories': 380,
        'protein': 25,
        'carbs': 12,
        'fat': 28,
        'fiber': 3,
        'sugar': 8,
        'sodium': 890,
        'description': 'Aromatic and spicy Thai curry with tender chicken and vegetables.'
    },
    {
        'name': 'Avocado Toast with Poached Egg',
        'cuisine': 'american',
        'ingredients': 'whole grain bread, avocado, eggs, lemon juice, salt, pepper, chili flakes',
        'instructions': 'Toast bread. Mash avocado with lemon and seasonings. Poach eggs. Spread avocado on toast, top with poached egg.',
        'prep_time': '5 min',
        'cook_time': '10 min',
        'servings': 2,
        'calories': 290,
        'protein': 14,
        'carbs': 20,
        'fat': 18,
        'fiber': 8,
        'sugar': 2,
        'sodium': 350,
        'description': 'Healthy breakfast with good fats and protein for sustained energy.'
    }
]

# To add these recipes:
# 1. Run: python add_recipe.py 
# 2. Or add them manually to setup_data.py in the SAMPLE_RECIPES list
# 3. Then run: python setup_data.py
