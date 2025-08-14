from typing import Dict, List, Any, Optional, Tuple
import logging
import streamlit as st
import os

# Check if we're in a cloud environment
IS_CLOUD_DEPLOYMENT = os.getenv('STREAMLIT_SHARING_MODE') is not None or '/mount/src/' in os.getcwd()

# Import with error handling for cloud deployment
try:
    from src.embeddings import RecipeEmbeddingGenerator
    from src.vector_store import RecipeVectorStore
    from src.health_logic import HealthLogicEngine
    from src.substitution import IngredientSubstitutionEngine
    from src.cloud_data import initialize_cloud_data
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    IMPORTS_SUCCESSFUL = False
    logger = logging.getLogger(__name__)
    logger.error(f"Import error: {e}")
    if not IS_CLOUD_DEPLOYMENT:  # Only show error in local environment
        st.error(f"Import error: {e}")
        st.stop()

import openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NutritionalRAGSystem:
    """
    Main RAG (Retrieval-Augmented Generation) system for nutritional recommendations.
    Combines recipe retrieval, health logic, substitutions, and AI generation.
    """
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 vector_db_path: str = "./chroma_db",
                 use_openai: bool = False):  # Disable OpenAI by default for cloud
        
        try:
            self.embedding_generator = RecipeEmbeddingGenerator(
                model_name=embedding_model, 
                use_openai=False  # Use local embeddings for cloud compatibility
            )
            self.vector_store = RecipeVectorStore(db_path=vector_db_path)
            self.health_engine = HealthLogicEngine()
            self.substitution_engine = IngredientSubstitutionEngine()
            
            # Initialize cloud data if no local data available
            self._ensure_data_availability()
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            # Initialize with minimal functionality for cloud deployment
            self._initialize_minimal_system()
        
        # OpenAI setup for text generation (optional)
        self.use_openai = use_openai
        if use_openai and os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            self.use_openai = False
    
    def _ensure_data_availability(self):
        """Ensure data is available for the system."""
        try:
            # Check if vector store has data
            if self.vector_store.collection and self.vector_store.collection.count() == 0:
                logger.info("No data in vector store, initializing with sample data...")
                recipes_df, nutrition_df = initialize_cloud_data()
                if not recipes_df.empty:
                    self._populate_from_dataframe(recipes_df)
        except Exception as e:
            logger.warning(f"Could not check/populate vector store: {e}")
    
    def _initialize_minimal_system(self):
        """Initialize minimal system for cloud deployment."""
        logger.warning("Initializing minimal RAG system for cloud deployment")
        self.embedding_generator = None
        self.vector_store = None
        self.health_engine = HealthLogicEngine()
        self.substitution_engine = IngredientSubstitutionEngine()
        self.use_openai = False
    
    def _populate_from_dataframe(self, recipes_df):
        """Populate vector store with sample data."""
        try:
            for _, recipe in recipes_df.iterrows():
                # Create recipe text for embedding
                recipe_text = f"{recipe['name']} {' '.join(recipe['ingredients'])} {recipe['instructions']}"
                
                # Generate embedding
                embedding = self.embedding_generator.generate_embedding(recipe_text)
                
                # Store in vector database
                self.vector_store.add_recipe(
                    recipe_id=recipe['name'],
                    recipe_data=recipe.to_dict(),
                    embedding=embedding
                )
            logger.info(f"Populated vector store with {len(recipes_df)} recipes")
        except Exception as e:
            logger.error(f"Error populating vector store: {e}")
    
    def process_user_query(self, 
                          query: str,
                          user_profile: Dict[str, Any],
                          max_results: int = 5) -> Dict[str, Any]:
        """
        Process a user query and return personalized recipe recommendations.
        
        Args:
            query: User's natural language query
            user_profile: User's health profile, restrictions, goals
            max_results: Maximum number of recipes to return
            
        Returns:
            Comprehensive response with recipes, recommendations, and explanations
        """
        logger.info(f"Processing query: {query}")
        
        try:
            # Step 1: Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            # Step 2: Extract dietary filters from user profile
            dietary_filters = user_profile.get('dietary_restrictions', [])
            health_conditions = user_profile.get('health_conditions', [])
            allergies = user_profile.get('allergies', [])
            
            # Step 3: Perform hybrid search
            search_results = self.vector_store.hybrid_search(
                query_embedding=query_embedding.tolist(),
                text_query=query,
                dietary_filters=dietary_filters,
                n_results=max_results * 2  # Get more results for filtering
            )
            
            # Step 4: Apply health and safety filtering
            safe_recipes = []
            logger.info(f"Total search results: {len(search_results['recipes'])}")
            
            for recipe_data in search_results['recipes']:
                recipe_metadata = recipe_data['metadata']
                
                # Get health recommendations
                health_assessment = self.health_engine.get_recipe_recommendations(
                    recipe_metadata, user_profile
                )
                
                logger.info(f"Recipe: {recipe_metadata.get('name', 'Unknown')}, Safe: {health_assessment['is_safe']}, Score: {health_assessment['overall_score']}")
                
                # Only include safe recipes with decent scores (more lenient for demo)
                if health_assessment['is_safe'] and health_assessment['overall_score'] > 0.1:
                    recipe_with_assessment = {
                        **recipe_data,
                        'health_assessment': health_assessment
                    }
                    safe_recipes.append(recipe_with_assessment)
            
            logger.info(f"Safe recipes after filtering: {len(safe_recipes)}")
            
            # Step 5: Sort by combined score (vector similarity + health score) and limit results
            # Combine vector similarity (distance) with health assessment
            for recipe in safe_recipes:
                vector_similarity = recipe.get('similarity_score', 0.5)  # Default similarity
                health_score = recipe['health_assessment']['overall_score']
                
                # Combined score: 70% vector similarity + 30% health score
                combined_score = (0.7 * vector_similarity) + (0.3 * health_score)
                recipe['combined_score'] = combined_score
            
            # Sort by combined score instead of just health score
            safe_recipes.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
            final_recipes = safe_recipes[:max_results]
            
            # Step 6: Generate substitution suggestions
            recipes_with_substitutions = []
            for recipe in final_recipes:
                substitution_suggestions = self.substitution_engine.get_substitution_suggestions(
                    recipe['metadata'], user_profile
                )
                recipe['substitution_suggestions'] = substitution_suggestions
                recipes_with_substitutions.append(recipe)
            
            # Step 7: Generate AI explanation and summary
            ai_summary = self._generate_ai_summary(
                query, recipes_with_substitutions, user_profile
            )
            
            # Step 8: Compile final response
            response = {
                'query': query,
                'total_found': len(search_results['recipes']),
                'safe_recipes_count': len(safe_recipes),
                'recommended_recipes': recipes_with_substitutions,
                'ai_summary': ai_summary,
                'user_profile_applied': {
                    'dietary_restrictions': dietary_filters,
                    'allergies': allergies,
                    'health_conditions': health_conditions
                },
                'search_metadata': {
                    'embedding_model': self.embedding_generator.model_name,
                    'query_embedding_dim': len(query_embedding),
                    'filters_applied': len(dietary_filters) + len(allergies) + len(health_conditions)
                }
            }
            
            logger.info(f"Successfully processed query. Found {len(final_recipes)} recommendations.")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'query': query,
                'error': str(e),
                'recommended_recipes': [],
                'ai_summary': "Sorry, I encountered an error processing your request. Please try again."
            }
    
    def _generate_ai_summary(self, 
                           query: str,
                           recipes: List[Dict[str, Any]],
                           user_profile: Dict[str, Any]) -> str:
        """Generate AI summary of recommendations."""
        
        if not recipes:
            return "I couldn't find any recipes that match your dietary requirements and health needs. Try adjusting your search terms or dietary preferences."
        
        # Prepare context for AI generation
        context = self._prepare_context_for_ai(query, recipes, user_profile)
        
        if self.use_openai and os.getenv("OPENAI_API_KEY"):
            return self._generate_openai_summary(context)
        else:
            return self._generate_template_summary(context)
    
    def _prepare_context_for_ai(self, 
                              query: str,
                              recipes: List[Dict[str, Any]],
                              user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context information for AI generation."""
        
        # Extract key information from recipes
        recipe_summaries = []
        for recipe in recipes:
            metadata = recipe['metadata']
            health_assessment = recipe['health_assessment']
            
            recipe_summary = {
                'name': metadata.get('name', 'Unknown'),
                'calories': metadata.get('calories', 0),
                'protein': metadata.get('protein', 0),
                'health_score': health_assessment['overall_score'],
                'recommendation_level': health_assessment['recommendation_level'],
                'key_benefits': health_assessment.get('health_recommendations', [])[:2]
            }
            recipe_summaries.append(recipe_summary)
        
        return {
            'query': query,
            'user_restrictions': user_profile.get('dietary_restrictions', []),
            'user_allergies': user_profile.get('allergies', []),
            'user_conditions': user_profile.get('health_conditions', []),
            'recipe_count': len(recipes),
            'top_recipes': recipe_summaries[:3],
            'avg_health_score': sum(r['health_score'] for r in recipe_summaries) / len(recipe_summaries)
        }
    
    def _generate_openai_summary(self, context: Dict[str, Any]) -> str:
        """Generate summary using OpenAI API."""
        try:
            prompt = f"""
You are a nutrition expert AI assistant. Based on the user's query and their health profile, provide a personalized summary of recipe recommendations.

User Query: "{context['query']}"

User Profile:
- Dietary Restrictions: {', '.join(context['user_restrictions']) if context['user_restrictions'] else 'None'}
- Allergies: {', '.join(context['user_allergies']) if context['user_allergies'] else 'None'}
- Health Conditions: {', '.join(context['user_conditions']) if context['user_conditions'] else 'None'}

Top Recipe Recommendations:
{self._format_recipes_for_prompt(context['top_recipes'])}

Please provide a warm, personalized summary that:
1. Acknowledges their health needs and restrictions
2. Highlights why these recipes are good choices for them
3. Mentions key nutritional benefits
4. Gives 1-2 practical tips for their health conditions
5. Keep it conversational and encouraging (2-3 paragraphs max)
"""
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return self._generate_template_summary(context)
    
    def _format_recipes_for_prompt(self, recipes: List[Dict[str, Any]]) -> str:
        """Format recipes for AI prompt."""
        formatted = []
        for i, recipe in enumerate(recipes, 1):
            formatted.append(
                f"{i}. {recipe['name']} - {recipe['calories']} calories, "
                f"{recipe['protein']}g protein ({recipe['recommendation_level']})"
            )
        return '\n'.join(formatted)
    
    def _generate_template_summary(self, context: Dict[str, Any]) -> str:
        """Generate summary using templates (fallback)."""
        
        # Start with greeting based on restrictions
        greeting_parts = []
        if context['user_restrictions']:
            greeting_parts.append(f"following your {', '.join(context['user_restrictions'])} diet")
        if context['user_conditions']:
            greeting_parts.append(f"supporting your {', '.join(context['user_conditions'])} management")
        
        greeting = f"Great! I found {context['recipe_count']} recipes"
        if greeting_parts:
            greeting += f" that are perfect for {' and '.join(greeting_parts)}"
        greeting += "."
        
        # Highlight top recipe
        if context['top_recipes']:
            top_recipe = context['top_recipes'][0]
            highlight = f" I especially recommend '{top_recipe['name']}' with {top_recipe['calories']} calories and {top_recipe['protein']}g of protein - it's {top_recipe['recommendation_level'].lower()}."
        else:
            highlight = ""
        
        # Add health tips based on conditions
        tips = []
        for condition in context['user_conditions']:
            if condition.lower() == 'diabetes':
                tips.append("focus on recipes with high fiber content to help manage blood sugar")
            elif condition.lower() == 'hypertension':
                tips.append("choose low-sodium options to support healthy blood pressure")
            elif condition.lower() == 'heart_disease':
                tips.append("prioritize recipes with healthy fats and lean proteins")
        
        tip_text = ""
        if tips:
            tip_text = f" For your health goals, I recommend you {tips[0]}."
        
        return greeting + highlight + tip_text
    
    def get_meal_plan_suggestions(self, 
                                user_profile: Dict[str, Any],
                                days: int = 3,
                                meals_per_day: int = 3) -> Dict[str, Any]:
        """Generate multi-day meal plan suggestions."""
        
        meal_types = ['breakfast', 'lunch', 'dinner']
        if meals_per_day > 3:
            meal_types.extend(['snack'] * (meals_per_day - 3))
        
        meal_plan = {}
        
        for day in range(1, days + 1):
            day_plan = {}
            
            for meal_type in meal_types:
                # Create meal-specific query
                query = f"healthy {meal_type} recipe"
                
                # Add variety by including different cuisines or cooking methods
                if day % 3 == 1:
                    query += " quick and easy"
                elif day % 3 == 2:
                    query += " mediterranean style"
                else:
                    query += " high protein"
                
                # Get recommendations for this meal
                meal_recommendations = self.process_user_query(
                    query=query,
                    user_profile=user_profile,
                    max_results=2
                )
                
                # Take the top recommendation
                if meal_recommendations['recommended_recipes']:
                    day_plan[meal_type] = meal_recommendations['recommended_recipes'][0]
                else:
                    day_plan[meal_type] = None
            
            meal_plan[f"day_{day}"] = day_plan
        
        # Calculate daily nutrition totals
        nutrition_summary = self._calculate_meal_plan_nutrition(meal_plan)
        
        return {
            'meal_plan': meal_plan,
            'nutrition_summary': nutrition_summary,
            'duration_days': days,
            'user_profile': user_profile
        }
    
    def _calculate_meal_plan_nutrition(self, meal_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate total nutrition for meal plan."""
        daily_totals = {}
        
        for day_key, day_meals in meal_plan.items():
            daily_nutrition = {
                'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0, 'fiber': 0, 'sodium': 0
            }
            
            for meal_type, meal_data in day_meals.items():
                if meal_data and 'metadata' in meal_data:
                    metadata = meal_data['metadata']
                    for nutrient in daily_nutrition.keys():
                        daily_nutrition[nutrient] += metadata.get(nutrient, 0)
            
            daily_totals[day_key] = daily_nutrition
        
        # Calculate averages
        if daily_totals:
            avg_nutrition = {}
            nutrient_keys = list(daily_totals.values())[0].keys()
            
            for nutrient in nutrient_keys:
                avg_nutrition[f"avg_{nutrient}"] = sum(
                    day_data[nutrient] for day_data in daily_totals.values()
                ) / len(daily_totals)
            
            return {
                'daily_totals': daily_totals,
                'averages': avg_nutrition
            }
        
        return {'daily_totals': {}, 'averages': {}}
    
    def analyze_recipe_compatibility(self, 
                                   recipe_data: Dict[str, Any],
                                   user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Provide detailed compatibility analysis for a specific recipe."""
        
        # Get health assessment
        health_assessment = self.health_engine.get_recipe_recommendations(
            recipe_data, user_profile
        )
        
        # Get substitution suggestions
        substitution_suggestions = self.substitution_engine.get_substitution_suggestions(
            recipe_data, user_profile
        )
        
        # Apply substitutions and get modified recipe
        modified_recipe = self.substitution_engine.apply_substitutions_to_recipe(
            recipe_data, user_profile
        )
        
        # Calculate nutritional comparison
        nutritional_comparison = self._compare_nutritional_profiles(
            recipe_data, modified_recipe
        )
        
        return {
            'original_recipe': recipe_data,
            'modified_recipe': modified_recipe,
            'health_assessment': health_assessment,
            'substitution_suggestions': substitution_suggestions,
            'nutritional_comparison': nutritional_comparison,
            'recommendation': self._generate_compatibility_recommendation(health_assessment, substitution_suggestions)
        }
    
    def _compare_nutritional_profiles(self, 
                                    original: Dict[str, Any],
                                    modified: Dict[str, Any]) -> Dict[str, Any]:
        """Compare nutritional profiles of original and modified recipes."""
        
        nutrients = ['calories', 'protein', 'carbs', 'fat', 'fiber', 'sodium', 'sugar']
        comparison = {}
        
        for nutrient in nutrients:
            original_value = original.get(nutrient, 0)
            modified_value = modified.get(nutrient, 0)
            
            if original_value > 0:
                change_percent = ((modified_value - original_value) / original_value) * 100
            else:
                change_percent = 0
            
            comparison[nutrient] = {
                'original': original_value,
                'modified': modified_value,
                'change_percent': change_percent,
                'change_direction': 'increase' if change_percent > 0 else 'decrease' if change_percent < 0 else 'no_change'
            }
        
        return comparison
    
    def _generate_compatibility_recommendation(self, 
                                             health_assessment: Dict[str, Any],
                                             substitution_suggestions: Dict[str, Any]) -> str:
        """Generate a compatibility recommendation."""
        
        if not health_assessment['is_safe']:
            return "❌ This recipe is not recommended due to safety concerns with your allergies or restrictions."
        
        score = health_assessment['overall_score']
        substitutions_count = len(substitution_suggestions.get('substitutions', []))
        
        if score >= 0.8:
            if substitutions_count == 0:
                return "🌟 This recipe is perfect for you as-is!"
            else:
                return f"🌟 Excellent choice! Consider {substitutions_count} simple substitutions to make it even better."
        elif score >= 0.6:
            return f"✅ Good recipe choice. I suggest {substitutions_count} substitutions to optimize it for your health needs."
        elif score >= 0.4:
            return f"⚠️ This recipe needs some modifications. Try the {substitutions_count} suggested substitutions to make it healthier for you."
        else:
            return "❌ This recipe may not be the best choice for your health profile. Consider looking for alternatives."
