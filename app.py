"""
NerveSpark - Intelligent Nutrition Assistant
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import logging
import sys
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import modules with error handling for cloud deployment
try:
    from src.rag_system import NutritionalRAGSystem
    from models.user_profile import UserProfile, UserProfileManager
    from utils.nutrition_calc import NutritionCalculator
    from utils.helpers import format_cooking_time, parse_cooking_time
    from config import APP_TITLE, APP_ICON, SIDEBAR_TITLE
except ImportError as e:
    st.error(f"""
    **Import Error**: {str(e)}
    
    This might be due to missing dependencies or cloud deployment issues.
    Please check that all required packages are installed.
    """)
    st.code(traceback.format_exc())
    st.stop()

# Page configuration
st.set_page_config(
    page_title=APP_TITLE if 'APP_TITLE' in locals() else "NerveSpark",
    page_icon=APP_ICON if 'APP_ICON' in locals() else "🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

class NerveSparkApp:
    """Main Streamlit application class for NerveSpark."""
    
    def __init__(self):
        try:
            self.rag_system = NutritionalRAGSystem()
            self.profile_manager = UserProfileManager()
            self.nutrition_calc = NutritionCalculator()
            
            # Initialize session state
            if 'user_profile' not in st.session_state:
                st.session_state.user_profile = None
            if 'query_history' not in st.session_state:
                st.session_state.query_history = []
                
        except Exception as e:
            st.error(f"""
            **Initialization Error**: {str(e)}
            
            The application failed to initialize properly. 
            This might be due to cloud deployment limitations.
            """)
            st.code(traceback.format_exc())
            # Continue with minimal functionality
            self.rag_system = None
            self.profile_manager = None
            self.nutrition_calc = None
    
    def render_header(self):
        """Render the application header."""
        st.title(f"{APP_ICON} {APP_TITLE}")
        st.markdown("""
        Welcome to **NerveSpark** - your intelligent nutrition assistant! 
        Get personalized recipe recommendations based on your health profile, 
        dietary restrictions, and nutritional goals.
        """)
        st.divider()
    
    def render_sidebar(self):
        """Render the sidebar with user profile management."""
        with st.sidebar:
            st.header(f"👤 {SIDEBAR_TITLE}")
            
            # User profile selection
            profiles = self.profile_manager.list_profiles()
            profile_options = ["Create New Profile"] + [f"{p['name']} ({p['user_id']})" for p in profiles]
            
            selected_profile = st.selectbox(
                "Select or Create Profile",
                profile_options,
                key="profile_selector"
            )
            
            if selected_profile == "Create New Profile":
                self.render_profile_creation_form()
            else:
                # Extract user_id from selection
                user_id = selected_profile.split("(")[-1].rstrip(")")
                profile = self.profile_manager.load_profile(user_id)
                if profile:
                    st.session_state.user_profile = profile
                    self.render_profile_display(profile)
                    
                    if st.button("Edit Profile", key="edit_profile"):
                        self.render_profile_edit_form(profile)
    
    def render_profile_creation_form(self):
        """Render form for creating a new user profile."""
        st.subheader("Create Your Profile")
        
        with st.form("create_profile_form"):
            # Basic information
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Name*", key="new_name")
                age = st.number_input("Age", min_value=1, max_value=120, value=30, key="new_age")
                gender = st.selectbox("Gender", ["female", "male", "other"], key="new_gender")
            
            with col2:
                height_cm = st.number_input("Height (cm)", min_value=100, max_value=250, value=170, key="new_height")
                weight_kg = st.number_input("Weight (kg)", min_value=30, max_value=300, value=70, key="new_weight")
                activity_level = st.selectbox(
                    "Activity Level",
                    ["sedentary", "light", "moderate", "active", "very_active"],
                    index=2,
                    key="new_activity"
                )
            
            # Dietary restrictions
            st.subheader("Dietary Preferences")
            dietary_restrictions = st.multiselect(
                "Dietary Restrictions",
                ["vegan", "vegetarian", "gluten_free", "dairy_free", "nut_free", "low_carb", "keto"],
                key="new_dietary"
            )
            
            allergies = st.multiselect(
                "Allergies",
                ["nuts", "dairy", "eggs", "soy", "shellfish", "fish"],
                key="new_allergies"
            )
            
            # Health conditions
            st.subheader("Health Information")
            health_conditions = st.multiselect(
                "Health Conditions",
                ["diabetes", "hypertension", "heart_disease", "kidney_disease"],
                key="new_health"
            )
            
            weight_goal = st.selectbox(
                "Weight Goal",
                ["lose", "maintain", "gain"],
                index=1,
                key="new_weight_goal"
            )
            
            # Preferences
            st.subheader("Cooking Preferences")
            col1, col2 = st.columns(2)
            with col1:
                cooking_skill = st.selectbox(
                    "Cooking Skill",
                    ["beginner", "intermediate", "advanced"],
                    key="new_skill"
                )
            with col2:
                cooking_time = st.selectbox(
                    "Preferred Cooking Time",
                    ["quick", "medium", "long"],
                    index=1,
                    key="new_time"
                )
            
            preferred_cuisines = st.multiselect(
                "Preferred Cuisines",
                ["american", "asian", "mediterranean", "italian", "mexican", "indian", "french"],
                key="new_cuisines"
            )
            
            submitted = st.form_submit_button("Create Profile")
            
            if submitted:
                if name:
                    # Create user profile
                    user_id = name.lower().replace(" ", "_") + "_profile"
                    
                    profile = UserProfile(
                        user_id=user_id,
                        name=name,
                        age=age,
                        gender=gender,
                        height_cm=height_cm,
                        weight_kg=weight_kg,
                        activity_level=activity_level,
                        dietary_restrictions=dietary_restrictions,
                        allergies=allergies,
                        health_conditions=health_conditions,
                        weight_goal=weight_goal,
                        cooking_skill_level=cooking_skill,
                        cooking_time_preference=cooking_time,
                        preferred_cuisines=preferred_cuisines
                    )
                    
                    if self.profile_manager.save_profile(profile):
                        st.session_state.user_profile = profile
                        st.success(f"Profile created for {name}!")
                        st.rerun()
                    else:
                        st.error("Failed to create profile. Please try again.")
                else:
                    st.error("Please enter your name.")
    
    def render_profile_display(self, profile: UserProfile):
        """Render user profile display."""
        st.subheader(f"Hello, {profile.name}! 👋")
        
        # Basic stats
        bmi = profile.calculate_bmi()
        if bmi:
            st.metric("BMI", f"{bmi}", f"{profile.get_bmi_category()}")
        
        if profile.calorie_goal:
            st.metric("Daily Calorie Goal", f"{profile.calorie_goal} cal")
        
        # Health summary
        if profile.health_conditions:
            st.write("🏥 **Health Conditions:**", ", ".join(profile.health_conditions))
        
        if profile.dietary_restrictions:
            st.write("🥗 **Dietary Restrictions:**", ", ".join(profile.dietary_restrictions))
        
        if profile.allergies:
            st.write("⚠️ **Allergies:**", ", ".join(profile.allergies))
    
    def render_profile_edit_form(self, profile: UserProfile):
        """Render profile editing form."""
        st.subheader("Edit Profile")
        # This would be similar to creation form but pre-populated
        # For brevity, showing a simplified version
        st.info("Profile editing feature coming soon!")
    
    def render_main_content(self):
        """Render the main content area."""
        if st.session_state.user_profile is None:
            # Create a temporary default profile for demo purposes
            from models.user_profile import UserProfile
            default_profile = UserProfile(
                user_id="demo_user",
                name="Demo User",
                age=30,
                gender="female",
                height_cm=165,
                weight_kg=65,
                activity_level="moderate",
                health_conditions=[],
                dietary_restrictions=[],
                allergies=[],
                nutrition_goals={
                    "daily_calories": 2000,
                    "protein_grams": 50,
                    "carb_grams": 250,
                    "fat_grams": 67
                }
            )
            st.session_state.user_profile = default_profile
            st.info("👈 Using demo profile. Please create your own profile for personalized recommendations.")
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Recipe Search", "📊 Nutrition Analysis", "📅 Meal Planning", "📈 Health Insights"])
        
        with tab1:
            self.render_recipe_search()
        
        with tab2:
            self.render_nutrition_analysis()
        
        with tab3:
            self.render_meal_planning()
        
        with tab4:
            self.render_health_insights()
    
    def render_recipe_search(self):
        """Render recipe search interface."""
        st.header("🔍 Find Your Perfect Recipe")
        
        # Search input
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "What would you like to cook today?",
                placeholder="e.g., healthy breakfast, low carb dinner, protein-rich snack",
                key="recipe_query"
            )
        
        with col2:
            max_results = st.selectbox("Results", [3, 5, 10], index=1, key="max_results")
        
        if st.button("Search Recipes", type="primary") or query:
            if query:
                with st.spinner("Finding personalized recipes for you..."):
                    try:
                        results = self.rag_system.process_user_query(
                            query=query,
                            user_profile=st.session_state.user_profile.to_dict(),
                            max_results=max_results
                        )
                        
                        # Store in history
                        st.session_state.query_history.append({
                            'query': query,
                            'results': results,
                            'timestamp': pd.Timestamp.now()
                        })
                        
                        self.display_search_results(results)
                        
                    except Exception as e:
                        st.error(f"Search failed: {str(e)}")
                        logger.error(f"Search error: {e}")
            else:
                st.warning("Please enter a search query.")
    
    def display_search_results(self, results: Dict[str, Any]):
        """Display search results."""
        if 'error' in results:
            st.error(f"Search error: {results['error']}")
            return
        
        # AI Summary
        if results.get('ai_summary'):
            st.info(f"💡 **AI Recommendation:** {results['ai_summary']}")
        
        # Results summary
        st.subheader(f"Found {len(results.get('recommended_recipes', []))} Personalized Recommendations")
        
        for i, recipe in enumerate(results.get('recommended_recipes', []), 1):
            self.render_recipe_card(recipe, i)
    
    def render_recipe_card(self, recipe: Dict[str, Any], index: int):
        """Render a single recipe card."""
        metadata = recipe['metadata']
        health_assessment = recipe['health_assessment']
        
        with st.expander(f"{index}. {metadata['name']} - {health_assessment['recommendation_level']}", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Recipe details
                st.write(f"**Description:** {metadata.get('description', 'No description available')}")
                st.write(f"**Cuisine:** {metadata.get('cuisine', 'Unknown').title()}")
                st.write(f"**Servings:** {metadata.get('servings', 1)}")
                
                cook_time = parse_cooking_time(metadata.get('cook_time', ''))
                prep_time = parse_cooking_time(metadata.get('prep_time', ''))
                total_time = cook_time + prep_time
                
                if total_time > 0:
                    st.write(f"**Total Time:** {format_cooking_time(total_time)}")
                
                # Ingredients
                st.write("**Ingredients:**")
                ingredients = metadata.get('ingredients', '').split('\n')
                for ingredient in ingredients[:5]:  # Show first 5 ingredients
                    if ingredient.strip():
                        st.write(f"• {ingredient.strip()}")
                if len(ingredients) > 5:
                    st.write(f"... and {len(ingredients) - 5} more ingredients")
            
            with col2:
                # Nutrition info
                st.write("**Nutrition (per serving):**")
                nutrition_data = {
                    'Calories': metadata.get('calories', 0),
                    'Protein': f"{metadata.get('protein', 0)}g",
                    'Carbs': f"{metadata.get('carbs', 0)}g",
                    'Fat': f"{metadata.get('fat', 0)}g",
                    'Fiber': f"{metadata.get('fiber', 0)}g"
                }
                
                for nutrient, value in nutrition_data.items():
                    st.metric(nutrient, value)
            
            # Health assessment
            self.render_health_assessment(health_assessment)
            
            # Substitution suggestions
            if recipe.get('substitution_suggestions', {}).get('substitutions'):
                self.render_substitution_suggestions(recipe['substitution_suggestions'])
    
    def render_health_assessment(self, assessment: Dict[str, Any]):
        """Render health assessment for a recipe."""
        st.subheader("🏥 Health Assessment")
        
        # Overall score
        score = assessment['overall_score']
        if score >= 0.8:
            score_color = "green"
        elif score >= 0.6:
            score_color = "orange"
        else:
            score_color = "red"
        
        st.markdown(f"**Overall Score:** <span style='color:{score_color}'>{score:.1f}/1.0</span>", unsafe_allow_html=True)
        st.write(f"**Recommendation:** {assessment['recommendation_level']}")
        
        # Warnings
        if assessment['safety_warnings']:
            st.warning("⚠️ **Safety Warnings:**")
            for warning in assessment['safety_warnings']:
                st.write(f"• {warning}")
        
        # Health recommendations
        if assessment['health_recommendations']:
            st.success("✅ **Health Benefits:**")
            for rec in assessment['health_recommendations'][:3]:  # Show top 3
                st.write(f"• {rec}")
        
        # Nutrition insights
        if assessment['nutrition_insights']:
            with st.expander("📊 Detailed Nutrition Insights"):
                for insight in assessment['nutrition_insights']:
                    st.write(f"• {insight}")
    
    def render_substitution_suggestions(self, suggestions: Dict[str, Any]):
        """Render ingredient substitution suggestions."""
        if not suggestions.get('substitutions'):
            return
        
        with st.expander("🔄 Suggested Ingredient Substitutions"):
            for sub in suggestions['substitutions'][:3]:  # Show top 3 substitutions
                st.write(f"**Replace:** {sub['original_ingredient']}")
                
                for alt in sub['alternatives'][:2]:  # Show top 2 alternatives
                    st.write(f"• **{alt['substitute']}** (ratio: {alt['ratio']}) - {alt['notes']}")
                st.write("---")
            
            if suggestions.get('benefits'):
                st.write("**Benefits of substitutions:**")
                for benefit in suggestions['benefits']:
                    st.write(f"✅ {benefit}")
    
    def render_nutrition_analysis(self):
        """Render nutrition analysis tab."""
        st.header("📊 Nutrition Analysis")
        
        if not st.session_state.query_history:
            st.info("Search for recipes first to see nutrition analysis.")
            return
        
        # Get latest search results
        latest_results = st.session_state.query_history[-1]['results']
        recipes = latest_results.get('recommended_recipes', [])
        
        if not recipes:
            st.info("No recipes to analyze.")
            return
        
        # Nutrition comparison chart
        st.subheader("Recipe Nutrition Comparison")
        
        nutrition_data = []
        for recipe in recipes:
            metadata = recipe['metadata']
            nutrition_data.append({
                'Recipe': metadata['name'][:20] + "..." if len(metadata['name']) > 20 else metadata['name'],
                'Calories': metadata.get('calories', 0),
                'Protein (g)': metadata.get('protein', 0),
                'Carbs (g)': metadata.get('carbs', 0),
                'Fat (g)': metadata.get('fat', 0),
                'Fiber (g)': metadata.get('fiber', 0)
            })
        
        df = pd.DataFrame(nutrition_data)
        
        # Create charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Calories comparison
            fig_calories = px.bar(df, x='Recipe', y='Calories', title="Calories per Serving")
            st.plotly_chart(fig_calories, use_container_width=True)
        
        with col2:
            # Macronutrient comparison
            macros_df = df[['Recipe', 'Protein (g)', 'Carbs (g)', 'Fat (g)']].set_index('Recipe')
            fig_macros = px.bar(macros_df, title="Macronutrients Comparison")
            st.plotly_chart(fig_macros, use_container_width=True)
        
        # Detailed nutrition table
        st.subheader("Detailed Nutrition Table")
        st.dataframe(df, use_container_width=True)
    
    def render_meal_planning(self):
        """Render meal planning tab."""
        st.header("📅 Meal Planning")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            planning_days = st.selectbox("Plan for how many days?", [1, 3, 5, 7], index=1)
        with col2:
            meals_per_day = st.selectbox("Meals per day?", [2, 3, 4], index=1)
        
        if st.button("Generate Meal Plan", type="primary"):
            with st.spinner("Creating your personalized meal plan..."):
                try:
                    meal_plan = self.rag_system.get_meal_plan_suggestions(
                        user_profile=st.session_state.user_profile.to_dict(),
                        days=planning_days,
                        meals_per_day=meals_per_day
                    )
                    
                    self.display_meal_plan(meal_plan)
                    
                except Exception as e:
                    st.error(f"Meal planning failed: {str(e)}")
                    logger.error(f"Meal planning error: {e}")
    
    def display_meal_plan(self, meal_plan: Dict[str, Any]):
        """Display generated meal plan."""
        st.subheader("Your Personalized Meal Plan")
        
        plan_data = meal_plan['meal_plan']
        nutrition_summary = meal_plan.get('nutrition_summary', {})
        
        # Display daily plans
        for day_key, day_meals in plan_data.items():
            day_num = day_key.replace('day_', '')
            
            with st.expander(f"Day {day_num}", expanded=True):
                cols = st.columns(len(day_meals))
                
                for i, (meal_type, meal_data) in enumerate(day_meals.items()):
                    with cols[i]:
                        st.write(f"**{meal_type.title()}**")
                        
                        if meal_data:
                            metadata = meal_data['metadata']
                            st.write(f"🍽️ {metadata['name']}")
                            st.write(f"⏱️ {metadata.get('prep_time', 'Unknown')} prep")
                            st.write(f"🔥 {metadata.get('calories', 0)} cal")
                            st.write(f"🥩 {metadata.get('protein', 0)}g protein")
                        else:
                            st.write("No suitable recipe found")
        
        # Nutrition summary
        if nutrition_summary.get('averages'):
            st.subheader("Daily Nutrition Averages")
            averages = nutrition_summary['averages']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Calories", f"{averages.get('avg_calories', 0):.0f}")
            with col2:
                st.metric("Avg Protein", f"{averages.get('avg_protein', 0):.0f}g")
            with col3:
                st.metric("Avg Carbs", f"{averages.get('avg_carbs', 0):.0f}g")
            with col4:
                st.metric("Avg Fat", f"{averages.get('avg_fat', 0):.0f}g")
    
    def render_health_insights(self):
        """Render health insights tab."""
        st.header("📈 Health Insights")
        
        profile = st.session_state.user_profile
        
        # Profile health summary
        st.subheader("Your Health Profile Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # BMI visualization
            bmi = profile.calculate_bmi()
            if bmi:
                bmi_categories = {
                    "Underweight": (0, 18.5, "blue"),
                    "Normal": (18.5, 25, "green"),
                    "Overweight": (25, 30, "orange"),
                    "Obese": (30, 40, "red")
                }
                
                fig_bmi = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = bmi,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "BMI"},
                    gauge = {
                        'axis': {'range': [None, 40]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 18.5], 'color': "lightblue"},
                            {'range': [18.5, 25], 'color': "lightgreen"},
                            {'range': [25, 30], 'color': "orange"},
                            {'range': [30, 40], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': bmi
                        }
                    }
                ))
                
                st.plotly_chart(fig_bmi, use_container_width=True)
        
        with col2:
            # Daily calorie goal vs recommended
            if profile.calorie_goal:
                st.metric("Daily Calorie Goal", f"{profile.calorie_goal} cal")
                
                # Calculate recommended calories
                recommended = profile.calculate_calorie_needs()
                difference = profile.calorie_goal - recommended
                
                st.metric(
                    "Recommended Calories", 
                    f"{recommended} cal",
                    f"{difference:+d} from goal"
                )
        
        # Health recommendations
        st.subheader("Personalized Health Recommendations")
        
        recommendations = []
        
        if "diabetes" in profile.health_conditions:
            recommendations.extend([
                "🍯 Focus on low-glycemic foods to manage blood sugar",
                "🥬 Increase fiber intake to help control glucose levels",
                "⏰ Consider smaller, more frequent meals"
            ])
        
        if "hypertension" in profile.health_conditions:
            recommendations.extend([
                "🧂 Limit sodium intake to less than 1500mg per day",
                "🍌 Include potassium-rich foods like bananas and leafy greens",
                "🥗 Follow the DASH diet principles"
            ])
        
        if "heart_disease" in profile.health_conditions:
            recommendations.extend([
                "🐟 Include omega-3 rich fish 2-3 times per week",
                "🥑 Choose healthy fats like avocados and olive oil",
                "🚫 Avoid trans fats and limit saturated fats"
            ])
        
        if profile.weight_goal == "lose":
            recommendations.extend([
                "⚖️ Create a moderate calorie deficit of 300-500 calories",
                "🏃 Combine diet with regular physical activity",
                "💧 Stay well hydrated throughout the day"
            ])
        
        if not recommendations:
            recommendations = [
                "🥗 Eat a variety of colorful fruits and vegetables",
                "🍗 Include lean proteins in every meal",
                "🌾 Choose whole grains over refined grains",
                "💧 Drink plenty of water throughout the day"
            ]
        
        for rec in recommendations:
            st.write(rec)
        
        # Progress tracking section
        st.subheader("Track Your Progress")
        st.info("🚧 Progress tracking features coming soon! You'll be able to log meals, track nutrition goals, and monitor health metrics.")
    
    def run(self):
        """Run the Streamlit application."""
        try:
            self.render_header()
            self.render_sidebar()
            self.render_main_content()
            
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            logger.error(f"App error: {e}")
            
            # Show debug information in development
            if st.checkbox("Show debug information"):
                st.exception(e)

def main():
    """Main function to run the Streamlit app."""
    app = NerveSparkApp()
    app.run()

if __name__ == "__main__":
    main()
