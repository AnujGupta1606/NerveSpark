"""
NerveSpark - Nutritional Recipe Assistant (Cloud Optimized)

Simple cloud-optimized version for demonstration purposes.
Uses sample data and minimal dependencies.
"""

import streamlit as st
import pandas as pd
import os
from typing import Dict, List, Any

# Simple cloud detection
IS_CLOUD_DEPLOYMENT = (
    os.getenv('STREAMLIT_SHARING_MODE') is not None or
    '/mount/src/' in os.getcwd() or
    'STREAMLIT_CLOUD' in os.environ
)

# Import appropriate modules based on environment
if IS_CLOUD_DEPLOYMENT:
    from src.simple_rag import SimpleNutritionalRAG
    RAG_SYSTEM = SimpleNutritionalRAG()
else:
    try:
        from src.rag_system import NutritionalRAG
        RAG_SYSTEM = NutritionalRAG()
    except ImportError:
        from src.simple_rag import SimpleNutritionalRAG
        RAG_SYSTEM = SimpleNutritionalRAG()

def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="NerveSpark - Nutritional Recipe Assistant",
        page_icon="🥗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .recipe-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #2E7D32;
    }
    .nutrition-card {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🥗 NerveSpark</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Your Intelligent Nutritional Recipe Assistant</p>', unsafe_allow_html=True)
    
    # Environment indicator
    env_type = "Cloud Demo" if IS_CLOUD_DEPLOYMENT else "Full Featured"
    st.info(f"🌟 Running in {env_type} mode")
    
    # Sidebar for user profile
    with st.sidebar:
        st.header("👤 User Profile")
        
        # Basic info
        age = st.number_input("Age", min_value=1, max_value=100, value=30)
        weight = st.number_input("Weight (kg)", min_value=1.0, max_value=200.0, value=70.0)
        height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)
        
        # Activity level
        activity_level = st.selectbox(
            "Activity Level",
            ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"]
        )
        
        # Health goals
        health_goals = st.multiselect(
            "Health Goals",
            ["Weight Loss", "Weight Gain", "Muscle Building", "General Health", "Athletic Performance"]
        )
        
        # Dietary restrictions
        dietary_restrictions = st.multiselect(
            "Dietary Restrictions",
            ["vegetarian", "vegan", "gluten-free", "dairy-free", "nut-free", "low-carb", "keto"]
        )
        
        # Health conditions
        health_conditions = st.multiselect(
            "Health Conditions",
            ["diabetes", "hypertension", "heart_disease", "high_cholesterol", "none"]
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Recipe Search")
        
        # Search input
        query = st.text_input(
            "What would you like to cook today?",
            placeholder="e.g., chicken salad, pasta, healthy breakfast...",
            help="Enter ingredients, dish names, or dietary preferences"
        )
        
        # Search button
        if st.button("🥘 Get Recommendations", type="primary") and query:
            with st.spinner("Finding perfect recipes for you..."):
                # Prepare user profile
                user_profile = {
                    'age': age,
                    'weight': weight,
                    'height': height,
                    'activity_level': activity_level,
                    'health_goals': health_goals,
                    'dietary_restrictions': dietary_restrictions,
                    'health_conditions': health_conditions
                }
                
                # Get recommendations
                try:
                    result = RAG_SYSTEM.generate_recommendation(query, user_profile)
                    
                    # Display results
                    st.success(f"Found {len(result['recipes'])} recipes for: **{query}**")
                    
                    # Display recipes
                    for i, recipe in enumerate(result['recipes'], 1):
                        with st.container():
                            st.markdown(f'<div class="recipe-card">', unsafe_allow_html=True)
                            
                            col_a, col_b = st.columns([3, 1])
                            
                            with col_a:
                                st.subheader(f"🍽️ {recipe['name']}")
                                st.write(f"**Cuisine:** {recipe.get('cuisine', 'International')}")
                                st.write(f"**Cooking Time:** {recipe.get('cooking_time', 'N/A')}")
                                st.write(f"**Difficulty:** {recipe.get('difficulty', 'Medium')}")
                                
                                # Ingredients
                                ingredients_text = ", ".join(recipe.get('ingredients', []))
                                st.write(f"**Ingredients:** {ingredients_text}")
                                
                                # Instructions
                                st.write(f"**Instructions:** {recipe.get('instructions', 'Not available')}")
                                
                                # Dietary tags
                                if recipe.get('dietary_tags'):
                                    tags_text = " | ".join([f"🏷️ {tag}" for tag in recipe['dietary_tags']])
                                    st.write(tags_text)
                            
                            with col_b:
                                st.markdown('<div class="nutrition-card">', unsafe_allow_html=True)
                                st.write("**Nutrition per serving:**")
                                st.metric("Calories", f"{recipe.get('calories', 0)}")
                                st.metric("Protein", f"{recipe.get('protein', 0)}g")
                                st.metric("Carbs", f"{recipe.get('carbs', 0)}g")
                                st.metric("Fat", f"{recipe.get('fat', 0)}g")
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.markdown("---")
                    
                    # Health recommendations
                    if result.get('recommendations'):
                        st.header("💡 Health Recommendations")
                        for rec in result['recommendations']:
                            st.info(f"🎯 {rec}")
                    
                    # Nutrition summary
                    if result.get('nutrition_summary'):
                        st.header("📊 Nutrition Summary")
                        summary = result['nutrition_summary']
                        
                        col_a, col_b, col_c, col_d = st.columns(4)
                        with col_a:
                            st.metric("Avg Calories", summary.get('avg_calories', 0))
                        with col_b:
                            st.metric("Avg Protein", f"{summary.get('avg_protein', 0)}g")
                        with col_c:
                            st.metric("Avg Carbs", f"{summary.get('avg_carbs', 0)}g")
                        with col_d:
                            st.metric("Avg Fat", f"{summary.get('avg_fat', 0)}g")
                
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")
                    st.info("Please try a different search term or check your connection.")
    
    with col2:
        st.header("📈 Quick Stats")
        
        # Calculate BMI
        if height > 0:
            bmi = weight / ((height/100) ** 2)
            st.metric("BMI", f"{bmi:.1f}")
            
            if bmi < 18.5:
                st.warning("Underweight")
            elif bmi < 25:
                st.success("Normal weight")
            elif bmi < 30:
                st.warning("Overweight")
            else:
                st.error("Obese")
        
        # Daily calorie estimate
        if age and weight and height:
            # Simple BMR calculation (Mifflin-St Jeor)
            bmr = 10 * weight + 6.25 * height - 5 * age + 5  # for males (simplified)
            activity_multiplier = {
                "Sedentary": 1.2,
                "Lightly Active": 1.375,
                "Moderately Active": 1.55,
                "Very Active": 1.725,
                "Extremely Active": 1.9
            }
            daily_calories = int(bmr * activity_multiplier.get(activity_level, 1.2))
            st.metric("Estimated Daily Calories", daily_calories)
        
        # Sample recipes quick access
        st.header("🎲 Quick Recipe Ideas")
        sample_queries = [
            "healthy breakfast",
            "low carb dinner",
            "vegetarian lunch",
            "high protein snack",
            "quick 15-minute meal"
        ]
        
        for sample_query in sample_queries:
            if st.button(f"🔍 {sample_query.title()}", key=f"sample_{sample_query}"):
                st.experimental_set_query_params(query=sample_query)
                st.experimental_rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p>🥗 <strong>NerveSpark</strong> - Your Intelligent Nutritional Recipe Assistant</p>
            <p>Powered by RAG (Retrieval-Augmented Generation) for personalized nutrition recommendations</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
