from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """User profile data structure for nutrition preferences and health information."""
    
    user_id: str
    name: str = ""
    age: Optional[int] = None
    gender: str = "female"  # "male", "female", "other"
    
    # Health Information
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    activity_level: str = "moderate"  # "sedentary", "light", "moderate", "active", "very_active"
    
    # Dietary Restrictions and Preferences
    dietary_restrictions: List[str] = field(default_factory=list)  # ["vegan", "gluten_free", etc.]
    allergies: List[str] = field(default_factory=list)  # ["nuts", "dairy", "shellfish", etc.]
    food_preferences: List[str] = field(default_factory=list)  # ["spicy", "sweet", "savory"]
    disliked_foods: List[str] = field(default_factory=list)
    
    # Health Conditions
    health_conditions: List[str] = field(default_factory=list)  # ["diabetes", "hypertension", etc.]
    medications: List[str] = field(default_factory=list)
    
    # Nutritional Goals
    nutrition_goals: Dict[str, float] = field(default_factory=dict)
    calorie_goal: Optional[int] = None
    weight_goal: str = "maintain"  # "lose", "gain", "maintain"
    
    # Preferences
    preferred_cuisines: List[str] = field(default_factory=list)
    cooking_time_preference: str = "medium"  # "quick", "medium", "long"
    cooking_skill_level: str = "beginner"  # "beginner", "intermediate", "advanced"
    
    # Meal Planning
    meals_per_day: int = 3
    snacks_per_day: int = 1
    meal_prep_days: int = 1
    
    def __post_init__(self):
        """Calculate derived properties after initialization."""
        if self.calorie_goal is None and self.height_cm and self.weight_kg and self.age:
            self.calorie_goal = self.calculate_calorie_needs()
        
        if not self.nutrition_goals:
            self.nutrition_goals = self.get_default_nutrition_goals()
    
    def calculate_calorie_needs(self) -> int:
        """Calculate daily calorie needs using Mifflin-St Jeor equation."""
        if not all([self.height_cm, self.weight_kg, self.age]):
            return 2000  # Default value
        
        # Base Metabolic Rate (BMR)
        if self.gender.lower() == "male":
            bmr = 10 * self.weight_kg + 6.25 * self.height_cm - 5 * self.age + 5
        else:
            bmr = 10 * self.weight_kg + 6.25 * self.height_cm - 5 * self.age - 161
        
        # Activity multipliers
        activity_multipliers = {
            "sedentary": 1.2,
            "light": 1.375,
            "moderate": 1.55,
            "active": 1.725,
            "very_active": 1.9
        }
        
        multiplier = activity_multipliers.get(self.activity_level, 1.55)
        daily_calories = int(bmr * multiplier)
        
        # Adjust for weight goals
        if self.weight_goal == "lose":
            daily_calories -= 500  # Deficit for 1 lb/week loss
        elif self.weight_goal == "gain":
            daily_calories += 500  # Surplus for 1 lb/week gain
        
        return daily_calories
    
    def get_default_nutrition_goals(self) -> Dict[str, float]:
        """Get default nutrition goals based on user profile."""
        calories = self.calorie_goal or 2000
        
        # Standard macronutrient ratios
        protein_calories = calories * 0.15  # 15% protein
        carb_calories = calories * 0.55     # 55% carbs
        fat_calories = calories * 0.30      # 30% fat
        
        return {
            "calories": calories,
            "protein": protein_calories / 4,  # 4 calories per gram
            "carbs": carb_calories / 4,
            "fat": fat_calories / 9,  # 9 calories per gram
            "fiber": 25 if self.gender == "female" else 38,
            "sodium": 2300,  # mg
            "sugar": 50  # grams
        }
    
    def calculate_bmi(self) -> Optional[float]:
        """Calculate Body Mass Index."""
        if self.height_cm and self.weight_kg:
            height_m = self.height_cm / 100
            return round(self.weight_kg / (height_m ** 2), 1)
        return None
    
    def get_bmi_category(self) -> str:
        """Get BMI category."""
        bmi = self.calculate_bmi()
        if bmi is None:
            return "Unknown"
        
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal weight"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"
    
    def has_dietary_restriction(self, restriction: str) -> bool:
        """Check if user has a specific dietary restriction."""
        return restriction.lower() in [r.lower() for r in self.dietary_restrictions]
    
    def has_allergy(self, allergen: str) -> bool:
        """Check if user has a specific allergy."""
        return allergen.lower() in [a.lower() for a in self.allergies]
    
    def has_health_condition(self, condition: str) -> bool:
        """Check if user has a specific health condition."""
        return condition.lower() in [c.lower() for c in self.health_conditions]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user profile to dictionary."""
        return {
            'user_id': self.user_id,
            'name': self.name,
            'age': self.age,
            'gender': self.gender,
            'height_cm': self.height_cm,
            'weight_kg': self.weight_kg,
            'activity_level': self.activity_level,
            'dietary_restrictions': self.dietary_restrictions,
            'allergies': self.allergies,
            'food_preferences': self.food_preferences,
            'disliked_foods': self.disliked_foods,
            'health_conditions': self.health_conditions,
            'medications': self.medications,
            'nutrition_goals': self.nutrition_goals,
            'calorie_goal': self.calorie_goal,
            'weight_goal': self.weight_goal,
            'preferred_cuisines': self.preferred_cuisines,
            'cooking_time_preference': self.cooking_time_preference,
            'cooking_skill_level': self.cooking_skill_level,
            'meals_per_day': self.meals_per_day,
            'snacks_per_day': self.snacks_per_day,
            'meal_prep_days': self.meal_prep_days
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """Create UserProfile from dictionary."""
        return cls(**data)
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the user profile."""
        summary_parts = [f"Profile for {self.name or 'User'}"]
        
        if self.age:
            summary_parts.append(f"Age: {self.age}")
        
        bmi = self.calculate_bmi()
        if bmi:
            summary_parts.append(f"BMI: {bmi} ({self.get_bmi_category()})")
        
        if self.dietary_restrictions:
            summary_parts.append(f"Dietary: {', '.join(self.dietary_restrictions)}")
        
        if self.allergies:
            summary_parts.append(f"Allergies: {', '.join(self.allergies)}")
        
        if self.health_conditions:
            summary_parts.append(f"Health: {', '.join(self.health_conditions)}")
        
        if self.calorie_goal:
            summary_parts.append(f"Daily calories: {self.calorie_goal}")
        
        return " | ".join(summary_parts)


class UserProfileManager:
    """Manages user profiles with SQLite database storage."""
    
    def __init__(self, db_path: str = "user_profiles.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the user profiles database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        user_id TEXT PRIMARY KEY,
                        profile_data TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        query TEXT NOT NULL,
                        recommendations TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
                    )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def save_profile(self, profile: UserProfile) -> bool:
        """Save or update a user profile."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                profile_json = json.dumps(profile.to_dict())
                
                cursor.execute('''
                    INSERT OR REPLACE INTO user_profiles (user_id, profile_data, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                ''', (profile.user_id, profile_json))
                
                conn.commit()
                logger.info(f"Profile saved for user {profile.user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving profile: {e}")
            return False
    
    def load_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load a user profile by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT profile_data FROM user_profiles WHERE user_id = ?',
                    (user_id,)
                )
                
                row = cursor.fetchone()
                if row:
                    profile_data = json.loads(row[0])
                    return UserProfile.from_dict(profile_data)
                
                return None
                
        except Exception as e:
            logger.error(f"Error loading profile: {e}")
            return None
    
    def delete_profile(self, user_id: str) -> bool:
        """Delete a user profile."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM user_profiles WHERE user_id = ?', (user_id,))
                cursor.execute('DELETE FROM user_history WHERE user_id = ?', (user_id,))
                
                conn.commit()
                logger.info(f"Profile deleted for user {user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting profile: {e}")
            return False
    
    def list_profiles(self) -> List[Dict[str, Any]]:
        """List all user profiles with basic info."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT user_id, profile_data, created_at, updated_at 
                    FROM user_profiles 
                    ORDER BY updated_at DESC
                ''')
                
                profiles = []
                for row in cursor.fetchall():
                    profile_data = json.loads(row[1])
                    profiles.append({
                        'user_id': row[0],
                        'name': profile_data.get('name', 'Unknown'),
                        'created_at': row[2],
                        'updated_at': row[3]
                    })
                
                return profiles
                
        except Exception as e:
            logger.error(f"Error listing profiles: {e}")
            return []
    
    def save_user_query(self, user_id: str, query: str, recommendations: Dict[str, Any]) -> bool:
        """Save user query and recommendations to history."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                recommendations_json = json.dumps(recommendations)
                
                cursor.execute('''
                    INSERT INTO user_history (user_id, query, recommendations)
                    VALUES (?, ?, ?)
                ''', (user_id, query, recommendations_json))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error saving user query: {e}")
            return False
    
    def get_user_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user query history."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT query, recommendations, timestamp 
                    FROM user_history 
                    WHERE user_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (user_id, limit))
                
                history = []
                for row in cursor.fetchall():
                    history.append({
                        'query': row[0],
                        'recommendations': json.loads(row[1]) if row[1] else {},
                        'timestamp': row[2]
                    })
                
                return history
                
        except Exception as e:
            logger.error(f"Error getting user history: {e}")
            return []
    
    def create_sample_profile(self, user_id: str = "sample_user") -> UserProfile:
        """Create a sample user profile for testing."""
        sample_profile = UserProfile(
            user_id=user_id,
            name="Jane Doe",
            age=32,
            gender="female",
            height_cm=165,
            weight_kg=65,
            activity_level="moderate",
            dietary_restrictions=["vegetarian"],
            allergies=["nuts"],
            health_conditions=["diabetes"],
            weight_goal="maintain",
            preferred_cuisines=["mediterranean", "asian"],
            cooking_skill_level="intermediate"
        )
        
        # Save the sample profile
        self.save_profile(sample_profile)
        return sample_profile
