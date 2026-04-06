import requests
import logging
from typing import Optional, Dict
import os

logger = logging.getLogger(__name__)

class NutritionCalculator:
    """
    Nutrition calculation engine for Indian food
    Data sources: USDA FoodData Central, OpenFoodFacts, Indian Food Composition Tables
    """
    
    def __init__(self, db_session=None):
        self.db = db_session
        self.usda_api_key = os.getenv("USDA_API_KEY")
        
        # Comprehensive nutrition database (per 100g)
        # Sources: USDA FoodData Central, Tarladalal, NutriScan, CalorieKing, IFCT
        # Covers all 40 model classes exactly
        self.indian_nutrition_db = {
            # ── Indian classes ──────────────────────────────────────────
            "ariselu":               {"calories": 340, "carbs": 45, "protein": 4,  "fat": 9,  "fiber": 0.8, "sodium": 20},
            "biryani":               {"calories": 185, "carbs": 28, "protein": 8,  "fat": 5,  "fiber": 1.5, "sodium": 350},
            "butter_chicken":        {"calories": 235, "carbs": 8,  "protein": 18, "fat": 15, "fiber": 1.5, "sodium": 500},
            "chapati":               {"calories": 297, "carbs": 51, "protein": 11, "fat": 7,  "fiber": 3.7, "sodium": 486},
            "dosa":                  {"calories": 168, "carbs": 28, "protein": 4,  "fat": 4,  "fiber": 1.5, "sodium": 200},
            "gulab_jamun":           {"calories": 315, "carbs": 50, "protein": 5,  "fat": 10, "fiber": 0.5, "sodium": 50},
            "idli":                  {"calories": 156, "carbs": 30, "protein": 5,  "fat": 1,  "fiber": 1.2, "sodium": 150},
            "jalebi":                {"calories": 450, "carbs": 70, "protein": 3,  "fat": 18, "fiber": 0.3, "sodium": 40},
            "lassi":                 {"calories": 98,  "carbs": 12, "protein": 4,  "fat": 4,  "fiber": 0,   "sodium": 60},
            "naan":                  {"calories": 262, "carbs": 45, "protein": 8,  "fat": 5,  "fiber": 2,   "sodium": 430},
            "palak_paneer":          {"calories": 180, "carbs": 8,  "protein": 12, "fat": 12, "fiber": 3,   "sodium": 400},
            "paneer_butter_masala":  {"calories": 220, "carbs": 10, "protein": 14, "fat": 15, "fiber": 2,   "sodium": 450},
            "pani_puri":             {"calories": 299, "carbs": 40, "protein": 5,  "fat": 13, "fiber": 2,   "sodium": 420},
            "pav_bhaji":             {"calories": 160, "carbs": 23, "protein": 4,  "fat": 6,  "fiber": 3,   "sodium": 514},
            "poori":                 {"calories": 375, "carbs": 45, "protein": 7,  "fat": 18, "fiber": 2.5, "sodium": 400},
            "rasgulla":              {"calories": 186, "carbs": 40, "protein": 4,  "fat": 1,  "fiber": 0,   "sodium": 30},
            "rasmalai":              {"calories": 265, "carbs": 34, "protein": 9,  "fat": 11, "fiber": 0,   "sodium": 80},
            "samosa":                {"calories": 262, "carbs": 24, "protein": 5,  "fat": 17, "fiber": 2.5, "sodium": 420},
            "vada_pav":              {"calories": 290, "carbs": 42, "protein": 7,  "fat": 10, "fiber": 3,   "sodium": 480},

            # ── Global / Food-101 classes ────────────────────────────────
            "caesar_salad":          {"calories": 158, "carbs": 8,  "protein": 4,  "fat": 13, "fiber": 2,   "sodium": 360},
            "cheesecake":            {"calories": 321, "carbs": 26, "protein": 6,  "fat": 22, "fiber": 0.4, "sodium": 220},
            "chocolate_cake":        {"calories": 371, "carbs": 50, "protein": 5,  "fat": 17, "fiber": 2,   "sodium": 300},
            "donuts":                {"calories": 452, "carbs": 51, "protein": 7,  "fat": 25, "fiber": 1.5, "sodium": 370},
            "french_fries":          {"calories": 312, "carbs": 41, "protein": 3,  "fat": 15, "fiber": 3.8, "sodium": 210},
            "fried_rice":            {"calories": 163, "carbs": 28, "protein": 4,  "fat": 4,  "fiber": 1,   "sodium": 450},
            "grilled_salmon":        {"calories": 208, "carbs": 0,  "protein": 28, "fat": 10, "fiber": 0,   "sodium": 75},
            "hamburger":             {"calories": 295, "carbs": 24, "protein": 17, "fat": 14, "fiber": 1.3, "sodium": 520},
            "hot_dog":               {"calories": 290, "carbs": 22, "protein": 11, "fat": 17, "fiber": 1,   "sodium": 670},
            "ice_cream":             {"calories": 207, "carbs": 24, "protein": 3,  "fat": 11, "fiber": 0.7, "sodium": 80},
            "macaroni_and_cheese":   {"calories": 164, "carbs": 22, "protein": 7,  "fat": 5,  "fiber": 1,   "sodium": 470},
            "onion_rings":           {"calories": 411, "carbs": 40, "protein": 5,  "fat": 26, "fiber": 2,   "sodium": 430},
            "pancakes":              {"calories": 227, "carbs": 28, "protein": 6,  "fat": 10, "fiber": 1,   "sodium": 430},
            "pizza":                 {"calories": 266, "carbs": 33, "protein": 11, "fat": 10, "fiber": 2.3, "sodium": 598},
            "popcorn":               {"calories": 375, "carbs": 74, "protein": 11, "fat": 4,  "fiber": 14,  "sodium": 7},
            "ramen":                 {"calories": 188, "carbs": 27, "protein": 8,  "fat": 5,  "fiber": 1,   "sodium": 890},
            "spaghetti_bolognese":   {"calories": 180, "carbs": 22, "protein": 10, "fat": 6,  "fiber": 2,   "sodium": 380},
            "steak":                 {"calories": 271, "carbs": 0,  "protein": 26, "fat": 18, "fiber": 0,   "sodium": 60},
            "sushi":                 {"calories": 143, "carbs": 28, "protein": 5,  "fat": 1,  "fiber": 1,   "sodium": 430},
            "tacos":                 {"calories": 226, "carbs": 20, "protein": 9,  "fat": 12, "fiber": 3,   "sodium": 400},
            "waffles":               {"calories": 291, "carbs": 37, "protein": 8,  "fat": 13, "fiber": 1.5, "sodium": 490},

            # ── Aliases (for partial-match fallback) ─────────────────────
            "roti":                  {"calories": 297, "carbs": 51, "protein": 11, "fat": 7,  "fiber": 3.7, "sodium": 486},
            "paratha":               {"calories": 320, "carbs": 42, "protein": 7,  "fat": 14, "fiber": 3,   "sodium": 500},
            "dal":                   {"calories": 116, "carbs": 20, "protein": 9,  "fat": 0.4,"fiber": 7.9, "sodium": 2},
            "rice":                  {"calories": 130, "carbs": 28, "protein": 2.7,"fat": 0.3,"fiber": 0.4, "sodium": 1},
            "donut":                 {"calories": 452, "carbs": 51, "protein": 7,  "fat": 25, "fiber": 1.5, "sodium": 370},
        }
    
    def calculate_nutrition(self, food_name: str, weight_grams: float) -> Dict:
        """
        Calculate nutritional values based on food weight
        Formula: nutrient_value = (weight / 100) × nutrient_per_100g
        
        Args:
            food_name: Name of Indian food item
            weight_grams: Weight in grams
            
        Returns:
            Dictionary with calculated nutritional values
        """
        food_key = food_name.lower().strip().replace(' ', '_')
        
        # Try exact match first
        base_nutrition = self.indian_nutrition_db.get(food_key)
        
        # Try partial match if exact match fails
        if not base_nutrition:
            for key in self.indian_nutrition_db.keys():
                if food_key in key or key in food_key:
                    base_nutrition = self.indian_nutrition_db[key]
                    logger.info(f"Matched '{food_name}' to '{key}'")
                    break
        
        # Try USDA API if not found in local database
        if not base_nutrition and self.usda_api_key:
            base_nutrition = self._fetch_from_usda(food_name)
        
        # Default fallback
        if not base_nutrition:
            logger.warning(f"Food '{food_name}' not found in database, using defaults")
            base_nutrition = {
                "calories": 150,
                "carbs": 25,
                "protein": 6,
                "fat": 4,
                "fiber": 2,
                "sodium": 200
            }
        
        # Calculate based on weight
        factor = weight_grams / 100
        
        return {
            "calories": round(base_nutrition["calories"] * factor, 1),
            "carbohydrates": round(base_nutrition["carbs"] * factor, 1),
            "protein": round(base_nutrition["protein"] * factor, 1),
            "fat": round(base_nutrition["fat"] * factor, 1),
            "fiber": round(base_nutrition.get("fiber", 0) * factor, 1),
            "sodium": round(base_nutrition.get("sodium", 0) * factor, 1),
            "weight_grams": weight_grams,
            "food_name": food_name
        }
    
    def _fetch_from_usda(self, food_name: str) -> Optional[Dict]:
        """
        Fetch nutrition data from USDA FoodData Central API
        
        Args:
            food_name: Name of food item
            
        Returns:
            Nutrition data dictionary or None
        """
        try:
            if not self.usda_api_key:
                return None
            
            # Search for food
            search_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
            params = {
                "api_key": self.usda_api_key,
                "query": food_name,
                "pageSize": 1
            }
            
            response = requests.get(search_url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("foods"):
                    food = data["foods"][0]
                    nutrients = {n["nutrientName"]: n["value"] 
                               for n in food.get("foodNutrients", [])}
                    
                    return {
                        "calories": nutrients.get("Energy", 150),
                        "carbs": nutrients.get("Carbohydrate, by difference", 25),
                        "protein": nutrients.get("Protein", 6),
                        "fat": nutrients.get("Total lipid (fat)", 4),
                        "fiber": nutrients.get("Fiber, total dietary", 2),
                        "sodium": nutrients.get("Sodium, Na", 200)
                    }
        except Exception as e:
            logger.error(f"USDA API error: {e}")
        
        return None
    
    def get_food_info(self, food_name: str) -> Optional[Dict]:
        """
        Get complete food information from database
        
        Args:
            food_name: Name of food item
            
        Returns:
            Complete nutrition data per 100g
        """
        food_key = food_name.lower().strip().replace(' ', '_')
        return self.indian_nutrition_db.get(food_key)
    
    def list_available_foods(self) -> list:
        """Get list of all available Indian foods in database"""
        return sorted(list(self.indian_nutrition_db.keys()))
