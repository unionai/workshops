"""Spoonacular API wrapper for recipe tools."""

import os
from typing import Any

import httpx


class SpoonacularClient:
    """Client for interacting with the Spoonacular Food API."""

    BASE_URL = "https://api.spoonacular.com"

    def __init__(self, api_key: str | None = None):
        """Initialize the Spoonacular client.
        
        Args:
            api_key: Spoonacular API key. If not provided, reads from environment.
        """
        self.api_key = api_key or os.getenv("SPOONACULAR_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Set SPOONACULAR_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.client = httpx.Client(timeout=30.0)

    def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make a request to the Spoonacular API.
        
        Args:
            endpoint: API endpoint path.
            params: Query parameters.
            
        Returns:
            JSON response data.
        """
        params = params or {}
        params["apiKey"] = self.api_key
        
        url = f"{self.BASE_URL}{endpoint}"
        response = self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def search_recipes(
        self,
        query: str = "",
        cuisine: str | None = None,
        diet: str | None = None,
        intolerances: str | None = None,
        type: str | None = None,
        max_ready_time: int | None = None,
        number: int = 10,
    ) -> list[dict[str, Any]]:
        """Search for recipes with various filters.
        
        Args:
            query: Natural language search query (e.g., "pasta", "chicken soup").
            cuisine: Filter by cuisine (e.g., "italian", "mexican", "asian").
            diet: Filter by diet (e.g., "vegan", "vegetarian", "gluten free", "keto").
            intolerances: Filter out recipes with intolerances (e.g., "dairy", "gluten").
            type: Filter by meal type (e.g., "main course", "dessert", "breakfast").
            max_ready_time: Maximum preparation time in minutes.
            number: Number of results to return (1-100).
            
        Returns:
            List of recipe summaries with id, title, and image.
        """
        params = {"number": min(number, 100)}
        
        if query:
            params["query"] = query
        if cuisine:
            params["cuisine"] = cuisine
        if diet:
            params["diet"] = diet
        if intolerances:
            params["intolerances"] = intolerances
        if type:
            params["type"] = type
        if max_ready_time:
            params["maxReadyTime"] = max_ready_time
        
        result = self._request("/recipes/complexSearch", params)
        return result.get("results", [])

    def search_by_ingredients(
        self,
        ingredients: list[str],
        number: int = 10,
        ranking: int = 1,
        ignore_pantry: bool = True,
    ) -> list[dict[str, Any]]:
        """Find recipes that use given ingredients.
        
        This is the "what's in my fridge" endpoint - finds recipes that maximize
        the use of ingredients you have and minimize what you need to buy.
        
        Args:
            ingredients: List of ingredients you have (e.g., ["chicken", "rice", "broccoli"]).
            number: Number of results to return (1-100).
            ranking: 1 = maximize used ingredients, 2 = minimize missing ingredients.
            ignore_pantry: Whether to ignore common pantry items (salt, water, etc.).
            
        Returns:
            List of recipes with used/missing ingredient details.
        """
        params = {
            "ingredients": ",".join(ingredients),
            "number": min(number, 100),
            "ranking": ranking,
            "ignorePantry": ignore_pantry,
        }
        
        return self._request("/recipes/findByIngredients", params)

    def search_by_nutrients(
        self,
        min_calories: int | None = None,
        max_calories: int | None = None,
        min_protein: int | None = None,
        max_protein: int | None = None,
        min_carbs: int | None = None,
        max_carbs: int | None = None,
        min_fat: int | None = None,
        max_fat: int | None = None,
        number: int = 10,
    ) -> list[dict[str, Any]]:
        """Find recipes by nutritional requirements.
        
        Args:
            min_calories: Minimum calories per serving.
            max_calories: Maximum calories per serving.
            min_protein: Minimum protein in grams.
            max_protein: Maximum protein in grams.
            min_carbs: Minimum carbohydrates in grams.
            max_carbs: Maximum carbohydrates in grams.
            min_fat: Minimum fat in grams.
            max_fat: Maximum fat in grams.
            number: Number of results to return (1-100).
            
        Returns:
            List of recipes matching nutritional criteria.
        """
        params = {"number": min(number, 100)}
        
        if min_calories is not None:
            params["minCalories"] = min_calories
        if max_calories is not None:
            params["maxCalories"] = max_calories
        if min_protein is not None:
            params["minProtein"] = min_protein
        if max_protein is not None:
            params["maxProtein"] = max_protein
        if min_carbs is not None:
            params["minCarbs"] = min_carbs
        if max_carbs is not None:
            params["maxCarbs"] = max_carbs
        if min_fat is not None:
            params["minFat"] = min_fat
        if max_fat is not None:
            params["maxFat"] = max_fat
        
        return self._request("/recipes/findByNutrients", params)

    def get_recipe_info(self, recipe_id: int, include_nutrition: bool = True) -> dict[str, Any]:
        """Get detailed information about a specific recipe.
        
        Args:
            recipe_id: The Spoonacular recipe ID.
            include_nutrition: Whether to include nutritional information.
            
        Returns:
            Detailed recipe information including ingredients, instructions, and nutrition.
        """
        params = {"includeNutrition": include_nutrition}
        return self._request(f"/recipes/{recipe_id}/information", params)

    def get_similar_recipes(self, recipe_id: int, number: int = 5) -> list[dict[str, Any]]:
        """Find recipes similar to a given recipe.
        
        Args:
            recipe_id: The Spoonacular recipe ID to find similar recipes for.
            number: Number of similar recipes to return.
            
        Returns:
            List of similar recipes.
        """
        params = {"number": min(number, 100)}
        return self._request(f"/recipes/{recipe_id}/similar", params)

    def autocomplete_recipe(self, query: str, number: int = 10) -> list[dict[str, Any]]:
        """Get recipe name suggestions based on a partial query.
        
        Args:
            query: Partial recipe name (e.g., "chick" for chicken recipes).
            number: Number of suggestions to return.
            
        Returns:
            List of recipe name suggestions with IDs.
        """
        params = {
            "query": query,
            "number": min(number, 100),
        }
        return self._request("/recipes/autocomplete", params)

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
