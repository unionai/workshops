"""Recipe MCP Server.

This server provides tools for AI agents to search and discover recipes
using the Spoonacular Food API.
"""

import os
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from tools.recipes import SpoonacularClient

# Initialize the MCP server
mcp = FastMCP(
    name="recipe-assistant",
    instructions="""
    You are a helpful recipe assistant that can search for and recommend recipes.
    
    You can help users:
    - Find recipes by name or description ("pasta carbonara", "quick dinner")
    - Discover recipes using ingredients they have on hand
    - Find recipes that match dietary requirements (vegan, keto, gluten-free)
    - Search for recipes by nutritional goals (high protein, low carb)
    - Get detailed recipe information including ingredients and instructions
    - Find similar recipes to ones they already like
    
    When presenting recipes:
    - Always include the recipe title and a brief description
    - Mention key details like prep time, servings, and dietary info
    - For detailed recipes, list ingredients and summarize instructions
    - Suggest related recipes when appropriate
    """,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
    stateless_http=True,
    json_response=True,

)

# Lazy initialization of the client
_client: SpoonacularClient | None = None


def get_client() -> SpoonacularClient:
    """Get or create the Spoonacular client."""
    load_dotenv()
    global _client
    if _client is None:
        api_key = os.getenv("SPOONACULAR_API_KEY")
        if not api_key:
            raise ValueError(
                "SPOONACULAR_API_KEY environment variable not set. "
                "Get a free API key at https://spoonacular.com/food-api"
            )
        _client = SpoonacularClient(api_key=api_key)
    return _client


# =============================================================================
# MCP Tools
# =============================================================================

@mcp.tool()
async def search_recipes(
    query: str = "",
    cuisine: str | None = None,
    diet: str | None = None,
    intolerances: str | None = None,
    meal_type: str | None = None,
    max_ready_time: int | None = None,
    number: int = 10,
) -> list[dict[str, Any]]:
    """Search for recipes with various filters.
    
    Args:
        query: What to search for (e.g., "pasta", "chicken soup", "healthy dinner").
        cuisine: Filter by cuisine - italian, mexican, asian, american, mediterranean, etc.
        diet: Filter by diet - vegan, vegetarian, gluten free, ketogenic, paleo, etc.
        intolerances: Exclude allergens - dairy, egg, gluten, peanut, seafood, etc.
        meal_type: Filter by type - main course, dessert, breakfast, snack, appetizer, etc.
        max_ready_time: Maximum total time in minutes (prep + cooking).
        number: How many recipes to return (default 10, max 100).
        
    Returns:
        List of recipes with id, title, and image URL.
    """
    client = get_client()
    recipes = client.search_recipes(
        query=query,
        cuisine=cuisine,
        diet=diet,
        intolerances=intolerances,
        type=meal_type,
        max_ready_time=max_ready_time,
        number=number,
    )
    return recipes


@mcp.tool()
async def search_by_ingredients(
    ingredients: list[str],
    number: int = 10,
    maximize_used: bool = True,
) -> list[dict[str, Any]]:
    """Find recipes using ingredients you have on hand.
    
    This is the "what's in my fridge" tool - it finds recipes that use
    the ingredients you have and tells you what else you might need.
    
    Args:
        ingredients: List of ingredients you have (e.g., ["chicken", "rice", "garlic"]).
        number: How many recipes to return (default 10).
        maximize_used: If True, prioritize recipes using more of your ingredients.
                      If False, prioritize recipes needing fewer additional ingredients.
        
    Returns:
        List of recipes showing which ingredients are used and which are missing.
    """
    client = get_client()
    recipes = client.search_by_ingredients(
        ingredients=ingredients,
        number=number,
        ranking=1 if maximize_used else 2,
    )
    
    # Format the results for clarity
    formatted = []
    for recipe in recipes:
        formatted.append({
            "id": recipe.get("id"),
            "title": recipe.get("title"),
            "image": recipe.get("image"),
            "used_ingredient_count": recipe.get("usedIngredientCount", 0),
            "missing_ingredient_count": recipe.get("missedIngredientCount", 0),
            "used_ingredients": [
                ing.get("name") for ing in recipe.get("usedIngredients", [])
            ],
            "missing_ingredients": [
                ing.get("name") for ing in recipe.get("missedIngredients", [])
            ],
        })
    return formatted


@mcp.tool()
async def search_by_nutrients(
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
    
    Perfect for users with specific dietary goals like high-protein,
    low-carb, or calorie-controlled diets.
    
    Args:
        min_calories: Minimum calories per serving.
        max_calories: Maximum calories per serving.
        min_protein: Minimum protein in grams per serving.
        max_protein: Maximum protein in grams per serving.
        min_carbs: Minimum carbohydrates in grams per serving.
        max_carbs: Maximum carbohydrates in grams per serving.
        min_fat: Minimum fat in grams per serving.
        max_fat: Maximum fat in grams per serving.
        number: How many recipes to return (default 10).
        
    Returns:
        List of recipes matching the nutritional criteria with nutrition info.
    """
    client = get_client()
    recipes = client.search_by_nutrients(
        min_calories=min_calories,
        max_calories=max_calories,
        min_protein=min_protein,
        max_protein=max_protein,
        min_carbs=min_carbs,
        max_carbs=max_carbs,
        min_fat=min_fat,
        max_fat=max_fat,
        number=number,
    )
    return recipes


@mcp.tool()
async def get_recipe_info(recipe_id: int) -> dict[str, Any]:
    """Get detailed information about a specific recipe.
    
    Use this after finding a recipe via search to get full details
    including ingredients, instructions, and nutrition facts.
    
    Args:
        recipe_id: The recipe ID (from a previous search result).
        
    Returns:
        Detailed recipe with title, ingredients, instructions, nutrition, and more.
    """
    client = get_client()
    recipe = client.get_recipe_info(recipe_id, include_nutrition=True)
    
    # Extract and format key information
    return {
        "id": recipe.get("id"),
        "title": recipe.get("title"),
        "image": recipe.get("image"),
        "servings": recipe.get("servings"),
        "ready_in_minutes": recipe.get("readyInMinutes"),
        "source_url": recipe.get("sourceUrl"),
        "summary": recipe.get("summary", "").replace("<b>", "").replace("</b>", ""),
        "diets": recipe.get("diets", []),
        "dish_types": recipe.get("dishTypes", []),
        "ingredients": [
            {
                "name": ing.get("name"),
                "amount": ing.get("amount"),
                "unit": ing.get("unit"),
                "original": ing.get("original"),
            }
            for ing in recipe.get("extendedIngredients", [])
        ],
        "instructions": recipe.get("instructions", ""),
        "nutrition": {
            "calories": next(
                (n["amount"] for n in recipe.get("nutrition", {}).get("nutrients", [])
                 if n["name"] == "Calories"), None
            ),
            "protein": next(
                (n["amount"] for n in recipe.get("nutrition", {}).get("nutrients", [])
                 if n["name"] == "Protein"), None
            ),
            "carbs": next(
                (n["amount"] for n in recipe.get("nutrition", {}).get("nutrients", [])
                 if n["name"] == "Carbohydrates"), None
            ),
            "fat": next(
                (n["amount"] for n in recipe.get("nutrition", {}).get("nutrients", [])
                 if n["name"] == "Fat"), None
            ),
        },
    }


@mcp.tool()
async def get_similar_recipes(recipe_id: int, number: int = 5) -> list[dict[str, Any]]:
    """Find recipes similar to one you like.
    
    Great for discovering new recipes based on ones the user already enjoys.
    
    Args:
        recipe_id: The recipe ID to find similar recipes for.
        number: How many similar recipes to return (default 5).
        
    Returns:
        List of similar recipes with basic info.
    """
    client = get_client()
    recipes = client.get_similar_recipes(recipe_id, number=number)
    return recipes


@mcp.tool()
async def autocomplete_recipe(query: str, number: int = 10) -> list[dict[str, Any]]:
    """Get recipe name suggestions as you type.
    
    Helps users discover recipes by providing autocomplete suggestions.
    
    Args:
        query: Partial recipe name (e.g., "chick" for chicken recipes).
        number: How many suggestions to return (default 10).
        
    Returns:
        List of recipe name suggestions with IDs.
    """
    client = get_client()
    suggestions = client.autocomplete_recipe(query, number=number)
    return suggestions


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Run the MCP server
    mcp.run(transport="streamable-http")
