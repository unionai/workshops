#!/usr/bin/env python
"""Deploy the Recipe MCP server to Union.

This script builds and deploys the MCP server as a Union task that can be
invoked by AI agents through the Union MCP interface.

Usage:
    python deploy.py --build    # Build the container image
    python deploy.py            # Deploy and run the server
"""
# /// script
# dependencies = [
#    "flyte>=2.0.0b56",
#    "httpx>=0.27.0",
# ]
# ///

import argparse
import json
import os

import flyte

# Define the task environment for the MCP server
env = flyte.TaskEnvironment(
    name="recipe-mcp",
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    image=flyte.Image.from_uv_script(
        __file__, 
        name="recipe-mcp-image", 
        python_version=(3, 11),
        pre=True
    )
)


def _spoonacular_request(api_key: str, endpoint: str, params: dict) -> dict:
    """Make a request to the Spoonacular API."""
    import httpx
    
    params["apiKey"] = api_key
    url = f"https://api.spoonacular.com{endpoint}"
    
    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        return response.json()


@env.task
async def search_recipes(
    api_key: str,
    query: str = "",
    cuisine: str = "",
    diet: str = "",
    meal_type: str = "",
    max_ready_time: int = 0,
    number: int = 10,
) -> str:
    """Search for recipes with various filters.
    
    Args:
        api_key: Spoonacular API key.
        query: Search query.
        cuisine: Filter by cuisine.
        diet: Filter by diet.
        meal_type: Filter by meal type.
        max_ready_time: Max prep time in minutes.
        number: Number of results.
        
    Returns:
        JSON string of recipe results.
    """
    params = {"number": min(number, 100)}
    
    if query:
        params["query"] = query
    if cuisine:
        params["cuisine"] = cuisine
    if diet:
        params["diet"] = diet
    if meal_type:
        params["type"] = meal_type
    if max_ready_time > 0:
        params["maxReadyTime"] = max_ready_time
    
    result = _spoonacular_request(api_key, "/recipes/complexSearch", params)
    return json.dumps(result.get("results", []), indent=2)


@env.task
async def search_by_ingredients(
    api_key: str,
    ingredients: str,  # Comma-separated
    number: int = 10,
    maximize_used: bool = True,
) -> str:
    """Find recipes using ingredients you have.
    
    Args:
        api_key: Spoonacular API key.
        ingredients: Comma-separated list of ingredients.
        number: Number of results.
        maximize_used: Prioritize using more ingredients.
        
    Returns:
        JSON string of recipe results.
    """
    params = {
        "ingredients": ingredients,
        "number": min(number, 100),
        "ranking": 1 if maximize_used else 2,
        "ignorePantry": True,
    }
    
    recipes = _spoonacular_request(api_key, "/recipes/findByIngredients", params)
    
    # Format results
    formatted = []
    for recipe in recipes:
        formatted.append({
            "id": recipe.get("id"),
            "title": recipe.get("title"),
            "image": recipe.get("image"),
            "used_ingredients": [ing.get("name") for ing in recipe.get("usedIngredients", [])],
            "missing_ingredients": [ing.get("name") for ing in recipe.get("missedIngredients", [])],
        })
    
    return json.dumps(formatted, indent=2)


@env.task
async def search_by_nutrients(
    api_key: str,
    min_calories: int = 0,
    max_calories: int = 0,
    min_protein: int = 0,
    max_protein: int = 0,
    number: int = 10,
) -> str:
    """Find recipes by nutritional requirements.
    
    Args:
        api_key: Spoonacular API key.
        min_calories: Minimum calories.
        max_calories: Maximum calories.
        min_protein: Minimum protein grams.
        max_protein: Maximum protein grams.
        number: Number of results.
        
    Returns:
        JSON string of recipe results.
    """
    params = {"number": min(number, 100)}
    
    if min_calories > 0:
        params["minCalories"] = min_calories
    if max_calories > 0:
        params["maxCalories"] = max_calories
    if min_protein > 0:
        params["minProtein"] = min_protein
    if max_protein > 0:
        params["maxProtein"] = max_protein
    
    recipes = _spoonacular_request(api_key, "/recipes/findByNutrients", params)
    return json.dumps(recipes, indent=2)


@env.task
async def get_recipe_info(api_key: str, recipe_id: int) -> str:
    """Get detailed recipe information.
    
    Args:
        api_key: Spoonacular API key.
        recipe_id: Recipe ID.
        
    Returns:
        JSON string of recipe details.
    """
    params = {"includeNutrition": True}
    recipe = _spoonacular_request(api_key, f"/recipes/{recipe_id}/information", params)
    
    # Extract key info
    result = {
        "id": recipe.get("id"),
        "title": recipe.get("title"),
        "servings": recipe.get("servings"),
        "ready_in_minutes": recipe.get("readyInMinutes"),
        "diets": recipe.get("diets", []),
        "ingredients": [
            {"name": ing.get("name"), "amount": ing.get("amount"), "unit": ing.get("unit")}
            for ing in recipe.get("extendedIngredients", [])
        ],
        "instructions": recipe.get("instructions", ""),
    }
    
    return json.dumps(result, indent=2)


@env.task
async def get_similar_recipes(api_key: str, recipe_id: int, number: int = 5) -> str:
    """Find similar recipes.
    
    Args:
        api_key: Spoonacular API key.
        recipe_id: Recipe ID to find similar recipes for.
        number: Number of results.
        
    Returns:
        JSON string of similar recipes.
    """
    params = {"number": min(number, 100)}
    recipes = _spoonacular_request(api_key, f"/recipes/{recipe_id}/similar", params)
    return json.dumps(recipes, indent=2)


@env.task
async def main(operation: str, params: str) -> str:
    """Main entry point for the MCP server operations.
    
    Args:
        operation: The operation to perform.
        params: JSON string of operation parameters.
        
    Returns:
        JSON string of operation result.
    """
    params_dict = json.loads(params)
    api_key = params_dict.pop("api_key", "")
    
    if operation == "search":
        return await search_recipes(api_key=api_key, **params_dict)
    elif operation == "search_by_ingredients":
        return await search_by_ingredients(api_key=api_key, **params_dict)
    elif operation == "search_by_nutrients":
        return await search_by_nutrients(api_key=api_key, **params_dict)
    elif operation == "get_recipe_info":
        return await get_recipe_info(api_key=api_key, **params_dict)
    elif operation == "get_similar":
        return await get_similar_recipes(api_key=api_key, **params_dict)
    else:
        return json.dumps({"error": f"Unknown operation: {operation}"})


if __name__ == "__main__":
    from flyte.remote import auth_metadata
    
    parser = argparse.ArgumentParser(description="Deploy Recipe MCP to Union")
    parser.add_argument("--build", action="store_true", help="Build the container image only")
    parser.add_argument("--operation", type=str, default="search", help="Operation to test")
    parser.add_argument("--params", type=str, default='{"query": "pasta"}', help="JSON params")
    args = parser.parse_args()
    
    # Initialize passthrough for Union
    flyte.init_passthrough(
        project=os.getenv("FLYTE_INTERNAL_EXECUTION_PROJECT"),
        domain=os.getenv("FLYTE_INTERNAL_EXECUTION_DOMAIN"),
    )
    
    with auth_metadata(("authorization", os.environ.get("FLYTE_PASSTHROUGH_API_KEY", ""))):
        if args.build:
            uri = flyte.build(env.image, wait=False)
            print(f"Build run URL: {uri}")
        else:
            # Test run with provided operation
            run = flyte.with_runcontext(mode="remote").run(
                main, 
                operation=args.operation,
                params=args.params
            )
            print(f"Run URL: {run.url}")
