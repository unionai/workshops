import httpx
import flyte
from utils.decorators import tool

@tool(agent="weather")
@flyte.trace
async def get_weather(location: str) -> dict:
    """
    Gets the current weather for a given location using the wttr.in API.
    
    Args:
        location (str): The location to get weather for (e.g., "London", "New York", "Tokyo")
    
    Returns:
        dict: Weather information including temperature, condition, and description
    """
    print(f"TOOL CALL: Getting weather for {location}")
    
    async with httpx.AsyncClient() as client:
        # Call wttr.in API with JSON format
        response = await client.get(
            f"https://wttr.in/{location}?format=j1",
            timeout=10.0
        )
        response.raise_for_status()
        data = response.json()
    
    # Extract key weather information
    current = data['current_condition'][0]
    weather_info = {
        "location": location,
        "temperature_c": current['temp_C'],
        "temperature_f": current['temp_F'],
        "condition": current['weatherDesc'][0]['value'],
        "feels_like_c": current['FeelsLikeC'],
        "feels_like_f": current['FeelsLikeF'],
        "humidity": current['humidity'],
        "wind_speed_kmph": current['windspeedKmph']
    }
    
    return weather_info
