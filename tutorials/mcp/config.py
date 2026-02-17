"""Configuration settings for the Recipe MCP server."""

import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Spoonacular API Configuration
SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY", "")
SPOONACULAR_BASE_URL = "https://api.spoonacular.com"

# Server Configuration
SERVER_NAME = "recipe-mcp"
SERVER_VERSION = "1.0.0"

# Union Configuration
UNION_ENDPOINT = os.getenv("UNION_ENDPOINT", "")
UNION_PROJECT = os.getenv("UNION_PROJECT", "default")
UNION_DOMAIN = os.getenv("UNION_DOMAIN", "development")
