# Building and Deploying an MCP Server on Union

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/unionai/workshops?quickstart=1)

This tutorial shows you how to build and deploy a **Model Context Protocol (MCP)** server on Union that helps AI agents find recipes! You'll create a recipe assistant that can search by ingredients, dietary needs, and more using the [Spoonacular Food API](https://spoonacular.com/food-api).

**What you'll learn:**
- How MCP servers work and their role in AI agent architectures
- Building MCP tools with Python using `fastmcp`
- Deploying your MCP server on Union
- Connecting the server to AI agents in Cursor or other MCP clients

## What is MCP?

The [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) is an open standard that enables AI assistants to securely connect with external data sources and tools. Think of it as a universal adapter that lets AI agents interact with APIs, databases, and services.

```
┌─────────────────┐     MCP Protocol     ┌─────────────────┐     API     ┌─────────────────┐
│   AI Agent      │◄───────────────────►│   MCP Server    │◄──────────►│   Spoonacular   │
│ (Claude, etc.)  │    Tools & Resources │ (Your Server)   │             │    Food API     │
└─────────────────┘                      └─────────────────┘             └─────────────────┘
```

## Prerequisites

- Python 3.11+
- A Spoonacular API key (free tier available)
- A Union account (sign up at [union.ai](https://union.ai))
- `uv` package manager (recommended)

## Setup

### 1. Get Your Spoonacular API Key (2 minutes)

1. Go to [spoonacular.com/food-api](https://spoonacular.com/food-api)
2. Click **Start Now** and create a free account
3. Copy your API key from the dashboard

The free tier includes **150 points/day** - plenty for development and testing!

### 2. Local Environment Setup

```bash
# Clone the repository
git clone https://github.com/unionai/workshops
cd workshops/tutorials/mcp

# Create virtual environment
uv venv .venv --python 3.11

# Activate the venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in this folder:

```bash
SPOONACULAR_API_KEY=your-api-key-here
```

## Project Structure

```
tutorials/mcp/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── config.py                           # Configuration settings
├── server.py                           # MCP server implementation
├── tools/                              # Spoonacular API tools
│   ├── __init__.py
│   └── recipes.py                      # Recipe API wrapper
├── deploy.py                           # Union deployment script
└── tutorial_recipe_mcp.ipynb           # Jupyter notebook tutorial
```

## Available Tools

The Recipe MCP server provides these tools for AI agents:

| Tool | Description |
|------|-------------|
| `search_recipes` | Search recipes by name, cuisine, diet, and more |
| `search_by_ingredients` | Find recipes using ingredients you have |
| `search_by_nutrients` | Find recipes by nutritional requirements |
| `get_recipe_info` | Get detailed recipe information |
| `get_similar_recipes` | Find recipes similar to one you like |
| `autocomplete_recipe` | Get recipe name suggestions |

## Quick Start

### Run the MCP Server Locally

```bash
# Test the server locally
python server.py
```

### Example Usage

Once connected to an AI agent, you can ask things like:

- *"What can I make with chicken, rice, and broccoli?"*
- *"Find me a vegan pasta recipe under 500 calories"*
- *"I want something similar to beef stroganoff"*
- *"Show me high-protein breakfast ideas"*
- *"What's a good gluten-free dessert?"*

## Deploying to Union

### 1. Connect to Union

```bash
# Configure Union CLI
union create config \
    --endpoint <your-union-endpoint> \
    --auth-type device-flow \
    --builder remote \
    --domain development \
    --project your-project

# Store your API key as a secret
union create secret SPOONACULAR_API_KEY
```

### 2. Deploy the MCP Server

```bash
# Build the container image
python deploy.py --build

# Run a test
python deploy.py --operation search --params '{"query": "pasta"}'
```

### 3. Configure Your MCP Client

Add the deployed server to your MCP client (e.g., Cursor):

**For Cursor** (`~/.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "recipe-assistant": {
      "command": "python",
      "args": ["server.py"],
      "cwd": "/path/to/tutorials/mcp",
      "env": {
        "SPOONACULAR_API_KEY": "your-api-key"
      }
    }
  }
}
```

## Tutorial Notebook

For a step-by-step walkthrough, open the Jupyter notebook:

```bash
jupyter notebook tutorial_recipe_mcp.ipynb
```

## How It Works

### MCP Server Architecture

```python
from fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("Recipe Assistant")

# Define a tool
@mcp.tool()
async def search_by_ingredients(ingredients: list[str], number: int = 5) -> list[dict]:
    """Find recipes using ingredients you have on hand."""
    # Call Spoonacular API
    ...
```

### Spoonacular API Integration

The server wraps these [Spoonacular endpoints](https://spoonacular.com/food-api/docs):

- **Complex Search**: Full-featured recipe search with filters
- **Search by Ingredients**: "What's in my fridge" functionality  
- **Search by Nutrients**: Find recipes by nutritional goals
- **Recipe Information**: Detailed recipe data with instructions
- **Similar Recipes**: Recommendation engine
- **Autocomplete**: Help users find recipe names

## Resources

- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [Union MCP Reference Implementation](https://github.com/unionai-oss/union-mcp)
- [Spoonacular API Documentation](https://spoonacular.com/food-api/docs)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)

## Troubleshooting

### "402 Payment Required" errors
- You've exceeded your daily API quota (150 points on free tier)
- Wait until midnight UTC for quota reset, or upgrade your plan

### "401 Unauthorized" errors
- Check that your API key is correct in `.env`
- Ensure the environment variable is being loaded

### Connection issues
- Verify your internet connection
- Check Spoonacular API status

## Next Steps

- Add more tools (meal planning, wine pairing, nutrition analysis)
- Build a recipe recommendation workflow
- Integrate with shopping list APIs
- Create a meal prep planning agent
