

## Project Structure

```
tutorials/multi-agent-workflows/
├── tutorial_planner_agent.ipynb  # Main tutorial notebook
├── config.py                      # Configuration settings
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables
├── agents/                       # Agent implementations
│   ├── __init__.py
│   ├── code_agent.py
│   ├── editor_agent.py
│   ├── math_agent.py
│   ├── planner_agent.py
│   ├── string_agent.py
│   ├── weather_agent.py
│   ├── web_search_agent.py
│   ├── web_search_reflexion_agent.py
│   └── writer_agent.py
├── tools/                        # Tool definitions for agents
│   ├── __init__.py
│   ├── code_tools.py
│   ├── math_tools.py
│   ├── string_tools.py
│   ├── weather_tools.py
│   └── web_search_tools.py
├── workflows/                    # Workflow orchestration
│   └── planner.py
└── utils/                        # Utility functions
    ├── __init__.py
    ├── decorators.py
    ├── file_viewer.py
    ├── logger.py
    ├── plan_executor.py
    └── summarizer.py
```

## Setup Instructions

```bash
# Clone the repository

# Create virtual environment
uv venv .venv --python 3.11

# Activate the venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -r requirements.txt
```

