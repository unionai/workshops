agent_tools = {}
agent_registry = {}

def agent(name):
    def decorator(fn):
        agent_registry[name] = fn
        return fn
    return decorator

def tool(agent):
    """
    Register a tool for a specific agent.

    Args:
        agent (str): The agent this tool belongs to (e.g., "math", "string")

    Example:
        @tool(agent="math")
        async def add(a, b):
            return a + b
    """
    def decorator(fn):
        name = fn.__name__
        agent_tools.setdefault(agent, {})[name] = fn
        return fn
    return decorator