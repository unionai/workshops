tool_registry = {}
agent_tools = {}
agent_registry = {}

def agent(name):
    def decorator(fn):
        agent_registry[name] = fn
        return fn
    return decorator

def tool(agent=None):
    def decorator(fn):
        name = fn.__name__
        if agent:
            agent_tools.setdefault(agent, {})[name] = fn
        else:
            tool_registry[name] = fn
        return fn
    return decorator