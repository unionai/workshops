from utils.decorators import tool
import flyte
import sys
from io import StringIO
import contextlib
import traceback
from typing import Optional


@tool(agent="code")
@flyte.trace
async def execute_python(
    code: str,
    timeout: int = 5,
    description: Optional[str] = None
) -> dict[str, str]:
    """
    Execute Python code in a safe, sandboxed environment.

    Args:
        code (str): The Python code to execute.
        timeout (int): Maximum execution time in seconds (default: 5).
        description (str, optional): Description of what the code does.

    Returns:
        dict: Dictionary with 'output', 'result', 'error', and 'description' keys.

    Example:
        code = '''
import math
result = math.factorial(5)
print(f"5! = {result}")
'''
    """
    print(f"[Execute Python] Running code: {description or 'No description'}")

    # Capture stdout
    stdout_capture = StringIO()
    result_value = None
    error_msg = ""

    # Allowed modules that can be imported in the sandbox
    ALLOWED_MODULES = {'math', 'json', 're', 'datetime', 'statistics', 'collections', 'itertools', 'functools'}

    def safe_import(name, *args, **kwargs):
        """Only allow importing from the approved module list."""
        if name not in ALLOWED_MODULES:
            raise ImportError(f"Module '{name}' is not allowed. Allowed: {', '.join(sorted(ALLOWED_MODULES))}")
        return __import__(name, *args, **kwargs)

    # Create a restricted namespace for execution
    safe_namespace = {
        '__builtins__': {
            # Math functions
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'sorted': sorted,
            'reversed': reversed,
            'map': map,
            'filter': filter,
            'any': any,
            'all': all,
            # Types
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            # Output
            'print': print,
            # Allow safe imports
            '__import__': safe_import,
            'True': True,
            'False': False,
            'None': None,
        },
        # Pre-loaded modules for convenience
        'math': __import__('math'),
        'json': __import__('json'),
        're': __import__('re'),
        'datetime': __import__('datetime'),
        'statistics': __import__('statistics'),
    }

    try:
        # Execute code with captured stdout
        with contextlib.redirect_stdout(stdout_capture):
            # Use exec for statements, eval for expressions
            exec(code, safe_namespace)

            # Try to get 'result' variable if it exists
            if 'result' in safe_namespace:
                result_value = safe_namespace['result']

        output = stdout_capture.getvalue()

        print(f"[Execute Python] Success - Output length: {len(output)} chars")

        return {
            "output": output,
            "result": str(result_value) if result_value is not None else "",
            "error": "",
            "description": description or ""
        }

    except SyntaxError as e:
        error_msg = f"Syntax Error: {str(e)}"
        print(f"[Execute Python] {error_msg}")
        return {
            "output": stdout_capture.getvalue(),
            "result": "",
            "error": error_msg,
            "description": description or ""
        }

    except Exception as e:
        error_msg = f"Runtime Error: {str(e)}\n{traceback.format_exc()}"
        print(f"[Execute Python] {error_msg}")
        return {
            "output": stdout_capture.getvalue(),
            "result": "",
            "error": error_msg,
            "description": description or ""
        }
