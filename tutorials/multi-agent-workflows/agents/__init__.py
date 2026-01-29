import importlib
import pkgutil
from pathlib import Path


def import_all_agents():
    """Auto-import all agent modules to trigger @agent and @tool decorator registration."""
    package_dir = Path(__file__).parent
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        importlib.import_module(f"agents.{module_info.name}")