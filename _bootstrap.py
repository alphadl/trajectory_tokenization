"""Set cwd and sys.path so scripts run from repo root. Call setup(__file__) at script start."""
import os
import sys


def setup(file_path: str) -> str:
    root = os.path.dirname(os.path.abspath(file_path))
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    return root
