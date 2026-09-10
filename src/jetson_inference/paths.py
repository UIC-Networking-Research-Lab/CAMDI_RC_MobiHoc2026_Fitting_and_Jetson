"""Resolve bundled assets and outputs at the unchanged repository root."""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
