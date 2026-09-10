"""Redraw figures from saved single-task or multi-task results."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from jetson_inference.entrypoints import plot

if __name__ == "__main__":
    plot()
