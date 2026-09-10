"""Collect accuracy measurements and fit accuracy estimators."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from fitting.cli import main

if __name__ == "__main__":
    main()
