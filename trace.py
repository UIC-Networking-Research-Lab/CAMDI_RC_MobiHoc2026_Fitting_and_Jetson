"""Collect four-node performance traces."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from traces.cli import main

if __name__ == "__main__":
    main()
