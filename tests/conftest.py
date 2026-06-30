"""Make the repo root importable so `from aerochrome import ...` works under any
pytest (including a system/anaconda pytest that lacks the installed package)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
