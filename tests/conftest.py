# tests/conftest.py — make the cbas package importable from tests/
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
