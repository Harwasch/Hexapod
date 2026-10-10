"""Isolated pinned upstream model process; see README for validation limits."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_support import serve

if __name__ == "__main__":
    from matrix_game.engine import Engine
    serve(Engine)
