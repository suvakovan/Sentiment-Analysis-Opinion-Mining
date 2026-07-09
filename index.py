import os
import sys


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(ROOT_DIR, "sentiment-analysis-project")

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from app import app  # noqa: E402
