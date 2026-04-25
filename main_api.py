"""Entry point for Docker deployment. Serves the 34-class IoT IDS pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "34 classes" / "deployment"))

from api_34 import app  # noqa: F401  (re-exported for uvicorn)
