"""Pytest fixtures for pipecat_voice tests."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

# Make the src/ layout importable without installing the package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Quiet down loguru so tests don't spam.
logger.remove()
logger.add(sys.stderr, level="WARNING")
