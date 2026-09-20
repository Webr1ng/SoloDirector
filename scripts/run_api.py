#!/usr/bin/env python3
"""Start the SoloDirector FastAPI service."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import uvicorn  # noqa: E402

from src.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402

if __name__ == "__main__":
    config = load_config(DEFAULT_CONFIG_PATH)
    api_config = config.get("api", {})
    uvicorn.run(
        "src.api.server:app",
        host=str(api_config.get("host", "127.0.0.1")),
        port=int(api_config.get("port", 8000)),
        reload=False,
    )
