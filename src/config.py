"""Configuration loading and path normalization for SoloDirector."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration and attach an absolute project root.

    The returned mapping is deliberately plain data so it can be passed to API, CLI,
    and tests without introducing another settings dependency.
    """

    config_path = Path(path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Config root must be a mapping: {config_path}")

    config = deepcopy(dict(loaded))
    configured_root = Path(str(config.get("project", {}).get("root", ".")))
    if not configured_root.is_absolute():
        # config/config.yaml uses ".." so the repository root remains correct when the
        # CLI is launched from any working directory.
        configured_root = (config_path.parent / configured_root).resolve()
    config.setdefault("project", {})["root"] = str(configured_root)
    config["project"]["config_path"] = str(config_path)
    return config


def resolve_path(config: Mapping[str, Any], value: str | Path) -> Path:
    """Resolve a config-relative path against the normalized project root."""

    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    root = Path(str(config.get("project", {}).get("root", REPO_ROOT)))
    return (root / candidate).resolve()


def output_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    """Return and create the four standard output directories."""

    output_config = config.get("outputs", {})
    paths = {
        key: resolve_path(config, output_config.get(key, f"outputs/{key}"))
        for key in ("events_dir", "frames_dir", "clips_dir", "highlights_dir")
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def ensure_runtime_directories(config: Mapping[str, Any]) -> None:
    """Create only project-owned runtime directories; never touch user media."""

    output_paths(config)
    resolve_path(config, config.get("api", {}).get("upload_dir", "data/input/uploads")).mkdir(
        parents=True, exist_ok=True
    )
