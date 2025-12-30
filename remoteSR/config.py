from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "gan.yaml"


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_raw_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError(
                "pyyaml is required for YAML config files. Install via `pip install pyyaml`."
            )
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError(f"Unsupported config format: {path.suffix}")


def _normalize_config_paths(
    path: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
) -> list[Path]:
    if path is None:
        return [DEFAULT_CONFIG_PATH]
    if isinstance(path, (list, tuple)):
        return [Path(p) for p in path]
    if isinstance(path, Path):
        return [path]
    parts = [p.strip() for p in str(path).split(",") if p.strip()]
    return [Path(p) for p in parts]


def load_config(
    path: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
    default_path: Path | None = None,
) -> dict[str, Any]:
    """
    Load one or more config files and merge them with the defaults.
    """
    base_path = default_path or DEFAULT_CONFIG_PATH
    merged = _load_raw_config(base_path)
    if path is None:
        return merged
    for cfg_path in _normalize_config_paths(path):
        if cfg_path.resolve() == base_path.resolve():
            continue
        raw = _load_raw_config(cfg_path)
        merged = _deep_merge(merged, raw)
    return merged
