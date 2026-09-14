from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str) -> dict:
    path = Path(path)
    with open(path) as f:
        model_cfg = yaml.safe_load(f) or {}

    base_path = path.parent / "base.yaml"
    if base_path.resolve() == path.resolve():
        return model_cfg

    with open(base_path) as f:
        base_cfg = yaml.safe_load(f) or {}

    return _deep_merge(base_cfg, model_cfg)
