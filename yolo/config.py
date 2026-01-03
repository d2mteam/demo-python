import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if config_path.suffix in {".yaml", ".yml"}:
        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    if config_path.suffix == ".json":
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    raise ValueError("Config must be .yaml/.yml or .json")
