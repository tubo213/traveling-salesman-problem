from typing import Optional

import yaml
from pytorch_pfn_extras.config import Config

from src.config_types import CONFIG_TYPES


def load_config(path: str, default_path: Optional[str] = None) -> Config:
    with open(path) as f:
        cfg: dict = yaml.safe_load(f)

    if default_path is not None:
        with open(default_path) as f:
            default_cfg: dict = yaml.safe_load(f)
        # merge default config
        for k, v in default_cfg.items():
            if k not in cfg:
                print(f"used default {k}: {v}")
                cfg[k] = v

    return Config(cfg, types=CONFIG_TYPES)  # type: ignore
