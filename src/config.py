from pytorch_pfn_extras.config import Config
import yaml
from typing import Optional
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


if __name__ == "__main__":
    test_path = "./yml/test.yml"
    default_path = "./yml/test_default.yml"
    cfg = load_config(test_path, default_path)
    print("exp_name: ", cfg["/exp_name"])
