import os
import yaml


def load_config(path):
    """
    Loading config
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"config file is empty: {path}")

        return config

    except Exception as e:
        raise ValueError(f"Error in loading config {path}: {e}")
