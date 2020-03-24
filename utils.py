"""Utilities module"""
from pathlib import Path
from collections import OrderedDict
import json


def read_config(path: Path):
    """Read model config"""
    with open(path, "r") as file_:
        config = json.load(file_, object_pairs_hook=OrderedDict)
    return config


if __name__ == "__main__":
    config = read_config("config.json")
    print(*config.values())
    print(type(config["b"]))
