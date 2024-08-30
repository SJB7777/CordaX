"""
This module provides functionality to load and save configuration files, 
with support for placeholder substitution in the configuration values.
"""

import re
from pathlib import Path
from functools import lru_cache

import yaml

from src.config.config_definitions import ExpConfig


def replace_placeholders(data, context):
    """
    Recursively replace placeholders in the given data structure.

    Placeholders are in the format `${key.subkey}`.

    Args:
        data (dict, list, str, or any): Data structure containing placeholders.
        context (dict): Context dictionary for placeholder resolution.

    Returns:
        Data structure with placeholders replaced.
    """
    if isinstance(data, dict):
        return {key: replace_placeholders(value, context) for key, value in data.items()}
    elif isinstance(data, list):
        return [replace_placeholders(item, context) for item in data]
    elif isinstance(data, str):
        return re.sub(r'\$\{([^}]+)\}', lambda match: resolve_placeholder(match.group(1), context), data)
    else:
        return data


def resolve_placeholder(placeholder, context):
    """
    Resolve a placeholder to its corresponding value from the context.

    Args:
        placeholder (str): Placeholder string in the format `${key.subkey}`.
        context (dict): Context dictionary containing the values.

    Returns:
        Value corresponding to the placeholder.

    Raises:
        ValueError: If the placeholder cannot be resolved.
    """
    keys = placeholder.split('.')
    value = context
    for key in keys:
        value = value.get(key)
        if value is None:
            raise ValueError(f"Placeholder '{placeholder}' could not be resolved")
    return value


@lru_cache(maxsize=1)
def load_config() -> ExpConfig:
    """
    Load the configuration file, replace placeholders, and return the configuration object.

    Returns:
        ExpConfig: Configuration object with placeholders resolved.
    """
    config_file = Path(__file__).resolve().parent.parent.parent / "config.yaml"
    with open(config_file, 'r', encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    context = config_dict.copy()
    replaced_data = replace_placeholders(config_dict, context)

    return ExpConfig(**replaced_data)


def save_config(config_dict: dict) -> None:
    """
    Save the given configuration dictionary to the configuration file.

    Args:
        config_dict (dict): Configuration dictionary to be saved.
    """
    config_file = Path(__file__).resolve().parent.parent.parent / "config.yaml"
    with open(config_file, 'w', encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)


if __name__ == '__main__':
    config = load_config()
    print(config)
