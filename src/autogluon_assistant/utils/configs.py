import importlib.resources
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

from ..constants import CONFIGS


def _get_default_config_path(
    presets: str,
) -> Path:
    """
    Get default config folder under package root
    Returns Path to the config.yaml file
    """
    try:
        # Get the package root directory using relative import
        package_root = Path(importlib.resources.files(__package__.split(".")[0]))

        # Construct path to configs directory
        config_path = package_root / CONFIGS / f"{presets}.yaml"

        if not config_path.exists():
            raise ValueError(
                f"Config file not found at expected location: {config_path}\n"
                f"Please ensure the config files are properly installed in the configs directory."
            )

        return config_path
    except Exception as e:
        logging.error(f"Error finding config file: {str(e)}")
        raise


def parse_override(override: str) -> tuple:
    """
    Parse a single override string in the format 'key=value' or 'key.nested=value'

    Args:
        override: String in format "key=value" or "key.nested=value"

    Returns:
        Tuple of (key, value)

    Raises:
        ValueError: If override string is not in correct format
    """
    if "=" not in override:
        raise ValueError(f"Invalid override format: {override}. Must be in format 'key=value' or 'key.nested=value'")
    key, value = override.split("=", 1)
    return key, value


def apply_overrides(config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """
    Apply command-line overrides to config

    Args:
        config: Base configuration
        overrides: List of overrides in format ["key1=value1", "key2.nested=value2"]

    Returns:
        Updated configuration
    """
    if not overrides:
        return config

    # Convert overrides to nested dict
    override_conf = {}
    overrides = ",".join(overrides)
    overrides = re.split(r"[,\s]+", overrides)
    for override in overrides:
        key, value = parse_override(override)

        # Try to convert value to appropriate type
        try:
            # Try to evaluate as literal (for numbers, bools, etc)
            value = eval(value)
        except:
            # Keep as string if eval fails
            pass

        # Handle nested keys
        current = override_conf
        key_parts = key.split(".")
        for part in key_parts[:-1]:
            current = current.setdefault(part, {})
        current[key_parts[-1]] = value

    # Convert override dict to OmegaConf and merge
    override_conf = OmegaConf.create(override_conf)
    return OmegaConf.merge(config, override_conf)


def load_config(
    presets: str, config_path: Optional[str] = None, overrides: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Load configuration from yaml file, merging with default config and applying overrides

    Args:
        config_path: Optional path to config file. If provided, will merge with and override default config
        overrides: Optional list of command-line overrides in format ["key1=value1", "key2.nested=value2"]

    Returns:
        Loaded and merged configuration

    Raises:
        ValueError: If config file not found or invalid
    """
    # Load default config
    default_config_path = _get_default_config_path(presets)
    logging.info(f"Loading default config from: {default_config_path}")
    config = OmegaConf.load(default_config_path)

    # If custom config provided, merge it
    if config_path:
        custom_config_path = Path(config_path)
        if not custom_config_path.is_file():
            raise ValueError(f"Custom config file not found at: {custom_config_path}")

        logging.info(f"Loading custom config from: {custom_config_path}")
        custom_config = OmegaConf.load(custom_config_path)
        config = OmegaConf.merge(config, custom_config)
        logging.info("Successfully merged custom config with default config")

    # Apply command-line overrides if any
    if overrides:
        logging.info(f"Applying command-line overrides: {overrides}")
        config = apply_overrides(config, overrides)
        logging.info("Successfully applied command-line overrides")

    return config
