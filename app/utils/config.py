import json
from typing import Dict, Any
from pathlib import Path

from app.utils.logger import AppLogger

def load_config(file_path: str) -> Dict[str, Any]:
    """
    Load the configuration settings from a JSON file.

    This function reads a configuration file, parses its JSON content,
    and returns the configuration as a dictionary.

    Args:
        file_path (str): The path to the configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration settings.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        json.JSONDecodeError: If there is an error parsing the JSON file.
    """
    logger = AppLogger.get_logger("app")
    proj_dir = str(Path.cwd()) + "\\"
    try:
        with open(proj_dir + file_path, 'r') as file:
            data = json.load(file)

        logger.info(f"{file_path} loaded!")

        # Updating paths relative to current working directory
        data['model']['paths']['proj_dir'] = proj_dir
        return data

    except FileNotFoundError:
        logger.error(f"file not found: {file_path}")
        raise

    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {file_path}")
        raise
