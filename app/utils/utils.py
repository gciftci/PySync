import os
from typing import NoReturn

from app.utils.logger import AppLogger

def set_env(config: dict) -> NoReturn:
    """
    Set environment variables based on the provided configuration.

    This function updates the system's environment variables with the values
    specified in the configuration dictionary under the 'environment' key.

    Args:
        config (dict): A dictionary containing the application's configuration,
                       including environment variables under the 'environment' key.
    """
    logger = AppLogger.get_logger("set_env")
    try:
        logger.info("Setting environment variables")
        for key, value in config.get('environment', {}).items():
            os.environ[key] = str(value)
            logger.debug(f"    {key}: {value}")
    except Exception as e:
        logger.error(f"Error setting environment variables: {e}")
        raise
