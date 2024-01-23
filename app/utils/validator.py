import os
from typing import Any, Optional

class Validator:
    """
    Utility class for various validation tasks within the application.

    This class provides static methods for validating different types of data,
    such as configuration settings, user inputs, or file paths.
    """

    @staticmethod
    def validate_config(config: Any) -> bool:
        """
        Validate the application's configuration.

        Checks if the provided configuration meets the expected structure and values.

        Args:
            config (Any): The configuration object to be validated.

        Returns:
            bool: True if the configuration is valid, False otherwise.
        """
        # @TODO: Implement validation logic for the configuration
        required_keys = ["model", "environment"]
        return all(key in config for key in required_keys)

    @staticmethod
    def validate_file_path(file_path: str) -> bool:
        """
        Validate a file path.

        Checks if the file path exists and is accessible.

        Args:
            file_path (str): The file path to be validated.

        Returns:
            bool: True if the file path is valid and accessible, False otherwise.
        """
        return os.path.exists(file_path) and os.path.isfile(file_path)
