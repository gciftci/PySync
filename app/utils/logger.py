from rich.logging import RichHandler
import logging

class AppLogger:
    """
    Utility class for application-wide logging.

    This class encapsulates the configuration and usage of the logging system
    for the application.
    """

    @staticmethod
    def configure_logging(config: dict) -> None:
        """
        Configure the application-wide logging settings.

        Sets up the logging format and level. This method should be called
        at the start of the application.
        """
        logging.basicConfig(
            level=config['logging']['level'],
            format=config['logging']['format'],
            datefmt="[%X]",
            handlers=[RichHandler()]
        )


    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Retrieve a logger with the given name.

        Args:
            name (str): The name of the logger to retrieve.

        Returns:
            logging.Logger: A configured logger with the specified name.
        """
        return logging.getLogger(name)
