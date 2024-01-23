from typing import NoReturn
from app.utils.logger import AppLogger
from app.utils.config import load_config
from app.utils.utils import set_env
from app.pysync import App

def main() -> NoReturn:
    """
    The main function of the PySync application.

    This function initiates the application by loading the configuration,
    setting environment variables, creating an instance of the App class,
    and then running the application.

    Returns:
        NoReturn: This function does not return anything.
    """
    try:
        # Load the configuration settings
        config = load_config("config.json")

        # Configure application-wide logging
        AppLogger.configure_logging(config)
        logger = AppLogger.get_logger("main")
        logger.debug('Configuration loaded!')

        # Set environment variables based on the loaded configuration
        set_env(config)

        # Initialize and run the application
        app = App(config)
        app.run()
    except Exception as e:
        # Log any exceptions that occur during the application initialization and runtime
        print(f"An error occurred: {e}")
        # logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
