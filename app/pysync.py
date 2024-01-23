from typing import NoReturn
from app.src.data_processing import DataProcessor
from app.utils.logger import AppLogger
logger = AppLogger.get_logger("app")

class App:
    """
    A class representing the main application.

    This class encapsulates the application logic, coordinating
    the initialization and execution of various components.

    Attributes:
        config (dict): Configuration settings for the application.
        data_processor (DataProcessor): Instance for managing data processing.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the App class with configuration settings.

        Args:
            config (dict): Configuration settings for the application.
        """
        self.config = config
        self.data_processor = None
        logger.info("App class with configuration settings initialized!")

    def init_data_processor(self) -> None:
        """
        Initialize the DataProcessor instance.

        This method lazily initializes the DataProcessor object, ensuring
        it is created only when needed.
        """
        if not self.data_processor:
            self.data_processor = DataProcessor(self.config)
            self.data_processor.process_data()

    def init_datasets(self) -> None:
        """
        Initialize the Data-Sets.

        This method lazily initializes the Data objects, ensuring
        it is created only when needed.
        """
        if not self.data_processor:
            self.data_processor = DataProcessor(self.config)

    def run(self) -> NoReturn:
        """
        Execute the main application logic.

        This method initializes the data processor and starts the application processes.
        """
        logger.debug("Executing main application logic!")
        self.init_data_processor()
        self.init_datasets()
