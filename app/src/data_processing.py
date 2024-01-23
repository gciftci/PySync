import logging
from typing import NoReturn

import keras
from app.src.model import KerasModel
from app.src.preprocesses.audio import AudioSet
from app.src.preprocesses.noise import NoiseSet
from app.src.preprocesses.test import TestSet

from app.utils.logger import AppLogger
logger = AppLogger.get_logger("data_processing")

class DataProcessor:
    """
    A class responsible for managing the data processing pipeline.

    This class handles the creation of datasets and models, and orchestrates
    the training and testing processes.

    Attributes:
        config (dict): Configuration settings for the data processing.
        model (KerasModel): The machine learning model for processing data.
        audio_set (AudioSet): Dataset for audio processing.
        noise_set (NoiseSet): Dataset for noise processing.
        test_set (TestSet): Dataset for testing.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the DataProcessor with configuration settings.

        Args:
            config (dict): Configuration settings for the data processing.
        """
        self.config = config
        self.model = None
        self.audio_set = None
        self.noise_set = None
        self.test_set = None
        logger.info("Initialized the DataProcessor!")

    def create_sets(self) -> NoReturn:
        """
        Create the datasets required for processing.

        Initializes AudioSet, NoiseSet, and TestSet based on the configuration.
        Raises exceptions in case of any failure during dataset creation.
        """
        try:
            logger.info("Creating datasets required for processing!")
            self.noise_set = NoiseSet(self.config)
            logger.debug("Initialized NoiseSet")

            self.audio_set = AudioSet(self.config, self.noise_set)
            logger.debug("Initialized AudioSet")

            self.test_set = TestSet(self.config, self.audio_set, self.noise_set)
            logger.debug("Initializ TestSeted")

        except Exception as e:
            logging.error(f"Error in creating data sets: {e}")
            raise

    def create_model(self) -> NoReturn:
        """
        Create and configure the machine learning model.

        Initializes a KerasModel based on the configuration and datasets.
        Raises exceptions in case of any failure during model creation.
        """
        try:
            # Gen
            logger.info("Creating and configuring the machine learning model!")
            self.model = KerasModel(self.config, self.audio_set.class_names)
            self.model.summary()

            # Compiling the model
            self.model.compile(optimizer="Adam",
                               loss="sparse_categorical_crossentropy",
                               metrics=["accuracy"])

            model_save_filename = "model.keras"

            # Add callbacks:
            earlystopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(
                model_save_filename, monitor="val_accuracy", save_best_only=True
            )

        except Exception as e:
            logging.error(f"Error in creating model: {e}")
            raise

    def process_data(self) -> NoReturn:
        """
        Execute the data processing tasks.

        This method calls the functions to create datasets and models,
        and can be extended to include training and testing logic.
        """
        self.create_sets()
        self.create_model()
