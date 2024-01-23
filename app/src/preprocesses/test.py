import tensorflow as tf
from typing import NoReturn
from app.src.preprocesses.audio import AudioSet
from app.src.preprocesses.noise import NoiseSet
import logging

class TestSet:
    """
    A class for managing the test dataset.

    This class handles the preparation and management of a test dataset
    for evaluating the model.

    Attributes:
        config (dict): Configuration settings for the test dataset.
        AudioSet (AudioSet): An instance of AudioSet for audio data.
        NoiseSet (NoiseSet): An instance of NoiseSet for noise data.
        test_ds (tf.data.Dataset): The test dataset.
    """

    def __init__(self, config: dict, AudioSet, NoiseSet) -> None:
        """
        Initialize the TestSet with configuration settings, AudioSet, and NoiseSet instances.

        Args:
            config (dict): Configuration settings for the test dataset.
            AudioSet (AudioSet): An instance of AudioSet for audio data.
            NoiseSet (NoiseSet): An instance of NoiseSet for noise data.
        """
        self.config = config
        self.AudioSet = AudioSet
        self.NoiseSet = NoiseSet
        self.test_ds = self.create_dataset()

    def create_dataset(self) -> NoReturn:
        """
        Create and return the test dataset.

        This method prepares the test dataset using the audio and noise data.

        Raises:
            Exception: If there is an error in creating the test dataset.
        """
        try:
            # @TODO: Dataset creation logic
            pass
        except Exception as e:
            logging.error(f"Error creating test dataset: {e}")
            raise

