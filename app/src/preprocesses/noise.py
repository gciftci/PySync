import os
from pathlib import Path
from typing import List

from app.utils.audio_pre import resample
from app.utils.logger import AppLogger
logger = AppLogger.get_logger("data_processing")


class NoiseSet:
    """
    A class for managing the noise dataset.

    This class handles the loading, preprocessing, and management of a noise dataset.

    Attributes:
        config (dict): Configuration settings for the noise dataset.
        noise_paths (List[str]): List of file paths for the noise data.
        DATASET_NOISE_PATH (str): File path to the noise dataset.
        noises (List[float]): Processed noise data.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the NoiseSet with configuration settings.

        Args:
            config (dict): Configuration settings for the noise dataset.
        """
        self.config = config
        self.noise_paths = []
        self.DATASET_NOISE_PATH = os.path.join(self.config['model']['paths']['dataset_dir'],
                                               self.config['model']['paths']['noise_dir'])
        self.noises = self.load_noises()
        logger.info("Initialized the NoiseSet with configuration settings")

    def get_noise_files(self):
        """
        Retrieve and store paths to all noise files in the dataset directory.

        Raises:
            FileNotFoundError: If no noise files are found in the dataset directory.
        """
        logger.debug("Retrieving and storing paths to all noise files in the dataset directory")
        for subdir in os.listdir(self.DATASET_NOISE_PATH):
            subdir_path = Path(self.DATASET_NOISE_PATH) / subdir
            if os.path.isdir(subdir_path):
                for filepath in os.listdir(subdir_path):
                    if filepath.endswith(".wav"):
                        logger.debug(f"Adding: {os.path.join(subdir_path, filepath)}")

                        self.noise_paths.append(os.path.join(subdir_path, filepath))
        if not self.noise_paths:
            raise RuntimeError(f"! No files at {self.DATASET_NOISE_PATH}")

    def load_noises(self) -> List[float]:
        """
        Load and process the noise files.

        This method retrieves noise files from the specified directory,
        processes them, and stores the result in the noises attribute.

        Returns:
            List[float]: Processed noise data.

        Raises:
            Exception: If there is an error in loading or processing noise files.
        """
        try:
            logger.debug("Loading and processing the noise files.")
            self.get_noise_files()
            return resample(self.DATASET_NOISE_PATH, self.noise_paths, self.config['model']['general']['sampling_rate'])
        except Exception as e:
            logger.error(f"Error loading noise files: {e}")
            raise
