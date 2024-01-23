import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple
import logging

class AudioSet:
    """
    A class for managing the audio dataset.

    This class handles the loading, preprocessing, and management of an audio dataset.

    Attributes:
        config (dict): Configuration settings for the audio dataset.
        NoiseSet (NoiseSet): An instance of NoiseSet for noise data.
        audio_paths (List[str]): List of file paths for the audio data.
        labels (List[int]): List of labels corresponding to the audio data.
        DATASET_AUDIO_PATH (str): File path to the audio dataset.
        class_names (List[str]): List of class names for the dataset.
        train_ds (tf.data.Dataset): The training dataset.
        valid_ds (tf.data.Dataset): The validation dataset.
    """

    def __init__(self, config: dict, NoiseSet) -> None:
        """
        Initialize the AudioSet with configuration settings and a NoiseSet instance.

        Args:
            config (dict): Configuration settings for the audio dataset.
            NoiseSet (NoiseSet): An instance of NoiseSet for noise data.
        """
        self.config = config
        self.NoiseSet = NoiseSet
        self.audio_paths = []
        self.labels = []
        self.DATASET_AUDIO_PATH = os.path.join(self.config['model']['paths']['dataset_dir'],
                                               self.config['model']['paths']['audio_dir'])
        self.class_names = self.get_class_names()
        self.train_ds, self.valid_ds = self.create_datasets()
        self.get_audio_files()
        self.shuffle_audio_files()
        self.split_samples_create_datasets()
        self.process_audio()

    def get_class_names(self) -> List[str]:
        """
        Retrieve class names from the audio dataset directory.

        Returns:
            List[str]: A list of class names.

        Raises:
            FileNotFoundError: If the audio dataset directory is not found.
        """
        try:
            return os.listdir(self.DATASET_AUDIO_PATH)
        except FileNotFoundError:
            logging.error(f"Audio directory not found: {self.DATASET_AUDIO_PATH}")
            raise

    def create_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Create the training and validation datasets.

        This method prepares the datasets for training and validation
        using the audio and noise data.

        Returns:
            Tuple[tf.data.Dataset, tf.data.Dataset]: A tuple containing the training and validation datasets.
        """
        # @TODO: Dataset creation logic, should return train_ds, valid_ds
        return tf.data.Dataset.from_tensors(0), tf.data.Dataset.from_tensors(0)

    def get_audio_files(self):
        for label, name in enumerate(self.class_names):
            print('Processing speaker {}'.format(name))

            dir_path = Path(self.DATASET_AUDIO_PATH) / name
            speaker_sample_paths = [
                os.path.join(dir_path, filepath)
                for filepath in os.listdir(dir_path)
                if filepath.endswith(".wav")
            ]
            self.audio_paths += speaker_sample_paths
            self.labels += [label] * len(speaker_sample_paths)

        print(
            "Found {} files belonging to {} classes.".format(len(self.audio_paths), len(self.class_names))
        )

    def shuffle_audio_files(self):
        rng = np.random.RandomState(self.config['MODEL']['GENERAL']['SHUFFLE_SEED'])
        rng.shuffle(self.audio_paths)
        rng = np.random.RandomState(self.config['MODEL']['GENERAL']['SHUFFLE_SEED'])
        rng.shuffle(self.labels)

    def split_samples_create_datasets(self):
        num_val_samples = int(self.config['MODEL']['GENERAL']['VALID_SPLIT'] * len(self.audio_paths))
        print("Using {} files for training.".format(len(self.audio_paths) - num_val_samples))
        self.train_audio_paths = self.audio_paths[:-num_val_samples]
        self.train_labels = self.labels[:-num_val_samples]

        print("Using {} files for validation.".format(num_val_samples))
        self.valid_audio_paths = self.audio_paths[-num_val_samples:]
        self.valid_labels = self.labels[-num_val_samples:]

        self.create_datasets(self.train_audio_paths, self.train_labels, self.valid_audio_paths, self.valid_labels)

    def paths_and_labels_to_dataset(self, audio_paths, labels):
        """Constructs a dataset of audios and labels."""
        self.path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
        self.audio_ds = self.path_ds.map(
            lambda x: self.path_to_audio(x), num_parallel_calls=tf.data.AUTOTUNE
        )
        self.label_ds = tf.data.Dataset.from_tensor_slices(self.labels)
        return tf.data.Dataset.zip((self.audio_ds, self.label_ds))

    def path_to_audio(self, path):
        """Reads and decodes an audio file."""
        audio = tf.io.read_file(path)
        audio, _ = tf.audio.decode_wav(audio, 1, self.config['MODEL']['GENERAL']['SAMPLING_RATE'])
        return audio

    def create_datasets(self, train_audio_paths, train_labels, valid_audio_paths, valid_labels):
        # Create 2 datasets, one for training and the other for validation
        self.train_ds = self.paths_and_labels_to_dataset(train_audio_paths, train_labels)
        self.train_ds = self.train_ds.shuffle(
            buffer_size=self.config['MODEL']['GENERAL']['BATCH_SIZE'] * 8,
            seed=self.config['MODEL']['GENERAL']['SHUFFLE_SEED']
            ).batch(self.config['MODEL']['GENERAL']['BATCH_SIZE'])

        self.valid_ds = self.paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
        self.valid_ds = self.valid_ds.shuffle(
            buffer_size=32 * 8,
            seed=self.config['MODEL']['GENERAL']['SHUFFLE_SEED']
            ).batch(32)

    def add_noise(self, audio, noises=None, scale=0.5):
        if noises is not None:
            # Create a random tensor of the same size as audio ranging from
            # 0 to the number of noise stream samples that we have.
            tf_rnd = tf.random.uniform(
                (tf.shape(audio)[0],), 0, noises.shape[0], dtype=tf.int32
            )
            noise = tf.gather(noises, tf_rnd, axis=0)

            # Get the amplitude proportion between the audio and the noise
            prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
            prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)

            # Adding the rescaled noise to audio
            audio = audio + noise * prop * scale

        return audio

    def audio_to_fft(self, audio):
        # Since tf.signal.fft applies FFT on the innermost dimension,
        # we need to squeeze the dimensions and then expand them again
        # after FFT
        audio = tf.squeeze(audio, axis=-1)
        fft = tf.signal.fft(
            tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
        )
        fft = tf.expand_dims(fft, axis=-1)

        # Return the absolute value of the first half of the FFT
        # which represents the positive frequencies
        return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])

    def process_audio(self):
        # Add noise to the training set
        self.train_ds = self.train_ds.map(
            lambda x, y: (self.add_noise(x, self.NoiseSet.noises, scale=
                                         self.config['MODEL']['GENERAL']['SCALE']), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Audio to FFT using audio_to_fft
        self.train_ds = self.train_ds.map(
            lambda x, y: (self.audio_to_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )
        self.train_ds = self.train_ds.prefetch(tf.data.AUTOTUNE)

        self.valid_ds = self.valid_ds.map(
            lambda x, y: (self.audio_to_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )
        self.valid_ds = self.valid_ds.prefetch(tf.data.AUTOTUNE)


