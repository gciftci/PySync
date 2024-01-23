import os
import subprocess
from typing import Optional, List
import tensorflow as tf
from pathlib import Path

from app.utils.logger import AppLogger
logger = AppLogger.get_logger("data_processing")

def load_sample(path: str, rate: int) -> Optional[List[tf.Tensor]]:
    """
    Load and process an audio sample.

    Reads an audio file from the specified path, decodes it, and processes it
    into slices if the sampling rate matches the expected rate.

    Args:
        path (str): The file path to the audio sample.
        rate (int): The expected sampling rate.

    Returns:
        Optional[List[tf.Tensor]]: A list of processed audio slices as tensors,
        or None if the sampling rate is incorrect or the file cannot be processed.

    Raises:
        Exception: If there is an error in loading or processing the audio file.
    """
    try:
        audio, sampling_rate = tf.audio.decode_wav(
            tf.io.read_file(path), desired_channels=1
        )
        if sampling_rate == rate:
            slices = int(audio.shape[0] / rate)
            return tf.split(audio[: slices * rate], slices)
        else:
            logger.warning(f"Sampling rate for {path} is incorrect. Ignoring it.")
            return None
    except Exception as e:
        logger.error(f"Error loading sample from {path}: {e}")
        return None

def exec_cmd(resampling_folder):
    resampling_folder = str(Path.cwd()) + "\\" + resampling_folder
    logger.debug(f"Resampling Folder: {resampling_folder}")
    command = f"""
    $ErrorActionPreference = 'SilentlyContinue'
    echo "{resampling_folder}"
    Get-ChildItem -Path "{resampling_folder}" -Directory | ForEach-Object {{
        $dir = "{resampling_folder}\\$_"
        echo "$dir"
        Get-ChildItem -Path "$dir\\*.wav" | ForEach-Object {{
            $file = $_
            echo "Uberprufe Datei: $file"
            $sample_rate = ffprobe -hide_banner -loglevel panic -show_streams $file.FullName | Select-String 'sample_rate' | ForEach-Object {{ $_.ToString().Split('=')[1] }}
            echo "Sample-Rate: $sample_rate"
            if ($sample_rate -ne 16000) {{
                echo "Konvertiere Datei: $file"
                $fileDir = $file.DirectoryName + "\\temp.wav"
                C:\\ffmpeg\\bin\\ffmpeg.exe -hide_banner -loglevel panic -y -i $file -ar 16000 $fileDir
                Move-Item -Path $fileDir -Destination $file -Force
                echo "Konvertierung abgeschlossen: $file"
            }}
        }}
    }}
    """
    process = subprocess.Popen(["powershell", "-command", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logger.info("Executed resampling")
    stdout, stderr = process.communicate()
    # print("out: ", stdout)
    # print("e: ",stderr)

def resample(audio_paths: List[str], sampling_rate: int, rate: int):
    """
    Resample audio files to a specified sampling rate.

    This function processes a list of audio file paths, loading each file,
    and resampling it to the given sampling rate.

    Args:
        audio_paths (List[str]): A list of paths to the audio files to be resampled.
        sampling_rate (int): The target sampling rate to resample the audio.

    Returns:
        Optional[List[tf.Tensor]]: A list of resampled audio tensors,
        or None if an error occurs during the resampling process.

    Raises:
        Exception: If there is an error in loading or processing the audio files.
    """
    from app.utils.config import load_config
    config = load_config("config.json")

    DATASET_NOISE_PATH = os.path.join(config['model']['paths']['dataset_dir'],
                                      config['model']['paths']['noise_dir'])
    resampled_audios = []
    exec_cmd(audio_paths)
    for path in os.listdir(DATASET_NOISE_PATH):
        try:
            logger.warning("{}{}".format(path, sampling_rate))
            audio_tensor = load_audio_file(path, target_sampling_rate=sampling_rate)
            resampled_audios.append(audio_tensor)
        except Exception as e:
            logger.error(f"Error resampling audio file ({path}): {e}")
            return None
    return resampled_audios

def load_audio_file(file_path: str, target_sampling_rate: int) -> tf.Tensor:
    """
    Load an audio file and resample it to a target sampling rate.

    Args:
        file_path (str): The file path to the audio file.
        target_sampling_rate (int): The target sampling rate.

    Returns:
        tf.Tensor: A tensor representing the resampled audio data.

    Raises:
        Exception: If there is an error in reading or resampling the audio file.
    """
    noise_paths = []
    for subdir in os.listdir(str(Path.cwd()) + "\\data\\noise\\" + file_path):
        subdir_path = Path(str(Path.cwd()) + "\\data\\noise\\" + file_path) / subdir
        if os.path.isdir(subdir_path):
            noise_paths += [
                os.path.join(subdir_path, filepath)
                for filepath in os.listdir(subdir_path)
                if filepath.endswith(".wav")
            ]
    if not noise_paths:
        raise RuntimeError(f"Could not find any files")
    try:
        logger.debug(("load_audio_file ERROR: {}".format(file_path)))
        audio, sampling_rate = tf.audio.decode_wav(
            tf.io.read_file(str(Path.cwd()) + "\\data\\noise\\" + file_path), desired_channels=1
        )
        # Resample if the current rate is different from the target rate
        if sampling_rate.numpy() != target_sampling_rate:
            audio = tf.signal.resample(audio, int(len(audio) / sampling_rate * target_sampling_rate), axis=0)
        return audio
    except Exception as e:
        logger.error(f"Error loading or resampling audio file ({file_path}): {e}")
        raise

