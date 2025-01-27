import os
import csv
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import csv
from pydub import AudioSegment
from datasets import load_dataset
from app.utils.audio_features import feature_extractor # Import the feature extraction function

# List of low-level audio feature names for extraction and analysis
FEATURE_NAMES = [
    'tonnetz', 'chroma', 'rms', 'spec_flux', 'spec_cont', 'spec_cent',
    'spec_band', 'roll_off', 'zcr'
]
# List of statistical metrics to calculate for each audio feature
STATS = ['mean', 'var', 'std', 'median', 'min', 'max']

def remove_features(df: pd.DataFrame, feat_list: list[str]) -> pd.DataFrame:
    """
    Removes features from a DataFrame that are not included in the feature list.

    Args:
        df (pd.DataFrame): DataFrame containing extracted features from audio files.
        feat_list (list[str]): List of features to retain in the DataFrame. Others will be removed.

    Returns:
        pd.DataFrame: Updated DataFrame with specified features removed.
    """
    # Determine features to remove
    feats_to_remove = list(set(FEATURE_NAMES).difference(feat_list))

    # Drop unnecessary columns
    for column in ['filename', 'audio_array', 'sampling_rate', 'feeling', 'emotion']:
        if column in df.columns:
            df = df.drop(column, axis=1)

    # Drop features not in the feature list
    for feat in feats_to_remove:
        cols_to_drop = df.columns[df.columns.str.contains(feat)]
        df.drop(cols_to_drop, axis=1, inplace=True)

    return df


def write_parquet_csv():
    """
    Creates a CSV file from a parquet dataset hosted on Hugging Face.
    The CSV includes audio file information and extracted features.

    Returns:
        str: The name of the generated CSV file.
    """
    filename = "parquet.csv"

    dataset = load_dataset(
        "parquet",
        data_files="/Users/christiandearman/PYTHON_Projects/Music_Classification/parquet/train-00000-of-00008-073bb2769492c6c6.parquet"
    )

    dataset = dataset.remove_columns([
        "Image1_filename", "Image1_tag", "Image1_text",
        "Image2_filename", "Image2_tag", "Image2_text",
        "Image3_filename", "Image3_tag", "Image3_text", "is_original_clip"
    ])

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ['filename', 'audio_array', 'sampling_rate', 'genre', 'feeling', 'emotion', 'tempo']

        for feature in FEATURE_NAMES:
            for stat in STATS:
                header.append(f'{feature}_{stat}')

        for i in range(1, 21):
            for stat in STATS:
                header.append(f'mfcc{i}_{stat}')

        writer.writerow(header)
        
        count = 0
        for data_row in dataset['train']:
            audio_filename = data_row['Audio_Filename']['path']
            if audio_filename.endswith(".mp3"):
                audio_filename = audio_filename[:-4] + '.wav'

            array = data_row['Audio_Filename']['array']
            sampling_rate = int(data_row['Audio_Filename']['sampling_rate'])
            genre = data_row['genre']
            feeling = data_row['feeling']
            emotion = data_row['emotion']

            row = [audio_filename, array, sampling_rate, genre, feeling, emotion]

            sf.write(f'audio/{audio_filename}', np.ravel(array), samplerate=sampling_rate)
            row.extend(feature_extractor(f'audio/{audio_filename}'))

            writer.writerow(row)
            count += 1
            print(f'{count} row(s) complete!')

    return filename

def create_base_csv(csv_filename):
    """
    Creates an empty CSV file with only a header row. The header includes
    all relevant audio features and their stats.

    Args:
        csv_filename (str): Name of the CSV file to create.

    Returns:
        None
    """
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        header = ['filename', 'tempo']
        for feature in FEATURE_NAMES:
            for stat in STATS:
                header.append(f'{feature}_{stat}')

        for i in range(1, 21):
            for stat in STATS:
                header.append(f'mfcc{i}_{stat}')

        writer.writerow(header)
        
def add_to_csv(csv_filename, audio_file, audio_name=""):
    """
    Appends a song and its features to an existing CSV file.

    Args:
        csv_filename (str): Name of the CSV file to update.
        audio_file (str): Path to the audio file to extract features from.
        audio_name (str, optional): Display name of the audio file.

    Returns:
        None
    """
    with open(csv_filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        try:
            row = feature_extractor(audio_file)
        except ValueError as e:
            print(f"Skipping {audio_file}: {e}")
            return

        row.insert(0, audio_name)
        writer.writerow(row)
        
        
def convert_to_wav(input_directory, output_directory):
    """
    Converts all .mp3 files in a directory to .wav format and saves them to an output directory.

    Args:
        input_directory (str): Path to the directory containing .mp3 files.
        output_directory (str): Path to the directory where .wav files will be saved.

    Returns:
        None
    """
    os.makedirs(output_directory, exist_ok=True)

    for file in os.listdir(input_directory):
        if file.endswith(".mp3"):
            input_path = os.path.join(input_directory, file)
            output_path = os.path.join(output_directory, file.replace(".mp3", ".wav"))

            # Convert to .wav
            audio = AudioSegment.from_file(input_path)
            audio.export(output_path, format="wav")
            print(f"Converted {file} to .wav format.")