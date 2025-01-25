import numpy as np
import librosa


# List of low-level audio feature names for extraction and analysis
FEATURE_NAMES = [
    'tonnetz', 'chroma', 'rms', 'spec_flux', 'spec_cont', 'spec_cent',
    'spec_band', 'roll_off', 'zcr'
]
# List of statistical metrics to calculate for each audio feature
STATS = ['mean', 'var', 'std', 'median', 'min', 'max']

def get_feature_data(name, val, feature_data):
    """
    Updates dictionary with a feature and its calculated statistical metrics.

    Args:
        name (str): Name of the feature (e.g., tonnetz, spec_flux).
        val (ndarray): The value of the feature extracted from the song.
        feature_data (dict): Dictionary to be updated.

    Returns:
        dict: Updated dictionary with feature statistics.
    """
    feature_data[name, 'mean'] = np.mean(val)
    feature_data[name, 'var'] = np.var(val)
    feature_data[name, 'std'] = np.std(val)
    feature_data[name, 'median'] = np.median(val)
    feature_data[name, 'min'] = np.min(val)
    feature_data[name, 'max'] = np.max(val)
    return feature_data

def mfccs_feature_extractor(audio_path):
    """
    Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from an audio file.

    Args:
        audio_path (str): Path to the audio file for feature extraction.

    Returns:
        ndarray: Array of MFCC features with a range of 1-20.
    """
    audio_data, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=20)
    return mfccs

def feature_extractor(audio_file):
    """
    Extracts low-level audio features (e.g., spectral centroid, zero-crossing rate) from an audio file.

    Args:
        audio_file (str): Path to the audio file.

    Returns:
        list: List of extracted audio features and their statistical metrics.
    """
    feature_data = {}
    feature_stats = []
    frame_size = 2048  # Frame size for feature extraction
    hop_length = 512  # Hop length for feature extraction

    # Load the audio file
    audio_data, sr = librosa.load(audio_file, res_type='kaiser_fast')

    # Ensure the audio signal is long enough for analysis
    if len(audio_data) < frame_size:
        raise ValueError(f"Audio file {audio_file} is too short for feature extraction.")

    # Extract tempo
    tempo = librosa.beat.tempo(y=audio_data, sr=sr)[0]
    feature_data['tempo', 'val'] = tempo

    # Extract various features and update the dictionary with their statistics
    tonnetz = librosa.feature.tonnetz(y=audio_data, sr=sr)
    feature_data.update(get_feature_data('tonnetz', tonnetz, feature_data))

    chroma_stft = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    feature_data.update(get_feature_data('chroma', chroma_stft, feature_data))

    rms = librosa.feature.rms(y=audio_data, frame_length=frame_size, hop_length=hop_length)[0]
    feature_data.update(get_feature_data('rms', rms, feature_data))

    spec_flux = librosa.onset.onset_strength(y=audio_data, sr=sr)
    feature_data.update(get_feature_data('spec_flux', spec_flux, feature_data))

    stft_magnitude = np.abs(librosa.stft(audio_data))
    spec_cont = librosa.feature.spectral_contrast(S=stft_magnitude, sr=sr)
    feature_data.update(get_feature_data('spec_cont', spec_cont, feature_data))

    spec_cent = librosa.feature.spectral_centroid(
        y=audio_data, sr=sr, n_fft=frame_size, hop_length=hop_length
    )[0]
    feature_data.update(get_feature_data('spec_cent', spec_cent, feature_data))

    spec_band = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
    feature_data.update(get_feature_data('spec_band', spec_band, feature_data))

    roll_off = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
    feature_data.update(get_feature_data('roll_off', roll_off, feature_data))

    zcr = librosa.feature.zero_crossing_rate(
        y=audio_data, frame_length=frame_size, hop_length=hop_length
    )[0]
    feature_data.update(get_feature_data('zcr', zcr, feature_data))

    # Compile feature statistics into a list for CSV writing
    feature_stats.append(tempo)
    for feature in FEATURE_NAMES:
        for stat in STATS:
            feature_stats.append(feature_data[feature, stat])

    # Extract MFCC features and add their statistics
    mfcc = mfccs_feature_extractor(audio_file)
    for coeff in mfcc:
        feature_stats.extend([
            np.mean(coeff), np.var(coeff), np.std(coeff),
            np.median(coeff), np.min(coeff), np.max(coeff)
        ])

    print('FEATURES EXTRACTED')
    return feature_stats
