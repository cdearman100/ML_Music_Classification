import os
import json
import librosa
import numpy as np
import csv
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler
from app.utils.file_utils import remove_features
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Dense, BatchNormalization, Input
# from tensorflow.keras import callbacks
# from tensorflow.keras import backend as K
from keras.models import load_model



# Emotion labels
EMOTIONS = [
    "Amusing", "Annoying", "Anxious, tense", "Beautiful",
    "Calm, relaxing, serene", "Dreamy", "Energizing, pump-up",
    "Erotic, desirous", "Indignant, defiant", "Joyful, cheerful",
    "Sad, depressing", "Scary, fearful", "Triumphant, heroic"
]

FEATURE_NAMES = [
    'tempo', 'tonnetz', 'chroma', 'rms', 'spec_flux', 'spec_cont',
    'spec_cent', 'spec_band', 'roll_off', 'zcr', 'mfcc'
]



def predict(csv: str, feat_list: list[str]):
    """
    Predicts the mood of songs using a pre-trained model.

    Args:
        csv (str): Path to the CSV file containing song features.
        feat_list (list[str]): List of features to include in the prediction.

    Returns:
        dict: Predictions for each song, including dominant emotion and probabilities.
    """
    model = load_model.load_model("data/models/final_model.keras")

    # Load 'base' dataset
    base_df = pd.read_csv('data/extracted_features/parquet.csv')
    # Load 'new' dataset
    new_df = pd.read_csv(csv)
    # Merge datasets together in order to have more accurate prediction
    df = pd.concat([base_df, new_df], axis=0)
    
    # List of filenames
    y = df['filename']
    
    # Remove features that are not used in model
    df = remove_features(df, feat_list)
    
    x = np.array(df.iloc[:, 1:], dtype=float)
    
    # Scale the features
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    
    # Predict emotions
    x_predict = model.predict(x_scaled)
    pred = np.argmax(x_predict, axis=1)
    
    # Convert numpy types to native Python types
    y = y.tolist()  # Convert filenames to a list
    pred = pred.tolist()  # Convert predictions to a list
    x_predict = x_predict.tolist()  # Convert probabilities to a list
    

    
    # Combine the list of filenames, predictions with labels, and predictions
    # with percentages into a dictionary
    pred_dict = dict(zip(y, zip(pred, x_predict)))
    
    # Remove any values from base dataframe
    pred_dict = dict(list(pred_dict.items())[1343:])
    
    
    # Transform the output to the desired format
    formatted_output = {}
    for filename, (predicted_index, percentages) in pred_dict.items():
        percentages_dict = {EMOTIONS[i]: round(p * 100, 2) for i, p in enumerate(percentages)}
        dominant_emotion = EMOTIONS[predicted_index]
        formatted_output[filename] = {
            "dominant_emotion": dominant_emotion,
            "percentages": percentages_dict
        }

    
    return formatted_output


