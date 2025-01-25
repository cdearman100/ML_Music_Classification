import os
import json
import librosa
import numpy as np
import csv
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Dense, BatchNormalization, Input
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from keras import models


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


def predict(csv: str, feat_list: list[str]):
    """
    Predicts the mood of songs using a pre-trained model.

    Args:
        csv (str): Path to the CSV file containing song features.
        feat_list (list[str]): List of features to include in the prediction.

    Returns:
        dict: Predictions for each song, including dominant emotion and probabilities.
    """
    model = models.load_model("data/models/final_model.keras")

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


def create_model(csv_file: str, feat_list: list[str], model_name: str):
    """
    Creates and trains a convolutional neural network (CNN) for emotion prediction.

    Args:
        csv_file (str): Path to the CSV file containing song features.
        feat_list (list[str]): List of features to include in the model.
        model_name (str): Name of the file to save the trained model.

    Returns:
        None
    """
    data = pd.read_csv('data/extracted_features/parquet.csv')
    data = remove_features(data, feat_list)

    # Prepare features and labels
    x = np.array(data.iloc[:, 1:], dtype=float)
    y = np.array(pd.get_dummies(data.iloc[:, 0]))

    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # Scale the features
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Train the model
    my_model = get_conv_model(x_train_scaled, x_test_scaled, y_train, y_test, 3, 64, 64, 64, 3, 5, 7, 32, 'categorical_crossentropy')
    my_model.save(model_name)


def get_conv_model(x_train, x_test, y_train, y_test, num_layers, f1, f2, f3, k1, k2, k3, d, loss, metrics='accuracy'):
    """
    Builds and trains a convolutional neural network (CNN) for emotion prediction.

    Args:
        x_train (ndarray): Training feature set.
        x_test (ndarray): Testing feature set.
        y_train (ndarray): Training labels.
        y_test (ndarray): Testing labels.
        num_layers (int): Number of convolutional layers in the network.
        f1, f2, f3 (int): Filters for each convolutional layer.
        k1, k2, k3 (int): Kernel sizes for each convolutional layer.
        d (int): Dropout rate.
        loss (str): Loss function for the model.
        metrics (str): Metrics to evaluate the model during training.

    Returns:
        Sequential: Trained Keras model.
    """
    K.clear_session()
    model = Sequential()

    # Input layer
    model.add(Input(shape=(x_train.shape[1], 1)))

    # Add convolutional layers
    for i in range(1, num_layers):
        model.add(Conv1D(f1, k1, activation='relu', padding='same'))
        model.add(BatchNormalization(name=f'BN{i}'))
        model.add(MaxPooling1D(pool_size=2, name=f'MaxPooling{i}'))

    # Add fully connected layers
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    # Compile the model
    model.compile(loss=loss, metrics=[metrics], optimizer='adam')

    # Early stopping callback
    earlystopping = callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10, restore_best_weights=True, verbose=1)

    # Train the model
    model.fit(x_train, y_train, epochs=100, batch_size=16, validation_data=(x_test, y_test), callbacks=earlystopping)

    return model
