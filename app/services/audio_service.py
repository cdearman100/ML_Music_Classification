from app.utils.file_utils import add_to_csv, create_base_csv
from app.services.prediction_service import predict


def process_audio(audio_name, file_path):
    """
    Process a manually uploaded audio file and return its emotional features.
    """
    
    # Initialize the CSV file for storing features
    csv_file = 'data/extracted_features/audio_features.csv'
    create_base_csv(csv_file)

    # Add the audio file to the CSV and extract features
    add_to_csv(csv_filename=csv_file, audio_file=file_path, audio_name=audio_name)

    # Predict emotions based on the audio features
    emotion_data = predict(
        csv_file, 
        ['tonnetz', 'chroma', 'rms', 'spec_flux', 'spec_cont', 'spec_band', 'roll_off', 'mfcc']
    )

    # Return the predicted emotion data as JSON
    return emotion_data




