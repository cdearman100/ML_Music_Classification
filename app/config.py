import os

class Config:
    """
    Base configuration class with default settings.
    """
    # SECRET_KEY = os.environ.get("SECRET_KEY", "default_secret_key")
    DEBUG = False
    TESTING = False

    # File storage settings
    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "data/uploaded_audio/")
    EXTRACTED_FEATURES_FOLDER = os.environ.get("EXTRACTED_FEATURES_FOLDER", "data/extracted_features/")
    DOWNLOADED_SONGS_FOLDER = os.environ.get("DOWNLOADED_SONGS_FOLDER", "data/downloaded_songs/")

    # Prediction model path
    MODEL_PATH = os.environ.get("MODEL_PATH", "data/models/emotion_model.pkl")

    # Spotify API credentials
    SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "your_spotify_client_id")
    SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "your_spotify_client_secret")


class DevelopmentConfig(Config):
    """
    Configuration for development environment.
    """
    DEBUG = True


class TestingConfig(Config):
    """
    Configuration for testing environment.
    """
    TESTING = True
    DEBUG = True

    # Temporary directories for testing
    UPLOAD_FOLDER = "data/test/uploaded_audio/"
    EXTRACTED_FEATURES_FOLDER = "data/test/extracted_features/"
    DOWNLOADED_SONGS_FOLDER = "data/test/downloaded_songs/"


class ProductionConfig(Config):
    """
    Configuration for production environment.
    """
    DEBUG = False
    SECRET_KEY = os.environ.get("SECRET_KEY", "production_secret_key")  # Should be set securely in production


# Map configurations for easy reference
config = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
