class AppError(Exception):
    """
    Base class for application-specific errors.
    """
    def __init__(self, message, details=None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self):
        """
        Convert the error to a dictionary for JSON responses.
        """
        return {
            "error": {
                "message": self.message,
                "details": self.details
            }
        }


class SpotifyDownloadError(AppError):
    """
    Raised when a Spotify track or playlist download fails.
    """
    def __init__(self, message="Failed to download Spotify track.", details=None):
        super().__init__(message, details)


class FeatureExtractionError(AppError):
    """
    Raised when feature extraction or CSV processing fails.
    """
    def __init__(self, message="Failed to extract features from the audio file.", details=None):
        super().__init__(message, details)


class PredictionError(AppError):
    """
    Raised when emotion prediction fails.
    """
    def __init__(self, message="Failed to predict emotions.", details=None):
        super().__init__(message, details)
