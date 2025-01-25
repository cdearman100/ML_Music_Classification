from flask import Blueprint, request, jsonify
from app.services.spotify_service import process_song
from app.errors import SpotifyDownloadError, FeatureExtractionError, PredictionError
import os

bp = Blueprint('song_routes', __name__)

@bp.route('/analyze-spotify-song', methods=['POST'])
def analyze_song():
    """
    Endpoint to analyze a single Spotify song and predict its emotional features.

    Returns:
        Response: JSON object containing predicted emotional features.
    """
    # Parse the request data
    data = request.json
    spotify_url = data.get('spotifyUrl')

    if not spotify_url:
        return jsonify({"error": "Spotify URL is required"}), 400
    
    try:
        emotion_data = process_song(spotify_url)
        return jsonify(emotion_data)
    except SpotifyDownloadError as e:
        return jsonify(e.to_dict()), 500
    except FeatureExtractionError as e:
        return jsonify(e.to_dict()), 500
    # except PredictionError as e:
    #     return jsonify(e.to_dict()), 500
    except Exception as e:
        return jsonify({
            "error": {
                "message": "An unexpected error occurred.",
                "details": str(e)
            }
        }), 500
