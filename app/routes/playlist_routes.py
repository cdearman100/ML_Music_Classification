from flask import Blueprint, request, jsonify
from app.services.spotify_service import process_playlist
from app.errors import SpotifyDownloadError, FeatureExtractionError, PredictionError

bp = Blueprint('playlist_routes', __name__)

@bp.route('/analyze-spotify-playlist', methods=['POST'])
def analyze_playlist():
    data = request.json
    playlist_url = data.get('playlistUrl')

    if not playlist_url:
        return jsonify({"error": "Playlist URL is required"}), 400

    try:
        emotion_data = process_playlist(playlist_url)
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
