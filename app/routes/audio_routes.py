from flask import Blueprint, request, jsonify
from app.services.audio_service import process_audio

bp = Blueprint('audio_routes', __name__)

@bp.route('/analyze-audio', methods=['POST'])
def analyze_audio():
    """
    Endpoint to analyze a manually uploaded audio file and predict its emotional features.
    """
    data = request.json
    audio_name = data.get('name')
    audio_file = data.get('filePath')

    if not audio_name or not audio_file:
        return jsonify({"error": "Audio name and file path are required"}), 400

    try:
        response = process_audio(audio_name, audio_file)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
