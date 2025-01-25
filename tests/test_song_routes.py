import pytest
from unittest.mock import patch
from app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    app.config['DEBUG'] = True
    with app.test_client() as client:
        yield client


@patch("app.services.spotify_service.download_spotify_track")
@patch("app.services.spotify_service.get_track_name")
@patch("app.services.spotify_service.add_to_csv")
@patch("app.services.spotify_service.predict")
@patch("app.services.spotify_service.create_base_csv")
def test_analyze_spotify_song_success(
    mock_create_base_csv,
    mock_predict,
    mock_add_to_csv,
    mock_get_track_name,
    mock_download_spotify_track,
    client,
):
    mock_create_base_csv.return_value = None
    mock_download_spotify_track.return_value = "mock_audio_file_path"
    mock_get_track_name.return_value = "mock_song_name"
    mock_add_to_csv.return_value = None
    mock_predict.return_value = {"mock_song_name": [0.1, 0.2, 0.3, 0.4, 0.5]}

    response = client.post(
        "/analyze-spotify-song",
        json={"spotifyUrl": "https://open.spotify.com/track/mock_url"},
    )

    assert response.status_code == 200
    data = response.get_json()
    assert "mock_song_name" in data
    assert isinstance(data["mock_song_name"], list)


def test_analyze_spotify_song_missing_url(client):
    response = client.post("/analyze-spotify-song", json={})
    assert response.status_code == 400  # Bad Request
    data = response.get_json()
    assert "error" in data


@patch("app.services.spotify_service.download_spotify_track", side_effect=Exception("Download error"))
def test_analyze_spotify_song_download_error(mock_download_spotify_track, client):
    response = client.post(
        "/analyze-spotify-song",
        json={"spotifyUrl": "https://open.spotify.com/track/mock_url"},
    )
    assert response.status_code == 500  # Internal Server Error
    data = response.get_json()
    assert "error" in data
