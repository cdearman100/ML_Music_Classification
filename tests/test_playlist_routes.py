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


@patch("app.services.spotify_service.get_playlist_tracks")
@patch("app.services.spotify_service.download_spotify_track")
@patch("app.services.spotify_service.add_to_csv")
@patch("app.services.spotify_service.predict")
@patch("app.services.spotify_service.create_base_csv")
def test_analyze_playlist_success(
    mock_create_base_csv,
    mock_predict,
    mock_add_to_csv,
    mock_download_spotify_track,
    mock_get_playlist_tracks,
    client,
):
    mock_create_base_csv.return_value = None
    mock_get_playlist_tracks.return_value = [
        "https://open.spotify.com/track/mock_track_1",
        "https://open.spotify.com/track/mock_track_2",
    ]
    mock_download_spotify_track.return_value = "mock_audio_file_path"
    mock_add_to_csv.return_value = None
    mock_predict.return_value = {
        "mock_track_1": [0.1, 0.2, 0.3],
        "mock_track_2": [0.4, 0.5, 0.6],
    }

    response = client.post(
        "/analyze-spotify-playlist",
        json={"playlistUrl": "https://open.spotify.com/playlist/mock_playlist_url"},
    )

    print('\n')
    print(response.data.decode())
    print('\n')
    assert response.status_code == 200
    data = response.get_json()
    assert "mock_track_1" in data
    assert "mock_track_2" in data


def test_analyze_playlist_missing_url(client):
    response = client.post("/analyze-spotify-playlist", json={})
    assert response.status_code == 400  # Bad Request
    data = response.get_json()
    assert "error" in data


@patch("app.services.spotify_service.get_playlist_tracks", side_effect=Exception("Playlist fetch error"))
def test_analyze_playlist_fetch_error(mock_get_playlist_tracks, client):
    response = client.post(
        "/analyze-spotify-playlist",
        json={"playlistUrl": "https://open.spotify.com/playlist/mock_playlist_url"},
    )
    assert response.status_code == 500  # Internal Server Error
    data = response.get_json()
    assert "error" in data
