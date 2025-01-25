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


@patch("app.services.audio_service.add_to_csv")
@patch("app.services.audio_service.predict")
@patch("app.services.audio_service.create_base_csv")
def test_analyze_audio_success(
    mock_create_base_csv,
    mock_predict,
    mock_add_to_csv,
    client,
):
    mock_create_base_csv.return_value = None
    mock_add_to_csv.return_value = None
    mock_predict.return_value = {"mock_audio": [0.1, 0.2, 0.3, 0.4]}

    response = client.post(
        "/analyze-audio",
        json={"name": "mock_audio", "filePath": "mock_file_path"},
    )

    assert response.status_code == 200
    data = response.get_json()
    assert "mock_audio" in data
    assert isinstance(data["mock_audio"], list)


def test_analyze_audio_missing_data(client):
    response = client.post("/analyze-audio", json={})
    assert response.status_code == 400  # Bad Request
    data = response.get_json()
    assert "error" in data


@patch("app.services.audio_service.add_to_csv", side_effect=Exception("CSV error"))
def test_analyze_audio_csv_error(mock_add_to_csv, client):
    response = client.post(
        "/analyze-audio",
        json={"name": "mock_audio", "filePath": "mock_file_path"},
    )
    assert response.status_code == 500  # Internal Server Error
    data = response.get_json()
    assert "error" in data
