import logging
import os
import subprocess
import time
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from app.services.prediction_service import predict
from app.utils.file_utils import add_to_csv, create_base_csv
from app.errors import SpotifyDownloadError, FeatureExtractionError, PredictionError
import shutil


# Set your Spotify API credentials
load_dotenv()
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")



# Initialize Spotify client
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))


def download_spotify_track(spotify_url):
    """
    Downloads a Spotify track or playlist using spotify-dl and stores it in a specified directory.

    Args:
        spotify_url (str): The URL of the Spotify track or playlist to be downloaded.

    Returns:
        str or None: Path to the downloaded audio file if successful, None otherwise.
        
    Raises:
        SpotifyDownloadError: If the download fails or no audio file is found.
    """
    # Define the output directory
    output_directory = "data/downloaded_songs"
    os.makedirs(output_directory, exist_ok=True)


    try:
        result = subprocess.run(
                ["spotdl", spotify_url, "--output", os.path.join(output_directory, "%(title)s.%(ext)s")],
            )
        

        if result.returncode != 0:
            raise SpotifyDownloadError(
                message="Failed to download the Spotify track.",
                details={"stderr": result.stderr}
            )
            
        folders = [
            f for f in os.listdir(output_directory) 
            if os.path.isdir(os.path.join(output_directory, f))
        ]

        if not folders:
            raise SpotifyDownloadError(
                message="Download completed, but no subfolder was found in the downloaded_songs directory."
            )

        subfolder = max(folders, key=lambda f: os.path.getctime(os.path.join(output_directory, f)))

        audio_extensions = ('.mp3', '.wav', '.flac', '.m4a', '.aac')
        subfolder_path = os.path.join(output_directory, subfolder)
        for root, _, files in os.walk(subfolder_path):
            for file in files:
                if file.endswith(audio_extensions):
                    return os.path.join(root, file)

  
        raise SpotifyDownloadError(
            message="No valid audio file was found after downloading the Spotify track."
        )

    except Exception as e:
        raise SpotifyDownloadError(
            message="An unexpected error occurred while downloading the Spotify track.",
            details={"exception": str(e)}
        )




def get_artist(name):
    """
    Retrieves the first matching artist's information based on the provided name.

    Args:
        name (str): Name of the artist to search for.

    Returns:
        dict: Spotify API response for the first matching artist, or None if no match is found.
    """
    results = sp.search(q='artist:' + name, type='artist')
    items = results['artists']['items']
    return items[0] if items else None


def get_track_name(track_url):
    """
    Retrieves the name of a track from its Spotify URL.

    Args:
        track_url (str): Spotify URL of the track.

    Returns:
        str: Name of the track.
    """
    track = sp.track(track_url)
    return track["name"]


def get_playlist_tracks(playlist_id):
    """
    Retrieves all track URLs from a Spotify playlist.

    Args:
        playlist_id (str): Spotify ID of the playlist.

    Returns:
        list: List of Spotify track URLs from the playlist.
    """
    track_urls = []
    tracks_response = sp.playlist_tracks(playlist_id)
    tracks = tracks_response["items"]

    # Paginate through the playlist tracks
    while tracks_response["next"]:
        tracks_response = sp.next(tracks_response)
        tracks.extend(tracks_response["items"])

    # Extract track URLs
    for track in tracks:
        track_urls.append(track['track']['external_urls']['spotify'])

    return track_urls


def show_album_tracks(album):
    """
    Logs and displays all track names from an album.

    Args:
        album (dict): Album details including its ID.

    Returns:
        None
    """
    tracks = []
    results = sp.album_tracks(album['id'])
    tracks.extend(results['items'])

    # Paginate through album tracks
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])

    for i, track in enumerate(tracks):
        print(f'{i + 1}. {track["name"]}')  # Replace logger.info with print for simplicity


def show_artist_albums(artist):
    """
    Displays all unique albums of an artist by their name.

    Args:
        artist (dict): Artist details including their Spotify ID.

    Returns:
        None
    """
    albums = []
    results = sp.artist_albums(artist['id'], album_type='album')
    albums.extend(results['items'])

    # Paginate through artist albums
    while results['next']:
        results = sp.next(results)
        albums.extend(results['items'])

    # Use a set to avoid duplicate album names
    unique = set()
    for album in albums:
        name = album['name'].lower()
        if name not in unique:
            unique.add(name)
            show_album_tracks(album)


def get_user_playlists(sp, username):
    """
    Retrieves a list of a user's playlists, including their names and URLs.

    Args:
        sp (Spotify): Spotify client object.
        username (str): Spotify username.

    Returns:
        list: List of tuples containing playlist names and their Spotify URLs.
    """
    playlist_info = []
    playlists = sp.user_playlists(username)

    for p in playlists['items']:
        playlist_url = p['external_urls']['spotify']
        playlist_name = p['name'].encode('ascii', 'ignore').decode('ascii')  # Remove non-ASCII characters
        playlist_info.append((playlist_name, playlist_url))

    return playlist_info


def process_song(spotify_url):
    """
    Process a single Spotify song and return its emotional features.
    
    Args:
        spotify_url (str): The URL of the Spotify track
        
    Returns:
        dict: Emotion data.
        
    Raises:
        SpotifyDownloadError: If downloading the track fails.
        FeatureExtractionError: If feature extraction fails.
        PredictionError: If emotion prediction fails.
    """
    # Initialize the CSV file for storing features
    csv_file = 'data/extracted_features/audio_features.csv'
    create_base_csv(csv_file)

    try:
        
        # Download the track
        audio_file_path = download_spotify_track(spotify_url)

        # Extract song name and add features to CSV
        song_name = get_track_name(spotify_url)
        add_to_csv(csv_file, audio_file=audio_file_path, audio_name=song_name)
        
        shutil.rmtree('data/downloaded_songs')
        
        
        
        # Predict emotions
        emotion_data = predict(
            csv_file,
            ['tonnetz', 'chroma', 'rms', 'spec_flux', 'spec_cont', 'spec_band', 'roll_off', 'mfcc']
        )

        return emotion_data

    except SpotifyDownloadError as e:
        raise e  # Propagate the error to the caller
    except Exception as e:
        raise FeatureExtractionError(
            message="An error occurred during feature extraction or CSV processing.",
            details={"exception": str(e)}
        )
    
    
def process_playlist(playlist_url):
    """
    Process all songs in a Spotify playlist and return their emotional features.

    Args:
        playlist_url (str): The URL of the Spotify playlist.

    Returns:
        dict: Emotion data for all songs in the playlist.

    Raises:
        SpotifyDownloadError: If downloading a track fails.
        FeatureExtractionError: If feature extraction fails for a track.
        PredictionError: If emotion prediction fails.
    """
    csv_file = 'data/extracted_features/audio_features.csv'
    create_base_csv(csv_file)

    try:
        # Retrieve all track URLs from the playlist
        track_urls = get_playlist_tracks(playlist_url)

        
        if not track_urls:
            raise SpotifyDownloadError(
                message="No tracks found in the Spotify playlist.",
                details={"playlist_url": playlist_url}
            )

        # Process each song
        for spotify_url in track_urls:
            try:
                
                # Download the track
                audio_file_path = download_spotify_track(spotify_url)

                # Extract song name and add features to CSV
                song_name = get_track_name(spotify_url)
                add_to_csv(csv_file, audio_file=audio_file_path, audio_name=song_name)

            except SpotifyDownloadError as e:
                # Log and continue with the next track
                logging.error(f"Error downloading track {spotify_url}: {e}")
                continue
            except FeatureExtractionError as e:
                # Log and continue with the next track
                logging.error(f"Error extracting features for track {spotify_url}: {e}")
                continue

        shutil.rmtree('data/downloaded_songs')
        # Predict emotions for all tracks
        emotion_data = predict(
            csv_file,
            ['tonnetz', 'chroma', 'rms', 'spec_flux', 'spec_cont', 'spec_band', 'roll_off', 'mfcc']
        )
        return emotion_data

    # except PredictionError as e:
    #     raise PredictionError(
    #         message="Failed to predict emotions for the playlist.",
    #         details={"playlist_url": playlist_url, "exception": str(e)}
    #     )
    except Exception as e:
        raise FeatureExtractionError(
            message="An unexpected error occurred while processing the playlist.",
            details={"playlist_url": playlist_url, "exception": str(e)}
        )

