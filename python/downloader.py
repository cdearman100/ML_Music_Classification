# Downloads a Spotify playlist into a folder of MP3 tracks
# Import dependencies
import os
import spotipy
import spotipy.oauth2 as oauth2
import yt_dlp
from youtube_search import YoutubeSearch
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials
# from sclib import SoundcloudAPI
import multiprocessing
from pydub import AudioSegment
# import environment variable dependencies
from dotenv import load_dotenv


load_dotenv()
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

def write_tracks(text_file: str, tracks: dict):
    # Writes the information of all tracks in the playlist to a text file. 
    # This includes the name, artist, and spotify URL. Each is delimited by a comma.
    with open(text_file, 'w+', encoding='utf-8') as file_out:
        
        while True:
            if tracks.get('items') == None:
                break
            for item in tracks['items']:
                if 'track' in item:
                    track = item['track']
                else:
                    track = item
                try:
                    
                    track_url = track['external_urls']['spotify']
                    track_name = track['name']
                    track_artist = track['artists'][0]['name']
                    csv_line = track_name + "," + track_artist + "," + track_url + "\n"
                    try:
                        file_out.write(csv_line)
                    except UnicodeEncodeError:  # Most likely caused by non-English song names
                        print("Track named {} failed due to an encoding error. This is \
                            most likely due to this song having a non-English name.".format(track_name))
                except KeyError:
                    print(u'Skipping track {0} by {1} (local only?)'.format(
                            track['name'], track['artists'][0]['name']))
                
            # 1 page = 50 results, check if there are more pages
            if tracks['next']:
                tracks = spotify.next(tracks)
            else:
                break
            



def write_playlist(playlist_id: str):



    results = spotify.playlist( playlist_id, fields='tracks,next,name')
    playlist_name = results['name']
    text_file = u'{0}.txt'.format(playlist_name, ok='-_()[]{}')
    print(u'Writing {0} tracks to {1}.'.format(results['tracks']['total'], text_file))
    tracks = results['tracks']
    write_tracks(text_file, tracks)
    return playlist_name



def find_and_download_songs_spotify(reference_file: str):
    TOTAL_ATTEMPTS = 10
    with open(reference_file, "r", encoding='utf-8') as file:
        for line in file:
            temp = line.split(",")
            name, artist = temp[0], temp[1]
            text_to_search = artist + " - " + name
            best_url = None
            attempts_left = TOTAL_ATTEMPTS
            while attempts_left > 0:
                try:
                    results_list = YoutubeSearch(text_to_search, max_results=1).to_dict()
                    best_url = "https://www.youtube.com{}".format(results_list[0]['url_suffix'])
                    break
                except IndexError:
                    attempts_left -= 1
                    print("No valid URLs found for {}, trying again ({} attempts left).".format(
                        text_to_search, attempts_left))
            if best_url is None:
                print("No valid URLs found for {}, skipping track.".format(text_to_search))
                continue
            # Run you-get to fetch and download the link's audio
            print("Initiating download for {}.".format(text_to_search))
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',        # wav + tremove preferredquality
                    'preferredquality': '320',
                }],
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([best_url])

               
# re-organize the files
def reorganize_files(folder_name):
    folder_path = os.getcwd()
    # loop through all files in the directory
    for filename in os.listdir(folder_path):
        # check if the file is an mp3 file
        if filename.endswith(".mp3"):
            if " - " in filename:
                artist, title = filename.split(" - ")[0], filename.split(" - ")[1]
                title = title[:-18]
                new_filename = f"{title} - {artist}.mp3"
            else:
                title = filename.split(" - ")[0][:-18]
                new_filename = f"{title}.mp3"
            # rename the file
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))

def download_playlist(client_id, client_secret,playlist_uri):
    auth_manager = oauth2.SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    spotify = spotipy.Spotify(auth_manager=auth_manager)
    playlist_name = write_playlist(playlist_uri)
    reference_file = "{}.txt".format(playlist_name)
    # Create the playlist folder
    if not os.path.exists(playlist_name):
        os.makedirs(playlist_name)
    os.rename(reference_file, playlist_name + "/" + reference_file)
    os.chdir(playlist_name)

    find_and_download_songs_spotify(reference_file)
    print("Mp3 Playlist Created.")
    print("Proceeding with reorganizing the files and converting them to WAV quality.")
   

    reorganize_files(playlist_name)


    folder_path = os.getcwd()
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp3"):
            file_path = folder_path + '/' + filename
            sound = AudioSegment.from_mp3(str(file_path))
            sound.export(folder_path + '/' + filename[:-4] + ".wav", format="wav")
            os.remove(str(file_path))

    print("Operation Complete!")


if __name__ == "__main__":
    print('hello world')



 