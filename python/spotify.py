import json
from dotenv import load_dotenv
import os
import base64
from requests import post
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

load_dotenv()

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")








def get_artist(name):
    results = sp.search(q='artist:' + name, type='artist')
    items = results['artists']['items']
    if len(items) > 0:
        return items[0]
    else:
        return None

 


def show_album_tracks(album):
    tracks = []
    results = sp.album_tracks(album['id'])
    tracks.extend(results['items'])
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    for i, track in enumerate(tracks):
        logger.info('%s. %s', i + 1, track['name'])


def show_artist_albums(artist):
    albums = []
    results = sp.artist_albums(artist['id'], album_type='album')
    albums.extend(results['items'])
    while results['next']:
        results = sp.next(results)
        albums.extend(results['items'])
   
    unique = set()  # skip duplicate albums
    for album in albums:
        name = album['name'].lower()
        if name not in unique:
            
            unique.add(name)
            show_album_tracks(album)
    
# all_artist_songs(sp)

def get_user_playlists(sp, username):
    '''
        Function: Get a list of users playlist names and their url
        Returns: A list of tuples: (playlist name, playlist url)
    '''

    playlist_info = []
    playlists = sp.user_playlists(username)

    for p in playlists['items']: # type: ignore
        playlist_url =  p['external_urls']['spotify']
        playlist_name = p['name'].encode('ascii', 'ignore').decode('ascii')

        playlist_info.append( (playlist_name, playlist_url) )
        

    return playlist_info

# def get_playlist_url(playlist_name, ):



if __name__=="__main__": 
    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

    print(get_user_playlists(sp, 'chrisdearman15'))

    # results = sp.playlist( '37i9dQZF1DX1lVhptIYRda', fields='name')
    # print(results)

    # playlist = sp.featured_playlists(country='US')
    # print(playlist)
    # print(sp.featured_playlists(country='US'))

    # spotify_categories = sp.categories(country='US')
    # print(spotify_categories['categories']['items'])
    # print(spotify_categories['categories']['items'][1])


    # name = 'Drake'
    # results = sp.search(q='artist:' + name, type='artist')
    # items = results['artists']['items']
    # print(items)
    