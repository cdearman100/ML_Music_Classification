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

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


import spotify
import downloader
import preproccesor




if __name__=="__main__": 
    feature_list = ['tonnetz', 'spec_flux', 'spec_cont', 'spec_cent', 'roll_off', 'zcr']
    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

    playlist = spotify.get_user_playlists(sp, 'chrisdearman15')

    url = ''

   
    
   
    downloader.download_playlist(client_id, client_secret, playlist[0][1])

    # preproccesor.write_csv(feature_list, 'Angry test playlist')
    # print(preproccesor.predict( 'Angry test playlist_features.csv'))
        

        




