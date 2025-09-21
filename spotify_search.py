import os
import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials

load_dotenv()

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
))

def build_queries(movie_name):
    base = movie_name.strip()
    return [
        f"{base} Original Motion Picture Soundtrack",
        f"{base} soundtrack",
        f"{base} movie soundtrack",
        f"{base} OST",
        f"{base} 2025 soundtrack"
    ]


def search_soundtrack(movie_name):
    queries = build_queries(movie_name)
    for query in queries:
        results = sp.search(q=query, type='album', limit=50)
        for album in results['albums']['items']:
            album_name = album['name'].lower()

            # Основна проверка дали това е саундтрак
            if any(keyword in album_name for keyword in ["soundtrack", "original motion picture", "ost"]):
                album_id = album['id']
                tracks = sp.album_tracks(album_id)
                return {
                    "album": album['name'],
                    "artist": album['artists'][0]['name'],
                    "tracks": [{
                        "title": t["name"],
                        "artist": t["artists"][0]["name"],
                        "scene": "",
                        "duration_ms": t["duration_ms"]
                    } for t in tracks['items']]
                }
    return None