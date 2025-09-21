import os
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")


def get_spotify_token():
    auth_str = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
    b64_auth_str = base64.b64encode(auth_str.encode()).decode()

    headers = {
        "Authorization": f"Basic {b64_auth_str}",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    data = {
        "grant_type": "client_credentials"
    }

    response = requests.post("https://accounts.spotify.com/api/token", headers=headers, data=data)

    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        print("❌ Spotify token error:", response.text)
        return None


def search_spotify(query, type="album"):
    token = get_spotify_token()
    if not token:
        return []

    headers = {
        "Authorization": f"Bearer {token}"
    }

    params = {
        "q": query,
        "type": type,
        "limit": 5
    }

    response = requests.get("https://api.spotify.com/v1/search", headers=headers, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print("❌ Spotify search error:", response.text)
        return []

def get_movie_songs(movie_name):
    # Търси албум със същото име като филма
    result = search_spotify(movie_name, type="album")

    if not result or "albums" not in result or not result["albums"]["items"]:
        print("❌ No album found for movie:", movie_name)
        return []

    album = result["albums"]["items"][0]
    album_id = album["id"]

    # Вземи песните от албума
    token = get_spotify_token()
    headers = {
        "Authorization": f"Bearer {token}"
    }

    tracks_response = requests.get(
        f"https://api.spotify.com/v1/albums/{album_id}/tracks",
        headers=headers
    )

    if tracks_response.status_code != 200:
        print("❌ Error fetching tracks:", tracks_response.text)
        return []

    tracks = tracks_response.json()["items"]

    # Подготви структурата
    song_data = []
    for track in tracks:
        song_data.append({
            "title": track["name"],
            "artist": track["artists"][0]["name"] if track["artists"] else "Unknown",
            "scene": "N/A"  # Нямаме сцена от Spotify
        })

    return song_data