import requests
import re
from typing import Optional

# 🔐 ВАЖНО: увери се, че този ключ е валиден
TMDB_API_KEY = "f0ff84eb0277780c7c064d1e426acba7"

def get_movie_genre(title: str) -> Optional[str]:
    def try_tmdb_query(title_query: str) -> Optional[dict]:
        try:
            search_url = "https://api.themoviedb.org/3/search/movie"
            params = {"api_key": TMDB_API_KEY, "query": title_query}
            response = requests.get(search_url, params=params).json()
            results = response.get("results", [])
            return results[0] if results else None
        except Exception as e:
            print(f"[Genre TMDb Query ERROR]: {e}")
            return None

    # 🌀 Опит 1: оригинално заглавие
    movie_data = try_tmdb_query(title)

    # 🌀 Опит 2: премахване на "_"
    if not movie_data and "_" in title:
        alt_title = title.replace("_", " ")
        print(f"[Genre Retry] Trying with underscores removed: '{alt_title}'")
        movie_data = try_tmdb_query(alt_title)

    # 🌀 Опит 3: премахване на година от края
    if not movie_data:
        match = re.match(r"^(.*?)(?:\s+|_)?(\d{4})$", title.strip())
        if match:
            title_wo_year = match.group(1).replace("_", " ").strip()
            print(f"[Genre Retry] Trying without year: '{title_wo_year}'")
            movie_data = try_tmdb_query(title_wo_year)

    # ❌ Ако няма резултат
    if not movie_data:
        print(f"[Genre ❌] Could not find movie '{title}' in TMDb")
        return None

    # 🎯 Вземаме genre_ids
    genre_ids = movie_data.get("genre_ids", [])
    if not genre_ids:
        print(f"[Genre ❌] No genres found for movie '{title}'")
        return None

    # ✅ Вземаме списъка с всички жанрове
    try:
        genre_list_url = "https://api.themoviedb.org/3/genre/movie/list"
        response = requests.get(genre_list_url, params={"api_key": TMDB_API_KEY}).json()
        genre_map = {g["id"]: g["name"] for g in response.get("genres", [])}
        genre_names = [genre_map.get(gid, "Unknown") for gid in genre_ids]
        genre_string = ", ".join(genre_names)
        print(f"[Genre ✅] Genres for '{title}': {genre_string}")
        return genre_string
    except Exception as e:
        print(f"[Genre Lookup ERROR]: {e}")
        return None


if __name__ == "__main__":
    test_titles = [
        "Inception",
        "The_Social_Network",
        "The_Social_Network_2010",
        "Avatar",
        "Interstellar",
        "Titanic_1997",
        "SUPERMAN_2025",  # тестов случай, вероятно няма да съществува
        "Shrek_2",
        "The_Lord_of_the_Rings_2001",
        "La_La_Land"
    ]

    for title in test_titles:
        print(f"\n🔍 Testing genre lookup for: {title}")
        genre = get_movie_genre(title)
        print(f"🎬 Result: {genre}")