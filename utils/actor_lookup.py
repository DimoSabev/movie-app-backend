import requests
import re
import json
import openai
from difflib import SequenceMatcher

# 👉 API ключове
import os
openai.api_key = os.getenv("OPENAI_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")


def is_valid_character_name(name: str) -> bool:
    if not name or len(name) < 2:
        return False
    invalids = {"he", "she", "they", "him", "her", "it", "voice", "voice-over"}
    if name.lower() in invalids:
        return False
    if not re.match(r"^[A-Za-z0-9 \-'.]+$", name):
        return False
    if re.match(r"^\d+$", name):
        return False
    if name.strip().lower() in {"the", "a", "an"}:
        return False
    return True

# ✅ TMDb API
def find_actor_tmdb(character_name: str, movie_title: str) -> str | None:
    def try_tmdb_query(title: str) -> dict | None:
        try:
            search_url = "https://api.themoviedb.org/3/search/movie"
            params = {"api_key": TMDB_API_KEY, "query": title}
            response = requests.get(search_url, params=params).json()
            results = response.get("results", [])
            return results[0] if results else None
        except Exception as e:
            print(f"[TMDb Query ERROR]: {e}")
            return None

    def normalize(text: str) -> str:
        return re.sub(r"[^\w\s]", "", text.lower().strip())

    norm_target = normalize(character_name)
    movie_data = try_tmdb_query(movie_title)

    if not movie_data and "_" in movie_title:
        alt_title = movie_title.replace("_", " ")
        print(f"[TMDb Retry] Trying with underscores removed: '{alt_title}'")
        movie_data = try_tmdb_query(alt_title)

    if not movie_data:
        match = re.match(r"^(.*?)(?:\s+|_)?(\d{4})$", movie_title.strip())
        if match:
            title_wo_year = match.group(1).replace("_", " ").strip()
            print(f"[TMDb Retry] Trying without year: '{title_wo_year}'")
            movie_data = try_tmdb_query(title_wo_year)

    if not movie_data:
        print(f"[TMDb] No results for movie '{movie_title}'")
        return None

    movie_id = movie_data["id"]

    try:
        credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits"
        credits_response = requests.get(credits_url, params={"api_key": TMDB_API_KEY}).json()
        cast = credits_response.get("cast", [])

        for actor in cast:
            character_field = actor.get("character", "")
            norm_field = normalize(character_field)

            # Строг match: ако целият character_name е част от пълното име
            if norm_target in norm_field.split() or norm_target in norm_field:
                print(f"[TMDb ✅] Exact/partial match: {character_field} → {actor['name']}")
                return actor["name"]

        # Ако нищо не е намерено, показваме debug информация
        print(f"[TMDb ❌] No match for '{character_name}' in cast list of '{movie_title}'")
        for actor in cast[:10]:  # показваме първите 10 за проверка
            print(f"  ↪️ {actor['name']} as {actor.get('character', '')}")
        return None

    except Exception as e:
        print(f"[TMDb ERROR]: {e}")
        return None

# GPT fallback
def find_actor_gpt(character_name: str, movie_title: str) -> str | None:
    try:
        prompt = f"""
Return only the full name of the actor or actress who plays the character \"{character_name}\" in the movie \"{movie_title}\".
Respond only in this JSON format:

{{
  "actor": "Full Name"
}}

Do not include any other text.
"""
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a movie expert who answers strictly in JSON."},
                {"role": "user", "content": prompt.strip()}
            ],
            temperature=0.2
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\\s*|```$", "", raw)
        data = json.loads(raw)
        return data.get("actor")
    except Exception as e:
        print(f"[GPT fallback error]: {e}")
        return None

# Wrapper
def get_actor_name(character_name: str, movie_title: str) -> str | None:
    if not is_valid_character_name(character_name):
        print(f"[SKIP] '{character_name}' is not a valid character name.")
        return None

    print(f"\n🔍 Looking up: {character_name} in '{movie_title}'")

    actor = find_actor_tmdb(character_name, movie_title)
    if actor:
        return actor

    actor = find_actor_gpt(character_name, movie_title)
    if actor:
        print(f"[GPT] Found actor for {character_name}: {actor}")
        return actor

    print("⚠️ Actor: None")
    return None

