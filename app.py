from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import numpy as np
import subprocess

import json
import openai
from spotify import get_movie_songs
from spotify_search import search_soundtrack

from huggingface_hub import hf_hub_download
os.environ["HF_HUB_OFFLINE"] = "1"

from utils.audio_to_text import transcribe_audio
from utils.subtitle_matcher import (
    load_index_and_mapping,
    find_best_match,
    SimpleEmbedder,
    get_scenes_up_to,  # 🆕 добавено
    get_movie_duration
)
from utils.subtitle_summarizer import (
    summarize_scene,
    summarize_until_now, # 🆕 добавено
    extract_character_profiles
)

from utils.generate_image import generate_images_from_summaries
from utils.search_description import find_best_movie
from utils.actor_lookup import get_actor_name
from utils.genre_lookup import get_movie_genre




load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
SECRET_TOKEN = os.getenv("FIREBASE_SYNC_SECRET")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

INDEX_PATH = "embeddings/subtitle_index.faiss"
MAPPING_PATH = "embeddings/subtitle_mapping.pkl"


@app.route('/')
def home():
    return 'Приложението работи!'

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        input_text = None

        # 🟦 1. JSON със "text"
        if request.is_json:
            data = request.get_json()
            input_text = data.get("text")

        # 🟦 2. Аудио файл
        elif "audio" in request.files:
            audio_file = request.files["audio"]
            audio_path = "temp_audio.wav"
            audio_file.save(audio_path)

            print("🎤 Получен е аудио файл:", audio_path)
            input_text = transcribe_audio(audio_path)
            print("📝 Извлечен текст:", input_text)

        # 🔴 Грешка при липса на текст
        if not input_text:
            return jsonify({"error": "No input text provided"}), 400

        # ✅ Зареждане на индекса и embedding-а в реално време
        index, mapping = load_index_and_mapping(INDEX_PATH, MAPPING_PATH)
        embedder = SimpleEmbedder()

        # 🟩 Търсене на сцена
        matches = find_best_match(input_text, index, mapping, embedder)
        if not matches:
            return jsonify({"error": "No match found"}), 404

        best = matches[0]
        lines = best["lines"]
        movie = best["movie"]
        timestamp = best["timestamp"]

        genre = get_movie_genre(movie)

        duration = get_movie_duration(movie, mapping)

        try:
            # 🔄 Извличане на всички сцени до момента
            scenes_until_now = get_scenes_up_to(timestamp, movie, mapping)
            print(f"[DEBUG] Извлечени {len(scenes_until_now)} сцени до момента за филм {movie}")
            for i, scene in enumerate(scenes_until_now):
                print(f"▶️ Сцена {i + 1}:\n{scene[:200]}...\n")

            # 🧠 Обобщение на всички сцени до момента
            summary = summarize_until_now(scenes_until_now, movie_name=movie)

            # 👤 Генериране на профили на герои
            character_profiles = extract_character_profiles(summary, movie_name=movie)

            # 🧩 Chunk-ване на сцените (напр. на всеки 5 сцени)
            scene_chunks = [
                "\n".join(scenes_until_now[i:i + 5])
                for i in range(0, len(scenes_until_now), 5)
            ]
            print(f"[DEBUG] Създадени {len(scene_chunks)} чънка (по 5 сцени)")

            # 🧠 Обобщаване на всеки чънк
            chunk_summaries = []
            for i, chunk in enumerate(scene_chunks):
                try:
                    chunk_summary = summarize_scene(chunk, movie_name=movie)
                    print(f"[CHUNK {i + 1}] Обобщение: {chunk_summary}")
                    chunk_summaries.append(chunk_summary)
                except Exception as e:
                    print(f"[ERROR] Грешка при обобщаване на чънк {i + 1}: {e}")
                    chunk_summaries.append("⚠️ Неуспешно обобщение на сцени.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            summary = "⚠️ Възникна грешка при обобщението."
            character_profiles = []
            chunk_summaries = []

        from flask import Response
        import json

        response_data = {
            "movie": movie,
            "genre": genre,
            "timestamp": timestamp,
            "duration": duration,  # 🆕 добавено
            "summary_until_now": summary,
            "character_profiles": character_profiles,
            "chunk_summaries": chunk_summaries
        }

        print("[DEBUG] Отговор към клиента:")
        print(json.dumps(response_data, ensure_ascii=False, indent=2))

        return Response(
            json.dumps(response_data, ensure_ascii=False),
            content_type="application/json"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate", methods=["POST"])
def generate_images():
    """
    Приема JSON с summaries (списък от текстови резюмета) и връща списък от генерирани изображения.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data"}), 400

        summaries = data.get("summaries", [])
        size = data.get("size", "1024x1024")

        if not summaries:
            return jsonify({"error": "No summaries provided"}), 400

        images = generate_images_from_summaries(summaries, chunk_size=1, size=size)

        return jsonify({"images": images})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/search_description", methods=["POST"])
def search_description():
    try:
        data = request.get_json()
        description = data.get("description", "").strip()


        if not description:
            return jsonify({"error": "Missing description"}), 400

        result = find_best_movie(description)

        print(f"🔍 Получено описание за търсене: {description}")
        print(f"🎬 Резултат: {result}")

        return jsonify({
            "movie": (
            result.get("matched_movie", "Unknown")[:-4]
            if result.get("matched_movie", "").lower().endswith(".srt")
            else result.get("matched_movie", "Unknown")
            ),
            "distance": result.get("distance"),
            "source": result.get("match_source")
        })



    except Exception as e:
        return jsonify({"error": str(e)}), 500





@app.route('/sync', methods=['POST'])
def trigger_indexing():
    try:
        print("📥 Получена заявка за /sync от Firebase Function.")
        print("🚀 Започва синхронизация и индексиране...")

        # 🧠 Стартиране на index скрипта
        index_result = subprocess.run(
            ["python", "generate_index.py"],
            capture_output=True,
            text=True
        )

        if index_result.returncode != 0:
            print("❌ Скриптът generate_index.py се провали:")
            print(index_result.stderr)
            return jsonify({"status": "error", "message": index_result.stderr}), 500

        print("✅ Индексиране на сцени завършено.")
        print("🧠 Започва генериране на описания...")

        # 🎬 Стартиране на описанията
        desc_result = subprocess.run(
            ["python", "generate_description_embeddings.py"],
            capture_output=True,
            text=True
        )

        if desc_result.returncode != 0:
            print("❌ Скриптът generate_description_embeddings.py се провали:")
            print(desc_result.stderr)
            return jsonify({"status": "error", "message": desc_result.stderr}), 500

        print("✅ Описанията са генерирани успешно.")
        return jsonify({
            "status": "success",
            "index_output": index_result.stdout,
            "desc_output": desc_result.stdout
        }), 200

    except Exception as e:
        print("❌ Грешка при изпълнение на /sync:", e)
        return jsonify({"status": "error", "message": str(e)}), 500




SONG_DB_PATH = "soundtracks.json"

def generate_songs_for_movie(movie_name):
    prompt = (
        f"List all notable songs from the movie '{movie_name}' in the following JSON format:\n\n"
        "[\n"
        "  {\n"
        '    "title": "Song title",\n'
        '    "artist": "Artist name",\n'
        '    "scene": "Scene where the song appears"\n'
        "  }\n"
        "]\n\n"
        "Only include music that actually appeared in the movie. Respond only with valid JSON. No explanation, no markdown."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        content = response["choices"][0]["message"]["content"]
        print("🎧 GPT-4 response:\n", content)

        # опит за декодиране
        songs = json.loads(content)
        return songs
    except Exception as e:
        print("❌ GPT-4 error:", e)
        return []

from spotify_search import search_soundtrack

@app.route("/songs", methods=["GET"])
def get_songs():
    movie_name = request.args.get("movie")
    if not movie_name:
        return jsonify({"error": "Missing 'movie' parameter"}), 400

    # Зареждане на кеш
    if os.path.exists(SONG_DB_PATH):
        with open(SONG_DB_PATH, "r", encoding="utf-8") as f:
            all_songs = json.load(f)
    else:
        all_songs = {}

    if movie_name in all_songs:
        return jsonify({
            "movie": movie_name,
            "songs": all_songs[movie_name],
            "source": "cache"
        })

    # 🧠 Първо опит с GPT
    print(f"🤖 Trying GPT for '{movie_name}'...")
    gpt_songs = generate_songs_for_movie(movie_name)
    if gpt_songs:
        all_songs[movie_name] = gpt_songs
        with open(SONG_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(all_songs, f, indent=2, ensure_ascii=False)
        return jsonify({
            "movie": movie_name,
            "songs": gpt_songs,
            "source": "gpt"
        })

    # 🎵 Fallback към Spotify
    print(f"🎵 GPT unavailable, falling back to Spotify for '{movie_name}'...")
    result = search_soundtrack(movie_name)
    if result:
        all_songs[movie_name] = result["tracks"]
        with open(SONG_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(all_songs, f, indent=2, ensure_ascii=False)
        return jsonify({
            "movie": movie_name,
            "songs": result["tracks"],
            "source": "spotify"
        })

    return jsonify({"error": "No songs found"}), 404



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)