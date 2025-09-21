from dotenv import load_dotenv
import os
import pickle
import numpy as np
import faiss
from utils.subtitle_parser import parse_srt
from langchain_community.embeddings import OpenAIEmbeddings
from firebase_utils import sync_subtitles_from_firebase

load_dotenv()
print("FIREBASE_CREDENTIALS_PATH =", os.getenv("FIREBASE_CREDENTIALS_PATH"))
print("BUCKET_NAME =", os.getenv("BUCKET_NAME"))
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("❗ OPENAI_API_KEY не е намерен – провери .env файла!")

SUBTITLES_FOLDER = "subtitles"
INDEX_PATH = "embeddings/subtitle_index.faiss"
MAPPING_PATH = "embeddings/subtitle_mapping.pkl"

embedder = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Зареждаме вече съществуващи индекси и mapping, ако ги има
if os.path.exists(INDEX_PATH) and os.path.exists(MAPPING_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(MAPPING_PATH, "rb") as f:
        mapping = pickle.load(f)
    print(f"🔁 Заредени {len(mapping)} стари embedding-и.")
else:
    index = None
    mapping = {}
    print("⚠️ Не са намерени съществуващи индекси – ще създадем нови.")

vectors = []
new_mapping = {}
start_i = max(mapping.keys(), default=-1) + 1
new_i = start_i

existing_texts = set(item["lines"] for item in mapping.values())

print("🟡 Обработка на нови субтитри...")

sync_subtitles_from_firebase()

for filename in os.listdir(SUBTITLES_FOLDER):
    if filename.endswith(".srt"):
        movie_name = filename[:-4]
        file_path = os.path.join(SUBTITLES_FOLDER, filename)
        scenes = parse_srt(file_path)

        # 🆕 Намираме последния timestamp за този филм
        last_scene_timestamp = scenes[-1]["timestamp"] if scenes else None

        for scene in scenes:
            text = scene["text"].strip()
            timestamp = scene["timestamp"]

            if not text or text in existing_texts:
                continue  # пропускаме дублирани или празни сцени

            try:
                vector = embedder.embed_query(text)
                vectors.append(vector)

                entry = {
                    "lines": text,
                    "timestamp": timestamp,
                    "movie": movie_name
                }

                # 🆕 Добавяме duration само веднъж (при първата сцена на този филм)
                if movie_name not in [m["movie"] for m in new_mapping.values()]:
                    entry["duration"] = last_scene_timestamp  # 🆕

                new_mapping[new_i] = entry
                new_i += 1
                existing_texts.add(text)

            except Exception as e:
                print(f"🔴 Грешка при сцена: {e}")

print(f"🆕 Нови embedding-и: {len(vectors)}")

if vectors:
    vectors_np = np.array(vectors).astype("float32")
    if index is None:
        index = faiss.IndexFlatL2(vectors_np.shape[1])
    index.add(vectors_np)

    # Актуализираме mapping-а
    mapping.update(new_mapping)

    # Записваме
    faiss.write_index(index, INDEX_PATH)
    with open(MAPPING_PATH, "wb") as f:
        pickle.dump(mapping, f)

    print(f"✅ Добавени {len(new_mapping)} нови сцени към индекса.")
else:
    print("ℹ️ Няма нови сцени за добавяне – всичко е актуално.")
