import faiss
import pickle
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class SimpleEmbedder:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("❌ Липсва OPENAI_API_KEY в .env файла!")
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    def embed_text(self, text):
        if not text or not isinstance(text, str):
            raise ValueError("❌ Не може да се създаде embedding: текстът е празен или невалиден.")
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            print(f"[ERROR] Embedding failed: {e}")
            raise


def load_index_and_mapping(index_path, mapping_path):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"❌ Не е намерен FAISS индекс на пътя: {index_path}")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"❌ Не е намерен pickle mapping файл на пътя: {mapping_path}")

    index = faiss.read_index(index_path)
    with open(mapping_path, "rb") as f:
        mapping = pickle.load(f)
    return index, mapping


def find_best_match(user_text, index, mapping, embedder, top_k=1):
    if not user_text or not isinstance(user_text, str):
        raise ValueError("❌ Входният текст за търсене е празен или невалиден.")

    vector = np.array([embedder.embed_text(user_text)]).astype("float32")
    distances, indices = index.search(vector, top_k)

    results = []
    for i, score in zip(indices[0], distances[0]):
        try:
            scene = mapping[i]
            if isinstance(scene, dict):
                result = scene.copy()
                result["score"] = float(score)
                results.append(result)
            else:
                print(f"⚠️ Очакван речник, но получен тип {type(scene)}")
        except Exception as e:
            print(f"❌ Грешка при обработка на mapping[{i}]: {e}")

    return results


def get_scenes_up_to(timestamp, movie_name, mapping):
    scenes = []
    entries = mapping.values() if isinstance(mapping, dict) else mapping

    def time_to_float(t):
        return float(t.replace(",", ".").replace(":", ""))

    try:
        current_time = time_to_float(timestamp)
    except Exception as e:
        print(f"[ERROR] Неуспешно парсване на текущия timestamp: {timestamp} → {e}")
        return []

    for entry in entries:
        if isinstance(entry, dict) and entry.get("movie") == movie_name:
            try:
                entry_time = time_to_float(entry["timestamp"])
                if entry_time <= current_time:
                    scenes.append((entry_time, entry["lines"]))
            except Exception as e:
                print(f"[⚠️] Грешка при entry: {e}")

    scenes.sort()
    return [lines for _, lines in scenes]

def get_movie_duration(movie_name, mapping):
    latest_time_str = "00:00:00,000"
    latest_time_float = 0.0

    def time_to_float(t):
        try:
            # Заменяме запетая с точка за универсалност
            t = t.replace(",", ".")

            # Разделяме на части: HH:MM:SS или HH:MM:SS.MS
            h, m, s = t.split(":")
            if "." in s:
                s, ms = s.split(".")
            else:
                ms = "0"

            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
        except Exception as e:
            print(f"[⚠️] time_to_float() грешка при '{t}': {e}")
            return 0.0

    for entry in mapping.values():
        if isinstance(entry, dict) and entry.get("movie") == movie_name:
            timestamp = entry.get("timestamp")
            if timestamp:
                current_time_float = time_to_float(timestamp)
                if current_time_float > latest_time_float:
                    latest_time_float = current_time_float
                    latest_time_str = timestamp

    return latest_time_str