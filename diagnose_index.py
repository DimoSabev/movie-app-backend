import pickle
import faiss
import os
from collections import defaultdict

# Пътища към файловете (ако са в папка embeddings)
INDEX_PATH = "embeddings/subtitle_index.faiss"
MAPPING_PATH = "embeddings/subtitle_mapping.pkl"

def load_mapping(mapping_path):
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"❌ Mapping файлът не съществува: {mapping_path}")
    with open(mapping_path, "rb") as f:
        return pickle.load(f)

def load_index(index_path):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"❌ FAISS индексът не съществува: {index_path}")
    return faiss.read_index(index_path)

def diagnose():
    print("📦 Зареждане на mapping и индекс...")
    mapping = load_mapping(MAPPING_PATH)
    index = load_index(INDEX_PATH)

    print(f"\n✅ Успешно заредени:")
    print(f" - 🧠 FAISS entries: {index.ntotal}")
    print(f" - 🗂️ Mapping entries: {len(mapping)}\n")

    # Броим сцените по филм
    movie_counts = defaultdict(int)
    for entry in mapping.values():
        movie = entry.get("movie", "❓ unknown")
        movie_counts[movie] += 1

    print("🎬 Сцени по филм:")
    for movie, count in sorted(movie_counts.items(), key=lambda x: -x[1]):
        status = "✅ OK" if count > 0 else "❌ Missing"
        print(f" - {movie}: {count} сцени ({status})")

    # Проверка за съответствие index <-> mapping
    if index.ntotal != len(mapping):
        print("\n⚠️ Несъответствие между индекс и mapping! Могат да се появят грешки при търсене.")

if __name__ == "__main__":
    diagnose()