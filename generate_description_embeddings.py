import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import openai

# 🔐 Зареждане на OpenAI API ключ
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 📁 Пътища
SUBTITLES_DIR = "subtitles"
INDEX_PATH = "description_embeddings/description_index.faiss"
MAPPING_PATH = "description_embeddings/description_mapping.pkl"

# 🔁 Firebase sync (ново!)
from firebase_utils import sync_subtitles_from_firebase


def get_embedding_model():
    # Зареждане на модела вътре във функция, за да избегнем segmentation fault
    return SentenceTransformer("all-MiniLM-L6-v2")


def generate_full_description_from_title(filename):
    title = filename.replace("_", " ").replace(".srt", "").strip()
    prompt = f"""
Generate a highly accurate, vivid, and concise 2-3 sentence description of the movie titled '{title}'.
Do not include the title in your response. The description should help users recognize the movie even from vague or short queries.
Focus on key plot elements, characters, and unique traits.
""".strip()

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error generating description for {filename}: {e}")
        return None


def main():
    print("🔄 Синхронизиране със Firebase...")
    sync_subtitles_from_firebase()

    # 🧠 Зареждане на embedding модела
    print("🧠 Зареждане на SentenceTransformer...")
    model = get_embedding_model()

    # 🧠 Създаване или зареждане на FAISS индекс и mapping
    if os.path.exists(INDEX_PATH) and os.path.exists(MAPPING_PATH):
        print("📂 Зареждане на съществуващ индекс и mapping...")
        faiss_index = faiss.read_index(INDEX_PATH)
        with open(MAPPING_PATH, "rb") as f:
            mapping = pickle.load(f)
    else:
        print("🆕 Създаване на нов индекс и mapping...")
        faiss_index = faiss.IndexFlatL2(384)
        mapping = {}

    # 🔄 Обработка на .srt файлове
    for filename in os.listdir(SUBTITLES_DIR):
        if filename.endswith(".srt") and filename not in mapping:
            print(f"\n🎬 Обработка на {filename}...")
            description = generate_full_description_from_title(filename)
            if description:
                print(f"✅ Описание: {description}")
                try:
                    embedding = model.encode([description])[0]
                    faiss_index.add(np.array([embedding], dtype=np.float32))
                    mapping[filename] = {"description": description}
                except Exception as e:
                    print(f"❌ Грешка при embedding за {filename}: {e}")
            else:
                print("⚠️ Пропуснат поради грешка.")

    # 💾 Записване
    os.makedirs("description_embeddings", exist_ok=True)
    faiss.write_index(faiss_index, INDEX_PATH)
    with open(MAPPING_PATH, "wb") as f:
        pickle.dump(mapping, f)

    print("\n🎉 Готово! Индексът и описанията са обновени.")


if __name__ == "__main__":
    main()