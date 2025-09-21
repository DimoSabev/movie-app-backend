import os
import pickle
import faiss
import openai
import numpy as np
from sentence_transformers import SentenceTransformer

# 🔑 Настройка на OpenAI API ключ
openai.api_key = os.getenv("OPENAI_API_KEY")

# 🧠 Зареждане на embedding модела (SentenceTransformer)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 📁 Пътища към файловете
EMBEDDINGS_PATH = "description_embeddings/description_index.faiss"
MAPPING_PATH = "description_embeddings/description_mapping.pkl"

# ✅ Проверка за съществуване
if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(MAPPING_PATH):
    raise FileNotFoundError("❌ FAISS индекс или mapping файл не са налични. Увери се, че си ги генерирал първо.")

# 🧠 Зареждане на индекса и описанията
def load_description_index():
    faiss_index = faiss.read_index(EMBEDDINGS_PATH)
    with open(MAPPING_PATH, "rb") as f:
        title_to_data = pickle.load(f)

    filenames = list(title_to_data.keys())
    descriptions = [data["description"] for data in title_to_data.values()]

    return faiss_index, filenames, descriptions, title_to_data

# 🔍 Търсене по embedding
def search_movie_by_description(description):
    faiss_index, filenames, descriptions, title_to_data = load_description_index()

    query_embedding = embedding_model.encode([description])
    D, I = faiss_index.search(np.array(query_embedding).astype("float32"), k=1)

    distance = D[0][0]
    index = I[0][0]

    if 0 <= index < len(filenames):
        matched_file = filenames[index]
        print(f"🎯 Най-близко съвпадение: {matched_file} с разстояние {distance:.4f}")
    else:
        matched_file = None

    return matched_file, distance

# 🧠 GPT-4 fallback (съвместим с по-стар OpenAI SDK)
def fallback_gpt(description):
    prompt = f"""
    A user described a movie as follows: "{description}"
    Suggest one real, well-known movie that best fits the description. Just return the name of the movie, nothing else.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt.strip()}
            ],
            max_tokens=20,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ GPT-4 fallback error: {e}")
        return "Unknown"

# 🎬 Главна функция
def find_best_movie(description, threshold=1.25, max_distance=1.5):
    best_match, dist = search_movie_by_description(description)

    if best_match and dist < threshold:
        print(f"✅ Best match (distance {dist:.2f}): {best_match}")
        return {
            "matched_movie": best_match,
            "match_source": "embedding",
            "distance": round(float(dist), 4)
        }

    elif dist <= max_distance:
        print(f"⚠️ Distance {dist:.2f} above threshold but within fallback range. Trying GPT-4...")
        gpt_response = fallback_gpt(description)
        return {
            "matched_movie": gpt_response,
            "match_source": "gpt4_fallback_due_to_uncertainty",
            "distance": round(float(dist), 4)
        }

    else:
        print(f"❌ No reliable match (distance {dist:.2f}). Fallback to GPT-4 only.")
        gpt_response = fallback_gpt(description)
        return {
            "matched_movie": gpt_response,
            "match_source": "gpt4_fallback_only",
            "distance": None
        }

# 🖨️ Показване на наличните описания
def print_available_movies():
    _, _, _, title_to_data = load_description_index()
    print("\n🎞️ Филми в индекса:")
    for filename, data in title_to_data.items():
        short_desc = data.get("description", "")[:100].strip()
        print(f"{filename} → {short_desc}")

# 🚀 Тест
if __name__ == "__main__":
    print_available_movies()
    user_input = input("📝 Describe the movie: ")
    result = find_best_movie(user_input)
    print(f"\n🎬 Result: {result}")