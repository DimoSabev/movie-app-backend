import os
import pickle
import faiss
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from utils.subtitle_parser import parse_srt_file

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


SUBTITLES_DIR = "subtitles"


INDEX_PATH = "embeddings/subtitle_index.faiss"
MAPPING_PATH = "embeddings/subtitle_mapping.pkl"

def main():
    embedder = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    all_embeddings = []
    mapping = {}
    idx = 0

    for file_name in os.listdir(SUBTITLES_DIR):
        if not file_name.endswith(".srt"):
            continue
        movie_name = file_name.replace(".srt", "")
        file_path = os.path.join(SUBTITLES_DIR, file_name)
        subtitles = parse_srt_file(file_path)

        for block in subtitles:
            text = block["text"]
            timestamp = block["timestamp"]
            vector = embedder.embed_query(text)
            all_embeddings.append(vector)

            mapping[idx] = {
                "lines": text,
                "timestamp": timestamp,
                "movie": movie_name
            }
            idx += 1


    dim = len(all_embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(all_embeddings).astype("float32"))


    faiss.write_index(index, INDEX_PATH)
    with open(MAPPING_PATH, "wb") as f:
        pickle.dump(mapping, f)

    print(" Индексът и mapping-а са успешно създадени.")

if __name__ == "__main__":
    main()
