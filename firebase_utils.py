import os
import firebase_admin
from firebase_admin import credentials, storage
from dotenv import load_dotenv

# 🟢 Първо зареди .env файла
load_dotenv()

# 🔑 Сега вече .env стойностите ще се прочетат коректно
FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH")
BUCKET_NAME = os.getenv("BUCKET_NAME")
LOCAL_SUBTITLE_DIR = "subtitles"

# 🛡️ Инициализирай Firebase само ако не е вече
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
    firebase_admin.initialize_app(cred, {
        'storageBucket': BUCKET_NAME
    })

# 🔎 Тестване: Изведи списък с blob-ове (само за отстраняване на грешки)
def test_list_blobs():
    bucket = storage.bucket()
    blobs = list(bucket.list_blobs())
    print(f"✅ FOUND {len(blobs)} FILES:")
    for blob in blobs:
        print(blob.name)

# 🔁 Основна функция
def sync_subtitles_from_firebase():
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix="subtitles/")
    os.makedirs(LOCAL_SUBTITLE_DIR, exist_ok=True)
    print("🔐 FIREBASE_CREDENTIALS_PATH:", FIREBASE_CREDENTIALS_PATH)
    print("🪣 BUCKET:", bucket.name)
    print("📦 STORAGE LIB FROM:", storage.__file__)

    downloaded = []

    for blob in blobs:
        if blob.name.endswith(".srt"):
            filename = os.path.basename(blob.name)
            local_path = os.path.join(LOCAL_SUBTITLE_DIR, filename)
            if not os.path.exists(local_path):
                blob.download_to_filename(local_path)
                print(f"⬇️ Свален нов файл: {filename}")
                downloaded.append(filename)

    if not downloaded:
        print("✅ Няма нови субтитри за сваляне.")

    return downloaded