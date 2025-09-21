
from faster_whisper import WhisperModel
import os


model_size = "base"
model = WhisperModel(model_size, compute_type="auto")

def transcribe_audio(audio_path):
    try:
        print(f"📂 Опит за транскрипция на: {audio_path}")
        segments, info = model.transcribe(audio_path)
        print("ℹ️ Информация за аудиото:", info)

        full_text = ""
        for segment in segments:
            print(f"🗣️ {segment.start:.2f}s - {segment.end:.2f}s: {segment.text}")
            full_text += segment.text.strip() + " "

        print("✅ Финален текст:", full_text.strip())
        return full_text.strip()

    except Exception as e:
        print(f"[ERROR] Faster-Whisper transcription failed: {e}")
        return None

