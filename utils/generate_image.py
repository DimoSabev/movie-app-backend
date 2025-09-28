import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_visual_prompt(summary: str, request_id=None) -> str:
    """
    Генерира изключително прост и стилово строго ограничен prompt, подходящ за DALL·E, с фокус върху плоски илюстрации в cartoon стил.
    """
    from app import cancelled_requests
    if request_id and request_id in cancelled_requests:
        print(f"[CANCEL] Прекъсване на функцията generate_visual_prompt за заявка {request_id}")
        cancelled_requests.discard(request_id)
        raise Exception(f"Request {request_id} was cancelled during generate_visual_prompt")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a prompt engineer for DALL·E. Your task is to convert scene summaries into extremely simplified, "
                        "visual prompts for image generation. All outputs must describe flat, 2D cartoon-style illustrations "
                        "with no realistic features. Avoid any photographic realism. Use keywords such as 'flat illustration', "
                        "'2D vector art', 'cartoon style', or 'icon-style' in the prompt. Do not include shadows, textures, "
                        "complex backgrounds or facial details. The prompt must focus on a single action or object in a clear and minimal composition. "
                        "Backgrounds must be simple, solid-colored, or omitted unless essential."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Convert the following movie scene summary into a visual prompt for DALL·E. "
                        f"The resulting image must look like a cartoon or flat illustration, in 2D vector style, suitable for a mobile app. "
                        f"Summary: {summary}"
                    )
                }
            ],
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[ERROR] Prompt generation failed: {e}")
        return "Flat illustration, cartoon-style, 2D vector art of a simple action with minimal background"


def generate_image(prompt: str, size="1024x1024", request_id=None) -> str:
    """
    Изпраща визуален prompt към DALL·E и връща URL на изображението.
    """
    from app import cancelled_requests
    if request_id and request_id in cancelled_requests:
        print(f"[CANCEL] Прекъсване на функцията generate_image за заявка {request_id}")
        cancelled_requests.discard(request_id)
        raise Exception(f"Request {request_id} was cancelled during generate_image")

    try:
        image_response = openai.Image.create(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality="standard",
            n=1
        )
        return image_response["data"][0]["url"]

    except Exception as e:
        print(f"[ERROR] Image generation failed: {e}")
        return None


def generate_images_from_summaries(summaries: list[str], chunk_size: int = 5, size="1024x1024", request_id=None) -> list[dict]:
    """
    Разделя списъка от summary фрагменти на парчета от `chunk_size` и генерира изображение за всяко.
    Връща списък от речници: {index, visual_prompt, image_url}.
    """
    images = []
    from app import cancelled_requests  # ✅ за да проверяваме cancel глобално

    for i in range(0, len(summaries), chunk_size):
        if request_id and request_id in cancelled_requests:
            print(f"[CANCEL] Image generation прекъснато за {request_id}")
            cancelled_requests.discard(request_id)
            break

        chunk = summaries[i:i + chunk_size]
        combined_summary = " ".join(chunk)

        visual_prompt = generate_visual_prompt(combined_summary, request_id=request_id)
        image_url = generate_image(visual_prompt, size=size, request_id=request_id)

        images.append({
            "index": i // chunk_size,
            "visual_prompt": visual_prompt,
            "image_url": image_url
        })

    return images