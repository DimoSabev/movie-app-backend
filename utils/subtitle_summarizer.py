
import openai
import os
import tiktoken  # ВАЖНО: да е най-отгоре с другите импорти
from textwrap import dedent
from utils.actor_lookup import get_actor_name



# Настройване на OpenAI API ключа
openai.api_key = os.getenv("OPENAI_API_KEY")

LANGUAGE_MAP = {
    "en": "English",
    "bg": "Bulgarian",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    # тук можеш да добавяш още езици
}


def summarize_scene(scene_text, movie_name=None, request_id=None, language="en"):
    from app import cancelled_requests
    language_name = LANGUAGE_MAP.get(language, "English")
    if request_id and request_id in cancelled_requests:
        cancelled_requests.discard(request_id)
        raise Exception("Request was cancelled during summarize_scene")

    prompt = f"""
You are a professional movie assistant. Your task is to write a short and coherent third-person summary of all the scenes up to this point in the movie. Focus on what has happened so far — key actions, settings, character interactions, and developments.
Do NOT quote or copy the dialogue. Just describe the scene like you're telling someone what happens in a film.

Movie: {movie_name if movie_name else 'Unknown'}
Scene:
\"\"\"
{scene_text}
\"\"\"
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

class CancelledEarlyException(Exception):
    pass

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def summarize_until_now(scenes, movie_name=None, max_tokens=15000, request_id=None, language="en"):
    scene_summaries = []
    context_so_far = ""
    model = "gpt-3.5-turbo"
    intro_tokens = count_tokens(f'Филмът "{movie_name or "Unknown"}" започва със сцената...\n\n', model=model)
    language_name = LANGUAGE_MAP.get(language, "English")

    for i, scene in enumerate(scenes):
        from app import cancelled_requests
        # 🟢 Прекъсваме *преди* започване на scene[i]
        if request_id and request_id in cancelled_requests:
            print(f"[CANCEL] Прекъсване между сцена {i} и {i + 1} за заявка {request_id}")
            cancelled_requests.discard(request_id)
            raise CancelledEarlyException()

        prompt = dedent(f"""
        You are a professional movie assistant helping summarize scenes for the film "{movie_name or 'Unknown'}".
        
        Your task is to produce the output entirely in {language_name}.  
        Do not include any words or sentences in any other language.  
        If the input is in another language, you must translate it fully into {language_name}.  
        The result must read as if it were originally written in {language_name}, not a translation.

        
        IMPORTANT:
        - Write the entire summary strictly in {language_name} language.
        - Do not use any other language, even partially.
        - Translate and adapt naturally into {language_name} language while fully preserving the meaning, tone, cinematic detail, and clarity of the text.
        - The output must sound like it was originally written in {language_name} language, not like a translation.

        This is Scene {i + 1}.

        Instructions:
        - Mention the location briefly **only when it's new or relevant**. Do not repeat the same setting in every scene.
        - Use short character names, not full names, unless needed for clarity.
        - Avoid repeating information already known from earlier scenes.
        - Add **more specific detail** about what happens (e.g., arguments, decisions, physical actions).
        - 3–6 sentences per scene. Clear. Cinematic. Detailed.
        - Focus on both the emotional impact and the factual details — include real names, details, terms, and developments from the scene.

Do not add commentary or interpretation — just narrate the story as it unfolds, clearly and tightly.

        Context so far:
        {context_so_far}

        Current Scene:
        \"\"\"
        {scene}
        \"\"\"
        """)

        total_tokens_if_added = count_tokens(prompt, model=model) + count_tokens(context_so_far, model=model) + intro_tokens

        if total_tokens_if_added > max_tokens:
            print(f"[LIMIT] Спираме обобщението на Scene {i + 1} – достигнат е лимит от {max_tokens} токена.")
            break

        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6
            )

            # ✅ Проверка веднага след като върне отговор, преди да го обработим
            if request_id and request_id in cancelled_requests:
                print(f"[CANCEL] Прекратен response след scene {i + 1} за заявка {request_id}")
                cancelled_requests.discard(request_id)
                raise Exception(f"Request {request_id} was cancelled after scene {i + 1} response")

            scene_summary = response.choices[0].message.content.strip()
            scene_summaries.append(f"———— Scene {i + 1} ————\n{scene_summary}")
            context_so_far += "\n" + scene_summary

        except Exception as e:
            print(f"[ERROR] Failed to summarize scene {i + 1}: {e}")
            continue

    if not scene_summaries:
        return "⚠️ Сцените са твърде дълги и не могат да бъдат обобщени в рамките на токен лимита."

    introduction = f'Филмът "{movie_name or "Unknown"}" започва със сцената...\n\n'

    final_prompt = dedent(f"""
    Combine all of these scene descriptions into a single structured summary.  
    The entire output must be written strictly in {language_name}.  
    Do not use any other language, even partially.  


    Each scene should:
    - Begin with a divider: "———— Scene X ————"
    - Include vivid but non-repetitive description
    - Reflect a natural narrative progression

    Introduction:
    {introduction}

    Scenes:
    {chr(10).join(scene_summaries)}
    """)

    # 🛑 Проверка преди финалния call
    from app import cancelled_requests
    if request_id and request_id in cancelled_requests:
        print(f"[CANCEL] Финално обобщение прекратено преди отправяне на заявката за {request_id}")
        cancelled_requests.discard(request_id)
        raise CancelledEarlyException()

    try:
        final_response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.6
        )

        # ✅ Проверка след финалния response
        if request_id and request_id in cancelled_requests:
            print(f"[CANCEL] Финалният response е прекратен за заявка {request_id}")
            cancelled_requests.discard(request_id)
            raise CancelledEarlyException()

        return final_response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[ERROR] Failed to generate final summary: {e}")
        return "⚠️ Error summarizing scenes."



import openai
import os
import json
import re

openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_character_profiles(summary_text: str, movie_name: str | None = None, max_characters: int = 10, request_id=None, language="en"):
    language_name = LANGUAGE_MAP.get(language, "English")
    if not summary_text or not isinstance(summary_text, str):
        return []

    from app import cancelled_requests
    if request_id and request_id in cancelled_requests:
        cancelled_requests.discard(request_id)
        raise Exception("Request was cancelled before extract_character_profiles")

    scene_numbers = [int(n) for n in re.findall(r'—+\s*Scene\s+(\d+)\s*—+', summary_text)]
    scene_hint = f"Known scene numbers: {scene_numbers}" if scene_numbers else "No scene markers found."

    prompt = f"""
Extract detailed character profiles from the movie summary below. Focus only on HUMAN characters.
Exclude clubs, places, schools, tools, brands, and non-human entities.

IMPORTANT:
- Write all character descriptions strictly in {language_name}.
- Do not mix languages. Use only {language_name} for the descriptive text.
- Always keep character names in English (do not translate them).
- The text must read as if it was originally written in {language_name}, not as a translation.
- Do not invent characters who are not explicitly present in the summary.
- Every field in the JSON must follow the format strictly. Do not add extra fields or explanations.



Return strict JSON in this format:
{{
  "characters": [
    {{
      "name": "string",
      "role": "string",
      "traits": ["string", "string"],
      "goals": ["string"],
      "relationships": [{{"with":"string","relation":"string","sentiment":"string"}}],
      "description": "Short paragraph describing the character: who they are, what drives them, and what makes them unique."
    }}
  ]
}}

Include up to {max_characters} characters. Use natural short names.
DO NOT include 'mentioned_in_scenes'.
DO NOT include clubs or schools as characters.

Movie: {movie_name or "Unknown"}
{scene_hint}

SUMMARY:
\"\"\"{summary_text}\"\"\"
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You extract structured profiles of characters from movie summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )

        # 🛑 Проверка след response, преди обработка
        if request_id and request_id in cancelled_requests:
            cancelled_requests.discard(request_id)
            raise Exception("Request was cancelled after extract_character_profiles response")

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)

        parsed = json.loads(raw)
        characters = parsed.get("characters", [])

        filtered = []
        for c in characters:
            name = (c.get("name") or "").lower()
            if any(x in name for x in ["club", "facebook", "harvard", "phoenix", "porcellian", "final"]):
                continue

            valid_relationships = []
            for rel in c.get("relationships", []):
                with_name = rel.get("with", "").lower()
                if not any(x in with_name for x in ["club", "facebook", "academic", "pressures", "system", "site"]):
                    valid_relationships.append(rel)
            c["relationships"] = valid_relationships

            try:
                actor_name = get_actor_name(c.get("name", ""), movie_name or "")
                c["actor"] = actor_name if actor_name else None
            except Exception as e:
                print(f"[WARN] Failed to get actor for {c.get('name', '')}: {e}")
                c["actor"] = None

            filtered.append(c)

        return filtered

    except Exception as e:
        print(f"[ERROR extract_character_profiles]: {e}")
        return []