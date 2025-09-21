
import openai
import os
from textwrap import dedent
from utils.actor_lookup import get_actor_name


openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()


def summarize_scene(scene_text, movie_name=None):
    prompt = f"""
You are a professional movie assistant. Your task is to write a short and coherent third-person summary of all the scenes up to this point in the movie. Focus on what has happened so far — key actions, settings, character interactions, and developments.
Do NOT quote or copy the dialogue. Just describe the scene like you're telling someone what happens in a film.

Movie: {movie_name if movie_name else 'Unknown'}
Scene:
\"\"\"
{scene_text}
\"\"\"
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content.strip()


import tiktoken  # ⬅️ ВАЖНО: да е най-отгоре с другите импорти

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def summarize_until_now(scenes, movie_name=None, max_tokens=15000):
    scene_summaries = []
    context_so_far = ""
    model = "gpt-3.5-turbo"
    intro_tokens = count_tokens(f'Филмът "{movie_name or "Unknown"}" започва със сцената...\n\n', model=model)

    for i, scene in enumerate(scenes):
        prompt = dedent(f"""
        You are a professional movie assistant helping summarize scenes for the film "{movie_name or 'Unknown'}".

        This is Scene {i + 1}.

        Instructions:
        - Mention the location briefly **only when it's new or relevant**. Do not repeat the same setting in every scene.
        - Use short character names (Mark, Eduardo), not full names, unless needed for clarity.
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
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6
            )
            scene_summary = response.choices[0].message.content.strip()
            scene_summaries.append(f"———— Scene {i + 1} ————\n{scene_summary}")
            context_so_far += "\n" + scene_summary

        except Exception as e:
            print(f"[ERROR] Failed to summarize scene {i + 1}: {e}")
            continue

    if not scene_summaries:
        return "⚠️ Сцените са твърде дълги и не могат да бъдат обобщени в рамките на токен лимита."

    # Въведение преди всички сцени
    introduction = f'Филмът "{movie_name or "Unknown"}" започва със сцената...\n\n'

    final_prompt = dedent(f"""
    Combine all of these scene descriptions into a single structured summary.

    Each scene should:
    - Begin with a divider: "———— Scene X ————"
    - Include vivid but non-repetitive description
    - Reflect a natural narrative progression

    Introduction:
    {introduction}

    Scenes:
    {chr(10).join(scene_summaries)}
    """)

    try:
        final_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.6
        )
        return final_response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[ERROR] Failed to generate final summary: {e}")
        return "⚠️ Error summarizing scenes."


import openai
import os
import json
import re

openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_character_profiles(summary_text: str, movie_name: str | None = None, max_characters: int = 10):
    """
    Извлича персонажи от готовия summary, с кратко разказно описание вместо 'evidence'.
    Премахва 'mentioned_in_scenes' и филтрира не-човешки обекти.
    """

    if not summary_text or not isinstance(summary_text, str):
        return []

    # За контекст, ако има сцени
    scene_numbers = [int(n) for n in re.findall(r'—+\s*Scene\s+(\d+)\s*—+', summary_text)]
    scene_hint = f"Known scene numbers: {scene_numbers}" if scene_numbers else "No scene markers found."

    prompt = f"""
Extract detailed character profiles from the movie summary below. Focus only on HUMAN characters.
Exclude clubs, places, schools, tools, brands, and non-human entities.

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

Include up to {max_characters} characters. Use natural short names (e.g., Mark, Eduardo).
DO NOT include 'mentioned_in_scenes'.
DO NOT include clubs or schools as characters.

Movie: {movie_name or "Unknown"}
{scene_hint}

SUMMARY:
\"\"\"{summary_text}\"\"\"
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You extract structured profiles of characters from movie summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)

        parsed = json.loads(raw)
        characters = parsed.get("characters", [])

        # Филтриране на не-човешки имена
        # Филтриране на не-човешки имена
        filtered = []
        for c in characters:
            name = (c.get("name") or "").lower()
            if any(x in name for x in ["club", "facebook", "harvard", "phoenix", "porcellian", "final"]):
                continue

            # Филтрирай само човешки relationships
            valid_relationships = []
            for rel in c.get("relationships", []):
                with_name = rel.get("with", "").lower()
                if not any(x in with_name for x in ["club", "facebook", "academic", "pressures", "system", "site"]):
                    valid_relationships.append(rel)
            c["relationships"] = valid_relationships

            # 🆕 Добавяне на actor, използвайки get_actor_name
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