import json
import os
from groq import Groq

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


SYSTEM_PROMPT = """You are an expert at distinguishing AI-generated text from human-written text.
Analyze the provided text and return ONLY a JSON object with this exact structure:
{"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}

Where score is:
- 0.0 = clearly human-written
- 1.0 = clearly AI-generated
- 0.5 = uncertain / could be either

Return only the JSON object, no other text."""

IMAGE_SYSTEM_PROMPT = """You are an expert at distinguishing AI-generated image descriptions from human-written captions.
AI-generated alt text tends to be formulaic: "A [adjective] [noun] [verb]ing in/on a [noun]", uses sweeping
landscape language ("serene", "majestic", "crystal-clear"), and sounds like stock photo metadata.
Human captions are informal, personal, specific, often incomplete sentences.

Analyze the provided image description and return ONLY a JSON object with this exact structure:
{"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}

Where score is:
- 0.0 = clearly human-written caption
- 1.0 = clearly AI-generated alt text / description
- 0.5 = uncertain

Return only the JSON object, no other text."""


def detect_with_llm(text: str, prompt_type: str = "text") -> dict:
    """Returns {"score": float, "reasoning": str, "raw": str}."""
    client = _get_client()
    system = IMAGE_SYSTEM_PROMPT if prompt_type == "image_description" else SYSTEM_PROMPT
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Analyze this text:\n\n{text}"},
        ],
        temperature=0.1,
        max_tokens=100,
    )
    raw = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw)
        score = float(parsed["score"])
        score = max(0.0, min(1.0, score))
        return {"score": score, "reasoning": parsed.get("reasoning", ""), "raw": raw}
    except (json.JSONDecodeError, KeyError, ValueError):
        # Fallback: treat unparseable response as uncertain
        return {"score": 0.5, "reasoning": "Could not parse LLM response.", "raw": raw}
