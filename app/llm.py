"""
Thin single-turn completion wrapper over Groq/Gemini, selected via
LLM_PROVIDER in .env. No multi-turn chat state to manage, so the whole
abstraction is just chat_completion(system, user, json_mode).
"""
import json
from functools import lru_cache

from app.config import (
    LLM_PROVIDER,
    GROQ_API_KEY,
    GROQ_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)


class LLMError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _groq_client():
    from groq import Groq

    if not GROQ_API_KEY:
        raise LLMError("GROQ_API_KEY is not set. Add it to your .env file.")
    return Groq(api_key=GROQ_API_KEY)


@lru_cache(maxsize=1)
def _gemini_model():
    import google.generativeai as genai

    if not GEMINI_API_KEY:
        raise LLMError("GEMINI_API_KEY is not set. Add it to your .env file.")
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(GEMINI_MODEL)


def _call_groq(system: str, user: str, temperature: float, json_mode: bool) -> str:
    client = _groq_client()
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            **kwargs,
        )
    except Exception as exc:  # network / API errors
        raise LLMError(f"Groq API call failed: {exc}") from exc
    return response.choices[0].message.content


def _call_gemini(system: str, user: str, temperature: float, json_mode: bool) -> str:
    model = _gemini_model()
    generation_config = {"temperature": temperature}
    if json_mode:
        generation_config["response_mime_type"] = "application/json"
    prompt = f"{system}\n\n{user}"
    try:
        response = model.generate_content(prompt, generation_config=generation_config)
    except Exception as exc:
        raise LLMError(f"Gemini API call failed: {exc}") from exc
    return response.text


def chat_completion(system: str, user: str, temperature: float = 0.0, json_mode: bool = False) -> str:
    if LLM_PROVIDER == "groq":
        return _call_groq(system, user, temperature, json_mode)
    if LLM_PROVIDER == "gemini":
        return _call_gemini(system, user, temperature, json_mode)
    raise LLMError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use 'groq' or 'gemini'.")


def chat_completion_json(system: str, user: str, temperature: float = 0.0) -> dict:
    """Calls the LLM in JSON mode and parses the result, with a defensive fallback."""
    raw = chat_completion(system, user, temperature=temperature, json_mode=True)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise LLMError(f"Could not parse JSON from LLM response: {raw[:300]}")
