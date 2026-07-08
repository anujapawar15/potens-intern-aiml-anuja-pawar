"""
Language detection + LLM-based translation for multilingual questions.
See README.md section 7 for the rationale.
"""
from langdetect import detect, LangDetectException

from app.llm import chat_completion


def detect_language(text: str) -> str:
    """Returns an ISO 639-1 language code, defaulting to 'en' on failure."""
    try:
        return detect(text)
    except LangDetectException:
        return "en"


def translate_to_english(text: str, source_lang: str) -> str:
    if source_lang == "en":
        return text
    system = "You are a precise translator. Translate the user's text to English. Return only the translation, with no extra commentary."
    return chat_completion(system, text, temperature=0.0).strip()


def translate_from_english(text: str, target_lang: str) -> str:
    if target_lang == "en":
        return text
    system = (
        f"You are a precise translator. Translate the user's text from English into the "
        f"language with ISO 639-1 code '{target_lang}'. Return only the translation, with "
        f"no extra commentary, and preserve any bracketed citation markers like [1] exactly."
    )
    return chat_completion(system, text, temperature=0.0).strip()
