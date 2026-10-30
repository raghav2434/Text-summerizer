import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TextTranslator:
    """
    Translation module for converting summarized text into Hindi and French.
    Restricted to Hindi and French languages only.
    """

    SUPPORTED_LANGUAGES: Dict[str, Dict[str, str]] = {
        "hindi": {
            "code": "hi",
            "name": "Hindi",
            "native_name": "हिंदी",
            "mymemory_code": "hi-IN"
        },
        "french": {
            "code": "fr",
            "name": "French",
            "native_name": "Français",
            "mymemory_code": "fr-FR"
        }
    }

    def get_supported_languages(self) -> List[str]:
        """Return list of supported language keys."""
        return list(self.SUPPORTED_LANGUAGES.keys())

    def _normalize_lang(self, target_lang: str) -> str:
        """Validate and normalize language identifier to 'hindi' or 'french'."""
        if not target_lang:
            raise ValueError("Target language must be specified.")
        
        normalized = target_lang.strip().lower()
        if normalized in ["hi", "hindi", "हिंदी"]:
            return "hindi"
        elif normalized in ["fr", "french", "français", "francais"]:
            return "french"
        else:
            raise ValueError(
                f"Unsupported language '{target_lang}'. Only 'hindi' and 'french' are supported."
            )

    def translate(self, text: str, target_lang: str) -> Dict[str, str]:
        """
        Translate the provided text into Hindi or French.

        :param text: Input text to translate.
        :param target_lang: Target language ('hindi' or 'french').
        :return: Dict containing translated_text, target_language, language_code, and status.
        """
        if not text or not text.strip():
            return {
                "translated_text": "",
                "target_language": target_lang,
                "language_code": "",
                "status": "empty"
            }

        lang_key = self._normalize_lang(target_lang)
        lang_meta = self.SUPPORTED_LANGUAGES[lang_key]
        clean_text = text.strip()

        # Attempt 1: MyMemoryTranslator via deep-translator
        try:
            from deep_translator import MyMemoryTranslator
            translator = MyMemoryTranslator(source="en-US", target=lang_meta["mymemory_code"])
            translated = translator.translate(clean_text)
            if translated:
                return {
                    "translated_text": translated,
                    "target_language": lang_meta["name"],
                    "native_name": lang_meta["native_name"],
                    "language_code": lang_meta["code"],
                    "status": "success",
                    "provider": "MyMemory"
                }
        except Exception as e:
            logger.warning(f"MyMemory translation failed: {e}. Trying GoogleTranslator fallback.")

        # Attempt 2: GoogleTranslator fallback
        try:
            from deep_translator import GoogleTranslator
            translator = GoogleTranslator(source="auto", target=lang_meta["code"])
            translated = translator.translate(clean_text)
            if translated:
                return {
                    "translated_text": translated,
                    "target_language": lang_meta["name"],
                    "native_name": lang_meta["native_name"],
                    "language_code": lang_meta["code"],
                    "status": "success",
                    "provider": "Google"
                }
        except Exception as e:
            logger.error(f"Translation failed: {e}")

        # Fallback if offline
        return {
            "translated_text": clean_text,
            "target_language": lang_meta["name"],
            "native_name": lang_meta["native_name"],
            "language_code": lang_meta["code"],
            "status": "fallback_original",
            "provider": "none"
        }
