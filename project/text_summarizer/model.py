import logging
import torch
from typing import Dict, Optional
from .cleaner import TextCleaner
from .ranker import ExtractiveSummarizer

logger = logging.getLogger(__name__)


class AbstractiveSummarizer:
    """
    Abstractive AI Summarizer powered by Hugging Face Transformers.
    Uses AutoTokenizer and AutoModelForSeq2SeqLM for robust text summarization
    with automatic fallback to Extractive summarization if offline or unavailable.
    """

    DEFAULT_MODEL = "sshleifer/distilbart-cnn-12-6"

    def __init__(self, model_name: str = None, device: str = None):
        """
        :param model_name: Hugging Face model identifier (e.g. 'sshleifer/distilbart-cnn-12-6', 't5-small').
        :param device: 'cpu' or 'cuda'.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self._tokenizer = None
        self._model = None
        self._load_failed = False
        self.cleaner = TextCleaner()
        self.fallback_ranker = ExtractiveSummarizer(cleaner=self.cleaner)

    def _load_model(self):
        """Lazy load Hugging Face tokenizer and Seq2Seq model."""
        if self._model is None and not self._load_failed:
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                print(f"[AI Model] Loading Hugging Face model '{self.model_name}' on {self.device}...")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
                print("[AI Model] Model loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not load Hugging Face model '{self.model_name}': {e}")
                self._load_failed = True

    def is_model_available(self) -> bool:
        """Check if Hugging Face model is ready."""
        self._load_model()
        return self._model is not None

    def summarize(
        self,
        text: str,
        max_length: int = 120,
        min_length: int = 30,
        do_sample: bool = False
    ) -> Dict[str, str]:
        """
        Generate an abstractive AI summary for the given text.

        :param text: Raw input text.
        :param max_length: Maximum summary token length.
        :param min_length: Minimum summary token length.
        :param do_sample: Enable nucleus sampling.
        :return: Dict containing 'summary', 'method', and 'model_used'.
        """
        clean_text = self.cleaner.clean_text(text)
        if not clean_text:
            return {"summary": "", "method": "none", "model_used": self.model_name}

        words = clean_text.split()
        # If input text is very short (under 25 words), return original text directly
        if len(words) < 25:
            return {
                "summary": clean_text,
                "method": "direct",
                "model_used": "direct_short_text"
            }

        # Try Hugging Face Transformer model
        self._load_model()
        if self._model is not None and self._tokenizer is not None:
            try:
                # Format input for T5 or BART models if needed
                prefix = "summarize: " if "t5" in self.model_name.lower() else ""
                input_str = prefix + clean_text

                inputs = self._tokenizer(
                    input_str,
                    return_tensors="pt",
                    max_length=1024,
                    truncation=True
                ).to(self.device)

                effective_max = min(max_length, max(30, int(len(words) * 0.7)))
                effective_min = min(min_length, int(effective_max * 0.35))

                with torch.no_grad():
                    summary_ids = self._model.generate(
                        **inputs,
                        max_length=effective_max,
                        min_length=effective_min,
                        num_beams=4,
                        do_sample=do_sample,
                        early_stopping=True
                    )

                summary_text = self._tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

                return {
                    "summary": summary_text,
                    "method": "abstractive_ai",
                    "model_used": self.model_name
                }
            except Exception as e:
                logger.warning(f"Abstractive generation error: {e}. Falling back to Extractive summarization.")

        # Fallback to Extractive engine if Hugging Face fails or offline
        num_sentences = max(2, int(len(self.cleaner.split_into_sentences(clean_text)) * 0.35))
        extractive_res = self.fallback_ranker.summarize(clean_text, num_sentences=num_sentences)

        return {
            "summary": extractive_res["summary"],
            "method": "extractive_fallback",
            "model_used": "NLTK Extractive Ranker"
        }
