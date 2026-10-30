from typing import Dict, List, Any
from .cleaner import TextCleaner


class SummaryEvaluator:
    """
    Evaluator tool for computing text metrics, compression ratios,
    readability estimates, and top keywords.
    """

    def __init__(self, cleaner: TextCleaner = None):
        self.cleaner = cleaner or TextCleaner()

    def evaluate(self, original_text: str, summary_text: str) -> Dict[str, Any]:
        """
        Compute comparative metrics between original text and generated summary.

        :param original_text: Input document.
        :param summary_text: Generated summary.
        :return: Dict containing metrics.
        """
        clean_orig = self.cleaner.clean_text(original_text)
        clean_sum = self.cleaner.clean_text(summary_text)

        orig_words = self.cleaner.tokenize_words(clean_orig)
        sum_words = self.cleaner.tokenize_words(clean_sum)

        orig_sentences = self.cleaner.split_into_sentences(clean_orig)
        sum_sentences = self.cleaner.split_into_sentences(clean_sum)

        orig_word_count = len(orig_words)
        sum_word_count = len(sum_words)

        if orig_word_count > 0:
            compression_ratio = round((1 - (sum_word_count / orig_word_count)) * 100, 1)
        else:
            compression_ratio = 0.0

        # Estimated reading time at average 200 words per minute
        orig_read_time_sec = round((orig_word_count / 200) * 60)
        sum_read_time_sec = round((sum_word_count / 200) * 60)
        time_saved_sec = max(0, orig_read_time_sec - sum_read_time_sec)

        # Extract top 5 key concepts/keywords from original text
        freq_map = self.cleaner.get_word_frequencies(clean_orig)
        sorted_keywords = sorted(freq_map, key=freq_map.get, reverse=True)[:5]

        return {
            "original_word_count": orig_word_count,
            "original_sentence_count": len(orig_sentences),
            "original_char_count": len(clean_orig),
            "summary_word_count": sum_word_count,
            "summary_sentence_count": len(sum_sentences),
            "summary_char_count": len(clean_sum),
            "compression_ratio_percent": max(0.0, compression_ratio),
            "original_reading_time_sec": orig_read_time_sec,
            "summary_reading_time_sec": sum_read_time_sec,
            "reading_time_saved_sec": time_saved_sec,
            "top_keywords": sorted_keywords
        }
