from typing import List, Dict
from .cleaner import TextCleaner


class ExtractiveSummarizer:
    """
    Extractive Summarizer.
    Scores sentences based on non-stopword term frequencies and extracts top sentences
    while preserving their original sequence in the text.
    """

    def __init__(self, cleaner: TextCleaner = None):
        self.cleaner = cleaner or TextCleaner()

    def summarize(self, text: str, num_sentences: int = 3) -> Dict[str, str]:
        """
        Generate an extractive summary with a target number of sentences.

        :param text: Raw input text.
        :param num_sentences: Target number of sentences in summary.
        :return: Dict containing 'summary', 'sentences', and 'sentence_scores'.
        """
        raw_sentences = self.cleaner.split_into_sentences(text)
        if not raw_sentences:
            return {
                "summary": "",
                "sentences": [],
                "scores": {}
            }

        # If document is shorter than requested sentence count, return full text
        if len(raw_sentences) <= num_sentences:
            return {
                "summary": " ".join(raw_sentences),
                "sentences": raw_sentences,
                "scores": {s: 1.0 for s in raw_sentences}
            }

        word_freqs = self.cleaner.get_word_frequencies(text)
        sentence_scores: Dict[int, float] = {}

        for idx, sentence in enumerate(raw_sentences):
            words = self.cleaner.tokenize_words(sentence, lower=True)
            if not words:
                continue
            
            score = 0.0
            word_count = 0
            for w in words:
                if w in word_freqs:
                    score += word_freqs[w]
                    word_count += 1

            # Length normalization to prevent long sentence bias
            sentence_scores[idx] = score / (word_count + 1e-5)

        # Select top sentence indices
        top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
        # Sort top indices chronologically to preserve document flow
        top_indices.sort()

        selected_sentences = [raw_sentences[i] for i in top_indices]
        summary_text = " ".join(selected_sentences)

        return {
            "summary": summary_text,
            "sentences": selected_sentences,
            "scores": {raw_sentences[i]: sentence_scores[i] for i in top_indices}
        }
