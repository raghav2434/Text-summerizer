import re
import string
import nltk
from typing import List, Dict, Set


class TextCleaner:
    """
    Text Preprocessing utility leveraging NLTK for sentence splitting,
    tokenization, stopword removal, and noise reduction.
    """

    def __init__(self):
        self._ensure_nltk_resources()
        self.stopwords = self._get_stopwords()

    def _ensure_nltk_resources(self):
        """Ensure required NLTK datasets are downloaded silently."""
        resources = ['punkt', 'stopwords', 'punkt_tab']
        for res in resources:
            try:
                nltk.data.find(f'tokenizers/{res}' if 'punkt' in res else f'corpora/{res}')
            except (LookupError, OSError):
                try:
                    nltk.download(res, quiet=True)
                except Exception:
                    pass

    def _get_stopwords(self) -> Set[str]:
        """Fetch English stopwords list or fallback to standard set."""
        try:
            from nltk.corpus import stopwords
            return set(stopwords.words('english'))
        except Exception:
            # Reliable built-in fallback English stopwords list
            return {
                'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
                'any', 'are', 'aren\'t', 'as', 'at', 'be', 'because', 'been', 'before', 'being',
                'below', 'between', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn\'t',
                'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during',
                'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t',
                'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here',
                'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i',
                'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it',
                'it\'s', 'its', 'itself', 'let\'s', 'me', 'more', 'most', 'mustn\'t', 'my',
                'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other',
                'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'shan\'t',
                'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 'shouldn\'t', 'so', 'some',
                'such', 'than', 'that', 'that\'s', 'the', 'their', 'theirs', 'them',
                'themselves', 'then', 'there', 'there\'s', 'these', 'they', 'they\'d',
                'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'through', 'to', 'too',
                'under', 'until', 'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll',
                'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s',
                'where', 'where\'s', 'which', 'while', 'who', 'who\'s', 'whom', 'why',
                'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll',
                'you\'re', 'you\'ve', 'your', 'yours', 'yourself', 'yourselves'
            }

    def clean_text(self, text: str) -> str:
        """Sanitize raw input text by stripping HTML, brackets, and redundant whitespace."""
        if not text:
            return ""
        # Remove HTML tags if present
        text = re.sub(r'<[^>]+>', ' ', text)
        # Normalize whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK or fallback regex splitter."""
        clean_input = self.clean_text(text)
        if not clean_input:
            return []
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(clean_input)
        except Exception:
            # Regex fallback splitter
            sentences = re.split(r'(?<=[.!?]) +', clean_input)
        
        return [s.strip() for s in sentences if s.strip()]

    def tokenize_words(self, text: str, lower: bool = True) -> List[str]:
        """Tokenize text into alphanumeric words."""
        if lower:
            text = text.lower()
        # Keep alphanumeric words
        words = re.findall(r'\b[a-zA-Z0-9]+\b', text)
        return words

    def get_word_frequencies(self, text: str) -> Dict[str, float]:
        """
        Calculate normalized word frequencies for non-stopword tokens.
        """
        words = self.tokenize_words(text, lower=True)
        freq_map: Dict[str, int] = {}
        
        for w in words:
            if w not in self.stopwords and w not in string.punctuation:
                freq_map[w] = freq_map.get(w, 0) + 1

        if not freq_map:
            return {}

        max_freq = max(freq_map.values())
        # Normalize frequencies by max frequency
        return {w: count / max_freq for w, count in freq_map.items()}
