import os
import json
from typing import List, Dict, Optional


class DatasetLoader:
    """
    Dataset Manager for accessing Hugging Face benchmark datasets
    (CNN/DailyMail, XSum) and built-in sample articles.
    """

    def __init__(self, data_path: str = None):
        if data_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, "data", "sample_articles.json")
        self.data_path = data_path
        self._sample_data = self._load_local_samples()

    def _load_local_samples(self) -> Dict[str, List[Dict[str, str]]]:
        """Load benchmark sample articles from JSON file."""
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def get_categories(self) -> List[str]:
        """Get available dataset categories."""
        return list(self._sample_data.keys())

    def get_articles_by_category(self, category: str) -> List[Dict[str, str]]:
        """Get sample articles belonging to a specific category."""
        return self._sample_data.get(category, [])

    def get_all_articles(self) -> List[Dict[str, str]]:
        """Flatten and return all sample articles across categories."""
        all_articles = []
        for category, articles in self._sample_data.items():
            for art in articles:
                art_copy = dict(art)
                art_copy["source_dataset"] = category
                all_articles.append(art_copy)
        return all_articles

    def try_load_hf_dataset(self, dataset_name: str = "cnn_dailymail", split: str = "test[:5]"):
        """
        Attempt to load a live Hugging Face dataset via `datasets` library.
        Returns a list of dicts or None if offline/unavailable.
        """
        try:
            from datasets import load_dataset
            if dataset_name == "cnn_dailymail":
                ds = load_dataset("cnn_dailymail", "3.0.0", split=split)
                result = []
                for idx, row in enumerate(ds):
                    result.append({
                        "id": f"hf_{idx}",
                        "title": f"CNN/DailyMail Article #{idx+1}",
                        "category": "CNN/DailyMail Live",
                        "text": row["article"],
                        "reference_summary": row["highlights"]
                    })
                return result
            elif dataset_name == "xsum":
                ds = load_dataset("xsum", split=split)
                result = []
                for idx, row in enumerate(ds):
                    result.append({
                        "id": f"hf_xsum_{idx}",
                        "title": f"XSum Article #{idx+1}",
                        "category": "XSum Live",
                        "text": row["document"],
                        "reference_summary": row["summary"]
                    })
                return result
        except Exception:
            pass
        return None
