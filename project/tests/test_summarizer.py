import pytest
from text_summarizer.cleaner import TextCleaner
from text_summarizer.ranker import ExtractiveSummarizer
from text_summarizer.model import AbstractiveSummarizer
from text_summarizer.dataset_loader import DatasetLoader
from text_summarizer.evaluator import SummaryEvaluator


@pytest.fixture
def sample_text():
    return (
        "Artificial Intelligence is advancing at an unprecedented pace across the globe. "
        "Researchers are developing state-of-the-art neural networks capable of processing text, audio, and vision. "
        "These models are transforming industries such as healthcare, automotive, and finance. "
        "However, ethical considerations and data privacy remain major topics of discussion among policymakers. "
        "Ensuring that AI technologies are built responsibly is essential for society's future."
    )


def test_cleaner_split_into_sentences(sample_text):
    cleaner = TextCleaner()
    sentences = cleaner.split_into_sentences(sample_text)
    assert len(sentences) == 5
    assert "Artificial Intelligence" in sentences[0]


def test_cleaner_word_frequencies(sample_text):
    cleaner = TextCleaner()
    freqs = cleaner.get_word_frequencies(sample_text)
    assert isinstance(freqs, dict)
    assert "intelligence" in freqs or "ai" in freqs


def test_extractive_summarizer(sample_text):
    cleaner = TextCleaner()
    ranker = ExtractiveSummarizer(cleaner=cleaner)
    res = ranker.summarize(sample_text, num_sentences=2)
    assert "summary" in res
    assert len(res["sentences"]) == 2
    assert len(res["summary"].split()) < len(sample_text.split())


def test_dataset_loader():
    loader = DatasetLoader()
    articles = loader.get_all_articles()
    assert len(articles) > 0
    assert "text" in articles[0]
    assert "title" in articles[0]


def test_summary_evaluator(sample_text):
    cleaner = TextCleaner()
    evaluator = SummaryEvaluator(cleaner=cleaner)
    summary = "Artificial Intelligence is transforming industries rapidly worldwide."
    metrics = evaluator.evaluate(sample_text, summary)
    
    assert metrics["original_word_count"] > metrics["summary_word_count"]
    assert metrics["compression_ratio_percent"] > 0
    assert len(metrics["top_keywords"]) > 0


def test_abstractive_summarizer_short_text():
    model = AbstractiveSummarizer()
    short_text = "This is a very short sentence for quick testing."
    res = model.summarize(short_text)
    assert res["method"] == "direct"
    assert res["summary"] == short_text


def test_translator_languages():
    from text_summarizer.translator import TextTranslator
    translator = TextTranslator()
    assert set(translator.get_supported_languages()) == {"hindi", "french"}
    
    with pytest.raises(ValueError):
        translator.translate("Hello world", "german")


def test_translator_translation():
    from text_summarizer.translator import TextTranslator
    translator = TextTranslator()
    sample = "Solar energy is vital for the future."
    
    hi_res = translator.translate(sample, "hindi")
    assert hi_res["target_language"] == "Hindi"
    assert hi_res["language_code"] == "hi"
    assert len(hi_res["translated_text"]) > 0

    fr_res = translator.translate(sample, "french")
    assert fr_res["target_language"] == "French"
    assert fr_res["language_code"] == "fr"
    assert len(fr_res["translated_text"]) > 0

