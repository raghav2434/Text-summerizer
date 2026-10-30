# ⚡ AI Text Summarizer Project

A complete, production-ready **AI/ML Text Summarization System** in Python. 

The application transforms long documents, news articles, and research papers into concise summaries preserving key information using both **Abstractive AI (Hugging Face Transformers)** and **Extractive (NLTK Sentence Ranker)** algorithms.

---

## 🌟 Features

* **Dual Summarization Engine**:
  * **Abstractive AI**: Uses pre-trained deep learning transformer models (`distilbart-cnn-12-6` / `t5-small`) via Hugging Face `transformers` to rewrite and synthesize key concepts.
  * **Extractive Summarizer**: Uses term frequency and sentence centrality scoring via `NLTK` to extract top sentences in narrative sequence.
* **Multi-Language Translation (Hindi & French)**:
  * Translates the generated summary into **Hindi (हिंदी)** and **French (Français)**.
* **Multiple User Interfaces**:
  * **Interactive Web App (`app.py`)**: Streamlit application with custom styling, token sliders, text/file inputs, sample benchmark articles, side-by-side comparison, and translation into Hindi & French.
  * **Command Line Interface (`cli.py`)**: Terminal tool to summarize and translate text files or direct strings quickly.
  * **Jupyter Notebook (`text_summarizer_demo.ipynb`)**: Colab/Jupyter notebook showcasing step-by-step text cleaning, sentence ranking, transformer models, and benchmark evaluation.
* **Benchmark Datasets**:
  * Integrated sample articles inspired by Hugging Face **CNN/DailyMail** and **XSum** datasets.
* **Comprehensive Metrics**:
  * Calculates original vs. summary word counts, compression ratio (%), estimated reading time saved, and top key concepts.

---

## 🏗️ Project Architecture

```
c:\project\
├── data/
│   └── sample_articles.json       # Benchmark datasets (CNN/DailyMail & XSum samples)
├── text_summarizer/
│   ├── __init__.py                # Package initializer
│   ├── cleaner.py                 # NLTK text processing, sentence splitting, tokenization
│   ├── ranker.py                  # Extractive sentence ranking algorithm
│   ├── model.py                   # Abstractive Hugging Face Transformers pipeline
│   ├── dataset_loader.py          # Benchmark dataset loader (CNN/DailyMail & XSum)
│   ├── evaluator.py               # Summary metrics & compression ratio calculator
│   └── translator.py              # Hindi & French translation module
├── app.py                         # Streamlit interactive web dashboard
├── cli.py                         # Command-line interface tool
├── text_summarizer_demo.ipynb     # Jupyter / Google Colab demonstration notebook
├── tests/
│   └── test_summarizer.py         # Pytest test suite
├── requirements.txt               # Project dependencies
└── README.md                      # Documentation guide
```

---

## 🚀 Quick Start Guide

### 1. Requirements & Installation

Ensure Python 3.10+ is installed on your system. Install dependencies via pip:

```bash
pip install -r requirements.txt
```

---

### 2. Run the Streamlit Web Application

Launch the web app interface:

```bash
streamlit run app.py
```

* Navigate to `http://localhost:8501` in your browser.
* Enter custom text, upload a `.txt` file, or select a sample CNN/DailyMail benchmark article.
* Adjust length sliders and generate summaries instantly using the Hugging Face AI engine.

---

### 3. Command Line Interface (CLI)

#### Summarize Sample Benchmark Article:
```bash
python cli.py --sample
```

#### Summarize a Local Text File:
```bash
python cli.py --file document.txt --max-length 120 --output summary.txt
```

#### Summarize Direct Text String:
```bash
python cli.py --text "Scientists have announced a major breakthrough in solar energy storage technology..."
```

#### Summarize and Translate (Hindi / French):
```bash
python cli.py --sample --translate hindi
python cli.py --sample --translate french
```

---

### 4. Interactive Jupyter / Colab Notebook

Open `text_summarizer_demo.ipynb` in VS Code, Jupyter Lab, or upload it to Google Colab:

```bash
jupyter notebook text_summarizer_demo.ipynb
```

---

### 5. Run Automated Tests

To execute unit tests and verify code health:

```bash
python -m pytest tests/
```

---

## 📊 Datasets & Performance Comparison

| Summarization Engine | Type | Strengths | Speed |
| :--- | :--- | :--- | :--- |
| **Hugging Face Transformers** | Abstractive AI | Rewrites human-like fluent summaries | High precision |
| **NLTK Sentence Ranker** | Extractive | Preserves exact original phrasing & fast execution | Ultra-fast (CPU) |

---

## 📜 License

Distributed under the MIT License.
