import time
import os
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="AI/ML Text Summarizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    /* Dark glassmorphic theme styling */
    .main {
        background-color: #0E1117;
        color: #E0E6ED;
    }
    .stAppHeader {
        background: transparent;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #4F46E5;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #6366F1;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #9CA3AF;
        margin-top: 4px;
    }
    .badge {
        display: inline-block;
        background: #312E81;
        color: #A5B4FC;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .summary-box {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #6366F1;
        border-radius: 8px;
        padding: 20px;
        font-size: 1.05rem;
        line-height: 1.6;
        color: #F8FAFC;
    }
    .original-box {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 20px;
        font-size: 0.95rem;
        line-height: 1.5;
        color: #CBD5E1;
        max-height: 400px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# Lazy import project modules
from text_summarizer.cleaner import TextCleaner
from text_summarizer.model import AbstractiveSummarizer
from text_summarizer.evaluator import SummaryEvaluator
from text_summarizer.dataset_loader import DatasetLoader


@st.cache_resource
def load_summarizer_models():
    """Cache models in memory for fast user interactions."""
    cleaner = TextCleaner()
    abstractive = AbstractiveSummarizer()
    evaluator = SummaryEvaluator(cleaner=cleaner)
    dataset_loader = DatasetLoader()
    return cleaner, abstractive, evaluator, dataset_loader


cleaner, abstractive_model, evaluator, dataset_loader = load_summarizer_models()

# App Header
st.title("⚡ AI & NLP Text Summarizer")
st.caption("Summarize long documents and articles using Hugging Face Transformers")

# Sidebar Controls
st.sidebar.header("⚙️ Configuration")
st.sidebar.subheader("🎛️ Parameters")

max_len = st.sidebar.slider("Max Summary Length (words)", 40, 250, 120, 10)
min_len = st.sidebar.slider("Min Summary Length (words)", 20, 100, 30, 5)

# Sample Data Selector or Custom Input
st.subheader("📝 Input Text")

input_mode = st.radio(
    "Choose Input Source",
    ["Type / Paste Text", "Upload Document (.txt / .md)", "Load Sample Benchmark Article"],
    horizontal=True
)

raw_text = ""

if input_mode == "Type / Paste Text":
    raw_text = st.text_area(
        "Paste article or document text below:",
        height=240,
        placeholder="Paste a long news article, research abstract, or document here..."
    )
elif input_mode == "Upload Document (.txt / .md)":
    uploaded_file = st.file_uploader("Upload a text file", type=["txt", "md"])
    if uploaded_file is not None:
        raw_text = uploaded_file.read().decode("utf-8")
        st.success(f"Loaded '{uploaded_file.name}' ({len(raw_text.split())} words)")
else: # Load Sample Benchmark Article
    sample_articles = dataset_loader.get_all_articles()
    if sample_articles:
        sample_titles = [f"[{art['category']}] {art['title']}" for art in sample_articles]
        selected_index = st.selectbox("Select Benchmark Article:", range(len(sample_titles)), format_func=lambda i: sample_titles[i])
        selected_article = sample_articles[selected_index]
        raw_text = selected_article["text"]
        
        st.info(f"**Selected Sample**: {selected_article['title']} ({selected_article['category']})")
        if "reference_summary" in selected_article:
            with st.expander("📌 Reference Human Summary"):
                st.write(selected_article["reference_summary"])

# Summarize Action Button
if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
    if not raw_text or len(raw_text.strip()) < 30:
        st.warning("Please enter or load a text with at least 30 characters.")
    else:
        with st.spinner("Analyzing text and generating summary..."):
            start_time = time.time()
            
            result = abstractive_model.summarize(
                raw_text,
                max_length=max_len,
                min_length=min_len
            )
            summary_text = result["summary"]
            engine_used = f"{result['method'].upper()} ({result['model_used']})"

            elapsed_time = round(time.time() - start_time, 2)
            metrics = evaluator.evaluate(raw_text, summary_text)

        st.markdown("---")
        
        # Display Metrics Grid
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['original_word_count']}</div>
                <div class="metric-label">Original Words</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['summary_word_count']}</div>
                <div class="metric-label">Summary Words</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['compression_ratio_percent']}%</div>
                <div class="metric-label">Reduction / Compression</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{elapsed_time}s</div>
                <div class="metric-label">Processing Time</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(" ")

        # Display Top Keywords
        if metrics["top_keywords"]:
            st.markdown("**🔑 Top Extracted Keywords:**")
            kw_html = "".join([f'<span class="badge">#{kw}</span>' for kw in metrics["top_keywords"]])
            st.markdown(kw_html, unsafe_allow_html=True)
            st.markdown(" ")

        # Side-by-side or Stacked View
        col_orig, col_sum = st.columns(2)
        
        with col_orig:
            st.subheader("📄 Original Text")
            st.markdown(f'<div class="original-box">{raw_text}</div>', unsafe_allow_html=True)
            
        with col_sum:
            st.subheader(f"✨ AI Summary ({engine_used})")
            st.markdown(f'<div class="summary-box">{summary_text}</div>', unsafe_allow_html=True)
            st.markdown(" ")
            st.download_button(
                label="📥 Download Summary (.txt)",
                data=summary_text,
                file_name="summary.txt",
                mime="text/plain"
            )

# Footer
st.markdown("---")
st.caption("AI/ML Text Summarizer Project • Python, Hugging Face Transformers, NLTK")
