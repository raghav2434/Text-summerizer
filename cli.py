import argparse
import sys
import os

# Ensure UTF-8 output on Windows consoles for multi-language text (Hindi, French)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from text_summarizer.cleaner import TextCleaner
from text_summarizer.model import AbstractiveSummarizer
from text_summarizer.evaluator import SummaryEvaluator
from text_summarizer.dataset_loader import DatasetLoader


def main():
    parser = argparse.ArgumentParser(
        description="AI/ML Text Summarizer CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--text", type=str, help="Raw text string to summarize.")
    group.add_argument("--file", type=str, help="Path to input text document.")
    group.add_argument("--sample", action="store_true", help="Use a sample benchmark article.")

    parser.add_argument("--max-length", type=int, default=120, help="Max length for AI summary.")
    parser.add_argument("--min-length", type=int, default=30, help="Min length for AI summary.")
    parser.add_argument(
        "--translate",
        choices=["hindi", "french"],
        help="Translate summary into Hindi or French."
    )
    parser.add_argument("--output", type=str, help="Save summary output to specified text file.")

    args = parser.parse_args()

    # Determine input text
    input_text = ""
    if args.text:
        input_text = args.text
    elif args.file:
        if not os.path.exists(args.file):
            print(f"Error: File '{args.file}' not found.", file=sys.stderr)
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            input_text = f.read()
    elif args.sample:
        dl = DatasetLoader()
        articles = dl.get_all_articles()
        if articles:
            sample_art = articles[0]
            print(f"--- Loading Sample: {sample_art['title']} ({sample_art['category']}) ---")
            input_text = sample_art["text"]
        else:
            print("No sample articles found.", file=sys.stderr)
            sys.exit(1)
    else:
        # If no argument given, prompt for text input
        print("--- Text Summarizer CLI ---")
        print("Enter or paste your text below (Press Ctrl+D or Ctrl+Z then Enter to submit):")
        input_text = sys.stdin.read()

    if not input_text or len(input_text.strip()) < 20:
        print("Error: Input text is empty or too short.", file=sys.stderr)
        sys.exit(1)

    cleaner = TextCleaner()
    evaluator = SummaryEvaluator(cleaner=cleaner)

    print("\nProcessing text and generating summary...")
    model = AbstractiveSummarizer()
    res = model.summarize(input_text, max_length=args.max_length, min_length=args.min_length)
    summary = res["summary"]
    method_str = f"Abstractive AI ({res['model_used']})"

    metrics = evaluator.evaluate(input_text, summary)

    print("\n" + "="*50)
    print(f"Engine: {method_str}")
    print(f"Original Words: {metrics['original_word_count']} | Summary Words: {metrics['summary_word_count']} | Reduction: {metrics['compression_ratio_percent']}%")
    print(f"Top Keywords: {', '.join(metrics['top_keywords'])}")
    print("="*50)
    print("\n--- GENERATED SUMMARY ---")
    print(summary)
    print("="*50)

    if args.translate:
        from text_summarizer.translator import TextTranslator
        translator = TextTranslator()
        print(f"\nTranslating summary to {args.translate.capitalize()}...")
        trans_res = translator.translate(summary, args.translate)
        print("\n" + "="*50)
        print(f"--- TRANSLATION ({trans_res['target_language']} - {trans_res['native_name']}) ---")
        try:
            print(trans_res["translated_text"])
        except UnicodeEncodeError:
            print(trans_res["translated_text"].encode("utf-8", errors="replace").decode("utf-8"))
        print("="*50)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(summary)
            if args.translate and 'trans_res' in locals():
                f.write(f"\n\n--- Translation ({trans_res['target_language']}) ---\n")
                f.write(trans_res["translated_text"])
        print(f"\nSaved summary to '{args.output}'")


if __name__ == "__main__":
    main()
