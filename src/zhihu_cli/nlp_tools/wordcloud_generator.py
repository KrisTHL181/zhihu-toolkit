"""Generate a word cloud from downloaded Zhihu Markdown files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import jieba
import jieba.analyse
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from zhihu_cli.content.handlers.cache_manager import cache_manager
from zhihu_cli.content.utils.text_clean import (
    body_after_frontmatter,
    extract_cjk,
    parse_frontmatter,
    strip_markdown_noise,
)
from zhihu_cli.nlp_tools import FONT_PATH, STOP_WORDS
from zhihu_cli.paths import zhihu_home

DATA_DIR = zhihu_home()
OUTPUT_FILE: str = str(DATA_DIR / "plots" / "zhihu_wordcloud.png")


def extract_text_from_md(file_path: str, skip_metadata: bool = False) -> str:
    """Extract metadata and body text from a Markdown file.

    Compatible with YAML frontmatter (current) and legacy JSON-line format.

    :param file_path: Path to the Markdown file.
    :param skip_metadata: When true, prepend the question metadata and drop the
        frontmatter block; otherwise return the file unchanged.
    :returns: The text to analyse, or an empty string on read failure.
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Read failed {file_path}: {e}")
        return ""

    if not skip_metadata:
        return content

    text_to_analyze = ""
    if "---" in content:
        meta = parse_frontmatter(content)
        if meta:
            text_to_analyze += meta.get("question_name", "") + " "
            text_to_analyze += meta.get("question_detail", "") + " "

    # Body is always the last segment. For current YAML files the helper strips
    # the frontmatter; for legacy JSON-line files (and plain files with a
    # ``---`` separator) fall back to the historical split so the body matches
    # the previous behaviour.
    body = body_after_frontmatter(content)
    text_to_analyze += body

    return text_to_analyze


def is_stop_word(word: str) -> bool:
    """Check if word is a stop word."""
    return word in STOP_WORDS or len(word) == 1


def main(topk_words: int = 200, source_dir: str | None = None, only_print: bool = False) -> None:
    if source_dir is None:
        source_dir = str(DATA_DIR / "downloads")
    all_text = []
    print(f"Scanning {source_dir} for Markdown files...")

    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                all_text.append(extract_text_from_md(file_path))

    text = " ".join(all_text)

    # Clean special chars, code blocks, etc.
    full_content = extract_cjk(strip_markdown_noise(text))

    keywords_raw = jieba.analyse.extract_tags(
        full_content,
        topK=topk_words * 2,
        withWeight=True,
    )

    keywords = [(word, weight) for word, weight in keywords_raw if not is_stop_word(word)]

    if len(keywords) < topk_words:
        print(f"Warning: only {len(keywords)} words left after filtering, fewer than {topk_words}")

    keywords = keywords[:topk_words]

    word_freq = {word: weight for word, weight in keywords}

    if only_print:
        print("\nKeywords:")
        for i, (word, weight) in enumerate(keywords, 1):
            print(f"{i}. {word} ({weight:.4f})")
        return

    print("\nTop 20 keywords (stop words filtered):")
    for i, (word, weight) in enumerate(keywords[:20], 1):
        print(f"{i}. {word} ({weight:.4f})")

    print("\nGenerating word cloud...")
    wordcloud = WordCloud(
        font_path=FONT_PATH if os.path.exists(FONT_PATH) else None,
        width=1600,
        height=900,
        background_color="white",
        max_words=topk_words,
        colormap="viridis",
        stopwords=STOP_WORDS,
    ).generate_from_frequencies(word_freq)

    plt.figure(figsize=(16, 9))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=cache_manager.get_plot_dpi())
    plt.show()
    print(f"Word cloud saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate word cloud image")
    parser.add_argument("--topk", type=int, default=200, help="Number of keywords (default: 200)")
    parser.add_argument("--source_dir", type=str, default=str(DATA_DIR / "downloads"), help="Markdown file directory")
    parser.add_argument("--only_print", action="store_true", help="Only print keywords, skip image generation")

    args = parser.parse_args()

    main(topk_words=args.topk, source_dir=args.source_dir, only_print=args.only_print)
