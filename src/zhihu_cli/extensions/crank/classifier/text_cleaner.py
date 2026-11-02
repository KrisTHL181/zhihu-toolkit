"""Unified text cleaning pipeline for BERT classification.

Unlike the existing nlp_tools/ pipeline, this does NOT use jieba tokenization
because BERT uses its own WordPiece tokenizer. Cleaning focuses on removing
markup noise while preserving Chinese + English characters for BERT.
"""

from __future__ import annotations

from zhihu_cli.content.utils.text_clean import (
    clean_excerpt,
    clean_text,
    strip_yaml_frontmatter,
)

__all__ = [
    "strip_yaml_frontmatter",
    "clean_text",
    "clean_excerpt",
    "prepare_for_bert",
    "prepare_bert_from_article_dict",
]


def prepare_for_bert(
    title: str,
    body: str,
    *,
    title_weight: float = 2.0,
) -> str:
    """Build input string for BERT tokenization.

    Title is repeated ``title_weight`` times for emphasis, since crank signals
    (grandiose claims, pseudoscientific terminology) are most concentrated there.
    """
    clean_title = clean_excerpt(title)
    clean_body = clean_text(body) if body else ""

    repeats = max(1, int(round(title_weight)))
    title_part = " ".join([clean_title] * repeats)

    if clean_body:
        return f"{title_part} [SEP] {clean_body}"
    return title_part


def prepare_bert_from_article_dict(article: dict) -> str:
    """Prepare BERT input from a Zhihu API article dict.

    Prefers ``_body`` (full markdown from scrape_article) over ``excerpt`` (API snippet).
    """
    title = article.get("title", "")
    body = article.get("_body", article.get("excerpt", ""))
    return prepare_for_bert(title, body)
