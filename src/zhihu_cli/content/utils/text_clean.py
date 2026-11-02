"""Shared text-cleaning helpers for downloaded Zhihu content.

These helpers operate on Markdown files written by the download pipeline and are
shared by the NLP tools and the crank classifier. The module only depends on the
standard library; ``yaml`` is imported lazily inside :func:`parse_frontmatter`
because the NLP tools may be used without the base dependency installed.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

__all__ = [
    "clean_excerpt",
    "clean_text",
    "extract_cjk",
    "parse_frontmatter",
    "strip_code_blocks",
    "strip_html_tags",
    "strip_latex",
    "strip_markdown_noise",
    "strip_yaml_frontmatter",
]


def strip_yaml_frontmatter(text: str) -> str:
    """Remove YAML frontmatter delimited by ``---...---``.

    Only text that starts with ``---`` and contains a closing ``---`` is
    stripped; anything else is returned unchanged.

    :param text: Raw file or document text.
    :returns: The text after the closing delimiter, or the original text.
    """
    if text.startswith("---"):
        idx = text.find("---", 3)
        if idx != -1:
            return text[idx + 3 :]
    return text


def parse_frontmatter(text: str) -> dict[str, Any]:
    """Parse the metadata block at the start of a downloaded Markdown file.

    Handles both the current YAML frontmatter format and the legacy format in
    which a JSON object (possibly pretty-printed over several lines) precedes
    the body. Any parse failure yields an empty dict.

    :param text: Raw file content.
    :returns: The parsed metadata as a dict, or ``{}`` when parsing fails.
    """
    if text.startswith("---"):
        idx = text.find("---", 3)
        if idx == -1:
            return {}
        try:
            import yaml

            meta = yaml.safe_load(text[3:idx])
        except Exception:
            return {}
        return meta if isinstance(meta, dict) else {}

    if "---" not in text:
        return {}

    # Legacy format: metadata precedes the first ``---`` delimiter. JSON is a
    # subset of YAML, so YAML parses single-line and multi-line JSON alike.
    prefix = text.split("---", 1)[0]
    try:
        import yaml

        meta = yaml.safe_load(prefix)
    except Exception:
        meta = None
    if isinstance(meta, dict):
        return meta

    try:
        meta = json.loads(prefix)
    except (TypeError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def strip_latex(text: str) -> str:
    """Remove LaTeX display math, inline math, and math environments.

    :param text: Raw text that may contain LaTeX markup.
    :returns: Text with ``$$...$$``, ``$...$``, and
        ``\\begin{...}...\\end{...}`` blocks removed.
    """
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    text = re.sub(r"\$.*?\$", "", text)
    text = re.sub(r"\\begin\{.*?\}.*?\\end\{.*?\}", "", text, flags=re.DOTALL)
    return text


def strip_code_blocks(text: str) -> str:
    """Remove fenced Markdown code blocks.

    :param text: Raw text that may contain fenced code blocks.
    :returns: Text with all ```` ```...``` ```` blocks removed.
    """
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def strip_html_tags(text: str) -> str:
    """Remove HTML tags from text.

    :param text: Raw text that may contain HTML markup.
    :returns: Text with every ``<...>`` tag removed.
    """
    return re.sub(r"<[^>]+>", "", text)


def strip_markdown_noise(text: str) -> str:
    """Remove LaTeX, fenced code blocks, and HTML tags from Markdown text.

    :param text: Raw Markdown text.
    :returns: Text without the common markup noise found in downloaded content.
    """
    text = strip_latex(text)
    text = strip_code_blocks(text)
    return strip_html_tags(text)


def extract_cjk(text: str, *, keep_latin: bool = False) -> str:
    """Extract CJK characters from text.

    :param text: Raw text.
    :param keep_latin: When true, also keep Latin letters and digits and drop
        separators; when false, keep only CJK characters.
    :returns: The extracted characters, concatenated without separators.
    """
    if keep_latin:
        return "".join(re.findall(r"[一-龥a-zA-Z0-9]+", text))
    return re.sub(r"[^一-龥]", "", text)


def clean_text(text: str) -> str:
    """Clean article text for BERT input.

    Pipeline: strip YAML → remove LaTeX math → remove code blocks → remove HTML
    tags → remove markdown image/link syntax → collapse whitespace.

    LaTeX environments (``\\begin{...}``) are deliberately left untouched here
    to preserve the historical output; use :func:`strip_latex` when they should
    be removed as well.

    :param text: Raw article text, optionally with YAML frontmatter.
    :returns: Cleaned single-line text.
    """
    text = strip_yaml_frontmatter(text)
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    text = re.sub(r"\$.*?\$", "", text)
    text = strip_code_blocks(text)
    text = strip_html_tags(text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[([^\]]*)\]\(.*?\)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_excerpt(text: str) -> str:
    """Light cleaning for API excerpt text.

    The excerpt is already plain text, so only ``<em>`` markup and URLs are
    removed.

    :param text: Excerpt text, possibly containing ``<em>`` tags and URLs.
    :returns: The cleaned excerpt.
    """
    text = re.sub(r"<em>|</em>", "", text)
    text = re.sub(r"http\S+", "", text)
    return text.strip()
