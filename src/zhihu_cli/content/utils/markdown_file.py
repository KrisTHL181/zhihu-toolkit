"""Filesystem helpers for writing downloaded content as Markdown.

The module is stdlib-only on import; ``yaml`` is imported lazily inside
:func:`build_yaml_frontmatter` because the download pipeline is the only
consumer that needs it. Keeping these helpers here (rather than in
``download_contents``) lets command modules build file names without pulling in
``curl_cffi`` through the download pipeline's session imports.
"""

from __future__ import annotations

import html
import os
import re
from typing import Any

__all__ = [
    "build_yaml_frontmatter",
    "get_safe_filename",
    "safe_path",
    "sanitize_filename",
    "write_markdown",
]


def sanitize_filename(name: str) -> str:
    """Sanitize filename by removing illegal characters.

    :param name: The raw file name or title.
    :returns: A filesystem-safe version of the name.
    """
    name = html.unescape(name)
    if os.name == "nt":
        illegal_chars = ["<", ">", ":", '"', "/", "\\", "|", "?", "*"]
        illegal_filenames = [
            "CON",
            "PRN",
            "AUX",
            "NUL",
            "COM1",
            "COM2",
            "COM3",
            "COM4",
            "COM5",
            "COM6",
            "COM7",
            "COM8",
            "COM9",
            "LPT1",
            "LPT2",
            "LPT3",
            "LPT4",
            "LPT5",
            "LPT6",
            "LPT7",
            "LPT8",
            "LPT9",
        ]
    else:
        illegal_chars = ["/", "\0"]
        illegal_filenames = []

    for char in illegal_chars:
        name = name.replace(char, "_")
    base_name = os.path.splitext(name)[0]
    if base_name.upper() in illegal_filenames:
        name = f"_{name}_"
    name = name.rstrip(". ")
    name = re.sub(r"_+", "_", name)
    return name.strip().strip("_")


def get_safe_filename(long_title: str, ext: str = ".md", max_bytes: int = 240) -> str:
    """Truncate a title to fit within ``max_bytes`` UTF-8 bytes.

    Truncation keeps whole characters and reserves room for the ``...`` marker
    and the extension, so the returned name never exceeds ``max_bytes``.

    :param long_title: The title (without extension).
    :param ext: The file extension to append.
    :param max_bytes: Maximum byte length of the returned name.
    :returns: The title plus extension, truncated when necessary.
    """
    total_bytes = len((long_title + ext).encode("utf-8"))
    if total_bytes <= max_bytes:
        return long_title + ext

    available_bytes = max(0, max_bytes - len(ext.encode("utf-8")) - 3)
    current_bytes = 0
    truncated_title = ""

    for char in long_title:
        char_bytes = len(char.encode("utf-8"))
        if current_bytes + char_bytes <= available_bytes:
            truncated_title += char
            current_bytes += char_bytes
        else:
            break

    return f"{truncated_title}...{ext}"


def safe_path(output_dir: str, stem: str, ext: str = ".md", max_bytes: int = 240) -> str:
    """Build a byte-safe path inside ``output_dir``.

    :param output_dir: Directory that will contain the file.
    :param stem: File name without extension.
    :param ext: The file extension.
    :param max_bytes: Maximum byte length of the file name.
    :returns: The joined path with a safely truncated file name.
    """
    return os.path.join(output_dir, get_safe_filename(stem, ext=ext, max_bytes=max_bytes))


def build_yaml_frontmatter(metadata: dict[str, Any]) -> str:
    """Build a YAML frontmatter block from a metadata dict.

    :param metadata: Fields to serialise.
    :returns: The frontmatter block, including trailing blank lines.
    """
    import yaml  # deferred: only needed when writing download files

    body = yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False).strip()
    return f"---\n{body}\n---\n\n"


def write_markdown(
    metadata: dict[str, Any] | None,
    markdown: str,
    output_dir: str,
    *,
    filename: str,
    with_frontmatter: bool = True,
) -> str:
    """Write Markdown to disk, optionally preceded by YAML frontmatter.

    :param metadata: Frontmatter fields. Required when ``with_frontmatter`` is
        true.
    :param markdown: The Markdown body.
    :param output_dir: Directory that will contain the file.
    :param filename: File name without extension; it is byte-safe truncated.
    :param with_frontmatter: Prepend :func:`build_yaml_frontmatter` output.
    :returns: The path of the written file.
    :raises ValueError: When ``with_frontmatter`` is true and ``metadata`` is
        ``None``.
    """
    if with_frontmatter and metadata is None:
        raise ValueError("metadata is required when with_frontmatter is True")

    os.makedirs(output_dir, exist_ok=True)
    filepath = safe_path(output_dir, filename)
    content = build_yaml_frontmatter(metadata) + markdown if with_frontmatter else markdown
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return filepath
