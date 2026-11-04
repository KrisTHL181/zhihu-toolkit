"""Tests for the shared Markdown file helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from zhihu_cli.content.utils.markdown_file import (
    build_yaml_frontmatter,
    get_safe_filename,
    safe_path,
    sanitize_filename,
    write_markdown,
)


def test_sanitize_filename_removes_illegal_chars() -> None:
    """Path separators and NUL bytes are replaced."""
    assert sanitize_filename("a/b\0c") == "a_b_c"


def test_sanitize_filename_collapses_underscores() -> None:
    """Repeated underscores collapse into one."""
    assert sanitize_filename("a___b") == "a_b"


@pytest.mark.skipif(os.name != "nt", reason="reserved names only apply on Windows")
def test_sanitize_filename_reserved_name() -> None:
    """Windows device names are wrapped."""
    assert sanitize_filename("CON") == "_CON_"


def test_get_safe_filename_short_title_untouched() -> None:
    """Titles that fit are returned unchanged."""
    assert get_safe_filename("hello") == "hello.md"


def test_get_safe_filename_truncates_utf8_with_extension() -> None:
    """Long CJK titles fit the byte budget and keep the extension."""
    name = get_safe_filename("中" * 300)

    assert len(name.encode("utf-8")) <= 240
    assert name.endswith(".md")
    assert name.endswith("....md")


def test_safe_path_uses_byte_safe_name() -> None:
    """The path is joined with a truncated file name."""
    path = safe_path("/tmp/out", "中" * 300)

    assert path.startswith("/tmp/out/")
    assert os.path.basename(path).endswith(".md")
    assert len(os.path.basename(path).encode("utf-8")) <= 240


def test_build_yaml_frontmatter() -> None:
    """Metadata is serialised between delimiters."""
    frontmatter = build_yaml_frontmatter({"title": "hello"})

    assert frontmatter == "---\ntitle: hello\n---\n\n"


def test_write_markdown_with_frontmatter(tmp_path) -> None:
    """Frontmatter and body are written together."""
    path = write_markdown({"title": "hello"}, "body", str(tmp_path), filename="note")

    assert path.endswith("note.md")
    assert Path(path).read_text(encoding="utf-8") == "---\ntitle: hello\n---\n\nbody"


def test_write_markdown_without_frontmatter(tmp_path) -> None:
    """``with_frontmatter=False`` writes the body verbatim."""
    path = write_markdown(None, "body", str(tmp_path), filename="note", with_frontmatter=False)

    assert Path(path).read_text(encoding="utf-8") == "body"


def test_write_markdown_requires_metadata_with_frontmatter(tmp_path) -> None:
    """Frontmatter output without metadata is rejected."""
    with pytest.raises(ValueError):
        write_markdown(None, "body", str(tmp_path), filename="note")
