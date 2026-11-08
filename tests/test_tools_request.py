"""Tests for ``zhihu tools request`` body handling.

Covers the streaming binary guard (peek the first 8 KiB, abort on NUL) and
the raw-byte write path used by ``--output``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import click
from click.testing import CliRunner

from zhihu_cli.commands.tools import _BINARY_PEEK_BYTES, _peek_body, _stream_text, register_tools
from zhihu_cli.content.handlers import request_tool

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any


class FakeStream:
    """Streaming response that yields canned chunks and counts what was read."""

    def __init__(self, chunks: list[bytes], encoding: str = "utf-8") -> None:
        self.chunks = chunks
        self.encoding = encoding
        self.consumed = 0

    def iter_content(self) -> Any:
        """Yield the canned chunks, recording how many bytes were consumed."""
        for chunk in self.chunks:
            self.consumed += len(chunk)
            yield chunk

    def close(self) -> None:
        """No-op: the fake owns no connection."""


class FakeResponse:
    """Minimal non-streaming response carrying binary content."""

    status_code = 200
    reason = "OK"
    headers = {"Content-Type": "application/octet-stream"}
    content = b"\x89PNG\r\n\x1a\n\x00\x00binary-payload"
    text = content.decode("utf-8", errors="replace")

    def close(self) -> None:
        """No-op: the fake owns no connection."""


def test_peek_body_stops_at_the_limit() -> None:
    """Only the first window is read; the rest stays unread for the caller."""
    resp = FakeStream([b"a" * 4096] * 10)

    head, chunks = _peek_body(resp)

    assert len(head) == _BINARY_PEEK_BYTES
    assert resp.consumed == _BINARY_PEEK_BYTES
    assert len(head) + sum(len(chunk) for chunk in chunks) == 40960


def test_peek_body_detects_nul_in_the_window() -> None:
    """A NUL in the leading chunks marks the payload as binary."""
    resp = FakeStream([b"header text", b"\x00binary"])

    head, _ = _peek_body(resp)

    assert b"\0" in head


def test_peek_body_returns_short_bodies_whole() -> None:
    """Bodies shorter than the window are returned in full."""
    resp = FakeStream([b"hello", b" ", b"world"])

    head, chunks = _peek_body(resp)

    assert head == b"hello world"
    assert list(chunks) == []


def test_stream_text_decodes_split_multibyte(capsys: Any) -> None:
    """A UTF-8 character split across the head/chunk boundary survives."""
    data = "中文".encode()

    _stream_text(FakeStream([]), data[:1], iter([data[1:]]))  # type: ignore[arg-type]

    assert capsys.readouterr().out == "中文\n"


def test_output_writes_raw_bytes(monkeypatch: Any, tmp_path: Path) -> None:
    """``--output`` must not run the body through a lossy text decode."""
    monkeypatch.setattr(request_tool, "send_request", lambda *args, **kwargs: FakeResponse())

    @click.group()
    def group() -> None:
        """Test root group."""

    register_tools(group)

    out = tmp_path / "blob.bin"
    result = CliRunner().invoke(group, ["tools", "request", "https://example.com/blob", "-o", str(out)])

    assert result.exit_code == 0, result.output
    assert out.read_bytes() == FakeResponse.content
