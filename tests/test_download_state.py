"""Tests for the incremental download state manifest."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

from zhihu_cli.content.download_state import DownloadState, delete_stale_file


def _write(path: Path, content: str = "body") -> Path:
    """Create *path* with *content*, including parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _state(base: Path) -> DownloadState:
    """Return a loaded state rooted at *base*."""
    state = DownloadState(str(base))
    state.load()
    return state


def test_record_save_load_round_trip(tmp_path: Path) -> None:
    """Recording an item persists kind, id, url, timestamps, and relpath."""
    _write(tmp_path / "answers" / "a.md")
    state = _state(tmp_path)
    state.record(
        "answers",
        "42",
        url="https://www.zhihu.com/question/1/answer/42",
        updated_time="2024-01-02 03:04:05",
        filepath="answers/a.md",
    )
    state.save()

    raw = json.loads((tmp_path / ".zhihu-download.json").read_text(encoding="utf-8"))
    assert raw["version"] == 1
    assert raw["items"]["answers:42"]["filepath"] == "answers/a.md"
    assert raw["items"]["answers:42"]["kind"] == "answers"

    reloaded = _state(tmp_path)
    entry = reloaded.get("answers", "42")
    assert entry is not None
    assert entry["url"] == "https://www.zhihu.com/question/1/answer/42"
    assert entry["updated_time"] == "2024-01-02 03:04:05"
    assert entry["downloaded_at"]


def test_is_fresh_without_entry(tmp_path: Path) -> None:
    """Unknown items are never fresh."""
    assert not _state(tmp_path).is_fresh("answers", "1", "2024-01-01 00:00:00")


def test_is_fresh_unchanged_timestamp(tmp_path: Path) -> None:
    """An unchanged listing timestamp means the file can be reused."""
    _write(tmp_path / "answers" / "a.md")
    state = _state(tmp_path)
    state.record("answers", "1", url="u", updated_time="2024-01-02 00:00:00", filepath="answers/a.md")

    assert state.is_fresh("answers", "1", "2024-01-02 00:00:00")
    assert state.is_fresh("answers", "1", "2024-01-01 00:00:00")


def test_is_fresh_newer_timestamp(tmp_path: Path) -> None:
    """A newer listing timestamp forces a re-download."""
    _write(tmp_path / "answers" / "a.md")
    state = _state(tmp_path)
    state.record("answers", "1", url="u", updated_time="2024-01-02 00:00:00", filepath="answers/a.md")

    assert not state.is_fresh("answers", "1", "2024-01-03 00:00:00")


def test_is_fresh_missing_file(tmp_path: Path) -> None:
    """A recorded file that vanished is not fresh."""
    state = _state(tmp_path)
    state.record("answers", "1", url="u", updated_time="2024-01-02 00:00:00", filepath="answers/a.md")

    assert not state.is_fresh("answers", "1", "2024-01-02 00:00:00")


def test_bootstrapped_entry_compares_against_mtime(tmp_path: Path) -> None:
    """Bootstrapped entries have no updated_time and fall back to downloaded_at."""
    path = _write(
        tmp_path / "answers" / "old.md",
        "---\nsource: https://www.zhihu.com/question/1/answer/7\n---\n\nbody",
    )
    old_ts = 1_600_000_000  # 2020-09-13 12:26:40 local
    os.utime(path, (old_ts, old_ts))

    state = _state(tmp_path)
    state.bootstrap("answers", str(tmp_path / "answers"))

    entry = state.get("answers", "7")
    assert entry is not None
    assert entry["updated_time"] is None
    assert entry["filepath"] == "answers/old.md"

    assert state.is_fresh("answers", "7", "2019-01-01 00:00:00")
    assert not state.is_fresh("answers", "7", "2030-01-01 00:00:00")


def test_bootstrap_parses_handwritten_markdown(tmp_path: Path) -> None:
    """The source URL in frontmatter is used to derive each item id."""
    _write(
        tmp_path / "articles" / "one.md",
        "---\nsource: https://zhuanlan.zhihu.com/p/12345\ntitle: One\n---\n\nbody",
    )
    _write(tmp_path / "pins" / "two.md", "---\nsource: https://www.zhihu.com/pin/99abc\n---\n\nbody")

    state = _state(tmp_path)
    state.bootstrap("articles", str(tmp_path / "articles"))
    state.bootstrap("pins", str(tmp_path / "pins"))

    assert state.get("articles", "12345") is not None
    assert state.get("pins", "99abc") is not None


def test_bootstrap_does_not_overwrite_existing_entries(tmp_path: Path) -> None:
    """Once a kind has entries, bootstrap is a no-op."""
    _write(tmp_path / "answers" / "a.md")
    state = _state(tmp_path)
    state.record("answers", "1", url="u", updated_time="2024-01-01 00:00:00", filepath="answers/a.md")

    _write(
        tmp_path / "answers" / "b.md",
        "---\nsource: https://www.zhihu.com/question/1/answer/2\n---\n\nbody",
    )
    state.bootstrap("answers", str(tmp_path / "answers"))

    assert state.get("answers", "2") is None
    assert state.get("answers", "1") is not None


def test_path_in_use_and_delete_stale_file(tmp_path: Path) -> None:
    """A renamed file is deleted only when no entry still references it."""
    old = _write(tmp_path / "answers" / "old.md")
    new = _write(tmp_path / "answers" / "new.md")

    state = _state(tmp_path)
    state.record("answers", "1", url="u", updated_time=None, filepath="answers/old.md")
    assert state.path_in_use("answers/old.md")

    # Another entry keeps the old path alive.
    state.record("answers", "2", url="u", updated_time=None, filepath="answers/old.md")
    delete_stale_file(str(tmp_path), "answers/old.md", "answers/new.md", state)
    assert old.exists()

    state.forget("answers", "2")
    state.record("answers", "1", url="u", updated_time=None, filepath="answers/new.md")
    assert not state.path_in_use("answers/old.md")
    delete_stale_file(str(tmp_path), "answers/old.md", "answers/new.md", state)
    assert not old.exists()
    assert new.exists()


def test_delete_stale_file_keeps_target(tmp_path: Path) -> None:
    """The file just written is never deleted."""
    target = _write(tmp_path / "answers" / "new.md")
    state = _state(tmp_path)

    delete_stale_file(str(tmp_path), "answers/new.md", "answers/new.md", state)
    assert target.exists()


def test_malformed_manifest_is_tolerated(tmp_path: Path) -> None:
    """A corrupt manifest warns and starts empty."""
    (tmp_path / ".zhihu-download.json").write_text("{not json", encoding="utf-8")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        state = DownloadState(str(tmp_path))
        state.load()

    assert state.get("answers", "1") is None
    assert caught


def test_forget_returns_previous_filepath(tmp_path: Path) -> None:
    """forget removes the entry and reports its file path."""
    state = _state(tmp_path)
    state.record("answers", "1", url="u", updated_time=None, filepath="answers/a.md")

    assert state.forget("answers", "1") == "answers/a.md"
    assert state.get("answers", "1") is None
