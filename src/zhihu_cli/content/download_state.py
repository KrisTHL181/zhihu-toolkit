"""Incremental download bookkeeping for the ``zhihu download`` commands.

The manifest lives in the download output directory as
``.zhihu-download.json`` and records, per item, the last listing timestamp and
the relative path of the file written for it.  Freshness is a plain string
comparison because every timestamp is a local naive ``%Y-%m-%d %H:%M:%S``
string, matching :func:`zhihu_cli.content.handlers.fmt_time`.

Only stdlib modules are imported at module level: ``markdown_file`` is
stdlib-only on import too (its ``yaml`` import is deferred), so the command
module can load this without pulling in ``curl_cffi`` or ``lxml``.
"""

from __future__ import annotations

import json
import os
import re
import threading
import warnings
from datetime import datetime
from typing import Any

from zhihu_cli.content.utils.markdown_file import read_frontmatter

__all__ = [
    "DownloadState",
    "delete_stale_file",
    "derive_id",
]

#: Format shared with ``handlers.fmt_time`` so string comparison is valid.
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

#: Support this many recorded items between automatic saves.
_SAVE_EVERY = 25

#: ``kind`` → regex extracting the item id from a saved ``source`` URL.
_SOURCE_ID_PATTERNS: dict[str, re.Pattern[str]] = {
    "answers": re.compile(r"/answer/(\d+)"),
    "articles": re.compile(r"/p/(\d+)"),
    "pins": re.compile(r"/pin/([^/?#]+)"),
}


def _now() -> str:
    """Return the current local time in :data:`TIME_FORMAT`."""
    return datetime.now().strftime(TIME_FORMAT)


def _relative(base_dir: str, filepath: str) -> str:
    """Return *filepath* relative to *base_dir* with POSIX separators."""
    return os.path.relpath(filepath, base_dir).replace(os.sep, "/")


def derive_id(kind: str, source: str) -> str | None:
    """Extract an item id from a Zhihu ``source`` URL or web URL.

    :param kind: One of ``"answers"``, ``"articles"``, or ``"pins"``.
    :param source: A URL such as ``https://www.zhihu.com/question/1/answer/2``.
    :returns: The id, or ``None`` when *kind* is unknown or *source* does not
        match the expected shape.
    """
    pattern = _SOURCE_ID_PATTERNS.get(kind)
    if pattern is None or not source:
        return None
    match = pattern.search(source)
    return match.group(1) if match else None


def delete_stale_file(base_dir: str, relpath: str, keep_relpath: str, state: DownloadState | None) -> None:
    """Delete a renamed download left behind by a re-download.

    The file is only removed when it is not the file that was just written
    (*keep_relpath*) and no other manifest entry still points at it.

    :param base_dir: The download state's root directory.
    :param relpath: POSIX relative path of the candidate file.
    :param keep_relpath: POSIX relative path of the file just written.
    :param state: Manifest used to check whether *relpath* is still in use.
    """
    if not relpath or relpath == keep_relpath:
        return
    if state is not None and state.path_in_use(relpath):
        return

    path = os.path.join(base_dir, *relpath.split("/")) if "/" in relpath else os.path.join(base_dir, relpath)
    try:
        os.remove(path)
    except OSError:
        # Missing files and permission failures are both non-fatal here: the
        # download itself succeeded, and the next sync will re-evaluate it.
        pass


class DownloadState:
    """Thread-safe incremental-download manifest for one output directory.

    :param base_dir: Directory containing the downloaded content.  The manifest
        is ``<base_dir>/.zhihu-download.json`` and all recorded file paths are
        relative to it.
    """

    def __init__(self, base_dir: str) -> None:
        self.base_dir = str(base_dir)
        self.path = os.path.join(self.base_dir, ".zhihu-download.json")
        self._lock = threading.RLock()
        self._items: dict[str, dict[str, Any]] = {}
        self._records_since_save = 0
        self.updated_at: str | None = None

    # ── persistence ────────────────────────────────────────────────────────

    def load(self) -> None:
        """Read the manifest, tolerating a missing or malformed file.

        A missing file is the normal first-run case and is silent.  A file that
        exists but cannot be parsed produces a :class:`RuntimeWarning`, and the
        state starts empty so a later :meth:`save` can rewrite it.
        """
        try:
            with open(self.path, encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            warnings.warn(f"could not read download state {self.path}: {exc}", RuntimeWarning, stacklevel=2)
            return

        if not isinstance(raw, dict):
            warnings.warn(f"download state {self.path} is not an object; starting empty", RuntimeWarning, stacklevel=2)
            return

        items = raw.get("items")
        if not isinstance(items, dict):
            if items is not None:
                warnings.warn(f"download state {self.path} has invalid items; starting empty", RuntimeWarning)
            items = {}

        with self._lock:
            self._items = {key: dict(value) for key, value in items.items() if isinstance(value, dict)}
            self.updated_at = raw.get("updated_at")

    def save(self) -> None:
        """Write the manifest atomically via a sibling ``.tmp`` file."""
        with self._lock:
            self.updated_at = _now()
            payload = {
                "version": 1,
                "updated_at": self.updated_at,
                "items": {key: dict(value) for key, value in self._items.items()},
            }
            os.makedirs(self.base_dir, exist_ok=True)
            tmp_path = self.path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.write("\n")
            os.replace(tmp_path, self.path)
            self._records_since_save = 0

    # ── queries ────────────────────────────────────────────────────────────

    def get(self, kind: str, item_id: str) -> dict[str, Any] | None:
        """Return a copy of the entry for *kind*/*item_id*, or ``None``."""
        with self._lock:
            entry = self._items.get(f"{kind}:{item_id}")
            return dict(entry) if entry is not None else None

    def is_fresh(self, kind: str, item_id: str, updated_time: str | None) -> bool:
        """Return whether a listing item can be skipped.

        An item is fresh when a manifest entry exists, its file still exists on
        disk, and the listing's ``updated_time`` has not advanced past the
        recorded one.  An unknown listing time (blank, ``"unknown time"``, or
        ``None``) is treated as "no evidence of change", so the existing file
        is kept.  A bootstrapped entry has no recorded ``updated_time`` and
        falls back to its ``downloaded_at`` timestamp.

        :param kind: Content kind (``"answers"``, ``"articles"``, ``"pins"``).
        :param item_id: The item's numeric (or pin) id.
        :param updated_time: Listing timestamp, or ``None`` when unavailable.
        :returns: ``True`` when the local file can be reused.
        """
        entry = self.get(kind, item_id)
        if entry is None:
            return False

        filepath = entry.get("filepath")
        if not filepath or not os.path.exists(os.path.join(self.base_dir, *str(filepath).split("/"))):
            return False

        listing = (updated_time or "").strip()
        if not listing or listing == "unknown time":
            return True

        recorded = entry.get("updated_time")
        if recorded:
            return listing <= str(recorded)
        return listing <= str(entry.get("downloaded_at") or "")

    def path_in_use(self, relpath: str) -> bool:
        """Return whether any manifest entry refers to *relpath*."""
        with self._lock:
            return any(entry.get("filepath") == relpath for entry in self._items.values())

    # ── mutation ───────────────────────────────────────────────────────────

    def record(
        self,
        kind: str,
        item_id: str,
        *,
        url: str,
        updated_time: str | None,
        filepath: str,
    ) -> None:
        """Record a successful download.

        :param kind: Content kind.
        :param item_id: The item's id.
        :param url: The Zhihu URL the item was downloaded from.
        :param updated_time: Listing timestamp, or ``None`` when unavailable.
        :param filepath: File path relative to ``base_dir``.
        """
        relpath = filepath.replace(os.sep, "/")
        with self._lock:
            self._items[f"{kind}:{item_id}"] = {
                "kind": kind,
                "id": item_id,
                "url": url,
                "updated_time": updated_time or None,
                "downloaded_at": _now(),
                "filepath": relpath,
            }
        self._maybe_save()

    def _maybe_save(self) -> None:
        """Save every :data:`_SAVE_EVERY` records to bound lost work."""
        with self._lock:
            self._records_since_save += 1
            if self._records_since_save < _SAVE_EVERY:
                return
            self._records_since_save = 0
        self.save()

    def forget(self, kind: str, item_id: str) -> str | None:
        """Remove an entry and return its previous relative filepath.

        :returns: The relative filepath that was recorded, or ``None``.
        """
        with self._lock:
            entry = self._items.pop(f"{kind}:{item_id}", None)
        return entry.get("filepath") if entry is not None else None

    def bootstrap(self, kind: str, directory: str) -> None:
        """Seed entries from Markdown files that predate the manifest.

        Runs only when the manifest has no entry of *kind* at all, so it never
        overwrites tracked state.  Each ``*.md`` file's YAML ``source`` URL is
        parsed into an item id, and the file's mtime becomes ``downloaded_at``;
        ``updated_time`` is left ``None`` so :meth:`is_fresh` falls back to the
        mtime.

        :param kind: Content kind to scan for.
        :param directory: Directory containing that kind's ``*.md`` files.
        """
        with self._lock:
            prefix = f"{kind}:"
            if any(key.startswith(prefix) for key in self._items):
                return

        if not os.path.isdir(directory):
            return

        for name in sorted(os.listdir(directory)):
            if not name.lower().endswith(".md"):
                continue
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue

            frontmatter = read_frontmatter(path)
            source = frontmatter.get("source")
            if not isinstance(source, str):
                continue
            item_id = derive_id(kind, source)
            if item_id is None:
                continue

            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue

            relpath = _relative(self.base_dir, path)
            with self._lock:
                key = f"{kind}:{item_id}"
                if key in self._items:
                    continue
                self._items[key] = {
                    "kind": kind,
                    "id": item_id,
                    "url": source,
                    "updated_time": None,
                    "downloaded_at": datetime.fromtimestamp(mtime).strftime(TIME_FORMAT),
                    "filepath": relpath,
                }
