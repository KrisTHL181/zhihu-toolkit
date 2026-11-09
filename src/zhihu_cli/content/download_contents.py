from __future__ import annotations

import html
import mimetypes
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from zhihu_cli.content.download_state import delete_stale_file, derive_id
from zhihu_cli.content.handlers.article import fetch_article_item, parse_article_metadata
from zhihu_cli.content.handlers.cache_manager import cache_manager
from zhihu_cli.content.handlers.pin import scrape_pin
from zhihu_cli.content.handlers.question import scrape_answer_page
from zhihu_cli.content.handlers.requests import reload_session, session
from zhihu_cli.content.utils.html2markdown import PageToMarkdown
from zhihu_cli.content.utils.markdown_file import (
    build_yaml_frontmatter,
    get_safe_filename,
    sanitize_filename,
    write_markdown,
)
from zhihu_cli.content.utils.wait import wait
from zhihu_cli.daemon.proxy import DaemonHTTPError
from zhihu_cli.paths import zhihu_home

if TYPE_CHECKING:
    from zhihu_cli.content.download_state import DownloadState

__all__ = [
    "ContentDownloader",
    "build_yaml_frontmatter",
    "download_media_files",
    "extract_config_from_curl",
    "extract_metadata_from_html",
    "get_safe_filename",
    "sanitize_filename",
    "save_article",
    "save_pin",
]

# ── helpers ──────────────────────────────────────────────────────────────────

_orig_print = print

#: Parallel workers print progress lines; serialise them so lines never mix.
_print_lock = threading.Lock()

#: Parallel workers also pick media filenames; the check-then-create loop below
#: is racy without a lock, and two items sharing an alt-text stem would
#: otherwise overwrite each other's image.
_media_lock = threading.Lock()


def _cli_print(*args: object, **kwargs: object) -> None:
    """Print to stderr (avoids stdout pollution during --json mode)."""
    with _print_lock:
        _orig_print(*args, file=sys.stderr, **kwargs)


# ── media download helpers ──────────────────────────────────────────────────

# Regex patterns for media references in markdown / html
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
_HTML_VIDEO_RE = re.compile(r'<video[^>]*\ssrc="([^"]+)"[^>]*>')
_HTML_IMG_RE = re.compile(r'<img[^>]*\ssrc="([^"]+)"[^>]*>')

_ZHIMG_HOST_RE = re.compile(r"\.zhimg\.com$", re.IGNORECASE)


def download_media_files(markdown: str, output_dir: str, progress_callback: object = None) -> tuple[str, int]:
    """Download media files referenced in *markdown* and rewrite URLs to local paths.

    Files are saved to ``<output_dir>/media/``.  Image alt-text is preferred as
    the filename stem; when unavailable a hash of the URL is used.  The session
    singleton is reused so Zhihu CDN images are fetched with auth cookies.

    If *progress_callback* is provided, it is called after each successful
    download as ``callback(current: int, total: int)``.

    Returns ``(updated_markdown, downloaded_count)``.
    """
    # ── collect (alt_text, url) pairs ──────────────────────────────────────
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()

    for m in _MD_IMAGE_RE.finditer(markdown):
        alt, url = m.group(1), m.group(2)
        if url and url not in seen:
            pairs.append((alt, url))
            seen.add(url)

    for pattern in (_HTML_IMG_RE, _HTML_VIDEO_RE):
        for m in pattern.finditer(markdown):
            url = m.group(1)
            if url and url not in seen:
                pairs.append(("", url))
                seen.add(url)

    if not pairs:
        return markdown, 0

    # ── download ───────────────────────────────────────────────────────────
    media_dir = os.path.join(output_dir, "media")
    os.makedirs(media_dir, exist_ok=True)

    url_to_local: dict[str, str] = {}
    downloaded = 0

    for alt, url in pairs:
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
        except (OSError, ValueError, DaemonHTTPError) as exc:
            _cli_print(f"  [media] skipped {url}: {exc}")
            continue

        # Guess extension: Content-Type → URL path → .jpg fallback
        ext = ""
        content_type = resp.headers.get("Content-Type", "")
        if content_type:
            ext = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ""

        if not ext:
            parsed = urlparse(url)
            path = parsed.path
            # Strip Zhihu query params that mask the real extension
            if _ZHIMG_HOST_RE.search(parsed.hostname or ""):
                path = path.split("?")[0]
            guess = os.path.splitext(path)[1]
            if guess:
                ext = guess.lower()

        if not ext:
            ext = ".jpg"

        # Prefer alt-text as filename stem, otherwise hash the URL
        stem = sanitize_filename(alt)[:80] if alt and alt.strip() else ""
        if not stem:
            import hashlib

            stem = hashlib.sha256(url.encode()).hexdigest()[:12]

        # Deduplicate within this batch.  Parallel downloads share the media
        # directory, so claiming a name and writing the bytes has to happen
        # under one lock; otherwise two threads can both see a free name and
        # overwrite each other.
        with _media_lock:
            candidate = f"{stem}{ext}"
            filepath = os.path.join(media_dir, candidate)
            counter = 1
            while os.path.exists(filepath):
                candidate = f"{stem}_{counter}{ext}"
                filepath = os.path.join(media_dir, candidate)
                counter += 1

            with open(filepath, "wb") as f:
                f.write(resp.content)

        rel_path = os.path.join("media", candidate)
        url_to_local[url] = rel_path
        downloaded += 1
        if progress_callback is not None:
            progress_callback(downloaded, len(pairs))  # type: ignore[call-arg]

    if not url_to_local:
        return markdown, 0

    # ── rewrite references ─────────────────────────────────────────────────
    updated = markdown
    for remote_url, local_path in url_to_local.items():
        updated = updated.replace(remote_url, local_path)

    return updated, downloaded


def extract_config_from_curl(curl_text: str) -> tuple[str, dict[str, str]]:
    """Extract URL and headers from a cURL command."""
    url_match = re.search(r"curl\s+'([^']+)'", curl_text)
    full_url = url_match.group(1) if url_match else ""

    headers = {}
    header_matches = re.findall(r"-H\s+'([^']+)'", curl_text)
    for h in header_matches:
        if ":" in h:
            k, v = h.split(":", 1)
            headers[k.strip()] = v.strip()

    # Let curl_cffi handle Accept-Encoding
    headers.pop("Accept-Encoding", None)

    return full_url, headers


def extract_metadata_from_html(html_content: str) -> dict[str, str]:
    """Extract metadata (title, author, time) from HTML."""
    metadata = {"title": "untitled", "author": "unknown", "created": ""}

    # Extract title from <meta itemprop="name"> or <h1 class="Post-Title">
    title_pattern = (
        r'<meta\s+itemprop="name"\s+content="([^"]+)"|<h1[^>]*class="[^"]*Post-Title[^"]*"[^>]*>([^<]+)</h1>'
    )
    match = re.search(title_pattern, html_content)

    if match:
        raw_title = match.group(1) or match.group(2)
        clean_title = html.unescape(raw_title).strip()
        metadata["title"] = sanitize_filename(clean_title)

    # Extract author
    author_match = re.search(r'<a[^>]*class="[^"]*UserLink-link[^"]*"[^>]*>([^<]+)</a>', html_content)
    if author_match:
        metadata["author"] = sanitize_filename(author_match.group(1))

    # Extract publish time from <meta itemprop="datePublished">
    created_match = re.search(r'<meta\s+itemProp="datePublished"\s+content="([^"]+)"', html_content)
    if created_match:
        created_raw = created_match.group(1)
        try:
            created_dt = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
            metadata["created"] = created_dt.strftime("%Y-%m-%d")
        except ValueError:
            metadata["created"] = created_raw[:10] if len(created_raw) >= 10 else created_raw

    # Fallback: extract from page text
    if not metadata["created"]:
        time_match = re.search(r"published\s+(\d{4}-\d{2}-\d{2})\s+\d{2}:\d{2}", html_content)
        if time_match:
            metadata["created"] = time_match.group(1)

    return metadata


def save_article(url: str, metadata: dict[str, str], markdown: str, output_dir: str, with_media: bool = False) -> str:
    """Save a downloaded article as Markdown with YAML frontmatter.

    Returns the file path.
    """
    if with_media:
        markdown, n = download_media_files(markdown, output_dir)
        if n:
            metadata = {**metadata, "media_files": n}

    title = sanitize_filename(metadata.get("title", "untitled"))
    author = sanitize_filename(metadata.get("author", "unknown"))
    created = metadata.get("created", "") or "unknown"
    full_name = f"{title}_{author}_{created}"

    meta = {**metadata, "source": url}
    return write_markdown(meta, markdown, output_dir, filename=full_name)


def save_pin(url: str, metadata: dict[str, str], markdown: str, output_dir: str, with_media: bool = False) -> str:
    """Save a downloaded pin as Markdown with YAML frontmatter.

    Returns the file path.
    """
    if with_media:
        markdown, n = download_media_files(markdown, output_dir)
        if n:
            metadata = {**metadata, "media_files": n}

    author = sanitize_filename(metadata.get("author", "unknown"))
    created = metadata.get("created", "unknown")
    preview = re.sub(r"\s+", " ", markdown)[:30]
    preview = sanitize_filename(preview)
    if not preview:
        preview = metadata.get("pin_id", "unknown")
    full_name = f"{author}_{created}_{preview}"

    meta = {**metadata, "source": url}
    return write_markdown(meta, markdown, output_dir, filename=full_name)


class ContentDownloader:
    """Zhihu content downloader."""

    def __init__(self, output_dir: str | None = None, with_media: bool = False) -> None:
        if output_dir is None:
            output_dir = str(zhihu_home() / "downloads")
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.headers: dict[str, str] = {}
        self.md_converter: PageToMarkdown = PageToMarkdown(skip_empty=True)
        self.with_media = with_media

    def load_headers_from_curl(self, quick_mode: bool = False) -> bool:
        """Load request headers from user-pasted cURL command."""
        if quick_mode:
            headers = cache_manager.load_headers()
            if headers:
                self.headers = headers
                _cli_print("[Success] Loaded cached headers")
                return True

        _cli_print("--- Paste cURL command for article page ---")
        _cli_print("(Copy as cURL from browser DevTools)")
        curl_input = sys.stdin.read()

        if not curl_input.strip():
            _cli_print("[Error] No valid cURL input")
            return False

        url, headers = extract_config_from_curl(curl_input)

        if not headers:
            _cli_print("[Error] Could not extract headers from cURL")
            return False

        self.headers = headers
        cache_manager.save_headers(headers)
        reload_session()
        _cli_print("[Success] Headers configured and cached.")
        return True

    def _record_state(self, state: DownloadState | None, kind: str, url: str, result: dict) -> None:
        """Record a successful download in the incremental manifest.

        :param state: Manifest, or ``None`` to skip bookkeeping.
        :param kind: Content kind used as the manifest key prefix.
        :param url: The URL that produced *result*.
        :param result: A result dict from one of the download methods.
        """
        if state is None or not result.get("success") or not result.get("filepath"):
            return
        item_id = derive_id(kind, url)
        if item_id is None:
            return
        relpath = os.path.relpath(result["filepath"], self.output_dir).replace(os.sep, "/")
        previous = state.get(kind, item_id)
        state.record(kind, item_id, url=url, updated_time=None, filepath=relpath)
        if previous is not None:
            delete_stale_file(self.output_dir, str(previous.get("filepath") or ""), relpath, state)

    def _pending_urls(
        self,
        urls: list[str],
        *,
        state: DownloadState | None,
        kind: str,
        force: bool,
    ) -> list[str]:
        """Drop URLs whose item is already downloaded, unless *force* is set.

        :param urls: Candidate URLs.
        :param state: Manifest, or ``None`` to keep every URL.
        :param kind: Content kind used to derive the item id.
        :param force: When true, keep every URL.
        :returns: The URLs that still need downloading.
        """
        if state is None or force:
            return urls
        return [url for url in urls if not state.is_fresh(kind, derive_id(kind, url) or "", None)]

    def _answer_result(self, url: str) -> dict:
        """Download one answer, returning the per-item result dict."""
        try:
            metadata, answer_markdown = scrape_answer_page(url)

            meta = {
                "title": metadata.get("title", "untitled"),
                "question_detail": metadata.get("question_detail", ""),
                "author": metadata.get("author", "unknown"),
                "created": metadata.get("created", "unknown"),
            }
            filepath = save_article(url, meta, answer_markdown, self.output_dir, with_media=self.with_media)

            _cli_print(f"[Success] {url}")
            _cli_print(f"  -> {filepath}")
            _cli_print(f"  Question: {meta['title']}")
            _cli_print(f"  Author: {meta['author']}")
            _cli_print(f"  Time: {meta['created']}")

            return {
                "url": url,
                "title": meta["title"],
                "author": meta["author"],
                "filepath": filepath,
                "success": True,
                "error": None,
            }
        except TimeoutError:
            _cli_print(f"[Timeout] {url}")
            return {
                "url": url,
                "title": None,
                "author": None,
                "filepath": None,
                "success": False,
                "error": "Timeout",
            }
        except Exception as e:
            _cli_print(f"[Error] Failed to process {url}: {e}")
            return {
                "url": url,
                "title": None,
                "author": None,
                "filepath": None,
                "success": False,
                "error": str(e),
            }

    def download_answers(
        self,
        urls: list[str],
        delay: float = 1.0,
        *,
        concurrency: int = 1,
        state: DownloadState | None = None,
        kind: str = "answers",
        force: bool = False,
    ) -> list[dict]:
        """Download answer pages and convert to Markdown.

        :param urls: Answer URLs to download.
        :param delay: Throttle delay before each item.
        :param concurrency: Number of worker threads; ``1`` runs sequentially.
        :param state: Optional incremental manifest to filter and record
            against.
        :param kind: Manifest key prefix for this content type.
        :param force: Re-download items even when the manifest says they are
            fresh.
        :returns: list of result dicts with keys ``url``, ``title``, ``author``,
            ``filepath``, ``success``, and ``error``.  In parallel mode the
            results are in completion order.
        """
        results: list[dict] = []
        if not self.headers:
            _cli_print("[Error] Load headers first")
            return results

        pending = self._pending_urls(urls, state=state, kind=kind, force=force)

        if concurrency <= 1:
            for url in pending:
                result = self._answer_result(url)
                results.append(result)
                if result["success"]:
                    self._record_state(state, kind, url, result)
                    wait(delay)
            return results

        def task(url: str) -> dict:
            wait(delay)
            return self._answer_result(url)

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(task, url): url for url in pending}
            for future in as_completed(futures):
                url = futures[future]
                result = future.result()
                results.append(result)
                if result["success"]:
                    self._record_state(state, kind, url, result)
        return results

    def fetch_article(self, url: str) -> tuple[dict[str, str], str]:
        """Fetch and convert a single article. Returns (metadata, markdown_content)."""
        if not self.headers:
            raise RuntimeError("Headers not loaded")
        article = fetch_article_item(url)
        metadata = parse_article_metadata(article)

        # ── metadata from structured entity data ──────────────
        title = sanitize_filename(metadata["title"])
        author = sanitize_filename(metadata["author"]["name"])

        created = ""
        created_time = metadata.get("created_time", "")
        if created_time and created_time[0].isdigit():
            created = created_time[:10]

        meta = {"title": title, "author": author, "created": created}

        content_html = article.get("content", "")
        markdown_content = self.md_converter.convert(content_html, url)
        return meta, markdown_content

    def _article_result(self, url: str) -> dict:
        """Download one article, returning the per-item result dict."""
        try:
            metadata, markdown_content = self.fetch_article(url)
            filepath = save_article(url, metadata, markdown_content, self.output_dir, with_media=self.with_media)

            _cli_print(f"[Success] {url}")
            _cli_print(f"  -> {filepath}")
            _cli_print(f"  Title: {metadata['title']}")
            _cli_print(f"  Author: {metadata['author']}")
            _cli_print(f"  Time: {metadata['created']}")

            return {
                "url": url,
                "title": metadata["title"],
                "author": metadata["author"],
                "filepath": filepath,
                "success": True,
                "error": None,
            }
        except TimeoutError:
            _cli_print(f"[Timeout] {url}")
            return {
                "url": url,
                "title": None,
                "author": None,
                "filepath": None,
                "success": False,
                "error": "Timeout",
            }
        except Exception as e:
            _cli_print(f"[Error] Failed to process {url}: {e}")
            return {
                "url": url,
                "title": None,
                "author": None,
                "filepath": None,
                "success": False,
                "error": str(e),
            }

    def download_articles(
        self,
        urls: list[str],
        delay: float = 1.0,
        *,
        concurrency: int = 1,
        state: DownloadState | None = None,
        kind: str = "articles",
        force: bool = False,
    ) -> list[dict]:
        """Download article pages and convert to Markdown.

        :param urls: Article URLs to download.
        :param delay: Throttle delay before each item.
        :param concurrency: Number of worker threads; ``1`` runs sequentially.
        :param state: Optional incremental manifest to filter and record
            against.
        :param kind: Manifest key prefix for this content type.
        :param force: Re-download items even when the manifest says they are
            fresh.
        :returns: list of result dicts with keys ``url``, ``title``, ``author``,
            ``filepath``, ``success``, and ``error``.  In parallel mode the
            results are in completion order.
        """
        results: list[dict] = []
        if not self.headers:
            _cli_print("[Error] Load headers first")
            return results

        pending = self._pending_urls(urls, state=state, kind=kind, force=force)

        if concurrency <= 1:
            for url in pending:
                result = self._article_result(url)
                results.append(result)
                if result["success"]:
                    self._record_state(state, kind, url, result)
                    wait(delay)
            return results

        def task(url: str) -> dict:
            wait(delay)
            return self._article_result(url)

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(task, url): url for url in pending}
            for future in as_completed(futures):
                url = futures[future]
                result = future.result()
                results.append(result)
                if result["success"]:
                    self._record_state(state, kind, url, result)
        return results

    def download_pins(self, urls: list[str], delay: float = 1.0) -> None:
        """Download pin pages and convert to Markdown."""
        if not self.headers:
            _cli_print("[Error] Load headers first")
            return

        for url in urls:
            try:
                metadata, markdown_content = scrape_pin(url)
            except TimeoutError:
                _cli_print(f"[Timeout] {url}")
                continue
            except Exception as e:
                _cli_print(f"[Error] Failed to process {url}: {e}")
                continue

            author = metadata["author"]["name"] or "unknown"
            created_time = metadata.get("created_time", "")
            created = created_time[:10] if created_time and created_time[0].isdigit() else "unknown"

            meta = {
                "author": author,
                "created": created,
                "ip": metadata.get("ip", ""),
                "pin_id": metadata.get("id", ""),
            }
            if not markdown_content:
                markdown_content = "(no content)"

            filepath = save_pin(url, meta, markdown_content, self.output_dir, with_media=self.with_media)

            _cli_print(f"[Success] {url}")
            _cli_print(f"  -> {filepath}")
            _cli_print(f"  Author: {meta['author']}")
            _cli_print(f"  Time: {meta['created']}")
            _cli_print(f"  IP: {meta['ip']}")

            wait(delay)


def main() -> None:
    import argparse  # only needed when this module is run as a standalone script

    parser = argparse.ArgumentParser(description="Zhihu content downloader (HTML → Markdown)")

    parser.add_argument(
        "--download-articles", nargs="+", metavar="URL", help="Download article pages and convert to Markdown"
    )
    parser.add_argument("--download-answers", nargs="+", metavar="URL", help="Download answer pages")
    parser.add_argument("--download-pins", nargs="+", metavar="URL", help="Download pin pages")
    parser.add_argument("--output-dir", type=str, default=str(zhihu_home() / "downloads"), help="Output directory")

    args = parser.parse_args()

    downloader = ContentDownloader(output_dir=args.output_dir)

    if not downloader.load_headers_from_curl(quick_mode=True):
        sys.exit(1)

    try:
        if args.download_articles:
            downloader.download_articles(args.download_articles)
        elif args.download_answers:
            downloader.download_answers(args.download_answers)
        elif args.download_pins:
            downloader.download_pins(args.download_pins)
        else:
            parser.print_help()
            sys.exit(1)
    except NotImplementedError as e:
        _cli_print(f"[Error] {e}")
        sys.exit(1)
    except TimeoutError:
        _cli_print("[Fatal] Operation timed out")
        sys.exit(1)


if __name__ == "__main__":
    main()
