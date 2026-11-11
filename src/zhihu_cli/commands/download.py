"""Download Zhihu content as Markdown files."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import click

from zhihu_cli.commands._helpers import _extract_url_token, _save_markdown
from zhihu_cli.content.assets import asset_url, load_assets
from zhihu_cli.content.download_contents import (
    ContentDownloader,
    download_media_files,
    save_article,
    save_pin,
)
from zhihu_cli.content.download_state import DownloadState, delete_stale_file
from zhihu_cli.content.handlers import get_data_dir, get_type_and_id
from zhihu_cli.content.handlers.article import scrape_article
from zhihu_cli.content.handlers.collection import get_collection_meta, list_collection_contents
from zhihu_cli.content.handlers.people import (
    fetch_member_answers,
    fetch_member_articles,
    fetch_member_pins,
    fetch_member_profile,
)
from zhihu_cli.content.handlers.pin import scrape_pin
from zhihu_cli.content.handlers.question import (
    scrape_answer_page,
    scrape_answers,
    scrape_question_data,
)
from zhihu_cli.content.handlers.requests import session
from zhihu_cli.content.handlers.zvideo import get_best_video_url, scrape_zvideo
from zhihu_cli.content.utils.markdown_file import safe_path, sanitize_filename
from zhihu_cli.content.utils.wait import wait
from zhihu_cli.output import (
    echo,
    error,
    f_dim,
    f_label,
    f_meta,
    f_name,
    f_num,
    f_path,
    f_title,
    file_saved,
    info,
    item_index,
    print_json,
    section,
    set_json_mode,
    success,
    warning,
)

#: Serialises progress lines emitted by parallel download workers.
_output_lock = threading.Lock()


def _parse_since(ctx: click.Context, param: click.Parameter, value: str | None) -> str | None:
    """Validate ``--since`` as a date or date-time.

    :raises click.BadParameter: When the value matches neither accepted shape.
    """
    if value is None:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            parsed = datetime.strptime(value, fmt)
        except ValueError:
            continue
        # Re-render so a non-padded value (``2024-1-1``) compares correctly
        # against the zero-padded timestamps in the listings.
        return parsed.strftime(fmt)
    raise click.BadParameter("expected YYYY-MM-DD or 'YYYY-MM-DD HH:MM:SS'", ctx=ctx, param=param)


def _run_items(
    items: list,
    concurrency: int,
    task,
    on_result,
) -> None:
    """Run *task* over *items*, sequentially or in a thread pool.

    :param items: Items to process.
    :param concurrency: Worker count; ``<= 1`` runs in the calling thread.
    :param task: ``task(index, item)`` returning an outcome dict.
    :param on_result: Called with each outcome (in completion order when
        parallel).
    """
    if concurrency <= 1:
        for index, item in enumerate(items, 1):
            on_result(task(index, item))
        return

    executor = ThreadPoolExecutor(max_workers=concurrency)
    try:
        futures = [executor.submit(task, index, item) for index, item in enumerate(items, 1)]
        for future in as_completed(futures):
            on_result(future.result())
    except BaseException:
        # Do not make Ctrl-C wait for in-flight network requests: cancel what
        # has not started and let the caller's finally save state first.
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)


#: Manifest kinds handled by :func:`download_user` and :func:`download_collection`.
_ITEM_TASK_KINDS = ("answers", "articles", "pins")

#: Collection items use singular ``type`` values; the manifest uses plurals.
_COLLECTION_TYPE_TO_KIND = {"answer": "answers", "article": "articles", "pin": "pins"}


def _answer_item_task(
    index: int,
    total: int,
    item: dict,
    output_dir: str,
    with_media: bool,
    *,
    url: str | None = None,
    item_id: str | None = None,
) -> dict:
    """Scrape one answer, save it as Markdown, and return a worker outcome.

    :param index: 1-based position in the current batch, for progress output.
    :param total: Size of the batch, for ``[i/n]`` progress output.
    :param item: Listing item; its ``updated_time`` is carried into the outcome.
    :param output_dir: Directory the Markdown file is written to.
    :param with_media: Download referenced images/videos alongside the file.
    :param url: Canonical source URL; defaults to ``item["url"]``.
    :param item_id: Item ID to record; defaults to ``item["id"]``.
    :returns: ``{"status": "ok", ...}`` on success, ``{"status": "error", ...}``
        on failure.
    """
    source_url = url if url is not None else item.get("url", "")
    answer_id = item_id if item_id is not None else str(item.get("id", ""))
    try:
        meta, md = scrape_answer_page(source_url)
        save_meta = {
            "title": meta.get("title", "untitled"),
            "author": meta.get("author", "unknown"),
            "created": meta.get("created", "unknown"),
        }
        filepath = save_article(source_url, save_meta, md, output_dir, with_media=with_media)
        with _output_lock:
            echo(f"  {item_index(index, total)} {save_meta['title'][:50]} -> {f_path(os.path.basename(filepath))}")
        return {
            "status": "ok",
            "kind": "answers",
            "id": answer_id,
            "url": source_url,
            "updated_time": item.get("updated_time"),
            "filepath": filepath,
        }
    except Exception as e:
        with _output_lock:
            error(f"  {item_index(index, total)} Error: {e}")
        return {"status": "error", "kind": "answers", "url": source_url, "error": str(e)}


def _article_item_task(
    index: int,
    total: int,
    item: dict,
    output_dir: str,
    with_media: bool,
    *,
    url: str | None = None,
    item_id: str | None = None,
) -> dict:
    """Scrape one article, save it as Markdown, and return a worker outcome.

    :param index: 1-based position in the current batch, for progress output.
    :param total: Size of the batch, for ``[i/n]`` progress output.
    :param item: Listing item; its ``updated_time`` is carried into the outcome.
    :param output_dir: Directory the Markdown file is written to.
    :param with_media: Download referenced images/videos alongside the file.
    :param url: Canonical source URL; defaults to ``item["url"]``.
    :param item_id: Item ID to record; defaults to ``item["id"]``.
    :returns: ``{"status": "ok", ...}`` on success, ``{"status": "error", ...}``
        on failure.
    """
    source_url = url if url is not None else item.get("url", "")
    article_id = item_id if item_id is not None else str(item.get("id", ""))
    try:
        meta, md = scrape_article(source_url)
        author = meta.get("author", {})
        author_name = author.get("name", "unknown") if isinstance(author, dict) else str(author)
        save_meta = {
            "title": meta.get("title", "untitled"),
            "author": author_name,
            "created": (meta.get("created_time", "unknown") or "unknown")[:10],
        }
        filepath = save_article(source_url, save_meta, md, output_dir, with_media=with_media)
        with _output_lock:
            echo(f"  {item_index(index, total)} {save_meta['title'][:50]} -> {f_path(os.path.basename(filepath))}")
        return {
            "status": "ok",
            "kind": "articles",
            "id": article_id,
            "url": source_url,
            "updated_time": item.get("updated_time"),
            "filepath": filepath,
        }
    except Exception as e:
        with _output_lock:
            error(f"  {item_index(index, total)} Error: {e}")
        return {"status": "error", "kind": "articles", "url": source_url, "error": str(e)}


def _pin_item_task(
    index: int,
    total: int,
    item: dict,
    output_dir: str,
    with_media: bool,
    *,
    url: str | None = None,
    item_id: str | None = None,
) -> dict:
    """Scrape one pin, save it as Markdown, and return a worker outcome.

    :param index: 1-based position in the current batch, for progress output.
    :param total: Size of the batch, for ``[i/n]`` progress output.
    :param item: Listing item; its ``updated_time`` is carried into the outcome.
    :param output_dir: Directory the Markdown file is written to.
    :param with_media: Download referenced images/videos alongside the file.
    :param url: Canonical source URL; defaults to ``item["url"]``.
    :param item_id: Item ID to record; defaults to ``item["id"]``.
    :returns: ``{"status": "ok", ...}`` on success, ``{"status": "error", ...}``
        on failure.
    """
    source_url = url if url is not None else item.get("url", "")
    pin_id = item_id if item_id is not None else str(item.get("id", ""))
    try:
        meta, md = scrape_pin(source_url)
        author = meta.get("author", {})
        author_name = author.get("name", "unknown") if isinstance(author, dict) else str(author)
        save_meta = {
            "author": author_name,
            "created": (meta.get("created_time", "unknown") or "unknown")[:10],
            "pin_id": str(meta.get("id", "")),
        }
        filepath = save_pin(source_url, save_meta, md, output_dir, with_media=with_media)
        preview = (meta.get("excerpt", "") or "")[:30]
        with _output_lock:
            echo(f"  {item_index(index, total)} {preview} -> {f_path(os.path.basename(filepath))}")
        return {
            "status": "ok",
            "kind": "pins",
            "id": pin_id,
            "url": source_url,
            "updated_time": item.get("updated_time"),
            "filepath": filepath,
        }
    except Exception as e:
        with _output_lock:
            error(f"  {item_index(index, total)} Error: {e}")
        return {"status": "error", "kind": "pins", "url": source_url, "error": str(e)}


#: Manifest kind → per-item worker shared by the user and collection commands.
_ITEM_TASK_HELPERS = {
    "answers": _answer_item_task,
    "articles": _article_item_task,
    "pins": _pin_item_task,
}


def _pending_items(items: list[dict], kind: str, state: DownloadState, force: bool) -> list[dict]:
    """Return the listing items that still need downloading.

    :param items: Listing items for *kind*.
    :param kind: Manifest kind (``"answers"``, ``"articles"``, or ``"pins"``).
    :param state: Loaded incremental-download manifest.
    :param force: When true every item is pending.
    :returns: The pending subset, in listing order.
    """
    if force:
        return list(items)
    return [item for item in items if not state.is_fresh(kind, str(item.get("id", "")), item.get("updated_time"))]


def _parse_collection_id(ref: str) -> str:
    """Resolve a collection URL or bare numeric ID to its numeric ID.

    :param ref: A ``https://www.zhihu.com/collection/<id>`` URL or a bare
        numeric collection ID.
    :returns: The numeric collection ID.
    :raises click.BadParameter: When *ref* is neither shape.
    """
    item_type, item_id = get_type_and_id(ref)
    if item_type == "collections" and item_id:
        return item_id
    candidate = ref.strip()
    if candidate.isdigit():
        return candidate
    raise click.BadParameter("expected a collection URL or numeric collection ID")


def _collection_kind(item_type: str) -> str | None:
    """Map a collection item's singular ``type`` to its manifest kind.

    :param item_type: ``"answer"``, ``"article"``, ``"pin"``, or an unsupported
        value such as ``"zvideo"``.
    :returns: ``"answers"``, ``"articles"``, ``"pins"``, or ``None``.
    """
    return _COLLECTION_TYPE_TO_KIND.get(item_type)


def _canonical_item_url(kind: str, item_id: str) -> str:
    """Build the canonical Zhihu URL for a collection item.

    Canonical URLs matter because :meth:`DownloadState.bootstrap` matches
    ``/answer/<id>``, ``/p/<id>``, and ``/pin/<id>`` in the saved frontmatter's
    ``source`` field.

    :param kind: Manifest kind.
    :param item_id: The item's numeric (or pin) ID.
    :returns: The canonical web URL.
    :raises ValueError: When *kind* is not a supported manifest kind.
    """
    if kind == "answers":
        return f"https://www.zhihu.com/answer/{item_id}"
    if kind == "articles":
        return f"https://zhuanlan.zhihu.com/p/{item_id}"
    if kind == "pins":
        return f"https://www.zhihu.com/pin/{item_id}"
    raise ValueError(f"unsupported collection item kind: {kind}")


def _bucket_collection_items(items: list[dict]) -> tuple[dict[str, list[dict]], int]:
    """Group collection items by download kind.

    Unknown types (e.g. ``zvideo``) and items without an ID are dropped and
    counted as unsupported.

    :param items: Items from :func:`list_collection_contents`.
    :returns: ``(buckets, unsupported_count)``, where *buckets* always has the
        keys ``"answers"``, ``"articles"``, and ``"pins"``.
    """
    buckets: dict[str, list[dict]] = {kind: [] for kind in _ITEM_TASK_KINDS}
    unsupported = 0
    for item in items:
        kind = _collection_kind(str(item.get("type", "")))
        if kind is None or not str(item.get("id", "")):
            unsupported += 1
            continue
        buckets[kind].append(item)
    return buckets, unsupported


@click.group()
def download() -> None:
    """Download Zhihu content as Markdown files."""


@download.command("article")
@click.argument("url")
@click.option(
    "--output-dir",
    "-o",
    default=str(get_data_dir() / "downloads" / "articles"),
    help="Output directory",
)
@click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON")
@click.option("--with-media", is_flag=True, default=False, help="Download images/videos alongside Markdown")
def download_article(url: str, output_dir: str, output_json: bool, with_media: bool) -> None:
    """Download a single Zhihu article as Markdown."""
    metadata, markdown = scrape_article(url)
    if with_media:
        markdown, n = download_media_files(markdown, output_dir)
        if n:
            metadata = {**metadata, "media_files": n}
    filepath = _save_markdown(metadata, markdown, output_dir)
    if output_json:
        print_json({"metadata": metadata, "filepath": filepath})
        return
    echo(f"  {f_title(str(metadata.get('title', 'untitled')))}")
    file_saved(filepath)


@download.command("question")
@click.argument("url")
@click.option(
    "--output-dir",
    "-o",
    default=str(get_data_dir() / "downloads" / "questions"),
    help="Output directory",
)
@click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON")
@click.option("--with-media", is_flag=True, default=False, help="Download images/videos alongside Markdown")
def download_question(url: str, output_dir: str, output_json: bool, with_media: bool) -> None:
    """Download a Zhihu question and all its answers as Markdown."""
    q_meta, q_detail_md = scrape_question_data(url)
    os.makedirs(output_dir, exist_ok=True)

    title = sanitize_filename(q_meta.get("title", "untitled"))

    # Question detail
    if with_media:
        q_detail_md, _ = download_media_files(q_detail_md, output_dir)
    filepath = safe_path(output_dir, f"{title}_question")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {q_meta['title']}\n\n{q_detail_md}\n")

    ans_dir = os.path.join(output_dir, f"{title}_answers")
    os.makedirs(ans_dir, exist_ok=True)
    count = 0
    for ans in scrape_answers(q_meta):
        count += 1
        content = ans["content"]
        if with_media:
            content, _ = download_media_files(content, ans_dir)
        afile = safe_path(ans_dir, f"{count:04d}_{sanitize_filename(ans['author'])}")
        with open(afile, "w", encoding="utf-8") as f:
            f.write(f"# Answer by {ans['author']} (+{ans['vote']})\n\n{content}\n")

    if output_json:
        print_json({"metadata": q_meta, "filepath": filepath, "answers_count": count, "answers_dir": ans_dir})
        return

    echo(f"  {f_title('Question:')} {q_meta['title']}")
    file_saved(filepath)
    echo(f"  {f_num(count)} answers saved to {f_path(ans_dir)}")


@download.command("pin")
@click.argument("url")
@click.option(
    "--output-dir",
    "-o",
    default=str(get_data_dir() / "downloads" / "pins"),
    help="Output directory",
)
@click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON")
@click.option("--with-media", is_flag=True, default=False, help="Download images/videos alongside Markdown")
def download_pin(url: str, output_dir: str, output_json: bool, with_media: bool) -> None:
    """Download a single Zhihu pin as Markdown.

    URL may be a full pin URL or a bare pin ID.
    """
    metadata, markdown = scrape_pin(url)
    if with_media:
        markdown, n = download_media_files(markdown, output_dir)
        if n:
            metadata = {**metadata, "media_files": n}
    filepath = _save_markdown(metadata, markdown, output_dir)
    if output_json:
        print_json({"metadata": metadata, "filepath": filepath})
        return
    echo(f"  {f_title('Pin')} by {f_name(str(metadata.get('author', 'unknown')))}")
    file_saved(filepath)


@download.command("video")
@click.argument("url")
@click.option(
    "--output-dir",
    "-o",
    default=str(get_data_dir() / "downloads" / "videos"),
    help="Output directory",
)
@click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON")
@click.option("--no-download-video", is_flag=True, default=False, help="Skip downloading the video file")
def download_video(url: str, output_dir: str, output_json: bool, no_download_video: bool) -> None:
    """Download a Zhihu zvideo and its metadata.

    URL may be a full zvideo URL or a bare numeric zvideo ID.
    """
    set_json_mode(output_json)
    metadata, markdown = scrape_zvideo(url)
    filepath = _save_markdown(metadata, markdown, output_dir)

    video_path: str | None = None
    if not no_download_video:
        video_url = get_best_video_url(metadata)
        if video_url:
            os.makedirs(output_dir, exist_ok=True)
            title = sanitize_filename(metadata.get("title", "video"))
            ext = ".mp4"
            video_path = safe_path(output_dir, title, ext=ext)
            info(f"Downloading video ({metadata.get('quality_tiers', [{}])[0].get('tier', 'best')} quality)...")
            try:
                resp = session.get(video_url, timeout=300, stream=True)
                resp.raise_for_status()
                with open(video_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                success(f"Video saved to {f_path(video_path)}")
            except Exception as e:
                warning(f"Video download failed: {e}")
                video_path = None

    if output_json:
        result: dict = {"metadata": metadata, "filepath": filepath}
        if video_path:
            result["video_path"] = video_path
        print_json(result)
        return

    echo(f"  {f_title(str(metadata.get('title', 'untitled')))}")
    file_saved(filepath)
    if video_path:
        file_saved(video_path)


@download.command("batch-answers")
@click.option(
    "--input",
    "-i",
    "input_file",
    default=str(get_data_dir() / "exports" / "all_assets_list.json"),
    help="Asset list JSON (all_assets_list.json or a scrape export)",
)
@click.option(
    "--output-dir",
    "-o",
    default=str(get_data_dir() / "downloads" / "answers"),
    help="Output directory",
)
@click.option("--delay", "-d", type=float, default=1.0, help="Delay between requests (seconds)")
@click.option(
    "--concurrency",
    "-j",
    type=click.IntRange(1, 32),
    default=4,
    show_default=True,
    help="Number of parallel downloads",
)
@click.option("--force", is_flag=True, default=False, help="Re-download items already recorded in the state file")
@click.option("--no-cache-headers", is_flag=True, help="Force re-paste of cURL")
@click.option("--with-media", is_flag=True, default=False, help="Download images/videos alongside Markdown")
@click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON")
def download_batch_answers(
    input_file: str,
    output_dir: str,
    delay: float,
    concurrency: int,
    force: bool,
    no_cache_headers: bool,
    with_media: bool,
    output_json: bool,
) -> None:
    """Batch download all answers listed in an asset/scrape JSON file."""
    set_json_mode(output_json)
    if not os.path.exists(input_file):
        error(f"file not found: {input_file}")
        raise SystemExit(1)

    assets = load_assets(input_file)
    pairs = [(a["id"], asset_url(a)) for a in assets if a["type"] == "answer"]
    if not pairs:
        info("No answers found in the assets file.")
        return

    dl = ContentDownloader(output_dir=output_dir, with_media=with_media)
    if not dl.load_headers_from_curl(quick_mode=not no_cache_headers):
        raise SystemExit(1)

    state = DownloadState(output_dir)
    state.load()
    try:
        state.bootstrap("answers", output_dir)
        if force:
            selected = pairs
        else:
            selected = [(item_id, url) for item_id, url in pairs if not state.is_fresh("answers", item_id, None)]
        skipped = len(pairs) - len(selected)

        extra = f" {f_dim(f'{len(selected)} pending, {skipped} skipped.')}" if skipped else ""
        echo(f"  {f_label('Found')} {f_num(len(pairs))} {f_dim('answers.')}{extra}")
        if not selected:
            if output_json:
                print_json([])
            return

        urls = [url for _, url in selected]
        results = dl.download_answers(
            urls, delay=delay, concurrency=concurrency, state=state, kind="answers", force=force
        )
        if output_json:
            print_json(results)
            return
    finally:
        state.save()


@download.command("batch-articles")
@click.option(
    "--input",
    "-i",
    "input_file",
    default=str(get_data_dir() / "exports" / "all_assets_list.json"),
    help="Asset list JSON (all_assets_list.json or a scrape export)",
)
@click.option(
    "--output-dir",
    "-o",
    default=str(get_data_dir() / "downloads" / "articles"),
    help="Output directory",
)
@click.option("--delay", "-d", type=float, default=1.0, help="Delay between requests (seconds)")
@click.option(
    "--concurrency",
    "-j",
    type=click.IntRange(1, 32),
    default=4,
    show_default=True,
    help="Number of parallel downloads",
)
@click.option("--force", is_flag=True, default=False, help="Re-download items already recorded in the state file")
@click.option("--no-cache-headers", is_flag=True, help="Force re-paste of cURL")
@click.option("--with-media", is_flag=True, default=False, help="Download images/videos alongside Markdown")
@click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON")
def download_batch_articles(
    input_file: str,
    output_dir: str,
    delay: float,
    concurrency: int,
    force: bool,
    no_cache_headers: bool,
    with_media: bool,
    output_json: bool,
) -> None:
    """Batch download all articles listed in an asset/scrape JSON file."""
    set_json_mode(output_json)
    if not os.path.exists(input_file):
        error(f"file not found: {input_file}")
        raise SystemExit(1)

    assets = load_assets(input_file)
    pairs = [(a["id"], asset_url(a)) for a in assets if a["type"] == "article"]
    if not pairs:
        info("No articles found in the assets file.")
        return

    dl = ContentDownloader(output_dir=output_dir, with_media=with_media)
    if not dl.load_headers_from_curl(quick_mode=not no_cache_headers):
        raise SystemExit(1)

    state = DownloadState(output_dir)
    state.load()
    try:
        state.bootstrap("articles", output_dir)
        if force:
            selected = pairs
        else:
            selected = [(item_id, url) for item_id, url in pairs if not state.is_fresh("articles", item_id, None)]
        skipped = len(pairs) - len(selected)

        extra = f" {f_dim(f'{len(selected)} pending, {skipped} skipped.')}" if skipped else ""
        echo(f"  {f_label('Found')} {f_num(len(pairs))} {f_dim('articles.')}{extra}")
        if not selected:
            if output_json:
                print_json([])
            return

        urls = [url for _, url in selected]
        results = dl.download_articles(
            urls, delay=delay, concurrency=concurrency, state=state, kind="articles", force=force
        )
        if output_json:
            print_json(results)
            return
    finally:
        state.save()


@download.command("batch-pins")
@click.option(
    "--input",
    "-i",
    "input_file",
    default=str(get_data_dir() / "exports" / "all_assets_list.json"),
    help="Asset list JSON (all_assets_list.json or a scrape export)",
)
@click.option(
    "--output-dir",
    "-o",
    default=str(get_data_dir() / "downloads" / "pins"),
    help="Output directory",
)
@click.option("--delay", "-d", type=float, default=1.0, help="Delay between requests (seconds)")
@click.option(
    "--concurrency",
    "-j",
    type=click.IntRange(1, 32),
    default=4,
    show_default=True,
    help="Number of parallel downloads",
)
@click.option("--force", is_flag=True, default=False, help="Re-download items already recorded in the state file")
@click.option("--with-media", is_flag=True, default=False, help="Download images/videos alongside Markdown")
@click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON")
def download_batch_pins(
    input_file: str,
    output_dir: str,
    delay: float,
    concurrency: int,
    force: bool,
    with_media: bool,
    output_json: bool,
) -> None:
    """Batch download all pins listed in an asset/scrape JSON file."""
    set_json_mode(output_json)
    if not os.path.exists(input_file):
        error(f"file not found: {input_file}")
        raise SystemExit(1)

    assets = load_assets(input_file)
    items = [{"id": a["id"], "url": asset_url(a)} for a in assets if a["type"] == "pin"]
    if not items:
        info("No pins found in the assets file.")
        return

    state = DownloadState(output_dir)
    state.load()
    try:
        state.bootstrap("pins", output_dir)
        pending = _pending_items(items, "pins", state, force)
        skipped = len(items) - len(pending)

        extra = f" {f_dim(f'{len(pending)} pending, {skipped} skipped.')}" if skipped else ""
        echo(f"  {f_label('Found')} {f_num(len(items))} {f_dim('pins.')}{extra}")
        if not pending:
            if output_json:
                print_json([])
            return

        def pin_task(index: int, item: dict) -> dict:
            wait(delay)
            return _pin_item_task(index, len(pending), item, output_dir, with_media)

        results: list[dict] = []

        def handle(outcome: dict) -> None:
            results.append(outcome)
            if outcome.get("status") != "ok":
                return
            relpath = os.path.relpath(outcome["filepath"], output_dir).replace(os.sep, "/")
            state.record(
                "pins",
                outcome["id"],
                url=outcome["url"],
                updated_time=outcome.get("updated_time"),
                filepath=relpath,
            )

        _run_items(pending, concurrency, pin_task, handle)
        if output_json:
            print_json(results)
            return
    finally:
        state.save()


@download.command("user")
@click.argument("user")
@click.option(
    "--output-dir",
    "-o",
    default=None,
    help="Base output directory (default: ~/.zhihu-cli/downloads/<username>)",
)
@click.option("--delay", "-d", type=float, default=1.0, help="Delay between requests in seconds")
@click.option("--max", "-n", "max_items", type=int, default=None, help="Max items per content type (default: all)")
@click.option(
    "--type",
    "content_types",
    default="all",
    type=click.Choice(["answers", "articles", "pins", "all"]),
    help="Content types to download (default: all)",
)
@click.option(
    "--since",
    default=None,
    callback=_parse_since,
    help="Only fetch items created on/after YYYY-MM-DD (or 'YYYY-MM-DD HH:MM:SS')",
)
@click.option(
    "--concurrency",
    "-j",
    type=click.IntRange(1, 32),
    default=4,
    show_default=True,
    help="Number of parallel downloads",
)
@click.option("--force", is_flag=True, default=False, help="Re-download items already recorded in the state file")
@click.option("--with-media", is_flag=True, default=False, help="Download images/videos alongside Markdown")
@click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON")
def download_user(
    user: str,
    output_dir: str | None,
    delay: float,
    max_items: int | None,
    content_types: str,
    since: str | None,
    concurrency: int,
    force: bool,
    with_media: bool,
    output_json: bool,
) -> None:
    """Download all answers, articles, and pins from a Zhihu user.

    Incremental sync is on by default: items whose local file exists and whose
    listing timestamp has not advanced are skipped. Use --force to re-download
    everything.
    """
    set_json_mode(output_json)
    url_token = _extract_url_token(user)

    profile = fetch_member_profile(url_token)
    user_name = profile["name"] if profile else url_token
    echo(f"  {f_label('User:')} {f_name(user_name)} (url_token: {f_meta(url_token)})")

    if output_dir is None:
        base_dir = get_data_dir() / "downloads" / sanitize_filename(user_name)
    else:
        base_dir = Path(output_dir)

    downloaded: dict[str, int] = {"answers": 0, "articles": 0, "pins": 0}
    skipped: dict[str, int] = {"answers": 0, "articles": 0, "pins": 0}
    failed: list[dict[str, str]] = []

    state = DownloadState(str(base_dir))
    state.load()

    def _handle(outcome: dict) -> None:
        """Update counters and the manifest from a worker outcome."""
        kind = outcome["kind"]
        if outcome["status"] != "ok":
            failed.append({"url": outcome.get("url", ""), "error": outcome["error"]})
            return

        downloaded[kind] += 1
        relpath = os.path.relpath(outcome["filepath"], str(base_dir)).replace(os.sep, "/")
        previous = state.get(kind, outcome["id"])
        state.record(
            kind,
            outcome["id"],
            url=outcome["url"],
            updated_time=outcome.get("updated_time"),
            filepath=relpath,
        )
        if previous is not None:
            delete_stale_file(str(base_dir), str(previous.get("filepath") or ""), relpath, state)

    try:
        if content_types in ("answers", "all"):
            answers_dir = str(base_dir / "answers")
            info(f"\nFetching answers list for {user_name}...")
            answer_items = fetch_member_answers(url_token, max_items=max_items, since=since)
            state.bootstrap("answers", answers_dir)
            pending = _pending_items(answer_items, "answers", state, force)
            skipped["answers"] = len(answer_items) - len(pending)
            detail = f"answers ({len(pending)} pending, {skipped['answers']} skipped)."
            echo(f"  {f_label('Found')} {f_num(len(answer_items))} {f_dim(detail)}")

            def answer_task(index: int, item: dict) -> dict:
                wait(delay)
                return _answer_item_task(index, len(pending), item, answers_dir, with_media)

            _run_items(pending, concurrency, answer_task, _handle)

        if content_types in ("articles", "all"):
            articles_dir = str(base_dir / "articles")
            info(f"\nFetching articles list for {user_name}...")
            article_items = fetch_member_articles(url_token, max_items=max_items, since=since)
            state.bootstrap("articles", articles_dir)
            pending = _pending_items(article_items, "articles", state, force)
            skipped["articles"] = len(article_items) - len(pending)
            detail = f"articles ({len(pending)} pending, {skipped['articles']} skipped)."
            echo(f"  {f_label('Found')} {f_num(len(article_items))} {f_dim(detail)}")

            def article_task(index: int, item: dict) -> dict:
                wait(delay)
                return _article_item_task(index, len(pending), item, articles_dir, with_media)

            _run_items(pending, concurrency, article_task, _handle)

        if content_types in ("pins", "all"):
            pins_dir = str(base_dir / "pins")
            info(f"\nFetching pins list for {user_name}...")
            pin_items = fetch_member_pins(url_token, max_items=max_items, since=since)
            state.bootstrap("pins", pins_dir)
            pending = _pending_items(pin_items, "pins", state, force)
            skipped["pins"] = len(pin_items) - len(pending)
            detail = f"pins ({len(pending)} pending, {skipped['pins']} skipped)."
            echo(f"  {f_label('Found')} {f_num(len(pin_items))} {f_dim(detail)}")

            def pin_task(index: int, item: dict) -> dict:
                wait(delay)
                return _pin_item_task(index, len(pending), item, pins_dir, with_media)

            _run_items(pending, concurrency, pin_task, _handle)
    finally:
        state.save()

    if output_json:
        print_json(
            {
                "downloaded": downloaded,
                "skipped": skipped,
                "concurrency": concurrency,
                "base_dir": str(base_dir),
                "user_name": user_name,
                "url_token": url_token,
                "failed": failed,
                "failure_count": len(failed),
            }
        )
        if failed and not any(downloaded.values()):
            raise SystemExit(1)
        return

    section(f"Done! Downloaded from {f_name(user_name)}:")
    echo(f"  {f_label('Answers:')}  {f_num(downloaded['answers'])}")
    echo(f"  {f_label('Articles:')} {f_num(downloaded['articles'])}")
    echo(f"  {f_label('Pins:')}     {f_num(downloaded['pins'])}")
    echo(f"  {f_label('Skipped:')}  {f_num(sum(skipped.values()))}")
    echo(f"  {f_label('Failed:')}   {f_num(len(failed))}")
    echo(f"  {f_label('Output:')}   {f_path(str(base_dir))}")
    if failed and not any(downloaded.values()):
        raise SystemExit(1)


@download.command("collection")
@click.argument("collection")
@click.option(
    "--output-dir",
    "-o",
    default=None,
    help="Base output directory (default: ~/.zhihu-cli/downloads/collections/<title>_<id>)",
)
@click.option("--delay", "-d", type=float, default=1.0, help="Delay between requests in seconds")
@click.option("--max", "-n", "max_items", type=int, default=None, help="Max collection items (default: all)")
@click.option(
    "--concurrency",
    "-j",
    type=click.IntRange(1, 32),
    default=4,
    show_default=True,
    help="Number of parallel downloads",
)
@click.option("--force", is_flag=True, default=False, help="Re-download items already recorded in the state file")
@click.option("--with-media", is_flag=True, default=False, help="Download images/videos alongside Markdown")
@click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON")
def download_collection(
    collection: str,
    output_dir: str | None,
    delay: float,
    max_items: int | None,
    concurrency: int,
    force: bool,
    with_media: bool,
    output_json: bool,
) -> None:
    """Download all answers, articles, and pins in a Zhihu collection.

    COLLECTION may be a ``https://www.zhihu.com/collection/<id>`` URL or a bare
    numeric collection ID. Incremental sync is on by default: items whose local
    file exists and whose listing timestamp has not advanced are skipped. Use
    --force to re-download everything.
    """
    set_json_mode(output_json)
    collection_id = _parse_collection_id(collection)

    meta = get_collection_meta(collection_id)
    title = meta.get("title", "") or ""
    creator = meta.get("creator", {})
    creator_name = creator.get("name", "") if isinstance(creator, dict) else str(creator)
    echo(f"  {f_label('Collection:')} {f_title(title or '(untitled)')} ({f_meta(creator_name or 'unknown')})")

    if output_dir is None:
        # Collection titles are not unique (several can be called "My
        # Collection"), so the id keeps their output directories apart.
        folder = sanitize_filename(title) if title else ""
        base_dir = (
            get_data_dir() / "downloads" / "collections" / (f"{folder}_{collection_id}" if folder else collection_id)
        )
    else:
        base_dir = Path(output_dir)

    downloaded: dict[str, int] = {"answers": 0, "articles": 0, "pins": 0}
    skipped: dict[str, int] = {"answers": 0, "articles": 0, "pins": 0}
    failed: list[dict[str, str]] = []

    items = list_collection_contents(collection_id, max_items=max_items)
    buckets, unsupported = _bucket_collection_items(items)
    item_count = meta.get("item_count")
    if item_count is None:
        item_count = len(items)
    if unsupported:
        warning(f"Skipped {unsupported} unsupported or unidentified item(s).")

    state = DownloadState(str(base_dir))
    state.load()

    def _handle(outcome: dict) -> None:
        """Update counters and the manifest from a worker outcome."""
        kind = outcome["kind"]
        if outcome["status"] != "ok":
            failed.append({"url": outcome.get("url", ""), "error": outcome["error"]})
            return

        downloaded[kind] += 1
        relpath = os.path.relpath(outcome["filepath"], str(base_dir)).replace(os.sep, "/")
        previous = state.get(kind, outcome["id"])
        state.record(
            kind,
            outcome["id"],
            url=outcome["url"],
            updated_time=outcome.get("updated_time"),
            filepath=relpath,
        )
        if previous is not None:
            delete_stale_file(str(base_dir), str(previous.get("filepath") or ""), relpath, state)

    try:
        for kind in _ITEM_TASK_KINDS:
            batch = buckets[kind]
            if not batch:
                continue

            kind_dir = str(base_dir / kind)
            state.bootstrap(kind, kind_dir)
            pending = _pending_items(batch, kind, state, force)
            skipped[kind] = len(batch) - len(pending)
            detail = f"{kind} ({len(pending)} pending, {skipped[kind]} skipped)."
            echo(f"  {f_label('Found')} {f_num(len(batch))} {f_dim(detail)}")
            if not pending:
                continue

            helper = _ITEM_TASK_HELPERS[kind]
            total = len(pending)

            def task(
                index: int,
                item: dict,
                _helper=helper,
                _kind=kind,
                _dir=kind_dir,
                _total=total,
            ) -> dict:
                wait(delay)
                item_id = str(item.get("id", ""))
                return _helper(
                    index,
                    _total,
                    item,
                    _dir,
                    with_media,
                    url=_canonical_item_url(_kind, item_id),
                    item_id=item_id,
                )

            _run_items(pending, concurrency, task, _handle)
    finally:
        state.save()

    if output_json:
        print_json(
            {
                "collection": {
                    "id": collection_id,
                    "title": title,
                    "creator": creator_name,
                    "item_count": item_count,
                },
                "downloaded": downloaded,
                "skipped": skipped,
                "unsupported": unsupported,
                "concurrency": concurrency,
                "base_dir": str(base_dir),
                "failed": failed,
                "failure_count": len(failed),
            }
        )
        if failed and not any(downloaded.values()):
            raise SystemExit(1)
        return

    section(f"Done! Downloaded from collection {f_title(title or collection_id)}:")
    echo(f"  {f_label('Answers:')}  {f_num(downloaded['answers'])}")
    echo(f"  {f_label('Articles:')} {f_num(downloaded['articles'])}")
    echo(f"  {f_label('Pins:')}     {f_num(downloaded['pins'])}")
    echo(f"  {f_label('Skipped:')}  {f_num(sum(skipped.values()))}")
    echo(f"  {f_label('Failed:')}   {f_num(len(failed))}")
    echo(f"  {f_label('Output:')}   {f_path(str(base_dir))}")
    if failed and not any(downloaded.values()):
        raise SystemExit(1)


def register_download(main_group) -> None:
    """Register the download command group on the main CLI group."""
    main_group.add_command(download)
