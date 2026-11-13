"""Pin fetching, page-state parsing, and Markdown rendering."""

from __future__ import annotations

import re
from typing import Any

from zhihu_cli.content.handlers import canonical_pin_url, fmt_time
from zhihu_cli.content.handlers.requests import fetch_page_html, get_page_state
from zhihu_cli.content.utils.html2markdown import converter
from zhihu_cli.content.utils.text_clean import strip_html_tags


def _canonical_pin_url(pin_url: str) -> str:
    """Normalise a pin reference to ``https://www.zhihu.com/pin/<id>``.

    Accepts a bare pin ID/token or a full ``/pin/<id>`` URL.  Other URLs are
    returned unchanged so the caller sees Zhihu's own error.
    """
    return canonical_pin_url(pin_url)


def _pick(item: dict[str, Any], *keys: str) -> Any:
    """Return the value of the first of *keys* that *item* actually carries.

    Pins reach this module in two shapes: API responses spell fields in
    snake_case, while page-state entities use camelCase.  Only one of the
    two spellings is ever present, and the missing one would otherwise be
    read as a default rather than as a mismatch.

    :param item: Pin entity to read from.
    :param keys: Candidate keys, most preferred first.
    :returns: The first non-``None`` value found, or ``None`` if there is none.
    """
    for key in keys:
        if item.get(key) is not None:
            return item[key]
    return None


def _content_plain_text(content: Any) -> str:
    """Flatten a pin's ``content`` field to plain, single-line text.

    Page-state pins carry ``content`` as a list of blocks (text, image, video,
    link cards), while API responses may carry a plain string.  Only the text
    a block actually holds is kept — images and videos contribute nothing —
    and the result is stripped of HTML marking, unescaped, and collapsed onto
    one line so it can be used as a title/preview.

    :param content: A pin's ``content`` value in either shape.
    :returns: The concatenated plain text, or an empty string.
    """
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            text = block.get("content") or block.get("ownText") or block.get("own_text") or ""
            if isinstance(text, str) and text:
                parts.append(text)
        content = " ".join(parts)
    if not isinstance(content, str):
        return ""
    return re.sub(r"\s+", " ", strip_html_tags(content, unescape=True)).strip()


def parse_pin_metadata(item: dict[str, Any], users: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a pin's metadata dict.

    :param item: Pin entity, either from the page state or from the API.
    :param users: The page state's ``users`` entity map. Page-scraped pins
        reference their author by url_token (a plain string) rather than
        inlining the author object, so the reference is resolved against
        this map; API responses inline the author as a dict instead.
    """
    pin_id = item.get("id", "")
    title = item.get("title", "") or f"pin {pin_id}"
    excerpt = item.get("excerpt")
    if not isinstance(excerpt, str):
        # Page-state pins have no ``excerpt`` field at all and the API leaves
        # it null, so fall back to the body.  The body is a list of blocks and
        # must not be sliced directly: callers expect a string and would
        # otherwise print the whole block list.
        excerpt = ""
    content_preview = excerpt or _content_plain_text(item.get("content", ""))[:200]

    # Pins do not separate 赞同 from 点赞, so the reaction count is the
    # vote count. ``voteup_count`` is what the field was originally read
    # as, but no pin response actually carries it.
    voteup_count = _pick(item, "voteup_count", "reaction_count", "reactionCount") or 0
    comment_count = _pick(item, "comment_count", "commentCount") or 0

    created = item.get("created", 0)
    updated = item.get("updated", 0)

    author = item.get("author") or {}
    if isinstance(author, str):
        # Page entities address users by url_token; the entry itself can
        # still be null when the user is not included in the payload.
        author = (users or {}).get(author) or {}
    elif not isinstance(author, dict):
        author = {}
    author_name = author.get("name") or "unknown"

    # Pin content may be in list format
    content_raw = item.get("content", "")
    if isinstance(content_raw, list) and content_raw:
        content_raw = content_raw[0].get("content", "")

    url = item.get("url", "")
    if not url and pin_id:
        url = f"https://www.zhihu.com/pin/{pin_id}"

    return {
        "id": pin_id,
        "title": title,
        "excerpt": content_preview,
        "url": url,
        "created_time": fmt_time(created),
        "updated_time": fmt_time(updated),
        "stats": {"voteup_count": voteup_count, "comment_count": comment_count},
        "author": {
            "name": author_name,
            "headline": author.get("headline") or "",
        },
        "comment_permission": _pick(item, "comment_permission", "commentPermission") or "",
        "ip": _pick(item, "ipInfo", "ip_info") or "",
    }


def _link_card_markdown(block: dict[str, Any]) -> str:
    """Render a repin's ``link_card`` block as a single Markdown link.

    The card carries the title and URL of whatever was reposted. The title
    is read through :func:`_pick` because API responses spell it
    ``data_draft_title`` where page-state entities use ``dataDraftTitle``.

    :param block: A ``link_card`` content block.
    :returns: A Markdown link, or an empty string when the block has no URL.
    """
    url = block.get("url") or ""
    if not url:
        return ""
    title = _pick(block, "data_draft_title", "dataDraftTitle") or ""
    return f"[{title or url}]({url})"


def _render_body(item: dict[str, Any]) -> str:
    """Render a pin's body as Markdown.

    A pin's ``content`` is a list of blocks rather than one document, rendered
    in order. Text blocks carry HTML and go through the usual converter;
    ``image`` blocks become ``![alt](url)``, ``video`` blocks become
    ``[video](url)``, and a repin's ``link_card`` block becomes a plain
    Markdown link to the reposted content. Poll and other unknown block types
    contribute nothing.

    :param item: Pin entity, either from the page state or from the API.
    :returns: The pin body as Markdown.
    """
    blocks = item.get("content")
    if not isinstance(blocks, list):
        return converter.convert(blocks or "")

    parts: list[str] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            text = converter.convert(block.get("content") or "")
            if text:
                parts.append(text)
        elif block_type == "image":
            url = block.get("url")
            if url:
                alt = block.get("alt") or "image"
                parts.append(f"![{alt}]({url})")
        elif block_type == "video":
            url = block.get("url")
            if url:
                parts.append(f"[video]({url})")
        elif block_type == "link_card":
            link = _link_card_markdown(block)
            if link:
                parts.append(link)

    return "\n\n".join(part for part in parts if part)


def scrape_pin(pin_url: str) -> tuple[dict[str, Any], str]:
    """Scrape a single pin page.

    :param pin_url: Full URL to a Zhihu pin, or a bare pin ID.
    :returns: ``(metadata, markdown)``.
    """
    entities = get_page_state(fetch_page_html(_canonical_pin_url(pin_url)))
    item = entities.get("pins", {})
    if not item:
        raise ValueError("No pins data found in entities")

    item_data = next(iter(item.values()))
    return parse_pin_metadata(item_data, entities.get("users", {})), _render_body(item_data)
