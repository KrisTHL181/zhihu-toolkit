from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from zhihu_cli.content.handlers import fmt_time, safe_json
from zhihu_cli.content.handlers.requests import fetch_json, session
from zhihu_cli.content.utils.html2markdown import converter

ARTICLE_API_URL = "https://www.zhihu.com/api/v4/articles/{article_id}"
ARTICLE_VOTERS_URL = "https://www.zhihu.com/api/v4/articles/{article_id}/voters"
ARTICLE_VOTERS_ACTION_URL = "https://www.zhihu.com/api/v4/articles/{article_id}/voters/{action}"
ARTICLE_LIKE_URL = "https://www.zhihu.com/api/v4/articles/{article_id}/like"
_ARTICLE_PATH_RE = re.compile(r"/(?:p|articles)/(\d+)/?$")


def extract_article_id(article_url: str) -> str:
    """Extract a numeric article ID from a Zhihu article URL or raw ID."""
    candidate = article_url.strip()
    if candidate.isdigit():
        return candidate

    match = _ARTICLE_PATH_RE.search(urlparse(candidate).path)
    if match:
        return match.group(1)

    raise ValueError(f"Could not extract a Zhihu article ID from {article_url!r}")


def fetch_article_item(article_url: str) -> dict[str, Any]:
    """Fetch raw article data from the JSON API."""
    article_id = extract_article_id(article_url)
    api_url = ARTICLE_API_URL.format(article_id=article_id)

    item = fetch_json(api_url)
    if "content" not in item:
        raise ValueError("Article API response does not contain content")
    return item


def parse_article_metadata(item: dict[str, Any]) -> dict[str, Any]:
    article_id = item.get("id", "")
    title = item.get("title", "untitled")
    excerpt = item.get("excerpt", "")
    content_preview = excerpt or (item.get("content", "")[:200] if item.get("content") else "")

    # Stats
    voteup_count = item.get("voteup_count", item.get("voteupCount", 0))
    comment_count = item.get("comment_count", item.get("commentCount", 0))
    favlists_count = item.get("favlists_count", item.get("favlistsCount", 0))

    # Timestamps
    created = item.get("created", 0)
    updated = item.get("updated", 0)

    # Author info
    author = item.get("author", {})
    author_name = author.get("name", "unknown")

    # Article URL
    url = item.get("url", "")
    if article_id and (not url or urlparse(url).netloc == "api.zhihu.com"):
        url = f"https://zhuanlan.zhihu.com/p/{article_id}"

    return {
        "id": article_id,
        "title": title,
        "excerpt": content_preview,
        "url": url,
        "created_time": fmt_time(created),
        "updated_time": fmt_time(updated),
        "stats": {"voteup_count": voteup_count, "comment_count": comment_count, "favlists_count": favlists_count},
        "author": {
            "name": author_name,
            "headline": author.get("headline", ""),
        },
        "comment_permission": item.get("comment_permission", ""),
    }


def scrape_article(article_url: str) -> tuple[dict[str, Any], str]:
    item_data = fetch_article_item(article_url)
    return parse_article_metadata(item_data), converter.convert(item_data["content"])


def upvote_article(article_id: str) -> dict[str, Any]:
    """Upvote (赞同) an article.

    Up and down votes must use the ``/voters/up`` and ``/voters/down``
    sub-paths.  Posting ``{"type": "up"}`` to the bare ``/voters`` endpoint
    answers ``{"success": true}`` but changes nothing: the response still
    reports ``is_up: false`` and ``voting: 0``.  Only ``"neutral"`` goes
    through the bare endpoint.

    This affects the 赞同 counter (``voteup_count`` / ``voting``) only; 喜欢
    is a separate counter — see :func:`like_article`.  Low-weight accounts
    may get ``heavy_up_result: "reviewing"``; the vote still lands, roughly
    one or two seconds later.

    :param article_id: Numeric article ID.
    :returns: The decoded JSON response.
    """
    resp = session.post(ARTICLE_VOTERS_ACTION_URL.format(article_id=article_id, action="up"))
    return resp.json()


def downvote_article(article_id: str) -> dict[str, Any]:
    """Downvote (反对) an article.

    Like :func:`upvote_article`, this uses the dedicated ``/voters/down``
    sub-path because the bare ``/voters`` endpoint ignores an ``"up"`` or
    ``"down"`` type even though it answers with ``success: true``.

    :param article_id: Numeric article ID.
    :returns: The decoded JSON response.
    """
    resp = session.post(ARTICLE_VOTERS_ACTION_URL.format(article_id=article_id, action="down"))
    return resp.json()


def neutral_article(article_id: str) -> dict[str, Any]:
    """Remove your vote (赞同/反对) from an article.

    ``"neutral"`` is the one action the bare ``/voters`` endpoint does honour.

    :param article_id: Numeric article ID.
    :returns: The decoded JSON response.
    """
    resp = session.post(ARTICLE_VOTERS_URL.format(article_id=article_id), json={"type": "neutral"})
    return resp.json()


def like_article(article_id: str) -> dict[str, Any]:
    """Like (喜欢) an article.

    Zhihu keeps 赞同 and 喜欢 as two independent counters: this changes
    ``liked_count`` / ``is_liked`` and leaves ``voteup_count`` / ``voting``
    untouched, and vice versa for :func:`upvote_article`.

    :param article_id: Numeric article ID.
    :returns: The decoded JSON response.
    """
    resp = session.post(ARTICLE_LIKE_URL.format(article_id=article_id))
    return resp.json()


def unlike_article(article_id: str) -> dict[str, Any]:
    """Remove your like (喜欢) from an article.

    The DELETE may answer with an empty body, so the response is decoded
    leniently.

    :param article_id: Numeric article ID.
    :returns: The decoded JSON response, or a success envelope when the body
        is empty.
    """
    resp = session.delete(ARTICLE_LIKE_URL.format(article_id=article_id))
    return safe_json(resp)
