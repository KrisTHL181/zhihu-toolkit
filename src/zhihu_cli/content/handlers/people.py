from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator
from functools import partial
from typing import Any

from zhihu_cli.content.handlers import MEMBER_API, fmt_time, safe_json
from zhihu_cli.content.handlers.requests import session
from zhihu_cli.content.handlers.waterfall import PAGE_SIZE, stream_handler
from zhihu_cli.content.utils.html2markdown import converter
from zhihu_cli.content.utils.text_clean import strip_html_tags


def follow(user_id: str) -> dict[str, Any]:
    resp = session.post(f"https://www.zhihu.com/api/v4/members/{user_id}/followers")

    data = safe_json(resp)
    if resp.status_code == 403 and isinstance(data, dict) and "error" in data:
        raise PermissionError(f"Failed to follow: {data['error']['message']}")

    return data


def unfollow(user_id: str) -> dict[str, Any]:
    resp = session.delete(f"https://www.zhihu.com/api/v4/members/{user_id}/followers")
    return safe_json(resp)


def block(user_id: str) -> None:
    session.post(f"https://www.zhihu.com/api/v4/members/{user_id}/actions/block")


def unblock(user_id: str) -> None:
    session.delete(f"https://www.zhihu.com/api/v4/members/{user_id}/actions/block")


# ── member profile ──────────────────────────────────────────────────────────

# ``include`` fields requested from the member profile API. ``gender``, the
# counts and the relational flags are only returned when explicitly listed.
_MEMBER_PROFILE_INCLUDE = (
    "gender,headline,description,educations,business,locations,employments,"
    "is_following,is_followed,follower_count,following_count,answer_count,"
    "articles_count,pins_count,question_count,voteup_count,thanked_count,"
    "cover_url,ip_info"
)


def fetch_member_profile(url_token: str) -> dict[str, Any] | None:
    """Fetch a member's public profile info (via the v4 members API).

    :param url_token: The user's url_token (e.g. ``"zhangsan"``).
    :returns: A flat dict of profile fields, or ``None`` on failure.
    """
    url = f"https://www.zhihu.com/api/v4/members/{url_token}?include={_MEMBER_PROFILE_INCLUDE}"
    try:
        resp = session.get(url)
        resp.raise_for_status()
        user = resp.json()
    except Exception:
        return None

    def _topic_names(items: Any) -> list[str]:
        """Collapse a list of {school|company: {name, ...}} into plain names."""
        if not isinstance(items, list):
            return []
        return [(t.get("school") or t.get("company") or {}).get("name", "") for t in items if isinstance(t, dict)]

    return {
        "id": user.get("id", ""),
        "name": user.get("name", ""),
        "url_token": user.get("url_token", url_token),
        "headline": user.get("headline", ""),
        "avatar_url": user.get("avatar_url", ""),
        "gender": user.get("gender", -1),
        "follower_count": user.get("follower_count", 0),
        "following_count": user.get("following_count", 0),
        "answer_count": user.get("answer_count", 0),
        "articles_count": user.get("articles_count", 0),
        "pins_count": user.get("pins_count", 0),
        "question_count": user.get("question_count", 0),
        "voteup_count": user.get("voteup_count", 0),
        "thanked_count": user.get("thanked_count", 0),
        "description": user.get("description", ""),
        # extra fields only available from the API
        "is_following": user.get("is_following", False),
        "is_followed": user.get("is_followed", False),
        "cover_url": user.get("cover_url", ""),
        "ip_info": user.get("ip_info", ""),
        "business": (user.get("business") or {}).get("name", ""),
        "educations": _topic_names(user.get("educations")),
        "employments": _topic_names(user.get("employments")),
        "locations": _topic_names(user.get("locations")),
    }


def fetch_member_error_message(url_token: str) -> str | None:
    """Return the platform's error message when a member profile is unavailable.

    Restricted or banned accounts (``UserRegulateError``) return HTTP 410 with
    an ``error.message`` body, e.g. ``由于严重违反知乎社区管理规定，该账号已被「停用」。``.
    Other failures (e.g. HTTP 404) may carry an ``error.message`` as well.
    Returns the message, or ``None`` when the profile is fetchable normally.
    """
    url = f"https://www.zhihu.com/api/v4/members/{url_token}"
    try:
        resp = session.get(url)
        if resp.status_code < 400:
            return None
        return resp.json().get("error", {}).get("message") or None
    except Exception:
        return None


# ── incremental listing helper ──────────────────────────────────────────────


def _until_since(items: Iterable[dict[str, Any]], since: str | None) -> Iterable[dict[str, Any]]:
    """Yield items until one predates *since*, relying on descending order.

    Member listings use ``sort_by=created`` and are newest-first, so the walk
    can stop at the first item older than *since* instead of paginating to the
    end.  The check is intentionally one-way: ascending input simply yields
    nothing once an old item is seen, because order is the caller's contract.

    Items whose ``created_time`` is not a date (e.g. ``"unknown time"``) are
    yielded rather than treated as the stop condition.

    :param items: Parsed listing items, newest first.
    :param since: Inclusive lower bound (``YYYY-MM-DD`` or
        ``YYYY-MM-DD HH:MM:SS``), or ``None`` to yield everything.
    :returns: A generator of the items at or after *since*.
    """
    if not since:
        yield from items
        return

    try:
        for item in items:
            created: Any = item.get("created_time") or ""
            # Listing items normally carry a formatted string, but the
            # full-content shape returned by ``stream_member_answers`` uses
            # integer timestamps (matching ``question.scrape_answers``).
            # Normalise both before the date comparison so ``since=`` keeps
            # working in either mode.
            if isinstance(created, (int, float)):
                created = fmt_time(created)
            created = str(created)
            if created[:1].isdigit() and created[:10] < since[:10]:
                return
            yield item
    finally:
        # Deterministic cleanup: a caller that breaks out of the wrapper (or
        # hits the date cutoff) must not leave the page generator suspended.
        # ``PageStream.close()`` is idempotent and lists have no ``close``.
        close = getattr(items, "close", None)
        if callable(close):
            close()


# ── answer list ─────────────────────────────────────────────────────────────


#: ``include`` fields for the lightweight answer listing (no body).
_MEMBER_ANSWER_INCLUDE = (
    "data%5B%2A%5D.excerpt%2Cdata%5B%2A%5D.voteup_count%2Cdata%5B%2A%5D.comment_count%2Cdata%5B%2A%5D.is_copyable"
)

#: ``include`` fields for ``include_content=True``.  The member-answers
#: endpoint returns the full HTML body only when ``content`` is requested;
#: ``favlists_count``/``thanks_count`` and the nested ``question`` object are
#: not part of the lightweight listing but are needed to build a corpus.
_MEMBER_ANSWER_INCLUDE_FULL = (
    "data%5B%2A%5D.content%2Cdata%5B%2A%5D.voteup_count"
    "%2Cdata%5B%2A%5D.comment_count%2Cdata%5B%2A%5D.is_copyable"
    "%2Cdata%5B%2A%5D.favlists_count%2Cdata%5B%2A%5D.thanks_count"
    "%2Cdata%5B%2A%5D.content_need_truncated%2Cdata%5B%2A%5D.author"
    "%2Cdata%5B%2A%5D.question"
)

#: ``sort_by`` values the member-answers endpoint documents.  ``voteups`` is
#: the one a corpus builder wants (high-vote answers first); ``created`` is the
#: only order for which ``since=`` can stop pagination early.
_MEMBER_ANSWER_SORT_BY = frozenset(("created", "voteups"))


def _parse_member_answer(
    item: dict[str, Any],
    *,
    include_content: bool = False,
    raw: bool = False,
) -> dict[str, Any]:
    """Normalise one member-answer listing item.

    The default shape is the lightweight listing (excerpt + formatted times)
    that existing callers consume.  ``include_content=True`` returns the full
    body instead, in the same shape as
    :func:`zhihu_cli.content.handlers.question.scrape_answers`: integer
    timestamps, ``vote``/``comment``/``favorite``/``thanks`` counts and the
    author.  ``raw`` only matters in that mode and leaves the HTML unconverted.
    """
    question = item.get("question") or {}
    answer_id = item.get("id", "")
    question_id = question.get("id", "")
    url = f"https://www.zhihu.com/question/{question_id}/answer/{answer_id}"

    if not include_content:
        return {
            "type": "answer",
            "id": answer_id,
            "question_id": question_id,
            "title": question.get("title", ""),
            "excerpt": item.get("excerpt", ""),
            "url": url,
            "created_time": fmt_time(item.get("created_time")),
            "updated_time": fmt_time(item.get("updated_time")),
            "voteup_count": item.get("voteup_count", 0),
            "comment_count": item.get("comment_count", 0),
            "is_copyable": item.get("is_copyable", True),
        }

    author = item.get("author") or {}
    content = item.get("content") or ""
    if not raw:
        content = converter.convert(content)
    return {
        "type": "answer",
        "id": str(answer_id),
        "question_id": str(question_id),
        "title": question.get("title", ""),
        "url": url,
        "content": content,
        "author": author.get("name", "anonymous"),
        "author_token": author.get("url_token", ""),
        "vote": item.get("voteup_count", 0),
        "comment": item.get("comment_count", 0),
        "favorite": item.get("favlists_count", 0),
        "thanks": item.get("thanks_count", 0),
        "created_time": int(item.get("created_time", 0) or 0),
        "updated_time": int(item.get("updated_time", 0) or 0),
        "question_created": int(question.get("created", 0) or 0),
        "truncated": bool(item.get("content_need_truncated", False)),
        "is_copyable": item.get("is_copyable", True),
    }


def _parse_answer_list(data: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for item in data.get("data", []):
        yield _parse_member_answer(item)


def _parse_full_answer_list(data: dict[str, Any], *, raw: bool = False) -> Iterable[dict[str, Any]]:
    for item in data.get("data", []):
        yield _parse_member_answer(item, include_content=True, raw=raw)


def stream_member_answers(
    url_token: str,
    limit: int = PAGE_SIZE,
    max_items: int | None = None,
    since: str | None = None,
    *,
    include_content: bool = False,
    sort_by: str = "created",
    raw: bool = False,
) -> Iterator[dict[str, Any]]:
    """Stream a member's answers, one dict per answer.

    The return value is the :class:`~zhihu_cli.content.handlers.waterfall.PageStream`
    itself when ``since`` is unset, so ``close()``/``totals``/``stalled`` stay
    reachable.  With ``since`` the result is an early-stopping wrapper iterator
    and those pagination attributes are not exposed.

    :param url_token: The member's url_token.
    :param limit: Page size for each API request.
    :param max_items: Maximum total items to return.
    :param since: Stop paginating at the first answer created before this
        ``YYYY-MM-DD`` / ``YYYY-MM-DD HH:MM:SS`` value.  Only valid with
        ``sort_by="created"`` because it relies on the newest-first order.
    :param include_content: Request the full HTML body and return the
        full-content shape (``content``, integer timestamps, vote/comment/
        favorite/thanks counts, author token, question creation time).
    :param sort_by: ``"created"`` (default) or ``"voteups"``.  Corpus builders
        usually want ``"voteups"`` so a per-author cap keeps the best answers
        rather than the newest.
    :param raw: Leave ``content`` as HTML instead of converting it to
        Markdown.  Ignored unless ``include_content`` is true.
    :raises ValueError: for an unknown ``sort_by``, or for ``since`` combined
        with a non-``created`` order.
    """
    if sort_by not in _MEMBER_ANSWER_SORT_BY:
        allowed = ", ".join(sorted(_MEMBER_ANSWER_SORT_BY))
        raise ValueError(f"sort_by must be one of {allowed}; got {sort_by!r}")
    if since and sort_by != "created":
        raise ValueError("since= relies on newest-first pagination; use sort_by='created' with it")

    include = _MEMBER_ANSWER_INCLUDE_FULL if include_content else _MEMBER_ANSWER_INCLUDE
    url = f"{MEMBER_API.format(token=url_token)}/answers?include={include}&offset=0&limit={limit}&sort_by={sort_by}"
    parser = partial(_parse_full_answer_list, raw=raw) if include_content else _parse_answer_list

    stream = stream_handler(url, parser, max_items=max_items)
    return _until_since(stream, since) if since else stream


def fetch_member_answers(
    url_token: str,
    limit: int = PAGE_SIZE,
    max_items: int | None = None,
    since: str | None = None,
    *,
    include_content: bool = False,
    sort_by: str = "created",
    raw: bool = False,
) -> list[dict[str, Any]]:
    """Fetch a member's answers list.

    This is the eager form of :func:`stream_member_answers`; see that function
    for the meaning of ``include_content``, ``sort_by`` and ``raw``.

    :param url_token: The member's url_token.
    :param limit: Page size for each API request.
    :param max_items: Maximum total items to return.
    :param since: Stop paginating at the first answer created before this
        ``YYYY-MM-DD`` / ``YYYY-MM-DD HH:MM:SS`` value.  Relies on the
        newest-first order of ``sort_by=created``.
    """
    return list(
        stream_member_answers(
            url_token,
            limit=limit,
            max_items=max_items,
            since=since,
            include_content=include_content,
            sort_by=sort_by,
            raw=raw,
        )
    )


# ── article list ────────────────────────────────────────────────────────────


def _parse_member_article(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "article",
        "id": item.get("id", ""),
        "title": item.get("title", ""),
        "excerpt": item.get("excerpt", ""),
        "url": (item.get("url") or f"https://zhuanlan.zhihu.com/p/{item.get('id', '')}").replace("http://", "https://"),
        "created_time": fmt_time(item.get("created")),
        "updated_time": fmt_time(item.get("updated")),
        "voteup_count": item.get("voteup_count", 0),
        "comment_count": item.get("comment_count", 0),
    }


def _parse_article_list(data: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for item in data.get("data", []):
        yield _parse_member_article(item)


def fetch_member_articles(
    url_token: str,
    limit: int = PAGE_SIZE,
    max_items: int | None = None,
    since: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch a member's articles list.

    The articles endpoint answers HTTP 500 when the ``x-app-za`` /
    ``x-app-version`` app-identity headers are present, so they are omitted
    here.

    :param url_token: The member's url_token.
    :param limit: Page size for each API request.
    :param max_items: Maximum total items to return.
    :param since: Stop paginating at the first article created before this
        ``YYYY-MM-DD`` / ``YYYY-MM-DD HH:MM:SS`` value.  Relies on the
        newest-first order of ``sort_by=created``.
    """
    url = (
        f"{MEMBER_API.format(token=url_token)}/articles"
        f"?include=data%5B%2A%5D.excerpt%2Cdata%5B%2A%5D.voteup_count"
        f"%2Cdata%5B%2A%5D.comment_count"
        f"&offset=0&limit={limit}&sort_by=created"
    )
    stream = stream_handler(url, _parse_article_list, max_items=max_items, skip_app_headers=True)
    return list(_until_since(stream, since))


# ── pin list ────────────────────────────────────────────────────────────────


def _parse_member_pin(item: dict[str, Any]) -> dict[str, Any]:
    content = item.get("content", "")
    if isinstance(content, list):
        content = "\n".join(block.get("content", "") if isinstance(block, dict) else str(block) for block in content)

    clean_title = strip_html_tags(item.get("excerpt", "") or content or "").strip()[:100]

    return {
        "type": "pin",
        "id": item.get("id", ""),
        "title": clean_title,
        "excerpt": item.get("excerpt", ""),
        "content_text": content,
        "url": f"https://www.zhihu.com/pin/{item.get('id', '')}",
        "created_time": fmt_time(item.get("created")),
        # The pin payload carries a real ``updated`` timestamp; exposing it is
        # what lets incremental downloads notice an edited pin.
        "updated_time": fmt_time(item.get("updated")),
        "voteup_count": item.get("voteup_count", 0),
        "comment_count": item.get("comment_count", 0),
    }


def _parse_pin_list(data: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for item in data.get("data", []):
        yield _parse_member_pin(item)


def fetch_member_pins(
    url_token: str,
    limit: int = PAGE_SIZE,
    max_items: int | None = None,
    since: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch a member's pins (想法) list.

    :param url_token: The member's url_token.
    :param limit: Page size for each API request.
    :param max_items: Maximum total items to return.
    :param since: Stop paginating at the first pin created before this
        ``YYYY-MM-DD`` / ``YYYY-MM-DD HH:MM:SS`` value.  Relies on the
        newest-first order of ``sort_by=created``.
    """
    url = (
        f"{MEMBER_API.format(token=url_token)}/pins"
        f"?include=data%5B%2A%5D.excerpt%2Cdata%5B%2A%5D.content"
        f"%2Cdata%5B%2A%5D.voteup_count%2Cdata%5B%2A%5D.comment_count"
        f"&offset=0&limit={limit}&sort_by=created"
    )
    return list(_until_since(stream_handler(url, _parse_pin_list, max_items=max_items), since))


# ── question list ───────────────────────────────────────────────────────────


def _parse_member_question(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "question",
        "id": item.get("id", ""),
        "title": item.get("title", ""),
        "excerpt": item.get("excerpt", ""),
        "url": f"https://www.zhihu.com/question/{item.get('id', '')}",
        "created_time": fmt_time(item.get("created")),
        "answer_count": item.get("answer_count", 0),
        "follower_count": item.get("follower_count", 0),
        "comment_count": item.get("comment_count", 0),
    }


def _parse_question_list(data: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for item in data.get("data", []):
        yield _parse_member_question(item)


def fetch_member_questions(
    url_token: str,
    limit: int = PAGE_SIZE,
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch questions a member has asked."""
    url = f"{MEMBER_API.format(token=url_token)}/questions?include=data%5B%2A%5D.excerpt&offset=0&limit={limit}"
    return list(stream_handler(url, _parse_question_list, max_items=max_items))


# ── activity feed ──────────────────────────────────────────────────────────

# Verbs that represent content creation (not votes, follows, etc.)
_CREATION_VERBS = frozenset(
    {
        "MEMBER_CREATE_PIN",
        "MEMBER_ANSWER_QUESTION",
        "MEMBER_CREATE_ARTICLE",
        "MEMBER_ASK_QUESTION",
        "MEMBER_CREATE_QUESTION",
    }
)


def _parse_activity_item(raw: dict[str, Any]) -> dict[str, Any]:
    """Parse a raw v3 moments/activities item into a structured dict.

    The v3 API returns items with ``verb``, ``action_text``, ``target``, ``actor``,
    and ``created_time``.  This normalizes them into a flat dict suitable for both
    display and JSON output.
    """
    verb = raw.get("verb", "")
    target = raw.get("target", {})
    actor = raw.get("actor", {})

    # Determine target type from verb or target structure
    target_type = ""
    if target.get("type") == "pin":
        target_type = "pin"
    elif target.get("type") == "answer":
        target_type = "answer"
    elif target.get("type") == "article":
        target_type = "article"
    elif target.get("type") == "question":
        target_type = "question"

    # Extract title (strip HTML tags for clean display)
    title = target.get("title", "") or target.get("excerpt_title", "")
    if not title and target.get("question"):
        title = target["question"].get("title", "")
    if isinstance(title, str):
        title = strip_html_tags(title).strip()

    # Build URL (strip query params like ?native=0)
    url = target.get("url", "")
    if url and url.startswith("https://api.zhihu.com/"):
        # Convert API URLs to web URLs
        url = url.replace("https://api.zhihu.com/answers/", "https://www.zhihu.com/question/")
        if target_type == "answer" and (target.get("question") or {}).get("id"):
            url = f"https://www.zhihu.com/question/{target['question']['id']}/answer/{target.get('id', '')}"
        elif target_type == "article":
            url = f"https://zhuanlan.zhihu.com/p/{target.get('id', '')}"
        elif target_type == "pin":
            url = f"https://www.zhihu.com/pin/{target.get('id', '')}"
        elif target_type == "question":
            url = f"https://www.zhihu.com/question/{target.get('id', '')}"
    # Strip query params from web URLs
    if url and "?" in url:
        url = url.split("?")[0]

    # Excerpt
    excerpt = target.get("excerpt", "") or target.get("excerpt_new", "")
    if isinstance(excerpt, str):
        excerpt = strip_html_tags(excerpt).strip()[:200]

    return {
        "id": raw.get("id", ""),
        "verb": verb,
        "action_text": raw.get("action_text", ""),
        "is_creation": verb in _CREATION_VERBS,
        "is_sticky": raw.get("is_sticky", False),
        "target_type": target_type,
        "target_id": target.get("id", ""),
        "title": title,
        "url": url,
        "excerpt": excerpt,
        "voteup_count": target.get("voteup_count", 0),
        "comment_count": target.get("comment_count", 0),
        "favorite_count": target.get("favlists_count", target.get("favorite_count", 0)),
        "created_time": fmt_time(raw.get("created_time")),
        "created_ts": raw.get("created_time", 0),
        # Actor info
        "actor_name": actor.get("name", ""),
        "actor_url_token": actor.get("url_token", ""),
    }


def _parse_activity_list(data: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for item in data.get("data", []):
        yield _parse_activity_item(item)


def fetch_member_activities(
    url_token: str,
    limit: int = PAGE_SIZE,
    max_items: int | None = None,
    creations_only: bool = False,
) -> list[dict[str, Any]]:
    """Fetch a member's activity feed from the v3 moments API.

    :param url_token: The user's url_token.
    :param limit: Page size for each API request.
    :param max_items: Maximum total items to return (None = follow pagination).
    :param creations_only: If True, only return content-creation activities
        (answers, pins, articles, questions), filtering out votes, follows, etc.
    """
    url = f"https://www.zhihu.com/api/v3/moments/{url_token}/activities?limit={limit}&desktop=True"
    if creations_only:
        # ``max_items`` caps the *filtered* count here, so it cannot be pushed
        # into ``stream_handler`` (which would count pre-filter items).  Filter
        # first, then cap the filtered stream.
        filtered = (item for item in stream_handler(url, _parse_activity_list) if item["is_creation"])
        if max_items is not None:
            filtered = itertools.islice(filtered, max_items)
        return list(filtered)
    return list(stream_handler(url, _parse_activity_list, max_items=max_items))


def get_my_url_token():
    resp = session.get("https://www.zhihu.com/api/v4/me")
    if resp.status_code == 200:
        me = resp.json()
        return me.get("url_token") or me.get("urlToken")
    return None
