"""Question and answer fetching, page-state parsing, and vote/thank actions."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from zhihu_cli.content.handlers import ANSWER_PATH_RE, canonical_answer_url, fmt_time, safe_json
from zhihu_cli.content.handlers.requests import fetch_json, fetch_page_html, get_page_state, session
from zhihu_cli.content.handlers.waterfall import PageStream, _set_query_param, stream_handler
from zhihu_cli.content.utils.html2markdown import converter

#: Default ``include`` fields for the answer feed.  ``content`` is the answer
#: body; the rest are feed annotations ``stream_answers`` parses.
ANSWER_FEEDS_INCLUDE = (
    "content,big_card_summary,media_detail,reaction_instruction,is_author,is_thanked,"
    "voting,is_favorited,label_info,content_text_length,reactions,"
    "is_collapsed,collapse_reason,collapsed_by"
)

ANSWER_FEEDS_URL = (
    f"https://api.zhihu.com/questions/{{question_id}}/feeds?include={ANSWER_FEEDS_INCLUDE}&order=default&show_detail=1"
)

# Appended to :data:`ANSWER_FEEDS_URL` to switch the feed from the default view
# to the collapsed answers section.  Zhihu's ``paging.next`` URLs carry the
# filter forward, so pagination needs no further handling.
COLLAPSED_FILTER = "&filter=collapsed"

QUESTION_API_URL = (
    "https://api.zhihu.com/questions/{question_id}"
    "?include=detail,answer_count,comment_count,follower_count,author,"
    "voteup_count,voting,can_vote,visit_count,relationship"
)
_QUESTION_PATH_RE = re.compile(r"/(?:question|questions)/(\d+)(?:/answer/\d+)?/?$")


def extract_question_id(question_url: str) -> str:
    """Extract a numeric question ID from a Zhihu question URL or raw ID.

    Both ``/question/<qid>`` and the composite
    ``/question/<qid>/answer/<aid>`` form are accepted.

    :param question_url: a question URL or a bare numeric question ID.
    :returns: the numeric question ID.
    :raises ValueError: if no question ID can be extracted.
    """
    candidate = question_url.strip()
    if candidate.isdigit():
        return candidate

    match = _QUESTION_PATH_RE.search(urlparse(candidate).path)
    if match:
        return match.group(1)

    raise ValueError(f"Could not extract a Zhihu question ID from {question_url!r}")


def is_http_404(exc: BaseException) -> bool:
    """Return whether *exc* represents an HTTP 404 from either session type.

    Direct sessions raise ``curl_cffi``'s ``HTTPError`` while daemon-backed
    sessions raise :class:`~zhihu_cli.daemon.proxy.DaemonHTTPError`; both
    expose the offending response as ``.response``.

    :param exc: the exception to inspect.
    :returns: True when the exception carries a 404 response.
    """
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 404


def parse_question_metadata(item: dict[str, Any]) -> dict[str, Any]:
    """Normalise a question entity to the metadata rendered by ``browse question``.

    :param item: a question dict from the JSON API or from a page's
        ``js-initialData``; the SSR camelCase keys are accepted alongside the
        API's snake_case ones.
    :returns: the display metadata dict.
    """
    author = item.get("author", {})
    question_id = item.get("id", "")
    url = item.get("url", "")
    parsed_url = urlparse(url)
    if question_id and (not url or parsed_url.netloc == "api.zhihu.com" or parsed_url.path.startswith("/api/")):
        url = f"https://www.zhihu.com/question/{question_id}"

    return {
        "id": question_id,
        "title": item.get("title", ""),
        "url": url,
        "created_time": fmt_time(item.get("created", 0)),
        "updated_time": fmt_time(item.get("updated_time", item.get("updatedTime", 0))),
        "answer_count": item.get("answer_count", item.get("answerCount", 0)),
        "comment_count": item.get("comment_count", item.get("commentCount", 0)),
        "visit_count": item.get("visit_count", item.get("visitCount", 0)),
        "follower_count": item.get("follower_count", item.get("followerCount", 0)),
        "voteup_count": item.get("voteup_count", item.get("voteupCount", 0)),
        "author": {
            "name": author.get("name", "anonymous"),
            "headline": author.get("headline", ""),
        },
    }


def fetch_question_item(question_url: str) -> dict[str, Any]:
    """Fetch raw question data from the JSON API.

    :param question_url: a question URL (including the composite
        ``/question/<qid>/answer/<aid>`` form) or a bare question ID.
    :returns: the question object returned by the API.
    :raises HTTPError: if the question resource is unavailable.  Deleted or
        hidden questions return 404 with ``{"error": {"code": 4041, ...}}``.
    """
    question_id = extract_question_id(question_url)
    api_url = QUESTION_API_URL.format(question_id=question_id)

    item = fetch_json(api_url)
    if not isinstance(item, dict):
        raise ValueError("Question API returned a non-object response")
    if "error" in item:
        raise ValueError(f"Question API returned an error: {item.get('error')}")
    if "title" not in item:
        raise ValueError("Question API response does not contain question data")
    return item


def scrape_question_data(question_url: str) -> tuple[dict[str, Any], str]:
    """Fetch a question's metadata and detail (as Markdown)."""
    item_data = fetch_question_item(question_url)
    return parse_question_metadata(item_data), converter.convert(item_data.get("detail") or "")


def stream_answers(
    question_data: dict[str, Any],
    raw: bool = False,
    limit: int = 5,
    max_items: int | None = None,
    collapsed: bool = False,
    *,
    order: str = "default",
    include: str | None = None,
    include_question: bool = False,
    max_pages: int | None = None,
    delay: float = 1.0,
) -> PageStream:
    """Stream answers for a question, one dict per answer.

    :param question_data: dict with at least an ``"id"`` key (the question ID).
    :param raw: when True, return raw HTML in the ``"content"`` field
        instead of converting to Markdown.  Also applies to
        ``question_description`` when ``include_question`` is set.
    :param limit: number of answers requested per API page.
    :param max_items: optional cap on the total number of answers yielded
        (stops pagination early when reached).
    :param collapsed: when True, fetch the collapsed (折叠) answers section
        instead of the default view.  Each yielded dict then also carries
        ``"is_collapsed"``, ``"collapse_reason"`` and ``"collapsed_by"``.
    :param order: feed order passed to the API.  ``"default"`` is Zhihu's
        ranking; ``"created"`` is chronological (needed for new-answer
        discovery).  Any non-empty value is forwarded, so new server-side
        orders can be used without a library change.
    :param include: advance escape hatch that replaces the whole ``include=``
        list.  The parser still only extracts the fields it knows, and must
        keep ``content`` for the body to be present.
    :param include_question: append ``question.detail`` to the request and add
        the question metadata a corpus builder needs to every answer dict:
        ``question_id``, ``question_title``, ``question_created``,
        ``question_description``, ``author_token``, ``thanks`` and
        ``truncated``.
    :param max_pages: optional cap on the number of API pages fetched.
    :param delay: inter-page delay passed to
        :func:`~zhihu_cli.content.handlers.waterfall.stream_handler`.  It is
        zeroed while a profile is logged in; use
        :func:`zhihu_cli.set_rate_limit` for real pacing.
    :raises ValueError: when ``order`` is empty.
    """
    if not order:
        raise ValueError("order must be a non-empty string")

    include_fields = ANSWER_FEEDS_INCLUDE if include is None else include
    if include_question:
        fields = [field.strip() for field in include_fields.split(",") if field.strip()]
        if "question.detail" not in fields:
            fields.append("question.detail")
        include_fields = ",".join(fields)

    url = ANSWER_FEEDS_URL.replace("{question_id}", str(question_data["id"]))
    if include_fields != ANSWER_FEEDS_INCLUDE:
        url = _set_query_param(url, "include", include_fields)
    if order != "default":
        url = _set_query_param(url, "order", order)
    if collapsed:
        url += COLLAPSED_FILTER

    question_id = str(question_data["id"])

    def parse_ans(data: dict[str, Any]) -> Iterable[dict[str, Any]]:
        for item in data.get("data", []):
            # /feeds endpoint wraps answers in a "target" field
            ans = item.get("target", item)
            if not isinstance(ans, dict) or not ans.get("id"):
                continue
            content = ans.get("content", "")
            if not raw:
                content = converter.convert(content)
            answer: dict[str, Any] = {
                "author": ans.get("author", {}).get("name", "anonymous"),
                "id": str(ans.get("id", ans.get("url", "/unknown").split("/")[-1])),
                "vote": ans.get("voteup_count", 0),
                "comment": ans.get("comment_count", 0),
                "favorite": ans.get("favlists_count", 0),
                "created_time": int(ans.get("created_time", 0) or 0),
                "updated_time": int(ans.get("updated_time", 0) or 0),
                "content": content,
                "is_collapsed": bool(ans.get("is_collapsed", False)),
                "collapse_reason": ans.get("collapse_reason") or "",
                "collapsed_by": ans.get("collapsed_by") or "",
            }
            if include_question:
                question = ans.get("question") or {}
                detail = question.get("detail")
                if isinstance(detail, str) and not raw:
                    detail = converter.convert(detail)
                answer.update(
                    {
                        "question_id": str(question.get("id") or question_id),
                        "question_title": question.get("title") or "",
                        "question_created": int(question.get("created", 0) or 0),
                        "question_description": detail,
                        "author_token": (ans.get("author") or {}).get("url_token", ""),
                        "thanks": ans.get("thanks_count", 0),
                        "truncated": bool(ans.get("content_need_truncated", False)),
                    }
                )
            yield answer

    return stream_handler(url, parse_ans, limit=limit, max_items=max_items, max_pages=max_pages, delay=delay)


def scrape_answers(
    question_data: dict[str, Any],
    raw: bool = False,
    limit: int = 5,
    max_items: int | None = None,
    collapsed: bool = False,
    *,
    order: str = "default",
    include: str | None = None,
    include_question: bool = False,
    max_pages: int | None = None,
    delay: float = 1.0,
) -> Iterable[dict[str, Any]]:
    """Yield answers for a question, one dict per answer.

    Backwards-compatible alias for :func:`stream_answers`; see that function
    for the full parameter documentation.
    """
    return stream_answers(
        question_data,
        raw=raw,
        limit=limit,
        max_items=max_items,
        collapsed=collapsed,
        order=order,
        include=include,
        include_question=include_question,
        max_pages=max_pages,
        delay=delay,
    )


def upvote_answer(answer_id: str) -> dict[str, Any]:
    resp = session.post(f"https://www.zhihu.com/api/v4/answers/{answer_id}/voters", json={"type": "up"})
    return resp.json()


def neutral_answer(answer_id: str) -> dict[str, Any]:
    resp = session.post(f"https://www.zhihu.com/api/v4/answers/{answer_id}/voters", json={"type": "neutral"})
    return resp.json()


def downvote_answer(answer_id: str) -> dict[str, Any]:
    resp = session.post(f"https://www.zhihu.com/api/v4/answers/{answer_id}/voters", json={"type": "down"})
    return resp.json()


def thank_answer(answer_id: str) -> dict[str, Any]:
    resp = session.post(f"https://www.zhihu.com/api/v4/answers/{answer_id}/thankers")
    return resp.json()


def unthank_answer(answer_id: str) -> dict[str, Any]:
    resp = session.delete(f"https://www.zhihu.com/api/v4/answers/{answer_id}/thankers")
    return safe_json(resp)


def upvote_question(question_id: str) -> dict[str, Any]:
    resp = session.post(f"https://www.zhihu.com/api/v4/questions/{question_id}/voters/up")
    return resp.json()


def downvote_question(question_id: str) -> dict[str, Any]:  # undocumented endpoint!
    resp = session.post(f"https://www.zhihu.com/api/v4/questions/{question_id}/voters/down")
    return resp.json()


def follow_question(question_id: str) -> dict[str, Any]:
    resp = session.post(f"https://www.zhihu.com/api/v4/questions/{question_id}/followers")
    return resp.json()


def unfollow_question(question_id: str) -> dict[str, Any]:
    resp = session.delete(f"https://www.zhihu.com/api/v4/questions/{question_id}/followers")
    return safe_json(resp)


def extract_answer_id(answer_url: str) -> str:
    """Extract a numeric answer ID from a Zhihu answer URL or bare ID.

    :param answer_url: A bare answer ID, ``/answer/<id>`` URL, or
        ``/question/<qid>/answer/<aid>`` URL.
    :returns: The numeric answer ID.
    :raises ValueError: If no answer ID can be extracted.
    """
    candidate = answer_url.strip()
    if candidate.isdigit():
        return candidate

    match = ANSWER_PATH_RE.search(urlparse(candidate).path)
    if match:
        return match.group(1)

    raise ValueError(f"Could not extract a Zhihu answer ID from {answer_url!r}")


def _canonical_answer_url(answer_url: str) -> str:
    """Normalise an answer reference to ``https://www.zhihu.com/answer/<id>``.

    Accepts a bare numeric answer ID, a short answer URL, or a
    ``/question/<qid>/answer/<aid>`` URL.  Anything else is returned unchanged
    so the caller sees Zhihu's own error rather than a parse failure.
    """
    return canonical_answer_url(answer_url)


def _fetch_answer_entities(answer_url: str) -> dict[str, Any]:
    """Fetch an answer page and return its ``js-initialData`` entity map.

    :param answer_url: an answer URL, bare answer ID, or composite
        ``/question/<qid>/answer/<aid>`` URL.
    :returns: the page's ``entities`` dict.
    :raises ValueError: if the page carries no answer data.
    """
    canonical = _canonical_answer_url(answer_url)
    entities = get_page_state(fetch_page_html(canonical))
    if not entities.get("answers"):
        raise ValueError(f"No answer data found in {canonical}")
    return entities


def _answer_author_name(answer_data: dict[str, Any], entities: dict[str, Any]) -> str:
    """Resolve an answer entity's author name.

    :param answer_data: the answer entity from ``js-initialData``.
    :param entities: the full entity map, used to resolve a string author ref.
    :returns: the author's display name, or ``"unknown"``.
    """
    author_ref = answer_data.get("author", "")
    if isinstance(author_ref, str):
        return entities.get("users", {}).get(author_ref, {}).get("name", "unknown")
    if isinstance(author_ref, dict):
        return author_ref.get("name", "unknown")
    return "unknown"


def _epoch_seconds(value: Any) -> int:
    """Normalise a timestamp (seconds or milliseconds) to integer seconds."""
    try:
        timestamp = float(value or 0)
    except (TypeError, ValueError):
        return 0
    if timestamp > 1e12:
        timestamp /= 1000
    return int(timestamp)


def scrape_answer_page(answer_url: str, raw: bool = False) -> tuple[dict[str, Any], str]:
    """Scrape full content from a single answer page.

    :param answer_url: full URL to a Zhihu answer page, or a bare answer ID.
    :param raw: when True, return raw HTML instead of converting to Markdown.
    :returns: (metadata, content) — content is Markdown by default, raw HTML if ``raw=True``.
        The metadata also carries ``question_detail``, the question body converted to Markdown.
    """
    entities = _fetch_answer_entities(answer_url)
    answer_data = next(iter(entities["answers"].values()))

    questions = entities.get("questions", {})
    question_key = next(iter(questions), "")
    question_data = questions.get(question_key, {})
    question_title = question_data.get("title", "untitled")
    question_id = str(question_data.get("id") or question_key or "")

    author_name = _answer_author_name(answer_data, entities)

    # Resolve created/updated timestamps from entity timestamps
    created_ts = answer_data.get("created_time") or answer_data.get("created") or answer_data.get("createdTime")
    if not created_ts:
        created_ts = question_data.get("created")
    updated_ts = answer_data.get("updated_time") or answer_data.get("updatedTime") or answer_data.get("updated")
    if isinstance(created_ts, (int, float)) and created_ts > 1e12:
        created_ts = created_ts / 1000
    if isinstance(updated_ts, (int, float)) and updated_ts > 1e12:
        updated_ts = updated_ts / 1000
    created_date = "unknown"
    if created_ts:
        try:
            created_date = datetime.fromtimestamp(created_ts).strftime("%Y-%m-%d")
        except (ValueError, OSError):
            pass

    metadata = {
        "id": str(answer_data.get("id", "")),
        "question_id": question_id,
        "title": question_title,
        "author": author_name,
        "vote": answer_data.get("voteupCount", 0),
        "comment": answer_data.get("commentCount", 0),
        "favorite": answer_data.get("favlistsCount", 0),
        "created": created_date,
        "created_time": fmt_time(created_ts),
        "updated_time": fmt_time(updated_ts),
        "question_detail": converter.convert(question_data.get("detail", "")),
    }

    content_html = answer_data.get("content", "")
    content = content_html if raw else converter.convert(content_html)

    return metadata, content


def scrape_question_from_answer_page(answer_url: str, raw: bool = False) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Recover a question snapshot from the answer page that links to it.

    Deleted or hidden questions answer 404 from the question API even though
    their answers stay reachable.  The answer page still embeds the question
    entity in ``js-initialData``, so it can supply the metadata and detail
    body — but not the full answer list.

    :param answer_url: an answer URL, bare answer ID, or the composite
        ``/question/<qid>/answer/<aid>`` URL supplied to ``browse question``.
    :param raw: when True, leave the answer body as raw HTML instead of
        converting it to Markdown.
    :returns: ``(question_metadata, detail_markdown, answer_item)``, where
        *answer_item* matches the dicts yielded by :func:`scrape_answers`.
    :raises ValueError: if the answer page has no question or answer entity.
    """
    entities = _fetch_answer_entities(answer_url)
    questions = entities.get("questions", {})
    if not questions:
        raise ValueError(f"No question data found in {answer_url}")
    question_data = next(iter(questions.values()))
    answer_data = next(iter(entities["answers"].values()))

    content_html = answer_data.get("content", "")
    content = content_html if raw else converter.convert(content_html)

    created_ts = _epoch_seconds(
        answer_data.get("created_time") or answer_data.get("createdTime") or answer_data.get("created")
    )
    updated_ts = _epoch_seconds(
        answer_data.get("updated_time") or answer_data.get("updatedTime") or answer_data.get("updated")
    )

    answer_item = {
        "author": _answer_author_name(answer_data, entities),
        "id": str(answer_data.get("id", "")),
        "vote": answer_data.get("voteupCount", 0),
        "comment": answer_data.get("commentCount", 0),
        "favorite": answer_data.get("favlistsCount", 0),
        "created_time": created_ts,
        "updated_time": updated_ts,
        "content": content,
        "is_collapsed": bool(answer_data.get("isCollapsed", False)),
        "collapse_reason": answer_data.get("collapseReason") or "",
        "collapsed_by": answer_data.get("collapsedBy") or "",
    }

    return (
        parse_question_metadata(question_data),
        converter.convert(question_data.get("detail") or ""),
        answer_item,
    )
