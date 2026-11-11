"""Summary interface: fetch engagement stats (voteup, favorite, comment, like,
share) for any Zhihu article / answer / pin URL."""

from __future__ import annotations

from typing import Any

from zhihu_cli.content.handlers import get_type_and_id
from zhihu_cli.content.handlers.requests import fetch_page_html, get_page_state, session

CREATOR_CONTENT_AGGR_URL = "https://www.zhihu.com/api/v4/creators/analysis/realtime/content/aggr"
CREATOR_CONTENT_DAILY_URL = "https://www.zhihu.com/api/v4/creators/analysis/realtime/content/daily"

# Map entity keys (plural) → creator API type param (singular)
_ENTITY_TO_CREATOR_TYPE = {"articles": "article", "answers": "answer", "pins": "pin"}


def _get_field(item: dict[str, Any], *keys: str) -> int | None:
    """Return the first matching key's value as int, or None if none match."""
    for k in keys:
        if k in item:
            val = item[k]
            return int(val) if val else 0
    return None


def _extract_stats(item: dict[str, Any]) -> dict[str, Any]:
    """Extract engagement metrics from a raw entity / API item dict."""
    return {
        "voteup_count": _get_field(item, "voteup_count", "voteupCount") or 0,
        "favlists_count": _get_field(item, "favlists_count", "favlistsCount") or 0,
        "comment_count": _get_field(item, "comment_count", "commentCount") or 0,
        "thanks_count": _get_field(item, "thanks_count", "thanksCount")
        or _get_field(item, "like_count", "likeCount")
        or _get_field(item, "liked_count", "likedCount")
        or 0,
    }


def _stats_from_entities(url: str, entity_key: str) -> dict[str, Any]:
    """Extract stats from a Zhihu page's js-initialData entities for articles or pins."""
    entities = get_page_state(fetch_page_html(url))
    items = entities.get(entity_key, {})
    if not items:
        raise ValueError(f"No {entity_key} data found in page entities for {url}")
    item_data = next(iter(items.values()))
    stats = _extract_stats(item_data)
    stats["title"] = item_data.get("title", "") or f"{entity_key.rstrip('s')} {item_data.get('id', '')}"
    stats["url"] = url
    stats["type"] = entity_key
    return stats


def _stats_for_answer(question_id: str, answer_id: str) -> dict[str, Any]:
    """Extract answer stats via the answer detail API."""
    resp = session.get(
        f"https://www.zhihu.com/api/v4/answers/{answer_id}",
        params={
            "include": ("voteup_count,favlists_count,comment_count,thanks_count,like_count,liked_count,question.id"),
        },
    )
    data = resp.json()
    if "error" in data:
        raise ValueError(f"API error for answer {answer_id}: {data['error']}")
    stats = _extract_stats(data)
    stats["title"] = (data.get("question") or {}).get("title", f"answer {answer_id}")
    stats["url"] = f"https://www.zhihu.com/question/{question_id}/answer/{answer_id}"
    stats["type"] = "answers"
    return stats


def fetch_creator_content_metrics(
    item_id: str,
    creator_type: str,
    *,
    start: str | None = None,
    end: str | None = None,
    aggregated: bool = False,
) -> dict[str, Any] | None:
    """Fetch creator-side metrics for one piece of content.

    The aggregated endpoint returns a single totals object, while the daily
    endpoint returns a per-day list.  Only the authenticated content author can
    read either one.

    :param item_id: Creator token of the content (the answer/article/pin ID).
    :param creator_type: Singular creator type: ``"answer"``, ``"article"`` or
        ``"pin"``.
    :param start: Optional start date (``YYYY-MM-DD``) for the daily endpoint.
    :param end: Optional end date (``YYYY-MM-DD``) for the daily endpoint.
    :param aggregated: When ``True`` query the aggregated endpoint, otherwise
        the daily one.
    :returns: The decoded JSON payload, or ``None`` when the API answers with a
        non-200 status.
    """
    url = CREATOR_CONTENT_AGGR_URL if aggregated else CREATOR_CONTENT_DAILY_URL
    params: dict[str, str] = {"type": creator_type, "token": item_id}
    if start is not None:
        params["start"] = start
    if end is not None:
        params["end"] = end

    response = session.get(url, params=params, timeout=15)
    if response.status_code != 200:
        return None
    return response.json()


def _fetch_creator_aggr(entity_key: str, item_id: str) -> dict[str, Any] | None:
    """Fetch aggregated metrics from creator API.

    Only works when the authenticated user is the content author.  Returns
    None when the entity has no creator type or the API answers non-200.
    """
    creator_type = _ENTITY_TO_CREATOR_TYPE.get(entity_key)
    if not creator_type:
        return None

    data = fetch_creator_content_metrics(item_id, creator_type, aggregated=True)
    if data is None:
        return None
    return {
        "share": int(data.get("share", 0)),
        "pv": int(data.get("pv", 0)),
        "show": int(data.get("show", 0)),
    }


def get_item_stats(
    url: str, with_share: bool = False, with_pv: bool = False, with_show: bool = False
) -> dict[str, Any]:
    """Return engagement summary for a Zhihu article, answer, or pin URL.

    Returns a dict with keys: type, title, url, voteup_count, favlists_count,
    comment_count, thanks_count. Creator-only flags (require author access):
    with_share adds share_count, with_pv adds pv (page views),
    with_show adds show (impressions).

    Raises ValueError if the URL type is not supported or data is unavailable.
    """
    item_type, item_id = get_type_and_id(url)
    if not item_type or not item_id:
        raise ValueError(f"Cannot parse Zhihu URL: {url}")

    if item_type == "articles":
        result = _stats_from_entities(url, "articles")
    elif item_type == "pins":
        result = _stats_from_entities(url, "pins")
    elif item_type == "answers":
        parts = item_id.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid answer ID format: {item_id}")
        result = _stats_for_answer(parts[0], parts[1])
    else:
        raise ValueError(f"Stats not supported for content type: {item_type}")

    if with_share or with_pv or with_show:
        token = item_id.split("/")[-1] if item_type == "answers" else item_id
        aggr = _fetch_creator_aggr(item_type, token)
        if with_share:
            result["share_count"] = aggr["share"] if aggr else None
        if with_pv:
            result["pv"] = aggr["pv"] if aggr else None
        if with_show:
            result["show"] = aggr["show"] if aggr else None

    return result
