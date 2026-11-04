from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from zhihu_cli.paths import zhihu_home

if TYPE_CHECKING:
    from typing import Any

USER_AGENT_FILE = zhihu_home() / "user-agent"


def get_data_dir() -> Path:
    """Return ~/.zhihu-cli/ data directory, creating it if needed."""
    data_dir = zhihu_home()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_user_agent() -> str | None:
    """Return the configured User-Agent from ~/.zhihu-cli/user-agent, or None."""
    if USER_AGENT_FILE.exists():
        ua = USER_AGENT_FILE.read_text(encoding="utf-8").strip()
        if ua:
            return ua
    return None


def set_user_agent(ua: str | None) -> None:
    """Set (or clear) the User-Agent in ~/.zhihu-cli/user-agent."""
    if ua is None:
        USER_AGENT_FILE.unlink(missing_ok=True)
    else:
        get_data_dir()  # ensure directory exists
        USER_AGENT_FILE.write_text(ua.strip(), encoding="utf-8")


ZHIHU_ARTICLE_PATTERN = r"https?://zhuanlan\.zhihu\.com/p/(\d+)"
ZHIHU_QUESTION_PATTERN = r"https?://(?:www\.)?zhihu\.com/question/(\d+)"
ZHIHU_QUESTION_WITH_ANSWER_PATTERN = r"https?://(?:www\.)?zhihu\.com/question/(\d+)(?:/answer/(\d+))?"
ZHIHU_ANSWER_SHORT_PATTERN = r"https?://(?:www\.)?zhihu\.com/answer/(\d+)"
ZHIHU_PIN_PATTERN = r"https?://(?:www\.)?zhihu\.com/pin/([^/?#]+)"
ZHIHU_ZVIDEO_PATTERN = r"https?://(?:www\.)?zhihu\.com/zvideo/(\d+)"
ZHIHU_COLLECTION_PATTERN = r"https?://(?:www\.)?zhihu\.com/collection/(\d+)"
ZHIHU_CONSULT_CONVERSATION_PATTERN = r"https?://(?:www\.)?zhihu\.com/consult/conversation/(\d+)(?:/answer)?"
ZHIHU_PAID_COLUMN_SECTION_PATTERN = r"https?://(?:www\.)?zhihu\.com/market/paid_column/(\d+)/section/(\d+)"

#: Matches the ``/answer/<id>`` suffix of an answer path.
ANSWER_PATH_RE = re.compile(r"/answer/(\d+)/?$")
#: Matches a ``/pin/<id>`` path segment.
PIN_PATH_RE = re.compile(r"/pin/([^/?#]+)")
#: Member profile API template; ``{token}`` is the user's ``url_token``.
MEMBER_API = "https://www.zhihu.com/api/v4/members/{token}"


def split_composite_id(item_id: str) -> str:
    """Extract the trailing ID from a composite ``parent/child`` ID.

    :param item_id: A bare ID or a composite ``question_id/answer_id`` style ID.
    :returns: The part after the slash, or the input unchanged when there is no
        slash.
    """
    if "/" in item_id:
        return item_id.split("/")[1]
    return item_id


def canonical_answer_url(ref: str) -> str:
    """Normalise an answer reference to ``https://www.zhihu.com/answer/<id>``.

    :param ref: A bare answer ID, a short answer URL, or a
        ``/question/<qid>/answer/<aid>`` URL.
    :returns: The canonical URL, or the stripped input when it cannot be
        parsed.
    """
    candidate = ref.strip()
    if candidate.isdigit():
        return f"https://www.zhihu.com/answer/{candidate}"

    match = ANSWER_PATH_RE.search(urlparse(candidate).path)
    if match:
        return f"https://www.zhihu.com/answer/{match.group(1)}"
    return candidate


def canonical_pin_url(ref: str) -> str:
    """Normalise a pin reference to ``https://www.zhihu.com/pin/<id>``.

    :param ref: A bare pin ID/token or a full ``/pin/<id>`` URL.
    :returns: The canonical URL, or the stripped input when it cannot be
        parsed.
    """
    candidate = ref.strip()
    if candidate.startswith(("http://", "https://")):
        match = PIN_PATH_RE.search(urlparse(candidate).path)
        if match:
            return f"https://www.zhihu.com/pin/{match.group(1)}"
        return candidate
    return f"https://www.zhihu.com/pin/{candidate}"


def get_type_and_id(url: str) -> tuple[str | None, str | None]:
    """
    Returns (type, id).

    type: ``"articles"``, ``"questions"``, ``"answers"``, ``"pins"``,
    ``"zvideos"``, ``"collections"``, ``"consult_conversations"``,
    ``"paid_column_section"``, or ``None``.

    For ``"answers"`` the id is ``"question_id/answer_id"`` (composite).
    For ``"paid_column_section"`` the id is ``"column_id/section_id"``.
    """
    match = re.search(ZHIHU_ARTICLE_PATTERN, url)
    if match:
        return ("articles", match.group(1))

    match = re.search(ZHIHU_QUESTION_WITH_ANSWER_PATTERN, url)
    if match and match.group(2):
        return ("answers", f"{match.group(1)}/{match.group(2)}")

    match = re.search(ZHIHU_ANSWER_SHORT_PATTERN, url)
    if match:
        # Short answer URL (e.g. https://www.zhihu.com/answer/123456) —
        # no question ID available; downstream callers use _resolve_answer_id()
        # which handles the bare answer ID just fine.
        return ("answers", match.group(1))

    match = re.search(ZHIHU_QUESTION_PATTERN, url)
    if match:
        return ("questions", match.group(1))

    match = re.search(ZHIHU_PIN_PATTERN, url)
    if match:
        return ("pins", match.group(1))

    match = re.search(ZHIHU_ZVIDEO_PATTERN, url)
    if match:
        return ("zvideos", match.group(1))

    match = re.search(ZHIHU_COLLECTION_PATTERN, url)
    if match:
        return ("collections", match.group(1))

    match = re.search(ZHIHU_CONSULT_CONVERSATION_PATTERN, url)
    if match:
        return ("consult_conversations", match.group(1))

    match = re.search(ZHIHU_PAID_COLUMN_SECTION_PATTERN, url)
    if match:
        # Composite ID: "column_id/section_id" (analogous to answers)
        return ("paid_column_section", f"{match.group(1)}/{match.group(2)}")

    return (None, None)


def fmt_time(ts: int | float | None) -> str:
    if ts:
        try:
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(ts)
    return "unknown time"


def safe_json(resp) -> Any:
    """Parse a JSON response body without raising on an empty or non-JSON one.

    Zhihu's write endpoints (follow, vote, thank, block, …) answer a
    successful DELETE with ``204 No Content`` and no body at all, so a bare
    ``resp.json()`` would fail with ``JSONDecodeError`` even though the
    action succeeded.

    :param resp: A response object returned by :class:`ZhihuSession`.
    :returns: The decoded JSON value, ``{"success": resp.ok, "status_code":
        ...}`` when the body is empty, or an ``error`` dict when the body is
        neither empty nor JSON.
    """
    if not resp.text.strip():
        return {"success": resp.ok, "status_code": resp.status_code}
    try:
        return resp.json()
    except Exception:
        return {"error": True, "status_code": resp.status_code, "message": resp.text[:200]}
