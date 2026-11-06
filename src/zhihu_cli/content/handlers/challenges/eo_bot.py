"""Automatic handling of Zhihu's EO_Bot JavaScript challenge.

When Zhihu rate-limits a client it may answer with **HTTP 200** and a small
``text/html`` page instead of JSON.  The page runs an obfuscated script that
writes two cookies and reloads the same URL:

* ``__tst_status=<sum of three random addends>#``
* ``EO_Bot_Ssid=<random 32-bit number>``

The challenge is stable in shape: the JS template (string table and rotation
constant) does not change, the three addends vary per response but always sum
to the same constant, and ``EO_Bot_Ssid`` is a literal in the script.  This
module parses those values and installs the cookies so the caller can retry
the original request once.

Detection is deliberately narrow (200 + ``text/html`` + the ``EO_Bot`` marker)
so ordinary HTML pages and real API JSON are never mistaken for a challenge.

This module is a :class:`~zhihu_cli.content.handlers.challenges.ResponseChallenge`
implementation: importing it registers :data:`EO_BOT_CHALLENGE`, and the
registry in :mod:`zhihu_cli.content.handlers.challenges` discovers and calls it
for every response before it is returned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from zhihu_cli.content.handlers.challenges import register
from zhihu_cli.content.handlers.cookie_utils import seed_cookie_jar, set_cookie

if TYPE_CHECKING:
    from zhihu_cli.content.handlers.challenges import ChallengeContext

#: Marker present in the challenge page.
_MARKER = b"EO_Bot"

#: How much of the body to scan for the marker (the page is < 1 KiB).
_SCAN_BYTES = 4096

#: Upper bound on the slice decoded for regex matching (64 KiB).
_MAX_DECODE_BYTES = 64 * 1024

#: The three addends whose sum becomes ``__tst_status``.
_ADDEND_RE = (
    re.compile(r"WTKkN:(\d+)"),
    re.compile(r"bOYDu:(\d+)"),
    re.compile(r"wyeCN:(\d+)"),
)

#: ``EO_Bot_Ssid`` is the literal passed to the add helper in case "3".
_SID_RE = re.compile(r'case"3"\s*:\s*t\s*=\s*a\[_0x649a\("0x7"\)\]\(t\s*,\s*(\d+)\)')


@dataclass(frozen=True)
class EoBotChallenge:
    """The two cookie values an EO_Bot challenge page asks the browser to set.

    :param tst_status: Value for ``__tst_status``, including the trailing ``#``.
    :param eo_bot_ssid: Value for ``EO_Bot_Ssid``.
    """

    tst_status: str
    eo_bot_ssid: str


def _is_zhihu_host(url: Any) -> bool:
    host = urlparse(str(url)).hostname or ""
    return host == "zhihu.com" or host.endswith(".zhihu.com")


def detect_eo_bot(response: Any, ctx: ChallengeContext) -> EoBotChallenge | None:
    """Detect and parse an EO_Bot challenge page.

    Streamed responses are skipped before their body is touched: reading
    ``response.content`` would consume a download that the caller has not asked
    for yet.

    :param response: A curl_cffi (or requests-like) response.
    :param ctx: Context of the request that produced it.
    :returns: An :class:`EoBotChallenge` when the response is a parsable
        challenge, ``None`` for a normal response or one whose template did not
        match.
    """
    if ctx.kwargs.get("stream"):
        return None
    if getattr(response, "status_code", None) != 200:
        return None
    if not _is_zhihu_host(getattr(response, "url", "")):
        return None

    headers = getattr(response, "headers", {}) or {}
    if "text/html" not in (headers.get("content-type") or "").lower():
        return None

    content = getattr(response, "content", None) or b""
    if _MARKER not in content[:_SCAN_BYTES]:
        return None

    text = content[:_MAX_DECODE_BYTES].decode("utf-8", "replace")
    addends = [re.search(pattern, text) for pattern in _ADDEND_RE]
    sid = _SID_RE.search(text)
    if not all(addends) or sid is None:
        # It is an EO_Bot page, but not the template we know how to parse.
        # There is no recovery to attempt, so behave like a normal response.
        return None

    total = sum(int(match.group(1)) for match in addends if match)
    return EoBotChallenge(tst_status=f"{total}#", eo_bot_ssid=sid.group(1))


def install_eo_bot_cookies(session: Any, challenge: EoBotChallenge) -> None:
    """Write the challenge cookies into *session*'s cookie jar.

    The jar is seeded from the session's static ``Cookie`` header first: the
    jar wins over an explicit header once it holds any cookie, so installing
    only the two challenge cookies would otherwise drop the profile's
    authentication cookies.

    :param session: The session that received the challenge.
    :param challenge: The parsed cookie values.
    """
    header = session.headers.get("Cookie", "")
    if isinstance(header, str) and header:
        seed_cookie_jar(session.cookies.jar, header)

    set_cookie(session.cookies.jar, "__tst_status", challenge.tst_status)
    set_cookie(session.cookies.jar, "EO_Bot_Ssid", challenge.eo_bot_ssid)


class EoBotChallengeHandler:
    """Registry handler for the EO_Bot JavaScript challenge."""

    name = "eo_bot"

    detect = staticmethod(detect_eo_bot)

    def solve(self, challenge: EoBotChallenge, ctx: ChallengeContext) -> bool:
        """Install the challenge cookies and ask for one retry.

        :param challenge: The parsed challenge.
        :param ctx: Context of the request that produced it.
        :returns: Always ``True`` — installing the cookies is enough for the
            retry to succeed.
        """
        install_eo_bot_cookies(ctx.session, challenge)
        return True


EO_BOT_CHALLENGE = EoBotChallengeHandler()
"""The singleton registered with the response-challenge registry."""

register(EO_BOT_CHALLENGE)
