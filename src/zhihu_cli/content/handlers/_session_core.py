"""Core session classes and helpers that depend on curl_cffi and lxml.

This module is imported *lazily* by ``requests.py`` — only when the daemon
is unavailable and a direct :class:`ZhihuSession` is needed, or when
``fetch_page_html`` / ``get_page_state`` are called.

Keeping these heavy imports (curl_cffi, lxml) out of the ``requests.py``
top-level allows the CLI to start quickly when the background daemon is
running, because it only needs :class:`~zhihu_cli.daemon.proxy.DaemonProxySession`
(which uses nothing but stdlib).
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from contextlib import nullcontext
from http.cookiejar import CookieJar
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from curl_cffi import CurlHttpVersion
from curl_cffi import requests as _requests
from curl_cffi.requests.exceptions import CookieConflict
from lxml import html as lxml_html

from zhihu_cli.content.handlers import get_user_agent
from zhihu_cli.content.handlers.cache_manager import cache_manager
from zhihu_cli.content.handlers.captcha_gate import captcha_gate
from zhihu_cli.content.handlers.challenges import ChallengeContext, run_challenges
from zhihu_cli.content.handlers.cookie_utils import seed_cookie_jar
from zhihu_cli.content.utils import rate_limit
from zhihu_cli.content.utils.zse import ZSE93

if TYPE_CHECKING:
    from collections.abc import Iterable


def get_browser(ua: str) -> _requests.BrowserTypeLiteral:
    """Detect browser family from a User-Agent string for curl_cffi TLS impersonation.

    Matching order is deliberate:
    1. Edge first — its UA also contains ``Chrome/``.
    2. Chrome Android — distinguish phone (``Mobile``) vs tablet.
    3. Chrome desktop — ``Chrome/`` without ``Mobile``.
    4. Firefox — ``Firefox/`` + ``Gecko/``.
    5. iOS Safari — ``iPhone/iPad/iPod`` + ``Safari/``.
    6. Desktop Safari — ``Safari/`` + ``Version/``, no ``Mobile``, no ``Chrome/``.

    Tor Browser is intentionally **not** detected — it forges a generic Firefox
    UA on all platforms and actively hides its fingerprint.
    """
    if not ua:
        return "chrome"

    # Edge — contains the Edg/ marker (newer Chromium-based Edge).
    if re.search(r"Edg/", ua):
        return "edge"

    # Chrome on Android — Linux + Android + Chrome/.
    if "Android" in ua and re.search(r"Chrome/", ua):
        return "chrome_android"

    # Chrome desktop — Chrome/ without Edg/, Android, or Mobile.
    if re.search(r"Chrome/", ua) and "Mobile" not in ua:
        return "chrome"

    # Firefox — Firefox/ + Gecko/.
    if re.search(r"Firefox/", ua) and re.search(r"Gecko/", ua):
        return "firefox"

    # iOS Safari — iPhone/iPad/iPod device token + Safari/.
    if re.search(r"(?:iPhone|iPad|iPod)", ua) and re.search(r"Safari/", ua):
        return "safari_ios"

    # Desktop Safari — Safari/ + Version/, no Mobile, no Chrome/.
    if re.search(r"Safari/", ua) and re.search(r"Version/", ua) and "Mobile" not in ua:
        return "safari"

    return "chrome"


class ZhihuSession(_requests.Session):
    """Session that auto-injects x-zse-93 / x-zse-96 signing headers.

    Equivalent to zhihu++'s ``signFetchRequest()``: builds a canonical
    source string from the ZSE version, request path, ``d_c0`` cookie, and
    JSON body (when present), hashes it with MD5, then encrypts the hash
    via the ZSEv4 block cipher.

    Set ``captcha_handler`` to "auto" (interactive), "prompt" (ask first),
    or "ignore" (skip handling) to control captcha / risk-control handling.
    Anti-crawl responses are handled by the registry in
    :mod:`zhihu_cli.content.handlers.challenges`; a handler whose name is in
    ``disabled_challenges`` is skipped for this session.  Assign a whole new
    set (``session.disabled_challenges = {"eo_bot"}``) — the getter returns a
    snapshot, so mutating it does not change the session.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._zse_cipher = None  # lazy init to avoid import-time side effects
        self.captcha_handler: str = "auto"  # "auto", "prompt", or "ignore"
        # Names of response-challenge handlers to skip, e.g. {"eo_bot"}.
        # Configure at setup time; the registry consults it on every response.
        self._disabled_challenges: set[str] = set()
        # Serialises the pop→request→restore window over the *shared*
        # ``self.headers`` dict for ``skip_app_headers`` requests (see
        # ``request``); only those callers ever take it.
        self._app_headers_lock = threading.Lock()

    @property
    def disabled_challenges(self) -> set[str]:
        """Names of response-challenge handlers this session skips.

        Returns a copy so a caller cannot mutate the live set through the
        snapshot; assign a new set to change it.

        :returns: A new set containing the current handler names.
        """
        return set(self._disabled_challenges)

    @disabled_challenges.setter
    def disabled_challenges(self, names: Iterable[str]) -> None:
        if isinstance(names, str):
            raise TypeError("disabled_challenges must be a set/list of handler names, not a string")
        self._disabled_challenges = {str(name) for name in names}

    @property
    def zse_cipher(self):
        if self._zse_cipher is None:
            from zhihu_cli.content.utils.zse import ZSECipher

            self._zse_cipher = ZSECipher()
        return self._zse_cipher

    def _get_dc0(self) -> str:
        # The jar wins once it holds cookies: libcurl replaces an explicit
        # ``Cookie`` header with the jar's contents, and a CAPTCHA resolution
        # refreshes ``d_c0`` on the jar only.  Sessions built without a
        # (seeded) jar keep using the static header, as before.
        try:
            value = self.cookies.get("d_c0")
        except CookieConflict:
            candidates = [cookie for cookie in self.cookies.jar if cookie.name == "d_c0"]
            value = next(
                (cookie.value for cookie in candidates if cookie.domain == ".zhihu.com"),
                candidates[0].value if candidates else None,
            )
        if value:
            return str(value)

        cookie_header = self.headers.get("Cookie", "")
        if isinstance(cookie_header, str):
            for part in cookie_header.split("; "):
                if part.startswith("d_c0="):
                    return part[5:]
        return ""

    def _build_sign_source(self, method: str, url: str, kwargs: dict) -> str | None:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""

        if not (hostname == "zhihu.com" or hostname.endswith(".zhihu.com")):
            return None

        path_and_query = parsed.path
        if parsed.query:
            path_and_query += "?" + parsed.query

        dc0 = self._get_dc0()

        body = None
        if "json" in kwargs and kwargs["json"] is not None:
            body = json.dumps(kwargs["json"], separators=(",", ":"), ensure_ascii=False)
        elif "data" in kwargs:
            data = kwargs["data"]
            if isinstance(data, str):
                try:
                    json.loads(data)
                    body = data
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

        parts = [ZSE93, path_and_query, dc0]
        if body is not None:
            parts.append(body)
        return "+".join(parts)

    def request(self, method, url, **kwargs):
        """Send a request, waiting at the captcha gate and handling challenges.

        The gate handles multi-threaded use: while any thread is resolving a
        captcha, every other thread parks here instead of sending requests
        Zhihu would only answer with another 403.  The wait is bounded (see
        :mod:`zhihu_cli.content.handlers.captcha_gate`), so a timed-out thread
        still sends and simply sees the original 403.

        Post-response anti-crawl handling is delegated to the registry in
        :mod:`zhihu_cli.content.handlers.challenges`: each registered handler
        may solve the challenge and have the request re-sent once.

        :param method: HTTP method.
        :param url: Full request URL.
        :param kwargs: Same keyword arguments as
            :meth:`curl_cffi.requests.Session.request`, plus
            ``skip_app_headers`` (suppress ``x-app-version``/``x-app-za``).
        :returns: The response, retried when a challenge was resolved.
        """
        # Order matters: the captcha gate can park many threads, so waiting
        # there first keeps the rate limiter's slots from being consumed by
        # threads that are not actually about to send.  This is the single
        # chokepoint for direct sessions, the daemon server's shared session,
        # and every ``_stream_direct`` fallback — no other call site needed.
        captcha_gate.acquire()
        rate_limit.limiter.acquire()
        skip_app = kwargs.pop("skip_app_headers", False)

        saved_version = None
        saved_za = None

        # Taking the app-headers lock for the whole request is the cheap,
        # obviously-correct way to keep two concurrent ``skip_app_headers``
        # requests from interleaving their pop/restore on the shared header
        # dict.  Residual (pre-existing, out of scope) limitation: a request
        # that does *not* skip the app headers can still observe them missing
        # while a skip_app request is in flight.
        with self._app_headers_lock if skip_app else nullcontext():
            if skip_app:
                saved_version = self.headers.pop("x-app-version", None)
                saved_za = self.headers.pop("x-app-za", None)

            try:
                sign_source = self._build_sign_source(method, url, kwargs)
                if sign_source is not None:
                    md5_hash = hashlib.md5(sign_source.encode()).hexdigest()
                    signature = self.zse_cipher.encrypt(md5_hash)

                    zse_headers = {
                        "x-zse-93": ZSE93,
                        "x-zse-96": "2.0_" + signature,
                        "x-requested-with": "fetch",
                    }
                    user_headers = kwargs.get("headers", {})
                    zse_headers.update(user_headers)
                    kwargs["headers"] = zse_headers

                # Read the gate's resolution counter immediately before the
                # send — deliberately not at request entry.  A resolution that
                # completed *before* this send already applies to the request
                # (post-resolution cookies went out with it), so a 403 on it is
                # a genuine new challenge and must still prompt; only a
                # resolution that completes *after* the send leaves us holding
                # a stale 403.
                epoch_at_send = captcha_gate.epoch

                resp = super().request(method, url, **kwargs)

                # Every registered challenge gets one look at the response.
                # ``retry`` re-sends with ``_raw_request`` so challenge
                # handling is bypassed and cannot recurse.  Running inside the
                # ``try`` keeps retries under the same app-header restore.
                ctx = ChallengeContext(
                    session=self,
                    method=method,
                    url=url,
                    kwargs=kwargs,
                    epoch_at_send=epoch_at_send,
                    retry=self._retry_challenge,
                )
                return run_challenges(resp, ctx, skip=self._disabled_challenges)
            finally:
                if skip_app:
                    if saved_version is not None:
                        self.headers["x-app-version"] = saved_version
                    if saved_za is not None:
                        self.headers["x-app-za"] = saved_za

    def _retry_challenge(self, ctx: ChallengeContext) -> Any:
        """Re-send a request after a challenge was solved.

        Reads the gate epoch immediately before the re-send, exactly like the
        original pre-send read; a stale value would make a genuine post-retry
        403 look like a pre-resolution one.

        :param ctx: Context of the challenged request; ``epoch_at_send`` is
            refreshed in place.
        :returns: The raw response, with no challenge handling applied.
        """
        ctx.epoch_at_send = captcha_gate.epoch
        return self._raw_request(ctx.method, ctx.url, ctx.kwargs)

    def _raw_request(self, method, url, kwargs):
        """Send a request directly, bypassing all challenge handling.

        Used as :attr:`ChallengeContext.retry`, so a solved challenge can
        re-send the original request exactly once without the registry (or this
        method) recursing.

        :param method: HTTP method.
        :param url: Full request URL.
        :param kwargs: Keyword arguments of the original request.
        :returns: The raw response.
        """
        return super().request(method, url, **kwargs)


def _resolve_user_agent(headers: dict[str, str]) -> str:
    """Return the effective User-Agent: configured override > profile UA > empty."""
    configured = get_user_agent()
    if configured:
        return configured
    return headers.get("User-Agent", "")


def _build_direct_session(
    profile_name: str | None = None,
    cookies: CookieJar | None = None,
    timeout: float | None = None,
) -> ZhihuSession:
    """Build a direct :class:`ZhihuSession` from a profile's headers.

    This is the original (non-daemon) session factory.

    :param profile_name: Profile whose headers to load.  ``None`` — the default —
        keeps the historical behaviour of using the *active* profile, so every
        existing caller is unaffected.  Naming a profile builds a session for a
        profile that is not active, which is what
        :func:`zhihu_cli.content.handlers.requests.use_profile` relies on.
    :param cookies: Jar the session should use.  When given, the session keeps
        this exact object (curl_cffi does not copy it) and the profile's cookies
        are seeded into it, so several sessions can share live cookie state.
        ``None`` keeps the historical per-session jar.
    :param timeout: Default request timeout in seconds.  ``None`` keeps
        curl_cffi's default.
    :returns: A session carrying that profile's headers.
    """
    hdrs = cache_manager.load_headers(profile_name)
    ua = _resolve_user_agent(hdrs)
    kwargs: dict[str, Any] = {
        "impersonate": get_browser(ua),
        "http_version": CurlHttpVersion.V3,
    }
    if cookies is not None:
        kwargs["cookies"] = cookies
    if timeout is not None:
        kwargs["timeout"] = timeout
    sess = ZhihuSession(**kwargs)
    sess.headers.update(hdrs)
    if ua:
        sess.headers["User-Agent"] = ua
    if cookies is not None:
        seed_cookie_jar(cookies, hdrs.get("Cookie", ""))
    return sess


def get_page_state(html_text: str, key: str = "entities") -> dict[str, Any]:
    """Extract ``js-initialData`` from a Zhihu SSR page.

    :param html_text: Raw HTML of a Zhihu page.
    :param key: Top-level key in ``initialState`` (default: ``"entities"``).
    :returns: The parsed JSON subtree.
    :raises ValueError: If the ``js-initialData`` script tag is missing.
    """
    doc = lxml_html.fromstring(html_text)

    script_tag = doc.find(".//script[@id='js-initialData']")
    if script_tag is None or not script_tag.text:
        raise ValueError("Could not find 'js-initialData' script tag")

    initial_data = json.loads(script_tag.text)
    return initial_data["initialState"][key]
