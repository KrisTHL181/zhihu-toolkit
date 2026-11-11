from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Iterable, Iterator
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from zhihu_cli.content.handlers.cache_manager import cache_manager
from zhihu_cli.content.handlers.requests import session
from zhihu_cli.content.handlers.risk_control import detect_gate
from zhihu_cli.content.utils.wait import wait

PAGE_SIZE: int = 20
"""Items requested per API page.

Page size is an API-level detail, not a user-facing knob: the CLI exposes only
``-n/--max`` (a cap on the *total* number of items).  This module owns the
value so every handler paginates identically.

It is a *request*, not a guarantee — several endpoints cap their own page size
and ignore ``limit`` entirely (the recommend/moments feeds return ~5-6 items
per page).  Anything that needs a page count should bound the walk with
``max_pages`` rather than dividing an item budget by this number.
"""

EMPTY_PAGE_LIMIT: int = 3
"""Consecutive empty pages tolerated before pagination is considered stalled.

Several ``/followers`` / ``/followees`` endpoints return an endless run of
empty pages once their server-side enumeration cap (~3020 items) is reached,
while still reporting ``is_end=false`` and a non-null ``paging.next``.  A
measured 3.4M-follower account produced roughly 173k such pages.  Three
consecutive empty pages in a row is a reliable signal that the cap was hit, so
the walk stops instead of following ``next`` forever.
"""


def _set_query_param(url: str, key: str, value: Any) -> str:
    """Return ``url`` with the query parameter ``key`` set to ``value``.

    Any existing value for ``key`` is replaced.  Blank query values are
    preserved (e.g. a trailing ``offset=``).
    """
    parts = urlparse(url)
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != key]
    query.append((key, str(value)))
    return urlunparse(parts._replace(query=urlencode(query)))


class _StreamStats:
    """Mutable bookkeeping shared between :class:`PageStream` and its page loop."""

    __slots__ = ("stalled", "totals", "truncated", "yielded")

    def __init__(self) -> None:
        self.totals: int | None = None
        self.yielded = 0
        self.truncated = False
        self.stalled = False


class PageStream(Iterator[Any]):
    """Iterable returned by :func:`stream_handler`, carrying pagination metadata.

    :ivar totals: Total number of items reported by the API, or ``None`` when the
        API never reported a (non-zero) total.
    :ivar yielded: Number of items produced so far.
    :ivar truncated: Whether pagination stopped early because ``max_items`` or
        ``max_pages`` was reached.
    :ivar stalled: Whether pagination stopped because the server returned
        :data:`EMPTY_PAGE_LIMIT` consecutive empty pages — a server-side
        enumeration cap, not a user cap.

    Call :meth:`close` to abandon a partially-read stream; it closes the page
    generator deterministically and leaves ``truncated`` / ``stalled`` alone,
    because a consumer's early stop is neither.
    """

    def __init__(self, source: Iterator[Any], stats: _StreamStats) -> None:
        self._source = source
        self._stats = stats
        self._exhausted = False

    def __iter__(self) -> PageStream:
        return self

    def __next__(self) -> Any:
        if self._exhausted:
            raise StopIteration
        try:
            return next(self._source)
        except StopIteration:
            self._exhausted = True
            # ── natural end of stream — check completeness ──────────────────
            # Skip the warning when pagination was intentionally capped by
            # ``max_items``.  A consumer that abandons the stream never gets
            # here at all, so it never sees the warning either.
            totals = self._stats.totals
            if totals is not None and self._stats.yielded < totals and not self._stats.truncated:
                missing = totals - self._stats.yielded
                if not cache_manager.get_suppress_incomplete_warning():
                    warnings.warn(
                        f"API reported {totals} total items but only {self._stats.yielded} were returned "
                        f"(missing {missing}).",
                        stacklevel=2,
                    )
            raise

    def close(self) -> None:
        """Stop pagination and release the underlying page generator.

        A consumer that abandons a partially-read stream (``for ... break``)
        leaves the generator suspended at its last ``yield`` until garbage
        collection reaches it.  Closing makes that deterministic and stops any
        further page fetches.

        Closing is a *consumer* decision, not a server-side condition: it
        leaves :attr:`truncated` and :attr:`stalled` untouched and does not
        emit the incomplete-stream warning.  The call is idempotent and safe
        on a stream that has already ended.

        :raises RuntimeError: If the page generator is currently executing in
            another thread — the same restriction as ``generator.close()``.
        """
        if self._exhausted:
            return
        # Set the flag first: a concurrent ``next()`` then sees StopIteration
        # instead of touching a generator this call is about to close.
        self._exhausted = True
        source_close = getattr(self._source, "close", None)
        if callable(source_close):
            source_close()

    @property
    def totals(self) -> int | None:
        """Total number of items reported by the API (``None`` when unknown)."""
        return self._stats.totals

    @property
    def yielded(self) -> int:
        """Number of items produced so far."""
        return self._stats.yielded

    @property
    def truncated(self) -> bool:
        """Whether the stream stopped early because ``max_items`` or ``max_pages`` was reached."""
        return self._stats.truncated

    @property
    def stalled(self) -> bool:
        """Whether pagination stopped because the server returned ``EMPTY_PAGE_LIMIT`` consecutive empty pages (a server-side enumeration cap, not a user cap)."""
        return self._stats.stalled


def _paginate(
    initial_url: str,
    parser: Callable[[dict[str, Any]], Iterable[Any]],
    extract_next: Callable[[dict[str, Any]], str | None] | None,
    delay: float,
    limit: int | None,
    max_items: int | None,
    max_pages: int | None,
    skip_app_headers: bool,
    stats: _StreamStats,
) -> Iterator[Any]:
    """Drive the pagination loop, yielding parsed items and updating ``stats``."""
    current_url = initial_url
    if limit is not None:
        current_url = _set_query_param(current_url, "limit", limit)

    api_totals = 0
    yielded_count = 0
    pages_fetched = 0
    stopped_by_limit = False
    consecutive_empty_pages = 0

    while current_url and (max_items is None or yielded_count < max_items):
        if max_pages is not None and pages_fetched >= max_pages:
            # Capping by page count is still an intentional early stop, so it must
            # set ``truncated`` — otherwise the incomplete-stream warning fires on
            # a stream that ended exactly where it was asked to.
            stats.truncated = True
            break

        resp = session.get(current_url, skip_app_headers=skip_app_headers)
        # Paginated walkers used to surface gates as bare HTTPErrors (and thus
        # tracebacks).  ``ZhihuSession`` handles API refusal codes itself; this
        # catches the response-shaped gates (``__zse_ck`` HTML, ``/unhuman``
        # redirects, risk-control JSON) for the daemon path and any handler
        # layered on top of the stream.
        gate = detect_gate(resp, current_url)
        if gate is not None:
            raise gate
        resp.raise_for_status()
        data = resp.json()
        pages_fetched += 1

        paging_raw = data.get("paging", {})
        # Some endpoints (e.g. next-content-render) return paging as a
        # JSON-encoded string rather than a dict — normalise it here.
        if isinstance(paging_raw, str):
            try:
                paging = json.loads(paging_raw)
            except (json.JSONDecodeError, TypeError):
                paging = {}
        else:
            paging = paging_raw
        # Capture totals from the first page that reports a non-zero value.
        if api_totals == 0:
            api_totals = paging.get("totals", 0) or 0
            stats.totals = api_totals or None

        page_has_items = False
        for item in parser(data):
            page_has_items = True
            yielded_count += 1
            stats.yielded = yielded_count
            yield item
            if max_items is not None and yielded_count >= max_items:
                stopped_by_limit = True
                stats.truncated = True
                break

        if stopped_by_limit:
            break

        if page_has_items:
            consecutive_empty_pages = 0
        else:
            consecutive_empty_pages += 1
            if consecutive_empty_pages >= EMPTY_PAGE_LIMIT:
                # The endpoint has hit its server-side enumeration cap and will
                # keep returning empty pages forever; stop following ``next``.
                # ``truncated`` stays False so the incomplete-stream warning
                # still reports the missing tail.
                stats.stalled = True
                break

        if extract_next:
            current_url = extract_next(data)
        else:
            if paging.get("is_end", True):
                current_url = None
            else:
                current_url = paging.get("next")

        if current_url:
            current_url = current_url.replace("http://", "https://")

        wait(delay)


def stream_handler(
    initial_url: str,
    parser: Callable[[dict[str, Any]], Iterable[Any]],
    extract_next: Callable[[dict[str, Any]], str | None] | None = None,
    delay: float = 1.0,
    limit: int | None = None,
    max_items: int | None = None,
    max_pages: int | None = None,
    skip_app_headers: bool = False,
) -> PageStream:
    """Paginate a Zhihu API endpoint, yielding parsed items one by one.

    :param initial_url: The first page URL (includes ``offset=0``).
    :param parser: Called on each page's JSON body; must yield zero or more
        parsed items per page.
    :param extract_next: Optional custom pagination resolver.  When omitted,
        ``paging.next`` / ``paging.is_end`` is used.
    :param limit: Optional per-page item count.  When provided, it is injected
        into the request URL's ``limit`` query parameter (overriding any
        existing value), so callers need not build it into the base URL.
    :param max_items: Optional cap on the total number of items yielded.  When
        reached, pagination stops early without issuing the incomplete-stream
        warning.
    :param max_pages: Optional cap on the number of *pages fetched*.  Prefer this
        over ``max_items`` when the caller means "N pages": endpoints do not
        necessarily honour ``limit`` (the recommend feed returns ~5-6 items per
        page however large ``limit`` is), so multiplying a page count by a
        nominal page size silently over-fetches several times over.  Reaching the
        cap sets ``truncated``, like ``max_items`` does.
    :param skip_app_headers: When :data:`True`, omit the ``x-app-za`` /
        ``x-app-version`` app-identity headers.  Needed for endpoints
        (e.g. the articles list) that return 500 when ``x-app-za`` is
        present.  Works through both the daemon proxy and direct session.
    :returns: A :class:`PageStream` — iterable over the parsed items, and
        carrying ``totals`` / ``yielded`` / ``truncated`` pagination metadata.
    """
    stats = _StreamStats()
    source = _paginate(
        initial_url,
        parser,
        extract_next,
        delay,
        limit,
        max_items,
        max_pages,
        skip_app_headers,
        stats,
    )
    return PageStream(source, stats)
