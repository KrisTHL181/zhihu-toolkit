"""Response challenge registry.

A *response challenge* is a detection plus solve step applied to every response
before it is returned to the caller.  Zhihu hides several anti-crawl mechanisms
behind ordinary-looking responses — an EO_Bot page served as HTTP 200
``text/html``, or a 403 captcha envelope — and this package is the single
extension point for them.

Adding a mechanism means dropping a module into this package: each module
implements :class:`ResponseChallenge` and calls :func:`register` at import
time, and :func:`registered` discovers every non-underscore module with
:func:`pkgutil.iter_modules`.  There is no central list to edit.

Dispatch is deliberately order-independent (:func:`run_challenges`): after a
handler solves a challenge the whole registry re-examines the retried response,
so which handler runs first cannot change the outcome.

This package is stdlib-only on purpose: it is imported at module level by
:mod:`zhihu_cli.content.handlers._session_core` and must not drag the heavy
dependencies in with it.  Handler modules may import their domain module (e.g.
``challenges.captcha`` imports the helpers from ``handlers.captcha``) but must
keep curl_cffi out.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import AbstractSet, Callable

__all__ = [
    "ChallengeContext",
    "ResponseChallenge",
    "register",
    "registered",
    "run_challenges",
]


@dataclass
class ChallengeContext:
    """Everything a challenge handler needs about the request it is judging.

    :param session: The session that sent the request (a
        :class:`~zhihu_cli.content.handlers._session_core.ZhihuSession` or a
        duck-typed stand-in).
    :param method: HTTP method of the request.
    :param url: URL of the request.
    :param kwargs: Keyword arguments the request was sent with.  Handlers
        consult ``stream`` so they never touch a streamed response's body.
    :param epoch_at_send: The captcha gate's resolution counter as read
        immediately before the send.  A higher current value means the response
        was produced with pre-resolution cookies and is stale.  The session
        refreshes it in place before each retry.
    :param retry: Re-send the original request once, bypassing all challenge
        handling.  Called as ``ctx.retry(ctx)`` so the session can refresh
        ``epoch_at_send`` immediately before the re-send.
    """

    session: Any
    method: str
    url: str
    kwargs: dict[str, Any]
    epoch_at_send: int
    retry: Callable[[ChallengeContext], Any]


class ResponseChallenge(Protocol):
    """One response challenge: detect it, then solve it.

    ``detect`` is called for every response; it must be cheap and must return
    ``None`` unless the response really is this handler's challenge.  ``solve``
    is called only when ``detect`` returned a value, and returning ``True``
    tells the registry to re-send the original request once.
    """

    name: str
    """Identifier used by ``session.disabled_challenges``."""

    def detect(self, response: Any, ctx: ChallengeContext) -> Any | None:
        """Return an opaque challenge value, or ``None`` when not applicable.

        :param response: The response to inspect.
        :param ctx: Context of the request that produced it.
        :returns: A value understood by :meth:`solve`, or ``None``.
        """

    def solve(self, challenge: Any, ctx: ChallengeContext) -> bool:
        """Try to solve *challenge*.

        :param challenge: The value returned by :meth:`detect`.
        :param ctx: Context of the request that produced it.
        :returns: ``True`` when the challenge was solved and the request should
            be re-sent once for this handler; the retried response is
            re-examined by the other handlers.
        """


#: Registered handlers, in module-discovery order.
_REGISTRY: list[ResponseChallenge] = []

#: Set before importing the handler modules so a re-entrant call cannot loop.
_loaded = False


def register(challenge: ResponseChallenge) -> ResponseChallenge:
    """Register a response challenge handler.

    Called at import time by the handler module.  Registration is idempotent
    per module import (Python caches modules), so no de-duplication is needed.

    :param challenge: The handler to append to the process-wide registry.
    :returns: *challenge*, so the module can keep it as a singleton.
    """
    _REGISTRY.append(challenge)
    return challenge


def registered() -> tuple[ResponseChallenge, ...]:
    """Return every registered handler, importing the package's modules once.

    Every non-underscore module in this package is imported on first call; each
    one calls :func:`register` at import time.  Dropping a new module into the
    package is therefore the whole extension step — no central list exists.

    ``_loaded`` is set *before* the imports so a re-entrant call (e.g. from a
    module imported during registration) cannot recurse.

    :returns: The registered handlers in discovery order.  Order carries no
        meaning for dispatch; see :func:`run_challenges`.
    """
    global _loaded
    if not _loaded:
        _loaded = True
        for info in pkgutil.iter_modules(__path__):
            if info.name.startswith("_"):
                continue
            importlib.import_module(f"{__name__}.{info.name}")
    return tuple(_REGISTRY)


def run_challenges(
    response: Any,
    ctx: ChallengeContext,
    *,
    skip: AbstractSet[str] = frozenset(),
) -> Any:
    """Run every registered challenge against *response* until none applies.

    Dispatch is order-independent.  The pass starts at the first unsolved
    handler; whenever one solves a challenge the original request is re-sent
    once through ``ctx.retry(ctx)`` and the whole registry re-examines the
    *new* response — starting over is what lets an earlier handler react to a
    challenge exposed by a later one.  A handler that has solved once is
    skipped for the rest of the request, which bounds the loop: every retry
    consumes one handler, and ``ctx.retry`` itself bypasses challenge handling.

    :param response: The response to inspect.
    :param ctx: Context of the request that produced it.
    :param skip: Names of handlers to ignore — ``session.disabled_challenges``.
    :returns: The final response, after every applicable handler has solved.
    """
    handlers = [handler for handler in registered() if handler.name not in skip]
    handled: set[str] = set()
    while True:
        for handler in handlers:
            if handler.name in handled:
                continue
            challenge = handler.detect(response, ctx)
            if challenge is None or not handler.solve(challenge, ctx):
                continue
            handled.add(handler.name)
            response = ctx.retry(ctx)
            break  # re-examine the NEW response with every unsolved handler
        else:
            return response
