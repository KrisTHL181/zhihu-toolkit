"""Captcha / risk-control response challenge.

Adapter between the registry and the captcha domain module
(:mod:`zhihu_cli.content.handlers.captcha`).  The detection helpers, banners
and interactive flow live there; this module only wires them into the
response-challenge protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from zhihu_cli.content.handlers.captcha import (
    _can_prompt_user,
    _is_captcha_endpoint,
    _print_captcha_warning,
    _print_non_interactive_captcha_warning,
    detect_captcha,
)
from zhihu_cli.content.handlers.captcha_gate import captcha_gate, resolve_captcha
from zhihu_cli.content.handlers.challenges import register

if TYPE_CHECKING:
    from zhihu_cli.content.handlers.challenges import ChallengeContext


class CaptchaChallenge:
    """Registry handler for 403 captcha / risk-control responses.

    The mode comes from ``session.captcha_handler``:

    * ``"ignore"`` — detection is skipped entirely;
    * ``"prompt"`` — print a warning and leave the 403 for the caller;
    * ``"auto"`` — resolve interactively once per challenge (coordinated by
      :data:`~zhihu_cli.content.handlers.captcha_gate.captcha_gate`) and retry.
    """

    name = "captcha"

    def detect(self, response: Any, ctx: ChallengeContext) -> dict | None:
        """Detect a captcha envelope, unless this session or URL opts out.

        :param response: The response to inspect.
        :param ctx: Context of the request that produced it.
        :returns: The ``detect_captcha()`` error dict, or ``None``.
        """
        if getattr(ctx.session, "captcha_handler", "ignore") == "ignore":
            return None
        # Avoid re-entrant handling for captcha-related endpoints.
        if _is_captcha_endpoint(urlparse(ctx.url).path):
            return None
        return detect_captcha(response)

    def solve(self, captcha_error: dict, ctx: ChallengeContext) -> bool:
        """Run the configured captcha handling and report whether to retry.

        :param captcha_error: Error dict returned by :func:`detect_captcha`.
        :param ctx: Context of the request that produced it.
        :returns: ``True`` when the challenge was resolved and the request
            should be re-sent once, ``False`` to return the 403 unchanged.
        """
        session = ctx.session
        if captcha_error.get("need_login"):
            # A login gate is not a captcha: do not open the unhuman page or
            # wait for input.  Report it once and let fetch_json/fetch_page_html
            # raise a ZhihuGateError with the next action.
            _print_non_interactive_captcha_warning(captcha_error)
            return False

        mode = getattr(session, "captcha_handler", "ignore")
        if mode == "prompt":
            _print_captcha_warning(captcha_error)
            return False
        if mode != "auto":
            return False

        if not _can_prompt_user():
            # ``handle_captcha`` prompts through ``input()``. With no terminal
            # to read from it would block forever (or die with EOFError), and
            # either way it consumes the 403 so the caller can no longer tell
            # "captcha" from "this item is unavailable". Degrade to the
            # "ignore" behaviour instead: hand the raw 403 back and let the
            # caller decide.
            _print_non_interactive_captcha_warning(captcha_error)
            return False
        if captcha_gate.is_resolver():
            # Re-entrant: this request came from the resolver thread's own
            # resolution flow — never prompt twice.
            return False
        if captcha_gate.epoch != ctx.epoch_at_send:
            # A resolution completed while our request was in flight, so this
            # 403 was produced before it and is stale.  We never parked at the
            # gate, so the wait branch below cannot see us: without this the
            # thread would become a *second* resolver and prompt the user for a
            # challenge that was just answered.  Retry through ``ctx.retry``
            # (bypassing challenge handling) so a second captcha cannot recurse.
            return True
        if captcha_gate.try_begin():
            try:
                result = resolve_captcha(session, captcha_error)
            finally:
                captcha_gate.finish()
            # Retry the original request once after verification.  For
            # "skipped"/"cancelled" the original 403 is returned as-is.
            return result == "resolved"
        if captcha_gate.acquire():
            # Another thread resolved the captcha while we waited at the gate:
            # our own in-flight request is already a stale 403, so re-send it
            # exactly once.
            return True
        # On gate timeout, return the original 403 so the caller can decide.
        return False


CAPTCHA_CHALLENGE = CaptchaChallenge()
"""The singleton registered with the response-challenge registry."""

register(CAPTCHA_CHALLENGE)
