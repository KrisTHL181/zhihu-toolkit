"""Tests for the response-challenge registry and the captcha handler."""

from __future__ import annotations

import subprocess
import sys
import types
from collections.abc import Callable

import pytest

from zhihu_cli.content.handlers import captcha, challenges
from zhihu_cli.content.handlers.challenges import captcha as challenge_captcha
from zhihu_cli.content.handlers.challenges.captcha import CaptchaChallenge

# ── registry ──────────────────────────────────────────────────────────────


class _FakeHandler:
    """Records the responses it is shown and returns canned results."""

    def __init__(
        self,
        name: str,
        detect_result: object = None,
        solve_result: bool = False,
        detect_fn: Callable[[object], object] | None = None,
    ) -> None:
        self.name = name
        self.detect_result = detect_result
        self.solve_result = solve_result
        self.detect_fn = detect_fn
        self.detect_calls: list[object] = []
        self.solve_calls: list[object] = []

    def detect(self, response: object, ctx: object) -> object:
        self.detect_calls.append(response)
        if self.detect_fn is not None:
            return self.detect_fn(response)
        return self.detect_result

    def solve(self, value: object, ctx: object) -> bool:
        self.solve_calls.append(value)
        return self.solve_result


def _ctx(retry: Callable[[object], object], *, kwargs: dict | None = None) -> challenges.ChallengeContext:
    """Build a ``ChallengeContext`` with a stand-in session."""
    return challenges.ChallengeContext(
        session=types.SimpleNamespace(captcha_handler="auto"),
        method="GET",
        url="https://www.zhihu.com/api/v4/me",
        kwargs={} if kwargs is None else kwargs,
        epoch_at_send=0,
        retry=retry,
    )


def _patch_registry(monkeypatch: pytest.MonkeyPatch, *handlers: _FakeHandler) -> None:
    monkeypatch.setattr(challenges, "registered", lambda: tuple(handlers))


def test_skip_set_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    first = _FakeHandler("first", detect_result="a", solve_result=True)
    second = _FakeHandler("second", detect_result="b", solve_result=True)
    _patch_registry(monkeypatch, first, second)

    response = object()
    retried = object()
    retries: list[int] = []

    def retry(ctx: object) -> object:
        retries.append(1)
        return retried

    result = challenges.run_challenges(response, _ctx(retry), skip={"first"})

    assert first.detect_calls == []
    assert first.solve_calls == []
    assert second.detect_calls == [response]
    assert retries == [1]
    assert result is retried


def test_detect_returning_none_leaves_response_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _FakeHandler("noop", detect_result=None, solve_result=True)
    _patch_registry(monkeypatch, handler)

    response = object()
    context = _ctx(lambda ctx: pytest.fail("retry must not be called"))

    assert challenges.run_challenges(response, context) is response
    assert handler.solve_calls == []


def test_solve_true_retries_once_and_next_handler_sees_new_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _FakeHandler("first", detect_result="challenge", solve_result=True)
    second = _FakeHandler("second", detect_result=None, solve_result=False)
    _patch_registry(monkeypatch, first, second)

    original = object()
    retried = object()
    retries: list[int] = []

    def retry(ctx: object) -> object:
        retries.append(1)
        return retried

    result = challenges.run_challenges(original, _ctx(retry))

    # The first handler solves the original, the pass restarts, and the second
    # sees only the retried response.
    assert first.detect_calls == [original]
    assert second.detect_calls == [retried]
    assert retries == [1]
    assert result is retried


def test_solve_false_does_not_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _FakeHandler("unsolved", detect_result="challenge", solve_result=False)
    _patch_registry(monkeypatch, handler)

    response = object()

    assert challenges.run_challenges(response, _ctx(lambda ctx: pytest.fail("retry must not happen"))) is response
    assert handler.solve_calls == ["challenge"]


def test_retry_restarts_the_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    original = object()
    after_first = object()
    after_second = object()
    retry_results = iter((after_first, after_second))
    retries: list[int] = []

    def retry(ctx: object) -> object:
        retries.append(1)
        return next(retry_results)

    first = _FakeHandler("first", detect_result="a", solve_result=True)
    # The second handler has nothing to do on the original response; it only
    # applies after the first retry produced an EO_Bot-like page.
    second = _FakeHandler(
        "second",
        solve_result=True,
        detect_fn=lambda response: "b" if response is after_first else None,
    )
    _patch_registry(monkeypatch, first, second)

    assert challenges.run_challenges(original, _ctx(retry)) is after_second
    assert len(retries) == 2
    # The first handler solved on the original, so the second is first examined
    # only after that retry — and that is where it detects its challenge.
    assert second.detect_calls == [after_first]


def test_handler_cannot_solve_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _FakeHandler("loopy", detect_result="always", solve_result=True)
    _patch_registry(monkeypatch, handler)

    response = object()
    retried = object()
    retries: list[int] = []

    def retry(ctx: object) -> object:
        retries.append(1)
        return retried

    assert challenges.run_challenges(response, _ctx(retry)) is retried
    assert len(retries) == 1
    assert handler.detect_calls == [response]
    assert len(handler.solve_calls) == 1


def test_registered_has_builtin_handlers() -> None:
    names = {handler.name for handler in challenges.registered()}
    assert {"captcha", "eo_bot"} <= names


def test_challenge_package_imports_without_curl_cffi() -> None:
    code = (
        "import sys; "
        "from zhihu_cli.content.handlers.challenges import registered; "
        "names = {h.name for h in registered()}; "
        "assert {'captcha', 'eo_bot'} <= names, names; "
        "assert 'curl_cffi' not in sys.modules, 'curl_cffi leaked'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ── captcha handler ───────────────────────────────────────────────────────


class _Response:
    """A captcha error envelope."""

    def __init__(self, payload: dict) -> None:
        self.status_code = 403
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _ExplodingResponse:
    status_code = 403

    def json(self) -> dict:
        raise AssertionError("body must not be parsed")


def _captcha_error(**overrides: object) -> dict:
    error = {
        "code": 40352,
        "message": "blocked",
        "redirect": "https://www.zhihu.com/account/unhuman",
        "need_login": False,
    }
    error.update(overrides)
    return error


def _captcha_ctx(
    session: object,
    *,
    url: str = "https://www.zhihu.com/api/v4/me",
    epoch_at_send: int | None = None,
) -> challenges.ChallengeContext:
    return challenges.ChallengeContext(
        session=session,
        method="GET",
        url=url,
        kwargs={},
        epoch_at_send=challenge_captcha.captcha_gate.epoch if epoch_at_send is None else epoch_at_send,
        retry=lambda ctx: pytest.fail("retry must not be called by solve"),
    )


@pytest.fixture(autouse=True)
def _reset_non_interactive_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the warn-once flag (owned by handlers.captcha) from leaking."""
    monkeypatch.setattr(captcha, "_non_interactive_captcha_warned", False)


def test_detect_ignores_when_captcha_handler_is_ignore() -> None:
    context = _captcha_ctx(types.SimpleNamespace(captcha_handler="ignore"))
    assert CaptchaChallenge().detect(_ExplodingResponse(), context) is None


def test_detect_ignores_captcha_endpoints() -> None:
    context = _captcha_ctx(
        types.SimpleNamespace(captcha_handler="auto"),
        url="https://www.zhihu.com/account/unhuman?type=x",
    )
    assert CaptchaChallenge().detect(_ExplodingResponse(), context) is None


def test_detect_returns_error_envelope() -> None:
    context = _captcha_ctx(types.SimpleNamespace(captcha_handler="auto"))
    response = _Response({"error": {"code": 40352, "message": "blocked"}})
    result = CaptchaChallenge().detect(response, context)
    assert result is not None
    assert result["code"] == 40352


def test_need_login_is_not_solved() -> None:
    context = _captcha_ctx(types.SimpleNamespace(captcha_handler="auto"))
    assert CaptchaChallenge().solve(_captcha_error(need_login=True), context) is False


def test_prompt_mode_only_warns() -> None:
    context = _captcha_ctx(types.SimpleNamespace(captcha_handler="prompt"))
    assert CaptchaChallenge().solve(_captcha_error(), context) is False


def test_auto_without_terminal_gives_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(challenge_captcha, "_can_prompt_user", lambda: False)
    context = _captcha_ctx(types.SimpleNamespace(captcha_handler="auto"))
    assert CaptchaChallenge().solve(_captcha_error(), context) is False


def test_auto_with_stale_epoch_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(challenge_captcha, "_can_prompt_user", lambda: True)
    context = _captcha_ctx(
        types.SimpleNamespace(captcha_handler="auto"),
        epoch_at_send=challenge_captcha.captcha_gate.epoch + 1,
    )
    assert CaptchaChallenge().solve(_captcha_error(), context) is True


def test_auto_resolved_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(challenge_captcha, "_can_prompt_user", lambda: True)
    monkeypatch.setattr(type(challenge_captcha.captcha_gate), "try_begin", lambda self: True)
    monkeypatch.setattr(type(challenge_captcha.captcha_gate), "finish", lambda self: None)
    monkeypatch.setattr(challenge_captcha, "resolve_captcha", lambda session, error: "resolved")

    context = _captcha_ctx(types.SimpleNamespace(captcha_handler="auto"))
    assert CaptchaChallenge().solve(_captcha_error(), context) is True


def test_auto_skipped_does_not_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(challenge_captcha, "_can_prompt_user", lambda: True)
    monkeypatch.setattr(type(challenge_captcha.captcha_gate), "try_begin", lambda self: True)
    monkeypatch.setattr(type(challenge_captcha.captcha_gate), "finish", lambda self: None)
    monkeypatch.setattr(challenge_captcha, "resolve_captcha", lambda session, error: "skipped")

    context = _captcha_ctx(types.SimpleNamespace(captcha_handler="auto"))
    assert CaptchaChallenge().solve(_captcha_error(), context) is False


# ── daemon proxy forwarding ───────────────────────────────────────────────


def test_proxy_disabled_challenges_property(monkeypatch: pytest.MonkeyPatch) -> None:
    from zhihu_cli.daemon.proxy import DaemonProxySession

    sent: list[tuple[str, object]] = []
    proxy = DaemonProxySession()
    monkeypatch.setattr(proxy, "_send_config", lambda key, value: sent.append((key, value)))
    proxy._direct = types.SimpleNamespace(disabled_challenges=set())

    proxy.disabled_challenges = {"eo_bot", "captcha"}

    assert sent == [("disabled_challenges", ["captcha", "eo_bot"])]
    assert proxy.disabled_challenges == {"captcha", "eo_bot"}
    assert proxy._direct.disabled_challenges == {"captcha", "eo_bot"}

    # The getter is a snapshot: mutating it must not touch the backing set.
    snapshot = proxy.disabled_challenges
    snapshot.add("other")
    assert proxy.disabled_challenges == {"captcha", "eo_bot"}


def test_proxy_disabled_challenges_rejects_bare_string(monkeypatch: pytest.MonkeyPatch) -> None:
    from zhihu_cli.daemon.proxy import DaemonProxySession

    proxy = DaemonProxySession()
    monkeypatch.setattr(proxy, "_send_config", lambda key, value: None)
    proxy.disabled_challenges = {"eo_bot"}

    with pytest.raises(TypeError):
        proxy.disabled_challenges = "captcha"

    assert proxy.disabled_challenges == {"eo_bot"}
