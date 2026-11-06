"""Tests for the EO_Bot challenge parser, cookie installer, and handler."""

from __future__ import annotations

import types
from http.cookiejar import CookieJar
from pathlib import Path

from zhihu_cli.content.handlers.challenges.eo_bot import (
    EoBotChallenge,
    EoBotChallengeHandler,
    detect_eo_bot,
    install_eo_bot_cookies,
)

FIXTURE = Path(__file__).parent / "fixtures" / "eo_bot_challenge.html"


class _Response:
    """Minimal response object accepted by ``detect_eo_bot``."""

    def __init__(
        self,
        status_code: int = 200,
        content_type: str = "text/html",
        body: bytes = b"",
        url: str = "https://www.zhihu.com/api/v4/me",
    ) -> None:
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        self.content = body
        self.url = url


class _NoContentResponse:
    """Response whose body must never be read (used for streamed requests)."""

    status_code = 200
    url = "https://www.zhihu.com/api/v4/me"
    headers = {"content-type": "text/html"}

    @property
    def content(self) -> bytes:
        raise AssertionError("streamed responses must not have their body read")


class _Session:
    """Minimal session object accepted by ``install_eo_bot_cookies``."""

    def __init__(self, cookie_header: str = "z_c0=abc; d_c0=def") -> None:
        self.headers = {"Cookie": cookie_header} if cookie_header else {}
        self.cookies = type("_Cookies", (), {"jar": CookieJar()})()


def _ctx(**kwargs: object) -> types.SimpleNamespace:
    """Build the subset of ``ChallengeContext`` that ``detect_eo_bot`` reads."""
    return types.SimpleNamespace(kwargs=kwargs)


def test_detects_and_parses_challenge() -> None:
    info = detect_eo_bot(_Response(body=FIXTURE.read_bytes()), _ctx())
    assert isinstance(info, EoBotChallenge)
    assert info.tst_status == "749332405#"
    assert info.eo_bot_ssid.isdigit()


def test_ignores_json() -> None:
    assert detect_eo_bot(_Response(body=b'{"ok": true}', content_type="application/json"), _ctx()) is None


def test_ignores_non_200() -> None:
    assert detect_eo_bot(_Response(status_code=403, body=FIXTURE.read_bytes()), _ctx()) is None


def test_ignores_non_zhihu_host() -> None:
    assert detect_eo_bot(_Response(body=FIXTURE.read_bytes(), url="https://example.com/"), _ctx()) is None


def test_unparsable_challenge_returns_none() -> None:
    assert detect_eo_bot(_Response(body=b"<script>EO_Bot</script>"), _ctx()) is None


def test_streamed_response_is_skipped_without_touching_body() -> None:
    assert detect_eo_bot(_NoContentResponse(), _ctx(stream=True)) is None


def test_install_seeds_profile_cookies_and_sets_challenge() -> None:
    session = _Session()
    challenge = EoBotChallenge(tst_status="749332405#", eo_bot_ssid="3552051200")

    install_eo_bot_cookies(session, challenge)

    values = {cookie.name: cookie.value for cookie in session.cookies.jar}
    assert values["z_c0"] == "abc"
    assert values["d_c0"] == "def"
    assert values["__tst_status"] == "749332405#"
    assert values["EO_Bot_Ssid"] == "3552051200"


def test_handler_solves_by_installing_cookies() -> None:
    session = _Session()
    handler = EoBotChallengeHandler()

    assert handler.name == "eo_bot"
    assert handler.solve(EoBotChallenge("1#", "2"), types.SimpleNamespace(session=session)) is True

    values = {cookie.name: cookie.value for cookie in session.cookies.jar}
    assert values["__tst_status"] == "1#"
    assert values["EO_Bot_Ssid"] == "2"
