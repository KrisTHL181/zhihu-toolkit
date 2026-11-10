"""Tests for API refusal codes and the top-level gate / HTTP error rendering."""

from __future__ import annotations

import json
from typing import Any

import pytest
from click.testing import CliRunner

from zhihu_cli.content.handlers import waterfall
from zhihu_cli.content.handlers._session_core import _api_refusal_gate
from zhihu_cli.content.handlers.risk_control import ZhihuGateError
from zhihu_cli.main import _ZhihuGroup


class _FakeResponse:
    """Minimal response stand-in for the gate helpers."""

    def __init__(
        self,
        status_code: int = 200,
        *,
        text: str = "",
        payload: Any = None,
        content_type: str | None = "application/json",
    ) -> None:
        self.status_code = status_code
        self.text = text
        self._payload = payload
        self.headers = {"content-type": content_type} if content_type else {}

    def json(self) -> Any:
        """Return the canned payload, or raise like a non-JSON body."""
        if self._payload is None:
            raise ValueError("not json")
        return self._payload

    def raise_for_status(self) -> None:
        """Fail the test if a gate was not detected before this call."""
        raise AssertionError("raise_for_status should not run after a gate")


def test_api_refusal_shape_becomes_gate() -> None:
    """Any 403 JSON error envelope is a gate, whatever its code."""
    resp = _FakeResponse(403, payload={"error": {"code": 99999, "message": "请求参数异常"}})

    gate = _api_refusal_gate(resp, "https://api.zhihu.com/questions/1")

    assert isinstance(gate, ZhihuGateError)
    assert gate.gate == "risk_control"
    assert "请求参数异常" in str(gate)


def test_api_refusal_without_message_is_still_a_gate() -> None:
    """The envelope shape, not the message, identifies a refusal."""
    resp = _FakeResponse(403, payload={"error": {"code": 99999}})

    gate = _api_refusal_gate(resp, "https://api.zhihu.com/questions/1")

    assert gate is not None
    assert gate.detail is None


def test_api_refusal_ignores_captcha_and_login_envelopes() -> None:
    """Captcha and login gates stay with the challenge registry."""
    captcha = _FakeResponse(403, payload={"error": {"code": 40352, "message": "验证码"}})
    login = _FakeResponse(
        403,
        payload={
            "error": {
                "code": 40353,
                "message": "请登录",
                "need_login": True,
                "redirect": "https://www.zhihu.com/account/unhuman?need_login=true",
            }
        },
    )

    assert _api_refusal_gate(captcha, "https://www.zhihu.com/api/v4/me") is None
    assert _api_refusal_gate(login, "https://www.zhihu.com/api/v4/me") is None


def test_api_refusal_ignores_non_json_and_non_403() -> None:
    """Only 403 responses with a JSON content type are inspected."""
    html = _FakeResponse(403, text="<meta id='zh-zse-ck'>", content_type="text/html")
    ok = _FakeResponse(200, payload={"error": {"code": 10003}})

    assert _api_refusal_gate(html, "https://www.zhihu.com/question/1") is None
    assert _api_refusal_gate(ok, "https://www.zhihu.com/question/1") is None


class _GateSession:
    """Session that always returns the same gated response."""

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        """Return the canned gated response."""
        return self._response


def _run_stream(monkeypatch: pytest.MonkeyPatch, response: _FakeResponse) -> list[Any]:
    """Drive ``stream_handler`` against *response* and return its items."""
    monkeypatch.setattr(waterfall, "session", _GateSession(response))
    monkeypatch.setattr(waterfall, "wait", lambda *args, **kwargs: None)
    stream = waterfall.stream_handler(
        "https://api.zhihu.com/topic/1/followers?offset=0",
        lambda data: data.get("data", []),
        delay=1.0,
    )
    return list(stream)


def test_waterfall_raises_zse_ck_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``__zse_ck`` HTML response stops the stream with a gate error."""
    response = _FakeResponse(
        403,
        text='<meta id="zh-zse-ck" content="x">',
        content_type="text/html",
    )

    with pytest.raises(ZhihuGateError) as excinfo:
        _run_stream(monkeypatch, response)

    assert excinfo.value.gate == "zse_ck"


def test_waterfall_raises_risk_control_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A JSON risk-control refusal stops the stream with a gate error."""
    payload = {"error": {"code": 40362, "message": "请求存在异常"}}
    response = _FakeResponse(403, text=json.dumps(payload), payload=payload)

    with pytest.raises(ZhihuGateError) as excinfo:
        _run_stream(monkeypatch, response)

    assert excinfo.value.gate == "risk_control"


def test_top_level_renders_http_error_without_traceback() -> None:
    """An uncaught ``HTTPError`` becomes one clean ``Error:`` line."""
    group = _ZhihuGroup("zhihu")

    @group.command("boom")
    def boom() -> None:
        """Raise the kind of error a handler produces on an unhandled 403."""
        from curl_cffi.requests.exceptions import HTTPError

        raise HTTPError("HTTP Error 403: boom")

    result = CliRunner().invoke(group, ["boom"])
    output = result.output + getattr(result, "stderr", "")

    assert result.exit_code == 1
    assert "Error: HTTP Error 403: boom" in output
    assert "Traceback" not in output
