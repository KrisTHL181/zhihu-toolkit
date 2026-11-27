"""Contract tests for the supported library surface in :mod:`zhihu_cli.api`.

These tests never touch the network: the wrappers are lazy, so patching
``_resolve`` or ``_call`` is enough to exercise the parts that matter — that
every exported name exists, that each wrapper names a real handler, and that
handler failures become the documented exception types.
"""

from __future__ import annotations

import importlib
import inspect
from types import SimpleNamespace
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

import zhihu_cli
import zhihu_cli.api as api
from zhihu_cli.content.handlers.risk_control import ZhihuGateError
from zhihu_cli.errors import AuthError, NotFoundError, RateLimitError, ZhihuError, ZhihuHTTPError

if TYPE_CHECKING:
    from collections.abc import Callable


def _data_wrappers() -> list[tuple[str, Callable[..., object]]]:
    """Return the ``*args, **kwargs`` wrappers defined by :mod:`zhihu_cli.api`.

    The context managers and pacing helpers take named parameters, so the
    varargs signature is what distinguishes a handler wrapper.
    """
    wrappers: list[tuple[str, Callable[..., object]]] = []
    for name in api.__all__:
        obj = getattr(api, name)
        if not inspect.isfunction(obj) or obj.__module__ != api.__name__:
            continue
        kinds = {param.kind for param in inspect.signature(obj).parameters.values()}
        if inspect.Parameter.VAR_POSITIONAL in kinds and inspect.Parameter.VAR_KEYWORD in kinds:
            wrappers.append((name, obj))
    return wrappers


def _http_error(status_code: int) -> Exception:
    """Build an exception that :func:`api._normalize` treats as curl_cffi's."""
    cls = type("HTTPError", (Exception,), {})
    cls.__module__ = "curl_cffi.requests"
    exc = cls(f"HTTP {status_code}")
    exc.response = SimpleNamespace(status_code=status_code)
    return exc


def test_every_exported_name_exists() -> None:
    missing = [name for name in api.__all__ if not hasattr(api, name)]
    assert not missing


def test_all_has_no_duplicates() -> None:
    assert len(api.__all__) == len(set(api.__all__))


def test_every_public_function_is_exported() -> None:
    defined = {
        name
        for name, obj in vars(api).items()
        if not name.startswith("_") and inspect.isfunction(obj) and obj.__module__ == api.__name__
    }
    assert defined <= set(api.__all__)


def test_every_data_wrapper_names_a_real_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_call(module_name: str, func_name: str, *args: object, **kwargs: object) -> list[object]:
        calls.append((module_name, func_name))
        return []

    monkeypatch.setattr(api, "_call", fake_call)

    wrappers = _data_wrappers()
    assert wrappers
    for _, wrapper in wrappers:
        wrapper()
    assert len(calls) == len(wrappers)

    for module_name, func_name in calls:
        module = importlib.import_module(f"zhihu_cli.content.handlers.{module_name}")
        assert callable(getattr(module, func_name, None)), f"{module_name}.{func_name} is missing"


def test_package_reexports_every_api_name() -> None:
    missing = [name for name in api.__all__ if not hasattr(zhihu_cli, name)]
    assert not missing


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        pytest.param(ValueError("missing payload"), NotFoundError, id="value-error"),
        pytest.param(PermissionError("denied"), AuthError, id="permission-error"),
        pytest.param(RuntimeError("risk control: 40352"), RateLimitError, id="risk-control-runtime"),
        pytest.param(RuntimeError("plain failure"), ZhihuError, id="plain-runtime"),
        pytest.param(ZhihuGateError("signin", "https://www.zhihu.com/signin"), AuthError, id="signin-gate"),
        pytest.param(ZhihuGateError("zse_ck", "https://www.zhihu.com/hot"), RateLimitError, id="zse-ck-gate"),
    ],
)
def test_handler_errors_are_normalized(
    monkeypatch: pytest.MonkeyPatch,
    exc: Exception,
    expected: type[ZhihuError],
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise exc

    monkeypatch.setattr(api, "_resolve", lambda module_name, func_name: boom)

    with pytest.raises(expected) as caught:
        api.fetch_hot_list()

    assert type(caught.value) is expected
    assert caught.value.__cause__ is exc


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        pytest.param(403, RateLimitError, id="forbidden"),
        pytest.param(404, NotFoundError, id="not-found"),
        pytest.param(500, ZhihuHTTPError, id="server-error"),
    ],
)
def test_http_errors_are_normalized(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    expected: type[ZhihuError],
) -> None:
    exc = _http_error(status_code)

    def boom(*args: object, **kwargs: object) -> None:
        raise exc

    monkeypatch.setattr(api, "_resolve", lambda module_name, func_name: boom)

    with pytest.raises(expected) as caught:
        api.fetch_hot_list()

    assert type(caught.value) is expected
    assert caught.value.__cause__ is exc


class _BrokenStream:
    """A lazy iterable whose first request fails, like a real ``PageStream``."""

    totals = 7

    def __iter__(self):
        raise ValueError("gone")
        yield  # pragma: no cover - makes this a generator


def test_stream_errors_are_normalized_during_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "_resolve", lambda module_name, func_name: lambda *args, **kwargs: _BrokenStream())

    stream = api.stream_answers()

    # Pagination metadata is forwarded through the proxy.
    assert stream.totals == 7
    with pytest.raises(NotFoundError) as caught:
        list(stream)
    assert isinstance(caught.value.__cause__, ValueError)


def test_missing_profile_is_reported_as_auth_error() -> None:
    name = f"missing-{uuid4().hex}"

    with pytest.raises(AuthError):
        with api.use_profile(name):
            pass


def test_rate_limit_context_restores_the_previous_value() -> None:
    previous = api.get_rate_limit()

    with api.rate_limit(2.5):
        assert api.get_rate_limit() == 2.5

    assert api.get_rate_limit() == previous
