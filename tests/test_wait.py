"""Tests for :mod:`zhihu_cli.content.utils.wait`."""

from __future__ import annotations

import pytest

from zhihu_cli.content.utils import wait as wait_module
from zhihu_cli.content.utils.wait import generate_lognormal, wait


@pytest.fixture
def sleep_calls(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace ``time.sleep`` with a recorder and return the recorded delays."""
    calls: list[float] = []
    monkeypatch.setattr(wait_module.time, "sleep", calls.append)
    return calls


@pytest.mark.parametrize("factor", [0.0, -1.0])
def test_wait_non_positive_returns_without_sleeping(sleep_calls: list[float], factor: float) -> None:
    """A non-positive factor must not raise (``math.log(0)``) or sleep."""
    wait(factor)
    wait(factor, forced_to_wait=True)
    assert sleep_calls == []


@pytest.mark.parametrize("mean_value", [0.0, -1.0])
def test_generate_lognormal_non_positive_returns_zero(mean_value: float) -> None:
    """Non-positive means are outside the log-normal domain and yield ``0.0``."""
    assert generate_lognormal(mean_value) == 0.0


def test_wait_positive_factor_sleeps_non_negative(sleep_calls: list[float]) -> None:
    """A positive factor still reaches ``time.sleep`` with a non-negative delay."""
    wait(2.0, forced_to_wait=True)
    assert len(sleep_calls) == 1
    assert sleep_calls[0] >= 0.0
