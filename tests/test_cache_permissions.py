"""POSIX permission tests for the credential cache.

Profile files hold full login cookies, so they must not be readable by other
local users.  Windows uses ACLs rather than the POSIX mode bits, so these tests
are skipped there.
"""

from __future__ import annotations

import os
import stat
from typing import TYPE_CHECKING

import pytest

from zhihu_cli.content.handlers.cache_manager import CacheManager

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits")


def _mode(path: Path) -> int:
    """Return the POSIX permission bits of *path*."""
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture
def cache(tmp_path: Path) -> Iterator[CacheManager]:
    """Give a test a private manager, restoring the process-wide singleton after.

    :param tmp_path: pytest's per-test temporary directory.
    :returns: A ``CacheManager`` rooted at ``<tmp_path>/cache``.
    """
    previous = CacheManager._instance
    CacheManager._instance = None
    try:
        yield CacheManager(cache_dir=tmp_path / "cache")
    finally:
        CacheManager._instance = previous


def test_credential_directories_are_owner_only(cache: CacheManager) -> None:
    assert _mode(cache.cache_dir) == 0o700
    assert _mode(cache.profiles_dir) == 0o700


def test_saved_profile_is_owner_only(cache: CacheManager) -> None:
    cache.save_headers({"Cookie": "SESSION=secret"}, profile_name="alice")

    assert _mode(cache.profiles_dir / "alice.json") == 0o600
    assert _mode(cache.active_profile_file) == 0o600


def test_initialization_tightens_permissions_from_older_versions(tmp_path: Path) -> None:
    previous = CacheManager._instance
    CacheManager._instance = None
    try:
        first = CacheManager(cache_dir=tmp_path / "cache")
        first.save_headers({"Cookie": "SESSION=secret"}, profile_name="alice")
        profile = first.profiles_dir / "alice.json"
        os.chmod(first.cache_dir, 0o755)
        os.chmod(first.profiles_dir, 0o755)
        os.chmod(profile, 0o644)

        CacheManager._instance = None
        second = CacheManager(cache_dir=tmp_path / "cache")

        assert _mode(second.cache_dir) == 0o700
        assert _mode(second.profiles_dir) == 0o700
        assert _mode(profile) == 0o600
    finally:
        CacheManager._instance = previous


def test_atomic_write_does_not_follow_a_planted_temp_symlink(cache: CacheManager, tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("untouched", encoding="utf-8")
    # The old implementation wrote to a fixed "<name>.tmp", so a symlink planted
    # there redirected the credential write outside the cache directory.
    (cache.profiles_dir / "bob.tmp").symlink_to(victim)

    cache.save_headers({"Cookie": "SESSION=secret"}, profile_name="bob")

    assert victim.read_text(encoding="utf-8") == "untouched"
    profile = cache.profiles_dir / "bob.json"
    assert profile.is_file() and not profile.is_symlink()
