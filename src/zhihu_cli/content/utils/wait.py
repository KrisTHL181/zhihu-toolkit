"""Unified sleep utility.

All delay/sleep calls in the project should go through this module so that
throttling behaviour can be adjusted in one place.
"""

from __future__ import annotations

import math
import random
import time


def generate_lognormal(mean_value: float, sigma=1.0):
    """Generate a random number from a log-normal distribution with a specified mean (k) and standard deviation (sigma).

    :param mean_value: Desired mean of the distribution.  Non-positive values
        have no log-normal interpretation and yield ``0.0`` (no delay).
    :param sigma: Standard deviation parameter of the underlying normal
        distribution.
    :returns: A random non-negative float, or ``0.0`` when ``mean_value <= 0``.
    """
    if mean_value <= 0:
        return 0.0
    mu = math.log(mean_value) - (sigma**2) / 2
    return random.lognormvariate(mu, sigma)


def wait(sleep_factor: float, forced_to_wait: bool = False) -> None:
    """Sleep for ``sleep_factor`` seconds.

    :param sleep_factor: Multiplied by 1.0 to get the actual delay in seconds.
        A non-positive value means "do not delay" and returns immediately.
    :param forced_to_wait: If True, force the wait regardless of throttling
        overrides.
    """
    if sleep_factor <= 0:
        # ``math.log`` is undefined at 0, and a negative factor is meaningless;
        # callers use 0.0 to disable throttling (e.g. ``stream_handler(delay=0)``).
        return

    delay = 1.0

    if not forced_to_wait:
        from zhihu_cli.content.handlers.cache_manager import cache_manager

        delay = 0.0 if cache_manager.get_active_profile() is not None else 1.0  # Don't delay if logged in

    time.sleep(delay * min(generate_lognormal(sleep_factor, 0.5), 3 * sleep_factor))
