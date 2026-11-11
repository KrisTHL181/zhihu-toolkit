"""zhihu CLI — unified entry point for all Zhihu operations."""

from __future__ import annotations

import click

from zhihu_cli.commands.agora import register_agora
from zhihu_cli.commands.ai import register_ai
from zhihu_cli.commands.auth import register_auth
from zhihu_cli.commands.browse import register_browse
from zhihu_cli.commands.chat import register_chat
from zhihu_cli.commands.config import register_config
from zhihu_cli.commands.consult import register_consult
from zhihu_cli.commands.daemon import register_daemon
from zhihu_cli.commands.daily import register_daily
from zhihu_cli.commands.download import register_download
from zhihu_cli.commands.draft import register_draft
from zhihu_cli.commands.interact import register_interact
from zhihu_cli.commands.listen import register_listen
from zhihu_cli.commands.people import register_people
from zhihu_cli.commands.profile import register_profile
from zhihu_cli.commands.publish import register_publish
from zhihu_cli.commands.scrape import register_scrape
from zhihu_cli.commands.search import register_search
from zhihu_cli.commands.stats import register_stats
from zhihu_cli.commands.tools import register_tools
from zhihu_cli.content.handlers.risk_control import ZhihuGateError
from zhihu_cli.extensions import discover_extensions


def _as_http_error(exc: BaseException) -> BaseException | None:
    """Return *exc* when it is a ``curl_cffi`` HTTP error, else ``None``.

    ``curl_cffi`` is imported lazily: command modules already import it on the
    request path, but ``zhihu --help`` and shell completion must not pay for
    it.  An exception is the only time this helper runs, so the import cost is
    irrelevant here.

    :param exc: The exception raised by a command.
    :returns: *exc* when it is a :class:`curl_cffi.requests.exceptions.HTTPError`,
        otherwise ``None`` so the caller can re-raise it unchanged.
    """
    try:
        from curl_cffi.requests.exceptions import HTTPError
    except Exception:  # noqa: BLE001 - curl_cffi is a hard dependency in practice
        return None
    return exc if isinstance(exc, HTTPError) else None


class _ZhihuGroup(click.Group):
    """Click group that renders anonymous-access gates as normal CLI errors.

    Handlers raise :class:`~zhihu_cli.content.handlers.risk_control.ZhihuGateError`
    when Zhihu answers with the ``__zse_ck`` challenge, the account-verification
    page, or a sign-in redirect.  Without this translation Click would print a
    traceback; the gate message already says how to authenticate.

    Uncaught ``curl_cffi`` HTTP errors (a 403 that no gate handler recognised)
    are rendered the same way: one ``Error:`` line with the server's wording,
    which is already folded into the exception by ``fetch_json``.
    """

    def invoke(self, ctx: click.Context) -> object:
        """Invoke the selected command, translating access-gate failures.

        :param ctx: Click context for the selected subcommand.
        :returns: Whatever the command returns.
        :raises click.ClickException: For a gate; Click prints its message and
            exits with status 1.
        """
        try:
            return super().invoke(ctx)
        except ZhihuGateError as exc:
            raise click.ClickException(str(exc)) from None
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is an HTTP error
            http_error = _as_http_error(exc)
            if http_error is None:
                raise
            raise click.ClickException(str(http_error).strip()) from None


@click.group(cls=_ZhihuGroup)
@click.version_option(package_name="zhihu-toolkit", prog_name="zhihu")
@click.option(
    "--qps",
    type=float,
    default=None,
    envvar="ZHIHU_QPS",
    help="Global request rate limit (requests/second; 0 disables).",
)
def main(qps: float | None) -> None:
    """zhihu — Zhihu scraping, automation, and analysis toolkit.

    Authenticate once with \033[1mzhihu auth paste\033[0m, then use any command.
    """
    # Per-invocation limit only: this configures the client process and is
    # never pushed to the daemon (that would make a supposedly local option
    # rewrite the shared account-wide setting).  ``zhihu config qps set`` is
    # the persistent, daemon-enforced counterpart.  The import stays inside
    # the branch so ``--help`` and shell completion do not pay for it.
    if qps is not None:
        from zhihu_cli.content.utils import rate_limit

        rate_limit.set_explicit_qps(qps)


# ── register built-in command groups ───────────────────────────────────────

register_agora(main)
register_ai(main)
register_auth(main)
register_browse(main)
register_chat(main)
register_config(main)
register_consult(main)
register_daemon(main)
register_daily(main)
register_draft(main)
register_download(main)
register_interact(main)
register_listen(main)
register_people(main)
register_profile(main)
register_publish(main)
register_scrape(main)
register_search(main)
register_stats(main)
register_tools(main)

# ── extensions ─────────────────────────────────────────────────────────────

# Auto-discover and register extension command groups.
for _ext_mod in discover_extensions():
    _ext_mod.register_cli(main)

if __name__ == "__main__":
    main()
