"""Tools command group — creator analytics, NLP text analysis, and periodic looping."""

from __future__ import annotations

import codecs
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

from zhihu_cli.content.handlers import get_data_dir
from zhihu_cli.content.handlers.cache_manager import cache_manager
from zhihu_cli.output import (
    echo,
    error,
    f_dim,
    f_label,
    f_num,
    f_path,
    heading,
    info,
    print_json,
    set_json_mode,
    success,
    warning,
)

#: How many leading bytes to inspect before printing a streamed body.
#: NUL in this window is the same binary heuristic git and grep use; a
#: payload that passes it is treated as text and streamed through.
_BINARY_PEEK_BYTES = 8192


def _peek_body(resp: Any, limit: int = _BINARY_PEEK_BYTES) -> tuple[bytes, Iterator[bytes]]:
    """Buffer the first *limit* bytes of a streaming response.

    Reading stops as soon as the buffer reaches *limit* (one chunk may push
    it slightly over), leaving the rest of the body unread.  Callers abort a
    binary payload at this point, so only a few KiB are ever downloaded.

    :param resp: A streaming response exposing ``iter_content()``.
    :param limit: Number of leading bytes to buffer.
    :returns: ``(head, chunks)`` — the buffered prefix and the iterator that
        yields the remainder of the body.
    """
    chunks = resp.iter_content()
    head = bytearray()
    for chunk in chunks:
        if not chunk:
            continue
        head += chunk
        if len(head) >= limit:
            break
    return bytes(head), chunks


def _stream_text(resp: Any, head: bytes, chunks: Iterator[bytes]) -> None:
    """Print a streamed body as text, reusing bytes already read into *head*.

    Decoding is incremental so a multi-byte character split across a chunk
    boundary is not mangled; ``errors="replace"`` matches how
    :attr:`curl_cffi.requests.Response.text` decodes.

    :param resp: The streaming response (only ``encoding`` is read).
    :param head: Leading bytes already consumed by :func:`_peek_body`.
    :param chunks: Iterator over the remaining body bytes.
    """
    encoding = getattr(resp, "encoding", None) or "utf-8"
    try:
        decoder = codecs.getincrementaldecoder(encoding)(errors="replace")
    except LookupError:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    click.echo(decoder.decode(head), nl=False)
    for chunk in chunks:
        click.echo(decoder.decode(chunk), nl=False)
    click.echo(decoder.decode(b"", final=True), nl=False)
    click.echo()  # echo() used to append this newline to the whole body


def register_tools(main_group):
    """Register the tools command group and all sub-commands onto *main_group*."""

    def _parse_period(period: str) -> int:
        """Parse a period string like '60', '5m', '1h', '2d' into seconds."""
        period = period.strip().lower()
        m = re.match(r"^(\d+)([smhd]?)$", period)
        if not m:
            raise click.BadParameter(
                f"Invalid period {period!r}. Use an integer (seconds) or suffix s/m/h/d, e.g. '60', '5m', '1h', '2d'."
            )
        value = int(m.group(1))
        if value <= 0:
            raise click.BadParameter(f"Invalid period {period!r} — must be greater than zero.")
        unit = m.group(2) or "s"
        multipliers: dict[str, int] = {"s": 1, "m": 60, "h": 3600, "d": 86400}
        return value * multipliers[unit]

    def _process_loop_result(result: subprocess.CompletedProcess, target_jsonl_file: str, override: bool) -> None:
        """Parse command stdout as JSON, annotate with a timestamp, and write to the JSONL file.

        :param result: Completed subprocess result with stdout/stderr.
        :param target_jsonl_file: Path to the output JSONL file.
        :param override: If True, overwrite the file instead of appending.
        """
        if result.returncode != 0:
            error(f"Command exited with code {result.returncode}")
            if result.stderr:
                error(result.stderr.strip())
            return

        stdout = result.stdout.strip()
        if not stdout:
            warning("Command produced no output")
            return

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            error("Command output is not valid JSON — skipping")
            return

        ts = time.time()
        if isinstance(data, dict):
            data["time"] = ts
        else:
            data = {"time": ts, "data": data}
        compact = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        Path(target_jsonl_file).parent.mkdir(parents=True, exist_ok=True)
        with open(target_jsonl_file, "a" if not override else "w", encoding="utf-8") as fh:
            fh.write(compact + "\n")
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        action = "Appended" if not override else "Written"
        success(f"[{ts_str}] {action} to {target_jsonl_file}")

    @main_group.group()
    def tools() -> None:
        """Analysis tools — creator analytics, NLP text analysis, and periodic looping."""

    @tools.command("loop")
    @click.argument("command", type=str)
    @click.argument("period", type=str)
    @click.argument("target_jsonl_file", type=click.Path())
    @click.option("--once", is_flag=True, default=False, help="Run once and exit (no looping).")
    @click.option("--override", is_flag=True, default=False, help="Override existing file instead of appending.")
    def tools_loop(command: str, period: str, target_jsonl_file: str, once: bool, override: bool) -> None:
        """Run a zhihu subcommand periodically, appending compact JSON to a JSONL file.

        COMMAND   – zhihu subcommand to run (e.g. "browse hot", "stats <url>").

        PERIOD    – interval between runs: an integer (seconds) or with suffix
                    s / m / h / d, e.g. "60", "5m", "1h", "2d".  Max unit is day.

        TARGET_JSONL_FILE – path to the output JSONL file (one compact JSON
                    object per line).
        """
        period_seconds = _parse_period(period)
        cmd_parts = ["zhihu"] + command.split() + ["--json"]

        heading(f"Loop: zhihu {command} --json  →  {target_jsonl_file}")
        info(f"Period: {period} ({period_seconds}s)")

        # Deadlines are anchored to the start time and advance by exactly one
        # period per round, so a slow run never pushes later runs back.  Sleeping
        # ``period_seconds`` *after* each run would instead make the real interval
        # ``period + run_duration``, drifting further every round.  ``monotonic``
        # keeps the schedule immune to wall-clock jumps (NTP, DST, suspend).
        deadline = time.monotonic() + period_seconds

        while True:
            try:
                result = subprocess.run(cmd_parts, capture_output=True, text=True, timeout=300)
            except subprocess.TimeoutExpired:
                error("Command timed out (300 s)")
            except FileNotFoundError:
                error("'zhihu' not found on PATH — is the CLI installed?")
                raise SystemExit(1)
            else:
                _process_loop_result(result, target_jsonl_file, override)

            if once:
                break

            now = time.monotonic()
            if now > deadline:
                # The run overran its slot — skip the slots it already missed
                # rather than firing them back to back, which would hammer the API.
                skipped = int((now - deadline) // period_seconds) + 1
                deadline += skipped * period_seconds
                warning(f"Run overran its interval — skipping {skipped} interval(s)")
            time.sleep(deadline - now)
            deadline += period_seconds

    @tools.command("markdown")
    @click.argument("path", type=click.Path(exists=True))
    @click.option(
        "--override",
        is_flag=True,
        default=False,
        help="Delete original HTML file(s) after conversion.",
    )
    def tools_markdown(path: str, override: bool) -> None:
        """Convert raw HTML files to Markdown.

        PATH may be a single .html file or a directory. When PATH is a
        directory, all .html / .htm files under it are converted recursively.

        By default the generated .md file sits alongside the original HTML.
        Use --override to delete the original HTML after conversion.
        """
        from zhihu_cli.content.utils.html2markdown import PageToMarkdown

        input_path = Path(path)
        converter = PageToMarkdown()

        # ── collect files ─────────────────────────────────────────────────
        if input_path.is_file():
            html_files = [input_path]
        else:
            html_files = sorted(
                p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in (".html", ".htm")
            )

        if not html_files:
            error(f"No HTML files found in {f_path(path)}")
            raise SystemExit(1)

        # ── convert each file ─────────────────────────────────────────────
        converted = 0
        skipped = 0
        for html_file in html_files:
            if html_file.suffix.lower() not in (".html", ".htm"):
                skipped += 1
                warning(f"Skipping non-HTML file: {f_path(str(html_file))}")
                continue

            md_path = html_file.with_suffix(".md")

            # Read HTML
            try:
                html_content = html_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                warning(f"Cannot read {f_path(str(html_file))}: {exc}")
                skipped += 1
                continue

            if not html_content.strip():
                warning(f"Skipping empty file: {f_path(str(html_file))}")
                skipped += 1
                continue

            # Convert
            markdown = converter.convert(html_content)

            # Write Markdown
            md_path.write_text(markdown, encoding="utf-8")
            converted += 1

            # Delete original if --override
            if override:
                html_file.unlink()
                success(f"{f_path(str(html_file))} {f_dim('→')} {f_path(str(md_path))}  (deleted)")
            else:
                success(f"{f_path(str(html_file))} {f_dim('→')} {f_path(str(md_path))}")

        # ── summary ──────────────────────────────────────────────────────
        msg = f"Converted {f_num(converted)} file(s)"
        if skipped:
            msg += f", {f_num(skipped)} skipped"
        if override:
            msg += f" {f_dim('(--override)')}"
        success(msg)

    @tools.command("request")
    @click.argument("url", type=str)
    @click.option(
        "--method", "-X", "method", default="GET", help="HTTP method (default: GET; auto-set to POST if --data given)"
    )
    @click.option("--data", "-d", default=None, help="Request body data")
    @click.option(
        "--header", "-H", "extra_headers", multiple=True, help="Extra header (repeatable), format 'Key: Value'"
    )
    @click.option(
        "--curl", "output_curl", is_flag=True, default=False, help="Print equivalent curl command instead of sending"
    )
    @click.option(
        "--include", "-i", "include_headers", is_flag=True, default=False, help="Include response headers in output"
    )
    @click.option(
        "--output",
        "-o",
        "output_file",
        default=None,
        type=click.Path(),
        help="Write the raw response body to file instead of stdout",
    )
    def tools_request(
        url: str,
        method: str,
        data: str | None,
        extra_headers: tuple[str, ...],
        output_curl: bool,
        include_headers: bool,
        output_file: str | None,
    ) -> None:
        """Send an HTTP request or generate an equivalent curl command.

        URL is the target URL.  For zhihu.com / api.zhihu.com URLs, ZSE
        signing headers (x-zse-93, x-zse-96) are added automatically — the
        active profile's auth cookies and User-Agent are included.

        Use --curl to see exactly what would be sent, without sending it.

        When the body is printed to a terminal, only the first 8 KiB are
        inspected for a NUL byte (the git/grep binary heuristic); a binary
        payload is aborted at that point and nothing is printed.  Use
        --output to save the raw body bytes instead.

        \b
        Examples:
          zhihu tools request https://api.zhihu.com/me
          zhihu tools request https://api.zhihu.com/me --curl
          zhihu tools request https://api.zhihu.com/v4/answer -X POST -d '{"id":123}'
          zhihu tools request https://example.com/api -H "Authorization: Bearer xxx"
        """
        # ── parse extra headers ──────────────────────────────────────────
        parsed_headers: dict[str, str] = {}
        for h in extra_headers:
            if ":" not in h:
                raise click.BadParameter(f"Invalid header format: {h!r}. Use 'Key: Value'.")
            key, _, value = h.partition(":")
            parsed_headers[key.strip()] = value.strip()

        # ── auto-set method when --data is given ─────────────────────────
        if data is not None and method == "GET":
            method = "POST"

        from zhihu_cli.content.handlers.request_tool import build_curl_command, send_request

        # ── --curl mode: print curl command and exit ─────────────────────
        if output_curl:
            profile_headers = cache_manager.load_headers()
            merged = {**profile_headers, **parsed_headers}
            curl_cmd = build_curl_command(url, method=method, headers=merged, data=data)
            echo(curl_cmd)
            return

        # ── send request ─────────────────────────────────────────────────
        # Stream when the body is headed for the terminal, so a binary
        # payload can be rejected after the first 8 KiB instead of being
        # downloaded in full.  File and pipe outputs have nothing to guard.
        stream_to_tty = sys.stdout.isatty() and output_file is None
        resp = send_request(
            url,
            method=method,
            headers=parsed_headers if parsed_headers else None,
            data=data,
            stream=stream_to_tty,
        )

        # ── output ───────────────────────────────────────────────────────
        if stream_to_tty:
            head, chunks = _peek_body(resp)
            if b"\0" in head:
                resp.close()
                error("Warning: Binary output can mess up your terminal. Use '--output <FILE>' to save to a file.")
                raise click.exceptions.Exit(1)

            try:
                if include_headers:
                    echo(f"HTTP/1.1 {resp.status_code} {getattr(resp, 'reason', '')}")
                    for k, v in resp.headers.items():
                        echo(f"{k}: {v}")
                    echo("")
                _stream_text(resp, head, chunks)
            finally:
                resp.close()
            return

        if output_file:
            out_path = Path(output_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Write the raw bytes: text-decoding here would replace every
            # non-UTF-8 byte with U+FFFD, which is exactly the binary payload
            # the warning tells the user to save with this flag.
            with open(out_path, "wb") as fh:
                if include_headers:
                    lines = [f"HTTP/1.1 {resp.status_code} {getattr(resp, 'reason', '')}"]
                    lines.extend(f"{k}: {v}" for k, v in resp.headers.items())
                    fh.write(("\n".join(lines) + "\n\n").encode("utf-8"))
                fh.write(resp.content)
            success(f"Saved {f_num(resp.status_code)} → {f_path(str(out_path))}")
        elif include_headers:
            echo(f"HTTP/1.1 {resp.status_code} {getattr(resp, 'reason', '')}")
            for k, v in resp.headers.items():
                echo(f"{k}: {v}")
            echo("")
            echo(resp.text)
        else:
            echo(resp.text)

    @tools.group("creator")
    def tools_creator() -> None:
        """Zhihu creator analytics."""

    @tools_creator.group("income")
    def tools_creator_income() -> None:
        """Income analytics (收益分析)."""

    @tools_creator_income.command("fetch")
    @click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON")
    def creator_income_fetch(output_json: bool) -> None:
        """Fetch incremental income data from Zhihu creator API."""
        from zhihu_cli.creator_tools.parse_zhihu_incomes import run_task

        run_task()
        if output_json:
            filepath = get_data_dir() / "exports" / "zhihu_income_report.json"
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
            except FileNotFoundError:
                print_json({"error": "file not found", "path": str(filepath)})
            else:
                print_json(data)
            return

    @tools_creator_income.command("monthly")
    @click.option(
        "--file",
        "-f",
        "file_path",
        default=str(get_data_dir() / "exports" / "zhihu_income_report.json"),
        help="Income report JSON",
    )
    @click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON")
    def creator_monthly(file_path: str, output_json: bool) -> None:
        """Print monthly income summary table."""
        from zhihu_cli.creator_tools.analyze_monthly_income import analyze_monthly_income, get_monthly_income_data

        if output_json:
            print_json(get_monthly_income_data(file_path))
            return
        analyze_monthly_income(file_path)

    @tools_creator_income.group("plot")
    def tools_creator_income_plot() -> None:
        """Income plot subcommands — basic, advanced, derivative."""

    @tools_creator_income_plot.command("basic")
    def creator_plot_basic() -> None:
        """Generate basic income plot (bar chart + EMA + trend)."""
        from zhihu_cli.creator_tools.plot_zhihu_incomes import plot_analysis

        plot_analysis()
        success(f"Saved {f_path(str(get_data_dir() / 'plots' / 'income_analysis.png'))}")

    @tools_creator_income_plot.command("advanced")
    def creator_plot_advanced() -> None:
        """Generate advanced analysis plot (Bollinger + MACD)."""
        from zhihu_cli.creator_tools.plot_zhihu_incomes_advanced import plot_advanced_analysis

        plot_advanced_analysis()
        success(f"Saved {f_path(str(get_data_dir() / 'plots' / 'income_advanced_analysis.png'))}")

    @tools_creator_income_plot.command("derivative")
    def creator_plot_derivative() -> None:
        """Generate derivative analysis plot (velocity, acceleration, jerk)."""
        from zhihu_cli.creator_tools.derivative_analysis import plot_derivative_analysis

        plot_derivative_analysis()
        success(f"Saved {f_path(str(get_data_dir() / 'plots' / 'derivative_analysis.png'))}")

    @tools_creator_income.command("weekday")
    def creator_weekday() -> None:
        """Generate weekday income distribution plot."""
        from zhihu_cli.creator_tools.weekday_income_analysis import plot_weekday_analysis

        plot_weekday_analysis()
        success(f"Saved {f_path(str(get_data_dir() / 'plots' / 'weekday_income_analysis.png'))}")

    @tools_creator.command("metrics")
    @click.option(
        "--aggr",
        is_flag=True,
        help="Use aggregated endpoint (single datapoint per content); saved as metrics_aggr_*.json",
    )
    @click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON")
    def creator_metrics(aggr: bool, output_json: bool) -> None:
        """Fetch per-content daily metrics from Zhihu API."""
        from zhihu_cli.creator_tools.parse_content_datas import run_batch_daily_analysis

        run_batch_daily_analysis(use_aggr=aggr)
        if output_json:
            metrics_dir = get_data_dir() / "exports" / "content_metrics"
            all_data = []
            for fp in sorted(metrics_dir.glob("*.json")):
                try:
                    all_data.append(json.loads(fp.read_text(encoding="utf-8")))
                except FileNotFoundError:
                    continue
            print_json(all_data)
            return

    @tools_creator.command("growth")
    @click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON")
    def creator_growth(output_json: bool) -> None:
        """Fetch and display creator growth-level (创作分) data."""
        from zhihu_cli.creator_tools.growth_level import show_growth_level

        show_growth_level(json_output=output_json)

    @tools_creator.command("salt")
    @click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON")
    def creator_salt(output_json: bool) -> None:
        """Fetch and display Zhihu salt value (盐值) data."""
        from zhihu_cli.creator_tools.salt import show_salt

        show_salt(json_output=output_json)

    @tools_creator.command("score")
    @click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON")
    def creator_score(output_json: bool) -> None:
        """Fetch incremental creator score detail (创作分明细) from Zhihu API."""
        from zhihu_cli.creator_tools.parse_score_detail import run_task

        run_task()
        if output_json:
            filepath = get_data_dir() / "exports" / "creator_score_detail.json"
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
            except FileNotFoundError:
                print_json({"error": "file not found", "path": str(filepath)})
            else:
                print_json(data)
            return

    @tools_creator.command("plot")
    @click.option("--no-pv", is_flag=True, default=False, help="Exclude PV (page view) metric from the plot")
    @click.option("--no-show", is_flag=True, default=False, help="Exclude Show metric from the plot")
    def creator_metrics_plot(no_pv: bool, no_show: bool) -> None:
        """Plot content metrics charts from content_metrics/*.json data."""
        from zhihu_cli.creator_tools.plot_content_metrics import plot_content_metrics

        plot_content_metrics(no_pv=no_pv, no_show=no_show)

    @tools_creator.command("stats")
    @click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON")
    @click.option(
        "--only",
        type=click.Choice(["daily", "aggr", "both"]),
        default="both",
        show_default=True,
        help="Which view to show: daily time series, aggregated lifetime totals, or both",
    )
    def creator_stats(output_json: bool, only: str) -> None:
        """Compute descriptive statistics from all content_metrics/*.json files.

        Scans every metrics file in ~/.zhihu-cli/exports/content_metrics/ and
        prints comprehensive statistics: per-metric summaries, daily aggregate
        stats, per-type breakdowns, and top-performing content.

        By default both views are shown: the daily time-series report followed
        by a "Lifetime Totals" section. Daily and aggregated files coexist in
        the same directory (``metrics_full_*`` and ``metrics_aggr_*``); the
        lifetime totals prefer the aggregated files and are synthesized from
        daily files when none are present.

        Use --only daily|aggr to show a single view, and --json for
        machine-readable output.
        """
        from zhihu_cli.creator_tools.stats_content_metrics import compute_and_display_stats

        compute_and_display_stats(output_json=output_json, only=only)

    @tools_creator_income.command("income")
    @click.option("--start-date", default=None, help="Start date (YYYY-MM-DD), default: 30 days ago")
    @click.option("--end-date", default=None, help="End date (YYYY-MM-DD), default: today")
    @click.option("--order-field", default="content_publish_at", help="Sort field (default: content_publish_at)")
    @click.option("--order-sort", default="desc", help="Sort direction (default: desc)")
    @click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON")
    def creator_income(
        start_date: str | None,
        end_date: str | None,
        order_field: str,
        order_sort: str,
        output_json: bool,
    ) -> None:
        """Fetch per-content income detail (单篇内容盐粒) from Zhihu API."""
        from zhihu_cli.creator_tools.parse_income_range import run_task

        run_task(
            start_date=start_date,
            end_date=end_date,
            order_field=order_field,
            order_sort=order_sort,
            json_output=output_json,
        )

    @tools_creator.group("follower")
    def tools_creator_follower() -> None:
        """Follower analytics (关注者分析)."""

    @tools_creator_follower.command("fetch")
    @click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON")
    def creator_follower_fetch(output_json: bool) -> None:
        """Fetch follower detail data from Zhihu API (from configured start-date to today)."""
        from datetime import date, datetime

        from zhihu_cli.creator_tools.parse_follower_detail import run_task

        start_str = cache_manager.get_start_date()
        start = datetime.strptime(start_str, "%Y-%m-%d").date()
        days = (date.today() - start).days
        run_task(days=days)
        if output_json:
            filepath = get_data_dir() / "exports" / "follower_detail.json"
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
            except FileNotFoundError:
                print_json({"error": "file not found", "path": str(filepath)})
            else:
                print_json(data)
            return

    @tools_creator_follower.command("analysis")
    @click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON")
    def creator_follower_analysis(output_json: bool) -> None:
        """Fetch follower profile/demographics (关注者画像) from Zhihu API."""
        from zhihu_cli.creator_tools.parse_follower_profile import run_task

        set_json_mode(output_json)
        run_task()
        if output_json:
            filepath = get_data_dir() / "exports" / "follower_profile.json"
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
            except FileNotFoundError:
                print_json({"error": "file not found", "path": str(filepath)})
            else:
                print_json(data)
            return

    @tools_creator_follower.command("plot")
    def creator_follower_plot() -> None:
        """Plot follower detail line chart from follower_detail.json."""
        from zhihu_cli.creator_tools.plot_follower_detail import plot_follower_detail

        plot_follower_detail()

    @tools.group("nlp")
    def tools_nlp() -> None:
        """NLP text analysis on downloaded Markdown files."""

    @tools_nlp.command("count")
    @click.option("--folder", default=str(get_data_dir() / "downloads" / "answers"), help="Folder with Markdown files")
    @click.option("--no-code", is_flag=True, help="Exclude code blocks")
    @click.option("--no-latex", is_flag=True, help="Exclude LaTeX blocks")
    @click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON")
    @click.option("--no-frontmatter", is_flag=True, default=False, help="Exclude YAML frontmatter from word count")
    @click.option(
        "--plot-cdf",
        "plot_cdf_flag",
        is_flag=True,
        default=False,
        help="Plot CDF (Cumulative Distribution Function) of character counts",
    )
    @click.option(
        "--log", "log_scale", is_flag=True, default=False, help="Use logarithmic scale for character count axis"
    )
    def nlp_count(
        folder: str,
        no_code: bool,
        no_latex: bool,
        output_json: bool,
        no_frontmatter: bool,
        plot_cdf_flag: bool,
        log_scale: bool,
    ) -> None:
        """Count characters in downloaded Markdown files."""
        import numpy as np

        from zhihu_cli.nlp_tools.count_answer_words import count_words, plot_cdf

        word_counts = []
        for filepath in Path(folder).rglob("*.md"):
            word_counts.append(
                count_words(str(filepath), no_code=no_code, no_latex=no_latex, no_frontmatter=no_frontmatter)
            )

        if not word_counts:
            if output_json:
                print_json({"files": 0})
            else:
                info("No markdown files found.")
            return

        wc = [int(x) for x in word_counts]
        if output_json:
            print_json(
                {
                    "files": len(wc),
                    "mean": round(float(np.mean(wc)), 1),
                    "min": int(min(wc)),
                    "std": round(float(np.std(wc)), 1),
                    "p50": int(np.percentile(wc, 50)),
                    "p90": int(np.percentile(wc, 90)),
                    "max": int(max(wc)),
                }
            )
            return

        echo(f"  {f_label('Files:')} {f_num(len(word_counts))}")
        echo(
            f"  {f_label('Mean:')} {f_num(f'{np.mean(word_counts):.0f}')}  {f_label('Std:')} {f_num(f'{np.std(word_counts):.0f}')}"
        )
        echo(
            f"  {f_label('P50:')} {f_num(f'{np.percentile(word_counts, 50):.0f}')}  {f_label('P90:')} {f_num(f'{np.percentile(word_counts, 90):.0f}')}"
        )
        echo(f"  {f_label('Min:')} {f_num(min(word_counts))}  {f_label('Max:')} {f_num(max(word_counts))}")

        if plot_cdf_flag:
            plot_cdf(word_counts, log_scale=log_scale)

    @tools_nlp.command("wordcloud")
    @click.option("--source-dir", default=str(get_data_dir() / "downloads"), help="Directory with Markdown files")
    @click.option("--topk", type=int, default=200, help="Top K keywords")
    @click.option("--only-print", is_flag=True, help="Only print keywords, skip image generation")
    def nlp_wordcloud(source_dir: str, topk: int, only_print: bool) -> None:
        """Generate a word cloud from downloaded content."""
        from zhihu_cli.nlp_tools.wordcloud_generator import main as wc_main

        wc_main(topk_words=topk, source_dir=source_dir, only_print=only_print)

    @tools_nlp.command("cluster")
    @click.option("--source-dir", default=str(get_data_dir() / "downloads"), help="Directory with Markdown files")
    @click.option("--output", "-o", default=str(get_data_dir() / "plots" / "zhihu_clusters.png"), help="Output image")
    @click.option("--n-clusters", type=int, default=8, help="Number of clusters")
    @click.option("--n-terms", type=int, default=10, help="Top terms per cluster")
    @click.option(
        "--mode", type=click.Choice(["pca", "tsne", "hybrid"]), default="pca", help="Dimensionality reduction"
    )
    @click.option("--evaluate-k", is_flag=True, help="Run elbow/silhouette analysis to help choose K")
    def nlp_cluster(source_dir: str, output: str, n_clusters: int, n_terms: int, mode: str, evaluate_k: bool) -> None:
        """KMeans cluster visualization of downloaded content."""
        from zhihu_cli.nlp_tools.cluster_visualizer import (
            find_best_k,
            load_and_clean_data,
            process_clusters,
            visualize_with_plotly,
        )

        documents, file_names = load_and_clean_data(source_dir)
        if not documents:
            error("No documents found.")
            return

        if evaluate_k:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(max_features=1500, min_df=2, max_df=0.95)
            X = vectorizer.fit_transform(documents)
            find_best_k(X, max_k=20)
            info("Check the elbow/silhouette plot to choose K.")
            return

        X, labels, vectorizer, kmeans = process_clusters(documents, n_clusters)
        visualize_with_plotly(
            X, labels, file_names, vectorizer, kmeans, n_clusters, mode=mode, output_path=output, n_terms=n_terms
        )  # type: ignore[arg-type]
        success(f"Saved {f_path(output)}")

    @tools_nlp.command("conetwork")
    @click.option("--source-dir", default=str(get_data_dir() / "downloads"), help="Directory with Markdown files")
    @click.option("--topk", type=int, default=80, help="Top N words to include in network")
    @click.option("--window-size", type=int, default=5, help="Co-occurrence window size within documents")
    @click.option("--min-edge-weight", type=int, default=3, help="Minimum co-occurrence count to show edge")
    @click.option(
        "--output", "-o", default=str(get_data_dir() / "plots" / "zhihu_conetwork.png"), help="Output image path"
    )
    def nlp_conetwork(source_dir: str, topk: int, window_size: int, min_edge_weight: int, output: str) -> None:
        """Word co-occurrence network visualization of downloaded content."""
        from zhihu_cli.nlp_tools.cooccurrence_network import main as conetwork_main

        conetwork_main(
            source_dir=source_dir,
            topk=topk,
            window_size=window_size,
            min_edge_weight=min_edge_weight,
            output=output,
        )

    @tools_nlp.command("graph")
    @click.option(
        "--url-token", default=None, help="User url_token to analyze (auto-detects logged-in user if omitted)"
    )
    @click.option("--max-followees", type=int, default=200, help="Max followees to fetch")
    @click.option("--max-followers", type=int, default=200, help="Max followers to fetch")
    @click.option(
        "--output", "-o", default="", help="Output image path (default: ~/.zhihu-cli/plots/zhihu_social_graph.png)"
    )
    @click.option(
        "--layout",
        type=click.Choice(["spring", "kamada_kawai", "circular", "shell"]),
        default="spring",
        help="Graph layout algorithm",
    )
    @click.option("--no-viz", is_flag=True, help="Print statistics only, skip image generation")
    @click.option("--depth", type=int, default=1, help="Graph depth: 1=ego-network, ≥2=recursively expand (default: 1)")
    @click.option("--max-expand", type=int, default=20, help="Max nodes to expand per hop level (default: 20)")
    @click.option("--max-per-node", type=int, default=50, help="Max followees fetched per expanded node (default: 50)")
    @click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON")
    def nlp_graph(
        url_token: str | None,
        max_followees: int,
        max_followers: int,
        output: str,
        layout: str,
        no_viz: bool,
        depth: int,
        max_expand: int,
        max_per_node: int,
        output_json: bool,
    ) -> None:
        """Social graph visualization of following/follower relationships."""
        from zhihu_cli.nlp_tools.graph import main as graph_main

        graph_main(
            url_token=url_token,
            max_followees=max_followees,
            max_followers=max_followers,
            output=output,
            layout=layout,
            no_viz=no_viz,
            depth=depth,
            max_expand=max_expand,
            max_per_node=max_per_node,
            output_json=output_json,
        )
