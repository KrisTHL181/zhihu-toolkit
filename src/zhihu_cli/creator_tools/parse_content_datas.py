"""Scan a creator's content and persist per-day or aggregated metrics."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from zhihu_cli.content.assets import asset_from_item, load_assets, write_assets
from zhihu_cli.content.handlers.stats import fetch_creator_content_metrics
from zhihu_cli.content.handlers.waterfall import stream_handler
from zhihu_cli.content.utils.wait import wait
from zhihu_cli.paths import zhihu_home

# ── helpers ──────────────────────────────────────────────────────────────────

_orig_print = print


def _cli_print(*args: object, **kwargs: object) -> None:
    """Print to stderr (avoids stdout pollution during --json mode)."""
    _orig_print(*args, file=sys.stderr, **kwargs)


def convert_percent(data: str | None) -> float:
    if data is None:
        return 0.0
    return round(float(data.rstrip("%")) / 100, 2)


def get_date(d: dict[str, Any]) -> str | None:
    if d.get("p_date"):
        return d["p_date"]

    if d.get("answer"):
        ts = d["answer"].get("created_time")
    elif d.get("pin"):
        ts = d["pin"].get("create_time")
    elif d.get("article"):
        ts = d["article"].get("created_time")
    else:
        ts = None

    if ts:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    return None


def generate_assets_file(output_path: Path) -> list[dict[str, Any]]:
    """Scrape all user creations and save to output_path. Returns the asset list."""
    initial_url = (
        "https://www.zhihu.com/api/v4/creators/creations/v2/all"
        "?start=0&end=0&limit=20&offset=0&need_co_creation=1&sort_type=created"
    )

    def parse_creations(data: dict[str, Any]) -> Iterable[dict[str, Any]]:
        for item in data.get("data", []):
            asset = asset_from_item(item, "creations")
            if asset is not None:
                yield asset

    _cli_print("Scanning your Zhihu creations...")
    all_assets: list[dict[str, Any]] = []
    for i, item in enumerate(stream_handler(initial_url, parse_creations)):
        all_assets.append(item)
        if (i + 1) % 20 == 0:
            _cli_print(f"  Collected {len(all_assets)} items...")

    write_assets(output_path, all_assets)
    _cli_print(f"Saved {len(all_assets)} assets → {output_path}")
    return all_assets


def _extract_daily_item(d: dict[str, Any] | None, content_type: str = "") -> dict | None:
    """Extract metrics from a single daily item (used for aggr yesterday/today)."""
    if not d:
        return None
    advanced = d.get("advanced") or {}
    upvote_key = "reaction" if content_type == "pin" else "upvote"
    return {
        "date": d.get("p_date"),
        # ── base metrics ──
        "pv": d.get("pv", 0),
        "show": d.get("show", 0),
        "play": d.get("play", 0),
        "upvote": d.get(upvote_key, d.get("upvote", 0)),
        "like": d.get("like", 0),
        "collect": d.get("collect", 0),
        "comment": d.get("comment", 0),
        "share": d.get("share", 0),
        "new_follow_uv": d.get("new_follow_uv", 0),
        "follower_conversion_rate": d.get("follower_conversion_rate", 0),
        "pageshow_uv": d.get("pageshow_uv", 0),
        "positive_interact_rate": d.get("positive_interact_rate", 0),
        "re_pin": d.get("re_pin", 0),
        # ── advanced ──
        "finish_read_percent": convert_percent(advanced.get("finish_read_percent", "0.0%")),
        "positive_interact_percent": convert_percent(advanced.get("positive_interact_percent", "0.0%")),
        "follower_translate": advanced.get("follower_translate", 0),
        "advanced_status": advanced.get("status"),
    }


def _migrate_metrics_filenames(metrics_dir: Path) -> int:
    """Rename legacy metrics files whose name prefix mismatches their content.

    Older versions wrote aggregated files as ``metrics_full_<type>_<id>.json``,
    the same name as daily files, so the two formats overwrote each other.
    Renaming them to ``metrics_aggr_*`` the first time keeps the aggregated
    data when the next daily fetch runs. Files whose name already matches
    their format are left alone.

    :param metrics_dir: Directory holding the metrics JSON files.
    :returns: Number of files renamed.
    """
    renamed = 0
    for fp in sorted(metrics_dir.glob("metrics_*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and "totals" in data:
            prefix, new_prefix = "metrics_full_", "metrics_aggr_"
        elif isinstance(data, list) and data and isinstance(data[0], dict) and "date" in data[0]:
            prefix, new_prefix = "metrics_aggr_", "metrics_full_"
        else:
            continue
        if not fp.stem.startswith(prefix):
            continue
        target = metrics_dir / f"{new_prefix}{fp.stem.removeprefix(prefix)}.json"
        if target.exists():
            _cli_print(f"  [Skip] Both {fp.name} and {target.name} exist")
            continue
        fp.rename(target)
        renamed += 1
    if renamed:
        _cli_print(f"  Renamed {renamed} legacy metrics file(s) to their correct prefix")
    return renamed


def run_batch_daily_analysis(use_aggr: bool = False) -> None:
    data_dir = zhihu_home() / "exports"
    metrics_dir = data_dir / "content_metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    _migrate_metrics_filenames(metrics_dir)
    assets_file = data_dir / "all_assets_list.json"

    answer_ids = []
    if assets_file.exists():
        answer_ids = load_assets(assets_file)
        _cli_print(f"Loaded {len(answer_ids)} assets to analyze")
    else:
        _cli_print("all_assets_list.json not found")
        _cli_print("  This file is the asset inventory of your Zhihu creations.")
        _cli_print()
        try:
            choice = input("  Generate it now? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            _cli_print()
            return
        if choice in ("", "y", "yes"):
            _cli_print()
            answer_ids = generate_assets_file(assets_file)
            if not answer_ids:
                return
        else:
            _cli_print()
            _cli_print("  Run: zhihu scrape creations")
            _cli_print("  Then: zhihu tools creator metrics")
            return

    success_count = 0
    for i, token in enumerate(answer_ids):
        _cli_print(f"\n[Task {i + 1}/{len(answer_ids)}] Processing ID: {token} ...")

        created_ts = token.get("created_time", 0)
        start_date = datetime.fromtimestamp(created_ts).strftime("%Y-%m-%d")

        try:
            data = fetch_creator_content_metrics(
                token["id"],
                token["type"],
                start=start_date,
                end=datetime.now().strftime("%Y-%m-%d"),
                aggregated=use_aggr,
            )

            if data is not None:
                if use_aggr:
                    # API returns per-day metrics inside yesterday/today,
                    # plus top-level answer metadata, content_mark, and advanced.
                    advanced_top = data.get("advanced") or {}
                    content_mark = data.get("content_mark") or {}
                    answer_meta = data.get("answer") or {}
                    clean_data = {
                        "type": token["type"],
                        "totals": {
                            "pv": data.get("pv", 0),
                            "show": data.get("show", 0),
                            "play": data.get("play", 0),
                            "upvote": data.get("reaction", data.get("upvote", 0))
                            if token["type"] == "pin"
                            else data.get("upvote", 0),
                            "like": data.get("like", 0),
                            "collect": data.get("collect", 0),
                            "comment": data.get("comment", 0),
                            "share": data.get("share", 0),
                            "new_follow_uv": data.get("new_follow_uv", 0),
                            "follower_conversion_rate": data.get("follower_conversion_rate", 0),
                            "pageshow_uv": data.get("pageshow_uv", 0),
                            "positive_interact_rate": data.get("positive_interact_rate", 0),
                            "re_pin": data.get("re_pin", 0),
                        },
                        "content_mark": {
                            "is_normal": content_mark.get("is_normal"),
                            "is_collapsed": content_mark.get("is_collapsed"),
                            "is_suggest": content_mark.get("is_suggest"),
                        },
                        "advanced_totals": {
                            "finish_read_percent": convert_percent(advanced_top.get("finish_read_percent", "0.0%")),
                            "positive_interact_percent": convert_percent(
                                advanced_top.get("positive_interact_percent", "0.0%")
                            ),
                            "follower_translate": advanced_top.get("follower_translate", 0),
                            "status": advanced_top.get("status"),
                        },
                        "yesterday": _extract_daily_item(data.get("yesterday"), token["type"]),
                        "today": _extract_daily_item(data.get("today"), token["type"]),
                    }
                    # Include answer metadata if present
                    if answer_meta:
                        clean_data["answer_meta"] = {
                            "id": answer_meta.get("id"),
                            "title": answer_meta.get("title"),
                            "question_id": answer_meta.get("question_id"),
                            "url_token": answer_meta.get("url_token"),
                            "excerpt": answer_meta.get("excerpt"),
                            "answer_type": answer_meta.get("answer_type"),
                            "created_time": answer_meta.get("created_time"),
                            "updated_time": answer_meta.get("updated_time"),
                        }
                    entries_label = "aggregated"
                else:
                    clean_data = []
                    for d in data:
                        advanced = d.get("advanced") or {}
                        upvote_key = "reaction" if token["type"] == "pin" else "upvote"

                        clean_data.append(
                            {
                                "type": token["type"],
                                "date": get_date(d),
                                "pv": d.get("pv", 0),
                                "show": d.get("show", 0),
                                "play": d.get("play", 0),
                                "upvote": d.get(upvote_key, d.get("upvote", 0)),
                                "like": d.get("like", 0),
                                "collect": d.get("collect", 0),
                                "comment": d.get("comment", 0),
                                "share": d.get("share", 0),
                                "new_follow_uv": d.get("new_follow_uv", 0),
                                "follower_conversion_rate": d.get("follower_conversion_rate", 0),
                                "pageshow_uv": d.get("pageshow_uv", 0),
                                "positive_interact_rate": d.get("positive_interact_rate", 0),
                                "re_pin": d.get("re_pin", 0),
                                "finish_read_percent": convert_percent(advanced.get("finish_read_percent", "0.0%")),
                                "positive_interact_percent": convert_percent(
                                    advanced.get("positive_interact_percent", "0.0%")
                                ),
                                "follower_translate": advanced.get("follower_translate", 0),
                                "advanced_status": advanced.get("status"),
                            }
                        )
                    entries_label = f"{len(clean_data)} entries"

                prefix = "metrics_aggr_" if use_aggr else "metrics_full_"
                output_file = metrics_dir / f"{prefix}{token['type']}_{token['id']}.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(clean_data, f, indent=4)

                _cli_print(f"  Saved: {output_file} ({entries_label} records)")
                success_count += 1
            else:
                _cli_print("  Fetch failed (non-200 response)")

        except Exception as e:
            _cli_print(f"  Exception: {e}")

        wait(1.2)

    _cli_print("\n" + "=" * 40)
    _cli_print("Batch harvest finished!")
    _cli_print(f"Successful fetches: {success_count} / {len(answer_ids)}")
    _cli_print(f"Data stored in: {metrics_dir}")
    _cli_print("=" * 40)


if __name__ == "__main__":
    run_batch_daily_analysis()
