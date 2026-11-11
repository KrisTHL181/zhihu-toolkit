"""Explicit asset model shared by the scrape and convert pipelines.

An asset is a plain dictionary of the form::

    {"id": str, "type": "answer" | "article" | "pin", "title": str,
     "url": str (optional), "created_time": int (optional)}

Assets are stored on disk as a plain JSON list of such dictionaries, so files
written by older versions of the tool keep working unchanged. The live
producers are ``zhihu scrape creations`` (creator API shape) and
``zhihu convert universal``; each source shape is handled by its own adapter
instead of the old heuristic guessing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

ASSET_TYPES = ("answer", "article", "pin")

SOURCE_FORMATS = ("answers", "typed", "generic_title", "activities", "creations", "unknown")


class Asset(TypedDict, total=False):
    """One piece of Zhihu content in the unified asset format."""

    id: str
    type: str
    title: str
    url: str
    created_time: int


def detect_format(items: list[dict[str, Any]]) -> str:
    """Heuristically detect the format from the first recognisable item.

    Scanning continues past items that match no known shape, so a leading
    malformed item (for example a deleted question serialized as ``null``)
    does not make the whole list look undetectable. Shapes are checked in
    priority order: raw API answers, activities, creator creations, generic
    typed items, and finally items carrying an ``id`` and ``title``.

    :param items: The exported items to inspect.
    :returns: One of the values in :data:`SOURCE_FORMATS`.
    """
    for item in items:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("question"), dict):
            return "answers"
        if "verb" in item and "target_type" in item:
            return "activities"
        if "type" in item and isinstance(item.get("data"), dict):
            return "creations"
        if "type" in item:
            return "typed"
        if "title" in item and "id" in item:
            return "generic_title"
    return "unknown"


def _asset_from_answers(item: dict[str, Any]) -> Asset | None:
    """Adapt a raw API answer list item.

    :param item: An answer item with an ``id`` and a ``question`` object.
    :returns: The asset, or ``None`` when the item has no usable id.
    """
    raw_id = item.get("id")
    if raw_id is None:
        return None
    asset_id = str(raw_id)
    if not asset_id:
        return None

    question = item.get("question")
    title = (question.get("title") if isinstance(question, dict) else None) or "untitled answer"

    asset: Asset = {"id": asset_id, "type": "answer", "title": title}
    if "url" in item:
        asset["url"] = item["url"]
    return asset


def _asset_from_typed(item: dict[str, Any]) -> Asset | None:
    """Adapt an item that already carries a ``type`` field.

    ``created_time`` is deliberately not copied: the scrape pipeline emits a
    formatted timestamp string there, not an epoch.

    :param item: A typed item with an ``id``.
    :returns: The asset, or ``None`` when the type or id is unusable.
    """
    asset_type = item.get("type")
    if not asset_type:
        return None
    raw_id = item.get("id")
    if raw_id is None:
        return None
    asset_id = str(raw_id)
    if not asset_id:
        return None

    asset: Asset = {"id": asset_id, "type": str(asset_type), "title": item.get("title") or f"{asset_type}_{asset_id}"}
    if "url" in item:
        asset["url"] = item["url"]
    return asset


def _asset_from_generic_title(item: dict[str, Any]) -> Asset | None:
    """Adapt an item carrying only an ``id`` and ``title`` as an article.

    :param item: An item with an ``id``.
    :returns: The asset, or ``None`` when the item has no usable id.
    """
    raw_id = item.get("id")
    if raw_id is None:
        return None
    asset_id = str(raw_id)
    if not asset_id:
        return None

    asset: Asset = {"id": asset_id, "type": "article", "title": item.get("title") or f"article_{asset_id}"}
    if "url" in item:
        asset["url"] = item["url"]
    return asset


def _asset_from_activities(item: dict[str, Any]) -> Asset | None:
    """Adapt an activity feed item, keeping only content-creation activities.

    Votes, follows and other non-creation activities have no downloadable
    target and are skipped.

    :param item: An activity item with ``target_type`` / ``target_id``.
    :returns: The asset, or ``None`` when the activity is not a creation.
    """
    target_type = item.get("target_type")
    if target_type not in ASSET_TYPES:
        return None
    if not item.get("is_creation", True):
        return None

    target_id = item.get("target_id")
    if target_id is None:
        return None
    asset_id = str(target_id)
    if not asset_id:
        return None

    asset: Asset = {
        "id": asset_id,
        "type": str(target_type),
        "title": item.get("title") or f"{target_type}_{asset_id}",
    }
    if "url" in item:
        asset["url"] = item["url"]
    return asset


def _asset_from_creations(item: dict[str, Any]) -> Asset | None:
    """Adapt a creator-API creation item (``{"type": ..., "data": {...}}``).

    :param item: A creation item whose ``data`` object holds the content.
    :returns: The asset, or ``None`` when the type or id is unsupported.
    """
    asset_type = item.get("type")
    if asset_type not in ASSET_TYPES:
        return None

    data = item.get("data")
    if not isinstance(data, dict):
        return None
    raw_id = data.get("id")
    if raw_id is None:
        return None
    asset_id = str(raw_id)
    if not asset_id:
        return None

    return {
        "id": asset_id,
        "type": str(asset_type),
        "title": data.get("title", ""),
        "created_time": data.get("created_time", 0),
    }


#: Adapter per detected source format. ``unknown`` has no adapter on purpose.
_ADAPTERS = {
    "answers": _asset_from_answers,
    "typed": _asset_from_typed,
    "generic_title": _asset_from_generic_title,
    "activities": _asset_from_activities,
    "creations": _asset_from_creations,
}


def asset_from_item(item: dict[str, Any], source: str) -> Asset | None:
    """Convert a single source item into an asset.

    :param item: The source item to adapt.
    :param source: One of the values in :data:`SOURCE_FORMATS`, describing the
        shape of *item*.
    :returns: The adapted asset, or ``None`` when the item has no usable id,
        is not the expected shape, or *source* is unknown.
    """
    if not isinstance(item, dict):
        return None
    adapter = _ADAPTERS.get(source)
    if adapter is None:
        return None
    return adapter(item)


def convert_items(
    items: list[dict[str, Any]],
    forced_type: str | None = None,
    source: str | None = None,
) -> list[Asset]:
    """Convert a list of source items into assets.

    :param items: Source items to convert.
    :param forced_type: When set, overrides the ``type`` of every emitted
        asset.
    :param source: The source format; ``None`` or ``"auto"`` detects it from
        *items*.
    :returns: The converted assets, in input order. Invalid items are skipped.
    """
    if source in (None, "auto"):
        detected = detect_format(items)
    else:
        detected = source

    result: list[Asset] = []
    for item in items:
        asset = asset_from_item(item, detected)
        if asset is None:
            continue
        if forced_type:
            asset["type"] = forced_type
        result.append(asset)
    return result


def dedupe_assets(assets: list[Asset]) -> list[Asset]:
    """Remove duplicate assets, keeping the first occurrence of each.

    :param assets: Assets in their original order.
    :returns: Assets deduplicated by ``(type, id)``, preserving order.
    """
    seen: set[tuple[str, str]] = set()
    unique: list[Asset] = []
    for asset in assets:
        key = (asset.get("type", ""), asset.get("id", ""))
        if key in seen:
            continue
        seen.add(key)
        unique.append(asset)
    return unique


def convert_files(
    paths: Iterable[str | Path],
    *,
    forced_type: str | None = None,
    source: str | None = None,
) -> list[Asset]:
    """Load and convert several export files, deduplicating the result.

    Format detection is performed per file when *source* is not given, so a
    directory of mixed exports converts correctly.

    :param paths: Files to load and convert.
    :param forced_type: When set, overrides the ``type`` of every asset.
    :param source: The source format; ``None`` or ``"auto"`` detects it per
        file.
    :returns: The deduplicated assets from all files.
    """
    result: list[Asset] = []
    for path in paths:
        result.extend(convert_items(load_json(path), forced_type=forced_type, source=source))
    return dedupe_assets(result)


def load_json(file_path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON file that holds a list or a ``{"data": [...]}`` wrapper.

    :param file_path: File to read.
    :returns: The list of source items.
    :raises ValueError: When the top-level structure is unsupported.
    """
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    raise ValueError(f"Unsupported JSON structure in {file_path}")


def load_assets(path: str | Path) -> list[Asset]:
    """Load an asset file, skipping entries that are not usable assets.

    Entries must be dictionaries with a non-empty ``id`` and ``type``; other
    entries (including non-dict values) are dropped. Extra keys such as
    ``url`` and ``created_time`` are preserved, and unknown types are kept
    because filtering is the caller's decision.

    :param path: Asset JSON file to read.
    :returns: The usable assets, in file order.
    :raises ValueError: When the top-level structure is unsupported.
    """
    assets: list[Asset] = []
    for entry in load_json(path):
        if not isinstance(entry, dict):
            continue
        raw_id = entry.get("id")
        raw_type = entry.get("type")
        if raw_id is None or not str(raw_id):
            continue
        if raw_type is None or not str(raw_type):
            continue
        asset: Asset = {**entry, "id": str(raw_id), "type": str(raw_type), "title": str(entry.get("title") or "")}
        assets.append(asset)
    return assets


def write_assets(path: str | Path, assets: list[Asset]) -> None:
    """Write assets to *path* as a plain JSON list, creating parent directories.

    :param path: Destination file.
    :param assets: Assets to write.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(list(assets), f, ensure_ascii=False, indent=2)


def asset_url(asset: Asset) -> str:
    """Build the canonical Zhihu URL for an asset.

    :param asset: The asset whose ``id`` and ``type`` are used.
    :returns: The canonical web URL.
    :raises ValueError: When the asset type is not a supported asset type.
    """
    asset_type = asset.get("type")
    if asset_type == "answer":
        return f"https://www.zhihu.com/answer/{asset['id']}"
    if asset_type == "article":
        return f"https://zhuanlan.zhihu.com/p/{asset['id']}"
    if asset_type == "pin":
        return f"https://www.zhihu.com/pin/{asset['id']}"
    raise ValueError(f"unsupported asset type: {asset_type}")
