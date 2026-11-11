"""Explicit asset model shared by the scrape and download pipelines.

An asset is a plain dictionary of the form::

    {"id": str, "type": "answer" | "article" | "pin", "title": str,
     "url": str (optional), "created_time": int (optional)}

Assets are stored on disk as a plain JSON list of such dictionaries, so files
written by older versions of the tool keep working unchanged. The live
producer is ``zhihu scrape creations`` (creator API shape);
``asset_from_item(item, "article")`` covers the raw article lists used by the
crank extension. There is no heuristic detection — every caller names the
source shape explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from typing import Any

ASSET_TYPES = ("answer", "article", "pin")


class Asset(TypedDict, total=False):
    """One piece of Zhihu content in the unified asset format."""

    id: str
    type: str
    title: str
    url: str
    created_time: int


def _asset_from_article(item: dict[str, Any]) -> Asset | None:
    """Adapt a raw article API item carrying an ``id`` and ``title``.

    :param item: An article item with an ``id``.
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


#: Adapter per source format.
_ADAPTERS = {
    "creations": _asset_from_creations,
    "article": _asset_from_article,
}


def asset_from_item(item: dict[str, Any], source: str) -> Asset | None:
    """Convert a single source item into an asset.

    :param item: The source item to adapt.
    :param source: Either ``"creations"`` (creator-API creation items) or
        ``"article"`` (raw article API items), describing the shape of *item*.
    :returns: The adapted asset, or ``None`` when the item has no usable id,
        is not the expected shape, or *source* is unknown.
    """
    if not isinstance(item, dict):
        return None
    adapter = _ADAPTERS.get(source)
    if adapter is None:
        return None
    return adapter(item)


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
