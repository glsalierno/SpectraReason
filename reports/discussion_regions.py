"""Discussion / manuscript spectral region definitions for offset stacks and chunks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reports.paper_peak_selection import PAPER_LABEL_REGIONS

DEFAULT_RANGE_COLORS = ("#0072bd", "#d95319", "#77ac30", "#7e2f8e", "#a2142f")


@dataclass
class DiscussionRegion:
    name: str
    wn_min: float
    wn_max: float
    title: str = ""
    color: str = "#0072bd"
    show_in_stacks: bool = True
    label_policy: str = "selected_only"

    @property
    def lo(self) -> float:
        return min(self.wn_min, self.wn_max)

    @property
    def hi(self) -> float:
        return max(self.wn_min, self.wn_max)


def default_ranges_config() -> dict[str, Any]:
    """Default custom FTIR discussion ranges (900–400 cm⁻¹ excluded from auto labels)."""
    specs = [
        ("OH_NH_stretch", 3000.0, 3700.0, "O–H / N–H stretch"),
        ("CH_stretch", 2800.0, 3000.0, "C–H stretch"),
        ("C_O_aromatic_NH", 1500.0, 1800.0, "C=O / aromatic / N–H bend"),
        ("ring_CN", 1200.0, 1500.0, "Ring / C–N"),
        ("CO_fingerprint", 900.0, 1200.0, "C–O / fingerprint"),
    ]
    ranges = []
    for i, (name, lo, hi, title) in enumerate(specs):
        ranges.append(
            {
                "name": name,
                "wn_min": lo,
                "wn_max": hi,
                "color": DEFAULT_RANGE_COLORS[i % len(DEFAULT_RANGE_COLORS)],
                "show_in_stacks": True,
                "label_policy": "selected_only",
                "title": title,
            }
        )
    return {"range_set_name": "Custom FTIR discussion ranges", "ranges": ranges}


def default_discussion_regions() -> list[DiscussionRegion]:
    return ranges_to_discussion_regions(default_ranges_config())


def _row_to_region(row: dict[str, Any], *, idx: int = 0) -> DiscussionRegion:
    name = str(row.get("name") or row.get("id") or f"region_{idx}")
    wn_min = float(row.get("wn_min", row.get("lo", 400)))
    wn_max = float(row.get("wn_max", row.get("hi", 4000)))
    title = str(row.get("title") or row.get("label") or name.replace("_", " "))
    color = str(row.get("color") or DEFAULT_RANGE_COLORS[idx % len(DEFAULT_RANGE_COLORS)])
    show_in_stacks = bool(row.get("show_in_stacks", True))
    label_policy = str(row.get("label_policy") or "selected_only")
    return DiscussionRegion(
        name=name,
        wn_min=wn_min,
        wn_max=wn_max,
        title=title,
        color=color,
        show_in_stacks=show_in_stacks,
        label_policy=label_policy,
    )


def ranges_to_discussion_regions(payload: dict[str, Any]) -> list[DiscussionRegion]:
    items = payload.get("ranges") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return default_discussion_regions()
    out = [_row_to_region(row, idx=i) for i, row in enumerate(items) if isinstance(row, dict)]
    return out or default_discussion_regions()


def load_ranges_config(path: Path | None) -> dict[str, Any]:
    if not path or not Path(path).is_file():
        return default_ranges_config()
    path = Path(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(f"PyYAML required for regions file {path}") from exc
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("ranges"), list):
        return raw
    if isinstance(raw, list):
        return {"range_set_name": "Custom FTIR discussion ranges", "ranges": raw}
    return default_ranges_config()


def load_discussion_regions(path: Path | None) -> list[DiscussionRegion]:
    return ranges_to_discussion_regions(load_ranges_config(path))


def save_ranges_config(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def legacy_paper_regions() -> list[DiscussionRegion]:
    """Legacy ids from paper_peak_selection (backward compatible)."""
    return [
        DiscussionRegion(
            name=str(r["id"]),
            wn_min=float(r["lo"]),
            wn_max=float(r["hi"]),
            title=str(r.get("title", r["id"])),
        )
        for r in PAPER_LABEL_REGIONS
    ]
