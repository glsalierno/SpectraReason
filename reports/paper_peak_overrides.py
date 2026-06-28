"""Optional per-spectrum peak label overrides (YAML or CSV)."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PaperPeakOverrides:
    required_peaks: list[float] = field(default_factory=list)
    preferred_peaks: list[float] = field(default_factory=list)
    suppressed_peaks: list[float] = field(default_factory=list)
    suppress_ranges: list[tuple[float, float]] = field(default_factory=list)
    region_quota_overrides: dict[str, int] = field(default_factory=dict)


def _norm_stem(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in Path(name).stem)


def _parse_range_pair(values: list[Any]) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    try:
        a, b = float(values[0]), float(values[1])
    except (TypeError, ValueError):
        return None
    return (min(a, b), max(a, b))


def _merge_entry(entry: dict[str, Any], out: PaperPeakOverrides) -> None:
    for key in ("required_peaks", "preferred_peaks", "suppressed_peaks"):
        raw = entry.get(key)
        if not raw:
            continue
        vals = [float(x) for x in raw if _is_number(x)]
        getattr(out, key).extend(vals)
    for sr in entry.get("suppress_ranges") or []:
        if isinstance(sr, (list, tuple)) and len(sr) >= 2:
            pair = _parse_range_pair(list(sr))
            if pair:
                out.suppress_ranges.append(pair)
    rq = entry.get("region_quota_overrides")
    if isinstance(rq, dict):
        for k, v in rq.items():
            try:
                out.region_quota_overrides[str(k)] = int(v)
            except (TypeError, ValueError):
                pass


def _is_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            f"PyYAML required to read override file {path}. Install pyyaml or use CSV format."
        ) from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _load_csv(path: Path) -> dict[str, dict[str, Any]]:
    """CSV columns: stem, kind, wn1, wn2 (optional)."""
    by_stem: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            stem = _norm_stem(str(row.get("stem") or row.get("spectrum") or ""))
            if not stem:
                continue
            entry = by_stem.setdefault(
                stem,
                {
                    "required_peaks": [],
                    "preferred_peaks": [],
                    "suppressed_peaks": [],
                    "suppress_ranges": [],
                    "region_quota_overrides": {},
                },
            )
            kind = str(row.get("kind") or row.get("type") or "").lower().strip()
            wn1 = row.get("wavenumber") or row.get("wn1") or row.get("wavenumber_cm1")
            wn2 = row.get("wn2")
            if kind in ("required", "required_peak", "require"):
                if _is_number(wn1):
                    entry["required_peaks"].append(float(wn1))
            elif kind in ("preferred", "preferred_peak"):
                if _is_number(wn1):
                    entry["preferred_peaks"].append(float(wn1))
            elif kind in ("suppressed", "suppress", "suppress_peak"):
                if _is_number(wn1):
                    entry["suppressed_peaks"].append(float(wn1))
            elif kind in ("suppress_range", "ignore_range"):
                pair = _parse_range_pair([wn1, wn2])
                if pair:
                    entry["suppress_ranges"].append(list(pair))
            elif kind.startswith("quota_"):
                region_id = kind.replace("quota_", "", 1)
                if region_id and _is_number(wn1):
                    entry["region_quota_overrides"][region_id] = int(float(wn1))
    return by_stem


def load_override_store(path: Path | None) -> dict[str, PaperPeakOverrides]:
    """Load all spectrum overrides from a YAML or CSV file."""
    if not path:
        return {}
    path = Path(path)
    if not path.is_file():
        return {}

    store: dict[str, PaperPeakOverrides] = {}
    if path.suffix.lower() in (".yaml", ".yml"):
        raw = _load_yaml(path)
        for stem_key, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            stem = _norm_stem(str(stem_key))
            ov = PaperPeakOverrides()
            _merge_entry(entry, ov)
            store[stem] = ov
    elif path.suffix.lower() == ".csv":
        raw = _load_csv(path)
        for stem, entry in raw.items():
            ov = PaperPeakOverrides()
            _merge_entry(entry, ov)
            store[stem] = ov
    else:
        raise ValueError(f"Unsupported override file format: {path.suffix}")
    return store


def resolve_overrides_for_stem(
    store: dict[str, PaperPeakOverrides],
    stem: str,
) -> PaperPeakOverrides:
    key = _norm_stem(stem)
    return store.get(key, PaperPeakOverrides())
