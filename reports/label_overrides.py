"""Per-spectrum label override JSON for interactive curation and static re-export."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

LabelMode = Literal["transmittance", "normalized_absorbance"]


def overrides_path(overrides_dir: Path, stem: str) -> Path:
    return Path(overrides_dir) / f"{stem}_label_overrides.json"


def load_label_overrides(path: Path | None) -> dict[str, Any]:
    if not path or not Path(path).is_file():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def save_label_overrides(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def build_auto_override_payload(
    *,
    stem: str,
    labels: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "spectrum_stem": stem,
        "labels": labels,
        "source": "auto",
    }


def label_record(
    *,
    mode: LabelMode,
    wavenumber_cm1: float,
    peak_y: float,
    label_text: str,
    show_label: bool = True,
    xshift_px: float = 0.0,
    yshift_px: float = 12.0,
    label_x_cm1: float | None = None,
    label_y_value: float | None = None,
    source: str = "auto",
    reason: str = "",
    region: str = "",
    score: float = 0.0,
    prominence: float = 0.0,
    comment: str = "",
    requested_wavenumber_cm1: float | None = None,
    snapped_wavenumber_cm1: float | None = None,
    snap_window_cm1: float | None = None,
    snap_target: str = "",
    added_by: str = "",
    snap_status: str = "",
) -> dict[str, Any]:
    snapped = snapped_wavenumber_cm1 if snapped_wavenumber_cm1 is not None else float(wavenumber_cm1)
    return {
        "mode": mode,
        "wavenumber_cm1": float(wavenumber_cm1),
        "peak_y": float(peak_y),
        "label_text": str(label_text),
        "show_label": bool(show_label),
        "xshift_px": float(xshift_px),
        "yshift_px": float(yshift_px),
        "label_x_cm1": label_x_cm1,
        "label_y_value": label_y_value,
        "source": source,
        "reason": reason,
        "region": region,
        "score": float(score),
        "prominence": float(prominence),
        "comment": comment,
        "requested_wavenumber_cm1": (
            float(requested_wavenumber_cm1)
            if requested_wavenumber_cm1 is not None
            else float(wavenumber_cm1)
        ),
        "snapped_wavenumber_cm1": snapped,
        "snap_window_cm1": snap_window_cm1,
        "snap_target": snap_target,
        "added_by": added_by,
        "snap_status": snap_status,
    }


def build_auto_curation_labels(
    selection: Any,
    t_minima: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build full candidate label list for curation (auto + suppressed visibility)."""
    selected_wn = {round(float(p["wavenumber_cm1"]), 1) for p in selection.selected}
    records: list[dict[str, Any]] = []
    for c in selection.candidates:
        reason = str(c.get("reason_not_selected") or "")
        if reason == "outside labeled region" and not c.get("required"):
            continue
        wn_v = float(c["wavenumber_cm1"])
        records.append(
            label_record(
                mode="normalized_absorbance",
                wavenumber_cm1=wn_v,
                peak_y=float(c.get("intensity", c.get("height", 0))),
                label_text=f"{wn_v:.0f}",
                show_label=round(wn_v, 1) in selected_wn or bool(c.get("label_selected")),
                yshift_px=12.0,
                source="manual" if c.get("required") else "auto",
                region=str(c.get("region", "")),
                score=float(c.get("score", 0)),
                prominence=float(c.get("prominence", 0)),
                comment=reason,
                snap_target="max_absorbance",
            )
        )
    for row in t_minima:
        abs_wn = float(row.get("absorbance_peak_cm1", row.get("wn", 0)))
        t_wn = float(row.get("wn", row.get("transmittance_label_cm1", abs_wn)))
        shown = str(row.get("label_shown", "")).lower() == "yes"
        records.append(
            label_record(
                mode="transmittance",
                wavenumber_cm1=t_wn,
                peak_y=float(row.get("y", row.get("transmittance_value", 0))),
                label_text=f"{t_wn:.0f}",
                show_label=shown,
                yshift_px=-12.0,
                source="auto",
                comment=str(row.get("note", "")),
                snap_target="min_transmittance",
            )
        )
    return records


def manual_label_from_snap(
    snap: dict[str, Any],
    *,
    added_by: str,
    show_label: bool = True,
) -> dict[str, Any]:
    mode = str(snap.get("mode", "normalized_absorbance"))
    wn = float(snap["wavenumber_cm1"])
    return label_record(
        mode=mode,  # type: ignore[arg-type]
        wavenumber_cm1=wn,
        peak_y=float(snap.get("peak_y", snap.get("intensity", 0))),
        label_text=f"{wn:.0f}",
        show_label=show_label,
        yshift_px=-12.0 if mode == "transmittance" else 12.0,
        source="manual",
        region=str(snap.get("region", "")),
        prominence=float(snap.get("prominence", 0)),
        comment="manual peak",
        requested_wavenumber_cm1=float(snap.get("requested_wavenumber_cm1", wn)),
        snapped_wavenumber_cm1=float(snap.get("snapped_wavenumber_cm1", wn)),
        snap_window_cm1=float(snap.get("snap_window_cm1", 25.0)),
        snap_target=str(snap.get("snap_target", "")),
        added_by=added_by,
        snap_status=str(snap.get("snap_status", "")),
    )


def selected_absorbance_peaks_from_labels(
    labels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Peaks for selected CSV / feedback when overrides are applied."""
    out: list[dict[str, Any]] = []
    for lab in labels:
        if str(lab.get("mode", "")) != "normalized_absorbance":
            continue
        if not lab.get("show_label", True):
            continue
        wn_v = float(lab["wavenumber_cm1"])
        out.append(
            {
                "wavenumber_cm1": wn_v,
                "intensity": float(lab.get("peak_y", 0)),
                "prominence": float(lab.get("prominence", 0)),
                "height": float(lab.get("peak_y", 0)),
                "region": lab.get("region", ""),
                "score": float(lab.get("score", 0)),
            }
        )
    return out


def merge_overrides_with_auto(
    auto_labels: list[dict[str, Any]],
    saved: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Apply saved overrides keyed by (mode, rounded wavenumber)."""
    if not saved or not saved.get("labels"):
        return list(auto_labels)
    key_map: dict[tuple[str, int], dict[str, Any]] = {}
    for row in saved.get("labels") or []:
        if not isinstance(row, dict):
            continue
        mode = str(row.get("mode", ""))
        try:
            wn = int(round(float(row.get("wavenumber_cm1", 0))))
        except (TypeError, ValueError):
            continue
        key_map[(mode, wn)] = row

    merged: list[dict[str, Any]] = []
    used_keys: set[tuple[str, int]] = set()
    for lab in auto_labels:
        mode = str(lab.get("mode", ""))
        wn_key = int(round(float(lab.get("wavenumber_cm1", 0))))
        ov = key_map.get((mode, wn_key))
        if ov:
            merged.append({**lab, **ov, "source": ov.get("source", lab.get("source", "auto"))})
            used_keys.add((mode, wn_key))
        else:
            merged.append(lab)

    for key, ov in key_map.items():
        if key in used_keys:
            continue
        merged.append(ov)
    return merged


def labels_for_mode(labels: list[dict[str, Any]], mode: LabelMode) -> list[dict[str, Any]]:
    return [l for l in labels if str(l.get("mode", "")) == mode and l.get("show_label", True)]


def _px_to_data_y(yshift_px: float, y_span: float) -> float:
    return (float(yshift_px) / 280.0) * max(y_span, 1e-9)


def overrides_to_laid_peaks(
    labels: list[dict[str, Any]],
    *,
    mode: LabelMode,
    y_min: float,
    y_max: float,
) -> list[dict[str, Any]]:
    """Convert override records to matplotlib/leader layout items."""
    y_span = max(float(y_max) - float(y_min), 1e-9)
    label_side = "above" if mode == "normalized_absorbance" else "below"
    direction = 1.0 if label_side == "above" else -1.0
    laid: list[dict[str, Any]] = []
    for lab in labels:
        if str(lab.get("mode", "")) != mode:
            continue
        if not lab.get("show_label", True):
            continue
        wn = float(lab["wavenumber_cm1"])
        peak_y = float(lab.get("peak_y", lab.get("y", 0)))
        base_offset = 0.08 * y_span * direction
        y_off = _px_to_data_y(float(lab.get("yshift_px", 12)), y_span) * direction + base_offset
        label_y = (
            float(lab["label_y_value"])
            if lab.get("label_y_value") is not None
            else peak_y + y_off
        )
        laid.append(
            {
                "wn": wn,
                "y": peak_y,
                "text": str(lab.get("label_text", f"{wn:.0f}")),
                "label_y": label_y,
                "prominence": float(lab.get("prominence", 0)),
                "show_label": True,
            }
        )
    return laid
