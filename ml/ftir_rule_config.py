"""
Optional rule/evidence configuration: presets and JSON overrides.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

_PRESETS_DIR = Path(__file__).resolve().parent.parent / "configs" / "rule_presets"
_BUILTIN_PRESETS = ("conservative", "sensitive", "phenol_alcohol_strict")


def list_presets() -> list[str]:
    names = list(_BUILTIN_PRESETS)
    if _PRESETS_DIR.is_dir():
        for p in sorted(_PRESETS_DIR.glob("*.json")):
            if p.stem not in names:
                names.append(p.stem)
    return names


def load_rules_config(
    *,
    preset: str | None = None,
    config_path: str | Path | None = None,
    inline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Merge optional preset + JSON file + inline dict.

    Returns dict with keys: ``evidence``, ``label_overrides``, ``post_rules``, ``description``.
    """
    merged: dict[str, Any] = {
        "evidence": {},
        "label_overrides": {},
        "post_rules": {},
        "description": "",
    }
    if preset:
        p = _PRESETS_DIR / f"{preset}.json"
        if not p.is_file():
            raise FileNotFoundError(f"Unknown rules preset: {preset!r} (expected {p})")
        merged = _deep_merge(merged, json.loads(p.read_text(encoding="utf-8")))
    if config_path:
        cp = Path(config_path)
        if not cp.is_file():
            raise FileNotFoundError(cp)
        merged = _deep_merge(merged, json.loads(cp.read_text(encoding="utf-8")))
    if inline:
        merged = _deep_merge(merged, inline)
    return merged


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def evidence_config_from_rules(rules_config: dict[str, Any] | None) -> dict[str, Any]:
    if not rules_config:
        return {}
    return dict(rules_config.get("evidence") or {})


def rules_assign_config_from_rules(rules_config: dict[str, Any] | None) -> dict[str, Any]:
    if not rules_config:
        return {}
    out: dict[str, Any] = {
        "label_overrides": dict(rules_config.get("label_overrides") or {}),
        "post_rules": dict(rules_config.get("post_rules") or {}),
    }
    if "ontology" in rules_config:
        out["ontology"] = str(rules_config.get("ontology") or "v3").lower()
    return out
