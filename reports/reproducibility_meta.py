"""
Reproducibility metadata embedded in report Technical details (collapsed in front mode).
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MARKER_REPRODUCIBILITY = "<!-- report-feature:reproducibility-meta -->"


def _sha256_file(path: Path, *, max_mb: int = 512) -> str:
    if not path.is_file():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        n = 0
        while chunk := f.read(1 << 20):
            h.update(chunk)
            n += len(chunk)
            if n > max_mb * (1 << 20):
                return h.hexdigest()[:16] + "…(truncated)"
    return h.hexdigest()[:16]


def git_commit_hash(repo_root: Path | None = None) -> str:
    root = repo_root or Path(__file__).resolve().parents[1]
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()[:12]
    except (OSError, subprocess.SubprocessError):
        pass
    return "unavailable"


def package_versions() -> dict[str, str]:
    out: dict[str, str] = {"python": platform.python_version()}
    for mod in ("numpy", "scipy", "sklearn", "plotly", "joblib", "pandas"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except ImportError:
            out[mod] = "not installed"
    return out


def band_library_version(repo_root: Path | None = None) -> str:
    root = repo_root or Path(__file__).resolve().parents[1]
    yml = root / "ml" / "ftir_band_library.yaml"
    py = root / "ml" / "ftir_band_library.py"
    if yml.is_file():
        return f"yaml sha256:{_sha256_file(yml)}"
    if py.is_file():
        return f"python sha256:{_sha256_file(py)}"
    return "unknown"


def _publish_model_path(path: Path, *, anonymize: bool, repo_root: Path) -> str:
    if anonymize:
        return path.name
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.name


def build_run_context(
    *,
    paths_line: list[str] | str,
    family_model: Path | None = None,
    specific_model: Path | None = None,
    legacy_model: Path | None = None,
    repo_root: Path | None = None,
    anonymize_paths: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = repo_root or Path(__file__).resolve().parents[1]
    ctx: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit_hash(root),
        "platform": platform.platform(),
        "packages": package_versions(),
        "ontology": "v4",
        "band_library": band_library_version(root),
        "run_settings": paths_line if isinstance(paths_line, str) else " | ".join(paths_line),
    }
    models: dict[str, str] = {}
    for key, p in (
        ("family", family_model),
        ("specific", specific_model),
        ("legacy", legacy_model),
    ):
        if p and Path(p).is_file():
            rp = Path(p).resolve()
            models[key] = _publish_model_path(rp, anonymize=anonymize_paths, repo_root=root)
            models[f"{key}_sha256"] = _sha256_file(rp)
    if models:
        ctx["models"] = models
    if extra:
        ctx.update(extra)
    return ctx


def build_reproducibility_html(ctx: dict[str, Any]) -> str:
    """Collapsed-friendly block for Technical details."""
    lines = [
        MARKER_REPRODUCIBILITY,
        "<details class='reproducibility-meta'><summary>Reproducibility metadata</summary>",
        "<pre class='mono repro-json'>",
        _esc(json.dumps(ctx, indent=2, sort_keys=True, default=str)),
        "</pre></details>",
    ]
    return "".join(lines)


def _esc(s: str) -> str:
    import html

    return html.escape(str(s), quote=True)
