#!/usr/bin/env python3
"""Wait for deconv NPZ build, then train deconv benchmark (memory-safe, no latest update)."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECONV_CANDIDATES: list[tuple[Path, Path]] = [
    (
        ROOT / "ml/runs/experiments/v4_deconv_specific/ds_deconv.npz",
        ROOT / "ml/runs/experiments/v4_deconv_specific/ds_deconv.meta.json",
    ),
    (
        ROOT / "ml/runs/experiments/v4_deconv_benchmark/ds_C_deconv.npz",
        ROOT / "ml/runs/experiments/v4_deconv_benchmark/ds_C_deconv.meta.json",
    ),
]
DATASET_PREFIX: Path | None = None
PC_OUT = ROOT / "ml/runs/experiments/v4_peakcodebook_specific"
B_PC_OUT = ROOT / "ml/runs/experiments/v4_classification_improvement/B_peakcodebook_specific"
PEAKCODEBOOK_OUT_DIRS = (B_PC_OUT, PC_OUT)
DECONV_OUT = ROOT / "ml/runs/experiments/v4_deconv_specific"
LOCK = DECONV_OUT / ".deconv_train_queued.lock"
LOG = DECONV_OUT / "wait_and_train.log"
POLL_SEC = 120
STABLE_SEC = 90


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    DECONV_OUT.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _resolve_ready_dataset() -> tuple[Path, Path] | None:
    global DATASET_PREFIX
    for npz, meta in DECONV_CANDIDATES:
        if not npz.is_file() or npz.stat().st_size < 1_000_000 or not meta.is_file():
            continue
        try:
            body = json.loads(meta.read_text(encoding="utf-8"))
            if int(body.get("feature_dim") or 0) >= 500:
                DATASET_PREFIX = npz.with_suffix("")
                return npz, meta
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            continue
    return None


def npz_stable() -> tuple[Path, Path] | None:
    hit = _resolve_ready_dataset()
    if not hit:
        return None
    npz, _ = hit
    s0 = npz.stat().st_size
    time.sleep(STABLE_SEC)
    hit2 = _resolve_ready_dataset()
    if hit2 and hit2[0].stat().st_size == s0:
        return hit2
    return None


def _train_done_in_dir(out_dir: Path) -> bool:
    if (out_dir / "train.log").is_file():
        txt = (out_dir / "train.log").read_text(encoding="utf-8", errors="replace")
        if '"model_out"' in txt or "model_out" in txt:
            return True
    return any(out_dir.glob("struct_fg_specific_v4_ontology_*.joblib"))


def peakcodebook_train_running() -> bool:
    try:
        import subprocess as sp

        out = sp.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                "Where-Object { $_.CommandLine -match 'structural_fg_svm train' -and "
                "$_.CommandLine -match 'peakcodebook' } | Measure-Object | Select-Object -ExpandProperty Count",
            ],
            text=True,
            timeout=30,
        )
        return int(out.strip() or "0") > 0
    except Exception:
        return False


def peakcodebook_train_done() -> bool:
    """Slot free when active peakcodebook benchmark finished (B or legacy v4_peakcodebook path)."""
    if peakcodebook_train_running():
        return False
    return any(_train_done_in_dir(d) for d in PEAKCODEBOOK_OUT_DIRS)


def deconv_already_trained() -> bool:
    return any(DECONV_OUT.glob("struct_fg_specific_v4_ontology_*.joblib"))


def wait_for_npz() -> None:
    log("Waiting for deconv dataset (v4_deconv_specific or v4_deconv_benchmark)")
    while True:
        hit = npz_stable()
        if hit:
            npz, meta_path = hit
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            log(
                f"Deconv NPZ ready: path={npz} n_rows={meta.get('n_rows')} "
                f"feature_dim={meta.get('feature_dim')}"
            )
            return
        for npz, _ in DECONV_CANDIDATES:
            if npz.is_file():
                log(f"NPZ present but not stable/complete yet ({npz}, {npz.stat().st_size} bytes)")
        time.sleep(POLL_SEC)


def wait_for_slot() -> None:
    """Wait until no active peakcodebook OvR train (success or failure frees the slot)."""
    log("Waiting for peakcodebook train slot (B_peakcodebook or v4_peakcodebook; max 1 heavy OvR train)")
    while peakcodebook_train_running():
        log("Peakcodebook train still running (polling)...")
        time.sleep(POLL_SEC)
    if peakcodebook_train_done():
        log("Peakcodebook train finished with model artifact; slot available for deconv train")
    else:
        log(
            "Peakcodebook train not running (no .joblib in benchmark dirs); "
            "slot available for deconv train"
        )


def run_deconv_train() -> int:
    if deconv_already_trained():
        log("Deconv model already exists; skipping train")
        return 0
    cmd = [
        sys.executable,
        "-m",
        "ml.structural_fg_svm",
        "train",
        "--dataset-prefix",
        str(DATASET_PREFIX or DECONV_CANDIDATES[0][0].with_suffix("")),
        "--model-kind",
        "specific",
        "--ontology",
        "v4",
        "--pipeline-version",
        "v4_ontology",
        "--label-source",
        "smarts",
        "--calibration",
        "sigmoid",
        "--split",
        "molecule",
        "--min-label-positives",
        "10",
        "--hard-negative-mode",
        "on",
        "--threshold-objective",
        "balanced_guarded",
        "--random-state",
        "13",
        "--memory-safe",
        "--n-jobs",
        "1",
        "--no-update-latest",
        "--deconv-mode",
        "fast",
        "--out",
        str(DECONV_OUT),
    ]
    log("Starting deconv train: " + " ".join(cmd))
    train_log = DECONV_OUT / "train.log"
    with train_log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
    log(f"Deconv train exit code: {proc.returncode}")
    return int(proc.returncode)


def main() -> int:
    DECONV_OUT.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        log(f"Lock exists ({LOCK}); another watcher may be running. Exiting.")
        return 0
    LOCK.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), encoding="utf-8")
    try:
        wait_for_npz()
        wait_for_slot()
        return run_deconv_train()
    finally:
        if LOCK.is_file():
            LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
