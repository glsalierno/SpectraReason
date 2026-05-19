"""
Spectral perturbation robustness evaluation for structural FG SVM models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ml.structural_fg_svm import predict_proba_row


@dataclass
class PerturbationSpec:
    name: str
    apply: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def _sort_xy(wn: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    o = np.argsort(wn)
    return wn[o], y[o]


def perturb_gaussian_noise(wn: np.ndarray, y: np.ndarray, *, sigma_frac: float) -> tuple[np.ndarray, np.ndarray]:
    wn, y = _sort_xy(np.asarray(wn, float), np.asarray(y, float))
    rng = float(np.nanstd(y)) or 1.0
    yn = y + np.random.default_rng(42).normal(0, sigma_frac * rng, size=y.shape)
    return wn, np.clip(yn, 0, None)


def perturb_baseline_slope(wn: np.ndarray, y: np.ndarray, *, slope: float) -> tuple[np.ndarray, np.ndarray]:
    wn, y = _sort_xy(np.asarray(wn, float), np.asarray(y, float))
    return wn, y + slope * (wn - float(np.mean(wn))) / max(float(wn.max() - wn.min()), 1.0)


def perturb_baseline_curve(wn: np.ndarray, y: np.ndarray, *, amp: float) -> tuple[np.ndarray, np.ndarray]:
    wn, y = _sort_xy(np.asarray(wn, float), np.asarray(y, float))
    x = (wn - wn.min()) / max(wn.max() - wn.min(), 1.0)
    return wn, y + amp * (x - 0.5) ** 2


def perturb_wavenumber_shift(wn: np.ndarray, y: np.ndarray, *, delta_cm1: float) -> tuple[np.ndarray, np.ndarray]:
    wn, y = _sort_xy(np.asarray(wn, float), np.asarray(y, float))
    return wn + delta_cm1, y.copy()


def perturb_intensity_scale(wn: np.ndarray, y: np.ndarray, *, scale: float) -> tuple[np.ndarray, np.ndarray]:
    wn, y = _sort_xy(np.asarray(wn, float), np.asarray(y, float))
    return wn, y * scale


def perturb_smooth(wn: np.ndarray, y: np.ndarray, *, window: int = 7) -> tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import uniform_filter1d

    wn, y = _sort_xy(np.asarray(wn, float), np.asarray(y, float))
    w = max(3, int(window) | 1)
    return wn, uniform_filter1d(y, size=w, mode="nearest")


def perturb_downsample(wn: np.ndarray, y: np.ndarray, *, factor: int = 2) -> tuple[np.ndarray, np.ndarray]:
    wn, y = _sort_xy(np.asarray(wn, float), np.asarray(y, float))
    f = max(2, int(factor))
    return wn[::f], y[::f]


def default_perturbations() -> list[PerturbationSpec]:
    specs: list[PerturbationSpec] = []
    for sig in (0.005, 0.01, 0.02, 0.05):
        specs.append(PerturbationSpec(f"noise_sigma_{sig}", lambda w, y, s=sig: perturb_gaussian_noise(w, y, sigma_frac=s)))
    specs.append(PerturbationSpec("baseline_slope_mild", lambda w, y: perturb_baseline_slope(w, y, slope=0.02)))
    specs.append(PerturbationSpec("baseline_curve", lambda w, y: perturb_baseline_curve(w, y, amp=0.03)))
    for d in (-10, -5, -2, 2, 5, 10):
        specs.append(
            PerturbationSpec(f"shift_{d:+d}cm1", lambda w, y, dd=d: perturb_wavenumber_shift(w, y, delta_cm1=dd))
        )
    for sc in (0.85, 1.15):
        specs.append(PerturbationSpec(f"scale_{sc}", lambda w, y, s=sc: perturb_intensity_scale(w, y, scale=s)))
    specs.append(PerturbationSpec("smooth_w7", lambda w, y: perturb_smooth(w, y, window=7)))
    specs.append(PerturbationSpec("downsample_x2", lambda w, y: perturb_downsample(w, y, factor=2)))
    return specs


def _top_k_labels(probs: dict[str, float], k: int) -> list[str]:
    return [lab for lab, _ in sorted(probs.items(), key=lambda kv: -kv[1])[:k]]


def evaluate_evidence_robustness_one_spectrum(
    wn: np.ndarray,
    y: np.ndarray,
    *,
    perturbations: list[PerturbationSpec] | None = None,
    top_k: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rule-assignment stability under perturbations (evidence-first, no ML)."""
    from ml.ftir_evidence import extract_spectral_evidence
    from ml.ftir_rules import assign_functional_groups_from_evidence

    perturbations = perturbations or default_perturbations()
    base_ev = extract_spectral_evidence(wn, y)
    base_rules = assign_functional_groups_from_evidence(base_ev)
    top_base = _top_k_labels(
        {k: float(v.get("score", 0)) for k, v in (base_rules.get("assignments") or {}).items()},
        top_k,
    )
    longform: list[dict[str, Any]] = []
    top1_hits = 0
    n_pert = 0
    for pspec in perturbations:
        try:
            wn_p, y_p = pspec.apply(wn, y)
            rules = assign_functional_groups_from_evidence(extract_spectral_evidence(wn_p, y_p))
            scores = {k: float(v.get("score", 0)) for k, v in (rules.get("assignments") or {}).items()}
            top = _top_k_labels(scores, top_k)
        except Exception as exc:
            longform.append({"layer": "evidence", "perturbation": pspec.name, "error": str(exc)})
            continue
        n_pert += 1
        top1_hits += int(top[:1] == top_base[:1])
        for lab, sc in scores.items():
            base_sc = float((base_rules.get("assignments") or {}).get(lab, {}).get("score", 0))
            longform.append(
                {
                    "layer": "evidence",
                    "perturbation": pspec.name,
                    "label": lab,
                    "rule_score": sc,
                    "delta_vs_original": sc - base_sc,
                    "top1_match": top[:1] == top_base[:1],
                }
            )
    summary = {
        "layer": "evidence",
        "n_perturbations": n_pert,
        "top1_stability": float(top1_hits / n_pert) if n_pert else 0.0,
        "original_top1": top_base[:1],
        "robustness_score": float(top1_hits / n_pert) if n_pert else 0.0,
    }
    return longform, summary


def evaluate_robustness_one_spectrum(
    wn: np.ndarray,
    y: np.ndarray,
    md: dict[str, Any],
    models: dict[str, dict[str, Any]],
    *,
    perturbations: list[PerturbationSpec] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Returns (longform_rows, summary_dict) for one spectrum and one or more models.
    """
    perturbations = perturbations or default_perturbations()
    longform: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"models": {}}

    for model_name, artifact in models.items():
        base_probs = predict_proba_row(artifact, wn=wn, y=y, md=md)
        top1_base = _top_k_labels(base_probs, 1)
        top3_base = _top_k_labels(base_probs, 3)
        per_label_probs: dict[str, list[float]] = {lab: [] for lab in base_probs}
        top1_hits = 0
        top3_hits = 0
        n_pert = 0

        for pspec in perturbations:
            try:
                wn_p, y_p = pspec.apply(wn, y)
                probs = predict_proba_row(artifact, wn=wn_p, y=y_p, md=md)
            except Exception as exc:
                longform.append(
                    {
                        "model": model_name,
                        "perturbation": pspec.name,
                        "error": str(exc),
                    }
                )
                continue
            n_pert += 1
            t1 = _top_k_labels(probs, 1)
            t3 = _top_k_labels(probs, 3)
            top1_hits += int(t1 == top1_base)
            top3_hits += int(set(t3) == set(top3_base))
            for lab, p in probs.items():
                per_label_probs.setdefault(lab, []).append(float(p))
                longform.append(
                    {
                        "model": model_name,
                        "perturbation": pspec.name,
                        "label": lab,
                        "probability": float(p),
                        "delta_vs_original": float(p) - float(base_probs.get(lab, 0.0)),
                        "top1_match": t1 == top1_base,
                        "top3_match": set(t3) == set(top3_base),
                    }
                )

        label_stats = {}
        for lab, vals in per_label_probs.items():
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            label_stats[lab] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "max_delta": float(np.max(np.abs(arr - base_probs.get(lab, 0.0)))),
            }

        robustness_score = float(top1_hits / n_pert) if n_pert else 0.0
        summary["models"][model_name] = {
            "n_perturbations": n_pert,
            "top1_stability": robustness_score,
            "top3_stability": float(top3_hits / n_pert) if n_pert else 0.0,
            "robustness_score": robustness_score,
            "per_label": label_stats,
            "original_top1": top1_base,
            "original_top3": top3_base,
        }

    overall = np.mean([m["robustness_score"] for m in summary["models"].values()]) if summary["models"] else 0.0
    summary["overall_robustness_score"] = float(overall)
    return longform, summary
