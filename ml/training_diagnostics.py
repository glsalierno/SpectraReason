"""Per-label threshold search and hard-negative diagnostics for structural FG SVM training."""



from __future__ import annotations



import csv

from pathlib import Path

from typing import Any



import numpy as np



from ml.ftir_ontology import HIGH_RISK_SPECIFIC_V4



THRESHOLD_OBJECTIVES = ("f1", "precision_biased", "balanced_guarded")



# Hard-negative label pairs: (positive_label, [competitor_labels that should not fire])

HARD_NEGATIVE_PAIRS: dict[str, list[str]] = {

    "phenol": ["alcohol", "aromatic", "aryl_ether"],

    "amide": [

        "primary_amine",

        "secondary_amine",

        "tertiary_amine",

        "ketone",

        "ester",

        "carboxylic_acid",

    ],

    "ester": ["ether", "ketone", "aldehyde", "amide", "carboxylic_acid"],

    "siloxane": ["ether", "aryl_ether", "alcohol", "phenol", "ester"],

    "silicone_or_silane": ["ether", "aryl_ether", "alcohol", "phenol", "ester"],

    "nitrile": ["alkyne"],

    "nitro": ["aromatic", "heteroaromatic"],

    "heteroaromatic": ["aromatic", "primary_amine", "secondary_amine", "pyrrole_like_NH"],

}





def is_high_risk_label(label: str) -> bool:

    return str(label) in HIGH_RISK_SPECIFIC_V4 or str(label) in {

        "nitro_family",

        "hydroxy_containing",

        "carbonyl_containing",

        "nitrogen_containing",

    }





def _objective_score(

    lab: str,

    precision: float,

    recall: float,

    f1: float,

    *,

    objective: str,

    model_kind: str,

) -> float:

    obj = str(objective or "balanced_guarded").lower()

    mk = str(model_kind or "").lower()

    if obj == "f1":

        return f1

    if obj == "precision_biased":

        return precision * 0.65 + f1 * 0.35

    # balanced_guarded (default for specific)

    if is_high_risk_label(lab) or mk in ("specific", "subtle"):

        return precision * 0.55 + f1 * 0.45

    if mk in ("family", "basic"):

        return f1 * 0.6 + precision * 0.4

    return f1





def tune_per_label_thresholds(

    label_names: list[str],

    y_test: np.ndarray,

    scores: np.ndarray,

    *,

    score_kind: str,

    default: float = 0.5,

    objective: str = "balanced_guarded",

    model_kind: str = "",

    label_supports: dict[str, int] | None = None,

) -> tuple[list[float], list[dict[str, Any]]]:

    """

    Search thresholds on test split. Objective and label risk drive precision vs F1 tradeoff.

    """

    from sklearn.metrics import f1_score, precision_score, recall_score



    thresholds: list[float] = []

    rows: list[dict[str, Any]] = []

    grid = np.linspace(0.15, 0.85, 15) if score_kind == "calibrated_probability" else np.linspace(-0.5, 1.5, 17)

    supports = label_supports or {}



    for j, lab in enumerate(label_names):

        y_t = y_test[:, j]

        s = scores[:, j]

        if len(np.unique(y_t)) < 2:

            thr = default

            rows.append(

                {

                    "label": lab,

                    "threshold": thr,

                    "precision": None,

                    "recall": None,

                    "f1": None,

                    "threshold_objective": objective,

                    "note": "single_class_test_split",

                }

            )

            thresholds.append(thr)

            continue



        sup = int(supports.get(lab, int(np.sum(y_t))))

        best_thr = default

        best_score = -1.0

        best_row: dict[str, Any] = {}

        for thr in grid:

            pred = (s >= thr).astype(int)

            pr = float(precision_score(y_t, pred, zero_division=0))

            rc = float(recall_score(y_t, pred, zero_division=0))

            f1 = float(f1_score(y_t, pred, zero_division=0))

            score = _objective_score(lab, pr, rc, f1, objective=objective, model_kind=model_kind)

            if score > best_score:

                best_score = score

                best_thr = float(thr)

                best_row = {

                    "label": lab,

                    "threshold": best_thr,

                    "precision": pr,

                    "recall": rc,

                    "f1": f1,

                    "threshold_objective": objective,

                }

        if is_high_risk_label(lab) and best_row:

            floor = 0.50 if sup < 40 else 0.45

            best_thr = min(0.88, max(best_thr, floor))

            best_row["threshold"] = best_thr

            best_row["high_risk_precision_bias"] = True

        elif sup < 30 and best_row:

            best_thr = min(0.90, max(best_thr, 0.48))

            best_row["threshold"] = best_thr

            best_row["low_support_stricter"] = True

        rows.append(best_row)

        thresholds.append(best_thr)

    return thresholds, rows





def hard_negative_false_positive_report(

    label_names: list[str],

    y_test: np.ndarray,

    scores: np.ndarray,

    thresholds: list[float],

    *,

    out_path: Path,

) -> dict[str, Any]:

    """Write CSV of FP rates among defined hard-negative competitor labels."""

    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    summary: dict[str, Any] = {"pairs": [], "total_fp": 0}



    for j, pos_lab in enumerate(label_names):

        competitors = HARD_NEGATIVE_PAIRS.get(pos_lab)

        if not competitors:

            continue

        pos_mask = y_test[:, j] == 0

        if not np.any(pos_mask):

            continue

        for comp in competitors:

            if comp not in label_names:

                continue

            cj = label_names.index(comp)

            neg_scores = scores[pos_mask, cj]

            thr_c = thresholds[cj] if cj < len(thresholds) else 0.5

            fp = int(np.sum(neg_scores >= thr_c))

            n = int(neg_scores.size)

            rate = float(fp / n) if n else 0.0

            med = float(np.median(neg_scores)) if n else 0.0

            rows.append(

                {

                    "positive_label": pos_lab,

                    "competitor_label": comp,

                    "hard_negative_n": n,

                    "false_positives": fp,

                    "fp_rate": round(rate, 4),

                    "median_ml_score": round(med, 4),

                    "threshold_competitor": thr_c,

                }

            )

            summary["total_fp"] += fp

            summary["pairs"].append(f"{pos_lab}->{comp}:{fp}/{n}")



    if rows:

        with out_path.open("w", newline="", encoding="utf-8") as fh:

            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))

            w.writeheader()

            w.writerows(rows)

    return summary





def hard_negative_metrics_by_label(

    label_names: list[str],

    y_test: np.ndarray,

    scores: np.ndarray,

    thresholds: list[float],

    *,

    out_path: Path,

    hard_negative_pairs: dict[str, list[str]] | None = None,

) -> list[dict[str, Any]]:

    """

    Per target label / hard-negative group metrics for audit.

    Columns: target_label, hard_negative_group, false_positives, fp_rate, median_ml_score, threshold_used

    """

    pairs = hard_negative_pairs or HARD_NEGATIVE_PAIRS

    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []



    for pos_lab, competitors in pairs.items():

        if pos_lab not in label_names:

            continue

        j = label_names.index(pos_lab)

        pos_mask = y_test[:, j] == 0

        if not np.any(pos_mask):

            continue

        for comp in competitors:

            if comp not in label_names:

                continue

            cj = label_names.index(comp)

            neg_scores = scores[pos_mask, cj]

            thr_c = thresholds[cj] if cj < len(thresholds) else 0.5

            fp = int(np.sum(neg_scores >= thr_c))

            n = int(neg_scores.size)

            rows.append(

                {

                    "target_label": pos_lab,

                    "hard_negative_group": comp,

                    "false_positives": fp,

                    "false_positive_rate": round(float(fp / n) if n else 0.0, 4),

                    "median_ml_score": round(float(np.median(neg_scores)) if n else 0.0, 4),

                    "threshold_used": thr_c,

                    "hard_negative_n": n,

                }

            )



    if rows:

        with out_path.open("w", newline="", encoding="utf-8") as fh:

            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))

            w.writeheader()

            w.writerows(rows)

    return rows





def compute_hard_negative_sample_weights(

    label_names: list[str],

    Y: np.ndarray,

    *,

    weight: float = 1.5,

    hard_negative_pairs: dict[str, list[str]] | None = None,

) -> np.ndarray:

    """

    Upweight rows that are hard negatives for high-risk positive labels (training only).

    Returns per-row multipliers (same length as Y.shape[0]).

    """

    pairs = hard_negative_pairs or HARD_NEGATIVE_PAIRS

    w = float(weight) if weight > 1.0 else 1.0

    if w <= 1.0:

        return np.ones(Y.shape[0], dtype=float)



    row_w = np.ones(Y.shape[0], dtype=float)

    for pos_lab, competitors in pairs.items():

        if pos_lab not in label_names:

            continue

        j = label_names.index(pos_lab)

        comp_idx = [label_names.index(c) for c in competitors if c in label_names]

        if not comp_idx:

            continue

        pos_neg = Y[:, j] == 0

        any_comp = np.any(Y[:, comp_idx] == 1, axis=1) if len(comp_idx) > 1 else (Y[:, comp_idx[0]] == 1)

        mask = pos_neg & any_comp

        row_w[mask] = np.maximum(row_w[mask], w)

    return row_w


