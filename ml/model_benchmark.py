"""Benchmark alternative classifiers on the same train/test split (report-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

BENCHMARK_MODEL_TYPES = (
    "linear_svm",
    "rbf_svm",
    "extra_trees",
    "hist_gradient_boosting",
    "logistic_elasticnet",
)


def _macro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import f1_score, precision_score, recall_score

    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _fit_ovr_scores(
    model_type: str,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    random_state: int,
    n_jobs: int,
) -> tuple[np.ndarray, str, str]:
    """Return (n_test x n_labels) scores in [0,1] or decision space, calibration_status, notes."""
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X_train)
    Xs_te = scaler.transform(X_test)
    mt = str(model_type).lower()
    cal_status = "uncalibrated"
    notes = ""

    if mt == "linear_svm":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import LinearSVC

        base = LinearSVC(class_weight="balanced", random_state=random_state, max_iter=100_000, dual=False)
        n_cv = min(5, max(2, int(X_train.shape[0] // 2000)))
        clf = OneVsRestClassifier(CalibratedClassifierCV(base, method="sigmoid", cv=n_cv, n_jobs=max(1, n_jobs)), n_jobs=1)
        clf.fit(Xs_tr, Y_train)
        raw = clf.predict_proba(Xs_te)
        if isinstance(raw, list):
            scores = np.column_stack(
                [np.asarray(r)[:, 1] if np.asarray(r).ndim == 2 else np.asarray(r).ravel() for r in raw]
            )
        else:
            scores = np.asarray(raw)
        cal_status = "sigmoid_ovr"
        notes = "default production family; linear coefficients interpretable"
        return scores, cal_status, notes

    if mt == "rbf_svm":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import SVC

        if X_train.shape[0] > 12_000:
            notes = "skipped: training rows > 12000 for RBF SVM"
            return np.zeros((X_test.shape[0], Y_train.shape[1])), "skipped", notes
        base = SVC(kernel="rbf", class_weight="balanced", probability=False, random_state=random_state)
        n_cv = min(3, max(2, int(X_train.shape[0] // 3000)))
        clf = OneVsRestClassifier(CalibratedClassifierCV(base, method="sigmoid", cv=n_cv, n_jobs=1), n_jobs=1)
        clf.fit(Xs_tr, Y_train)
        raw = clf.predict_proba(Xs_te)
        if isinstance(raw, list):
            scores = np.column_stack(
                [np.asarray(r)[:, 1] if np.asarray(r).ndim == 2 else np.asarray(r).ravel() for r in raw]
            )
        else:
            scores = np.asarray(raw)
        cal_status = "sigmoid_ovr"
        notes = "nonlinear; slower; less interpretable"
        return scores, cal_status, notes

    if mt == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        base = ExtraTreesClassifier(
            n_estimators=120,
            max_depth=24,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=max(1, n_jobs),
        )
        clf = OneVsRestClassifier(base, n_jobs=1)
        clf.fit(Xs_tr, Y_train)
        raw = clf.predict_proba(Xs_te)
        if isinstance(raw, list):
            scores = np.column_stack(
                [np.asarray(r)[:, 1] if np.asarray(r).ndim == 2 else np.asarray(r).ravel() for r in raw]
            )
        else:
            scores = np.asarray(raw)
        notes = "tree ensemble; feature importances available but black-box relative to linear SVM"
        return scores, cal_status, notes

    if mt == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        clf = OneVsRestClassifier(
            HistGradientBoostingClassifier(max_iter=80, random_state=random_state),
            n_jobs=1,
        )
        clf.fit(Xs_tr, Y_train)
        raw = clf.predict_proba(Xs_te)
        if isinstance(raw, list):
            scores = np.column_stack(
                [np.asarray(r)[:, 1] if np.asarray(r).ndim == 2 else np.asarray(r).ravel() for r in raw]
            )
        else:
            scores = np.asarray(raw)
        notes = "boosted trees; not default for publication interpretability"
        return scores, cal_status, notes

    if mt == "logistic_elasticnet":
        from sklearn.linear_model import LogisticRegression

        base = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
            n_jobs=max(1, n_jobs),
        )
        clf = OneVsRestClassifier(base, n_jobs=1)
        clf.fit(Xs_tr, Y_train)
        raw = clf.predict_proba(Xs_te)
        if isinstance(raw, list):
            scores = np.column_stack(
                [np.asarray(r)[:, 1] if np.asarray(r).ndim == 2 else np.asarray(r).ravel() for r in raw]
            )
        else:
            scores = np.asarray(raw)
        notes = "sparse linear alternative; coefficients interpretable"
        return scores, cal_status, notes

    raise ValueError(f"Unknown benchmark model_type: {model_type!r}")


def run_model_benchmark(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    label_names: list[str],
    *,
    model_types: list[str] | None = None,
    feature_set: str = "",
    model_kind: str = "",
    random_state: int = 13,
    n_jobs: int = 1,
    threshold_objective: str = "balanced_guarded",
) -> list[dict[str, Any]]:
    from ml.training_diagnostics import tune_per_label_thresholds

    if model_types is None:
        model_types = list(BENCHMARK_MODEL_TYPES)

    rows: list[dict[str, Any]] = []
    for mt in model_types:
        try:
            scores, cal_status, notes = _fit_ovr_scores(
                mt, X_train, Y_train, X_test, random_state=random_state, n_jobs=n_jobs
            )
            if cal_status == "skipped":
                rows.append(
                    {
                        "model_type": mt,
                        "feature_set": feature_set,
                        "model_kind": model_kind,
                        "macro_f1": None,
                        "micro_f1": None,
                        "precision_macro": None,
                        "recall_macro": None,
                        "calibration_status": cal_status,
                        "interpretability_notes": notes,
                    }
                )
                continue
            thresholds, _ = tune_per_label_thresholds(
                label_names,
                Y_test,
                scores,
                score_kind="calibrated_probability",
                objective=threshold_objective,
                model_kind=model_kind,
            )
            pred = np.column_stack([(scores[:, j] >= thresholds[j]).astype(int) for j in range(len(label_names))])
            m = _macro_metrics(Y_test, pred)
            rows.append(
                {
                    "model_type": mt,
                    "feature_set": feature_set,
                    "model_kind": model_kind,
                    **m,
                    "calibration_status": cal_status,
                    "interpretability_notes": notes,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model_type": mt,
                    "feature_set": feature_set,
                    "model_kind": model_kind,
                    "macro_f1": None,
                    "micro_f1": None,
                    "precision_macro": None,
                    "recall_macro": None,
                    "calibration_status": "error",
                    "interpretability_notes": str(exc)[:200],
                }
            )
    return rows
