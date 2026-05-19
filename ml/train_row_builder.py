"""Rebuild feature rows from raw spectra (augmentation / train-time refresh)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ml.structural_fg_svm import build_feature_row_layout


def featurize_spectrum_arrays(
    wn: np.ndarray,
    y: np.ndarray,
    *,
    feature_set: str,
    has_structure: bool = True,
    evidence: dict[str, Any] | None = None,
) -> np.ndarray:
    md: dict[str, Any] = {}
    if has_structure:
        md["structure_parse_ok"] = True
    row, _ = build_feature_row_layout(
        np.asarray(wn, float),
        np.asarray(y, float),
        md,
        feature_set=feature_set,
        calc=None,
        mordred_dim=0,
        smarts_feature_labels=[],
        evidence=evidence,
    )
    return row


def expand_train_with_augmentation(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    wn_train: list[np.ndarray],
    y_train: list[np.ndarray],
    has_mol_train: np.ndarray,
    *,
    feature_set: str,
    mode: str,
    random_state: int,
    copies_per_row: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Append augmented training rows. Returns (X_new, Y_new, n_augmented_added).
    """
    from ml.spectrum_augmentation import augment_spectrum

    mode = str(mode or "none").lower()
    if mode == "none" or not wn_train:
        return X_train, Y_train, 0

    n_copy = copies_per_row
    if n_copy is None:
        n_copy = 1 if mode == "light" else 2

    X_parts = [X_train]
    Y_parts = [Y_train]
    n_added = 0
    gbase = int(random_state)

    for i, (wn, y) in enumerate(zip(wn_train, y_train)):
        hm = bool(has_mol_train[i] > 0.5) if i < has_mol_train.size else True
        for k in range(n_copy):
            wn_a, y_a = augment_spectrum(wn, y, mode=mode, seed=gbase + i * 17 + k * 997)
            row = featurize_spectrum_arrays(wn_a, y_a, feature_set=feature_set, has_structure=hm)
            X_parts.append(row.reshape(1, -1))
            Y_parts.append(Y_train[i : i + 1])
            n_added += 1

    return np.vstack(X_parts), np.vstack(Y_parts), n_added
