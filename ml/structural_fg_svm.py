#!/usr/bin/env python3
"""
Structural FTIR functional-group SVM (v2): **spectra + RDKit + Mordred**.

Separate from ``ml.ftir_fg_svm`` (spectrum-only 14-D features). This module
concatenates:

1. Same **14-D spectral** statistics as ``ftir_fg_svm.featurize`` (aligned bands).
2. **RDKit** scalar molecular descriptors (when ``Mol`` is built from metadata).
3. **Mordred** descriptors (``ignore_3D=True``), optionally capped for tractability.

**Labels** reuse the same weak supervision as v1: keyword rules on
``metadata_json`` name/title/formula (see ``ml.ftir_fg_svm.FG_RULES``).
Rows without parseable structure still contribute **spectral + zero-padded**
structure columns plus ``has_structure`` flag (unless ``--require-structure``).

By default, **NIST FTIR x-axis preparation** keeps only JCAMP spectra whose
stored X/Y arrays can be interpreted as **wavenumber (cm⁻¹)** for the fixed
band featurizer: **micrometer** axes (``MICROMETERS`` in ``xunits``) are
converted with ``ν = 10000/λ``, then rows with too few points or too narrow a
span are skipped. Use ``--no-nist-ftir-cm1-prep`` to disable (legacy behavior).

**Atom-aware inference:** ``predict`` / ``predict_proba_row`` pass outputs through
``ml.atom_content_mask``: if ``formula``, ``elements``, or ``no_halogens`` in
metadata implies **no F/Cl/Br/I/At**, the **halide** probability is set to **0**
(spectral false positives from weak keyword training).

Optional **PubChem PUG REST** enrichment (``--enrich-pubchem``) fills **SMILES /
InChI** from **CAS** (metadata or path) then **compound name** (``title`` / ``name``
or path stem), cached on disk (see ``ml.pubchem_structure_lookup``).

After a successful PubChem hit, **SMILES are round-tripped through RDKit**
(``MolToSmiles(..., canonical=True, isomericSmiles=True)``) so Mordred sees a
stable, parser-friendly string. **Open Babel is not required** — Mordred uses
RDKit internally; RDKit normalization is the usual approach here.

Dependencies::

    pip install scikit-learn joblib numpy
    pip install rdkit          # or conda install rdkit
    pip install mordred

Example::

    python -m ml.structural_fg_svm build-dataset \\
      --nist-index NIST/reference_libraries/nistchemdata_ir_index_v7_fresh.sqlite \\
      --out-prefix ml/runs/struct_fg_ds \\
      --enrich-pubchem --pubchem-delay 0.25

    python -m ml.structural_fg_svm train \\
      --dataset-prefix ml/runs/struct_fg_ds \\
      --model-out ml/runs/struct_fg_svm_rdkit_mordred.joblib

    python -m ml.structural_fg_svm predict \\
      --model ml/runs/struct_fg_svm_rdkit_mordred.joblib \\
      --spectrum path/to/sample.csv --title "Dopamine"

    python -m ml.structural_fg_svm explain \\
      --model ml/runs/struct_fg_svm_rdkit_mordred.joblib --top 12 --format text

Production (v3) examples::

    # Evidence-only batch report (ML off)
    python reports/structural_fg_svm_kronecker_report.py batch \\
      --inputs ../data/raw/*.CSV --ml-mode none --rules-preset conservative \\
      --include-evidence --include-consensus --no-include-ml \\
      --export-csv reports/production_run/csv --out reports/production_run

    # Train basic SMARTS-label spectral SVM (recommended X: spectral only)
    python -m ml.structural_fg_svm build-dataset --nist-index ... --out-prefix ml/runs/ds_basic \\
      --model-kind basic --label-source smarts --feature-set spectral --require-structure --enrich-pubchem
    python -m ml.structural_fg_svm train --dataset-prefix ml/runs/ds_basic \\
      --model-kind basic --label-source smarts \\
      --calibration sigmoid --split molecule --min-label-positives 20 --out ml/runs/

    # Train subtle SMARTS-label spectral SVM
    python -m ml.structural_fg_svm train --dataset-prefix ml/runs/ds_subtle \\
      --model-kind subtle --calibration sigmoid --split molecule --min-label-positives 20 --out ml/runs/

    # ML-assisted explainable batch report
    python reports/structural_fg_svm_kronecker_report.py batch \\
      --inputs ../data/raw/*.CSV --ml-mode both \\
      --basic-model ml/runs/struct_fg_basic_smarts_latest.joblib \\
      --subtle-model ml/runs/struct_fg_subtle_smarts_latest.joblib \\
      --fusion-mode annotate --include-evidence --include-ml --include-consensus \\
      --export-csv reports/ml_assisted_run/csv --out reports/ml_assisted_run
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ml.fg_label_configs import (
    MODEL_KIND_BASIC,
    MODEL_KIND_COMBINED,
    MODEL_KIND_FAMILY,
    MODEL_KIND_SPECIFIC,
    MODEL_KIND_SUBTLE,
    MODEL_KINDS,
    basic_label_names,
    filter_labels_by_counts,
    infer_fg_vector as infer_fg_vector_kind,
    subtle_label_names,
)
from ml.fg_smarts_library import (
    basic_training_labels,
    binary_vector_for_labels,
    infer_multilabel_smarts,
    smarts_library_version_hash,
    subtle_training_labels,
)
from ml.ftir_ontology import (
    evidence_feature_vector,
    infer_v4_basic_smarts,
    infer_v4_family_smarts,
    infer_v4_specific_smarts,
    is_v4,
    trainable_labels_v4,
)
from ml.ftir_evidence_features import (
    EVIDENCE_FEATURE_VERSION_V2,
    align_evidence_vector,
    evidence_feature_version_for_feature_set,
    feature_prefix_counts,
    stable_evidence_feature_names,
)
from ml.ftir_fg_svm import FG_RULES, featurize
from ml.atom_content_mask import mask_fg_probs_by_atom_content
from ml.pubchem_structure_lookup import (
    apply_canonical_structure_to_cache,
    cache_key,
    enrich_metadata_pubchem,
    load_json_cache,
    preview_pubchem_queries,
    save_json_cache,
    validate_structure_smiles,
)


def _git_hash_short() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        if r.returncode == 0:
            return r.stdout.strip() or None
    except Exception:
        pass
    return None


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _resolve_model_out_path(model_out: str | None, model_kind: str) -> Path:
    if model_out:
        return Path(model_out)
    return Path(f"ml/runs/struct_fg_{model_kind}_{_utc_timestamp()}.joblib")


def _label_names_for_kind(model_kind: str) -> list[str]:
    if model_kind == MODEL_KIND_SUBTLE:
        return subtle_label_names()
    return basic_label_names()


LABEL_SOURCES = ("smarts", "existing", "hybrid")
FEATURE_SET_CHOICES = (
    "legacy",
    "spectral",
    "spectral+smarts",
    "spectral+mordred",
    "spectral+smarts+mordred",
    "spectral+evidence",
    "spectral+evidence_v2",
    "spectral+evidence_v2+peakcodebook",
    "spectral+evidence_v2+deconv",
    "spectral+evidence_v2+peakcodebook+deconv",
    "spectral+smarts+evidence",
    "spectral+mordred+evidence",
    "spectral+smarts+mordred+evidence",
)
CALIBRATION_CHOICES = ("none", "sigmoid", "isotonic")
AUGMENT_CHOICES = ("none", "light", "moderate")
THRESHOLD_OBJECTIVE_CHOICES = ("f1", "precision_biased", "balanced_guarded")
BENCHMARK_MODEL_CHOICES = (
    "linear_svm",
    "rbf_svm",
    "extra_trees",
    "hist_gradient_boosting",
    "logistic_elasticnet",
)


def _normalize_feature_set(
    fs: str | None,
    *,
    meta: dict[str, Any] | None = None,
) -> str:
    if fs and str(fs).strip():
        return str(fs).strip().lower().replace(" ", "")
    if meta and meta.get("feature_set"):
        return str(meta["feature_set"]).strip().lower().replace(" ", "")
    return "legacy"


def resolve_training_label_names(
    model_kind: str,
    label_source: str,
    *,
    ontology: str | None = None,
) -> list[str]:
    """Label column order for Y; depends on weak-label source and model kind."""
    mk = str(model_kind).lower()
    ls = str(label_source).lower()
    ont = str(ontology or "v3").lower()
    from ml.specialist_configs import is_specialist_model_kind, specialist_label_names

    if is_specialist_model_kind(mk):
        return specialist_label_names(mk)
    if ont == "v4" and ls == "smarts":
        if mk in (MODEL_KIND_FAMILY, MODEL_KIND_BASIC):
            return list(trainable_labels_v4("family"))
        if mk in (MODEL_KIND_SPECIFIC, MODEL_KIND_SUBTLE):
            allowed = set(subtle_training_labels())
            return [x for x in trainable_labels_v4("specific") if x in allowed or x in trainable_labels_v4("specific")]
        if mk == MODEL_KIND_COMBINED:
            return list(trainable_labels_v4("combined"))
    if mk == MODEL_KIND_SUBTLE:
        if ls == "existing":
            return subtle_label_names()
        return subtle_training_labels()
    if mk == MODEL_KIND_BASIC:
        if ls == "existing":
            return basic_label_names()
        return basic_training_labels()
    raise ValueError(f"Unknown model_kind: {model_kind!r}")


def _infer_y_vector(
    md: dict[str, Any],
    mol: Any,
    *,
    model_kind: str,
    label_source: str,
    label_names: list[str],
    ontology: str | None = None,
) -> list[int]:
    ls = str(label_source).lower()
    mk = str(model_kind).lower()
    ont = str(ontology or "v3").lower()
    if ls == "existing":
        fg = infer_fg_vector_kind(md, model_kind=mk, mol=mol if mk == MODEL_KIND_SUBTLE else None)
        return [int(fg.get(k, 0)) for k in label_names]
    if ls == "smarts":
        if ont == "v4":
            if mk in (MODEL_KIND_FAMILY, MODEL_KIND_BASIC):
                sm = infer_v4_family_smarts(mol)
            elif mk == MODEL_KIND_COMBINED:
                sm = {**infer_v4_family_smarts(mol), **infer_v4_specific_smarts(mol, label_names)}
            else:
                sm = infer_v4_specific_smarts(mol, label_names)
            return [int(sm.get(k, 0)) for k in label_names]
        sm = infer_multilabel_smarts(mol, label_names)
        return [int(sm.get(k, 0)) for k in label_names]
    # hybrid
    if mk == MODEL_KIND_BASIC:
        kw = infer_fg_vector_kind(md, model_kind=MODEL_KIND_BASIC)
        sm = infer_v4_basic_smarts(mol) if ont == "v4" else infer_multilabel_smarts(mol, label_names)
        out_h: list[int] = []
        for lab in label_names:
            k = int(kw.get(lab, 0))
            if lab == "ether_or_ester":
                k = max(int(kw.get("ether", 0)), int(kw.get("ester", 0)), k)
            if lab == "silicon_oxygen":
                k = max(int(kw.get("silicone_or_siloxane", 0)), k)
            out_h.append(int(max(k, int(sm.get(lab, 0)))))
        return out_h
    kw_only = infer_fg_vector_kind(md, model_kind=MODEL_KIND_SUBTLE, mol=None)
    sm = infer_multilabel_smarts(mol, label_names)
    return [int(max(int(kw_only.get(lab, 0)), int(sm.get(lab, 0)))) for lab in label_names]


def _build_feature_names_for_layout(
    *,
    feature_set: str,
    mordred_names: list[str],
    smarts_labels: list[str],
) -> tuple[list[str], dict[str, int]]:
    sp = spectral_feature_names()
    fs = _normalize_feature_set(feature_set)
    parts: list[str] = [*sp]
    counts: dict[str, int] = {
        "n_spectral": len(sp),
        "n_smarts": 0,
        "n_rdkit": 0,
        "n_mordred": 0,
        "n_evidence": 0,
        "n_has_flag": 1,
    }
    if fs == "legacy":
        rk = [n for n, _ in _rdkit_descriptor_spec()]
        parts.extend([f"rdkit_{n}" for n in rk])
        counts["n_rdkit"] = len(rk)
        parts.extend([f"mordred_{n}" for n in mordred_names])
        counts["n_mordred"] = len(mordred_names)
    else:
        if "smarts" in fs:
            parts.extend([f"smarts_{x}" for x in smarts_labels])
            counts["n_smarts"] = len(smarts_labels)
        if "mordred" in fs:
            parts.extend([f"mordred_{n}" for n in mordred_names])
            counts["n_mordred"] = len(mordred_names)
        if "evidence" in fs:
            evn = stable_evidence_feature_names(feature_set=fs)
            parts.extend(evn)
            counts["n_evidence"] = len(evn)
            counts["evidence_feature_version"] = evidence_feature_version_for_feature_set(fs)
        if "peakcodebook" in fs:
            from ml.ftir_peakcodebook_features import peakcodebook_meta, stable_peakcodebook_feature_names

            pc_names = stable_peakcodebook_feature_names()
            parts.extend(pc_names)
            pcm = peakcodebook_meta()
            counts["n_peakcodebook"] = int(pcm["n_peakcodebook"])
            counts["peakcodebook_bin_width"] = pcm["peakcodebook_bin_width"]
            counts["peakcodebook_n_bins"] = pcm["peakcodebook_n_bins"]
        if "deconv" in fs:
            from ml.ftir_deconvolution import deconv_meta, stable_deconv_feature_names

            dv_names = stable_deconv_feature_names()
            parts.extend(dv_names)
            dvm = deconv_meta()
            counts["n_deconv"] = int(dvm["n_deconv"])
            counts["deconv_profile_type"] = dvm["deconv_profile_type"]
            counts["deconv_regions"] = dvm["deconv_regions"]
    parts.append("has_structure_flag")
    return parts, counts


def build_feature_row_layout(
    wn: np.ndarray,
    y: np.ndarray,
    md: dict[str, Any],
    *,
    feature_set: str,
    calc: Any,
    mordred_dim: int,
    smarts_feature_labels: list[str],
    evidence: dict[str, Any] | None = None,
    deconv_mode: str = "fast",
) -> tuple[np.ndarray, bool]:
    """Assemble X row per ``feature_set`` (v3); ``legacy`` matches historical rdkit+mordret layout."""
    sp = featurize(np.asarray(wn, float), np.asarray(y, float)).ravel()
    mol = mol_from_metadata(md)
    fs = _normalize_feature_set(feature_set)
    parts: list[np.ndarray] = [sp]
    if fs == "legacy":
        rk, _ = featurize_rdkit(mol)
        mo = featurize_mordred(mol, calc, n_cols=mordred_dim)
        parts.extend([rk, mo])
    else:
        if "smarts" in fs:
            vec, _ = binary_vector_for_labels(mol, smarts_feature_labels)
            parts.append(np.asarray(vec, dtype=float))
        if "mordred" in fs:
            mo = featurize_mordred(mol, calc, n_cols=mordred_dim)
            parts.append(mo)
        if "evidence" in fs:
            from ml.ftir_evidence import extract_spectral_evidence

            ev_use = evidence
            if ev_use is None:
                ev_use = extract_spectral_evidence(
                    np.asarray(wn, float),
                    np.asarray(y, float),
                    peaks=None,
                    config={"ontology": "v4"},
                )
            else:
                ev_use = dict(ev_use)
            try:
                from ml.ftir_artifacts import detect_spectral_artifacts

                if not ev_use.get("artifacts"):
                    ev_use["artifacts"] = detect_spectral_artifacts(
                        np.asarray(wn, float), np.asarray(y, float), ev_use
                    )
            except Exception:
                pass
            ev_use["_wn_cache"] = np.asarray(wn, float)
            ev_use["_y_cache"] = np.asarray(y, float)
            ev_vec, evn = evidence_feature_vector(ev_use, feature_set=fs)
            tmpl = stable_evidence_feature_names(feature_set=fs)
            vec = align_evidence_vector(list(ev_vec), evn, tmpl)
            parts.append(np.asarray(vec, dtype=float))
            ev_for_pc = ev_use
        else:
            ev_for_pc = evidence
        if "peakcodebook" in fs:
            from ml.ftir_peakcodebook_features import peakcodebook_feature_vector

            pc_vec, _pcn = peakcodebook_feature_vector(
                np.asarray(wn, float),
                np.asarray(y, float),
                evidence=ev_for_pc if "evidence" in fs else evidence,
            )
            parts.append(np.asarray(pc_vec, dtype=float))
        if "deconv" in fs:
            from ml.ftir_deconvolution import (
                deconv_to_evidence_dict,
                deconvolve_spectrum,
                extract_deconv_features,
            )

            dm = str(deconv_mode or "fast").lower()
            dv_vec, _dvn, _dfail = extract_deconv_features(
                np.asarray(wn, float),
                np.asarray(y, float),
                mode=dm,
                track_stats=True,
            )
            if "evidence" in fs and isinstance(ev_for_pc, dict):
                try:
                    ev_for_pc["deconv"] = deconv_to_evidence_dict(
                        deconvolve_spectrum(np.asarray(wn, float), np.asarray(y, float), mode=dm)
                    )
                except Exception:
                    pass
            parts.append(np.asarray(dv_vec, dtype=float))
    flag = 1.0 if mol is not None else 0.0
    parts.append(np.asarray([flag], dtype=float))
    return np.concatenate(parts), mol is not None


def trim_x_after_label_filter(
    X: np.ndarray,
    *,
    meta: dict[str, Any],
    old_label_names: list[str],
    new_label_names: list[str],
) -> np.ndarray:
    """Drop SMARTS feature columns to match label columns after ``filter_labels_by_counts``."""
    fs = _normalize_feature_set(meta.get("feature_set", "legacy"))
    n_sm_old = len(old_label_names)
    if "smarts" not in fs or n_sm_old == 0 or not new_label_names:
        return X
    n_sp = int(meta.get("n_spectral", 14))
    n_mo = int(meta.get("n_mordred", len(meta.get("mordred_names") or [])))
    n_ev = int(meta.get("n_evidence", 0)) if "evidence" in fs else 0
    if X.shape[1] < n_sp + n_sm_old + n_mo + n_ev + 1:
        return X
    keep_idx = [old_label_names.index(lab) for lab in new_label_names]
    rest_before = X[:, :n_sp]
    sm_old = X[:, n_sp : n_sp + n_sm_old]
    mo_part = X[:, n_sp + n_sm_old : n_sp + n_sm_old + n_mo]
    ev_part = X[:, n_sp + n_sm_old + n_mo : n_sp + n_sm_old + n_mo + n_ev] if n_ev else None
    flag_part = X[:, -1:]
    sm_new = sm_old[:, keep_idx]
    if ev_part is not None:
        return np.hstack([rest_before, sm_new, mo_part, ev_part, flag_part])
    return np.hstack([rest_before, sm_new, mo_part, flag_part])


def _package_versions_meta() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        import joblib
        import sklearn

        out["sklearn"] = getattr(sklearn, "__version__", "unknown")
        out["joblib"] = getattr(joblib, "__version__", "unknown")
    except Exception:
        pass
    try:
        import numpy as np

        out["numpy"] = np.__version__
    except Exception:
        pass
    try:
        from rdkit import Chem

        out["rdkit"] = str(getattr(Chem, "__version__", "unknown"))
    except Exception:
        pass
    try:
        import mordred

        out["mordred"] = getattr(mordred, "__version__", "unknown")
    except Exception:
        pass
    return out


def _require_sklearn():
    try:
        import joblib  # noqa: F401
        from sklearn.multiclass import OneVsRestClassifier  # noqa: F401
        from sklearn.preprocessing import StandardScaler  # noqa: F401
        from sklearn.svm import SVC  # noqa: F401
    except ImportError as e:
        print("Install: pip install scikit-learn joblib", file=sys.stderr)
        raise SystemExit(1) from e


def _require_rdkit():
    try:
        from rdkit import Chem  # noqa: F401
        from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors  # noqa: F401
    except ImportError as e:
        print("Install RDKit: pip install rdkit   (or conda install -c conda-forge rdkit)", file=sys.stderr)
        raise SystemExit(1) from e


def _require_mordred():
    try:
        from mordred import Calculator, descriptors  # noqa: F401
    except ImportError as e:
        print("Install: pip install mordred", file=sys.stderr)
        raise SystemExit(1) from e


def _rdkit_mol_from_smiles_for_mordred(smi: str) -> Any:
    """
    PubChem (and JCAMP) SMILES occasionally need a second parse path. RDKit must
    return a **fully sanitized** mol for reliable Mordred descriptor evaluation.
    """
    _require_rdkit()
    from rdkit import Chem

    s = str(smi).strip()
    if not s:
        return None
    m = Chem.MolFromSmiles(s, sanitize=True)
    if m is not None:
        return m
    m = Chem.MolFromSmiles(s, sanitize=False)
    if m is None:
        return None
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    return m


def mol_from_metadata(md: dict[str, Any]) -> Any:
    """Build RDKit Mol from JCAMP-style metadata if possible (sanitized, Mordred-safe)."""
    _require_rdkit()
    from rdkit import Chem

    if not md:
        return None
    for key in ("SMILES", "smiles", "CAN_SMILES", "csi_smiles", "IsomericSMILES", "isomeric_smiles"):
        s = md.get(key)
        if s and str(s).strip():
            m = _rdkit_mol_from_smiles_for_mordred(str(s))
            if m is not None:
                return m
    for key in ("INCHI", "inchi", "InChI"):
        s = md.get(key)
        if s and str(s).strip():
            m = Chem.MolFromInchi(str(s).strip())
            if m is not None:
                return m
    return None


def canonicalize_structure_metadata(md: dict[str, Any]) -> dict[str, Any]:
    """
    If metadata contains parseable SMILES or InChI, set **SMILES** to RDKit's
    canonical isomeric SMILES (stable for Mordred / training vs inference).

    No-op when neither parses; returns a shallow copy when nothing changes.
    """
    if not md:
        return md
    out = dict(md)
    smi_keys = ("SMILES", "smiles", "CAN_SMILES", "csi_smiles", "IsomericSMILES", "isomeric_smiles")
    inchi_keys = ("INCHI", "inchi", "InChI")
    has_smi = any(out.get(k) and str(out.get(k)).strip() for k in smi_keys)
    has_inchi = any(out.get(k) and str(out.get(k)).strip() for k in inchi_keys)
    if not has_smi and not has_inchi:
        return out

    _require_rdkit()
    from rdkit import Chem

    mol = None
    for key in smi_keys:
        s = out.get(key)
        if s and str(s).strip():
            mol = _rdkit_mol_from_smiles_for_mordred(str(s))
            if mol is not None:
                break
    if mol is None:
        for key in inchi_keys:
            s = out.get(key)
            if s and str(s).strip():
                mol = Chem.MolFromInchi(str(s).strip())
                if mol is not None:
                    break
    if mol is None:
        return out

    try:
        can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return out
    if not can:
        return out
    out["SMILES"] = can
    return out


def _rdkit_descriptor_spec() -> list[tuple[str, Callable[..., float]]]:
    _require_rdkit()
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

    # Stable, common 2D scalar descriptors (no 3D conformer requirement).
    return [
        ("MolWt", Descriptors.MolWt),
        ("ExactMolWt", Descriptors.ExactMolWt),
        ("MolLogP", Descriptors.MolLogP),
        ("TPSA", Descriptors.TPSA),
        ("LabuteASA", Descriptors.LabuteASA),
        ("HeavyAtomCount", Descriptors.HeavyAtomCount),
        ("NumHDonors", Lipinski.NumHDonors),
        ("NumHAcceptors", Lipinski.NumHAcceptors),
        ("NumRotatableBonds", Lipinski.NumRotatableBonds),
        ("RingCount", Lipinski.RingCount),
        ("NumAromaticRings", Lipinski.NumAromaticRings),
        ("NumAliphaticRings", Lipinski.NumAliphaticRings),
        ("NumSaturatedRings", Lipinski.NumSaturatedRings),
        ("FractionCSP3", Lipinski.FractionCSP3),
        ("NumHeteroatoms", Descriptors.NumHeteroatoms),
        ("NumValenceElectrons", Descriptors.NumValenceElectrons),
        ("NumRadicalElectrons", Descriptors.NumRadicalElectrons),
        ("Chi0n", Descriptors.Chi0n),
        ("Chi1n", Descriptors.Chi1n),
        ("Chi2n", Descriptors.Chi2n),
        ("HallKierAlpha", Descriptors.HallKierAlpha),
        ("Kappa1", Descriptors.Kappa1),
        ("Kappa2", Descriptors.Kappa2),
        ("Kappa3", Descriptors.Kappa3),
        ("BertzCT", Descriptors.BertzCT),
        ("NumAmideBonds", rdMolDescriptors.CalcNumAmideBonds),
        ("NumAliphaticCarbocycles", rdMolDescriptors.CalcNumAliphaticCarbocycles),
        ("NumAromaticCarbocycles", rdMolDescriptors.CalcNumAromaticCarbocycles),
        ("NumAliphaticHeterocycles", rdMolDescriptors.CalcNumAliphaticHeterocycles),
        ("NumAromaticHeterocycles", rdMolDescriptors.CalcNumAromaticHeterocycles),
        ("NumSpiroAtoms", rdMolDescriptors.CalcNumSpiroAtoms),
        ("NumBridgeheadAtoms", rdMolDescriptors.CalcNumBridgeheadAtoms),
    ]


def featurize_rdkit(mol: Any | None) -> tuple[np.ndarray, list[str]]:
    spec = _rdkit_descriptor_spec()
    names = [n for n, _ in spec]
    if mol is None:
        return np.zeros(len(spec), dtype=float), names
    vec: list[float] = []
    for _, fn in spec:
        try:
            v = float(fn(mol))
            if not math.isfinite(v):
                v = 0.0
        except Exception:
            v = 0.0
        vec.append(v)
    return np.asarray(vec, dtype=float), names


def make_mordred_calculator(*, max_descriptors: int) -> tuple[Any, list[str]]:
    """Return (Calculator, ordered descriptor names used). Reproducible given same mordred version."""
    _require_mordred()
    from mordred import Calculator, descriptors as md_desc

    # Current mordred exposes ``descriptors.all`` modules; ``Calculator(descriptors, ...)`` expands
    # to ~1613 concrete descriptors (see ``mordred/descriptors/__init__.py``). Older docs mentioned
    # ``all_descriptors`` which is absent on some installs.
    full_calc = Calculator(md_desc, ignore_3D=True)
    all_inst = list(full_calc.descriptors)
    all_inst.sort(key=str)
    if max_descriptors > 0 and len(all_inst) > max_descriptors:
        scored = sorted(range(len(all_inst)), key=lambda i: (hash(str(all_inst[i])) % (2**32), i))
        pick = sorted(scored[:max_descriptors])
        chosen = [all_inst[i] for i in pick]
    else:
        chosen = all_inst
    calc = Calculator(chosen, ignore_3D=True)
    names = [str(d) for d in chosen]
    return calc, names


def prepare_nist_ftir_cm1(
    wn: np.ndarray,
    y: np.ndarray,
    md: dict[str, Any],
    *,
    min_points: int = 32,
    min_span_cm1: float = 200.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Return ``(wavenumber_cm1, y_aligned)`` for NIST JCAMP spectra indexed in SQLite.

    - Uses ``xunits`` / ``XUNITS`` when present (``1/CM`` vs ``MICROMETERS``).
    - Converts **micrometers → cm⁻¹** via ``10000 / λ`` so ``featurize`` band windows match FTIR.
    - Drops rows that are empty, mis-sized, non-finite, or lack a usable cm⁻¹ span.

    Rows already in the DB are successfully parsed JDX; this step keeps only spectra whose
    **x-axis is interpretable as wavenumber** for the fixed cm⁻¹ feature extractor.
    """
    wn = np.asarray(wn, dtype=float).reshape(-1)
    yy = np.asarray(y, dtype=float).reshape(-1)
    if wn.size < min_points or yy.size != wn.size:
        return None
    if not np.all(np.isfinite(wn)) or not np.all(np.isfinite(yy)):
        return None

    xu = str(md.get("xunits") or md.get("XUNITS") or "").upper().replace(" ", "")
    is_um = "MICROM" in xu or "MICRON" in xu
    is_cm = (
        "1/CM" in xu
        or "CM-1" in xu
        or "CM^-1" in xu
        or "WAVENUMBER" in xu
        or xu == "CM-1"
    )
    if not is_um and not is_cm:
        mx = float(np.nanmax(wn))
        mn = float(np.nanmin(wn))
        if mx < 200 and mn > 0.2:
            is_um = True
        elif mx > 400:
            is_cm = True
        else:
            return None

    if is_um:
        if np.any(wn <= 0):
            return None
        w_cm = 10000.0 / wn
    else:
        w_cm = wn.astype(float, copy=True)

    order = np.argsort(w_cm)
    w_cm = w_cm[order]
    y_al = yy[order]
    span = float(w_cm[-1] - w_cm[0])
    if span < min_span_cm1:
        return None
    return w_cm, y_al


def featurize_mordred(mol: Any | None, calc: Any, *, n_cols: int) -> np.ndarray:
    if mol is None:
        return np.zeros(n_cols, dtype=float)
    try:
        from rdkit import Chem

        mol_use = Chem.Mol(mol)
        df = calc.pandas([mol_use])
        arr = df.fillna(0).to_numpy(dtype=float).ravel()
        # Coerce non-finite
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.size != n_cols:
            out = np.zeros(n_cols, dtype=float)
            out[: min(arr.size, n_cols)] = arr[:n_cols]
            return out
        return arr
    except Exception:
        return np.zeros(n_cols, dtype=float)


def build_feature_row(
    wn: np.ndarray,
    y: np.ndarray,
    md: dict[str, Any],
    *,
    calc: Any,
    mordred_dim: int,
) -> tuple[np.ndarray, bool]:
    """Single row: [spectral(14), rdkit(R), mordred(M), has_structure(1)]."""
    sp = featurize(np.asarray(wn, float), np.asarray(y, float)).ravel()
    mol = mol_from_metadata(md)
    rk, _ = featurize_rdkit(mol)
    mo = featurize_mordred(mol, calc, n_cols=mordred_dim)
    flag = 1.0 if mol is not None else 0.0
    x = np.concatenate([sp, rk, mo, np.asarray([flag], dtype=float)])
    return x, mol is not None


# Wavenumber windows match ``ml.ftir_fg_svm.featurize`` (14-D spectral block).
_SPECTRAL_CM1_BANDS: tuple[tuple[int, int], ...] = (
    (2500, 3700),
    (1650, 1820),
    (1200, 1700),
    (900, 1400),
    (650, 900),
)


_STABLE_EVIDENCE_FEATURE_NAMES: list[str] | None = None


def _stable_evidence_feature_names(feature_set: str | None = None) -> list[str]:
    """Stable column order for spectral+evidence feature blocks (delegates to ftir_evidence_features)."""
    return stable_evidence_feature_names(feature_set=feature_set or "spectral+evidence")


def spectral_feature_names() -> list[str]:
    """Human-readable names for the 14-D spectral block (same order as ``featurize``)."""
    names = [
        "spectral_y_global_mean_absorbance",
        "spectral_y_global_std_absorbance",
        "spectral_y_global_max_absorbance",
        "spectral_y_global_min_absorbance",
    ]
    for lo, hi in _SPECTRAL_CM1_BANDS:
        names.append(f"spectral_cm1_{lo}_{hi}_mean_absorbance")
        names.append(f"spectral_cm1_{lo}_{hi}_std_absorbance")
    return names


def spectral_band_reference() -> list[dict[str, str]]:
    """Short chemistry-oriented notes for each spectral window (cm⁻¹)."""
    notes = {
        (2500, 3700): "O-H / N-H stretches (alcohol, acid, amine, amide) and C-H stretches.",
        (1650, 1820): "C=O / amide I region (carbonyl, ester, acid salt, conjugated ketones).",
        (1200, 1700): "C-O / C-N / aromatic ring modes (ester, ether, amine, aromatic).",
        (900, 1400): "Fingerprint / out-of-plane bends (substitution patterns, aromatics).",
        (650, 900): "Low-frequency bends and heavy-atom stretches (halogenated aromatics, some inorganics).",
    }
    return [{"cm1_min": str(lo), "cm1_max": str(hi), "note": notes[(lo, hi)]} for lo, hi in _SPECTRAL_CM1_BANDS]


def structural_feature_names(meta: dict[str, Any]) -> list[str]:
    fn = meta.get("feature_names")
    if isinstance(fn, list) and fn:
        return [str(x) for x in fn]
    sp = spectral_feature_names()
    rk = [f"rdkit_{n}" for n in (meta.get("rdkit_names") or [])]
    mo = [f"mordred_{n}" for n in (meta.get("mordred_names") or [])]
    sm = [str(x) for x in (meta.get("smarts_feature_names") or [])]
    ev = [str(x) for x in (meta.get("evidence_feature_names") or [])]
    if sm or ev:
        return [*sp, *sm, *mo, *ev, "has_structure_flag"]
    return [*sp, *rk, *mo, "has_structure_flag"]


def _feature_block_ranges(meta: dict[str, Any]) -> dict[str, tuple[int, int]]:
    ns = int(meta.get("n_spectral", 14))
    nk = int(meta.get("n_smarts", 0))
    nr = int(meta.get("n_rdkit", len(meta.get("rdkit_names") or [])))
    nm = int(meta.get("n_mordred", len(meta.get("mordred_names") or [])))
    ne = int(meta.get("n_evidence", 0))
    s1 = ns
    s2 = s1 + nk
    s3 = s2 + nr
    s4 = s3 + nm
    s5 = s4 + ne
    s6 = s5 + int(meta.get("n_has_flag", 1))
    out: dict[str, tuple[int, int]] = {
        "spectral": (0, s1),
        "smarts": (s1, s2),
        "rdkit": (s2, s3),
        "mordred": (s3, s4),
        "has_structure": (s5, s6),
    }
    if ne > 0:
        out["evidence"] = (s4, s5)
    return out


def _ovr_linear_coef_vector(estimator: Any) -> np.ndarray:
    """One OvR hyperplane row (``n_features``); supports calibrated and raw ``LinearSVC``."""
    est = estimator
    if hasattr(est, "calibrated_classifiers_"):
        sub = est.calibrated_classifiers_[0]
        base = getattr(sub, "estimator", None) or getattr(sub, "base_estimator", None)
        if base is None or not hasattr(base, "coef_"):
            raise TypeError("Expected CalibratedClassifierCV wrapping LinearSVC")
        return np.asarray(base.coef_, dtype=float).ravel()
    if hasattr(est, "coef_"):
        return np.asarray(est.coef_, dtype=float).ravel()
    raise TypeError("Unsupported estimator for linear explainability")


def explain_structural_fg_model(artifact: dict[str, Any], *, topk: int = 12) -> dict[str, Any]:
    """
    Summarize **linear** explainability for each FG OvR head.

    Coefficients are for **StandardScaler-normalized** inputs (training pipeline). They describe
    the separating hyperplane of the inner ``LinearSVC``; ``CalibratedClassifierCV`` rescales
    margins to probabilities but keeps the same linear boundary in feature space.
    """
    clf = artifact.get("model")
    meta = artifact.get("meta") or {}
    labels = [str(x).lower() for x in artifact.get("labels") or []]
    if clf is None or not hasattr(clf, "estimators_"):
        raise ValueError("Artifact missing OneVsRestClassifier 'model'")

    names = structural_feature_names(meta)
    ranges = _feature_block_ranges(meta)
    w_all = np.vstack([_ovr_linear_coef_vector(est) for est in clf.estimators_])
    if w_all.shape[1] != len(names):
        names = [f"f{i}" for i in range(w_all.shape[1])]

    def block_l1(w: np.ndarray, sl: tuple[int, int]) -> float:
        a, b = sl
        seg = w[a:b]
        return float(np.sum(np.abs(seg)))

    per_label: dict[str, Any] = {}
    for j, lab in enumerate(labels):
        w = w_all[j]
        pos_i = np.where(w > 0)[0]
        neg_i = np.where(w < 0)[0]
        pos_i = pos_i[np.argsort(-w[pos_i])][:topk] if pos_i.size else pos_i
        neg_i = neg_i[np.argsort(w[neg_i])][:topk] if neg_i.size else neg_i
        top_pos = [{"feature": names[i], "coef_z": float(w[i])} for i in pos_i]
        top_neg = [{"feature": names[i], "coef_z": float(w[i])} for i in neg_i]
        per_label[lab] = {
            "top_positive_z": top_pos,
            "top_negative_z": top_neg,
            "block_abs_coef_mass": {
                k: round(block_l1(w, v), 6) for k, v in ranges.items()
            },
        }

    return {
        "n_features": int(w_all.shape[1]),
        "feature_order": (
            ["spectral_14", "smarts", "rdkit", "mordred", "evidence", "has_structure"]
            if int(meta.get("n_evidence", 0) or 0) > 0
            else ["spectral_14", "smarts", "rdkit", "mordred", "has_structure"]
        ),
        "spectral_windows_cm1_reference": spectral_band_reference(),
        "per_functional_group": per_label,
        "interpretation_note": (
            "Positive coef on a z-scored feature raises the score for that FG OvR head; "
            "negative lowers it. Spectral bands are absorbance statistics after your preprocessing; "
            "RDKit/Mordred columns are often zero when PubChem/structure was missing at training time, "
            "so the model may lean heavily on spectral bands for some labels. "
            "If the bundle includes SMARTS binary features, treat positive/negative weights as structural hints, "
            "not as independent FTIR evidence."
        ),
    }


def cmd_explain(args: argparse.Namespace) -> int:
    import joblib

    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}")
    artifact = joblib.load(model_path)
    rep = explain_structural_fg_model(artifact, topk=max(1, int(args.top)))
    rep["model"] = str(model_path)
    if str(getattr(args, "format", "json")).lower() == "text":
        print(rep["interpretation_note"])
        print("\nSpectral windows (cm-1):")
        for row in rep["spectral_windows_cm1_reference"]:
            print(f"  {row['cm1_min']}-{row['cm1_max']}: {row['note']}")
        for lab, block in rep["per_functional_group"].items():
            print(f"\n=== {lab} ===")
            print("  block |L1| mass (std features):", block["block_abs_coef_mass"])
            print("  top positive:")
            for r in block["top_positive_z"][:8]:
                print(f"    {r['coef_z']:+.5f}  {r['feature']}")
            print("  top negative:")
            for r in block["top_negative_z"][:8]:
                print(f"    {r['coef_z']:+.5f}  {r['feature']}")
        return 0
    print(json.dumps(rep, indent=2))
    return 0


def cmd_build_dataset(args: argparse.Namespace) -> int:
    db_path = Path(args.nist_index).resolve()
    if not db_path.is_file():
        raise SystemExit(f"SQLite not found: {db_path}")

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = Path(str(prefix) + ".npz")
    meta_path = Path(str(prefix) + ".meta.json")

    model_kind = str(getattr(args, "model_kind", MODEL_KIND_BASIC))
    calc: Any = None
    feature_set = _normalize_feature_set(getattr(args, "feature_set", None) or "legacy")
    label_source = str(getattr(args, "label_source", "existing") or "existing").lower()
    ontology_arg = str(getattr(args, "ontology", None) or "v3").lower()
    pipeline_version_arg = str(getattr(args, "pipeline_version", None) or "").strip()
    label_names = resolve_training_label_names(model_kind, label_source, ontology=ontology_arg)
    smarts_labels = list(label_names) if "smarts" in feature_set else []
    use_mordred = feature_set == "legacy" or "mordred" in feature_set
    if use_mordred:
        calc, mordred_names = make_mordred_calculator(max_descriptors=int(args.mordred_max))
    else:
        mordred_names = []
        calc = None
    mordred_dim = len(mordred_names)
    fe_names, f_counts = _build_feature_names_for_layout(
        feature_set=feature_set, mordred_names=mordred_names, smarts_labels=smarts_labels
    )

    pe = max(0, int(getattr(args, "progress_every", 0) or 0))
    print(
        "[build-dataset] Descriptor calc ready; scanning SQLite"
        + (f" (heartbeat every {pe} rows on stderr)" if pe > 0 else " (progress heartbeats disabled)"),
        file=sys.stderr,
        flush=True,
    )
    rk_names = [n for n, _ in _rdkit_descriptor_spec()]
    spectral_dim = 14
    min_label_pos = max(1, int(getattr(args, "min_label_positives", 50) or 50))

    leakage = str(label_source).lower() == "smarts" and "smarts" in feature_set
    if leakage:
        print(
            "WARNING: possible structural label/feature leakage: label-source=smarts and feature-set includes SMARTS. "
            "Training diagnostics may be optimistically biased. Prefer spectral-only X for publication models.",
            file=sys.stderr,
            flush=True,
        )

    pc_cache: dict[str, Any] = {}
    pc_cache_path = Path(args.pubchem_cache).resolve() if getattr(args, "enrich_pubchem", False) else None
    if pc_cache_path is not None:
        pc_cache = load_json_cache(pc_cache_path)

    pubchem_stats: dict[str, int] = {
        "cache_hit": 0,
        "network_resolved": 0,
        "network_miss": 0,
        "network_ambiguous": 0,
        "network_error": 0,
        "offline_miss": 0,
        "cas_attempt": 0,
        "cas_resolved": 0,
        "name_attempt": 0,
        "name_resolved": 0,
        "filename_attempt": 0,
        "filename_resolved": 0,
        "invalid_cas": 0,
        "stale_cache_retry": 0,
        "smiles_parse_failed": 0,
    }
    pubchem_debug_n = max(0, int(getattr(args, "pubchem_debug", 0) or 0))
    pubchem_debug_left = pubchem_debug_n

    conn = sqlite3.connect(str(db_path))
    try:
        try:
            cur = conn.execute(
                "SELECT reference_id, metadata_json, wn_json, y_json, source_path FROM spectra"
            )
            rows_iter = cur
        except sqlite3.OperationalError:
            cur = conn.execute(
                "SELECT reference_id, metadata_json, wn_json, y_json FROM spectra"
            )
            rows_iter = ((a, b, c, d, None) for (a, b, c, d) in cur)
        X_list: list[np.ndarray] = []
        Y_list: list[list[int]] = []
        ids: list[str] = []
        has_mol_flags: list[float] = []
        wn_raw_list: list[np.ndarray] = []
        y_raw_list: list[np.ndarray] = []
        store_raw = bool(getattr(args, "store_raw_spectra", False)) or (
            "peakcodebook" in feature_set or "deconv" in feature_set
        )
        skipped_parse = 0
        skipped_no_label = 0
        skipped_structure = 0
        skipped_invalid_ftir_spectrum = 0
        nist_ftir_cm1_prep = not bool(getattr(args, "no_nist_ftir_cm1_prep", False))

        progress_every = max(0, int(getattr(args, "progress_every", 0) or 0))
        pubchem_save_every = max(0, int(getattr(args, "pubchem_cache_save_every", 0) or 0))
        deconv_mode = str(getattr(args, "deconv_mode", "fast") or "fast").lower()
        if "deconv" in feature_set and deconv_mode != "off":
            from ml.ftir_deconvolution import reset_deconv_stats

            reset_deconv_stats()
            print(
                json.dumps({"build_dataset_deconv_mode": deconv_mode}),
                file=sys.stderr,
                flush=True,
            )

        for row_idx, (rid, mj, wj, yj, src_path) in enumerate(rows_iter):
            try:
                try:
                    md = json.loads(mj) if mj else {}
                    wn = np.asarray(json.loads(wj), dtype=float)
                    yy = np.asarray(json.loads(yj), dtype=float)
                except (json.JSONDecodeError, TypeError, ValueError):
                    skipped_parse += 1
                    continue

                if nist_ftir_cm1_prep:
                    pr = prepare_nist_ftir_cm1(wn, yy, md)
                    if pr is None:
                        skipped_invalid_ftir_spectrum += 1
                        continue
                    wn, yy = pr

                md_work = dict(md)
                if getattr(args, "enrich_pubchem", False):
                    md_work, pc_status = enrich_metadata_pubchem(
                        md_work,
                        path_hint=str(src_path or ""),
                        cache=pc_cache,
                        delay_s=float(args.pubchem_delay),
                        offline_cache_only=bool(args.pubchem_offline_only),
                        metrics=pubchem_stats,
                    )
                    pubchem_stats[pc_status] = pubchem_stats.get(pc_status, 0) + 1
                    if pubchem_debug_left > 0 and pc_status in (
                        "network_miss",
                        "network_ambiguous",
                        "network_error",
                    ):
                        ck_dbg = cache_key(md, str(src_path or ""))
                        preview = preview_pubchem_queries(md, str(src_path or ""))
                        dbg = {
                            "reference_id": str(rid),
                            "pc_status": pc_status,
                            "cache_key_hint": ck_dbg[:200],
                            "source_path_tail": str(src_path or "").replace("\\", "/").split("/")[-1][:120],
                            "metadata_keys": sorted(md.keys())[:48],
                            "cas_candidates_preview": preview.get("cas_candidates"),
                            "formula_raw": preview.get("formula_raw"),
                            "inchi_raw_snip": str(preview.get("inchi_raw") or "")[:120],
                            "inchikey_raw_snip": str(preview.get("inchikey_raw") or "")[:80],
                            "title_snip": str(md.get("title") or md.get("TITLE") or "")[:120],
                            "name_snip": str(md.get("name") or md.get("NAME") or "")[:120],
                            "preview_queries": preview,
                            "last_attempts": (md_work.get("pubchem_attempts") or [])[-4:],
                        }
                        print(json.dumps({"pubchem_debug": dbg}), file=sys.stderr, flush=True)
                        pubchem_debug_left -= 1

                md_work = canonicalize_structure_metadata(md_work)
                sp_ok, sp_reason = validate_structure_smiles(md_work)
                if md_work.get("SMILES") and not sp_ok:
                    md_work["structure_parse_ok"] = False
                    md_work["structure_parse_reason"] = sp_reason
                    pubchem_stats["smiles_parse_failed"] = pubchem_stats.get("smiles_parse_failed", 0) + 1
                elif md_work.get("SMILES"):
                    md_work["structure_parse_ok"] = True
                if pc_cache_path is not None and getattr(args, "enrich_pubchem", False):
                    ck = cache_key(md, str(src_path or ""))
                    apply_canonical_structure_to_cache(pc_cache, ck, md_work)

                mol = mol_from_metadata(md_work)
                if args.require_structure and mol is None:
                    skipped_structure += 1
                    continue

                ev_for_row = None
                if "evidence" in feature_set:
                    from ml.ftir_evidence import extract_spectral_evidence

                    ev_cfg = {"ontology": "v4"} if ontology_arg == "v4" else {}
                    ev_for_row = extract_spectral_evidence(wn, yy, peaks=None, config=ev_cfg)

                yvec = _infer_y_vector(
                    md_work,
                    mol,
                    model_kind=model_kind,
                    label_source=label_source,
                    label_names=label_names,
                    ontology=ontology_arg,
                )
                if sum(yvec) == 0:
                    skipped_no_label += 1
                    continue

                x_row, hm = build_feature_row_layout(
                    wn,
                    yy,
                    md_work,
                    feature_set=feature_set,
                    calc=calc,
                    mordred_dim=mordred_dim,
                    smarts_feature_labels=smarts_labels,
                    evidence=ev_for_row,
                    deconv_mode=deconv_mode if "deconv" in feature_set else "off",
                )
                X_list.append(x_row)
                Y_list.append(yvec)
                ids.append(str(rid))
                has_mol_flags.append(1.0 if hm else 0.0)
                if store_raw:
                    wn_raw_list.append(np.asarray(wn, dtype=float).copy())
                    y_raw_list.append(np.asarray(yy, dtype=float).copy())
            finally:
                if (
                    pc_cache_path is not None
                    and getattr(args, "enrich_pubchem", False)
                    and pubchem_save_every > 0
                    and (row_idx + 1) % pubchem_save_every == 0
                ):
                    try:
                        save_json_cache(pc_cache_path, pc_cache)
                        print(
                            "[build-dataset] pubchem_cache_checkpoint "
                            f"spectra_rows={row_idx + 1} cache_keys={len(pc_cache)} path={pc_cache_path}",
                            file=sys.stderr,
                            flush=True,
                        )
                    except OSError as exc:
                        print(
                            f"[build-dataset] pubchem_cache_checkpoint FAILED: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                if progress_every > 0 and (row_idx + 1) % progress_every == 0:
                    print(
                        "[build-dataset] "
                        f"spectra_rows={row_idx + 1} kept={len(X_list)} "
                        f"skipped parse/invalid_ftir/no_fg/require_structure="
                        f"{skipped_parse}/{skipped_invalid_ftir_spectrum}/{skipped_no_label}/{skipped_structure} "
                        f"pubchem={pubchem_stats}",
                        file=sys.stderr,
                        flush=True,
                    )

        if len(X_list) < 5:
            raise SystemExit(
                f"Too few rows ({len(X_list)}). "
                f"skipped_parse={skipped_parse} skipped_invalid_ftir_spectrum={skipped_invalid_ftir_spectrum} "
                f"skipped_no_label={skipped_no_label} skipped_structure={skipped_structure}"
            )

        X = np.vstack(X_list)
        Y = np.asarray(Y_list, dtype=int)

        label_counts = {lab: int(Y[:, j].sum()) for j, lab in enumerate(label_names)}
        dropped_build: list[str] = []
        old_label_names = list(label_names)
        label_names, Y, dropped_build, _ = filter_labels_by_counts(
            label_names, Y, min_positives=min_label_pos
        )
        final_counts = {lab: int(Y[:, j].sum()) for j, lab in enumerate(label_names)}
        low_support_labels = [
            {"label": lab, "count": final_counts[lab], "low_support_label": True}
            for lab in label_names
            if final_counts[lab] < 2 * min_label_pos
        ]
        if dropped_build:
            print(json.dumps({"dropped_labels_at_build": dropped_build, "label_counts_before_filter": label_counts}, indent=2))

        if smarts_labels and "smarts" in feature_set:
            n_sp = int(f_counts["n_spectral"])
            n_sm_old = len(old_label_names)
            n_mo = int(f_counts["n_mordred"])
            if n_sm_old and X.shape[1] >= n_sp + n_sm_old + n_mo + 1:
                keep_idx = [old_label_names.index(lab) for lab in label_names]
                rest_before = X[:, :n_sp]
                sm_old = X[:, n_sp : n_sp + n_sm_old]
                mo_part = X[:, n_sp + n_sm_old : n_sp + n_sm_old + n_mo]
                flag_part = X[:, -1:]
                sm_new = sm_old[:, keep_idx]
                X = np.hstack([rest_before, sm_new, mo_part, flag_part])
            smarts_labels = list(label_names)
            fe_names, f_counts = _build_feature_names_for_layout(
                feature_set=feature_set, mordred_names=mordred_names, smarts_labels=smarts_labels
            )

        if getattr(args, "enrich_pubchem", False) and int(X.shape[0]) > 0:
            ntot = int(X.shape[0])
            nr = int(pubchem_stats.get("network_resolved", 0))
            na = int(pubchem_stats.get("network_ambiguous", 0))
            nm = int(pubchem_stats.get("network_miss", 0))
            ne = int(pubchem_stats.get("network_error", 0))
            ch = int(pubchem_stats.get("cache_hit", 0))
            spf = int(pubchem_stats.get("smiles_parse_failed", 0))
            pubchem_stats["resolution_rate"] = round(nr / ntot, 6)
            pubchem_stats["ambiguity_rate"] = round(na / ntot, 6)
            pubchem_stats["network_miss_rate"] = round(nm / ntot, 6)
            pubchem_stats["network_failure_rate"] = round(ne / ntot, 6)
            pubchem_stats["cache_hit_rate"] = round(ch / ntot, 6)
            pubchem_stats["smiles_parse_fail_rate"] = round(spf / ntot, 6)

        npz_kw: dict[str, Any] = {
            "X": X,
            "Y": Y,
            "reference_id": np.asarray(ids, dtype=object),
            "has_mol": np.asarray(has_mol_flags, dtype=float),
        }
        if store_raw and wn_raw_list:
            npz_kw["Wn_raw"] = np.asarray(wn_raw_list, dtype=object)
            npz_kw["Yspec_raw"] = np.asarray(y_raw_list, dtype=object)
        np.savez_compressed(npz_path, **npz_kw)

        n_rdkit_meta = len(rk_names) if feature_set == "legacy" else 0
        rk_meta = rk_names if feature_set == "legacy" else []
        meta = {
            "pipeline_version": pipeline_version_arg or None,
            "ontology": ontology_arg,
            "model_kind": model_kind,
            "label_source": label_source,
            "feature_set": feature_set,
            "label_names": label_names,
            "label_counts": {lab: int(Y[:, j].sum()) for j, lab in enumerate(label_names)},
            "dropped_labels": dropped_build,
            "dropped_label_records": [
                {"label": (s.split(":", 1)[0] if ":" in s else s), "reason": (s.split(":", 1)[1] if ":" in s else s)}
                for s in dropped_build
            ],
            "possible_structural_label_feature_leakage": bool(leakage),
            "label_feature_smarts_leakage": bool(leakage),
            "smarts_library_version_hash": smarts_library_version_hash(),
            "min_label_positives": min_label_pos,
            "n_spectral": spectral_dim,
            "n_smarts": int(f_counts.get("n_smarts", 0)),
            "n_evidence": int(f_counts.get("n_evidence", 0)),
            "n_rdkit": n_rdkit_meta,
            "n_mordred": mordred_dim,
            "n_has_flag": 1,
            "feature_dim": int(X.shape[1]),
            "feature_names": fe_names,
            "evidence_feature_version": evidence_feature_version_for_feature_set(feature_set)
            if "evidence" in feature_set
            else None,
            "low_support_labels": low_support_labels,
            "evidence_feature_names": list(_stable_evidence_feature_names(feature_set=feature_set))
            if "evidence" in feature_set
            else [],
            "smarts_feature_names": [f"smarts_{x}" for x in smarts_labels],
            "rdkit_names": rk_meta,
            "mordred_names": mordred_names,
            "mordred_max_requested": int(args.mordred_max),
            "mordred_ignore_3d": True,
            "require_structure": bool(args.require_structure),
            "n_rows": int(X.shape[0]),
            "n_with_structure": int(sum(has_mol_flags)),
            "skipped_parse": skipped_parse,
            "skipped_invalid_ftir_spectrum": skipped_invalid_ftir_spectrum,
            "skipped_no_label": skipped_no_label,
            "skipped_structure": skipped_structure,
            "nist_ftir_cm1_prep_applied": nist_ftir_cm1_prep,
            "sqlite": str(db_path),
            "pubchem_enriched": bool(getattr(args, "enrich_pubchem", False)),
            "pubchem_cache": str(pc_cache_path) if pc_cache_path else None,
            "pubchem_cache_save_every": int(pubchem_save_every)
            if getattr(args, "enrich_pubchem", False) and pc_cache_path
            else 0,
            "pubchem_stats": pubchem_stats if getattr(args, "enrich_pubchem", False) else {},
            "smiles_canonicalization": "rdkit_MolToSmiles_canonical_isomeric",
            "store_raw_spectra": store_raw,
        }
        if "peakcodebook" in feature_set:
            from ml.ftir_peakcodebook_features import peakcodebook_meta

            meta.update(peakcodebook_meta())
        if "deconv" in feature_set:
            from ml.ftir_deconvolution import deconv_meta, get_deconv_stats

            meta.update(deconv_meta(mode=deconv_mode))
            meta.update(get_deconv_stats().to_dict())
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        if pc_cache_path is not None:
            save_json_cache(pc_cache_path, pc_cache)

        print(json.dumps({"wrote_npz": str(npz_path), "wrote_meta": str(meta_path), **{k: meta[k] for k in ("n_rows", "n_with_structure", "feature_dim")}}, indent=2))
        if getattr(args, "enrich_pubchem", False):
            print(json.dumps({"pubchem_stats": pubchem_stats, "pubchem_cache_saved": str(pc_cache_path)}, indent=2))
            print(
                json.dumps({"pubchem_summary_rates": {k: pubchem_stats[k] for k in (
                    "resolution_rate", "ambiguity_rate", "network_miss_rate",
                    "network_failure_rate", "cache_hit_rate", "smiles_parse_fail_rate",
                )}}, indent=2),
                file=sys.stderr,
                flush=True,
            )
    finally:
        conn.close()

    if getattr(args, "enrich_pubchem", False):
        nr = int(pubchem_stats.get("network_resolved", 0))
        nm = int(pubchem_stats.get("network_miss", 0))
        na = int(pubchem_stats.get("network_ambiguous", 0))
        if nr == 0 and (nm > 0 or na > 0):
            print(
                "Warning: --enrich-pubchem resolved 0 structures from network hits. "
                "Common causes: metadata has no CAS (NIST rows often use title-only names), "
                "name queries miss PubChem, or Phase-3 thresholds marked candidates ambiguous. "
                "Re-run with --pubchem-debug 15 for sample rows; dry-run: "
                "python -m ml.structural_fg_svm pubchem-resolve --title \"...\". "
                "HTTPS smoke test: python -m ml.structural_fg_svm pubchem-smoke --verbose",
                file=sys.stderr,
                flush=True,
            )
    return 0


def cmd_pubchem_smoke(args: argparse.Namespace) -> int:
    """Quick check that PubChem PUG REST returns data (same HTTP stack as build-dataset)."""
    from ml.pubchem_structure_lookup import (
        diagnose_pubchem_cas_xref,
        resolve_pubchem,
        resolve_pubchem_phase1,
    )

    cas_q = str(getattr(args, "cas", None) or "64-17-5").strip()
    delay_s = float(args.pubchem_delay)
    r = resolve_pubchem(cas=cas_q, name=None, delay_s=delay_s)
    summary: dict[str, Any] = {
        "cas_query": cas_q,
        "resolved": r is not None,
        "SMILES": (r or {}).get("SMILES"),
        "INCHI": (r or {}).get("INCHI"),
        "pubchem_cid": (r or {}).get("pubchem_cid"),
    }
    has_struct = bool(
        r
        and (
            str((r.get("SMILES") or "")).strip()
            or str((r.get("INCHI") or "")).strip()
        )
    )
    if getattr(args, "verbose", False) or not has_struct:
        summary["diagnose_cas_xref"] = diagnose_pubchem_cas_xref(cas_q, delay_s=0.0)
    if getattr(args, "show_candidates", False):
        md_dbg: dict[str, Any] = {"cas": cas_q, "title": cas_q, "name": cas_q}
        _resolved_dbg, attempts, _tf = resolve_pubchem_phase1(md=md_dbg, path_hint=None, delay_s=delay_s)
        max_attempts = int(getattr(args, "max_candidate_attempts", 3))
        max_ranked = int(getattr(args, "max_ranked_candidates", 3))
        compact_attempts: list[dict[str, Any]] = []
        for a in attempts[: max(1, max_attempts)]:
            item: dict[str, Any] = {
                "query_type": a.get("query_type"),
                "query": a.get("query"),
                "source": a.get("source"),
                "n_cids": a.get("n_cids"),
                "decision": a.get("decision"),
            }
            ranked = a.get("ranked_candidates")
            if isinstance(ranked, list):
                item["ranked_candidates"] = ranked[: max(1, max_ranked)]
            compact_attempts.append(item)
        summary["phase2_candidate_report"] = compact_attempts
    print(json.dumps(summary, indent=2))
    if not has_struct:
        print(
            f"PubChem smoke test FAILED for CAS {cas_q!r}. "
            "TLS: pip install truststore (uses Windows cert store) or certifi. "
            "See diagnose_cas_xref.http_error in the JSON. "
            "Proxy: set HTTPS_PROXY. Broken MITM PKI: PUBCHEM_INSECURE_SSL=1 (insecure).",
            file=sys.stderr,
        )
        return 1
    print("PubChem smoke test OK.", file=sys.stderr)
    return 0


def cmd_pubchem_resolve(args: argparse.Namespace) -> int:
    """Debug resolver: show candidate queries + ranked attempts (same stack as build-dataset)."""
    from ml.pubchem_structure_lookup import resolve_pubchem_phase1

    md: dict[str, Any] = {}
    if getattr(args, "cas", None) and str(args.cas).strip():
        md["cas"] = str(args.cas).strip()
    if getattr(args, "title", None) and str(args.title).strip():
        t = str(args.title).strip()
        md["title"] = t
        md["name"] = t
    path = str(getattr(args, "path", "") or "").strip() or None
    delay_s = float(getattr(args, "pubchem_delay", 0.25))
    preview = preview_pubchem_queries(md, path)
    resolved, attempts, transport_failed = resolve_pubchem_phase1(md=md, path_hint=path, delay_s=delay_s)
    print(
        json.dumps(
            {
                "preview_queries": preview,
                "resolved": resolved,
                "resolved_ok": resolved is not None,
                "transport_failed": transport_failed,
                "attempts": attempts[:16],
            },
            indent=2,
        )
    )
    return 0


def _resolve_model_out_path_v3(
    model_out: str | None,
    *,
    model_kind: str,
    label_source: str,
    train_ts: str,
    out_dir: str | None,
    pipeline_version: str = "",
    ontology: str = "",
) -> Path:
    if model_out:
        return Path(model_out)
    pv = str(pipeline_version or "").strip().lower().replace("-", "_")
    ls = str(label_source).lower()
    ont = str(ontology or "").strip().lower()
    if pv == "v4_ontology" or ont == "v4":
        stem = f"struct_fg_{model_kind}_v4_ontology_{train_ts}"
    elif pv == "v3_guarded":
        stem = (
            f"struct_fg_{model_kind}_smarts_v3_guarded_{train_ts}"
            if ls == "smarts"
            else f"struct_fg_{model_kind}_v3_guarded_{train_ts}"
        )
    elif ls == "smarts":
        stem = f"struct_fg_{model_kind}_smarts_{train_ts}"
    else:
        stem = f"struct_fg_{model_kind}_{train_ts}"
    if out_dir:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{stem}.joblib"
    root = Path(__file__).resolve().parent / "runs"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{stem}.joblib"


def _write_training_report_html(
    path: Path,
    *,
    title: str,
    training_meta: dict[str, Any],
    interpret_summary: dict[str, Any] | None,
) -> None:
    import html as html_mod

    path.parent.mkdir(parents=True, exist_ok=True)
    inter_html = ""
    if interpret_summary:
        inter_html = f"<pre>{html_mod.escape(json.dumps(interpret_summary, indent=2)[:12000])}</pre>"
    dropped = training_meta.get("dropped_labels") or []
    dropped_ul = "<ul>" + "".join(f"<li>{html_mod.escape(str(x))}</li>" for x in dropped[:200]) + "</ul>"
    body = f"""<!doctype html><html><head><meta charset="utf-8"/><title>{html_mod.escape(title)}</title>
<style>body{{font-family:system-ui;margin:24px;max-width:960px}} table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #ccc;padding:6px}}</style>
</head><body>
<h1>{html_mod.escape(title)}</h1>
<p>Timestamp: {html_mod.escape(str(training_meta.get('training_timestamp_utc')))}</p>
<h2>Split</h2>
<p>{html_mod.escape(json.dumps(training_meta.get('split'), indent=2))}</p>
<h2>Calibration</h2>
<p>{html_mod.escape(json.dumps(training_meta.get('calibration'), indent=2))}</p>
<h2>Dropped labels</h2>
{dropped_ul}
<h2>Warnings</h2>
<ul>{"".join(f"<li>{html_mod.escape(str(w))}</li>" for w in (training_meta.get('warnings') or [])[:80])}</ul>
<h2>Global interpretability (linear)</h2>
{inter_html}
</body></html>"""
    path.write_text(body, encoding="utf-8")


def cmd_train(args: argparse.Namespace) -> int:
    import csv

    _require_sklearn()
    import joblib
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_fscore_support, roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    prefix = Path(args.dataset_prefix)
    npz_path = Path(str(prefix) + ".npz")
    meta_path = Path(str(prefix) + ".meta.json")
    if not npz_path.is_file():
        raise SystemExit(f"Dataset not found: {npz_path}")
    meta0 = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = dict(meta0)
    label_names: list[str] = list(meta["label_names"])
    model_kind = str(getattr(args, "model_kind", None) or meta.get("model_kind") or MODEL_KIND_BASIC)
    ls_arg = str(getattr(args, "label_source", "") or "").strip().lower()
    label_source = ls_arg if ls_arg else str(meta.get("label_source") or "existing").lower()
    min_label_pos = max(1, int(getattr(args, "min_label_positives", meta.get("min_label_positives", 20)) or 20))
    test_size = float(getattr(args, "test_size", 0.2) or 0.2)
    split_mode = str(getattr(args, "split", "molecule") or "molecule").lower()
    random_state = int(getattr(args, "random_state", 13) or 13)
    calibration = str(getattr(args, "calibration", "sigmoid") or "sigmoid").lower()
    out_dir_arg = str(getattr(args, "out", "") or "").strip()

    data = np.load(npz_path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=float)
    Y = np.asarray(data["Y"], dtype=int)
    ref_ids = np.asarray(data["reference_id"], dtype=object) if "reference_id" in data.files else None
    has_mol_all = np.asarray(data["has_mol"], dtype=float) if "has_mol" in data.files else None
    wn_raw_all = data["Wn_raw"] if "Wn_raw" in data.files else None
    yspec_raw_all = data["Yspec_raw"] if "Yspec_raw" in data.files else None
    feature_set_train = _normalize_feature_set(
        str(getattr(args, "feature_set", "") or meta.get("feature_set") or "legacy")
    )
    augment_mode = str(getattr(args, "augment", "none") or "none").lower()
    threshold_objective = str(getattr(args, "threshold_objective", "balanced_guarded") or "balanced_guarded").lower()
    hn_weight = float(getattr(args, "hard_negative_weight", 1.5) or 1.0)
    run_benchmark = bool(getattr(args, "benchmark", False))
    no_update_latest = bool(getattr(args, "no_update_latest", False))

    if getattr(args, "remap_legacy_labels", False) or (
        meta.get("model_kind") is None and model_kind == MODEL_KIND_BASIC and "label_source" not in meta
    ):
        from ml.fg_label_configs import remap_legacy_y_columns

        label_names, Y = remap_legacy_y_columns(label_names, Y, target_kind=MODEL_KIND_BASIC)
        meta["remapped_from_legacy"] = True

    old_names_pre = list(label_names)
    label_counts_before = {lab: int(Y[:, j].sum()) for j, lab in enumerate(label_names)}
    label_names, Y, dropped_train, label_counts = filter_labels_by_counts(
        label_names, Y, min_positives=min_label_pos
    )
    if dropped_train:
        X = trim_x_after_label_filter(X, meta=meta, old_label_names=old_names_pre, new_label_names=label_names)
        smarts_labs = list(label_names) if "smarts" in _normalize_feature_set(meta.get("feature_set", "legacy")) else []
        fe_names, f_counts = _build_feature_names_for_layout(
            feature_set=str(meta.get("feature_set", "legacy")),
            mordred_names=list(meta.get("mordred_names") or []),
            smarts_labels=smarts_labs,
        )
        meta["feature_names"] = fe_names
        meta["n_smarts"] = int(f_counts.get("n_smarts", 0))
        meta["feature_dim"] = int(X.shape[1])
        print(json.dumps({"dropped_labels_at_train": dropped_train, "label_counts": label_counts_before}, indent=2))

    if Y.shape[1] == 0:
        raise SystemExit("No label columns left after variance filter.")

    if ref_ids is not None and split_mode == "molecule":
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        tr_idx, te_idx = next(gss.split(X, Y, groups=ref_ids))
    else:
        tr_idx, te_idx = train_test_split(
            np.arange(X.shape[0]),
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

    X_train, X_test = X[tr_idx], X[te_idx]
    Y_train, Y_test = Y[tr_idx], Y[te_idx]
    n_augmented = 0
    if augment_mode != "none" and wn_raw_all is not None and yspec_raw_all is not None:
        from ml.train_row_builder import expand_train_with_augmentation

        wn_tr = [np.asarray(wn_raw_all[i], dtype=float) for i in tr_idx]
        y_tr = [np.asarray(yspec_raw_all[i], dtype=float) for i in tr_idx]
        hm_tr = has_mol_all[tr_idx] if has_mol_all is not None else np.ones(len(tr_idx))
        X_train, Y_train, n_augmented = expand_train_with_augmentation(
            X_train,
            Y_train,
            wn_tr,
            y_tr,
            hm_tr,
            feature_set=feature_set_train,
            mode=augment_mode,
            random_state=random_state,
        )
    elif augment_mode != "none":
        warn_list_pre = list(meta.get("warnings") or [])
        warn_list_pre.append(
            "Augmentation requested but dataset lacks Wn_raw/Yspec_raw; rebuild with --store-raw-spectra or peakcodebook feature-set."
        )
        meta["warnings"] = warn_list_pre

    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X_train)
    Xs_te = scaler.transform(X_test)

    hn_weight_applied = False
    if hn_weight > 1.0:
        from ml.specialist_configs import SPECIALIST_HARD_NEGATIVES, is_specialist_model_kind
        from ml.training_diagnostics import HARD_NEGATIVE_PAIRS, compute_hard_negative_sample_weights

        hn_pairs = (
            SPECIALIST_HARD_NEGATIVES.get(model_kind, HARD_NEGATIVE_PAIRS)
            if is_specialist_model_kind(model_kind)
            else HARD_NEGATIVE_PAIRS
        )
        row_w = compute_hard_negative_sample_weights(
            label_names, Y_train, weight=hn_weight, hard_negative_pairs=hn_pairs
        )
        dup_mask = row_w >= (hn_weight - 1e-6)
        if np.any(dup_mask):
            X_train = np.vstack([X_train, X_train[dup_mask]])
            Y_train = np.vstack([Y_train, Y_train[dup_mask]])
            hn_weight_applied = True
            Xs_tr = scaler.fit_transform(X_train)

    n_jobs = int(args.n_jobs)
    if n_jobs == 0:
        n_jobs = 1
    n_features_train = int(X_train.shape[1])
    memory_safe = bool(getattr(args, "memory_safe", False)) or (
        n_features_train >= int(getattr(args, "memory_safe_feature_threshold", 500) or 500)
    )
    if memory_safe:
        n_jobs = 1
        cal_n_jobs = 1
        n_cv = min(3, max(2, int(X_train.shape[0] // 2500)))
        warn_list_pre = list(meta.get("warnings") or [])
        warn_list_pre.append(
            f"memory_safe training: n_jobs=1, calibration_cv={n_cv}, sequential OvR "
            f"(n_features={n_features_train})."
        )
        meta["warnings"] = warn_list_pre
    else:
        cal_n_jobs = max(1, n_jobs)
        n_cv = min(5, max(2, int(X_train.shape[0] // 2000)))

    skip_cal = bool(getattr(args, "skip_calibration", False))
    if memory_safe and skip_cal and calibration in ("sigmoid", "isotonic"):
        warn_list_pre = list(meta.get("warnings") or [])
        warn_list_pre.append("skip_calibration: using LinearSVC decision scores (memory-safe benchmark).")
        meta["warnings"] = warn_list_pre
        calibration = "none"

    base = LinearSVC(
        class_weight="balanced",
        random_state=random_state,
        max_iter=200_000,
        dual=False,
    )

    cal_meta: dict[str, Any] = {
        "method": calibration,
        "fitted": False,
        "cv": n_cv,
        "strategy": f"OneVsRest + per-label calibration ({calibration})"
        if calibration in ("sigmoid", "isotonic")
        else "OneVsRest LinearSVC (uncalibrated decision scores)",
    }
    warn_list: list[str] = list(meta.get("warnings") or [])
    if calibration == "isotonic" and X_train.shape[0] < 800:
        warn_list.append("Isotonic calibration requested but training split is small; falling back to sigmoid.")
        calibration = "sigmoid"
    if calibration in ("sigmoid", "isotonic") and X_train.shape[0] < 200:
        warn_list.append("Calibration may be unreliable: very few training rows after split.")

    if calibration in ("sigmoid", "isotonic"):
        clf = OneVsRestClassifier(
            CalibratedClassifierCV(base, method=calibration, cv=n_cv, n_jobs=cal_n_jobs),
            n_jobs=1,
        )
        clf.fit(Xs_tr, Y_train)
        cal_meta["fitted"] = True
        cal_meta["method_effective"] = calibration
        score_kind = "calibrated_probability"
    else:
        clf = OneVsRestClassifier(base, n_jobs=1)
        clf.fit(Xs_tr, Y_train)
        cal_meta["method_effective"] = "none"
        if skip_cal:
            cal_meta["calibration_status"] = "skipped_memory_safe"
            cal_meta["strategy"] = "OneVsRest LinearSVC (uncalibrated; memory-safe skip_calibration)"
        score_kind = "svm_decision_score"

    train_ts = _utc_timestamp()
    pipeline_version = str(getattr(args, "pipeline_version", "") or "").strip()
    ontology_train = str(getattr(args, "ontology", None) or meta.get("ontology") or "v3").lower()
    model_out = _resolve_model_out_path_v3(
        str(getattr(args, "model_out", "") or "") or None,
        model_kind=model_kind,
        label_source=label_source,
        train_ts=train_ts,
        out_dir=out_dir_arg or None,
        pipeline_version=pipeline_version,
        ontology=ontology_train,
    )
    model_out.parent.mkdir(parents=True, exist_ok=True)
    eval_prefix = model_out.with_suffix("")

    # --- metrics on test ---
    label_metrics_rows: list[dict[str, Any]] = []
    if score_kind == "calibrated_probability":
        pred = clf.predict_proba(Xs_te)
        if isinstance(pred, list):
            pred_m = np.column_stack(
                [np.asarray(r)[:, 1] if np.asarray(r).ndim == 2 else np.asarray(r).ravel() for r in pred]
            )
        else:
            pred_m = np.asarray(pred)
    else:
        raw_df = clf.decision_function(Xs_te)
        pred_m = np.asarray(raw_df, dtype=float)
        if pred_m.ndim == 1:
            pred_m = pred_m.reshape(-1, 1)

    from ml.specialist_configs import SPECIALIST_HARD_NEGATIVES, is_specialist_model_kind
    from ml.training_diagnostics import (
        hard_negative_false_positive_report,
        hard_negative_metrics_by_label,
        tune_per_label_thresholds,
    )

    label_supports = {lab: int(Y[:, j].sum()) for j, lab in enumerate(label_names)}
    thresholds, threshold_rows = tune_per_label_thresholds(
        label_names,
        Y_test,
        pred_m,
        score_kind=score_kind,
        objective=threshold_objective,
        model_kind=model_kind,
        label_supports=label_supports,
    )
    hn_summary: dict[str, Any] = {}
    hn_pairs_use = (
        SPECIALIST_HARD_NEGATIVES.get(model_kind, None)
        if is_specialist_model_kind(model_kind)
        else None
    )
    if str(getattr(args, "hard_negative_mode", "off") or "off").lower() in ("on", "true", "1"):
        hn_summary = hard_negative_false_positive_report(
            label_names,
            Y_test,
            pred_m,
            thresholds,
            out_path=Path(str(eval_prefix) + "_hard_negative_false_positives.csv"),
        )
        hard_negative_metrics_by_label(
            label_names,
            Y_test,
            pred_m,
            thresholds,
            out_path=Path(str(eval_prefix) + "_hard_negative_metrics_by_label.csv"),
            hard_negative_pairs=hn_pairs_use,
        )

    label_metrics_rows = []
    for j, lab in enumerate(label_names):
        y_t = Y_test[:, j]
        s = pred_m[:, j]
        thr = thresholds[j]
        pr, rc, f1, _sup = precision_recall_fscore_support(
            y_t, (s >= thr).astype(int), average="binary", zero_division=0
        )
        sup_pos = int(np.sum(y_t))
        ap = None
        try:
            if len(np.unique(y_t)) > 1:
                ap = float(average_precision_score(y_t, s))
        except Exception:
            ap = None
        roc = None
        try:
            if len(np.unique(y_t)) > 1:
                roc = float(roc_auc_score(y_t, s))
        except Exception:
            roc = None
        brier = None
        if score_kind == "calibrated_probability" and len(np.unique(y_t)) > 1:
            try:
                brier = float(brier_score_loss(y_t, np.clip(s, 0.0, 1.0)))
            except Exception:
                brier = None
        label_metrics_rows.append(
            {
                "label": lab,
                "precision": float(pr),
                "recall": float(rc),
                "f1": float(f1),
                "support": sup_pos,
                "average_precision": ap,
                "roc_auc": roc,
                "brier": brier,
                "threshold": thr,
            }
        )

    metrics_summary = {
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X.shape[1]),
        "n_labels": int(Y.shape[1]),
        "score_kind": score_kind,
        "macro_f1": float(
            sum(r["f1"] for r in label_metrics_rows) / max(len(label_metrics_rows), 1)
        ),
    }

    csv_labels = Path(str(eval_prefix) + "_label_metrics.csv")
    with csv_labels.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(label_metrics_rows[0].keys()))
        w.writeheader()
        for row in label_metrics_rows:
            w.writerow(row)

    with Path(str(eval_prefix) + "_metrics_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for k, v in metrics_summary.items():
            w.writerow([k, v])

    with Path(str(eval_prefix) + "_threshold_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        th_base = [
            "label",
            "threshold",
            "precision",
            "recall",
            "f1",
            "threshold_objective",
            "high_risk_precision_bias",
            "note",
        ]
        th_extra: list[str] = []
        for row in threshold_rows:
            for k in row:
                if k not in th_base and k not in th_extra:
                    th_extra.append(k)
        th_fields = th_base + th_extra
        w = csv.DictWriter(fh, fieldnames=th_fields, extrasaction="ignore")
        w.writeheader()
        for row in threshold_rows:
            w.writerow(row)

    with Path(str(eval_prefix) + "_confusion_or_threshold_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "threshold", "tp", "fp", "tn", "fn"])
        for j, lab in enumerate(label_names):
            y_t = Y_test[:, j]
            s = pred_m[:, j]
            thr = thresholds[j]
            pred_bin = (s >= thr).astype(int)
            tp = int(np.sum((pred_bin == 1) & (y_t == 1)))
            fp = int(np.sum((pred_bin == 1) & (y_t == 0)))
            tn = int(np.sum((pred_bin == 0) & (y_t == 0)))
            fn = int(np.sum((pred_bin == 0) & (y_t == 1)))
            w.writerow([lab, thr, tp, fp, tn, fn])

    dropped_all = list(meta.get("dropped_labels") or []) + list(dropped_train)
    dropped_records = [
        {"raw": s, "label": (s.split(":", 1)[0] if ":" in str(s) else s), "reason": (s.split(":", 1)[1] if ":" in str(s) else "dropped")}
        for s in dropped_all
    ]
    Path(str(eval_prefix) + "_dropped_labels.json").write_text(
        json.dumps({"dropped": dropped_records}, indent=2),
        encoding="utf-8",
    )

    imbalance = {lab: {"positives": int(Y[:, j].sum()), "negatives": int(Y.shape[0] - int(Y[:, j].sum()))} for j, lab in enumerate(label_names)}

    training_meta_out: dict[str, Any] = {
        "training_timestamp_utc": train_ts,
        "dataset_prefix": str(prefix),
        "model_kind": model_kind,
        "label_source": label_source,
        "feature_set": meta.get("feature_set"),
        "calibration": cal_meta,
        "split": {
            "mode": split_mode,
            "test_size": test_size,
            "random_state": random_state,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "train_molecules": int(len(np.unique(ref_ids[tr_idx]))) if ref_ids is not None else None,
            "test_molecules": int(len(np.unique(ref_ids[te_idx]))) if ref_ids is not None else None,
        },
        "label_counts": {lab: int(Y[:, j].sum()) for j, lab in enumerate(label_names)},
        "dropped_labels": dropped_all,
        "class_imbalance": imbalance,
        "ml_score_kind": score_kind,
        "possible_structural_label_feature_leakage": bool(meta.get("possible_structural_label_feature_leakage")),
        "smarts_library_version_hash": meta.get("smarts_library_version_hash"),
        "package_versions": _package_versions_meta(),
        "git_hash": _git_hash_short(),
        "warnings": warn_list,
        "metrics_summary": metrics_summary,
        "per_label_thresholds": {lab: float(t) for lab, t in zip(label_names, thresholds)},
        "pipeline_version": str(getattr(args, "pipeline_version", "") or "").strip() or None,
        "ontology": ontology_train,
        "hard_negative_mode": str(getattr(args, "hard_negative_mode", "off") or "off"),
        "hard_negative_summary": hn_summary,
        "hard_negative_weight": hn_weight,
        "hard_negative_weight_applied": hn_weight_applied,
        "threshold_objective": threshold_objective,
        "augmentation": {
            **({"mode": augment_mode, "n_augmented_rows": n_augmented} if augment_mode != "none" else {"mode": "none"}),
        },
        "evidence_feature_version": meta.get("evidence_feature_version"),
        "n_peakcodebook": meta.get("n_peakcodebook"),
        "peakcodebook_bin_width": meta.get("peakcodebook_bin_width"),
        "n_deconv": meta.get("n_deconv"),
        "deconv_profile_type": meta.get("deconv_profile_type"),
        "deconv_mode": str(getattr(args, "deconv_mode", "") or meta.get("deconv_mode") or ""),
        "memory_safe_training": bool(memory_safe),
        "calibration_status": cal_meta.get("calibration_status", cal_meta.get("method_effective")),
        "feature_prefix_counts": feature_prefix_counts(meta.get("feature_names") or []),
        "low_support_labels": meta.get("low_support_labels") or [],
    }
    Path(str(eval_prefix) + "_training_metadata.json").write_text(
        json.dumps(training_meta_out, indent=2, default=str),
        encoding="utf-8",
    )

    if ontology_train == "v4":
        from dataclasses import asdict

        from ml.ftir_ontology import ONTOLOGY_V4, is_v4

        if is_v4(ontology_train):
            on_sum = {k: asdict(v) for k, v in ONTOLOGY_V4.items() if k in set(label_names) or v.trainable_basic or v.trainable_subtle}
            Path(str(eval_prefix) + "_ontology_summary.json").write_text(
                json.dumps({"ontology": "v4", "entries": on_sum}, indent=2, default=str),
                encoding="utf-8",
            )
        cat_rows = []
        for row in label_metrics_rows:
            lab = str(row.get("label", ""))
            ent = ONTOLOGY_V4.get(lab)
            cat = ent.category if ent else "unknown"
            cat_rows.append({**row, "ontology_category": cat})
        p_cat = Path(str(eval_prefix) + "_label_category_metrics.csv")
        if cat_rows:
            with p_cat.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(cat_rows[0].keys()))
                w.writeheader()
                for r in cat_rows:
                    w.writerow(r)
        dropped_ev = [lab for lab in ONTOLOGY_V4 if lab not in label_names and ONTOLOGY_V4[lab].reportable]
        Path(str(eval_prefix) + "_dropped_or_evidence_only_labels.csv").write_text(
            "label,note\n"
            + "\n".join(f"{lab},not_in_training_columns_or_evidence_only" for lab in dropped_ev[:400]),
            encoding="utf-8",
        )

    interpret: dict[str, Any] | None = None
    try:
        art_tmp = {
            "model": clf,
            "labels": label_names,
            "meta": {
                **meta,
                "feature_names": meta.get("feature_names") or structural_feature_names(meta),
                "n_spectral": meta.get("n_spectral", 14),
                "n_smarts": meta.get("n_smarts", 0),
                "n_rdkit": meta.get("n_rdkit", 0),
                "n_mordred": meta.get("n_mordred", 0),
                "n_evidence": meta.get("n_evidence", 0),
            },
        }
        interpret = explain_structural_fg_model(art_tmp, topk=8)
    except Exception as exc:
        warn_list.append(f"Global interpretability unavailable: {exc}")

    report_dir = Path("reports") / (f"v4_ontology_{train_ts}" if ontology_train == "v4" else f"model_training_{train_ts}")
    _write_training_report_html(
        report_dir / "REPORT.html",
        title=f"Structural FG SVM training report ({model_kind})",
        training_meta=training_meta_out,
        interpret_summary=interpret,
    )

    meta.update(
        {
            "train_estimator": str(cal_meta.get("strategy")),
            "model_kind": model_kind,
            "label_source": label_source,
            "training_timestamp_utc": train_ts,
            "label_counts": {lab: int(Y[:, j].sum()) for j, lab in enumerate(label_names)},
            "dropped_labels": dropped_all,
            "dataset_prefix": str(prefix),
            "git_hash": training_meta_out.get("git_hash"),
            "feature_names": meta.get("feature_names") or structural_feature_names(meta),
            "calibration": cal_meta,
            "ml_score_kind": score_kind,
            "split": training_meta_out["split"],
            "package_versions": training_meta_out["package_versions"],
            "warnings": warn_list,
            "per_label_thresholds": training_meta_out["per_label_thresholds"],
            "ontology": ontology_train,
        }
    )

    artifact: dict[str, Any] = {
        "version": 3,
        "kind": "structural_fg_svm",
        "model_kind": model_kind,
        "model": clf,
        "scaler": scaler,
        "labels": [str(x).lower() for x in label_names],
        "label_counts": meta["label_counts"],
        "dropped_labels": dropped_all,
        "feature_names": meta["feature_names"],
        "training_timestamp_utc": train_ts,
        "dataset_provenance": str(prefix.resolve()),
        "git_hash": meta.get("git_hash"),
        "meta": meta,
        "ml_score_kind": score_kind,
        "calibration": cal_meta,
    }
    joblib.dump(artifact, model_out)

    if run_benchmark:
        import csv as csv_mod

        from ml.model_benchmark import run_model_benchmark

        bench_types = list(
            getattr(args, "benchmark_models", None) or BENCHMARK_MODEL_CHOICES
        )
        bench_rows = run_model_benchmark(
            X_train,
            Y_train,
            X_test,
            Y_test,
            label_names,
            model_types=bench_types,
            feature_set=feature_set_train,
            model_kind=model_kind,
            random_state=random_state,
            n_jobs=n_jobs,
            threshold_objective=threshold_objective,
        )
        bench_path = Path(str(eval_prefix) + "_model_benchmark_summary.csv")
        if bench_rows:
            with bench_path.open("w", newline="", encoding="utf-8") as fh:
                w = csv_mod.DictWriter(fh, fieldnames=list(bench_rows[0].keys()))
                w.writeheader()
                for br in bench_rows:
                    w.writerow(br)

    if not no_update_latest:
        latest = model_out.parent / f"struct_fg_{model_kind}_smarts_latest.joblib"
        if str(label_source).lower() == "smarts":
            try:
                import shutil

                shutil.copyfile(model_out, latest)
            except OSError:
                pass
        if ontology_train == "v4":
            from ml.specialist_configs import is_specialist_model_kind

            v4_latest = model_out.parent / f"struct_fg_{model_kind}_v4_ontology_latest.joblib"
            try:
                import shutil

                shutil.copyfile(model_out, v4_latest)
            except OSError:
                pass
            if is_specialist_model_kind(model_kind):
                spec_latest = model_out.parent / f"struct_fg_specialist_{model_kind}_v4_ontology_latest.joblib"
                try:
                    import shutil

                    shutil.copyfile(model_out, spec_latest)
                except OSError:
                    pass
            if model_kind in (MODEL_KIND_FAMILY, "family"):
                fam_latest = model_out.parent / "struct_fg_family_v4_ontology_latest.joblib"
                try:
                    import shutil

                    shutil.copyfile(model_out, fam_latest)
                except OSError:
                    pass
            if model_kind in (MODEL_KIND_SPECIFIC, "specific"):
                spec_latest = model_out.parent / "struct_fg_specific_v4_ontology_latest.joblib"
                try:
                    import shutil

                    shutil.copyfile(model_out, spec_latest)
                except OSError:
                    pass

    print(
        json.dumps(
            {
                "model_out": str(model_out.resolve()),
                "artifacts_prefix": str(eval_prefix),
                "training_report": str((report_dir / "REPORT.html").resolve()),
                **metrics_summary,
            },
            indent=2,
        )
    )
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Print label list and counts from a saved model bundle."""
    import joblib

    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}")
    artifact = joblib.load(model_path)
    rep = {
        "model": str(model_path),
        "model_kind": artifact.get("model_kind") or (artifact.get("meta") or {}).get("model_kind"),
        "labels": artifact.get("labels"),
        "label_counts": artifact.get("label_counts") or (artifact.get("meta") or {}).get("label_counts"),
        "dropped_labels": artifact.get("dropped_labels") or (artifact.get("meta") or {}).get("dropped_labels"),
        "training_timestamp_utc": artifact.get("training_timestamp_utc"),
        "n_features": len(artifact.get("feature_names") or []),
    }
    print(json.dumps(rep, indent=2))
    return 0


def predict_proba_row(
    artifact: dict[str, Any],
    *,
    wn: np.ndarray,
    y: np.ndarray,
    md: dict[str, Any],
) -> dict[str, float]:
    """Return label -> score for one spectrum + metadata (layout matches training)."""
    _require_sklearn()
    meta = artifact.get("meta") or {}
    mrq = int(meta.get("mordred_max_requested", 256))
    fs = _normalize_feature_set(meta.get("feature_set", None))
    if not meta.get("feature_set") and not meta.get("feature_names"):
        fs = "legacy"

    use_mordred = fs == "legacy" or "mordred" in fs
    calc: Any = None
    mordred_dim = 0
    if use_mordred:
        calc, mordred_names = make_mordred_calculator(max_descriptors=mrq)
        mordred_dim = len(mordred_names)
    smarts_labels: list[str] = []
    if "smarts" in fs:
        smarts_labels = [
            str(x).replace("smarts_", "") for x in (meta.get("smarts_feature_names") or []) if x
        ]
        if not smarts_labels:
            smarts_labels = [str(x) for x in (meta.get("label_names") or artifact.get("labels") or [])]

    md_use = canonicalize_structure_metadata(dict(md))
    wn_arr = np.asarray(wn, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if bool(meta.get("nist_ftir_cm1_prep_applied", False)):
        pr = prepare_nist_ftir_cm1(wn_arr, y_arr, md_use)
        if pr is not None:
            wn_arr, y_arr = pr

    if fs == "legacy":
        assert calc is not None
        x_row, _ = build_feature_row(
            wn_arr,
            y_arr,
            md_use,
            calc=calc,
            mordred_dim=mordred_dim,
        )
    else:
        if calc is None and use_mordred:
            calc, mordred_names = make_mordred_calculator(max_descriptors=mrq)
            mordred_dim = len(mordred_names)
        ev_extra = None
        if "evidence" in fs:
            from ml.ftir_evidence import extract_spectral_evidence

            ev_cfg = {"ontology": "v4"} if str(meta.get("ontology") or "").lower() == "v4" else {}
            ev_extra = extract_spectral_evidence(wn_arr, y_arr, peaks=None, config=ev_cfg)
        x_row, _ = build_feature_row_layout(
            wn_arr,
            y_arr,
            md_use,
            feature_set=fs,
            calc=calc,
            mordred_dim=mordred_dim,
            smarts_feature_labels=smarts_labels,
            evidence=ev_extra,
            deconv_mode=str(meta.get("deconv_mode") or "fast"),
        )

    scaler = artifact["scaler"]
    clf = artifact["model"]
    labels = [str(x).lower() for x in artifact["labels"]]
    xt = scaler.transform(x_row.reshape(1, -1))
    score_kind = str(artifact.get("ml_score_kind") or meta.get("ml_score_kind") or "calibrated_probability").lower()
    use_proba = score_kind == "calibrated_probability" and hasattr(clf, "predict_proba")
    if use_proba:
        raw = clf.predict_proba(xt)
        if isinstance(raw, list):
            probs = np.column_stack(
                [
                    np.asarray(r)[:, 1] if np.asarray(r).ndim == 2 and np.asarray(r).shape[1] > 1 else np.asarray(r).ravel()
                    for r in raw
                ]
            )
        else:
            probs = np.asarray(raw)
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        out_vec = np.clip(probs[0], 0.0, 1.0)
    else:
        df = clf.decision_function(xt)
        arr = np.asarray(df, dtype=float).ravel()
        if arr.size != len(labels):
            arr = np.asarray(df, dtype=float).reshape(-1)
        out_vec = arr

    out: dict[str, float] = {}
    for i, lab in enumerate(labels):
        out[lab] = float(out_vec[i] if i < out_vec.size else 0.0)
    out = mask_fg_probs_by_atom_content(out, md_use)
    return out


def cmd_predict(args: argparse.Namespace) -> int:
    """Load one spectrum (.csv / .jdx), run the same featurizer + model as training, print JSON."""
    import joblib
    from lib.ftir_foundation import preprocess_spectrum, read_spectrum

    spec_path = Path(args.spectrum).resolve()
    if not spec_path.is_file():
        raise SystemExit(f"Spectrum not found: {spec_path}")
    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}")

    wn_raw, inten_raw, hint = read_spectrum(spec_path)
    wn, y, _ = preprocess_spectrum(wn_raw, inten_raw, intensity_mode=hint)
    title = (args.title or spec_path.stem).strip()
    md: dict[str, Any] = {"title": title, "name": title, "xunits": "1/CM"}
    if getattr(args, "cas", None) and str(args.cas).strip():
        md["cas"] = str(args.cas).strip()
    if getattr(args, "formula", None) and str(args.formula).strip():
        md["formula"] = str(args.formula).strip()
    if getattr(args, "elements", None) and str(args.elements).strip():
        md["elements"] = str(args.elements).strip()
    if getattr(args, "skip_atom_fg_mask", False):
        md["skip_atom_fg_mask"] = True

    artifact = joblib.load(model_path)
    probs = predict_proba_row(artifact, wn=wn, y=y, md=md)
    out = {
        "spectrum": str(spec_path),
        "model": str(model_path),
        "title": title,
        "functional_group_probabilities": probs,
    }
    print(json.dumps(out, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Structural FG SVM: spectra + RDKit + Mordred")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ds = sub.add_parser("build-dataset", help="Build NPZ + meta.json from SQLite index")
    p_ds.add_argument("--nist-index", required=True)
    p_ds.add_argument("--out-prefix", required=True, help="Prefix for .npz and .meta.json")
    from ml.specialist_configs import SPECIALIST_MODEL_KINDS

    _build_model_kinds = tuple(dict.fromkeys([*MODEL_KINDS, *SPECIALIST_MODEL_KINDS]))
    p_ds.add_argument(
        "--model-kind",
        choices=_build_model_kinds,
        default=MODEL_KIND_BASIC,
        help="basic/subtle/family/specific/combined or v4 specialist kinds",
    )
    p_ds.add_argument(
        "--label-source",
        choices=LABEL_SOURCES,
        default="existing",
        help="Weak labels: SMARTS on structure, legacy keywords, or hybrid merge.",
    )
    p_ds.add_argument(
        "--feature-set",
        choices=FEATURE_SET_CHOICES,
        default="legacy",
        help="Descriptor layout (legacy = spectral+RDKit+Mordred+flag).",
    )
    p_ds.add_argument(
        "--min-label-positives",
        type=int,
        default=50,
        help="Drop labels with fewer positive training examples at build time.",
    )
    p_ds.add_argument(
        "--mordred-max",
        type=int,
        default=256,
        help="Max Mordred descriptors (2D, ignore_3D); deterministic subsample if smaller than full set.",
    )
    p_ds.add_argument(
        "--require-structure",
        action="store_true",
        help="Keep only rows where INCHI/SMILES parses with RDKit.",
    )
    p_ds.add_argument(
        "--enrich-pubchem",
        action="store_true",
        help="Fill SMILES/InChI via PubChem PUG REST from CAS (metadata/filename) and compound name.",
    )
    p_ds.add_argument(
        "--pubchem-cache",
        type=str,
        default="ml/runs/pubchem_structure_cache.json",
        help="JSON cache of PubChem lookups (retry-safe, respects PubChem load).",
    )
    p_ds.add_argument(
        "--pubchem-delay",
        type=float,
        default=0.25,
        help="Seconds to sleep between PubChem HTTP calls (be polite on large builds).",
    )
    p_ds.add_argument(
        "--pubchem-offline-only",
        action="store_true",
        help="Use only entries already in --pubchem-cache (no network).",
    )
    p_ds.add_argument(
        "--pubchem-cache-save-every",
        type=int,
        default=500,
        help="With --enrich-pubchem, flush --pubchem-cache to disk every N SQLite spectra rows (0=only at successful exit). "
        "Use the same --pubchem-cache path when re-running after a crash to resume lookups without re-querying PubChem.",
    )
    p_ds.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print heartbeat lines to stderr every N SQLite spectra rows (0 disables).",
    )
    p_ds.add_argument(
        "--no-nist-ftir-cm1-prep",
        action="store_true",
        help="Use stored X/Y as-is (no um->cm^-1 conversion, no span/point validity filter).",
    )
    p_ds.add_argument(
        "--pubchem-debug",
        type=int,
        default=0,
        help="Print first N network_miss/network_ambiguous rows as JSON on stderr (0 disables).",
    )
    p_ds.add_argument(
        "--ontology",
        choices=("v3", "v4"),
        default="v3",
        help="v3 legacy; v4 enables ontology buckets, v4 SMARTS basic labels when label-source=smarts, and spectral+evidence helpers.",
    )
    p_ds.add_argument(
        "--pipeline-version",
        default="",
        help="Optional tag stored in dataset meta (e.g. v4_ontology) for training artifacts.",
    )
    p_ds.add_argument(
        "--store-raw-spectra",
        action="store_true",
        help="Store Wn_raw/Yspec_raw in NPZ for train-time augmentation (auto when peakcodebook in feature-set).",
    )
    p_ds.add_argument(
        "--deconv-mode",
        choices=("off", "fast", "full"),
        default="fast",
        help="Region deconvolution speed/quality when feature-set includes deconv (default fast).",
    )
    p_ds.set_defaults(func=cmd_build_dataset)

    p_smoke = sub.add_parser(
        "pubchem-smoke",
        help="Verify HTTPS access to PubChem PUG REST (default CAS 64-17-5 ethanol).",
        epilog="Example: python -m ml.structural_fg_svm pubchem-smoke --cas 100-00-5 --verbose",
    )
    p_smoke.add_argument(
        "--cas",
        type=str,
        default="64-17-5",
        help="CAS registry number to test (default: 64-17-5 ethanol).",
    )
    p_smoke.add_argument(
        "--verbose",
        action="store_true",
        help="Print xref URL + HTTP/TLS diagnostic JSON to stderr before the summary.",
    )
    p_smoke.add_argument(
        "--show-candidates",
        action="store_true",
        help="Include compact ranked candidate report (scores/reasons) from phase-2/3 resolver.",
    )
    p_smoke.add_argument(
        "--max-candidate-attempts",
        type=int,
        default=3,
        help="Max query attempts to include in --show-candidates report.",
    )
    p_smoke.add_argument(
        "--max-ranked-candidates",
        type=int,
        default=3,
        help="Max ranked CIDs per attempt in --show-candidates report.",
    )
    p_smoke.add_argument(
        "--pubchem-delay",
        type=float,
        default=0.0,
        help="Optional delay before the request (seconds).",
    )
    p_smoke.set_defaults(func=cmd_pubchem_smoke)

    p_resolve = sub.add_parser(
        "pubchem-resolve",
        help="Preview PubChem query candidates and run Phase1/2/3 resolve (debug; same HTTP stack as build-dataset).",
    )
    p_resolve.add_argument("--cas", default="", help="Optional CAS (metadata)")
    p_resolve.add_argument("--title", default="", help="Compound title/name")
    p_resolve.add_argument("--path", default="", help="Optional source path or filename stem hint")
    p_resolve.add_argument(
        "--pubchem-delay",
        type=float,
        default=0.25,
        help="Delay between PubChem HTTP calls (seconds).",
    )
    p_resolve.set_defaults(func=cmd_pubchem_resolve)

    p_tr = sub.add_parser("train", help="Train OvR SVM on dataset from build-dataset")
    p_tr.add_argument("--dataset-prefix", required=True, help="Same prefix passed to build-dataset")
    p_tr.add_argument(
        "--model-out",
        default="",
        help="Output .joblib (default: ml/runs/struct_fg_{kind}_{timestamp}.joblib)",
    )
    from ml.specialist_configs import SPECIALIST_MODEL_KINDS

    _train_model_kinds = tuple(dict.fromkeys([*MODEL_KINDS, *SPECIALIST_MODEL_KINDS]))
    p_tr.add_argument("--model-kind", choices=_train_model_kinds, default=None, help="Override kind from dataset meta")
    p_tr.add_argument(
        "--label-source",
        default="",
        help="Override label provenance recorded in dataset meta (optional)",
    )
    p_tr.add_argument("--min-label-positives", "--min-positives", type=int, default=20)
    p_tr.add_argument(
        "--feature-set",
        default="",
        help="Override feature_set recorded in dataset meta (must match X dimension).",
    )
    p_tr.add_argument("--augment", choices=AUGMENT_CHOICES, default="none")
    p_tr.add_argument(
        "--threshold-objective",
        choices=THRESHOLD_OBJECTIVE_CHOICES,
        default="balanced_guarded",
    )
    p_tr.add_argument("--hard-negative-weight", type=float, default=1.5)
    p_tr.add_argument(
        "--benchmark",
        action="store_true",
        help="Run alternative classifier benchmarks on same split; writes model_benchmark_summary.csv",
    )
    p_tr.add_argument(
        "--benchmark-models",
        nargs="*",
        default=None,
        help="Subset of benchmark model types (default: all).",
    )
    p_tr.add_argument(
        "--no-update-latest",
        action="store_true",
        help="Do not copy trained model to *_latest.joblib symlinks (for experiments).",
    )
    p_tr.add_argument(
        "--deconv-mode",
        choices=("off", "fast", "full"),
        default="",
        help="Record deconv mode in training metadata (features come from dataset NPZ).",
    )
    p_tr.add_argument(
        "--memory-safe",
        action="store_true",
        help="Reduce calibration CV to 3, force n_jobs=1 (auto-on when n_features>=500).",
    )
    p_tr.add_argument(
        "--memory-safe-feature-threshold",
        type=int,
        default=500,
        help="Auto-enable memory-safe training when feature_dim >= this value.",
    )
    p_tr.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Train uncalibrated LinearSVC only (sets ml_score_kind=svm_decision_score).",
    )
    p_tr.add_argument(
        "--calibration",
        choices=CALIBRATION_CHOICES,
        default="sigmoid",
        help="Probability calibration for LinearSVM OvR heads (none = decision scores on inference).",
    )
    p_tr.add_argument("--test-size", type=float, default=0.2)
    p_tr.add_argument("--split", choices=("molecule", "random"), default="molecule")
    p_tr.add_argument("--random-state", type=int, default=13)
    p_tr.add_argument(
        "--out",
        default="",
        help="Output directory for model .joblib, metrics CSV/JSON, and training report stem (default: ml/runs/)",
    )
    p_tr.add_argument(
        "--remap-legacy-labels",
        action="store_true",
        help="Map v7 12-label NPZ columns to basic broad labels before training.",
    )
    p_tr.add_argument("--n-jobs", type=int, default=1)
    p_tr.add_argument(
        "--pipeline-version",
        "--version",
        dest="pipeline_version",
        default="",
        help="Optional tag for artifact naming (e.g. v3_guarded -> struct_fg_<kind>_v3_guarded_<timestamp>.joblib)",
    )
    p_tr.add_argument(
        "--hard-negative-mode",
        choices=("on", "off"),
        default="on",
        help="Per-label threshold tuning + hard-negative FP CSV on test split.",
    )
    p_tr.add_argument(
        "--ontology",
        choices=("v3", "v4"),
        default="",
        help="Override ontology for model naming/metadata (default: use dataset meta).",
    )
    p_tr.set_defaults(func=cmd_train)

    p_ev = sub.add_parser("evaluate", help="Summarize labels/metadata in a trained model")
    p_ev.add_argument("--model", required=True)
    p_ev.set_defaults(func=cmd_evaluate)

    p_pr = sub.add_parser(
        "predict",
        help="Run trained structural model on one .csv or .jdx spectrum (same IO as ftir_foundation).",
    )
    p_pr.add_argument("--model", required=True, help="Path to structural_fg_svm *.joblib")
    p_pr.add_argument("--spectrum", required=True, help="Path to .csv or .jdx spectrum file")
    p_pr.add_argument("--title", default="", help="Compound title for FG keyword rules (default: filename stem)")
    p_pr.add_argument("--cas", default="", help="Optional CAS for PubChem-enriched metadata if present in model")
    p_pr.add_argument(
        "--formula",
        default="",
        help="Empirical formula (e.g. C8H11NO2) to infer elements; zeros halide if no F/Cl/Br/I/At.",
    )
    p_pr.add_argument(
        "--elements",
        default="",
        help="Comma-separated elements (e.g. C,H,N,O); zeros halide if no halogens listed.",
    )
    p_pr.add_argument(
        "--skip-atom-fg-mask",
        action="store_true",
        help="Do not apply atom-content halide masking.",
    )
    p_pr.set_defaults(func=cmd_predict)

    p_ex = sub.add_parser(
        "explain",
        help="Per-FG linear coefficients (OvR LinearSVC) + spectral window reference.",
    )
    p_ex.add_argument("--model", required=True, help="structural_fg_svm *.joblib")
    p_ex.add_argument("--top", type=int, default=12, help="Top positive/negative features per FG")
    p_ex.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="json (default) or compact text tables",
    )
    p_ex.set_defaults(func=cmd_explain)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
