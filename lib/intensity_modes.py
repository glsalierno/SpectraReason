"""Explicit FTIR intensity categories, conversions, and transmittance export rules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from lib.ftir_foundation import infer_intensity_mode

INTENSITY_EPS = 1e-9

RawIntensityCategory = Literal[
    "transmittance_percent",
    "transmittance_fraction",
    "absorbance",
    "absorbance_difference",
]

ForceIntensityMode = Literal[
    "transmittance_percent",
    "absorbance",
    "absorbance_difference",
]

DIFFERENCE_FILENAME_MARKERS = (
    "blank_subtracted",
    "minus_air",
    "_minus_",
    "difference",
    "minus",
    "diff",
    "scaled",
)

FORCED_DIFFERENCE_STEMS = frozenset(
    {
        "pda_eg_con_new_minus_air_scaled",
        "oda_in_ethanol_blank_subtracted",
    }
)

DEFAULT_APPARENT_TRANSMITTANCE_LABEL = "Apparent Transmittance (%)"
DEFAULT_APPARENT_WARNING = (
    "Computed from absorbance-like data; not native measured %T."
)
DIFFERENCE_APPARENT_WARNING = (
    "Blank-subtracted difference: absorbance offset (minimum → 0) before "
    "T_app = 100·10^(−A); not native measured %T."
)
DEFAULT_TRANSMITTANCE_SKIP_BANNER = (
    "Native transmittance curation is disabled because this spectrum was detected as "
    "absorbance/difference data. Use normalized absorbance for interpretation. "
    "Apparent transmittance can be exported with --allow-apparent-transmittance, "
    "but it is not native measured %T."
)


def transmittance_to_absorbance(t_percent: np.ndarray, *, eps: float = INTENSITY_EPS) -> np.ndarray:
    """A = -log10(clip(T_percent, eps, None) / 100)."""
    t = np.clip(np.asarray(t_percent, dtype=float), eps, None)
    return -np.log10(t / 100.0)


def absorbance_to_apparent_transmittance(a: np.ndarray, *, eps: float = INTENSITY_EPS) -> np.ndarray:
    """T_app = 100 * 10^(-A). Not chemically meaningful for difference spectra unless flagged."""
    a = np.asarray(a, dtype=float)
    return 100.0 * np.power(10.0, -a)


def is_difference_filename(stem: str) -> bool:
    s = stem.lower()
    if s in FORCED_DIFFERENCE_STEMS:
        return True
    return any(marker in s for marker in DIFFERENCE_FILENAME_MARKERS)


def is_native_transmittance_category(category: str) -> bool:
    return category in ("transmittance_percent", "transmittance_fraction")


def mostly_non_negative(y: np.ndarray, *, negative_fraction_max: float = 0.05) -> bool:
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return False
    return float(np.mean(y < 0.0)) <= negative_fraction_max


def apparent_transmittance_plausible(
    t_app: np.ndarray,
    *,
    lo: float = 0.0,
    hi: float = 120.0,
    margin: float = 5.0,
) -> bool:
    y = np.asarray(t_app, dtype=float)
    if y.size == 0:
        return False
    p1, p99 = np.percentile(y, [1, 99])
    return float(p1) >= lo - margin and float(p99) <= hi + margin


@dataclass(frozen=True)
class IntensityClassification:
    category: RawIntensityCategory
    preprocess_mode: str
    is_native_transmittance: bool
    is_absorbance_difference: bool
    forced: bool = False


@dataclass(frozen=True)
class TransmittancePanelPlan:
    show_panel: bool
    y_values: np.ndarray | None
    is_apparent: bool
    ylabel: str
    warning: str | None
    skip_reason: str | None
    banner_html: str | None = None


def classify_intensity(
    path: Path | str,
    hint: str,
    raw: np.ndarray,
    *,
    force_mode: ForceIntensityMode | None = None,
) -> IntensityClassification:
    """Classify raw spectrum intensity for preprocessing and export rules."""
    stem = Path(path).stem.lower()
    forced = force_mode is not None

    if force_mode == "absorbance_difference" or (
        not forced and is_difference_filename(stem)
    ):
        return IntensityClassification(
            category="absorbance_difference",
            preprocess_mode="absorbance",
            is_native_transmittance=False,
            is_absorbance_difference=True,
            forced=forced and force_mode == "absorbance_difference",
        )

    if force_mode == "transmittance_percent":
        return IntensityClassification(
            category="transmittance_percent",
            preprocess_mode="transmittance_percent",
            is_native_transmittance=True,
            is_absorbance_difference=False,
            forced=True,
        )

    if force_mode == "absorbance":
        return IntensityClassification(
            category="absorbance",
            preprocess_mode="absorbance",
            is_native_transmittance=False,
            is_absorbance_difference=False,
            forced=True,
        )

    if hint in ("transmittance_percent", "transmittance_fraction", "absorbance"):
        category: RawIntensityCategory = hint  # type: ignore[assignment]
        return IntensityClassification(
            category=category,
            preprocess_mode=category if category != "absorbance_difference" else "absorbance",
            is_native_transmittance=is_native_transmittance_category(category),
            is_absorbance_difference=False,
        )

    inferred = infer_intensity_mode(raw)
    return IntensityClassification(
        category=inferred,  # type: ignore[arg-type]
        preprocess_mode=inferred,
        is_native_transmittance=is_native_transmittance_category(inferred),
        is_absorbance_difference=False,
    )


def native_transmittance_percent(raw: np.ndarray, classification: IntensityClassification) -> np.ndarray:
    y = np.asarray(raw, dtype=float)
    if classification.category == "transmittance_fraction":
        return y * 100.0
    return y


def explain_transmittance_skip(
    classification: IntensityClassification,
    raw: np.ndarray,
    *,
    allow_apparent: bool,
    apparent_rejected_reason: str | None = None,
) -> str:
    if classification.is_native_transmittance:
        mn, mx = float(np.nanmin(raw)), float(np.nanmax(raw))
        if mn < -1.0 or mx > 120.0:
            return (
                f"Native %T not detected: raw values (min={mn:.2g}, max={mx:.2g}) "
                "are outside the valid transmittance range."
            )
        return "Native %T not detected."

    if classification.is_absorbance_difference:
        if not allow_apparent:
            return (
                "Transmittance export skipped: absorbance/difference spectrum "
                "(blank-subtracted, scaled-difference, or filename marker)."
            )
        if apparent_rejected_reason:
            return apparent_rejected_reason
        return "Apparent transmittance disabled or not plausible for this difference spectrum."

    if not allow_apparent:
        return (
            "Transmittance export skipped: spectrum classified as absorbance; "
            "native measured %T not detected. "
            "Use normalized absorbance, or pass --allow-apparent-transmittance for T_app = 100·10^(−A)."
        )

    if apparent_rejected_reason:
        return apparent_rejected_reason

    return "Apparent transmittance disabled (--allow-apparent-transmittance not set)."


def plan_transmittance_panel(
    raw: np.ndarray,
    classification: IntensityClassification,
    *,
    allow_apparent: bool = False,
    apparent_label: str = DEFAULT_APPARENT_TRANSMITTANCE_LABEL,
) -> TransmittancePanelPlan:
    """Decide whether to show/export transmittance (native or apparent)."""
    raw = np.asarray(raw, dtype=float)

    if classification.is_native_transmittance:
        y_native = native_transmittance_percent(raw, classification)
        mn, mx = float(np.nanmin(y_native)), float(np.nanmax(y_native))
        if mn < -1.0 or mx > 120.0:
            reason = explain_transmittance_skip(
                classification, raw, allow_apparent=allow_apparent
            )
            return TransmittancePanelPlan(
                show_panel=False,
                y_values=None,
                is_apparent=False,
                ylabel="Transmittance (%T)",
                warning=None,
                skip_reason=reason,
                banner_html=_skip_banner(classification, reason, allow_apparent),
            )
        return TransmittancePanelPlan(
            show_panel=True,
            y_values=y_native,
            is_apparent=False,
            ylabel="Transmittance (%T)",
            warning=None,
            skip_reason=None,
            banner_html=None,
        )

    if not allow_apparent:
        reason = explain_transmittance_skip(
            classification, raw, allow_apparent=False
        )
        return TransmittancePanelPlan(
            show_panel=False,
            y_values=None,
            is_apparent=False,
            ylabel=apparent_label,
            warning=None,
            skip_reason=reason,
            banner_html=_skip_banner(classification, reason, allow_apparent=False),
        )

    warning = DEFAULT_APPARENT_WARNING
    absorbance_for_t = raw
    if classification.is_absorbance_difference:
        absorbance_for_t = raw - float(np.nanmin(raw))
        warning = DIFFERENCE_APPARENT_WARNING
    elif not mostly_non_negative(raw):
        rej = (
            "Apparent transmittance skipped: absorbance-like values are not mostly non-negative."
        )
        reason = explain_transmittance_skip(
            classification, raw, allow_apparent=True, apparent_rejected_reason=rej
        )
        return TransmittancePanelPlan(
            show_panel=False,
            y_values=None,
            is_apparent=True,
            ylabel=apparent_label,
            warning=DEFAULT_APPARENT_WARNING,
            skip_reason=reason,
            banner_html=_skip_banner(classification, reason, allow_apparent=True),
        )

    t_app = absorbance_to_apparent_transmittance(absorbance_for_t)
    if not apparent_transmittance_plausible(t_app):
        rej = (
            "Apparent transmittance skipped: T_app = 100·10^(−A) falls outside a plausible 0–120 %T range."
        )
        reason = explain_transmittance_skip(
            classification, raw, allow_apparent=True, apparent_rejected_reason=rej
        )
        return TransmittancePanelPlan(
            show_panel=False,
            y_values=None,
            is_apparent=True,
            ylabel=apparent_label,
            warning=DEFAULT_APPARENT_WARNING,
            skip_reason=reason,
            banner_html=_skip_banner(classification, reason, allow_apparent=True),
        )

    return TransmittancePanelPlan(
        show_panel=True,
        y_values=t_app,
        is_apparent=True,
        ylabel=apparent_label,
        warning=warning,
        skip_reason=None,
        banner_html=f"<p class='curation-warn intensity-warn'>{warning}</p>",
    )


def _skip_banner(
    classification: IntensityClassification,
    reason: str,
    *,
    allow_apparent: bool,
) -> str:
    if classification.is_native_transmittance:
        return f"<p class='curation-warn intensity-warn'>{html_escape(reason)}</p>"
    if allow_apparent:
        return f"<p class='curation-warn intensity-warn'>{html_escape(reason)}</p>"
    return f"<p class='curation-warn intensity-warn'>{DEFAULT_TRANSMITTANCE_SKIP_BANNER}</p>"


def html_escape(text: str) -> str:
    import html

    return html.escape(str(text), quote=True)


def resolve_intensity_mode(path: Path, hint: str, raw: np.ndarray, *, force_mode: ForceIntensityMode | None = None) -> str:
    """Backward-compatible preprocess mode string."""
    return classify_intensity(path, hint, raw, force_mode=force_mode).preprocess_mode


def transmittance_valid(category_or_mode: str) -> bool:
    """True when category is native transmittance (legacy helper)."""
    return is_native_transmittance_category(category_or_mode)


def has_transmittance_panel(plan: TransmittancePanelPlan) -> bool:
    return bool(plan.show_panel and plan.y_values is not None)
