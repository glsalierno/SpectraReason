"""Load and preprocess a single FTIR spectrum (CSV/JDX)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lib.ftir_foundation import preprocess_spectrum, read_spectrum


@dataclass(frozen=True)
class ProcessedSpectrum:
    name: str
    path: Path
    wn: np.ndarray
    y: np.ndarray


def load_processed_spectrum(path: Path) -> ProcessedSpectrum:
    wn_raw, inten_raw, hint = read_spectrum(path)
    wn, y, _ = preprocess_spectrum(wn_raw, inten_raw, intensity_mode=hint)
    return ProcessedSpectrum(
        name=path.name,
        path=path.resolve(),
        wn=np.asarray(wn, float),
        y=np.asarray(y, float),
    )
