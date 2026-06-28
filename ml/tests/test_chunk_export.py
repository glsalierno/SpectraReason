"""Tests for spectral chunk export."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reports.chunk_export import ChunkExportConfig, export_spectral_chunks
from reports.region_stack_export import StackSpectrum, spectra_from_batch


def test_export_spectral_chunks_smoke(tmp_path: Path) -> None:
    wn = np.linspace(4000, 400, 200)
    ab = 0.1 + 0.5 * np.exp(-((wn - 1600) ** 2) / (2 * 80**2))
    p1 = tmp_path / "a.CSV"
    p2 = tmp_path / "b.CSV"
    p1.write_text(
        "Wavenumber,Absorbance\n" + "\n".join(f"{a:.1f},{b:.4f}" for a, b in zip(wn, ab)),
        encoding="utf-8",
    )
    p2.write_text(
        "Wavenumber,Absorbance\n" + "\n".join(f"{a:.1f},{b:.4f}" for a, b in zip(wn, ab * 0.8)),
        encoding="utf-8",
    )
    specs = spectra_from_batch([p1, p2])
    manifest = export_spectral_chunks(
        spectra=specs,
        out_dir=tmp_path / "out",
        config=ChunkExportConfig(
            stack_modes=("normalized_absorbance",),
            formats=("png",),
            export_collage=True,
            export_chunk_data=True,
        ),
    )
    assert Path(manifest["ranges_config_path"]).is_file()
    assert manifest.get("outputs")
    assert manifest.get("chunk_data")
    assert manifest.get("collage")


def test_export_removes_legacy_stack_artifacts(tmp_path: Path) -> None:
    stacks = tmp_path / "out" / "stacks"
    stacks.mkdir(parents=True)
    (stacks / "oh_nh_normalized_absorbance_stack.svg").write_text("legacy", encoding="utf-8")
    (stacks / "region_stacks_manifest.json").write_text("{}", encoding="utf-8")

    wn = np.linspace(4000, 400, 120)
    ab = 0.2 + 0.4 * np.exp(-((wn - 3300) ** 2) / (2 * 120**2))
    p = tmp_path / "a.CSV"
    p.write_text(
        "Wavenumber,Absorbance\n" + "\n".join(f"{a:.1f},{b:.4f}" for a, b in zip(wn, ab)),
        encoding="utf-8",
    )
    export_spectral_chunks(
        spectra=spectra_from_batch([p]),
        out_dir=tmp_path / "out",
        config=ChunkExportConfig(stack_modes=("normalized_absorbance",), formats=("png",)),
    )
    assert not (stacks / "oh_nh_normalized_absorbance_stack.svg").exists()
    assert not (stacks / "region_stacks_manifest.json").exists()
    assert (stacks / "chunks_manifest.json").is_file()
