"""Shared figure paths, style, and deterministic export helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Results"
FIGURES = ROOT / "Figures"
FIGURE_DIRS = (FIGURES,)

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#666666",
}

FIG4_FONT_RC = {
    "font.size": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
}


def configure_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.top": True,
        "ytick.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "lines.linewidth": 1.35,
        "lines.markersize": 4.0,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def ensure_output_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    for figure_dir in FIGURE_DIRS:
        figure_dir.mkdir(parents=True, exist_ok=True)


def save_named_figure(fig: plt.Figure, figure_name: str) -> None:
    figure_path = Path(figure_name)
    if figure_path.suffix.lower() != ".png":
        figure_path = figure_path.with_suffix(".png")
    for figure_dir in FIGURE_DIRS:
        figure_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_dir / figure_path.name, format="png", dpi=300)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _relative_file_hashes(paths: list[Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix()):
        hashes[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    return hashes


def refresh_output_manifest() -> None:
    """Record hashes for generated result and figure files."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    result_files = [
        path for path in RESULTS.iterdir()
        if path.is_file() and path.name != "output_manifest.json"
    ]
    figure_files: list[Path] = []
    for figure_dir in FIGURE_DIRS:
        if figure_dir.exists():
            figure_files.extend(figure_dir.glob("figure*.png"))
    manifest = {
        "hash_algorithm": "sha256",
        "result_files": _relative_file_hashes(result_files),
        "figure_files": _relative_file_hashes(figure_files),
    }
    with open(RESULTS / "output_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def refresh_random_mixed_hashes() -> None:
    summary_path = RESULTS / "numerical_summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as handle:
            summary = json.load(handle)
        summary["figure4_source_csv_sha256"] = file_sha256(
            RESULTS / "random_mixed_state_povm_equality.csv"
        )
        summary["figure4_png_sha256"] = file_sha256(FIGURE_DIRS[0] / "figure4.png")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
    refresh_output_manifest()
