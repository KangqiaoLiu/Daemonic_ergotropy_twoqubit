"""Generate the closed-form figure and associated numerical data."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

from closed_form_daemonic_ergotropy import (
    daemonic_gain_star,
    daemonic_gain_visibility,
    pure_schmidt_bloch,
    save_csv,
)
from figure_io import FIGURE_DIRS, OKABE_ITO, RESULTS, refresh_output_manifest


def configure_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 11,
        "axes.linewidth": 1.2,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.top": True,
        "ytick.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "lines.linewidth": 1.5,
        "lines.markersize": 4.0,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def main() -> None:
    configure_plot_style()
    RESULTS.mkdir(parents=True, exist_ok=True)
    for figure_dir in FIGURE_DIRS:
        figure_dir.mkdir(parents=True, exist_ok=True)

    C_values = np.linspace(0.0, 1.0, 101)
    pure_rows = []
    for C in C_values:
        a, T = pure_schmidt_bloch(C)
        pure_rows.append((C, daemonic_gain_star(a, T)))
    save_csv(RESULTS / "pure_state_concurrence_gain_closed_form.csv",
             ["concurrence", "gain_over_omega"], pure_rows)

    eta_values = np.linspace(0.0, 1.0, 101)
    state_specs = [
        ("pure C=0.8", "pure, $C=0.8$", np.array([0.0, 0.0, np.sqrt(1.0 - 0.8**2)]), np.diag([0.8, -0.8, 1.0]), OKABE_ITO["blue"], "-"),
        ("Bell diagonal", "Bell diag.", np.zeros(3), np.diag([0.55, 0.25, 0.15]), OKABE_ITO["vermillion"], "--"),
        ("aligned mixed", "aligned", np.array([0.0, 0.0, 0.20]), np.diag([0.22, 0.22, 0.50]), OKABE_ITO["green"], "-."),
    ]
    visibility_rows = []
    for eta in eta_values:
        for csv_label, _plot_label, a, T, _color, _ls in state_specs:
            visibility_rows.append((eta, csv_label, daemonic_gain_visibility(a, T, eta)))
    # Manual CSV write because one column is text.
    with open(RESULTS / "visibility_gain_closed_form.csv", "w", encoding="utf-8", newline="") as handle:
        handle.write("eta,state,gain_over_omega\n")
        for eta, label, gain in visibility_rows:
            handle.write(f"{eta:.12g},{label},{gain:.12g}\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.06, 2.82))
    pure = np.array(pure_rows, dtype=float)
    ax1.plot(pure[:, 0], pure[:, 1], color=OKABE_ITO["black"], label=r"closed form")
    sample_idx = np.linspace(0, len(pure) - 1, 11).astype(int)
    ax1.plot(pure[sample_idx, 0], pure[sample_idx, 1], linestyle="none",
             marker="o", markerfacecolor="none", color=OKABE_ITO["blue"],
             label="samples")
    ax1.set_xlabel(r"concurrence $C$")
    ax1.set_ylabel(r"$\Delta W_D^\star/\omega$")
    ax1.text(0.02, 0.94, "(a)", transform=ax1.transAxes, va="top", ha="left")
    ax1.minorticks_on()
    ax1.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.15, 1.02),
               handlelength=1.5, borderaxespad=0.2)

    for _csv_label, plot_label, a, T, color, ls in state_specs:
        gains = [daemonic_gain_visibility(a, T, eta) for eta in eta_values]
        ax2.plot(eta_values, gains, color=color, linestyle=ls, label=plot_label)
    ax2.set_xlabel(r"readout visibility $\eta$")
    ax2.set_ylabel(r"$\Delta W_{D,\eta}^\star/\omega$")
    ax2.text(0.02, 0.94, "(b)", transform=ax2.transAxes, va="top", ha="left")
    ax2.minorticks_on()
    ax2.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.16, 1.02),
               handlelength=1.5, borderaxespad=0.2)
    fig.subplots_adjust(wspace=0.40)

    for figure_dir in FIGURE_DIRS:
        fig.savefig(figure_dir / "figure2.png", format="png", dpi=300)
    plt.close(fig)
    refresh_output_manifest()


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("This script takes no command-line arguments.")
    main()
