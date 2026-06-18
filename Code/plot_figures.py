"""Plot paper figures from the numerical result files."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

from figure_io import (
    FIG4_FONT_RC,
    OKABE_ITO,
    RESULTS,
    configure_plot_style,
    refresh_random_mixed_hashes,
    save_named_figure,
)


def plot_aligned_region() -> None:
    data = np.load(RESULTS / "aligned_diagonal_projective_regions.npz")
    x = data["t3_abs_grid"]
    y = data["tperp_grid"]
    region = data["region"]
    a_abs = float(data["a_abs"])
    threshold_x = data["threshold_t3_abs_values"]
    threshold_y = data["threshold_tperp_values"]

    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    ax.contourf(x, y, region, levels=[-0.5, 0.5, 1.5],
                colors=["#E8E8E8", "#C8DCEC"], alpha=1.0)
    ax.plot(threshold_x, threshold_y, color=OKABE_ITO["black"], linestyle="--",
            linewidth=1.2, label=r"$t_\perp^2=t_3^2-a^2$")
    ax.plot([a_abs, a_abs], [0, 0.02], color=OKABE_ITO["black"], linewidth=0.8)
    ax.text(0.17, 0.86, "equatorial\nmaximizer", transform=ax.transAxes,
            ha="center", va="center")
    ax.text(0.76, 0.17, "axial\nmaximizer", transform=ax.transAxes,
            ha="center", va="center")
    ax.set_xlabel(r"$|t_3|$")
    ax.set_ylabel(r"$t_\perp$")
    ax.set_xlim(0, 1.15)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, loc="upper right", handlelength=2.4)
    save_named_figure(fig, "figure1.png")
    plt.close(fig)


def plot_werner_check() -> None:
    werner = np.genfromtxt(RESULTS / "werner_line_projective_optimum.csv", delimiter=",", names=True)
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(3.45, 3.15), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.18},
    )
    ax.plot(werner["p"], werner["L_projective_exact"],
            color=OKABE_ITO["black"], linestyle="-", label=r"$|p|$")
    ax.plot(werner["p"], werner["L_projective_numerical"],
            color=OKABE_ITO["blue"], linestyle="none", marker="o",
            markerfacecolor="none", label="numerical")
    ax.set_ylabel(r"$\mathcal{L}_{\rm proj}^{\star}$")
    ax.legend(frameon=False, loc="upper center", ncol=2, handlelength=2.2)
    ax.text(0.02, 0.94, "(a)", transform=ax.transAxes, va="top", ha="left")
    axr.axhline(0.0, color=OKABE_ITO["black"], linewidth=0.8)
    axr.plot(werner["p"], werner["residual"] * 1e16,
              color=OKABE_ITO["vermillion"], linestyle="none", marker="s",
              markerfacecolor="none")
    axr.set_xlabel(r"$p$")
    axr.set_ylabel(r"res. ($10^{-16}$)")
    axr.text(0.02, 0.88, "(b)", transform=axr.transAxes, va="top", ha="left")
    save_named_figure(fig, "figure3.png")
    plt.close(fig)


def plot_random_mixed_check() -> None:
    data = np.genfromtxt(RESULTS / "random_mixed_state_povm_equality.csv", delimiter=",", names=True)
    idx = data["index"].astype(int)
    abs_delta = np.maximum(data["abs_delta_L"], 1e-18)
    closure = np.maximum(data["partial4_closure_residual"], 1e-18)

    with plt.rc_context(FIG4_FONT_RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.06, 2.795), sharex=False)
        ax1.semilogy(idx, abs_delta, color=OKABE_ITO["blue"], marker="o",
                     markerfacecolor="none", linestyle="-", linewidth=1.0)
        ax1.axhline(1e-15, color=OKABE_ITO["gray"], linestyle="--", linewidth=0.9)
        ax1.set_xlabel("state index")
        ax1.set_ylabel(r"$|\Delta\mathcal{L}|$")
        ax1.text(0.02, 0.94, "(a)", transform=ax1.transAxes, va="top", ha="left")
        ax2.semilogy(idx, closure, color=OKABE_ITO["green"], marker="s",
                     markerfacecolor="none", linestyle="-", linewidth=1.0)
        ax2.axhline(1e-15, color=OKABE_ITO["gray"], linestyle="--", linewidth=0.9)
        ax2.set_xlabel("state index")
        ax2.set_ylabel(r"$\varepsilon_{\rm comp}$", fontsize=20)
        ax2.text(0.02, 0.94, "(b)", transform=ax2.transAxes, va="top", ha="left")
        fig.subplots_adjust(wspace=0.40)
        save_named_figure(fig, "figure4.png")
        plt.close(fig)
    refresh_random_mixed_hashes()


def main(argv: list[str] | None = None) -> None:
    if argv:
        raise SystemExit("This script does not take command-line arguments.")
    configure_plot_style()
    plot_aligned_region()
    plot_werner_check()
    plot_random_mixed_check()


if __name__ == "__main__":
    main(sys.argv[1:])
