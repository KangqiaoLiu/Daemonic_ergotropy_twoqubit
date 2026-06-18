# Two-Qubit Daemonic Ergotropy

This repository contains Python code and numerical outputs for reproducing the
two-qubit daemonic-ergotropy calculations.

## Layout

- `Code/closed_form_daemonic_ergotropy.py`: closed-form functions for the
  concurrence and visibility calculations.
- `Code/twoqubit_daemonic_ergotropy_numerics.py`: full numerical validation for
  analytic families, random pure states, and random Hilbert-Schmidt mixed
  states.
- `Code/plot_closed_form_figure.py`: regenerates the closed-form figure and its
  associated CSV outputs.
- `Code/plot_figures.py`: regenerates the remaining figure files
  from existing files in `Results/`.
- `Code/figure_io.py`: shared output paths, plotting style, and deterministic
  hash helpers.
- `Results/`: generated CSV, NPZ, and JSON outputs.
- `Figures/`: generated figure PNG files.

The plotting scripts write figure PNG files to `Figures/` when they are run.

## Environment

The pinned Python packages are listed in `Code/requirements.txt`. From the
repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r Code/requirements.txt
```

## Regeneration

Regenerate the closed-form figure and visibility CSV:

```bash
python Code/plot_closed_form_figure.py
```

Regenerate figure files from existing numerical outputs:

```bash
python Code/plot_figures.py
```

Run the full numerical validation:

```bash
python Code/twoqubit_daemonic_ergotropy_numerics.py
```
