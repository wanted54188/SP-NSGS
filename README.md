# SP-NSGS

This repository contains the public code and configuration package for the
structure-preserving neural subgrid-scale (SP-NSGS) closure used in the
accompanying two-dimensional filtered-vorticity study. The implemented closure
is

\[
\mathbf q_{\mathrm{sgs}}=\mathbf q_d+\mathbf q_b.
\]

The dissipative term combines a positive local diffusivity with a
symmetric-positive-definite anisotropy tensor. The bounded signed correction
permits local reverse SGS transfer while limiting its magnitude. Frozen Full
SP-NSGS and independently trained isotropic-dissipative (ISO) checkpoints are
included for evaluation and reproducibility.

## Contents

```text
spnsgs/       Solver, closure, data interface, training, and validation code
configs/      Full SP-NSGS, ISO, and sparse-observation configurations
scripts/      Training, evaluation, and formal experiment entry points
checkpoints/  Frozen Full SP-NSGS and ISO checkpoints
tables/       Frozen summary tables reported in the manuscript
tests/        Solver and structural-constraint tests
```

The manuscript, final figures, and archived frozen field data are distributed
separately from this source package.

## Installation

The formal runs used Python 3.11 and PyTorch with CUDA. Install the minimal
Python dependencies with:

```bash
python -m venv .venv
.venv\\Scripts\\activate  # Windows
pip install -r requirements.txt
```

For GPU runs, install the PyTorch wheel compatible with your CUDA driver from
the official PyTorch installation selector.

## Reference data

The filtered reference snapshots are not included. Place CSV files matching
`XYZ_内部表_table_*.csv` in a local `data/` directory at the repository root,
or change `data.data_dir` in a configuration file. The `data/` directory is
ignored by Git to prevent accidental publication of the reference data.

## Reproducing a run

Train Full SP-NSGS:

```bash
python scripts/train.py --config configs/main.json
```

Train the independent ISO baseline:

```bash
python scripts/train.py --config configs/iso.json
```

Evaluate an included checkpoint:

```bash
python scripts/evaluate.py --config configs/main.json --checkpoint checkpoints/full_sp_nsgs.pt
python scripts/evaluate.py --config configs/iso.json --checkpoint checkpoints/iso_dissipative.pt
```

The formal sparse protocol uses flow-sensor fractions of 1%, 2%, and 4% with
seeds 2026, 2027, and 2028. Its reported aggregate values are preserved in
`tables/sparse_results.csv`.


