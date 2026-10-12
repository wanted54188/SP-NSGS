# SP-NSGS: Physics of Fluids reproducibility package (review branch)

**Author-approved PoF reproducibility upgrade, prepared on a review branch.**
This branch is not a tagged release and does not claim acceptance of the paper.
Existing Git history is preserved. The default branch's old two-model package
is superseded only after this pull request is reviewed and merged.
See [release audit](RELEASE_AUDIT.md) and [licensing status](docs/LICENSING_STATUS.md).

## Method and evidence

The vorticity SGS flux is decomposed as `q = q_d + q_b`. The dissipative
component `q_d = -nu_d A grad(omega)` has an SPD tensor and pointwise
nonnegative enstrophy transfer. The bounded, sign-indefinite correction can
carry local forward and reverse transfer. This branch guarantee is distinct
from stability of the complete closure.

The formal paired factorial uses ISO, A1B0, A0B1 and Full, seeds 2026–2028.
The signed correction supplies the dominant reproducible gain. Clark has
strong instantaneous SGS fidelity; A0B1 has lower mean free-running trajectory
error in this benchmark. Full/2027's high-error realization remains in every
frozen evidence table. The tests comprise IC-1/IC-2, two unseen filter widths,
two viscosity changes, and known periodic forcing under the recorded settings.

## Installation

From this repository's root, in Python 3.10 or later:

```console
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m spnsgs --help
python -m pytest -q
```

Install the appropriate PyTorch CPU/CUDA build for your hardware. Local tests
used PyTorch 2.5.1+cu121, NumPy 2.1.3, pandas 2.2.3 and Matplotlib 3.10.1.
GPU training is the practical default for full runs; CPU is supported for
preflight and light checks. Full training/rollout hardware requirements were
not re-benchmarked during integration.

## Layout

```text
spnsgs/             solver, closure, training, protocols, metrics, reference, CLI
configs/            12 path-sanitized frozen effective configurations
models/             hash-verified weights and original training records
data/reference/     canonical physical archives, diagnostics, one-frame initial states
data/splits/        exact 400/25/100 spatial layouts for each formal seed
data/calibration/   original IC-1 Leith and Mixed coefficients
data/figure_inputs/ frozen numeric plotting inputs
results/frozen/     per-seed/split results, including failure realizations
figures/published/  exact manuscript-selected figure files
manifests/          hashes, lineage, availability and proposed public-file ledger
outputs/            new runs only; never overwrite frozen material
```

Approved processed archives and all eleven verifiable checkpoints are included
on this branch. Git LFS is used only for the two 100-frame assisted physical
archives (about 51.4 MiB each). After cloning, run `git lfs install` and
`git lfs pull`; the verifier detects unhydrated pointers by hash mismatch.
Other data and weights are ordinary Git files. Raw STAR CSV collections and
`.sim` projects are excluded from the new tree. Old CSVs remain accessible in
historical commits because history was preserved; no history rewrite occurred.

## Reproduce frozen evidence

```console
python -m spnsgs verify
python -m spnsgs figures --output outputs/paper_figures --dpi 600
python -m spnsgs tables --output outputs/paper_tables
```

The figure command recovers the 8 main and 17 Supplement selected originals
in `published_exact/` and regenerates numeric evidence in `redrawn/` as
PNG/PDF/SVG. Original assets are byte-identical; numerical redraws have
explicit new layouts and are **not** advertised as pixel-identical copies.
It performs no model inference. Full per-seed and paired-difference tables
retain Clark's deterministic status and classical calibration replicates.

## Evaluate an available frozen model

```console
python -m spnsgs evaluate --model A0B1 --seed 2026 --ic ic1 --mode apriori_reference_state --output outputs/ic1_A0B1_2026_apriori
python -m spnsgs evaluate --model A0B1 --seed 2026 --ic ic1 --mode apriori_reference_state --output outputs/ic1_A0B1_2026_apriori --device cuda --execute
python -m spnsgs evaluate --model A0B1 --seed 2026 --ic ic2 --mode free_running_continuation --output outputs/ic2_A0B1_2026_free --device cuda --execute
python -m spnsgs evaluate --model A0B1 --seed 2026 --ic ic1 --mode observation_assisted --output outputs/ic1_A0B1_2026_assisted --device cuda --execute
python -m spnsgs evaluate --model clark --ic ic1 --regime filter_large --mode free_running_continuation --output outputs/ic1_Clark_large --device cuda --execute
```

Without `--execute`, evaluation only checks inputs and hashes. Genuine
free-running reads only the separately saved initial state before advancing;
future truth is opened after trajectory completion for scoring. Assisted
reconstruction uses future observations at the frozen 400 sites every 0.1 s.
Neither its error nor the old assisted error of about 0.098 is autonomous
prediction accuracy. Cross-regime forcing is a known equation term supplied
equally to neural and classical models; no coefficient is fitted on a test regime.

## Training

```console
python -m spnsgs train --variant A0B1 --seed 2026 --output outputs/new_training/A0B1_2026
python -m spnsgs train --variant A0B1 --seed 2026 --output outputs/new_training/A0B1_2026 --device cuda --execute
```

This uses the original random initialization and complete original curriculum,
losses, sparse split and held-out closure-only checkpoint rule. It defaults to
the processed frozen IC-1 fields, without refiltering or adding observations.
`--raw-dir` can instead point to original hash-verified STAR exports. Training
produces a **new** run; it is never substituted for the released formal weights.

## Checkpoint availability

Eleven validated files are available: all ISO/A1B0/A0B1 seeds, Full/2028, and
the explicitly identified Full/2026 `seed_2026_recovery01` used by its formal
result manifests. The original Full/2026 file is empty. Full/2027's exact
frozen file is missing (the present research file is empty). Its stored results
can be tabulated/plotted, but its original inference cannot be rerun. The
evaluator stops with `MISSING EXACT FROZEN CHECKPOINT`; no seed/recovery
substitution occurs. New training cannot guarantee recovery of the same weights.

The local trusted checkpoints are hash-checked before pickle-based PyTorch
loading. Do not load untrusted replacement checkpoint files.

## Citation and licenses

This candidate corresponds to the latest author-reviewed PoF revision, not an
accepted publication. No paper DOI is available in the verified release inputs.
Use the manuscript's confirmed citation when available. Publication permission is confirmed, but a reuse license remains unselected.
See `manifests/public_release_inventory.csv` and `docs/LICENSING_STATUS.md`.

Detailed commands and numerical conventions:
[reproducibility](docs/REPRODUCIBILITY.md).

## New spectral, event and fixed-Hybrid evidence

```console
python -m spnsgs verify-paper-evidence
python -m spnsgs paper-evidence --output outputs/s14_s17 --dpi 600
python -m spnsgs hybrid-evaluate --ic ic1 --seed 2026 --mode apriori_reference_state --output outputs/hybrid_ic1_ap
python -m spnsgs hybrid-evaluate --ic ic2 --seed 2026 --mode free_running_continuation --output outputs/hybrid_ic2_fr --device cuda --execute
```

Only fixed `alpha=0.5` is offered as a full evaluation command. Alpha 0/1 are
unit-test endpoint checks, not a public tuning/sweep feature. The blend is
computed from the current resolved state at every RK4 stage, never by mixing
trajectories. All formal weights, normalization and integration settings stay
frozen. IC-1's v3 grid was exploratory and its sparse-only rule selected 1.0.
V4 adopted 0.5 after inspecting IC-1 a-priori and free-running results, before
six new IC-2 runs. Historical IC-2 endpoints were already viewed; this was not
a completely blind test. IC-1 mean vorticity error improves from 0.589±0.009
to 0.563±0.029, with only 0.0023 improvement for seed 2026. The composite
correction does not inherit the original A0B1 limiter radius.

The offline commands reproduce Table S5, Fig. S17, all 50-frame source and
signed-transfer spectra (S14/S15), and physical conditional PDFs/overlap at
the fixed 1 s/5 s snapshots (S16), including ISO and recorded Full/2026
recovery01. Clark's stronger a-priori results and every neural seed remain.
No inference is performed by these plotting/verification commands. No gate,
alpha search or unpublished mechanism study is included.
