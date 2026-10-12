# Reproducibility

Run commands from the repository root. `python -m spnsgs COMMAND --help`
documents the actual supported parameters. All full computational commands
require explicit `--execute`; verification, plotting and tabulation do not.
Output targets must be new and outside frozen/input directories.

## 1. Prepare reference from author-supplied raw exports

```console
python -m spnsgs prepare-reference --raw-dir "PATH_TO_RAW_IC1" --ic ic1 --regime baseline --output outputs/rebuilt_ic1
python -m spnsgs prepare-reference --raw-dir "PATH_TO_RAW_IC1" --ic ic1 --regime baseline --output outputs/rebuilt_ic1 --execute
python -m spnsgs prepare-reference --raw-dir "PATH_TO_RAW_IC1" --ic ic1 --regime filter_large --output outputs/rebuilt_large --execute
python -m spnsgs prepare-reference --raw-dir "PATH_TO_RAW_STRONG_FORCED" --ic ic1 --regime forced_strong --output outputs/rebuilt_forced --execute
```

Expected STAR columns: `Position[X] (m)`, `Position[Y] (m)`, `Velocity[i] (m/s)`,
`Velocity[j] (m/s)`, `Vorticity[k] (/s)` (the exact reader's column map is in
`data.py`). Filename pattern comes from the frozen config. Required times are
0.1,0.2,...,5.0 s (or `--end-time 10`). Preflight checks complete cadence,
first-grid construction and raw identities for baseline/refiltered data.
Reference construction is no interpolation, exp(−delta²|k|²/24) filtering,
physical product residuals and full Fourier derivatives. Existing canonical
archives are not rewritten. The builder's exports are new artifacts; it does
not register an arbitrary new archive as an official model-selection dataset.

Strong source 10 N/m³ divided by rho=1000 gives F=0.01 m/s². With k=4 m⁻¹,
filtered curl is −k F exp(−delta²k²/24) cos(ky). The solver includes it once
outside SGS labels. Re cases keep U*=0.15 sqrt(70) m/s and L*=2pi m fixed;
Re/Re0 = nu0/nu. These reporting scales do not alter network features.

## 2. Train with the original sparse protocol

```console
python -m spnsgs train --variant iso --seed 2026 --output outputs/train/iso_2026 --device cuda --execute
python -m spnsgs train --variant A1B0 --seed 2026 --output outputs/train/A1B0_2026 --device cuda --execute
python -m spnsgs train --variant A0B1 --seed 2026 --output outputs/train/A0B1_2026 --device cuda --execute
python -m spnsgs train --variant full --seed 2026 --output outputs/train/full_2026 --device cuda --execute
```

Repeat each command for seeds 2027 and 2028 with unique output paths. No frozen
checkpoint is loaded as training initialization. Preflight reconstructs all
spatial schedules from IC-1 and compares them against released split indices
and weights. The processed-archive adapter supplies exactly the seven fields
used by the original trainer; no new supervision is added. An optional
`--raw-dir` must contain the source-hash-verified IC-1 exports.

Formal trainer function bodies are extracted unchanged from the source whose
hash was frozen in original run manifests, split into `training.py`,
`protocols.py` and `evolution.py`. AdamW, cosine learning-rate schedule,
curriculum, EMA, loss weights and 12-feature architecture remain unchanged.
The effective configs expose all settings. Primary selection is minimum
held-out reference-state closure score:

`0.40*mean_div_L2 + 0.15*endpoint_div_L2 + 0.15*(1-r_div) + 0.15*(1-r_Pi) + 0.15*abs(negative_fraction_error)`.

Assisted/free-running/test outcomes are excluded. Re-training produces a new
realization and is not claimed to recover missing original weights exactly.

## 3. Frozen evaluation matrix

Supported models: `iso`, `A1B0`, `A0B1`, `full`, `leith`, `clark`, `mixed`.
Use formal neural seed or classical calibration split `--seed 2026/2027/2028`;
omit seed for deterministic Clark. Every output is independent.

```console
python -m spnsgs evaluate --model leith --seed 2026 --ic ic1 --mode apriori_reference_state --output outputs/leith_ic1_ap --execute
python -m spnsgs evaluate --model mixed --seed 2026 --ic ic2 --mode free_running_continuation --output outputs/mixed_ic2_fr --execute
python -m spnsgs evaluate --model A0B1 --seed 2026 --ic ic1 --regime forced_strong --mode apriori_reference_state --output outputs/A0B1_force_ap --device cuda --execute
python -m spnsgs evaluate --model A0B1 --seed 2026 --ic ic1 --regime half_Re --mode free_running_continuation --output outputs/A0B1_halfRe_fr --device cuda --execute
```

Regimes: `baseline`, `filter_small`, `filter_large`, `half_Re`, `double_Re`,
`forced_strong`. IC-2 transfer reference exists for filter tests, not the
viscosity/forced tests. Original 12 neural configurations are frozen; only
explicit delta/viscosity/known-force physical mappings change. Leith and Mixed
reuse the three original IC-1 sparse coefficients. Clark's gradient coefficient
is one (delta²/12 support), without fitting.

Genuine free-running initializes once from the separate first-frame NPY. It
cannot access reference arrays until evolution ends. No sensors or adaptation
enter this path. RK4, five substeps/stored interval, and rectangular 2/3
projection are unchanged. Baseline neural/classical stored timestamps use
float32 differences as in formal evaluation; transfer tests retain their
original float64 schedule. This distinction is preserved rather than silently
standardized. CPU/CUDA float32 kernels can differ slightly; original and
refactored CPU single-frame q/div/Pi outputs matched exactly in local checks.

The shared scorer retains physical versus operator-matched source errors,
state/global metrics, transfer-spectrum L1, negative-region classification and
scale-aware budget validation. Additional amplitude/source diagnostics retain
the formal definitions. A nonfinite continuation records its first failed
time and scores only completed frames, without resetting or censoring failures.

## 4. Observation-assisted reconstruction

```console
python -m spnsgs evaluate --model A0B1 --seed 2026 --ic ic2 --mode observation_assisted --output outputs/A0B1_ic2_assisted --device cuda --execute
```

Baseline neural models only, t=0.1–10 s. Frozen 400 flow sites, omega/u/v every
0.1 s. Corrections occur after each stored interval's RK4 advances, before
storing the corrected state; no correction at initial frame. The exact gains
and lifting length scales come from each original effective config. Windows
0.1–5, 5–10 and 0.1–10 share the t=5 boundary as recorded; uncertainty is
across seeds, not time frames. This is reconstruction, not autonomous prediction.

## 5. Classical calibration reproduction (optional)

```console
python -m spnsgs calibrate --seed 2026 --output outputs/calibration_2026
python -m spnsgs calibrate --seed 2026 --output outputs/calibration_2026 --execute
```

This reproduces the original weighted nonnegative scalar fit on original IC-1
25-site/time vector labels only, checking the result against the frozen
coefficients. Never use new-regime labels to calibrate a zero-shot comparison.
The evaluator always uses `data/calibration/`, not newly generated output.

## 6. Figures, tables and verification

```console
python -m spnsgs verify
python -m pytest -q
python -m spnsgs figures --selection all --output outputs/figures --dpi 600
python -m spnsgs tables --output outputs/tables
```

Eight main and seventeen Supplement assets are selected by actual current TeX
references, including filename gaps inherited from manuscript numbering.
The registry records current TeX/Supplement hashes; no layout inferred from
dates or old numbering is used. Figures use frozen numeric CSVs and reference
arrays. `published_exact/` copies original bytes; `redrawn/` regenerates
scientific panels with explicit layouts, and records that pixel identity is
not claimed. Tables retain n=3 sample SD, deterministic Clark and paired
differences; archived LaTeX tables are also provided unchanged.

The read-only verifier validates archive shapes/time grids/hashes, all available
weights/config hashes, 400/25/100 splits, 72 factorial records, 60 classical/
neural cross-regime rows, and the 12 rounded amplitude-table rows. Missing
Full/2027 is a disclosed availability limit, not a removed result.


## Fixed Hybrid and 50-frame spectral evidence

See README new evidence commands. Three source definitions are distinct: physical-target compares projected model to unprojected reference (the original frozen score); operator-matched compares both projected; full-Fourier constitutive compares neither projected. All 72 shells are retained; official transfer L1 averages per-frame relative discrepancies over shells 1–33, not the discrepancy of mean spectra. Positive T means resolved enstrophy gain and the complete shell sum equals -mean(Pi). Fifty correlated frames are not independent uncertainty replicates. Table S5 and S17 use the two ICs and all three paired seeds, with sample SD across seed-level values only.

Scientific hashes are byte-based. Git attributes intentionally disable automatic line-ending conversion for code, frozen records and assets.
