# Data and model records

All numerical archives are local candidate copies pending permission, with
unchanged bytes and original SHA-256. `manifests/datasets.json` identifies
canonical paths, fields, dtypes, time lists and shapes. Identical diagnostic
archives are shared by hash, instead of copied into each regime folder.

| Dataset | Frames | Times | Physical change |
|---|---:|---|---|
| baseline/ic1, baseline/ic2 | 50 each | 0.1–5.0 s | Independent IC phases |
| assisted/ic1, assisted/ic2 | 100 each | 0.1–10.0 s | Same reference settings |
| filter_small/ic1, ic2 | 50 each | 0.1–5.0 s | delta = 0.1884955592 m |
| filter_large/ic1, ic2 | 50 each | 0.1–5.0 s | delta = 0.3769911184 m |
| half_Re/ic1 | 50 | 0.1–5.0 s | mu = 0.002 Pa s, nu = 2e-6 m²/s |
| double_Re/ic1 | 50 | 0.1–5.0 s | mu = 0.0005 Pa s, nu = 5e-7 m²/s |
| forced_strong/ic1 | 50 | 0.1–5.0 s | Source 10 sin(4y/1m) N/m³ |

The principal Gaussian width is 0.25132741228718347 m. Exports have 101×101
periodic endpoint points; the reader drops duplicate endpoints to 100×100
without interpolation. STAR mesh base-size parameter 0.296 m is not a uniform
sample spacing. The reference is a grid-refined numerical reference, not DNS
or exact truth. Mesh cell count is 566,048, dt=0.005 s, STAR-CCM+2510,
Segregated Flow, second-order convection/implicit time, and five fixed inner
iterations per physical step. Those five iterations are not a residual tolerance.

## Physical fields

| Field | Shape | Unit / convention |
|---|---|---|
| `u`, `v` | (time,100,100) | Filtered velocity, m/s |
| `omega` | (time,100,100) | Filtered dv/dx − du/dy, 1/s |
| `q_sgs_x`, `q_sgs_y` | (time,100,100) | G(u_i omega) − G(u_i)G(omega), m/s² |
| `div_q_sgs` | (time,100,100) | Physical full spectral divergence, 1/s² |
| `pi_sgs` | (time,100,100) | −q · grad(filtered omega), 1/s³ |
| `K`, `Z` | (time,) | Half mean squared speed (m²/s²), half mean omega² (1/s²) |
| `E`, `T_sgs` | (time,72) | Radial shell energy and enstrophy source spectrum |
| `time_s` | (time,) | Stored physical time in seconds |
| `x`, `y` | (100,) | Unique periodic coordinates, m |

`initial_state.npy` is exactly the first frozen omega frame, separately stored
so free-running cannot accidentally decompress future states. Diagnostic
archives hold `div_q_solver_matched`, reported separately from physical SGS.
Transfer spectra obey sum(T_sgs) = −mean(Pi) with the stated spectral convention.
Relative spectral L1 scores use shells 1–33. Known forcing curl is outside
SGS flux/source labels.

## Information budget and models

For each seed: 400 fixed flow sites (omega,u,v), 25 SGS sites per time selected
from the same training cloud, and 100 disjoint held-out spatial sites. Importance
weights, SGS schedules, and full time lists are stored in `data/splits/`.
All 50 times belong to one IC-1 trajectory; spatial validation is not temporal
or independent-flow validation. IC-2 and transfer regimes never select weights.

`configs/` preserves every effective numerical setting, with only data/output
paths edited; original/config-candidate hashes are recorded. The original
solver normalizes its 12 input maps by detached current-state RMS as defined
in `solver.py`. Checkpoints retain saved scales and closure output scale.
There is no new normalization fit on IC-2 or transfer regimes. CLI changes
only delta's explicit physical maps, molecular viscosity, and known forcing.

Stored formal results include Full/2027 despite missing exact weights. Full/2026
uses the provenance-declared recovery01 identity; it is not represented as
recovery of the original empty file. Registry checks prevent substitution.

## Access and approved distribution scope

The authors authorized publication of code, approved processed reference data,
verifiable weights and original figures on 12 October 2026. These files are
included in this review branch; it is not yet a merged/tagged release.
Two 100-frame physical archives use Git LFS; all other approved assets are
ordinary Git files. No raw STAR CSV collection or proprietary .sim is added.
Raw file hashes/names remain in the processing provenance. Existing old CSVs
remain in Git history, which is preserved. A specific reuse license is not
finally selected; no MIT/CC BY grant or invented archive URL/DOI is asserted.
See the per-file ledger and docs/LICENSING_STATUS.md.

## Fixed Hybrid and 50-frame spectral evidence

See README new evidence commands. Three source definitions are distinct: physical-target compares projected model to unprojected reference (the original frozen score); operator-matched compares both projected; full-Fourier constitutive compares neither projected. All 72 shells are retained; official transfer L1 averages per-frame relative discrepancies over shells 1–33, not the discrepancy of mean spectra. Positive T means resolved enstrophy gain and the complete shell sum equals -mean(Pi). Fifty correlated frames are not independent uncertainty replicates. Table S5 and S17 use the two ICs and all three paired seeds, with sample SD across seed-level values only.
