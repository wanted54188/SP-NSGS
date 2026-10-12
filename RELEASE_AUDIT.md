# PoF final reproducibility upgrade - release audit

Date: 12 October 2026. Author publication permission is confirmed. This is a
review-branch upgrade, not a merged/tagged release. No training, STAR run,
complete new neural evaluation, coefficient calibration or checkpoint reselection
was performed. The original scientific project and release candidate are retained.

## Remote baseline and history

Remote main was fetched and verified as
`e88fbdc77369642a56fa89c923a8f7517ca9710a`.
It contains the old Full/ISO code and two old weights. Its README says reference
data are not included, but its tree has 202 raw STAR CSV paths representing 101
unique CSV blobs in two duplicate directories. Those paths and the old weights
are removed from this branch's current tree; both existing commits and all
historical objects are preserved. Historical raw data and existing author/e-mail
commit metadata remain publicly recoverable. No history rewrite/force push occurs.

## Approved shipped scope

- Original four-cell neural factorial, all three seeds, frozen trainer,
  normalization, 400/25/100 split indices and held-out closure-only selection.
- Eleven verifiable weights: ISO/A1B0/A0B1 2026-2028; Full/2028; explicitly
  identified Full/2026 recovery01. Exact Full/2027 is missing; all its historical
  scores remain, and evaluation refuses substitution.
- Eleven processed reference/diagnostic datasets (ICs, 10 s assisted references,
  filter tests, viscosity changes and known forcing) with SHA-256 identities.
- Leith/Clark/Mixed, original sparse calibration coefficients and all frozen results.
- Eight main and seventeen Supplement publication assets, byte-matched to current
  author-reviewed TeX dependencies; complete available plotting inputs.
- M19 physical Pi snapshots, own-negative/reference-negative conditional PDFs and
  TP/FP/FN/TN overlap; ISO empty own-negative PDF remains undefined.
- M21/M22: four complete 50-frame spectral records, 72 shells, 200 model states.
  Three source conventions remain separate. Positive T means resolved enstrophy
  gain; complete shell sum equals -mean(Pi). Official L1 is the mean per-frame
  relative discrepancy over shells 1-33, not L1 of mean spectra.
- Fixed Clark-A0B1 alpha=.5 flux mixing at every current-state RK4 stage;
  twelve archived IC/mode/seed results, Table S5 and Fig. S17 inputs and renderer.
  No adaptive gate, alpha sweep interface, unpublished mechanism study, internal
  submission/reviewer correspondence or proprietary .sim project is included.

## Reproduction gates

|Gate|Result|
|---|---|
|Static current/history secret-pattern scan|PASS; 128 historical blobs, no listed credential patterns|
|Local absolute paths in runtime Python literals|No matches|
|Original available model/config/reference SHA identities|PASS|
|Independent repository snapshot|PASS; outside research project, no borrowed scientific assets|
|Installation|PASS; existing isolated third-party environment reused, package reinstalled from new snapshot|
|Isolated-mode import (-I)|PASS; resolves only the new snapshot package|
|Dependency check|PASS; no broken requirements|
|Lightweight tests|10 passed; includes original A0B1 single-frame metrics|
|Hybrid endpoints|Alpha 0/1 single-frame q/source/Pi/RHS exactly equal A0B1/Clark|
|Hybrid .5|Single-frame frozen metrics within float32 tolerance; all archived Table S5 means and Fig. S17 curves verified|
|M21/M22|All 200 source Parseval/transfer records reproduce frozen per-frame metrics; model and reference budget signs verified|
|M19/M21/M22 plotting|Four exported CSVs exactly reproduce original figure data|
|Fig.8 redraw|PASS; common full range also includes mean+SD (upper limit 4.398315); frozen publication asset unchanged|
|Hybrid CLI|Preflight PASS; full computations require --execute|

Windows test setup required a short temporary build directory to avoid long-path
wheel creation failures. This is installation scaffolding, not a scientific
data dependency. Missing wheel build support was corrected in pyproject.toml.
The original numerical code, reference archives and weights are not rewritten.

## Alpha provenance and scientific limits

V3 explored alpha=0,.05,.1,.25,.5,1 on IC-1 in both evaluation modes. Its original
sparse-only selection chose alpha=1. V4 adopted .5 after viewing IC-1 a-priori
and free-running results, locked it before the six new IC-2 runs, and did not
claim a preregistered IC-1 optimum. Historical IC-2 endpoints had already been
inspected: the phase-realization check is not completely blind. IC-1 mean omega
error .589+/- .009 becomes .563+/- .029, with only .0023 improvement for seed2026.
The combined correction does not inherit A0B1's limiter radius or global stability.
Clark's stronger reference-state fidelity and Full/2027's unfavorable history
remain in the archived evidence. Time frames are correlated; sample SD uses
the three actual seed-level values, not 50 independent realizations.

## Size and delivery

Pre-ledger/pre-audit snapshot: 1076 files, 435,169,843 bytes (415.010 MiB).

Two assisted physical archives use Git LFS; all remaining code/assets/weights are normal Git. The mandatory SHA verifier detects missing LFS hydration. No file exceeds 100 MiB. GitHub warns above 50 MiB and blocks ordinary blobs above 100 MiB ([official limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)).

- `data/reference/assisted/ic1/physical.npz`: 53,882,541 bytes; SHA-256 `9d43c963a39501bbd3da1b7ab3b64f94ee87a9b1a7e398156be31e9a55ed701d`.
- `data/reference/assisted/ic2/physical.npz`: 53,888,223 bytes; SHA-256 `1e8b69a51fce4bce778e216a50cbb7bcb45e96428d74fdeab84cbd56a628b25d`.

## Outstanding before a tagged release

1. Review/merge the PR; no direct main push, tag or GitHub Release is created here.
2. Copyright holders must select actual reuse licenses. No LICENSE, MIT or CC BY
   grant is invented; distribution permission and reuse licensing are distinct.
3. Exact Full/2027 weights remain unavailable; archived results do not restore them.
4. Old raw CSVs remain in history. Any history cleanup requires a separate decision.
5. Manuscript availability/E08/S1 must be synchronized to the actual merged
   commit/tag and accessible assets after formal publication, not preemptively.

Suggested tag after merge and checks: `v2.0.0-pof-revision-20261012`.
No accepted-publication DOI or universal accuracy/generalization claim is made.

## Post-push clone correction

A real anonymous GitHub clone and remote LFS hydration found two historical
Python files stored with LF bytes instead of staging CRLF. Their parsed code
and every scientific asset were unchanged. Provenance was aligned to actual
committed bytes, and verify now checks every shipped source-lineage artifact.
No frozen numerical result or checkpoint changed.
