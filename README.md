# LENS: minimal paper reproduction package

This snapshot contains the numerical implementation and the original numerical
outputs needed for Sections 4 and 5 of the accompanying planar Monge--Ampere paper.
It is organized around the experiments actually retained in the paper, including
unfavorable results and all five noise/PINN seeds used in each reported ensemble.
It does not contain the manuscript, old code, superseded experiments, exploratory
parameter searches, temporary files, or previously generated figures.

## Quick start: verify and export, without training

Run these commands from the extracted `LENS_reproducibility` directory:

```sh
python -m pip install -r requirements.txt
python tools/verify.py
python reproduce.py export
```

The last command reconstructs the tables and all eight paper figures under
`reproduced/`. It reads only the packaged original records/models. It does not
run training or require LaTeX, a manuscript file, a GPU, the original computer,
or access to the original project directory. Reproducing figures requires the
numerical packages because several plots evaluate saved model coefficients.

For the included mathematical regression checks:

```sh
python reproduce.py check
```

The original runs used Python 3.14.6, NumPy 2.5.1, SciPy 1.18.1,
PyTorch 2.14.0+xpu and Matplotlib 3.11.1. All computations used CPU float64
and four threads; the XPU extension was not used. `requirements.txt` specifies
the corresponding public package versions, including ordinary PyTorch 2.14.0.
This package has been exercised using the recorded local environment; installing
into a completely new Python environment has not been tested.

## Where the results come from

| Paper content | Original records retained here |
|---|---|
| Stable coordinates, constant-source Dirichlet accuracy, order/sample sweeps, square/disk baselines | `exp/release_2026-11-05/dirichlet/` |
| Peanut PINN and convex FDM baselines | `exp/Section4.2_Peanut/out/supplement_2026-11-06/` and `fdm_continuation_2026-11-06/` |
| Non-family variable-source Dirichlet examples on S, D, T and U | `exp/Section4.3_RHS/out/nonfamily_2026-11-09/` |
| Two in-family OT examples | `exp/release_2026-11-05/transport_K16/` |
| Current OT cross-geometry table | `exp/release_2026-11-05/transport_K24/` plus `exp/Section4.4_Transport/out/superellipse_2026-11-11/main/` |
| Nine OT order/weight cells | `exp/release_2026-11-05/transport_parameters/`, with the K=24, lambda=0.4 cell read from the main cross-geometry run |
| Corner comparisons | `exp/release_2026-11-05/corners/` plus `exp/Section4.4_Transport/out/superellipse_2026-11-11/corners/` |
| OT monotone-FDM comparison on S to D, f1 | `exp/Section4.4_Transport/out/fdm_S_D_f1_2026-11-11/` |
| Inverse LENS/FDM comparison: two sources, 0 and 1% noise, 24 runs | `exp/Section5_Inverse/out/budget200_2026-11-19/` |
| Six independent n=640 reference fields and boundary observations | `exp/Section5_Inverse/out/reference_v1/` |

Region codes in the paper are S=square, D=disk, T=triangle, U=fourth-order
superellipse, and P=peanut. Immutable historical OT run IDs use P for the
triangle; the exporter relabels it as T. Historical ellipse results are excluded
from the current cross-geometry and corner tables. The disk quadratic source
record in the Dirichlet folder is retained because it supplies the common box
for the coordinate comparison, rather than a separate current variable-source test.

Raw JSONL lines were selected by experiment identity, not by achieved accuracy;
the retained lines are byte-for-byte original. Saved NPZ arrays and individual
JSON files were copied byte-for-byte. `PROVENANCE.json` records original file
checksums, every retained run ID and all packaging-only code adjustments.
For each retained experiment, the available model checkpoints and required
sampled fields are included. Redundant boundary fields are omitted when they
can be reconstructed from the retained model coefficients. All inverse snapshots
are retained because they support the error-time curves and the 100-second table.

The inverse fits originally ran for up to **200 seconds**. The paper table uses
the last saved feasible state at or before **100 seconds** (records are about
five seconds apart); the time curves display 0--100 seconds. Early numerical
stopping holds the final value. The source-profile figure uses the 200-second
final states and shows LENS only; the table and time curves compare both methods.
Noisy means and sample standard deviations include all five seeds. Normal
observations are noisy; the prescribed Dirichlet data and evaluation responses
are clean. Source A is legacy reference C, and source B is legacy reference A.

## Start fresh numerical fits

Always use a fresh output directory. Saved timings vary with hardware and system
load. In a fixed-time inverse run, hardware speed also changes the number of
completed iterates, so newly trained errors need not equal the original table.
The recorded numerical stopping rules and initialization budgets are preserved.
On Windows, inverse timing runs require AC power when the battery API is available;
on other systems the unavailable power status is recorded without claiming AC power.

Inspect the complete sequential command plan first:

```sh
python reproduce.py train --out reruns --plan
```

The OT monotone-FDM baseline additionally requires the authors' external
`ma-ot-monotone` implementation and `agd==0.2.16`. They are not vendored here.
The helper fetches only `main/ma.py` at the exact recorded commit and verifies its
SHA-256 before writing it. Internet access is required only for dependency setup.

```sh
python -m pip install -r requirements-ot.txt
python tools/fetch_ot_baseline.py
python reproduce.py train --out reruns
python reproduce.py export --rerun-root reruns --out rerun_results
```

Alternatively, supply an existing checkout using `--upstream /path/to/ma-ot-monotone`.
The original commit is `d84acdc3d18614a174ff6b4b39b3d3aaa01396ee`:
<https://github.com/gbonnet1/ma-ot-monotone>.
The method is described in G. Bonnet and J.-M. Mirebeau,
*Monotone discretization of the Monge--Ampere equation of optimal transport*,
ESAIM: M2AN 56(3), 815--865 (2022). The bundled adapter retains the original
support-function boundary discretization and the documented grid-gradient
reconstruction used for the comparison.

To run a subset, use `--section forward`, `--section inverse`, or
`--section ot-baseline`. These subsets use the same output directory layout and
can be completed successively before exporting the full fresh experiment set.
The inverse rerun reads the supplied independent n=640 observations. Recomputing
those six reference fields is optional and potentially expensive:

```sh
python -B exp/Section5_Inverse/reference.py --sources C,A --ns 640 --out new_reference
```

This last command creates reference data only. To train against newly generated
references, explicitly change `REFERENCE` in `run_budget.py` and the reference
path in `audit_budget.py`; use a new fit output directory. The default reproduces
the published observations exactly, rather than silently replacing the labels.

## Integrity and scope

`MANIFEST.json` contains SHA-256 checksums of every packaged file except itself.
`EXPECTED_TABLES.json` checks the two complete frozen table exports. Numerical
tables are verified independently of manuscript layout; generated forward tables
may combine blocks that the manuscript presents as separate tables.

`lens/` and `refs/` contain the numerical implementation; `exp/` contains the
required experiment/plot entry points; `tests/` contains the relevant mathematical
regression scripts. Packaging adjustments are limited to output selection,
standalone exports, optional Windows power telemetry and avoiding unused pilot
runs. Numerical solver modules are unmodified. Historical fingerprints inside raw
records describe the original runs and are not rewritten to match this packaging.

This archive is a reproducibility snapshot, not an upload or a license grant.
No new license has been assigned by packaging. External dependencies retain
their own licensing terms. See `VALIDATION.json` for the checks actually completed
when the archive was prepared.
