# Fitting datasets with different energy zeros

This feature fits one shared ACE potential while allowing a constant
**per-atom energy offset for each source/method group**. It is intended for
datasets whose absolute energy references differ, for example because of
different electronic-structure codes or pseudopotentials. It is disabled
unless `fit.source_energy_offsets` is set.

The offsets are solved analytically at every ACE coefficient vector, not
added as nonlinear optimization variables. One source is the fixed reference.
At fixed ACE coefficients, adding an offset does not change forces. This
does **not** make incompatible physics or poor
numerical settings equivalent; curate the groups before fitting.

## Install

Use the `pacemaker-cpu-energy-offsets` branch of
[`ondracka/python-ace`](https://github.com/ondracka/python-ace/tree/pacemaker-cpu-energy-offsets)
and the isolated environment recipe in [CPU-FITTING.md](CPU-FITTING.md#install-a-separate-environment).
It installs the pinned legacy TensorPotential without local dependency
patches. Both `pacemaker` and `pacemaker-cpu` accept the settings below;
the latter also enables native CPU kernels and persistent batch workers.

## 1. Label the training data

Every row needs a non-missing label in a column such as `source_id`, plus the
normal `ase_atoms`, `forces`, and energy columns. Energies are **total eV per
structure**, forces are eV/angstrom, and offsets are eV/atom. Do not replace
total energies with per-atom energies in the dataframe.

For two *internally consistent* datasets, using only trusted pickle files:

```python
import pandas as pd

first = pd.read_pickle("method_A.pckl.gzip", compression="gzip")
second = pd.read_pickle("method_B.pckl.gzip", compression="gzip")
first["source_id"] = "method_A"
second["source_id"] = "method_B"
combined = pd.concat([first, second], ignore_index=True)
assert not combined["source_id"].isna().any()
combined.to_pickle("combined.pckl.gzip", compression="gzip", protocol=4)
```

Keep original upload, entry and method identifiers as separate provenance
columns. An upload is not necessarily one consistent method: subdivide it
when needed. Conversely, a group may span uploads if its energy reference
and methodology really match. Do not give every structure its own group;
that would remove nearly all energy constraints. Prefer stable labels of a
single type, and keep the reference label's YAML type consistent with the
dataframe (integer `1` and string `"1"` are different labels).

Offsets remove group means, not geometry-dependent or composition-dependent
errors. Within-group energy differences still constrain the fit. Groups
covering disjoint structures can leave relative energies weakly constrained;
overlapping configurations and independent physical validation are valuable.

## 2. Configure the fit

For example, save the following as `input.yaml`, with an existing initial
basis/potential `initial.yaml` and the labelled dataset above. Your normal
inline `potential:` definition can be used instead. To generate a starting
input interactively, run `pacemaker -t` in a new directory and add the source
offset setting to its `fit:` section.

```yaml
seed: 42
potential: initial.yaml

data:
  filename: combined.pckl.gzip
  # One common reference, not a different subtraction for each source.
  # Zero is explicit here so automatic reference estimation is not invoked.
  reference_energy: {Si: 0.0}

fit:
  optimizer: BFGS
  maxiter: 200
  loss:
    kappa: 0.1
    L1_coeffs: 4.0e-7
    L2_coeffs: 4.0e-7
  source_energy_offsets:
    column: source_id
    reference: method_A

backend:
  evaluator: tensorpot
  batch_size: 2
  display_step: 1
```

The reference source must occur in training; its offset stays exactly zero.
If `reference` is omitted, the first encountered training label is used, so
an explicit choice is preferable for reproducibility. The common elemental
`reference_energy` is ordinary Pacemaker preprocessing and is distinct from
the fitted source offsets. Use an appropriate common reference for your
system, and keep it unchanged when restarting/comparing fits. Existing
`energy_corrected` values can instead be used without this option, but check
the log for inherited or automatically estimated reference energies.
Large legitimate cross-code offsets can still trigger Pacemaker's generic
energy-range warning; check units and provenance rather than treating that
warning alone as evidence of an incorrect offset fit.

The loss settings are illustrative, not validated universal parameters.
`kappa=0` fits only energies, `kappa=1` only forces. Use `kappa<1` to include
energy information in coefficient fitting. Existing energy/force weights are
retained and normalized through the usual Pacemaker preparation; source
offsets do not balance group sizes automatically. Avoid selecting or
weighting heterogeneous sources by their unaligned absolute energies.

## 3. Run with 12 single-thread workers

After activating the environment, run from a new fit directory containing
`input.yaml` (and adjust paths to the dataset/initial basis as necessary):

```bash
export PACEMAKER_CPU_THREADS=12 PACEMAKER_CPU_WORKERS=12
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export CUDA_VISIBLE_DEVICES=-1
pacemaker-cpu input.yaml
```

Twelve allowed physical cores and at least twelve batches are required.
Reduce workers/threads for smaller allocations or datasets. Each worker
consumes additional memory; fewer workers may be preferable for large frames.
The startup log records the selected kernel, ISA and worker configuration.
For the ordinary serial TensorPotential route, use `pacemaker input.yaml`
with the same input.

Source-offset TensorPotential fitting supports BFGS, L-BFGS-B and CG.
Persistent-worker fitting currently requires graph mode and **no test split**
(`test_size`/`test_filename`). Ordinary `pacemaker` supports a test dataset:
only training offsets are applied, and test-only source labels are rejected.
Do not fit offsets on validation labels to make their errors smaller.

The CPU implementation uses an extra **energy-only** prediction pass,
reduces results in dataset order, fits offsets globally, and sends adjusted
targets back to the same workers. It never fits separate offsets per worker
or batch. Timing includes prediction, offset preparation and loss/gradient
work; historical gradient-only timings are not directly comparable.

## Offset definition and sign

For structure `i` with `N_i` atoms and source `s`, the corrected prediction is:

```text
E_pred(i) = E_ACE(i) + N_i * offset(s)

offset(s) = weighted_mean_s[(E_target(i) - E_ACE(i)) / N_i]
offset(reference) = 0
```

`E_target` here is `energy_corrected`, after the common reference subtraction.
The mean uses the structure energy weights directly, not an extra atom-count
weight. A group whose labels lie 100 eV/atom above the shared model therefore
gets an offset near **+100 eV/atom**. Zero-total-weight groups do not constrain
the energy loss; their reported offset uses an unweighted mean convention.

At each fitted non-reference offset, the weighted mean per-atom residual is
zero. The usual coefficient gradient at these adjusted targets is therefore
the exact gradient of the profiled objective. Accepted-step offsets are
restored after rejected line-search trials.

## Outputs and use of the fitted potential

`log.txt` records the fitted and best-fit offsets. `train_pred.pckl.gzip`
retains the ordinary data/provenance columns and:

| Column | Meaning | Units |
| --- | --- | --- |
| `energy_pred_raw` | Shared ACE prediction before the source correction | eV/structure |
| `energy_source_offset_fit` | `N_i * offset(s)` | eV/structure |
| `energy_source_offset_fit_per_atom` | `offset(s)` | eV/atom |
| `energy_pred` | Raw prediction plus the fitted correction | eV/structure |
| `forces_pred` | Unchanged force prediction | eV/angstrom |

For example, extract and retain the final offsets alongside the model:

```python
import numpy as np
import pandas as pd

prediction = pd.read_pickle("train_pred.pckl.gzip", compression="gzip")
np.testing.assert_allclose(
    prediction.energy_pred,
    prediction.energy_pred_raw + prediction.energy_source_offset_fit,
)
offsets = prediction.groupby("source_id")["energy_source_offset_fit_per_atom"].first()
offsets.rename("offset_eV_per_atom").to_csv("source_offsets.csv")
```

`metrics.txt` reports source-corrected energy errors; these are not raw
cross-code energy errors. Full offset vectors are in logs, prediction
columns and in-memory metric snapshots, not columns of `metrics.txt`.

`output_potential.yaml` contains the **shared ACE model**, not a calculator
that chooses a source offset. ASE/LAMMPS evaluations use that shared model
and do not need source labels. Keep the input, label mapping, common reference
and offset table for provenance. When comparing relative phase energies or
EOS curves, use one common energy alignment, not a separate offset per phase.

## Restarting and known limits

For the pinned legacy TensorPotential, restart from the **pre-core**
`interim_potential_best_cycle.yaml`, not a final model decorated with
post-fit automatic ZBL. The legacy fitting and exported-ZBL representations
are not generally interchangeable. To request another optimizer run, make
a new checkpoint without completed-cycle bookkeeping:

```python
from pyace.basis import BBasisConfiguration

basis = BBasisConfiguration("previous/interim_potential_best_cycle.yaml")
for key in ("_fit_cycles", "_loss"):
    if key in basis.metadata:
        del basis.metadata[key]
basis.save("restart_pre_core.yaml")
```

Use `potential: restart_pre_core.yaml` in a **new** run directory, preserving
the dataset, reference and labels. The offsets are recomputed during fitting;
the optimizer's inverse Hessian is not restored. Removing bookkeeping does
not convert a ZBL-decorated final model into a compatible pre-core model.
Do not use `pacemaker --clean` to manage old runs you want to retain.

Reaching `maxiter` is not proof of convergence. Small corrected training
errors and well-determined offsets do not establish transferable phase
energies, repulsive behavior or a physically stable EOS. A constant offset
cannot repair mixed XC methods, charge states, inconsistent numerical errors
or bad force/energy pairing within a group.

## Synthetic verification

These checks generate synthetic data or use repository test fixtures, not
archive uploads. `pytest` is an optional development tool, not included in
the runtime environment recipe; install it with `python -m pip install
pytest` if needed. Run from the checkout with the installed environment:

```bash
python -m pytest -q --runtensorpot \
  tests/test_source_energy_offsets.py tests/test_tensorpotential_energy.py
python tests/cpu/smoke.py --directory /tmp/pacemaker-offset-smoke --source-offsets
```

The smoke-test destination must not already exist. It compares ordinary and
CPU two-step fits with unequal atom counts, nonuniform weights and three
source labels, including recovery of a known relative 137.5 eV/atom shift.
Unit tests cover finite-difference gradients, missing labels, live coefficient
updates, held-out behavior, accepted offsets, auxiliary smoothness graphs and
complete objective timing. These establish implementation correctness in the
tested cases, not physical validity of a potential trained on real data.
