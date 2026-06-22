# HCAR — Hybrid Constraint Acquisition with Refinement

Reference implementation for the paper:

> **Overcoming Over-Fitting in Constraint Acquisition via Query-Driven Interactive Refinement**
> Vasileios Balafas, Dimos Tsouros, Nikolaos Ploskas, Kostas Stergiou.

HCAR learns constraint models (including the global constraints `AllDifferent`,
`Sum`, and `Count`) from a few positive examples, then removes the *over-fitted*
candidates that passive learning produces. It does so with a three-phase pipeline:

1. **Phase 1 — Passive learning.** Generate candidate global constraints
   (`B_globals`) and a fixed-arity bias (`B_fixed`) from a handful of solutions.
2. **Phase 2 — Query-driven Bayesian refinement.** Vet each global candidate
   against an oracle. Confidence scores start from a machine-learning prior
   (XGBoost) and are updated with Bayesian reasoning; rejected over-scoped
   constraints are recovered via *subset exploration*.
3. **Phase 3 — Active learning.** Complete the model with `MQuAcq-2` (from
   [PyConA](https://github.com/CPMpy/PyConA)) over the remaining fixed-arity bias.

## Repository structure

```
main.py                     Entry point: Phase 2 (Bayesian refinement) + Phase 3 (MQuAcq-2)
benchmark_runner.py         Runs main.py over all five benchmarks
phase1_passive_learning.py  Phase 1 passive learning (produces the passive pickles)

bayesian_ca_env.py          Bayesian active-CA environment (confidence bookkeeping)
bayesian_quacq.py           Bayesian QuAcq learner used in Phase 2
enhanced_bayesian_pqgen.py  Violation-driven query generation for Phase 2
feature_extraction.py       Structural features fed to the XGBoost prior model

vm_allocation_model.py      PM/VM data for the vm_allocation benchmark (10 PMs, 40 VMs)
oracles.py                  CSPLib model zoo (training set for the prior model)
train_prior_model.py        Regenerates the XGBoost prior model from oracles.py
constraint_classifier_xgb.joblib  Trained XGBoost prior model (Phase 2)
xgb_feature_columns.txt           Feature-column order for the prior model

benchmarks/                 Binary (fixed-arity) encodings of the 5 benchmarks
benchmarks_global/          Global-constraint encodings of the 5 benchmarks
```

## Requirements

Python **3.10–3.12** recommended, plus the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The solver backend is Google OR-Tools CP-SAT. `ortools` is pinned `<9.15`: newer
releases changed the `CpSolver.Solve()` API that `cpmpy 0.9.x` calls (and ship no
wheels for Python ≤3.12). The ML prior uses `xgboost` (loaded via `joblib`).

## Data and model artifacts

Everything needed to run the pipeline is included in the repository:

| File | Used for |
|------|----------|
| `vm_allocation_model.py` (`PM_DATA`, `VM_DATA`) | the `vm_allocation` benchmark |
| `constraint_classifier_xgb.joblib` + `xgb_feature_columns.txt` | ML confidence priors (Phase 2) |

> **Reproduction note.** The original experiment-machine `vm_allocation_model.py`
> and XGBoost model were not preserved in version control, so both are
> *reconstructed* here: the VM instance follows the paper's Appendix B spec
> (10 PMs / 40 VMs, GPU-equipped PMs, HA zones; verified satisfiable), and the
> prior model is retrained from the CSPLib problems in `oracles.py`. Reported
> numbers therefore approximate, but will not be bit-identical to, the paper's.
> HCAR is robust to prior noise (the Bayesian update corrects wrong priors), so
> this mainly affects query efficiency, not final accuracy. If the XGBoost
> artifacts are absent, priors fall back to a flat `0.5` (the "No ML Priors"
> ablation).

To retrain the prior model from scratch:

```bash
python train_prior_model.py     # writes constraint_classifier_xgb.joblib + xgb_feature_columns.txt
```

## Quick start

Run a single benchmark (Phase 2 + Phase 3):

```bash
python main.py --experiment sudoku        --use_bayesian
python main.py --experiment examtt        --use_bayesian
python main.py --experiment nurse         --use_bayesian
python main.py --experiment uefa          --use_bayesian
python main.py --experiment vm_allocation --use_bayesian
```

Or run all five through the runner:

```bash
python benchmark_runner.py
```

Per-benchmark results are appended to `results.csv`; the runner also writes logs
under `benchmark_logs/` and summaries under `benchmark_results/`.

## The five benchmarks

| `--experiment` | Problem | Global constraints |
|----------------|---------|--------------------|
| `sudoku`        | 9×9 Sudoku | `AllDifferent` |
| `uefa`          | UEFA Champions League group draw | `Count`, `AllDifferent` |
| `vm_allocation` | Cloud VM-to-PM allocation | `Sum`, `Count`, `AllDifferent` |
| `examtt`        | University exam timetabling | `AllDifferent`, `Count` |
| `nurse`         | Nurse rostering | `AllDifferent`, `Count` |

Each problem is defined twice: a binary encoding under `benchmarks/` and a
global-constraint encoding under `benchmarks_global/`.

## Hyperparameters

The Bayesian hyperparameters reported in the paper are set inside
`active_learning_system` in `main.py`:

- `alpha = 0.42` (Bayesian noise),
- `theta_max = 0.9` (acceptance threshold),
- `theta_min = 0.1` (rejection threshold).

These are the effective values used at run time. The `--alpha`, `--theta_max`,
`--theta_min`, and `--prior` command-line flags are parsed but do not override
the hardcoded paper configuration.

Other useful flags: `--timeout <seconds>` (default 600), `--phase2_only` (stop
after refinement), `--use_passive_constraints --passive_output_dir output`
(feed Phase 1 pickles into Phase 2).

## Phase 1 (optional)

To regenerate the passive-learning pickles consumed by
`--use_passive_constraints`:

```bash
python phase1_passive_learning.py --benchmark sudoku
```

Phase 1 supports `sudoku`, `examtt`, `nurse`, and `uefa`. Output pickles are
written to `output/` and loaded by `main.py` when `--use_passive_constraints`
is set.

## Paper appendices

- [Appendix B — Benchmark Problem Details](docs/appendix_b_benchmarks.md)
- [Appendix C — Bayesian Parameter Validation and Sensitivity Analysis](docs/appendix_c_parameter_validation.md)

## Citation

If you use this code, please cite the paper above.
