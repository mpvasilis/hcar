# Reproducing the HCAR results

This guide reproduces the paper's **solution-space accuracy** (100% precision /
100% recall) on all five benchmarks.

## Environment

Tested on **Python 3.11** with the exact versions in `requirements.txt`:

| Package | Version |
|---|---|
| cpmpy | 0.10.1 |
| pycona | 0.4.1 |
| ortools | 9.14.6206 |
| xgboost | 3.2.0 |
| scikit-learn | 1.9.0 |

```bash
# from the repository root
py -3.11 -m venv .venv            # Windows;  python3.11 -m venv .venv on Linux/macOS
.venv/Scripts/python -m pip install -U pip
.venv/Scripts/python -m pip install -r requirements.txt
```

(On Linux/macOS use `.venv/bin/python`.)

## Run

Use `run_benchmark.py` (a thin wrapper around `main.py` that applies a
non-functional ortools `solution_hint` compatibility shim — see the file header):

```bash
.venv/Scripts/python run_benchmark.py --experiment sudoku        --use_bayesian
.venv/Scripts/python run_benchmark.py --experiment uefa          --use_bayesian
.venv/Scripts/python run_benchmark.py --experiment nurse         --use_bayesian
.venv/Scripts/python run_benchmark.py --experiment examtt        --use_bayesian
.venv/Scripts/python run_benchmark.py --experiment vm_allocation --use_bayesian
```

Each run prints `Solution-space Precision: ...%  Recall: ...%` and appends a row
to `results.csv`.

## Expected results

All five benchmarks reach **100% / 100%** solution-space precision/recall
(confirmed over repeated runs):

| Benchmark | S-Prec. | S-Rec. |
|---|:---:|:---:|
| Sudoku (9×9) | 100% | 100% |
| UEFA Scheduling | 100% | 100% |
| Nurse Rostering | 100% | 100% |
| Exam Timetabling | 100% | 100% |
| VM Allocation | 100% | 100% |


- **Determinism:** Phase 2 uses randomness; accuracy is stable across runs, but
  exact query counts vary slightly run to run.
- **Solver versions:** `ortools` is pinned `< 9.15` because newer releases
  changed the `CpSolver.Solve()` / `AddHint` APIs that the pinned `cpmpy` uses.
