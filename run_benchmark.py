"""Reproduction runner: applies a solver-compat shim, then runs main.py.

pycona's query generation passes solution hints that may be ``None`` for
variables it leaves unassigned; recent ortools' ``AddHint`` rejects ``None``.
Solution hints are non-binding search heuristics, so we simply drop the ``None``
entries. This does not change any solver result or learned model.

Usage:
    python run_benchmark.py --experiment sudoku        --use_bayesian
    python run_benchmark.py --experiment uefa          --use_bayesian
    python run_benchmark.py --experiment nurse         --use_bayesian
    python run_benchmark.py --experiment examtt        --use_bayesian
    python run_benchmark.py --experiment vm_allocation --use_bayesian

Results (including solution-space precision/recall) are appended to results.csv.
"""
import sys
import warnings

warnings.filterwarnings("ignore")

from cpmpy.solvers.ortools import CPM_ortools
from cpmpy.expressions.utils import flatlist

_orig_solution_hint = CPM_ortools.solution_hint


def _solution_hint(self, cpm_vars, vals):
    cpm_vars = flatlist(cpm_vars)
    vals = flatlist(vals)
    self.ort_model.ClearHints()
    for v, x in zip(cpm_vars, vals):
        if x is not None:
            self.ort_model.AddHint(self.solver_var(v), x)
    return self


CPM_ortools.solution_hint = _solution_hint

import runpy

sys.argv = ["main.py"] + sys.argv[1:]
runpy.run_path("main.py", run_name="__main__")
