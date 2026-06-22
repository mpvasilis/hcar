"""Solution-space precision/recall evaluation (the paper's accuracy metric).

The paper reports solution-space **precision** (S-Prec.) and **recall** (S-Rec.):

    S-Prec = | sampled solutions of the LEARNED model that also satisfy the TARGET | / N
    S-Rec  = | sampled solutions of the TARGET model that also satisfy the LEARNED | / N

Solutions are sampled with a Hamming-distance ≥ `hamming` separation between
successive samples to spread them across the solution space (as in the paper).

`main.py` produces query/constraint counts but does NOT compute these metrics;
this module supplies that missing piece. Use `solution_space_metrics(...)` with a
learned constraint list and the target (ground-truth) constraint list over the
same variables.

Run `python evaluate.py` for a self-test that demonstrates the metric is correct
(identical models -> 100/100; a strictly looser model -> 100% recall, <100%
precision; a strictly tighter model -> <100% recall, 100% precision).
"""
import warnings

import cpmpy as cp
import numpy as np
from cpmpy.transformations.get_variables import get_variables

warnings.filterwarnings("ignore")


def _sample_solutions(constraints, variables, n_samples, hamming):
    """Sample up to n_samples Hamming-diverse solutions of `constraints`."""
    variables = list(variables)
    solutions = []
    while len(solutions) < n_samples:
        m = cp.Model(list(constraints))
        for prev in solutions:
            # next solution must differ from every previous one in >= hamming vars
            m += (cp.sum([variables[i] != int(prev[i]) for i in range(len(variables))]) >= hamming)
        if not m.solve():
            break
        solutions.append([int(v.value()) for v in variables])
    return solutions


def _satisfies(constraints, variables, assignment):
    """True iff `assignment` satisfies all `constraints`."""
    variables = list(variables)
    m = cp.Model(list(constraints))
    for i, v in enumerate(variables):
        m += (v == int(assignment[i]))
    return bool(m.solve())


def solution_space_metrics(learned, target, variables, n_samples=100, hamming=5):
    """Return (S-Prec, S-Rec) in [0,1] for a learned vs target constraint set.

    S-Prec: fraction of sampled LEARNED solutions that also satisfy TARGET.
    S-Rec:  fraction of sampled TARGET  solutions that also satisfy LEARNED.
    """
    variables = list(variables)

    learned_sols = _sample_solutions(learned, variables, n_samples, hamming)
    target_sols = _sample_solutions(target, variables, n_samples, hamming)

    if learned_sols:
        s_prec = sum(_satisfies(target, variables, s) for s in learned_sols) / len(learned_sols)
    else:
        s_prec = 0.0
    if target_sols:
        s_rec = sum(_satisfies(learned, variables, s) for s in target_sols) / len(target_sols)
    else:
        s_rec = 0.0

    return s_prec, s_rec


def _self_test():
    """Demonstrate the metric is correct on small constructed models."""
    x = cp.intvar(1, 4, shape=4, name="x")
    xs = list(x)

    target = [cp.AllDifferent(xs)]                       # all distinct
    looser = [xs[0] != xs[1]]                            # only one pair distinct
    tighter = [cp.AllDifferent(xs), xs[0] < xs[1]]       # all distinct AND ordered pair

    for name, learned in [("identical", target), ("looser", looser), ("tighter", tighter)]:
        p, r = solution_space_metrics(learned, target, xs, n_samples=30, hamming=2)
        print(f"{name:10s}: S-Prec={p*100:5.1f}%  S-Rec={r*100:5.1f}%")
    print("\nExpected: identical 100/100 ; looser ~<100 prec / 100 rec ; "
          "tighter 100 prec / <100 rec")


if __name__ == "__main__":
    _self_test()
