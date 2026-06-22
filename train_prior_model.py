"""Train the XGBoost constraint-prior model used by Phase 2 of HCAR.

This regenerates the two artifacts ``main.py`` loads to initialise per-constraint
confidence priors:

    constraint_classifier_xgb.joblib   - the trained XGBoost classifier
    xgb_feature_columns.txt            - the (post one-hot) feature column order

Training data is built from the CSPLib problems in ``oracles.py`` (the paper's
training set). For each problem the true global constraints are positive
examples; spurious / over-scoped variants are generated as negatives. Features
are extracted with the SAME ``feature_extraction.extract_constraint_features``
that ``main.py`` uses at inference time, and one-hot encoded identically, so the
saved model and column order plug directly into the pipeline.

Note: the original experiment-machine model was not preserved in version
control. This script reconstructs an equivalent prior; the resulting
probabilities approximate, but will not exactly equal, the paper's. HCAR is
robust to prior noise (the Bayesian update corrects wrong priors via oracle
evidence), so this primarily affects query efficiency, not final accuracy.

Usage:
    python train_prior_model.py
"""
import random
import warnings

import cpmpy as cp
import joblib
import pandas as pd
from cpmpy.transformations.get_variables import get_variables
from cpmpy.expressions.globalconstraints import AllDifferent
from xgboost import XGBClassifier

from feature_extraction import extract_constraint_features
from oracles import CSP_LIBS, flatten_vars

warnings.filterwarnings("ignore")
random.seed(42)

MODEL_PATH = "constraint_classifier_xgb.joblib"
FEATURE_COLUMNS_PATH = "xgb_feature_columns.txt"

# The paper's training problems (CSPLib ids present in oracles.CSP_LIBS).
TRAINING_PROBLEMS = [
    "prob001_car_sequencing",
    "prob003_quasigroup_existence",
    "prob007_all_interval_series",
    "prob012_nonogram",
    "prob014_solitaire_battleships",
    "prob019_magic_square",
    "prob022_bus_driver_scheduling",
    "prob030_bacp",
    "prob038_steel_mill_slab_design",
    "prob045_covering_array",
    "prob049_number_partitioning",
    "prob054_nqueens",
    "prob067_quasigroup_completion",
    "prob076_costas_arrays",
]

GLOBAL_TYPES = {"AllDifferent", "Sum", "Count"}


def _relation(constraint):
    """Return the constraint's relation label as used by extract_constraint_features."""
    return extract_constraint_features(constraint, [])["relation"]


def _generate_negatives(true_constraints, all_vars):
    """Build spurious / over-scoped constraints as negative training examples.

    For each true constraint we synthesise one same-type spurious sibling (an
    over-scoped scope and/or a perturbed bound) that is unlikely to hold, giving
    a roughly class-balanced dataset across AllDifferent/Sum/Count.
    """
    negatives = []
    named_vars = [v for v in all_vars if hasattr(v, "name")]
    if len(named_vars) < 3:
        return negatives

    for c in true_constraints:
        rel = _relation(c)
        scope = [v for v in get_variables(c) if hasattr(v, "name")]
        if not scope:
            continue
        extras = [v for v in named_vars if v not in scope]

        if rel == "AllDifferent":
            if extras:                                    # over-scoped variant
                k = min(len(extras), random.randint(1, 3))
                negatives.append(AllDifferent(scope + random.sample(extras, k)))
            else:                                         # random-scope variant
                size = max(2, min(len(named_vars), len(scope)))
                negatives.append(AllDifferent(random.sample(named_vars, size)))
        elif rel == "Sum":                                # over-scoped + wrong bound
            sc = scope + (random.sample(extras, min(len(extras), 2)) if extras else [])
            negatives.append(cp.sum(sc) == random.randint(1, 3 * max(2, len(sc))))
        elif rel == "Count":                              # spurious count target
            size = max(2, min(len(named_vars), len(scope)))
            negatives.append(cp.Count(random.sample(named_vars, size), 1)
                             == random.randint(1, size))

    return negatives


def build_dataset():
    """Collect (feature_dict, label) rows across the training problems."""
    rows, labels = [], []
    used = []

    for name in TRAINING_PROBLEMS:
        constructor = CSP_LIBS.get(name)
        if constructor is None:
            continue
        try:
            *var_parts, model = constructor()
        except Exception as exc:  # constructor may be environment-sensitive
            print(f"  skip {name}: {exc}")
            continue

        all_vars = flatten_vars(var_parts)
        constraints = list(model.constraints)

        positives = [c for c in constraints if _relation(c) in GLOBAL_TYPES]
        if not positives:
            continue
        negatives = _generate_negatives(positives, all_vars)

        for c in positives:
            rows.append(extract_constraint_features(c, all_vars))
            labels.append(1)
        for c in negatives:
            rows.append(extract_constraint_features(c, all_vars))
            labels.append(0)

        used.append(f"{name} (+{len(positives)}/-{len(negatives)})")

    print("Training problems used:")
    for u in used:
        print(f"  {u}")
    return rows, labels


def main():
    rows, labels = build_dataset()
    if not rows:
        raise SystemExit("No training data could be generated.")

    feat_df = pd.DataFrame(rows)
    # One-hot encode exactly as main.py does at inference time.
    cat_cols = feat_df.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        feat_df = pd.get_dummies(feat_df, columns=cat_cols)
    feature_cols = list(feat_df.columns)

    X = feat_df.values
    y = labels
    print(f"\nDataset: {len(y)} constraints "
          f"({sum(y)} valid / {len(y) - sum(y)} spurious), {len(feature_cols)} features")

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        eval_metric="logloss",
        random_state=42,
    )
    clf.fit(X, y)
    train_acc = clf.score(X, y)
    print(f"Training accuracy: {train_acc:.3f}")

    joblib.dump(clf, MODEL_PATH)
    with open(FEATURE_COLUMNS_PATH, "w") as f:
        for col in feature_cols:
            f.write(f"{col}\n")

    print(f"\nSaved model      -> {MODEL_PATH}")
    print(f"Saved feature cols -> {FEATURE_COLUMNS_PATH}")


if __name__ == "__main__":
    main()
