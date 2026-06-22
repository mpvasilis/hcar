# Original experiment artifacts (archived for provenance)

These are the **genuine** artifacts recovered from the original project repository,
not the reconstructions used as the repo defaults.

**Source:** `github.com/mpvasilis/violationLearnedGlobalCstrs` @ branch
`v1.1-UpdatedMLFeatures` (the artifacts were committed on feature branches, not on
`master`, which is why they were not found by earlier default-branch searches).

| File | What it is |
|------|------------|
| `constraint_classifier_xgb.joblib` | The original trained XGBoost prior model (30 features) |
| `xgb_feature_columns.txt` | Its feature-column order (30 columns — consistent with the model) |
| `feature_extraction.py` | The feature extractor that produced the model's 30 features |
| `train_xgboost.py` | The original training script |
| `vm_allocation_model.py` | The original VM/PM data (9 PMs / 19 VMs, with GPU) |

## Why these are archived, not used as defaults

They are **not compatible** with the cleaned, paper-faithful code in this repo:

1. **Feature-schema mismatch.** This model expects the 30 features produced by the
   `feature_extraction.py` in *this folder*. The repo's top-level
   `feature_extraction.py` produces 42 one-hot columns matching the paper's
   **Table 1** feature set. `main.py` would therefore not load this model
   correctly (column mismatch).

2. **VM data infeasibility.** The original `vm_allocation_model.py` (9 PMs /
   19 VMs) is satisfiable for the global encoding but **UNSAT** for this repo's
   binary encoding (`benchmarks/vm_allocation.py`): every VM has `memory > 8`,
   which that encoding forces to be pairwise on distinct PMs — needing 19 PMs but
   only 9 are available. It belongs to an earlier benchmark-code variant.

Other branches also carry models but with internally inconsistent metadata
(`v1.0-FinalMethodology-UpdatedML`: 51-feature model vs 19-column file;
`v1.2`: 30-feature model vs 613-column file), so `v1.1-UpdatedMLFeatures` was
chosen here as the one self-consistent set.

## Repo defaults (what `main.py` actually uses)

The repository root ships **reconstructed** artifacts that are tuned to and
verified against the cleaned paper code:

- `constraint_classifier_xgb.joblib` + `xgb_feature_columns.txt` — retrained from
  `oracles.py` via `train_prior_model.py`, matching the paper's Table-1 features.
- `vm_allocation_model.py` — a paper-spec instance (10 PMs / 40 VMs, GPU PMs, HA
  zones) verified satisfiable for both the binary and global encodings.

## Using the originals instead

To run with the original model: copy this folder's `feature_extraction.py`,
`constraint_classifier_xgb.joblib`, and `xgb_feature_columns.txt` to the repo root
(overwriting the defaults). To use the original VM data you must also adopt the
matching benchmark encoding from the source branch, otherwise the binary instance
is UNSAT.
