# Appendix C — Bayesian Parameter Validation and Sensitivity Analysis

This appendix details the validation process for the Bayesian noise parameter
$\alpha$ and the decision thresholds $\theta_{min}, \theta_{max}$ used by HCAR.

## Validation methodology

To ensure that the selected parameters generalize across diverse domains, we
performed a grid search on five CSPLib problems that are **distinct** from the
primary evaluation benchmarks. The validation set consisted of:

- **Magic Square** (prob019)
- **Car Sequencing** (prob001)
- **N-Queens** (prob054)
- **Balanced Academic Curriculum Problem (BACP)** (prob030)
- **Steel Mill Slab Design** (prob038)

These problems span a wide range of constraint types (`AllDifferent`, `Sum`,
`Count`) and structures (scheduling, assignment, packing). Each validation
instance was initialized with 5 positive examples, mirroring the experimental
setup of the main evaluation.

## Noise parameter selection ($\alpha$)

We evaluated $\alpha \in [0.1, 0.7]$. For each value we tracked the average number
of queries to reach a decision, the false acceptance rate (constraints
erroneously added to $C'_G$), and the solution-space recall. After the grid
search we ran an additional random search over $[0.4, 0.5]$.

| $\alpha$ | Avg. Queries / Constraint | False Accept. Rate (%) | Query Eff. (Q / \|B_globals\|) | S-Rec. (Avg. %) |
|:---:|:---:|:---:|:---:|:---:|
| 0.1  | 12.4 | 0.0 | 8.2 | 100 |
| 0.2  | 9.8  | 0.0 | 6.7 | 100 |
| 0.3  | 7.2  | 0.0 | 5.1 | 100 |
| 0.4  | 5.8  | 0.0 | 4.3 | 100 |
| **0.42** | **5.5** | **0.0** | **4.1** | **100** |
| 0.45 | 5.6  | 0.0 | 4.2 | 100 |
| 0.5  | 5.9  | 0.8 | 4.4 | 98  |
| 0.6  | 6.8  | 2.1 | 5.0 | 95  |
| 0.7  | 8.3  | 4.5 | 6.2 | 91  |

The value $\alpha = 0.42$ was identified as the optimal setting, minimizing the
number of queries (4.1 queries per constraint) while maintaining zero false
acceptances and 100% recall. Lower values (0.1–0.3) proved overly conservative,
requiring up to three times as many queries to build sufficient confidence.
Higher values (0.5–0.7) introduced instability: from $\alpha = 0.5$ the system
began accepting spurious constraints (0.8% false acceptance rate), and at
$\alpha = 0.7$ recall dropped to 91%.

## Decision threshold selection

We validated the acceptance ($\theta_{max}$) and rejection ($\theta_{min}$)
thresholds with $\alpha$ fixed at 0.42.

| $\theta_{min}$ | $\theta_{max}$ | Avg. $Q_2$ | S-Prec. (%) | S-Rec. (%) | Undecided Constr. (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.05 | 0.95 | 287 | 100 | 100 | 2.1 |
| **0.1** | **0.9** | **215** | **100** | **100** | **0.8** |
| 0.15 | 0.85 | 168 | 100 | 96 | 1.2 |
| 0.2  | 0.8  | 142 | 98  | 93 | 3.5 |

The selected thresholds $\theta_{min} = 0.1$ and $\theta_{max} = 0.9$ represent
the most effective compromise. Tighter thresholds (0.05/0.95) increased query
consumption by roughly 33% and left more undecided constraints (2.1%) without
improving accuracy. Looser thresholds (0.2/0.8) reduced query cost but dropped
recall to 93% and precision to 98%. The rationale: a high acceptance bar (0.9) is
crucial because accepting an invalid constraint corrupts the model, whereas a
lower rejection bar (0.1) is acceptable because valid constraints rejected by
mistake can often be recovered via the subset exploration mechanism.

## Sensitivity analysis across the test benchmarks

We repeated the analysis on the five **main** test benchmarks (Sudoku, UEFA
Scheduling, Exam Timetabling, VM Allocation, Nurse Rostering), varying $\alpha$
with the thresholds fixed.

- Solution-space recall remains at **100% for $\alpha \le 0.42$** across all five
  test benchmarks, confirming that the tuning on the validation set generalizes
  to unseen domains.
- For $\alpha \ge 0.5$ recall degrades, most sharply on the complex problems
  (VM Allocation and Nurse Rostering).
- Query cost decreases as $\alpha$ grows, but beyond $\alpha = 0.42$ this saving
  comes at the expense of accuracy.
- Conservative values ($\alpha = 0.3$) maintain accuracy but increase query cost
  by roughly 29%.

The selected value $\alpha = 0.42$ is the optimal balance point: full recall is
maintained with near-minimal query overhead.
