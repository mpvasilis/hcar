# Appendix B — Benchmark Problem Details

This appendix describes the five benchmark problems used in the HCAR evaluation.
For each benchmark we give a short description, the decision variables, and the
ground-truth target model $\mathcal{C}_T$.

## Sudoku (9×9)

- **Description.** A standard 9×9 Sudoku: fill the grid so that each row,
  column, and 3×3 subgrid contains all digits 1–9.
- **Variables.** A 9×9 integer matrix `grid[i,j]`, each in `{1,…,9}`;
  81 variables total.
- **Target constraints ($|\mathcal{C}_T| = 27$).** 9 row, 9 column, and 9 block
  `AllDifferent` constraints.

## UEFA Champions League Scheduling

- **Description.** Assign 32 teams to 8 groups under UEFA rules; teams have
  associated countries and pot (coefficient) groupings.
- **Variables.** An integer `group_assignment[t]` for each of the 32 teams,
  giving its group (1–8).
- **Target constraints ($|\mathcal{C}_T| = 19$).** 8 `Count` (4 teams per group),
  4 `AllDifferent` (one per pot of 8 teams), and 7 `AllDifferent` (teams from the
  same country).

## Cloud VM Allocation

- **Description.** Assign 40 VMs to 10 PMs, respecting resource capacities
  (CPU, memory, disk) and high-availability rules.
- **Variables.** An integer `vm_assignment[v]` for each of the 40 VMs, giving its
  PM (1–10).
- **Target constraints ($|\mathcal{C}_T| = 72$).** 30 `Sum` constraints (three per
  PM, capping the total CPU/memory/disk of co-located VMs); 18 `Count` constraints
  enforcing maximum capacity and minimum utilization on the GPU-equipped PMs; and
  24 `AllDifferent` constraints ensuring that critical VMs in the same
  high-availability group are placed on distinct PMs.

## University Exam Timetabling

- **Description.** Schedule 54 exams (9 semesters × 6 courses) into 126 timeslots
  (9 per day over 14 days), avoiding conflicts and respecting daily capacity.
- **Variables.** A 2D integer matrix `exam_slots[s,c]` giving each exam's
  timeslot; 54 variables total.
- **Target constraints ($|\mathcal{C}_T| = 24$).** 1 global `AllDifferent` over all
  54 exams (no two in the same slot), 9 constraints placing each semester's exams
  on different days, and 14 `Count` constraints capping exams per day.

## Nurse Rostering

- **Description.** Build a weekly schedule for 8 nurses across 3 daily shifts,
  respecting work regulations.
- **Variables.** A 3D integer matrix `roster[d,s,p]`; 42 variables total.
- **Target constraints ($|\mathcal{C}_T| = 21$).** 7 `AllDifferent` (a nurse works
  at most one shift per day), 6 `AllDifferent` (rest period between days), and
  8 `Count` (max weekly workdays per nurse).
