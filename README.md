# Augmented Weakly-Hard Scheduling for Mixed-Criticality Multicore Systems

Implementation of an augmented weakly-hard importance-aware scheduling framework
for fully-partitioned mixed-criticality systems.

## Overview

LO-criticality tasks are assigned baseline weakly-hard constraints $(m_i, k_i)$.
Available slack is exploited by augmenting the number of mandatory jobs to
$(m_i + x_i, k_i)$. The augmented level $x_i$ varies across three modes:

- **LO mode** ($x_i^{\mathrm{L}}$): promote according to augmented importance density.
- **Mode switch** ($x_i^{\mathrm{S}}$): degrade to preserve schedulability under HI-mode execution.
- **Stable HI mode** ($x_i^{\mathrm{H}}$): recover suspended tasks by sacrificing augmented levels from retained tasks; then re-augment retained tasks using remaining slack.

The importance achieved by each LO task is

$$ I_i(x_i) = \mu_i \left(1 + \beta \frac{x_i}{k_i - m_i}\right), \quad 0 < \beta < 1. $$

Total system performance is $I^{\chi} = \sum a_i^{\chi} I_i(x_i^{\chi})$, normalised by $I^{\max}$.

## Code Structure

```
core/                   Task, MKPattern, Processor, Job, importance model
scheduling/
  sched_test.py          RTA: LO-mode, mode-switch, stable-HI WCRT
  priority_assignment.py DMPO priority assignment
  task_partitioning.py   Baseline-first WFD partitioning + classification
  augmentation.py        LO-mode augmentation & mode-switch degradation
  recovery.py            Stable-HI recovery & post-augmentation
experiments/
  comparison/            Baseline methods (Static, AugOnly, MaxCount)
  vary_utilization/      H-mode performance vs. utilisation
  vary_cf/               H-mode performance vs. C_HI/C_LO ratio
  vary_mk/               H-mode performance vs. (m,k) constraint
  mode_performance/      Per-mode performance (L, S, H) + feasibility rate
  beta_sensitivity/      Performance sensitivity to beta
  suspend_recovery/      Suspend count & recovery success rate vs. utilisation
  performance.py         Shared performance metric functions
utils/
  generate_taskset.py    Task set generation via DRS
  drs.py                 Dirichlet Rescale Algorithm (Griffin et al., 2020)
  logger.py              Logging utility
```

## Dependencies

Python 3.10+, NumPy, SciPy, Matplotlib.

```bash
pip install numpy scipy matplotlib
```

## Running Experiments

```bash
# HI-mode performance vs utilisation
python -m experiments.vary_utilization.exp

# HI-mode performance vs C_HI/C_LO ratio
python -m experiments.vary_cf.exp

# HI-mode performance vs (m,k) constraint
python -m experiments.vary_mk.exp

# Per-mode (L, S, H) performance + Static baseline
python -m experiments.mode_performance.exp_vary_utilization

# Beta sensitivity
python -m experiments.beta_sensitivity.exp_vary_beta

# Suspend count & recovery rate
python -m experiments.suspend_recovery.exp_vary_utilization

# Quick pipeline test (1 run per point)
python -m experiments.test_pipeline
```

Append `--test` to any experiment for a single-run quick check.

Configuration (task count, utilisation range, beta, threads, runs) is at the top of each experiment file.

## Comparison Methods

| Method | Augmentation | Recovery | Description |
|---|---|---|---|
| Static-(m,k) | $x \equiv 0$ | None | Baseline lower bound |
| Augmented-Only | L + S | None | Post-augment in H only |
| MaxCount Recovery | L + S | Low-util-first | Maximises recovery count |
| **Proposed** | L + S + H | Importance-based | Maximises importance gain |

## License

© 2026 The Authors. All rights reserved.

This repository contains unpublished research work. Redistribution, modification,
or commercial use is not permitted without explicit permission.
