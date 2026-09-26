"""Hypothesis test for the post-hoc AT trajectories at their final epoch.

    python run_hypothesis_test_at_traj.py

Nothing statistical is re-derived here. Like run_hypothesis_test_knockoff2,
this file only supplies a POSITIVE_FAMILIES configuration to
`run_hypothesis_test_positives` and calls its `main()`, so every checkpoint is
judged against exactly the reference sets the negative run used: the same
frozen manifest (80-model ResNet-18 pool, 50 rounds, k in {5,...,30}), the
same mu / Sigma estimator, the same chi2 and exact-F laws.

What is scored
--------------
The 12 mix=off trajectories of main_at.py (run tag traj2): four sources
{FT-AL, Prune20-FT-AL, DKD, Knockoff Same10 -> RN18} x three eps
{2, 4, 8}/255, each at its last checkpoint epoch_29. Every suspect is a
ResNet-18 derived from the ResNet-18 victim, so H0 is the CIFAR-10 ResNet-18
pool, the same H0 the trajectory plot draws.

Selection is plan-based (saved_exp_plan/hypo_at_traj/, one Scenario_Name per
source) and then narrowed by `filters` on the MI table's own columns:
run_tag = traj2 keeps out the earlier single-checkpoint `traj` run of the same
models, ckpt_kind picks the epoch. `group_by` reports every
(Scenario, eps, ckpt_kind) as its own group, so the summary has one line per
trajectory. To score more of a trajectory, list more epochs in CKPT_KINDS and
bump TAG (main() refuses to overwrite a finished run).

Reading the result
------------------
chi2 underflows to exactly 0 on a strong suspect; the exact F law keeps
resolution, so read `p_F`. Per group and k the summary gives the distribution
of p_F over the 50 reference splits (min / p05 / median / p95 / max) and the
rejection rate at each alpha; `In_rate1_bins50/per_model.csv` holds every
(model, round, k) value.
"""
from pathlib import Path

import run_hypothesis_test_positives as H

BASE = Path(__file__).resolve().parent

CKPT_KINDS = ["epoch_29"]          # e.g. ["epoch_0", "epoch_14", "epoch_29"]
RUN_TAGS = ["traj2"]                # the every-epoch run; "traj" was the last-epoch-only pilot

H.POSITIVE_FAMILIES = [
    {"label": "at_traj",
     "plan_dir": BASE / "saved_exp_plan/hypo_at_traj",
     "mi_csv": BASE / "saved_logs/at_evasion/MI_master_table_at_traj.csv",
     "h0_scenario": "CIFAR-10_ResNet-18_25000",
     "filters": {"run_tag": RUN_TAGS, "ckpt_kind": CKPT_KINDS},
     "group_by": ("Scenario", "eps", "ckpt_kind")},
]

# One configuration: the operating point every d and every trajectory plot
# already uses. Widen IN_SIZE_RATES / BINS to sweep the grid; each combination
# writes its own result directory.
H.IN_SIZE_RATES = [1.00]
H.BINS = [50]
H.MI_KIND = "In"
H.ALPHAS = [0.05, 0.01]
H.RUN_MODE = "evaluate"
TAG = "ep29_v1"
H.OUTPUT_DIR = BASE / f"saved_logs/vanilla/Hypo_Test_AT_Traj_{TAG}"

if __name__ == "__main__":
    H.main()
