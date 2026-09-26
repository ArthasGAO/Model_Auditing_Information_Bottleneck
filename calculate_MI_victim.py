"""
Victim-model MI on the negative-pool-0.0 grid.

Why this script exists
----------------------
The seed-42 / rate-1.0 victims only have MI rows in
saved_logs/vanilla/MI_master_table.csv, computed 2026-05-08 on the OLD grid
in_size {1000, 5000, 10000, 15000, 20000, 25000}. The rate-0.0 negative pool
(MI_master_table_neg_pool0.csv) and the FT positives (ft_final/MI_master_table_ft.csv)
use the NEW grid POOL0_IN_SIZE_RATES x group_size = {250, 1250, 2500, 5000,
12500, 18750, 25000} x POOL0_BINS. Only 5000 and 25000 overlap, so a victim
marker cannot be placed at the other sizes. This script fills the full new grid
for every victim and stores it in a SEPARATE table (VICTIM_MASTER_CSV); the two
existing tables are never written.

How it computes
---------------
It calls MI_check.main_nega_pool0 with rate=1.0 and the victim model directory,
so the checkpoint (best_epoch.pth), the in-sample group (group_A seed 42), the
nested subsets (Indices/<dataset>/nested_subsets_seed42.npz, subset_seed 42),
the bins and the MI estimator are exactly those of the pool and of the FT rows.
Schema is identical to MI_master_table.csv (epoch column = 99 placeholder).
Models already complete in VICTIM_MASTER_CSV are skipped (missing_mi_grid
resume protocol), so re-running is safe; partial rows raise.

Victims (2026-09-20 on disk: all six)
-------------------------------------
CNN  : saved_models/vanilla/CNN_Models/<Scenario>_42_1.0            (family="cnn")
DeiT : saved_models/vanilla/Transformer_Models/<Scenario>_42_1.0    (family="deit")
CIFAR-100_DeiT_Distill_25000_42_1.0 arrived 2026-09-20 (trained on the second
machine) and is now active in VICTIMS; before that its line was commented out
because Transformer_Models held only the rate-0.0 Distill pool. Its plan lives
in train_plan/ rather than train_plan/old_plan/ like the other five. The
distilled student has head + head_dist, but MI_check switches Distillation off
in a copy of the plan (so no teacher is built -- the plan's teacher_ckpt is a
{seed} template that does not resolve here) and evaluates in eval mode, where
the two heads are returned averaged as one [B, 100] tensor. Nothing else in
this script is family- or dataset-specific.

Usage (from E:\Experiment, pytorch_env)
--------------------------------------
    python calculate_MI_victim.py                 # all victims in VICTIMS
    python calculate_MI_victim.py CIFAR10_RES18   # only yamls whose basename
                                                  # contains one of the args
"""
import sys

import torch

from MI_check import (
    main_nega_pool0, set_seed,
    POOL0_BINS, POOL0_IN_SIZE_RATES,
)

# ---- configuration -------------------------------------------------------
VICTIM_MASTER_CSV  = "./saved_logs/vanilla/MI_master_table_victim.csv"
VICTIM_VERBOSE_DIR = "./saved_logs/vanilla/MI_verbose_victim"
VICTIM_SEED        = 42
VICTIM_RATE        = 1.0
VICTIM_SUBSET_SEED = 42                     # same nested subsets as pool / FT
VICTIM_CNN_DIR     = "./saved_models/vanilla/CNN_Models"
VICTIM_DEIT_DIR    = "./saved_models/vanilla/Transformer_Models"

# (training-plan yaml, family). The yaml's Scenario_Name names the folder
# <Scenario_Name>_<VICTIM_SEED>_<VICTIM_RATE> under the family directory.
VICTIMS = [
    ("./saved_exp_plan/train_plan/old_plan/CIFAR10_RES18_SGD_SMALL.yaml",  "cnn"),   # CIFAR-10_ResNet-18_25000_42_1.0
    ("./saved_exp_plan/train_plan/old_plan/CIFAR10_VGG16_SGD_SMALL.yaml",  "cnn"),   # CIFAR-10_VGG16_25000_42_1.0
    ("./saved_exp_plan/train_plan/old_plan/CIFAR10_DeiT_Plain_SMALL.yaml", "deit"),  # CIFAR-10_DeiT_Plain_25000_42_1.0
    ("./saved_exp_plan/train_plan/old_plan/CIFAR100_RES18_SGD_SMALL.yaml", "cnn"),   # CIFAR-100_ResNet-18_25000_42_1.0
    ("./saved_exp_plan/train_plan/old_plan/CIFAR100_VGG16_SGD_SMALL.yaml", "cnn"),   # CIFAR-100_VGG16_25000_42_1.0
    ("./saved_exp_plan/train_plan/CIFAR100_DeiT_Distill_SMALL.yaml",       "deit"),  # CIFAR-100_DeiT_Distill_25000_42_1.0
]


def run_victim_mi(victims=VICTIMS, master_csv_path=VICTIM_MASTER_CSV,
                  verbose_dir=VICTIM_VERBOSE_DIR, record_verbose=False):
    """
    Compute the full (POOL0_IN_SIZE_RATES x POOL0_BINS) MI grid for each
    (yaml, family) in `victims` and append the rows to `master_csv_path`.
    Returns {yaml_path: True if rows were written, False if skipped/missing}.
    """
    print(f"Victim MI sweep: {len(victims)} model(s)")
    print(f"  output : {master_csv_path}")
    print(f"  grid   : in_size_rates={list(POOL0_IN_SIZE_RATES)}  bins={list(POOL0_BINS)}")
    print(f"  seed={VICTIM_SEED} rate={VICTIM_RATE} subset_seed={VICTIM_SUBSET_SEED}")

    tally = {}
    for yaml_path, family in victims:
        model_dir = VICTIM_DEIT_DIR if family == "deit" else VICTIM_CNN_DIR
        set_seed(VICTIM_SEED)
        wrote = main_nega_pool0(
            VICTIM_SEED, yaml_path,
            in_sizes=None,                          # -> sizes_from_rates(group_size, POOL0_IN_SIZE_RATES)
            num_intervals_list=list(POOL0_BINS),
            record_verbose=record_verbose,
            model_dir=model_dir,
            master_csv_path=master_csv_path,
            verbose_dir=verbose_dir,
            subset_seed=VICTIM_SUBSET_SEED,
            rate=VICTIM_RATE,
            skip_existing=True,
            family=family,
        )
        tally[yaml_path] = wrote

    print("\nVictim MI sweep finished:")
    for yaml_path, wrote in tally.items():
        print(f"  {'computed' if wrote else 'skipped/missing':<16} {yaml_path}")
    print(f"  -> {master_csv_path}")
    return tally


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    selectors = sys.argv[1:]
    chosen = [
        (y, f) for y, f in VICTIMS
        if not selectors or any(s in y.rsplit("/", 1)[-1] for s in selectors)
    ]
    if not chosen:
        raise SystemExit(f"No victim yaml matches {selectors}; available: "
                         f"{[y.rsplit('/', 1)[-1] for y, _ in VICTIMS]}")
    run_victim_mi(chosen)
