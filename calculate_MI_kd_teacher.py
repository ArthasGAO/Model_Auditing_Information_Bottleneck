"""MI for the KD teachers (the bigger same-family victims of the KD row).

In the extraction plots the star on the information plane is the victim. The
KD analogue is the TEACHER, and the shared victim table
(saved_logs/vanilla/MI_master_table_victim.csv) does not contain it: that table
holds exactly the six standard victims, and Gate 1 in
run_hypothesis_test_fixed_splits.py keys off that set. So the teachers get their
own table inside the kd_final tree rather than being appended there.

Nothing in calculate_MI_victim.py or MI_check.py is modified - this only calls
MI_check.main_nega_pool0 with a different model_dir / master_csv_path, the same
way calculate_MI_victim.py does.

Two things differ from calculate_MI_victim.py and are the reason this file
exists at all:
  * the teachers sit in saved_models/vanilla/ ITSELF, not in the CNN_Models
    subfolder that VICTIM_CNN_DIR points at (main_train_teacher.py writes
    ./saved_models/vanilla/<Scenario>_<seed>_<rate>/);
  * their plans live in train_plan/teacher_plan/.

Grid, seed, rate and subset seed match the victim / pool / FT / extraction /
KD tables exactly, so a teacher point is directly comparable to all of them.

Usage (from E:\\Experiment, pytorch_env)
    python calculate_MI_kd_teacher.py                # every teacher below
    python calculate_MI_kd_teacher.py CIFAR10_VGG19  # basename substring filter
"""
import os
import sys

import torch

from MI_check import main_nega_pool0, set_seed, POOL0_BINS, POOL0_IN_SIZE_RATES

# ---- configuration -------------------------------------------------------
TEACHER_MASTER_CSV  = "./saved_logs/kd_final/MI_master_table_teacher.csv"
TEACHER_VERBOSE_DIR = "./saved_logs/kd_final/MI_verbose_teacher"
TEACHER_MODEL_DIR   = "./saved_models/vanilla"   # NOT .../CNN_Models
TEACHER_SEED        = 42
TEACHER_RATE        = 1.0
TEACHER_SUBSET_SEED = 42                          # same nested subsets as the pool

# (training-plan yaml, family). Scenario_Name names the folder
# <Scenario_Name>_<TEACHER_SEED>_<TEACHER_RATE> under TEACHER_MODEL_DIR.
TEACHERS = [
    ("./saved_exp_plan/train_plan/teacher_plan/CIFAR10_RES34_SGD_SMALL.yaml", "cnn"),   # CIFAR-10_ResNet-34_25000_42_1.0
    ("./saved_exp_plan/train_plan/teacher_plan/CIFAR10_VGG19_SGD_SMALL.yaml", "cnn"),   # CIFAR-10_VGG19_25000_42_1.0
    # CIFAR-100 teachers, trained 2026-09-22 (RN34 73.07%, VGG19 67.60%).
    ("./saved_exp_plan/train_plan/teacher_plan/CIFAR100_RES34_SGD_SMALL.yaml", "cnn"),  # CIFAR-100_ResNet-34_25000_42_1.0
    ("./saved_exp_plan/train_plan/teacher_plan/CIFAR100_VGG19_SGD_SMALL.yaml", "cnn"),  # CIFAR-100_VGG19_25000_42_1.0
]


def run_teacher_mi(teachers=TEACHERS, master_csv_path=TEACHER_MASTER_CSV,
                   verbose_dir=TEACHER_VERBOSE_DIR, record_verbose=False):
    """Full (POOL0_IN_SIZE_RATES x POOL0_BINS) grid per teacher, appended once.

    main_nega_pool0(skip_existing=True) resumes, so re-running is free.
    """
    os.makedirs(os.path.dirname(master_csv_path) or ".", exist_ok=True)
    print(f"Teacher MI sweep: {len(teachers)} model(s)")
    print(f"  models : {TEACHER_MODEL_DIR}")
    print(f"  output : {master_csv_path}")
    print(f"  grid   : in_size_rates={list(POOL0_IN_SIZE_RATES)}  bins={list(POOL0_BINS)}")
    print(f"  seed={TEACHER_SEED} rate={TEACHER_RATE} subset_seed={TEACHER_SUBSET_SEED}")

    tally = {}
    for yaml_path, family in teachers:
        set_seed(TEACHER_SEED)
        tally[yaml_path] = main_nega_pool0(
            TEACHER_SEED, yaml_path,
            in_sizes=None,                    # -> sizes_from_rates(group_size, rates)
            num_intervals_list=list(POOL0_BINS),
            record_verbose=record_verbose,
            model_dir=TEACHER_MODEL_DIR,
            master_csv_path=master_csv_path,
            verbose_dir=verbose_dir,
            subset_seed=TEACHER_SUBSET_SEED,
            rate=TEACHER_RATE,
            skip_existing=True,
            family=family,
        )

    print("\nTeacher MI sweep finished:")
    for yaml_path, wrote in tally.items():
        print(f"  {'computed' if wrote else 'skipped/missing':<16} {yaml_path}")
    print(f"  -> {master_csv_path}")
    return tally


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    selectors = sys.argv[1:]
    chosen = [
        (y, f) for y, f in TEACHERS
        if not selectors or any(s in y.rsplit("/", 1)[-1] for s in selectors)
    ]
    if not chosen:
        print(f"No teacher plan matches {selectors}")
    else:
        run_teacher_mi(teachers=chosen)
