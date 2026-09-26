"""
Train stand-alone teacher / victim models for the "bigger teacher -> pooled
student" distillation row (e.g. the ResNet-34 victim behind
saved_exp_plan/kd_plan_rn34/).

This is a separate entry so that the production entry in main_train_nega.py -
which runs the DeiT pool plans in saved_exp_plan/train_plan/ through
main_deit_overlap and reads SEED_START / SEED_END - stays untouched. It reuses
main_train_nega.main_overlap unchanged, which trains a CNN on group_B(rate)
(rate 1.0 == group_A, the victim's own split) and writes
    ./saved_models/vanilla/{Scenario_Name}_{seed}_{rate}/best_epoch.pth
    ./saved_logs/vanilla/Performance/training_log_{Scenario_Name}_{seed}_{rate}.csv

Plans : every *.yaml in saved_exp_plan/train_plan/teacher_plan/, narrowed by
        any command-line arguments - each is a case-insensitive substring
        matched against the file name, so `python main_train_teacher.py VGG19`
        trains only the VGG19 victims. No arguments means every plan.
Seeds : TEACHER_SEEDS, comma separated, default "42"
Rates : TEACHER_RATES, comma separated, default "1.0" (victim on group_A);
        "0.0" gives the independent teacher used by the control cells.

    PowerShell   python main_train_teacher.py
                 python main_train_teacher.py CIFAR10_VGG19
                 $env:TEACHER_RATES="0.0"; python main_train_teacher.py
    bash         TEACHER_RATES="0.0" python main_train_teacher.py

A (plan, seed, rate) whose best_epoch.pth already exists is skipped, so the
script can be re-run without retraining or appending to a finished log.

DeiT plans (the ones whose `Model:` is a dict rather than a string) are
reported and skipped: they need main_train_nega.main_deit_overlap, which this
entry does not call yet. Without that guard the first such plan aborted the
whole sweep inside process_experiment_setup.
"""
import os
import sys
import glob

import torch

from main_train_nega import main_overlap, set_seed, DETERMINISTIC
from util import process_yaml_file


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    exp_folder = "./saved_exp_plan/train_plan/teacher_plan"
    yaml_files = sorted(glob.glob(os.path.join(exp_folder, "*.yaml")))
    seeds = [int(s) for s in os.environ.get("TEACHER_SEEDS", "42").split(",")]
    rates = [float(r) for r in os.environ.get("TEACHER_RATES", "1.0").split(",")]

    # Optional positional filters on the file name, e.g. "VGG19" or "CIFAR10".
    filters = [a.lower() for a in sys.argv[1:]]
    if filters:
        yaml_files = [p for p in yaml_files
                      if any(f in os.path.basename(p).lower() for f in filters)]

    if not yaml_files:
        print(f"No YAML files found in {exp_folder}"
              + (f" matching {filters}" if filters else ""))
    else:
        print(f"Found {len(yaml_files)} teacher plan(s)"
              + (f" matching {filters}" if filters else "") + ":")
        for f in yaml_files:
            print(" -", f)
    print(f"Seeds: {seeds}   Rates: {rates}   Deterministic: {DETERMINISTIC}")

    for yaml_path in yaml_files:
        plan = process_yaml_file(yaml_path)
        scenario = plan["Scenario_Name"]
        print(f"\n========== {os.path.basename(yaml_path)}  ({scenario}) ==========")

        if isinstance(plan.get("Model"), dict):
            # A timm/DeiT plan. main_overlap builds CNNs through
            # process_experiment_setup, which only understands a string Model
            # and would raise "Unsupported model: {...}" here.
            print(f"[SKIP] {os.path.basename(yaml_path)}: Model is a dict "
                  f"(timm/DeiT). This entry only drives main_overlap; wire "
                  f"main_deit_overlap before running it.")
            continue

        for seed in seeds:
            for rate in rates:
                out_dir = f"./saved_models/vanilla/{scenario}_{seed}_{round(rate, 2)}"
                if os.path.exists(os.path.join(out_dir, "best_epoch.pth")):
                    print(f"[SKIP] {out_dir}/best_epoch.pth already exists")
                    continue

                print(f"\n>>> Training seed {seed}, rate {rate}  ->  {out_dir}")
                set_seed(seed, deterministic=DETERMINISTIC)
                main_overlap(seed, rate, yaml_path)
