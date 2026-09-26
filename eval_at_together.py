import os
import glob
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import csv
from torch.utils.data import DataLoader
from util import load_best_checkpoint, process_yaml_file, build_dataset_from_yaml
from util_adv import (NormalizedModel, at_one_epoch, build_model, create_at_logger, load_stolen_model, build_at_dataset_from_yaml, initialize_optimizer_scheduler,
                      compute_clean_accuracy, compute_robust_test_accuracy, log_at_epoch, parse_attack_configs, pgd_attack_v2, fgsm_attack)

# =====================================================
# 1. Global setup
# =====================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_eval_model(exp_yaml, dataset_obj, num_classes, model_path):
    model_name = exp_yaml.get("Model", "ResNet-18")
    net = build_model(model_name, num_classes).to(device)

    ckpt_path, _ = load_best_checkpoint(model_path)
    if ckpt_path is None:
        raise FileNotFoundError(f"No best_epoch.pth found in {model_path}")

    state = torch.load(ckpt_path, map_location=device)

    # Strip NormalizedModel wrapper prefix if present
    if any(k.startswith('base_model.') for k in state.keys()):
        state = {k.replace('base_model.', ''): v
                 for k, v in state.items()
                 if k not in ('mean', 'std')}

    net.load_state_dict(state)
    net = NormalizedModel(net, dataset_obj.mean, dataset_obj.std).to(device)
    net.eval()
    return net


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True

    

# =====================================================
# 7. Main Adversarial Train
# =====================================================
def main_cross_eval(seed, yaml_file_path):
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    exp_yaml = process_yaml_file(yaml_file_path)

    # ---------- Dataset ----------
    dataset_obj, num_classes, _ = build_dataset_from_yaml(exp_yaml["Dataset"])
    _, at_val = build_at_dataset_from_yaml(dataset_obj, exp_yaml["Dataset"])

    at_val_loader = DataLoader(at_val, batch_size=128, shuffle=False,
                               num_workers=8, worker_init_fn=seed_worker,
                               generator=g, persistent_workers=True,
                               pin_memory=True)

    # ---------- Models to evaluate ----------
    model_dir = Path("./saved_models/at_train1")
    model_entries = [(p.name, str(p)) for p in sorted(model_dir.iterdir()) if p.is_dir()]

    # NEW: Filter directories to only include the current seed
    seed_tag = f"atseed={seed}"
    model_entries = [
        (p.name, str(p)) for p in sorted(model_dir.iterdir()) 
        if p.is_dir() and seed_tag in p.name
    ]

    # Add vanilla baseline at the top
    model_entries.insert(0, (
        "Pre-AT_Baseline",
        "./saved_models/vanilla/CIFAR-10_ResNet-18_25000_42_1.0/"
    ))

    # ---------- All attack configs to evaluate against ----------
    eval_attack_configs = []

    # FGSM sweeps
    for eps in [0.005, 0.01, 0.02, 0.03, 0.05, 0.07]:
        eval_attack_configs.append(
            (fgsm_attack, f"FGSM_eps={eps}", {"eps": eps})
        )

    # PGD sweeps — adjust steps/alpha/restarts to match your training settings
    for eps in [0.005, 0.01, 0.02, 0.03, 0.05, 0.07]:
        eval_attack_configs.append(
            (pgd_attack_v2, f"PGD_eps={eps}_steps=10",
             {"eps": eps, "steps": 10})
        )

    # ---------- Results matrix ----------
    results = []

    for display_name, model_path in model_entries:
        print(f"\n{'='*60}")
        print(f"Evaluating model: {display_name}")
        print(f"{'='*60}")

        suspect_model = load_eval_model(exp_yaml, dataset_obj, num_classes, model_path)

        clean_acc = compute_clean_accuracy(suspect_model, at_val_loader)
        print(f"  Clean Acc: {clean_acc*100:.2f}%")

        row = {"model": display_name, "clean_acc": clean_acc}

        for attack_fn, attack_label, attack_kwargs in eval_attack_configs:
            rob_acc = compute_robust_test_accuracy(
                suspect_model, at_val_loader, attack_fn, attack_kwargs
            )
            print(f"  {attack_label}: {rob_acc*100:.2f}%")
            row[attack_label] = rob_acc

        results.append(row)

    # ---------- Write single cross-eval CSV ----------
    log_dir = Path("./saved_logs/at_train1/Performance")
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / f"cross_eval_seed={seed}.csv"

    fieldnames = (["model", "clean_acc"]
                  + [cfg[1] for cfg in eval_attack_configs])

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nResults saved to {csv_path}")


def main_transfer_eval(seed, yaml_file_path):
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    exp_yaml = process_yaml_file(yaml_file_path)

    # ---------- Dataset ----------
    dataset_obj, num_classes, _ = build_dataset_from_yaml(exp_yaml["Dataset"])
    _, at_val = build_at_dataset_from_yaml(dataset_obj, exp_yaml["Dataset"])

    at_val_loader = DataLoader(at_val, batch_size=128, shuffle=False,
                               num_workers=8, worker_init_fn=seed_worker,
                               generator=g, persistent_workers=True,
                               pin_memory=True)

    # ---------- Models to evaluate ----------
    model_dir = Path("./saved_models/at_train1")
    model_entries = [(p.name, str(p)) for p in sorted(model_dir.iterdir()) if p.is_dir()]

    # NEW: Filter directories to only include the current seed
    seed_tag = f"atseed={seed}"
    model_entries = [
        (p.name, str(p)) for p in sorted(model_dir.iterdir()) 
        if p.is_dir() and seed_tag in p.name
    ]

    # Add vanilla baseline at the top
    model_entries.insert(0, (
        "Pre-AT_Baseline",
        "./saved_models/vanilla/CIFAR-10_ResNet-18_25000_42_1.0/"
    ))

    # ---------- All attack configs to evaluate against ----------
    eval_attack_configs = []

    # FGSM sweeps
    for eps in [0.005, 0.01, 0.02, 0.03, 0.05, 0.07]:
        eval_attack_configs.append(
            (fgsm_attack, f"FGSM_eps={eps}", {"eps": eps})
        )

    # PGD sweeps — adjust steps/alpha/restarts to match your training settings
    for eps in [0.005, 0.01, 0.02, 0.03, 0.05, 0.07]:
        eval_attack_configs.append(
            (pgd_attack_v2, f"PGD_eps={eps}_steps=20",
             {"eps": eps, "steps": 20})
        )

    # ---------- Results matrix ----------
    results = []

    for display_name, model_path in model_entries:
        print(f"\n{'='*60}")
        print(f"Evaluating model: {display_name}")
        print(f"{'='*60}")

        suspect_model = load_eval_model(exp_yaml, dataset_obj, num_classes, model_path)

        clean_acc = compute_clean_accuracy(suspect_model, at_val_loader)
        print(f"  Clean Acc: {clean_acc*100:.2f}%")

        row = {"model": display_name, "clean_acc": clean_acc}

        for attack_fn, attack_label, attack_kwargs in eval_attack_configs:
            rob_acc = compute_robust_test_accuracy(
                suspect_model, at_val_loader, attack_fn, attack_kwargs
            )
            print(f"  {attack_label}: {rob_acc*100:.2f}%")
            row[attack_label] = rob_acc

        results.append(row)

    # ---------- Write single cross-eval CSV ----------
    log_dir = Path("./saved_logs/at_train1/Performance")
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / f"cross_eval_seed={seed}.csv"

    fieldnames = (["model", "clean_acc"]
                  + [cfg[1] for cfg in eval_attack_configs])

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nResults saved to {csv_path}")



# =====================================================
# 8. Entry point
# =====================================================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    exp_dir    = "./saved_exp_plan/at_plan"
    yaml_files = sorted(glob.glob(os.path.join(exp_dir, "*.yaml")))

    if not yaml_files:
        print(f"No YAML files found in {exp_dir}")
    else:
        print(f"Found {len(yaml_files)} experiment plan(s):")
        for f in yaml_files:
            print(" -", f)

     # Iterate over YAML files and seeds
    for yaml_path in yaml_files:
        print(f"\n========== Starting experiments from {yaml_path} ==========")
        for seed in range(42, 45): # entry point for main_nega
            print(f"\n>>> Running seed {seed} for {os.path.basename(yaml_path)}")
            set_seed(seed)

            main_cross_eval(seed, yaml_path)
    