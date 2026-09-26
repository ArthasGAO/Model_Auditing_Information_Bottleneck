"""
Explore the Certainty Score (CS) distribution of CIFAR-10/100 test samples
under a trained victim model.

CS(f, x) = sum_i f_i(x)^2  (DeepGini Gini-purity)

For a C-class classifier:
    - CS in [1/C, 1.0]
    - high CS = peaked softmax = model is confident, sample is far from boundary
    - low  CS = flat softmax  = model is uncertain, sample is near a boundary

Usage:
    Drop a YAML containing only a Victim block into ./saved_exp_plan/cs_explore/
    and run this script. No suspect / attack / training keys are required.
"""

import os
import glob
import random
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from util import process_yaml_file, build_dataset_from_yaml, load_best_checkpoint

from Model.ResNet_18 import ResNet18
from Model.VGG16 import ModifiedVGG16
from Model.MLP import MNIST_MLP


# =====================================================
# Setup (mirrors the main framework)
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def build_model(model_name, num_classes):
    if model_name == "MLP":
        return MNIST_MLP()
    elif model_name == "ResNet-18":
        return ResNet18(num_classes=num_classes)
    elif model_name == "VGG16":
        return ModifiedVGG16(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


# =====================================================
# Same NormalizedModel wrapper used in the main framework
# =====================================================
class NormalizedModel(torch.nn.Module):
    def __init__(self, base_model, mean, std):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        x_norm = (x - self.mean) / self.std
        return self.base_model(x_norm)


def load_victim_model(victim_cfg, dataset_obj, num_classes, model_seed=42):
    model_name = victim_cfg.get("Model", "ResNet-18")
    net = build_model(model_name, num_classes).to(device)

    folder = victim_cfg["Model_Name"] + f"_{model_seed}_{1.0}"
    model_dir = Path("./saved_models/vanilla/") / folder
    ckpt, _ = load_best_checkpoint(model_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No victim checkpoint in {model_dir}")

    net.load_state_dict(torch.load(ckpt, map_location=device))
    net = NormalizedModel(net, dataset_obj.mean, dataset_obj.std).to(device)
    net.eval()
    print(f"  Victim loaded from: {ckpt}")
    return net


def build_raw_test_loader(dataset_obj, batch_size=1000, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset_obj.raw_test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=True,
        pin_memory=True,
    )


# =====================================================
# CS computation
# =====================================================
@torch.no_grad()
def compute_cs_with_correctness(model, data_loader, device="cuda"):
    """
    Returns
    -------
    cs       : (N,) float tensor  — Certainty Scores
    correct  : (N,) bool tensor   — whether prediction matches the true label
    labels   : (N,) long tensor   — true labels
    preds    : (N,) long tensor   — predicted labels
    """
    model.eval()
    all_cs, all_correct, all_y, all_pred = [], [], [], []

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        probs = F.softmax(model(x), dim=1)
        cs    = (probs ** 2).sum(dim=1)
        pred  = probs.argmax(dim=1)
        correct = (pred == y)

        all_cs.append(cs.cpu())
        all_correct.append(correct.cpu())
        all_y.append(y.cpu())
        all_pred.append(pred.cpu())

    return (torch.cat(all_cs),
            torch.cat(all_correct),
            torch.cat(all_y),
            torch.cat(all_pred))


# =====================================================
# Reporting
# =====================================================
def report_cs_distribution(cs, correct, labels, num_classes, scenario_name,
                           top_ks=(100, 500, 1000, 2000)):
    """Print summary stats + per-class breakdown + top-K cutoffs."""
    cs_np    = cs.numpy()
    cor_np   = correct.numpy()
    lbl_np   = labels.numpy()
    n        = len(cs_np)
    cs_floor = 1.0 / num_classes

    print("\n" + "=" * 60)
    print(f"  CS distribution — {scenario_name}")
    print("=" * 60)
    print(f"  Samples:        {n}")
    print(f"  Num classes:    {num_classes}")
    print(f"  CS theoretical: [{cs_floor:.4f}, 1.0000]")
    print()
    print("  --- Overall stats ---")
    print(f"    min    = {cs_np.min():.4f}")
    print(f"    p25    = {np.percentile(cs_np, 25):.4f}")
    print(f"    median = {np.median(cs_np):.4f}")
    print(f"    p75    = {np.percentile(cs_np, 75):.4f}")
    print(f"    p95    = {np.percentile(cs_np, 95):.4f}")
    print(f"    p99    = {np.percentile(cs_np, 99):.4f}")
    print(f"    max    = {cs_np.max():.4f}")
    print(f"    mean   = {cs_np.mean():.4f}")
    print(f"    std    = {cs_np.std():.4f}")

    print("\n  --- Correctly vs incorrectly classified ---")
    cs_cor = cs_np[cor_np]
    cs_inc = cs_np[~cor_np]
    print(f"    correct  (n={cor_np.sum():>5d}): "
          f"mean={cs_cor.mean():.4f}, median={np.median(cs_cor):.4f}")
    if len(cs_inc):
        print(f"    incorrect (n={(~cor_np).sum():>4d}): "
              f"mean={cs_inc.mean():.4f}, median={np.median(cs_inc):.4f}")
    else:
        print("    incorrect (n=0): victim is 100% accurate on this test set")

    print("\n  --- Top-K cutoffs (CS at the K-th-ranked seed) ---")
    sorted_desc = np.sort(cs_np)[::-1]
    for k in top_ks:
        if k <= n:
            cutoff = sorted_desc[k - 1]
            avg_topk = sorted_desc[:k].mean()
            n_correct_topk = cor_np[np.argsort(-cs_np)[:k]].sum()
            print(f"    top-{k:>5d}: CS_cutoff = {cutoff:.4f}, "
                  f"mean_CS = {avg_topk:.4f}, "
                  f"correct = {n_correct_topk}/{k} "
                  f"({100*n_correct_topk/k:.1f}%)")

    print("\n  --- Per-class summary (top 10 classes) ---")
    print(f"    {'class':>6s}  {'count':>6s}  {'mean':>7s}  {'median':>7s}  {'min':>7s}  {'acc':>6s}")
    classes_to_show = sorted(np.unique(lbl_np))[:10]
    for c in classes_to_show:
        mask  = lbl_np == c
        cs_c  = cs_np[mask]
        cor_c = cor_np[mask]
        print(f"    {c:>6d}  {mask.sum():>6d}  "
              f"{cs_c.mean():>7.4f}  {np.median(cs_c):>7.4f}  "
              f"{cs_c.min():>7.4f}  {cor_c.mean()*100:>5.1f}%")
    if num_classes > 10:
        print(f"    ... and {num_classes - 10} more classes")


def plot_cs_histogram(cs, correct, num_classes, scenario_name, save_path,
                      top_ks=(100, 1000)):
    """Histogram of CS, with correct/incorrect overlay and top-K cutoffs marked."""
    cs_np  = cs.numpy()
    cor_np = correct.numpy()
    cs_floor = 1.0 / num_classes

    fig, ax = plt.subplots(figsize=(10, 5))

    bins = np.linspace(cs_floor, 1.0, 60)
    ax.hist(cs_np[cor_np],  bins=bins, alpha=0.7,
            label=f"correct (n={cor_np.sum()})", color="tab:blue")
    if (~cor_np).any():
        ax.hist(cs_np[~cor_np], bins=bins, alpha=0.7,
                label=f"incorrect (n={(~cor_np).sum()})", color="tab:red")

    sorted_desc = np.sort(cs_np)[::-1]
    for k in top_ks:
        if k <= len(cs_np):
            cutoff = sorted_desc[k - 1]
            ax.axvline(cutoff, linestyle="--", linewidth=1.2,
                       label=f"top-{k} cutoff @ CS={cutoff:.3f}")

    ax.set_xlabel("Certainty Score (CS)")
    ax.set_ylabel("Count")
    ax.set_title(f"CS distribution — {scenario_name}")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  Histogram saved -> {save_path}")


def save_cs_dump(cs, correct, labels, preds, scenario_name, save_path):
    """Dump raw per-sample arrays for downstream analysis (sortable in pandas)."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "scenario_name": scenario_name,
        "cs":      cs,
        "correct": correct,
        "labels":  labels,
        "preds":   preds,
    }, save_path)
    print(f"  Raw CS dump saved -> {save_path}")


# =====================================================
# Per-YAML driver
# =====================================================
def explore_cs(yaml_file_path, output_dir="./saved_logs/cs_explore"):
    print(f"\nDevice: {device}")
    print(f"YAML  : {yaml_file_path}")

    exp_yaml = process_yaml_file(yaml_file_path)
    scenario_name = exp_yaml.get("Scenario_Name", Path(yaml_file_path).stem)

    victim_cfg    = exp_yaml["Victim"]
    victim_ds_cfg = victim_cfg["Dataset"]
    dataset_obj, num_classes, _ = build_dataset_from_yaml(victim_ds_cfg)

    print("==> Loading victim model..")
    victim_model = load_victim_model(victim_cfg, dataset_obj, num_classes)

    print("==> Loading raw [0,1] test data..")
    raw_loader = build_raw_test_loader(dataset_obj, batch_size=1000)

    print("==> Computing CS for each test sample..")
    cs, correct, labels, preds = compute_cs_with_correctness(
        victim_model, raw_loader, device=device
    )

    report_cs_distribution(cs, correct, labels, num_classes, scenario_name)

    hist_path = f"{output_dir}/{scenario_name}_cs_hist.png"
    dump_path = f"{output_dir}/{scenario_name}_cs_dump.pt"
    plot_cs_histogram(cs, correct, num_classes, scenario_name, hist_path)
    save_cs_dump(cs, correct, labels, preds, scenario_name, dump_path)


# =====================================================
# Entry point
# =====================================================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    set_seed(42)

    # Drop minimal Victim-only YAMLs in this folder.
    exp_dir = "./saved_exp_plan/cs_explore"
    yaml_files = sorted(glob.glob(os.path.join(exp_dir, "*.yaml")))

    if not yaml_files:
        print(f"No YAML files found in {exp_dir}")
        print("Drop a Victim-only YAML there and re-run.")
    else:
        print(f"Found {len(yaml_files)} YAML(s):")
        for f in yaml_files:
            print(" -", f)

    for yaml_path in yaml_files:
        explore_cs(yaml_path)