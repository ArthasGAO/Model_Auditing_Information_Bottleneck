"""MI measurement for substitutes extracted from a DeiT victim.

Fork of `calculate_MI_extraction.py`. Identical measurement protocol - the
In set is still the victim's group_A and the Out set the victim's test set,
so these rows are directly comparable to the CNN ones. Two differences:

  * the substitute may be a DeiT (dict-valued `Substitute.Model`), so it is
    built through `build_model_any`;
  * substitutes are read from saved_models/extraction_vanilla/Transformer_Models/.

Rows are appended to the SAME master table as the CNN runs, so the plotting
framework picks up DeiT scenarios with no changes.
"""
from datetime import datetime
import os
import glob
import time
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # needed for full CUDA determinism
from pathlib import Path
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import csv
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
import torchvision
import torchvision.transforms as transforms
from util import (process_yaml_file, build_dataset_from_yaml, evaluate2,
                  create_or_load_group_A, load_best_checkpoint, build_deit_student)
from Model.ResNet_18 import ResNet18
from Model.VGG16 import ModifiedVGG16
from Model.MLP import MNIST_MLP
import torch.nn.functional as F

# =====================================================
# 1. Global setup
# =====================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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


def build_model(model_name, num_classes):
    if model_name == "MLP":
        return MNIST_MLP()
    elif model_name == "ResNet-18":
        return ResNet18(num_classes=num_classes)
    elif model_name == "VGG16":
        return ModifiedVGG16(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def build_model_any(model_cfg, num_classes):
    """Build either a CNN (str config) or a DeiT (dict config). See
    main_knockoff_extraction_deit.build_model_any for the rationale."""
    if isinstance(model_cfg, dict):
        return build_deit_student({"Model": model_cfg}, num_classes)
    return build_model(model_cfg, num_classes)
    

def create_nested_balanced_subsets(
    dataset, group_A, save_dir, subset_sizes,
    num_classes=10, seed=42, force_rebuild=False,
):
    """Create nested class-balanced subsets of group_A. See previous docstring."""
    save_path = Path(save_dir) / f"nested_subsets_seed{seed}.npz"
    if save_path.exists() and not force_rebuild:
        print(f"[INFO] Loading nested subsets from {save_path}")
        data = np.load(save_path)
        return {int(k.split("_")[1]): data[k] for k in data.files}

    print(f"[INFO] Building nested subsets for sizes {subset_sizes}")
    rng = np.random.default_rng(seed)

    class_indices = {c: [] for c in range(num_classes)}
    for idx in group_A:
        _, y = dataset[idx]
        class_indices[int(y)].append(idx)
    for c in range(num_classes):
        arr = np.array(class_indices[c])
        rng.shuffle(arr)
        class_indices[c] = arr

    max_per_class = min(len(v) for v in class_indices.values())
    for s in subset_sizes:
        if s % num_classes != 0:
            raise ValueError(
                f"subset_size {s} not divisible by num_classes {num_classes}"
            )
        if s // num_classes > max_per_class:
            raise ValueError(
                f"subset_size {s} requires {s // num_classes} per class, "
                f"but smallest class has only {max_per_class}"
            )

    subsets = {}
    for s in subset_sizes:
        n_per_class = s // num_classes
        idx = np.concatenate([
            class_indices[c][:n_per_class] for c in range(num_classes)
        ])
        order_rng = np.random.default_rng(seed + s)
        order_rng.shuffle(idx)
        subsets[s] = idx

    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, **{f"size_{s}": subsets[s] for s in subset_sizes})
    print(f"[INFO] Saved nested subsets to {save_path}")
    return subsets


# ===========================================================================
# Step 1: Inference -- run the model ONCE per loader, cache logits + labels
# ===========================================================================

def _infer_num_classes(net):
    if hasattr(net, "fc"):
        return net.fc.out_features
    elif hasattr(net, "classifier"):
        last_linear = [m for m in net.classifier.modules()
                       if isinstance(m, nn.Linear)][-1]
        return last_linear.out_features
    elif hasattr(net, "num_classes"):
        return net.num_classes
    raise ValueError("Cannot automatically infer num_classes from model.")


def collect_logits(net, data_loader, device):
    """Run the model over the loader once and return cached logits + one-hot labels.

    This is the expensive step. The returned tensors can be reused for any
    number of MI computations at different bin counts.

    Returns:
        layer_T:      (N, K) float32 logits tensor on `device`.
        label_matrix: (N, K_y) float32 one-hot label tensor on `device`.
    """
    start_time = time.time()
    num_classes = _infer_num_classes(net)
    layer_T_list, label_list = [], []

    net.eval()
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            layer_T_list.append(outputs.detach())
            label_list.append(F.one_hot(targets.detach(),
                                        num_classes=num_classes).float())

    layer_T = torch.cat(layer_T_list, dim=0).to(dtype=torch.float32)
    label_matrix = torch.cat(label_list, dim=0).to(dtype=torch.float32)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"logits inference costs: {elapsed_time}")
    return layer_T, label_matrix


# ===========================================================================
# Step 2: MI computation -- cheap, can be called many times on cached logits
# ===========================================================================

def MI_formula_cal(matrix, p1, p2):
    mask = matrix > 0
    denom = p1[:, None] * p2[None, :]
    ratio = matrix / denom
    log_ratio = torch.log2(ratio)
    return (matrix * log_ratio)[mask].sum()


def mi_from_logits(layer_T, label_matrix, num_intervals=50, verbose=False):
    """Compute (I(X;T), I(T;Y)) from cached logits at a given bin count.

    This is the function to call repeatedly for bin sweeps. It does NOT run
    the model; it only does softmax + binning + counting + entropy.

    Returns (I_X_T, I_T_Y) when verbose=False, or
            (I_X_T, I_T_Y, debug_data) when verbose=True.
    """
    start_time = time.time()
    device = layer_T.device
    N = layer_T.shape[0]

    T_soft = torch.softmax(layer_T, dim=1)
    bins = torch.linspace(0, 1, num_intervals + 1, device=device, dtype=torch.float32)
    T_discrete = torch.bucketize(T_soft, bins, right=True) - 1
    T_discrete = T_discrete.clamp(0, num_intervals - 1).contiguous()

    unique_T, inverse_idx = torch.unique(T_discrete, dim=0, return_inverse=True)
    K_unique = unique_T.shape[0]

    T_counts = torch.zeros(K_unique, device=device, dtype=torch.float32)
    T_counts.index_add_(0, inverse_idx,
                        torch.ones(N, device=device, dtype=torch.float32))
    p_T = T_counts / N
    mask_T = p_T > 0
    I_X_T = -(p_T[mask_T] * torch.log2(p_T[mask_T])).sum()

    K_y = label_matrix.shape[1]
    TY_counts = torch.zeros((K_unique, K_y), device=device, dtype=torch.float32)
    TY_counts.index_add_(0, inverse_idx, label_matrix)
    TY_matrix = TY_counts / N
    P_T_marg = TY_matrix.sum(dim=1)
    P_Y_marg = TY_matrix.sum(dim=0)
    I_T_Y = MI_formula_cal(TY_matrix, P_T_marg, P_Y_marg)

    if not verbose:
        return I_X_T.item(), I_T_Y.item()

    debug_data = {
        "T_soft":         T_soft.detach().cpu().numpy().astype(np.float32),
        "T_discrete":     T_discrete.detach().cpu().numpy().astype(np.int16),
        "unique_T":       unique_T.detach().cpu().numpy().astype(np.int16),
        "inverse_idx":    inverse_idx.detach().cpu().numpy().astype(np.int32),
        "pattern_counts": T_counts.detach().cpu().numpy().astype(np.int64),
        "label_counts":   TY_counts.detach().cpu().numpy().astype(np.int64),
        "N":              np.int64(N),
        "K":              np.int64(layer_T.shape[1]),
        "K_y":            np.int64(K_y),
        "U":              np.int64(K_unique),
        "num_intervals":  np.int64(num_intervals),
        "I_X_T":          np.float64(I_X_T.item()),
        "I_T_Y":          np.float64(I_T_Y.item()),
    }

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"MI cal costs: {elapsed_time}")
    return I_X_T.item(), I_T_Y.item(), debug_data


def save_verbose_data(debug_data, save_dir, model_name, split_name,
                      num_intervals, in_size=None):
    """Save verbose debug data to a structured .npz file."""
    out_dir = Path(save_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if split_name == "in" and in_size is not None:
        fname = f"in_size{in_size}_bins{num_intervals}.npz"
    else:
        fname = f"{split_name}_bins{num_intervals}.npz"
    out_path = out_dir / fname
    np.savez_compressed(out_path, **debug_data)
    return out_path


# ===========================================================================
# Master CSV
# ===========================================================================

MASTER_CSV_COLUMNS = [
    "Scenario", "seed", "rate", "model_name", "epoch", "bins",
    "in_size",  "I(X;T)-In",  "I(T;Y)-In",
    "out_size", "I(X;T)-Out", "I(T;Y)-Out",
    "timestamp",
]


def ensure_master_csv(csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(MASTER_CSV_COLUMNS)


def append_master_row(csv_path, row_dict):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([row_dict[c] for c in MASTER_CSV_COLUMNS])


# =====================================================
# 3. Main MI calculation logic
# =====================================================
def main_extraction_mi_check(
    seed,
    yaml_file_path,
    in_sizes=None,
    num_intervals_list=None,
    record_verbose=True,
    master_csv_path="./saved_logs/extraction_vanilla/MI_master_table_extraction.csv",
    verbose_dir="./saved_logs/extraction_vanilla/MI_verbose",
    subset_seed=42,
):
    """
    Calculate MI metrics for a knockoff/extraction substitute model
    using the new MI-check pipeline.
    """

    if in_sizes is None:
        in_sizes = [5000, 10000, 15000, 20000, 25000]

    if num_intervals_list is None:
        num_intervals_list = [50, 75, 100, 125, 150]

    print(f"Device: {device}")

    # ============================================================
    # 1. Parse YAML and victim dataset information
    # ============================================================
    exp_yaml = process_yaml_file(yaml_file_path)

    victim_cfg = exp_yaml["Victim"]
    victim_ds_cfg = victim_cfg["Dataset"]

    victim_dataset_obj, victim_num_classes, victim_group_size = build_dataset_from_yaml(
        victim_ds_cfg
    )

    # ============================================================
    # 2. Load extracted substitute model
    # ============================================================
    print("==> Loading extracted substitute model..")

    sub_model_cfg = exp_yaml["Substitute"].get(
        "Model",
        victim_cfg.get("Model", "ResNet-18")
    )

    model_name = exp_yaml["Scenario_Name"] + f"_{seed}_{1.0}"
    print(f"\nThis time extraction scenario is: {model_name}")

    model_dir = "./saved_models/extraction_vanilla/Transformer_Models/"
    model_folder = Path(model_dir) / model_name

    ckpt_path, _ = load_best_checkpoint(model_folder)
    if ckpt_path is None:
        print(f"[WARNING] No .pth files found in {model_folder}")
        return

    net = build_model_any(sub_model_cfg, victim_num_classes).to(device)

    state = torch.load(ckpt_path, map_location=device)

    net.load_state_dict(state)

    net.eval()

    print(f"Substitute model loaded from: {ckpt_path}")

    # ============================================================
    # 3. Prepare victim in-sample query data
    # ============================================================
    print("==> Preparing in-sample query data..")

    in_sample_set = victim_dataset_obj.in_sample_set
    out_sample_set = victim_dataset_obj.test_set

    # This group_A should be the victim training group D_V.
    group_A = create_or_load_group_A(
        dataset=in_sample_set,
        save_dir=f'./Indices/{victim_ds_cfg["name"]}/',
        group_size=victim_group_size,
        num_classes=victim_num_classes,
        seed=42,
        force_rebuild=False,
    )

    nested_subsets = create_nested_balanced_subsets(
        dataset=in_sample_set,
        group_A=group_A,
        save_dir=f'./Indices/{victim_ds_cfg["name"]}/',
        subset_sizes=in_sizes,
        num_classes=victim_num_classes,
        seed=subset_seed,
        force_rebuild=False,
    )

    out_size = len(out_sample_set)
    out_sample_loader = DataLoader(
        out_sample_set, batch_size=128, shuffle=False, num_workers=0,
        pin_memory=True,
    )

    # ============================================================
    # 4. Prepare logging
    # ============================================================
    ensure_master_csv(master_csv_path)
    os.makedirs(verbose_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ============================================================
    # 5. Phase 1: inference once per in-sample subset
    # ============================================================
    print("\n==> [Phase 1] Running inference once per in-sample subset")

    # Out-sample: infer once, reuse for all bin counts
    print(f"  Inferring out-sample (size={out_size})...")
    out_layer_T, out_labels = collect_logits(net, out_sample_loader, device)
    print(f"    -> cached logits shape: {tuple(out_layer_T.shape)}")

    in_cache = {}

    for in_size in in_sizes:
        subset_indices = nested_subsets[in_size].tolist()

        in_sample_subset = victim_dataset_obj.subset(
            "train",
            subset_indices,
            clean=True,
        )

        in_loader = DataLoader(
            in_sample_subset, batch_size=128, shuffle=False, 
            pin_memory=True,
        )

        print(f"  Inferring in-sample subset, size={in_size}...")
        layer_T, labels = collect_logits(net, in_loader, device)
        in_cache[in_size] = (layer_T, labels)
        print(f"    -> cached logits shape: {tuple(layer_T.shape)}")

    # ============================================================
    # 6. Phase 2: MI computation on cached logits
    # ============================================================
    print("\n==> [Phase 2] Computing MI on cached logits")

    # Out-sample MI for each bin count
    out_results = {}
    for nb in num_intervals_list:
        ixt_out, ity_out = mi_from_logits(
            out_layer_T, out_labels, num_intervals=nb, verbose=False,
        )
        print(f"  out  bins={nb:>3}: I(X;T)={ixt_out:.4f}, "
                f"I(T;Y)={ity_out:.4f}")
        out_results[nb] = (ixt_out, ity_out)

    # In-sample MI for each (in_size, bin) combination, plus CSV row writing
    for in_size in in_sizes:
        layer_T, labels = in_cache[in_size]
        for nb in num_intervals_list:
            ixt_in, ity_in = mi_from_logits(
                layer_T, labels, num_intervals=nb, verbose=False,
            )
            print(f"  in   size={in_size:>5}  bins={nb:>3}: "
                    f"I(X;T)={ixt_in:.4f}, I(T;Y)={ity_in:.4f}")

            ixt_out, ity_out = out_results[nb]

            row = {
                "Scenario":    exp_yaml["Scenario_Name"],
                "seed":        seed,
                "rate":        1.0,
                "model_name":  model_name,
                "epoch":       99,
                "bins":        nb,
                "in_size":     in_size,
                "I(X;T)-In":   f"{ixt_in:.6f}",
                "I(T;Y)-In":   f"{ity_in:.6f}",
                "out_size":    out_size,
                "I(X;T)-Out":  f"{ixt_out:.6f}",
                "I(T;Y)-Out":  f"{ity_out:.6f}",
                "timestamp":   timestamp,
            }
            append_master_row(master_csv_path, row)

    # ============================================================
    # 7. Clean memory
    # ============================================================
    del in_cache, out_layer_T, out_labels

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n==> Done. Extraction MI master CSV: {master_csv_path}")



# =====================================================
# 4. Entry point
# =====================================================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Folder containing all YAML experiment plans
    exp_dir = "./saved_exp_plan/extraction_plan_crossarch"
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

        for seed in range(0, 1):
            print(f"\n>>> Running seed {seed} for {os.path.basename(yaml_path)}")
            set_seed(seed)

            main_extraction_mi_check(seed, yaml_path, 
                          in_sizes = [25000],#1000, 5000, 10000, 15000, 20000, 25000],
                          num_intervals_list = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200],
                          record_verbose=False,
                          master_csv_path="./saved_logs/extraction_vanilla/MI_master_table_extraction.csv",
                          verbose_dir="./saved_logs/extraction_vanilla/MI_verbose",
            )
