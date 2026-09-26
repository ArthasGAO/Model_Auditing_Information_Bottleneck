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
                  create_or_load_group_A, load_best_checkpoint,
                  build_deit_student)
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
    """Build either a CNN or a DeiT, depending on how the YAML declares it.

    str  -> "ResNet-18" / "VGG16" / "MLP", the unchanged CNN path.
    dict -> {model_name, img_size, patch_size, pretrained, drop_path_rate},
            built through the same helper the vanilla/knockoff training used,
            so the structure matches the saved checkpoint exactly.
    Mirrors main_knockoff_extraction_deit.build_model_any.
    """
    if isinstance(model_cfg, dict):
        return build_deit_student({"Model": model_cfg}, num_classes)
    return build_model(model_cfg, num_classes)


def model_cfg_name(model_cfg):
    """Stable string for a Model field that may be a str (CNN) or dict (DeiT).

    The identity columns are compared as strings by _missing_extraction_grid, so
    a raw dict must never reach the CSV -- it would serialise as a Python repr
    and make the resume check depend on dict ordering.
    """
    if isinstance(model_cfg, dict):
        return str(model_cfg.get("model_name", "deit"))
    return str(model_cfg)


def model_cfg_family(model_cfg):
    """'deit' for a dict-declared timm model, 'cnn' otherwise."""
    return "deit" if isinstance(model_cfg, dict) else "cnn"



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

    sub_model_name = exp_yaml["Substitute"].get(
        "Model",
        victim_cfg.get("Model", "ResNet-18")
    )

    model_name = exp_yaml["Scenario_Name"] + f"_{seed}_{1.0}"
    print(f"\nThis time extraction scenario is: {model_name}")

    model_dir = "./saved_models/extraction_vanilla/"
    model_folder = Path(model_dir) / model_name

    ckpt_path, _ = load_best_checkpoint(model_folder)
    if ckpt_path is None:
        print(f"[WARNING] No .pth files found in {model_folder}")
        return

    net = build_model(sub_model_name, victim_num_classes).to(device)

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



# ===========================================================================
# NEW (2026-09-17): best_epoch MI on the negative-pool-0.0 grid.
#
# Why this path exists
# --------------------
# `main_extraction_mi_check` above is kept untouched. It hardcodes the model
# directory to ./saved_models/extraction_vanilla/, takes absolute in_sizes,
# appends unconditionally (re-running duplicates rows) and writes the thin
# 13-column schema. The negative pool (MI_master_table_neg_pool0.csv), the
# victim (MI_master_table_victim.csv) and the FT positives
# (ft_final/MI_master_table_ft.csv) are all on the rate-based grid
# POOL0_IN_SIZE_RATES x group_size = {250, 1250, 2500, 5000, 12500, 18750,
# 25000} x POOL0_BINS, with a resume protocol and provenance columns.
#
# `main_extraction_best` below matches that form for extraction substitutes:
#   * grid from POOL0_IN_SIZE_RATES / POOL0_BINS (or explicit in_sizes)
#   * nested subsets from mi_pool_support (the SHARED builder), so the probe
#     sets are byte-identical to the pool / victim / FT ones. The local
#     builder above is left in place for the old path, but it is NOT used
#     here: it rewrites the shared npz down to only the requested sizes on a
#     rebuild, and it rejects sizes not divisible by num_classes (250, 1250
#     and 18750 are not divisible by 100, so CIFAR-100 fails there).
#   * missing_mi_grid resume: exact (in_size, bins) pairs, complete models
#     skipped, partial/mismatched rows raise instead of being overwritten
#   * ft-style provenance columns plus attack / victim_model /
#     substitute_model / aux_dataset so a plot spec can filter by variant
#   * a SEPARATE output CSV; the 2026-09-08 extraction_vanilla table is never
#     written to
#
# The MI estimator is unchanged: collect_logits / mi_from_logits in this file
# are byte-identical to MI_check's, verified 2026-09-17, and one model
# re-measured on the new grid reproduced the old table at in_size=25000 with
# a maximum difference of 0.0 across all four MI columns.
#
# DeiT substitutes are not handled here (build_model has no DeiT branch); use
# calculate_MI_extraction_deit.py for those.
#
# Usage (from E:\Experiment, pytorch_env)
#   python calculate_MI_extraction.py                # every plan in EXTRACTION_BEST_PLANS
#   python calculate_MI_extraction.py KNOCKOFF_Same10_Cross16   # substring filter
# ===========================================================================
import hashlib

from MI_check import POOL0_BINS, POOL0_IN_SIZE_RATES
from mi_pool_support import (
    create_nested_balanced_subsets as create_nested_balanced_subsets_shared,
    missing_mi_grid, positive_ints, sizes_from_rates,
)

EXTRACTION_BASE_DIR = Path(__file__).resolve().parent
EXTRACTION_BEST_MODEL_DIR = EXTRACTION_BASE_DIR / "saved_models/extraction_final"
EXTRACTION_BEST_MASTER_CSV = (EXTRACTION_BASE_DIR
                              / "saved_logs/extraction_final/MI_master_table_extraction.csv")
EXTRACTION_BEST_VERBOSE_DIR = (EXTRACTION_BASE_DIR
                               / "saved_logs/extraction_final/MI_verbose_best")
EXTRACTION_BEST_INDEX_DIR = EXTRACTION_BASE_DIR / "Indices"
# The CIFAR-10 Same10 3x3 matrix: 3 victim archs x 3 surrogate archs, all nine
# cells, both halves of the split (CNN-only plans and DeiT-involving plans are
# separate FOLDERS only because two different training scripts produce them --
# MI reads one flat model tree, so one sweep covers all nine).
# Seeds differ per cell (RN18 row has 0..4, the rest 0..2); the sweep reports
# a checkpoint that is not there instead of aborting, so one list is fine.
_C10 = EXTRACTION_BASE_DIR / "saved_exp_plan/knockoff_3x3_c10_same10"
_C10D = EXTRACTION_BASE_DIR / "saved_exp_plan/knockoff_3x3_c10_same10_deit"
# The CIFAR-100 Same100 3x3, same shape, trained on the second machine and
# copied into the same flat extraction_final/ tree. Its DeiT is the DISTILLED
# tiny (head + head_dist) because that dataset's negative pool is distilled;
# in eval mode it returns the averaged logits, so MI reads one [B,100] tensor
# exactly as for the plain DeiT. Every cell has seeds 0..2.
_C100 = EXTRACTION_BASE_DIR / "saved_exp_plan/knockoff_3x3_c100_same100"
_C100D = EXTRACTION_BASE_DIR / "saved_exp_plan/knockoff_3x3_c100_same100_deit"
_DFMS = EXTRACTION_BASE_DIR / "saved_exp_plan/dfms_plan/matrix"
# Double knockoff (two-hop extraction) on the CIFAR-10 ResNet-18 victim:
# victim -> S1 -> S2. The suspect is S2, so nothing about the schema changes -- the attack block is still `Knockoff`, the query
# set is still group_B, and the surrogate is still what MI reads logits from.
# Two consequences worth knowing when reading the table:
#   * `victim_model` holds arch(S1), NOT ResNet-18, because S1 is what the
#     attack queried. The Scenario string
#     (`Knockoff2_<q1>_Via<S1>_<q2>_<S2>`) is the authority on the chain.
#   * every chain's own hop-1 row is already in this CSV under
#     `..._Knockoff_Same10_{Same18,Cross16,CrossDeiT}`, so the before/after
#     comparison needs no extra sweep.
# Chain seed s means hop-1 seed s feeding hop-2 seed s; seeds 0..2.
_K2 = EXTRACTION_BASE_DIR / "saved_exp_plan/knockoff_double_c10/matrix"
EXTRACTION_BEST_PLANS = [
    _C10 / "CIFAR10_RES18_KNOCKOFF_Same10_Same18.yaml",     # RN18  -> RN18
    _C10 / "CIFAR10_RES18_KNOCKOFF_Same10_Cross16.yaml",    # RN18  -> VGG16
    _C10 / "CIFAR10_VGG16_KNOCKOFF_Same10_Cross18.yaml",    # VGG16 -> RN18
    _C10 / "CIFAR10_VGG16_KNOCKOFF_Same10_Same16.yaml",     # VGG16 -> VGG16
    _C10D / "CIFAR10_RES18_KNOCKOFF_Same10_CrossDeiT.yaml",  # RN18  -> DeiT
    _C10D / "CIFAR10_VGG16_KNOCKOFF_Same10_CrossDeiT.yaml",  # VGG16 -> DeiT
    _C10D / "CIFAR10_DEIT_KNOCKOFF_Same10_Cross18.yaml",     # DeiT  -> RN18
    _C10D / "CIFAR10_DEIT_KNOCKOFF_Same10_Cross16.yaml",     # DeiT  -> VGG16
    _C10D / "CIFAR10_DEIT_KNOCKOFF_Same10_SameDeiT.yaml",    # DeiT  -> DeiT
    _C100 / "CIFAR100_RES18_KNOCKOFF_Same100_Same18.yaml",    # RN18  -> RN18
    _C100 / "CIFAR100_RES18_KNOCKOFF_Same100_Cross16.yaml",   # RN18  -> VGG16
    _C100 / "CIFAR100_VGG16_KNOCKOFF_Same100_Cross18.yaml",   # VGG16 -> RN18
    _C100 / "CIFAR100_VGG16_KNOCKOFF_Same100_Same16.yaml",    # VGG16 -> VGG16
    _C100D / "CIFAR100_RES18_KNOCKOFF_Same100_CrossDeiT.yaml",  # RN18  -> DeiT
    _C100D / "CIFAR100_VGG16_KNOCKOFF_Same100_CrossDeiT.yaml",  # VGG16 -> DeiT
    _C100D / "CIFAR100_DEIT_KNOCKOFF_Same100_Cross18.yaml",     # DeiT  -> RN18
    _C100D / "CIFAR100_DEIT_KNOCKOFF_Same100_Cross16.yaml",     # DeiT  -> VGG16
    _C100D / "CIFAR100_DEIT_KNOCKOFF_Same100_SameDeiT.yaml",    # DeiT  -> DeiT
    # DFMS-HL (data-free, hard label), the CIFAR-10 3x3 with the 40-class
    # CIFAR-100 proxy. Same flat model tree, same schema: the attack column
    # comes from the plan's DFMS block and the query set from its Proxy block,
    # both already handled. Attacker seed 0 only.
    # All nine are listed even though most are not trained yet -- a missing
    # checkpoint is reported as [ABSENT] and skipped, so results coming back
    # from another machine need no edit here, just a re-run.
    # Paths point at matrix/ because that folder is the immutable output of
    # make_dfms_plans.py; dfms_plan/ itself only holds what is queued for
    # training, cases/ holds the per-machine copies, and finished plans move to
    # dfms_plan/done/. All copies are byte-identical, so plan_sha256 is stable
    # whichever one a run used.
    _DFMS / "CIFAR10_RES18_DFMS_C100-40C_Same18.yaml",      # RN18  -> RN18  (done)
    _DFMS / "CIFAR10_RES18_DFMS_C100-40C_Cross16.yaml",     # RN18  -> VGG16
    _DFMS / "CIFAR10_RES18_DFMS_C100-40C_CrossDeiT.yaml",   # RN18  -> DeiT  (done, pilot)
    _DFMS / "CIFAR10_VGG16_DFMS_C100-40C_Cross18.yaml",     # VGG16 -> RN18
    _DFMS / "CIFAR10_VGG16_DFMS_C100-40C_Same16.yaml",      # VGG16 -> VGG16
    _DFMS / "CIFAR10_VGG16_DFMS_C100-40C_CrossDeiT.yaml",   # VGG16 -> DeiT
    _DFMS / "CIFAR10_DEIT_DFMS_C100-40C_Cross18.yaml",      # DeiT  -> RN18
    _DFMS / "CIFAR10_DEIT_DFMS_C100-40C_Cross16.yaml",      # DeiT  -> VGG16
    _DFMS / "CIFAR10_DEIT_DFMS_C100-40C_SameDeiT.yaml",     # DeiT  -> DeiT
    # Two-hop cells: victim -> S1 -> S2, a 2x2 of (chain type) x (hop-2 query
    # set). Paths point at matrix/ (the generator's immutable output); run/
    # holds byte-identical copies, so plan_sha256 is the same whichever one
    # trained the model. Untrained cells report [ABSENT] and are skipped.
    #
    # A two-hop row must be compared against the 1-HOP surrogate of the victim
    # with the SAME query set and the SAME architecture, not against the chain's
    # own hop-1 model. The query set alone moves d by 2-5x (RN18 1-hop: 27.00
    # on CIFAR-10 group_B vs 61.37 on CIFAR-100 group_B), almost all of it on
    # I(X;T). Each plan's header names its own baseline; `aux_dataset` in this
    # CSV is the hop-2 query set and is what separates the two arms.
    _K2 / "CIFAR10_RES18_Knockoff2_Same10_Via18_Same10_Same18.yaml",      # RN18->RN18->RN18, in-dist
    _K2 / "CIFAR10_RES18_Knockoff2_Same10_Via18_Cross100_Same18.yaml",    # RN18->RN18->RN18, OOD
    _K2 / "CIFAR10_RES18_Knockoff2_Same10_Via16_Same10_CrossDeiT.yaml",   # RN18->VGG16->DeiT, in-dist
    _K2 / "CIFAR10_RES18_Knockoff2_Same10_Via16_Cross100_CrossDeiT.yaml", # RN18->VGG16->DeiT, OOD
    _K2 / "CIFAR10_RES18_Knockoff2_Same10_ViaDeiT_Same10_Cross16.yaml",   # RN18->DeiT->VGG16, in-dist
    _K2 / "CIFAR10_RES18_Knockoff2_Same10_ViaDeiT_Cross100_Cross16.yaml", # RN18->DeiT->VGG16, OOD
]
EXTRACTION_BEST_SEEDS = [0, 1, 2, 3, 4]
EXTRACTION_BEST_RATE = 1.0            # folder suffix of every extraction model
EXTRACTION_BEST_IN_SIZE_RATES = list(POOL0_IN_SIZE_RATES)
EXTRACTION_BEST_BINS = list(POOL0_BINS)
EXTRACTION_BEST_SUBSET_SEED = 42      # same nested subsets as pool / victim / FT
EXTRACTION_BEST_GROUP_SEED = 42       # group_A seed, fixed upstream
EXTRACTION_BEST_RECORD_VERBOSE = False
EXTRACTION_BEST_BATCH_SIZE = 128
EXTRACTION_BEST_NUM_WORKERS = 0
# Exactly one of these blocks names the attack in an extraction plan.
EXTRACTION_ATTACK_KEYS = ("Knockoff", "JBA", "DFMS")
# CNN substitutes are named by string. A DeiT substitute is declared as a dict
# (model_name/img_size/patch_size/...) and is accepted too: since the knockoff
# output tree was flattened into extraction_final/, there is no longer any path
# or schema difference between the families, so one sweep and one master CSV
# cover both. calculate_MI_extraction_deit.py is left untouched -- it still
# serves the older extraction_vanilla/Transformer_Models results.
EXTRACTION_BEST_SUBSTITUTES = ("ResNet-18", "VGG16", "MLP")

EXTRACTION_BEST_CSV_COLUMNS = [
    "Scenario", "seed", "rate", "model_name", "epoch", "bins", "in_size",
    "I(X;T)-In", "I(T;Y)-In", "out_size", "I(X;T)-Out", "I(T;Y)-Out",
    "timestamp", "attack", "victim_model", "substitute_model", "aux_dataset",
    "training_size", "in_size_rate", "family", "subset_seed", "group_seed",
    "checkpoint", "checkpoint_sha256", "plan_sha256",
]


def _extraction_file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extraction_attack_name(exp_yaml):
    """Name the attack from the single attack block present in the plan."""
    present = [key for key in EXTRACTION_ATTACK_KEYS if key in exp_yaml]
    if len(present) != 1:
        raise ValueError(
            f"Expected exactly one of {EXTRACTION_ATTACK_KEYS} in the plan, found {present}"
        )
    return present[0]


def _extraction_aux_dataset(exp_yaml):
    """Auxiliary/proxy query-set name; '' when the plan has neither block."""
    for key in ("Auxiliary_Dataset", "Proxy"):
        block = exp_yaml.get(key)
        if isinstance(block, dict) and block.get("name"):
            return str(block["name"])
        if isinstance(block, dict) and isinstance(block.get("Dataset"), dict):
            return str(block["Dataset"].get("name", ""))
    return ""


def _missing_extraction_grid(csv_path, identity, in_sizes, bins):
    """Reject mixed checkpoint/configuration identities before reusing MI."""
    path = Path(csv_path)
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != EXTRACTION_BEST_CSV_COLUMNS:
                raise ValueError(
                    f"Incompatible extraction best-MI CSV header: {path}; use a separate CSV"
                )
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Incomplete/malformed MI row in {path}")
                if row["model_name"] != identity["model_name"]:
                    continue
                for key, expected in identity.items():
                    if row[key] != str(expected):
                        raise ValueError(
                            f"Extraction best-MI identity mismatch: "
                            f"{identity['model_name']}, {key}; preserve the existing CSV and "
                            "use a new output for changed checkpoints/configuration"
                        )
                size = float(row["in_size"])
                fraction = float(row["in_size_rate"])
                if not (0 < size <= identity["training_size"] and np.isfinite(fraction)
                        and abs(fraction - size / identity["training_size"]) < 1e-12):
                    raise ValueError(
                        f"Invalid training-size fraction: {identity['model_name']}"
                    )
    return missing_mi_grid(
        path, identity["model_name"], identity["Scenario"], identity["seed"],
        identity["rate"], in_sizes, bins,
    )


def _append_extraction_best_row(csv_path, row):
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("x", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=EXTRACTION_BEST_CSV_COLUMNS).writeheader()
    with path.open("a", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=EXTRACTION_BEST_CSV_COLUMNS).writerow(row)


def main_extraction_best(
        seed, yaml_file_path,
        in_sizes=None, num_intervals_list=None, record_verbose=None,
        master_csv_path=None, verbose_dir=None, subset_seed=None,
        model_dir=None, in_size_rates=None, index_dir=None, rate=None,
        skip_existing=True,
):
    """Record best_epoch.pth In/Out MI for ONE extraction substitute model.

    Defaults are the editable EXTRACTION_BEST_* globals. The probe sets are
    the victim's own nested subsets (group_A seed 42, subset seed 42), so the
    points are directly comparable to the negative pool, the victim and the
    FT positives. Each missing (in_size, bins) cell is appended once; matching
    Out MI already in the CSV is reused by bin count. `epoch=best` does not
    invent an unknown epoch number. Returns the number of appended rows.
    This is a single-writer CSV.
    """
    exp_yaml = process_yaml_file(yaml_file_path)
    scenario = exp_yaml["Scenario_Name"]
    victim_cfg = exp_yaml["Victim"]
    victim_ds_cfg = victim_cfg["Dataset"]

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rate = EXTRACTION_BEST_RATE if rate is None else rate
    if not np.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("rate must be a finite fraction in [0, 1]")
    rate = round(float(rate), 2)

    substitute_model = exp_yaml["Substitute"].get(
        "Model", victim_cfg.get("Model", "ResNet-18")
    )
    if not isinstance(substitute_model, dict) and \
            substitute_model not in EXTRACTION_BEST_SUBSTITUTES:
        raise ValueError(
            f"Substitute.Model {substitute_model!r} is not one of "
            f"{EXTRACTION_BEST_SUBSTITUTES}, and is not a dict-declared "
            "timm model either"
        )
    # Family is a property of the SUBSTITUTE -- that is the model whose logits
    # the MI is read from. A DeiT surrogate stolen from a CNN victim is 'deit'.
    family = model_cfg_family(substitute_model)

    training_size = victim_ds_cfg["group_size"]
    sizes_from_rates(training_size, [1.0])   # validates the denominator
    if in_sizes is not None and in_size_rates is not None:
        raise ValueError("Choose in_sizes or in_size_rates, not both")
    if in_sizes is None:
        in_sizes = sizes_from_rates(
            training_size,
            EXTRACTION_BEST_IN_SIZE_RATES if in_size_rates is None else in_size_rates,
        )
    in_sizes = positive_ints(in_sizes, "in_sizes")
    if max(in_sizes) > training_size:
        raise ValueError("in_sizes cannot exceed the victim training group_size")
    bins = positive_ints(
        EXTRACTION_BEST_BINS if num_intervals_list is None else num_intervals_list, "bins"
    )

    model_dir = Path(EXTRACTION_BEST_MODEL_DIR if model_dir is None else model_dir)
    csv_path = Path(EXTRACTION_BEST_MASTER_CSV if master_csv_path is None else master_csv_path)
    verbose_dir = Path(EXTRACTION_BEST_VERBOSE_DIR if verbose_dir is None else verbose_dir)
    index_dir = Path(EXTRACTION_BEST_INDEX_DIR if index_dir is None else index_dir)
    subset_seed = EXTRACTION_BEST_SUBSET_SEED if subset_seed is None else subset_seed
    record_verbose = (EXTRACTION_BEST_RECORD_VERBOSE if record_verbose is None
                      else record_verbose)

    model_name = f"{scenario}_{seed}_{rate}"
    checkpoint = model_dir / model_name / "best_epoch.pth"
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"Missing/empty best checkpoint: {checkpoint}")

    identity = dict(
        Scenario=scenario, seed=seed, rate=rate, model_name=model_name, epoch="best",
        attack=_extraction_attack_name(exp_yaml),
        victim_model=model_cfg_name(victim_cfg.get("Model", "")),
        substitute_model=model_cfg_name(substitute_model),
        aux_dataset=_extraction_aux_dataset(exp_yaml),
        training_size=training_size, family=family, subset_seed=subset_seed,
        group_seed=EXTRACTION_BEST_GROUP_SEED, checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=_extraction_file_sha256(checkpoint),
        plan_sha256=_extraction_file_sha256(yaml_file_path),
    )

    missing, existing_out = _missing_extraction_grid(csv_path, identity, in_sizes, bins)
    if not skip_existing and len(missing) != len(in_sizes) * len(bins):
        raise ValueError("skip_existing=False would duplicate MI rows; use a new CSV")
    if not missing:
        print(f"[SKIP] {model_name}: all requested best-MI cells already exist")
        return 0

    set_seed(seed)
    victim_dataset_obj, victim_num_classes, _ = build_dataset_from_yaml(victim_ds_cfg)
    in_sample_set = victim_dataset_obj.in_sample_set
    out_sample_set = victim_dataset_obj.test_set
    idx_dir = index_dir / victim_ds_cfg["name"]

    group_A = create_or_load_group_A(
        dataset=in_sample_set, save_dir=idx_dir, group_size=training_size,
        num_classes=victim_num_classes, seed=EXTRACTION_BEST_GROUP_SEED,
        force_rebuild=False,
    )
    nested_subsets = create_nested_balanced_subsets_shared(
        dataset=in_sample_set, group_A=group_A, save_dir=idx_dir,
        subset_sizes=in_sizes, num_classes=victim_num_classes,
        seed=subset_seed, force_rebuild=False,
    )
    absent = [s for s in in_sizes if s not in nested_subsets]
    if absent:
        raise ValueError(
            f"nested_subsets_seed{subset_seed}.npz in {idx_dir} has no entry for "
            f"in_size {absent}; available: {sorted(nested_subsets)}"
        )

    out_size = len(out_sample_set)
    needed_bins = sorted({b for _, b in missing})
    for nb in needed_bins:
        if nb in existing_out and existing_out[nb][2] != out_size:
            raise ValueError(f"Cached out_size differs for {model_name}, bins={nb}")

    def loader(data):
        return DataLoader(
            data, batch_size=EXTRACTION_BEST_BATCH_SIZE, shuffle=False,
            num_workers=EXTRACTION_BEST_NUM_WORKERS,
            pin_memory=(str(device).startswith("cuda")), worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(subset_seed),
        )

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if _extraction_file_sha256(checkpoint) != identity["checkpoint_sha256"]:
        raise ValueError(
            f"Checkpoint changed during loading: {checkpoint}; rerun after training finishes"
        )
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net = build_model_any(substitute_model, victim_num_classes).to(device)
    net.load_state_dict(state, strict=True)
    del state
    net.eval()
    print(f"[BEST] {model_name}: {len(missing)} missing MI cells on {device}")

    def measure(logits, labels, nb, split, size=None):
        result = mi_from_logits(logits, labels, num_intervals=nb, verbose=record_verbose)
        if not np.isfinite(result[:2]).all():
            raise ValueError(f"Nonfinite MI: {model_name}, {split}, {size}, bins={nb}")
        if record_verbose:
            save_verbose_data(result[2], verbose_dir, model_name, split, nb, in_size=size)
        return result[:2]

    written = 0
    try:
        out_results = {b: existing_out[b][:2] for b in needed_bins if b in existing_out}
        new_out_bins = [b for b in needed_bins if b not in existing_out]
        if new_out_bins:
            out_logits, out_labels = collect_logits(net, loader(out_sample_set), device)
            for nb in new_out_bins:
                out_results[nb] = measure(out_logits, out_labels, nb, "out")
            del out_logits, out_labels
        for size in sorted({s for s, _ in missing}):
            probe = victim_dataset_obj.subset(
                "train", nested_subsets[size].tolist(), clean=True,
            )
            logits, labels = collect_logits(net, loader(probe), device)
            for nb in sorted(b for s, b in missing if s == size):
                ixt_in, ity_in = measure(logits, labels, nb, "in", size)
                ixt_out, ity_out = out_results[nb]
                row = dict(
                    identity, bins=nb, in_size=size, out_size=out_size,
                    in_size_rate=format(size / training_size, ".12g"),
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
                row.update({"I(X;T)-In": f"{ixt_in:.6f}", "I(T;Y)-In": f"{ity_in:.6f}",
                            "I(X;T)-Out": f"{ixt_out:.6f}", "I(T;Y)-Out": f"{ity_out:.6f}"})
                _append_extraction_best_row(csv_path, row)
                written += 1
                print(f"  size={size:>5} bins={nb:>3}: In=({ixt_in:.6f}, {ity_in:.6f}) "
                      f"Out=({ixt_out:.6f}, {ity_out:.6f})")
            del logits, labels
    finally:
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"[DONE] Appended {written} best-MI rows: {csv_path}")
    return written


def run_extraction_best_sweep(
        plans=None, seeds=None, master_csv_path=None, model_dir=None, **kwargs
):
    """Evaluate every (plan, seed); prints a per-plan tally. Returns row counts."""
    plans = list(EXTRACTION_BEST_PLANS if plans is None else plans)
    seeds = list(EXTRACTION_BEST_SEEDS if seeds is None else seeds)
    csv_path = Path(EXTRACTION_BEST_MASTER_CSV if master_csv_path is None else master_csv_path)
    print(f"Extraction best-MI sweep: {len(plans)} plan(s) x {len(seeds)} seed(s)")
    print(f"  model_dir : {Path(EXTRACTION_BEST_MODEL_DIR if model_dir is None else model_dir)}")
    print(f"  output    : {csv_path}")
    print(f"  grid      : in_size_rates={EXTRACTION_BEST_IN_SIZE_RATES}  "
          f"bins={EXTRACTION_BEST_BINS}")
    tally, absent = {}, {}
    for plan in plans:
        total = 0
        for seed in seeds:
            # Seed coverage differs per cell (the RN18 row has 0..4, the rest
            # 0..2). A checkpoint that was never trained is reported and
            # skipped, not fatal -- otherwise one short cell would abort the
            # whole sweep and the later plans would never be reached.
            try:
                total += main_extraction_best(
                    seed, plan, master_csv_path=csv_path, model_dir=model_dir, **kwargs
                )
            except FileNotFoundError as exc:
                absent.setdefault(str(plan), []).append(seed)
                print(f"[ABSENT] seed={seed} {Path(plan).name}: {exc}")
        tally[str(plan)] = total
    print("\nExtraction best-MI sweep finished:")
    for plan, total in tally.items():
        miss = absent.get(plan)
        note = f"   (no checkpoint for seeds {miss})" if miss else ""
        print(f"  appended {total:>4} rows  {Path(plan).name}{note}")
    print(f"  -> {csv_path}")
    return tally


# =====================================================
# 4. Entry point
# =====================================================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    # New default: best_epoch MI for the extraction_final models on the
    # negative-pool-0.0 grid. Optional arguments filter EXTRACTION_BEST_PLANS
    # by yaml basename substring, e.g. "KNOCKOFF_Same10_Cross16".
    import sys

    selectors = sys.argv[1:]
    chosen = [p for p in EXTRACTION_BEST_PLANS
              if not selectors or any(s in Path(p).name for s in selectors)]
    if not chosen:
        raise SystemExit(
            f"No extraction plan matches {selectors}; available: "
            f"{[Path(p).name for p in EXTRACTION_BEST_PLANS]}"
        )
    run_extraction_best_sweep(plans=chosen)

    """ Previous entry (main_extraction_mi_check on extraction_vanilla,
    absolute in_sizes, thin schema) -- kept verbatim for reference.
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Folder containing all YAML experiment plans
    exp_dir = "./saved_exp_plan/extraction_plan"
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
    """
