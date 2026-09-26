import copy
import os
import glob
from datetime import datetime
import time

from Model.MLP import MNIST_MLP
from Model.ResNet_18 import ResNet18
from Model.VGG16 import ModifiedVGG16
from util_adv import NormalizedModel, collect_checkpoints, parse_attack_configs
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # needed for full CUDA determinism
from pathlib import Path
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import csv
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from util import build_dataset_from_yaml, load_last_checkpoint, process_yaml_file, process_experiment_setup, create_train_subset, evaluate2, prepare_group_subset, \
                 process_experiment_setup_deit, create_or_load_group_A, create_class_balanced_mix_train_test, MixedSplitDataset, \
                 load_best_checkpoint, create_or_load_subset_from_group

# =====================================================
# 1. Global setup
# =====================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_worker(worker_id):
    # Worker gets a different, but deterministic, seed derived from the main seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # all GPUs

    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
        # Raises error if a non-deterministic op is used; great for debugging reproducibility
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
    

def find_epoch_checkpoints(model_dir):
    """
    Find all epoch checkpoints in a directory.
    Returns list of (epoch_number, checkpoint_path) sorted by epoch.
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        print(f"  [SKIP] Directory not found: {model_dir}")
        return []

    checkpoints = []
    # Match common patterns: epoch_10.pth, checkpoint_10.pt, model_epoch10.pth, etc.
    for pattern in ['epoch_*.pth', 'epoch_*.pt', 'checkpoint_*.pth', 'checkpoint_*.pt',
                    'model_epoch*.pth', 'model_*.pth']:
        for ckpt in model_dir.glob(pattern):
            # Extract epoch number from filename
            stem = ckpt.stem
            # Try to find a number in the filename
            nums = [int(s) for s in stem.replace('_', ' ').replace('-', ' ').split() if s.isdigit()]
            if nums:
                checkpoints.append((nums[-1], ckpt))

    # Also include best checkpoint if it exists
    best, _ = load_best_checkpoint(model_dir)
    if best is not None:
        checkpoints.append(('best', best))

    # Sort by epoch number (put 'best' at the end)
    checkpoints.sort(key=lambda x: (isinstance(x[0], str), x[0]))
    return checkpoints


# ===========================================================================
# Subset generation: nested class-balanced subsets from group_A
# (unchanged from previous version)
# ===========================================================================

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
    # 如果是 NormalizedModel 这样的 wrapper，先穿透到真正的 backbone
    if hasattr(net, "base_model"):
        net = net.base_model
    if hasattr(net, "fc"):
        return net.fc.out_features
    if hasattr(net, "classifier"):
        last_linear = [m for m in net.classifier.modules() if isinstance(m, nn.Linear)][-1]
        return last_linear.out_features
    if hasattr(net, "num_classes"):
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


import time
import numpy as np
import torch


def mi_from_logits(layer_T, label_matrix, num_intervals=50, verbose=False):
    """
    Compute (I(X;T), I(T;Y)) from cached logits at a given bin count.

    This function does NOT run the model. It only performs:
        logits -> softmax -> binning -> pattern counting -> entropy / MI.

    Args:
        layer_T:
            Tensor of shape [N, K], usually cached logits.
        label_matrix:
            One-hot label matrix of shape [N, K_y].
        num_intervals:
            Number of bins used to discretize softmax outputs.
        verbose:
            If False, return only (I_X_T, I_T_Y).
            If True, also return debug_data.

    Returns:
        If verbose=False:
            I_X_T, I_T_Y

        If verbose=True:
            I_X_T, I_T_Y, debug_data
    """

    start_time = time.time()

    device = layer_T.device
    N = layer_T.shape[0]
    K = layer_T.shape[1]
    K_y = label_matrix.shape[1]

    # Make sure label_matrix is on the same device and float type.
    label_matrix = label_matrix.to(device=device, dtype=torch.float32)

    # ---------------------------------------------------------
    # 1. Softmax output T
    # ---------------------------------------------------------
    T_soft = torch.softmax(layer_T, dim=1)

    # ---------------------------------------------------------
    # 2. Discretize softmax output
    # ---------------------------------------------------------
    bins = torch.linspace(
        0.0,
        1.0,
        num_intervals + 1,
        device=device,
        dtype=torch.float32
    )

    T_discrete = torch.bucketize(T_soft, bins, right=True) - 1
    T_discrete = T_discrete.clamp(0, num_intervals - 1).contiguous()

    # ---------------------------------------------------------
    # 3. Count unique T patterns
    # ---------------------------------------------------------
    unique_T, inverse_idx = torch.unique(
        T_discrete,
        dim=0,
        return_inverse=True
    )

    K_unique = unique_T.shape[0]

    T_counts = torch.zeros(
        K_unique,
        device=device,
        dtype=torch.float32
    )

    T_counts.index_add_(
        0,
        inverse_idx,
        torch.ones(N, device=device, dtype=torch.float32)
    )

    p_T = T_counts / N

    # ---------------------------------------------------------
    # 4. I(X;T)
    # Since T is a deterministic function of X after forward + binning:
    #     I(X;T) = H(T)
    # ---------------------------------------------------------
    mask_T = p_T > 0
    H_T = -(p_T[mask_T] * torch.log2(p_T[mask_T])).sum()
    I_X_T = H_T

    # ---------------------------------------------------------
    # 5. Joint distribution P(T,Y)
    # ---------------------------------------------------------
    TY_counts = torch.zeros(
        (K_unique, K_y),
        device=device,
        dtype=torch.float32
    )

    TY_counts.index_add_(0, inverse_idx, label_matrix)

    TY_matrix = TY_counts / N
    P_T_marg = TY_matrix.sum(dim=1)
    P_Y_marg = TY_matrix.sum(dim=0)

    # ---------------------------------------------------------
    # 6. I(T;Y)
    # This assumes your existing MI_formula_cal() is already defined.
    # ---------------------------------------------------------
    I_T_Y = MI_formula_cal(TY_matrix, P_T_marg, P_Y_marg)

    if not verbose:
        return I_X_T.item(), I_T_Y.item()

    # ---------------------------------------------------------
    # 7. Extra diagnostic statistics
    # ---------------------------------------------------------
    eps = 1e-12

    # Softmax confidence: max probability per sample.
    confidence = T_soft.max(dim=1).values

    # Softmax entropy per sample:
    #     H(p(y|x)) = - sum_c p_c log2 p_c
    soft_entropy = -(
        T_soft * torch.log2(T_soft.clamp_min(eps))
    ).sum(dim=1)

    # Top-1 / Top-2 softmax margin.
    if K >= 2:
        top2_values = torch.topk(T_soft, k=2, dim=1).values
        top1_top2_margin = top2_values[:, 0] - top2_values[:, 1]
    else:
        top1_top2_margin = torch.zeros(N, device=device, dtype=torch.float32)

    # Label entropy H(Y)
    mask_Y = P_Y_marg > 0
    H_Y = -(
        P_Y_marg[mask_Y] * torch.log2(P_Y_marg[mask_Y])
    ).sum()

    # Conditional entropy H(Y|T)
    # Since I(T;Y) = H(Y) - H(Y|T),
    #     H(Y|T) = H(Y) - I(T;Y)
    H_Y_given_T = H_Y - I_T_Y

    # Pattern size
    pattern_size = T_counts

    # Pattern purity:
    # For each T pattern, purity = majority-label-count / pattern-size.
    pattern_purity = (
        TY_counts.max(dim=1).values / pattern_size.clamp_min(1.0)
    )

    pattern_purity_mean = pattern_purity.mean()

    # Weighted pattern purity:
    # Gives larger patterns more influence.
    pattern_purity_weighted = (
        pattern_purity * pattern_size / float(N)
    ).sum()

    pattern_purity_min = pattern_purity.min()
    pattern_purity_max = pattern_purity.max()

    # Singleton patterns:
    # Patterns that contain exactly one sample.
    singleton_mask = T_counts == 1
    singleton_count = singleton_mask.sum()

    singleton_pattern_ratio = singleton_count.float() / float(K_unique)

    singleton_sample_ratio = (
        T_counts[singleton_mask].sum() / float(N)
        if singleton_count.item() > 0
        else torch.tensor(0.0, device=device)
    )

    # Collision rate:
    # 0 means every sample has a unique T pattern.
    # Larger means more samples collide into shared patterns.
    collision_rate = 1.0 - (float(K_unique) / float(N))

    # Pattern count distribution summary.
    pattern_count_min = T_counts.min()
    pattern_count_max = T_counts.max()
    pattern_count_mean = T_counts.mean()
    pattern_count_std = T_counts.std(unbiased=False)
    pattern_count_median = T_counts.median()

    # Effective number of patterns:
    # exp2(H(T)) gives the effective support size under entropy H(T).
    effective_num_patterns = torch.pow(
        torch.tensor(2.0, device=device),
        H_T
    )

    # Prediction-level statistics.
    pred_labels = T_soft.argmax(dim=1)

    if label_matrix.shape[1] > 1:
        true_labels = label_matrix.argmax(dim=1)
        clean_acc_from_logits = (
            pred_labels == true_labels
        ).float().mean()
    else:
        clean_acc_from_logits = torch.tensor(float("nan"), device=device)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # ---------------------------------------------------------
    # 8. Debug data
    # ---------------------------------------------------------
    debug_data = {
        # Original saved arrays
        "T_soft": T_soft.detach().cpu().numpy().astype(np.float32),
        "T_discrete": T_discrete.detach().cpu().numpy().astype(np.int16),
        "unique_T": unique_T.detach().cpu().numpy().astype(np.int16),
        "inverse_idx": inverse_idx.detach().cpu().numpy().astype(np.int32),
        "pattern_counts": T_counts.detach().cpu().numpy().astype(np.int64),
        "label_counts": TY_counts.detach().cpu().numpy().astype(np.int64),

        # Basic metadata
        "N": np.int64(N),
        "K": np.int64(K),
        "K_y": np.int64(K_y),
        "U": np.int64(K_unique),
        "num_intervals": np.int64(num_intervals),

        # Main MI values
        "I_X_T": np.float64(I_X_T.item()),
        "I_T_Y": np.float64(I_T_Y.item()),

        # Entropy decomposition
        "H_T": np.float64(H_T.item()),
        "H_Y": np.float64(H_Y.item()),
        "H_Y_given_T": np.float64(H_Y_given_T.item()),
        "effective_num_patterns": np.float64(effective_num_patterns.item()),

        # Softmax confidence statistics
        "confidence_mean": np.float64(confidence.mean().item()),
        "confidence_std": np.float64(confidence.std(unbiased=False).item()),
        "confidence_min": np.float64(confidence.min().item()),
        "confidence_max": np.float64(confidence.max().item()),

        # Softmax entropy statistics
        "soft_entropy_mean": np.float64(soft_entropy.mean().item()),
        "soft_entropy_std": np.float64(soft_entropy.std(unbiased=False).item()),
        "soft_entropy_min": np.float64(soft_entropy.min().item()),
        "soft_entropy_max": np.float64(soft_entropy.max().item()),

        # Top-1 / top-2 margin statistics
        "top1_top2_margin_mean": np.float64(top1_top2_margin.mean().item()),
        "top1_top2_margin_std": np.float64(top1_top2_margin.std(unbiased=False).item()),
        "top1_top2_margin_min": np.float64(top1_top2_margin.min().item()),
        "top1_top2_margin_max": np.float64(top1_top2_margin.max().item()),

        # Pattern purity statistics
        "pattern_purity_mean": np.float64(pattern_purity_mean.item()),
        "pattern_purity_weighted": np.float64(pattern_purity_weighted.item()),
        "pattern_purity_min": np.float64(pattern_purity_min.item()),
        "pattern_purity_max": np.float64(pattern_purity_max.item()),

        # Pattern collision / singleton statistics
        "singleton_count": np.int64(singleton_count.item()),
        "singleton_pattern_ratio": np.float64(singleton_pattern_ratio.item()),
        "singleton_sample_ratio": np.float64(singleton_sample_ratio.item()),
        "collision_rate": np.float64(collision_rate),

        # Pattern count summary
        "pattern_count_min": np.float64(pattern_count_min.item()),
        "pattern_count_max": np.float64(pattern_count_max.item()),
        "pattern_count_mean": np.float64(pattern_count_mean.item()),
        "pattern_count_std": np.float64(pattern_count_std.item()),
        "pattern_count_median": np.float64(pattern_count_median.item()),

        # Prediction statistic from cached logits
        "clean_acc_from_logits": np.float64(clean_acc_from_logits.item()),

        # Timing
        "mi_elapsed_time": np.float64(elapsed_time),
    }

    print(f"MI cal costs: {elapsed_time:.4f}s")
    print(
        "[MI-debug] "
        f"N={N}, U={K_unique}, "
        f"I(X;T)={I_X_T.item():.6f}, "
        f"I(T;Y)={I_T_Y.item():.6f}, "
        f"H(Y|T)={H_Y_given_T.item():.6f}, "
        f"conf={confidence.mean().item():.4f}, "
        f"entropy={soft_entropy.mean().item():.4f}, "
        f"margin={top1_top2_margin.mean().item():.4f}, "
        f"purity_w={pattern_purity_weighted.item():.4f}, "
        f"singleton={singleton_sample_ratio.item():.4f}, "
        f"collision={collision_rate:.4f}"
    )

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


def _extract_epoch_from_ckpt(ckpt_name):
    """
    Parse the epoch integer from a checkpoint filename of the form
    'epoch_N.pth' (or any name whose stem ends with '_N').
 
    Returns the integer N. Assumes the 'epoch_N.pth' convention is enforced
    upstream by the training pipeline.
    """
    stem = Path(ckpt_name).stem            # 'epoch_50.pth' -> 'epoch_50'
    return int(stem.split("_")[-1])        # 'epoch_50'     -> 50


# =====================================================
# 3. Main training logic
# =====================================================
def main_at_pos(seed, yaml_file_path):
    """
    Calculate MI metrics for adversarially trained (positive suspect) models.
    Reads YAML in the same format as the at_eval pipeline:
      - Victim.Dataset       defines the dataset
      - Positive.Model       defines the architecture
      - Positive.Model_Path  lists checkpoint directories under saved_models/
      - Positive.State       'best' | 'last' | 'all' selects which checkpoint(s)
    """
    print(device)

    exp_yaml = process_yaml_file(yaml_file_path)
    scenario_root = exp_yaml.get("Scenario_Name", Path(yaml_file_path).stem)

    victim_ds_cfg = exp_yaml["Victim"]["Dataset"]
    dataset_obj, num_classes, _ = build_dataset_from_yaml(victim_ds_cfg)

    positive_cfg = exp_yaml.get("Positive", {})
    model_arch   = positive_cfg.get("Model", "ResNet-18")
    eval_mode    = positive_cfg.get("State", "best")
    model_paths  = positive_cfg.get("Model_Path", [])

    if not model_paths:
        print("No positive model paths found in YAML.")
        return

    # ---- Prepare data loaders (independent of attack / cs_chunks) -------
    g = torch.Generator()
    g.manual_seed(seed)

    in_sample_set  = dataset_obj.in_sample_set
    out_sample_set = dataset_obj.test_set

    group_A = create_or_load_group_A(
        dataset=in_sample_set,
        save_dir=f'./Indices/{victim_ds_cfg["name"]}/',
        group_size=victim_ds_cfg["group_size"],
        num_classes=num_classes,
        seed=42,
        force_rebuild=False,
    )
    in_sample_subset = dataset_obj.subset("train", group_A, clean=True)

    in_sample_loader = DataLoader(
        in_sample_subset, batch_size=128, shuffle=False, num_workers=8,
        worker_init_fn=seed_worker, generator=g,
        persistent_workers=True, pin_memory=True,
    )
    out_sample_loader = DataLoader(
        out_sample_set, batch_size=128, shuffle=False, num_workers=8,
        worker_init_fn=seed_worker, generator=g,
        persistent_workers=True, pin_memory=True,
    )

    # ---- Logging ---------------------------------------------------------
    log_dir = "./saved_logs/at_eval/MI/Positive"
    os.makedirs(log_dir, exist_ok=True)

    # ---- Iterate over each model_path × selected checkpoints -------------
    base_dir = Path("./saved_models")

    for dir_name in model_paths:
        scenario_name = dir_name.split("/")[-1]
        print(f"\n--- MI Evaluation: {scenario_name} ---")

        model_dir = base_dir / dir_name.lstrip("/")

        checkpoints = collect_checkpoints(model_dir, eval_mode)
        if not checkpoints:
            print(f"  [SKIP] No checkpoints found in {model_dir}")
            continue

        # One CSV per scenario (matches your existing convention)
        log_file = os.path.join(log_dir, f"training_log_{scenario_name}_MI.csv")
        if not os.path.exists(log_file):
            with open(log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Scenario", "Checkpoint",
                    "I(X;T)-InAug", "I(T;Y)-InAug",
                    "I(X;T)-In",    "I(T;Y)-In",
                    "I(X;T)-Out",   "I(T;Y)-Out",
                ])

        for ckpt_name, ckpt_path in checkpoints:
            print(f"  Evaluating: {ckpt_name}  ({ckpt_path.name})")

            net = build_model(model_arch, num_classes).to(device)
            state = torch.load(ckpt_path, map_location=device)

            # Strip NormalizedModel wrapping if present.
            if any(k.startswith("base_model.") for k in state.keys()):
                state = {
                    k.replace("base_model.", ""): v
                    for k, v in state.items()
                    if k not in ("mean", "std")
                }

            net.load_state_dict(state)
            net.eval()

            value_xt_in, value_ty_in = evaluate2(net, in_sample_loader, device)
            # value_xt_out, value_ty_out = evaluate2(net, out_sample_loader, device)

            print(f"    I(X;T)-In: {value_xt_in:.4f}, I(T;Y)-In: {value_ty_in:.4f}")

            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    scenario_name, ckpt_name,
                    0, 0,
                    value_xt_in, value_ty_in,
                    0, 0,
                ])

            del net
            torch.cuda.empty_cache()

        print(f"  Logged to: {log_file}")

    print(f"\n  MI evaluation complete.")


def main_at_neg(
    seed, yaml_file_path,
    in_sizes=None,
    num_intervals_list=None,
    record_verbose=True,
    master_csv_path="./saved_logs/at_vanilla/MI_master_table_at_neg1.csv",
    subset_seed=42,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Calculate MI metrics for adversarially fine-tuned negative models.
    """
    if in_sizes is None:
        in_sizes = [5000, 10000, 15000, 20000, 25000]
    if num_intervals_list is None:
        num_intervals_list = [50, 75, 100, 125, 150]
    
    print(f"Device: {device}")

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_setup(exp_yaml)
    
    # ---------- Prepare dataset ----------
    print('==> Preparing data..')
    # Note: Using your standard build_dataset_from_yaml to get dataset_obj
    dataset_obj, num_classes, _ = build_dataset_from_yaml(exp_yaml["Dataset"])

    in_sample_set = dataset_obj.raw_train_clean_set
    out_sample_set = dataset_obj.raw_test_set

    # Create fixed indices for MI calculation evaluation
    group_A = create_or_load_group_A(
        dataset=in_sample_set,
        save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
        group_size=exp_yaml.get("GroupSize", 25000), # Default fallback if missing
        num_classes=num_classes,
        seed=42, force_rebuild=False,
    )
    nested_subsets = create_nested_balanced_subsets(
        dataset=in_sample_set, group_A=group_A,
        save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
        subset_sizes=in_sizes,
        num_classes=num_classes,
        seed=subset_seed, force_rebuild=False,
    )

    out_size = len(out_sample_set)
    out_sample_loader = DataLoader(
        out_sample_set, batch_size=128, shuffle=False, num_workers=0,
        pin_memory=True,
    )

    print('==> Building model..')
    # load experiment information
    init_model = exp_setup["Model"]

    # ---------- Prepare Attack Configurations ----------
    attack_configs = parse_attack_configs(exp_yaml)
    print(f"Found {len(attack_configs)} attack config(s)")

    model_path_ls = exp_yaml.get("Model_Path", [])
    
    ensure_master_csv(master_csv_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---------- Iterate over Models and Attacks ----------
    for model_path in model_path_ls:
        base_name = model_path.split('/')[-1]
        
        for attack_fn, attack_name, attack_kwargs in attack_configs:
            print(f"\n{'='*60}")
            print(f"Evaluating MI for Attack: {attack_name}, {attack_kwargs}")
            print(f"Base Model: {base_name}")
            print(f"{'='*60}")

            param_str = "_".join(f"{k}={v}" for k, v in sorted(attack_kwargs.items()))
            
            # --- CRITICAL CHANGE: Match the new scenario_name logic exactly ---
            scenario_name = base_name + f"_{attack_name}_{param_str}_ftseed={seed}"

            model_dir = './saved_models/at_vanilla/'
            model_folder = Path(model_dir) / scenario_name
            ckpt_path, filename = load_best_checkpoint(model_folder, filename="best_clean_epoch.pth")
            if ckpt_path is None:
                print(f"[⚠️] No .pth files found in {model_folder}")
                return
            
            # --- Load the model architecture and state ---
            print(f"==> Building and loading model from {ckpt_path}..")
            net = copy.deepcopy(init_model).to(device)
            state = torch.load(ckpt_path, map_location=device)
            # Strip NormalizedModel wrapping if present.
            if any(k.startswith("base_model.") for k in state.keys()):
                state = {
                    k.replace("base_model.", ""): v
                    for k, v in state.items()
                    if k not in ("mean", "std")
                }
            net.load_state_dict(state)   # bare backbone keys -> backbone, not the wrapper
            net = NormalizedModel(net, exp_setup["Dataset"].mean, exp_setup["Dataset"].std).to(device)

            # =============================================================
            # PHASE 1: Inference (the expensive part) -- ONCE per data set
            # =============================================================
            print(f"\n==> [Phase 1] Running inference once per loader")

            # Out-sample: infer once, reuse for all bin counts
            print(f"  Inferring out-sample (size={out_size})...")
            out_layer_T, out_labels = collect_logits(net, out_sample_loader, device)
            print(f"    -> cached logits shape: {tuple(out_layer_T.shape)}")

            ## In-sample: infer once per in_size, reuse for all bin counts
            in_cache = {}   # in_size -> (layer_T, label_matrix)
            for in_size in in_sizes:
                subset_indices = nested_subsets[in_size].tolist()
                in_sample_subset = exp_setup["Dataset"].subset(
                    "raw_train_clean", subset_indices, # this should be unnormalized form
                )
                loader = DataLoader(
                    in_sample_subset, batch_size=128, shuffle=False, 
                    pin_memory=True,
                )
                print(f"  Inferring in-sample (size={in_size})...")
                layer_T, labels = collect_logits(net, loader, device)
                in_cache[in_size] = (layer_T, labels)
                print(f"    -> cached logits shape: {tuple(layer_T.shape)}")

            # =============================================================
            # PHASE 2: MI computation (the cheap part) -- bin sweep on cached logits
            # =============================================================
            print(f"\n==> [Phase 2] Computing MI on cached logits")

            # Out-sample MI for each bin count
            out_results = {}
            for nb in num_intervals_list:
                ixt_out, ity_out = mi_from_logits(
                    out_layer_T, out_labels, num_intervals=nb, verbose=False,
                )
                print(f"  out  bins={nb:>3}: I(X;T)={ixt_out:.4f}, I(T;Y)={ity_out:.4f}")
                out_results[nb] = (ixt_out, ity_out)

            # In-sample MI for each (in_size, bin) combination, plus CSV row writing
            # Extract overlap rate for logging
            overlap_rate_str = base_name.split('_')[-1]
            try:
                rate_val = round(float(overlap_rate_str), 2)
            except ValueError:
                rate_val = -1.0 # fallback

            for in_size in in_sizes:
                layer_T, labels = in_cache[in_size]
                for nb in num_intervals_list:
                    ixt_in, ity_in = mi_from_logits(
                        layer_T, labels, num_intervals=nb, verbose=False,
                    )
                    print(f"  in   size={in_size:>5}  bins={nb:>3}: I(X;T)={ixt_in:.4f}, I(T;Y)={ity_in:.4f}")

                    ixt_out, ity_out = out_results[nb]

                    row = {
                        "Scenario":    base_name,
                        "seed":        seed,
                        "rate":        rate_val, 
                        "model_name":  scenario_name,
                        "epoch":       filename.split(".")[0],
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

            # Free the cached logits for this specific model+attack combo
            del in_cache, out_layer_T, out_labels
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n==> Done. Master CSV: {master_csv_path}")


def main_at_neg_scratch(
    seed, r, yaml_file_path,
    in_sizes=None,
    num_intervals_list=None,
    record_verbose=True,
    master_csv_path="./saved_logs/at_vanilla/MI_master_table_at_neg1.csv",
    verbose_dir="./saved_logs/vanilla/MI_verbose",
    subset_seed=42,
):
    """
    Calculate MI metrics for adversarially trained (negative suspect) models.
    Loads only the best_epoch checkpoint for each model × attack combo.
    Writes one CSV per scenario.
    """
    if in_sizes is None:
        in_sizes = [5000, 10000, 15000, 20000, 25000]
    if num_intervals_list is None:
        num_intervals_list = [50, 75, 100, 125, 150]
    
    print(f"Device: {device}")

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_setup(exp_yaml)
    base_name = exp_yaml["Scenario_Name"] + f"_{seed}_{round(r, 2)}"

    print('==> Preparing data..')
    in_sample_set = exp_setup["Dataset"].raw_train_clean_set # it fits for the normalized model wrapper
    out_sample_set = exp_setup["Dataset"].raw_test_set

    group_A = create_or_load_group_A(
        dataset=in_sample_set,
        save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
        group_size=exp_setup["GroupSize"],
        num_classes=exp_setup["NumClasses"],
        seed=42, force_rebuild=False,
    )
    nested_subsets = create_nested_balanced_subsets(
        dataset=in_sample_set, group_A=group_A,
        save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
        subset_sizes=in_sizes,
        num_classes=exp_setup["NumClasses"],
        seed=subset_seed, force_rebuild=False,
    )

    out_size = len(out_sample_set)
    out_sample_loader = DataLoader(
        out_sample_set, batch_size=128, shuffle=False, num_workers=0,
        pin_memory=True,
    )

    print('==> Building model..')
    # load experiment information
    init_model = exp_setup["Model"]
    #init_model = NormalizedModel(init_model, exp_setup["Dataset"].mean, exp_setup["Dataset"].std).to(device)

    # ---------- Prepare Attack Configurations (must match main_at_neg) ----------
    attack_configs = parse_attack_configs(exp_yaml)
    print(f"Found {len(attack_configs)} attack config(s)")

    ensure_master_csv(master_csv_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for attack_fn, attack_name, attack_kwargs in attack_configs:
        print(f"\n{'='*60}")
        print(f"Attack: {attack_name}, {attack_kwargs}")
        print(f"Model:  {base_name}")
        print(f"{'='*60}")

        param_str = "_".join(f"{k}={v}" for k, v in sorted(attack_kwargs.items()))
        scenario_name = base_name + f"_{attack_name}_{param_str}_atseed={seed}"

        model_dir = './saved_models/at_vanilla/'
        model_folder = Path(model_dir) / scenario_name
        ckpt_path, filename = load_best_checkpoint(model_folder, filename="best_clean_epoch.pth")
        if ckpt_path is None:
            print(f"[⚠️] No .pth files found in {model_folder}")
            return
        
        net = copy.deepcopy(init_model).to(device)
        state = torch.load(ckpt_path, map_location=device)
        # Strip NormalizedModel wrapping if present.
        if any(k.startswith("base_model.") for k in state.keys()):
            state = {
                k.replace("base_model.", ""): v
                for k, v in state.items()
                if k not in ("mean", "std")
            }
        net.load_state_dict(state)   # bare backbone keys -> backbone, not the wrapper

        net = NormalizedModel(net, exp_setup["Dataset"].mean, exp_setup["Dataset"].std).to(device)

        # =============================================================
        # PHASE 1: Inference (the expensive part) -- ONCE per data set
        # =============================================================
        print(f"\n==> [Phase 1] Running inference once per loader")

        # Out-sample: infer once, reuse for all bin counts
        print(f"  Inferring out-sample (size={out_size})...")
        out_layer_T, out_labels = collect_logits(net, out_sample_loader, device)
        print(f"    -> cached logits shape: {tuple(out_layer_T.shape)}")

        # In-sample: infer once per in_size, reuse for all bin counts
        in_cache = {}   # in_size -> (layer_T, label_matrix)
        for in_size in in_sizes:
            subset_indices = nested_subsets[in_size].tolist()
            in_sample_subset = exp_setup["Dataset"].subset(
                "raw_train_clean", subset_indices, # this should be unnormalized form
            )
            loader = DataLoader(
                in_sample_subset, batch_size=128, shuffle=False, 
                pin_memory=True,
            )
            print(f"  Inferring in-sample (size={in_size})...")
            layer_T, labels = collect_logits(net, loader, device)
            in_cache[in_size] = (layer_T, labels)
            print(f"    -> cached logits shape: {tuple(layer_T.shape)}")

        # =============================================================
        # PHASE 2: MI computation (the cheap part) -- bin sweep on cached logits
        # =============================================================
        print(f"\n==> [Phase 2] Computing MI on cached logits")

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
                    "rate":        round(r, 2),
                    "model_name":  scenario_name,
                    "epoch":       filename.split(".")[0],
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

    # Free the cached logits to keep GPU memory tidy if many models in sequence
    del in_cache, out_layer_T, out_labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n==> Done. Master CSV: {master_csv_path}")


def main_deit(seed, r, yaml_file_path): # this function is used to calculate the model from main_train_nega.py
    print(device)

    g = torch.Generator()
    g.manual_seed(seed)

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_setup_deit(exp_yaml) 

    print('==> Loading model..')
    model_name = exp_yaml["Scenario_Name"] + f"_{seed}_{r}"
    model_dir = './saved_models/vanilla/'
    model_folder = Path(model_dir)/model_name

    #ckpt_path = load_last_checkpoint(model_folder)
    ckpt_path, _ = load_best_checkpoint(model_folder)
    if ckpt_path is None:
        print(f"[⚠️] No .pth files found in {model_folder}")
    
    net = exp_setup["Student Model"].to(device) # load model
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state)

    print('==> Preparing data..')
    in_sample_set = exp_setup["Dataset"].in_sample_set # No data augmentation here
    out_sample_set =  exp_setup["Dataset"].test_set
    
    # Important: the in sample evaluation set should be the same each round
    group_A = create_or_load_group_A(dataset=in_sample_set, save_dir=f'./Indices/{exp_yaml['Dataset']['name']}/',
                                                  group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"], seed=42, force_rebuild=False)
    in_sample_subset = exp_setup["Dataset"].subset("train", group_A, clean=True)

    in_sample_loader = DataLoader(in_sample_subset,batch_size=128,shuffle=False,num_workers=8, # in sample without augmentation
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)
    
    out_sample_loader = DataLoader(out_sample_set,batch_size=128,shuffle=False,num_workers=8, # out sample
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)

    log_dir = './saved_logs/vanilla/MI/DeiT/'
    os.makedirs(log_dir, exist_ok=True)
    log_name = model_name
    log_file = os.path.join(log_dir, f"training_log_{log_name}_MI.csv")

    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Scenario','Epoch','I(X;T)-InAug', 'I(T;Y)-InAug', 'I(X;T)-In', 'I(T;Y)-In','I(X;T)-Out', 'I(T;Y)-Out']) 

    # Evaluation loop
    print(f"This is the {seed} round on {r} rate 99 epoch!") # here we only consider calculating the last epoch's metric
    # value_xt_inAug, value_ty_inAug = evaluate2(net, trainloader, device)
    value_xt_in, value_ty_in = evaluate2(net, in_sample_loader, device)
    value_xt_out, value_ty_out = evaluate2(net, out_sample_loader, device)

    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([log_name, 99,
                            0, 0, # default value for non-calculated groups
                            value_xt_in, value_ty_in,
                            value_xt_out, value_ty_out
                            ])
        

                                                    # based on different overlapping rate
    print(device)

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_setup(exp_yaml) 

    print('==> Loading model..')
    model_name = exp_yaml["Scenario_Name"] + f"_{seed}_{1.0}"
    model_dir = './saved_models/vanilla/'
    model_folder = Path(model_dir)/model_name

    # ckpt_path = load_last_checkpoint(model_folder)
    ckpt_path, _ = load_best_checkpoint(model_folder)
    if ckpt_path is None:
        print(f"[⚠️] No .pth files found in {model_folder}")
    
    net = exp_setup["Model"].to(device) # load model
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state)

    print('==> Preparing data..')
    g = torch.Generator()
    g.manual_seed(seed)

    in_sample_set = exp_setup["Dataset"].in_sample_set # No data augmentation here
    out_sample_set =  exp_setup["Dataset"].test_set
    #train_set = exp_setup["Dataset"].train_set
    
    # Important: the in sample evaluation set should be the same each round
    group_A = create_or_load_group_A(dataset=in_sample_set, save_dir=f'./Indices/{exp_yaml['Dataset']['name']}/',
                                                  group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"], seed=42, force_rebuild=False)
    
    mixed, picked_train, picked_test = create_class_balanced_mix_train_test(
                                        train_dataset=in_sample_set,
                                        test_dataset=out_sample_set,
                                        train_in_indices=group_A,
                                        num_classes=exp_setup["NumClasses"],
                                        total_size=10000, # here we temporarily set the total size of mixed probe dataset as 10000
                                        frac_in_from_train=round(r,2),
                                        seed=42,
                                        return_parts=True,
                                    )
    mixed_subset = MixedSplitDataset(in_sample_set, out_sample_set, mixed, return_split=False)
    mixed_sample_loader = DataLoader(mixed_subset,batch_size=128,shuffle=False,num_workers=8, # in sample without augmentation
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)

    log_dir = './saved_logs/vanilla/MI'
    os.makedirs(log_dir, exist_ok=True)
    log_name = model_name + f"_overlap_{round(r,2)}_size_{10000}"
    log_file = os.path.join(log_dir, f"training_log_{log_name}_MI.csv")

    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Scenario','Epoch','I(X;T)-InAug', 'I(T;Y)-InAug', 'I(X;T)-In', 'I(T;Y)-In','I(X;T)-Out', 'I(T;Y)-Out']) 

    # Evaluation loop
    print(f"This is the {seed} round on {r} rate 99 epoch!") # here we only consider calculating the last epoch's metric
    # value_xt_inAug, value_ty_inAug = evaluate2(net, trainloader, device)
    value_xt_in, value_ty_in = evaluate2(net, mixed_sample_loader, device)

    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([log_name, 99,
                            0, 0, # default value for non-calculated groups
                            value_xt_in, value_ty_in,
                            0, 0 # in this case, no need to compute the out-sample MI
                            ])


# ===========================================================================
# NEW (2026-09-18): best-checkpoint MI on the negative-pool-0.0 grid.
#
# Why this path exists
# --------------------
# `main_at_pos` / `main_at_neg` / `main_at_neg_scratch` above are kept
# untouched. They hardcode the model directory, take absolute in_sizes, append
# unconditionally (re-running duplicates rows) and write the thin 13-column
# schema. The negative pool (MI_master_table_neg_pool0.csv), the victim
# (MI_master_table_victim.csv), the FT positives (ft_final/...) and the
# extraction positives (extraction_final/...) are all on the rate-based grid
# POOL0_IN_SIZE_RATES x group_size = {250, 1250, 2500, 5000, 12500, 18750,
# 25000} x POOL0_BINS, with a resume protocol and provenance columns.
#
# `main_at_best` below matches that form for adversarially trained suspects:
#   * grid from POOL0_IN_SIZE_RATES / POOL0_BINS (or explicit in_sizes)
#   * nested subsets from mi_pool_support (the SHARED builder), so the probe
#     sets are byte-identical to the pool / victim / FT / extraction ones. The
#     local builder above is left in place for the old paths but is NOT used
#     here: it rewrites the shared npz down to only the requested sizes on a
#     rebuild, and rejects sizes not divisible by num_classes.
#   * missing_mi_grid resume: exact (in_size, bins) pairs, complete models
#     skipped, partial/mismatched rows raise instead of being overwritten
#   * provenance columns plus the parsed AT identity (base_model, attack, eps,
#     steps, bn, at_size, at_epochs, run_tag, at_seed, mix, ckpt_kind) so a
#     plot spec can filter by variant
#   * a SEPARATE output CSV; the old at_vanilla / at_train_new tables are
#     never written to
#
# Two checkpoints per model
# -------------------------
# AT runs save best_clean_epoch.pth AND best_rob_epoch.pth. Both are measured
# by default. They are distinct models, so each gets its own `model_name`
# (folder name plus "_ckpt=<kind>") -- a shared model_name would collide in the
# (model_name, in_size, bins) resume key. `epoch` carries the kind
# ("best_clean" / "best_rob"), which makes it directly filterable through
# TableGroupSpec's `epochs` field in the notebook.
#
# Normalization
# -------------
# AT checkpoints are NormalizedModel state_dicts: a `base_model.` prefixed
# backbone plus `mean` / `std` buffers, fed raw [0,1] images. The pool / victim
# / FT / extraction paths instead feed normalized images to a bare backbone.
# Verified 2026-09-18 on a real at_final checkpoint: the two conventions give
# bit-identical logits (max |difference| 0.0 on both the 2500-sample in-probe
# and the 10000-sample out-probe), so the MI values share one coordinate
# system. This path keeps the AT convention and checks the checkpoint's own
# mean/std against the dataset's, raising on a mismatch rather than silently
# producing incomparable numbers.
#
# The MI estimator is unchanged: `collect_logits` here is byte-identical to
# MI_check's and this file's `mi_from_logits` is numerically identical to it
# (max difference 0.0 over 60 shape/scale/bin configurations, 2026-09-18).
# `_infer_num_classes` here additionally sees through the NormalizedModel
# wrapper, which MI_check's version cannot.
#
# Usage (from E:\Experiment, pytorch_env)
#   python calculate_MI_at.py                      # every model in at_final
#   python calculate_MI_at.py Knockoff_Same10      # substring filter on folder
# ===========================================================================
import hashlib

from MI_check import POOL0_BINS, POOL0_IN_SIZE_RATES
from mi_pool_support import (
    create_nested_balanced_subsets as create_nested_balanced_subsets_shared,
    missing_mi_grid, positive_ints, sizes_from_rates,
)

AT_BASE_DIR = Path(__file__).resolve().parent
AT_BEST_MODEL_DIR = AT_BASE_DIR / "saved_models/at_final"
AT_BEST_MASTER_CSV = AT_BASE_DIR / "saved_logs/at_final/MI_master_table_at.csv"
AT_BEST_VERBOSE_DIR = AT_BASE_DIR / "saved_logs/at_final/MI_verbose_best"
AT_BEST_INDEX_DIR = AT_BASE_DIR / "Indices"
# The plan that produced at_final; supplies the victim dataset and the arch.
AT_BEST_PLAN = (AT_BASE_DIR
                / "saved_exp_plan/at_plan/CIFAR10_RES18_Same_PGD.yaml")
AT_BEST_CKPT_KINDS = ["best_clean", "best_rob"]
AT_BEST_IN_SIZE_RATES = list(POOL0_IN_SIZE_RATES)
AT_BEST_BINS = list(POOL0_BINS)
AT_BEST_SUBSET_SEED = 42       # same nested subsets as pool / victim / FT
AT_BEST_GROUP_SEED = 42        # group_A seed, fixed upstream
AT_BEST_RATE = 1.0             # rate suffix of the base extraction model
AT_BEST_RECORD_VERBOSE = False
AT_BEST_BATCH_SIZE = 128
AT_BEST_NUM_WORKERS = 0
AT_BEST_NORM_TOL = 1e-6        # checkpoint mean/std vs dataset mean/std

AT_BEST_CSV_COLUMNS = [
    "Scenario", "seed", "rate", "model_name", "epoch", "bins", "in_size",
    "I(X;T)-In", "I(T;Y)-In", "out_size", "I(X;T)-Out", "I(T;Y)-Out",
    "timestamp", "at_scenario", "base_model", "attack", "eps", "steps",
    "bn", "at_size", "at_epochs", "run_tag", "at_seed", "mix", "ckpt_kind",
    "training_size", "in_size_rate", "family", "subset_seed", "group_seed",
    "checkpoint", "checkpoint_sha256", "plan_sha256",
]


def _at_file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# main_at.py writes two kinds of checkpoint into one folder and names them
# differently: the selected ones as <kind>_epoch.pth (best_clean_epoch.pth,
# best_rob_epoch.pth) and the per-epoch trajectory as epoch_<N>.pth. One
# resolver keeps both reachable through the same `ckpt_kinds` argument, so the
# trajectory needs no second copy of main_at_best.
AT_EPOCH_KIND_PREFIX = "epoch_"


def at_ckpt_filename(kind):
    """File for a ckpt_kind: 'best_clean' -> best_clean_epoch.pth, 'epoch_7' -> epoch_7.pth."""
    kind = str(kind)
    return f"{kind}.pth" if kind.startswith(AT_EPOCH_KIND_PREFIX) else f"{kind}_epoch.pth"


def at_ckpt_epoch(kind):
    """Value for the `epoch` column: the integer N for 'epoch_N', the kind itself otherwise.

    Keeping it numeric for the trajectory is what lets a plot sort and join on
    it; 'best_clean' / 'best_rob' are not epochs and stay as they were.
    """
    kind = str(kind)
    if kind.startswith(AT_EPOCH_KIND_PREFIX):
        return int(kind[len(AT_EPOCH_KIND_PREFIX):])
    return kind


def at_epoch_kinds(model_dir, at_scenario):
    """Sorted ['epoch_0', ...] for the epoch_<N>.pth actually present on disk."""
    folder = Path(model_dir) / at_scenario
    if not folder.is_dir():
        raise FileNotFoundError(f"AT scenario folder not found: {folder}")
    found = []
    for path in folder.glob("epoch_*.pth"):
        try:
            found.append(int(path.stem[len(AT_EPOCH_KIND_PREFIX):]))
        except ValueError:
            continue
    return [f"{AT_EPOCH_KIND_PREFIX}{n}" for n in sorted(found)]


def parse_at_scenario_name(name):
    """Split an at_final folder name into its identity fields.

    Inverse of main_at.build_at_pos_scenario_name, which produces
        <base_model>_<attack>_<k=v sorted>_bn=<bn>_atn=<n>_atepochs=<e>
        [_run=<tag>]_atseed=<s>_mix=<mix>
    `Scenario` drops the base model's trailing _<seed>_<rate>, mirroring the
    extraction table where Scenario is the scenario and model_name adds them.
    """
    head, marker, tail = name.partition("_bn=")
    if not marker:
        raise ValueError(f"Not an AT scenario name (no '_bn='): {name}")
    tokens = head.split("_")
    first_kv = next((i for i, t in enumerate(tokens) if "=" in t), None)
    if first_kv is None or first_kv < 2:
        raise ValueError(f"Cannot locate the attack/parameter boundary in: {name}")
    base_model = "_".join(tokens[:first_kv - 1])
    attack = tokens[first_kv - 1]
    attack_params = dict(t.split("=", 1) for t in tokens[first_kv:])

    bn, marker, rest = tail.partition("_atn=")
    if not marker:
        raise ValueError(f"Missing '_atn=' in: {name}")
    rest_kv = {}
    for token in ("atn=" + rest).split("_"):
        if "=" not in token:
            raise ValueError(f"Unexpected token {token!r} in: {name}")
        key, value = token.split("=", 1)
        rest_kv[key] = value
    for required in ("atn", "atepochs", "atseed", "mix"):
        if required not in rest_kv:
            raise ValueError(f"Missing '{required}=' in: {name}")

    base_parts = base_model.split("_")
    if len(base_parts) < 3:
        raise ValueError(f"Base model name too short to carry seed/rate: {base_model}")
    return {
        "at_scenario": name,
        "Scenario": "_".join(base_parts[:-2]),
        "base_model": base_model,
        "base_seed": base_parts[-2],
        "rate": round(float(base_parts[-1]), 2),
        "attack": attack,
        "eps": attack_params.get("eps", ""),
        "steps": attack_params.get("steps", ""),
        "bn": bn,
        "at_size": int(rest_kv["atn"]),
        "at_epochs": int(rest_kv["atepochs"]),
        "run_tag": rest_kv.get("run", ""),
        "at_seed": int(rest_kv["atseed"]),
        "mix": rest_kv["mix"],
    }


def discover_at_models(model_dir=None, selectors=None, ckpt_kinds=None):
    """List (folder_name, parsed_identity) for every AT model on disk.

    at_final is a curated subset of at_evasion, so the inventory comes from the
    directory rather than from a product over the plan's attack grid.
    `selectors` keeps only folders containing one of the given substrings.
    Folders missing any requested checkpoint are reported and skipped.
    """
    model_dir = Path(AT_BEST_MODEL_DIR if model_dir is None else model_dir)
    kinds = list(AT_BEST_CKPT_KINDS if ckpt_kinds is None else ckpt_kinds)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"AT model directory not found: {model_dir}")
    found = []
    for folder in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        if selectors and not any(s in folder.name for s in selectors):
            continue
        missing = [k for k in kinds
                   if not (folder / at_ckpt_filename(k)).is_file()
                   or (folder / at_ckpt_filename(k)).stat().st_size == 0]
        if missing:
            print(f"[SKIP] {folder.name}: missing/empty checkpoint(s) {missing}")
            continue
        found.append((folder.name, parse_at_scenario_name(folder.name)))
    return found



# ---------------------------------------------------------------------------
# Plan-driven model selection (2026-09-18)
#
# calculate_MI_ft.py, calculate_MI_extraction.py and calculate_MI_victim.py all
# take their model set from the plans listed in the script, never from whatever
# happens to sit in saved_models. This section gives calculate_MI_at.py the same
# contract, so dropping new checkpoints into at_final does not silently enlarge
# the next run: point AT_BEST_PLANS at the plan(s) you want measured and only
# those models are computed.
#
# Names are rebuilt with main_at.build_at_pos_scenario_name, the very function
# that named the folders during training, so the two cannot drift apart. Each
# model is then measured with the plan that declared it, which also makes the
# plan_sha256 column correct per model rather than per sweep.
#
# `discover_at_models` above is kept: it is the directory view, useful for
# auditing what is on disk against what the plans declare, and still reachable
# through run_at_best_sweep(from_disk=True).
# ---------------------------------------------------------------------------
AT_BEST_PLANS = [
    AT_BASE_DIR / "saved_exp_plan/at_plan/CIFAR10_RES18_Same_PGD.yaml",
]
AT_BEST_AT_SEEDS = [0, 1, 2, 3, 4]   # the --seeds passed to main_at.py
AT_BEST_RUN_TAG = "v1"               # the --run-tag passed to main_at.py


def expand_at_plan(yaml_path, at_seeds=None, run_tag=None):
    """Folder names one AT plan produces: Model_Path x attack grid x at_seeds."""
    # Imported lazily: main_at pulls in the training stack, which the MI paths
    # in this file never need.
    from main_at import build_at_pos_scenario_name, parse_at_sources

    plan = process_yaml_file(yaml_path)
    # [(path, alias)]: an entry may be a plain string (alias None) or a
    # {Path, Alias} mapping. The alias is part of the folder name, so it has to
    # reach build_at_pos_scenario_name or the rebuilt names miss on disk.
    paths = parse_at_sources(plan)
    if not paths:
        raise ValueError(f"Plan declares no Model_Path entries: {yaml_path}")
    at_cfg = plan.get("AdversarialTraining", {}) or {}
    use_mixed = bool(at_cfg.get("use_mixed", False))
    mix_rate = float(at_cfg.get("mix_rate", 0.0))
    seeds = list(AT_BEST_AT_SEEDS if at_seeds is None else at_seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(
            type(s) is not int or s < 0 for s in seeds):
        raise ValueError("at_seeds must be unique nonnegative integers")
    run_tag = AT_BEST_RUN_TAG if run_tag is None else run_tag

    names = []
    for model_path, alias in paths:
        for _, attack_name, attack_kwargs in parse_attack_configs(plan):
            for seed in seeds:
                names.append(build_at_pos_scenario_name(
                    model_path, attack_name, attack_kwargs, seed,
                    use_mixed, mix_rate,
                    train_size=plan["Dataset"]["group_size"],
                    epochs=plan["Optimizer"]["Epochs"],
                    run_tag=run_tag, alias=alias,
                ))
    if len(set(names)) != len(names):
        raise ValueError(f"Plan expands to duplicate scenario names: {yaml_path}")
    return names


def models_from_at_plans(plans=None, at_seeds=None, run_tag=None, model_dir=None,
                         ckpt_kinds=None, selectors=None, missing="skip"):
    """[(folder_name, identity, plan_path)] for every model the plans declare.

    A plan may list a wider attack grid than was actually trained, so entries
    with no checkpoints on disk are reported and skipped by default; pass
    missing="raise" to refuse to run a partial set. `selectors` keeps only names
    containing one of the given substrings.
    """
    if missing not in ("skip", "raise"):
        raise ValueError("missing must be 'skip' or 'raise'")
    plans = list(AT_BEST_PLANS if plans is None else plans)
    if not plans:
        raise ValueError("No AT plan configured; set AT_BEST_PLANS")
    kinds = list(AT_BEST_CKPT_KINDS if ckpt_kinds is None else ckpt_kinds)
    root = Path(AT_BEST_MODEL_DIR if model_dir is None else model_dir)

    found, absent, seen = [], [], set()
    for plan in plans:
        for name in expand_at_plan(plan, at_seeds, run_tag):
            if selectors and not any(s in name for s in selectors):
                continue
            gaps = [k for k in kinds
                    if not (root / name / f"{k}_epoch.pth").is_file()
                    or (root / name / f"{k}_epoch.pth").stat().st_size == 0]
            if gaps:
                absent.append((name, gaps))
                continue
            if name in seen:
                raise ValueError(f"Two plans declare the same model: {name}")
            seen.add(name)
            found.append((name, parse_at_scenario_name(name), plan))

    if absent:
        summary = (f"{len(absent)} model(s) declared by the plan(s) have no "
                   f"checkpoint under {root}")
        if missing == "raise":
            raise FileNotFoundError(
                summary + ": " + ", ".join(n for n, _ in absent[:5])
            )
        print(f"[PLAN] {summary}; skipping them")
        for name, gaps in absent[:5]:
            print(f"        missing {gaps}: {name}")
        if len(absent) > 5:
            print(f"        ... and {len(absent) - 5} more")
    return found


def _missing_at_grid(csv_path, identity, in_sizes, bins):
    """Reject mixed checkpoint/configuration identities before reusing MI."""
    path = Path(csv_path)
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != AT_BEST_CSV_COLUMNS:
                raise ValueError(
                    f"Incompatible AT best-MI CSV header: {path}; use a separate CSV"
                )
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Incomplete/malformed MI row in {path}")
                if row["model_name"] != identity["model_name"]:
                    continue
                for key, expected in identity.items():
                    if row[key] != str(expected):
                        raise ValueError(
                            f"AT best-MI identity mismatch: {identity['model_name']}, "
                            f"{key}; preserve the existing CSV and use a new output "
                            "for changed checkpoints/configuration"
                        )
                size = float(row["in_size"])
                fraction = float(row["in_size_rate"])
                if not (0 < size <= identity["training_size"] and np.isfinite(fraction)
                        and abs(fraction - size / identity["training_size"]) < 1e-12):
                    raise ValueError(f"Invalid training-size fraction: {identity['model_name']}")
    return missing_mi_grid(
        path, identity["model_name"], identity["Scenario"], identity["seed"],
        identity["rate"], in_sizes, bins,
    )


def _append_at_best_row(csv_path, row):
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("x", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=AT_BEST_CSV_COLUMNS).writeheader()
    with path.open("a", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=AT_BEST_CSV_COLUMNS).writerow(row)


def _load_at_checkpoint(path, arch, num_classes, mean, std):
    """Return an eval-mode NormalizedModel for an AT checkpoint.

    Accepts both a NormalizedModel state_dict (base_model. prefix + mean/std)
    and a bare backbone state_dict. The checkpoint's own mean/std must match
    the dataset's, otherwise the logits would not be comparable to the pool.
    """
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net = NormalizedModel(build_model(arch, num_classes), mean, std).to(device)
    if any(k.startswith("base_model.") for k in state):
        net.load_state_dict(state, strict=True)
        for field, expected in (("mean", mean), ("std", std)):
            got = getattr(net, field).flatten().tolist()
            if max(abs(a - b) for a, b in zip(got, expected)) > AT_BEST_NORM_TOL:
                raise ValueError(
                    f"Checkpoint {field} {got} differs from dataset {field} "
                    f"{list(expected)}: {path}; MI would not be comparable"
                )
    else:
        net.base_model.load_state_dict(state, strict=True)
    net.eval()
    return net


def main_at_best(
        at_scenario, yaml_file_path=None,
        in_sizes=None, num_intervals_list=None, record_verbose=None,
        master_csv_path=None, verbose_dir=None, subset_seed=None,
        model_dir=None, in_size_rates=None, index_dir=None,
        ckpt_kinds=None, skip_existing=True,
):
    """Record best-checkpoint In/Out MI for ONE adversarially trained model.

    `at_scenario` is the folder name under `model_dir`. Both checkpoint kinds
    in `ckpt_kinds` are measured and stored as separate model_name entries with
    `epoch` set to the kind. Defaults are the editable AT_BEST_* globals. The
    probe sets are the victim's own nested subsets (group_A seed 42, subset
    seed 42), so the points are comparable to the negative pool, the victim,
    the FT positives and the extraction positives. Each missing
    (in_size, bins) cell is appended once; matching Out MI already in the CSV
    is reused by bin count. Returns the number of appended rows. This is a
    single-writer CSV.
    """
    yaml_file_path = AT_BEST_PLAN if yaml_file_path is None else yaml_file_path
    exp_yaml = process_yaml_file(yaml_file_path)
    ds_cfg = exp_yaml["Dataset"]
    arch = exp_yaml.get("Model", "ResNet-18")

    parsed = parse_at_scenario_name(at_scenario)
    kinds = list(AT_BEST_CKPT_KINDS if ckpt_kinds is None else ckpt_kinds)
    if not kinds or len(set(kinds)) != len(kinds):
        raise ValueError("ckpt_kinds must be a nonempty list of unique kinds")

    training_size = ds_cfg["group_size"]
    sizes_from_rates(training_size, [1.0])   # validates the denominator
    if in_sizes is not None and in_size_rates is not None:
        raise ValueError("Choose in_sizes or in_size_rates, not both")
    if in_sizes is None:
        in_sizes = sizes_from_rates(
            training_size,
            AT_BEST_IN_SIZE_RATES if in_size_rates is None else in_size_rates,
        )
    in_sizes = positive_ints(in_sizes, "in_sizes")
    if max(in_sizes) > training_size:
        raise ValueError("in_sizes cannot exceed the victim training group_size")
    bins = positive_ints(
        AT_BEST_BINS if num_intervals_list is None else num_intervals_list, "bins")

    model_dir = Path(AT_BEST_MODEL_DIR if model_dir is None else model_dir)
    csv_path = Path(AT_BEST_MASTER_CSV if master_csv_path is None else master_csv_path)
    verbose_dir = Path(AT_BEST_VERBOSE_DIR if verbose_dir is None else verbose_dir)
    index_dir = Path(AT_BEST_INDEX_DIR if index_dir is None else index_dir)
    subset_seed = AT_BEST_SUBSET_SEED if subset_seed is None else subset_seed
    record_verbose = AT_BEST_RECORD_VERBOSE if record_verbose is None else record_verbose
    plan_hash = _at_file_sha256(yaml_file_path)

    # Preflight every requested checkpoint before touching data or the CSV.
    pending = []
    for kind in kinds:
        checkpoint = model_dir / at_scenario / at_ckpt_filename(kind)
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty AT checkpoint: {checkpoint}")
        identity = dict(
            parsed,
            seed=parsed["at_seed"],
            model_name=f"{at_scenario}_ckpt={kind}",
            epoch=at_ckpt_epoch(kind), ckpt_kind=kind,
            training_size=training_size, family="cnn", subset_seed=subset_seed,
            group_seed=AT_BEST_GROUP_SEED, checkpoint=str(checkpoint.resolve()),
            checkpoint_sha256=_at_file_sha256(checkpoint), plan_sha256=plan_hash,
        )
        identity.pop("base_seed", None)
        missing, existing_out = _missing_at_grid(csv_path, identity, in_sizes, bins)
        if not skip_existing and len(missing) != len(in_sizes) * len(bins):
            raise ValueError("skip_existing=False would duplicate MI rows; use a new CSV")
        if missing:
            pending.append((checkpoint, identity, missing, existing_out))
        else:
            print(f"[SKIP] {identity['model_name']}: all requested best-MI cells already exist")
    if not pending:
        return 0

    set_seed(parsed["at_seed"])
    dataset_obj, num_classes, _ = build_dataset_from_yaml(ds_cfg)
    idx_dir = index_dir / ds_cfg["name"]
    group_A = create_or_load_group_A(
        dataset=dataset_obj.in_sample_set, save_dir=idx_dir, group_size=training_size,
        num_classes=num_classes, seed=AT_BEST_GROUP_SEED, force_rebuild=False,
    )
    nested_subsets = create_nested_balanced_subsets_shared(
        dataset=dataset_obj.in_sample_set, group_A=group_A, save_dir=idx_dir,
        subset_sizes=in_sizes, num_classes=num_classes,
        seed=subset_seed, force_rebuild=False,
    )
    absent = [s for s in in_sizes if s not in nested_subsets]
    if absent:
        raise ValueError(
            f"nested_subsets_seed{subset_seed}.npz in {idx_dir} has no entry for "
            f"in_size {absent}; available: {sorted(nested_subsets)}"
        )

    # AT checkpoints expect raw [0,1] inputs; the NormalizedModel wrapper
    # normalizes internally. Verified identical to the normalized-input /
    # bare-backbone convention used by the pool and the victim.
    out_sample_set = dataset_obj.raw_test_set
    out_size = len(out_sample_set)

    def loader(data):
        return DataLoader(
            data, batch_size=AT_BEST_BATCH_SIZE, shuffle=False,
            num_workers=AT_BEST_NUM_WORKERS,
            pin_memory=(str(device).startswith("cuda")), worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(subset_seed),
        )

    written = 0
    for checkpoint, identity, missing, existing_out in pending:
        name = identity["model_name"]
        needed_bins = sorted({b for _, b in missing})
        for nb in needed_bins:
            if nb in existing_out and existing_out[nb][2] != out_size:
                raise ValueError(f"Cached out_size differs for {name}, bins={nb}")
        net = _load_at_checkpoint(checkpoint, arch, num_classes,
                                  dataset_obj.mean, dataset_obj.std)
        if _at_file_sha256(checkpoint) != identity["checkpoint_sha256"]:
            raise ValueError(
                f"Checkpoint changed during loading: {checkpoint}; rerun when training ends"
            )
        print(f"[BEST] {name}: {len(missing)} missing MI cells on {device}")

        def measure(logits, labels, nb, split, size=None):
            result = mi_from_logits(logits, labels, num_intervals=nb, verbose=record_verbose)
            if not np.isfinite(result[:2]).all():
                raise ValueError(f"Nonfinite MI: {name}, {split}, {size}, bins={nb}")
            if record_verbose:
                save_verbose_data(result[2], verbose_dir, name, split, nb, in_size=size)
            return result[:2]

        try:
            out_results = {b: existing_out[b][:2] for b in needed_bins if b in existing_out}
            new_out_bins = [b for b in needed_bins if b not in existing_out]
            if new_out_bins:
                out_logits, out_labels = collect_logits(net, loader(out_sample_set), device)
                for nb in new_out_bins:
                    out_results[nb] = measure(out_logits, out_labels, nb, "out")
                del out_logits, out_labels
            for size in sorted({s for s, _ in missing}):
                probe = dataset_obj.subset("raw_train_clean", nested_subsets[size].tolist())
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
                    _append_at_best_row(csv_path, row)
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


def run_at_best_sweep(
        plans=None, selectors=None, model_dir=None, master_csv_path=None,
        at_seeds=None, run_tag=None, ckpt_kinds=None, missing="skip",
        from_disk=False, yaml_file_path=None, **kwargs
):
    """Evaluate the AT models the configured plans declare; per-model tally.

    Plan-driven by default, matching calculate_MI_ft.py / _extraction.py /
    _victim.py: the model set comes from AT_BEST_PLANS (times AT_BEST_AT_SEEDS
    and the plan's attack grid), so new checkpoints appearing in at_final do not
    enlarge the run, and every model is measured with the plan that produced it.

    from_disk=True restores the directory-enumeration behaviour; in that mode
    `yaml_file_path` (default AT_BEST_PLAN) supplies the plan for every model.
    """
    csv_path = Path(AT_BEST_MASTER_CSV if master_csv_path is None else master_csv_path)
    root = Path(AT_BEST_MODEL_DIR if model_dir is None else model_dir)
    kinds = AT_BEST_CKPT_KINDS if ckpt_kinds is None else ckpt_kinds

    if from_disk:
        selected = [(name, ident, yaml_file_path)
                    for name, ident in discover_at_models(model_dir, selectors, ckpt_kinds)]
        source = f"directory {root}"
    else:
        selected = models_from_at_plans(
            plans, at_seeds, run_tag, model_dir, ckpt_kinds, selectors, missing)
        source = ", ".join(Path(p).name for p in (AT_BEST_PLANS if plans is None else plans))
    if not selected:
        raise SystemExit(f"No AT model selected (selectors={selectors}) from {source}")

    print(f"AT best-MI sweep: {len(selected)} model(s) x {len(kinds)} checkpoint(s)")
    print(f"  source    : {source}")
    print(f"  model_dir : {root}")
    print(f"  output    : {csv_path}")
    print(f"  grid      : in_size_rates={AT_BEST_IN_SIZE_RATES}  bins={AT_BEST_BINS}")
    tally = {}
    for name, _, plan in selected:
        tally[name] = main_at_best(
            name, plan, model_dir=model_dir, master_csv_path=csv_path,
            ckpt_kinds=ckpt_kinds, **kwargs
        )
    print("\nAT best-MI sweep finished:")
    for name, total in tally.items():
        print(f"  appended {total:>4} rows  {name[-46:]}")
    print(f"  -> {csv_path}")
    return tally


# =====================================================
# 4. Entry point
# =====================================================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    # New default: best-checkpoint MI on the negative-pool-0.0 grid for the
    # models declared by AT_BEST_PLANS (x AT_BEST_AT_SEEDS x the plan's attack
    # grid). Only those are computed, whatever else sits in at_final. Optional
    # arguments narrow that set by substring, e.g. "eps=0.031373" or "atseed=0".
    import sys

    run_at_best_sweep(selectors=sys.argv[1:] or None)

    """ Previous entry (main_at_neg_scratch on at_vanilla, absolute
    in_sizes, thin schema) -- kept verbatim for reference.
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Folder containing all YAML experiment plans
    exp_dir = "./saved_exp_plan/at_plan"
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

        for seed in range(42, 43): # entry point for main_nega
            for rate in [1.0, 0.5, 0.0]:#np.linspace(0, 1, 11): #overlapping rate: [0, 0.1, 0.2, 0.3,...,0.9, 1]
                print(f"\n>>> Running seed {seed} for {os.path.basename(yaml_path)}")
                set_seed(seed)

                main_at_neg_scratch(seed, rate, yaml_path,
                                    in_sizes = [25000], #1000, 5000, 10000, 15000, 20000, 
                                    num_intervals_list = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200],
                                    record_verbose=False)
                
        '''for seed in range(42, 43): # entry point for main_nega
            print(f"\n>>> Running seed {seed} for {os.path.basename(yaml_path)}")
            set_seed(seed)

            main_at_neg(seed, yaml_path,
                        in_sizes = [25000], #1000, 5000, 10000, 15000, 20000, 
                        num_intervals_list = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200],
                        record_verbose=False)'''
                
        
    """
