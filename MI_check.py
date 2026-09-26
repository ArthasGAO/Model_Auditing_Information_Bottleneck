import os
import glob
import time

from util_adv import build_model, collect_checkpoints
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

import os
import csv
from datetime import datetime
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from util import build_dataset_from_yaml, create_or_load_group_A, load_best_checkpoint, process_experiment_setup, process_experiment_setup_deit, process_yaml_file

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
        
# ===========================================================================
# Subset generation and incremental MI coverage (shared, Torch-free helpers).
# ===========================================================================
from mi_pool_support import (
    create_nested_balanced_subsets, missing_mi_grid, sizes_from_rates,
)


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


def _extract_epoch_from_ckpt(ckpt_name):
    """
    Parse the epoch integer from a checkpoint filename of the form
    'epoch_N.pth' (or any name whose stem ends with '_N').
 
    Returns the integer N. Assumes the 'epoch_N.pth' convention is enforced
    upstream by the training pipeline.
    """
    stem = Path(ckpt_name).stem            # 'epoch_50.pth' -> 'epoch_50'
    return int(stem.split("_")[-1])        # 'epoch_50'     -> 50


# ===========================================================================
# Updated main_nega: inference and MI cleanly separated
# ===========================================================================

def main_nega(
    seed, r, yaml_file_path,
    in_sizes=None,
    num_intervals_list=None,
    record_verbose=True,
    master_csv_path="./saved_logs/vanilla/MI_master_table.csv",
    verbose_dir="./saved_logs/vanilla/MI_verbose",
    subset_seed=42,
):
    """ this function corresponds to the train_plan."""
    if in_sizes is None:
        in_sizes = [5000, 10000, 15000, 20000, 25000]
    if num_intervals_list is None:
        num_intervals_list = [50, 75, 100, 125, 150]

    print(f"Device: {device}")

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_setup(exp_yaml)

    # ---- Load model ----
    print('==> Loading model..')
    model_name = exp_yaml["Scenario_Name"] + f"_{seed}_{round(r, 2)}"
    print(f"\n this time scenario is: {model_name}")

    model_dir = './saved_models/vanilla/'
    model_folder = Path(model_dir) / model_name
    ckpt_path, _ = load_best_checkpoint(model_folder)
    if ckpt_path is None:
        print(f"[⚠️] No .pth files found in {model_folder}")
        return
    net = exp_setup["Model"].to(device)
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state)

    # ---- Prepare datasets ----
    print('==> Preparing data..')
    in_sample_set = exp_setup["Dataset"].in_sample_set
    out_sample_set = exp_setup["Dataset"].test_set

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

    ensure_master_csv(master_csv_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
            "train", subset_indices, clean=True,
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
        if record_verbose:
            ixt_out, ity_out, dbg_out = mi_from_logits(
                out_layer_T, out_labels, num_intervals=nb, verbose=True,
            )
            path_out = save_verbose_data(
                dbg_out, verbose_dir, model_name, "out", nb,
            )
            print(f"  out  bins={nb:>3}: I(X;T)={ixt_out:.4f}, "
                  f"I(T;Y)={ity_out:.4f}  -> {path_out.name}")
        else:
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
            if record_verbose:
                ixt_in, ity_in, dbg_in = mi_from_logits(
                    layer_T, labels, num_intervals=nb, verbose=True,
                )
                path_in = save_verbose_data(
                    dbg_in, verbose_dir, model_name, "in", nb, in_size=in_size,
                )
                print(f"  in   size={in_size:>5}  bins={nb:>3}: "
                      f"I(X;T)={ixt_in:.4f}, I(T;Y)={ity_in:.4f}  "
                      f"-> {path_in.name}")
            else:
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

    # Free the cached logits to keep GPU memory tidy if many models in sequence
    del in_cache, out_layer_T, out_labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n==> Done. Master CSV: {master_csv_path}")


def main_nega_deit(
    seed, r, yaml_file_path,
    in_sizes=None,
    num_intervals_list=None,
    record_verbose=True,
    master_csv_path="./saved_logs/vanilla/MI_master_table.csv",
    verbose_dir="./saved_logs/vanilla/MI_verbose",
    subset_seed=42,
):
    """
    DeiT (Transformer) 版本的负样本 MI 计算。
    与 main_nega(CNN) 的区别:
      - 用 process_experiment_setup_deit + "Student Model"(而非 "Model")
      - 模型目录指向 Transformer_Models
      - collect_logits 后用 split_deit_outputs 防御性处理(plain 返回单张量,
        distilled 万一返回 tuple 也能正确取用)
      - 确保 eval 模式(关 drop_path,plain 下返回单张量)
    """
    if in_sizes is None:
        in_sizes = [5000, 10000, 15000, 20000, 25000]
    if num_intervals_list is None:
        num_intervals_list = [50, 75, 100, 125, 150]
 
    print(f"Device: {device}")
 
    exp_yaml = process_yaml_file(yaml_file_path)
    # ---- 关键差异 1:用 DeiT 的 setup 函数 ----
    exp_setup = process_experiment_setup_deit(exp_yaml)
 
    # ---- Load model ----
    print('==> Loading model..')
    model_name = exp_yaml["Scenario_Name"] + f"_{seed}_{round(r, 2)}"
    print(f"\n this time scenario is: {model_name}")
 
    # ---- 关键差异 2:DeiT 模型存在 Transformer_Models 目录 ----
    model_dir = './saved_models/vanilla/Transformer_Models'
    model_folder = Path(model_dir) / model_name
    ckpt_path, _ = load_best_checkpoint(model_folder)
    if ckpt_path is None:
        print(f"[⚠️] No .pth files found in {model_folder}")
        return
 
    # ---- 关键差异 3:DeiT setup 返回的是 "Student Model" ----
    net = exp_setup["Student Model"].to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # 兼容可能的 checkpoint 包裹格式
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state)
 
    # ---- 关键差异 4:确保 eval 模式 ----
    # MI 必须在 eval 模式下算:关 drop_path,且 plain DeiT 在 eval 下返回单张量。
    net.eval()
 
    # ---- Prepare datasets ----
    print('==> Preparing data..')
    in_sample_set = exp_setup["Dataset"].in_sample_set
    out_sample_set = exp_setup["Dataset"].test_set
 
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
 
    ensure_master_csv(master_csv_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
 
    # =============================================================
    # PHASE 1: Inference (the expensive part) -- ONCE per data set
    # =============================================================
    print(f"\n==> [Phase 1] Running inference once per loader")
 
    # Out-sample: infer once, reuse for all bin counts
    print(f"  Inferring out-sample (size={out_size})...")
    out_layer_T, out_labels = collect_logits(net, out_sample_loader, device)
    print(f"    -> cached logits shape: {tuple(out_layer_T.shape)}")
 
    # In-sample: infer once per in_size, reuse for all bin counts
    in_cache = {}
    for in_size in in_sizes:
        subset_indices = nested_subsets[in_size].tolist()
        in_sample_subset = exp_setup["Dataset"].subset(
            "train", subset_indices, clean=True,   # clean=True: MI 用无增强数据
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
 
    out_results = {}
    for nb in num_intervals_list:
        if record_verbose:
            ixt_out, ity_out, dbg_out = mi_from_logits(
                out_layer_T, out_labels, num_intervals=nb, verbose=True,
            )
            path_out = save_verbose_data(
                dbg_out, verbose_dir, model_name, "out", nb,
            )
            print(f"  out  bins={nb:>3}: I(X;T)={ixt_out:.4f}, "
                  f"I(T;Y)={ity_out:.4f}  -> {path_out.name}")
        else:
            ixt_out, ity_out = mi_from_logits(
                out_layer_T, out_labels, num_intervals=nb, verbose=False,
            )
            print(f"  out  bins={nb:>3}: I(X;T)={ixt_out:.4f}, "
                  f"I(T;Y)={ity_out:.4f}")
        out_results[nb] = (ixt_out, ity_out)
 
    for in_size in in_sizes:
        layer_T, labels = in_cache[in_size]
        for nb in num_intervals_list:
            if record_verbose:
                ixt_in, ity_in, dbg_in = mi_from_logits(
                    layer_T, labels, num_intervals=nb, verbose=True,
                )
                path_in = save_verbose_data(
                    dbg_in, verbose_dir, model_name, "in", nb, in_size=in_size,
                )
                print(f"  in   size={in_size:>5}  bins={nb:>3}: "
                      f"I(X;T)={ixt_in:.4f}, I(T;Y)={ity_in:.4f}  "
                      f"-> {path_in.name}")
            else:
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
 
    del in_cache, out_layer_T, out_labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
 
    print(f"\n==> Done. Master CSV: {master_csv_path}")

 
def main_at(
    seed, r, yaml_file_path,
    in_sizes=None,
    num_intervals_list=None,
    record_verbose=True,
    master_csv_path="./saved_logs/at_train_new/MI_master_table.csv",
    verbose_dir="./saved_logs/at_train_new/MI_verbose",
    subset_seed=42,
):
    """
    MI evaluation for AT-trained suspects, one row per
    (scenario, epoch, in_size, bins) combination.
 
    When the YAML's Positive.State is 'all', every checkpoint in each
    model_path directory produces its own master-CSV row. The 'Scenario'
    and 'model_name' columns include an '_epoch=N' suffix so different
    epochs are distinguishable downstream (and verbose-file paths don't
    collide across epochs).
    """
    if in_sizes is None:
        in_sizes = [5000, 10000, 15000, 20000, 25000]
    if num_intervals_list is None:
        num_intervals_list = [50, 75, 100, 125, 150]
 
    print(f"Device: {device}")
 
    exp_yaml = process_yaml_file(yaml_file_path)
 
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
    nested_subsets = create_nested_balanced_subsets(
        dataset=in_sample_set, group_A=group_A,
        save_dir=f'./Indices/{victim_ds_cfg["name"]}/',
        subset_sizes=in_sizes,
        num_classes=num_classes,
        seed=subset_seed, force_rebuild=False,
    )
 
    out_size = len(out_sample_set)
    out_sample_loader = DataLoader(
        out_sample_set, batch_size=128, shuffle=False, num_workers=0,
        pin_memory=True,
    )
 
    ensure_master_csv(master_csv_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
 
    # ---- Iterate over each model_path x selected checkpoints -------------
    base_dir = Path("./saved_models")
 
    for dir_name in model_paths:
        # Base scenario name = leaf segment of the model directory.
        # Per-epoch scenario/model names get '_epoch=N' appended inside the
        # checkpoint loop below.
        dir_leaf = dir_name.rstrip("/").split("/")[-1]
 
        print(f"\n--- MI Evaluation: {dir_leaf} ---")
 
        model_dir = base_dir / dir_name.lstrip("/")
 
        checkpoints = collect_checkpoints(model_dir, eval_mode)
        if not checkpoints:
            print(f"  [SKIP] No checkpoints found in {model_dir}")
            continue
 
        for ckpt_name, ckpt_path in checkpoints:
            # Per-checkpoint identity: parsed integer + suffixed names.
            epoch         = _extract_epoch_from_ckpt(ckpt_name)
            scenario_name = f"{dir_leaf}_epoch={epoch}"
            model_name    = scenario_name   # keep tied; controls verbose-file paths
 
            print(f"  Evaluating: {ckpt_name}  ({ckpt_path.name})  "
                  f"-> epoch={epoch}")
 
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
                in_sample_subset = dataset_obj.subset(
                    "train", subset_indices, clean=True,
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
                if record_verbose:
                    ixt_out, ity_out, dbg_out = mi_from_logits(
                        out_layer_T, out_labels, num_intervals=nb, verbose=True,
                    )
                    path_out = save_verbose_data(
                        dbg_out, verbose_dir, model_name, "out", nb,
                    )
                    print(f"  out  bins={nb:>3}: I(X;T)={ixt_out:.4f}, "
                          f"I(T;Y)={ity_out:.4f}  -> {path_out.name}")
                else:
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
                    if record_verbose:
                        ixt_in, ity_in, dbg_in = mi_from_logits(
                            layer_T, labels, num_intervals=nb, verbose=True,
                        )
                        path_in = save_verbose_data(
                            dbg_in, verbose_dir, model_name, "in", nb, in_size=in_size,
                        )
                        print(f"  in   size={in_size:>5}  bins={nb:>3}: "
                              f"I(X;T)={ixt_in:.4f}, I(T;Y)={ity_in:.4f}  "
                              f"-> {path_in.name}")
                    else:
                        ixt_in, ity_in = mi_from_logits(
                            layer_T, labels, num_intervals=nb, verbose=False,
                        )
                        print(f"  in   size={in_size:>5}  bins={nb:>3}: "
                              f"I(X;T)={ixt_in:.4f}, I(T;Y)={ity_in:.4f}")
 
                    ixt_out, ity_out = out_results[nb]
 
                    row = {
                        "Scenario":    scenario_name,   # includes _epoch=N
                        "seed":        seed,
                        "rate":        round(r, 2),     # at case, this rate is always 1.0
                        "model_name":  model_name,      # tied to Scenario
                        "epoch":       epoch,           # parsed integer (no longer hardcoded)
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



# ===========================================================================
# Negative model pool, overlap rate = 0.0  (80 seeds per scenario)
#
# Same two-phase pipeline as main_nega (inference once per loader, then a
# bin sweep on the cached logits), with three differences:
#   * checkpoints are read from POOL0_MODEL_DIR instead of
#     ./saved_models/vanilla/<model_name>
#   * rows are appended to a separate CSV (POOL0_MASTER_CSV) that has the
#     same schema as MI_master_table.csv
#   * a model whose full (in_size x bins) grid is already in that CSV is
#     skipped, so the sweep can be interrupted and resumed safely
# ===========================================================================

POOL0_MODEL_DIR   = "./saved_models/vanilla/Negative_Model_Pool_0.0/CNN"
POOL0_MASTER_CSV  = "./saved_logs/vanilla/MI_master_table_neg_pool0.csv"
POOL0_VERBOSE_DIR = "./saved_logs/vanilla/MI_verbose_neg_pool0"
POOL0_RATE        = 0.0
# DeiT pool: scenario names match the yaml Scenario_Name, as for CNNs.
POOL0_DEIT_MODEL_DIR = "./saved_models/vanilla/Negative_Model_Pool_0.0/Transformer"

# Training plans of the four CNN scenarios in the pool. train_plan/ itself
# only holds the DeiT plan now; the CNN plans live in old_plan/.
POOL0_YAMLS = [
    "./saved_exp_plan/train_plan/old_plan/CIFAR10_RES18_SGD_SMALL.yaml",
    "./saved_exp_plan/train_plan/old_plan/CIFAR10_VGG16_SGD_SMALL.yaml",
    "./saved_exp_plan/train_plan/old_plan/CIFAR100_RES18_SGD_SMALL.yaml",
    "./saved_exp_plan/train_plan/old_plan/CIFAR100_VGG16_SGD_SMALL.yaml",
]

# DeiT plans (rate 0.0 only): CIFAR-10 -> plain DeiT-tiny, CIFAR-100 -> the
# hard-distilled DeiT-tiny pool. Evaluated with family="deit".
POOL0_DEIT_YAMLS = [
    "./saved_exp_plan/train_plan/old_plan/CIFAR10_DeiT_Plain_SMALL.yaml",
    "./saved_exp_plan/train_plan/CIFAR100_DeiT_Distill_SMALL.yaml",
]

# New MI sizes are exact fractions of each model's actual training group_size.
POOL0_IN_SIZE_RATES = [0.01, 0.05, 0.10, 0.20, 0.50, 0.75, 1.00]
POOL0_SEEDS = range(42, 122)
POOL0_RUN_CNN = True
POOL0_RUN_DEIT = True
POOL0_BINS     = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]


import copy


def main_nega_pool0(
    seed, yaml_file_path,
    in_sizes=None,
    num_intervals_list=None,
    record_verbose=False,
    model_dir=None,
    master_csv_path=POOL0_MASTER_CSV,
    verbose_dir=POOL0_VERBOSE_DIR,
    subset_seed=42,
    rate=POOL0_RATE,
    skip_existing=True,
    family="cnn",
):
    """
    MI evaluation for ONE model of the rate-0.0 negative pool.

    Loads <model_dir>/<Scenario_Name>_<seed>_<rate>/best_epoch.pth, runs
    inference once on the out-sample set and once per in_size on the
    nested in-sample subsets, then computes (I(X;T), I(T;Y)) for every
    (in_size, bins) pair and appends one row per pair to master_csv_path.

    family: "cnn" (process_experiment_setup, key "Model", default dir
    POOL0_MODEL_DIR) or "deit" (process_experiment_setup_deit, key
    "Student Model", default dir POOL0_DEIT_MODEL_DIR). model_dir=None
    picks the family default.

    Returns True if rows were written, False if the model was skipped
    (already complete in the CSV) or its checkpoint is missing.
    """
    if family not in ("cnn", "deit"):
        raise ValueError(f"family must be 'cnn' or 'deit', got {family!r}")
    if model_dir is None:
        model_dir = POOL0_DEIT_MODEL_DIR if family == "deit" else POOL0_MODEL_DIR
    if num_intervals_list is None:
        num_intervals_list = list(POOL0_BINS)

    exp_yaml = process_yaml_file(yaml_file_path)
    if in_sizes is None:
        in_sizes = sizes_from_rates(exp_yaml["Dataset"]["group_size"], POOL0_IN_SIZE_RATES)
    scenario = exp_yaml["Scenario_Name"]
    model_name = f"{scenario}_{seed}_{round(rate, 2)}"
    model_folder = Path(model_dir) / model_name

    # Check exact pairs, not row count. Preserve all existing valid records.
    missing_pairs, existing_out = missing_mi_grid(
        master_csv_path, model_name, scenario, seed, round(rate, 2),
        in_sizes, num_intervals_list,
    )
    if not skip_existing and len(missing_pairs) != len(in_sizes) * len(num_intervals_list):
        raise ValueError("skip_existing=False would duplicate MI rows; use a new CSV for recomputation")
    if not missing_pairs:
        print(f"[SKIP] {model_name}: all requested MI combinations already exist")
        return False
    needed_sizes = sorted({s for s, _ in missing_pairs})
    needed_bins = sorted({b for _, b in missing_pairs})
    missing_set = set(missing_pairs)
    print(f"[RESUME] {model_name}: {len(missing_pairs)} missing pairs, sizes={needed_sizes}")

    # ---- model ----
    print(f"\n==> [{model_name}]  device={device}")
    ckpt_path, _ = load_best_checkpoint(model_folder)
    if ckpt_path is None:
        print(f"[WARN] No best_epoch.pth found in {model_folder}")
        return False
    if family == "deit":
        # Inference needs only the student. Distillation is switched off in
        # a copy of the plan so no teacher is built (the Distill plan's
        # teacher_ckpt is a {seed} template that does not resolve here);
        # the student architecture comes from the Model block and is
        # unaffected. In eval mode the distilled head returns the averaged
        # (cls + dist) logits as a single tensor.
        mi_yaml = copy.deepcopy(exp_yaml)
        if "Distillation" in mi_yaml:
            mi_yaml["Distillation"]["enabled"] = False
        exp_setup = process_experiment_setup_deit(mi_yaml)
        net = exp_setup["Student Model"].to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
    else:
        exp_setup = process_experiment_setup(exp_yaml)
        net = exp_setup["Model"].to(device)
        state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state)
    net.eval()

    # ---- data (same group_A / nested subsets as MI_master_table.csv) ----
    in_sample_set = exp_setup["Dataset"].in_sample_set
    out_sample_set = exp_setup["Dataset"].test_set
    idx_dir = f'./Indices/{exp_yaml["Dataset"]["name"]}/'
    group_A = create_or_load_group_A(
        dataset=in_sample_set, save_dir=idx_dir,
        group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"],
        seed=42, force_rebuild=False,
    )
    nested_subsets = create_nested_balanced_subsets(
        dataset=in_sample_set, group_A=group_A, save_dir=idx_dir,
        subset_sizes=in_sizes, num_classes=exp_setup["NumClasses"],
        seed=subset_seed, force_rebuild=False,
    )
    missing_sizes = [s for s in in_sizes if s not in nested_subsets]
    if missing_sizes:
        raise ValueError(
            f"nested_subsets_seed{subset_seed}.npz in {idx_dir} has no entry "
            f"for in_size {missing_sizes}; available: {sorted(nested_subsets)}"
        )

    out_size = len(out_sample_set)
    out_sample_loader = DataLoader(
        out_sample_set, batch_size=128, shuffle=False, num_workers=0,
        pin_memory=True,
    )

    for nb in needed_bins:
        if nb in existing_out and existing_out[nb][2] != out_size:
            raise ValueError(f"Cached out_size differs for {model_name}, bins={nb}")
    ensure_master_csv(master_csv_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---- phase 1: inference, once per loader ----
    out_bins_to_compute = [b for b in needed_bins if b not in existing_out]
    out_layer_T = out_labels = None
    if out_bins_to_compute:
        print(f"  [Phase 1] out-sample (size={out_size})")
        out_layer_T, out_labels = collect_logits(net, out_sample_loader, device)

    in_cache = {}
    for in_size in needed_sizes:
        subset_indices = nested_subsets[in_size].tolist()
        in_sample_subset = exp_setup["Dataset"].subset(
            "train", subset_indices, clean=True,
        )
        loader = DataLoader(
            in_sample_subset, batch_size=128, shuffle=False, pin_memory=True,
        )
        print(f"  [Phase 1] in-sample (size={in_size})")
        in_cache[in_size] = collect_logits(net, loader, device)

    # ---- phase 2: MI on cached logits ----
    out_results = {b: existing_out[b][:2] for b in needed_bins if b in existing_out}
    for nb in out_bins_to_compute:
        if record_verbose:
            ixt_out, ity_out, dbg_out = mi_from_logits(
                out_layer_T, out_labels, num_intervals=nb, verbose=True,
            )
            save_verbose_data(dbg_out, verbose_dir, model_name, "out", nb)
        else:
            ixt_out, ity_out = mi_from_logits(
                out_layer_T, out_labels, num_intervals=nb, verbose=False,
            )
        out_results[nb] = (ixt_out, ity_out)

    for in_size in needed_sizes:
        layer_T, labels = in_cache[in_size]
        for nb in needed_bins:
            if (in_size, nb) not in missing_set:
                continue
            if record_verbose:
                ixt_in, ity_in, dbg_in = mi_from_logits(
                    layer_T, labels, num_intervals=nb, verbose=True,
                )
                save_verbose_data(
                    dbg_in, verbose_dir, model_name, "in", nb, in_size=in_size,
                )
            else:
                ixt_in, ity_in = mi_from_logits(
                    layer_T, labels, num_intervals=nb, verbose=False,
                )
            ixt_out, ity_out = out_results[nb]
            print(f"  in_size={in_size:>5} bins={nb:>3}: "
                  f"I(X;T)={ixt_in:.4f} I(T;Y)={ity_in:.4f} | "
                  f"out I(X;T)={ixt_out:.4f} I(T;Y)={ity_out:.4f}")
            if not np.isfinite([ixt_in, ity_in, ixt_out, ity_out]).all():
                raise ValueError(f"Nonfinite computed MI: {model_name}, {in_size}, {nb}")
            row = {
                "Scenario":    scenario,
                "seed":        seed,
                "rate":        round(rate, 2),
                "model_name":  model_name,
                "epoch":       99,   # placeholder, same convention as MI_master_table.csv
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

    del in_cache, out_layer_T, out_labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


def run_pool0_sweep(
    yamls=POOL0_YAMLS,
    seeds=POOL0_SEEDS,
    in_sizes=None,
    num_intervals_list=POOL0_BINS,
    record_verbose=False,
    master_csv_path=POOL0_MASTER_CSV,
    model_dir=None,
    family="cnn",
):
    """
    Evaluate every (scenario yaml, seed) of the rate-0.0 pool.
    family selects the CNN or DeiT loading path (see main_nega_pool0).

    Iterates scenario-major so one architecture / dataset is finished
    before the next starts. Prints a per-scenario tally at the end.
    """
    seeds = list(seeds)
    if model_dir is None:
        model_dir = POOL0_DEIT_MODEL_DIR if family == "deit" else POOL0_MODEL_DIR
    print(f"Negative pool 0.0 sweep [{family}]: {len(yamls)} scenario(s) x {len(seeds)} seed(s)")
    print(f"  model_dir : {model_dir}")
    print(f"  output    : {master_csv_path}")
    print(f"  grid      : in_sizes={in_sizes if in_sizes is not None else POOL0_IN_SIZE_RATES}  bins={list(num_intervals_list)}")

    tally = {}
    for yaml_path in yamls:
        scenario = process_yaml_file(yaml_path)["Scenario_Name"]
        done = skipped = 0
        for seed in seeds:
            set_seed(seed)
            wrote = main_nega_pool0(
                seed, yaml_path,
                in_sizes=in_sizes,
                num_intervals_list=num_intervals_list,
                record_verbose=record_verbose,
                model_dir=model_dir,
                master_csv_path=master_csv_path,
                family=family,
            )
            if wrote:
                done += 1
            else:
                skipped += 1
        tally[scenario] = (done, skipped)

    print("\nNegative pool 0.0 sweep finished:")
    for scenario, (done, skipped) in tally.items():
        print(f"  {scenario:<28} computed={done:<3} skipped/missing={skipped}")
    print(f"  -> {master_csv_path}")
    return tally



if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Resume all 480 pool models. Defaults resolve rates from each YAML group_size.
    if POOL0_RUN_CNN:
        run_pool0_sweep(yamls=POOL0_YAMLS, family="cnn")
    if POOL0_RUN_DEIT:
        run_pool0_sweep(yamls=POOL0_DEIT_YAMLS, family="deit")


    """ Previous entry (DeiT negatives via main_nega_deit) -- kept for reference.
    # Folder containing all YAML experiment plans
    exp_dir = "./saved_exp_plan/train_plan"
    #exp_dir = "./saved_exp_plan/tmp_plan"
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
            
        for seed in range(53, 54): # entry point for main_nega
            for rate in [0.3]:#np.linspace(0, 1, 11): #overlapping rate: [0, 0.1, 0.2, 0.3,...,0.9, 1]
                print(f"\n>>> Running seed {seed} for {os.path.basename(yaml_path)}")
                set_seed(seed)

                '''main_nega(seed, rate, yaml_path, 
                          in_sizes = [1000, 5000, 10000, 15000, 20000, 25000],
                          num_intervals_list = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200],
                          record_verbose=False)'''
                
                main_nega_deit(seed, rate, yaml_path, 
                          in_sizes = [1000, 5000, 10000, 15000, 20000, 25000],
                          num_intervals_list = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200],
                          record_verbose=False)
    """

    '''exp_dir = "./saved_exp_plan/at_eval_plan"
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
            
        for seed in range(42, 43): # entry point for main_at
            print(f"\n>>> Running seed {seed} for {os.path.basename(yaml_path)}")
            set_seed(seed)

            main_at(seed, 1.0, yaml_path, 
                    in_sizes = [25000], #1000, 5000, 10000, 15000, 20000, 
                    num_intervals_list = [50],#5, 10, 15, 20, 30, 50, 75, 100, 150, 200],
                    master_csv_path="./saved_logs/at_train_new1/MI_master_table.csv",
                    verbose_dir="./saved_logs/at_train_new1/MI_verbose",)'''