from datetime import datetime
import copy
import hashlib
import os
import glob
import time
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
from util import create_or_load_subset_from_group, process_experiment_ft_setup_deit, process_yaml_file, process_experiment_ft_setup, create_train_subset, evaluate2, prepare_group_subset,\
                 load_last_checkpoint, load_checkpoint_from_epoch, build_epoch_to_ckpt_map, create_or_load_group_A
import torch.nn.functional as F
from MI_check import (
    collect_logits, MI_formula_cal, mi_from_logits,
    POOL0_IN_SIZE_RATES, POOL0_BINS,
)
from mi_pool_support import (
    create_nested_balanced_subsets, missing_mi_grid, sizes_from_rates, positive_ints,
)

# Editable configuration for main_best / the script entry point.
BASE_DIR = Path(__file__).resolve().parent
# FT_PLAN_DIR overrides the plan folder (default saved_exp_plan/ft_plan). main_ft.py reads the same variable, so training and MI stay on one batch.
BEST_PLAN_DIR = Path(os.environ.get("FT_PLAN_DIR", BASE_DIR / "saved_exp_plan/ft_plan"))
BEST_MODEL_DIR = BASE_DIR / "saved_models/ft_final"
BEST_MASTER_CSV = BASE_DIR / "saved_logs/ft_final/MI_master_table_ft.csv"
BEST_VERBOSE_DIR = BASE_DIR / "saved_logs/ft_final/MI_verbose_best"
BEST_INDEX_DIR = BASE_DIR / "Indices"
BEST_MODEL_SEEDS = [42]
# SEED_START / SEED_END mirror main_ft.py, so training and MI can be pointed at
# the same batch. The default keeps the historical 0..4 range; with missing=
# "skip" an untrained seed in that range is simply reported and passed over.
BEST_FT_SEEDS = list(range(int(os.environ.get("SEED_START", 0)),
                           int(os.environ.get("SEED_END", 5))))
BEST_RATES = [1.0]
BEST_IN_SIZE_RATES = list(POOL0_IN_SIZE_RATES)
BEST_BINS = list(POOL0_BINS)
BEST_STRATEGIES = ["FT-LL", "FT-AL", "RT-AL"]
BEST_SUBSET_SEED = 42
BEST_RECORD_VERBOSE = False
BEST_BATCH_SIZE = 128
BEST_NUM_WORKERS = 0

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



# MI inference and computation are shared with MI_check.py.
def save_verbose_data(debug_data, save_dir, model_name, split_name,
                      num_intervals, in_size=None, epoch=None):
    """Save verbose debug data to a structured .npz file."""
    out_dir = Path(save_dir) / model_name
    if epoch is not None:
        out_dir = out_dir / f"epoch_{epoch}"
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
    "in_size", "I(X;T)-In", "I(T;Y)-In",
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



# ---------------- CONFIGURATION ----------------
STRATEGY = ['FT-LL', 'FT-AL', 'RT-AL']#]#, 

# =====================================================
# 3. Main training logic
# =====================================================
def main_last(model_seed, ft_seed, r, yaml_file_path): 
    print(device)

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_ft_setup(exp_yaml)

    g = torch.Generator()
    g.manual_seed(ft_seed)

    print('==> Preparing data..')
    in_sample_set = exp_setup["Dataset"].in_sample_set # No data augmentation here
    
    # Important: the in sample evaluation set should be the same each round
    group_A = create_or_load_group_A(dataset=in_sample_set, save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
                                    group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"], seed=42, force_rebuild=False)
    in_sample_subset = exp_setup["Dataset"].subset("train", group_A, clean=True)

    in_sample_loader = DataLoader(in_sample_subset,batch_size=128,shuffle=False,num_workers=8, # in sample without augmentation
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, pin_memory=True)

    for strategy in STRATEGY:
        model_name = (
            f'{exp_yaml["Scenario_Name"]}_{model_seed}_{round(r, 2)}'
            f'_{strategy}_{ft_seed}'
        )
        print(model_name)

        model_dir = './saved_models/ft_vanilla/'
        model_folder = Path(model_dir)/model_name

        ckpt_path = load_last_checkpoint(model_folder)
        if ckpt_path is None:
            raise FileNotFoundError(f"No .pth files found in {model_folder}")

        # Fresh model instance
        net = exp_setup["Model_Factory"]().to(device)
        state = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(state)

        log_dir = './saved_logs/ft_vanilla/MI'
        os.makedirs(log_dir, exist_ok=True)
        log_name = model_name
        log_file = os.path.join(log_dir, f"training_log_{log_name}_MI.csv")

        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Scenario','Epoch','I(X;T)-InAug', 'I(T;Y)-InAug', 'I(X;T)-In', 'I(T;Y)-In','I(X;T)-Out', 'I(T;Y)-Out'])

        # Evaluation loop
        value_xt_in, value_ty_in = evaluate2(net, in_sample_loader, device)

        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([log_name, 99,
                                0, 0,
                                value_xt_in, value_ty_in,
                                0, 0
                                ])
            
            
def main_10000(seed, r, yaml_file_path): 
    print(device)

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_ft_setup(exp_yaml)

    g = torch.Generator()
    g.manual_seed(seed)

    print('==> Preparing data..')
    in_sample_set = exp_setup["Dataset"].in_sample_set # No data augmentation here
    out_sample_set =  exp_setup["Dataset"].test_set
    #train_set = exp_setup["Dataset"].train_set
    
    # Important: the in sample evaluation set should be the same each round
    group_A = create_or_load_group_A(dataset=in_sample_set, save_dir=f'./Indices/{exp_yaml['Dataset']['name']}/',
                                                  group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"], seed=42, force_rebuild=False)
    subset_A = create_or_load_subset_from_group(
                dataset=in_sample_set,
                group_A=group_A,
                save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
                subset_size=10000,
                num_classes=exp_setup["NumClasses"],
                seed=42,
                force_rebuild=False
            )
    
    in_sample_subset = exp_setup["Dataset"].subset("train", subset_A, clean=True)

    in_sample_loader = DataLoader(in_sample_subset,batch_size=128,shuffle=False,num_workers=8, # in sample without augmentation
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)

    for strategy in STRATEGY:
        model_name = exp_yaml["Scenario_Name"] + f"_{seed}_{round(r,2)}" + f"_{strategy}"
        print(model_name)
        model_dir = './saved_models/ft_vanilla/'
        model_folder = Path(model_dir)/model_name

        ckpt_path = load_last_checkpoint(model_folder)
        if ckpt_path is None:
            print(f"[⚠️] No .pth files found in {model_folder}")

        net = exp_setup["Model"].to(device) # load model
        state = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(state)

        log_dir = './saved_logs/ft_vanilla/MI'
        os.makedirs(log_dir, exist_ok=True)
        log_name = model_name + "_10000"
        log_file = os.path.join(log_dir, f"training_log_{log_name}_MI.csv")

        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Scenario','Epoch','I(X;T)-InAug', 'I(T;Y)-InAug', 'I(X;T)-In', 'I(T;Y)-In','I(X;T)-Out', 'I(T;Y)-Out']) #,

        # Evaluation loop
        value_xt_in, value_ty_in = evaluate2(net, in_sample_loader, device)

        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([log_name, 99,
                                0, 0,
                                value_xt_in, value_ty_in,
                                0, 0
                                ])


def main_epochs(
        model_seed, ft_seed, r, yaml_file_path,
        in_sizes=None,
        num_intervals_list=None,
        record_verbose=True,
        master_csv_path="./saved_logs/ft_vanilla/MI_master_table.csv",
        verbose_dir="./saved_logs/ft_vanilla/MI_verbose",
        subset_seed=42,
):
    """Fine-tuned model MI evaluation across epochs, in_sizes, and bin counts."""
    if in_sizes is None:
        in_sizes = [5000, 10000, 15000, 20000, 25000]
    if num_intervals_list is None:
        num_intervals_list = [50, 75, 100, 125, 150]

    print(f"Device: {device}")

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_ft_setup(exp_yaml)

    g = torch.Generator()
    g.manual_seed(ft_seed)

    # ---- Prepare datasets (shared across all strategies and epochs) ----
    print('==> Preparing data..')
    in_sample_set = exp_setup["Dataset"].in_sample_set  # no augmentation
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

    # Build in-sample loaders once (size -> loader)
    in_loaders = {}
    for in_size in in_sizes:
        subset_indices = nested_subsets[in_size].tolist()
        in_sample_subset = exp_setup["Dataset"].subset(
            "train", subset_indices, clean=True,
        )
        in_loaders[in_size] = DataLoader(
            in_sample_subset, batch_size=128, shuffle=False, num_workers=0,
            pin_memory=True,
        )

    out_size = len(out_sample_set)
    out_sample_loader = DataLoader(
        out_sample_set, batch_size=128, shuffle=False, num_workers=0,
        pin_memory=True,
    )
    ensure_master_csv(master_csv_path)

    # =============================================================
    # Loop over strategies and epoch checkpoints
    # =============================================================
    for strategy in STRATEGY:
        model_name = (
            f'{exp_yaml["Scenario_Name"]}_{model_seed}_{round(r, 2)}'
            f'_{strategy}_ftsize={exp_setup["FT_GroupSize"]}_ftseed={ft_seed}'
        )
        print(f"\n========== {model_name} ==========")

        model_dir = './saved_models/ft_vanilla/'
        model_folder = Path(model_dir) / model_name
        ckpt_map = build_epoch_to_ckpt_map(model_dir=model_folder, gap=1)

        for epoch in sorted(ckpt_map.keys()):
            ckpt_path = ckpt_map[epoch]
            print(f"\n--- Strategy={strategy}, Epoch={epoch} ---")

            # Fresh model instance per checkpoint
            net = exp_setup["Model_Factory"]().to(device)
            state = torch.load(ckpt_path, map_location=device)
            net.load_state_dict(state)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # =========================================================
            # PHASE 1: Inference once per loader for this checkpoint
            # =========================================================
            print(f"  ==> [Phase 1] Inference for epoch {epoch}")

            # Out-sample: infer once, reuse for all bin counts
            print(f"  Inferring out-sample (size={out_size})...")
            out_layer_T, out_labels = collect_logits(net, out_sample_loader, device)
            print(f"    -> cached logits shape: {tuple(out_layer_T.shape)}")

            in_cache = {}
            for in_size in in_sizes:
                print(f"    Inferring in-sample (size={in_size})...")
                layer_T, labels = collect_logits(net, in_loaders[in_size], device)
                in_cache[in_size] = (layer_T, labels)
                print(f"      -> cached logits shape: {tuple(layer_T.shape)}")

            # =========================================================
            # PHASE 2: MI computation on cached logits
            # =========================================================
            print(f"  ==> [Phase 2] MI computation for epoch {epoch}")

            # Out-sample MI for each bin count
            out_results = {}
            for nb in num_intervals_list:
                ixt_out, ity_out = mi_from_logits(
                    out_layer_T, out_labels, num_intervals=nb, verbose=False,
                )
                print(f"  out  bins={nb:>3}: I(X;T)={ixt_out:.4f}, "
                    f"I(T;Y)={ity_out:.4f}")
            out_results[nb] = (ixt_out, ity_out)

            # In-sample MI + CSV row writing
            for in_size in in_sizes:
                layer_T, labels = in_cache[in_size]
                for nb in num_intervals_list:
                    ixt_in, ity_in = mi_from_logits(
                        layer_T, labels, num_intervals=nb, verbose=False,
                    )
                    print(f"    in   size={in_size:>5}  bins={nb:>3}: "
                        f"I(X;T)={ixt_in:.4f}, I(T;Y)={ity_in:.4f}")

                    row = {
                        "Scenario": exp_yaml["Scenario_Name"],
                        "seed": ft_seed,
                        "rate": round(r, 2),
                        "model_name": model_name,
                        "epoch": epoch,
                        "bins": nb,
                        "in_size": in_size,
                        "I(X;T)-In": f"{ixt_in:.6f}",
                        "I(T;Y)-In": f"{ity_in:.6f}",
                        "out_size":    out_size,
                        "I(X;T)-Out":  f"{ixt_out:.6f}",
                        "I(T;Y)-Out":  f"{ity_out:.6f}",
                        "timestamp": timestamp,
                    }
                    append_master_row(master_csv_path, row)

            # Free per-epoch caches
            del in_cache, net
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n==> Done. Master CSV: {master_csv_path}")



def main_epochs_deit(
        model_seed, ft_seed, r, yaml_file_path,
        in_sizes=None,
        num_intervals_list=None,
        record_verbose=True,
        master_csv_path="./saved_logs/ft_vanilla/MI_master_table.csv",
        verbose_dir="./saved_logs/ft_vanilla/MI_verbose",
        subset_seed=42,
):
    """
    DeiT 版本:对 fine-tune 后的 DeiT,逐 epoch checkpoint 计算 in-sample MI。
    与 CNN 版 main_epochs 的差异:
      - 用 process_experiment_ft_setup_deit(工厂构建 DeiT)
      - model_dir 指向 ft_vanilla/Transformer_Models
      - torch.load 加 weights_only=False
      - collect_logits 直接复用(对 plain DeiT 天然兼容,无需 DeiT 专用版)
    """
    if in_sizes is None:
        in_sizes = [5000, 10000, 15000, 20000, 25000]
    if num_intervals_list is None:
        num_intervals_list = [50, 75, 100, 125, 150]
 
    print(f"Device: {device}")
 
    exp_yaml = process_yaml_file(yaml_file_path)
    # ---- 差异:DeiT 的 ft setup(工厂建 DeiT)----
    exp_setup = process_experiment_ft_setup_deit(exp_yaml)
 
    g = torch.Generator()
    g.manual_seed(ft_seed)
 
    # ---- 数据准备(所有 strategy / epoch 共用)----
    print('==> Preparing data..')
    in_sample_set = exp_setup["Dataset"].in_sample_set  # no augmentation
 
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
 
    # in-sample loaders 一次性构建(size -> loader)
    in_loaders = {}
    for in_size in in_sizes:
        subset_indices = nested_subsets[in_size].tolist()
        in_sample_subset = exp_setup["Dataset"].subset(
            "train", subset_indices, clean=True,
        )
        in_loaders[in_size] = DataLoader(
            in_sample_subset, batch_size=128, shuffle=False, num_workers=0,
            pin_memory=True,
        )
 
    ensure_master_csv(master_csv_path)
 
    # =============================================================
    # 遍历 strategy 和 epoch checkpoint
    # =============================================================
    for strategy in STRATEGY:
        model_name = (
            f'{exp_yaml["Scenario_Name"]}_{model_seed}_{round(r, 2)}'
            f'_{strategy}_ftsize={exp_setup["FT_GroupSize"]}_ftseed={ft_seed}'
        )
        print(f"\n========== {model_name} ==========")
 
        # ---- 差异:DeiT 的 ft 模型在 Transformer_Models 目录 ----
        model_dir = './saved_models/ft_vanilla/Transformer_Models/'
        model_folder = Path(model_dir) / model_name
        ckpt_map = build_epoch_to_ckpt_map(model_dir=model_folder, gap=1)
 
        for epoch in sorted(ckpt_map.keys()):
            ckpt_path = ckpt_map[epoch]
            print(f"\n--- Strategy={strategy}, Epoch={epoch} ---")
 
            # 每个 checkpoint 新建模型实例
            net = exp_setup["Model_Factory"]().to(device)
            # ---- 差异:weights_only=False + 解包 ----
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            elif isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            net.load_state_dict(state)
 
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
 
            # =========================================================
            # PHASE 1: 每个 loader 推理一次(缓存 logits)
            # =========================================================
            print(f"  ==> [Phase 1] Inference for epoch {epoch}")
 
            in_cache = {}
            for in_size in in_sizes:
                print(f"    Inferring in-sample (size={in_size})...")
                # collect_logits 直接复用:plain DeiT eval 返回单张量,
                # 其内部 isinstance(outputs, tuple) 防御对 distilled 也安全
                layer_T, labels = collect_logits(net, in_loaders[in_size], device)
                in_cache[in_size] = (layer_T, labels)
                print(f"      -> cached logits shape: {tuple(layer_T.shape)}")
 
            # =========================================================
            # PHASE 2: 在缓存 logits 上算 MI
            # =========================================================
            print(f"  ==> [Phase 2] MI computation for epoch {epoch}")
 
            for in_size in in_sizes:
                layer_T, labels = in_cache[in_size]
                for nb in num_intervals_list:
                    if record_verbose:
                        ixt_in, ity_in, dbg_in = mi_from_logits(
                            layer_T, labels, num_intervals=nb, verbose=True,
                        )
                        path_in = save_verbose_data(
                            dbg_in, verbose_dir, model_name, "in", nb,
                            in_size=in_size, epoch=epoch,
                        )
                        print(f"    in   size={in_size:>5}  bins={nb:>3}: "
                              f"I(X;T)={ixt_in:.4f}, I(T;Y)={ity_in:.4f}  "
                              f"-> {path_in.name}")
                    else:
                        ixt_in, ity_in = mi_from_logits(
                            layer_T, labels, num_intervals=nb, verbose=False,
                        )
                        print(f"    in   size={in_size:>5}  bins={nb:>3}: "
                              f"I(X;T)={ixt_in:.4f}, I(T;Y)={ity_in:.4f}")
 
                    row = {
                        "Scenario": exp_yaml["Scenario_Name"],
                        "seed": ft_seed,
                        "rate": round(r, 2),
                        "model_name": model_name,
                        "epoch": epoch,
                        "bins": nb,
                        "in_size": in_size,
                        "I(X;T)-In": f"{ixt_in:.6f}",
                        "I(T;Y)-In": f"{ity_in:.6f}",
                        "timestamp": timestamp,
                    }
                    append_master_row(master_csv_path, row)
 
            # 释放本 epoch 缓存
            del in_cache, net
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
 
    print(f"\n==> Done. Master CSV: {master_csv_path}")
                

# =====================================================
# Best-checkpoint evaluation: same MI and subsets as MI_check.py.
# Kept separate from the historical per-epoch CSV schema above.
# =====================================================
BEST_CSV_COLUMNS = [
    "Scenario", "seed", "rate", "model_name", "epoch", "bins", "in_size",
    "I(X;T)-In", "I(T;Y)-In", "out_size", "I(X;T)-Out", "I(T;Y)-Out",
    "timestamp", "model_seed", "ft_seed", "strategy", "ft_size",
    "training_size", "in_size_rate", "family", "subset_seed", "group_seed",
    "checkpoint", "checkpoint_sha256", "plan_sha256",
]


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _missing_best_grid(csv_path, identity, in_sizes, bins):
    """Reject mixed checkpoint/configuration identities before reusing MI."""
    path = Path(csv_path)
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != BEST_CSV_COLUMNS:
                raise ValueError(f"Incompatible best-MI CSV header: {path}; use a separate CSV")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Incomplete/malformed MI row in {path}")
                if row["model_name"] != identity["model_name"]:
                    continue
                for key, expected in identity.items():
                    if row[key] != str(expected):
                        raise ValueError(
                            f"Best-MI identity mismatch: {identity['model_name']}, {key}; "
                            "preserve the existing CSV and use a new output for changed checkpoints/configuration"
                        )
                size = float(row["in_size"])
                fraction = float(row["in_size_rate"])
                if not (0 < size <= identity["training_size"] and np.isfinite(fraction)
                        and abs(fraction - size / identity["training_size"]) < 1e-12):
                    raise ValueError(f"Invalid training-size fraction: {identity['model_name']}")
    return missing_mi_grid(
        path, identity["model_name"], identity["Scenario"], identity["ft_seed"],
        identity["rate"], in_sizes, bins,
    )


def _append_best_row(csv_path, row):
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("x", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=BEST_CSV_COLUMNS).writeheader()
    with path.open("a", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=BEST_CSV_COLUMNS).writerow(row)


def main_best(
        model_seed, ft_seed, r, yaml_file_path,
        in_sizes=None, num_intervals_list=None, record_verbose=None,
        master_csv_path=None, verbose_dir=None, subset_seed=None,
        model_dir=None, strategies=None, in_size_rates=None, family="auto",
        missing="skip",
        index_dir=None, skip_existing=True,
):
    """Record best_epoch.pth In/Out MI for the requested FT strategies.

    Defaults are the editable BEST_* globals. Rate denominators are the
    original Dataset.group_size, NOT FT_Dataset.group_size. Each missing
    (in_size, bins) cell is appended once; matching Out MI is reused by bin.
    `seed` retains the old FT-table convention (ft_seed); both seeds also
    have explicit columns. `epoch=best` does not invent an unknown epoch.
    Returns the number of newly appended rows. This is a single-writer CSV.
    """
    exp_yaml = process_yaml_file(yaml_file_path)
    training_size = exp_yaml["Dataset"]["group_size"]
    ft_size = exp_yaml["FT_Dataset"]["group_size"]
    sizes_from_rates(training_size, [1.0])  # validate even for explicit sizes
    if type(ft_size) is not int or ft_size <= 0:
        raise ValueError("FT_Dataset.group_size must be a positive integer")
    if in_sizes is not None and in_size_rates is not None:
        raise ValueError("Choose in_sizes or in_size_rates, not both")
    if in_sizes is None:
        in_sizes = sizes_from_rates(
            training_size, BEST_IN_SIZE_RATES if in_size_rates is None else in_size_rates,
        )
    in_sizes = positive_ints(in_sizes, "in_sizes")
    if max(in_sizes) > training_size:
        raise ValueError("in_sizes cannot exceed the original training group_size")
    bins = positive_ints(BEST_BINS if num_intervals_list is None else num_intervals_list, "bins")
    strategies = list(BEST_STRATEGIES if strategies is None else strategies)
    if not strategies or len(set(strategies)) != len(strategies) or any(
            strategy not in {"FT-LL", "FT-AL", "RT-AL"} for strategy in strategies):
        raise ValueError("strategies must be unique FT-LL / FT-AL / RT-AL entries")
    if family == "auto":
        family = "deit" if isinstance(exp_yaml["Model"], dict) else "cnn"
    if family not in {"cnn", "deit"}:
        raise ValueError("family must be auto, cnn or deit")
    for name, value in [("model_seed", model_seed), ("ft_seed", ft_seed)]:
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not np.isfinite(r) or not 0 <= r <= 1:
        raise ValueError("r must be a finite fraction in [0, 1]")
    rate = round(float(r), 2)
    model_dir = Path(BEST_MODEL_DIR if model_dir is None else model_dir)
    csv_path = Path(BEST_MASTER_CSV if master_csv_path is None else master_csv_path)
    verbose_dir = Path(BEST_VERBOSE_DIR if verbose_dir is None else verbose_dir)
    index_dir = Path(BEST_INDEX_DIR if index_dir is None else index_dir)
    subset_seed = BEST_SUBSET_SEED if subset_seed is None else subset_seed
    record_verbose = BEST_RECORD_VERBOSE if record_verbose is None else record_verbose
    scenario = exp_yaml["Scenario_Name"]
    plan_hash = _file_sha256(yaml_file_path)

    if missing not in {"skip", "raise"}:
        raise ValueError('missing must be "skip" or "raise"')
    # `missing` is rebound below by the _missing_*_grid call, so keep the
    # policy in its own name -- otherwise missing="raise" stops raising
    # after the first checkpoint.
    on_missing = missing

    # Preflight every requested strategy before dataset/cache/CSV mutation.
    # A plan x seed grid is routinely wider than what has been trained so far
    # (seeds are run in batches), so an absent checkpoint is REPORTED AND
    # SKIPPED by default rather than aborting the sweep - same convention as
    # calculate_MI_at.py. missing="raise" refuses to run a partial set.
    pending = []
    for strategy in strategies:
        name = f"{scenario}_{model_seed}_{rate}_{strategy}_ftsize={ft_size}_ftseed={ft_seed}"
        checkpoint = model_dir / name / "best_epoch.pth"
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            if on_missing == "raise":
                raise FileNotFoundError(f"Missing/empty best checkpoint: {checkpoint}")
            print(f"[ABSENT] {name}: no best_epoch.pth under {model_dir}; skipping")
            continue
        identity = dict(
            Scenario=scenario, seed=ft_seed, rate=rate, model_name=name, epoch="best",
            model_seed=model_seed, ft_seed=ft_seed, strategy=strategy, ft_size=ft_size,
            training_size=training_size, family=family, subset_seed=subset_seed,
            group_seed=42, checkpoint=str(checkpoint.resolve()),
            checkpoint_sha256=_file_sha256(checkpoint), plan_sha256=plan_hash,
        )
        missing, existing_out = _missing_best_grid(csv_path, identity, in_sizes, bins)
        if not skip_existing and len(missing) != len(in_sizes) * len(bins):
            raise ValueError("skip_existing=False would duplicate MI rows; use a new CSV")
        if missing:
            pending.append((checkpoint, identity, missing, existing_out))
        else:
            print(f"[SKIP] {name}: all requested best-MI cells already exist")
    if not pending:
        return 0

    set_seed(ft_seed)
    # MI probes use the original dataset. Do not construct the FT/pseudo-label
    # training dataset just to load an already-trained model.
    mi_yaml = copy.deepcopy(exp_yaml)
    mi_yaml.pop("FT_Dataset", None)
    if family == "deit":
        mi_yaml["Model"]["pretrained"] = False
    setup_fn = process_experiment_ft_setup_deit if family == "deit" else process_experiment_ft_setup
    setup = setup_fn(mi_yaml)
    dataset = setup["Dataset"]
    idx_dir = index_dir / exp_yaml["Dataset"]["name"]
    group_a = create_or_load_group_A(
        dataset=dataset.in_sample_set, save_dir=idx_dir,
        group_size=training_size, num_classes=setup["NumClasses"], seed=42, force_rebuild=False,
    )
    subsets = create_nested_balanced_subsets(
        dataset.in_sample_set, group_a, idx_dir, in_sizes,
        num_classes=setup["NumClasses"], seed=subset_seed, force_rebuild=False,
    )
    out_size = len(dataset.test_set)

    def loader(data):
        return DataLoader(
            data, batch_size=BEST_BATCH_SIZE, shuffle=False, num_workers=BEST_NUM_WORKERS,
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
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if _file_sha256(checkpoint) != identity["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed during loading: {checkpoint}; rerun after training finishes")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        net = setup["Model_Factory"]().to(device)
        net.load_state_dict(state, strict=True)
        del state
        net.eval()
        print(f"[BEST] {name}: {len(missing)} missing MI cells on {device}")

        def measure(logits, labels, nb, split, size=None):
            result = mi_from_logits(logits, labels, num_intervals=nb, verbose=record_verbose)
            if not np.isfinite(result[:2]).all():
                raise ValueError(f"Nonfinite MI: {name}, {split}, {size}, bins={nb}")
            if record_verbose:
                save_verbose_data(
                    result[2], verbose_dir, name, split, nb, in_size=size, epoch="best",
                )
            return result[:2]

        try:
            out_results = {b: existing_out[b][:2] for b in needed_bins if b in existing_out}
            new_out_bins = [b for b in needed_bins if b not in existing_out]
            if new_out_bins:
                out_logits, out_labels = collect_logits(net, loader(dataset.test_set), device)
                for nb in new_out_bins:
                    out_results[nb] = measure(out_logits, out_labels, nb, "out")
                del out_logits, out_labels
            for size in sorted({s for s, _ in missing}):
                probe = dataset.subset("train", subsets[size].tolist(), clean=True)
                logits, labels = collect_logits(net, loader(probe), device)
                for nb in [b for s, b in missing if s == size]:
                    ixt_in, ity_in = measure(logits, labels, nb, "in", size)
                    ixt_out, ity_out = out_results[nb]
                    row = dict(
                        identity, bins=nb, in_size=size, out_size=out_size,
                        in_size_rate=format(size / training_size, ".12g"),
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    )
                    row.update({"I(X;T)-In": f"{ixt_in:.6f}", "I(T;Y)-In": f"{ity_in:.6f}",
                                "I(X;T)-Out": f"{ixt_out:.6f}", "I(T;Y)-Out": f"{ity_out:.6f}"})
                    _append_best_row(csv_path, row)
                    written += 1
                    print(f"  size={size} bins={nb}: In=({ixt_in:.6f}, {ity_in:.6f}) "
                          f"Out=({ixt_out:.6f}, {ity_out:.6f})")
                del logits, labels
        finally:
            del net
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"[DONE] Appended {written} best-MI rows: {csv_path}")
    return written


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    yaml_files = sorted(Path(BEST_PLAN_DIR).glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No YAML experiment plans in {BEST_PLAN_DIR}")
    for _k in ("FT_PLAN_DIR", "SEED_START", "SEED_END"):
        if _k in os.environ:
            print(f"[ENV] {_k}={os.environ[_k]}")
    print(f"Plans: {len(yaml_files)} in {BEST_PLAN_DIR}")
    print(f"ft_seeds: {BEST_FT_SEEDS}   strategies: {BEST_STRATEGIES}")
    print("Untrained cells are reported as [ABSENT] and skipped "
          "(set missing='raise' to refuse a partial set).")
    total = 0
    for yaml_path in yaml_files:
        for model_seed in BEST_MODEL_SEEDS:
            for ft_seed in BEST_FT_SEEDS:
                for rate in BEST_RATES:
                    total += main_best(model_seed, ft_seed, rate, yaml_path)
    print(f"\n==> All done. {total} new row(s) in {BEST_MASTER_CSV}")
