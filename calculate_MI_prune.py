import os
import glob
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
from util import create_or_load_group_A, load_best_checkpoint, load_last_checkpoint, process_yaml_file, process_experiment_prune_setup, create_train_subset, evaluate2, prepare_group_subset, check_pruned_weights,\
                 load_checkpoint_from_epoch, prune_model_global, remove_prune_mask, build_epoch_to_ckpt_map,\
                 sparsity_levels_from_setup   # sparsity 档位以 YAML plan 为准
from Dataset.CIFAR_10 import CIFAR10Dataset
from Dataset.CIFAR_100 import CIFAR100Dataset
from Model.ResNet_18 import ResNet18
from Model.VGG16 import ModifiedVGG16
from MI_check import append_master_row, ensure_master_csv, create_nested_balanced_subsets, collect_logits, mi_from_logits
from datetime import datetime

# ---- Extra imports used only by main_prune_best (the best-checkpoint path below) ----
import copy
import hashlib
from util import process_experiment_prune_setup_deit
from MI_check import POOL0_IN_SIZE_RATES, POOL0_BINS
from mi_pool_support import missing_mi_grid, sizes_from_rates, positive_ints

BASE_DIR = Path(__file__).resolve().parent

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

# ---------------- CONFIGURATION ----------------
STRATEGY = ["FT-AL"] #"RT-AL", "FT-LL", 
SPARSITY = [0.2, 0.4, 0.6, 0.8]

# =====================================================
# 3. Main training logic
# =====================================================
def main_last(model_seed, ft_seed, r, yaml_file_path): 
    print(device)

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_prune_setup(exp_yaml)

    g = torch.Generator()
    g.manual_seed(ft_seed)

    print('==> Preparing data..')
    in_sample_set = exp_setup["Dataset"].in_sample_set # No data augmentation here

    # Important: the in sample evaluation set should be the same each round
    group_A = create_or_load_group_A(dataset=in_sample_set, save_dir=f'./Indices/{exp_yaml['Dataset']['name']}/',
                                    group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"], seed=42, force_rebuild=False)
    in_sample_subset = exp_setup["Dataset"].subset("train", group_A, clean=True)

    in_sample_loader = DataLoader(in_sample_subset,batch_size=128,shuffle=False,num_workers=8, # in sample without augmentation
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, pin_memory=True)

    for s in sparsity_levels_from_setup(exp_setup, SPARSITY):
        s_key = round(float(s), 6)

        for strategy in STRATEGY:
            scenario_name = (
                f'{exp_yaml["Scenario_Name"]}_{model_seed}_{round(r, 2)}'
                f'_sparsity={s_key}_{strategy}_{ft_seed}'
            )
            print(scenario_name)

            model_dir = "./saved_models/prune_vanilla/"
            model_folder = Path(model_dir) / scenario_name

            log_dir = './saved_logs/prune_vanilla/MI'
            os.makedirs(log_dir, exist_ok=True)
            log_name = scenario_name
            log_file = os.path.join(log_dir, f"training_log_{log_name}_MI.csv")

            if not os.path.exists(log_file):
                with open(log_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Scenario','Epoch','I(X;T)-InAug', 'I(T;Y)-InAug', 'I(X;T)-In', 'I(T;Y)-In','I(X;T)-Out', 'I(T;Y)-Out']) #,

            # -----------------------------
            # (1): Pre-FT Model MI Evaluation
            # -----------------------------
            pre_ckpt_path = model_folder / "epoch_-1.pth"
            pre_net = exp_setup["Model_Factory"]().to(device)
            pre_state = torch.load(pre_ckpt_path, map_location=device)
            pre_net.load_state_dict(pre_state)
            check_pruned_weights(pre_net)

            value_xt_in, value_ty_in = evaluate2(pre_net, in_sample_loader, device)
    
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([log_name, -1,
                                    0, 0,
                                    value_xt_in, value_ty_in,
                                    0, 0
                                    ])
                
            # -----------------------------
            # (2): FT Model MI Evaluation
            # -----------------------------
            ckpt_path = load_last_checkpoint(model_folder)
            if ckpt_path is None:
                raise FileNotFoundError(f"No .pth files found in {model_folder}")
            
            # For the revised pruning pipeline, saved checkpoints are mask-removed.
            # So load them into a fresh plain model WITHOUT pruning again.
            net = exp_setup["Model_Factory"]().to(device)
            state = torch.load(ckpt_path, map_location=device)
            net.load_state_dict(state)
            check_pruned_weights(net)

            print(
                f"[INFO] Calculating MI for model_seed={model_seed}, "
                f"ft_seed={ft_seed}, sparsity={s_key}, strategy={strategy}, "
                f"checkpoint={ckpt_path.name}"
            )

            value_xt_in, value_ty_in = evaluate2(net, in_sample_loader, device)
    
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([log_name, 99,
                                    0, 0,
                                    value_xt_in, value_ty_in,
                                    0, 0
                                    ])
            

def main_multiple_epochs(
        model_seed, ft_seed, r, yaml_file_path,
        in_sizes=None,
        num_intervals_list=None,
        record_verbose=True,
        master_csv_path="./saved_logs/pruning_vanilla/MI_master_table_pruning.csv",
        verbose_dir="./saved_logs/pruning_vanilla/MI_verbose",
        subset_seed=42,
):
    """
    Pruned/Fine-tuned model MI evaluation across sparsities, strategies, epochs, 
    in_sizes, and bin counts using efficient Phase 1 (Cache) / Phase 2 (Compute) logic.
    """
    if in_sizes is None:
        in_sizes = [5000, 10000, 15000, 20000, 25000]
    if num_intervals_list is None:
        num_intervals_list = [50, 75, 100, 125, 150]

    print(f"Device: {device}")

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_prune_setup(exp_yaml)

    g = torch.Generator()
    g.manual_seed(ft_seed)

    # =============================================================
    # Prepare datasets & DataLoaders (shared across sparsities, strategies, and epochs)
    # =============================================================
    print('==> Preparing data..')
    in_sample_set = exp_setup["Dataset"].in_sample_set # No data augmentation here
    # Assuming test_set is accessible similarly to ft_setup
    out_sample_set = exp_setup["Dataset"].test_set 

    # Important: the in sample evaluation set should be the same each round
    group_A = create_or_load_group_A(
        dataset=in_sample_set, 
        save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
        group_size=exp_setup["GroupSize"], 
        num_classes=exp_setup["NumClasses"], 
        seed=42, force_rebuild=False
    )
    nested_subsets = create_nested_balanced_subsets(
        dataset=in_sample_set, group_A=group_A,
        save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
        subset_sizes=in_sizes,
        num_classes=exp_setup["NumClasses"],
        seed=subset_seed, force_rebuild=False,
    )

    # Build in-sample loaders once (in_size -> loader)
    in_loaders = {}
    for in_size in in_sizes:
        subset_indices = nested_subsets[in_size].tolist()
        in_sample_subset = exp_setup["Dataset"].subset("train", subset_indices, clean=True)
        in_loaders[in_size] = DataLoader(
            in_sample_subset, batch_size=128, shuffle=False, num_workers=4, 
            worker_init_fn=seed_worker, generator=g, persistent_workers=True, pin_memory=True
        )

    out_size = len(out_sample_set)
    out_sample_loader = DataLoader(
        out_sample_set, batch_size=128, shuffle=False, num_workers=0,
        pin_memory=True,
    )

    ensure_master_csv(master_csv_path)

    # =============================================================
    # Loop over Sparsity, Strategies and Epoch Checkpoints
    # =============================================================
    for s in sparsity_levels_from_setup(exp_setup, SPARSITY):
        s_key = round(float(s), 6)

        for strategy in STRATEGY:
            scenario_name = (
                f'{exp_yaml["Scenario_Name"]}_{model_seed}_{round(r, 2)}'
                f'_sparsity={s_key}_{strategy}_ftsize={exp_setup["FT_GroupSize"]}_ftseed={ft_seed}'
            )
            print(f"\n========== {scenario_name} ==========")

            model_dir = "./saved_models/pruning_vanilla/"
            model_folder = Path(model_dir) / scenario_name
            
            # Use gap=3 as in your original function
            ckpt_map = build_epoch_to_ckpt_map(model_dir=model_folder, gap=1) # gap

            for epoch in sorted(ckpt_map.keys()):
                ckpt_path = ckpt_map[epoch]

                print(
                    f"\n--- Strategy={strategy}, Sparsity={s_key}, Epoch={epoch} ---\n"
                    f"[INFO] Checkpoint={ckpt_path.name}"
                )

                # Fresh model instance per checkpoint
                net = exp_setup["Model_Factory"]().to(device)
                state = torch.load(ckpt_path, map_location=device)
                net.load_state_dict(state)
                check_pruned_weights(net)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # =========================================================
                # PHASE 1: Inference once per loader for this checkpoint
                # =========================================================
                print(f"  ==> [Phase 1] Inference for epoch {epoch}")

                # Out-sample: infer once, reuse for all bin counts
                print(f"  Inferring out-sample (size={out_size})...")
                with torch.no_grad():
                    out_layer_T, out_labels = collect_logits(net, out_sample_loader, device)
                print(f"    -> cached logits shape: {tuple(out_layer_T.shape)}")

                in_cache = {}
                for in_size in in_sizes:
                    print(f"    Inferring in-sample (size={in_size})...")
                    with torch.no_grad():
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
                    print(f"  out  bins={nb:>3}: I(X;T)={ixt_out:.4f}, I(T;Y)={ity_out:.4f}")
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

                        # Retrieve matched out-sample MI
                        ixt_out, ity_out = out_results[nb]

                        row = {
                            "Scenario": exp_yaml["Scenario_Name"],
                            "seed": ft_seed,
                            "rate": round(r, 2),
                            "model_name": scenario_name, # note: sparsity is captured here
                            "epoch": epoch,
                            "bins": nb,
                            "in_size": in_size,
                            "I(X;T)-In": f"{ixt_in:.6f}",
                            "I(T;Y)-In": f"{ity_in:.6f}",
                            "out_size": out_size,
                            "I(X;T)-Out": f"{ixt_out:.6f}",
                            "I(T;Y)-Out": f"{ity_out:.6f}",
                            "timestamp": timestamp,
                        }
                        append_master_row(master_csv_path, row)

                # Free per-epoch caches to keep memory utilization low
                del in_cache, out_layer_T, out_labels, net
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"\n==> Done. Master CSV: {master_csv_path}")


def main_best_epoch(
        model_seed, ft_seed, r, yaml_file_path,
        in_sizes=None,
        num_intervals_list=None,
        record_verbose=True,
        master_csv_path="./saved_logs/pruning_vanilla/MI_master_table_pruning_best.csv",
        verbose_dir="./saved_logs/pruning_vanilla/MI_verbose",
        subset_seed=42,
):
    """
    Pruned/Fine-tuned model MI evaluation ONLY for the BEST epoch ('best_clean_epoch.pth') 
    across sparsities, strategies, in_sizes, and bin counts.
    """
    if in_sizes is None:
        in_sizes = [5000, 10000, 15000, 20000, 25000]
    if num_intervals_list is None:
        num_intervals_list = [50, 75, 100, 125, 150]

    print(f"Device: {device}")

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_prune_setup(exp_yaml)

    g = torch.Generator()
    g.manual_seed(ft_seed)

    # =============================================================
    # Prepare datasets & DataLoaders (shared across sparsities, strategies)
    # =============================================================
    print('==> Preparing data..')
    in_sample_set = exp_setup["Dataset"].in_sample_set # No data augmentation here
    out_sample_set = exp_setup["Dataset"].test_set 

    # Important: the in sample evaluation set should be the same each round
    group_A = create_or_load_group_A(
        dataset=in_sample_set, 
        save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
        group_size=exp_setup["GroupSize"], 
        num_classes=exp_setup["NumClasses"], 
        seed=42, force_rebuild=False
    )
    nested_subsets = create_nested_balanced_subsets(
        dataset=in_sample_set, group_A=group_A,
        save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
        subset_sizes=in_sizes,
        num_classes=exp_setup["NumClasses"],
        seed=subset_seed, force_rebuild=False,
    )

    # Build in-sample loaders once (in_size -> loader)
    in_loaders = {}
    for in_size in in_sizes:
        subset_indices = nested_subsets[in_size].tolist()
        in_sample_subset = exp_setup["Dataset"].subset("train", subset_indices, clean=True)
        in_loaders[in_size] = DataLoader(
            in_sample_subset, batch_size=128, shuffle=False, num_workers=0, # Set to 0 to avoid Windows MP issues if needed
            worker_init_fn=seed_worker, generator=g, persistent_workers=False, pin_memory=True
        )

    out_size = len(out_sample_set)
    out_sample_loader = DataLoader(
        out_sample_set, batch_size=128, shuffle=False, num_workers=0,
        pin_memory=True,
    )

    ensure_master_csv(master_csv_path)

    # =============================================================
    # Loop over Sparsity and Strategies (ONLY evaluate best epoch)
    # =============================================================
    for s in sparsity_levels_from_setup(exp_setup, SPARSITY):
        s_key = round(float(s), 6)

        for strategy in STRATEGY:
            scenario_name = (
                f'{exp_yaml["Scenario_Name"]}_{model_seed}_{round(r, 2)}'
                f'_sparsity={s_key}_{strategy}_ftsize={exp_setup["FT_GroupSize"]}_ftseed={ft_seed}'
            )
            print(f"\n========== Evaluating BEST Epoch for {scenario_name} ==========")

            model_dir = "./saved_models/pruning_vanilla/"
            model_folder = Path(model_dir) / scenario_name
            
            ckpt_path, _ = load_best_checkpoint(model_folder, filename="best_clean_epoch.pth")
            if ckpt_path is None:
                print(f"[⚠️] No .pth files found in {model_folder}")
                return

            epoch_label = "best"

            print(
                f"\n--- Strategy={strategy}, Sparsity={s_key} ---\n"
                f"[INFO] Loading Checkpoint={ckpt_path.name}"
            )

            # Fresh model instance
            net = exp_setup["Model_Factory"]().to(device)
            state = torch.load(ckpt_path, map_location=device)
            net.load_state_dict(state)
            check_pruned_weights(net)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # =========================================================
            # PHASE 1: Inference once per loader for the best checkpoint
            # =========================================================
            print(f"  ==> [Phase 1] Inference for {epoch_label}")

            # Out-sample: infer once, reuse for all bin counts
            print(f"  Inferring out-sample (size={out_size})...")
            with torch.no_grad():
                out_layer_T, out_labels = collect_logits(net, out_sample_loader, device)
            print(f"    -> cached logits shape: {tuple(out_layer_T.shape)}")

            in_cache = {}
            for in_size in in_sizes:
                print(f"    Inferring in-sample (size={in_size})...")
                with torch.no_grad():
                    layer_T, labels = collect_logits(net, in_loaders[in_size], device)
                in_cache[in_size] = (layer_T, labels)
                print(f"      -> cached logits shape: {tuple(layer_T.shape)}")

            # =========================================================
            # PHASE 2: MI computation on cached logits
            # =========================================================
            print(f"  ==> [Phase 2] MI computation for {epoch_label}")

            # Out-sample MI for each bin count
            out_results = {}
            for nb in num_intervals_list:
                ixt_out, ity_out = mi_from_logits(
                    out_layer_T, out_labels, num_intervals=nb, verbose=False,
                )
                print(f"  out  bins={nb:>3}: I(X;T)={ixt_out:.4f}, I(T;Y)={ity_out:.4f}")
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

                    # Retrieve matched out-sample MI
                    ixt_out, ity_out = out_results[nb]

                    row = {
                        "Scenario": exp_yaml["Scenario_Name"],
                        "seed": ft_seed,
                        "rate": round(r, 2),
                        "model_name": scenario_name,
                        "epoch": epoch_label, 
                        "bins": nb,
                        "in_size": in_size,
                        "I(X;T)-In": f"{ixt_in:.6f}",
                        "I(T;Y)-In": f"{ity_in:.6f}",
                        "out_size": out_size,
                        "I(X;T)-Out": f"{ixt_out:.6f}",
                        "I(T;Y)-Out": f"{ity_out:.6f}",
                        "timestamp": timestamp,
                    }
                    append_master_row(master_csv_path, row)

            # Free caches
            del in_cache, out_layer_T, out_labels, net
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n==> Done. Master CSV (Best Epoch): {master_csv_path}")


# =====================================================
# Best-checkpoint evaluation for the pruning attack.
#
# Port of calculate_MI_ft.main_best to the prune family, following the same
# house pattern calculate_MI_at.main_at_best uses: editable *_BEST_* globals,
# an identity-carrying CSV with checkpoint/plan hashes, a resume protocol that
# refuses to mix identities, and a preflight that validates every checkpoint
# before touching datasets or the CSV.
#
# Differences from the FT version, all forced by what main_prune.py writes:
#   * the loop has TWO axes (sparsity x strategy), not just strategy, and the
#     sparsity levels come from the YAML via sparsity_levels_from_setup -- the
#     same single source of truth main_prune.py now uses;
#   * checkpoints live under saved_models/pruning_final/ (CNN) or
#     .../pruning_final/Transformer_Models/ (DeiT), NOT pruning_vanilla/;
#   * the checkpoint file is best_epoch.pth (the project-wide convention that
#     main_prune.py now follows), NOT the best_clean_epoch.pth that the older
#     main_best_epoch below still looks for;
#   * two checkpoint kinds are supported. "preft" = epoch_-1.pth, the model
#     right after pruning and before the restoring fine-tune; "best" =
#     best_epoch.pth. Recording both separates "what pruning did to the MI
#     signal" from "what the restoring fine-tune did", which is the whole point
#     of the experiment. Mirrors AT_BEST_CKPT_KINDS.
#   * every loaded checkpoint is verified to actually carry its nominal
#     sparsity. main_prune.py folds the mask into the weights before saving, so
#     a checkpoint's zero fraction must equal the sparsity encoded in its own
#     folder name. If it does not, the scenario name is lying about which model
#     it holds, and every row derived from it would be silently mislabelled.
#
# The older main_last / main_multiple_epochs / main_best_epoch entry points
# above are left untouched; they still read the legacy pruning_vanilla/ tree.
# =====================================================
# PRUNE_PLAN_DIR overrides the plan folder (default saved_exp_plan/prune_plan). main_prune.py reads the same variable, so training and MI stay on one batch.
PRUNE_BEST_PLAN_DIR = Path(os.environ.get("PRUNE_PLAN_DIR", BASE_DIR / "saved_exp_plan/prune_plan"))
PRUNE_BEST_MODEL_DIR = BASE_DIR / "saved_models/pruning_final"
PRUNE_BEST_MASTER_CSV = BASE_DIR / "saved_logs/pruning_final/MI_master_table_prune.csv"
PRUNE_BEST_VERBOSE_DIR = BASE_DIR / "saved_logs/pruning_final/MI_verbose_best"
PRUNE_BEST_INDEX_DIR = BASE_DIR / "Indices"
PRUNE_BEST_MODEL_SEEDS = [42]
# SEED_START / SEED_END mirror main_prune.py. Default 0..4 keeps the five
# ft_seeds the CIFAR-10 / CIFAR-100 ResNet-18 rows were trained with; with
# missing="skip" an untrained seed in that range is reported and passed over.
PRUNE_BEST_FT_SEEDS = list(range(int(os.environ.get("SEED_START", 0)),
                                 int(os.environ.get("SEED_END", 5))))
PRUNE_BEST_RATES = [1.0]
PRUNE_BEST_STRATEGIES = ["FT-AL"]        # pruning only ever fine-tunes all layers
PRUNE_BEST_CKPT_KINDS = ["preft", "best"]
PRUNE_BEST_IN_SIZE_RATES = list(POOL0_IN_SIZE_RATES)
PRUNE_BEST_BINS = list(POOL0_BINS)
PRUNE_BEST_SUBSET_SEED = 42              # same nested subsets as pool / victim / FT
PRUNE_BEST_GROUP_SEED = 42               # group_A seed, fixed upstream
PRUNE_BEST_RECORD_VERBOSE = False
PRUNE_BEST_BATCH_SIZE = 128
PRUNE_BEST_NUM_WORKERS = 0
PRUNE_BEST_SPARSITY_TOL = 0.5            # percentage points, measured vs nominal

# epoch_-1.pth is written by main_prune.py right after prune_model_global and
# before the first fine-tune step; best_epoch.pth is the best restoring-FT epoch.
PRUNE_CKPT_FILES = {"preft": "epoch_-1.pth", "best": "best_epoch.pth"}
PRUNE_CKPT_EPOCH_LABEL = {"preft": "-1", "best": "best"}

PRUNE_BEST_CSV_COLUMNS = [
    "Scenario", "seed", "rate", "model_name", "epoch", "bins", "in_size",
    "I(X;T)-In", "I(T;Y)-In", "out_size", "I(X;T)-Out", "I(T;Y)-Out",
    "timestamp", "model_seed", "ft_seed", "sparsity", "strategy", "ckpt_kind",
    "ft_size", "training_size", "in_size_rate", "family", "subset_seed",
    "group_seed", "achieved_sparsity", "checkpoint", "checkpoint_sha256",
    "plan_sha256",
]


def _prune_file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_verbose_data_prune(debug_data, save_dir, model_name, split_name,
                            num_intervals, in_size=None, epoch=None):
    """Save verbose MI debug data to a structured .npz (same layout as the FT script)."""
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


def measured_sparsity(net, family, extra_exclude=None):
    """Zero fraction (%) over exactly the parameter set prune_model_global targets.

    prune_model_global skips DeiT's patch_embed Conv2d, so the denominator has
    to skip it too -- otherwise a DeiT model pruned to 80% reads back as less.

    extra_exclude mirrors a plan's optional Prune_Exclude list (layers main_prune
    kept dense). Those weights were never candidates, so counting them here
    would dilute the measurement and trip the sparsity tolerance. Omitting it
    reproduces the original behaviour.
    """
    exclude = ["patch_embed"] if family == "deit" else []
    exclude = exclude + list(extra_exclude or [])
    return check_pruned_weights(net, exclude_patterns=exclude or None)


def _missing_prune_best_grid(csv_path, identity, in_sizes, bins):
    """Reject mixed checkpoint/configuration identities before reusing MI."""
    path = Path(csv_path)
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != PRUNE_BEST_CSV_COLUMNS:
                raise ValueError(f"Incompatible prune-MI CSV header: {path}; use a separate CSV")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Incomplete/malformed MI row in {path}")
                if row["model_name"] != identity["model_name"]:
                    continue
                for key, expected in identity.items():
                    if row[key] != str(expected):
                        raise ValueError(
                            f"Prune-MI identity mismatch: {identity['model_name']}, {key}; "
                            "preserve the existing CSV and use a new output for changed "
                            "checkpoints/configuration"
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


def _append_prune_best_row(csv_path, row):
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("x", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=PRUNE_BEST_CSV_COLUMNS).writeheader()
    with path.open("a", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=PRUNE_BEST_CSV_COLUMNS).writerow(row)


def main_prune_best(
        model_seed, ft_seed, r, yaml_file_path,
        in_sizes=None, num_intervals_list=None, record_verbose=None,
        master_csv_path=None, verbose_dir=None, subset_seed=None,
        model_dir=None, strategies=None, sparsities=None, ckpt_kinds=None,
        in_size_rates=None, family="auto", index_dir=None, skip_existing=True,
        missing="skip",
        sparsity_tol=None,
):
    """Record In/Out MI for every (sparsity, strategy, checkpoint kind) of a prune plan.

    Defaults come from the editable PRUNE_BEST_* globals. Rate denominators are
    the original Dataset.group_size, NOT FT_Dataset.group_size -- the MI probe
    measures the victim's training set, which pruning does not change. Each
    missing (in_size, bins) cell is appended once; matching Out MI is reused by
    bin. `seed` keeps the FT-table convention (ft_seed); both seeds also have
    explicit columns. Returns the number of newly appended rows.
    This is a single-writer CSV.
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
            training_size, PRUNE_BEST_IN_SIZE_RATES if in_size_rates is None else in_size_rates,
        )
    in_sizes = positive_ints(in_sizes, "in_sizes")
    if max(in_sizes) > training_size:
        raise ValueError("in_sizes cannot exceed the original training group_size")
    bins = positive_ints(PRUNE_BEST_BINS if num_intervals_list is None else num_intervals_list, "bins")

    strategies = list(PRUNE_BEST_STRATEGIES if strategies is None else strategies)
    if not strategies or len(set(strategies)) != len(strategies) or any(
            strategy not in {"FT-LL", "FT-AL", "RT-AL"} for strategy in strategies):
        raise ValueError("strategies must be unique FT-LL / FT-AL / RT-AL entries")
    if missing not in {"skip", "raise"}:
        raise ValueError('missing must be "skip" or "raise"')
    # `missing` is rebound below by the _missing_*_grid call, so keep the
    # policy in its own name -- otherwise missing="raise" stops raising
    # after the first checkpoint.
    on_missing = missing
    ckpt_kinds = list(PRUNE_BEST_CKPT_KINDS if ckpt_kinds is None else ckpt_kinds)
    if not ckpt_kinds or len(set(ckpt_kinds)) != len(ckpt_kinds) or any(
            kind not in PRUNE_CKPT_FILES for kind in ckpt_kinds):
        raise ValueError(f"ckpt_kinds must be unique entries of {sorted(PRUNE_CKPT_FILES)}")

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

    model_dir = Path(PRUNE_BEST_MODEL_DIR if model_dir is None else model_dir)
    if model_dir == Path(PRUNE_BEST_MODEL_DIR) and family == "deit":
        # main_prune_deit writes one level deeper, exactly like main_ft_deit.
        model_dir = model_dir / "Transformer_Models"
    csv_path = Path(PRUNE_BEST_MASTER_CSV if master_csv_path is None else master_csv_path)
    verbose_dir = Path(PRUNE_BEST_VERBOSE_DIR if verbose_dir is None else verbose_dir)
    index_dir = Path(PRUNE_BEST_INDEX_DIR if index_dir is None else index_dir)
    subset_seed = PRUNE_BEST_SUBSET_SEED if subset_seed is None else subset_seed
    record_verbose = PRUNE_BEST_RECORD_VERBOSE if record_verbose is None else record_verbose
    sparsity_tol = PRUNE_BEST_SPARSITY_TOL if sparsity_tol is None else sparsity_tol
    scenario = exp_yaml["Scenario_Name"]
    plan_hash = _prune_file_sha256(yaml_file_path)
    # Optional plan key, same one main_prune.py honours: layers deliberately
    # kept dense. Needed here only so achieved_sparsity uses main_prune's
    # denominator. Absent -> None -> unchanged behaviour.
    prune_exclude = exp_yaml.get("Prune_Exclude") or None

    # MI probes use the original dataset. Do not build the FT/pseudo-label
    # training set just to load an already-trained model -- for the CIFAR-10
    # plan that would load the 500K pseudo-label pickle for nothing.
    mi_yaml = copy.deepcopy(exp_yaml)
    mi_yaml.pop("FT_Dataset", None)
    if family == "deit":
        mi_yaml["Model"]["pretrained"] = False
    setup_fn = process_experiment_prune_setup_deit if family == "deit" else process_experiment_prune_setup
    setup = setup_fn(mi_yaml)

    # Sparsity levels from the plan, same single source of truth main_prune.py uses.
    if sparsities is None:
        sparsities = sparsity_levels_from_setup(setup, SPARSITY)
    sparsities = [round(float(s), 6) for s in sparsities]
    if not sparsities or len(set(sparsities)) != len(sparsities):
        raise ValueError("sparsities must be a non-empty list of unique levels")
    if any(not 0 <= s < 1 for s in sparsities):
        raise ValueError("each sparsity must be a fraction in [0, 1)")

    # Preflight every (sparsity, strategy, kind) before dataset/cache/CSV mutation.
    pending = []
    for s_key in sparsities:
        for strategy in strategies:
            # Must match main_prune.py / main_prune_deit scenario_name byte for byte.
            name = (f"{scenario}_{model_seed}_{rate}"
                    f"_sparsity={s_key}_{strategy}_ftsize={ft_size}_ftseed={ft_seed}")
            for kind in ckpt_kinds:
                checkpoint = model_dir / name / PRUNE_CKPT_FILES[kind]
                if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                    # A plan x seed grid is routinely wider than what has been
                    # trained so far, so an absent checkpoint is reported and
                    # skipped by default (calculate_MI_at.py's convention).
                    # missing="raise" refuses to run a partial set.
                    if on_missing == "raise":
                        raise FileNotFoundError(
                            f"Missing/empty {kind} checkpoint: {checkpoint}")
                    print(f"[ABSENT] {name} ({kind}): no {PRUNE_CKPT_FILES[kind]} "
                          f"under {model_dir}; skipping")
                    continue
                # One CSV identity per (scenario, sparsity, strategy, kind): the
                # kind is folded into model_name so preft and best never collide
                # in the resume protocol's model_name lookup.
                row_model_name = f"{name}_ckpt={kind}"
                identity = dict(
                    Scenario=scenario, seed=ft_seed, rate=rate, model_name=row_model_name,
                    epoch=PRUNE_CKPT_EPOCH_LABEL[kind], model_seed=model_seed, ft_seed=ft_seed,
                    sparsity=s_key, strategy=strategy, ckpt_kind=kind, ft_size=ft_size,
                    training_size=training_size, family=family, subset_seed=subset_seed,
                    group_seed=PRUNE_BEST_GROUP_SEED, checkpoint=str(checkpoint.resolve()),
                    checkpoint_sha256=_prune_file_sha256(checkpoint), plan_sha256=plan_hash,
                )
                missing, existing_out = _missing_prune_best_grid(csv_path, identity, in_sizes, bins)
                if not skip_existing and len(missing) != len(in_sizes) * len(bins):
                    raise ValueError("skip_existing=False would duplicate MI rows; use a new CSV")
                if missing:
                    pending.append((checkpoint, identity, missing, existing_out))
                else:
                    print(f"[SKIP] {row_model_name}: all requested MI cells already exist")
    if not pending:
        return 0

    set_seed(ft_seed)
    dataset = setup["Dataset"]
    idx_dir = index_dir / exp_yaml["Dataset"]["name"]
    group_a = create_or_load_group_A(
        dataset=dataset.in_sample_set, save_dir=idx_dir,
        group_size=training_size, num_classes=setup["NumClasses"],
        seed=PRUNE_BEST_GROUP_SEED, force_rebuild=False,
    )
    subsets = create_nested_balanced_subsets(
        dataset.in_sample_set, group_a, idx_dir, in_sizes,
        num_classes=setup["NumClasses"], seed=subset_seed, force_rebuild=False,
    )
    out_size = len(dataset.test_set)

    def loader(data):
        return DataLoader(
            data, batch_size=PRUNE_BEST_BATCH_SIZE, shuffle=False,
            num_workers=PRUNE_BEST_NUM_WORKERS,
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
        if _prune_file_sha256(checkpoint) != identity["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed during loading: {checkpoint}; "
                             "rerun after training finishes")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        net = setup["Model_Factory"]().to(device)
        # main_prune.py calls remove_prune_mask before saving, so the checkpoint
        # holds plain `weight` tensors with the pruned entries already zeroed --
        # a fresh unpruned model takes it directly, no re-pruning needed.
        net.load_state_dict(state, strict=True)
        del state
        net.eval()

        # Integrity: the zero fraction must match the sparsity in the folder name.
        achieved = measured_sparsity(net, identity["family"], prune_exclude)
        nominal = 100.0 * identity["sparsity"]
        if not np.isfinite(achieved) or abs(achieved - nominal) > sparsity_tol:
            raise ValueError(
                f"Sparsity mismatch for {name}: checkpoint carries {achieved:.4f}% zeros "
                f"but its scenario name claims {nominal:.4f}% (tol {sparsity_tol}pp). "
                f"Checkpoint: {checkpoint}"
            )
        print(f"[PRUNE-MI] {name}: {len(missing)} missing MI cells on {device} "
              f"(sparsity {achieved:.2f}%)")

        def measure(logits, labels, nb, split, size=None):
            result = mi_from_logits(logits, labels, num_intervals=nb, verbose=record_verbose)
            if not np.isfinite(result[:2]).all():
                raise ValueError(f"Nonfinite MI: {name}, {split}, {size}, bins={nb}")
            if record_verbose:
                save_verbose_data_prune(
                    result[2], verbose_dir, name, split, nb,
                    in_size=size, epoch=identity["epoch"],
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
                        achieved_sparsity=f"{achieved:.6f}",
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    )
                    row.update({"I(X;T)-In": f"{ixt_in:.6f}", "I(T;Y)-In": f"{ity_in:.6f}",
                                "I(X;T)-Out": f"{ixt_out:.6f}", "I(T;Y)-Out": f"{ity_out:.6f}"})
                    _append_prune_best_row(csv_path, row)
                    written += 1
                    print(f"  size={size} bins={nb}: In=({ixt_in:.6f}, {ity_in:.6f}) "
                          f"Out=({ixt_out:.6f}, {ity_out:.6f})")
                del logits, labels
        finally:
            del net
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"[DONE] Appended {written} prune-MI rows: {csv_path}")
    return written


# =====================================================
# 4. Entry point
# =====================================================
# Drives main_prune_best over every plan in PRUNE_BEST_PLAN_DIR. Point that at
# saved_exp_plan/prune_plan_deit to do the DeiT runs -- family, model root and
# setup function are all auto-detected per plan from whether Model is a dict.
#
# Previous entry point (legacy pruning_vanilla/ tree, per-epoch grid):
#     main_multiple_epochs(model_seed, ft_seed, 1.0, yaml_path,
#                          in_sizes=[25000],
#                          num_intervals_list=[5, 10, 15, 20, 30, 50, 75, 100, 150, 200],
#                          record_verbose=False)
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    yaml_files = sorted(Path(PRUNE_BEST_PLAN_DIR).glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No YAML experiment plans in {PRUNE_BEST_PLAN_DIR}")
    print(f"Found {len(yaml_files)} experiment plan(s):")
    for f in yaml_files:
        print(" -", f)

    total = 0
    for yaml_path in yaml_files:
        print(f"\n========== Starting MI evaluation from {yaml_path} ==========")
        for model_seed in PRUNE_BEST_MODEL_SEEDS:
            for ft_seed in PRUNE_BEST_FT_SEEDS:
                for rate in PRUNE_BEST_RATES:
                    print(f"\n>>> model_seed={model_seed}, ft_seed={ft_seed}, rate={rate}")
                    total += main_prune_best(model_seed, ft_seed, rate, yaml_path)
    print(f"\n==> All done. {total} new rows in {PRUNE_BEST_MASTER_CSV}")
