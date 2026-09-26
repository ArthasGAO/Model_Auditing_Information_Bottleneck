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
import os
import csv
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
import torchvision
import torchvision.transforms as transforms
from util import create_or_load_subset_from_group, process_yaml_file, process_experiment_ft_setup, create_train_subset, evaluate2, prepare_group_subset,\
                 load_last_checkpoint, load_checkpoint_from_epoch, build_epoch_to_ckpt_map, create_or_load_group_A
import torch.nn.functional as F

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
    """Standard discrete mutual information formula.

    matrix: joint probability table P(A,B)
    p1: marginal P(A)
    p2: marginal P(B)
    """
    mask = matrix > 0
    denom = p1[:, None] * p2[None, :]

    # Only evaluate entries where joint probability is non-zero.
    # This avoids 0 * log(0) numerical issues.
    ratio = matrix[mask] / denom[mask]
    log_ratio = torch.log2(ratio)
    return (matrix[mask] * log_ratio).sum()


def _entropy_from_probs(probs):
    """Entropy H(P) in bits for a probability vector."""
    mask = probs > 0
    return -(probs[mask] * torch.log2(probs[mask])).sum()


def _print_mi_debug_summary(
        *,
        layer_T,
        label_matrix,
        T_soft,
        T_discrete,
        unique_T,
        inverse_idx,
        T_counts,
        TY_counts,
        I_X_T,
        I_T_Y,
        num_intervals,
        prefix="",
        topk=8,
):
    """Print diagnostic information for checking whether I(T;Y) is saturated.

    The main hypothesis to verify is:
        I(T;Y) == H(Y) because H(Y|T) ~= 0,
    which happens when each discretized T-pattern is label-pure.
    """
    with torch.no_grad():
        device = layer_T.device
        N = layer_T.shape[0]
        K_y = label_matrix.shape[1]
        K_unique = unique_T.shape[0]

        true_y = label_matrix.argmax(dim=1)
        pred_y = T_soft.argmax(dim=1)
        pred_acc = (pred_y == true_y).float().mean()
        conf = T_soft.max(dim=1).values
        entropy_pred = -(T_soft.clamp_min(1e-12) * torch.log2(T_soft.clamp_min(1e-12))).sum(dim=1)

        TY_matrix = TY_counts / N
        P_T_marg = TY_matrix.sum(dim=1)
        P_Y_marg = TY_matrix.sum(dim=0)

        H_Y = _entropy_from_probs(P_Y_marg)
        H_T = _entropy_from_probs(P_T_marg)

        # H(Y|T) = sum_t P(t) H(Y|T=t)
        p_y_given_t = TY_counts / T_counts[:, None].clamp_min(1.0)
        mask = p_y_given_t > 0
        H_Y_given_each_T = torch.zeros(K_unique, device=device)
        H_Y_given_each_T = -(
            p_y_given_t.clamp_min(1e-12) * torch.where(
                mask,
                torch.log2(p_y_given_t.clamp_min(1e-12)),
                torch.zeros_like(p_y_given_t),
            )
        ).sum(dim=1)
        H_Y_given_T = (P_T_marg * H_Y_given_each_T).sum()
        I_T_Y_by_entropy = H_Y - H_Y_given_T

        # Pattern purity diagnostics.
        dominant_counts, dominant_labels = TY_counts.max(dim=1)
        purity = dominant_counts / T_counts.clamp_min(1.0)
        nonzero_label_per_pattern = (TY_counts > 0).sum(dim=1)
        is_pure = nonzero_label_per_pattern == 1
        is_ambiguous = ~is_pure

        pure_pattern_count = is_pure.sum()
        ambiguous_pattern_count = is_ambiguous.sum()
        pure_pattern_ratio = pure_pattern_count.float() / max(K_unique, 1)
        ambiguous_pattern_ratio = ambiguous_pattern_count.float() / max(K_unique, 1)
        ambiguous_sample_mass = T_counts[is_ambiguous].sum() / N

        label_counts = label_matrix.sum(dim=0)
        pred_counts = torch.bincount(pred_y, minlength=K_y).float()

        print("\n" + "=" * 90)
        print(f"[MI-DEBUG] {prefix} bins={num_intervals}, N={N}, K_y={K_y}")
        print("-" * 90)
        print(f"  Unique discretized T patterns U       : {K_unique} / {N}  ({K_unique / N:.6f})")
        print(f"  I(X;T) = H(T) from pattern counts      : {I_X_T.item():.6f}")
        print(f"  H(T) recomputed from TY marginal       : {H_T.item():.6f}")
        print(f"  H(Y)                                  : {H_Y.item():.6f}")
        print(f"  H(Y|T)                                : {H_Y_given_T.item():.8f}")
        print(f"  I(T;Y) direct MI formula               : {I_T_Y.item():.6f}")
        print(f"  I(T;Y) via H(Y)-H(Y|T)                 : {I_T_Y_by_entropy.item():.6f}")
        print(f"  H(Y) - I(T;Y)                          : {(H_Y - I_T_Y).item():.8f}")
        print(f"  |direct - entropy-form|                : {abs(I_T_Y - I_T_Y_by_entropy).item():.10f}")
        print("-" * 90)
        print(f"  Pure T-pattern count                   : {pure_pattern_count.item()} / {K_unique}")
        print(f"  Pure T-pattern ratio                   : {pure_pattern_ratio.item():.6f}")
        print(f"  Ambiguous T-pattern count              : {ambiguous_pattern_count.item()} / {K_unique}")
        print(f"  Ambiguous T-pattern ratio              : {ambiguous_pattern_ratio.item():.6f}")
        print(f"  Ambiguous sample mass                  : {ambiguous_sample_mass.item():.8f}")
        print(f"  Avg nonzero labels per T-pattern       : {nonzero_label_per_pattern.float().mean().item():.6f}")
        print(f"  Max nonzero labels in one T-pattern    : {nonzero_label_per_pattern.max().item()}")
        print("-" * 90)
        print(f"  Prediction accuracy on this loader      : {pred_acc.item():.6f}")
        print(f"  Softmax confidence mean / min / max     : "
              f"{conf.mean().item():.6f} / {conf.min().item():.6f} / {conf.max().item():.6f}")
        print(f"  Softmax entropy mean / min / max        : "
              f"{entropy_pred.mean().item():.6f} / {entropy_pred.min().item():.6f} / {entropy_pred.max().item():.6f}")
        print(f"  True label counts                       : {label_counts.detach().cpu().to(torch.int64).tolist()}")
        print(f"  Pred label counts                       : {pred_counts.detach().cpu().to(torch.int64).tolist()}")

        # Show the largest ambiguous patterns, if any exist.
        if ambiguous_pattern_count.item() > 0:
            print("-" * 90)
            print(f"  Top {topk} ambiguous T-patterns by sample count:")
            amb_idx = torch.where(is_ambiguous)[0]
            order = torch.argsort(T_counts[amb_idx], descending=True)
            chosen = amb_idx[order[:topk]]
            for rank, idx in enumerate(chosen.tolist(), start=1):
                counts = TY_counts[idx]
                nz = torch.where(counts > 0)[0]
                label_mix = {int(c.item()): int(counts[c].item()) for c in nz}
                print(
                    f"    #{rank}: pattern_id={idx}, count={int(T_counts[idx].item())}, "
                    f"purity={purity[idx].item():.6f}, dominant_label={int(dominant_labels[idx].item())}, "
                    f"label_mix={label_mix}, T_bins={unique_T[idx].detach().cpu().tolist()}"
                )
        else:
            print("-" * 90)
            print("  No ambiguous T-patterns found. Every discretized T-pattern maps to one label only.")
            print("  This directly explains I(T;Y) = H(Y) saturation.")
        print("=" * 90 + "\n")


def mi_from_logits(
        layer_T,
        label_matrix,
        num_intervals=50,
        verbose=False,
        debug=False,
        debug_prefix="",
        debug_topk=8,
):
    """Compute (I(X;T), I(T;Y)) from cached logits at a given bin count.

    This function does NOT run the model; it only does softmax + binning
    + counting + entropy/MI computation.

    Args:
        debug: if True, print saturation diagnostics for I(T;Y).
        debug_prefix: text prefix to identify strategy/epoch/in_size in logs.
        debug_topk: number of ambiguous T-patterns to print.

    Returns:
        (I_X_T, I_T_Y) when verbose=False,
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

    if debug:
        _print_mi_debug_summary(
            layer_T=layer_T,
            label_matrix=label_matrix,
            T_soft=T_soft,
            T_discrete=T_discrete,
            unique_T=unique_T,
            inverse_idx=inverse_idx,
            T_counts=T_counts,
            TY_counts=TY_counts,
            I_X_T=I_X_T,
            I_T_Y=I_T_Y,
            num_intervals=num_intervals,
            prefix=debug_prefix,
            topk=debug_topk,
        )

    if not verbose:
        return I_X_T.item(), I_T_Y.item()

    # Extra debug quantities saved into the .npz file.
    with torch.no_grad():
        P_T_marg = TY_matrix.sum(dim=1)
        P_Y_marg = TY_matrix.sum(dim=0)
        H_Y = _entropy_from_probs(P_Y_marg)
        p_y_given_t = TY_counts / T_counts[:, None].clamp_min(1.0)
        H_Y_given_each_T = -(
            p_y_given_t.clamp_min(1e-12) * torch.where(
                p_y_given_t > 0,
                torch.log2(p_y_given_t.clamp_min(1e-12)),
                torch.zeros_like(p_y_given_t),
            )
        ).sum(dim=1)
        H_Y_given_T = (P_T_marg * H_Y_given_each_T).sum()
        dominant_counts = TY_counts.max(dim=1).values
        nonzero_label_per_pattern = (TY_counts > 0).sum(dim=1)
        pure_pattern_ratio = (nonzero_label_per_pattern == 1).float().mean()
        ambiguous_sample_mass = T_counts[nonzero_label_per_pattern > 1].sum() / N

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
        "H_Y":            np.float64(H_Y.item()),
        "H_Y_given_T":    np.float64(H_Y_given_T.item()),
        "H_Y_minus_I_T_Y": np.float64((H_Y - I_T_Y).item()),
        "pure_pattern_ratio": np.float64(pure_pattern_ratio.item()),
        "ambiguous_sample_mass": np.float64(ambiguous_sample_mass.item()),
        "dominant_counts": dominant_counts.detach().cpu().numpy().astype(np.int64),
        "nonzero_label_per_pattern": nonzero_label_per_pattern.detach().cpu().numpy().astype(np.int16),
    }

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"MI cal costs: {elapsed_time}")
    return I_X_T.item(), I_T_Y.item(), debug_data


def save_verbose_data(debug_data, save_dir, model_name, split_name,
                      num_intervals, in_size=None, epoch=None):
    """Save verbose debug data to a structured .npz file."""
    out_dir = Path(save_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    epoch_tag = f"epoch{epoch}_" if epoch is not None else ""
    if split_name == "in" and in_size is not None:
        fname = f"{epoch_tag}in_size{in_size}_bins{num_intervals}.npz"
    else:
        fname = f"{epoch_tag}{split_name}_bins{num_intervals}.npz"

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
        debug_mi=True,
        debug_topk=8,
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

            # In-sample MI + CSV row writing
            for in_size in in_sizes:
                layer_T, labels = in_cache[in_size]
                for nb in num_intervals_list:
                    if record_verbose:
                        ixt_in, ity_in, dbg_in = mi_from_logits(
                            layer_T, labels, num_intervals=nb, verbose=True,
                            debug=debug_mi,
                            debug_prefix=f"strategy={strategy}, epoch={epoch}, in_size={in_size}",
                            debug_topk=debug_topk,
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
                            debug=debug_mi,
                            debug_prefix=f"strategy={strategy}, epoch={epoch}, in_size={in_size}",
                            debug_topk=debug_topk,
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

            # Free per-epoch caches
            del in_cache, net
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n==> Done. Master CSV: {master_csv_path}")
                

# =====================================================
# 4. Entry point
# =====================================================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Folder containing all YAML experiment plans
    exp_dir = "./saved_exp_plan/ft_plan"
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

        for model_seed in range(42, 43):
            for ft_seed in range(0, 1):
                print(f"\n>>> Running model seed {model_seed}, ft seed {ft_seed} for {os.path.basename(yaml_path)}")
                set_seed(ft_seed)
                main_epochs(model_seed, ft_seed, 1.0, yaml_path,
                            in_sizes = [25000], #5000, 10000, 15000, 20000,
                            num_intervals_list = [50],# 10, 25, 50, 75, 100, 125, 150],
                            record_verbose = False,
                            master_csv_path="./saved_logs/ft_vanilla/MI_master_table_ft.csv",
                            verbose_dir="./saved_logs/ft_vanilla/MI_verbose",
                            subset_seed=42,
                            debug_mi=True,
                            debug_topk=8,
                            )

