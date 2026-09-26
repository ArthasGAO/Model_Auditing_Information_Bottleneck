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
from util import create_or_load_group_B, partition_indices_class_balanced, process_yaml_file, process_experiment_setup, create_train_subset, evaluate2, prepare_group_subset, \
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

"""
QQ-Plot Validation for Theorem 3
=================================
Tests whether the leave-one-out Hotelling T² values from the
auxiliary MI vectors follow a chi-squared(d=2) distribution,
as predicted by the asymptotic result in Theorem 3.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, probplot, kstest, anderson
from sklearn.covariance import ledoit_wolf


# ══════════════════════════════════════════════════════════════════════
# Hotelling T² functions (from your existing codebase)
# ══════════════════════════════════════════════════════════════════════

def hotelling_T2_raw_fit(aux_matrix, ddof=1, shrinkage="auto"):
    X = np.asarray(aux_matrix, dtype=float)
    k, p = X.shape
    mu = X.mean(axis=0)

    S_raw = np.cov(X, rowvar=False, ddof=ddof)
    evals_raw = np.linalg.eigvalsh(S_raw)
    cond_raw = float(evals_raw.max() / max(evals_raw.min(), 1e-12))

    S_final = S_raw.copy()
    evals_final = evals_raw.copy()
    cond_final = cond_raw
    applied_lambda = 0.0

    if shrinkage == "auto":
        _, optimal_lambda = ledoit_wolf(X)
        applied_lambda = optimal_lambda
    elif isinstance(shrinkage, float) and shrinkage > 0.0:
        applied_lambda = min(max(shrinkage, 0.0), 1.0)

    if applied_lambda > 0.0:
        avg_var = np.trace(S_raw) / p
        target = avg_var * np.eye(p)
        S_final = (1.0 - applied_lambda) * S_raw + applied_lambda * target
        evals_final = np.linalg.eigvalsh(S_final)
        cond_final = float(evals_final.max() / max(evals_final.min(), 1e-12))

    info = {
        "mu": mu, "k": k, "p": p,
        "shrinkage_lambda": applied_lambda,
        "S": S_final, "eigvals": evals_final, "cond": cond_final,
        "S_raw": S_raw, "eigvals_raw": evals_raw, "cond_raw": cond_raw,
    }
    return mu, S_final, info


def hotelling_T2_raw_score(mu, S, x, k=None, scaling="none"):
    x = np.asarray(x, dtype=float).ravel()
    diff = x - mu
    #Sinv = np.linalg.inv(S)
    md2 = float(diff @ np.linalg.solve(S, diff))

    if scaling == "none":
        T2 = md2
    elif scaling == "paper":
        if k is None:
            raise ValueError("k must be provided for scaling='paper'")
        T2 = float((k / (k + 1.0)) * md2)
    elif scaling == "k":
        if k is None:
            raise ValueError("k must be provided for scaling='k'")
        T2 = float(k * md2)
    else:
        raise ValueError("scaling must be in {'none','paper','k'}")

    p_value = 1.0 - chi2.cdf(T2, df=len(mu))
    return T2, p_value, {"diff": diff, "md2": md2, "scaling": scaling}


# ══════════════════════════════════════════════════════════════════════
# Leave-One-Out T² computation
# ══════════════════════════════════════════════════════════════════════

def leave_one_out_T2(mi_vectors, scaling="paper", shrinkage="auto"):
    aux_matrix = np.array(mi_vectors, dtype=float)
    k = len(aux_matrix)

    T2_values = np.zeros(k)
    p_values = np.zeros(k)
    details = []

    for i in range(k):
        held_out = aux_matrix[i]
        remaining = np.delete(aux_matrix, i, axis=0)

        mu_loo, S_loo, info_loo = hotelling_T2_raw_fit(
            remaining, ddof=1, shrinkage=shrinkage
        )
        T2_i, p_i, score_info = hotelling_T2_raw_score(
            mu_loo, S_loo, held_out,
            k=info_loo["k"], scaling=scaling,
        )

        T2_values[i] = T2_i
        p_values[i] = p_i
        details.append({
            "partition": i,
            "held_out": held_out,
            "mu": mu_loo,
            "shrinkage_lambda": info_loo["shrinkage_lambda"],
            "cond": info_loo["cond"],
            "T2": T2_i,
            "p_value": p_i,
        })

    return T2_values, p_values, details


# ══════════════════════════════════════════════════════════════════════
# QQ-Plot + Goodness-of-Fit Testing
# ══════════════════════════════════════════════════════════════════════
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"

def qq_plot_chi2(
    T2_values,
    df=2,
    title="QQ Plot: T² vs χ²(2)",
    figsize=(6, 6),
    save_path=None,
    dpi=150,
    show=True,
    annotate=False,
):
    """
    Draw a QQ plot of empirical T² values against theoretical chi-squared(df) quantiles.

    Parameters
    ----------
    T2_values : array-like
        The empirical T² statistics.
    df : int
        Degrees of freedom of the reference chi-squared distribution.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save the figure.
    dpi : int
        Resolution for saved figure.
    show : bool
        Whether to display the plot.
    annotate : bool
        Whether to annotate sorted point ranks.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    T2_values = np.sort(np.asarray(T2_values, dtype=float))
    n = len(T2_values)

    if n == 0:
        raise ValueError("T2_values must contain at least one value.")

    # Theoretical chi-square quantiles using plotting positions
    plotting_positions = (np.arange(1, n + 1) - 0.5) / n
    theoretical_quantiles = chi2.ppf(plotting_positions, df=df)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # QQ points
    ax.scatter(
        theoretical_quantiles,
        T2_values,
        s=60,
        zorder=3,
        edgecolors="white",
        linewidth=0.8,
        label=f"Empirical T² values (n={n})",
    )

    # Reference diagonal
    max_val = max(theoretical_quantiles.max(), T2_values.max()) * 1.15
    ax.plot(
        [0, max_val],
        [0, max_val],
        "r--",
        linewidth=1.2,
        alpha=0.7,
        label="y = x",
    )

    # Optional annotation by sorted rank
    if annotate:
        for i, (tq, tv) in enumerate(zip(theoretical_quantiles, T2_values)):
            ax.annotate(
                f"{i}",
                (tq, tv),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                color="gray",
            )

    ax.set_xlabel(f"Theoretical Quantiles — χ²({df})", fontsize=20)
    ax.set_ylabel("Empirical T² Values", fontsize=20)
    ax.tick_params(axis="both", labelsize=16)
    #ax.set_title(title, fontsize=13)
    ax.legend(fontsize=16, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    ax.set_box_aspect(1)

    # -----------------------------
    # Save / show
    # -----------------------------
    import re
    from pathlib import Path
    if save_path is not None:
        output_dir = Path(save_path)
    else:
        output_dir = Path(".")
    
    safe_title = re.sub(r'[\\/*?:"<>|]+', "", title)   # remove illegal filename chars
    safe_title = re.sub(r"\s+", "_", safe_title.strip())  # replace spaces/newlines with underscores
    pdf_path = output_dir / f"{safe_title}.pdf"
    
    fig.savefig(pdf_path, format="pdf", dpi=dpi, bbox_inches="tight")
    print(f"Figure saved to {pdf_path}")

    if show:
        plt.show()

    return fig, ax


# ══════════════════════════════════════════════════════════════════════
# Full Pipeline: from MI vectors -> LOO T² -> QQ-plot
# ══════════════════════════════════════════════════════════════════════

def validate_theorem3(
    mi_vectors,
    scaling="paper",
    shrinkage="auto",
    title="Theorem 3 Validation: T² vs χ²(2)",
    save_path=None,
    show=True,
    verbose=True,
):

    # ── Step 1: Leave-one-out T² ───────────────────────────────────
    T2_values, p_values, details = leave_one_out_T2(
        mi_vectors, scaling=scaling, shrinkage=shrinkage
    )


    # ── Step 2: QQ-plot + GoF ──────────────────────────────────────
    fig, ax = qq_plot_chi2(
        T2_values, df=2, title=title,
        save_path=save_path, show=show,
    )


# =====================================================
# 3. Main training logic
# =====================================================
def main_nega(seed, r, num_partitions, yaml_file_path): # this function is used to calculate the model from main_train_nega.py
    print(device)

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_setup(exp_yaml) 

    print('==> Loading model..')
    model_name = exp_yaml["Scenario_Name"]
    model_dir = './saved_models/vanilla/'
    model_folder = Path(model_dir)/model_name

    # ckpt_path = load_last_checkpoint(model_folder)
    ckpt_path = load_best_checkpoint(model_folder)
    if ckpt_path is None:
        print(f"[⚠️] No .pth files found in {model_folder}")
    
    net = exp_setup["Model"].to(device) # load model
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state)

    print('==> Preparing data..')
    g = torch.Generator()
    g.manual_seed(seed)

    train_set = exp_setup["Dataset"].train_set
    group_A = create_or_load_group_A(dataset=train_set, save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
                                                    group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"], seed=42, force_rebuild=False)

    group_B = create_or_load_group_B(dataset=train_set, save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/', group_A_indices=group_A,
                                        group_size = exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"], 
                                        overlap_rate=0.0, seed=42, force_rebuild=False)
    
    partitions = partition_indices_class_balanced(
                    dataset=train_set,
                    indices=group_B,
                    num_partitions=num_partitions,
                    seed=42,
                )
    
    eval_loaders = {}
    for i, part_indices in enumerate(partitions):
        subset = exp_setup["Dataset"].subset("train", part_indices, clean=True)
        loader = DataLoader(
            subset,
            batch_size=exp_setup.get("BatchSize", 64),
            shuffle=False,       # no need to shuffle for evaluation
            num_workers=4,
            pin_memory=True,
        )
        eval_loaders[f"partition_{i}"] = {
            "indices": part_indices,
            "loader": loader,
        }

    mi_ls = []

    for name, entry in eval_loaders.items():
        loader = entry["loader"]
        indices = entry["indices"]
        
        # your MI computation here
        value_xt_out, value_ty_out = evaluate2(net, loader, device)
        print(f"{name}: I(X;T)={value_xt_out:.4f}, I(T;Y)={value_ty_out:.4f}")
        mi_ls.append((value_xt_out, value_ty_out)) 

    validate_theorem3(
        mi_ls,
        scaling="paper",
        shrinkage="auto",
        title=f"{model_name} {num_partitions} ",
        save_path="./figures",
        show=False,
    )


def main_nega1(seed, r, yaml_file_path): # this function is used to calculate the model from main_train_nega.py
    print(device)

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_setup(exp_yaml) 

    print('==> Loading model..')
    model_name = exp_yaml["Scenario_Name"]
    model_dir = './saved_models/vanilla/'
    model_folder = Path(model_dir)/model_name

    # ckpt_path = load_last_checkpoint(model_folder)
    ckpt_path = load_best_checkpoint(model_folder)
    if ckpt_path is None:
        print(f"[⚠️] No .pth files found in {model_folder}")
    
    net = exp_setup["Model"].to(device) # load model
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state)

    print('==> Preparing data..')
    g = torch.Generator()
    g.manual_seed(seed)

    train_set = exp_setup["Dataset"].train_set
    group_A = create_or_load_group_A(dataset=train_set, save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
                                                    group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"], seed=42, force_rebuild=False)

    group_B = create_or_load_group_B(dataset=train_set, save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/', group_A_indices=group_A,
                                        group_size = exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"], 
                                        overlap_rate=0.0, seed=42, force_rebuild=False)
    
    base_num_partitions = 10

    base_partitions = partition_indices_class_balanced(
        dataset=train_set,
        indices=group_B,
        num_partitions=base_num_partitions,
        seed=42,
    )
    
    mi_ls = []

    for i, part_indices in enumerate(base_partitions):
        subset = exp_setup["Dataset"].subset("train", part_indices, clean=True)
        loader = DataLoader(
            subset,
            batch_size=exp_setup.get("BatchSize", 64),
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        value_xt_out, value_ty_out = evaluate2(net, loader, device)
        print(f"partition_{i}: I(X;T)={value_xt_out:.4f}, I(T;Y)={value_ty_out:.4f}")
        mi_ls.append((value_xt_out, value_ty_out))

    for use_k in  [4, 5, 6, 7, 8, 9 ,10]:

        selected_mi = mi_ls[:use_k]

        validate_theorem3(
            selected_mi,
            scaling="paper",
            shrinkage="auto",
            title=f"{model_name}_base10_use{use_k}",
            save_path="./figures",
            show=False,
        )
        
    


# =====================================================
# 4. Entry point
# =====================================================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Folder containing all YAML experiment plans
    exp_dir = "./saved_exp_plan/train_plan"
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
            print(f"\n>>> Running seed {seed} for {os.path.basename(yaml_path)}")
            set_seed(seed)

            for num in [4, 5, 6, 7, 8, 9 ,10]:
                main_nega(seed, 1.0, num, yaml_path)
            #main_nega1(seed, 1.0, yaml_path)

                
        