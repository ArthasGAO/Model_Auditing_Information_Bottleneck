"""
MI metrics for knowledge-distillation students.

Restructured to the current master-table format used by calculate_MI_at.py:

  * Phase 1 / Phase 2 split - inference runs ONCE per loader and the cached
    logits are reused for every bin count, instead of re-running the model per
    bin (the old version called the long-deprecated 3-argument `evaluate2`).
  * Bin sweep over `num_intervals_list` and subset sweep over `in_sizes`.
  * Both in-sample AND out-sample MI (the old version wrote hardcoded zeros for
    the out-sample and the InAug columns).
  * One row per (model, in_size, bins) appended to a single master CSV with the
    shared MASTER_CSV_COLUMNS schema, instead of one CSV per scenario holding a
    single row labelled with a fake epoch of 99.

The MI estimator, the master-CSV schema and the nested-subset builder are
imported from calculate_MI_at.py rather than copied, so this script cannot
drift away from the canonical implementation the way the old one did.

IMPORTANT - normalization convention differs from the AT scripts:
KD students are NOT wrapped in NormalizedModel; normalization lives in the
dataset transforms (main_kd.py trains on subset("train", ..., clean=False)).
So the evaluation splits here must be the NORMALIZED ones -
`subset("train", group_A, clean=True)` and `dataset.test_set` - not the raw
[0,1] splits that calculate_MI_at.py feeds to its wrapped models.

The in-sample set is always group_A (the victim/teacher split), regardless of
which transfer set the student was distilled on. That is the point of the
measurement: MI is evaluated with respect to the owner's data.
"""

import os
import glob
from datetime import datetime

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # needed for full CUDA determinism

from pathlib import Path
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from util import (create_or_load_group_A, process_yaml_file,
                  process_experiment_kd_setup, load_best_checkpoint,
                  collect_logits)

# Shared MI machinery - single source of truth, see module docstring.
from calculate_MI_at import (mi_from_logits, ensure_master_csv, append_master_row,
                             save_verbose_data)
# The (in_size, bins) grid every other MI table already uses.
from MI_check import POOL0_BINS, POOL0_IN_SIZE_RATES
# Resume protocol shared with the other MI drivers: missing_mi_grid reports
# which (in_size, bins) cells a model still needs and refuses to treat a table
# with duplicate or inconsistent rows as complete. The nested-subset builder is
# the SHARED one, not calculate_MI_at's local copy: the local copy rewrites the
# cached npz down to whatever sizes it was asked for, and rejects sizes that are
# not divisible by num_classes (250 / 1250 / 18750 all fail on CIFAR-100).
from mi_pool_support import (
    create_nested_balanced_subsets as create_nested_balanced_subsets_shared,
    missing_mi_grid, positive_ints, sizes_from_rates,
)

# Same grid as the pool / victim / FT / extraction tables, so a KD point can be
# overlaid on their in_size sweeps. Before 2026-09-21 this driver only ever
# wrote in_size=25000, which left KD as the one method with a single column.
KD_IN_SIZE_RATES = list(POOL0_IN_SIZE_RATES)   # {0.01 .. 1.00} x group_size
KD_BINS = list(POOL0_BINS)

# Output tree, moved off kd_vanilla on 2026-09-21 to match main_kd.py. The old
# tree still holds the to10 / to8 / ResNet-18-teacher rows and their MI table,
# which distribution_check.ipynb reads by name; this table covers exactly the
# models in saved_models/kd_final.
KD_MODEL_ROOT = './saved_models/kd_final'
KD_LOG_ROOT = './saved_logs/kd_final'
KD_MASTER_CSV = f'{KD_LOG_ROOT}/MI_master_table_kd.csv'
KD_VERBOSE_DIR = f'{KD_LOG_ROOT}/MI_verbose'

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


# =====================================================
# 2. Main calculation logic
# =====================================================
def main_kd(
    seed, r, yaml_file_path,
    in_sizes=None,
    num_intervals_list=None,
    record_verbose=False,
    master_csv_path=KD_MASTER_CSV,
    verbose_dir=KD_VERBOSE_DIR,
    subset_seed=42,
):
    """
    Calculate MI metrics for every distillation method listed in one KD plan.

    Args:
        seed:  training seed of the student to evaluate (part of its folder name).
        r:     transfer-set overlap rate of the student to evaluate.
                 1.0 -> Cell A (student distilled on the teacher's own split)
                 0.0 -> Cell C (student distilled on a disjoint transfer set)
               Recorded in the `rate` column of the master table.
        yaml_file_path: the same plan file main_kd.py trained from, so the
               scenario names line up exactly.
    """
    # in_sizes defaults to the shared rate grid resolved against this plan's own
    # group_size, so a CIFAR-100 KD cell gets {250 .. 25000} the same way a
    # CIFAR-10 one does. Pass in_sizes=[25000] to reproduce the pre-2026-09-21
    # single-column behaviour.
    if num_intervals_list is None:
        num_intervals_list = list(KD_BINS)
    num_intervals_list = positive_ints(num_intervals_list, "bins")

    print(f"Device: {device}")

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_kd_setup(exp_yaml)

    # ---------- Prepare dataset ----------
    print('==> Preparing data..')
    # Normalized, un-augmented splits: KD students carry no NormalizedModel wrapper.
    in_sample_set = exp_setup["Dataset"].in_sample_set   # train images, no aug, normalized
    out_sample_set = exp_setup["Dataset"].test_set       # test images, normalized

    group_A = create_or_load_group_A(
        dataset=in_sample_set,
        save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
        group_size=exp_setup["GroupSize"],
        num_classes=exp_setup["NumClasses"],
        seed=42, force_rebuild=False,
    )
    if in_sizes is None:
        in_sizes = sizes_from_rates(int(exp_setup["GroupSize"]), KD_IN_SIZE_RATES)
    in_sizes = positive_ints(in_sizes, "in_sizes")
    if max(in_sizes) > int(exp_setup["GroupSize"]):
        raise ValueError("in_sizes cannot exceed the plan's group_size")
    print(f"[GRID] in_sizes={in_sizes}  bins={num_intervals_list}")

    nested_subsets = create_nested_balanced_subsets_shared(
        dataset=in_sample_set, group_A=group_A,
        save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
        subset_sizes=in_sizes,
        num_classes=exp_setup["NumClasses"],
        seed=subset_seed, force_rebuild=False,
    )
    absent = [s for s in in_sizes if s not in nested_subsets]
    if absent:
        raise ValueError(
            f"nested_subsets_seed{subset_seed}.npz in ./Indices/"
            f"{exp_yaml['Dataset']['name']}/ has no entry for {absent}"
        )

    out_size = len(out_sample_set)
    out_sample_loader = DataLoader(
        out_sample_set, batch_size=128, shuffle=False, num_workers=0,
        pin_memory=True,
    )

    # ensure_master_csv only creates the file, not its parent directory.
    os.makedirs(os.path.dirname(master_csv_path) or ".", exist_ok=True)
    ensure_master_csv(master_csv_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    kd_methods = list(exp_setup["KD_Setups"].keys())
    print(f"Found {len(kd_methods)} distillation method(s): {kd_methods}")

    # ---------- Iterate over distillation methods ----------
    for method_name in kd_methods:
        print(f"\n{'='*60}")
        print(f"Method: {method_name}  |  seed={seed}  rate={r}")
        print(f"{'='*60}")

        # Must match main_kd.py's scenario_name exactly.
        scenario_name = exp_yaml["Scenario_Name"] + f"_{method_name}_{seed}_{r}"

        model_dir = KD_MODEL_ROOT
        model_folder = Path(model_dir) / scenario_name
        ckpt_path, filename = load_best_checkpoint(model_folder)
        if ckpt_path is None:
            # Skip rather than abort: one missing run should not stop the sweep.
            print(f"[!] No best_epoch.pth in {model_folder} - skipping.")
            continue

        # ---- Resume: which (in_size, bins) cells does this model still need?
        # Checked BEFORE the builder runs, because the builder also constructs
        # the teacher and loads its checkpoint. Without this, a second sweep
        # appended a duplicate copy of every row (the table already carried 100
        # such rows from a 2026-08-26 re-run before the guard existed).
        todo_pairs, _ = missing_mi_grid(
            master_csv_path, scenario_name, exp_yaml["Scenario_Name"],
            seed, round(float(r), 2), in_sizes, num_intervals_list,
        )
        if not todo_pairs:
            print(f"[SKIP] {scenario_name}: all "
                  f"{len(in_sizes) * len(num_intervals_list)} (in_size, bins) "
                  f"row(s) already in {master_csv_path}")
            continue
        todo_set = set(todo_pairs)
        needed_sizes = [s for s in in_sizes if any(sz == s for sz, _ in todo_pairs)]
        if len(todo_pairs) < len(in_sizes) * len(num_intervals_list):
            print(f"[RESUME] {scenario_name}: {len(todo_pairs)} of "
                  f"{len(in_sizes) * len(num_intervals_list)} cell(s) missing; "
                  f"inferring in-sample only for {needed_sizes}")

        # Rebuild the student architecture through the plan's own builder so the
        # architecture is guaranteed to match what training used.
        print(f"==> Building and loading student from {ckpt_path}..")
        distiller = exp_setup["KD_Setups"][method_name]["Builder"]()
        net = distiller.student.to(device)
        net.load_state_dict(torch.load(ckpt_path, map_location=device))
        net.eval()

        # =============================================================
        # PHASE 1: Inference (the expensive part) -- ONCE per data set
        # =============================================================
        print(f"\n==> [Phase 1] Running inference once per loader")

        print(f"  Inferring out-sample (size={out_size})...")
        out_layer_T, out_labels = collect_logits(net, out_sample_loader, device)
        print(f"    -> cached logits shape: {tuple(out_layer_T.shape)}")

        in_cache = {}   # in_size -> (layer_T, label_matrix)
        for in_size in needed_sizes:
            subset_indices = nested_subsets[in_size].tolist()
            in_sample_subset = exp_setup["Dataset"].subset(
                "train", subset_indices, clean=True,   # normalized, no augmentation
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
        # PHASE 2: MI computation (the cheap part) -- bin sweep on cache
        # =============================================================
        print(f"\n==> [Phase 2] Computing MI on cached logits")

        out_results = {}
        for nb in num_intervals_list:
            if record_verbose:
                ixt_out, ity_out, dbg = mi_from_logits(
                    out_layer_T, out_labels, num_intervals=nb, verbose=True,
                )
                save_verbose_data(dbg, verbose_dir, scenario_name, "out", nb)
            else:
                ixt_out, ity_out = mi_from_logits(
                    out_layer_T, out_labels, num_intervals=nb, verbose=False,
                )
            print(f"  out  bins={nb:>3}: I(X;T)={ixt_out:.4f}, I(T;Y)={ity_out:.4f}")
            out_results[nb] = (ixt_out, ity_out)

        for in_size in needed_sizes:
            layer_T, labels = in_cache[in_size]
            for nb in num_intervals_list:
                if (in_size, nb) not in todo_set:
                    continue          # already in the master CSV
                if record_verbose:
                    ixt_in, ity_in, dbg = mi_from_logits(
                        layer_T, labels, num_intervals=nb, verbose=True,
                    )
                    save_verbose_data(dbg, verbose_dir, scenario_name, "in", nb, in_size)
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
                    "rate":        round(float(r), 2),
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

        # Free cached logits so a long sweep does not accumulate GPU memory.
        del in_cache, out_layer_T, out_labels, distiller, net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n==> Done. Master CSV: {master_csv_path}")


# =====================================================
# 3. Entry point
# =====================================================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Mirror main_kd.py: same plan folder, same seed range, same overlap rates,
    # so every trained student gets a matching MI row.
    # KD_PLAN_DIR overrides the folder from the environment (same variable
    # main_kd.py reads), default unchanged.
    exp_dir = os.environ.get("KD_PLAN_DIR", "./saved_exp_plan/kd_plan_arch")
    # Same three variables main_kd.py reads, so training and MI stay in step.
    # Previously SEEDS was range(42, 43) and OVERLAP_RATES was [1.0, 0.0],
    # which no longer matched main_kd.py's KD_RATES default of "0.0".
    SEEDS = [int(x) for x in os.environ.get("KD_SEEDS", "0,1,2").split(",")]
    OVERLAP_RATES = [float(x) for x in os.environ.get("KD_RATES", "0.0").split(",")]
    for _k in ("KD_PLAN_DIR", "KD_SEEDS", "KD_RATES"):
        if _k in os.environ:
            print(f"[ENV] {_k}={os.environ[_k]}")
    print(f"Seeds: {SEEDS}   Rates: {OVERLAP_RATES}")

    yaml_files = sorted(glob.glob(os.path.join(exp_dir, "*.yaml")))

    if not yaml_files:
        print(f"No YAML files found in {exp_dir}")
    else:
        print(f"Found {len(yaml_files)} experiment plan(s):")
        for f in yaml_files:
            print(" -", f)

    for yaml_path in yaml_files:
        print(f"\n========== Starting experiments from {yaml_path} ==========")
        for seed in SEEDS:
            for rate in OVERLAP_RATES:
                print(f"\n>>> MI for seed {seed}, overlap_rate {rate} "
                      f"for {os.path.basename(yaml_path)}")
                set_seed(seed)

                # in_sizes=None -> the shared rate grid resolved against the
                # plan's own group_size. KD_IN_SIZES="25000" restores the old
                # single-column behaviour.
                _env_sizes = os.environ.get("KD_IN_SIZES")
                main_kd(
                    seed, rate, yaml_path,
                    in_sizes=[int(s) for s in _env_sizes.split(",")] if _env_sizes else None,
                    num_intervals_list=None,
                    record_verbose=False,
                )
