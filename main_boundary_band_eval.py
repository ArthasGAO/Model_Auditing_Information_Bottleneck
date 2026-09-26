"""
main_boundary_band_eval.py

Phase-1 decision-boundary distance gate -- YAML-driven runner.

For the served (victim) model, run the two-phase band pipeline in raw-pixel L2 units
(Option A: gradients flow through NormalizedModel with raw [0,1] inputs):

  1. collect_boundary_stats  -- ONE expensive pass over benign test queries, cached
                                to ./Indices/{ds}/BoundaryBand/... so re-tuning d is free.
  2. calibrate_d             -- threshold from a low percentile of the benign Delta
                                distribution (never a hard-coded constant), unless the
                                YAML pins d explicitly.
  3. band_from_stats         -- cheap band-membership decision at d:
                                 * benign band rate  = false-positive rate of the gate
                                 * probe  band rate  = detection rate, if a probe .pt is given

Results are appended to a dedup-keyed master CSV, mirroring main_IPGUARD_eval.py.

This runner only reads models/data and writes logs; it modifies no existing module.
It reuses util_adv loaders (NormalizedModel wrapping) and util YAML/dataset helpers.
"""

import os
import csv
import glob
import random
from pathlib import Path
from datetime import datetime

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from util import process_yaml_file, build_dataset_from_yaml
from util_adv import load_victim_model
from AdvAttack.boundary_band import (
    collect_boundary_stats, band_from_stats, calibrate_d, summarize,
)


# =====================================================
# 1. Global setup
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"


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


# =====================================================
# 2. Master table configuration (dedup-keyed, IPGuard convention)
# =====================================================
BB_MASTER_CSV_PATH = Path("./saved_logs/at_eval/BoundaryBand/master.csv")

BB_MASTER_COLUMNS = [
    "Run_Timestamp", "Scenario_Name",
    "Victim_Arch", "Dataset",
    "Order", "TopK", "Percentile", "d",
    "N_Benign", "Benign_Band_Rate",
    "N_Probe", "Probe_Band_Rate",
    "Mean_Delta_Benign", "Median_Delta_Benign",
    "P01_Delta_Benign", "N_Ill_Benign",
]

# Same scenario/victim with a different order/topk/percentile/d coexist as rows.
BB_DEDUP_KEY = (
    "Scenario_Name", "Victim_Arch", "Dataset",
    "Order", "TopK", "Percentile", "d",
)


def _read_master_rows(csv_path):
    if not csv_path.exists():
        return []
    with open(csv_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _write_master_rows(csv_path, columns, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_master_row(csv_path, row_dict, columns, dedup_key):
    """Append a row with dedup on `dedup_key` (new row overwrites same-key row)."""
    missing = set(columns) - set(row_dict.keys())
    extra = set(row_dict.keys()) - set(columns)
    if missing:
        raise ValueError(f"append_master_row missing columns: {missing}")
    if extra:
        raise ValueError(f"append_master_row unknown columns: {extra}")

    existing = _read_master_rows(csv_path)
    new_key = tuple(str(row_dict[k]) for k in dedup_key)
    kept = [r for r in existing
            if tuple(r.get(k, "") for k in dedup_key) != new_key]
    kept.append(row_dict)
    _write_master_rows(csv_path, columns, kept)


# =====================================================
# 3. Loaders
# =====================================================
def build_raw_data_loader(dataset_obj):
    return DataLoader(
        dataset_obj,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


def _build_victim_id(victim_cfg, victim_ds_cfg):
    ds_name = victim_ds_cfg["name"]
    train_size = victim_ds_cfg.get("group_size", "unknown")
    seed = victim_cfg.get("Seed", 42)
    overlap = victim_cfg.get("Overlap", 1.0)
    return f"{ds_name}_{train_size}_seed={seed}_overlap={overlap}"


# =====================================================
# 4. Stats caching (the "collect once" side of the two-phase split)
# =====================================================
def _stats_cache_path(ds_name, victim_arch, victim_id, split, topk, order, n_cap):
    topk_tag = "all" if topk is None else str(topk)
    cur = "2nd" if order == "second" else "1st"
    cap = "all" if n_cap is None else str(n_cap)
    d = Path(f"./Indices/{ds_name}/BoundaryBand/"
             f"victim={victim_arch}_{victim_id}")
    d.mkdir(parents=True, exist_ok=True)
    return d / f"stats_{split}_topk={topk_tag}_{cur}_n={cap}.pt"


def _get_or_collect_stats(model, loader, cache_path, topk, compute_curvature,
                          fd_r, corr_tol, max_samples):
    if cache_path.exists():
        print(f"==> Loading cached boundary stats from {cache_path}")
        return torch.load(cache_path, map_location="cpu")
    print(f"==> Collecting boundary stats (topk={topk}, "
          f"curvature={compute_curvature}, max_samples={max_samples})..")
    stats = collect_boundary_stats(
        model, loader, device=device, topk=topk,
        compute_curvature=compute_curvature, fd_r=fd_r, corr_tol=corr_tol,
        max_samples=max_samples,
    )
    torch.save(stats, cache_path)
    print(f"  Cached stats -> {cache_path}")
    return stats


def _load_probe_images(probe_cfg):
    """Load a probe set of raw [0,1] images from a .pt file.

    Accepts either a dict with a 'test_x' tensor (IPGuard fingerprint format) or a
    bare image tensor. Returns (X, None) so collect_boundary_stats treats it as an
    unlabeled query stream.
    """
    path = probe_cfg["Path"]
    blob = torch.load(path, map_location="cpu")
    if isinstance(blob, dict):
        key = probe_cfg.get("Key", "test_x")
        X = blob[key]
    else:
        X = blob
    return X.float()


# =====================================================
# 5. Main
# =====================================================
def main_boundary_band(yaml_file_path):
    print(f"Device: {device}")
    exp_yaml = process_yaml_file(yaml_file_path)
    scenario_name = exp_yaml.get("Scenario_Name", Path(yaml_file_path).stem)

    # --- config block (with safe defaults) ---
    bb = exp_yaml.get("BoundaryBand", {})
    topk = bb.get("topk", None)                     # None => all competitors (CIFAR-10)
    order = bb.get("order", "first")                # "first" | "second"
    percentile = float(bb.get("percentile", 1.0))   # low pct of benign Delta -> d
    d_override = bb.get("d", None)                    # pin d explicitly (skips calibration)
    max_samples = bb.get("max_samples", 1000)        # benign queries to score
    fd_r = float(bb.get("fd_r", 1e-2))               # curvature finite-diff radius
    corr_tol = float(bb.get("corr_tol", 0.0))        # skip 2nd order if correction below
    compute_curvature = (order == "second")

    # --- victim (served) model ---
    victim_cfg = exp_yaml["Victim"]
    victim_ds_cfg = victim_cfg["Dataset"]
    victim_ds_obj, num_classes, _ = build_dataset_from_yaml(victim_ds_cfg)

    print("==> Loading victim model..") # the model loaded here is already with Normalized Wrapper
    victim_model = load_victim_model(victim_cfg, victim_ds_obj, num_classes, model_seed=42)
    victim_arch = victim_cfg["Model"]
    victim_id = _build_victim_id(victim_cfg, victim_ds_cfg)
    ds_name = victim_ds_cfg["name"]
    print(f"  Victim arch: {victim_arch}   Victim ID: {victim_id}")
    print(f"  Config: order={order} topk={topk} percentile={percentile} "
          f"d_override={d_override} max_samples={max_samples}")

    # --- Phase 1a: benign stats (collect once, cached) ---
    raw_loader = build_raw_test_loader(victim_ds_obj, batch_size=1000)
    benign_cache = _stats_cache_path(
        ds_name, victim_arch, victim_id, "benign", topk, order, max_samples)
    benign_stats = _get_or_collect_stats(
        victim_model, raw_loader, benign_cache, topk, compute_curvature,
        fd_r, corr_tol, max_samples)

    # --- threshold d: calibrate from benign distribution unless pinned ---
    if d_override is not None:
        d = float(d_override)
        print(f"==> Using pinned d = {d}")
    else:
        d = calibrate_d(benign_stats, percentile=percentile, order=order)
        print(f"==> Calibrated d = {d:.6f}  (benign p{percentile}, {order}-order)")

    # --- Phase 1b: cheap band decisions ---
    benign_band = band_from_stats(benign_stats, d, order=order)
    bsum = summarize(benign_stats, order=order)
    print(f"  Benign band rate (FPR) = {benign_band['band_rate']:.4f} "
          f"over N={bsum['n']} (mean Delta={bsum['mean']:.4f}, "
          f"median={bsum['median']:.4f}, n_ill={bsum['n_ill']})")

    # --- optional probe set: detection rate at the same d ---
    n_probe, probe_rate = 0, ""
    probe_cfg = exp_yaml.get("Probe")
    if probe_cfg and probe_cfg.get("Path"):
        print(f"==> Scoring probe set from {probe_cfg['Path']}")
        probe_X = _load_probe_images(probe_cfg)
        probe_cache = _stats_cache_path(
            ds_name, victim_arch, victim_id,
            f"probe-{Path(probe_cfg['Path']).stem}", topk, order, None)
        probe_stats = _get_or_collect_stats(
            victim_model, probe_X, probe_cache, topk, compute_curvature,
            fd_r, corr_tol, None)
        probe_band = band_from_stats(probe_stats, d, order=order)
        n_probe = int(len(probe_band["flags"]))
        probe_rate = round(probe_band["band_rate"], 4)
        print(f"  Probe band rate (detection) = {probe_rate} over N={n_probe}")

    # --- log to master CSV (dedup) ---
    append_master_row(BB_MASTER_CSV_PATH, {
        "Run_Timestamp":       datetime.now().isoformat(timespec="seconds"),
        "Scenario_Name":       scenario_name,
        "Victim_Arch":         victim_arch,
        "Dataset":             ds_name,
        "Order":               order,
        "TopK":                "all" if topk is None else topk,
        "Percentile":          percentile,
        "d":                   round(d, 6),
        "N_Benign":            bsum["n"],
        "Benign_Band_Rate":    round(benign_band["band_rate"], 4),
        "N_Probe":             n_probe,
        "Probe_Band_Rate":     probe_rate,
        "Mean_Delta_Benign":   round(bsum["mean"], 6),
        "Median_Delta_Benign": round(bsum["median"], 6),
        "P01_Delta_Benign":    round(bsum["p01"], 6),
        "N_Ill_Benign":        bsum["n_ill"],
    }, columns=BB_MASTER_COLUMNS, dedup_key=BB_DEDUP_KEY)

    print(f"\n  Results -> {BB_MASTER_CSV_PATH}")


# =====================================================
# 6. Entry point
# =====================================================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    set_seed(42)

    exp_dir = "./saved_exp_plan/boundary_band_plan"
    yaml_files = sorted(glob.glob(os.path.join(exp_dir, "*.yaml")))

    if not yaml_files:
        print(f"No YAML files found in {exp_dir}")
    else:
        print(f"Found {len(yaml_files)} experiment plan(s):")
        for f in yaml_files:
            print(" -", f)

    for yaml_path in yaml_files:
        print(f"\n{'=' * 60}\n  {yaml_path}\n{'=' * 60}")
        main_boundary_band(yaml_path)
