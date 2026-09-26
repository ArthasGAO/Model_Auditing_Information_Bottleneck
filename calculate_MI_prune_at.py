"""Best-checkpoint MI for the pruning + added-adversarial-term models (saved_models/pruning_at).

Same measurement as calculate_MI_prune.main_prune_best / calculate_MI_ft.main_best
(MI_check's collect_logits / mi_from_logits, the pool-0.0 grid, group_A clean
subset as the In probe, full test set as the Out probe, epoch="best"),
through the shared calculate_MI_ft_at.measure_pending. Differences from the
pruning_final table are the model directory (saved_models/pruning_at), the
model name (the pruning_final name plus the AT suffix main_prune_at.py appends,
rebuilt from the plan through at_family_common) and the provenance columns
appended after the 25 ft columns: base_model_name (the pruning_final name),
sparsity, achieved_sparsity (zero fraction of the dense checkpoint), attack,
eps, steps, lambda, bn_policy, recal, run_tag. Output:
saved_logs/pruning_at/MI_master_table_prune_at.csv. Single-writer CSV, same
resume/identity rules as the other best-MI tables.

Usage (globals below; SEED_START/SEED_END, PRUNE_AT_SPARSITIES, PRUNE_AT_RUN_TAG,
PRUNE_AT_PLAN_DIR env vars override them):
  python calculate_MI_prune_at.py
"""
import os
from pathlib import Path

import numpy as np
import torch

from at_family_common import at_suffix, parse_additive_at_config
from calculate_MI_ft import BEST_CSV_COLUMNS, BEST_SUBSET_SEED, _file_sha256
from calculate_MI_ft_at import (FT_AT_BINS, FT_AT_IN_SIZE_RATES, FT_AT_INDEX_DIR, NO_AT, _missing_grid,
                                measure_pending)
from main_prune import prune_scenario_name
from main_prune_at import parse_recal
from mi_pool_support import positive_ints, sizes_from_rates
from util import process_yaml_file

BASE_DIR = Path(__file__).resolve().parent
PRUNE_AT_PLAN_DIR = Path(os.environ.get("PRUNE_AT_PLAN_DIR", BASE_DIR / "saved_exp_plan/prune_at_plan"))
PRUNE_AT_MODEL_DIR = BASE_DIR / "saved_models/pruning_at"
PRUNE_AT_MASTER_CSV = BASE_DIR / "saved_logs/pruning_at/MI_master_table_prune_at.csv"
PRUNE_AT_VERBOSE_DIR = BASE_DIR / "saved_logs/pruning_at/MI_verbose_best"
PRUNE_AT_MODEL_SEEDS = [42]
PRUNE_AT_FT_SEEDS = list(range(int(os.environ.get("SEED_START", 0)), int(os.environ.get("SEED_END", 1))))
PRUNE_AT_RATES = [1.0]
PRUNE_AT_SPARSITIES = [float(s) for s in os.environ.get("PRUNE_AT_SPARSITIES", "0.8").split(",") if s]
PRUNE_AT_RUN_TAG = os.environ.get("PRUNE_AT_RUN_TAG", "v1")
PRUNE_AT_STRATEGY = "FT-AL"
PRUNE_AT_RECORD_VERBOSE = False

PRUNE_AT_PROVENANCE_COLUMNS = ["base_model_name", "sparsity", "achieved_sparsity", "attack", "eps", "steps",
                               "lambda", "bn_policy", "recal", "run_tag"]
PRUNE_AT_CSV_COLUMNS = BEST_CSV_COLUMNS + PRUNE_AT_PROVENANCE_COLUMNS
NO_AT_PRUNE = dict(NO_AT, recal="")


def prune_plan_suffixes(exp_yaml, run_tag):
    attacks, _, lam, bn_policy = parse_additive_at_config(exp_yaml)
    recal = parse_recal(exp_yaml)
    out = []
    for _, attack_name, kwargs in attacks:
        out.append((at_suffix(attack_name, kwargs, lam, run_tag, bn_policy,
                              extra={"recal": "clean" if recal else "none"}),
                    dict(attack=attack_name, eps=kwargs.get("eps", ""), steps=kwargs.get("steps", ""),
                         **{"lambda": format(lam, "g")}, bn_policy=bn_policy,
                         recal="clean" if recal else "none", run_tag=run_tag)))
    return out


def achieved_sparsity(checkpoint):
    """Zero fraction over all conv/linear weight tensors of a dense (mask-removed) state_dict."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    total = zero = 0
    for k, v in state.items():
        if k.endswith("weight") and v.ndim in (2, 4):
            total += v.numel()
            zero += (v == 0).sum().item()
    return zero / total if total else float("nan")


def main_prune_at_best(model_seed, ft_seed, r, yaml_file_path, *, sparsities=None, run_tag=None,
                       name_suffixes=None, in_sizes=None, num_intervals_list=None, in_size_rates=None,
                       record_verbose=None, master_csv_path=None, verbose_dir=None, subset_seed=None,
                       model_dir=None, index_dir=None, family="auto", missing="skip", skip_existing=True):
    """Record best_epoch.pth In/Out MI for the pruning+AT models of one plan x ft_seed x sparsities.

    `name_suffixes` defaults to the plan's attack grid; pass [("", NO_AT_PRUNE)]
    with model_dir=saved_models/pruning_final to measure a plain pruning
    checkpoint through this exact code path (cross-check against
    MI_master_table_prune.csv). Returns the number of newly appended rows.
    """
    exp_yaml = process_yaml_file(yaml_file_path)
    training_size = exp_yaml["Dataset"]["group_size"]
    ft_size = exp_yaml["FT_Dataset"]["group_size"]
    sizes_from_rates(training_size, [1.0])
    if in_sizes is not None and in_size_rates is not None:
        raise ValueError("Choose in_sizes or in_size_rates, not both")
    if in_sizes is None:
        in_sizes = sizes_from_rates(training_size, FT_AT_IN_SIZE_RATES if in_size_rates is None else in_size_rates)
    in_sizes = positive_ints(in_sizes, "in_sizes")
    if max(in_sizes) > training_size:
        raise ValueError("in_sizes cannot exceed the original training group_size")
    bins = positive_ints(FT_AT_BINS if num_intervals_list is None else num_intervals_list, "bins")
    if family == "auto":
        family = "deit" if isinstance(exp_yaml["Model"], dict) else "cnn"
    if family not in {"cnn", "deit"}:
        raise ValueError("family must be auto, cnn or deit")
    for name, value in [("model_seed", model_seed), ("ft_seed", ft_seed)]:
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not np.isfinite(r) or not 0 <= r <= 1:
        raise ValueError("r must be a finite fraction in [0, 1]")
    if missing not in {"skip", "raise"}:
        raise ValueError('missing must be "skip" or "raise"')
    rate = round(float(r), 2)
    run_tag = PRUNE_AT_RUN_TAG if run_tag is None else run_tag
    sparsities = list(PRUNE_AT_SPARSITIES if sparsities is None else sparsities)
    plan_sparsities = [round(float(o["sparsity"]), 6) for o in exp_yaml.get("Optimizers", []) or []]
    suffixes = prune_plan_suffixes(exp_yaml, run_tag) if name_suffixes is None else list(name_suffixes)
    model_dir = Path(PRUNE_AT_MODEL_DIR if model_dir is None else model_dir)
    csv_path = Path(PRUNE_AT_MASTER_CSV if master_csv_path is None else master_csv_path)
    verbose_dir = Path(PRUNE_AT_VERBOSE_DIR if verbose_dir is None else verbose_dir)
    index_dir = Path(FT_AT_INDEX_DIR if index_dir is None else index_dir)
    subset_seed = BEST_SUBSET_SEED if subset_seed is None else subset_seed
    record_verbose = PRUNE_AT_RECORD_VERBOSE if record_verbose is None else record_verbose
    scenario = exp_yaml["Scenario_Name"]
    plan_hash = _file_sha256(yaml_file_path)

    pending = []
    for s in sparsities:
        s_key = round(float(s), 6)
        if s_key not in plan_sparsities:
            raise ValueError(f"{Path(yaml_file_path).name}: no Optimizers entry for sparsity {s}")
        base = prune_scenario_name(scenario, s_key, model_seed, rate, ft_seed, ft_size, PRUNE_AT_STRATEGY)
        for suffix, provenance in suffixes:
            name = base + suffix
            checkpoint = model_dir / name / "best_epoch.pth"
            if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                if missing == "raise":
                    raise FileNotFoundError(f"Missing/empty best checkpoint: {checkpoint}")
                print(f"[ABSENT] {name}: no best_epoch.pth under {model_dir}; skipping")
                continue
            identity = dict(
                Scenario=scenario, seed=ft_seed, rate=rate, model_name=name, epoch="best",
                model_seed=model_seed, ft_seed=ft_seed, strategy=PRUNE_AT_STRATEGY, ft_size=ft_size,
                training_size=training_size, family=family, subset_seed=subset_seed, group_seed=42,
                checkpoint=str(checkpoint.resolve()), checkpoint_sha256=_file_sha256(checkpoint),
                plan_sha256=plan_hash, base_model_name=base, sparsity=s_key,
                achieved_sparsity=format(achieved_sparsity(checkpoint), ".6f"), **provenance,
            )
            missing_cells, existing_out = _missing_grid(csv_path, identity, in_sizes, bins, PRUNE_AT_CSV_COLUMNS)
            if not skip_existing and len(missing_cells) != len(in_sizes) * len(bins):
                raise ValueError("skip_existing=False would duplicate MI rows; use a new CSV")
            if missing_cells:
                pending.append((checkpoint, identity, missing_cells, existing_out))
            else:
                print(f"[SKIP] {name}: all requested MI cells already exist")
    if not pending:
        return 0
    return measure_pending(exp_yaml, pending, family=family, in_sizes=in_sizes, bins=bins, index_dir=index_dir,
                           subset_seed=subset_seed, record_verbose=record_verbose, verbose_dir=verbose_dir,
                           csv_path=csv_path, columns=PRUNE_AT_CSV_COLUMNS, seed=ft_seed, tag="PRUNE_AT")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    yaml_files = sorted(Path(PRUNE_AT_PLAN_DIR).glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No YAML experiment plans in {PRUNE_AT_PLAN_DIR}")
    for _k in ("PRUNE_AT_PLAN_DIR", "PRUNE_AT_SPARSITIES", "PRUNE_AT_RUN_TAG", "SEED_START", "SEED_END"):
        if _k in os.environ:
            print(f"[ENV] {_k}={os.environ[_k]}")
    print(f"Plans: {len(yaml_files)} in {PRUNE_AT_PLAN_DIR}")
    print(f"ft_seeds: {PRUNE_AT_FT_SEEDS}   sparsities: {PRUNE_AT_SPARSITIES}   run_tag: {PRUNE_AT_RUN_TAG}")
    print(f"grid: in_size rates {FT_AT_IN_SIZE_RATES} x bins {FT_AT_BINS}")
    total = 0
    for yaml_path in yaml_files:
        for model_seed in PRUNE_AT_MODEL_SEEDS:
            for ft_seed in PRUNE_AT_FT_SEEDS:
                for rate in PRUNE_AT_RATES:
                    total += main_prune_at_best(model_seed, ft_seed, rate, yaml_path)
    print(f"\n==> All done. {total} new row(s) in {PRUNE_AT_MASTER_CSV}")
