"""Best-checkpoint MI for the FT + added-adversarial-term models (saved_models/ft_at).

Same measurement as calculate_MI_ft.main_best (the ft_final table): MI_check's
collect_logits / mi_from_logits, the negative-pool-0.0 grid (in_size rates x
bins), the original Dataset.group_size as the rate denominator, the group_A
clean subset as the In probe and the full test set as the Out probe, the
nested subset cache under Indices/<dataset>/, epoch="best". Only three things
differ, and they are the reason this is a separate entry point with its own
CSV rather than rows appended to MI_master_table_ft.csv:

  * model directory: saved_models/ft_at, and the model name is the ft_final
    name plus the AT suffix main_ft_at.py appends
    (_AT<attack>_<params>_lambda=<l>_bn=<policy>_run=<tag>), rebuilt from the
    plan through at_family_common so names cannot drift from training;
  * seven provenance columns appended AFTER the 25 ft_final columns:
    base_model_name (the ft_final name, for joining the two tables), attack,
    eps, steps, lambda, bn_policy, run_tag;
  * output: saved_logs/ft_at/MI_master_table_ft_at.csv.

Resume/identity rules are those of calculate_MI_ft.main_best: exact
(in_size, bins) cells are skipped when present, Out MI is reused by bin, a
changed checkpoint / plan / configuration is refused rather than mixed.
Single-writer CSV.

Verified 2026-09-22: pointed at the ft_final FT-AL seed-0 checkpoint (suffix
"") this path reproduces the MI_master_table_ft.csv values with diff 0.0.

Usage (globals below; SEED_START/SEED_END, FT_AT_STRATEGIES, FT_AT_RUN_TAG,
FT_AT_PLAN_DIR env vars override them):
  python calculate_MI_ft_at.py
"""
from datetime import datetime
import copy
import csv
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from at_family_common import at_suffix, parse_additive_at_config
from calculate_MI_ft import (BEST_BATCH_SIZE, BEST_BINS, BEST_CSV_COLUMNS, BEST_IN_SIZE_RATES,
                             BEST_NUM_WORKERS, BEST_SUBSET_SEED, _file_sha256, save_verbose_data,
                             seed_worker, set_seed)
from MI_check import collect_logits, mi_from_logits
from mi_pool_support import create_nested_balanced_subsets, missing_mi_grid, positive_ints, sizes_from_rates
from util import (create_or_load_group_A, process_experiment_ft_setup, process_experiment_ft_setup_deit,
                  process_yaml_file)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_DIR = Path(__file__).resolve().parent

# Editable configuration for main_ft_at_best / the script entry point.
FT_AT_PLAN_DIR = Path(os.environ.get("FT_AT_PLAN_DIR", BASE_DIR / "saved_exp_plan/ft_at_plan"))
FT_AT_MODEL_DIR = BASE_DIR / "saved_models/ft_at"
FT_AT_MASTER_CSV = BASE_DIR / "saved_logs/ft_at/MI_master_table_ft_at.csv"
FT_AT_VERBOSE_DIR = BASE_DIR / "saved_logs/ft_at/MI_verbose_best"
FT_AT_INDEX_DIR = BASE_DIR / "Indices"
FT_AT_MODEL_SEEDS = [42]
FT_AT_FT_SEEDS = list(range(int(os.environ.get("SEED_START", 0)), int(os.environ.get("SEED_END", 3))))
FT_AT_RATES = [1.0]
FT_AT_STRATEGIES = [s for s in os.environ.get("FT_AT_STRATEGIES", "FT-AL").split(",") if s]
FT_AT_RUN_TAG = os.environ.get("FT_AT_RUN_TAG", "v1")
FT_AT_IN_SIZE_RATES = list(BEST_IN_SIZE_RATES)
FT_AT_BINS = list(BEST_BINS)
FT_AT_RECORD_VERBOSE = False

AT_PROVENANCE_COLUMNS = ["base_model_name", "attack", "eps", "steps", "lambda", "bn_policy", "run_tag"]
FT_AT_CSV_COLUMNS = BEST_CSV_COLUMNS + AT_PROVENANCE_COLUMNS
NO_AT = dict(attack="none", eps="", steps="", **{"lambda": 0}, bn_policy="", run_tag="")


def ft_base_name(scenario, model_seed, rate, strategy, ft_size, ft_seed):
    """The ft_final model name (identical to calculate_MI_ft.main_best)."""
    return f"{scenario}_{model_seed}_{rate}_{strategy}_ftsize={ft_size}_ftseed={ft_seed}"


def plan_suffixes(exp_yaml, run_tag):
    """[(suffix, provenance dict)] for every attack config the plan declares."""
    attacks, _, lam, bn_policy = parse_additive_at_config(exp_yaml)
    out = []
    for _, attack_name, kwargs in attacks:
        out.append((at_suffix(attack_name, kwargs, lam, run_tag, bn_policy),
                    dict(attack=attack_name, eps=kwargs.get("eps", ""), steps=kwargs.get("steps", ""),
                         **{"lambda": format(lam, "g")}, bn_policy=bn_policy, run_tag=run_tag)))
    return out


def _missing_grid(csv_path, identity, in_sizes, bins, columns):
    """calculate_MI_ft._missing_best_grid generalised to a table with `columns`."""
    path = Path(csv_path)
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(columns):
                raise ValueError(f"Incompatible MI CSV header: {path}; use a separate CSV")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Incomplete/malformed MI row in {path}")
                if row["model_name"] != identity["model_name"]:
                    continue
                for key, expected in identity.items():
                    if row[key] != str(expected):
                        raise ValueError(
                            f"ft_at MI identity mismatch: {identity['model_name']}, {key}; "
                            "preserve the existing CSV and use a new output for changed checkpoints/configuration")
                size = float(row["in_size"])
                fraction = float(row["in_size_rate"])
                if not (0 < size <= identity["training_size"] and np.isfinite(fraction)
                        and abs(fraction - size / identity["training_size"]) < 1e-12):
                    raise ValueError(f"Invalid training-size fraction: {identity['model_name']}")
    return missing_mi_grid(path, identity["model_name"], identity["Scenario"], identity["ft_seed"],
                           identity["rate"], in_sizes, bins)


def _missing_ft_at_grid(csv_path, identity, in_sizes, bins):
    return _missing_grid(csv_path, identity, in_sizes, bins, FT_AT_CSV_COLUMNS)


def _append_row(csv_path, row, columns):
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("x", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=list(columns)).writeheader()
    with path.open("a", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=list(columns)).writerow(row)


def _append_ft_at_row(csv_path, row):
    _append_row(csv_path, row, FT_AT_CSV_COLUMNS)


def measure_pending(exp_yaml, pending, *, family, in_sizes, bins, index_dir, subset_seed, record_verbose,
                    verbose_dir, csv_path, columns, seed, tag="FT_AT"):
    """Measure In/Out MI for every (checkpoint, identity, missing cells, cached Out) in `pending`.

    This is the measurement half of calculate_MI_ft.main_best, shared by the
    ft_at and pruning_at tables: same probes, same subset cache, same
    mi_from_logits, one CSV row per (in_size, bins). Returns the row count.
    """
    set_seed(seed)
    training_size = exp_yaml["Dataset"]["group_size"]
    mi_yaml = copy.deepcopy(exp_yaml)
    mi_yaml.pop("FT_Dataset", None)
    # Only Dataset / NumClasses / Model_Factory are needed; the per-strategy or
    # per-sparsity optimizer entries (FT vs pruning plans) are irrelevant here.
    mi_yaml.pop("Optimizers", None)
    if family == "deit":
        mi_yaml["Model"]["pretrained"] = False
    setup_fn = process_experiment_ft_setup_deit if family == "deit" else process_experiment_ft_setup
    setup = setup_fn(mi_yaml)
    dataset = setup["Dataset"]
    idx_dir = Path(index_dir) / exp_yaml["Dataset"]["name"]
    group_a = create_or_load_group_A(dataset=dataset.in_sample_set, save_dir=idx_dir,
                                     group_size=training_size, num_classes=setup["NumClasses"],
                                     seed=42, force_rebuild=False)
    subsets = create_nested_balanced_subsets(dataset.in_sample_set, group_a, idx_dir, in_sizes,
                                             num_classes=setup["NumClasses"], seed=subset_seed,
                                             force_rebuild=False)
    out_size = len(dataset.test_set)

    def loader(data):
        return DataLoader(data, batch_size=BEST_BATCH_SIZE, shuffle=False, num_workers=BEST_NUM_WORKERS,
                          pin_memory=(str(device).startswith("cuda")), worker_init_fn=seed_worker,
                          generator=torch.Generator().manual_seed(subset_seed))

    written = 0
    for checkpoint, identity, missing_cells, existing_out in pending:
        name = identity["model_name"]
        needed_bins = sorted({b for _, b in missing_cells})
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
        print(f"[{tag}] {name}: {len(missing_cells)} missing MI cells on {device}")

        def measure(logits, labels, nb, split, size=None):
            result = mi_from_logits(logits, labels, num_intervals=nb, verbose=record_verbose)
            if not np.isfinite(result[:2]).all():
                raise ValueError(f"Nonfinite MI: {name}, {split}, {size}, bins={nb}")
            if record_verbose:
                save_verbose_data(result[2], verbose_dir, name, split, nb, in_size=size, epoch="best")
            return result[:2]

        try:
            out_results = {b: existing_out[b][:2] for b in needed_bins if b in existing_out}
            new_out_bins = [b for b in needed_bins if b not in existing_out]
            if new_out_bins:
                out_logits, out_labels = collect_logits(net, loader(dataset.test_set), device)
                for nb in new_out_bins:
                    out_results[nb] = measure(out_logits, out_labels, nb, "out")
                del out_logits, out_labels
            for size in sorted({s for s, _ in missing_cells}):
                probe = dataset.subset("train", subsets[size].tolist(), clean=True)
                logits, labels = collect_logits(net, loader(probe), device)
                for nb in [b for s, b in missing_cells if s == size]:
                    ixt_in, ity_in = measure(logits, labels, nb, "in", size)
                    ixt_out, ity_out = out_results[nb]
                    row = dict(identity, bins=nb, in_size=size, out_size=out_size,
                               in_size_rate=format(size / training_size, ".12g"),
                               timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    row.update({"I(X;T)-In": f"{ixt_in:.6f}", "I(T;Y)-In": f"{ity_in:.6f}",
                                "I(X;T)-Out": f"{ixt_out:.6f}", "I(T;Y)-Out": f"{ity_out:.6f}"})
                    _append_row(csv_path, row, columns)
                    written += 1
                    print(f"  size={size} bins={nb}: In=({ixt_in:.6f}, {ity_in:.6f}) Out=({ixt_out:.6f}, {ity_out:.6f})")
                del logits, labels
        finally:
            del net
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"[DONE] Appended {written} {tag} MI rows: {csv_path}")
    return written


def main_ft_at_best(
        model_seed, ft_seed, r, yaml_file_path, *,
        strategies=None, run_tag=None, name_suffixes=None,
        in_sizes=None, num_intervals_list=None, in_size_rates=None, record_verbose=None,
        master_csv_path=None, verbose_dir=None, subset_seed=None, model_dir=None, index_dir=None,
        family="auto", missing="skip", skip_existing=True,
):
    """Record best_epoch.pth In/Out MI for the FT+AT models of one plan x ft_seed.

    `name_suffixes` defaults to the plan's attack grid (plan_suffixes); pass
    [("", NO_AT)] with model_dir=saved_models/ft_final to measure a plain FT
    checkpoint through this exact code path (the cross-check in the docstring).
    Returns the number of newly appended rows.
    """
    exp_yaml = process_yaml_file(yaml_file_path)
    training_size = exp_yaml["Dataset"]["group_size"]
    ft_size = exp_yaml["FT_Dataset"]["group_size"]
    sizes_from_rates(training_size, [1.0])
    if type(ft_size) is not int or ft_size <= 0:
        raise ValueError("FT_Dataset.group_size must be a positive integer")
    if in_sizes is not None and in_size_rates is not None:
        raise ValueError("Choose in_sizes or in_size_rates, not both")
    if in_sizes is None:
        in_sizes = sizes_from_rates(training_size, FT_AT_IN_SIZE_RATES if in_size_rates is None else in_size_rates)
    in_sizes = positive_ints(in_sizes, "in_sizes")
    if max(in_sizes) > training_size:
        raise ValueError("in_sizes cannot exceed the original training group_size")
    bins = positive_ints(FT_AT_BINS if num_intervals_list is None else num_intervals_list, "bins")
    strategies = list(FT_AT_STRATEGIES if strategies is None else strategies)
    if not strategies or len(set(strategies)) != len(strategies) or any(
            s not in {"FT-LL", "FT-AL", "RT-AL"} for s in strategies):
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
    if missing not in {"skip", "raise"}:
        raise ValueError('missing must be "skip" or "raise"')
    rate = round(float(r), 2)
    run_tag = FT_AT_RUN_TAG if run_tag is None else run_tag
    suffixes = plan_suffixes(exp_yaml, run_tag) if name_suffixes is None else list(name_suffixes)
    model_dir = Path(FT_AT_MODEL_DIR if model_dir is None else model_dir)
    csv_path = Path(FT_AT_MASTER_CSV if master_csv_path is None else master_csv_path)
    verbose_dir = Path(FT_AT_VERBOSE_DIR if verbose_dir is None else verbose_dir)
    index_dir = Path(FT_AT_INDEX_DIR if index_dir is None else index_dir)
    subset_seed = BEST_SUBSET_SEED if subset_seed is None else subset_seed
    record_verbose = FT_AT_RECORD_VERBOSE if record_verbose is None else record_verbose
    scenario = exp_yaml["Scenario_Name"]
    plan_hash = _file_sha256(yaml_file_path)
    on_missing = missing

    # Preflight every strategy x attack before touching dataset/cache/CSV.
    pending = []
    for strategy in strategies:
        base = ft_base_name(scenario, model_seed, rate, strategy, ft_size, ft_seed)
        for suffix, provenance in suffixes:
            name = base + suffix
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
                base_model_name=base, **provenance,
            )
            missing_cells, existing_out = _missing_ft_at_grid(csv_path, identity, in_sizes, bins)
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
                           csv_path=csv_path, columns=FT_AT_CSV_COLUMNS, seed=ft_seed, tag="FT_AT")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    yaml_files = sorted(Path(FT_AT_PLAN_DIR).glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No YAML experiment plans in {FT_AT_PLAN_DIR}")
    for _k in ("FT_AT_PLAN_DIR", "FT_AT_STRATEGIES", "FT_AT_RUN_TAG", "SEED_START", "SEED_END"):
        if _k in os.environ:
            print(f"[ENV] {_k}={os.environ[_k]}")
    print(f"Plans: {len(yaml_files)} in {FT_AT_PLAN_DIR}")
    print(f"ft_seeds: {FT_AT_FT_SEEDS}   strategies: {FT_AT_STRATEGIES}   run_tag: {FT_AT_RUN_TAG}")
    print(f"grid: in_size rates {FT_AT_IN_SIZE_RATES} x bins {FT_AT_BINS}")
    print("Untrained cells are reported as [ABSENT] and skipped (missing='raise' refuses a partial set).")
    total = 0
    for yaml_path in yaml_files:
        for model_seed in FT_AT_MODEL_SEEDS:
            for ft_seed in FT_AT_FT_SEEDS:
                for rate in FT_AT_RATES:
                    total += main_ft_at_best(model_seed, ft_seed, rate, yaml_path)
    print(f"\n==> All done. {total} new row(s) in {FT_AT_MASTER_CSV}")
