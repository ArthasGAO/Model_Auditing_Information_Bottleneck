import os
import csv
import json
import glob
import random
from pathlib import Path
from datetime import datetime

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from scipy import stats

from util_adv import (
    _append_csv_row, _csv_path_for, _ensure_csv_with_header,
    _suspect_id_fields, _suspect_id_header, _suspect_id_values,
    collect_checkpoints, fgsm_attack, load_adv_examples,
    load_negative_suspect, load_positive_suspect, load_victim_model,
    pgd_attack_v2,
)
from util import process_yaml_file, build_dataset_from_yaml
from AdvAttack.IP_Guard import IPGuardGenerator, verify_fingerprint


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


ATTACK_REGISTRY = {
    "PGD": pgd_attack_v2,
    "FGSM": fgsm_attack,
}


# =====================================================
# 2b. Master table configuration — IPGuard
# =====================================================
IPGUARD_MASTER_CSV_PATH = Path("./saved_logs/at_eval/IPGuard/master.csv")

# Architecture is now first-class: Victim_Arch and Suspect_Arch are dedicated
# columns. Scenario_Name is architecture-free and captures only the
# experimental condition (dataset, seed, overlap, AT params, epoch).
IPGUARD_MASTER_COLUMNS = [
    "Run_Timestamp", "Scenario_Name",
    "Victim_Arch", "Suspect_Arch",
    "Suspect_Type", "Checkpoint",
    "FP_Config", "k_Param", "Size",
    "Matching_Rate",
]

# Dedup key includes both architectures so cross-architecture sweeps with
# the same scenario string coexist as distinct rows.
IPGUARD_DEDUP_KEY = (
    "Scenario_Name",
    "Victim_Arch", "Suspect_Arch",
    "Suspect_Type", "Checkpoint",
    "FP_Config", "k_Param", "Size",
)


def format_eval_attack_string(attack_name, attack_kwargs):
    """Human-readable eval-attack column value: 'PGD,eps=0.03,steps=10'."""
    params = ",".join(f"{k}={v}" for k, v in sorted(attack_kwargs.items()))
    return f"{attack_name},{params}"


def _read_master_rows(csv_path):
    """Read existing master CSV as list of dicts. Returns [] if file doesn't exist."""
    if not csv_path.exists():
        return []
    with open(csv_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _write_master_rows(csv_path, columns, rows):
    """Write rows back, preserving the given column order."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_master_row(csv_path, row_dict, columns, dedup_key):
    """
    Append a row to the master table at `csv_path` with dedup on `dedup_key`.
    New row overwrites any existing row with the same key. `columns` defines
    the on-disk schema and ordering.
    """
    missing = set(columns) - set(row_dict.keys())
    extra   = set(row_dict.keys()) - set(columns)
    if missing:
        raise ValueError(f"append_master_row missing columns: {missing}")
    if extra:
        raise ValueError(f"append_master_row unknown columns: {extra}")

    existing = _read_master_rows(csv_path)
    new_key  = tuple(str(row_dict[k]) for k in dedup_key)

    # Drop any existing row with the same key (new row wins)
    kept = [r for r in existing
            if tuple(r.get(k, "") for k in dedup_key) != new_key]
    kept.append(row_dict)

    _write_master_rows(csv_path, columns, kept)


# =====================================================
# 3. Loaders
# =====================================================
def build_raw_test_loader(dataset_obj, batch_size=1000, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset_obj.raw_test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=True,
        pin_memory=True,
    )


# =====================================================
# 7. Shared boilerplate (victim + adv + IPGuard fp + DI regressor)
# =====================================================
def _build_victim_id(victim_cfg, victim_ds_cfg):
    """
    Build an architecture-free victim identifier.

    Example output: 'CIFAR-10_25000_seed=42_overlap=1.0'

    Used in both the master CSV's Scenario_Name (so rows for the same
    victim group together regardless of suspect architecture) and as
    part of the fingerprint cache directory name.

    Reads from:
      - victim_ds_cfg['name']        -> dataset (e.g. 'CIFAR-10')
      - victim_ds_cfg['group_size']  -> training set size (e.g. 25000)
      - victim_cfg['Seed']           -> seed (default 42)
      - victim_cfg['Overlap']        -> overlap rate (default 1.0)

    Victim seed/overlap default to (42, 1.0) — the standard victim setup
    in this experimental framework.
    """
    ds_name    = victim_ds_cfg["name"]
    train_size = victim_ds_cfg.get("group_size", "unknown")
    seed       = victim_cfg.get("Seed", 42)
    overlap    = victim_cfg.get("Overlap", 1.0)
    return f"{ds_name}_{train_size}_seed={seed}_overlap={overlap}"


def _setup_victim_context(yaml_file_path, build_raw_loader=True):
    """
    Common setup performed by every main_* function:
      - print device, parse YAML
      - build victim dataset object
      - load victim model (seed=42)
      - optionally build raw [0,1] test loader
      - extract victim architecture and build the architecture-free
        victim identifier used throughout the pipeline
    """
    print(f"Device: {device}")

    exp_yaml = process_yaml_file(yaml_file_path)
    scenario_name = exp_yaml.get("Scenario_Name", Path(yaml_file_path).stem)

    victim_cfg = exp_yaml["Victim"]
    victim_ds_cfg = victim_cfg["Dataset"]
    victim_dataset_obj, victim_num_classes, _ = build_dataset_from_yaml(victim_ds_cfg)

    print("==> Loading victim model..")
    victim_model = load_victim_model(
        victim_cfg, victim_dataset_obj, victim_num_classes, model_seed=42
    )

    raw_loader = None
    if build_raw_loader:
        print("==> Loading raw [0,1] test data..")
        raw_loader = build_raw_test_loader(victim_dataset_obj, batch_size=1000)

    # Explicit victim architecture and architecture-free victim ID. These
    # flow through the rest of the pipeline and end up as structured columns
    # in the master CSV.
    victim_arch = victim_cfg["Model"]
    victim_id   = _build_victim_id(victim_cfg, victim_ds_cfg)
    print(f"  Victim arch: {victim_arch}")
    print(f"  Victim ID:   {victim_id}")

    return {
        "exp_yaml": exp_yaml,
        "scenario_name": scenario_name,
        "victim_cfg": victim_cfg,
        "victim_ds_cfg": victim_ds_cfg,
        "victim_dataset_obj": victim_dataset_obj,
        "victim_num_classes": victim_num_classes,
        "victim_model": victim_model,
        "victim_arch": victim_arch,
        "victim_id": victim_id,
        "raw_loader": raw_loader,
    }


# =====================================================
# 8. Suspect block detection + unified iteration
# =====================================================
NEG_KEY = "Negative Suspect"
POS_KEY = "Positive"

_DEFAULT_SUSPECT_SEEDS = list(range(42, 52))
_DEFAULT_OVERLAP_RATES = [round(r * 0.1, 1) for r in range(0, 11)]


def _detect_suspect_block(exp_yaml):
    """
    Return ('negative', cfg) or ('positive', cfg) based on which block the
    YAML contains. Raises if both or neither are present.
    """
    has_neg = bool(exp_yaml.get(NEG_KEY))
    has_pos = bool(exp_yaml.get(POS_KEY))

    if has_neg and has_pos:
        raise ValueError(
            "YAML contains both 'Negative Suspect' and 'Positive' blocks. "
            "Each plan file should declare exactly one suspect type."
        )
    if has_neg:
        return "negative", exp_yaml[NEG_KEY]
    if has_pos:
        return "positive", exp_yaml[POS_KEY]
    raise ValueError(
        f"YAML must contain either '{NEG_KEY}' or '{POS_KEY}' block."
    )


def _parse_epoch_from_ckpt(ckpt_name):
    """
    Extract the integer epoch from a checkpoint filename of the form
    'epoch_N.pth'. Returns the int, or None if the name doesn't match
    (e.g. 'best_epoch.pth', 'last.pth').

    Example: 'epoch_47.pth' -> 47
    """
    stem = Path(ckpt_name).stem  # 'epoch_47'
    if not stem.startswith("epoch_"):
        return None
    try:
        return int(stem.split("_")[-1])
    except ValueError:
        return None


def _iter_suspects(suspect_type, suspect_cfg, victim_dataset_obj, num_classes,
                   victim_arch, victim_id,
                   suspect_seeds=None, overlap_rates=None):
    """
    Yield uniform suspect records regardless of suspect type.

    Each record:
        {
          "type":           "negative" | "positive",
          "model":          loaded NormalizedModel,
          "seed":           int or None,
          "overlap":        float or None,
          "dir_name":       str or None,
          "ckpt_name":      str or None,
          "ckpt_path":      Path or None,
          "eval_mode":      "best" | "last" | "all",
          "scenario_name":  per-suspect scenario string (architecture-free)
          "victim_arch":    str  (e.g. "ResNet-18")
          "suspect_arch":   str  (e.g. "VGG16")
          "epoch":          int or None
        }

    `scenario_name` per record is architecture-free:
      - grid negatives: f"{victim_id}_seed={s}_overlap={o}"
      - path suspects:  dir_leaf (with optional _epoch=N suffix)

    Architecture lives in `victim_arch` and `suspect_arch`, which become
    dedicated columns in the master CSV.

    Negative suspects support TWO modes:
      - default: iterate seeds x overlap_rates (loaded from
        ./saved_models/vanilla/{Model_Name}_{seed}_{overlap}/best_epoch.pth)
      - explicit Model_Path: iterate directories like the positive branch,
        with full checkpoint selection via 'State'
    """
    suspect_arch = suspect_cfg.get("Model", "UnknownArch")

    if suspect_type == "negative":
        # Explicit Model_Path takes precedence if present.
        if suspect_cfg.get("Model_Path"):
            yield from _iter_suspects_by_paths(
                suspect_type="negative",
                suspect_cfg=suspect_cfg,
                victim_dataset_obj=victim_dataset_obj,
                num_classes=num_classes,
                victim_arch=victim_arch,
                suspect_arch=suspect_arch,
            )
            return

        # Fall back to the seed x overlap grid.
        seeds    = suspect_seeds or suspect_cfg.get("seeds")         or _DEFAULT_SUSPECT_SEEDS
        overlaps = overlap_rates or suspect_cfg.get("overlap_rates") or _DEFAULT_OVERLAP_RATES

        for seed in seeds:
            for overlap in overlaps:
                print(f"\n--- Negative Suspect ({suspect_arch}): "
                      f"seed={seed}, overlap={overlap} ---")
                net = load_negative_suspect(
                    suspect_cfg, victim_dataset_obj, num_classes, seed, overlap
                )
                if net is None:
                    print(f"  [SKIP] No checkpoint for seed={seed}, overlap={overlap}")
                    continue
                yield {
                    "type": "negative", "model": net,
                    "seed": seed, "overlap": overlap,
                    "dir_name": None, "ckpt_name": None, "ckpt_path": None,
                    "eval_mode": "best",  # grid-mode negatives always load best_epoch.pth
                    "scenario_name": f"{victim_id}_seed={seed}_overlap={overlap}",
                    "victim_arch": victim_arch,
                    "suspect_arch": suspect_arch,
                    "epoch": None,
                }
                del net
                torch.cuda.empty_cache()

    elif suspect_type == "positive":
        yield from _iter_suspects_by_paths(
            suspect_type="positive",
            suspect_cfg=suspect_cfg,
            victim_dataset_obj=victim_dataset_obj,
            num_classes=num_classes,
            victim_arch=victim_arch,
            suspect_arch=suspect_arch,
        )

    else:
        raise ValueError(f"Unknown suspect type: {suspect_type!r}")


def _iter_suspects_by_paths(suspect_type, suspect_cfg,
                            victim_dataset_obj, num_classes,
                            victim_arch, suspect_arch):
    """
    Shared path-based iteration for both 'positive' and explicit-path
    'negative' suspects. Yields records tagged with `suspect_type`, the
    YAML-supplied `eval_mode`, and a per-suspect `scenario_name` extracted
    from the leaf of `dir_name` (with optional _epoch=N suffix when the
    checkpoint follows the `epoch_N.pth` convention).

    Checkpoint selection (same rules as main_DI_eval.py):
      - `Checkpoint: <file>`  -> evaluate exactly `<dir>/<file>` (e.g.
        `best_clean_epoch.pth` saved by main_at.py, which never writes
        `best_epoch.pth`); missing file raises FileNotFoundError.
      - otherwise `State: best|last|all` via collect_checkpoints.
      - `Require_Checkpoints: true` turns an empty selection into an error
        instead of a silent [SKIP].
    """
    model_name = suspect_cfg.get("Model", "ResNet-18")
    eval_mode  = suspect_cfg.get("State", "best")
    model_dirs = suspect_cfg.get("Model_Path", [])
    base_dir   = Path("./saved_models")

    type_label = suspect_type.capitalize()

    for dir_name in model_dirs:
        model_dir = base_dir / dir_name.lstrip("/")
        print(f"\n--- {type_label} Suspect ({suspect_arch}): {dir_name} ---")

        if suspect_cfg.get("Checkpoint"):
            checkpoint = model_dir / suspect_cfg["Checkpoint"]
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            checkpoints = [(checkpoint.stem, checkpoint)]
        else:
            checkpoints = collect_checkpoints(model_dir, eval_mode)
        if not checkpoints:
            if suspect_cfg.get("Require_Checkpoints", False):
                raise FileNotFoundError(f"No checkpoints found in {model_dir}")
            print(f"  [SKIP] No checkpoints found in {model_dir}")
            continue

        # Leaf segment of the model directory (epoch suffix added per-ckpt below).
        # Strip trailing slash so paths ending in '/' resolve cleanly.
        dir_leaf = dir_name.rstrip("/").split("/")[-1]

        for ckpt_name, ckpt_path in checkpoints:
            print(f"  Evaluating: {ckpt_name}")
            net = load_positive_suspect(
                model_name, num_classes, victim_dataset_obj, ckpt_path,
                checkpoint_format=suspect_cfg.get("Checkpoint_Format"),
                model_config=suspect_cfg.get("Model_Config"),
            )

            # Parse epoch from filename. None when eval_mode is 'best'/'last'
            # (single-checkpoint cases), int when iterating epoch_N.pth files.
            epoch = _parse_epoch_from_ckpt(ckpt_name)

            # Per-row scenario: append _epoch=N when we have a numeric epoch,
            # otherwise stay clean (single-checkpoint cases).
            per_row_scenario = (
                f"{dir_leaf}_epoch={epoch}" if epoch is not None else dir_leaf
            )

            yield {
                "type": suspect_type, "model": net,
                "seed": None, "overlap": None,
                "dir_name": dir_name, "ckpt_name": ckpt_name, "ckpt_path": ckpt_path,
                "eval_mode": eval_mode,
                "scenario_name": per_row_scenario,
                "victim_arch": victim_arch,
                "suspect_arch": suspect_arch,
                "epoch": epoch,
            }
            del net
            torch.cuda.empty_cache()


VICTIM_SEED = 42
VICTIM_OVERLAP = 1.0


# =====================================================
# IPGuard — fingerprint generation + matching-rate logging
# =====================================================
IPGUARD_CONFIGS = [
    ("T", "R"),  # Training init + Random target
    #("T", "L"),  # Training init + Least-likely target
    #("R", "R"),  # Random init + Random target
    #("R", "L"),  # Random init + Least-likely target
]


def _format_convergence_log(scenario_dir, victim_id, victim_arch,
                            k_param, size, fingerprints):
    """
    Write a human-readable convergence.log inside `scenario_dir`.
    Style A: one section per config, each with aggregate header + tab-separated
    per-point table. Called only when at least one config was freshly generated.

    Sections appear in canonical IPGUARD_CONFIGS order (TR, TL, RR, RL) so the
    layout is stable across runs.
    """
    log_path = scenario_dir / "convergence.log"

    config_names = {
        "T": "Training example", "R": "Random",
        "L": "Least-likely",
    }

    lines = []
    lines.append("# IP-Guard fingerprint convergence log")
    lines.append(f"# Victim ID:   {victim_id}")
    lines.append(f"# Victim Arch: {victim_arch}")
    lines.append(f"# k_param:     {k_param}")
    lines.append(f"# size:        {size}")
    lines.append("")

    for init_strat, target_strat in IPGUARD_CONFIGS:
        tag = f"{init_strat}{target_strat}"
        fp = fingerprints[tag]

        n_total     = len(fp["converged"])
        n_converged = sum(1 for c in fp["converged"] if c)
        conv_rate   = 100.0 * n_converged / n_total if n_total else 0.0
        mean_loss   = float(np.mean(fp["final_losses"])) if n_total else 0.0
        mean_iters  = float(np.mean(fp["iters"]))       if n_total else 0.0

        lines.append("=" * 70)
        lines.append(f"=== {tag} "
                     f"({config_names[init_strat]} init, "
                     f"{config_names[target_strat]} target) ===")
        lines.append("=" * 70)
        lines.append(f"Aggregate: {n_converged}/{n_total} converged "
                     f"({conv_rate:.1f}%), "
                     f"mean_final_loss={mean_loss:.6f}, "
                     f"mean_iters={mean_iters:.1f}")
        lines.append("")
        lines.append("# Per-point detail (tab-separated)")
        lines.append("idx\tsource\ttarget\tconverged\tfinal_loss\titers\tvictim_pred")

        for idx in range(n_total):
            lines.append(
                f"{idx}\t"
                f"{fp['source_labels'][idx]}\t"
                f"{fp['target_labels'][idx]}\t"
                f"{int(fp['converged'][idx])}\t"
                f"{fp['final_losses'][idx]:.6f}\t"
                f"{fp['iters'][idx]}\t"
                f"{fp['victim_preds'][idx]}"
            )
        lines.append("")

    log_path.write_text("\n".join(lines))
    print(f"  Wrote convergence log -> {log_path}")


def _get_or_generate_ipguard_fingerprints(
    victim_model, victim_num_classes, victim_ds_cfg,
    victim_arch, victim_id,
    raw_train_set, k_param, size,
):
    """
    Generate (or load) IP-Guard fingerprints for all four (init, target)
    configurations. Returns dict {tag -> fp_data}.

    Cache layout: each (victim_arch, victim_id, k, size) gets its own
    subdirectory containing 4 .pt files (one per config) plus a
    convergence.log written only when at least one config was freshly
    generated. The cache is keyed strictly by victim identity — suspect
    architecture does not affect fingerprint generation.

        ./Indices/{ds}/IPGuard/
            victim={victim_arch}_{victim_id}_k={k}_size={size}/
                TR.pt
                TL.pt
                RR.pt
                RL.pt
                convergence.log

    Partial caches are handled gracefully: any missing .pt files trigger
    regeneration for just those configs.
    """
    ds_name  = victim_ds_cfg["name"]
    img_size = victim_ds_cfg["img_size"]

    scenario_dir = Path(
        f"./Indices/{ds_name}/IPGuard/"
        f"victim={victim_arch}_{victim_id}_k={k_param}_size={size}"
    )
    scenario_dir.mkdir(parents=True, exist_ok=True)

    fingerprints = {}
    any_freshly_generated = False

    for init_strat, target_strat in IPGUARD_CONFIGS:
        tag = f"{init_strat}{target_strat}"
        fp_path = scenario_dir / f"{tag}.pt"

        if fp_path.exists():
            print(f"==> Loading existing {tag} fingerprints from {fp_path}")
            fingerprints[tag] = IPGuardGenerator.load_fingerprints(fp_path)
            continue

        print(f"==> Generating {tag} fingerprints (k={k_param}, size={size})..")
        gen = IPGuardGenerator(
            model=victim_model,
            num_classes=victim_num_classes,
            k=k_param,
            max_iters=1000,
            lr=0.01,
            init_strategy=init_strat,
            target_strategy=target_strat,
            device=device,
        )
        train_data = raw_train_set if init_strat == "T" else None
        fingerprints[tag] = gen.generate(
            n_points=size,
            train_dataset=train_data,
            input_shape=(3, img_size, img_size),
        )
        gen.save_fingerprints(fingerprints[tag], fp_path)
        any_freshly_generated = True

    # Write the convergence log only when something new was generated.
    # If all four configs hit cache, the existing log on disk is already
    # consistent with the .pt files and we leave it alone.
    if any_freshly_generated:
        _format_convergence_log(
            scenario_dir, victim_id, victim_arch,
            k_param, size, fingerprints,
        )
    else:
        print(f"  All 4 configs loaded from cache; convergence.log left as-is.")

    return fingerprints


def main_ipguard(yaml_file_path, k_params=(5, 10, 20), sizes=(100, 500),
                 suspect_seeds=None, overlap_rates=None):
    """
    IP-Guard matching-rate evaluation between victim and every declared suspect,
    swept over a Cartesian grid of (k_param, size) pairs.

    For each (k_param, size) pair, generates the four fingerprint sets on the
    victim (TR, TL, RR, RL), then for each suspect computes the matching rate
    under all four configs. Produces:
        4 configs x N suspects x |k_params| x |sizes|
    rows in ./saved_logs/at_eval/IPGuard/master.csv.

    Each row carries Victim_Arch and Suspect_Arch as structured columns so
    cross-architecture sweeps (e.g. ResNet-18 victim vs VGG16 negatives)
    produce unambiguous, queryable data.

    Backward-compatible defaults preserve the old single-point evaluation
    behavior by setting k_params and sizes to one-element tuples. To restrict
    a run to a single (k, size), pass e.g. k_params=(5,), sizes=(100,).
    """
    # IPGuard doesn't need adversarial examples or a test loader; build only
    # the dataset object and victim model.
    ctx = _setup_victim_context(yaml_file_path, build_raw_loader=False)
    exp_yaml      = ctx["exp_yaml"]
    victim_cfg    = ctx["victim_cfg"]
    victim_ds_cfg = ctx["victim_ds_cfg"]
    victim_ds_obj = ctx["victim_dataset_obj"]
    num_classes   = ctx["victim_num_classes"]
    victim_model  = ctx["victim_model"]
    victim_arch   = ctx["victim_arch"]
    victim_id     = ctx["victim_id"]

    settings = exp_yaml.get("IPGuard", {})
    k_params = settings.get("k_Params", k_params)
    sizes = settings.get("Sizes", sizes)
    master_path = Path(settings.get("Master_CSV", IPGUARD_MASTER_CSV_PATH))

    s_type, suspect_cfg = _detect_suspect_block(exp_yaml)
    suspect_arch = suspect_cfg.get("Model", "UnknownArch")
    print(f"  Suspect block: {s_type}")
    print(f"  Suspect arch:  {suspect_arch}")
    print(f"  Sweep grid: k_params={list(k_params)}, sizes={list(sizes)} "
          f"({len(k_params) * len(sizes)} (k, size) pairs)")
    print(f"  Master table -> {master_path}")

    # ---- Pre-generate all fingerprint sets across the sweep grid ----------
    # Cache: {(k, size) -> {tag -> fp_data}}
    # Disk-backed by _get_or_generate_ipguard_fingerprints, so re-runs are free.
    # Fingerprints depend only on the victim, so they're keyed strictly by
    # victim_arch + victim_id (no suspect info in the cache path).
    fingerprint_grid = {}
    for k_param in k_params:
        for size in sizes:
            print(f"\n--- Preparing fingerprints for k={k_param}, size={size} ---")
            fingerprint_grid[(k_param, size)] = _get_or_generate_ipguard_fingerprints(
                victim_model, num_classes, victim_ds_cfg,
                victim_arch, victim_id,
                victim_ds_obj.raw_train_set, k_param, size,
            )

    # ---- Per-suspect evaluation across the sweep grid ---------------------
    # Outer: suspects (so each suspect model is loaded once).
    # Inner: (k, size) grid, then four FP configs per (k, size).
    for rec in _iter_suspects(s_type, suspect_cfg, victim_ds_obj, num_classes,
                              victim_arch, victim_id,
                              suspect_seeds, overlap_rates):
        for (k_param, size), fingerprints in fingerprint_grid.items():
            for tag, fp_data in fingerprints.items():
                result = verify_fingerprint(rec["model"], fp_data, device=device)
                matching_rate = result["matching_rate"]

                print(f"    [{rec['scenario_name'][:60]}...] "
                      f"victim={rec['victim_arch']} suspect={rec['suspect_arch']} "
                      f"k={k_param} size={size} {tag}: "
                      f"matching_rate = {matching_rate:.4f}")

                append_master_row(master_path, {
                    "Run_Timestamp":  datetime.now().isoformat(timespec="seconds"),
                    "Scenario_Name":  rec["scenario_name"],
                    "Victim_Arch":    rec["victim_arch"],
                    "Suspect_Arch":   rec["suspect_arch"],
                    "Suspect_Type":   s_type,
                    "Checkpoint":     rec["eval_mode"],
                    "FP_Config":      tag,
                    "k_Param":        k_param,
                    "Size":           size,
                    "Matching_Rate":  round(matching_rate, 4),
                }, columns=IPGUARD_MASTER_COLUMNS, dedup_key=IPGUARD_DEDUP_KEY)

    print(f"\n  Results -> {master_path}")


"""-------------------------- Main Execution Script---------------------------------"""

# =====================================================
# 11. Name-based dispatcher
# =====================================================
EVAL_REGISTRY = {
    "ipguard":  main_ipguard,
    # "robd": main_robd,  # superseded by main_robd_jsd
}


def run_eval(name, yaml_path, **kwargs):
    """Run one evaluation method by name."""
    name = name.lower().strip()
    if name not in EVAL_REGISTRY:
        raise ValueError(
            f"Unknown evaluation '{name}'. "
            f"Available: {sorted(EVAL_REGISTRY.keys())}"
        )

    print(f"\n{'#' * 60}")
    print(f"#  Running: {name}")
    print(f"#  YAML  : {yaml_path}")
    print(f"{'#' * 60}")

    return EVAL_REGISTRY[name](yaml_path, **kwargs)


def run_evals(names, yaml_path, **kwargs):
    """Run a sequence of evaluations by name on the same YAML."""
    results = {}
    for name in names:
        results[name] = run_eval(name, yaml_path, **kwargs)
    return results


# =====================================================
# 12. Entry point
# =====================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IPGuard evaluation from YAML plans")
    parser.add_argument("--yaml", action="append")
    parser.add_argument("--plan-dir", default="./saved_exp_plan/at_eval_plan")
    cli = parser.parse_args()
    torch.multiprocessing.set_start_method("spawn", force=True)
    set_seed(42)

    # Folder containing all YAML experiment plans
    exp_dir = cli.plan_dir
    yaml_files = cli.yaml or sorted(glob.glob(os.path.join(exp_dir, "*.yaml")))

    if not yaml_files:
        print(f"No YAML files found in {exp_dir}")
    else:
        print(f"Found {len(yaml_files)} experiment plan(s):")
        for f in yaml_files:
            print(" -", f)

    methods_to_run = [
        "ipguard",
    ]

    for yaml_path in yaml_files:
        print(f"\n{'=' * 60}")
        print(f"  {yaml_path}")
        print(f"{'=' * 60}")

        run_evals(methods_to_run, yaml_path, k_params=(0.01, 5, ), sizes=(100,))
