import os
import csv
import json
import glob
import random
import argparse
import hashlib
from pathlib import Path
from datetime import datetime

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from util_adv import (
    collect_checkpoints,
    load_negative_suspect, load_positive_suspect, load_victim_model,
)
from util import process_yaml_file, build_dataset_from_yaml
from AdvAttack.DI import DatasetInferencePipeline


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
# 2. Master table configuration — DI
# =====================================================
DI_MASTER_CSV_PATH = Path("./saved_logs/at_eval/DI/master_DI.csv")

# Architecture is first-class: Victim_Arch and Suspect_Arch are dedicated
# columns. Scenario_Name is architecture-free and captures only the
# experimental condition (dataset, seed, overlap, AT params, epoch).
DI_MASTER_COLUMNS = [
    "Run_Timestamp", "Scenario_Name",
    "Victim_Arch", "Suspect_Arch",
    "Suspect_Type", "Checkpoint",
    "N_Train_Samples", "N_Test_Samples", "Alpha",
    "Mean_Private", "Mean_Public", "Delta",
    "T_Stat", "P_Value", "Stolen",
]

# Dedup key includes both architectures + the (n_train, n_test) sweep axes
# so repeated runs at the same sweep point overwrite cleanly while different
# sweep points coexist as distinct rows.
DI_DEDUP_KEY = (
    "Scenario_Name",
    "Victim_Arch", "Suspect_Arch",
    "Suspect_Type", "Checkpoint",
    "N_Train_Samples", "N_Test_Samples", "Alpha",
)


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
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=True,
        pin_memory=True,
    )


# =====================================================
# 7. Shared boilerplate (victim + DI regressor)
# =====================================================
def _build_victim_id(victim_cfg, victim_ds_cfg):
    """
    Build an architecture-free victim identifier.

    Example output: 'CIFAR-10_25000_seed=42_overlap=1.0'

    Used in both the master CSV's Scenario_Name (so rows for the same
    victim group together regardless of suspect architecture) and as
    part of the DI regressor cache filename.

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


def _setup_victim_context(yaml_file_path, build_raw_loader=True, load_victim=True):
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

    victim_model = None
    if load_victim:
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
    Extract the integer epoch from a checkpoint name. Returns the int when
    the name follows the 'epoch_N' convention (from `State: all` iteration
    over real epoch_N.pth files, or `State: last` with the updated
    collect_checkpoints that returns the real stem), or None for the
    'best_epoch' alias returned by collect_checkpoints in `State: best`.

    Examples:
      'epoch_47'   -> 47
      'best_epoch' -> None
    """
    stem = Path(ckpt_name).stem
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
        }

    `scenario_name` per record is architecture-free:
      - path-based suspects:  dir_leaf (with optional _epoch=N suffix when
                              the checkpoint follows the epoch_N convention)
      - grid-based negatives: f"{victim_id}_seed={s}_overlap={o}"

    Architecture lives in `victim_arch` and `suspect_arch`, which become
    dedicated columns in the master CSV.

    Negative suspects support TWO modes:
      - default: iterate seeds x overlap_rates (loaded from
        ./saved_models/vanilla/{Model_Name}_{seed}_{overlap}/best_epoch.pth)
      - explicit Model_Path: iterate directories like the positive branch,
        with full checkpoint selection via 'State'
    """
    # A list allows one evaluation YAML to declare several suspect architectures.
    if isinstance(suspect_cfg, list):
        for block in suspect_cfg:
            yield from _iter_suspects(suspect_type, block, victim_dataset_obj, num_classes,
                                      victim_arch, victim_id, suspect_seeds, overlap_rates)
        return
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
    YAML-supplied `eval_mode`, and a per-suspect `scenario_name`.

    Per-row scenario_name = "{dir_leaf}_epoch={N}" when the checkpoint
    name parses as an epoch integer, else just "{dir_leaf}". With State: all
    this gives one distinct row per epoch in the master CSV.
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

        # Base scenario name = leaf segment of the model directory.
        # Per-checkpoint scenario gets '_epoch=N' appended inside the loop
        # whenever the checkpoint name parses as an epoch integer.
        dir_leaf = dir_name.rstrip("/").split("/")[-1]

        for ckpt_name, ckpt_path in checkpoints:
            # epoch is int for 'all' mode (and for 'last' once collect_checkpoints
            # returns the real stem), None for 'best' (alias 'best_epoch').
            epoch = _parse_epoch_from_ckpt(ckpt_name)
            per_row_scenario = (
                f"{dir_leaf}_epoch={epoch}" if epoch is not None else dir_leaf
            )

            print(f"  Evaluating: {ckpt_name}"
                  + (f" (epoch={epoch})" if epoch is not None else ""))

            # Checkpoint_Format / Model_Config: same optional suspect keys as the
            # IPGuard / DeepJudge plans; absent keys keep the old CNN / plain-DeiT load.
            net = load_positive_suspect(
                model_name, num_classes, victim_dataset_obj, ckpt_path,
                checkpoint_format=suspect_cfg.get("Checkpoint_Format"),
                model_config=suspect_cfg.get("Model_Config"),
            )

            yield {
                "type": suspect_type, "model": net,
                "seed": None, "overlap": None,
                "dir_name": dir_name, "ckpt_name": ckpt_name, "ckpt_path": ckpt_path,
                "eval_mode": eval_mode,
                "scenario_name": per_row_scenario,
                "victim_arch": victim_arch,
                "suspect_arch": suspect_arch,
            }
            del net
            torch.cuda.empty_cache()


VICTIM_SEED = 42
VICTIM_OVERLAP = 1.0


# =====================================================
# DI — regressor cache + verification
# =====================================================
def di_regressor_path(victim_arch, victim_id, n_train_samples, private_transform="clean"):
    """
    Where the trained DI regressor for a given (victim_arch, victim_id,
    n_train_samples, private_transform) is cached.

    The path includes `victim_arch` because the regressor is trained on the
    victim's Blind Walk embeddings — a ResNet-18 victim and a VGG16 victim
    produce genuinely different embedding distributions, and the cache
    must not conflate them. `n_train_samples` is also part of the key
    because the embedding distribution learned by the regressor depends on
    how many training points it saw.

    `private_transform` names how the PRIVATE images were fed to the victim
    when the regressor was trained: "clean" (ToTensor only, the protocol used
    since 2026-09-13; both private and public images are un-augmented) gets a
    `_clean` suffix. The older un-suffixed file
    `victim=..._n=1000.pt` predates this tag and its private transform is not
    recorded, so it is deliberately NOT picked up; a clean regressor is
    trained on first use (about two minutes for n=1000).
    """
    tag = "" if private_transform == "legacy" else f"_{private_transform}"
    return Path(
        f"./saved_models/di_regressor/"
        f"victim={victim_arch}_{victim_id}_n={n_train_samples}{tag}.pt"
    )


def _get_or_train_di_pipeline(
    victim_model, victim_dataset_obj, victim_ds_cfg,
    victim_arch, victim_id, n_train_samples, regressor_epochs=30,
):
    """
    Load or train a DI regressor pipeline against the victim. Returns the
    pipeline plus the dataset handles needed for suspect verification.

    The regressor cache is keyed by (victim_arch, victim_id, n_train_samples).
    Suspect architecture does not affect the regressor — it learns "what
    private vs public looks like through the victim's lens", and that lens
    is fully determined by the victim model.
    """
    # Private set uses the CLEAN (un-augmented, ToTensor-only) train images so
    # that private vs public differs ONLY by membership — matching the original
    # Dataset Inference protocol, which drops train augmentation at feature-
    # generation time (funcs.py: transform_train = transform_test). Using the
    # augmented raw_train_set here would confound the distance signal with crop/
    # flip noise and make the embeddings nondeterministic across runs.
    # The same clean set is returned for verification, so regressor training
    # and suspect scoring see identically pre-processed private images.
    raw_train_set = victim_dataset_obj.raw_train_clean_set
    raw_test_set  = victim_dataset_obj.raw_test_set

    train_subset_indices = np.load(
        f"./Indices/{victim_ds_cfg['name']}/group_A_subset_10000_from_25000_seed42.npy"
    )

    pipeline = DatasetInferencePipeline(victim_model, device=device)

    regressor_path = di_regressor_path(victim_arch, victim_id, n_train_samples, private_transform="clean")

    if regressor_path.exists():
        print(f"==> Loading existing regressor from {regressor_path}")
        pipeline.load_regressor(regressor_path)
    else:
        print(f"==> Training new regressor g_V on victim embeddings "
              f"(n_train={n_train_samples})..")
        pipeline.train_regressor(
            private_dataset=raw_train_set,
            public_dataset=raw_test_set,
            private_indices=train_subset_indices,
            n_train_samples=n_train_samples,
            regressor_epochs=regressor_epochs,
        )
        regressor_path.parent.mkdir(parents=True, exist_ok=True)
        pipeline.save_regressor(regressor_path)
        print(f"  Saved regressor -> {regressor_path}")

    return pipeline, raw_train_set, raw_test_set, train_subset_indices


def _load_fixed_di_pipeline(di_cfg, dataset, dataset_cfg):
    """Explicit YAML checkpoint means load-only: never fall back to training."""
    from AdvAttack.DI import ConfidenceRegressor
    path = Path(di_cfg["Regressor_Path"]).resolve(strict=True)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    walk = {key: checkpoint[key] for key in
            ("n_samples", "noise_uniform", "noise_gaussian", "noise_laplace", "max_steps")}
    if checkpoint["embedding_dim"] != 3 * walk["n_samples"]:
        raise ValueError("Saved DI dimensions disagree with saved walk parameters.")
    point_batch = int(di_cfg.get("Point_Batch_Size", 8))
    if point_batch < 1:
        raise ValueError("DI.Point_Batch_Size must be positive.")
    pipeline = DatasetInferencePipeline(None, **walk, point_batch_size=point_batch, device=device)
    pipeline.regressor = ConfidenceRegressor(checkpoint["embedding_dim"], checkpoint["embedding_dim"])
    pipeline.regressor.load_state_dict(checkpoint["state_dict"], strict=True)
    pipeline.regressor = pipeline.regressor.to(device).eval()
    transform = di_cfg["Private_Transform"]
    if transform not in ("legacy_augmented", "clean"):
        raise ValueError("DI.Private_Transform must be legacy_augmented or clean.")
    raw_train = dataset.raw_train_set if transform == "legacy_augmented" else dataset.raw_train_clean_set
    indices_path = Path(di_cfg.get("Private_Indices",
        f"Indices/{dataset_cfg['name']}/group_A_subset_10000_from_25000_seed42.npy"))
    indices = np.load(indices_path)
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or len(indices) == 0
            or len(np.unique(indices)) != len(indices) or indices.min() < 0 or indices.max() >= len(raw_train)):
        raise ValueError("Invalid private indices for DI evaluation.")
    print(f"==> Existing DI regressor: {path}; no training; private input: {transform}")
    return pipeline, raw_train, dataset.raw_test_set, indices


def _file_sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main_di(yaml_file_path,
            n_train_samples_grid=(500, 1000),
            n_test_samples_grid=(200, 500, 1000),
            alpha=0.05,
            suspect_seeds=None, overlap_rates=None, preflight=False):
    """
    Dataset-Inference T-test between victim and every declared suspect,
    swept over a Cartesian grid of (n_train, n_test) pairs.

    When YAML DI.Regressor_Path is set, load that checkpoint only, restore its
    walk parameters, and use the DI settings and Positive groups from YAML.
    This mode never loads the victim or trains a regressor. N_Train_Samples is
    descriptive metadata; N_Test_Samples controls the verification sample grid.
    The cache/train behavior below applies to plans without an explicit path.

    For each n_train value, trains (or loads) a regressor g_V on n_train
    victim Blind Walk embeddings. For each suspect, evaluates against every
    (n_train, n_test) pair: produces |n_train_grid| x |n_test_grid| rows
    per suspect in the master CSV.

    Loop structure:
        for n_train in n_train_grid:           # pre-train all regressors
            pipeline = load_or_train(n_train)
        for suspect in iter_suspects():        # load each suspect once
            for n_train in n_train_grid:       # cheap (regressors already cached)
                for n_test in n_test_grid:     # quick re-verify
                    run verify_suspect()

    The regressor cache is keyed by (victim_arch, victim_id, n_train), so
    each unique n_train produces exactly one regressor file on disk,
    reusable across all subsequent runs.

    Each row carries Victim_Arch and Suspect_Arch as structured columns so
    cross-architecture sweeps (e.g. ResNet-18 victim vs VGG16 negatives)
    produce unambiguous, queryable data.

    For path-based suspects, the Scenario_Name column is
    "{dir_leaf}_epoch={N}" so each checkpoint epoch produces a distinct,
    non-colliding row in the master CSV.
    """
    # DI doesn't need adversarial examples or a separate test loader; the
    # pipeline manages its own data sampling internally.
    plan = process_yaml_file(yaml_file_path)
    di_cfg = plan.get("DI", {})
    fixed = bool(di_cfg.get("Regressor_Path"))
    if preflight and not fixed:
        raise ValueError("--preflight requires an explicit DI.Regressor_Path; it never trains.")
    if fixed:
        # N_Train_Samples describes the saved checkpoint, not a training request.
        n_train_samples_grid = (int(di_cfg["N_Train_Samples"]),)
        n_test_samples_grid = di_cfg.get("N_Test_Samples", [1000])
        if isinstance(n_test_samples_grid, int):
            n_test_samples_grid = [n_test_samples_grid]
        if not n_test_samples_grid or any(not isinstance(n, int) or n < 2 for n in n_test_samples_grid):
            raise ValueError("DI.N_Test_Samples must contain integers >= 2.")
        alpha = float(di_cfg.get("Alpha", alpha))
        if not 0 < alpha < 1:
            raise ValueError("DI.Alpha must be between 0 and 1.")
    elif di_cfg:
        # Train path (no Regressor_Path): the same DI keys are honoured as plan-
        # level overrides of the caller's grids, so a plan can drop
        # Regressor_Path / Private_Transform to switch to the clean protocol and
        # keep its N_Train_Samples / N_Test_Samples / Alpha / Master_CSV.
        # Plans without a DI block behave exactly as before.
        def _as_grid(v):
            return tuple(int(x) for x in (v if isinstance(v, (list, tuple)) else [v]))
        if "N_Train_Samples" in di_cfg:
            n_train_samples_grid = _as_grid(di_cfg["N_Train_Samples"])
        if "N_Test_Samples" in di_cfg:
            n_test_samples_grid = _as_grid(di_cfg["N_Test_Samples"])
            if any(n < 2 for n in n_test_samples_grid):
                raise ValueError("DI.N_Test_Samples must contain integers >= 2.")
        if "Alpha" in di_cfg:
            alpha = float(di_cfg["Alpha"])
            if not 0 < alpha < 1:
                raise ValueError("DI.Alpha must be between 0 and 1.")
        if di_cfg.get("Private_Transform", "clean") != "clean":
            raise ValueError("Without DI.Regressor_Path the train path always uses clean private "
                             "images; remove Private_Transform or set it to 'clean'.")
    master_path = Path(di_cfg["Master_CSV"]) if di_cfg.get("Master_CSV") else DI_MASTER_CSV_PATH
    ctx = _setup_victim_context(yaml_file_path, build_raw_loader=False, load_victim=not fixed)
    exp_yaml      = ctx["exp_yaml"]
    victim_ds_cfg = ctx["victim_ds_cfg"]
    victim_ds_obj = ctx["victim_dataset_obj"]
    num_classes   = ctx["victim_num_classes"]
    victim_model  = ctx["victim_model"]
    victim_arch   = ctx["victim_arch"]
    victim_id     = ctx["victim_id"]

    s_type, suspect_cfg = _detect_suspect_block(exp_yaml)
    suspect_arch = ([block.get("Model", "UnknownArch") for block in suspect_cfg]
                    if isinstance(suspect_cfg, list) else suspect_cfg.get("Model", "UnknownArch"))
    print(f"  Suspect block: {s_type}")
    print(f"  Suspect arch:  {suspect_arch}")
    print(f"  Sweep grid: n_train={list(n_train_samples_grid)}, "
          f"n_test={list(n_test_samples_grid)} "
          f"({len(n_train_samples_grid) * len(n_test_samples_grid)} "
          f"(n_train, n_test) pairs per suspect)")
    print(f"  Master table -> {master_path}")

    # ---- Pre-train (or load) regressors for every n_train in the grid ----
    # Cache: {n_train -> (pipeline, raw_train_set, raw_test_set, indices)}
    # Disk-backed by _get_or_train_di_pipeline, so re-runs are free.
    # Regressors depend only on the victim, so they're keyed strictly by
    # victim_arch + victim_id + n_train (no suspect info in the cache path).
    pipelines_by_n_train = {}
    for n_train in n_train_samples_grid:
        print(f"\n--- Preparing DI pipeline for n_train={n_train} ---")
        if fixed:
            pipelines_by_n_train[n_train] = _load_fixed_di_pipeline(di_cfg, victim_ds_obj, victim_ds_cfg)
            _, private_data, public_data, indices = pipelines_by_n_train[n_train]
            if max(n_test_samples_grid) > min(len(indices), len(public_data)):
                raise ValueError("Requested DI sample count exceeds the available private/public pools.")
        else:
            pipelines_by_n_train[n_train] = _get_or_train_di_pipeline(
                victim_model, victim_ds_obj, victim_ds_cfg,
                victim_arch, victim_id, n_train, regressor_epochs=30,
            )

    provenance = {}
    if fixed:
        provenance = {"Regressor_Path": str(Path(di_cfg["Regressor_Path"]).resolve()),
                      "Regressor_SHA256": _file_sha256(di_cfg["Regressor_Path"]),
                      "Private_Transform": di_cfg["Private_Transform"],
                      "Evaluation_Seed": int(di_cfg.get("Seed", 42)),
                      "Point_Batch_Size": int(di_cfg.get("Point_Batch_Size", 8)),
                      "Private_Indices_SHA256": hashlib.sha256(indices.tobytes()).hexdigest(),
                      "Plan_Path": str(Path(yaml_file_path).resolve())}
    columns = DI_MASTER_COLUMNS + list(provenance) + (["Checkpoint_Path", "Checkpoint_SHA256"] if fixed else [])
    dedup_key = DI_DEDUP_KEY + (tuple(k for k in provenance if k != "Plan_Path") +
                              ("Checkpoint_Path", "Checkpoint_SHA256") if fixed else ())
    evaluated = 0

    # ---- Per-suspect evaluation across the (n_train, n_test) grid --------
    # Outer: suspects (so each suspect model is loaded once).
    # Inner: (n_train, n_test) grid — n_train picks the regressor, n_test
    # controls how many Blind Walk embeddings to draw on the suspect.
    for rec in _iter_suspects(s_type, suspect_cfg, victim_ds_obj, num_classes,
                              victim_arch, victim_id,
                              suspect_seeds, overlap_rates):
        if preflight:
            with torch.no_grad():
                logits = rec["model"](victim_ds_obj.raw_test_set[0][0].unsqueeze(0).to(device))
            if logits.shape != (1, num_classes) or not torch.isfinite(logits).all():
                raise ValueError(f"Invalid suspect output: {rec['ckpt_path']}")
            evaluated += 1
            continue
        for n_train in n_train_samples_grid:
            pipeline, raw_train_set, raw_test_set, train_subset_indices = (
                pipelines_by_n_train[n_train]
            )

            for n_test in n_test_samples_grid:
                if fixed:
                    set_seed(int(di_cfg.get("Seed", 42)))
                result = pipeline.verify_suspect(
                    suspect_model=rec["model"],
                    private_dataset=raw_train_set,
                    public_dataset=raw_test_set,
                    private_indices=train_subset_indices,
                    n_test_samples=n_test,
                    alpha=alpha,
                )

                print(f"    [{rec['scenario_name'][:60]}...] "
                      f"victim={rec['victim_arch']} suspect={rec['suspect_arch']} "
                      f"n_train={n_train} n_test={n_test}: "
                      f"delta={result['delta']:.4f}  "
                      f"t={result['t_stat']:.3f}  "
                      f"p={result['p_value']:.3e}  "
                      f"stolen={int(result['stolen'])}")

                if not all(np.isfinite(result[k]) for k in ("mean_private", "mean_public", "delta", "t_stat", "p_value")):
                    raise ValueError(f"Non-finite DI statistics for {rec['scenario_name']}: {result}")
                checkpoint_info = ({"Checkpoint_Path": str(Path(rec["ckpt_path"]).resolve()),
                                    "Checkpoint_SHA256": _file_sha256(rec["ckpt_path"])}
                                   if fixed and rec["ckpt_path"] else
                                   {"Checkpoint_Path": "", "Checkpoint_SHA256": ""} if fixed else {})
                append_master_row(master_path, {
                    **provenance, **checkpoint_info,
                    "Run_Timestamp":     datetime.now().isoformat(timespec="seconds"),
                    "Scenario_Name":     rec["scenario_name"],
                    "Victim_Arch":       rec["victim_arch"],
                    "Suspect_Arch":      rec["suspect_arch"],
                    "Suspect_Type":      s_type,
                    "Checkpoint":        rec["eval_mode"],
                    "N_Train_Samples":   n_train,
                    "N_Test_Samples":    n_test,
                    "Alpha":             alpha,
                    "Mean_Private":      round(result["mean_private"], 6),
                    "Mean_Public":       round(result["mean_public"], 6),
                    "Delta":             round(result["delta"], 6),
                    "T_Stat":            round(result["t_stat"], 6),
                    # Scientific notation for very small p-values (Excel display
                    # caveat: it may strip the exponent — paste-as-text if needed).
                    "P_Value":           f"{result['p_value']:.6e}",
                    "Stolen":            int(result["stolen"]),
                }, columns=columns, dedup_key=dedup_key)
                evaluated += 1

    if evaluated == 0:
        raise ValueError("No suspects were evaluated. Check the YAML model paths.")
    if preflight:
        print(f"\n  Preflight passed: {evaluated} suspects; no training or DI results written.")
    else:
        print(f"\n  Results -> {master_path}")


"""-------------------------- Main Execution Script---------------------------------"""

# =====================================================
# 11. Name-based dispatcher
# =====================================================
EVAL_REGISTRY = {
    "di": main_di,
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
    parser = argparse.ArgumentParser(description="Run DI evaluation from YAML plans.")
    parser.add_argument("--yaml", action="append", help="Evaluate only this YAML; repeat for multiple plans.")
    parser.add_argument("--plan-dir", default="./saved_exp_plan/di_eval_plan",
                        help="Plan folder used when --yaml is omitted.")
    parser.add_argument("--preflight", action="store_true", help="Validate fixed regressor and suspect loads without evaluating DI.")
    args = parser.parse_args()
    torch.multiprocessing.set_start_method("spawn", force=True)
    set_seed(42)

    # Folder containing all YAML experiment plans
    exp_dir = args.plan_dir
    yaml_files = args.yaml or sorted(glob.glob(os.path.join(exp_dir, "*.yaml")))

    if not yaml_files:
        print(f"No YAML files found in {exp_dir}")
    else:
        print(f"Found {len(yaml_files)} experiment plan(s):")
        for f in yaml_files:
            print(" -", f)

    methods_to_run = [
        "di",
    ]

    for yaml_path in yaml_files:
        print(f"\n{'=' * 60}")
        print(f"  {yaml_path}")
        print(f"{'=' * 60}")

        run_evals(
            methods_to_run, yaml_path,
            n_train_samples_grid=(1000,),
            n_test_samples_grid=(1000,),
            alpha=0.01,
            preflight=args.preflight,
        )
