import os
import re
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
# 2. Master table configuration
# =====================================================
MASTER_CSV_DIR = Path("./saved_logs/at_eval/DeepJudge")

# Architecture is first-class: Victim_Arch and Suspect_Arch are dedicated
# columns. Scenario_Name is architecture-free and captures only the
# experimental condition (dataset, seed, overlap, AT params, epoch).
MASTER_COLUMNS = [
    "Run_Timestamp", "Scenario_Name",
    "Victim_Arch", "Suspect_Arch",
    "Suspect_Type", "Checkpoint", "Eval_Attack",
    "Rob_Victim", "Rob_Suspect", "RobD", "JSD_Suspect",
    "Tau_RobD", "Tau_JSD",
    "RobD_Vote_Positive", "JSD_Vote_Positive",
    "P_Copy", "Stolen",
]

# Dedup key includes both architectures so cross-architecture sweeps with
# the same scenario string coexist as distinct rows.
MASTER_DEDUP_KEY = (
    "Scenario_Name",
    "Victim_Arch", "Suspect_Arch",
    "Suspect_Type", "Checkpoint", "Eval_Attack",
)


def master_csv_path_for(selection_tag):
    """
    Map a selection_tag from _get_or_generate_adv_examples to its master CSV path.

      selection_tag = ""                 -> master.csv                 (full test set)
      selection_tag = "_cschunks_1-2-3"  -> master_cschunks_1-2-3.csv  (top-3000 by CS)

    Each selection mode gets its own master CSV so rows from different
    sample selections never collide on the dedup key.
    """
    stem = "master"
    if selection_tag:
        # selection_tag has a leading underscore (e.g. "_cschunks_1-2-3");
        # the underscore between "master" and the tag is supplied by the tag itself.
        stem = f"master{selection_tag}"
    return MASTER_CSV_DIR / f"{stem}.csv"


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


def _write_master_rows(csv_path, rows):
    """Write rows back, preserving column order."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MASTER_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_master_row(csv_path, row_dict):
    """
    Append a row to the master table at `csv_path` with dedup on MASTER_DEDUP_KEY.
    New row overwrites any existing row with the same key.
    """
    missing = set(MASTER_COLUMNS) - set(row_dict.keys())
    extra   = set(row_dict.keys()) - set(MASTER_COLUMNS)
    if missing:
        raise ValueError(f"append_master_row missing columns: {missing}")
    if extra:
        raise ValueError(f"append_master_row unknown columns: {extra}")

    existing = _read_master_rows(csv_path)
    new_key  = tuple(str(row_dict[k]) for k in MASTER_DEDUP_KEY)

    # Drop any existing row with the same key (new row wins)
    kept = [r for r in existing
            if tuple(r.get(k, "") for k in MASTER_DEDUP_KEY) != new_key]
    kept.append(row_dict)

    _write_master_rows(csv_path, kept)


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
# 7. Shared boilerplate (victim + adv + IPGuard fp + DI regressor)
# =====================================================
def _build_victim_id(victim_cfg, victim_ds_cfg):
    """
    Build an architecture-free victim identifier.

    Example output: 'CIFAR-10_25000_seed=42_overlap=1.0'

    Used in both the master CSV's Scenario_Name (so rows for the same
    victim group together regardless of suspect architecture) and as
    part of the threshold cache filename.

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
        raw_loader = build_raw_test_loader(victim_dataset_obj,
            batch_size=int(exp_yaml.get("DeepJudge", {}).get("Batch_Size", 1000)))

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
    over real epoch_N.pth files), or None for the logical aliases
    'best_epoch' / 'last_epoch' returned by collect_checkpoints in
    `State: best` / `State: last` modes.

    Examples:
      'epoch_47'   -> 47
      'best_epoch' -> None
      'last_epoch' -> None
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
      - path-based suspects:  "{dir_leaf}_epoch={N}" where N is parsed from
                              the checkpoint filename. With State: all this
                              gives one distinct, non-colliding row per epoch.
      - grid-based negatives: f"{victim_id}_seed={s}_overlap={o}"

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


def _infer_arch_from_dir(dir_name, default_arch):
    """
    Infer the suspect architecture from the directory name when it encodes
    the surrogate architecture via the knockoff/extraction naming convention.

    Convention: a '{Cross|Same}{NN}' token names the surrogate architecture,
    where NN is the architecture's signature depth:
        16 -> VGG16        (e.g. 'Cross16', 'Same16')
        18 -> ResNet-18    (e.g. 'Same18',  'Cross18')

    Returns the matched architecture string, or `default_arch` (the YAML
    `Model`) when no token is present, so non-extraction paths are unaffected.
    """
    if re.search(r"(?:Cross|Same)16(?:[_/]|$)", dir_name):
        return "VGG16"
    if re.search(r"(?:Cross|Same)18(?:[_/]|$)", dir_name):
        return "ResNet-18"
    return default_arch


def _iter_suspects_by_paths(suspect_type, suspect_cfg,
                            victim_dataset_obj, num_classes,
                            victim_arch, suspect_arch):
    """
    Shared path-based iteration for both 'positive' and explicit-path
    'negative' suspects. Yields records tagged with `suspect_type`, the
    YAML-supplied `eval_mode`, and a per-suspect `scenario_name`.

    Per-row scenario_name = "{dir_leaf}_epoch={N}", where N is parsed from
    the checkpoint filename. With State: all this gives one distinct row
    per epoch in the master CSV (no dedup collisions across epochs).

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
        # Architecture can vary per path (e.g. knockoff surrogates trained as
        # ResNet-18 'Same18' vs VGG16 'Cross16'); infer it from the dir name,
        # falling back to the YAML `Model` when no arch token is present.
        path_arch = _infer_arch_from_dir(dir_name, model_name)
        print(f"\n--- {type_label} Suspect ({path_arch}): {dir_name} ---")

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
        # Per-checkpoint scenario gets '_epoch=N' appended inside the loop.
        dir_leaf = dir_name.rstrip("/").split("/")[-1]

        for ckpt_name, ckpt_path in checkpoints:
            # epoch is an int for 'all' mode (epoch_N), None for 'best'/'last'.
            epoch = _parse_epoch_from_ckpt(ckpt_name)
            per_row_scenario = (
                f"{dir_leaf}_epoch={epoch}" if epoch is not None else dir_leaf
            )

            print(f"  Evaluating: {ckpt_name}"
                  + (f" (epoch={epoch})" if epoch is not None else ""))

            net = load_positive_suspect(
                path_arch, num_classes, victim_dataset_obj, ckpt_path,
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
                "suspect_arch": path_arch,
            }
            del net
            torch.cuda.empty_cache()


'''--------------Metrics (Deep-Judge family)----------------------'''

def compute_robustness(model, adv_dataset, device="cuda", batch_size=256):
    """Rob(f, T) = accuracy of model on adversarial test set T."""
    model.eval()
    x_adv = adv_dataset["adv_images"]
    y_true = adv_dataset["labels"]
    correct, total = 0, len(y_true)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        xb = x_adv[start:end].to(device)
        yb = y_true[start:end].to(device)
        with torch.no_grad():
            correct += (model(xb).argmax(1) == yb).sum().item()

    return correct / total


def compute_robd(rob_victim, rob_suspect):
    return abs(rob_suspect - rob_victim)


def compute_jsd(victim_model, suspect_model, data_loader, device="cuda"):
    """
    Jensen-Shannon Distance between victim and suspect output distributions,
    averaged over the provided data_loader.
    """
    victim_model.eval()
    suspect_model.eval()

    total_jsd = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            p = F.softmax(victim_model(images), dim=1)
            q = F.softmax(suspect_model(images), dim=1)
            m = (p + q) / 2

            eps = 1e-10
            kl_p_m = (p * (torch.log(p + eps) - torch.log(m + eps))).sum(dim=1)
            kl_q_m = (q * (torch.log(q + eps) - torch.log(m + eps))).sum(dim=1)
            batch_jsd = (kl_p_m + kl_q_m) / 2

            total_jsd += batch_jsd.sum().item()
            total_samples += len(images)

    return total_jsd / total_samples


# ---- Constants from the DeepJudge paper -----------------------------------
DEEPJUDGE_ALPHA_BLACKBOX = 0.9    # relaxing factor for black-box metrics
DEEPJUDGE_ALPHA_WHITEBOX = 0.6    # (kept for completeness; unused here)
DEEPJUDGE_CONFIDENCE     = 0.99   # one-tailed T-test confidence level

# Canonical negative pool used to derive tau (paper Sec. V-A3).
# Same architecture / dataset as the victim, varying only seed and overlap.
THRESHOLD_NEG_SEEDS    = list(range(42, 52))    # 42..51 inclusive
THRESHOLD_NEG_OVERLAPS = [0.0, 1.0]             # disjoint + identical training data

VICTIM_SEED = 42
VICTIM_OVERLAP = 1.0


# =====================================================
# Threshold computation + I/O
# =====================================================
def compute_deepjudge_threshold(neg_values, alpha, confidence=DEEPJUDGE_CONFIDENCE):
    """
    Compute the DeepJudge threshold tau from negative-suspect metric values.

    Steps:
      1. Sample mean mu_hat and standard error SE.
      2. One-tailed lower bound at `confidence` level:
             LB = mu_hat - t_{confidence, n-1} * SE
      3. tau = alpha * LB

    Returns
    -------
    dict with keys: mean, se, n, t_crit, lb, alpha, tau
    """
    arr = np.asarray(neg_values, dtype=float)
    n = len(arr)
    if n < 2:
        raise ValueError(
            f"Need at least 2 negative-suspect values to compute T-test bound; got {n}."
        )

    mean   = float(arr.mean())
    se     = float(arr.std(ddof=1) / np.sqrt(n))
    t_crit = float(stats.t.ppf(confidence, df=n - 1))   # one-tailed
    lb     = mean - t_crit * se
    tau    = alpha * lb

    return {
        "mean": mean, "se": se, "n": n, "t_crit": t_crit,
        "lb": lb, "alpha": alpha, "tau": tau,
    }


def deepjudge_threshold_path(metric_name, victim_arch, victim_id, attack_suffix):
    """
    Where the threshold for a given (metric, victim_arch, victim_id, attack)
    is cached.

    The path includes `victim_arch` because the threshold is calibrated on
    the negative pool of the victim's architecture — a ResNet-18 victim
    and a VGG16 victim have genuinely different thresholds, and the cache
    must not conflate them.
    """
    return Path(
        f"./saved_logs/at_eval/DeepJudge/Threshold/{metric_name}/"
        f"victim={victim_arch}_{victim_id}_{metric_name}_"
        f"{attack_suffix}_threshold.json"
    )


def save_deepjudge_threshold(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_deepjudge_threshold(path):
    if not Path(path).exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


# =====================================================
# Threshold builder — metric-agnostic
# =====================================================
def build_metric_threshold(
    metric_name,                    # "RobD", "JSD", ...
    metric_fn,                      # callable(suspect_net) -> float
    victim_cfg, victim_dataset_obj, victim_num_classes,
    victim_arch, victim_id,         # architecture + identifier for cache path
    attack_suffix,                  # attack_suffix used only for filename
    extra_payload=None,             # dict of metric-specific fields to embed in JSON
    alpha=DEEPJUDGE_ALPHA_BLACKBOX,
):
    """
    Construct the DeepJudge threshold tau for `metric_name` from the canonical
    negative pool. Generic over the metric: caller supplies `metric_fn` which
    takes a suspect network and returns a scalar.

    Pool: 20 models = (seeds 42..51) x (overlap {0.0, 1.0}), minus the victim.
    The pool inherits the victim's architecture and Model_Name — this is part
    of DeepJudge's methodology: the threshold is calibrated on models matching
    the victim's training setup, then applied to suspects of any architecture.

    Saves the threshold record (including per-model values) to disk and returns it.
    """
    print("\n" + "=" * 50)
    print(f"  Building DeepJudge {metric_name} threshold from negative pool")
    print(f"  Victim arch: {victim_arch}, victim_id: {victim_id}")
    print("=" * 50)
    print(f"  Pool: seeds {THRESHOLD_NEG_SEEDS}, overlaps {THRESHOLD_NEG_OVERLAPS}")
    print(f"  Expected size: "
          f"{len(THRESHOLD_NEG_SEEDS) * len(THRESHOLD_NEG_OVERLAPS) - 1} models "
          f"(victim excluded)")

    # The pool config inherits the victim's architecture and Model_Name,
    # since these negatives share the victim's training setup (apart from
    # seed and data overlap).
    threshold_neg_cfg = {
        "Model":      victim_cfg.get("Model", "ResNet-18"),
        "Model_Name": victim_cfg["Model_Name"],
    }

    neg_values    = []
    per_model_log = []

    for seed in THRESHOLD_NEG_SEEDS:
        for overlap in THRESHOLD_NEG_OVERLAPS:

            if seed == VICTIM_SEED and float(overlap) == float(VICTIM_OVERLAP):
                print(f"\n  --- Threshold pool: seed={seed}, overlap={overlap} ---")
                print("    [SKIP] This is the victim model; exclude from threshold pool.")
                continue

            print(f"\n  --- Threshold pool: seed={seed}, overlap={overlap} ---")
            net = load_negative_suspect(
                threshold_neg_cfg, victim_dataset_obj, victim_num_classes,
                seed, overlap,
            )
            if net is None:
                print(f"    [SKIP] No checkpoint for seed={seed}, overlap={overlap}")
                continue

            value = metric_fn(net)
            print(f"    {metric_name} = {value:.4f}")

            neg_values.append(value)
            per_model_log.append({
                "seed": seed, "overlap": overlap,
                metric_name.lower(): round(value, 4),
            })

            del net
            torch.cuda.empty_cache()

    if len(neg_values) < 2:
        raise RuntimeError(
            f"Threshold pool has only {len(neg_values)} usable models for {metric_name}. "
            f"Need at least 2 for a T-test. Check that the negative checkpoints "
            f"exist under ./saved_models/vanilla/{victim_cfg['Model_Name']}_*_*."
        )

    info = compute_deepjudge_threshold(neg_values, alpha=alpha)

    payload = {
        "victim_arch":    victim_arch,
        "victim_id":      victim_id,
        "metric":         metric_name,
        "attack_suffix":  attack_suffix,
        "pool_seeds":     THRESHOLD_NEG_SEEDS,
        "pool_overlaps":  THRESHOLD_NEG_OVERLAPS,
        "excluded_from_pool": [{
            "reason":  "victim_model",
            "seed":    VICTIM_SEED,
            "overlap": VICTIM_OVERLAP,
        }],
        "pool_size_used": len(neg_values),
        "pool_models":    per_model_log,
        **(extra_payload or {}),
        **info,
    }

    threshold_path = deepjudge_threshold_path(
        metric_name, victim_arch, victim_id, attack_suffix
    )
    save_deepjudge_threshold(threshold_path, payload)

    print(f"\n  {metric_name} threshold pool stats (n={info['n']}):")
    print(f"    mean = {info['mean']:.4f},  SE = {info['se']:.4f}")
    print(f"    t_crit({DEEPJUDGE_CONFIDENCE}, df={info['n']-1}) = {info['t_crit']:.4f}")
    print(f"    LB   = {info['lb']:.4f},  alpha = {info['alpha']},  "
          f"tau = {info['tau']:.4f}")
    print(f"  Saved -> {threshold_path}")

    return payload


# =====================================================
# 6. Adversarial example helpers
# =====================================================
def generate_and_save_adv_examples(
    victim_model, data_loader, save_path, attack_fn, attack_kwargs, device="cuda"
):
    """Generate adv examples on the victim and save them; reused for all suspects."""
    victim_model.eval()
    all_adv, all_clean, all_labels = [], [], []

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = attack_fn(victim_model, images, labels, **attack_kwargs)
        all_adv.append(adv_images.cpu())
        all_clean.append(images.cpu())
        all_labels.append(labels.cpu())

    adv_dataset = {
        "adv_images": torch.cat(all_adv, dim=0),
        "clean_images": torch.cat(all_clean, dim=0),
        "labels": torch.cat(all_labels, dim=0),
        "parameters": attack_kwargs,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(adv_dataset, save_path)
    print(f"Saved {len(adv_dataset['labels'])} adversarial examples to {save_path}")
    return adv_dataset


def build_attack_suffix(attack_name, attack_kwargs, selection_tag=""):
    param_str = "_".join(f"{k}={v}" for k, v in sorted(attack_kwargs.items()))
    return f"{attack_name}_{param_str}{selection_tag}"


def build_adv_save_path(victim_cfg, attack_name, attack_kwargs, selection_tag=""):
    """
    Adv examples are generated on the victim, so the cache path is keyed by
    the victim's dataset and architecture. Naturally architecture-aware.
    """
    return (f"./Indices/adv_examples/"
            f"{victim_cfg['Dataset']['name']}_{victim_cfg['Model']}_"
            f"{build_attack_suffix(attack_name, attack_kwargs, selection_tag)}.pt")


def _get_or_generate_adv_examples(
    victim_model, victim_cfg, raw_loader, exp_yaml, victim_dataset_obj=None,
):
    if not exp_yaml.get("Attack"):
        raise KeyError(
            "DeepJudge (RobD) needs an 'Attack' block in the plan YAML, e.g.\n"
            "  Attack:\n    name: PGD\n    eps: 0.03\n    steps: 10\n    step_size: 0.003\n"
            "It selects the adversarial test set and the RobD threshold cache "
            "(Indices/adv_examples/*, saved_logs/at_eval/DeepJudge/Threshold/RobD/*)."
        )
    attack_cfg  = dict(exp_yaml["Attack"])
    attack_name = attack_cfg.pop("name")
    cs_chunks   = attack_cfg.pop("cs_chunks", None)
    if isinstance(cs_chunks, int):
        cs_chunks = [cs_chunks]

    attack_kwargs = attack_cfg
    attack_fn = ATTACK_REGISTRY[attack_name]

    selection_tag = (f"_cschunks_{'-'.join(map(str, cs_chunks))}"
                     if cs_chunks else "")
    print(f"Attack: {attack_name}, {attack_kwargs}, selection={selection_tag or 'full'}")

    adv_save_path = build_adv_save_path(
        victim_cfg, attack_name, attack_kwargs, selection_tag
    )

    if Path(adv_save_path).exists():
        adv_dataset = load_adv_examples(adv_save_path)
    else:
        if cs_chunks:
            # Build cache on first use.
            chunks_dir = Path(f"./Indices/cs_cache/"
                              f"{victim_cfg['Dataset']['name']}_{victim_cfg['Model']}/chunks")
            if not chunks_dir.exists():
                build_cs_cache(victim_model, victim_cfg, raw_loader)

            indices = load_cs_chunks(victim_cfg, cs_chunks).tolist()
            subset  = torch.utils.data.Subset(victim_dataset_obj.raw_test_set, indices)
            gen_loader = DataLoader(subset, batch_size=1000, shuffle=False,
                                    num_workers=4, pin_memory=True)
            print(f"  Using {len(indices)} samples from chunks {cs_chunks}")
        else:
            gen_loader = raw_loader

        adv_dataset = generate_and_save_adv_examples(
            victim_model, gen_loader, adv_save_path, attack_fn, attack_kwargs
        )

    return adv_dataset, attack_name, attack_kwargs, selection_tag


# =====================================================
# CS-ranked sample cache (disjoint chunks)
# =====================================================
# A one-time per-victim cache:
#   - cs_ranking.pt        : full descending CS ranking (all N indices)
#   - chunks/chunk_0001.pt : ranks 0..999     (top 1000 by CS)
#   - chunks/chunk_0002.pt : ranks 1000..1999 (next 1000)
#   - ...
#   - chunks/chunk_0010.pt : ranks 9000..9999 (lowest 1000 by CS)
#
# Each chunk is DISJOINT. Compose any range by listing the chunk IDs you want:
#   top-3000      -> [1, 2, 3]
#   middle slice  -> [4, 5, 6, 7]
#   bottom-1000   -> [10]
#   high + low    -> [1, 10]
#   full set      -> [1, 2, ..., 10]

CHUNK_STEP = 1000


@torch.no_grad()
def compute_cs(model, data_loader, device="cuda"):
    """CS(f, x) = sum_i f_i(x)^2  (DeepGini Gini-purity)"""
    model.eval()
    out = []
    for x, _ in data_loader:
        probs = F.softmax(model(x.to(device)), dim=1)
        out.append((probs ** 2).sum(dim=1).cpu())
    return torch.cat(out)


def build_cs_cache(victim_model, victim_cfg, raw_loader, device="cuda"):
    """Compute CS, sort descending, save disjoint 1000-sample chunks.

    CS values depend on the victim model, so the cache is keyed by the
    victim's dataset and architecture — naturally architecture-aware.
    """
    cache_dir = Path(f"./Indices/cs_cache/"
                     f"{victim_cfg['Dataset']['name']}_{victim_cfg['Model']}")
    chunks_dir = cache_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    cs = compute_cs(victim_model, raw_loader, device)
    rank_order = torch.argsort(cs, descending=True)
    n = len(cs)
    print(f"  CS over {n} samples: min={cs.min():.4f}, max={cs.max():.4f}")

    for cid in range(1, (n + CHUNK_STEP - 1) // CHUNK_STEP + 1):
        start, end = CHUNK_STEP * (cid - 1), min(CHUNK_STEP * cid, n)
        torch.save(rank_order[start:end], chunks_dir / f"chunk_{cid:04d}.pt")
        print(f"  chunk_{cid:04d}: ranks [{start}, {end}), "
              f"CS [{cs[rank_order[start:end]].min():.4f}, "
              f"{cs[rank_order[start:end]].max():.4f}]")


def load_cs_chunks(victim_cfg, chunk_ids):
    """Load and concatenate disjoint chunks. chunk_ids: int or list of ints.
    Chunk size is fixed at build time (CHUNK_STEP = 1000)."""
    if isinstance(chunk_ids, int):
        chunk_ids = [chunk_ids]
    chunks_dir = Path(f"./Indices/cs_cache/"
                      f"{victim_cfg['Dataset']['name']}_{victim_cfg['Model']}/chunks")
    return torch.cat([torch.load(chunks_dir / f"chunk_{c:04d}.pt") for c in chunk_ids])


"--------------------- Unified main_* per method ------------------------"

def main_robd_jsd(yaml_file_path, suspect_seeds=None, overlap_rates=None):
    """
    Joint RobD + JSD evaluation against the DeepJudge threshold framework.

    Writes one row per suspect (or per epoch, when State: all) to a master
    table under ./saved_logs/at_eval/DeepJudge/. The exact file depends on
    the sample selection mode read from the YAML's Attack block:
      - cs_chunks omitted      -> master.csv                  (full test set)
      - cs_chunks: [1, 2, 3]   -> master_cschunks_1-2-3.csv   (top-3000 by CS)
    P_Copy is the mean of the two binary votes; Stolen = int(P_Copy > 0.5).

    Each row carries Victim_Arch and Suspect_Arch as structured columns so
    cross-architecture sweeps (e.g. ResNet-18 victim vs VGG16 negatives)
    produce unambiguous, queryable data.

    Negative suspects support two modes:
      - default                -> seed x overlap grid
      - explicit Model_Path    -> directory iteration

    For path-based suspects (positive, or negative with Model_Path), the
    Scenario_Name column is "{dir_leaf}_epoch={N}" so each checkpoint epoch
    produces a distinct, non-colliding row in the master CSV.
    """
    ctx = _setup_victim_context(yaml_file_path, build_raw_loader=True)
    exp_yaml          = ctx["exp_yaml"]
    victim_cfg        = ctx["victim_cfg"]
    victim_ds_obj     = ctx["victim_dataset_obj"]
    num_classes       = ctx["victim_num_classes"]
    victim_model      = ctx["victim_model"]
    victim_arch       = ctx["victim_arch"]
    victim_id         = ctx["victim_id"]
    raw_loader        = ctx["raw_loader"]

    s_type, suspect_cfg = _detect_suspect_block(exp_yaml)
    suspect_arch = suspect_cfg.get("Model", "UnknownArch")
    print(f"  Suspect block: {s_type}")
    print(f"  Suspect arch:  {suspect_arch}")

    # ---- Adv examples + victim robustness ---------------------------------
    adv_dataset, attack_name, attack_kwargs, selection_tag = (
        _get_or_generate_adv_examples(
            victim_model, victim_cfg, raw_loader, exp_yaml,
            victim_dataset_obj=victim_ds_obj,
        )
    )
    eval_batch = int(exp_yaml.get("DeepJudge", {}).get("Batch_Size", 256))
    rob_victim = compute_robustness(victim_model, adv_dataset, batch_size=eval_batch)
    print(f"  Rob(victim) = {rob_victim:.4f}  ({rob_victim * 100:.2f}%)")

    eval_attack_str = format_eval_attack_string(attack_name, attack_kwargs)
    attack_suffix   = build_attack_suffix(attack_name, attack_kwargs, selection_tag)

    # Resolve which master CSV this run writes to (one CSV per selection mode).
    master_csv = Path(exp_yaml.get("DeepJudge", {}).get("Master_CSV", master_csv_path_for(selection_tag)))
    print(f"  Master table -> {master_csv}")

    # ---- RobD threshold ---------------------------------------------------
    robd_threshold_path = deepjudge_threshold_path(
        "RobD", victim_arch, victim_id, attack_suffix
    )
    robd_threshold = load_deepjudge_threshold(robd_threshold_path)
    if robd_threshold is None:
        print(f"\n  No RobD threshold cache at {robd_threshold_path}; building it now.")

        def robd_metric_fn(suspect_net):
            rob_s = compute_robustness(suspect_net, adv_dataset)
            return compute_robd(rob_victim, rob_s)

        robd_threshold = build_metric_threshold(
            metric_name="RobD",
            metric_fn=robd_metric_fn,
            victim_cfg=victim_cfg,
            victim_dataset_obj=victim_ds_obj,
            victim_num_classes=num_classes,
            victim_arch=victim_arch,
            victim_id=victim_id,
            attack_suffix=attack_suffix,
            extra_payload={
                "attack_name":   attack_name,
                "attack_kwargs": attack_kwargs,
                "selection_tag": selection_tag,
                "rob_victim":    round(rob_victim, 4),
            },
        )
    else:
        print(f"\n  Loaded cached RobD threshold from {robd_threshold_path}")
        print(f"    tau_RobD = {robd_threshold['tau']:.4f}  "
              f"(LB = {robd_threshold['lb']:.4f}, "
              f"alpha = {robd_threshold['alpha']}, "
              f"n_neg = {robd_threshold['n']})")

    tau_robd = robd_threshold["tau"]

    # ---- JSD threshold (attack-independent; suffix='clean') ---------------
    jsd_threshold_path = deepjudge_threshold_path(
        "JSD", victim_arch, victim_id, "clean"
    )
    jsd_threshold = load_deepjudge_threshold(jsd_threshold_path)
    if jsd_threshold is None:
        print(f"\n  No JSD threshold cache at {jsd_threshold_path}; building it now.")

        def jsd_metric_fn(suspect_net):
            return compute_jsd(victim_model, suspect_net, raw_loader)

        jsd_threshold = build_metric_threshold(
            metric_name="JSD",
            metric_fn=jsd_metric_fn,
            victim_cfg=victim_cfg,
            victim_dataset_obj=victim_ds_obj,
            victim_num_classes=num_classes,
            victim_arch=victim_arch,
            victim_id=victim_id,
            attack_suffix="clean",
        )
    else:
        print(f"\n  Loaded cached JSD threshold from {jsd_threshold_path}")
        print(f"    tau_JSD = {jsd_threshold['tau']:.4f}  "
              f"(LB = {jsd_threshold['lb']:.4f}, "
              f"alpha = {jsd_threshold['alpha']}, "
              f"n_neg = {jsd_threshold['n']})")

    tau_jsd = jsd_threshold["tau"]

    # ---- Per-suspect evaluation ------------------------------------------
    for rec in _iter_suspects(s_type, suspect_cfg, victim_ds_obj, num_classes,
                              victim_arch, victim_id,
                              suspect_seeds, overlap_rates):
        # RobD
        rob_s = compute_robustness(rec["model"], adv_dataset, batch_size=eval_batch)
        robd  = compute_robd(rob_victim, rob_s)

        # JSD
        jsd_val = compute_jsd(victim_model, rec["model"], raw_loader)

        # Votes (low value = suspicious for both metrics)
        robd_vote = int(robd    <= tau_robd)
        jsd_vote  = int(jsd_val <= tau_jsd)

        # Ensemble decision
        p_copy = (robd_vote + jsd_vote) / 2.0
        stolen = int(p_copy > 0.5)

        print(f"    [{rec['scenario_name'][:60]}...] "
              f"victim={rec['victim_arch']} suspect={rec['suspect_arch']} "
              f"Rob_s={rob_s:.4f}  RobD={robd:.4f}  JSD={jsd_val:.4f}  "
              f"votes=({robd_vote},{jsd_vote})  P_copy={p_copy:.2f}  stolen={stolen}")

        append_master_row(master_csv, {
            "Run_Timestamp":       datetime.now().isoformat(timespec="seconds"),
            "Scenario_Name":       rec["scenario_name"],
            "Victim_Arch":         rec["victim_arch"],
            "Suspect_Arch":        rec["suspect_arch"],
            "Suspect_Type":        s_type,
            "Checkpoint":          rec["eval_mode"],
            "Eval_Attack":         eval_attack_str,
            "Rob_Victim":          round(rob_victim, 4),
            "Rob_Suspect":         round(rob_s, 4),
            "RobD":                round(robd, 4),
            "JSD_Suspect":         round(jsd_val, 4),
            "Tau_RobD":            round(tau_robd, 4),
            "Tau_JSD":             round(tau_jsd, 4),
            "RobD_Vote_Positive":  robd_vote,
            "JSD_Vote_Positive":   jsd_vote,
            "P_Copy":              p_copy,
            "Stolen":              stolen,
        })

    print(f"\n  Results -> {master_csv}")


"""-------------------------- Main Execution Script---------------------------------"""

# =====================================================
# 11. Name-based dispatcher
# =====================================================
EVAL_REGISTRY = {
    "robd_jsd": main_robd_jsd,
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
    parser = argparse.ArgumentParser(description="DeepJudge evaluation from YAML plans")
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
        "robd_jsd",
    ]

    for yaml_path in yaml_files:
        print(f"\n{'=' * 60}")
        print(f"  {yaml_path}")
        print(f"{'=' * 60}")

        run_evals(methods_to_run, yaml_path)
