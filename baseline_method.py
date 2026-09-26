import os
import csv
import glob
import random
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from AdvAttack.DI import DatasetInferencePipeline
from AdvAttack.IP_Guard import IPGuardGenerator, verify_fingerprint
from AdvAttack.pgd import PGD

from util_adv import fgsm_attack, load_adv_examples, pgd_attack_v2
from util import (
    load_last_checkpoint,
    process_yaml_file,
    build_dataset_from_yaml,
    create_or_load_group_A,
    load_best_checkpoint,
)

from Model.ResNet_18 import ResNet18
from Model.VGG16 import ModifiedVGG16
from Model.MLP import MNIST_MLP


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



"------------Common Utility Part-----------------"

def build_model(model_name, num_classes):
    if model_name == "MLP":
        return MNIST_MLP()
    elif model_name == "ResNet-18":
        return ResNet18(num_classes=num_classes)
    elif model_name == "VGG16":
        return ModifiedVGG16(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


ATTACK_REGISTRY = {
    "PGD": pgd_attack_v2,
    "FGSM": fgsm_attack,
}


# =====================================================
# 2. NormalizedModel wrapper
# =====================================================
class NormalizedModel(nn.Module):
    """Wraps a model so it accepts raw [0,1] inputs and normalizes internally."""

    def __init__(self, base_model, mean, std):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        x_norm = (x - self.mean) / self.std
        return self.base_model(x_norm)


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
# 4. Model loading helpers
# =====================================================
def load_victim_model(victim_cfg, dataset_obj, num_classes, model_seed):
    """Load victim model and wrap with NormalizedModel."""
    model_name = victim_cfg.get("Model", "ResNet-18")
    net = build_model(model_name, num_classes).to(device)

    folder = victim_cfg["Model_Name"] + f"_{model_seed}_{1.0}"
    model_dir = Path("./saved_models/vanilla/") / folder
    ckpt = load_best_checkpoint(model_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No victim checkpoint in {model_dir}")

    net.load_state_dict(torch.load(ckpt, map_location=device))
    net = NormalizedModel(net, dataset_obj.mean, dataset_obj.std).to(device)
    net.eval()
    print(f"  Victim loaded from: {ckpt}")
    return net


def load_negative_suspect(suspect_cfg, dataset_obj, num_classes, seed, overlap):
    """Load a negative suspect (independently trained vanilla model)."""
    model_name = suspect_cfg.get("Model", "ResNet-18")
    net = build_model(model_name, num_classes).to(device)

    folder = suspect_cfg["Model_Name"] + f"_{seed}_{overlap}"
    model_dir = Path("./saved_models/vanilla/") / folder
    ckpt = load_best_checkpoint(model_dir)
    if ckpt is None:
        print(f"  [SKIP] No checkpoint found in {model_dir}")
        return None

    net.load_state_dict(torch.load(ckpt, map_location=device))
    net = NormalizedModel(net, dataset_obj.mean, dataset_obj.std).to(device)
    net.eval()
    print(f"  Negative suspect loaded from: {ckpt}")
    return net


def load_positive_suspect(model_name, num_classes, dataset_obj, ckpt_path):
    """Load a positive suspect, transparently unwrapping NormalizedModel checkpoints."""
    net = build_model(model_name, num_classes).to(device)
    state = torch.load(ckpt_path, map_location=device)

    if any(k.startswith("base_model.") for k in state.keys()):
        state = {
            k.replace("base_model.", ""): v
            for k, v in state.items()
            if k not in ("mean", "std")
        }

    net.load_state_dict(state)
    net = NormalizedModel(net, dataset_obj.mean, dataset_obj.std).to(device)
    net.eval()
    return net


def collect_checkpoints(model_dir, eval_mode):
    """Collect checkpoint paths based on evaluation mode."""
    model_dir = Path(model_dir)

    if eval_mode == "best":
        return [("best_epoch", load_best_checkpoint(model_dir))]

    if eval_mode == "last":
        return [("last_epoch", load_last_checkpoint(model_dir))]

    if eval_mode == "all":
        epoch_files = sorted(
            model_dir.glob("epoch_*.pth"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        return [(f.stem, f) for f in epoch_files]

    raise ValueError(f"Unknown eval_mode '{eval_mode}'. Use 'best', 'last', or 'all'.")


# =====================================================
# 7. Shared boilerplate (victim + adv + IPGuard fp + DI regressor)
# =====================================================
def _setup_victim_context(yaml_file_path, build_raw_loader=True):
    """
    Common setup performed by every main_* function:
      - print device, parse YAML
      - build victim dataset object
      - load victim model (seed=42)
      - optionally build raw [0,1] test loader
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

    return {
        "exp_yaml": exp_yaml,
        "scenario_name": scenario_name,
        "victim_cfg": victim_cfg,
        "victim_ds_cfg": victim_ds_cfg,
        "victim_dataset_obj": victim_dataset_obj,
        "victim_num_classes": victim_num_classes,
        "victim_model": victim_model,
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


def _iter_suspects(suspect_type, suspect_cfg, victim_dataset_obj, num_classes,
                   suspect_seeds=None, overlap_rates=None):
    """
    Yield uniform suspect records regardless of suspect type.

    Each record:
        {
          "type":      "negative" | "positive",
          "model":     loaded NormalizedModel,
          "seed":      int or None,
          "overlap":   float or None,
          "dir_name":  str or None,
          "ckpt_name": str or None,
          "ckpt_path": Path or None,
        }
    """
    if suspect_type == "negative":
        seeds    = suspect_seeds  or suspect_cfg.get("seeds")         or _DEFAULT_SUSPECT_SEEDS
        overlaps = overlap_rates  or suspect_cfg.get("overlap_rates") or _DEFAULT_OVERLAP_RATES

        for seed in seeds:
            for overlap in overlaps:
                print(f"\n--- Negative Suspect: seed={seed}, overlap={overlap} ---")
                net = load_negative_suspect(
                    suspect_cfg, victim_dataset_obj, num_classes, seed, overlap
                )
                if net is None:
                    continue
                yield {
                    "type": "negative", "model": net,
                    "seed": seed, "overlap": overlap,
                    "dir_name": None, "ckpt_name": None, "ckpt_path": None,
                }
                del net
                torch.cuda.empty_cache()

    elif suspect_type == "positive":
        model_name = suspect_cfg.get("Model", "ResNet-18")
        eval_mode  = suspect_cfg.get("State", "best")
        model_dirs = suspect_cfg.get("Model_Path", [])
        base_dir   = Path("./saved_models")

        for dir_name in model_dirs:
            model_dir = base_dir / dir_name.lstrip("/")
            print(f"\n--- Positive Suspect: {dir_name} ---")
            checkpoints = collect_checkpoints(model_dir, eval_mode)
            if not checkpoints:
                print(f"  [SKIP] No checkpoints found in {model_dir}")
                continue

            for ckpt_name, ckpt_path in checkpoints:
                print(f"  Evaluating: {ckpt_name}")
                net = load_positive_suspect(
                    model_name, num_classes, victim_dataset_obj, ckpt_path
                )
                yield {
                    "type": "positive", "model": net,
                    "seed": None, "overlap": None,
                    "dir_name": dir_name, "ckpt_name": ckpt_name, "ckpt_path": ckpt_path,
                }
                del net
                torch.cuda.empty_cache()

    else:
        raise ValueError(f"Unknown suspect type: {suspect_type!r}")


# =====================================================
# 9. CSV helpers
# =====================================================
def _ensure_csv_with_header(log_file, header):
    if not os.path.exists(log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "w", newline="") as f:
            csv.writer(f).writerow(header)


def _append_csv_row(log_file, row):
    with open(log_file, "a", newline="") as f:
        csv.writer(f).writerow(row)


def _csv_path_for(method, suspect_type, scenario_name, suffix):
    """Build the per-method, per-suspect-type CSV path."""
    folder = "Negative" if suspect_type == "negative" else "Positive"
    return f"./saved_logs/at_eval/{method}/{folder}/{scenario_name}_{suffix}.csv"


def _suspect_id_fields(suspect_type, record=None):
    """
    Return (header_fields, row_values) describing the suspect identity.
    Pass record=None to get just the header.
    """
    if suspect_type == "negative":
        header = ["Suspect_Seed", "Overlap_Rate"]
        values = [] if record is None else [record["seed"], record["overlap"]]
    else:  # positive
        header = ["Model_Dir", "Checkpoint"]
        values = [] if record is None else [record["dir_name"], record["ckpt_name"]]
    return header, values




import json
from scipy import stats


VICTIM_SEED = 42
VICTIM_OVERLAP = 1.0







"----------------------Dataset Inference Family-----------------------"

def _get_or_load_di_pipeline(
    victim_model, victim_dataset_obj, victim_ds_cfg,
    regressor_path, n_train_samples, regressor_epochs=30,
):
    """Load or train a DI regressor pipeline against the victim.

    Private images are the CLEAN (ToTensor-only) train split, so private and
    public differ only by membership; the same clean set is returned for
    verification. Before 2026-09-13 this used the augmented raw_train_set.
    """
    raw_train_set = victim_dataset_obj.raw_train_clean_set
    raw_test_set = victim_dataset_obj.raw_test_set

    train_subset_indices = np.load(
        f"./Indices/{victim_ds_cfg['name']}/group_A_subset_10000_from_25000_seed42.npy"
    )

    pipeline = DatasetInferencePipeline(victim_model, device=device)

    if os.path.exists(regressor_path):
        print(f"==> Loading existing regressor from {regressor_path}")
        pipeline.load_regressor(regressor_path)
    else:
        print("==> Training new regressor g_V on victim embeddings..")
        pipeline.train_regressor(
            private_dataset=raw_train_set,
            public_dataset=raw_test_set,
            private_indices=train_subset_indices,
            n_train_samples=n_train_samples,
            regressor_epochs=regressor_epochs,
        )
        pipeline.save_regressor(regressor_path)

    return pipeline, raw_train_set, raw_test_set, train_subset_indices


"--------------------- Unified main_* per method ------------------------"


def main_di(yaml_file_path, n_train_samples=1000, n_test_samples=1000, alpha=0.05,
            suspect_seeds=None, overlap_rates=None):
    """Dataset-Inference T-test between victim and every declared suspect."""

    ctx = _setup_victim_context(yaml_file_path, build_raw_loader=False)
    exp_yaml      = ctx["exp_yaml"]
    scenario_name = ctx["scenario_name"]
    victim_ds_cfg = ctx["victim_ds_cfg"]
    victim_ds_obj = ctx["victim_dataset_obj"]
    num_classes   = ctx["victim_num_classes"]
    victim_model  = ctx["victim_model"]

    s_type, suspect_cfg = _detect_suspect_block(exp_yaml)
    print(f"  Suspect block: {s_type}")

    # `_clean` tag: regressor trained with un-augmented private images. The
    # un-suffixed legacy file (augmented private images) is intentionally not
    # reused.
    regressor_path = (
        f"./saved_models/di_regressor/{scenario_name}_n={n_train_samples}_clean.pt"
    )
    pipeline, raw_train_set, raw_test_set, train_subset_indices = (
        _get_or_load_di_pipeline(
            victim_model, victim_ds_obj, victim_ds_cfg,
            regressor_path, n_train_samples, regressor_epochs=30,
        )
    )

    suffix   = f"DI_n={n_test_samples}_alpha={alpha}"
    log_file = _csv_path_for("DI", s_type, scenario_name, suffix)

    id_header, _ = _suspect_id_fields(s_type)
    _ensure_csv_with_header(log_file, [
        "Scenario", "Suspect_Type", *id_header,
        "Mean_Private", "Mean_Public", "Delta", "T_Stat",
        "P_Value", "Alpha", "Stolen",
    ])

    for rec in _iter_suspects(s_type, suspect_cfg, victim_ds_obj, num_classes,
                              suspect_seeds, overlap_rates):
        result = pipeline.verify_suspect(
            suspect_model=rec["model"],
            private_dataset=raw_train_set,
            public_dataset=raw_test_set,
            private_indices=train_subset_indices,
            n_test_samples=n_test_samples,
            alpha=alpha,
        )

        _, id_vals = _suspect_id_fields(s_type, rec)
        _append_csv_row(log_file, [
            scenario_name, s_type, *id_vals,
            round(result["mean_private"], 6),
            round(result["mean_public"], 6),
            round(result["delta"], 6),
            round(result["t_stat"], 6),
            f"{result['p_value']:.6e}",
            alpha,
            int(result["stolen"]),
        ])

    print(f"\n  Results -> {log_file}")



"""-------------------------- Main Execution Script---------------------------------"""

# =====================================================
# 11. Name-based dispatcher
# =====================================================
EVAL_REGISTRY = {
    "robd":    main_robd,
    "ipguard": main_ipguard,
    "di":      main_di,
    # JSD intentionally omitted for now.
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
    torch.multiprocessing.set_start_method("spawn", force=True)
    set_seed(42)

    # Folder containing all YAML experiment plans
    exp_dir = "./saved_exp_plan/at_eval_plan"
    yaml_files = sorted(glob.glob(os.path.join(exp_dir, "*.yaml")))

    if not yaml_files:
        print(f"No YAML files found in {exp_dir}")
    else:
        print(f"Found {len(yaml_files)} experiment plan(s):")
        for f in yaml_files:
            print(" -", f)

    methods_to_run = [
        "robd",
        # "ipguard",
        #"di",
    ]

    for yaml_path in yaml_files:
        print(f"\n{'=' * 60}")
        print(f"  {yaml_path}")
        print(f"{'=' * 60}")

        run_evals(methods_to_run, yaml_path)