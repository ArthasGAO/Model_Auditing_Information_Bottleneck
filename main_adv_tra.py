import os
import glob

from AdvAttack.DI import DatasetInferencePipeline
from AdvAttack.IP_Guard import IPGuardGenerator, verify_fingerprint
from util_adv import fgsm_attack, load_adv_examples, pgd_attack_v2
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import csv
from torch.utils.data import DataLoader
from util import (load_last_checkpoint, process_yaml_file, build_dataset_from_yaml,
                  create_or_load_group_A, load_best_checkpoint)
from Model.ResNet_18 import ResNet18
from Model.VGG16 import ModifiedVGG16
from Model.MLP import MNIST_MLP
import torch.nn.functional as F

from AdvAttack.advtra.adv_tra_adapter import build_args, run_extraction, run_verification_pretty

# =====================================================
# 1. Global setup
# =====================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    'PGD': pgd_attack_v2,
    'FGSM': fgsm_attack,
}

# =====================================================
# 2. NormalizedModel wrapper
# =====================================================
class NormalizedModel(nn.Module):
    """Wraps a model so it accepts raw [0,1] inputs and normalizes internally."""
    def __init__(self, base_model, mean, std):
        super().__init__()
        self.base_model = base_model
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        x_norm = (x - self.mean) / self.std
        return self.base_model(x_norm)




# =====================================================
# 4. Raw test loader (ToTensor only, no normalization)
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
        pin_memory=True
    )


# =====================================================
# 5. Model loading helpers
# =====================================================
def load_victim_model(victim_cfg, dataset_obj, num_classes, model_seed):
    """Load victim model and wrap with NormalizedModel."""
    from util_adv import load_victim_model as load_shared
    return load_shared(victim_cfg, dataset_obj, num_classes, model_seed)


def load_negative_suspect(suspect_cfg, dataset_obj, num_classes, seed, overlap):
    """
    Load a negative suspect (independently trained vanilla model).
    Naming: {Model_Name}_{seed}_{overlap}
    """
    model_name = suspect_cfg.get("Model", "ResNet-18")
    net = build_model(model_name, num_classes).to(device)

    folder = suspect_cfg["Model_Name"] + f"_{seed}_{overlap}"
    model_dir = Path('./saved_models/vanilla/') / folder
    ckpt, _ = load_best_checkpoint(model_dir)
    if ckpt is None:
        print(f"  [SKIP] No checkpoint found in {model_dir}")
        return None

    net.load_state_dict(torch.load(ckpt, map_location=device))
    net = NormalizedModel(net, dataset_obj.mean, dataset_obj.std).to(device)
    net.eval()
    print(f"  Negative suspect loaded from: {ckpt}")
    return net




def find_epoch_checkpoints(model_dir):
    """
    Find all epoch checkpoints in a directory.
    Returns list of (epoch_number, checkpoint_path) sorted by epoch.
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        print(f"  [SKIP] Directory not found: {model_dir}")
        return []

    checkpoints = []
    # Match common patterns: epoch_10.pth, checkpoint_10.pt, model_epoch10.pth, etc.
    for pattern in ['epoch_*.pth', 'epoch_*.pt', 'checkpoint_*.pth', 'checkpoint_*.pt',
                    'model_epoch*.pth', 'model_*.pth']:
        for ckpt in model_dir.glob(pattern):
            # Extract epoch number from filename
            stem = ckpt.stem
            # Try to find a number in the filename
            nums = [int(s) for s in stem.replace('_', ' ').replace('-', ' ').split() if s.isdigit()]
            if nums:
                checkpoints.append((nums[-1], ckpt))

    # Also include best checkpoint if it exists
    best, _ = load_best_checkpoint(model_dir)
    if best is not None:
        checkpoints.append(('best', best))

    # Sort by epoch number (put 'best' at the end)
    checkpoints.sort(key=lambda x: (isinstance(x[0], str), x[0]))
    return checkpoints


def load_positive_suspect(model_name, num_classes, dataset_obj, ckpt_path,
                          checkpoint_format=None, model_config=None):
    if checkpoint_format:
        from util_adv import load_positive_suspect as load_shared
        return load_shared(model_name, num_classes, dataset_obj, ckpt_path,
                           checkpoint_format=checkpoint_format, model_config=model_config)
    net = build_model(model_name, num_classes).to(device)
    state = torch.load(ckpt_path, map_location=device)

    # Check if saved from NormalizedModel wrapper
    if any(k.startswith('base_model.') for k in state.keys()):
        # Strip 'base_model.' prefix and remove mean/std buffers
        state = {k.replace('base_model.', ''): v
                 for k, v in state.items()
                 if not k in ('mean', 'std')}

    net.load_state_dict(state)
    net = NormalizedModel(net, dataset_obj.mean, dataset_obj.std).to(device)
    net.eval()
    return net



def collect_checkpoints(model_dir, eval_mode):
    """
    Collect checkpoint paths based on evaluation mode.

    Returns:
        list of (checkpoint_name, checkpoint_path) tuples
    """
    model_dir = Path(model_dir)

    if eval_mode == 'best':
        best_ckpt, _ = load_best_checkpoint(model_dir)
        # Empty list (not [('best_epoch', None)]) when best_epoch.pth is absent,
        # so the caller can [SKIP] instead of torch.load(None) crashing.
        return [('best_epoch', best_ckpt)] if best_ckpt is not None else []

    elif eval_mode == 'last':
        # Find the highest epoch number
        last_ckpt = load_last_checkpoint(model_dir)
        return [('last_epoch', last_ckpt)] if last_ckpt is not None else []

    elif eval_mode == 'all':
        checkpoints = []

        # Then all epoch checkpoints in order
        epoch_files = sorted(model_dir.glob('epoch_*.pth'),
                             key=lambda p: int(p.stem.split('_')[-1]))
        for f in epoch_files:
            checkpoints.append((f.stem, f))

        return checkpoints

    else:
        raise ValueError(f"Unknown eval_mode '{eval_mode}'. Use 'best', 'last', or 'all'.")


def sanity_check_source_accuracy(wrapped_model, raw_dataset, n: int = 200) -> float:
    """Quick accuracy check: the source should classify its raw training data
    with high accuracy (otherwise the NormalizedModel wrapping is misconfigured)."""
    correct = 0
    with torch.no_grad():
        for i in range(min(n, len(raw_dataset))):
            x, y = raw_dataset[i]
            x = x.unsqueeze(0).to(device)
            pred = wrapped_model(x).argmax(dim=1).item()
            correct += int(pred == int(y))
    acc = correct / min(n, len(raw_dataset))
    return acc



def main_adv_tra_neg(yaml_file_path):
    print(f"Device: {device}")
 
    exp_yaml = process_yaml_file(yaml_file_path)
    scenario_name = exp_yaml.get("Scenario_Name", Path(yaml_file_path).stem)
 
    # ---------- Parse configs ----------
    victim_cfg    = exp_yaml["Victim"]
    victim_ds_cfg = victim_cfg["Dataset"]
    victim_dataset_obj, victim_num_classes, _ = build_dataset_from_yaml(victim_ds_cfg)
 
    suspect_cfg         = exp_yaml["Negative Suspect"]
    suspect_num_classes = victim_num_classes     # same dataset
 
    # ---------- Load victim (fixed: seed=42) ----------
    print("==> Loading victim model..")
    victim_model = load_victim_model(victim_cfg, victim_dataset_obj,
                                     victim_num_classes, model_seed=42)
 
    # ---------- Raw [0,1] training data for base samples ----------
    print("==> Loading raw [0,1] train data..")
    raw_train_set = victim_dataset_obj.raw_train_clean_set  # ToTensor only, no augmentation
    acc = sanity_check_source_accuracy(victim_model, raw_train_set)
    print(f"[sanity] source accuracy on first 200 raw_train samples: {acc:.2%}")
    if acc < 0.7:
        raise RuntimeError(
            "Source model accuracy is unexpectedly low. Check that:\n"
            "  (a) The checkpoint actually matches the model arch;\n"
            "  (b) NormalizedModel is using the correct (mean, std);\n"
            "  (c) raw_train_clean_set yields images in [0, 1] as a Tensor."
        )
 
    train_subset_indices = np.load(
        f'./Indices/{victim_ds_cfg["name"]}/group_A_subset_10000_from_25000_seed42.npy'
    )
 
    # ---------- ADV-TRA args (paper / code defaults) ----------
    ADVTRA_ROOT = Path("./results/advtra1") / scenario_name.rsplit('_', 1)[0]
 
    # Use 2x the number of trajectories as base samples (adapter will
    # skip failed ones, reference code does the same)
    num_trajectories = 100
    base_eval_number = 2 * num_trajectories
    base_indices = train_subset_indices[:base_eval_number].tolist()
 
    args = build_args(
        dataset_name="cifar10",
        num_classes=victim_num_classes,
        data_path=str(ADVTRA_ROOT / "data"),
        model_path=str(ADVTRA_ROOT / "_model_paths"),
        fingerprint_path=str(ADVTRA_ROOT / "fingerprints"),
        num_trajectories=num_trajectories,
        length=8,
        tra_classes=10,
        max_iteration=1000,
        initial_stepsize=0.05,
        tra_lr=0.05,
        factor_lc=0.9,
        factor_re=0.95,
        threshold=0.5,
        device=device,
        # NEW: enable per-attempt extraction logging
        extraction_log_path=str(
            ADVTRA_ROOT / "fingerprints" / "cifar10" / f"trajectory_{8}" / "extraction_log.txt"
        ),
    )
 
 
    # ---------- Extract the fingerprint (run once, then cached on disk) ----------
    # ---------- Extract the fingerprint (run once, then cached on disk) ----------
    fp_trajectory_dir = (
        Path(args.fingerprint_path) / args.dataset / f"trajectory_{args.length}"
    )

    # Count how many surface trajectories are actually on disk. The vendored
    # code saves fewer than num_trajectories when some base samples fail
    # boundary probing (failures print "This basic sample cannot generate...")
    saved_count = 0
    if fp_trajectory_dir.exists():
        saved_count = sum(
            1 for d in fp_trajectory_dir.iterdir()
            if d.is_dir() and (d / "tra_log.pth").exists()
        )

    # Accept any cached fingerprint with >= 20 trajectories. Paper App. D.3.1
    # shows detection AUC stable at 1.0 from 20 to 180 trajectories, so we
    # don't need to hit the exact target to have a valid fingerprint.
    MIN_ACCEPTABLE_TRAJECTORIES = 20

    if saved_count >= MIN_ACCEPTABLE_TRAJECTORIES:
        print(f"\nFingerprint already present at {fp_trajectory_dir} "
              f"({saved_count} trajectories) -- skipping extraction")
        args.num_trajectories = saved_count      # <-- key: sync args to disk reality
    else:
        print(f"\n==> Extracting {args.num_trajectories} trajectories "
              f"from {len(base_indices)} base samples (this takes ~5 min on GPU)...")
        print("-" * 70)
        run_extraction(
            args,
            wrapped_source_model=victim_model,
            raw_dataset=raw_train_set,
            base_indices=base_indices,
        )
        print("-" * 70)
        print(f"Extraction complete. {args.num_trajectories} trajectories saved "
              f"under {fp_trajectory_dir}")
        # Note: the adapter itself has already synced args.num_trajectories
        # after extraction, so we don't need to do it again here.
 
    # ---------- Sanity verification: source vs source ----------
    print("\n" + "=" * 70)
    print(" Sanity check: source vs source (expected detection_rate = 1.0)")
    print("=" * 70)
    sanity_result = run_verification_pretty(
        args,
        wrapped_suspect_model=victim_model,
        suspect_path=str(ADVTRA_ROOT / "_model_paths" / args.dataset / "_sanity.pth"),
    )
    print(sanity_result)
    if sanity_result.detection_rate < 0.99:
        print(
            "[warning] sanity check failed -- detection rate should be 1.0 "
            "when verifying the source against its own fingerprint.\n"
            "This typically indicates a model-wrapping or device mismatch."
        )
 
    # ---------- Prepare CSV ----------
    log_dir = './saved_logs/at_eval/ADV_TRA/Negative'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        f"{scenario_name}_ADV_TRA_n={args.num_trajectories}_thr={args.threshold}.csv"
    )
 
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                'Scenario', 'Suspect_Type', 'Suspect_Seed', 'Overlap_Rate',
                'Detection_Rate', 'Mean_Mutation_Rate',
                'Threshold', 'Num_Trajectories', 'Stolen',
            ])
 
    # ---------- Evaluate all negative suspects ----------
    suspect_seeds = range(42, 52)
    # overlap_rates = [round(r * 0.1, 1) for r in range(0, 11)]
    overlap_rates = [0.0, 1.0]
 
    print("\n==> Evaluating negative suspects..")
    for seed in suspect_seeds:
        for overlap in overlap_rates:
            print(f"\n--- Negative Suspect: seed={seed}, overlap={overlap} ---")
 
            suspect_net = load_negative_suspect(
                suspect_cfg, victim_dataset_obj,
                suspect_num_classes, seed, overlap
            )
            if suspect_net is None:
                continue
 
            # Per-suspect dummy checkpoint path so runs don't collide
            suspect_dummy = str(
                ADVTRA_ROOT / "_model_paths" / args.dataset /
                f"_neg_seed{seed}_ov{overlap}.pth"
            )
            result = run_verification_pretty(
                args,
                wrapped_suspect_model=suspect_net,
                suspect_path=suspect_dummy,
            )
            print(result)
 
            with open(log_file, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    scenario_name, 'negative', seed, overlap,
                    round(result.detection_rate,      6),
                    round(result.mean_mutation_rate, 6),
                    result.threshold,
                    result.num_trajectories,
                    int(result.detection_rate > result.threshold),
                ])
 
            del suspect_net
            torch.cuda.empty_cache()
 
    print(f"\n  All results logged to: {log_file}")



def main_adv_tra_pos(yaml_file_path):
    print(f"Device: {device}")
 
    exp_yaml = process_yaml_file(yaml_file_path)
    scenario_name = exp_yaml.get("Scenario_Name", Path(yaml_file_path).stem)
 
    # ---------- Parse configs ----------
    victim_cfg    = exp_yaml["Victim"]
    victim_ds_cfg = victim_cfg["Dataset"]
    victim_dataset_obj, victim_num_classes, _ = build_dataset_from_yaml(victim_ds_cfg)
 
    suspect_cfg         = exp_yaml["Positive"]
    suspect_num_classes = victim_num_classes
    suspect_model_name  = suspect_cfg.get("Model", "ResNet-18")
 
    # ---------- Load victim (fixed: seed=42) ----------
    print("==> Loading victim model..")
    victim_model = load_victim_model(victim_cfg, victim_dataset_obj,
                                     victim_num_classes, model_seed=42)
 
    # ---------- Raw [0,1] training data for base samples ----------
    print("==> Loading raw [0,1] train data..")
    raw_train_set = victim_dataset_obj.raw_train_clean_set  # ToTensor only, no augmentation
    acc = sanity_check_source_accuracy(victim_model, raw_train_set)
    print(f"[sanity] source accuracy on first 200 raw_train samples: {acc:.2%}")
    if acc < 0.7:
        raise RuntimeError(
            "Source model accuracy is unexpectedly low. Check that:\n"
            "  (a) The checkpoint actually matches the model arch;\n"
            "  (b) NormalizedModel is using the correct (mean, std);\n"
            "  (c) raw_train_clean_set yields images in [0, 1] as a Tensor."
        )
 
    train_subset_indices = np.load(
        f'./Indices/{victim_ds_cfg["name"]}/group_A_subset_10000_from_25000_seed42.npy'
    )
 
    # ---------- ADV-TRA args (same path layout as the negative run, so the
    #            fingerprint is shared) ----------
    # The fingerprint is a property of the VICTIM, so its cache root must not
    # follow a per-experiment Scenario_Name. Resolution order:
    #   1. ADV_TRA.Fingerprint_Root in the YAML (relative to results/advtra1)
    #   2. Scenario_Name (legacy convention: Scenario_Name == victim id,
    #      e.g. 'CIFAR-10_ResNet-18_25000_42_1.0')
    # If the resolved root has no cached fingerprint but the victim-id root
    # does, warn loudly: continuing would extract a NEW fingerprint (~5 min)
    # whose detection rates are not comparable with earlier runs.
    advtra_cfg = exp_yaml.get("ADV_TRA", {}) or {}
    fp_root_name = advtra_cfg.get("Fingerprint_Root") or scenario_name
    ADVTRA_ROOT = Path("./results/advtra1") / fp_root_name
    victim_root_name = (f"{victim_cfg['Model_Name']}_"
                        f"{victim_cfg.get('Seed', 42)}_{victim_cfg.get('Overlap', 1.0)}")
    victim_fp_dir = (Path("./results/advtra1") / victim_root_name
                     / "fingerprints" / "cifar10" / "trajectory_8")
    own_fp_dir = ADVTRA_ROOT / "fingerprints" / "cifar10" / "trajectory_8"
    print(f"  ADV_TRA fingerprint root: {ADVTRA_ROOT}")
    if fp_root_name != victim_root_name and not own_fp_dir.exists() and victim_fp_dir.exists():
        print(f"  [WARNING] No fingerprint under {own_fp_dir}, but the victim's "
              f"cached fingerprint exists at {victim_fp_dir}.\n"
              f"  A fresh extraction would NOT be comparable with earlier results. "
              f"Set 'Scenario_Name: {victim_root_name}' or "
              f"'ADV_TRA: {{Fingerprint_Root: {victim_root_name}}}' to reuse it.")

    num_trajectories = 100
    base_eval_number = 2 * num_trajectories
    base_indices = train_subset_indices[:base_eval_number].tolist()
 
    args = build_args(
            dataset_name="cifar10",
            num_classes=victim_num_classes,
            data_path=str(ADVTRA_ROOT / "data"),
            model_path=str(ADVTRA_ROOT / "_model_paths"),
            fingerprint_path=str(ADVTRA_ROOT / "fingerprints"),
            num_trajectories=num_trajectories,
            length=8,
            tra_classes=10,
            max_iteration=1000,
            initial_stepsize=0.05,
            tra_lr=0.05,
            factor_lc=0.9,
            factor_re=0.95,
            threshold=0.5,
            device=device,
            # NEW: enable per-attempt extraction logging
            extraction_log_path=str(
                ADVTRA_ROOT / "fingerprints" / "cifar10" / f"trajectory_{8}" / "extraction_log.txt"
            ),
        )
 
    # ---------- Extract the fingerprint (reuses the negative run's cache) ----------
    fp_trajectory_dir = (
        Path(args.fingerprint_path) / args.dataset / f"trajectory_{args.length}"
    )
 
    saved_count = 0
    if fp_trajectory_dir.exists():
        saved_count = sum(
            1 for d in fp_trajectory_dir.iterdir()
            if d.is_dir() and (d / "tra_log.pth").exists()
        )
 
    MIN_ACCEPTABLE_TRAJECTORIES = 20
 
    if saved_count >= MIN_ACCEPTABLE_TRAJECTORIES:
        print(f"\nFingerprint already present at {fp_trajectory_dir} "
              f"({saved_count} trajectories) -- skipping extraction")
        args.num_trajectories = saved_count
    else:
        print(f"\n==> Extracting {args.num_trajectories} trajectories "
              f"from {len(base_indices)} base samples (this takes ~5 min on GPU)...")
        print("-" * 70)
        run_extraction(
            args,
            wrapped_source_model=victim_model,
            raw_dataset=raw_train_set,
            base_indices=base_indices,
        )
        print("-" * 70)
        print(f"Extraction complete. {args.num_trajectories} trajectories saved "
              f"under {fp_trajectory_dir}")
 
    # ---------- Sanity verification: source vs source ----------
    print("\n" + "=" * 70)
    print(" Sanity check: source vs source (expected detection_rate = 1.0)")
    print("=" * 70)
    sanity_result = run_verification_pretty(
        args,
        wrapped_suspect_model=victim_model,
        suspect_path=str(ADVTRA_ROOT / "_model_paths" / args.dataset / "_sanity.pth"),
    )
    print(sanity_result)
    if sanity_result.detection_rate < 0.99:
        print(
            "[warning] sanity check failed -- detection rate should be 1.0 "
            "when verifying the source against its own fingerprint."
        )
 
    # ---------- Iterate over positive suspects ----------
    model_dirs = suspect_cfg.get("Model_Path", [])
    eval_mode  = suspect_cfg.get("State", "best")   # 'best' | 'last' | 'all'
    base_dir   = Path('./saved_models')
 
    log_dir = exp_yaml.get("ADV_TRA", {}).get("Log_Dir", './saved_logs/at_eval/ADV_TRA/Positive')
    os.makedirs(log_dir, exist_ok=True)
 
    for dir_name in model_dirs:
        suspect_scenario = dir_name.split('/')[-1]
        log_file = os.path.join(
            log_dir,
            f"{suspect_scenario}_ADV_TRA_n={args.num_trajectories}_thr={args.threshold}.csv"
        )
 
        # Write header only once per file
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    'Scenario', 'Suspect_Type', 'Model_Dir', 'Checkpoint',
                    'Detection_Rate', 'Mean_Mutation_Rate',
                    'Threshold', 'Num_Trajectories', 'Stolen',
                ])
 
        model_dir = base_dir / dir_name.lstrip('/')
        print(f"\n--- Positive Suspect: {dir_name} ---")

        # Same checkpoint rules as main_DI_eval.py / main_IPGUARD_eval.py /
        # main_DEEPJUDGE_eval.py: an explicit `Checkpoint: <file>` wins over
        # `State`; `Require_Checkpoints: true` makes an empty selection fatal.
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
 
        for ckpt_name, ckpt_path in checkpoints:
            print(f"\n  Evaluating checkpoint: {ckpt_name}")
 
            try:
                suspect_net = load_positive_suspect(
                    suspect_model_name, suspect_num_classes,
                    victim_dataset_obj, ckpt_path,
                    checkpoint_format=suspect_cfg.get("Checkpoint_Format"),
                    model_config=suspect_cfg.get("Model_Config"),
                )
            except (FileNotFoundError, RuntimeError) as e:
                if suspect_cfg.get("Require_Checkpoints", False):
                    raise
                print(f"  [SKIP] Could not load {ckpt_path}: {e}")
                continue
 
            if suspect_net is None:
                continue
 
            # Per-checkpoint dummy path so concurrent runs don't collide
            suspect_dummy = str(
                ADVTRA_ROOT / "_model_paths" / args.dataset /
                f"_pos_{suspect_scenario}_{ckpt_name}.pth"
            )
            result = run_verification_pretty(
                args,
                wrapped_suspect_model=suspect_net,
                suspect_path=suspect_dummy,
            )
            print(result)
 
            with open(log_file, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    suspect_scenario, 'positive', dir_name, ckpt_name,
                    round(result.detection_rate,     6),
                    round(result.mean_mutation_rate, 6),
                    result.threshold,
                    result.num_trajectories,
                    int(result.detection_rate > result.threshold),
                ])
 
            del suspect_net
            torch.cuda.empty_cache()
 
        print(f"\n  Positive results logged to: {log_file}")





# =====================================================
# 8. Entry point
# =====================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ADV_TRA positive evaluation from YAML plans")
    parser.add_argument("--yaml", action="append")
    parser.add_argument("--plan-dir", default="./saved_exp_plan/at_eval_plan")
    cli = parser.parse_args()
    torch.multiprocessing.set_start_method("spawn", force=True)
    set_seed(42)

    exp_dir    = cli.plan_dir
    yaml_files = cli.yaml or sorted(glob.glob(os.path.join(exp_dir, "*.yaml")))

    if not yaml_files:
        print(f"No YAML files found in {exp_dir}")
    else:
        print(f"Found {len(yaml_files)} experiment plan(s):")
        for f in yaml_files:
            print(" -", f)

    for yaml_path in yaml_files:
        print(f"\n{'='*60}")
        print(f"  {yaml_path}")
        print(f"{'='*60}")

        #main_adv_tra_neg(yaml_path)
        main_adv_tra_pos(yaml_path)
