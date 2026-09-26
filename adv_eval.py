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
from AdvAttack.pgd import PGD
import torch.nn.functional as F

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
# 3. Robustness metrics
# =====================================================
def compute_robustness(model, adv_dataset, device='cuda', batch_size=256):
    """
    Rob(f, T) = accuracy of model on adversarial test set T.
    Args:
        model:  nn.Module (NormalizedModel), accepts raw [0,1] inputs
        x_adv:  torch.Tensor (N, C, H, W), raw [0,1] adversarial images
        y_true: torch.Tensor (N,), ground-truth labels
    """
    model.eval()
    x_adv = adv_dataset['adv_images']
    y_true = adv_dataset['labels']
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


# =====================================================
# KL Divergence Distance
# =====================================================
def compute_jsd(victim_model, suspect_model, data_loader, device='cuda'):
    """
    Compute Jensen-Shannon Distance (JSD) between victim and suspect model
    output distributions over a dataset.
    """
    victim_model.eval()
    suspect_model.eval()

    total_jsd = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)

            # Get softmax probability distributions from both models
            p = F.softmax(victim_model(images), dim=1)   # f^L(x)
            q = F.softmax(suspect_model(images), dim=1)   # f_hat^L(x)

            # Mixture distribution: m = (p + q) / 2
            m = (p + q) / 2

            # KL(p || m) + KL(q || m), summed over classes, averaged over batch
            # Using log for numerical stability, add small epsilon to avoid log(0)
            eps = 1e-10
            kl_p_m = (p * (torch.log(p + eps) - torch.log(m + eps))).sum(dim=1)
            kl_q_m = (q * (torch.log(q + eps) - torch.log(m + eps))).sum(dim=1)

            batch_jsd = (kl_p_m + kl_q_m) / 2  # per-sample JSD

            total_jsd += batch_jsd.sum().item()
            total_samples += len(images)

    jsd = total_jsd / total_samples
    return jsd



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
    model_name = victim_cfg.get("Model", "ResNet-18")
    net = build_model(model_name, num_classes).to(device)

    folder = victim_cfg["Model_Name"] + f"_{model_seed}_{1.0}"
    model_dir = Path('./saved_models/vanilla/') / folder
    ckpt = load_best_checkpoint(model_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No victim checkpoint in {model_dir}")

    net.load_state_dict(torch.load(ckpt, map_location=device))
    net = NormalizedModel(net, dataset_obj.mean, dataset_obj.std).to(device)
    net.eval()
    print(f"  Victim loaded from: {ckpt}")
    return net


def load_negative_suspect(suspect_cfg, dataset_obj, num_classes, seed, overlap):
    """
    Load a negative suspect (independently trained vanilla model).
    Naming: {Model_Name}_{seed}_{overlap}
    """
    model_name = suspect_cfg.get("Model", "ResNet-18")
    net = build_model(model_name, num_classes).to(device)

    folder = suspect_cfg["Model_Name"] + f"_{seed}_{overlap}"
    model_dir = Path('./saved_models/vanilla/') / folder
    ckpt = load_best_checkpoint(model_dir)
    if ckpt is None:
        print(f"  [SKIP] No checkpoint found in {model_dir}")
        return None

    net.load_state_dict(torch.load(ckpt, map_location=device))
    net = NormalizedModel(net, dataset_obj.mean, dataset_obj.std).to(device)
    net.eval()
    print(f"  Negative suspect loaded from: {ckpt}")
    return net


# =====================================================
# 6. PGD adversarial generation
# =====================================================
def generate_and_save_adv_examples(victim_model, data_loader, save_path,
                                    attack_fn, attack_kwargs, device='cuda'):
    """
    Generate adversarial examples from the victim model and save them.
    These are fixed and reused for all suspect evaluations.
    """
    victim_model.eval()
    all_adv_images = []
    all_labels = []
    all_clean_images = []

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        adv_images = attack_fn(victim_model, images, labels, **attack_kwargs)

        all_adv_images.append(adv_images.cpu())
        all_clean_images.append(images.cpu())
        all_labels.append(labels.cpu())

    adv_dataset = {
        'adv_images': torch.cat(all_adv_images, dim=0),
        'clean_images': torch.cat(all_clean_images, dim=0),
        'labels': torch.cat(all_labels, dim=0),
        'parameters': attack_kwargs
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(adv_dataset, save_path)
    print(f"Saved {len(adv_dataset['labels'])} adversarial examples to {save_path}")

    return adv_dataset


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
    best = load_best_checkpoint(model_dir)
    if best is not None:
        checkpoints.append(('best', best))

    # Sort by epoch number (put 'best' at the end)
    checkpoints.sort(key=lambda x: (isinstance(x[0], str), x[0]))
    return checkpoints

'''def load_positive_suspect(model_name, num_classes, dataset_obj, ckpt_path):
    """
    Load a positive suspect (fine-tuned/extracted model) from a specific checkpoint.
    Wraps with NormalizedModel.
    """
    net = build_model(model_name, num_classes).to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net = NormalizedModel(net, dataset_obj.mean, dataset_obj.std).to(device)
    net.eval()
    return net'''


def load_positive_suspect(model_name, num_classes, dataset_obj, ckpt_path):
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


def evaluate_positive_suspects(adv_dataset, rob_victim,
                                positive_cfg, victim_dataset_obj,
                                victim_num_classes, attack_name, attack_kwargs,
                                eval_mode='best'):
    """
    Evaluate positive (stolen) suspects against fixed adversarial examples.

    Args:
        adv_dataset:         pre-generated adversarial examples dict
        rob_victim:          victim's Rob score (precomputed)
        positive_cfg:        YAML config for positive suspects
        victim_dataset_obj:  dataset object for building models
        victim_num_classes:  number of classes
        attack_name:         name of the attack used for adv generation
        attack_kwargs:       attack hyperparameters (for logging)
        eval_mode:           'best'  — evaluate only best_epoch.pth
                             'last'  — evaluate only the final epoch checkpoint
                             'all'   — evaluate every epoch checkpoint
    """
    model_name = positive_cfg.get("Model", "ResNet-18")
    model_dirs = positive_cfg.get("Model_Path", [])
    base_dir = Path('./saved_models')

    # Prepare CSV
    attack_suffix = build_attack_suffix(attack_name, attack_kwargs)
    log_dir = './saved_logs/at_eval/RobD/Positive'
    os.makedirs(log_dir, exist_ok=True)

    for dir_name in model_dirs:
        scenario_name = dir_name.split('/')[-1]
        log_file = os.path.join(log_dir, f"robd_{scenario_name}_{attack_suffix}1.csv")

        write_header = not os.path.exists(log_file)
        csv_file = open(log_file, 'a', newline='')
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(['Scenario', 'Suspect_Type', 'Model_Dir',
                             'Checkpoint', 'Attack_Method',
                             'Rob_Victim', 'Rob_Suspect', 'RobD'])

        model_dir = base_dir / dir_name
        print(f"\n--- Positive Suspect: {dir_name} ---")

        # Collect checkpoints based on eval_mode
        checkpoints = collect_checkpoints(model_dir, eval_mode)
        if not checkpoints:
            print(f"  [SKIP] No checkpoints found in {model_dir}")
            csv_file.close()
            continue

        for ckpt_name, ckpt_path in checkpoints:
            print(f"  Evaluating: {ckpt_name}")

            suspect_net = load_positive_suspect(
                model_name, victim_num_classes,
                victim_dataset_obj, ckpt_path
            )

            rob_suspect = compute_robustness(suspect_net, adv_dataset)
            robd = compute_robd(rob_victim, rob_suspect)

            print(f"    Rob(suspect) = {rob_suspect:.4f}  ({rob_suspect * 100:.2f}%)")
            print(f"    RobD         = {robd:.4f}")

            writer.writerow([scenario_name, 'positive', dir_name,
                             ckpt_name, attack_name,
                             round(rob_victim, 4), round(rob_suspect, 4), round(robd, 4)])

            del suspect_net
            torch.cuda.empty_cache()

        csv_file.close()
        print(f"\n  Positive results logged to: {log_file}")


def collect_checkpoints(model_dir, eval_mode):
    """
    Collect checkpoint paths based on evaluation mode.

    Returns:
        list of (checkpoint_name, checkpoint_path) tuples
    """
    model_dir = Path(model_dir)

    if eval_mode == 'best':
        best_ckpt = load_best_checkpoint(model_dir)
        return [('best_epoch', best_ckpt)]

    elif eval_mode == 'last':
        # Find the highest epoch number
        last_ckpt = load_last_checkpoint(model_dir)
        return [('last_epoch', last_ckpt)]

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


def build_attack_suffix(attack_name, attack_kwargs):
    param_str = "_".join(f"{k}={v}" for k, v in sorted(attack_kwargs.items()))
    return f"{attack_name}_{param_str}"


def build_adv_save_path(exp_yaml, attack_name, attack_kwargs):
    dataset_name = exp_yaml['Dataset']['name']
    model_name = exp_yaml['Model']
    suffix = build_attack_suffix(attack_name, attack_kwargs)
    return f"./Indices/adv_examples/{dataset_name}_{model_name}_{suffix}.pt"


# =====================================================
# 7. Main RobD evaluation
# =====================================================
def main_robd_neg(yaml_file_path):
    """
    Compute RobD between victim and all negative suspects
    across different seeds and overlapping rates.
    """
    print(f"Device: {device}")

    exp_yaml = process_yaml_file(yaml_file_path)
    scenario_name = exp_yaml.get("Scenario_Name", Path(yaml_file_path).stem)

    # ---------- Parse configs ----------
    victim_cfg    = exp_yaml["Victim"]
    victim_ds_cfg = victim_cfg["Dataset"]
    victim_dataset_obj, victim_num_classes, _ = build_dataset_from_yaml(victim_ds_cfg)

    suspect_cfg    = exp_yaml["Negative Suspect"]
    suspect_num_classes = victim_num_classes  # same dataset

    # ---------- Load victim (fixed: seed=42) ----------
    print("==> Loading victim model..")
    victim_model = load_victim_model(victim_cfg, victim_dataset_obj,
                                   victim_num_classes, model_seed=42)

    # ---------- Build raw test loader ----------
    print("==> Loading raw [0,1] test data..")
    raw_loader = build_raw_test_loader(victim_dataset_obj, batch_size=1000)

    # ---------- Generate adversarial set once on victim ----------
    attack_cfg = dict(exp_yaml['Attack']) # copy to avoid mutating the original
    attack_name = attack_cfg.pop('name')  # extract name, rest are kwargs

    attack_fn = ATTACK_REGISTRY.get(attack_name)
    if attack_fn is None:
        raise ValueError(
            f"Unknown attack '{attack_name}'. "
            f"Available: {list(ATTACK_REGISTRY.keys())}"
        )
    attack_kwargs = attack_cfg  # everything remaining is hyperparameters

    print(f"Attack Name:{attack_name}, {attack_kwargs}")

    adv_save_path = build_adv_save_path(victim_cfg, attack_name, attack_kwargs)

    if Path(adv_save_path).exists():
        print(f"Loading adv examples from {adv_save_path}")
        adv_dataset = load_adv_examples(adv_save_path)
    else:
        print("Generating new adv examples:")
        adv_dataset = generate_and_save_adv_examples(
            victim_model, raw_loader, adv_save_path, attack_fn, attack_kwargs
        )

    # ---------- Victim robustness (computed once) ----------
    rob_victim = compute_robustness(victim_model, adv_dataset)
    print(f"  Rob(victim) = {rob_victim:.4f}  ({rob_victim * 100:.2f}%)")

    # ---------- Prepare CSV logging ----------
    log_dir  = './saved_logs/at_eval/RobD/Negative'
    os.makedirs(log_dir, exist_ok=True)
    attack_suffix = build_attack_suffix(attack_name, attack_kwargs)
    log_file = os.path.join(log_dir, f"{scenario_name}_RobD_{attack_suffix}.csv")

    write_header = not os.path.exists(log_file)
    csv_file = open(log_file, 'a', newline='')
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(['Scenario', 'Suspect_Type', 'Suspect_Seed',
                         'Overlap_Rate', 'Rob_Victim', 'Rob_Suspect', 'RobD'])

    # ---------- Evaluate all negative suspects ----------
    suspect_seeds = range(42, 52)          # 42 to 51 inclusive
    overlap_rates = [round(r * 0.1, 1) for r in range(0, 11)]  # 0.0 to 1.0

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

            rob_suspect = compute_robustness(suspect_net, adv_dataset)
            robd = compute_robd(rob_victim, rob_suspect)

            print(f"  Rob(suspect) = {rob_suspect:.4f}  ({rob_suspect * 100:.2f}%)")
            print(f"  RobD         = {robd:.4f}")

            writer.writerow([scenario_name, 'negative', seed, 
                             overlap, rob_victim, rob_suspect, robd])

            # Free GPU memory for next model
            del suspect_net
            torch.cuda.empty_cache()

    csv_file.close()
    print(f"\n  All results logged to: {log_file}")


def main_robd_pos(yaml_file_path):
    """
    Compute RobD between victim and all suspects (positive).
    """
    print(f"Device: {device}")

    exp_yaml = process_yaml_file(yaml_file_path)
    #scenario_name = exp_yaml.get("Scenario_Name", Path(yaml_file_path).stem)

    # ---------- Parse configs ----------
    victim_cfg    = exp_yaml["Victim"]
    victim_ds_cfg = victim_cfg["Dataset"]
    victim_dataset_obj, victim_num_classes, _ = build_dataset_from_yaml(victim_ds_cfg)

    # ---------- Load victim (fixed: seed=42) ----------
    print("==> Loading victim model..")
    victim_model = load_victim_model(victim_cfg, victim_dataset_obj,
                                   victim_num_classes, model_seed=42)

    # ---------- Build raw test loader ----------
    print("==> Loading raw [0,1] test data..")
    raw_loader = build_raw_test_loader(victim_dataset_obj, batch_size=1000)

    # ---------- Generate adversarial set once on victim ----------
    attack_cfg = dict(exp_yaml['Attack']) # copy to avoid mutating the original
    attack_name = attack_cfg.pop('name')  # extract name, rest are kwargs

    attack_fn = ATTACK_REGISTRY.get(attack_name)
    if attack_fn is None:
        raise ValueError(
            f"Unknown attack '{attack_name}'. "
            f"Available: {list(ATTACK_REGISTRY.keys())}"
        )
    attack_kwargs = attack_cfg  # everything remaining is hyperparameters

    print(f"Attack Name:{attack_name}, {attack_kwargs}")

    adv_save_path = build_adv_save_path(victim_cfg, attack_name, attack_kwargs)

    if Path(adv_save_path).exists():
        print(f"Loading adv examples from {adv_save_path}")
        adv_dataset = load_adv_examples(adv_save_path)
    else:
        print("Generating new adv examples:")
        adv_dataset = generate_and_save_adv_examples(
            victim_model, raw_loader, adv_save_path, attack_fn, attack_kwargs
        )

    # ---------- Victim robustness (computed once) ----------
    rob_victim = compute_robustness(victim_model, adv_dataset)
    print(f"  Rob(victim) = {rob_victim:.4f}  ({rob_victim * 100:.2f}%)")

    # ---------- Evaluate positive suspects ----------
    if exp_yaml.get("Positive"):
        print("\n" + "=" * 50)
        print("  POSITIVE SUSPECT EVALUATION")
        print("=" * 50)
        evaluate_positive_suspects(adv_dataset, rob_victim,
                                    exp_yaml["Positive"], victim_dataset_obj,victim_num_classes,
                                    attack_name, attack_kwargs, eval_mode="all")


def main_jsd_neg(yaml_file_path):
    """
    Compute JSD between victim and all negative suspects
    across different seeds and overlapping rates.
    """
    print(f"Device: {device}")

    exp_yaml = process_yaml_file(yaml_file_path)
    scenario_name = exp_yaml.get("Scenario_Name", Path(yaml_file_path).stem)

    # ---------- Parse configs ----------
    victim_cfg    = exp_yaml["Victim"]
    victim_ds_cfg = victim_cfg["Dataset"]
    victim_dataset_obj, victim_num_classes, _ = build_dataset_from_yaml(victim_ds_cfg)

    suspect_cfg    = exp_yaml["Negative Suspect"]
    suspect_num_classes = victim_num_classes  # same dataset

    # ---------- Load victim (fixed: seed=42) ----------
    print("==> Loading victim model..")
    victim_model = load_victim_model(victim_cfg, victim_dataset_obj,
                                   victim_num_classes, model_seed=42)

    # ---------- Build raw test loader ----------
    print("==> Loading raw [0,1] test data..")
    raw_loader = build_raw_test_loader(victim_dataset_obj, batch_size=1000)

    # ---------- Generate adversarial set once on victim ----------
    attack_cfg = dict(exp_yaml['Attack']) # copy to avoid mutating the original
    attack_name = attack_cfg.pop('name')  # extract name, rest are kwargs

    attack_fn = ATTACK_REGISTRY.get(attack_name)
    if attack_fn is None:
        raise ValueError(
            f"Unknown attack '{attack_name}'. "
            f"Available: {list(ATTACK_REGISTRY.keys())}"
        )
    attack_kwargs = attack_cfg  # everything remaining is hyperparameters

    print(f"Attack Name:{attack_name}, {attack_kwargs}")

    adv_save_path = build_adv_save_path(victim_cfg, attack_name, attack_kwargs)

    if Path(adv_save_path).exists():
        print(f"Loading adv examples from {adv_save_path}")
        adv_dataset = load_adv_examples(adv_save_path)
    else:
        print("Generating new adv examples:")
        adv_dataset = generate_and_save_adv_examples(
            victim_model, raw_loader, adv_save_path, attack_fn, attack_kwargs
        )

    # ---------- Victim robustness (computed once) ----------
    rob_victim = compute_robustness(victim_model, adv_dataset)
    print(f"  Rob(victim) = {rob_victim:.4f}  ({rob_victim * 100:.2f}%)")

    # ---------- Prepare CSV logging ----------
    log_dir  = './saved_logs/at_eval/RobD/Negative'
    os.makedirs(log_dir, exist_ok=True)
    attack_suffix = build_attack_suffix(attack_name, attack_kwargs)
    log_file = os.path.join(log_dir, f"{scenario_name}_RobD_{attack_suffix}.csv")

    write_header = not os.path.exists(log_file)
    csv_file = open(log_file, 'a', newline='')
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(['Scenario', 'Suspect_Type', 'Suspect_Seed',
                         'Overlap_Rate', 'Rob_Victim', 'Rob_Suspect', 'RobD'])

    # ---------- Evaluate all negative suspects ----------
    suspect_seeds = range(42, 52)          # 42 to 51 inclusive
    overlap_rates = [round(r * 0.1, 1) for r in range(0, 11)]  # 0.0 to 1.0

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

            rob_suspect = compute_robustness(suspect_net, adv_dataset)
            robd = compute_robd(rob_victim, rob_suspect)

            print(f"  Rob(suspect) = {rob_suspect:.4f}  ({rob_suspect * 100:.2f}%)")
            print(f"  RobD         = {robd:.4f}")

            writer.writerow([scenario_name, 'negative', seed, 
                             overlap, rob_victim, rob_suspect, robd])

            # Free GPU memory for next model
            del suspect_net
            torch.cuda.empty_cache()

    csv_file.close()
    print(f"\n  All results logged to: {log_file}")


def main_ipguard_neg(yaml_file_path):
    """
    Compute IP Guard generated samples matching rate between victim and all negative suspects
    across different seeds and overlapping rates.
    """
    print(f"Device: {device}")

    exp_yaml = process_yaml_file(yaml_file_path)
    scenario_name = exp_yaml.get("Scenario_Name", Path(yaml_file_path).stem)

    # ---------- Parse configs ----------
    victim_cfg    = exp_yaml["Victim"]
    victim_ds_cfg = victim_cfg["Dataset"]
    victim_dataset_obj, victim_num_classes, _ = build_dataset_from_yaml(victim_ds_cfg)

    suspect_cfg    = exp_yaml["Negative Suspect"]
    suspect_num_classes = victim_num_classes  # same dataset

    # ---------- Load victim (fixed: seed=42) ----------
    print("==> Loading victim model..")
    victim_model = load_victim_model(victim_cfg, victim_dataset_obj,
                                   victim_num_classes, model_seed=42)

    # ---------- Build raw test loader ----------
    print("==> Loading raw [0,1] train data..")
    raw_train_set = victim_dataset_obj.raw_train_set

    k_param = 5
    size = 100
    ds_name = victim_ds_cfg["name"]
    model_name = victim_cfg["Model"]

    # ---------- All four (init, target) configurations ----------
    configs = [
        ("T", "R"),  # Training init + Random target
        ("T", "L"),  # Training init + Least-likely target
        ("R", "R"),  # Random init + Random target
        ("R", "L"),  # Random init + Least-likely target
    ]

    fingerprints = {}
    for init_strat, target_strat in configs:
        tag = f"{init_strat}{target_strat}"
        fp_path = f'./Indices/{ds_name}/{model_name}_{tag}_k={k_param}_size={size}.pt'

        if os.path.exists(fp_path):
            print(f"==> Loading existing {tag} fingerprints..")
            fingerprints[tag] = IPGuardGenerator.load_fingerprints(fp_path)
        else:
            print(f"==> Generating {tag} fingerprints..")
            gen = IPGuardGenerator(
                model=victim_model, num_classes=victim_num_classes,
                k=k_param, max_iters=1000, lr=0.01,
                init_strategy=init_strat, target_strategy=target_strat,
                device=device,
            )
            # Training init needs train_dataset; Random init does not
            train_data = raw_train_set if init_strat == "T" else None
            fingerprints[tag] = gen.generate(
                n_points=size, train_dataset=train_data,
                input_shape=(3, victim_ds_cfg["img_size"], victim_ds_cfg["img_size"]),
            )
            gen.save_fingerprints(fingerprints[tag], fp_path)

    # ---------- Prepare CSV ----------
    log_dir  = './saved_logs/at_eval/IPGuard/Negative'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{scenario_name}_IPGuard_k={k_param}_size={size}_MatchingRate.csv")

    write_header = not os.path.exists(log_file)

    # ---------- Evaluate all negative suspects ----------
    suspect_seeds = range(42, 52)
    overlap_rates = [round(r * 0.1, 1) for r in range(0, 11)]

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

            # Evaluate all four configurations
            with open(log_file, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                if write_header:
                    writer.writerow(['Scenario', 'Suspect_Type', 'Suspect_Seed',
                                     'Overlap_Rate', 'Initialization_Type',
                                     'Matching Rate'])
                    write_header = False  # only write header once

                for tag, fp_data in fingerprints.items():
                    result = verify_fingerprint(victim_model, suspect_net,
                                               fp_data, device=device)
                    print(f"  Matching rate {tag}: {result['matching_rate']:.4f}")

                    writer.writerow([scenario_name, 'negative', seed,
                                     overlap, tag, result['matching_rate']])

            del suspect_net
            torch.cuda.empty_cache()

    print(f"\n  All results logged to: {log_file}")


def main_ipguard_pos(yaml_file_path):
    """
    Compute IP Guard generated samples matching rate between victim and all positive suspects
    across different settings.
    """
    print(f"Device: {device}")

    exp_yaml = process_yaml_file(yaml_file_path)
    scenario_name = exp_yaml.get("Scenario_Name", Path(yaml_file_path).stem)

    # ---------- Parse configs ----------
    victim_cfg    = exp_yaml["Victim"]
    victim_ds_cfg = victim_cfg["Dataset"]
    victim_dataset_obj, victim_num_classes, _ = build_dataset_from_yaml(victim_ds_cfg)

    suspect_cfg    = exp_yaml["Positive"]
    suspect_num_classes = victim_num_classes  # same dataset

    # ---------- Load victim (fixed: seed=42) ----------
    print("==> Loading victim model..")
    victim_model = load_victim_model(victim_cfg, victim_dataset_obj,
                                   victim_num_classes, model_seed=42)

    # ---------- Build raw test loader ----------
    print("==> Loading raw [0,1] train data..")
    raw_train_set = victim_dataset_obj.raw_train_set

    k_param = 5
    size = 100
    ds_name = victim_ds_cfg["name"]
    model_name = victim_cfg["Model"]

    # ---------- All four (init, target) configurations ----------
    configs = [
        ("T", "R"),  # Training init + Random target
        ("T", "L"),  # Training init + Least-likely target
        ("R", "R"),  # Random init + Random target
        ("R", "L"),  # Random init + Least-likely target
    ]

    fingerprints = {}
    for init_strat, target_strat in configs:
        tag = f"{init_strat}{target_strat}"
        fp_path = f'./Indices/{ds_name}/{model_name}_{tag}_k={k_param}_size={size}.pt'

        if os.path.exists(fp_path):
            print(f"==> Loading existing {tag} fingerprints..")
            fingerprints[tag] = IPGuardGenerator.load_fingerprints(fp_path)
        else:
            print(f"==> Generating {tag} fingerprints..")
            gen = IPGuardGenerator(
                model=victim_model, num_classes=victim_num_classes,
                k=k_param, max_iters=1000, lr=0.01,
                init_strategy=init_strat, target_strategy=target_strat,
                device=device,
            )
            # Training init needs train_dataset; Random init does not
            train_data = raw_train_set if init_strat == "T" else None
            fingerprints[tag] = gen.generate(
                n_points=100, train_dataset=train_data,
                input_shape=(3, victim_ds_cfg["img_size"], victim_ds_cfg["img_size"]),
            )
            gen.save_fingerprints(fingerprints[tag], fp_path)

    # ---------- Prepare CSV ----------
    model_dirs = suspect_cfg.get("Model_Path", [])
    eval_mode = suspect_cfg.get("State", "")
    base_dir = Path('./saved_models')

    log_dir  = './saved_logs/at_eval/IPGuard/Positive'
    os.makedirs(log_dir, exist_ok=True)

    for dir_name in model_dirs:
        scenario_name = dir_name.split('/')[-1]
        log_file = os.path.join(log_dir, f"{scenario_name}_IPGuard_k={k_param}_size={size}_MatchingRate.csv")

        write_header = not os.path.exists(log_file)
        csv_file = open(log_file, 'a', newline='')
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(['Scenario', 'Suspect_Type', 'Model_Dir',
                             'Checkpoint', 'Initialization Type', 'k_param', 'Size', 'Matching Rate'])
            
        model_dir = base_dir / dir_name
        print(f"\n--- Positive Suspect: {dir_name} ---")
        
        # Collect checkpoints based on eval_mode
        checkpoints = collect_checkpoints(model_dir, eval_mode)
        if not checkpoints:
            print(f"  [SKIP] No checkpoints found in {model_dir}")
            csv_file.close()
            continue
        print(checkpoints)
        for ckpt_name, ckpt_path in checkpoints:
            print(f"  Evaluating: {ckpt_name}")

            suspect_net = load_positive_suspect(
                model_name, suspect_num_classes,
                victim_dataset_obj, ckpt_path
            )

            for tag, fp_data in fingerprints.items():
                result = verify_fingerprint(victim_model, suspect_net,
                                            fp_data, device=device)
                print(f"  Matching rate {tag}: {result['matching_rate']:.4f}")

                writer.writerow([scenario_name, 'positive', dir_name,
                                    ckpt_name, tag, k_param, size, result['matching_rate']])

            del suspect_net
            torch.cuda.empty_cache()

        csv_file.close()
        print(f"\n  Positive results logged to: {log_file}")


def main_di_neg(yaml_file_path):
    print(f"Device: {device}")

    exp_yaml = process_yaml_file(yaml_file_path)
    scenario_name = exp_yaml.get("Scenario_Name", Path(yaml_file_path).stem)

    # ---------- Parse configs ----------
    victim_cfg    = exp_yaml["Victim"]
    victim_ds_cfg = victim_cfg["Dataset"]
    victim_dataset_obj, victim_num_classes, _ = build_dataset_from_yaml(victim_ds_cfg)

    suspect_cfg    = exp_yaml["Negative Suspect"]
    suspect_num_classes = victim_num_classes  # same dataset

    # ---------- Load victim (fixed: seed=42) ----------
    print("==> Loading victim model..")
    victim_model = load_victim_model(victim_cfg, victim_dataset_obj,
                                    victim_num_classes, model_seed=42)

    # ---------- Build raw test loader ----------
    print("==> Loading raw [0,1] train data..")
    # Clean (ToTensor-only) private images since 2026-09-13; both sides un-augmented.
    raw_train_set = victim_dataset_obj.raw_train_clean_set  # contains S_V (victim's training data)
    raw_test_set  = victim_dataset_obj.raw_test_set          # public/held-out data

    train_subset_indices = np.load(f'./Indices/{victim_ds_cfg["name"]}/group_A_subset_10000_from_25000_seed42.npy') # Load the evaled training subset

    # Hyperparameters
    n_train_samples = 1000    # points per set for regressor training (500 priv + 500 pub)
    n_test_samples  = 1000     # points per set for suspect verification
    alpha = 0.05

    # ---------- Initialize or load pre-trained regressor ----------
    pipeline = DatasetInferencePipeline(victim_model, device=device)
    regressor_path = f'./saved_models/di_regressor/{scenario_name.rsplit('_', 1)[0]}_n={n_train_samples}_clean.pt'

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
            regressor_epochs=30
        )
        pipeline.save_regressor(regressor_path)

    # ---------- Prepare CSV ----------
    log_dir = './saved_logs/at_eval/DI/Negative'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        f"{scenario_name}_DI_n={n_test_samples}_alpha={alpha}.csv"
    )

    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                'Scenario', 'Suspect_Type', 'Suspect_Seed', 'Overlap_Rate',
                'Mean_Private', 'Mean_Public', 'Delta', 'T_Stat',
                'P_Value', 'Alpha', 'Stolen'
            ])

    # ---------- Evaluate all negative suspects ----------
    suspect_seeds = range(42, 52)
    #overlap_rates = [round(r * 0.1, 1) for r in range(0, 11)]
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

            result = pipeline.verify_suspect(
                suspect_model=suspect_net,
                private_dataset=raw_train_set,
                public_dataset=raw_test_set,
                private_indices=train_subset_indices,
                n_test_samples=n_test_samples,
                alpha=alpha,
            )

            with open(log_file, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    scenario_name, 'negative', seed, overlap,
                    round(result['mean_private'], 6),
                    round(result['mean_public'],  6),
                    round(result['delta'],        6),
                    round(result['t_stat'],       6),
                    result['p_value'],
                    alpha,
                    int(result['stolen']),
                ])
                
            del suspect_net
            torch.cuda.empty_cache()

    print(f"\n  All results logged to: {log_file}")


def main_di_pos(yaml_file_path):
    print(f"Device: {device}")

    exp_yaml = process_yaml_file(yaml_file_path)
    scenario_name = exp_yaml.get("Scenario_Name", Path(yaml_file_path).stem)

    # ---------- Parse configs ----------
    victim_cfg    = exp_yaml["Victim"]
    victim_ds_cfg = victim_cfg["Dataset"]
    victim_dataset_obj, victim_num_classes, _ = build_dataset_from_yaml(victim_ds_cfg)

    suspect_cfg    = exp_yaml["Positive"]
    suspect_num_classes = victim_num_classes
    suspect_model_name = suspect_cfg.get("Model", "ResNet-18")

    # ---------- Load victim ----------
    print("==> Loading victim model..")
    victim_model = load_victim_model(victim_cfg, victim_dataset_obj,
                                     victim_num_classes, model_seed=42)

    # ---------- Raw datasets ----------
    print("==> Loading raw [0,1] data..")
    # Clean (ToTensor-only) private images since 2026-09-13; both sides un-augmented.
    raw_train_set = victim_dataset_obj.raw_train_clean_set   # contains S_V
    raw_test_set  = victim_dataset_obj.raw_test_set          # public/held-out

    train_subset_indices = np.load(
        f'./Indices/{victim_ds_cfg["name"]}/group_A_subset_10000_from_25000_seed42.npy'
    )

    # Hyperparameters — should match what you used in main_di_neg
    n_train_samples = 1000
    n_test_samples  = 1000
    alpha = 0.05

    # ---------- Initialize or load pre-trained regressor ----------
    pipeline = DatasetInferencePipeline(victim_model, device=device)
    regressor_path = f'./saved_models/di_regressor/{scenario_name}_n={n_train_samples}_clean.pt'

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
        )
        pipeline.save_regressor(regressor_path)

    # ---------- Iterate over positive suspects ----------
    model_dirs = suspect_cfg.get("Model_Path", [])
    eval_mode  = suspect_cfg.get("State", "best")   # 'best' | 'last' | 'all'
    base_dir   = Path('./saved_models')

    log_dir = './saved_logs/at_eval/DI/Positive'
    os.makedirs(log_dir, exist_ok=True)

    for dir_name in model_dirs:
        suspect_scenario = dir_name.split('/')[-1]
        log_file = os.path.join(
            log_dir,
            f"{suspect_scenario}_DI_n={n_test_samples}_alpha={alpha}.csv"
        )

        # Write header only once — open with 'w' if new, else append
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                csv.writer(f).writerow([
                    'Scenario', 'Suspect_Type', 'Model_Dir', 'Checkpoint',
                    'Mean_Private', 'Mean_Public', 'Delta', 'T_Stat',
                    'P_Value', 'Alpha', 'Stolen'
                ])

        model_dir = base_dir / dir_name.lstrip('/')
        print(f"\n--- Positive Suspect: {dir_name} ---")

        checkpoints = collect_checkpoints(model_dir, eval_mode)
        if not checkpoints:
            print(f"  [SKIP] No checkpoints found in {model_dir}")
            continue

        for ckpt_name, ckpt_path in checkpoints:
            print(f"\n  Evaluating checkpoint: {ckpt_name}")

            suspect_net = load_positive_suspect(
                suspect_model_name, suspect_num_classes,
                victim_dataset_obj, ckpt_path
            )

            result = pipeline.verify_suspect(
                suspect_model=suspect_net,
                private_dataset=raw_train_set,
                public_dataset=raw_test_set,
                private_indices=train_subset_indices,
                n_test_samples=n_test_samples,
                alpha=alpha,
            )

            with open(log_file, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    suspect_scenario, 'positive', dir_name, ckpt_name,
                    round(result['mean_private'], 6),
                    round(result['mean_public'],  6),
                    round(result['delta'],        6),
                    round(result['t_stat'],       6),
                    f"{result['p_value']:.6e}",   # force scientific notation
                    alpha,
                    int(result['stolen']),
                ])

            del suspect_net
            torch.cuda.empty_cache()

        print(f"\n  Positive results logged to: {log_file}")




# =====================================================
# 8. Entry point
# =====================================================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    set_seed(42)

    exp_dir    = "./saved_exp_plan/at_eval_plan"
    yaml_files = sorted(glob.glob(os.path.join(exp_dir, "*.yaml")))

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
        #main_robd_neg(yaml_path)
        #main_ipguard_neg(yaml_path)
        #main_ipguard_pos(yaml_path)
        main_di_neg(yaml_path)
        #main_di_pos(yaml_path)