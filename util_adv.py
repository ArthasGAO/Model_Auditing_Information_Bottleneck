import os
import glob
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
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    

# DeiT eval build config — fixed to the framework's CIFAR DeiT-Plain victim
# (deit_tiny_patch16_224 @ img_size=32, patch_size=4). build_model only receives
# (name, num_classes), so the architecture hyperparameters live in this constant.
# Update here if you evaluate a different DeiT variant.
DEIT_EVAL_CFG = {
    "model_name": "deit_tiny_patch16_224",
    "img_size": 32,
    "patch_size": 4,
    "drop_path_rate": 0.0,
}


def _build_deit_eval(num_classes, cfg=None):
    """Build a DeiT for evaluation via timm (mirrors util.build_deit_student)."""
    import timm
    cfg = cfg or DEIT_EVAL_CFG
    kwargs = dict(pretrained=False, num_classes=num_classes,
                  drop_path_rate=float(cfg.get("drop_path_rate", 0.0)),
                  img_size=int(cfg.get("img_size", 32)),
                  patch_size=int(cfg.get("patch_size", 4)))
    try:
        return timm.create_model(cfg["model_name"], **kwargs)
    except TypeError:
        kwargs.pop("img_size", None)
        return timm.create_model(cfg["model_name"], **kwargs)


def build_model(model_name, num_classes):
    if model_name == "MLP":
        return MNIST_MLP()
    elif model_name == "ResNet-18":
        return ResNet18(num_classes=num_classes)
    elif model_name == "VGG16":
        return ModifiedVGG16(num_classes=num_classes)
    elif model_name == "DeiT":          # backward-compatible addition (CNN callers unaffected)
        return _build_deit_eval(num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    

def load_stolen_model(exp_yaml, dataset_obj, num_classes, model_path):
    """Load stolen model and wrap with NormalizedModel."""
    model_name = exp_yaml.get("Model", "ResNet-18")
    net = build_model(model_name, num_classes).to(device)

    folder = "./saved_models" + model_path
    model_dir = Path(folder)
    ckpt, _ = load_best_checkpoint(model_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No victim checkpoint in {model_dir}")

    net.load_state_dict(torch.load(ckpt, map_location=device))
    net = NormalizedModel(net, dataset_obj.mean, dataset_obj.std).to(device)
    net.eval()
    print(f"  Victim loaded from: {ckpt}")
    return net


def build_at_dataset_from_yaml(dataset_obj, ds_cfg, rate):
    if ds_cfg["name"] == "PseudoLabelCIFAR-10":
        pass
    elif ds_cfg["name"] == "CIFARNet":
        pass
    else:
        save_dir = f"./Indices/{ds_cfg['name']}"
        group_size = ds_cfg['group_size']

        save_path = Path(save_dir + f"/group_B_25000_{rate}_{group_size}_seed42.npy")
        group_B = np.load(save_path).tolist()

        print(f"AT training set loading from: {save_path} !")

        at_train_set = dataset_obj.subset("raw_train", group_B)
        at_test_set = dataset_obj.raw_test_set

        print(f"  AT training set size: {len(at_train_set)}")
    
    return at_train_set, at_test_set
    

def initialize_optimizer_scheduler(exp_yaml, net):
    # Fresh optimizer 
    optimizer_class = getattr(optim, exp_yaml["Optimizer"]["name"])
    optimizer = optimizer_class(
        filter(lambda p: p.requires_grad, net.parameters()),
        **exp_yaml["Optimizer"]["params"]
    )

    epochs = exp_yaml["Optimizer"]["Epochs"]

    # Fresh scheduler 
    scheduler = None
    if exp_yaml["Scheduler"]["name"] is not None:
        scheduler_params = exp_yaml["Scheduler"]["params"].copy()
        if scheduler_params.get("T_max") == "auto":
            scheduler_params["T_max"] = epochs

        scheduler_class = getattr(lr_sched, exp_yaml["Scheduler"]["name"])
        scheduler = scheduler_class(optimizer, **scheduler_params)

    return optimizer, scheduler, epochs
    

def compute_clean_accuracy(model, raw_test_loader):
    """Evaluate clean accuracy on raw [0,1] test images."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in raw_test_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += len(labels)
    return correct / total


def compute_robust_test_accuracy(model, raw_test_loader, attack_fn, attack_kwargs):
    """Evaluate accuracy on attacked test images."""

    model.eval()
    correct, total = 0, 0

    for images, labels in raw_test_loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = attack_fn(model, images, labels, **attack_kwargs)
        with torch.no_grad():
            correct += (model(adv_images).argmax(1) == labels).sum().item()
        total += len(labels)

    return correct / total


def at_one_epoch(
    model,
    train_loader,
    optimizer,
    loss_fn,
    attack_fn,
    attack_kwargs,
    device='cuda',
    use_mixed=False,
    clean_weight=0.5,
    adv_weight=0.5,
    train_acc_on="adv",   # "adv", "clean", or "mixed"
):
    """
    One epoch of adversarial fine-tuning.

    Default behavior:
        use_mixed=False
        loss = CE(model(x_adv), y)

    Mixed clean + adversarial behavior:
        use_mixed=True
        loss = clean_weight * CE(model(x_clean), y)
             + adv_weight   * CE(model(x_adv), y)

    train_acc_on:
        "adv"   -> report training accuracy on adversarial examples
        "clean" -> report training accuracy on clean examples
        "mixed" -> report average of clean and adversarial accuracy
    """

    model.train()
    total_loss, correct, total = 0.0, 0.0, 0

    if use_mixed:
        weight_sum = clean_weight + adv_weight
        if weight_sum <= 0:
            raise ValueError("clean_weight + adv_weight must be positive.")

        clean_weight = clean_weight / weight_sum
        adv_weight = adv_weight / weight_sum

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        batch_size = labels.size(0)

        # -------------------------------------------------
        # Inner max: generate adversarial examples
        # -------------------------------------------------
        # Use eval mode during attack generation to avoid updating
        # BatchNorm statistics inside FGSM/PGD generation.
        model.eval()
        adv_images = attack_fn(model, images, labels, **attack_kwargs)
        adv_images = adv_images.detach()
        model.train()

        # -------------------------------------------------
        # Outer min: update model
        # -------------------------------------------------
        optimizer.zero_grad(set_to_none=True)

        if use_mixed:
            # Clean branch
            clean_logits = model(images)
            clean_loss = loss_fn(clean_logits, labels)

            # Adversarial branch
            adv_logits = model(adv_images)
            adv_loss = loss_fn(adv_logits, labels)

            # Mixed objective
            loss = clean_weight * clean_loss + adv_weight * adv_loss

            # Choose what training accuracy means
            with torch.no_grad():
                clean_correct = (clean_logits.argmax(1) == labels).sum().item()
                adv_correct = (adv_logits.argmax(1) == labels).sum().item()

                if train_acc_on == "clean":
                    batch_correct = clean_correct
                elif train_acc_on == "mixed":
                    batch_correct = clean_weight * clean_correct + adv_weight * adv_correct
                else:
                    # default: keep comparable with previous AT logs
                    batch_correct = adv_correct

        else:
            # Original pure adversarial fine-tuning
            adv_logits = model(adv_images)
            loss = loss_fn(adv_logits, labels)

            with torch.no_grad():
                batch_correct = (adv_logits.argmax(1) == labels).sum().item()

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        correct += batch_correct
        total += batch_size

    train_loss = total_loss / total
    train_acc = correct / total

    return train_loss, train_acc


import torch
import torch.nn as nn
from contextlib import contextmanager


@contextmanager
def freeze_bn_stats(model):
    """
    Keep BatchNorm layers in train mode so forward uses batch statistics,
    but freeze running_mean, running_var, and num_batches_tracked.

    This avoids polluting BN running statistics during PGD/FGSM generation.
    """
    saved = {}

    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            saved[m] = {
                "momentum": m.momentum,
                "num_batches_tracked": (
                    m.num_batches_tracked.clone()
                    if m.num_batches_tracked is not None
                    else None
                ),
            }
            m.momentum = 0.0

    try:
        yield
    finally:
        for m, state in saved.items():
            m.momentum = state["momentum"]

            if state["num_batches_tracked"] is not None:
                m.num_batches_tracked.copy_(state["num_batches_tracked"])


def at_one_epoch_clean_bn(
    model,
    train_loader,
    optimizer,
    loss_fn,
    attack_fn,
    attack_kwargs,
    device="cuda",
    use_mixed=False,
    clean_weight=0.5,
    adv_weight=0.5,
    train_acc_on="adv",
):
    """Run one AT epoch while calibrating BatchNorm on clean inputs.

    In mixed mode, the clean forward is the only forward allowed to update
    BatchNorm running statistics. The adversarial forward still uses batch
    statistics and contributes gradients to the mixed objective, but its
    running-stat updates are frozen. This keeps inference-time BatchNorm
    aligned with the clean distribution whose utility mixed fine-tuning is
    intended to retain.

    The edge cases have explicit semantics:
      * clean_weight == 0: pure adversarial training; update BN on adversarial
        inputs because no clean branch is active.
      * adv_weight == 0: pure clean fine-tuning; skip adversarial generation.
    """

    if train_acc_on not in {"adv", "clean", "mixed"}:
        raise ValueError("train_acc_on must be 'adv', 'clean', or 'mixed'.")

    if use_mixed:
        weight_sum = clean_weight + adv_weight
        if weight_sum <= 0:
            raise ValueError("clean_weight + adv_weight must be positive.")

        clean_weight = clean_weight / weight_sum
        adv_weight = adv_weight / weight_sum

    model.train()
    total_loss, correct, total = 0.0, 0.0, 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.size(0)

        clean_only = use_mixed and adv_weight == 0.0

        # Generate attacks in eval mode so PGD/FGSM cannot modify BN running
        # statistics. Pure clean fine-tuning does not need this computation.
        adv_images = None
        if not clean_only:
            model.eval()
            adv_images = attack_fn(
                model,
                images,
                labels,
                **attack_kwargs,
            ).detach()
            model.train()

        optimizer.zero_grad(set_to_none=True)

        if use_mixed and clean_weight > 0.0:
            # Clean data are the calibration distribution: this is the only
            # forward that updates BN running_mean/running_var in mixed mode.
            clean_logits = model(images)
            clean_loss = loss_fn(clean_logits, labels)

            if adv_weight > 0.0:
                # Keep adversarial gradients and train-mode batch statistics,
                # but do not let this branch contaminate clean BN state.
                with freeze_bn_stats(model):
                    adv_logits = model(adv_images)
                adv_loss = loss_fn(adv_logits, labels)
                loss = clean_weight * clean_loss + adv_weight * adv_loss
            else:
                adv_logits = None
                loss = clean_loss

            with torch.no_grad():
                clean_correct = (clean_logits.argmax(1) == labels).sum().item()
                adv_correct = (
                    (adv_logits.argmax(1) == labels).sum().item()
                    if adv_logits is not None
                    else clean_correct
                )

                if train_acc_on == "clean":
                    batch_correct = clean_correct
                elif train_acc_on == "mixed":
                    batch_correct = (
                        clean_weight * clean_correct
                        + adv_weight * adv_correct
                    )
                else:
                    # At clean_weight == 1 there is no adversarial branch, so
                    # report the only meaningful training accuracy available.
                    batch_correct = adv_correct

        else:
            # Pure adversarial mode: the active training distribution must
            # update BN because there is no clean calibration branch.
            model.train()
            adv_logits = model(adv_images)
            loss = loss_fn(adv_logits, labels)

            with torch.no_grad():
                batch_correct = (adv_logits.argmax(1) == labels).sum().item()

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        correct += batch_correct
        total += batch_size

    return total_loss / total, correct / total


def at_one_epoch_clean_bn_eval_aligned(
    model,
    train_loader,
    optimizer,
    loss_fn,
    attack_fn,
    attack_kwargs,
    device="cuda",
    use_mixed=False,
    clean_weight=0.5,
    adv_weight=0.5,
    train_acc_on="adv",
):
    """Run mixed AT with clean BN state and eval-aligned adversarial loss.

    In mixed mode, PGD/FGSM generation and the adversarial-loss forward both
    run in eval mode, so they use the same clean running BatchNorm statistics
    that robust evaluation will use. The clean-loss forward then runs in train
    mode and is the only forward allowed to update BN running statistics.

    This differs from :func:`at_one_epoch_clean_bn`, where the adversarial-loss
    forward remains in train mode and therefore uses adversarial batch
    statistics even though their running-stat updates are frozen.

    Edge cases preserve the intended training semantics:
      * clean_weight == 0 or use_mixed=False: pure adversarial training, with
        BN updated by the adversarial forward as in the original function.
      * adv_weight == 0: pure clean fine-tuning, with no attack generation.
    """

    if train_acc_on not in {"adv", "clean", "mixed"}:
        raise ValueError("train_acc_on must be 'adv', 'clean', or 'mixed'.")

    if use_mixed:
        weight_sum = clean_weight + adv_weight
        if weight_sum <= 0:
            raise ValueError("clean_weight + adv_weight must be positive.")

        clean_weight = clean_weight / weight_sum
        adv_weight = adv_weight / weight_sum

    model.train()
    total_loss, correct, total = 0.0, 0.0, 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.size(0)

        mixed_with_clean = use_mixed and clean_weight > 0.0
        clean_only = mixed_with_clean and adv_weight == 0.0
        adv_images = None

        if not clean_only:
            # PGD/FGSM sees the inference-time model and cannot update BN.
            model.eval()
            adv_images = attack_fn(
                model,
                images,
                labels,
                **attack_kwargs,
            ).detach()

        optimizer.zero_grad(set_to_none=True)

        if mixed_with_clean:
            if adv_weight > 0.0:
                # Keep eval mode after attack generation. This forward uses
                # exactly the same BN state as the inner maximization and as
                # robust evaluation, while retaining gradients for all
                # trainable parameters (including BN affine parameters).
                adv_logits = model(adv_images)
                adv_loss = loss_fn(adv_logits, labels)
            else:
                adv_logits = None
                adv_loss = None

            # Clean data are the sole BN calibration distribution. Run this
            # branch last so the model exits the batch in train mode and only
            # clean inputs update running_mean/running_var.
            model.train()
            clean_logits = model(images)
            clean_loss = loss_fn(clean_logits, labels)

            loss = clean_loss
            if adv_loss is not None:
                loss = clean_weight * clean_loss + adv_weight * adv_loss

            with torch.no_grad():
                clean_correct = (clean_logits.argmax(1) == labels).sum().item()
                adv_correct = (
                    (adv_logits.argmax(1) == labels).sum().item()
                    if adv_logits is not None
                    else clean_correct
                )

                if train_acc_on == "clean":
                    batch_correct = clean_correct
                elif train_acc_on == "mixed":
                    batch_correct = (
                        clean_weight * clean_correct
                        + adv_weight * adv_correct
                    )
                else:
                    batch_correct = adv_correct

        else:
            # Pure adversarial mode intentionally retains the original AT
            # behavior: the adversarial training distribution updates BN.
            model.train()
            adv_logits = model(adv_images)
            loss = loss_fn(adv_logits, labels)

            with torch.no_grad():
                batch_correct = (adv_logits.argmax(1) == labels).sum().item()

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        correct += batch_correct
        total += batch_size

    return total_loss / total, correct / total


def at_one_epoch_scratch(
    model,
    train_loader,
    optimizer,
    loss_fn,
    attack_fn,
    attack_kwargs,
    device="cuda",
    use_mixed=False,
    clean_weight=0.5,
    adv_weight=0.5,
    train_acc_on="adv",
    update_bn_on="adv",  # "adv" or "both"
):
    """
    One epoch of adversarial training or adversarial fine-tuning.

    use_mixed=False:
        loss = CE(model(x_adv), y)

    use_mixed=True:
        loss = clean_weight * CE(model(x_clean), y)
             + adv_weight   * CE(model(x_adv), y)

    train_acc_on:
        "adv"   -> report accuracy on adversarial examples
        "clean" -> report accuracy on clean examples
        "mixed" -> report weighted clean/adv accuracy

    update_bn_on:
        "adv"  -> in mixed mode, only adversarial forward updates BN running stats
        "both" -> in mixed mode, both clean and adversarial forwards update BN stats
    """

    if train_acc_on not in {"adv", "clean", "mixed"}:
        raise ValueError("train_acc_on must be 'adv', 'clean', or 'mixed'.")

    if update_bn_on not in {"adv", "both"}:
        raise ValueError("update_bn_on must be 'adv' or 'both'.")

    model.train()
    total_loss, correct, total = 0.0, 0.0, 0

    if use_mixed:
        s = clean_weight + adv_weight
        if s <= 0:
            raise ValueError("clean_weight + adv_weight must be positive.")

        clean_weight = clean_weight / s
        adv_weight = adv_weight / s

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.size(0)

        # -------------------------------------------------
        # Inner max: generate adversarial examples.
        # Keep model in train mode, but freeze BN running stats.
        # -------------------------------------------------
        model.train()
        with freeze_bn_stats(model):
            adv_images = attack_fn(
                model,
                images,
                labels,
                **attack_kwargs
            ).detach()

        # -------------------------------------------------
        # Outer min: update model.
        # -------------------------------------------------
        optimizer.zero_grad(set_to_none=True)

        if use_mixed:
            if update_bn_on == "adv":
                # Clean branch contributes gradients but does not update BN stats.
                with freeze_bn_stats(model):
                    clean_logits = model(images)

                # Adversarial branch updates BN stats.
                adv_logits = model(adv_images)

            else:
                # Both clean and adversarial forwards update BN stats.
                clean_logits = model(images)
                adv_logits = model(adv_images)

            clean_loss = loss_fn(clean_logits, labels)
            adv_loss = loss_fn(adv_logits, labels)

            loss = clean_weight * clean_loss + adv_weight * adv_loss

            with torch.no_grad():
                clean_correct = (clean_logits.argmax(1) == labels).sum().item()
                adv_correct = (adv_logits.argmax(1) == labels).sum().item()

                if train_acc_on == "clean":
                    batch_correct = clean_correct
                elif train_acc_on == "adv":
                    batch_correct = adv_correct
                else:
                    batch_correct = (
                        clean_weight * clean_correct
                        + adv_weight * adv_correct
                    )

        else:
            adv_logits = model(adv_images)
            loss = loss_fn(adv_logits, labels)

            with torch.no_grad():
                batch_correct = (adv_logits.argmax(1) == labels).sum().item()

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        correct += batch_correct
        total += batch_size

    return total_loss / total, correct / total


# =====================================================
# Logging to CSV Files
# =====================================================

def build_scenario_name(base_name, attack_name, attack_kwargs, seed):
    """Build a descriptive scenario name from attack config."""
    param_str = "_".join(f"{k}={v}" for k, v in sorted(attack_kwargs.items()))
    return f"{base_name}_{attack_name}_{param_str}_atseed={seed}"


def create_at_logger(log_dir, scenario_name, attack_name, attack_kwargs):
    """
    Create a CSV logger that adapts columns to the attack type.
    
    Returns:
        (csv_file, writer, header) — caller is responsible for closing csv_file.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"at_log_{scenario_name}.csv")

    # Fixed columns every attack shares
    fixed_columns = ['Scenario', 'AT_Epoch', 'Attack_Method',
                     'AT_Train_Loss', 'AT_Train_Acc',
                     'Clean_Acc', 'Robust_Acc']

    # Dynamic columns from attack kwargs (eps, steps, step_size, etc.)
    param_columns = [f"AT_{k}" for k in sorted(attack_kwargs.keys())]

    header = fixed_columns + param_columns

    write_header = not os.path.exists(log_file)
    csv_file = open(log_file, 'a', newline='')
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(header)

    return csv_file, writer


def log_at_epoch(writer, scenario_name, epoch, attack_name, attack_kwargs,
                 train_loss, train_acc, clean_acc, rob_acc):
    """Write one row to the AT log."""
    fixed_values = [scenario_name, epoch, attack_name,
                    round(train_loss, 4), round(train_acc, 4),
                    round(clean_acc, 4), round(rob_acc, 4)]

    param_values = [attack_kwargs[k] for k in sorted(attack_kwargs.keys())]

    writer.writerow(fixed_values + param_values)


# =====================================================
# Exact Adversarial Attacks Implementation
# =====================================================

def pgd_attack_v1(model, x, y, eps, steps, step_size):
    """
    Lightweight PGD for adversarial training.
    Operates on raw [0,1] inputs with NormalizedModel.
    Returns adversarial tensor (no logging).
    """
    x_adv = x.clone().detach()
    x_adv += torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = nn.CrossEntropyLoss()(model(x_adv), y)
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + step_size * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach()


def pgd_attack_v2(model, x, y, eps, steps, step_size=None):
    if step_size is None:
        step_size = 2.5 * eps / steps

    x_adv = x.detach().clone()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for _ in range(steps):
        x_adv.requires_grad_(True)

        logits = model(x_adv)
        loss = nn.functional.cross_entropy(logits, y)

        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0] # to avoid gradient accumulation

        with torch.no_grad():
            x_adv = x_adv + step_size * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach()


def fgsm_attack(model, images, labels, eps):
    images_adv = images.detach().clone().requires_grad_(True)

    logits = model(images_adv)
    loss = nn.functional.cross_entropy(logits, labels)

    grad = torch.autograd.grad(loss, images_adv, only_inputs=True)[0]

    with torch.no_grad():
        adv_images = images + eps * grad.sign()
        adv_images = torch.max(torch.min(adv_images, images + eps), images - eps)
        adv_images = adv_images.clamp(0.0, 1.0)

    return adv_images.detach()


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


def load_adv_examples(save_path):
    """Load pre-generated adversarial examples."""
    adv_dataset = torch.load(save_path, map_location='cpu')
    print(f"Loaded {len(adv_dataset['labels'])} adversarial examples ")
    return adv_dataset


ATTACK_REGISTRY = {
    'PGD': pgd_attack_v2,
    'FGSM': fgsm_attack,
}

def parse_attack_configs(exp_yaml):
    """
    Parse attack configuration from YAML, expanding list-valued 
    hyperparameters into multiple (attack_fn, attack_name, attack_kwargs) tuples.

    Supports:
        Attack:
          name: FGSM
          eps: 0.03            # single config

        Attack:
          name: FGSM
          eps: [0.005, 0.01]   # expands to 2 configs

        Attack:
          name: PGD
          eps: [0.01, 0.03]
          steps: 10
          step_size: [0.001, 0.003]  # must match length of eps
    """
    attack_cfg = dict(exp_yaml['Attack'])
    attack_name = attack_cfg.pop('name')

    attack_fn = ATTACK_REGISTRY.get(attack_name)
    if attack_fn is None:
        raise ValueError(
            f"Unknown attack '{attack_name}'. "
            f"Available: {list(ATTACK_REGISTRY.keys())}"
        )

    # Find which params are lists (to sweep) vs scalars (fixed)
    list_params = {k: v for k, v in attack_cfg.items() if isinstance(v, list)}
    scalar_params = {k: v for k, v in attack_cfg.items() if not isinstance(v, list)}

    if not list_params:
        # Single config, no sweep
        return [(attack_fn, attack_name, attack_cfg)]

    # Validate: all list params must have the same length
    lengths = [len(v) for v in list_params.values()]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"All list-valued attack params must have the same length. "
            f"Got: {list_params}"
        )

    num_configs = lengths[0]
    configs = []
    for i in range(num_configs):
        kwargs = dict(scalar_params)  # copy fixed params
        for k, v_list in list_params.items():
            kwargs[k] = v_list[i]
        configs.append((attack_fn, attack_name, kwargs))

    return configs


def load_victim_model(victim_cfg, dataset_obj, num_classes, model_seed):
    """Load victim model and wrap with NormalizedModel."""
    model_name = victim_cfg.get("Model", "ResNet-18")
    if model_name == "DeiT" and victim_cfg.get("Model_Config"):
        # Optional Victim.Model_Config (the training plan's Model block), e.g. the
        # distilled CIFAR-100 DeiT victim; absent -> plain DEIT_EVAL_CFG as before.
        net = _build_deit_eval(num_classes, victim_cfg["Model_Config"]).to(device)
    else:
        net = build_model(model_name, num_classes).to(device)

    folder = victim_cfg["Model_Name"] + f"_{model_seed}_{1.0}"
    subdir = "Transformer_Models" if model_name == "DeiT" else "CNN_Models"
    model_dir = Path("./saved_models/vanilla/") / subdir / folder
    ckpt, _ = load_best_checkpoint(model_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No victim checkpoint in {model_dir}")

    if model_name == "DeiT":
        state = torch.load(ckpt, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        net.load_state_dict(state)
    else:
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
    if model_name == "DeiT":
        model_dir = Path("./saved_models/vanilla/Transformer_Models/") / folder
    else:
        model_dir = Path("./saved_models/vanilla/") / folder
    ckpt, _ = load_best_checkpoint(model_dir)
    if ckpt is None:
        print(f"  [SKIP] No checkpoint found in {model_dir}")
        return None

    if model_name == "DeiT":
        state = torch.load(ckpt, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        net.load_state_dict(state)
    else:
        net.load_state_dict(torch.load(ckpt, map_location=device))
    net = NormalizedModel(net, dataset_obj.mean, dataset_obj.std).to(device)
    net.eval()
    print(f"  Negative suspect loaded from: {ckpt}")
    return net

def load_positive_suspect(model_name, num_classes, dataset_obj, ckpt_path,
                          checkpoint_format=None, model_config=None):
    """Load a positive suspect, transparently unwrapping NormalizedModel checkpoints.

    Architecture-agnostic (ResNet-18 / VGG16 / DeiT). DeiT/timm checkpoints may
    be saved as a {"model"|"state_dict": ...} container and may pickle non-tensor
    objects, so we unwrap the container and use weights_only=False (also required
    on torch>=2.6 where weights_only defaults to True).
    """
    if checkpoint_format == "kd":
        from Model.kd_eval import build_kd_student, KDLogitsOnly
        student = build_kd_student(model_name, num_classes, model_config)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # main_kd.py saves the student's state dict, including DeiT's backbone
        # prefix. Use the training class and load strictly without key remapping.
        student.load_state_dict(state, strict=True)
        return NormalizedModel(KDLogitsOnly(student), dataset_obj.mean, dataset_obj.std).to(device).eval()
    if checkpoint_format not in (None, "standard"):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_format}")
    if model_name == "DeiT" and model_config:
        # Model_Config (the training plan's Substitute.Model block) picks the timm
        # variant, e.g. the distilled DeiT of the CIFAR-100 knockoff clones.
        # Without it, DEIT_EVAL_CFG (plain DeiT) is used exactly as before.
        net = _build_deit_eval(num_classes, model_config).to(device)
    else:
        net = build_model(model_name, num_classes).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Unwrap common checkpoint containers (DeiT/timm save {"model"|"state_dict": ...}).
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Strip the NormalizedModel wrapper prefix if the checkpoint was saved wrapped.
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
        ckpt, _ = load_best_checkpoint(model_dir)
        return [("best_epoch", ckpt)] if ckpt is not None else []

    if eval_mode == "last":
        ckpt = load_last_checkpoint(model_dir)
        return [("last_epoch", ckpt)] if ckpt is not None else []

    if eval_mode == "all":
        epoch_files = sorted(
            model_dir.glob("epoch_*.pth"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        return [(f.stem, f) for f in epoch_files]

    raise ValueError(f"Unknown eval_mode '{eval_mode}'. Use 'best', 'last', or 'all'.")


def _ensure_csv_with_header(log_file, header):
    if not os.path.exists(log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "w", newline="") as f:
            csv.writer(f).writerow(header)


def _append_csv_row(log_file, row):
    with open(log_file, "a", newline="") as f:
        csv.writer(f).writerow(row)


def _csv_path_for(method, suspect_type, scenario_name, suffix, by_paths=False):
    """Build the per-method, per-suspect-type CSV path.

    `by_paths=True` puts path-based negative results in a separate CSV
    so the schema stays consistent within each file.
    """
    folder = "Negative" if suspect_type == "negative" else "Positive"
    sub    = "_paths" if by_paths else ""
    return f"./saved_logs/at_eval/{method}/{folder}/{scenario_name}_{suffix}{sub}.csv"


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


def _suspect_id_header(suspect_type, by_paths=False):
    """CSV header columns identifying a suspect."""
    if suspect_type == "negative" and not by_paths:
        return ["Suspect_Seed", "Overlap_Rate"]
    return ["Model_Dir", "Checkpoint"]


def _suspect_id_values(suspect_type, record, by_paths=False):
    """CSV row values identifying a suspect."""
    if suspect_type == "negative" and not by_paths:
        return [record["seed"], record["overlap"]]
    return [record["dir_name"], record["ckpt_name"]]
