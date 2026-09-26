import os
import csv
import math
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from sklearn.metrics import precision_recall_fscore_support

# These two modules come from earlier in this project:
from Dataset.tiny_ImageNet import TinyImageNetDataset
from Model.ResNet_Factory import ResNet50
from util_imagenet import create_or_load_group_A


# =============================================================
# 1. CONFIG
# =============================================================
CONFIG = {
    # Experiment identity
    "scenario_name": "tin_resnet50_victim_50k1",
    "seed": 42,
    "round": 0,

    # Data
    "data_root": "./data/tiny-imagenet-200",
    "num_classes": 200,
    "img_size": 64,
    "group_size": 50000,              # 50k of the 100k training images
    "subset_seed": 42,                # fixed across runs so splits match for victim/suspect
    "indices_dir": "./Indices/tiny_imagenet",
    "batch_size": 128,
    "num_workers": 8,

    # Model
    "model_name": "ResNet50",
    "stem_type": "hybrid",            # recommended for 64x64
    "zero_init_residual": True,

    # Optimization
    "epochs": 250,                    # longer training compensates for less data
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 1e-3,             # higher than full-data (5e-4) for regularization
    "nesterov": True,
    "warmup_epochs": 5,
    "min_lr": 1e-6,
    "grad_clip": 1.0,

    # Loss
    "label_smoothing": 0.1,

    # Mixup / CutMix (batch-level aug)
    "mixup_enabled": False,
    "mixup_alpha": 0.2,
    "cutmix_alpha": 1.0,
    "mixup_apply_prob": 0.5,

    # Evaluation
    "use_tta": False,                  # horizontal-flip averaging

    # Output
    "log_dir": "./saved_logs/vanilla_imagenet/Performance",
    "ckpt_dir": "./saved_models/vanilla_imagenet",
}


# =============================================================
# 2. REPRODUCIBILITY
# =============================================================
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================
# 3. DATA LOADING
# =============================================================
def build_dataset(cfg):
    """
    Build the TinyImageNetDataset with the strong-augmentation recipe
    for the 50k setting: RandomCrop + Flip + RandAugment + ColorJitter +
    RandomErasing on top of Normalize.
    """
    train_transforms = [
        {"name": "RandomCrop", "params": {"size": cfg["img_size"], "padding": 8}},
        {"name": "RandomHorizontalFlip", "params": {"p": 0.5}},
        {"name": "RandAugment", "params": {"num_ops": 2, "magnitude": 12}},
        {"name": "ColorJitter", "params": {
            "brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.05
        }},
        {"name": "ToTensor"},
        {"name": "Normalize"},
        {"name": "RandomErasing", "params": {"p": 0.35}},
    ]

    # Test pipeline is just ToTensor + Normalize (no augmentation at eval time)
    test_transforms = [
        {"name": "ToTensor"},
        {"name": "Normalize"},
    ]

    dataset = TinyImageNetDataset(
        normalization="tiny_imagenet",
        loading="file",
        root_dir=cfg["data_root"],
        img_size=cfg["img_size"],
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        build_dataset=True,
        download=True,
    )
    return dataset

# =============================================================
# 5. MODEL
# =============================================================
def build_model(cfg, device):
    net = ResNet50(
        num_classes=cfg["num_classes"],
        stem_type=cfg["stem_type"],
        zero_init_residual=cfg["zero_init_residual"],
    ).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[model] {cfg['model_name']} (stem={cfg['stem_type']}): {n_params/1e6:.2f}M params")
    return net


# =============================================================
# 6. OPTIMIZER + LOSS + SCHEDULER
# =============================================================
def build_optimizer(net, cfg):
    return SGD(
        net.parameters(),
        lr=cfg["lr"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
        nesterov=cfg["nesterov"],
    )


def build_scheduler(optimizer, cfg):
    """
    Linear warmup for `warmup_epochs`, then cosine decay for the rest.
    Call .step() once per epoch regardless of phase.
    """
    warmup = LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0,
        total_iters=cfg["warmup_epochs"],
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cfg["epochs"] - cfg["warmup_epochs"],
        eta_min=cfg["min_lr"],
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[cfg["warmup_epochs"]])


def build_criterion(cfg):
    # Label smoothing is a native arg of nn.CrossEntropyLoss in modern PyTorch.
    return nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])


# =============================================================
# 7. MIXUP / CUTMIX (batch-level augmentation)
# =============================================================
def _rand_bbox(size, lam):
    """Random CutMix bounding box. Returns (x1, y1, x2, y2)."""
    H, W = size[-2], size[-1]
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def mixup_cutmix(x, y, cfg):
    """
    Apply Mixup OR CutMix to a batch with probability cfg['mixup_apply_prob'].
    Returns (x_mixed, y_a, y_b, lam). lam=1.0 means no mixing occurred.
    """
    if not cfg["mixup_enabled"] or np.random.rand() >= cfg["mixup_apply_prob"]:
        return x, y, y, 1.0

    use_cutmix = np.random.rand() < 0.5
    alpha = cfg["cutmix_alpha"] if use_cutmix else cfg["mixup_alpha"]
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0

    perm = torch.randperm(x.size(0), device=x.device)
    y_a, y_b = y, y[perm]

    if use_cutmix:
        x1, y1_, x2, y2_ = _rand_bbox(x.size(), lam)
        x = x.clone()
        x[:, :, y1_:y2_, x1:x2] = x[perm, :, y1_:y2_, x1:x2]
        # Recompute lam from actual pasted area (edge-clipping can shrink bbox)
        lam = 1.0 - ((x2 - x1) * (y2_ - y1_) / (x.size(-1) * x.size(-2)))
    else:
        x = lam * x + (1.0 - lam) * x[perm]

    return x, y_a, y_b, lam


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


# =============================================================
# 8. TRAINING LOOP (one epoch)
# =============================================================
def train_one_epoch(net, loader, optimizer, criterion, epoch, device, cfg):
    """
    Single training epoch.

    Mixup/CutMix is optional — controlled by cfg["mixup_enabled"].
      - If True:  apply Mixup or CutMix per batch (with cfg["mixup_apply_prob"]),
                  loss = lam * CE(logits, y_a) + (1-lam) * CE(logits, y_b),
                  train accuracy is computed against the dominant label.
      - If False: standard supervised training, train accuracy against the true label.
    """
    net.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    use_mixup = cfg.get("mixup_enabled", False)

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # ---- Forward pass: branch on whether Mixup/CutMix is enabled ----
        optimizer.zero_grad(set_to_none=True)

        if use_mixup:
            inputs, y_a, y_b, lam = mixup_cutmix(inputs, targets, cfg)
            logits = net(inputs)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            # Under Mixup/CutMix, the dominant label is what we score against.
            # This is only a rough sanity metric — trust test accuracy for true quality.
            label_for_acc = y_a if lam >= 0.5 else y_b
        else:
            logits = net(inputs)
            loss = criterion(logits, targets)
            label_for_acc = targets

        # ---- Backward + step ----
        loss.backward()
        if cfg.get("grad_clip") is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
        optimizer.step()

        # ---- Bookkeeping ----
        total_loss += loss.item() * inputs.size(0)
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(label_for_acc).sum().item()
        all_preds.append(predicted.detach().cpu().numpy())
        all_labels.append(label_for_acc.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return {
        "train_loss": total_loss / total,
        "train_acc": 100.0 * correct / total,
        "train_precision": precision,
        "train_recall": recall,
        "train_f1": f1,
    }


# =============================================================
# 9. EVALUATION LOOP (with optional TTA)
# =============================================================
@torch.no_grad()
def evaluate(net, loader, criterion, device, use_tta=False):
    net.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = net(inputs)
        if use_tta:
            logits_flip = net(torch.flip(inputs, dims=[-1]))
            logits = (logits + logits_flip) / 2.0

        loss = criterion(logits, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        all_preds.append(predicted.detach().cpu().numpy())
        all_labels.append(targets.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return {
        "test_loss": total_loss / total,
        "test_acc": 100.0 * correct / total,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
    }


# =============================================================
# 10. CSV LOGGING HELPERS
# =============================================================
def init_log_file(log_file):
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Scenario", "Epoch",
                "Train_Loss", "Train_Acc", "Train_Precision", "Train_Recall", "Train_F1",
                "Test_Loss", "Test_Acc", "Test_Precision", "Test_Recall", "Test_F1",
                "LR",
            ])


def log_epoch(log_file, scenario_name, epoch, train_result, test_result, lr):
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            scenario_name, epoch,
            train_result["train_loss"], train_result["train_acc"],
            train_result["train_precision"], train_result["train_recall"], train_result["train_f1"],
            test_result["test_loss"], test_result["test_acc"],
            test_result["test_precision"], test_result["test_recall"], test_result["test_f1"],
            lr,
        ])


# =============================================================
# 11. MAIN
# =============================================================
def main(cfg=None):
    if cfg is None:
        cfg = CONFIG

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    set_global_seed(cfg["seed"])
    g = torch.Generator()
    g.manual_seed(cfg["seed"])

    scenario_name = f"{cfg['scenario_name']}_s{cfg['seed']}_r{cfg['round']}"

    # --- Logging + checkpoint dirs
    os.makedirs(cfg["log_dir"], exist_ok=True)
    log_file = os.path.join(cfg["log_dir"], f"training_log_{scenario_name}.csv")
    init_log_file(log_file)

    ckpt_dir = Path(cfg["ckpt_dir"]) / scenario_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Data
    print("==> Preparing data..")
    dataset = build_dataset(cfg)
    train_set_full = dataset.train_set
    test_set = dataset.test_set

    group_A = create_or_load_group_A(
        dataset=train_set_full,
        save_dir=cfg["indices_dir"],
        group_size=cfg["group_size"],
        num_classes=cfg["num_classes"],
        seed=cfg["subset_seed"],
        force_rebuild=False,
    )
    train_subset = dataset.subset("train", group_A, clean=False)
    print(f"[data] train subset: {len(train_subset)} | test set: {len(test_set)}")

    train_loader = DataLoader(
        train_subset, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], worker_init_fn=seed_worker,
        generator=g, persistent_workers=True, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], worker_init_fn=seed_worker,
        generator=g, persistent_workers=True, pin_memory=True,
    )

    # --- Model, optimizer, criterion, scheduler
    print("==> Building model..")
    net = build_model(cfg, device)
    criterion = build_criterion(cfg)
    optimizer = build_optimizer(net, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # --- Training loop
    best_test_acc = -1.0
    for epoch in range(cfg["epochs"]):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n[epoch {epoch}/{cfg['epochs']}] lr={current_lr:.5f}")

        train_result = train_one_epoch(net, train_loader, optimizer, criterion, epoch, device, cfg)
        test_result = evaluate(net, test_loader, criterion, device, use_tta=cfg["use_tta"])

        scheduler.step()

        print(f"  train loss={train_result['train_loss']:.4f}  acc={train_result['train_acc']:.2f}%")
        print(f"  test  loss={test_result['test_loss']:.4f}  acc={test_result['test_acc']:.2f}%"
              f"{' (TTA)' if cfg['use_tta'] else ''}")

        log_epoch(log_file, scenario_name, epoch, train_result, test_result, current_lr)

        # Best-checkpoint tracking
        if test_result["test_acc"] > best_test_acc:
            best_test_acc = test_result["test_acc"]
            torch.save(net.state_dict(), ckpt_dir / "best_epoch.pth")
            print(f"  ** new best: {best_test_acc:.2f}%  -> saved best_epoch.pth")

    # Final checkpoint
    torch.save(net.state_dict(), ckpt_dir / f"epoch_{cfg['epochs']-1}.pth")
    print(f"\n[done] best test acc: {best_test_acc:.2f}%")
    print(f"[done] checkpoints in: {ckpt_dir}")


if __name__ == "__main__":
    main()