import os
import csv
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import precision_recall_fscore_support

from torchvision.models import resnet50, ResNet50_Weights

# Your project modules
from Dataset.tiny_ImageNet import TinyImageNetDataset
from util_imagenet import create_or_load_group_A, create_or_load_group_B


# =============================================================
# 1. CONFIG
# =============================================================
CONFIG = {
    # Experiment identity
    "scenario_name": "tinyIN_pretrained_resnet50_ft",
    "seed": 42,
    "round": 0,

    # Group mode:
    #   "A" = victim model trained on group A
    #   "B" = negative models trained on group B with controlled overlap
    "group_mode": "B",

    # Used only when group_mode == "B"
    "overlap_rates": [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],

    # Data
    "data_root": "./data/tiny-imagenet-200",
    "num_classes": 200,
    "img_size": 224,
    "group_size": 50000,
    "subset_seed": 42,
    "indices_dir": "./Indices/tiny_imagenet",
    "batch_size": 128,
    "num_workers": 8,

    # Pretrained model
    "model_name": "torchvision_resnet50",
    "weights": "IMAGENET1K_V2",

    # Fine-tuning schedule
    "fc_warmup_epochs": 5,
    "full_ft_epochs": 45,
    "total_epochs": 50,

    # Optimizer
    "lr_fc_stage1": 1e-2,
    "lr_backbone_stage2": 1e-3,
    "lr_fc_stage2": 1e-2,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
    "min_lr": 1e-6,
    "grad_clip": 1.0,

    # Loss
    "label_smoothing": 0.0,

    # Evaluation
    "use_tta": False,

    # Output
    "log_dir": "./saved_logs/pretrained_tiny_imagenet/Performance",
    "ckpt_dir": "./saved_models/pretrained_tiny_imagenet",
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
# 3. DATASET
# =============================================================
def build_dataset(cfg):
    """
    Tiny-ImageNet fine-tuning pipeline for ImageNet-pretrained ResNet50.

    Important:
    The pretrained torchvision ResNet50 expects ImageNet-style 224x224 inputs
    and ImageNet normalization.

    If your TinyImageNetDataset class does not support normalization="imagenet",
    add it inside the dataset class as:

        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
    """

    train_transforms = [
        {"name": "Resize", "params": {"size": 256}},
        {"name": "RandomCrop", "params": {"size": 224}},
        {"name": "RandomHorizontalFlip", "params": {"p": 0.5}},
        {"name": "ToTensor"},
        {"name": "Normalize"},
    ]

    test_transforms = [
        {"name": "Resize", "params": {"size": 256}},
        {"name": "CenterCrop", "params": {"size": 224}},
        {"name": "ToTensor"},
        {"name": "Normalize"},
    ]

    dataset = TinyImageNetDataset(
        normalization="imagenet",
        loading="file",
        root_dir=cfg["data_root"],
        img_size=cfg["img_size"],
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        build_dataset=True,
        download=True,
    )
    return dataset


def build_train_indices(cfg, train_set_full, overlap_rate=None):
    """
    Build either Group A or Group B.

    Group A:
        victim fine-tuning subset.

    Group B:
        negative fine-tuning subset with controlled overlap relative to Group A.
    """

    group_A = create_or_load_group_A(
        dataset=train_set_full,
        save_dir=cfg["indices_dir"],
        group_size=cfg["group_size"],
        num_classes=cfg["num_classes"],
        seed=cfg["subset_seed"],
        force_rebuild=False,
    )

    if cfg["group_mode"].upper() == "A":
        print("[indices] Using Group A: victim subset")
        return group_A

    elif cfg["group_mode"].upper() == "B":
        if overlap_rate is None:
            raise ValueError("overlap_rate must be provided when group_mode == 'B'.")

        print(f"[indices] Using Group B: overlap_rate={overlap_rate}")

        group_B = create_or_load_group_B(
            save_dir=cfg["indices_dir"],
            overlap_rate=overlap_rate,
            group_A_indices=group_A,
            dataset=train_set_full,
            group_size=cfg["group_size"],
            num_classes=cfg["num_classes"],
        )
        return group_B

    else:
        raise ValueError(f"Unknown group_mode: {cfg['group_mode']}. Use 'A' or 'B'.")


# =============================================================
# 4. MODEL
# =============================================================
def build_model(cfg, device):
    """
    Build ImageNet-pretrained torchvision ResNet50 and replace the classifier
    with a Tiny-ImageNet 200-class head.
    """

    if cfg["weights"] == "IMAGENET1K_V2":
        weights = ResNet50_Weights.IMAGENET1K_V2
    elif cfg["weights"] == "IMAGENET1K_V1":
        weights = ResNet50_Weights.IMAGENET1K_V1
    elif cfg["weights"] == "DEFAULT":
        weights = ResNet50_Weights.DEFAULT
    else:
        raise ValueError(f"Unsupported weights setting: {cfg['weights']}")

    net = resnet50(weights=weights)

    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features, cfg["num_classes"])

    net = net.to(device)

    n_params = sum(p.numel() for p in net.parameters())
    n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print(f"[model] torchvision ResNet50 pretrained={cfg['weights']}")
    print(f"[model] replaced fc: {in_features} -> {cfg['num_classes']}")
    print(f"[model] total params: {n_params / 1e6:.2f}M")
    print(f"[model] trainable params: {n_trainable / 1e6:.2f}M")

    return net


# =============================================================
# 5. FREEZE / UNFREEZE HELPERS
# =============================================================
def freeze_backbone_train_fc_only(net):
    """
    Stage 1:
    Freeze all pretrained layers and train only the new fc layer.
    """

    for name, param in net.named_parameters():
        param.requires_grad = name.startswith("fc.")

    trainable = [name for name, p in net.named_parameters() if p.requires_grad]
    print(f"[stage 1] trainable layers: {trainable}")


def unfreeze_all_layers(net):
    """
    Stage 2:
    Unfreeze the whole network for full fine-tuning.
    """

    for param in net.parameters():
        param.requires_grad = True

    print("[stage 2] all layers are trainable")


# =============================================================
# 6. OPTIMIZER + SCHEDULER + LOSS
# =============================================================
def build_stage1_optimizer(net, cfg):
    """
    Stage 1 optimizer: train only fc.
    """

    return SGD(
        net.fc.parameters(),
        lr=cfg["lr_fc_stage1"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
        nesterov=cfg["nesterov"],
    )


def build_stage2_optimizer(net, cfg):
    """
    Stage 2 optimizer:
    smaller LR for pretrained backbone, larger LR for the new fc layer.
    """

    backbone_params = []
    head_params = []

    for name, param in net.named_parameters():
        if name.startswith("fc."):
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = SGD(
        [
            {"params": backbone_params, "lr": cfg["lr_backbone_stage2"], "name": "backbone"},
            {"params": head_params, "lr": cfg["lr_fc_stage2"], "name": "fc"},
        ],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
        nesterov=cfg["nesterov"],
    )

    return optimizer


def build_stage2_scheduler(optimizer, cfg):
    return CosineAnnealingLR(
        optimizer,
        T_max=cfg["full_ft_epochs"],
        eta_min=cfg["min_lr"],
    )


def build_criterion(cfg):
    return nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])


def get_lr_string(optimizer):
    pieces = []
    for i, group in enumerate(optimizer.param_groups):
        name = group.get("name", f"group{i}")
        pieces.append(f"{name}:{group['lr']:.6g}")
    return " | ".join(pieces)


# =============================================================
# 7. TRAINING LOOP
# =============================================================

def run_single_training(cfg, dataset, overlap_rate=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reset seed before each run so each overlap-rate run is reproducible.
    set_global_seed(cfg["seed"])

    g = torch.Generator()
    g.manual_seed(cfg["seed"])

    group_mode = cfg["group_mode"].upper()

    if group_mode == "A":
        scenario_name = (
            f"{cfg['scenario_name']}_victim_A"
            f"_s{cfg['seed']}_r{cfg['round']}"
        )
    else:
        overlap_tag = str(overlap_rate).replace(".", "p")
        scenario_name = (
            f"{cfg['scenario_name']}_neg_B_ov{overlap_tag}"
            f"_s{cfg['seed']}_r{cfg['round']}"
        )

    print("\n" + "=" * 80)
    print(f"[run] scenario: {scenario_name}")
    print("=" * 80)

    os.makedirs(cfg["log_dir"], exist_ok=True)
    log_file = os.path.join(cfg["log_dir"], f"training_log_{scenario_name}.csv")
    init_log_file(log_file)

    ckpt_dir = Path(cfg["ckpt_dir"]) / scenario_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Data
    # -----------------------------
    train_set_full = dataset.train_set
    test_set = dataset.test_set

    train_indices = build_train_indices(
        cfg=cfg,
        train_set_full=train_set_full,
        overlap_rate=overlap_rate,
    )

    train_subset = dataset.subset("train", train_indices, clean=False)

    print(f"[data] train subset: {len(train_subset)}")
    print(f"[data] test set: {len(test_set)}")

    use_persistent = cfg["num_workers"] > 0

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=use_persistent,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=use_persistent,
        pin_memory=True,
    )

    # -----------------------------
    # Model + loss
    # -----------------------------
    print("==> Building model...")
    net = build_model(cfg, device)
    criterion = build_criterion(cfg)

    best_test_acc = -1.0

    # =========================================================
    # Stage 1: train fc only
    # =========================================================
    print("\n==> Stage 1: train fc only")
    freeze_backbone_train_fc_only(net)

    optimizer = build_stage1_optimizer(net, cfg)

    for local_epoch in range(cfg["fc_warmup_epochs"]):
        global_epoch = local_epoch
        lr_string = get_lr_string(optimizer)

        print(f"\n[epoch {global_epoch}/{cfg['total_epochs']}] stage=fc_only lr={lr_string}")

        train_result = train_one_epoch(
            net, train_loader, optimizer, criterion, device, cfg
        )
        test_result = evaluate(
            net, test_loader, criterion, device, use_tta=cfg["use_tta"]
        )

        print(f"  train loss={train_result['train_loss']:.4f} acc={train_result['train_acc']:.2f}%")
        print(f"  test  loss={test_result['test_loss']:.4f} acc={test_result['test_acc']:.2f}%")

        log_epoch(
            log_file,
            scenario_name,
            global_epoch,
            "fc_only",
            train_result,
            test_result,
            lr_string,
        )

        if test_result["test_acc"] > best_test_acc:
            best_test_acc = test_result["test_acc"]
            torch.save(net.state_dict(), ckpt_dir / "best_epoch.pth")
            print(f"  ** new best: {best_test_acc:.2f}% -> saved best_epoch.pth")

    # =========================================================
    # Stage 2: full fine-tuning
    # =========================================================
    print("\n==> Stage 2: full network fine-tuning")
    unfreeze_all_layers(net)

    optimizer = build_stage2_optimizer(net, cfg)
    scheduler = build_stage2_scheduler(optimizer, cfg)

    for local_epoch in range(cfg["full_ft_epochs"]):
        global_epoch = cfg["fc_warmup_epochs"] + local_epoch
        lr_string = get_lr_string(optimizer)

        print(f"\n[epoch {global_epoch}/{cfg['total_epochs']}] stage=full_ft lr={lr_string}")

        train_result = train_one_epoch(
            net, train_loader, optimizer, criterion, device, cfg
        )
        test_result = evaluate(
            net, test_loader, criterion, device, use_tta=cfg["use_tta"]
        )

        scheduler.step()

        print(f"  train loss={train_result['train_loss']:.4f} acc={train_result['train_acc']:.2f}%")
        print(f"  test  loss={test_result['test_loss']:.4f} acc={test_result['test_acc']:.2f}%")

        log_epoch(
            log_file,
            scenario_name,
            global_epoch,
            "full_ft",
            train_result,
            test_result,
            lr_string,
        )

        if test_result["test_acc"] > best_test_acc:
            best_test_acc = test_result["test_acc"]
            torch.save(net.state_dict(), ckpt_dir / "best_epoch.pth")
            print(f"  ** new best: {best_test_acc:.2f}% -> saved best_epoch.pth")

    torch.save(net.state_dict(), ckpt_dir / f"epoch_{cfg['total_epochs'] - 1}.pth")

    print("\n[done]")
    print(f"[done] scenario: {scenario_name}")
    print(f"[done] best test acc: {best_test_acc:.2f}%")
    print(f"[done] log file: {log_file}")
    print(f"[done] checkpoints in: {ckpt_dir}")

    return {
        "scenario_name": scenario_name,
        "best_test_acc": best_test_acc,
        "log_file": log_file,
        "ckpt_dir": str(ckpt_dir),
    }


def train_one_epoch(net, loader, optimizer, criterion, device, cfg):
    net.train()

    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = net(inputs)
        loss = criterion(logits, targets)

        loss.backward()

        if cfg.get("grad_clip") is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for p in net.parameters() if p.requires_grad],
                cfg["grad_clip"],
            )

        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        all_preds.append(predicted.detach().cpu().numpy())
        all_labels.append(targets.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    return {
        "train_loss": total_loss / total,
        "train_acc": 100.0 * correct / total,
        "train_precision": precision,
        "train_recall": recall,
        "train_f1": f1,
    }


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
            logits = 0.5 * (logits + logits_flip)

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
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    return {
        "test_loss": total_loss / total,
        "test_acc": 100.0 * correct / total,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
    }


# =============================================================
# 8. LOGGING
# =============================================================
def init_log_file(log_file):
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Scenario", "Epoch", "Stage",
                "Train_Loss", "Train_Acc", "Train_Precision", "Train_Recall", "Train_F1",
                "Test_Loss", "Test_Acc", "Test_Precision", "Test_Recall", "Test_F1",
                "LR",
            ])


def log_epoch(log_file, scenario_name, epoch, stage, train_result, test_result, lr_string):
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            scenario_name,
            epoch,
            stage,
            train_result["train_loss"],
            train_result["train_acc"],
            train_result["train_precision"],
            train_result["train_recall"],
            train_result["train_f1"],
            test_result["test_loss"],
            test_result["test_acc"],
            test_result["test_precision"],
            test_result["test_recall"],
            test_result["test_f1"],
            lr_string,
        ])


# =============================================================
# 9. MAIN
# =============================================================
def main(cfg=None):
    if cfg is None:
        cfg = CONFIG

    assert cfg["total_epochs"] == cfg["fc_warmup_epochs"] + cfg["full_ft_epochs"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    set_global_seed(cfg["seed"])

    # -----------------------------
    # Build dataset once
    # -----------------------------
    print("==> Preparing dataset once...")
    dataset = build_dataset(cfg)

    results = []

    group_mode = cfg["group_mode"].upper()

    if group_mode == "A":
        # Train one victim model on Group A.
        result = run_single_training(
            cfg=cfg,
            dataset=dataset,
            overlap_rate=None,
        )
        results.append(result)

    elif group_mode == "B":
        # Train one negative model for each overlap rate.
        overlap_rates = cfg.get("overlap_rates", None)

        if overlap_rates is None:
            raise ValueError(
                "When group_mode == 'B', cfg must contain 'overlap_rates', "
                "for example [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]."
            )

        for overlap_rate in overlap_rates:
            result = run_single_training(
                cfg=cfg,
                dataset=dataset,
                overlap_rate=overlap_rate,
            )
            results.append(result)

            # Clear GPU memory between runs.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    else:
        raise ValueError(f"Unknown group_mode: {cfg['group_mode']}. Use 'A' or 'B'.")

    print("\n" + "=" * 80)
    print("[summary]")
    print("=" * 80)

    for item in results:
        print(
            f"{item['scenario_name']}: "
            f"best_test_acc={item['best_test_acc']:.2f}%"
        )


if __name__ == "__main__":
    main()