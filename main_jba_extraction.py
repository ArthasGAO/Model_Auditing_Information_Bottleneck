"""main_jba_pos.py

Jacobian-Based Augmentation (JBA) model extraction attack, framework-aligned
with main_knockoff_pos.py.

Faithful to Papernot et al. 2017 ("Practical Black-Box Attacks against ML"):
  - Hard-label only access to victim
  - Iterative dataset growth via Jacobian of *victim-predicted class logit*
  - Periodic sign-flip schedule on lambda (every tau rounds)
  - Sample efficiency: starts from N seeds, grows to N * 2^rho samples
"""
import os
import glob
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import csv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from util import (process_yaml_file, build_dataset_from_yaml,
                  create_or_load_group_A, create_or_load_group_B,
                  load_best_checkpoint, evaluate1, evaluate_fidelity)
from Model.ResNet_18 import ResNet18
from Model.VGG16 import ModifiedVGG16
from Model.MLP import MNIST_MLP


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


def build_model(model_name, num_classes):
    if model_name == "MLP":      return MNIST_MLP()
    if model_name == "ResNet-18": return ResNet18(num_classes=num_classes)
    if model_name == "VGG16":     return ModifiedVGG16(num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_name}")


# =====================================================
# Core JBA augmentation -- the only piece that's new vs Knockoff
# =====================================================
def jacobian_augment(substitute, X, y_victim, scale, device, clip_range=None):
    """Generate new samples via sign(Jacobian) of the victim-predicted class.

    Faithful to Papernot 2017:
        x_new = x + scale * sign( d F_{y_v(x)}(x) / d x )

    where F_{y_v(x)} is the substitute's logit at the victim-predicted class.
    Crucially, gradient is of the *predicted-class logit only*, NOT the sum of
    all logits (a common reproduction bug; when the substitute outputs softmax
    probabilities, the latter is identically zero).

    Args:
        substitute: PyTorch model returning logits (pre-softmax).
        X: (N, C, H, W) tensor in the input space the substitute expects.
        y_victim: (N,) int tensor of victim hard labels.
        scale: signed step size (sign flips per the round schedule).
        device: cuda / cpu.
        clip_range: optional (min, max) for valid pixel range in the input space.

    Returns:
        X_new: (N, C, H, W) augmented tensor on CPU.
    """
    substitute.eval()
    X = X.clone().detach().to(device).requires_grad_(True)
    y_victim = y_victim.to(device).long()

    logits = substitute(X)
    # gather(1, y_v) picks logits[i, y_victim[i]]; summing then taking grad
    # gives per-sample gradient of that sample's predicted-class logit.
    selected = logits.gather(1, y_victim.view(-1, 1)).squeeze(1).sum()
    grad = torch.autograd.grad(selected, X)[0]

    X_new = X.detach() + scale * grad.sign()
    if clip_range is not None:
        X_new = X_new.clamp(min=clip_range[0], max=clip_range[1])
    return X_new.detach().cpu()


# =====================================================
# Train substitute for one round (hard-label CE)
# =====================================================
def train_substitute_round(substitute, X_data, y_data, optimizer, criterion,
                           epochs, batch_size, generator, device):
    """Train substitute for `epochs` epochs on the current (X, y) dataset.
    Returns the final-epoch metrics."""
    train_ds = TensorDataset(X_data, y_data)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, worker_init_fn=seed_worker,
                              generator=generator, pin_memory=True)
    substitute.train()
    final_loss = final_acc = 0.0
    for epoch in range(epochs):
        running_loss, correct, total = 0.0, 0, 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).long()
            optimizer.zero_grad()
            logits = substitute(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_batch.size(0)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += x_batch.size(0)
        final_loss = running_loss / total
        final_acc = 100. * correct / total
    return final_loss, final_acc


# =====================================================
# Main JBA extraction
# =====================================================
def main_jba(model_seed, extract_seed, yaml_file_path):
    print(device)
    g = torch.Generator(); g.manual_seed(extract_seed)
    exp_yaml = process_yaml_file(yaml_file_path)

    # ----- Victim -----
    victim_cfg = exp_yaml["Victim"]
    victim_ds_cfg = victim_cfg["Dataset"]
    victim_dataset_obj, victim_num_classes, _ = build_dataset_from_yaml(victim_ds_cfg)
    victim_model_name = victim_cfg.get("Model", "ResNet-18")

    print('==> Loading victim..')
    victim_net = build_model(victim_model_name, victim_num_classes).to(device)
    victim_folder_name = victim_cfg["Model_Name"] + f"_{model_seed}_{1.0}"
    victim_model_dir = Path('./saved_models/vanilla/CNN_Models/') / victim_folder_name
    ckpt_path, _ = load_best_checkpoint(victim_model_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"No victim checkpoint in {victim_model_dir}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    victim_net.load_state_dict(state)
    victim_net.eval()
    for p in victim_net.parameters():
        p.requires_grad_(False)
    print(f'   Loaded: {ckpt_path}')

    # ----- Auxiliary (seed pool) -----
    aux_ds_cfg = exp_yaml.get("Auxiliary_Dataset", victim_ds_cfg)
    aux_dataset_obj, aux_num_classes, aux_group_size = build_dataset_from_yaml(aux_ds_cfg)
    train_set = aux_dataset_obj.train_set

    group_A = create_or_load_group_A(
        dataset=train_set, save_dir=f'./Indices/{aux_ds_cfg["name"]}/',
        group_size=aux_group_size, num_classes=aux_num_classes,
        seed=42, force_rebuild=False)
    group_B = create_or_load_group_B(
        dataset=train_set, save_dir=f'./Indices/{aux_ds_cfg["name"]}/',
        group_A_indices=group_A, group_size=aux_group_size,
        num_classes=aux_num_classes, overlap_rate=0.0,
        seed=42, force_rebuild=False)

    # ----- JBA-specific config -----
    jba_cfg = exp_yaml.get("JBA", {})
    num_seeds      = jba_cfg.get("num_seeds", 150)
    extract_rounds = jba_cfg.get("extract_rounds", 6)
    scale_const    = jba_cfg.get("scale_const", 0.1)
    tau            = jba_cfg.get("tau", 2)
    epochs_per_rd  = jba_cfg.get("epochs_per_round", 10)
    batch_size     = jba_cfg.get("batch_size", 128)
    clip_range_yaml = jba_cfg.get("clip_range", None)
    clip_range = tuple(clip_range_yaml) if clip_range_yaml else None

    # ----- Collect seed samples + victim hard labels -----
    print(f'==> Building seed set ({num_seeds} samples from group_B)..')
    seed_indices = list(group_B[:num_seeds])
    # clean=True -> ToTensor+Normalize only, no augmentation -> reproducible queries
    seed_subset = aux_dataset_obj.subset("train", seed_indices, clean=True)
    seed_loader = DataLoader(seed_subset, batch_size=128, shuffle=False,
                             num_workers=4, worker_init_fn=seed_worker,
                             generator=g, pin_memory=True)
    X_list, y_list = [], []
    with torch.no_grad():
        for x_batch, _ in seed_loader:
            x_batch = x_batch.to(device)
            y_pred = victim_net(x_batch).argmax(dim=1).cpu()
            X_list.append(x_batch.cpu())
            y_list.append(y_pred)
    X_data = torch.cat(X_list, dim=0)
    y_data = torch.cat(y_list, dim=0)
    print(f'   X_data {X_data.shape}, y_data {y_data.shape}')

    # ----- Substitute -----
    print('==> Building substitute..')
    sub_model_name = exp_yaml["Substitute"].get("Model", victim_model_name)
    substitute_net = build_model(sub_model_name, victim_num_classes).to(device)

    optimizer_cfg = exp_yaml.get("Optimizer", {"name": "Adam", "params": {"lr": 1e-3}})
    optimizer = getattr(optim, optimizer_cfg.get("name", "Adam"))(
        substitute_net.parameters(), **optimizer_cfg.get("params", {}))

    # Hard-label CE per the JBA paper (vs. Knockoff which uses KL with soft labels)
    criterion = nn.CrossEntropyLoss()

    # ----- Test set -----
    test_set = victim_dataset_obj.test_set
    testloader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4,
                            worker_init_fn=seed_worker, generator=g, pin_memory=True)
    victim_test_result = evaluate1(victim_net, testloader, criterion, device)
    print(f'   Victim Test Acc: {victim_test_result["test_acc"]:.2f}%')

    # ----- Logging -----
    scenario_name = exp_yaml["Scenario_Name"] + f"_{extract_seed}_{1.0}"
    log_dir = './saved_logs/extraction_vanilla/Performance'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_log_{scenario_name}.csv')
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            csv.writer(f).writerow(['Scenario', 'Round', 'Dataset_Size', 'Scale',
                                    'Train_Loss', 'Train_Fidelity',
                                    'Test_Loss', 'Test_Acc'])

    save_dir = f'./saved_models/extraction_vanilla/{scenario_name}'
    os.makedirs(save_dir, exist_ok=True)
    best_test_acc = -1.0

    # ===== Main JBA loop =====
    for round_idx in range(1, extract_rounds + 1):
        # Periodic sign-flip schedule (Papernot 2017): lambda_rho = lambda * (-1)^(rho // tau)
        rho = round_idx - 1
        scale = scale_const * ((-1) ** (rho // tau))
        print(f'\n=== Round {round_idx}/{extract_rounds} | '
              f'scale={scale:+.3f} | dataset={len(X_data)} ===')

        # Train substitute on current dataset
        train_loss, train_fid = train_substitute_round(
            substitute_net, X_data, y_data, optimizer, criterion,
            epochs=epochs_per_rd, batch_size=batch_size, generator=g, device=device)

        # Evaluate
        test_result = evaluate1(substitute_net, testloader, criterion, device)
        print(f'   train_loss={train_loss:.4f} train_fid={train_fid:.2f}% '
              f'test_acc={test_result["test_acc"]:.2f}%')

        with open(log_file, 'a', newline='') as f:
            csv.writer(f).writerow([scenario_name, round_idx, len(X_data), scale,
                                    train_loss, train_fid,
                                    test_result["test_loss"], test_result["test_acc"]])

        if test_result["test_acc"] > best_test_acc:
            best_test_acc = test_result["test_acc"]
            torch.save(substitute_net.state_dict(), f'{save_dir}/best_epoch.pth')

        # Augment (skip after final round; we already evaluated above)
        if round_idx < extract_rounds:
            print(f'   Augmenting via Jacobian..')
            aug_loader = DataLoader(TensorDataset(X_data, y_data),
                                    batch_size=500, shuffle=False)
            X_new_list, y_new_list = [], []
            for x_batch, y_batch in aug_loader:
                x_new = jacobian_augment(substitute_net, x_batch, y_batch,
                                         scale=scale, device=device,
                                         clip_range=clip_range)
                with torch.no_grad():
                    y_new = victim_net(x_new.to(device)).argmax(dim=1).cpu()
                X_new_list.append(x_new)
                y_new_list.append(y_new)
            X_data = torch.cat([X_data] + X_new_list, dim=0)
            y_data = torch.cat([y_data] + y_new_list, dim=0)
            print(f'   Dataset grew to {len(X_data)}')

    # ----- Final: reload best checkpoint and report consistent metrics -----
    torch.save(substitute_net.state_dict(), f'{save_dir}/final_round.pth')
    substitute_net.load_state_dict(
        torch.load(f'{save_dir}/best_epoch.pth', map_location=device, weights_only=False))
    substitute_net.eval()

    print(f'\n==> JBA extraction complete.')
    fidelity = evaluate_fidelity(victim_net, substitute_net, testloader, device)
    acc_recovery = 100.0 * best_test_acc / victim_test_result["test_acc"]
    total_queries = num_seeds * (2 ** (extract_rounds - 1))  # seeds + augments per round

    print(f'Victim Test Acc:    {victim_test_result["test_acc"]:.2f}%')
    print(f'Substitute Best:    {best_test_acc:.2f}%')
    print(f'Accuracy Recovery:  {acc_recovery:.1f}%')
    print(f'Fidelity:           {fidelity:.1f}%')
    print(f'Total queries:      {total_queries} '
          f'(seeds + {extract_rounds - 1} rounds of augmentation)')


# =====================================================
# Entry point (matches main_knockoff_pos.py structure)
# =====================================================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    exp_folder = "./saved_exp_plan/extraction_plan"
    yaml_files = sorted(glob.glob(os.path.join(exp_folder, "*.yaml")))
    print(f"Found {len(yaml_files)} JBA experiment plan(s)")

    for yaml_path in yaml_files:
        print(f"\n========== {yaml_path} ==========")
        for model_seed in range(42, 43):
            for extraction_seed in range(0, 1):
                print(f"\n>>> model_seed={model_seed} extract_seed={extraction_seed}")
                set_seed(extraction_seed)
                main_jba(model_seed, extraction_seed, yaml_path)