"""Knockoff (soft-label) model extraction with a DeiT victim.

Fork of `main_knockoff_extraction.py`. The attack itself is unchanged - the
victim is still queried as a black box and the substitute is still trained on
the returned soft labels with a KL objective. Only three things differ:

  * the victim is a timm DeiT, built through `build_deit_student` so its
    structure matches the vanilla-training checkpoint, and loaded from
    saved_models/vanilla/Transformer_Models/ (CNN victims resolve to
    CNN_Models/ automatically, so a DeiT surrogate can also steal a CNN);
  * `Victim.Model` / `Substitute.Model` may be a dict (DeiT) as well as a
    string (CNN), so the surrogate can be either family;
  * the scheduler dispatch understands `WarmupCosineAnnealingLR`, which the
    DeiT recipe needs and which does not exist in torch.optim.lr_scheduler.

Each surrogate is trained with the SAME recipe used to train that architecture
normally (SGD/cosine/160ep for ResNet-18, AdamW/warmup-cosine/200ep for DeiT);
that recipe lives entirely in the YAML, so nothing here is architecture-specific.

Results are written under saved_models/extraction_final/ and
saved_logs/extraction_final/Performance/ -- the SAME flat tree
main_knockoff_extraction.py writes to, one directory per scenario, with no
Transformer_Models sub-level. This deliberately departs from how main_ft.py and
main_removalnet_deit.py separate transformer runs: here the two families share a
scenario namespace already (the surrogate suffix keeps names unique), and most
runs driven by this script have a CNN victim with only a DeiT surrogate, so a
"Transformer_Models" folder would have described them wrongly. The MI master
table is shared with the CNN runs so all scenarios plot together.
"""
import os
import glob
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # needed for full CUDA determinism

import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import csv
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader, TensorDataset, Dataset
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from util import (create_or_load_group_B, evaluate_fidelity, process_yaml_file, build_dataset_from_yaml, query_victim, train_one_epoch, evaluate1,
                  create_or_load_group_A, load_best_checkpoint, train_one_epoch_knockoff,
                  build_deit_student, build_warmup_cosine_scheduler)
from Model.ResNet_18 import ResNet18
from Model.VGG16 import ModifiedVGG16
from Model.MLP import MNIST_MLP
import torch.optim.lr_scheduler as lr_sched


class SoftLabeledSubset(Dataset):
    """Pairs (augmented_image, soft_label) by index.
    
    aug_dataset[i] returns the augmented image whose underlying raw image
    is the same as the one used to produce soft_labels[i]. Both must come
    from the same indices list (group_B here), iterated in the same order.
    """
    def __init__(self, aug_dataset, soft_labels):
        assert len(aug_dataset) == len(soft_labels), \
            f"length mismatch: {len(aug_dataset)} vs {len(soft_labels)}"
        self.aug_dataset = aug_dataset
        self.soft_labels = soft_labels

    def __len__(self):
        return len(self.aug_dataset)

    def __getitem__(self, idx):
        img, _ = self.aug_dataset[idx]   # discard the hard label
        return img, self.soft_labels[idx]


# =====================================================
# 1. Global setup
# =====================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Skip any (plan, extraction_seed) whose best_epoch.pth is already on disk.
# This script has no resume logic of its own: every call trains from scratch and
# overwrites. Without this guard, re-running to top up seed coverage would redo
# -- and silently replace -- the runs that already finished (cudnn.benchmark is
# on, so a redo does not even reproduce the same weights). Set False to force.
SKIP_EXISTING = True

# Output roots. Everything for the final runs goes under extraction_final/, FLAT --
# deliberately no Transformer_Models sub-level, unlike the old extraction_vanilla
# layout. Two reasons: scenario names are already unique across families (the
# surrogate suffix distinguishes them: _Same18 / _Cross16 / _CrossDeiT / _SameDeiT),
# and "Transformer_Models" was misleading here anyway -- most of what landed in it
# has a CNN victim and only a DeiT surrogate. One flat tree for downstream MI.
# Previous roots: './saved_models/extraction_vanilla/Transformer_Models'
#                 './saved_logs/extraction_vanilla/Performance/Transformer_Models'
EXTRACTION_ROOT = './saved_models/extraction_final'
EXTRACTION_LOG_ROOT = './saved_logs/extraction_final/Performance'

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
# 2. Knockoff attack helpers
# =====================================================
def build_model(model_name, num_classes):
    if model_name == "MLP":
        return MNIST_MLP()
    elif model_name == "ResNet-18":
        return ResNet18(num_classes=num_classes)
    elif model_name == "VGG16":
        return ModifiedVGG16(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def build_model_any(model_cfg, num_classes):
    """Build either a CNN or a DeiT, depending on how the YAML declares it.

    str  -> "ResNet-18" / "VGG16" / "MLP", the unchanged CNN path.
    dict -> {model_name, img_size, patch_size, pretrained, drop_path_rate},
            built through timm by the same helper the vanilla DeiT training
            used, so the structure matches the saved checkpoint exactly.
            For CIFAR the victim is img_size=32 / patch_size=4, i.e. the
            substitute consumes the very same 32x32 tensors as a CNN would.
    """
    if isinstance(model_cfg, dict):
        return build_deit_student({"Model": model_cfg}, num_classes)
    return build_model(model_cfg, num_classes)


# =====================================================
# 3. Main knockoff extraction logic
# =====================================================
def main_knockoff(model_seed, extract_seed, yaml_file_path):
    print(device)
    start_epoch = 0

    g = torch.Generator()
    g.manual_seed(extract_seed)

    exp_yaml = process_yaml_file(yaml_file_path)

    # ----- Skip-existing guard -----
    # Checked before the victim, the auxiliary dataset and the query pass, so a
    # skipped cell costs nothing but a YAML parse.
    scenario_name = exp_yaml["Scenario_Name"] + f"_{extract_seed}_{1.0}"
    save_dir = f'{EXTRACTION_ROOT}/{scenario_name}'
    best_ckpt_path = f'{save_dir}/best_epoch.pth'
    if SKIP_EXISTING and os.path.isfile(best_ckpt_path) and os.path.getsize(best_ckpt_path) > 0:
        print(f"[SKIP] {scenario_name}: best_epoch.pth already exists -> {best_ckpt_path}")
        return

    # ----- Parse victim configuration -----
    victim_cfg = exp_yaml["Victim"]
    victim_ds_cfg = victim_cfg["Dataset"]
    victim_dataset_obj, victim_num_classes, victim_group_size = build_dataset_from_yaml(victim_ds_cfg)
    victim_model_cfg = victim_cfg.get("Model", "ResNet-18")

    # ----- Load pre-trained victim model -----
    print('==> Loading victim model..')
    victim_net = build_model_any(victim_model_cfg, victim_num_classes).to(device)

    # Construct victim model path: saved_models/vanilla/{Model_Name}_{seed}_{r}/best_epoch.pth
    victim_folder_name = victim_cfg["Model_Name"] + f"_{model_seed}_{1.0}"
    # Victim root: DeiT victims live in Transformer_Models, CNN victims in
    # CNN_Models. Inferred from how the YAML declares Victim.Model (dict = DeiT
    # built through timm, str = CNN), so the existing DeiT plans keep resolving
    # exactly as before. An explicit `Victim.Model_Dir` overrides the guess.
    default_victim_root = ('./saved_models/vanilla/Transformer_Models'
                           if isinstance(victim_model_cfg, dict)
                           else './saved_models/vanilla/CNN_Models')
    victim_model_dir = Path(victim_cfg.get("Model_Dir", default_victim_root)) / victim_folder_name
    ckpt_path, _ = load_best_checkpoint(victim_model_dir)
    if ckpt_path is None:
        print(f"[WARNING] No checkpoint found in {victim_model_dir}, trying without seed suffix...")
        raise FileNotFoundError(f"No victim model checkpoint found in {victim_model_dir}")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    victim_net.load_state_dict(state)
    victim_net.eval()
    print(f'Victim model loaded from: {ckpt_path}')

    # ----- Parse auxiliary dataset -----
    print('==> Preparing auxiliary dataset..')
    aux_ds_cfg = exp_yaml.get("Auxiliary_Dataset", victim_ds_cfg)
    aux_dataset_obj, aux_num_classes, aux_group_size = build_dataset_from_yaml(aux_ds_cfg)
    print(f"aux_dataset classes: {aux_num_classes}")
    print(f"axu_group_size: {aux_group_size}")

    # Use in_sample_set (clean transforms, no augmentation) for querying the victim.
    # This ensures soft labels are computed on non-augmented images for reliability.
    train_set = aux_dataset_obj.train_set

    group_A = create_or_load_group_A(dataset=train_set, save_dir=f'./Indices/{aux_ds_cfg["name"]}/',
                                    group_size=aux_group_size, num_classes=aux_num_classes, seed=42, force_rebuild=False)
    
    group_B = create_or_load_group_B(dataset=train_set, save_dir=f'./Indices/{aux_ds_cfg["name"]}/', 
                                     group_A_indices=group_A, group_size=aux_group_size, num_classes=aux_num_classes, 
                                     overlap_rate=0.0, seed=42, force_rebuild=False)

    query_subset =  aux_dataset_obj.subset("train", group_B, clean=True) # without augmentation

    # Optionally subsample the auxiliary data
    knockoff_cfg = exp_yaml.get("Knockoff", {})
    sampling_size = knockoff_cfg.get("sampling_size", 1.0)

    # Create a loader for querying (clean transforms, no augmentation)
    query_loader = DataLoader(query_subset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    # ----- Query victim to get soft labels -----
    print('==> Querying victim model (blackbox)..')
    stolen_inputs, stolen_labels = query_victim(victim_net, query_loader, device)
    print(f' Collected {stolen_inputs.shape[0]} query-response pairs')
    print(f' Input shape: {stolen_inputs.shape}, Label shape: {stolen_labels.shape}')

    # Sanity check: verify that the underlying raw images align between
    # the clean query path and the augmented training path. We compare the
    # raw (unaugmented) images at a few sample indices.
    for i in [0, len(group_B)//2, len(group_B)-1]:
        img_clean, _ = query_subset[i]
        # If subset returns PIL Images at this stage, compare differently;
        # if it already applied ToTensor, hash the tensor.
        assert torch.allclose(img_clean, stolen_inputs[i], atol=1e-6), \
            f"index alignment broken at i={i}"
    print("Index alignment verified.")

    # ----- Build substitute training set with augmentation -----
    augmented_subset = aux_dataset_obj.subset("train", group_B, clean=False)
    stolen_dataset = SoftLabeledSubset(augmented_subset, stolen_labels)
    stolen_loader = DataLoader(stolen_dataset, batch_size=128, shuffle=True,
                                num_workers=8, worker_init_fn=seed_worker,
                                generator=g, persistent_workers=True, pin_memory=True)


    # ----- Prepare test set (same as victim's test set for fair evaluation) -----
    test_set = victim_dataset_obj.test_set
    testloader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    # ----- Build substitute model -----
    print('==> Building substitute model..')
    sub_model_cfg = exp_yaml["Substitute"].get("Model", victim_model_cfg)
    substitute_net = build_model_any(sub_model_cfg, victim_num_classes).to(device)

    # ----- Optimizer & scheduler -----
    optimizer_cfg = exp_yaml.get("Optimizer", {})
    optimizer_name = optimizer_cfg.get("name", "Adam")
    optimizer_params = optimizer_cfg.get("params", {"lr": 1e-3})
    optimizer_class = getattr(optim, optimizer_name)
    optimizer = optimizer_class(substitute_net.parameters(), **optimizer_params)

    scheduler_cfg = exp_yaml.get("Scheduler", {})
    scheduler = None
    if scheduler_cfg:
        scheduler_name = scheduler_cfg.get("name", None)
        scheduler_params = scheduler_cfg.get("params", {})
        if scheduler_name == "WarmupCosineAnnealingLR":
            # Not a torch.optim.lr_scheduler class - the DeiT recipe needs the
            # repo's warmup+cosine helper, the same one the vanilla DeiT
            # training used (see util.process_experiment_setup_deit).
            scheduler = build_warmup_cosine_scheduler(
                optimizer,
                total_epochs=int(scheduler_params.get("T_max", exp_yaml.get("Epochs", 100))),
                warmup_epochs=int(scheduler_params.get("warmup_epochs", 10)),
                warmup_start_factor=float(scheduler_params.get("warmup_start_factor", 0.1)),
                eta_min=float(scheduler_params.get("eta_min", 1e-6)),
            )
        elif scheduler_name is not None:
            scheduler_class = getattr(lr_sched, scheduler_name)
            scheduler = scheduler_class(optimizer, **scheduler_params)

    # ----- Soft-label loss (KL divergence) -----
    # Since victim outputs are soft probabilities, we use KL divergence
    def soft_label_loss(logits, soft_targets):
        log_probs = torch.log_softmax(logits, dim=1)
        return nn.KLDivLoss(reduction='batchmean')(log_probs, soft_targets)

    eval_criterion = nn.CrossEntropyLoss()

    # ----- Logging -----  (scenario_name / save_dir defined above, with the skip guard)
    log_dir = EXTRACTION_LOG_ROOT
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{scenario_name}.csv")

    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Scenario', 'Epoch', 'Train_Loss', 'Train_Acc', 'Train_Precision', 'Train_Recall', 'Train_F1',
                             'Test_Loss', 'Test_Acc', 'Test_Precision', 'Test_Recall', 'Test_F1'])

    # ----- Also evaluate victim on test set for reference -----
    print('==> Evaluating victim model on test set (reference)..')
    victim_test_result = evaluate1(victim_net, testloader, eval_criterion, device)
    print(f'    Victim Test Acc: {victim_test_result["test_acc"]:.2f}%')

    # ----- Training loop -----
    best_test_acc = -1.0
    os.makedirs(save_dir, exist_ok=True)

    num_epochs = exp_yaml.get("Epochs", 100)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        print(f"This is the model seed{model_seed} extraction seed {extract_seed} round, {epoch} epoch!")
        train_result = train_one_epoch_knockoff(substitute_net, stolen_loader, optimizer,
                                                 soft_label_loss, epoch, device)
        if scheduler is not None:
            scheduler.step()
        print(f"Epoch {epoch} | LR = {optimizer.param_groups[0]['lr']}")

        test_result = evaluate1(substitute_net, testloader, eval_criterion, device)

        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([scenario_name, epoch,
                             train_result["train_loss"], train_result["train_acc"],
                             train_result["train_precision"], train_result["train_recall"], train_result["train_f1"],
                             test_result["test_loss"], test_result["test_acc"],
                             test_result["test_precision"], test_result["test_recall"], test_result["test_f1"],
                             ])

        current_acc = test_result["test_acc"]
        if current_acc > best_test_acc:
            best_test_acc = current_acc
            torch.save(substitute_net.state_dict(), best_ckpt_path)

    torch.save(substitute_net.state_dict(), f'{save_dir}/epoch_{epoch}.pth')

    # Reload best checkpoint so that fidelity, acc_recovery, and the model used
    # downstream (RobD / IPGuard / Dataset Inference) all refer to the same weights.
    substitute_net.load_state_dict(
        torch.load(best_ckpt_path, map_location=device, weights_only=False)
    )
    substitute_net.eval()

    print(f'\n==> Knockoff extraction complete.')
    fidelity = evaluate_fidelity(victim_net, substitute_net, testloader, device)
    acc_recovery = 100.0 * best_test_acc / victim_test_result["test_acc"]

    print(f"Victim Test Acc:      {victim_test_result['test_acc']:.2f}%")
    print(f"Substitute Best Acc:  {best_test_acc:.2f}%")
    print(f"Accuracy Recovery:    {acc_recovery:.1f}%")
    print(f"Fidelity:             {fidelity:.1f}%")


# =====================================================
# 4. Entry point
# =====================================================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Folder containing all YAML experiment plans.
    # The DeiT half of the CIFAR-10 Same10 3x3 matrix: the 5 cells with a DeiT
    # on either side (RN18->DeiT, VGG16->DeiT, DeiT->RN18, DeiT->VGG16,
    # DeiT->DeiT). The 4 CNN-only cells live in knockoff_3x3_c10_same10 and are
    # run by main_knockoff_extraction.py instead.
    # Previous folder: "./saved_exp_plan/extraction_plan_crossarch"
    exp_folder = "./saved_exp_plan/knockoff_3x3_c10_same10_deit"
    yaml_files = sorted(glob.glob(os.path.join(exp_folder, "*.yaml")))

    if not yaml_files:
        print(f"No YAML files found in {exp_folder}")
    else:
        print(f"Found {len(yaml_files)} experiment plan(s):")
        for f in yaml_files:
            print(" -", f)

    # Iterate over YAML files and seeds
    for yaml_path in yaml_files:
        print(f"\n========== Starting experiments from {yaml_path} ==========")

        for model_seed in range(42, 43):
            for extraction_seed in range(0, 3):   # seeds 0,1,2; finished cells are skipped
                print(f"\n>>> Running model seed {model_seed} , extraction seed {extraction_seed} for {os.path.basename(yaml_path)}")
                set_seed(extraction_seed)

                main_knockoff(model_seed, extraction_seed, yaml_path)
