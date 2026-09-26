import copy
import os
import glob
import yaml
DETERMINISTIC = False

if DETERMINISTIC:
    # cuBLAS needs a fixed workspace, chosen before torch is imported.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from pathlib import Path
import torch
import torch.nn as nn
from Model.ResNet_18 import ResNet18
from Model.VGG16 import ModifiedVGG16
import torch.backends.cudnn as cudnn
import random
import numpy as np
from Dataset.CIFAR_10 import CIFAR10Dataset, CIFAR10PseudoLabelDataset
from Dataset.CIFAR_100 import CIFAR100Dataset
from torch.utils.data import Subset, DataLoader
from util import create_or_load_group_A, create_or_load_group_B, determine_ft_dataset, extract_ft_balanced_train_val_indices, load_best_checkpoint, setup_finetune, snapshot_params, verify_params_changed
import torch.optim as optim
import csv
from util import process_yaml_file, process_experiment_setup, train_one_epoch, evaluate1, create_train_subset,\
                 prune_model_global, check_pruned_weights, ft_one_epoch, load_pickel_dataset,\
                 prepare_group_subset, load_checkpoint_from_epoch, remove_prune_mask, process_experiment_prune_setup,\
                 verify_optimizer_param_binding
from util import sparsity_levels_from_setup   # sparsity 档位以 YAML plan 为准,不再用模块级常量
# ---- DeiT-only helpers (used exclusively by main_prune_deit; CNN path untouched) ----
from util import process_experiment_prune_setup_deit, remove_prune_mask_safe, evaluate_deit_ft
import torch.optim.lr_scheduler as lr_sched

# =====================================================
# 1. Global setup
# =====================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Default AT seeds: 0, 1, 2. Explicit --seeds overrides the environment range.
# SEED_START/SEED_END remain available for splitting work across shells.
SEED_START = int(os.environ.get("SEED_START", 0))
SEED_END   = int(os.environ.get("SEED_END", 3))

# ---- skip guard ---------------------------------------------------------
# A finished (plan, sparsity, model_seed, rate, ft_seed) leaves a non-empty
# best_epoch.pth under PRUNE_MODEL_ROOT (CNN) or its Transformer_Models
# subfolder (DeiT). The guard is per (plan, ft_seed) and requires EVERY
# sparsity in the plan to be done, because one call loops all sparsities.
# PRUNE_SKIP_EXISTING=0 forces a rebuild.
SKIP_EXISTING = os.environ.get("PRUNE_SKIP_EXISTING", "1") != "0"
PRUNE_MODEL_ROOT = "./saved_models/pruning_final"


def prune_scenario_name(scenario, sparsity, model_seed, r, ft_seed, ft_group_size,
                        strategy="FT-AL"):
    """Folder name main_prune / main_prune_deit write to."""
    return (f"{scenario}_{model_seed}_{round(r, 2)}_sparsity={sparsity}"
            f"_{strategy}_ftsize={ft_group_size}_ftseed={ft_seed}")


def prune_run_done(yaml_path, model_seed, r, ft_seed):
    """True only if EVERY sparsity in the plan already has best_epoch.pth."""
    with open(yaml_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    root = PRUNE_MODEL_ROOT
    if isinstance(data.get("Model"), dict):      # DeiT writes one level deeper
        root = os.path.join(root, "Transformer_Models")
    for opt in data.get("Optimizers", []) or []:
        name = prune_scenario_name(data["Scenario_Name"], opt["sparsity"], model_seed,
                                   r, ft_seed, data["FT_Dataset"]["group_size"])
        ckpt = os.path.join(root, name, "best_epoch.pth")
        if not (os.path.isfile(ckpt) and os.path.getsize(ckpt) > 0):
            return False
    return True
ROOT = Path(__file__).resolve().parent

def seed_worker(worker_id):
    # Worker gets a different, but deterministic, seed derived from the main seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # all GPUs

    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
        # Raises error if a non-deterministic op is used; great for debugging reproducibility
        torch.use_deterministic_algorithms(True)
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True
        # This flag is a sticky global: it must be cleared explicitly, or a
        # previous deterministic run in the same process keeps it enabled.
        torch.use_deterministic_algorithms(False)


# ---------------- CONFIGURATION ----------------
STRATEGY = ["FT-AL"] #"RT-AL", "FT-LL", In pruning task, we only consider fine tune all layers now.
                     # Since the model layer architecture has been destroyed, so it is recommended to train all together.
SPARSITY = [0.2, 0.4, 0.6, 0.8] #

BEST_CKPT_START_FRAC = 0.2

# Which arch family the entry point drives:
#   False -> main_prune      (CNN,  ./saved_exp_plan/prune_plan)
#   True  -> main_prune_deit (DeiT, ./saved_exp_plan/prune_plan_deit)
# Overridable per shell with the RUN_DEIT environment variable.
RUN_DEIT = bool(int(os.environ.get("RUN_DEIT", 0)))

# DeiT 的 patch_embed 不参与剪枝(见 util.prune_model_global),统计稀疏度时必须
# 用同一份排除规则,否则打印出来的 GLOBAL SPARSITY 会被 patch_embed 的权重稀释。
DEIT_PRUNE_EXCLUDE = ["patch_embed"]

# ------------------------------------------------
# ---------------- MAIN EXECUTION ----------------
def main_prune(model_seed, ft_seed, r, yaml_file_path):
    print(device)
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_prune_setup(exp_yaml) 

    # Optional plan key: layer-name substrings kept OUT of the pruned set (see
    # util.prune_model_global). Absent -> None -> the original behaviour. The
    # same list is used for the sparsity printout so `amount` and the measured
    # GLOBAL SPARSITY share a denominator.
    prune_exclude = exp_yaml.get("Prune_Exclude") or None

    print('==> Loading model..')
    model_name = exp_yaml["Model_Name"] + f"_{model_seed}_{round(r,2)}"
    model_dir = './saved_models/vanilla/CNN_Models/'
    model_folder = Path(model_dir)/model_name

    ckpt_path, _ = load_best_checkpoint(model_folder) # find best checkpoint
    if ckpt_path is None:
        raise FileNotFoundError(f"No .pth files found in {model_folder}")

    print('==> Preparing data..')
    g = torch.Generator()
    g.manual_seed(ft_seed)

    group_A = create_or_load_group_A(dataset=exp_setup["Dataset"].train_set , save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
                                    group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"], seed=42, force_rebuild=False)
                                    # always used for evaluation in different case

    # -----------------------------
    # FT dataset preparation
    # -----------------------------
    ft_data_train, ft_data_val = determine_ft_dataset(exp_yaml, exp_setup, group_A)

    trainloader = DataLoader(ft_data_train, batch_size=128, shuffle=True, num_workers=4,
                worker_init_fn=seed_worker,generator=g, persistent_workers=True,
                pin_memory=True)
    
    testloader = DataLoader(ft_data_val, batch_size=128, shuffle=False, num_workers=2,
                    worker_init_fn=seed_worker,generator=g, persistent_workers=True,
                    pin_memory=True)
    

    criterion = nn.CrossEntropyLoss()

    # Load the checkpoint state once (reused across all sparsity levels)
    ckpt_state = torch.load(ckpt_path, map_location=device)

    # -----------------------------
    # Pruning loop
    # -----------------------------
    for s in sparsity_levels_from_setup(exp_setup, SPARSITY):
        s_key = round(float(s), 6)
        if s_key not in exp_setup["Prune_Setups"]:
            raise KeyError(f"No pruning config found for sparsity={s_key}")
        
        prune_cfg = exp_setup["Prune_Setups"][s_key]
        epochs = prune_cfg["Epochs"]
 
        for strategy in STRATEGY: # we only consider FT-AL for pruning
            print(f"\n[🔧] Starting fine-tuning strategy: {strategy} with sparsity={s}.")

            scenario_name = (
                f'{exp_yaml["Scenario_Name"]}_{model_seed}_{round(r, 2)}'
                f'_sparsity={s_key}_{strategy}_ftsize={exp_setup["FT_GroupSize"]}_ftseed={ft_seed}'
            )
            log_dir = "./saved_logs/pruning_final/Performance"
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"training_log_{scenario_name}.csv")

            if not os.path.exists(log_file):
                with open(log_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Scenario", "Epoch",
                        "Train_Loss", "Train_Acc", "Train_Precision", "Train_Recall", "Train_F1",
                        "Test_Loss", "Test_Acc", "Test_Precision", "Test_Recall", "Test_F1"
                    ])

            # ----- Fresh model → prune → finetune setup -----
            print("==> Building fresh model..")
            net = exp_setup["Model_Factory"]().to(device)
            net.load_state_dict(ckpt_state)
            check_pruned_weights(net, exclude_patterns=prune_exclude)

            pruned_net = prune_model_global(model=net, amount=s_key,
                                            exclude_patterns=prune_exclude)
            print(f"[✅] Pruned model for {model_name} with sparsity={s_key}"
                  + (f" (sparing {prune_exclude})" if prune_exclude else ""))
            check_pruned_weights(pruned_net, exclude_patterns=prune_exclude)
            
            pruned_net = setup_finetune(
                model=pruned_net, strategy=strategy, device=device
            )
            check_pruned_weights(pruned_net, exclude_patterns=prune_exclude)

            # Fresh optimizer per strategy/run
            optimizer_class = getattr(optim, prune_cfg["Optimizer_Name"])
            optimizer = optimizer_class(
                filter(lambda p: p.requires_grad, pruned_net.parameters()),
                **prune_cfg["Optimizer_Params"]
            )

            # Fresh scheduler per strategy/run
            scheduler = None
            if prune_cfg["Scheduler_Name"] is not None:
                scheduler_params = prune_cfg["Scheduler_Params"].copy()
                if scheduler_params.get("T_max") == "auto":
                    scheduler_params["T_max"] = epochs

                scheduler_class = getattr(lr_sched, prune_cfg["Scheduler_Name"])
                scheduler = scheduler_class(optimizer, **scheduler_params)

            # Log pre-FT state into strategy log too
            result_pre_ft_eval = evaluate1(pruned_net, testloader, criterion, device)

            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    scenario_name, -1,
                    0, 0, 0, 0, 0,
                    result_pre_ft_eval["test_loss"], result_pre_ft_eval["test_acc"],
                    result_pre_ft_eval["test_precision"], result_pre_ft_eval["test_recall"], result_pre_ft_eval["test_f1"],
                ])

            # ----- Checkpoint directory + best-epoch tracking (mirrors main_ft.py) -----
            save_dir = f'./saved_models/pruning_final/{scenario_name}'
            os.makedirs(save_dir, exist_ok=True)

            best_test_acc = -1.0
            best_ckpt_from = int(epochs * BEST_CKPT_START_FRAC)

            # Save pre-FT checkpoint for this strategy (epoch -1 is consumed by calculate_MI_prune.py)
            temp_net = copy.deepcopy(pruned_net)
            remove_prune_mask(temp_net)
            torch.save(temp_net.state_dict(), f'{save_dir}/epoch_-1.pth')
            del temp_net

            # -----------------------------
            # Training loop
            # -----------------------------
            for epoch in range(start_epoch, start_epoch + epochs):
                print(
                    f"[INFO] model_seed={model_seed}, ft_seed={ft_seed}, "
                    f"sparsity={s_key}, strategy={strategy}, epoch={epoch}"
                )

                train_result = ft_one_epoch(
                    pruned_net, trainloader, optimizer, criterion, epoch, device, strategy
                )

                if scheduler is not None:
                    scheduler.step()

                print(f"Epoch {epoch} | LR = {optimizer.param_groups[0]['lr']}")

                eval_result = evaluate1(pruned_net, testloader, criterion, device)

                with open(log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        scenario_name, epoch,
                        train_result["train_loss"], train_result["train_acc"],
                        train_result["train_precision"], train_result["train_recall"], train_result["train_f1"],
                        eval_result["test_loss"], eval_result["test_acc"],
                        eval_result["test_precision"], eval_result["test_recall"], eval_result["test_f1"],
                    ])

                print(f"Epoch {epoch}: model sparsity checking:")
                check_pruned_weights(pruned_net, exclude_patterns=prune_exclude)

                # ----- Best-epoch checkpoint only (mirrors main_ft.py) -----
                current_acc = eval_result["test_acc"]
                if epoch - start_epoch >= best_ckpt_from and current_acc > best_test_acc:
                    best_test_acc = current_acc
                    temp_net = copy.deepcopy(pruned_net)
                    remove_prune_mask(temp_net)
                    torch.save(temp_net.state_dict(), f'{save_dir}/best_epoch.pth')
                    del temp_net

            # ----- Last-epoch checkpoint, saved once after the loop (mirrors main_ft.py) -----
            temp_net = copy.deepcopy(pruned_net)
            remove_prune_mask(temp_net)
            torch.save(temp_net.state_dict(), f'{save_dir}/epoch_{epoch}.pth')
            del temp_net

            print(
                f"==> Finished sparsity={s_key}, strategy={strategy}. "
                f"Best Test Acc: {best_test_acc:.2f}% (from epoch >= {best_ckpt_from})"
            )

    


def main_prune_deit(model_seed, ft_seed, r, yaml_file_path):
    """
    DeiT 版本的 pruning 主函数,覆盖两种 arch:
      - CIFAR-10 : deit_tiny_patch16_224           (plain,     只有 head)
      - CIFAR-100: deit_tiny_distilled_patch16_224 (distilled, head + head_dist)

    与 CNN 版 main_prune 的差异(逐条对应 DeiT 的接口适配):
      1. setup     : process_experiment_prune_setup_deit —— Model 字段是 dict,用
                     timm.create_model 构建,参数与 build_deit_student 一致
      2. model_dir : saved_models/vanilla/Transformer_Models/(不是 CNN_Models)
      3. torch.load: weights_only=False + 解包 {"model"} / {"state_dict"} 包裹格式
      4. 剪枝范围  : prune_model_global 已内置跳过 patch_embed 的 Conv2d(打 patch 的
                     那一层剪掉会直接破坏 token 化);其余 Linear(qkv/proj/fc1/fc2/
                     head[/head_dist])按全局 |w| 统一阈值剪
      5. 去 mask   : remove_prune_mask_safe —— 旧的 remove_prune_mask 会对没有 mask 的
                     patch_embed.proj 调 prune.remove 而抛 ValueError
      6. 稀疏统计  : check_pruned_weights(..., exclude_patterns=deit_exclude),
                     与剪枝集合同口径
      7. 评估      : evaluate_deit_ft —— 处理 distilled 的双头(按推理口径取
                     (cls+dist)/2),且指标口径与 ft_one_epoch 对齐(0-100 / weighted /
                     per-sample),保证同一行 CSV 的 Train_* 与 Test_* 可比
      8. 落盘      : saved_models/pruning_final/Transformer_Models/{scenario}/,
                     日志 saved_logs/pruning_final/Performance/Transformer_Models/

    其余骨架(sparsity 循环、FT-AL、optimizer/scheduler、epoch_-1 / best_epoch /
    末轮 checkpoint)与 CNN 版 main_prune 完全一致。旧入口 main_prune 保留不动。
    """
    print(device)
    start_epoch = 0

    exp_yaml = process_yaml_file(yaml_file_path)
    # ---- 差异 1:DeiT 的 setup(Model 是 dict,用 timm 构建)----
    exp_setup = process_experiment_prune_setup_deit(exp_yaml)

    # 同 CNN 版:plan 里可选的 Prune_Exclude,叠加在 patch_embed 之上。
    prune_exclude = exp_yaml.get("Prune_Exclude") or None
    deit_exclude = DEIT_PRUNE_EXCLUDE + list(prune_exclude or [])

    print('==> Loading model..')
    model_name = exp_yaml["Model_Name"] + f"_{model_seed}_{round(r,2)}"
    print(f"Model Name is {model_name}.")
    # ---- 差异 2:DeiT 模型在 Transformer_Models 目录 ----
    model_dir = './saved_models/vanilla/Transformer_Models/'
    model_folder = Path(model_dir) / model_name

    ckpt_path, _ = load_best_checkpoint(model_folder)  # find best checkpoint
    if ckpt_path is None:
        raise FileNotFoundError(f"No .pth files found in {model_folder}")

    print('==> Preparing data..')
    g = torch.Generator()
    g.manual_seed(ft_seed)

    group_A = create_or_load_group_A(dataset=exp_setup["Dataset"].train_set, save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
                                     group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"], seed=42, force_rebuild=False)
                                     # always used for evaluation in different case

    # -----------------------------
    # FT dataset preparation
    # -----------------------------
    ft_data_train, ft_data_val = determine_ft_dataset(exp_yaml, exp_setup, group_A)

    trainloader = DataLoader(ft_data_train, batch_size=128, shuffle=True, num_workers=4,
                worker_init_fn=seed_worker, generator=g, persistent_workers=True,
                pin_memory=True)

    testloader = DataLoader(ft_data_val, batch_size=128, shuffle=False, num_workers=2,
                    worker_init_fn=seed_worker, generator=g, persistent_workers=True,
                    pin_memory=True)

    criterion = nn.CrossEntropyLoss()

    # ---- 差异 3:weights_only=False + 兼容包裹格式(与 main_ft_deit 一致)----
    # Load the checkpoint state once (reused across all sparsity levels)
    ckpt_state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt_state, dict) and "model" in ckpt_state:
        ckpt_state = ckpt_state["model"]
    elif isinstance(ckpt_state, dict) and "state_dict" in ckpt_state:
        ckpt_state = ckpt_state["state_dict"]

    # -----------------------------
    # Pruning loop
    # -----------------------------
    for s in sparsity_levels_from_setup(exp_setup, SPARSITY):
        s_key = round(float(s), 6)
        if s_key not in exp_setup["Prune_Setups"]:
            raise KeyError(f"No pruning config found for sparsity={s_key}")

        prune_cfg = exp_setup["Prune_Setups"][s_key]
        epochs = prune_cfg["Epochs"]

        for strategy in STRATEGY:  # we only consider FT-AL for pruning
            print(f"\n[🔧] Starting fine-tuning strategy: {strategy} with sparsity={s}.")

            scenario_name = (
                f'{exp_yaml["Scenario_Name"]}_{model_seed}_{round(r, 2)}'
                f'_sparsity={s_key}_{strategy}_ftsize={exp_setup["FT_GroupSize"]}_ftseed={ft_seed}'
            )
            # ---- 差异 8:DeiT 的日志目录 ----
            log_dir = "./saved_logs/pruning_final/Performance/Transformer_Models"
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"training_log_{scenario_name}.csv")

            if not os.path.exists(log_file):
                with open(log_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Scenario", "Epoch",
                        "Train_Loss", "Train_Acc", "Train_Precision", "Train_Recall", "Train_F1",
                        "Test_Loss", "Test_Acc", "Test_Precision", "Test_Recall", "Test_F1"
                    ])

            # ----- Fresh model -> prune -> finetune setup -----
            print("==> Building fresh DeiT..")
            net = exp_setup["Model_Factory"]().to(device)
            net.load_state_dict(ckpt_state)
            print(f"[✅] Loaded model from {ckpt_path.name}")
            check_pruned_weights(net, exclude_patterns=deit_exclude)

            # ---- 差异 4:prune_model_global 内部跳过 patch_embed 的 Conv2d ----
            pruned_net = prune_model_global(model=net, amount=s_key,
                                            exclude_patterns=prune_exclude)
            print(f"[✅] Pruned model for {model_name} with sparsity={s_key} (patch_embed excluded)")
            check_pruned_weights(pruned_net, exclude_patterns=deit_exclude)

            # setup_finetune 已识别 DeiT 的 .head[/.head_dist];FT-AL 下只是全解冻。
            # deit_finetune_norm_in_ll=False 与 main_ft_deit 保持一致(只对 FT-LL 生效)。
            pruned_net = setup_finetune(
                model=pruned_net, strategy=strategy, device=device,
                deit_finetune_norm_in_ll=False
            )
            check_pruned_weights(pruned_net, exclude_patterns=deit_exclude)

            # Fresh optimizer per strategy/run
            # 注意:剪枝后可训练的是 weight_orig(weight_mask 是 buffer,不进优化器),
            # 所以被剪掉的位置在整个 FT 过程中恒为 0。
            optimizer_class = getattr(optim, prune_cfg["Optimizer_Name"])
            optimizer = optimizer_class(
                filter(lambda p: p.requires_grad, pruned_net.parameters()),
                **prune_cfg["Optimizer_Params"]
            )

            # Fresh scheduler per strategy/run
            scheduler = None
            if prune_cfg["Scheduler_Name"] is not None:
                scheduler_params = prune_cfg["Scheduler_Params"].copy()
                if scheduler_params.get("T_max") == "auto":
                    scheduler_params["T_max"] = epochs

                scheduler_class = getattr(lr_sched, prune_cfg["Scheduler_Name"])
                scheduler = scheduler_class(optimizer, **scheduler_params)

            # ---- 差异 7:用 evaluate_deit_ft ----
            # Log pre-FT state into strategy log too
            result_pre_ft_eval = evaluate_deit_ft(pruned_net, testloader, criterion, device)

            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    scenario_name, -1,
                    0, 0, 0, 0, 0,
                    result_pre_ft_eval["test_loss"], result_pre_ft_eval["test_acc"],
                    result_pre_ft_eval["test_precision"], result_pre_ft_eval["test_recall"], result_pre_ft_eval["test_f1"],
                ])

            # ----- Checkpoint directory + best-epoch tracking (mirrors main_ft.py) -----
            save_dir = f'./saved_models/pruning_final/Transformer_Models/{scenario_name}'
            os.makedirs(save_dir, exist_ok=True)

            best_test_acc = -1.0
            best_ckpt_from = int(epochs * BEST_CKPT_START_FRAC)

            # Save pre-FT checkpoint for this strategy (epoch -1 is consumed by calculate_MI_prune.py)
            # ---- 差异 5:remove_prune_mask_safe,跳过没有 mask 的 patch_embed.proj ----
            temp_net = copy.deepcopy(pruned_net)
            remove_prune_mask_safe(temp_net)
            torch.save(temp_net.state_dict(), f'{save_dir}/epoch_-1.pth')
            del temp_net

            # -----------------------------
            # Training loop
            # -----------------------------
            for epoch in range(start_epoch, start_epoch + epochs):
                print(
                    f"[INFO] model_seed={model_seed}, ft_seed={ft_seed}, "
                    f"sparsity={s_key}, strategy={strategy}, epoch={epoch}"
                )

                # ft_one_epoch 已处理 DeiT distilled 的双头输出((cls+dist)/2)
                train_result = ft_one_epoch(
                    pruned_net, trainloader, optimizer, criterion, epoch, device, strategy
                )

                if scheduler is not None:
                    scheduler.step()

                print(f"Epoch {epoch} | LR = {optimizer.param_groups[0]['lr']}")

                eval_result = evaluate_deit_ft(pruned_net, testloader, criterion, device)

                with open(log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        scenario_name, epoch,
                        train_result["train_loss"], train_result["train_acc"],
                        train_result["train_precision"], train_result["train_recall"], train_result["train_f1"],
                        eval_result["test_loss"], eval_result["test_acc"],
                        eval_result["test_precision"], eval_result["test_recall"], eval_result["test_f1"],
                    ])

                print(f"Epoch {epoch}: model sparsity checking:")
                check_pruned_weights(pruned_net, exclude_patterns=deit_exclude)

                # ----- Best-epoch checkpoint only (mirrors main_ft.py) -----
                current_acc = eval_result["test_acc"]
                if epoch - start_epoch >= best_ckpt_from and current_acc > best_test_acc:
                    best_test_acc = current_acc
                    temp_net = copy.deepcopy(pruned_net)
                    remove_prune_mask_safe(temp_net)
                    torch.save(temp_net.state_dict(), f'{save_dir}/best_epoch.pth')
                    del temp_net

            # ----- Last-epoch checkpoint, saved once after the loop (mirrors main_ft.py) -----
            temp_net = copy.deepcopy(pruned_net)
            remove_prune_mask_safe(temp_net)
            torch.save(temp_net.state_dict(), f'{save_dir}/epoch_{epoch}.pth')
            del temp_net

            print(
                f"==> Finished sparsity={s_key}, strategy={strategy}. "
                f"Best Test Acc: {best_test_acc:.2f}% (from epoch >= {best_ckpt_from})"
            )


# =====================================================
# 4. Entry point
# =====================================================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Folder containing all YAML experiment plans.
    #   CNN  -> prune_plan       -> main_prune
    #   DeiT -> prune_plan_deit  -> main_prune_deit   (set RUN_DEIT=1)
    if RUN_DEIT:
        default_dir = "./saved_exp_plan/prune_plan_deit"
        run_fn = main_prune_deit
    else:
        default_dir = "./saved_exp_plan/prune_plan"
        run_fn = main_prune
    # PRUNE_PLAN_DIR overrides the folder, like KD_PLAN_DIR / FT_PLAN_DIR. The
    # defaults are globbed wholesale by this script AND by
    # calculate_MI_prune.py, and still hold CIFAR-10 work that must not re-run.
    exp_folder = os.environ.get("PRUNE_PLAN_DIR", default_dir)

    for _k in ("PRUNE_PLAN_DIR", "RUN_DEIT", "SEED_START", "SEED_END", "PRUNE_SKIP_EXISTING"):
        if _k in os.environ:
            print(f"[ENV] {_k}={os.environ[_k]}")
    yaml_files = sorted(glob.glob(os.path.join(exp_folder, "*.yaml")))

    if not yaml_files:
            print(f"No YAML files found in {exp_folder}")
    else:
        print(f"Found {len(yaml_files)} experiment plan(s) for {run_fn.__name__}:")
        for f in yaml_files:
            print(" -", f)

    print(f"Seed range: [{SEED_START}, {SEED_END})  ->  {SEED_END - SEED_START} model(s)")

    # Iterate over YAML files and seeds
    for yaml_path in yaml_files:
        print(f"\n========== Starting experiments from {yaml_path} ==========")

        for model_seed in [42]:
            for ft_seed in range(SEED_START, SEED_END):
                if SKIP_EXISTING and prune_run_done(yaml_path, model_seed, 1.0, ft_seed):
                    print(f"[SKIP] {os.path.basename(yaml_path)} ft_seed={ft_seed}: "
                          f"every sparsity already has best_epoch.pth")
                    continue
                print(f"\n>>> Running model seed {model_seed}, ft seed {ft_seed} for {os.path.basename(yaml_path)}")
                set_seed(ft_seed, deterministic=DETERMINISTIC)
                run_fn(model_seed, ft_seed, 1.0, yaml_path)
