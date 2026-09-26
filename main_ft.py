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
from util import create_or_load_group_A, determine_ft_dataset, extract_ft_balanced_train_val_indices, process_experiment_ft_setup_deit, setup_finetune
import torch.optim as optim
import csv
from util import process_yaml_file, process_experiment_ft_setup, train_one_epoch, evaluate1, create_train_subset\
                ,ft_one_epoch, extract_ft_balanced_train_val_indices, load_pickel_dataset, setup_finetune_deit, evaluate_deit_ft, create_or_load_group_B\
                ,prepare_group_subset, load_best_checkpoint, load_last_checkpoint, sanity_check_finetune
import torch.optim.lr_scheduler as lr_sched

# Default AT seeds: 0, 1, 2. Explicit --seeds overrides the environment range.
# SEED_START/SEED_END remain available for splitting work across shells.
SEED_START = int(os.environ.get("SEED_START", 0))
SEED_END   = int(os.environ.get("SEED_END", 3))

# ---- skip guard ---------------------------------------------------------
# A finished (plan, strategy, model_seed, rate, ft_seed) leaves a non-empty
# best_epoch.pth in FT_MODEL_ROOT/<scenario>/. Without this, re-running a plan
# folder retrained every cell and appended a second run's rows to the training
# log that main_ft only creates when absent. FT_SKIP_EXISTING=0 forces a
# rebuild; a run killed mid-training also leaves a best_epoch.pth, so delete
# that folder rather than turning the guard off for the whole sweep.
SKIP_EXISTING = os.environ.get("FT_SKIP_EXISTING", "1") != "0"
FT_MODEL_ROOT = "./saved_models/ft_final"


def ft_scenario_name(scenario, strategy, model_seed, r, ft_seed, ft_group_size):
    """Folder name main_ft / main_ft_deit write to. One source of truth."""
    return (f"{scenario}_{model_seed}_{round(r, 2)}"
            f"_{strategy}_ftsize={ft_group_size}_ftseed={ft_seed}")


def ft_run_done(yaml_path, strategy, model_seed, r, ft_seed):
    """True if that cell already has a non-empty best_epoch.pth.

    Reads the plan with yaml.safe_load rather than process_experiment_ft_setup:
    the latter builds both datasets and the model, which is exactly the cost
    this guard exists to avoid.
    """
    with open(yaml_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    name = ft_scenario_name(data["Scenario_Name"], strategy, model_seed, r, ft_seed,
                            data["FT_Dataset"]["group_size"])
    ckpt = os.path.join(FT_MODEL_ROOT, name, "best_epoch.pth")
    return os.path.isfile(ckpt) and os.path.getsize(ckpt) > 0
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

STRATEGY = ['FT-LL', 'FT-AL', 'RT-AL'] #,
device = 'cuda' if torch.cuda.is_available() else 'cpu'

BEST_CKPT_START_FRAC = 0.3

# ------------------------------------------------
# ---------------- MAIN EXECUTION ----------------
def main_ft(model_seed, ft_seed, r, yaml_file_path): # this method is used to fine-tune the saved CNN model in a ft dataset
    print(device)
    start_epoch = 0

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_ft_setup(exp_yaml) 

    print('==> Loading model..')
    model_name = exp_yaml["Model_Name"] + f"_{model_seed}_{round(r,2)}"
    print(f"Model Name is {model_name}.")
    model_dir = './saved_models/vanilla/CNN_Models'
    model_folder = Path(model_dir)/model_name

    #ckpt_path = load_last_checkpoint(model_folder) # find last checkpoint
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
    # Dataset Preparation
    # -----------------------------
    ft_data_train, ft_data_val = determine_ft_dataset(exp_yaml, exp_setup, group_A)

    trainloader = DataLoader(ft_data_train, batch_size=128, shuffle=True, num_workers=4,
                worker_init_fn=seed_worker,generator=g, persistent_workers=True,
                pin_memory=True)
    
    testloader = DataLoader(ft_data_val, batch_size=128, shuffle=False, num_workers=2,
                worker_init_fn=seed_worker,generator=g, persistent_workers=True,
                pin_memory=True)

    criterion = nn.CrossEntropyLoss()

    # -----------------------------
    # Fine-tuning loop
    # -----------------------------
    for strategy in STRATEGY:
        if strategy not in exp_setup["FT_Setups"]:
            raise KeyError(f"No fine-tune config found for strategy={strategy}")
        
        ft_cfg = exp_setup["FT_Setups"][strategy]
        epochs = ft_cfg["Epochs"]

        print(f"==> Building fresh model for strategy={strategy} ..")
        net = exp_setup["Model_Factory"]().to(device)

        state = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(state)
        print(f"[✅] Loaded model from {ckpt_path.name}")

        net = setup_finetune(model=net, strategy=strategy, device=device)

        scenario_name = (
            f'{exp_yaml["Scenario_Name"]}_{model_seed}_{round(r, 2)}'
            f'_{strategy}_ftsize={exp_setup["FT_GroupSize"]}_ftseed={ft_seed}'
        )
        print(scenario_name)

        log_dir = './saved_logs/ft_final/Performance'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"training_log_{scenario_name}.csv")

        # Make sure log file has header
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Scenario','Epoch', 'Train_Loss', 'Train_Acc', 'Train_Precision', 'Train_Recall','Train_F1'
                                , 'Test_Loss', 'Test_Acc', 'Test_Precision', 'Test_Recall','Test_F1'])
        
        # Pre-FT evaluation
        pre_test_result = evaluate1(net, testloader, criterion, device) # this part gets the metrics before fine tuning.
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([scenario_name, -1, 
                             0, 0, 0, 0, 0, 
                             pre_test_result["test_loss"], pre_test_result["test_acc"], pre_test_result["test_precision"], pre_test_result["test_recall"],pre_test_result["test_f1"], 
                            ])
        
        # Fresh optimizer AFTER setup_finetune
        optimizer_class = getattr(optim, ft_cfg["Optimizer_Name"])
        optimizer = optimizer_class(
            filter(lambda p: p.requires_grad, net.parameters()),
            **ft_cfg["Optimizer_Params"]
        )

        # Fresh scheduler AFTER optimizer creation
        scheduler = None
        if ft_cfg["Scheduler_Name"] is not None:
            scheduler_params = ft_cfg["Scheduler_Params"].copy()
            if scheduler_params.get("T_max") == "auto":
                scheduler_params["T_max"] = epochs

            scheduler_class = getattr(lr_sched, ft_cfg["Scheduler_Name"])
            scheduler = scheduler_class(optimizer, **scheduler_params)

        best_test_acc = -1.0
        best_ckpt_from = int(epochs * BEST_CKPT_START_FRAC)
        os.makedirs(f'./saved_models/ft_final/{scenario_name}', exist_ok=True)

        for epoch in range(start_epoch, start_epoch + epochs): 
            print(f"[INFO] ft_seed={ft_seed}, strategy={strategy}, epoch={epoch}")
            train_result = ft_one_epoch(net, trainloader, optimizer, criterion, epoch, device, strategy)       
            # ---- scheduler step (per epoch) ----
            if scheduler is not None:
                scheduler.step()
            print(f"Epoch {epoch} | LR = {optimizer.param_groups[0]['lr']}")

            test_result = evaluate1(net,testloader,criterion,device)

            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([scenario_name, epoch, 
                                train_result["train_loss"], train_result["train_acc"], train_result["train_precision"], train_result["train_recall"],train_result["train_f1"], 
                                test_result["test_loss"], test_result["test_acc"], test_result["test_precision"], test_result["test_recall"],test_result["test_f1"], 
                                ])

            current_acc = test_result["test_acc"]
            if epoch - start_epoch >= best_ckpt_from and current_acc > best_test_acc:
                best_test_acc = current_acc
                torch.save(net.state_dict(), f'./saved_models/ft_final/{scenario_name}/best_epoch.pth')
        

        torch.save(net.state_dict(), f'./saved_models/ft_final/{scenario_name}/epoch_{epoch}.pth') 

        # At the end of the strategy loop, after all epochs
        sanity_check_finetune(net, ckpt_path, strategy, device)


def main_ft_deit(model_seed, ft_seed, r, yaml_file_path):
    """
    DeiT 版本的 fine-tune 主函数。
    与 CNN 版 main_ft 的差异:
      - 用 process_experiment_ft_setup_deit(工厂构建 DeiT)
      - model_dir 指向 Transformer_Models
      - 评估用 evaluate_deit_ft（DeiT 多头处理 + 与 ft_one_epoch 一致的指标口径），而非 evaluate1
      - torch.load 加 weights_only=False
    其余骨架(strategy 循环、optimizer/scheduler、日志、checkpoint)与 CNN 版一致。
    """
    print(device)
    start_epoch = 0
 
    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_ft_setup_deit(exp_yaml)
 
    print('==> Loading model..')
    model_name = exp_yaml["Model_Name"] + f"_{model_seed}_{round(r, 2)}"
    # ---- 差异:DeiT 模型在 Transformer_Models 目录 ----
    model_dir = './saved_models/vanilla/Transformer_Models/'
    model_folder = Path(model_dir) / model_name
 
    ckpt_path, _ = load_best_checkpoint(model_folder)
    if ckpt_path is None:
        raise FileNotFoundError(f"No .pth files found in {model_folder}")
 
    print('==> Preparing data..')
    g = torch.Generator()
    g.manual_seed(ft_seed)
 
    group_A = create_or_load_group_A(
        dataset=exp_setup["Dataset"].train_set,
        save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
        group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"],
        seed=42, force_rebuild=False,
    )
 
    ft_data_train, ft_data_val = determine_ft_dataset(exp_yaml, exp_setup, group_A)
 
    trainloader = DataLoader(ft_data_train, batch_size=128, shuffle=True, num_workers=4,
                             worker_init_fn=seed_worker, generator=g, persistent_workers=True,
                             pin_memory=True)
    testloader = DataLoader(ft_data_val, batch_size=128, shuffle=False, num_workers=4,
                            worker_init_fn=seed_worker, generator=g, persistent_workers=True,
                            pin_memory=True)
 
    criterion = nn.CrossEntropyLoss()
 
    # -----------------------------
    # Fine-tuning loop(每个 strategy 一轮)
    # -----------------------------
    for strategy in STRATEGY:
        if strategy not in exp_setup["FT_Setups"]:
            raise KeyError(f"No fine-tune config found for strategy={strategy}")
 
        ft_cfg = exp_setup["FT_Setups"][strategy]
        epochs = ft_cfg["Epochs"]
 
        print(f"==> Building fresh DeiT for strategy={strategy} ..")
        net = exp_setup["Model_Factory"]().to(device)
 
        # ---- 差异:weights_only=False + 兼容包裹格式 ----
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        net.load_state_dict(state)
        print(f"[✅] Loaded model from {ckpt_path.name}")
 
        # 配置 fine-tune 策略(默认 deit_finetune_norm_in_ll=False,严格 FT-LL)
        net = setup_finetune(model=net, strategy=strategy, device=device,
                             deit_finetune_norm_in_ll=False)
 
        scenario_name = (
            f'{exp_yaml["Scenario_Name"]}_{model_seed}_{round(r, 2)}'
            f'_{strategy}_ftsize={exp_setup["FT_GroupSize"]}_ftseed={ft_seed}'
        )
        print(scenario_name)
 
        log_dir = './saved_logs/ft_final/Performance'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"training_log_{scenario_name}.csv")
 
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Scenario', 'Epoch', 'Train_Loss', 'Train_Acc', 'Train_Precision', 'Train_Recall', 'Train_F1',
                                 'Test_Loss', 'Test_Acc', 'Test_Precision', 'Test_Recall', 'Test_F1'])
 
        # ---- Pre-FT evaluation(差异:用 evaluate_deit_ft)----
        pre_test_result = evaluate_deit_ft(net, testloader, criterion, device)
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([scenario_name, -1,
                             0, 0, 0, 0, 0,
                             pre_test_result["test_loss"], pre_test_result["test_acc"],
                             pre_test_result["test_precision"], pre_test_result["test_recall"],
                             pre_test_result["test_f1"]])
 
        # Fresh optimizer AFTER setup_finetune(只收集可训练参数,FT-LL 关键)
        optimizer_class = getattr(optim, ft_cfg["Optimizer_Name"])
        optimizer = optimizer_class(
            filter(lambda p: p.requires_grad, net.parameters()),
            **ft_cfg["Optimizer_Params"],
        )
 
        # Fresh scheduler
        scheduler = None
        if ft_cfg["Scheduler_Name"] is not None:
            scheduler_params = ft_cfg["Scheduler_Params"].copy()
            if scheduler_params.get("T_max") == "auto":
                scheduler_params["T_max"] = epochs
            scheduler_class = getattr(lr_sched, ft_cfg["Scheduler_Name"])
            scheduler = scheduler_class(optimizer, **scheduler_params)
 
        best_test_acc = -1.0
        best_ckpt_from = int(epochs * BEST_CKPT_START_FRAC)
        save_dir = f'./saved_models/ft_final/{scenario_name}'
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(start_epoch, start_epoch + epochs):
            print(f"[INFO] ft_seed={ft_seed}, strategy={strategy}, epoch={epoch}")
            # ft_one_epoch 新版:freeze_backbone_norm 默认 True(FT-LL 严格)
            train_result = ft_one_epoch(net, trainloader, optimizer, criterion, epoch, device, strategy)
 
            if scheduler is not None:
                scheduler.step()
            print(f"Epoch {epoch} | LR = {optimizer.param_groups[0]['lr']}")
 
            # ---- 每 epoch 评估(差异:用 evaluate_deit_ft)----
            test_result = evaluate_deit_ft(net, testloader, criterion, device)
 
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([scenario_name, epoch,
                                 train_result["train_loss"], train_result["train_acc"],
                                 train_result["train_precision"], train_result["train_recall"], train_result["train_f1"],
                                 test_result["test_loss"], test_result["test_acc"],
                                 test_result["test_precision"], test_result["test_recall"], test_result["test_f1"]])
 
            # Mirror the CNN path: keep the best checkpoint (from the same
            # BEST_CKPT_START_FRAC window) plus the final one, in ft_final.
            # Before 2026-09-21 this wrote EVERY epoch into
            # ft_vanilla/Transformer_Models/ and never produced best_epoch.pth,
            # which calculate_MI_ft's load_best_checkpoint needs.
            current_acc = test_result["test_acc"]
            if epoch - start_epoch >= best_ckpt_from and current_acc > best_test_acc:
                best_test_acc = current_acc
                torch.save(net.state_dict(), f'{save_dir}/best_epoch.pth')
 
        torch.save(net.state_dict(), f'{save_dir}/epoch_{epoch}.pth')

        # strategy 结束后做 sanity check(开关与 setup_finetune 一致:False)
        sanity_check_finetune(net, ckpt_path, strategy, device,
                              deit_finetune_norm_in_ll=False) 


# =====================================================
# 4. Entry point
# =====================================================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Folder containing all YAML experiment plans. FT_PLAN_DIR overrides it, the
    # same way KD_PLAN_DIR does for main_kd.py: ft_plan/ is globbed wholesale by
    # this script AND by calculate_MI_ft.py, so a dedicated folder is the only
    # way to run one batch without touching the CIFAR-10 work already in there.
    #
    # RUN_DEIT=1 switches to main_ft_deit, mirroring main_prune.py's convention.
    # A DeiT plan CANNOT go through main_ft: its `Model` is a dict and
    # process_experiment_ft_setup would reject it.
    RUN_DEIT = os.environ.get("RUN_DEIT", "0") == "1"
    default_dir = "./saved_exp_plan/ft_plan_deit" if RUN_DEIT else "./saved_exp_plan/ft_plan"
    exp_folder = os.environ.get("FT_PLAN_DIR", default_dir)
    run_fn = main_ft_deit if RUN_DEIT else main_ft
    yaml_files = sorted(glob.glob(os.path.join(exp_folder, "*.yaml")))

    for _k in ("FT_PLAN_DIR", "RUN_DEIT", "SEED_START", "SEED_END", "FT_SKIP_EXISTING"):
        if _k in os.environ:
            print(f"[ENV] {_k}={os.environ[_k]}")
    if not yaml_files:
        print(f"No YAML files found in {exp_folder}")
    else:
        print(f"Found {len(yaml_files)} experiment plan(s) in {exp_folder}:")
        for f in yaml_files:
            print(" -", f)

    print(f"Driver: {run_fn.__name__}   strategies: {STRATEGY}")
    print(f"Seed range: [{SEED_START}, {SEED_END})  ->  {SEED_END - SEED_START} model(s)")
    _planned = len(yaml_files) * (SEED_END - SEED_START) * len(STRATEGY)
    _todo = sum(
        1
        for y in yaml_files
        for s in range(SEED_START, SEED_END)
        for st in STRATEGY
        if not (SKIP_EXISTING and ft_run_done(y, st, 42, 1.0, s))
    )
    print(f"TOTAL: {_planned} cell(s) = {len(yaml_files)} plan(s) x "
          f"{SEED_END - SEED_START} seed(s) x {len(STRATEGY)} strategy(ies)  ->  "
          f"{_planned - _todo} already trained, {_todo} to train")

    # Iterate over YAML files and seeds
    for yaml_path in yaml_files:
        print(f"\n========== Starting experiments from {yaml_path} ==========")

        for model_seed in [42]:
            for ft_seed in range(SEED_START, SEED_END):
                if SKIP_EXISTING and all(
                        ft_run_done(yaml_path, st, model_seed, 1.0, ft_seed)
                        for st in STRATEGY):
                    print(f"[SKIP] {os.path.basename(yaml_path)} ft_seed={ft_seed}: "
                          f"all {len(STRATEGY)} strategy(ies) already trained")
                    continue
                print(f"\n>>> Running model seed {model_seed}, ft seed {ft_seed} "
                      f"for {os.path.basename(yaml_path)}")
                set_seed(ft_seed, deterministic=DETERMINISTIC)
                run_fn(model_seed, ft_seed, 1.0, yaml_path)
