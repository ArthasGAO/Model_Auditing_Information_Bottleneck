import os
import glob
# =====================================================
# 0. SETTINGS
# =====================================================
# Bitwise reproducibility vs. training speed.
#   True  -> rerunning a seed reproduces the exact same weights. Costs ~1.4x
#            wall-clock on CNNs (~1.25x on DeiT): cuDNN cannot use its fastest
#            kernels and cannot autotune for the input shape.
#   False -> the seed still fully determines weight initialisation and data
#            order, so different seeds keep producing distinct, comparable
#            models. Only rerun-identity is given up; the saved .pth remains
#            the record of any individual model.
DETERMINISTIC = False

if DETERMINISTIC:
    # cuBLAS needs a fixed workspace, chosen before torch is imported.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Skip "best checkpoint" writes during the opening stretch of training. Measured
# across 612 completed runs in saved_logs/: the final best never landed in the
# first 40% of epochs, while those epochs produced ~36% of the writes -- and one
# write is a full state_dict (60 MB for VGG16).
BEST_CKPT_START_FRAC = 0.7

# Half-open seed range [SEED_START, SEED_END) this process is responsible for.
# Both are overridable from the environment so that two shells can split the
# sweep without a second copy of this file -- copies drift silently, and a
# checkpoint trained under a stale setting is indistinguishable from a good one.
#
# Split across two windows:
#   PowerShell   $env:SEED_START=42; $env:SEED_END=67; python main_train_nega.py
#                $env:SEED_START=67; $env:SEED_END=92; python main_train_nega.py
#   bash         SEED_START=42 SEED_END=67 python main_train_nega.py
#                SEED_START=67 SEED_END=92 python main_train_nega.py
#
# Two is the limit: 3+ concurrent processes measured no extra throughput, since
# the bottleneck is kernel-launch overhead rather than compute. Running with no
# environment variables set keeps the single-process 42..91 behaviour.
#
# These are read once at import. A value left over in the shell from an earlier
# sweep silently wins over the defaults, so check the "Seed range:" line the
# entry point prints before walking away from a launch.
SEED_START = int(os.environ.get("SEED_START", 42))
SEED_END   = int(os.environ.get("SEED_END", 92))
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import csv
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
import torchvision
import torchvision.transforms as transforms
from util import evaluate2, process_yaml_file, process_experiment_setup, train_one_epoch, evaluate1, create_train_subset, \
        create_or_load_group_A, create_or_load_group_B, prepare_group_subset, train_one_epoch_mix, process_experiment_setup_deit, \
        train_one_epoch_deit, evaluate_deit, save_checkpoint
from torchvision.transforms import v2
from torch.utils.data.dataloader import default_collate
from Model.DeiT import DistillationLoss

# =====================================================
# 1. Global setup
# =====================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

mixup_cutmix = v2.RandomChoice([
    v2.MixUp(num_classes=100),
    v2.CutMix(num_classes=100),
])

def collate_fn(batch):
    inputs, targets = default_collate(batch)   # inputs: [B,3,32,32], targets: [B] ints
    inputs, targets = mixup_cutmix(inputs, targets)
    return inputs, targets

# =====================================================
# 3. Main training logic
# =====================================================
def main(seed, r, yaml_file_path): # this method is made for CNN normal training.
    print(device)
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    g = torch.Generator()
    g.manual_seed(seed)

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_setup(exp_yaml) 

    scenario_name = exp_yaml["Scenario_Name"] + f"_{seed}_{r}"

    log_dir = './saved_logs/vanilla/Performance'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{scenario_name}.csv")

    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Scenario','Epoch', 'Train_Loss', 'Train_Acc', 'Train_Precision', 'Train_Recall','Train_F1'
                             , 'Test_Loss', 'Test_Acc', 'Test_Precision', 'Test_Recall','Test_F1'])
            
    print('==> Preparing data..')

    train_set = exp_setup["Dataset"].train_set
    test_set =  exp_setup["Dataset"].test_set

    # 25000 vs. 250000 （50000）
    group_A = create_or_load_group_A(dataset=train_set, save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
                                                  group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"], seed=42, force_rebuild=False)
    train_subset1 =  exp_setup["Dataset"].subset("train", group_A, clean=False) # with augmentation
    
    trainloader = DataLoader(train_subset1, batch_size=128, shuffle=True, num_workers=4,
                    worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                    pin_memory=True)

    testloader = DataLoader(test_set,batch_size=128,shuffle=False,num_workers=4,
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)

    print('==> Building model..')
    # load experiment information
    net = exp_setup["Model"].to(device) # load model 
    criterion = nn.CrossEntropyLoss() 
    optimizer = exp_setup["Optimizer"]
    scheduler = exp_setup["Scheduler"]

    # Training loop
    best_test_acc = -1.0
    best_ckpt_from = int(exp_yaml["Epochs"] * BEST_CKPT_START_FRAC)
    os.makedirs(f'./saved_models/vanilla/{scenario_name}', exist_ok=True)

    for epoch in range(start_epoch, start_epoch + exp_yaml["Epochs"]): 
        print(f"This is the {seed} round, {epoch} epoch!")
        train_result = train_one_epoch(net, trainloader, optimizer, criterion, epoch, device)       
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
            torch.save(net.state_dict(), f'./saved_models/vanilla/{scenario_name}/best_epoch.pth')
    
    torch.save(net.state_dict(), f'./saved_models/vanilla/{scenario_name}/epoch_{epoch}.pth') 


def main_mix(seed, r, yaml_file_path): # Use mixup/cutmix for CIFAR-100 small sample case
    print(device)
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    g = torch.Generator()
    g.manual_seed(seed)

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_setup(exp_yaml) 

    scenario_name = f"{exp_yaml['Dataset']}_{exp_yaml['Model']}_{seed}_{r}_Mix" 

    log_dir = './saved_logs/normal_nega/performance'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{scenario_name}.csv")

    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Scenario','Epoch', 'Train_Loss', 'Train_Acc', 'Train_Precision', 'Train_Recall','Train_F1'
                             , 'Test_Loss', 'Test_Acc', 'Test_Precision', 'Test_Recall','Test_F1'])
            
    print('==> Preparing data..')

    train_set = exp_setup["Dataset"].train_set
    test_set =  exp_setup["Dataset"].test_set

    # 25000 vs. 250000 （50000）
    train_subset = prepare_group_subset(train_set, r, f"./Indices/{exp_yaml['Dataset']}")
    trainloader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=4,
                    worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                    pin_memory=True, collate_fn=collate_fn)

    testloader = DataLoader(test_set,batch_size=128,shuffle=False,num_workers=4,
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)

    print('==> Building model..')
    # load experiment information
    net = exp_setup["Model"].to(device) # load model 
    #criterion = nn.CrossEntropyLoss()
    import torch.nn.functional as F
    def soft_target_cross_entropy(logits, soft_targets):
        # soft_targets: [B, C] floats, sum to 1 across C
        log_probs = F.log_softmax(logits, dim=1)
        return -(soft_targets * log_probs).sum(dim=1).mean()
    train_criterion = soft_target_cross_entropy
    eval_criterion = nn.CrossEntropyLoss()
    
    optimizer = exp_setup["Optimizer"]
    scheduler = exp_setup["Scheduler"]
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Training loop
    for epoch in range(start_epoch, start_epoch + exp_yaml["Epochs"]): 
        print(f"This is the {seed} round, {epoch} epoch!")
        train_result = train_one_epoch_mix(net, trainloader, optimizer, train_criterion, epoch, device)       
        # ---- scheduler step (per epoch) ----
        if scheduler is not None:
            scheduler.step()
        print(f"Epoch {epoch} | LR = {optimizer.param_groups[0]['lr']}")

        test_result = evaluate1(net,testloader,eval_criterion,device)

        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([scenario_name, epoch, 
                             train_result["train_loss"], train_result["train_acc"], train_result["train_precision"], train_result["train_recall"],train_result["train_f1"], 
                             test_result["test_loss"], test_result["test_acc"], test_result["test_precision"], test_result["test_recall"],test_result["test_f1"], 
                            ])
            
    os.makedirs(f'./saved_models/normal_nega/{scenario_name}', exist_ok=True)
    torch.save(net.state_dict(), f'./saved_models/normal_nega/{scenario_name}/epoch_{epoch}.pth') 


def main_overlap(seed, r, yaml_file_path): # this method is made for CNN normal training with different overlapping rate of training data.
    print(device)
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    g = torch.Generator()
    g.manual_seed(seed)

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_setup(exp_yaml) 

    scenario_name = exp_yaml["Scenario_Name"] + f"_{seed}_{round(r, 2)}"

    log_dir = './saved_logs/vanilla/Performance'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{scenario_name}.csv")

    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Scenario','Epoch', 'Train_Loss', 'Train_Acc', 'Train_Precision', 'Train_Recall','Train_F1'
                             , 'Test_Loss', 'Test_Acc', 'Test_Precision', 'Test_Recall','Test_F1'])
            
    print('==> Preparing data..')

    train_set = exp_setup["Dataset"].train_set
    test_set =  exp_setup["Dataset"].test_set

    # 25000 vs. 250000 （50000）exp_setup["GroupSize"]
    group_A = create_or_load_group_A(dataset=train_set, save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
                                                  group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"], seed=42, force_rebuild=False)

    group_B = create_or_load_group_B(dataset=train_set, save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/', group_A_indices=group_A,
                                     group_size = exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"], 
                                     overlap_rate=r, seed=42, force_rebuild=False)
    
    train_subset1 =  exp_setup["Dataset"].subset("train", group_B, clean=False) # with augmentation
    
    trainloader = DataLoader(train_subset1, batch_size=128, shuffle=True, num_workers=4,
                    worker_init_fn=seed_worker, generator=g, persistent_workers=True, 
                    pin_memory=True)

    testloader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2,
                    persistent_workers=True, pin_memory=True)


    print('==> Building model..')
    # load experiment information
    net = exp_setup["Model"].to(device) # load model 
    criterion = nn.CrossEntropyLoss()
    optimizer = exp_setup["Optimizer"]
    scheduler = exp_setup["Scheduler"]

    # Training loop
    best_test_acc = -1.0
    best_ckpt_from = int(exp_yaml["Epochs"] * BEST_CKPT_START_FRAC)
    os.makedirs(f'./saved_models/vanilla/{scenario_name}', exist_ok=True)

    for epoch in range(start_epoch, start_epoch + exp_yaml["Epochs"]): 
        print(f"This is the {seed} round, {epoch} epoch!")
        train_result = train_one_epoch(net, trainloader, optimizer, criterion, epoch, device)       
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
            torch.save(net.state_dict(), f'./saved_models/vanilla/{scenario_name}/best_epoch.pth') 

    torch.save(net.state_dict(), f'./saved_models/vanilla/{scenario_name}/epoch_{epoch}.pth')


def main_deit(seed, r, yaml_file_path): # this method is made for the DeiT Model(Transformer) normal training.
    print(device)
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    g = torch.Generator()
    g.manual_seed(seed)

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_setup_deit(exp_yaml) 

    scenario_name = exp_yaml["Scenario_Name"] + f"_{seed}_{round(r, 2)}"

    log_dir = './saved_logs/vanilla/Performance'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{scenario_name}.csv")

    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Scenario','Epoch', 'Train_Loss', 'Train_Acc', 'Train_Precision', 'Train_Recall','Train_F1'
                             , 'Test_Loss', 'Test_Acc', 'Test_Precision', 'Test_Recall','Test_F1'])

    print('==> Preparing data..')

    train_set = exp_setup["Dataset"].train_set
    test_set =  exp_setup["Dataset"].test_set

    # Prepare class balanced training subset
    group_A = create_or_load_group_A(dataset=train_set, save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
                                                  group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"], seed=42, force_rebuild=False)
    train_subset1 =  exp_setup["Dataset"].subset("train", group_A, clean=False)
    
    trainloader = DataLoader(train_subset1, batch_size=128, shuffle=True, num_workers=4,
                    worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                    pin_memory=True)

    testloader = DataLoader(test_set,batch_size=128,shuffle=False,num_workers=4,
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)

    print('==> Building model..')
    # load experiment information
    student_net = exp_setup["Student Model"].to(device) # load model
    #teacher_net = exp_setup["Teacher Model"].to(device)

    optimizer = exp_setup["Optimizer"]
    scheduler = exp_setup["Scheduler"]
    mixup_fn = exp_setup.get("MixupFn", None)

    # load distillaiton information
    criterion = exp_setup["Distillation"]
    eval_criterion = nn.CrossEntropyLoss()

    # Training loop
    best_test_acc = -1.0
    best_ckpt_from = int(exp_yaml["Epochs"] * BEST_CKPT_START_FRAC)
    os.makedirs(f'./saved_models/vanilla/Transformer_Models/{scenario_name}', exist_ok=True)

    for epoch in range(start_epoch, start_epoch + exp_yaml["Epochs"]): 
        print(f"This is the {seed} round, {epoch} epoch!")
        train_result = train_one_epoch_deit(student_net, trainloader, optimizer, criterion, epoch, device, mixup_fn=mixup_fn)      
        # ---- scheduler step (per epoch) ----
        if scheduler is not None:
            scheduler.step()
        print(f"Epoch {epoch} | LR = {optimizer.param_groups[0]['lr']}")

        test_result = evaluate_deit(student_net,testloader,eval_criterion,device)

        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([scenario_name, epoch, 
                             train_result["train_loss"], train_result["train_acc"], train_result["train_precision"], train_result["train_recall"],train_result["train_f1"], 
                             test_result["test_loss"], test_result["test_acc"], test_result["test_precision"], test_result["test_recall"],test_result["test_f1"], 
                            ])
            
        current_acc = test_result["test_acc"]
        if epoch - start_epoch >= best_ckpt_from and current_acc > best_test_acc:
            best_test_acc = current_acc
            torch.save(student_net.state_dict(), f'./saved_models/vanilla/Transformer_Models/{scenario_name}/best_epoch.pth') 
            
    torch.save(student_net.state_dict(), f'./saved_models/vanilla/Transformer_Models/{scenario_name}/epoch_{epoch}.pth')


def main_deit_overlap(seed, r, yaml_file_path):
    """
    DeiT (Transformer) 训练,支持不同 overlapping rate。
    与 main_overlap 对齐:用 GroupB(相对 GroupA 有 overlap_rate r)作为训练子集。
    """
    print(device)
    start_epoch = 0

    g = torch.Generator()
    g.manual_seed(seed)

    exp_yaml = process_yaml_file(yaml_file_path)

    # Seed-paired teacher. A plan may template teacher_ckpt with {seed} so that
    # every student distils from the CNN trained under its own seed. All of those
    # teachers saw the identical subset -- create_or_load_group_B below is called
    # with seed=42 fixed, so group_B depends only on (group_size, overlap_rate)
    # -- so pairing changes no data, it only stops the whole population from
    # inheriting one shared decision boundary, which would correlate the models
    # and shrink the H0 covariance they are pooled into.
    teacher_ckpt = exp_yaml.get("Distillation", {}).get("teacher_ckpt", "")
    if "{seed}" in teacher_ckpt:
        exp_yaml["Distillation"]["teacher_ckpt"] = teacher_ckpt.format(seed=seed)
        print(f"teacher: {exp_yaml['Distillation']['teacher_ckpt']}")

    exp_setup = process_experiment_setup_deit(exp_yaml)

    # 命名与 main_overlap 对齐:用 round(r, 2) 保证文件名稳定
    scenario_name = exp_yaml["Scenario_Name"] + f"_{seed}_{round(r, 2)}"

    log_dir = './saved_logs/vanilla/Performance/Transformer_Models'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{scenario_name}.csv")

    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Scenario', 'Epoch', 'Train_Loss', 'Train_Acc', 'Train_Precision', 'Train_Recall', 'Train_F1',
                             'Test_Loss', 'Test_Acc', 'Test_Precision', 'Test_Recall', 'Test_F1'])

    print('==> Preparing data..')

    train_set = exp_setup["Dataset"].train_set
    test_set = exp_setup["Dataset"].test_set

    # ---- GroupA + GroupB(带 overlap rate),与 main_overlap 完全一致 ----
    group_A = create_or_load_group_A(
        dataset=train_set, save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
        group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"],
        seed=42, force_rebuild=False,
    )

    group_B = create_or_load_group_B(
        dataset=train_set, save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
        group_A_indices=group_A, group_size=exp_setup["GroupSize"],
        num_classes=exp_setup["NumClasses"], overlap_rate=r,
        seed=42, force_rebuild=False,
    )

    # 用 GroupB 作为训练子集(带增强)
    train_subset1 = exp_setup["Dataset"].subset("train", group_B, clean=False)

    trainloader = DataLoader(train_subset1, batch_size=128, shuffle=True, num_workers=8,
                             worker_init_fn=seed_worker, generator=g, persistent_workers=True,
                             pin_memory=True)

    testloader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2,
                            persistent_workers=True, pin_memory=True)


    print('==> Building model..')
    student_net = exp_setup["Student Model"].to(device)

    optimizer = exp_setup["Optimizer"]
    scheduler = exp_setup["Scheduler"]
    mixup_fn = exp_setup.get("MixupFn", None)

    # ---- 蒸馏 criterion ----
    criterion = exp_setup["Distillation"]
    eval_criterion = nn.CrossEntropyLoss()

    # ---- 关键修复:确保 criterion 内部的 teacher 在正确 device ----
    if getattr(criterion, "teacher", None) is not None:
        criterion.teacher.to(device)
        criterion.teacher.eval()
        for p in criterion.teacher.parameters():
            p.requires_grad = False

    # ---- 设备 sanity 打印,确认 student / teacher 同设备 ----
    print("student device:", next(student_net.parameters()).device)
    if getattr(criterion, "teacher", None) is not None:
        print("teacher device:", next(criterion.teacher.parameters()).device)

    # Training loop
    best_test_acc = -1.0
    best_ckpt_from = int(exp_yaml["Epochs"] * BEST_CKPT_START_FRAC)
    os.makedirs(f'./saved_models/vanilla/Transformer_Models/{scenario_name}', exist_ok=True)

    for epoch in range(start_epoch, start_epoch + exp_yaml["Epochs"]):
        print(f"This is the {seed} round, {epoch} epoch!")
        train_result = train_one_epoch_deit(student_net, trainloader, optimizer, criterion, epoch, device, mixup_fn=mixup_fn)

        if scheduler is not None:
            scheduler.step()
        print(f"Epoch {epoch} | LR = {optimizer.param_groups[0]['lr']}")

        test_result = evaluate_deit(student_net, testloader, eval_criterion, device)

        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([scenario_name, epoch,
                             train_result["train_loss"], train_result["train_acc"], train_result["train_precision"], train_result["train_recall"], train_result["train_f1"],
                             test_result["test_loss"], test_result["test_acc"], test_result["test_precision"], test_result["test_recall"], test_result["test_f1"],
                             ])

        current_acc = test_result["test_acc"]
        if epoch - start_epoch >= best_ckpt_from and current_acc > best_test_acc:
            best_test_acc = current_acc
            torch.save(student_net.state_dict(), f'./saved_models/vanilla/Transformer_Models/{scenario_name}/best_epoch.pth')

    torch.save(student_net.state_dict(), f'./saved_models/vanilla/Transformer_Models/{scenario_name}/epoch_{epoch}.pth')



def main_test(seed, r, yaml_file_path): # this method is made for CNN normal training test part.
    print(device)
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    g = torch.Generator()
    g.manual_seed(seed)

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_setup(exp_yaml) 

    scenario_name = exp_yaml["Scenario_Name"] + f"_{seed}_{r}_test"

    log_dir = './saved_logs/vanilla/Performance'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{scenario_name}.csv")

    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Scenario','Epoch', 'Train_Loss', 'Train_Acc', 'Train_Precision', 'Train_Recall','Train_F1'
                             , 'Test_Loss', 'Test_Acc', 'Test_Precision', 'Test_Recall','Test_F1'])
            
    print('==> Preparing data..')

    train_set = exp_setup["Dataset"].train_set
    test_set =  exp_setup["Dataset"].test_aug_set

    # 25000 vs. 250000 （50000）
    group_A = create_or_load_group_A(dataset=train_set, save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
                                                  group_size=10000, num_classes=exp_setup["NumClasses"], seed=42, force_rebuild=False)
    test_subset1 =  exp_setup["Dataset"].subset("train", group_A, clean=True) # with augmentation
    
    trainloader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=4,
                    worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                    pin_memory=True)

    testloader = DataLoader(test_subset1,batch_size=128,shuffle=False,num_workers=4,
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)

    print('==> Building model..')
    # load experiment information
    net = exp_setup["Model"].to(device) # load model 
    criterion = nn.CrossEntropyLoss() 
    optimizer = exp_setup["Optimizer"]
    scheduler = exp_setup["Scheduler"]

    # Training loop
    best_test_acc = -1.0
    best_ckpt_from = int(exp_yaml["Epochs"] * BEST_CKPT_START_FRAC)
    os.makedirs(f'./saved_models/vanilla/{scenario_name}', exist_ok=True)

    for epoch in range(start_epoch, start_epoch + exp_yaml["Epochs"]): 
        print(f"This is the {seed} round, {epoch} epoch!")
        train_result = train_one_epoch(net, trainloader, optimizer, criterion, epoch, device)       
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
            torch.save(net.state_dict(), f'./saved_models/vanilla/{scenario_name}/best_epoch.pth')
    
    torch.save(net.state_dict(), f'./saved_models/vanilla/{scenario_name}/epoch_{epoch}.pth') 

# =====================================================
# 4. Entry point
# =====================================================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Folder containing all YAML experiment plans
    exp_folder = "./saved_exp_plan/train_plan"
    yaml_files = sorted(glob.glob(os.path.join(exp_folder, "*.yaml")))

    if not yaml_files:
            print(f"No YAML files found in {exp_folder}")
    else:
        print(f"Found {len(yaml_files)} experiment plan(s):")
        for f in yaml_files:
            print(" -", f)

    # Echo the range up front: with two shells open this is how you tell them
    # apart, and it catches a mistyped SEED_START before hours of training.
    print(f"Seed range: [{SEED_START}, {SEED_END})  ->  {SEED_END - SEED_START} model(s)")

    # Fail on a missing teacher checkpoint now rather than partway through a
    # multi-hour sweep. Plans without a {seed}-templated teacher are skipped.
    for yaml_path in yaml_files:
        tmpl = process_yaml_file(yaml_path).get("Distillation", {}).get("teacher_ckpt", "")
        if "{seed}" not in tmpl:
            continue
        absent = [s for s in range(SEED_START, SEED_END)
                  if not os.path.exists(tmpl.format(seed=s))]
        if absent:
            raise FileNotFoundError(
                f"{os.path.basename(yaml_path)}: no teacher checkpoint for seed(s) {absent}"
            )
        print(f"  teachers OK: {SEED_END - SEED_START} checkpoints for {os.path.basename(yaml_path)}")

    # Iterate over YAML files and seeds
    for yaml_path in yaml_files:
        print(f"\n========== Starting experiments from {yaml_path} ==========")

        for seed in range(SEED_START, SEED_END):
            for rate in [0.0]:#np.linspace(0,1,11): #overlapping rate: [0, 0.1, 0.2, 0.3,...,0.9, 1]
                print(f"\n>>> Running seed {seed} for {os.path.basename(yaml_path)}")
                set_seed(seed, deterministic=DETERMINISTIC)

                #main_overlap(seed, rate, yaml_path)
                main_deit_overlap(seed, rate, yaml_path)
                #main_test(seed, rate, yaml_path)
        '''for seed in range(42, 45): 
            print(f"\n>>> Running seed {seed} for {os.path.basename(yaml_path)}")
            set_seed(seed, deterministic=DETERMINISTIC)

            #main_deit(seed, 1.0, yaml_path)    
            main(seed, 1.0, yaml_path)
            #main_epoch(seed, 1.0, yaml_path)'''
            
            #
        '''for num in range(5000, 25001, 5000):
            main(seed, num, yaml_path)'''
        '''for rate in np.linspace(0,1,11): #overlapping rate: [0, 0.1, 0.2, 0.3,...,0.9, 1]
            main_overlap(seed, rate, yaml_path)
            break'''