import os
import glob
import sys
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # needed for full CUDA determinism

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
import yaml
from util import process_yaml_file, process_experiment_setup, train_one_epoch, evaluate1, create_train_subset, \
        create_or_load_group_A, create_or_load_group_B, prepare_group_subset, train_one_epoch_mix, \
        process_experiment_kd_setup, train_one_epoch_kd, build_warmup_cosine_scheduler
from torchvision.transforms import v2
from torch.utils.data.dataloader import default_collate
from Model.DeiT import DistillationLoss
import torch.optim.lr_scheduler as lr_sched


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


# =====================================================
# 2. Skip guard
# =====================================================
# A finished (scenario, method, seed, rate) leaves a non-empty best_epoch.pth
# in <KD_MODEL_ROOT>/<scenario>_<method>_<seed>_<rate>/. Before this
# guard existed, re-running a plan folder retrained every cell from scratch and
# overwrote both the checkpoint and the training log, so a partially finished
# sweep could not be resumed and finished rows had to be kept out of the plan
# folder by hand.
#
# Two levels, because they save different amounts of work:
#   * __main__ skips a (plan, seed, rate) whose methods are ALL done, which
#     avoids building the dataset and the group_A / group_B subsets;
#   * the per-method loop in main() skips one method, which avoids building the
#     distiller - that step loads the teacher checkpoint and, for a DeiT cell,
#     creates two timm backbones.
#
# KD_SKIP_EXISTING=0 forces a rebuild of everything. Note that a run killed
# mid-training also leaves a best_epoch.pth (it is written whenever test acc
# improves), so that cell looks finished: delete its folder rather than turning
# the guard off for the whole sweep.
SKIP_EXISTING = os.environ.get("KD_SKIP_EXISTING", "1") != "0"
# Output tree. Moved off kd_vanilla on 2026-09-21: kd_final holds the
# bigger-teacher matrix; the old tree keeps the to10 / to8 / ResNet-18-teacher
# rows that distribution_check.ipynb still reads by name.
KD_MODEL_ROOT = './saved_models/kd_final'
KD_LOG_ROOT = './saved_logs/kd_final'


def kd_scenario_name(scenario, method_name, seed, r):
    """Folder name main() writes to. Single source of truth for the guard."""
    return f"{scenario}_{method_name}_{seed}_{r}"


def kd_run_done(scenario, method_name, seed, r):
    ckpt = os.path.join(KD_MODEL_ROOT,
                        kd_scenario_name(scenario, method_name, seed, r),
                        "best_epoch.pth")
    return os.path.isfile(ckpt) and os.path.getsize(ckpt) > 0


def kd_plan_methods(yaml_file_path):
    """(Scenario_Name, [method names]) read straight off a plan.

    Deliberately not process_experiment_kd_setup: that builds the dataset and
    every distiller, which is exactly the cost the __main__ guard exists to
    avoid.
    """
    with open(yaml_file_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    methods = [d.get("name") for d in data.get("Distillation", []) or []]
    return data.get("Scenario_Name"), methods


# =====================================================
# 3. Main training logic
# =====================================================
def main(seed, r, yaml_file_path): 
    print(device)
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    g = torch.Generator()
    g.manual_seed(seed)

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_kd_setup(exp_yaml) 

    print('==> Preparing data..')

    train_set = exp_setup["Dataset"].train_set
    test_set =  exp_setup["Dataset"].test_set

    # 25000 vs. 250000 （50000）
    group_A = create_or_load_group_A(dataset=train_set, save_dir=f'./Indices/{exp_yaml['Dataset']['name']}/',
                                    group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"], seed=42, force_rebuild=False)

    # Transfer set: the images the student is distilled on. `r` is the overlap
    # rate with the victim/teacher split (group_A).
    #   r = 1.0 -> group_B == group_A  : Cell A, student shares the teacher's data
    #   r = 0.0 -> group_B disjoint    : Cell C, student uses its own transfer set
    # Intermediate rates give the dose-response curve between the two.
    group_B = create_or_load_group_B(dataset=train_set, save_dir=f'./Indices/{exp_yaml['Dataset']['name']}/',
                                     group_A_indices=group_A, group_size=exp_setup["GroupSize"],
                                     num_classes=exp_setup["NumClasses"], overlap_rate=r,
                                     seed=42, force_rebuild=False)
    print(f"[DATA] transfer set: overlap_rate={r}, size={len(group_B)}, "
          f"overlap with group_A={len(set(group_B) & set(group_A))}")

    train_subset1 =  exp_setup["Dataset"].subset("train", group_B, clean=False) # with augmentation
    
    trainloader = DataLoader(train_subset1, batch_size=256, shuffle=True, num_workers=4,
                    worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                    pin_memory=True)

    testloader = DataLoader(test_set,batch_size=256,shuffle=False,num_workers=0,
                pin_memory=True)
    
    kd_methods = list(exp_setup["KD_Setups"].keys())

    for method_name in kd_methods:
        print(f"\n{'='*50}")
        print(f"==> Starting Experiment for Method: {method_name}")
        print(f"{'='*50}")

        # Unique naming and logging for this specific method
        scenario_name = exp_yaml["Scenario_Name"] + f"_{method_name}_{seed}_{r}"

        # Before the log file is touched and before the distiller (and its
        # teacher checkpoint) is built.
        if SKIP_EXISTING and kd_run_done(exp_yaml["Scenario_Name"], method_name, seed, r):
            print(f"[SKIP] {scenario_name}: best_epoch.pth already exists")
            continue

        log_dir = f'{KD_LOG_ROOT}/Performance'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"training_log_{scenario_name}.csv")

        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Scenario','Epoch', 'Train_Loss', 'Train_Acc', 'Train_Precision', 'Train_Recall','Train_F1', 
                                 'Test_Loss', 'Test_Acc', 'Test_Precision', 'Test_Recall','Test_F1'])
                
        kd_cfg = exp_setup["KD_Setups"][method_name]

        # Fresh distiller per method run
        distiller = kd_cfg["Builder"]().to(device)

        # Fresh optimizer from distiller's official learnable params
        trainable_params = distiller.get_learnable_parameters()
        if len(trainable_params) == 0:
            raise ValueError(f"No trainable parameters found for method {method_name}")

        optimizer_class = getattr(optim, kd_cfg["Optimizer_Name"])
        optimizer = optimizer_class(trainable_params, **kd_cfg["Optimizer_Params"])

        # Fresh scheduler
        scheduler = None
        if kd_cfg["Scheduler_Name"] is not None:
            s_params = kd_cfg["Scheduler_Params"].copy()
            if s_params.get("T_max") == "auto":
                s_params["T_max"] = exp_setup["Epochs"]

            if kd_cfg["Scheduler_Name"] == "WarmupCosineAnnealingLR":
                # Not a torch.optim.lr_scheduler class - the DeiT recipe needs
                # the repo's warmup+cosine helper, the same one the vanilla DeiT
                # training used (see util.process_experiment_setup_deit
                # and main_train_nega.main_deit).
                scheduler = build_warmup_cosine_scheduler(
                    optimizer,
                    total_epochs=int(s_params.get("T_max", exp_setup["Epochs"])),
                    warmup_epochs=int(s_params.get("warmup_epochs", 10)),
                    warmup_start_factor=float(s_params.get("warmup_start_factor", 0.1)),
                    eta_min=float(s_params.get("eta_min", 1e-6)),
                )
            else:
                scheduler_class = getattr(lr_sched, kd_cfg["Scheduler_Name"])
                scheduler = scheduler_class(optimizer, **s_params)
        
        # Note: criterion is handled internally by distiller, but evaluate1 might still need it
        criterion = nn.CrossEntropyLoss()

        # Training loop
        best_test_acc = -1.0
        os.makedirs(f'{KD_MODEL_ROOT}/{scenario_name}', exist_ok=True)

        for epoch in range(start_epoch, start_epoch + exp_yaml["Epochs"]): 
            print(f"This is the {method_name} {seed} round, {epoch} epoch!")
            train_result = train_one_epoch_kd(distiller, trainloader, optimizer, epoch, device)       
            # ---- scheduler step (per epoch) ----
            if scheduler is not None:
                scheduler.step()
            print(f"Epoch {epoch} | LR = {optimizer.param_groups[0]['lr']}")

            test_result = evaluate1(distiller.student, testloader, criterion, device)

            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([scenario_name, epoch, 
                                train_result["train_loss"], train_result["train_acc"], train_result["train_precision"], train_result["train_recall"],train_result["train_f1"], 
                                test_result["test_loss"], test_result["test_acc"], test_result["test_precision"], test_result["test_recall"],test_result["test_f1"], 
                                ])
                
            current_acc = test_result["test_acc"]
            if current_acc > best_test_acc:
                best_test_acc = current_acc
                torch.save(distiller.student.state_dict(), f'{KD_MODEL_ROOT}/{scenario_name}/best_epoch.pth') 
        
        # Save final epoch
        torch.save(distiller.student.state_dict(), f'{KD_MODEL_ROOT}/{scenario_name}/epoch_{epoch}.pth') 
        print(f"==> Finished {method_name}. Best Acc: {best_test_acc:.2f}%\n")


# =====================================================
# 4. Entry point
# =====================================================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Folder containing all YAML experiment plans
    # kd_plan_arch holds the architecture-pair plans for the Cell A / Cell C
    # grid (25000 group size, so a same-size disjoint group_B exists).
    # Switch back to "./saved_exp_plan/kd_plan" to re-run the original
    # 18to10 / 16to8 compression plans - but note those include 40000-group
    # configs, for which no disjoint transfer set can be built.
    # KD_PLAN_DIR overrides the folder from the environment, default unchanged.
    # The ResNet-34 teacher row lives in ./saved_exp_plan/kd_plan_rn34.
    exp_folder = os.environ.get("KD_PLAN_DIR", "./saved_exp_plan/kd_plan_arch")
    yaml_files = sorted(glob.glob(os.path.join(exp_folder, "*.yaml")))

    # Optional positional filters on the plan FILE NAME, case-insensitive
    # substrings -- same convention as main_train_teacher.py and
    # calculate_MI_extraction.py:
    #     python main_kd.py RES34to18
    #     python main_kd.py RES34to18 VGG19toVGG16
    # This is how one cell is given to one shell so several can train at once.
    # Different cells write different folders under KD_MODEL_ROOT and different
    # log files, so parallel shells do not collide. calculate_MI_kd.py is a
    # SINGLE-WRITER CSV, though -- run the MI step once, afterwards.
    _filters = [a.lower() for a in sys.argv[1:]]
    if _filters:
        yaml_files = [f for f in yaml_files
                      if any(x in os.path.basename(f).lower() for x in _filters)]

    if not yaml_files:
            print(f"No YAML files found in {exp_folder}"
                  + (f" matching {_filters}" if _filters else ""))
    else:
        print(f"Found {len(yaml_files)} experiment plan(s)"
              + (f" matching {_filters}" if _filters else "") + ":")
        for f in yaml_files:
            print(" -", f)

    # ---- Experiment grid -------------------------------------------------
    # OVERLAP_RATES selects which cell of the transfer-set design runs:
    #   1.0 -> Cell A : student distilled on the teacher's own split (group_A)
    #   0.0 -> Cell C : student distilled on a disjoint transfer set
    # Both cells keep the ground-truth CE term on (labels available); the
    # label-free cells B / D are a YAML change (CE_WEIGHT: 0.0), not a code one.
    # NOTE: a same-size disjoint group_B only exists for group_size = 25000,
    # so run these plans on the *_SMALL / 25000 configurations.
    # KD_SEEDS lists the repeated experiments, comma separated.
    #
    # A seed changes the student's initialisation and the transfer-set shuffle
    # order and NOTHING ELSE: main() builds group_A and group_B with seed=42
    # hard-coded, and the teacher is a fixed checkpoint path. So seed 1 is a
    # genuine repeat of seed 0's experiment, exactly like the negative pools
    # (whose 50 members also share one data split and differ only by training
    # seed). That is what makes per-seed spread comparable between the KD
    # positives and their null.
    #
    # Indexed from 0 since 2026-09-21, matching knockoff / DFMS / AT, whose
    # repeats are 0,1,2.. (KD used to start at 42). The six already-trained
    # ResNet-34-row cells were relabelled 42->0 in place; that checkpoint was
    # produced with torch RNG seed 42 even though it is now filed as seed 0,
    # so re-deriving it byte-for-byte needs KD_SEEDS=42, not 0.
    #
    # Same caveat for two more, copied in on 2026-09-21 rather than relabelled:
    #   kd_vanilla/CIFAR-10_ResNet-18to18_25000_{KD,DKD}_42_0.0
    #     -> kd_final/CIFAR-10_ResNet-18to18_25000_{KD,DKD}_0_0.0
    # They predate the kd_plan_matrix self-distillation cells but were trained
    # on a byte-compatible recipe (kd_plan_arch/old_plan/CIFAR10_RES18to18_KD
    # differs only by also listing FitNet), so they count as that cell's seed 0.
    # The originals are still in kd_vanilla.
    SEEDS = [int(x) for x in os.environ.get("KD_SEEDS", "0,1,2").split(",")]
    # KD_RATES overrides the overlap rates from the environment, comma separated:
    # KD_RATES="1.0,0.0" runs Cell A then Cell C. Default unchanged (0.0 only).
    OVERLAP_RATES = [float(x) for x in os.environ.get("KD_RATES", "0.0").split(",")]

    # Announce the grid and what the guard will do with it, so a stale shell
    # variable cannot quietly change the run count.
    for _k in ("KD_PLAN_DIR", "KD_SEEDS", "KD_RATES", "KD_SKIP_EXISTING"):
        if _k in os.environ:
            print(f"[ENV] {_k}={os.environ[_k]}")
    _planned = _todo = 0
    for _p in yaml_files:
        _scen, _ms = kd_plan_methods(_p)
        for _s in SEEDS:
            for _r in OVERLAP_RATES:
                for _m in _ms:
                    _planned += 1
                    if not (SKIP_EXISTING and kd_run_done(_scen, _m, _s, _r)):
                        _todo += 1
    print(f"Seeds: {SEEDS}   Rates: {OVERLAP_RATES}   SKIP_EXISTING: {SKIP_EXISTING}")
    print(f"TOTAL: {_planned} cell(s) = {len(yaml_files)} plan(s) x {len(SEEDS)} seed(s) "
          f"x {len(OVERLAP_RATES)} rate(s) x method(s)  ->  "
          f"{_planned - _todo} already trained, {_todo} to train")

    for yaml_path in yaml_files:
        print(f"\n========== Starting experiments from {yaml_path} ==========")

        for seed in SEEDS:
            for rate in OVERLAP_RATES:
                print(f"\n>>> Running seed {seed}, overlap_rate {rate} "
                      f"for {os.path.basename(yaml_path)}")

                if SKIP_EXISTING:
                    scen, methods = kd_plan_methods(yaml_path)
                    if methods and all(kd_run_done(scen, m, seed, rate) for m in methods):
                        print(f"[SKIP] {scen} seed {seed} rate {rate}: all "
                              f"{len(methods)} method(s) already trained "
                              f"({', '.join(methods)})")
                        continue

                set_seed(seed)

                main(seed, rate, yaml_path)
