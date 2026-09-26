import argparse
import copy
import json
import os
import re
DETERMINISTIC = False

if DETERMINISTIC:
    # cuBLAS needs a fixed workspace, chosen before torch is imported.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from pathlib import Path
import random
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import csv
from torch.utils.data import DataLoader
from util import create_or_load_group_A, create_or_load_group_B, process_experiment_setup, process_yaml_file, build_dataset_from_yaml, wait_for_cool_gpu
from util_adv import (NormalizedModel, at_one_epoch, at_one_epoch_clean_bn_eval_aligned, at_one_epoch_scratch, create_at_logger, load_stolen_model, build_at_dataset_from_yaml, initialize_optimizer_scheduler,
                      compute_clean_accuracy, compute_robust_test_accuracy, log_at_epoch, parse_attack_configs, pgd_attack_v2, fgsm_attack)

# =====================================================
# 1. Global setup
# =====================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Default AT seeds: 0, 1, 2. Explicit --seeds overrides the environment range.
# SEED_START/SEED_END remain available for splitting work across shells.
SEED_START = int(os.environ.get("SEED_START", 0))
SEED_END   = int(os.environ.get("SEED_END", 3))
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


def parse_mixed_training_config(exp_yaml):
    """Read and validate the clean/adversarial mixing settings from YAML."""
    at_cfg = exp_yaml.get("AdversarialTraining", {})
    if at_cfg is None:
        at_cfg = {}
    if not isinstance(at_cfg, dict):
        raise ValueError("AdversarialTraining must be a YAML mapping.")

    use_mixed = at_cfg.get("use_mixed", True)
    if not isinstance(use_mixed, bool):
        raise ValueError("AdversarialTraining.use_mixed must be true or false.")

    try:
        mix_rate = float(at_cfg.get("mix_rate", 0.8))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "AdversarialTraining.mix_rate must be a number between 0 and 1."
        ) from exc

    if not 0.0 <= mix_rate <= 1.0:
        raise ValueError(
            "AdversarialTraining.mix_rate must be between 0 and 1."
        )

    return use_mixed, mix_rate


BEST_CKPT_START_FRAC = 0.2

# Keep a checkpoint for EVERY epoch, not just the last one and the two selected
# ones. Off by default: 30 extra files of ~45 MB per run is 1.4 GB, which is
# only worth paying when the intermediate models are the point -- e.g. reading
# MI at each epoch to draw a trajectory (calculate_MI_at_traj.py). Turned on
# from the CLI with --save-all-epochs; main() sets it before any training runs.
SAVE_ALL_EPOCHS = False


def parse_at_sources(exp_yaml):
    """[(model_path, alias_or_None)] for every Model_Path entry.

    An entry is either a path string (the original form, alias None) or a
    mapping {Path, Alias}. The alias replaces the source folder name in the
    output folder, for the same two reasons main_at_posthoc.parse_sources has
    one: the un-aliased pruning name can exceed the Windows path limit, and
    calculate_MI_at.parse_at_scenario_name splits the AT suffix on '=' pairs,
    so any '=' inside the source part (ftsize=25000, sparsity=0.2, ftseed=0)
    makes the finished run unreadable downstream. An alias must therefore carry
    no '=' and keep a trailing _<seed>_<rate>, which is what that parser expects.
    """
    out = []
    for entry in exp_yaml.get("Model_Path", []):
        if isinstance(entry, str):
            out.append((entry, None))
            continue
        if not isinstance(entry, dict) or "Path" not in entry:
            raise ValueError(f"Model_Path entry must be a string or a mapping with Path: {entry!r}")
        alias = entry.get("Alias")
        if alias is not None:
            alias = str(alias)
            if "=" in alias or "/" in alias or not alias.strip():
                raise ValueError(f"Model_Path Alias must be non-empty and contain no '=' or '/': {alias!r}")
        out.append((str(entry["Path"]), alias))
    return out


def write_at_source_record(scenario_name, model_path, alias, root="./saved_models/at_evasion"):
    """Record the real source next to an aliased run, so the alias stays traceable."""
    if alias is None:
        return
    folder = Path(root) / scenario_name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "source.json").write_text(
        json.dumps({"Path": model_path, "Alias": alias,
                    "checkpoint": str(Path("./saved_models" + model_path) / "best_epoch.pth")},
                   indent=2),
        encoding="utf-8")


def build_at_pos_scenario_name(
    model_path, attack_name, attack_kwargs, seed, use_mixed, mix_rate,
    *, train_size=None, epochs=None, run_tag=None, alias=None,
):
    """Distinguish AT data sizes/budgets; keep seed/mix last for notebook parsing.

    `alias`, when given, replaces the source folder name (see parse_at_sources);
    omitting it reproduces the original name exactly.
    """
    base_name = alias if alias is not None else model_path.split('/')[-1]
    param_str = "_".join(f"{k}={v}" for k, v in sorted(attack_kwargs.items()))
    mix_label = f"{mix_rate:g}" if use_mixed else "off"
    bn_label = "clean_eval" if use_mixed and mix_rate > 0.0 else "adv"
    training_label = ""
    if train_size is not None or epochs is not None or run_tag is not None:
        if train_size is None or epochs is None or train_size <= 0 or epochs <= 0:
            raise ValueError("Run names require positive train_size and epochs.")
        training_label = f"_atn={train_size}_atepochs={epochs}"
    if run_tag is not None:
        if not re.fullmatch(r"[A-Za-z0-9-]{1,32}", run_tag):
            raise ValueError("run_tag must contain 1-32 letters, digits or hyphens.")
        training_label += f"_run={run_tag}"
    return (
        f"{base_name}_{attack_name}_{param_str}_bn={bn_label}"
        f"{training_label}_atseed={seed}_mix={mix_label}"
    )


def check_at_pos_outputs_available(scenario_name):
    """Do not append a new experiment to an old CSV or overwrite its models."""
    csv_path = Path('saved_logs/at_evasion/Performance') / f'at_log_{scenario_name}.csv'
    ckpt_path = Path('saved_models/at_evasion') / scenario_name
    if csv_path.exists() or ckpt_path.exists():
        raise FileExistsError(
            f"Outputs already exist for {scenario_name}; choose a new --run-tag "
            "or select only unstarted plans/seeds. This entry point does not resume runs."
        )


# =====================================================
# 7. Main Adversarial Train
# =====================================================
def main_at_pos(seed, yaml_file_path, *, run_tag=None, reseed_each_case=True):
    g = torch.Generator()
    g.manual_seed(seed)

    print(f"Device: {device}")

    exp_yaml = process_yaml_file(yaml_file_path)

    # ---------- Prepare dataset ----------
    dataset_obj, num_classes, _ = build_dataset_from_yaml(exp_yaml["Dataset"])
    at_train, at_val = build_at_dataset_from_yaml(dataset_obj, exp_yaml["Dataset"], rate=0.0)
    if len(at_train) != exp_yaml["Dataset"]["group_size"]:
        raise ValueError("AT index count does not match Dataset.group_size.")

    # ---------- Prepare Attack Configurations (may be multiple) ----------
    attack_configs = parse_attack_configs(exp_yaml)
    print(f"Found {len(attack_configs)} attack config(s)")

    # ---------- Prepare clean/adversarial mixing configuration ----------
    use_mixed, mix_rate = parse_mixed_training_config(exp_yaml)
    adv_rate = 1.0 - mix_rate
    if use_mixed:
        print(f"Mixed training enabled: clean={mix_rate:.2f}, adversarial={adv_rate:.2f}")
        if mix_rate > 0.0:
            print("BatchNorm policy: clean-only updates; adversarial loss uses clean running statistics")
        else:
            print("BatchNorm policy: update running statistics from adversarial inputs")
    else:
        print("Mixed training disabled: using adversarial examples only")
        print("BatchNorm policy: update running statistics from adversarial inputs")

    # ---------- Load and train for each model × attack combo ----------
    model_path_ls = parse_at_sources(exp_yaml)

    for model_path, model_alias in model_path_ls:
        for attack_fn, attack_name, attack_kwargs in attack_configs:
            if reseed_each_case:
                # Match each model's random starting stream across epoch budgets.
                set_seed(seed, deterministic=DETERMINISTIC)
                g = torch.Generator().manual_seed(seed)
            print(f"\n{'='*60}")
            print(f"Attack: {attack_name}, {attack_kwargs}")
            print(f"Model:  {model_path}")
            print(f"{'='*60}")

            scenario_name = build_at_pos_scenario_name(
                model_path, attack_name, attack_kwargs, seed, use_mixed, mix_rate,
                train_size=len(at_train), epochs=exp_yaml["Optimizer"]["Epochs"],
                run_tag=run_tag, alias=model_alias,
            )
            check_at_pos_outputs_available(scenario_name)
            write_at_source_record(scenario_name, model_path, model_alias)

            # Reload fresh model for each attack config
            suspect_model = load_stolen_model(exp_yaml, dataset_obj, num_classes, model_path)

            log_dir = './saved_logs/at_evasion/Performance'
            csv_file, writer = create_at_logger(log_dir, scenario_name, attack_name, attack_kwargs)

            at_train_loader = DataLoader(at_train, batch_size=128, shuffle=True, num_workers=4,
                            worker_init_fn=seed_worker, generator=g, persistent_workers=True,
                            pin_memory=True)

            at_val_loader = DataLoader(at_val, batch_size=128, shuffle=False, num_workers=2,
                            worker_init_fn=seed_worker, generator=g, persistent_workers=True,
                            pin_memory=True)

            # Pre-AT evaluation
            clean_acc = compute_clean_accuracy(suspect_model, at_val_loader)
            rob_acc = compute_robust_test_accuracy(suspect_model, at_val_loader, attack_fn, attack_kwargs)

            print(f" [Epoch -1] Clean Acc: {clean_acc*100:.2f}%,"
                  f" Robustness Acc: {rob_acc*100:.2f}%")

            log_at_epoch(writer, scenario_name, -1, attack_name, attack_kwargs,
                         0, 0, clean_acc, rob_acc)

            # Fresh optimizer for each attack config
            at_optimizer, at_scheduler, at_epochs = initialize_optimizer_scheduler(exp_yaml, suspect_model)
            loss_fn = nn.CrossEntropyLoss()

            best_rob_test_acc, best_clean_acc = -1.0, -1.0
            best_ckpt_from = int(at_epochs * BEST_CKPT_START_FRAC)
            os.makedirs(f'./saved_models/at_evasion/{scenario_name}', exist_ok=True)

            for epoch in range(0, at_epochs):
                train_loss, train_acc = at_one_epoch_clean_bn_eval_aligned(
                                        suspect_model,
                                        at_train_loader,
                                        at_optimizer,
                                        loss_fn,
                                        attack_fn,
                                        attack_kwargs,
                                        device,
                                        use_mixed=use_mixed,
                                        clean_weight=mix_rate,
                                        adv_weight=adv_rate,
                                        train_acc_on="adv",
                                    )

                wait_for_cool_gpu(threshold=89.5) # to check gpu temperature during at

                if at_scheduler is not None:
                    at_scheduler.step()

                clean_acc = compute_clean_accuracy(suspect_model, at_val_loader)
                rob_acc = compute_robust_test_accuracy(suspect_model, at_val_loader, attack_fn, attack_kwargs)

                print(f"  [Epoch {epoch}/{at_epochs}],"
                      f" LR: {at_optimizer.param_groups[0]['lr']},"
                      f" Loss: {train_loss:.4f},"
                      f" Train Acc: {train_acc*100:.2f}%,"
                      f" Clean Acc: {clean_acc*100:.2f}%,"
                      f" Robust Acc: {rob_acc*100:.2f}%")

                log_at_epoch(writer, scenario_name, epoch, attack_name, attack_kwargs,
                            train_loss, train_acc, clean_acc, rob_acc)

                if epoch >= best_ckpt_from and rob_acc > best_rob_test_acc:
                    best_rob_test_acc = rob_acc
                    torch.save(suspect_model.state_dict(), f'./saved_models/at_evasion/{scenario_name}/best_rob_epoch.pth')

                if epoch >= best_ckpt_from and clean_acc > best_clean_acc:
                    best_clean_acc = clean_acc
                    torch.save(suspect_model.state_dict(), f'./saved_models/at_evasion/{scenario_name}/best_clean_epoch.pth')

                # The final epoch is written once below, outside the loop, as it
                # always was; this only adds the ones before it.
                if SAVE_ALL_EPOCHS and epoch < at_epochs - 1:
                    torch.save(suspect_model.state_dict(),
                               f'./saved_models/at_evasion/{scenario_name}/epoch_{epoch}.pth')

            torch.save(suspect_model.state_dict(),
                    f'./saved_models/at_evasion/{scenario_name}/epoch_{epoch}.pth')

            csv_file.close()


def main_at_neg(seed, yaml_file_path): # this is used to test the AT fine-tune on existing negative models
    g = torch.Generator()
    g.manual_seed(seed)

    print(f"Device: {device}")

    exp_yaml = process_yaml_file(yaml_file_path)

    # ---------- Prepare dataset ----------
    dataset_obj, num_classes, _ = build_dataset_from_yaml(exp_yaml["Dataset"])

    # ---------- Prepare Attack Configurations (may be multiple) ----------
    attack_configs = parse_attack_configs(exp_yaml)
    print(f"Found {len(attack_configs)} attack config(s)")

    # ---------- Load and train for each model × attack combo ----------
    model_path_ls = exp_yaml.get("Model_Path", [])

    for model_path in model_path_ls:
        for attack_fn, attack_name, attack_kwargs in attack_configs:
            print(f"\n{'='*60}")
            print(f"Attack: {attack_name}, {attack_kwargs}")
            print(f"Model:  {model_path}")
            print(f"{'='*60}")

            base_name = model_path.split('/')[-1]
            overlap_rate = base_name.split('_')[-1]

            at_train, at_val = build_at_dataset_from_yaml(dataset_obj, exp_yaml["Dataset"], rate=overlap_rate)

            param_str = "_".join(f"{k}={v}" for k, v in sorted(attack_kwargs.items()))
            scenario_name = base_name + f"_{attack_name}_{param_str}_ftseed={seed}"

            # Reload fresh model for each attack config
            suspect_model = load_stolen_model(exp_yaml, dataset_obj, num_classes, model_path)

            log_dir = './saved_logs/at_vanilla/Performance'
            csv_file, writer = create_at_logger(log_dir, scenario_name, attack_name, attack_kwargs)

            at_train_loader = DataLoader(at_train, batch_size=128, shuffle=True, num_workers=8,
                            worker_init_fn=seed_worker, generator=g, persistent_workers=True,
                            pin_memory=True)

            at_val_loader = DataLoader(at_val, batch_size=128, shuffle=False, num_workers=0,
                            pin_memory=True)

            # Pre-AT evaluation
            clean_acc = compute_clean_accuracy(suspect_model, at_val_loader)
            rob_acc = compute_robust_test_accuracy(suspect_model, at_val_loader, attack_fn, attack_kwargs)

            print(f" [Epoch -1] Clean Acc: {clean_acc*100:.2f}%,"
                  f" Robustness Acc: {rob_acc*100:.2f}%")

            log_at_epoch(writer, scenario_name, -1, attack_name, attack_kwargs,
                         0, 0, clean_acc, rob_acc)

            # Fresh optimizer for each attack config
            at_optimizer, at_scheduler, at_epochs = initialize_optimizer_scheduler(exp_yaml, suspect_model)
            loss_fn = nn.CrossEntropyLoss()

            best_rob_test_acc, best_clean_acc = -1.0, -1.0
            os.makedirs(f'./saved_models/at_vanilla/{scenario_name}', exist_ok=True)

            for epoch in range(0, at_epochs):
                train_loss, train_acc = at_one_epoch(
                    suspect_model, at_train_loader, at_optimizer, loss_fn,
                    attack_fn, attack_kwargs, device
                )

                wait_for_cool_gpu(threshold=89.5) # to check gpu temperature during at

                if at_scheduler is not None:
                    at_scheduler.step()

                clean_acc = compute_clean_accuracy(suspect_model, at_val_loader)
                rob_acc = compute_robust_test_accuracy(suspect_model, at_val_loader, attack_fn, attack_kwargs)

                print(f"  [Epoch {epoch}/{at_epochs}],"
                      f" LR: {at_optimizer.param_groups[0]['lr']},"
                      f" Loss: {train_loss:.4f},"
                      f" Train Acc: {train_acc*100:.2f}%,"
                      f" Clean Acc: {clean_acc*100:.2f}%,"
                      f" Robust Acc: {rob_acc*100:.2f}%")

                log_at_epoch(writer, scenario_name, epoch, attack_name, attack_kwargs,
                            train_loss, train_acc, clean_acc, rob_acc)

                if rob_acc > best_rob_test_acc:
                    best_rob_test_acc = rob_acc
                    torch.save(suspect_model.state_dict(), f'./saved_models/at_vanilla/{scenario_name}/best_rob_epoch.pth')

                if clean_acc > best_clean_acc:
                    best_clean_acc = clean_acc
                    torch.save(suspect_model.state_dict(), f'./saved_models/at_vanilla/{scenario_name}/best_clean_epoch.pth')

            torch.save(suspect_model.state_dict(), f'./saved_models/at_vanilla/{scenario_name}/epoch_{epoch}.pth')

            csv_file.close()


def main_at_neg_scratch(seed, r, yaml_file_path):
    print(device)
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    g = torch.Generator()
    g.manual_seed(seed)

    print(f"Device: {device}")

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_setup(exp_yaml) 
    base_name = exp_yaml["Scenario_Name"] + f"_{seed}_{round(r, 2)}"

    # ---------- Prepare dataset ----------
    print('==> Preparing data..')

    train_set = exp_setup["Dataset"].train_set
    test_set =  exp_setup["Dataset"].raw_test_set

    # 25000 vs. 250000 （50000）exp_setup["GroupSize"]
    group_A = create_or_load_group_A(dataset=train_set, save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
                                                  group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"], seed=42, force_rebuild=False)

    group_B = create_or_load_group_B(dataset=train_set, save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/', group_A_indices=group_A,
                                     group_size = exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"], 
                                     overlap_rate=r, seed=42, force_rebuild=False)
    
    train_subset1 =  exp_setup["Dataset"].subset("raw_train", group_B) # with augmentation
    
    at_trainloader = DataLoader(train_subset1, batch_size=128, shuffle=True, num_workers=8,
                    worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                    pin_memory=True)

    at_testloader = DataLoader(test_set,batch_size=128,shuffle=False,num_workers=0, pin_memory=True)
    
    print('==> Building model..')
    # load experiment information
    init_model = exp_setup["Model"]
    init_model = NormalizedModel(init_model, exp_setup["Dataset"].mean, exp_setup["Dataset"].std).to(device)

    # ---------- Prepare Attack Configurations (may be multiple) ----------
    attack_configs = parse_attack_configs(exp_yaml)
    print(f"Found {len(attack_configs)} attack config(s)")

    for attack_fn, attack_name, attack_kwargs in attack_configs:
        print(f"\n{'='*60}")
        print(f"Attack: {attack_name}, {attack_kwargs}")
        print(f"Model:  {base_name}")
        print(f"{'='*60}")

        param_str = "_".join(f"{k}={v}" for k, v in sorted(attack_kwargs.items()))
        scenario_name = base_name + f"_{attack_name}_{param_str}_atseed={seed}"

        log_dir = './saved_logs/at_vanilla/Performance'
        csv_file, writer = create_at_logger(log_dir, scenario_name, attack_name, attack_kwargs)

        net = copy.deepcopy(init_model).to(device)
        # Fresh optimizer for each attack config
        at_optimizer, at_scheduler, at_epochs = initialize_optimizer_scheduler(exp_yaml, net)
        loss_fn = nn.CrossEntropyLoss()

        best_rob_test_acc, best_clean_acc = -1.0, -1.0
        os.makedirs(f'./saved_models/at_vanilla/{scenario_name}', exist_ok=True)

        for epoch in range(0, at_epochs):
            train_loss, train_acc = at_one_epoch_scratch(
                net, at_trainloader, at_optimizer, loss_fn,
                attack_fn, attack_kwargs, device
            )

            wait_for_cool_gpu(threshold=89.5) # to check gpu temperature during at

            if at_scheduler is not None:
                at_scheduler.step()

            clean_acc = compute_clean_accuracy(net, at_testloader)
            rob_acc = compute_robust_test_accuracy(net, at_testloader, attack_fn, attack_kwargs)

            print(f"  [Epoch {epoch}/{at_epochs}],"
                    f" LR: {at_optimizer.param_groups[0]['lr']},"
                    f" Loss: {train_loss:.4f},"
                    f" Train Acc: {train_acc*100:.2f}%,"
                    f" Clean Acc: {clean_acc*100:.2f}%,"
                    f" Robust Acc: {rob_acc*100:.2f}%")

            log_at_epoch(writer, scenario_name, epoch, attack_name, attack_kwargs,
                        train_loss, train_acc, clean_acc, rob_acc)

            if rob_acc > best_rob_test_acc:
                best_rob_test_acc = rob_acc
                torch.save(net.state_dict(), f'./saved_models/at_vanilla/{scenario_name}/best_rob_epoch.pth')

            if clean_acc > best_clean_acc:
                best_clean_acc = clean_acc
                torch.save(net.state_dict(), f'./saved_models/at_vanilla/{scenario_name}/best_clean_epoch.pth')

        torch.save(net.state_dict(),
                f'./saved_models/at_vanilla/{scenario_name}/epoch_{epoch}.pth')

        csv_file.close()



# =====================================================
# 8. Entry point
# =====================================================
def prepare_at_pos_plans(yaml_files, seeds, run_tag, epoch_filter=None):
    """Read-only preflight, using the same attack expansion and names as training."""
    plans = []
    for path in yaml_files:
        with path.open(encoding="utf-8") as file:
            cfg = yaml.safe_load(file)
        epochs = cfg["Optimizer"]["Epochs"]
        if epoch_filter is None or epochs in epoch_filter:
            plans.append((path, cfg))
    # Complete all shorter-budget cases before starting the next budget.
    plans.sort(key=lambda plan: (plan[1]["Optimizer"]["Epochs"], str(plan[0])))
    if not plans:
        raise FileNotFoundError("No YAML plans matched the selected paths/epoch budgets.")
    if epoch_filter is not None:
        missing = set(epoch_filter) - {cfg["Optimizer"]["Epochs"] for _, cfg in plans}
        if missing:
            raise ValueError(f"No YAML plan found for epoch budgets: {sorted(missing)}")

    scenarios = []
    seen = set()
    for path, cfg in plans:
        use_mixed, mix_rate = parse_mixed_training_config(cfg)
        ds = cfg["Dataset"]
        # Mirror main_at_pos's rate=0.0 index selection without loading images.
        index_path = Path("Indices") / ds["name"] / (
            f'group_B_25000_0.0_{ds["group_size"]}_seed42.npy'
        )
        indices = np.load(index_path, allow_pickle=False)
        if (indices.shape != (ds["group_size"],)
                or not np.issubdtype(indices.dtype, np.integer)
                or np.unique(indices).size != ds["group_size"]
                or np.any(indices < 0)
                or (ds["name"] == "CIFAR-10" and np.any(indices >= 50000))):
            raise ValueError(f"Invalid AT subset indices: {index_path}")
        attacks = parse_attack_configs(cfg)
        if not cfg.get("Model_Path") or not attacks:
            raise ValueError(f"{path.name}: at least one model and attack are required.")
        sources = parse_at_sources(cfg)
        for model_path, _alias in sources:
            # load_stolen_model uses load_best_checkpoint's default filename.
            checkpoint = Path("./saved_models" + model_path) / "best_epoch.pth"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"No initial checkpoint for {model_path}")
        for seed in seeds:
            for model_path, alias in sources:
                for _, attack_name, kwargs in attacks:
                    scenario = build_at_pos_scenario_name(
                        model_path, attack_name, kwargs, seed, use_mixed, mix_rate,
                        train_size=len(indices), epochs=cfg["Optimizer"]["Epochs"],
                        run_tag=run_tag, alias=alias,
                    )
                    if scenario in seen:
                        raise ValueError(f"Duplicate output name across selected plans: {scenario}")
                    seen.add(scenario)
                    check_at_pos_outputs_available(scenario)
                    scenarios.append(scenario)
    return plans, scenarios


def main(argv=None):
    parser = argparse.ArgumentParser(description="Unified YAML-driven adversarial fine-tuning.")
    parser.add_argument("--plans", nargs="+", type=Path,
                        help="Specific YAML files; default: all top-level saved_exp_plan/at_plan/*.yaml.")
    parser.add_argument("--epochs", nargs="+", type=int,
                        help="Select plans by Optimizer.Epochs; does not override YAML settings.")
    parser.add_argument("--seeds", nargs="+", type=int,
                        help="AT seeds; default: SEED_START..SEED_END (exclusive), normally 0 1 2.")
    parser.add_argument("--run-tag", default="v1", help="Independent repetition label, e.g. v1 or v2.")
    parser.add_argument("--save-all-epochs", action="store_true",
                        help="Keep epoch_<N>.pth for every epoch (~1.4 GB per 30-epoch run), "
                             "not just the last; needed for per-epoch MI trajectories.")
    parser.add_argument("--check-only", action="store_true",
                        help="Check indices, checkpoint paths and output names without training.")
    args = parser.parse_args(argv)
    seeds = args.seeds if args.seeds is not None else list(range(SEED_START, SEED_END))
    if not seeds or len(seeds) != len(set(seeds)) or any(s < 0 or s >= 2**32 for s in seeds):
        parser.error("Seeds must be distinct integers in [0, 2**32), with at least one seed.")
    if args.epochs is not None and (
            len(args.epochs) != len(set(args.epochs)) or any(e <= 0 for e in args.epochs)):
        parser.error("Epoch budgets must be distinct positive integers.")
    if not re.fullmatch(r"[A-Za-z0-9-]{1,32}", args.run_tag):
        parser.error("--run-tag must contain 1-32 letters, digits or hyphens.")

    # Existing dataset/model paths are project-relative, even from another shell cwd.
    os.chdir(ROOT)
    yaml_files = args.plans or sorted((ROOT / "saved_exp_plan/at_plan").glob("*.yaml"))
    global SAVE_ALL_EPOCHS
    SAVE_ALL_EPOCHS = bool(args.save_all_epochs)
    plans, scenarios = prepare_at_pos_plans(yaml_files, seeds, args.run_tag, args.epochs)
    print(f"Plans: {len(plans)}; AT seeds: {seeds}; run tag: {args.run_tag}")
    for path, cfg in plans:
        print(f'  {cfg["Optimizer"]["Epochs"]} epochs, {cfg["Dataset"]["group_size"]} images: {path.name}')
    print(f"Total: {len(scenarios)} independent cases; each reloads its original checkpoint.")
    if args.check_only:
        for scenario in scenarios:
            print("  " + scenario)
        print("Preflight passed. No training started and no experiment outputs written.")
        return

    torch.multiprocessing.set_start_method("spawn", force=True)
    for yaml_path, _ in plans:
        print(f"\n========== Starting experiments from {yaml_path} ==========")
        for seed in seeds:
            print(f"\n>>> Running seed {seed} for {os.path.basename(yaml_path)}")
            set_seed(seed, deterministic=DETERMINISTIC)
            main_at_pos(seed, str(yaml_path), run_tag=args.run_tag, reseed_each_case=True)


if __name__ == "__main__":
    main()
