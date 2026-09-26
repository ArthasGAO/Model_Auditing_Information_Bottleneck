import os
import glob
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # needed for full CUDA determinism
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
import numpy as np
from torch.utils.data import DataLoader

from util import (process_yaml_file, process_experiment_ft_setup, create_or_load_group_A,
                  determine_ft_dataset, load_best_checkpoint, evaluate1)
from AdvAttack.removalnet import RemovalNet, attach_resnet_feature_methods


# ---- determinism helpers (identical to main_ft.py) ----
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


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---- default RemovalNet hyperparameters (author's CIFAR-10 values). ----
# Any of these can be overridden by a `RemovalNet:` block in the YAML.
REMOVALNET_DEFAULTS = dict(
    layer=2,            # author ResNet scheme: 2 -> layer1 (configurable 1..5)
    logit_only=False,   # True -> ablation: skip the layer/latent term (decision-boundary only)
    ce_only=False,      # True -> ablation: CE only (alpha=beta=0); plain fine-tune on victim labels
    ydist="l2",
    alpha=0.2,
    beta=2.0,
    gamma=0.6,
    T=20,
    poison_steps=20,
    shuffle_ratio=0.02,
    lr=0.008,
    momentum=0.9,
    weight_decay=0.0001,
    iterations=1000,
    test_interval=20,
    save_interval=200,
    batch_size=128,
)


def build_cfg(exp_yaml):
    cfg_dict = dict(REMOVALNET_DEFAULTS)
    cfg_dict.update(exp_yaml.get("RemovalNet", {}) or {})
    return SimpleNamespace(**cfg_dict)


def _load_victim_copy(factory, ckpt_path):
    net = factory().to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    net.load_state_dict(state)
    attach_resnet_feature_methods(net)
    return net


def main_removalnet(model_seed, attack_seed, r, yaml_file_path):
    print(device)
    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_ft_setup(exp_yaml)   # reuse FT data/model pipeline (no util changes)
    cfg = build_cfg(exp_yaml)

    print('==> Loading victim model..')
    model_name = exp_yaml["Model_Name"] + f"_{model_seed}_{round(r, 2)}"
    model_dir = './saved_models/vanilla/CNN_Models'
    model_folder = Path(model_dir) / model_name
    ckpt_path, _ = load_best_checkpoint(model_folder)
    if ckpt_path is None:
        raise FileNotFoundError(f"No .pth files found in {model_folder}")
    print(f"[victim] {model_name} -> {ckpt_path}")

    print('==> Preparing data..')
    g = torch.Generator()
    g.manual_seed(attack_seed)

    group_A = create_or_load_group_A(
        dataset=exp_setup["Dataset"].train_set,
        save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
        group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"],
        seed=42, force_rebuild=False)

    # attacker surrogate data (disjoint group_B) + victim clean test set, exactly as FT does
    atk_data_train, atk_data_val = determine_ft_dataset(exp_yaml, exp_setup, group_A)

    trainloader = DataLoader(atk_data_train, batch_size=int(cfg.batch_size), shuffle=True, num_workers=4,
                             worker_init_fn=seed_worker, generator=g, persistent_workers=True, pin_memory=True)
    testloader = DataLoader(atk_data_val, batch_size=int(cfg.batch_size), shuffle=False, num_workers=4,
                            worker_init_fn=seed_worker, generator=g, persistent_workers=True, pin_memory=True)

    # two copies: frozen victim oracle (model_T) + trainable surrogate (model_t)
    factory = exp_setup["Model_Factory"]
    model_T = _load_victim_copy(factory, ckpt_path)
    model_t = _load_victim_copy(factory, ckpt_path)
    model_T.eval()
    for p in model_T.parameters():
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss()

    if bool(getattr(cfg, "ce_only", False)):
        mode_tag = "ceonly"
    elif bool(getattr(cfg, "logit_only", False)):
        mode_tag = "logitonly"
    else:
        mode_tag = f"l{cfg.layer}"
    scenario_name = (
        f'{exp_yaml["Scenario_Name"]}_{model_seed}_{round(r, 2)}'
        f'_{mode_tag}_iters{cfg.iterations}_aseed={attack_seed}'
    )
    print(scenario_name)

    save_dir = f'./saved_models/removalnet_vanilla/{scenario_name}'
    log_dir = './saved_logs/removalnet_vanilla/Performance'
    plot_dir = f'./saved_logs/removalnet_vanilla/Plots/{scenario_name}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{scenario_name}.csv")

    attack = RemovalNet(
        model_T=model_T, model_t=model_t,
        train_loader=trainloader, test_loader=testloader,
        cfg=cfg, device=device, criterion=criterion, evaluate_fn=evaluate1,
        log_file=log_file, save_dir=save_dir, scenario_name=scenario_name, plot_dir=plot_dir)
    attack.deepremoval()


# =====================================================
# Entry point (mirror of main_ft.py)
# =====================================================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    exp_folder = "./saved_exp_plan/removalnet_plan"
    yaml_files = sorted(glob.glob(os.path.join(exp_folder, "*.yaml")))

    if not yaml_files:
        print(f"No YAML files found in {exp_folder}")
    else:
        print(f"Found {len(yaml_files)} experiment plan(s):")
        for f in yaml_files:
            print(" -", f)

    # victim overlap-rate to attack (any trained victim is fine); 0.0 = standard victim
    VICTIM_R = 1.0
    for yaml_path in yaml_files:
        print(f"\n========== RemovalNet on {yaml_path} ==========")
        for model_seed in range(42, 43):
            for attack_seed in range(0, 1):
                print(f"\n>>> model_seed={model_seed} attack_seed={attack_seed} r={VICTIM_R}")
                set_seed(attack_seed)
                main_removalnet(model_seed, attack_seed, VICTIM_R, yaml_path)
