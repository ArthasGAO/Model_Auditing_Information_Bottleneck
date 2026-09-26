import os
import glob
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
import numpy as np
from torch.utils.data import DataLoader

from util import (process_yaml_file, process_experiment_ft_setup_deit, create_or_load_group_A,
                  determine_ft_dataset, load_best_checkpoint, evaluate1)
from AdvAttack.removalnet import RemovalNet

# NOTE: DeiT path mirrors main_ft_deit.py. It uses the SAME RemovalNet class as the
# CNN path, but forces logit_only=True (the latent/feature term does not apply to a
# ViT in the author's layer scheme), loads the victim from Transformer_Models, builds
# via timm (process_experiment_ft_setup_deit), and does NOT attach ResNet feature
# methods. The ResNet runner (main_removalnet.py) is untouched.


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


# ---- DeiT logit-only defaults (AdamW, transformer-appropriate). ----
# logit_only is forced True regardless of YAML (DeiT supports decision-boundary
# removal only here). Override any field via the YAML `RemovalNet:` block.
REMOVALNET_DEIT_DEFAULTS = dict(
    logit_only=True,
    optimizer="adamw",   # transformers need adaptive optimizer, not SGD
    layer=2,             # ignored (logit_only)
    ydist="l2",
    alpha=0.8,           # weight on logit-level KL removal
    gamma=0.2,           # weight on accuracy-preserving CE
    T=20,
    lr=5e-4,             # AdamW lr (DeiT-appropriate, much smaller than ResNet SGD)
    momentum=0.9,        # unused with adamw
    weight_decay=0.03,
    iterations=1000,
    test_interval=20,
    save_interval=200,
    batch_size=128,
    # feature-only fields kept for cfg completeness (ignored in logit_only):
    beta=2.0, poison_steps=20, shuffle_ratio=0.02,
)


def build_cfg(exp_yaml):
    cfg_dict = dict(REMOVALNET_DEIT_DEFAULTS)
    cfg_dict.update(exp_yaml.get("RemovalNet", {}) or {})
    cfg_dict["logit_only"] = True  # enforce: DeiT path is logit-only
    return SimpleNamespace(**cfg_dict)


def _load_victim_copy_deit(factory, ckpt_path):
    net = factory().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state)
    # NO attach_resnet_feature_methods: DeiT logit-only never needs intermediate layers.
    return net


def main_removalnet_deit(model_seed, attack_seed, r, yaml_file_path):
    print(device)
    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_ft_setup_deit(exp_yaml)   # timm-based DeiT factory
    cfg = build_cfg(exp_yaml)

    print('==> Loading victim DeiT model..')
    model_name = exp_yaml["Model_Name"] + f"_{model_seed}_{round(r, 2)}"
    model_dir = './saved_models/vanilla/Transformer_Models'   # <-- DeiT lives here
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

    atk_data_train, atk_data_val = determine_ft_dataset(exp_yaml, exp_setup, group_A)

    trainloader = DataLoader(atk_data_train, batch_size=int(cfg.batch_size), shuffle=True, num_workers=4,
                             worker_init_fn=seed_worker, generator=g, persistent_workers=True, pin_memory=True)
    testloader = DataLoader(atk_data_val, batch_size=int(cfg.batch_size), shuffle=False, num_workers=4,
                            worker_init_fn=seed_worker, generator=g, persistent_workers=True, pin_memory=True)

    factory = exp_setup["Model_Factory"]
    model_T = _load_victim_copy_deit(factory, ckpt_path)
    model_t = _load_victim_copy_deit(factory, ckpt_path)
    model_T.eval()
    for p in model_T.parameters():
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss()

    scenario_name = (
        f'{exp_yaml["Scenario_Name"]}_{model_seed}_{round(r, 2)}'
        f'_logitonly_iters{cfg.iterations}_aseed={attack_seed}'
    )
    print(scenario_name)

    save_dir = f'./saved_models/removalnet_vanilla/Transformer_Models/{scenario_name}'
    log_dir = './saved_logs/removalnet_vanilla/Performance/Transformer_Models'
    plot_dir = f'./saved_logs/removalnet_vanilla/Plots/Transformer_Models/{scenario_name}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{scenario_name}.csv")

    attack = RemovalNet(
        model_T=model_T, model_t=model_t,
        train_loader=trainloader, test_loader=testloader,
        cfg=cfg, device=device, criterion=criterion, evaluate_fn=evaluate1,
        log_file=log_file, save_dir=save_dir, scenario_name=scenario_name, plot_dir=plot_dir)
    attack.deepremoval()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    exp_folder = "./saved_exp_plan/removalnet_plan_deit"
    yaml_files = sorted(glob.glob(os.path.join(exp_folder, "*.yaml")))

    if not yaml_files:
        print(f"No YAML files found in {exp_folder}")
    else:
        print(f"Found {len(yaml_files)} experiment plan(s):")
        for f in yaml_files:
            print(" -", f)

    VICTIM_R = 1.0
    for yaml_path in yaml_files:
        print(f"\n========== RemovalNet (DeiT, logit-only) on {yaml_path} ==========")
        for model_seed in range(42, 43):
            for attack_seed in range(0, 1):
                print(f"\n>>> model_seed={model_seed} attack_seed={attack_seed} r={VICTIM_R}")
                set_seed(attack_seed)
                main_removalnet_deit(model_seed, attack_seed, VICTIM_R, yaml_path)
