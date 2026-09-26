"""Validity check for an adversarially trained checkpoint: is its robustness real?

Loads one checkpoint and measures clean / robust test accuracy under
  1. the running statistics as saved (eval mode) = the deployed model;
  2. train-mode batch statistics on the test batches;
  3. BatchNorm recalibrated on CLEAN training images (no gradient, cumulative average);
  4. BatchNorm recalibrated on ADVERSARIAL training images (PGD against the model);
  5. BatchNorm recalibrated on a 50/50 mix.

How to read it (2026-09-22 findings):
  * |row 2 - row 1| of several tens of points = the network learned a
    batch-statistics "switch" instead of robust features (seen with
    bn_policy=both when fine-tuning the victim at lr 0.001): INVALID model.
  * rows 3-5 changing the clean accuracy by tens of points = running
    statistics drifted away from the weights (bn_policy=clean_eval): INVALID.
  * for a frozen-BN model (bn_policy=frozen) rows 3-5 legitimately lower the
    robustness (the frozen statistics are part of the learned function); the
    validity criterion is row 1 vs row 2 only.
  * a healthy from-scratch or pure-adversarial model keeps all five rows
    within a few points of each other (at_final mix=off: 81/45 everywhere).

Usage
  python diagnose_bn_modes.py --ckpt <state_dict.pth> [--arch resnet18|resnet18_dist]
         [--eps 0.031373] [--steps 10] [--robust-n 2000] [--calib-batches 60] [--plan <yaml>]
         [--policy frozen|both|joint]
  --policy joint (main_at_posthoc.py's mixed-batch BN): the verdict compares row 1 with row 5
  (mixture recalibration) instead of row 2, because joint models never see clean-only batches.
The plan only supplies the dataset (default: the CIFAR-10 ft_at plan). Checkpoints
saved as NormalizedModel state_dicts (base_model. prefix) are unwrapped.
"""
import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from util import (create_or_load_group_A, create_or_load_group_B, evaluate1, process_experiment_ft_setup,
                  process_yaml_file)
from util_adv import NormalizedModel, compute_robust_test_accuracy, pgd_attack_v2

device = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(__file__).resolve().parent
DEFAULT_PLAN = ROOT / "saved_exp_plan/ft_at_plan/CIFAR10_RES18_FT_Same_25000_ATPGD8.yaml"


def build_net(arch, num_classes):
    if arch == "resnet18":
        from Model.ResNet_18 import ResNet18
        return ResNet18(num_classes=num_classes)
    if arch == "resnet18_dist":
        from Model.kd_eval import KDLogitsOnly
        from Model.ResNet_18_dist import ResNet18_dist
        return KDLogitsOnly(ResNet18_dist(num_classes=num_classes))
    raise ValueError(f"unsupported arch {arch}")


def load_state(path):
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(k.startswith("base_model.") for k in state):
        state = {k[len("base_model."):]: v for k, v in state.items() if k not in ("mean", "std")}
    return state


def bn_layers(m):
    return [b for b in m.modules() if isinstance(b, nn.modules.batchnorm._BatchNorm)]


def recalibrate(model, calib_loader, mode, attack, calib_batches):
    m = copy.deepcopy(model)
    for b in bn_layers(m):
        b.reset_running_stats()
        b.momentum = None                     # cumulative average over the calibration pass
    for i, (x, y) in enumerate(calib_loader):
        if i >= calib_batches:
            break
        x, y = x.to(device), y.to(device)
        if mode in ("adv", "mix"):
            m.eval()
            x_adv = pgd_attack_v2(m, x, y, **attack)
        m.train()
        with torch.no_grad():
            if mode == "clean":
                m(x)
            elif mode == "adv":
                m(x_adv)
            else:
                m(x)
                m(x_adv)
    for b in bn_layers(m):
        b.momentum = 0.1
    return m


@torch.no_grad()
def train_mode_batchstats_acc(model, testloader):
    m = copy.deepcopy(model)
    for b in bn_layers(m):
        b.momentum = 0.0
    m.train()
    correct = total = 0
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        correct += (m.base_model(x).argmax(1) == y).sum().item()
        total += len(y)
    return 100.0 * correct / total


def diagnose(ckpt, arch="resnet18", eps=0.031373, steps=10, robust_n=2000, calib_batches=60, plan=DEFAULT_PLAN,
             policy="frozen"):
    cfg = process_yaml_file(str(plan))
    setup = process_experiment_ft_setup(cfg)
    ds = setup["Dataset"]
    group_a = create_or_load_group_A(dataset=ds.train_set, save_dir=f'./Indices/{cfg["Dataset"]["name"]}/',
                                     group_size=setup["GroupSize"], num_classes=setup["NumClasses"],
                                     seed=42, force_rebuild=False)
    group_b = create_or_load_group_B(save_dir=f'./Indices/{cfg["Dataset"]["name"]}/', overlap_rate=0.0,
                                     group_A_indices=group_a, dataset=ds.train_set, group_size=setup["GroupSize"],
                                     num_classes=setup["NumClasses"], seed=42, force_rebuild=False)
    calib_loader = DataLoader(ds.subset("raw_train", group_b), batch_size=128, shuffle=True,
                              generator=torch.Generator().manual_seed(0), num_workers=0)
    testloader = DataLoader(ds.test_set, batch_size=128, shuffle=False, num_workers=0)
    raw_test = ds.raw_test_set
    if robust_n and robust_n < len(raw_test):
        raw_test = Subset(raw_test, list(range(robust_n)))
    raw_testloader = DataLoader(raw_test, batch_size=128, shuffle=False, num_workers=0)
    attack = {"eps": eps, "steps": steps}
    crit = nn.CrossEntropyLoss()

    net = build_net(arch, setup["NumClasses"]).to(device)
    state = load_state(ckpt)
    target = net.student if arch == "resnet18_dist" else net
    target.load_state_dict(state, strict=True)
    model = NormalizedModel(net, ds.mean, ds.std).to(device).eval()

    def report(tag, m):
        m.eval()
        clean = evaluate1(m.base_model, testloader, crit, device)["test_acc"]
        rob = 100.0 * compute_robust_test_accuracy(m, raw_testloader, pgd_attack_v2, attack)
        print(f"{tag:<46s} clean {clean:6.2f}%   robust {rob:6.2f}%")
        return clean, rob

    print(f"checkpoint: {ckpt}\narch={arch}  eps={eps}  steps={steps}  robust on {len(raw_test)} test images")
    c1, r1 = report("1. running stats as saved (deployed model)", model)
    tm = train_mode_batchstats_acc(model, testloader)
    print(f"{'2. train-mode batch statistics on test set':<46s} clean {tm:6.2f}%   (gap vs deployed {tm - c1:+.1f} pt)")
    report("3. BN recalibrated on clean train images", recalibrate(model, calib_loader, "clean", attack, calib_batches))
    report("4. BN recalibrated on adversarial images", recalibrate(model, calib_loader, "adv", attack, calib_batches))
    c5, r5 = report("5. BN recalibrated on clean+adv mix", recalibrate(model, calib_loader, "mix", attack,
                                                                        calib_batches))
    if policy == "joint":
        # joint models are trained on mixed clean+adv batches, so their running statistics are
        # mixture statistics: row 2 (clean-only test batches) is not their training condition,
        # row 5 is. Valid when re-estimating the mixture statistics reproduces the deployed model.
        ok = abs(c5 - c1) <= 3.0 and abs(r5 - r1) <= 3.0
        verdict = "PASS" if ok else "WARN: deployed model differs from mix-recalibrated statistics by > 3 pt"
    else:
        verdict = "PASS" if abs(tm - c1) <= 3.0 else "WARN: train/eval gap > 3 pt (batch-statistics switch?)"
    print(f"verdict ({policy}): {verdict}")
    return {"clean": c1, "robust": r1, "train_mode_clean": tm, "gap": tm - c1, "mix_clean": c5, "mix_robust": r5}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", required=True, nargs="+", help="one or more state_dict paths")
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet18_dist"])
    p.add_argument("--eps", type=float, default=0.031373)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--robust-n", type=int, default=2000)
    p.add_argument("--calib-batches", type=int, default=60)
    p.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    p.add_argument("--policy", default="frozen", choices=["frozen", "both", "joint"],
                   help="BN policy the checkpoint was trained with; selects the verdict rule")
    a = p.parse_args()
    import os
    os.chdir(ROOT)
    for ck in a.ckpt:
        diagnose(ck, a.arch, a.eps, a.steps, a.robust_n, a.calib_batches, a.plan, a.policy)
        print()
