"""Pruning with an added adversarial term in the recovery fine-tuning (one-stage AT, pruning family).

Same attack as main_prune.py (victim checkpoint -> global L1 magnitude pruning
of every Conv2d/Linear weight at the plan's sparsity -> FT-AL recovery on the
disjoint 25k auxiliary split with the plan's optimizer / lr / epochs, best
checkpoint = highest clean Test_Acc after 20% of the epochs), with one extra
loss term per batch of the recovery:

    L = CE(f(x), y) + lambda * CE(f(x_adv), y)        (lambda = 1 by default)

    x_adv = PGD / FGSM against the pruned model being recovered, maximising CE.
    The pruning masks (torch.nn.utils.prune reparameterisation) stay in place
    during the whole recovery, so the sparsity is preserved exactly.

BatchNorm policy: AdversarialTraining.bn_policy, default "frozen" (the model
starts from the victim, see at_family_common.py): the running statistics stay
the victim's, exactly the state main_prune.py's recovery starts from.
AdversarialTraining.recalibrate_bn_after_prune (default false) optionally
recomputes them ONCE on the clean auxiliary images right after pruning before
freezing. Measured at sparsity 0.8 / eps 8/255 / 10 epochs: recalibration lowers
the pre-recovery accuracy (77.2% vs 82.3%; weights and statistics are
co-adapted) and the recovered models are indistinguishable (80.8/35.8 vs
80.8/35.4 clean/robust), so it is off. The scenario name records the choice as
_recal=none / _recal=clean.

Reused unchanged from main_prune.py / util.py
  * process_experiment_prune_setup, sparsity_levels_from_setup, prune_model_global,
    check_pruned_weights, remove_prune_mask, setup_finetune, BEST_CKPT_START_FRAC (0.2),
    prune_scenario_name, the group_B(0.0) FT data, evaluate1 on the normalised test set,
    epoch_-1.pth (pruned, before recovery), best_epoch.pth, epoch_<last>.pth saved WITHOUT
    masks (remove_prune_mask on a copy), i.e. in the pruning_final format;
  * the epoch loop is main_ft_at.ft_at_one_epoch (shared template).

Deliberate deviations
  * raw [0,1] images + NormalizedModel wrapper (eps in pixel units); robust accuracy and
    the measured global sparsity logged every epoch; best_rob_epoch.pth added;
  * CNN plans only (no DeiT); DETERMINISTIC=False as in main_at.py.

Outputs
  saved_models/pruning_at/<scenario>/{epoch_-1,best_epoch,best_rob_epoch,epoch_N}.pth
  saved_logs/pruning_at/Performance/training_log_<scenario>.csv
  <scenario> = <pruning scenario name>_AT<attack>_<params>_lambda=<l>_bn=<policy>_recal=<clean|none>_run=<tag>

Usage
  python main_prune_at.py --check-only --sparsity 0.8 --seeds 0
  python main_prune_at.py --seeds 0 --sparsity 0.8 --eps 0.031373 --epochs 2 --max-batches 30 --robust-n 1000 --run-tag smoke --force
  python main_prune_at.py --seeds 0 --sparsity 0.8 --eps 0.007843 0.015686 0.031373
"""
import argparse
import copy
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import yaml
from torch.utils.data import DataLoader, Subset

from at_family_common import (AT_EXTRA_COLUMNS, at_suffix, filter_attacks, make_adv_generator,
                              parse_additive_at_config)
from main_ft_at import build_ft_raw_data, ft_at_one_epoch
from main_prune import BEST_CKPT_START_FRAC, prune_scenario_name
from util import (check_pruned_weights, create_or_load_group_A, evaluate1, load_best_checkpoint,
                  process_experiment_prune_setup, process_yaml_file, prune_model_global, remove_prune_mask,
                  setup_finetune, sparsity_levels_from_setup, wait_for_cool_gpu)
from util_adv import NormalizedModel, compute_robust_test_accuracy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = Path(__file__).resolve().parent
DETERMINISTIC = False
SEED_START = int(os.environ.get("SEED_START", 0))
SEED_END = int(os.environ.get("SEED_END", 3))
STRATEGY = "FT-AL"            # main_prune.py: recovery always fine-tunes all layers

PRUNE_AT_ROOT = './saved_models/pruning_at'
PRUNE_AT_LOG_ROOT = './saved_logs/pruning_at/Performance'
DEFAULT_PLAN_DIR = ROOT / "saved_exp_plan/prune_at_plan"

PRUNE_COLUMNS = ['Scenario', 'Epoch', 'Train_Loss', 'Train_Acc', 'Train_Precision', 'Train_Recall',
                 'Train_F1', 'Test_Loss', 'Test_Acc', 'Test_Precision', 'Test_Recall', 'Test_F1']
LOG_COLUMNS = PRUNE_COLUMNS + AT_EXTRA_COLUMNS + ['Global_Sparsity', 'AT_recal']


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = deterministic
    cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)


# =====================================================
# 1. Pruning-mask helpers
# =====================================================
@torch.no_grad()
def refresh_pruned_weights(net):
    """Re-materialise weight = weight_orig * weight_mask as a leaf tensor.

    prune's forward pre-hook rebuilds `weight` on every forward; after a
    grad-enabled forward (training, PGD) it is a non-leaf tensor and
    copy.deepcopy raises. Calling this before every deepcopy/save removes the
    dependence on the order of evaluation calls (see pruning-path notes).
    """
    for m in net.modules():
        if hasattr(m, "weight_orig") and hasattr(m, "weight_mask"):
            m.weight = m.weight_orig * m.weight_mask


def global_sparsity(net):
    """Fraction of exactly-zero weights over all Conv2d/Linear weights (masked view)."""
    total = zero = 0
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                w = m.weight
                total += w.numel()
                zero += (w == 0).sum().item()
    return zero / total if total else float('nan')


def dense_state_dict(net):
    """State dict of a mask-free copy (the pruning_final checkpoint format)."""
    refresh_pruned_weights(net)
    tmp = copy.deepcopy(net)
    remove_prune_mask(tmp)
    state = tmp.state_dict()
    del tmp
    return state


@torch.no_grad()
def recalibrate_bn(model, loader, device, max_batches=None):
    """Recompute BN running statistics on clean images (cumulative average, no gradient).

    Leaves the model in eval mode with momentum restored to 0.1.
    """
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for b in bns:
        b.reset_running_stats()
        b.momentum = None
    model.train()
    n = 0
    for i, (x, _) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        model(x.to(device, non_blocking=True))
        n += 1
    for b in bns:
        b.momentum = 0.1
    model.eval()
    return n


# =====================================================
# 2. Config, naming, preflight
# =====================================================
def parse_recal(exp_yaml):
    at_cfg = exp_yaml.get("AdversarialTraining") or {}
    flag = at_cfg.get("recalibrate_bn_after_prune", False)
    if not isinstance(flag, bool):
        raise ValueError("AdversarialTraining.recalibrate_bn_after_prune must be true or false.")
    return flag


def prune_at_scenario_name(exp_yaml, exp_setup, model_seed, r, sparsity, ft_seed, attack_name, attack_kwargs,
                           lam, bn_policy, recal, run_tag):
    base = prune_scenario_name(exp_yaml["Scenario_Name"], sparsity, model_seed, r, ft_seed,
                               exp_setup["FT_GroupSize"], STRATEGY)
    return base + at_suffix(attack_name, attack_kwargs, lam, run_tag, bn_policy,
                            extra={"recal": "clean" if recal else "none"})


def expand_plan(yaml_path, seeds, sparsities, run_tag, eps_filter=None, model_seed=42, r=1.0):
    with Path(yaml_path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    attacks, _, lam, bn_policy = parse_additive_at_config(cfg)
    attacks = filter_attacks(attacks, eps_filter)
    recal = parse_recal(cfg)
    plan_sparsities = [round(float(o["sparsity"]), 6) for o in cfg.get("Optimizers", []) or []]
    names = []
    for s in sparsities:
        s_key = round(float(s), 6)
        if s_key not in plan_sparsities:
            raise ValueError(f"{Path(yaml_path).name}: no Optimizers entry for sparsity {s}")
        base = prune_scenario_name(cfg["Scenario_Name"], s_key, model_seed, r, 0, cfg["FT_Dataset"]["group_size"], STRATEGY)
        for seed in seeds:
            base = prune_scenario_name(cfg["Scenario_Name"], s_key, model_seed, r, seed,
                                       cfg["FT_Dataset"]["group_size"], STRATEGY)
            for _, name, kwargs in attacks:
                names.append(base + at_suffix(name, kwargs, lam, run_tag, bn_policy,
                                              extra={"recal": "clean" if recal else "none"}))
    return cfg, names


# =====================================================
# 3. Main
# =====================================================
def main_prune_at(model_seed, ft_seed, r, yaml_file_path, *, sparsities=(0.8,), run_tag="v1", eps_filter=None,
                  epochs_override=None, max_batches=None, robust_n=None, force=False):
    print(f"Device: {device}")
    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_prune_setup(exp_yaml)
    attack_configs, clean_w, adv_w, bn_policy = parse_additive_at_config(exp_yaml)
    attack_configs = filter_attacks(attack_configs, eps_filter)
    recal = parse_recal(exp_yaml)
    prune_exclude = exp_yaml.get("Prune_Exclude") or None
    print(f"Additive AT: L = {clean_w:g} * CE(x) + {adv_w:g} * CE(x_adv); attacks: {len(attack_configs)};"
          f" bn_policy={bn_policy}; recalibrate BN after pruning={recal}")

    # ----- Victim checkpoint (identical to main_prune) -----
    model_name = exp_yaml["Model_Name"] + f"_{model_seed}_{round(r, 2)}"
    model_folder = Path('./saved_models/vanilla/CNN_Models') / model_name
    ckpt_path, _ = load_best_checkpoint(model_folder)
    if ckpt_path is None:
        raise FileNotFoundError(f"No .pth files found in {model_folder}")
    ckpt_state = torch.load(ckpt_path, map_location=device)

    # ----- Data: same group_B(0.0) indices as main_prune, served raw -----
    set_seed(ft_seed, deterministic=DETERMINISTIC)
    g = torch.Generator()
    g.manual_seed(ft_seed)
    dataset_obj = exp_setup["Dataset"]
    group_A = create_or_load_group_A(dataset=dataset_obj.train_set,
                                     save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
                                     group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"],
                                     seed=42, force_rebuild=False)
    ft_train_raw, test_set, raw_test_set, group_B = build_ft_raw_data(exp_yaml, exp_setup, group_A)
    print(f"FT data: {len(ft_train_raw)} raw images (group_B 0.0); {len(set(group_B) & set(group_A))} overlap with group_A")
    trainloader = DataLoader(ft_train_raw, batch_size=128, shuffle=True, num_workers=4,
                             worker_init_fn=seed_worker, generator=g, persistent_workers=True, pin_memory=True)
    calib_loader = DataLoader(ft_train_raw, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    testloader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2,
                            worker_init_fn=seed_worker, generator=g, persistent_workers=True, pin_memory=True)
    if robust_n is not None and robust_n < len(raw_test_set):
        raw_test_set = Subset(raw_test_set, list(range(robust_n)))
    robust_n_used = len(raw_test_set)
    raw_testloader = DataLoader(raw_test_set, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    criterion = nn.CrossEntropyLoss()
    os.makedirs(PRUNE_AT_LOG_ROOT, exist_ok=True)

    plan_levels = sparsity_levels_from_setup(exp_setup, None)
    for s in sparsities:
        s_key = round(float(s), 6)
        if s_key not in exp_setup["Prune_Setups"] or s_key not in [round(float(x), 6) for x in plan_levels]:
            raise KeyError(f"No pruning config found for sparsity={s_key}")
        prune_cfg = exp_setup["Prune_Setups"][s_key]
        epochs = int(epochs_override) if epochs_override is not None else int(prune_cfg["Epochs"])

        for attack_fn, attack_name, attack_kwargs in attack_configs:
            scenario_name = prune_at_scenario_name(exp_yaml, exp_setup, model_seed, r, s_key, ft_seed, attack_name,
                                                   attack_kwargs, adv_w, bn_policy, recal, run_tag)
            save_dir = Path(PRUNE_AT_ROOT) / scenario_name
            best_ckpt_path = save_dir / 'best_epoch.pth'
            log_file = Path(PRUNE_AT_LOG_ROOT) / f"training_log_{scenario_name}.csv"
            if (best_ckpt_path.is_file() and best_ckpt_path.stat().st_size > 0) or log_file.exists():
                if not force:
                    print(f"[SKIP] {scenario_name}: outputs already exist (use --force or a new --run-tag)")
                    continue
                print(f"[FORCE] overwriting outputs of {scenario_name}")
                if log_file.exists():
                    log_file.unlink()
            print(f"\n{'=' * 60}\nSparsity {s_key} | {attack_name} {attack_kwargs} | lambda={adv_w:g} | bn={bn_policy}"
                  f" | recal={'clean' if recal else 'none'}\nScenario: {scenario_name}\n{'=' * 60}")

            # ----- Fresh model -> load victim -> prune -> FT-AL setup (identical to main_prune) -----
            set_seed(ft_seed, deterministic=DETERMINISTIC)
            g.manual_seed(ft_seed)
            net = exp_setup["Model_Factory"]().to(device)
            net.load_state_dict(ckpt_state)
            net = prune_model_global(model=net, amount=s_key, exclude_patterns=prune_exclude)
            net = setup_finetune(model=net, strategy=STRATEGY, device=device)
            check_pruned_weights(net, exclude_patterns=prune_exclude)
            model = NormalizedModel(net, dataset_obj.mean, dataset_obj.std).to(device)
            adv_generator = make_adv_generator(attack_fn, attack_kwargs)

            if recal:
                n_cal = recalibrate_bn(model, calib_loader, device)
                print(f"[BN] running statistics recalibrated on {n_cal} clean batches after pruning")
            model.eval()

            os.makedirs(save_dir, exist_ok=True)
            with open(log_file, 'w', newline='') as f:
                csv.writer(f).writerow(LOG_COLUMNS)

            def log_row(epoch, train_result, test_result, rob_acc, lr_now, seconds, sparsity_now):
                tr = train_result or {}
                with open(log_file, 'a', newline='') as f:
                    csv.writer(f).writerow([
                        scenario_name, epoch,
                        tr.get("train_loss", 0), tr.get("train_acc", 0), tr.get("train_precision", 0),
                        tr.get("train_recall", 0), tr.get("train_f1", 0),
                        test_result["test_loss"], test_result["test_acc"], test_result["test_precision"],
                        test_result["test_recall"], test_result["test_f1"],
                        round(rob_acc, 4), round(tr.get("train_clean_acc", float('nan')), 4),
                        round(tr.get("train_adv_acc", float('nan')), 4), lr_now, round(seconds, 1),
                        attack_name, attack_kwargs.get("eps"), attack_kwargs.get("steps"), adv_w, bn_policy,
                        robust_n_used, round(sparsity_now, 6), "clean" if recal else "none",
                    ])

            # ----- Pre-recovery state (epoch -1): pruned model, saved dense as main_prune does -----
            pre = evaluate1(net, testloader, criterion, device)
            pre_rob = 100.0 * compute_robust_test_accuracy(model, raw_testloader, attack_fn, attack_kwargs)
            sp = global_sparsity(net)
            print(f"  [Epoch -1] Test clean {pre['test_acc']:.2f}% | Test robust {pre_rob:.2f}% | sparsity {sp:.4f}")
            log_row(-1, None, pre, pre_rob, 0, 0, sp)
            torch.save(dense_state_dict(net), save_dir / 'epoch_-1.pth')

            optimizer = getattr(optim, prune_cfg["Optimizer_Name"])(
                filter(lambda p: p.requires_grad, net.parameters()), **prune_cfg["Optimizer_Params"])
            scheduler = None
            if prune_cfg["Scheduler_Name"] is not None:
                s_params = prune_cfg["Scheduler_Params"].copy()
                if s_params.get("T_max") == "auto":
                    s_params["T_max"] = epochs
                scheduler = getattr(lr_sched, prune_cfg["Scheduler_Name"])(optimizer, **s_params)

            best_test_acc, best_rob_acc = -1.0, -1.0
            best_ckpt_from = int(epochs * BEST_CKPT_START_FRAC)
            for epoch in range(epochs):
                t0 = time.time()
                train_result = ft_at_one_epoch(model, net, trainloader, optimizer, criterion, adv_generator,
                                               STRATEGY, clean_w, adv_w, device, max_batches, bn_policy)
                wait_for_cool_gpu(threshold=89.5)
                lr_now = optimizer.param_groups[0]['lr']
                if scheduler is not None:
                    scheduler.step()
                test_result = evaluate1(net, testloader, criterion, device)
                rob_acc = 100.0 * compute_robust_test_accuracy(model, raw_testloader, attack_fn, attack_kwargs)
                sp = global_sparsity(net)
                seconds = time.time() - t0
                print(f"  [Epoch {epoch}/{epochs}] LR {lr_now:.5f} | Loss {train_result['train_loss']:.4f}"
                      f" | Train clean {train_result['train_clean_acc']:.2f}% adv {train_result['train_adv_acc']:.2f}%"
                      f" | Test clean {test_result['test_acc']:.2f}% | Test robust {rob_acc:.2f}%"
                      f" | sparsity {sp:.4f} | {seconds:.0f}s")
                log_row(epoch, train_result, test_result, rob_acc, lr_now, seconds, sp)

                if epoch >= best_ckpt_from and test_result["test_acc"] > best_test_acc:
                    best_test_acc = test_result["test_acc"]
                    torch.save(dense_state_dict(net), best_ckpt_path)
                if epoch >= best_ckpt_from and rob_acc > best_rob_acc:
                    best_rob_acc = rob_acc
                    torch.save(dense_state_dict(net), save_dir / 'best_rob_epoch.pth')

            torch.save(dense_state_dict(net), save_dir / f'epoch_{epoch}.pth')
            check_pruned_weights(net, exclude_patterns=prune_exclude)
            print(f"==> sparsity {s_key} done. Best clean {best_test_acc:.2f}% | best robust {best_rob_acc:.2f}%"
                  f" (window from epoch {best_ckpt_from}); pre-recovery clean {pre['test_acc']:.2f}% robust {pre_rob:.2f}%")


# =====================================================
# 4. Entry point
# =====================================================
def main(argv=None):
    parser = argparse.ArgumentParser(description="Pruning recovery with an added adversarial loss term.")
    parser.add_argument("--plans", nargs="+", type=Path, help="YAML plans; default: all saved_exp_plan/prune_at_plan/*.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, help="ft seeds; default SEED_START..SEED_END.")
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--sparsity", nargs="+", type=float, default=[0.8], help="Plan sparsities to run (default 0.8).")
    parser.add_argument("--eps", nargs="+", type=float, help="Only run the plan's attack configs with these eps.")
    parser.add_argument("--run-tag", default="v1")
    parser.add_argument("--epochs", type=int, help="Override the recovery epoch budget (smoke tests).")
    parser.add_argument("--max-batches", type=int, help="Smoke tests: train on at most N batches per epoch.")
    parser.add_argument("--robust-n", type=int, help="Robust accuracy on the first N test images (default all).")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)

    seeds = args.seeds if args.seeds is not None else list(range(SEED_START, SEED_END))
    if not seeds or len(seeds) != len(set(seeds)):
        parser.error("Seeds must be distinct and non-empty.")
    os.chdir(ROOT)
    yaml_files = args.plans or sorted(DEFAULT_PLAN_DIR.glob("*.yaml"))
    if not yaml_files:
        parser.error(f"No YAML plans found in {DEFAULT_PLAN_DIR}")

    print(f"Plans: {[str(p) for p in yaml_files]}; seeds: {seeds}; sparsity: {args.sparsity}; eps: {args.eps or 'all'};"
          f" run tag: {args.run_tag}")
    for path in yaml_files:
        _, names = expand_plan(path, seeds, args.sparsity, args.run_tag, args.eps, args.model_seed, args.rate)
        for n in names:
            exists = (Path(PRUNE_AT_ROOT) / n / 'best_epoch.pth').is_file()
            print(f"  {'[exists] ' if exists else ''}{n}")
    if args.check_only:
        print("Preflight passed. No training started.")
        return

    torch.multiprocessing.set_start_method("spawn", force=True)
    for yaml_path in yaml_files:
        for seed in seeds:
            print(f"\n>>> ft seed {seed} for {yaml_path.name}")
            main_prune_at(args.model_seed, seed, args.rate, str(yaml_path), sparsities=tuple(args.sparsity),
                          run_tag=args.run_tag, eps_filter=args.eps, epochs_override=args.epochs,
                          max_batches=args.max_batches, robust_n=args.robust_n, force=args.force)


if __name__ == "__main__":
    main()
