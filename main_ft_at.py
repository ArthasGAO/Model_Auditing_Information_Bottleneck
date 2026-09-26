"""Fine-tuning with an added adversarial term (one-stage AT for the FT family).

Same attack as main_ft.py (victim checkpoint -> FT-LL / FT-AL / RT-AL on the
disjoint 25k auxiliary split, the plan's optimizer / lr / epochs / checkpoint
rule), with one extra loss term per batch:

    L = CE(f(x), y) + lambda * CE(f(x_adv), y)         (lambda = 1 by default)

    x_adv = PGD / FGSM against the model being fine-tuned, maximising CE(., y).
    The weights are used as given: no normalisation, no rescaling.
    BatchNorm policy: AdversarialTraining.bn_policy (default "frozen": every
    BatchNorm stays in eval mode for both forwards, running statistics fixed
    to the victim's, affine parameters train; see at_family_common.py for the
    diagnosis that rules out "both" and "clean_eval" in this fine-tuning regime).

Reused unchanged from main_ft.py / util.py
  * process_experiment_ft_setup (plan parsing, per-strategy optimizer/scheduler),
    setup_finetune (freeze / re-init rules), set_backbone_eval_norm_dropout
    (FT-LL keeps the frozen backbone's BN in eval), sanity_check_finetune,
    evaluate1 on the normalised test set, BEST_CKPT_START_FRAC and the
    best_epoch.pth rule (highest clean Test_Acc after 30% of the epochs);
  * the FT data are the same group_B(0.0) indices determine_ft_dataset uses.

Deliberate deviations
  * the FT loader serves raw [0,1] images (subset "raw_train": crop+flip, no
    Normalize) and the network is wrapped in NormalizedModel, so PGD's eps is
    in pixel units exactly as in main_at.py; checkpoints are saved UNWRAPPED,
    in the same format as ft_final, so calculate_MI_ft.py can read them;
  * robust accuracy (same attack, ground-truth labels, raw test set) is logged
    at epoch -1 and every epoch; best_rob_epoch.pth is saved alongside;
  * only the plain-CIFAR FT_Dataset branch is supported (not PseudoLabel /
    CIFARNet); CNN models only (no DeiT);
  * DETERMINISTIC=False (cudnn.benchmark) as in main_at.py.

Outputs
  saved_models/ft_at/<scenario>/{best_epoch,best_rob_epoch,epoch_N}.pth
  saved_logs/ft_at/Performance/training_log_<scenario>.csv
  <scenario> = <ft scenario name>_AT<attack>_<params>_lambda=<l>_run=<tag>

Usage
  python main_ft_at.py --check-only
  python main_ft_at.py --seeds 0 --strategies FT-AL --epochs 2 --max-batches 30 --robust-n 1000 --run-tag smoke --force
  python main_ft_at.py --seeds 0 1 2
"""
import argparse
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

from at_family_common import (AT_EXTRA_COLUMNS, at_suffix, batch_accuracy, make_adv_generator,
                              mixed_adversarial_step, parse_additive_at_config)
from main_ft import BEST_CKPT_START_FRAC, STRATEGY
from util import (create_or_load_group_A, create_or_load_group_B, evaluate1, load_best_checkpoint,
                  process_experiment_ft_setup, process_yaml_file, sanity_check_finetune,
                  set_backbone_eval_norm_dropout, setup_finetune, wait_for_cool_gpu)
from util_adv import NormalizedModel, compute_robust_test_accuracy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = Path(__file__).resolve().parent
DETERMINISTIC = False
SEED_START = int(os.environ.get("SEED_START", 0))
SEED_END = int(os.environ.get("SEED_END", 3))

FT_AT_ROOT = './saved_models/ft_at'
FT_AT_LOG_ROOT = './saved_logs/ft_at/Performance'
DEFAULT_PLAN_DIR = ROOT / "saved_exp_plan/ft_at_plan"

FT_COLUMNS = ['Scenario', 'Epoch', 'Train_Loss', 'Train_Acc', 'Train_Precision', 'Train_Recall',
              'Train_F1', 'Test_Loss', 'Test_Acc', 'Test_Precision', 'Test_Recall', 'Test_F1']
LOG_COLUMNS = FT_COLUMNS + AT_EXTRA_COLUMNS


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
# 1. One epoch: FT loss on x plus lambda * FT loss on x_adv
# =====================================================
def ft_at_one_epoch(model, base_net, train_loader, optimizer, criterion, adv_generator, strategy,
                    clean_weight=1.0, adv_weight=1.0, device='cuda', max_batches=None, bn_policy="frozen"):
    """One fine-tuning epoch with the added adversarial term.

    `model` is NormalizedModel(base_net). Train_Loss is the summed objective;
    Train_Acc / precision / recall / F1 are measured on the CLEAN forward so the
    leading columns stay comparable with ft_final logs (adv_weight == 0
    reproduces util.ft_one_epoch exactly; test_ft_at.py checks that). In pure
    adversarial mode (clean_weight == 0) they fall back to the adversarial forward.
    """
    hook = (lambda: set_backbone_eval_norm_dropout(base_net)) if strategy == "FT-LL" else None
    running_loss, total, clean_correct, adv_correct = 0.0, 0, 0, 0
    all_preds, all_targets = [], []

    for batch_idx, (images, targets) in enumerate(train_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = mixed_adversarial_step(model, images, targets, criterion, adv_generator,
                                     clean_weight, adv_weight, post_train_hook=hook, bn_policy=bn_policy)
        out["loss"].backward()
        optimizer.step()

        n = targets.size(0)
        running_loss += out["loss"].item() * n
        total += n
        if out["clean_logits"] is not None:
            clean_correct += batch_accuracy(out["clean_logits"], targets)
        if out["adv_logits"] is not None:
            adv_correct += batch_accuracy(out["adv_logits"], targets)
        report = out["clean_logits"] if out["clean_logits"] is not None else out["adv_logits"]
        all_preds.append(report.argmax(1).detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    from sklearn.metrics import precision_score, recall_score, f1_score
    has_clean = clean_weight > 0
    return {
        "train_loss": running_loss / total,
        "train_acc": 100.0 * (clean_correct if has_clean else adv_correct) / total,
        "train_precision": precision_score(all_targets, all_preds, average="weighted", zero_division=0),
        "train_recall": recall_score(all_targets, all_preds, average="weighted", zero_division=0),
        "train_f1": f1_score(all_targets, all_preds, average="weighted", zero_division=0),
        "train_clean_acc": (100.0 * clean_correct / total) if has_clean else float('nan'),
        "train_adv_acc": (100.0 * adv_correct / total) if adv_weight > 0 else float('nan'),
    }


# =====================================================
# 2. Naming, data, preflight
# =====================================================
def ft_scenario_name(exp_yaml, exp_setup, model_seed, r, strategy, ft_seed):
    """Identical to main_ft.py's scenario name."""
    return (f'{exp_yaml["Scenario_Name"]}_{model_seed}_{round(r, 2)}'
            f'_{strategy}_ftsize={exp_setup["FT_GroupSize"]}_ftseed={ft_seed}')


def build_ft_raw_data(exp_yaml, exp_setup, group_A):
    """The group_B(0.0) split of determine_ft_dataset, served raw for PGD."""
    name = exp_yaml["FT_Dataset"]["name"]
    if name in ("PseudoLabelCIFAR-10", "CIFARNet"):
        raise NotImplementedError(f"FT_Dataset {name} is not supported by main_ft_at.py")
    dataset_obj = exp_setup["Dataset"]
    group_B = create_or_load_group_B(
        save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/', overlap_rate=0.0,
        group_A_indices=group_A, dataset=dataset_obj.train_set,
        group_size=exp_setup["FT_GroupSize"], num_classes=exp_setup["NumClasses"],
        seed=42, force_rebuild=False)
    # The raw path must describe the same images as the normalised path main_ft uses.
    mean = torch.tensor(dataset_obj.mean).view(3, 1, 1)
    std = torch.tensor(dataset_obj.std).view(3, 1, 1)
    norm_clean = dataset_obj.subset("train", group_B, clean=True)
    raw_clean = dataset_obj.subset("raw_train_clean", group_B)
    for i in [0, len(group_B) // 2, len(group_B) - 1]:
        (xn, yn), (xr, yr) = norm_clean[i], raw_clean[i]
        assert int(yn) == int(yr) and torch.allclose((xr - mean) / std, xn, atol=1e-5), \
            f"raw/normalised FT data misaligned at {i}"
    return dataset_obj.subset("raw_train", group_B), dataset_obj.test_set, dataset_obj.raw_test_set, group_B


def expand_plan(yaml_path, seeds, strategies, run_tag, model_seed=42, r=1.0):
    with Path(yaml_path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    attacks, _, lam, bn_policy = parse_additive_at_config(cfg)
    plan_strategies = [o["strategy"] for o in cfg.get("Optimizers", [])]
    names = []
    for s in strategies:
        if s not in plan_strategies:
            raise ValueError(f"{Path(yaml_path).name}: no Optimizers entry for strategy {s}")
    ft_size = cfg["FT_Dataset"]["group_size"]
    for seed in seeds:
        for strategy in strategies:
            for _, attack_name, kwargs in attacks:
                base = (f'{cfg["Scenario_Name"]}_{model_seed}_{round(r, 2)}'
                        f'_{strategy}_ftsize={ft_size}_ftseed={seed}')
                names.append(base + at_suffix(attack_name, kwargs, lam, run_tag, bn_policy))
    return cfg, names


# =====================================================
# 3. Main
# =====================================================
def main_ft_at(model_seed, ft_seed, r, yaml_file_path, *, strategies=None, run_tag="v1",
               epochs_override=None, max_batches=None, robust_n=None, force=False):
    print(f"Device: {device}")
    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_ft_setup(exp_yaml)
    attack_configs, clean_w, adv_w, bn_policy = parse_additive_at_config(exp_yaml)
    strategies = strategies or STRATEGY
    print(f"Additive AT: L = {clean_w:g} * CE(x) + {adv_w:g} * CE(x_adv); attacks: {len(attack_configs)};"
          f" bn_policy={bn_policy}")

    # ----- Victim checkpoint (identical to main_ft) -----
    model_name = exp_yaml["Model_Name"] + f"_{model_seed}_{round(r, 2)}"
    model_folder = Path('./saved_models/vanilla/CNN_Models') / model_name
    ckpt_path, _ = load_best_checkpoint(model_folder)
    if ckpt_path is None:
        raise FileNotFoundError(f"No .pth files found in {model_folder}")

    # ----- Data -----
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
    testloader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2,
                            worker_init_fn=seed_worker, generator=g, persistent_workers=True, pin_memory=True)
    if robust_n is not None and robust_n < len(raw_test_set):
        raw_test_set = Subset(raw_test_set, list(range(robust_n)))
    robust_n_used = len(raw_test_set)
    raw_testloader = DataLoader(raw_test_set, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    criterion = nn.CrossEntropyLoss()
    os.makedirs(FT_AT_LOG_ROOT, exist_ok=True)

    for strategy in strategies:
        if strategy not in exp_setup["FT_Setups"]:
            raise KeyError(f"No fine-tune config found for strategy={strategy}")
        ft_cfg = exp_setup["FT_Setups"][strategy]
        epochs = int(epochs_override) if epochs_override is not None else int(ft_cfg["Epochs"])

        for attack_fn, attack_name, attack_kwargs in attack_configs:
            scenario_name = ft_scenario_name(exp_yaml, exp_setup, model_seed, r, strategy, ft_seed) \
                + at_suffix(attack_name, attack_kwargs, adv_w, run_tag, bn_policy)
            save_dir = Path(FT_AT_ROOT) / scenario_name
            best_ckpt_path = save_dir / 'best_epoch.pth'
            log_file = Path(FT_AT_LOG_ROOT) / f"training_log_{scenario_name}.csv"
            if (best_ckpt_path.is_file() and best_ckpt_path.stat().st_size > 0) or log_file.exists():
                if not force:
                    print(f"[SKIP] {scenario_name}: outputs already exist (use --force or a new --run-tag)")
                    continue
                print(f"[FORCE] overwriting outputs of {scenario_name}")
                if log_file.exists():
                    log_file.unlink()
            print(f"\n{'=' * 60}\nStrategy: {strategy} | {attack_name} {attack_kwargs} | lambda={adv_w:g}"
                  f" | bn={bn_policy}\nScenario: {scenario_name}\n{'=' * 60}")

            # ----- Fresh model -> load victim -> strategy setup -> wrap (identical order to main_ft) -----
            net = exp_setup["Model_Factory"]().to(device)
            net.load_state_dict(torch.load(ckpt_path, map_location=device))
            net = setup_finetune(model=net, strategy=strategy, device=device)
            model = NormalizedModel(net, dataset_obj.mean, dataset_obj.std).to(device)
            adv_generator = make_adv_generator(attack_fn, attack_kwargs)

            os.makedirs(save_dir, exist_ok=True)
            with open(log_file, 'w', newline='') as f:
                csv.writer(f).writerow(LOG_COLUMNS)

            def log_row(epoch, train_result, test_result, rob_acc, lr_now, seconds):
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
                        robust_n_used,
                    ])

            # ----- Pre-FT evaluation (epoch -1), as in main_ft, plus robust accuracy -----
            pre = evaluate1(net, testloader, criterion, device)
            pre_rob = 100.0 * compute_robust_test_accuracy(model, raw_testloader, attack_fn, attack_kwargs)
            print(f"  [Epoch -1] Test clean {pre['test_acc']:.2f}% | Test robust {pre_rob:.2f}%")
            log_row(-1, None, pre, pre_rob, 0, 0)

            optimizer = getattr(optim, ft_cfg["Optimizer_Name"])(
                filter(lambda p: p.requires_grad, net.parameters()), **ft_cfg["Optimizer_Params"])
            scheduler = None
            if ft_cfg["Scheduler_Name"] is not None:
                s_params = ft_cfg["Scheduler_Params"].copy()
                if s_params.get("T_max") == "auto":
                    s_params["T_max"] = epochs
                scheduler = getattr(lr_sched, ft_cfg["Scheduler_Name"])(optimizer, **s_params)

            best_test_acc, best_rob_acc = -1.0, -1.0
            best_ckpt_from = int(epochs * BEST_CKPT_START_FRAC)
            for epoch in range(epochs):
                t0 = time.time()
                train_result = ft_at_one_epoch(model, net, trainloader, optimizer, criterion, adv_generator,
                                               strategy, clean_w, adv_w, device, max_batches, bn_policy)
                wait_for_cool_gpu(threshold=89.5)
                lr_now = optimizer.param_groups[0]['lr']
                if scheduler is not None:
                    scheduler.step()
                test_result = evaluate1(net, testloader, criterion, device)
                rob_acc = 100.0 * compute_robust_test_accuracy(model, raw_testloader, attack_fn, attack_kwargs)
                seconds = time.time() - t0
                print(f"  [Epoch {epoch}/{epochs}] LR {lr_now:.5f} | Loss {train_result['train_loss']:.4f}"
                      f" | Train clean {train_result['train_clean_acc']:.2f}% adv {train_result['train_adv_acc']:.2f}%"
                      f" | Test clean {test_result['test_acc']:.2f}% | Test robust {rob_acc:.2f}% | {seconds:.0f}s")
                log_row(epoch, train_result, test_result, rob_acc, lr_now, seconds)

                if epoch >= best_ckpt_from and test_result["test_acc"] > best_test_acc:
                    best_test_acc = test_result["test_acc"]
                    torch.save(net.state_dict(), best_ckpt_path)
                if epoch >= best_ckpt_from and rob_acc > best_rob_acc:
                    best_rob_acc = rob_acc
                    torch.save(net.state_dict(), save_dir / 'best_rob_epoch.pth')

            torch.save(net.state_dict(), save_dir / f'epoch_{epoch}.pth')
            sanity_check_finetune(net, ckpt_path, strategy, device)
            print(f"==> {strategy} done. Best clean {best_test_acc:.2f}% | best robust {best_rob_acc:.2f}%"
                  f" (window from epoch {best_ckpt_from}); pre-FT clean {pre['test_acc']:.2f}% robust {pre_rob:.2f}%")


# =====================================================
# 4. Entry point
# =====================================================
def main(argv=None):
    parser = argparse.ArgumentParser(description="Fine-tuning with an added adversarial loss term.")
    parser.add_argument("--plans", nargs="+", type=Path,
                        help="YAML plans; default: all saved_exp_plan/ft_at_plan/*.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, help="ft seeds; default SEED_START..SEED_END.")
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--rate", type=float, default=1.0, help="Victim overlap rate suffix (1.0).")
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGY), choices=list(STRATEGY))
    parser.add_argument("--run-tag", default="v1")
    parser.add_argument("--epochs", type=int, help="Override every strategy's epoch budget (smoke tests).")
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

    print(f"Plans: {[str(p) for p in yaml_files]}; seeds: {seeds}; strategies: {args.strategies}; run tag: {args.run_tag}")
    for path in yaml_files:
        _, names = expand_plan(path, seeds, args.strategies, args.run_tag, args.model_seed, args.rate)
        for n in names:
            exists = (Path(FT_AT_ROOT) / n / 'best_epoch.pth').is_file()
            print(f"  {'[exists] ' if exists else ''}{n}")
    if args.check_only:
        print("Preflight passed. No training started.")
        return

    torch.multiprocessing.set_start_method("spawn", force=True)
    for yaml_path in yaml_files:
        for seed in seeds:
            print(f"\n>>> ft seed {seed} for {yaml_path.name}")
            main_ft_at(args.model_seed, seed, args.rate, str(yaml_path), strategies=args.strategies,
                       run_tag=args.run_tag, epochs_override=args.epochs, max_batches=args.max_batches,
                       robust_n=args.robust_n, force=args.force)


if __name__ == "__main__":
    main()
