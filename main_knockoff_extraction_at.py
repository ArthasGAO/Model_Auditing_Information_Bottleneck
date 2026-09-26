"""Knockoff extraction with an added adversarial term (one-stage AT, extraction family).

Same attack as main_knockoff_extraction.py (victim queried once on the 25k
disjoint auxiliary images, substitute trained from scratch on the soft
labels with the plan's optimizer / lr / epochs), with one extra loss term
per batch:

    L = KL( f(x) || p_victim(x) ) + lambda * KL( f(x_adv) || p_victim(x) )

    x_adv = PGD / FGSM against the SUBSTITUTE f (never the black-box victim),
            maximising CE against argmax p_victim(x)   (inner_target=hard, default)
            or KL against p_victim(x)                  (inner_target=soft).
    Weights are used as given (no normalisation). The soft label of x_adv is
    the one queried on the clean x: the query budget is unchanged.

BatchNorm policy: AdversarialTraining.bn_policy, default "both" (both loss
forwards in train mode, BN updated by both). The substitute starts from
random weights, so the "frozen" policy of the fine-tuning families has no
statistics to freeze; "both" is the standard from-scratch practice
(Madry / TRADES / ARD). Every trained model must pass diagnose_bn_modes.py
(train-mode vs eval-mode gap, recalibration invariance) before it is used.

Reused unchanged from main_knockoff_extraction.py
  * victim loading, group_A / group_B index files, the clean query pass and
    the index-alignment check; the KL soft-label loss on clean inputs;
  * util.evaluate1 / util.evaluate_fidelity on the normalised test loader, so
    Test_Acc / Fidelity are on the same footing as the extraction_final logs.

Deliberate deviations
  * the substitute trains on raw [0,1] images (subset "raw_train") wrapped in
    NormalizedModel so eps is in pixel units as in main_at.py; checkpoints are
    saved UNWRAPPED, in the extraction_final format;
  * robust accuracy (same attack, ground-truth labels, raw test set) and the
    fidelity to the victim are logged every epoch; best_rob_epoch.pth added;
  * best_epoch.pth keeps the family rule (highest clean Test_Acc, any epoch);
  * DETERMINISTIC=False (cudnn.benchmark) as in main_at.py;
  * the scheduler accepts T_max: auto (= the epoch budget).

Outputs
  saved_models/extraction_at/<scenario>/{best_epoch,best_rob_epoch,epoch_N}.pth
  saved_logs/extraction_at/Performance/training_log_<scenario>.csv
  <scenario> = <Scenario_Name>_<extract_seed>_1.0_AT<attack>_<params>_lambda=<l>
               _bn=<policy>_inner=<hard|soft>_run=<tag>

Usage
  python main_knockoff_extraction_at.py --check-only
  python main_knockoff_extraction_at.py --seeds 0 --eps 0.031373 --epochs 2 --max-batches 30 --robust-n 1000 --run-tag smoke --force
  python main_knockoff_extraction_at.py --seeds 0 1 2 --eps 0.031373
"""
import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import yaml
from torch.utils.data import DataLoader, Subset

from at_family_common import (at_suffix, batch_accuracy, filter_attacks, make_adv_generator,
                              mixed_adversarial_step, parse_additive_at_config)
from main_knockoff_extraction import SoftLabeledSubset, build_model, seed_worker, set_seed
from util import (build_dataset_from_yaml, create_or_load_group_A, create_or_load_group_B,
                  evaluate1, evaluate_fidelity, load_best_checkpoint, process_yaml_file,
                  query_victim, wait_for_cool_gpu)
from util_adv import NormalizedModel, compute_robust_test_accuracy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = Path(__file__).resolve().parent

DETERMINISTIC = False
SEED_START = int(os.environ.get("SEED_START", 0))
SEED_END = int(os.environ.get("SEED_END", 3))

EXTRACTION_AT_ROOT = './saved_models/extraction_at'
EXTRACTION_AT_LOG_ROOT = './saved_logs/extraction_at/Performance'
DEFAULT_PLAN_DIR = ROOT / "saved_exp_plan/extraction_at_plan"

KNOCKOFF_COLUMNS = ['Scenario', 'Epoch', 'Train_Loss', 'Train_Acc', 'Train_Precision', 'Train_Recall',
                    'Train_F1', 'Test_Loss', 'Test_Acc', 'Test_Precision', 'Test_Recall', 'Test_F1']
LOG_COLUMNS = KNOCKOFF_COLUMNS + ['Robust_Acc', 'Fidelity', 'Train_Clean_Acc', 'Train_Adv_Acc', 'LR',
                                  'Epoch_Seconds', 'AT_Attack', 'AT_eps', 'AT_steps', 'AT_lambda', 'AT_bn',
                                  'AT_inner', 'Robust_N']


# =====================================================
# 1. Losses and inner maximisation
# =====================================================
def soft_label_loss(logits, soft_targets):
    """KL(soft_targets || softmax(logits)); identical to main_knockoff_extraction."""
    log_probs = torch.log_softmax(logits, dim=1)
    return nn.KLDivLoss(reduction='batchmean')(log_probs, soft_targets)


def pgd_attack_soft(model, x, soft_targets, eps, steps, step_size=None):
    """L-inf PGD maximising KL(p_victim || f(x_adv)); mirrors util_adv.pgd_attack_v2."""
    if step_size is None:
        step_size = 2.5 * eps / steps
    x_adv = x.detach().clone()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = soft_label_loss(model(x_adv), soft_targets)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        with torch.no_grad():
            x_adv = x_adv + step_size * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.detach()


def fgsm_attack_soft(model, x, soft_targets, eps):
    x_adv = x.detach().clone().requires_grad_(True)
    loss = soft_label_loss(model(x_adv), soft_targets)
    grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
    with torch.no_grad():
        x_adv = torch.clamp(x + eps * grad.sign(), 0.0, 1.0)
    return x_adv.detach()


SOFT_ATTACKS = {'PGD': pgd_attack_soft, 'FGSM': fgsm_attack_soft}


def parse_inner_target(exp_yaml):
    at_cfg = exp_yaml.get("AdversarialTraining") or {}
    inner = str(at_cfg.get("inner_target", "hard")).lower()
    if inner not in ("hard", "soft"):
        raise ValueError("AdversarialTraining.inner_target must be 'hard' or 'soft'.")
    return inner


def make_knockoff_adv_generator(attack_fn, attack_name, attack_kwargs, inner_target):
    """gen(model, x_raw, soft_targets) -> x_adv for the chosen inner target."""
    if inner_target == "hard":
        return make_adv_generator(attack_fn, attack_kwargs, target_fn=lambda t: t.argmax(1))
    soft_fn = SOFT_ATTACKS.get(attack_name)
    if soft_fn is None:
        raise ValueError(f"inner_target=soft is only implemented for {list(SOFT_ATTACKS)}")
    return make_adv_generator(soft_fn, attack_kwargs)


# =====================================================
# 2. One epoch: KL(x) + lambda * KL(x_adv), shared BN policy
# =====================================================
def knockoff_at_one_epoch(model, train_loader, optimizer, adv_generator, device='cuda',
                          clean_weight=1.0, adv_weight=1.0, bn_policy="both", max_batches=None):
    """One knockoff epoch with the added adversarial term.

    The loader yields (raw [0,1] augmented image, soft label); `model` is a
    NormalizedModel. Train_Loss is the summed objective; Train_Acc / precision
    / recall / F1 are the clean predictions vs argmax soft label, exactly as
    util.train_one_epoch_knockoff reports them (adv_weight == 0 with
    bn_policy="both" reproduces that function; test_knockoff_at.py checks it).
    """
    running_loss, total, clean_correct, adv_correct = 0.0, 0, 0, 0
    all_preds, all_targets = [], []
    for batch_idx, (images, soft_targets) in enumerate(train_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        soft_targets = soft_targets.to(device, non_blocking=True)
        hard_targets = soft_targets.argmax(1)
        optimizer.zero_grad(set_to_none=True)
        out = mixed_adversarial_step(model, images, soft_targets, soft_label_loss, adv_generator,
                                     clean_weight, adv_weight, bn_policy=bn_policy)
        out["loss"].backward()
        optimizer.step()

        n = images.size(0)
        running_loss += out["loss"].item() * n
        total += n
        if out["clean_logits"] is not None:
            clean_correct += batch_accuracy(out["clean_logits"], hard_targets)
        if out["adv_logits"] is not None:
            adv_correct += batch_accuracy(out["adv_logits"], hard_targets)
        report = out["clean_logits"] if out["clean_logits"] is not None else out["adv_logits"]
        all_preds.append(report.argmax(1).detach().cpu().numpy())
        all_targets.append(hard_targets.detach().cpu().numpy())

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
# 3. Naming, preflight
# =====================================================
def knockoff_at_scenario_name(base_scenario, extract_seed, attack_name, attack_kwargs, lam, bn_policy,
                              inner_target, run_tag):
    return (f"{base_scenario}_{extract_seed}_{1.0}"
            + at_suffix(attack_name, attack_kwargs, lam, run_tag, bn_policy, extra={"inner": inner_target}))


def resolve_epochs(exp_yaml, epochs_override):
    epochs = int(epochs_override) if epochs_override is not None else int(exp_yaml.get("Epochs", 100))
    if epochs <= 0:
        raise ValueError("Epochs must be positive.")
    return epochs


def expand_plan(yaml_path, seeds, run_tag, eps_filter=None):
    """Names one plan produces; read-only, same expansion as training."""
    with Path(yaml_path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    attacks, _, lam, bn_policy = parse_additive_at_config(cfg)
    attacks = filter_attacks(attacks, eps_filter)
    inner = parse_inner_target(cfg)
    names = [knockoff_at_scenario_name(cfg["Scenario_Name"], seed, name, kwargs, lam, bn_policy, inner, run_tag)
             for seed in seeds for _, name, kwargs in attacks]
    return cfg, names


# =====================================================
# 4. Main
# =====================================================
def main_knockoff_at(model_seed, extract_seed, yaml_file_path, *, run_tag="v1", eps_filter=None,
                     epochs_override=None, max_batches=None, robust_n=None, force=False):
    print(f"Device: {device}")
    exp_yaml = process_yaml_file(yaml_file_path)
    attack_configs, clean_w, adv_w, bn_policy = parse_additive_at_config(exp_yaml)
    attack_configs = filter_attacks(attack_configs, eps_filter)
    inner_target = parse_inner_target(exp_yaml)
    num_epochs = resolve_epochs(exp_yaml, epochs_override)
    print(f"Additive AT: L = {clean_w:g} * KL(x) + {adv_w:g} * KL(x_adv); attacks: {len(attack_configs)}; "
          f"inner={inner_target}; bn_policy={bn_policy}; epochs={num_epochs}")

    # ----- Victim (identical to main_knockoff_extraction) -----
    victim_cfg = exp_yaml["Victim"]
    victim_ds_cfg = victim_cfg["Dataset"]
    victim_dataset_obj, victim_num_classes, _ = build_dataset_from_yaml(victim_ds_cfg)
    victim_model_name = victim_cfg.get("Model", "ResNet-18")
    victim_net = build_model(victim_model_name, victim_num_classes).to(device)
    victim_model_dir = Path('./saved_models/vanilla/CNN_Models') / (victim_cfg["Model_Name"] + f"_{model_seed}_{1.0}")
    ckpt_path, _ = load_best_checkpoint(victim_model_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"No victim model checkpoint found in {victim_model_dir}")
    victim_net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    victim_net.eval()
    print(f'Victim model loaded from: {ckpt_path}')

    # ----- Auxiliary data and the clean query pass (identical) -----
    aux_ds_cfg = exp_yaml.get("Auxiliary_Dataset", victim_ds_cfg)
    aux_dataset_obj, aux_num_classes, aux_group_size = build_dataset_from_yaml(aux_ds_cfg)
    train_set = aux_dataset_obj.train_set
    group_A = create_or_load_group_A(dataset=train_set, save_dir=f'./Indices/{aux_ds_cfg["name"]}/',
                                     group_size=aux_group_size, num_classes=aux_num_classes, seed=42,
                                     force_rebuild=False)
    group_B = create_or_load_group_B(dataset=train_set, save_dir=f'./Indices/{aux_ds_cfg["name"]}/',
                                     group_A_indices=group_A, group_size=aux_group_size,
                                     num_classes=aux_num_classes, overlap_rate=0.0, seed=42,
                                     force_rebuild=False)
    query_subset = aux_dataset_obj.subset("train", group_B, clean=True)
    query_loader = DataLoader(query_subset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    print('==> Querying victim model (blackbox)..')
    stolen_inputs, stolen_labels = query_victim(victim_net, query_loader, device)
    print(f' Collected {stolen_inputs.shape[0]} query-response pairs')
    for i in [0, len(group_B) // 2, len(group_B) - 1]:
        img_clean, _ = query_subset[i]
        assert torch.allclose(img_clean, stolen_inputs[i], atol=1e-6), f"index alignment broken at i={i}"
    mean = torch.tensor(aux_dataset_obj.mean).view(3, 1, 1)
    std = torch.tensor(aux_dataset_obj.std).view(3, 1, 1)
    raw_clean_subset = aux_dataset_obj.subset("raw_train_clean", group_B)
    for i in [0, len(group_B) // 2, len(group_B) - 1]:
        img_raw, _ = raw_clean_subset[i]
        assert torch.allclose((img_raw - mean) / std, stolen_inputs[i], atol=1e-5), \
            f"raw/normalised alignment broken at i={i}"
    print("Index alignment verified (normalised and raw paths).")

    # ----- Substitute training set: raw [0,1] with augmentation -----
    g = torch.Generator()
    g.manual_seed(extract_seed)
    stolen_dataset = SoftLabeledSubset(aux_dataset_obj.subset("raw_train", group_B), stolen_labels)
    stolen_loader = DataLoader(stolen_dataset, batch_size=128, shuffle=True, num_workers=8,
                               worker_init_fn=seed_worker, generator=g, persistent_workers=True,
                               pin_memory=True)

    # ----- Test loaders: normalised (family metrics) and raw (robust accuracy) -----
    test_set = victim_dataset_obj.test_set
    testloader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    raw_test_set = victim_dataset_obj.raw_test_set
    if robust_n is not None and robust_n < len(raw_test_set):
        raw_test_set = Subset(raw_test_set, list(range(robust_n)))
    robust_n_used = len(raw_test_set)
    raw_testloader = DataLoader(raw_test_set, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    eval_criterion = nn.CrossEntropyLoss()
    victim_test_result = evaluate1(victim_net, testloader, eval_criterion, device)
    print(f'    Victim Test Acc: {victim_test_result["test_acc"]:.2f}%')

    sub_model_name = exp_yaml["Substitute"].get("Model", victim_model_name)
    os.makedirs(EXTRACTION_AT_LOG_ROOT, exist_ok=True)

    for attack_fn, attack_name, attack_kwargs in attack_configs:
        scenario_name = knockoff_at_scenario_name(exp_yaml["Scenario_Name"], extract_seed, attack_name,
                                                  attack_kwargs, adv_w, bn_policy, inner_target, run_tag)
        save_dir = Path(EXTRACTION_AT_ROOT) / scenario_name
        best_ckpt_path = save_dir / 'best_epoch.pth'
        log_file = Path(EXTRACTION_AT_LOG_ROOT) / f"training_log_{scenario_name}.csv"
        if (best_ckpt_path.is_file() and best_ckpt_path.stat().st_size > 0) or log_file.exists():
            if not force:
                print(f"[SKIP] {scenario_name}: outputs already exist (use --force or a new --run-tag)")
                continue
            print(f"[FORCE] overwriting outputs of {scenario_name}")
            if log_file.exists():
                log_file.unlink()

        print(f"\n{'=' * 60}\nAttack: {attack_name} {attack_kwargs} | lambda={adv_w:g} | bn={bn_policy}"
              f" | inner={inner_target}\nScenario: {scenario_name}\n{'=' * 60}")
        set_seed(extract_seed, deterministic=DETERMINISTIC)
        g.manual_seed(extract_seed)
        adv_generator = make_knockoff_adv_generator(attack_fn, attack_name, attack_kwargs, inner_target)

        base_net = build_model(sub_model_name, victim_num_classes).to(device)
        substitute = NormalizedModel(base_net, aux_dataset_obj.mean, aux_dataset_obj.std).to(device)

        optimizer_cfg = exp_yaml.get("Optimizer", {})
        optimizer = getattr(optim, optimizer_cfg.get("name", "Adam"))(
            substitute.parameters(), **optimizer_cfg.get("params", {"lr": 1e-3}))
        scheduler = None
        scheduler_cfg = exp_yaml.get("Scheduler") or {}
        if scheduler_cfg.get("name"):
            s_params = dict(scheduler_cfg.get("params", {}))
            if s_params.get("T_max") == "auto":
                s_params["T_max"] = num_epochs
            scheduler = getattr(lr_sched, scheduler_cfg["name"])(optimizer, **s_params)

        os.makedirs(save_dir, exist_ok=True)
        with open(log_file, 'w', newline='') as f:
            csv.writer(f).writerow(LOG_COLUMNS)

        best_test_acc, best_rob_acc = -1.0, -1.0
        for epoch in range(num_epochs):
            t0 = time.time()
            train_result = knockoff_at_one_epoch(substitute, stolen_loader, optimizer, adv_generator, device,
                                                 clean_w, adv_w, bn_policy, max_batches)
            wait_for_cool_gpu(threshold=89.5)
            lr_now = optimizer.param_groups[0]['lr']
            if scheduler is not None:
                scheduler.step()
            test_result = evaluate1(base_net, testloader, eval_criterion, device)
            rob_acc = 100.0 * compute_robust_test_accuracy(substitute, raw_testloader, attack_fn, attack_kwargs)
            fidelity = evaluate_fidelity(victim_net, base_net, testloader, device)
            seconds = time.time() - t0
            print(f"  [Epoch {epoch}/{num_epochs}] LR {lr_now:.5f} | Loss {train_result['train_loss']:.4f}"
                  f" | Train clean {train_result['train_clean_acc']:.2f}% adv {train_result['train_adv_acc']:.2f}%"
                  f" | Test clean {test_result['test_acc']:.2f}% | Test robust {rob_acc:.2f}%"
                  f" | Fidelity {fidelity:.2f}% | {seconds:.0f}s")
            with open(log_file, 'a', newline='') as f:
                csv.writer(f).writerow([
                    scenario_name, epoch,
                    train_result["train_loss"], train_result["train_acc"], train_result["train_precision"],
                    train_result["train_recall"], train_result["train_f1"],
                    test_result["test_loss"], test_result["test_acc"], test_result["test_precision"],
                    test_result["test_recall"], test_result["test_f1"],
                    round(rob_acc, 4), round(fidelity, 4), round(train_result["train_clean_acc"], 4),
                    round(train_result["train_adv_acc"], 4), lr_now, round(seconds, 1),
                    attack_name, attack_kwargs.get("eps"), attack_kwargs.get("steps"), adv_w, bn_policy,
                    inner_target, robust_n_used,
                ])
            if test_result["test_acc"] > best_test_acc:
                best_test_acc = test_result["test_acc"]
                torch.save(base_net.state_dict(), best_ckpt_path)
            if rob_acc > best_rob_acc:
                best_rob_acc = rob_acc
                torch.save(base_net.state_dict(), save_dir / 'best_rob_epoch.pth')

        torch.save(base_net.state_dict(), save_dir / f'epoch_{epoch}.pth')
        base_net.load_state_dict(torch.load(best_ckpt_path, map_location=device, weights_only=False))
        base_net.eval()
        fidelity = evaluate_fidelity(victim_net, base_net, testloader, device)
        rob_best = 100.0 * compute_robust_test_accuracy(substitute, raw_testloader, attack_fn, attack_kwargs)
        print(f'\n==> Knockoff+AT complete: {scenario_name}')
        print(f"Victim Test Acc:            {victim_test_result['test_acc']:.2f}%")
        print(f"Substitute Best Clean Acc:  {best_test_acc:.2f}%  (robust at that ckpt {rob_best:.2f}%)")
        print(f"Best Robust Acc (any ckpt): {best_rob_acc:.2f}%")
        print(f"Accuracy Recovery:          {100.0 * best_test_acc / victim_test_result['test_acc']:.1f}%")
        print(f"Fidelity (best_epoch):      {fidelity:.1f}%")


# =====================================================
# 5. Entry point
# =====================================================
def main(argv=None):
    parser = argparse.ArgumentParser(description="Knockoff extraction with an added adversarial loss term.")
    parser.add_argument("--plans", nargs="+", type=Path,
                        help="YAML plans; default: all saved_exp_plan/extraction_at_plan/*.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, help="Extraction seeds; default SEED_START..SEED_END.")
    parser.add_argument("--model-seed", type=int, default=42, help="Victim seed (folder suffix).")
    parser.add_argument("--eps", nargs="+", type=float, help="Only run the plan's attack configs with these eps.")
    parser.add_argument("--run-tag", default="v1")
    parser.add_argument("--epochs", type=int, help="Override the YAML epoch budget (T_max: auto follows).")
    parser.add_argument("--max-batches", type=int, help="Smoke tests: train on at most N batches per epoch.")
    parser.add_argument("--robust-n", type=int, help="Robust accuracy on the first N test images (default all).")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs of the same name.")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)

    seeds = args.seeds if args.seeds is not None else list(range(SEED_START, SEED_END))
    if not seeds or len(seeds) != len(set(seeds)):
        parser.error("Seeds must be distinct and non-empty.")
    os.chdir(ROOT)
    yaml_files = args.plans or sorted(DEFAULT_PLAN_DIR.glob("*.yaml"))
    if not yaml_files:
        parser.error(f"No YAML plans found in {DEFAULT_PLAN_DIR}")

    print(f"Plans: {[str(p) for p in yaml_files]}; seeds: {seeds}; eps: {args.eps or 'all'}; run tag: {args.run_tag}")
    for path in yaml_files:
        _, names = expand_plan(path, seeds, args.run_tag, args.eps)
        for n in names:
            exists = (Path(EXTRACTION_AT_ROOT) / n / 'best_epoch.pth').is_file()
            print(f"  {'[exists] ' if exists else ''}{n}")
    if args.check_only:
        print("Preflight passed. No training started.")
        return

    torch.multiprocessing.set_start_method("spawn", force=True)
    for yaml_path in yaml_files:
        for seed in seeds:
            print(f"\n>>> extraction seed {seed} for {yaml_path.name}")
            set_seed(seed, deterministic=DETERMINISTIC)
            main_knockoff_at(args.model_seed, seed, str(yaml_path), run_tag=args.run_tag, eps_filter=args.eps,
                             epochs_override=args.epochs, max_batches=args.max_batches,
                             robust_n=args.robust_n, force=args.force)


if __name__ == "__main__":
    main()
