"""Knowledge distillation with an added adversarial term (one-stage AT, KD family).

Same attack as main_kd.py for the KD method (RN34 teacher -> ResNet-18
student trained from scratch on the transfer set group_B with overlap rate
r, batch 256, the plan's optimizer / lr / epochs, grad-norm clipping 1.0),
with one extra loss term per batch:

    L_kd(s, t, y) = ce_w * CE(s, y) + kd_w * T^2 * KL( softmax(s/T) || softmax(t/T) )
    L = L_kd( f_s(x), f_t(x), y ) + lambda * L_kd( f_s(x_adv), f_t(x), y )

    x_adv = PGD / FGSM against the STUDENT, maximising CE(f_s(.), y) (ARD's
            inner maximisation). The teacher only ever sees the clean x; its
            logits are computed once per batch and shared by both terms.
    Weights are used as given (no normalisation).

BatchNorm policy: AdversarialTraining.bn_policy, default "both" (the student
starts from random weights, so there are no statistics to freeze). Every
trained model must pass diagnose_bn_modes.py --arch resnet18_dist before use.

Reused unchanged from main_kd.py / util.py
  * process_experiment_kd_setup (teacher checkpoint, student factory, the KD
    distiller with its temperature / ce_weight / kd_weight / ce_criterion,
    per-plan optimizer and scheduler), group_A / group_B with the plan seed,
    util.evaluate1 on the normalised test loader, best_epoch.pth = highest
    Test_Acc from epoch 0, epoch_<last>.pth, student.state_dict() format.
  * The two loss terms are computed OUTSIDE KD.forward_train (which would
    feed the teacher whatever image the student gets) with the same formula,
    KnowledgeDistillation.KD.kd_loss and the distiller's own weights.

Deliberate deviations
  * the student trains on raw [0,1] images (subset "raw_train") and is
    wrapped as NormalizedModel(KDLogitsOnly(student)) for PGD / robust
    evaluation; the teacher receives the normalised clean batch;
  * robust accuracy (same attack, ground-truth labels, raw test set) and the
    student-teacher agreement on the test set are logged every epoch;
    best_rob_epoch.pth added;
  * KD method only (DKD needs its own logit decomposition; not in v1);
    CNN students only (no DeiT); DETERMINISTIC=False as in main_at.py.

Outputs
  saved_models/kd_at/<scenario>/{best_epoch,best_rob_epoch,epoch_N}.pth
  saved_logs/kd_at/Performance/training_log_<scenario>.csv
  <scenario> = <Scenario_Name>_KD_<seed>_<r>_AT<attack>_<params>_lambda=<l>_bn=<policy>_run=<tag>

Usage
  python main_kd_at.py --check-only
  python main_kd_at.py --seeds 0 --eps 0.031373 --epochs 2 --max-batches 20 --robust-n 1000 --run-tag smoke --force
  python main_kd_at.py --seeds 0 1 2 --eps 0.031373
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

from at_family_common import (at_suffix, batch_accuracy, filter_attacks, make_adv_generator,
                              mixed_adversarial_step, parse_additive_at_config)
from KnowledgeDistillation.KD import kd_loss
from main_kd import kd_scenario_name
from Model.kd_eval import KDLogitsOnly
from util import (build_warmup_cosine_scheduler, create_or_load_group_A, create_or_load_group_B, evaluate1,
                  evaluate_fidelity, process_experiment_kd_setup, process_yaml_file, wait_for_cool_gpu)
from util_adv import NormalizedModel, compute_robust_test_accuracy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = Path(__file__).resolve().parent
DETERMINISTIC = False
SEED_START = int(os.environ.get("SEED_START", 0))
SEED_END = int(os.environ.get("SEED_END", 3))

KD_AT_ROOT = './saved_models/kd_at'
KD_AT_LOG_ROOT = './saved_logs/kd_at/Performance'
DEFAULT_PLAN_DIR = ROOT / "saved_exp_plan/kd_at_plan"
BATCH_SIZE = 256            # main_kd.py hard-codes 256 (the yaml BatchSize is not read)
CLIP_GRAD_NORM = 1.0        # util.train_one_epoch_kd clips the distiller's gradients to 1.0

KD_COLUMNS = ['Scenario', 'Epoch', 'Train_Loss', 'Train_Acc', 'Train_Precision', 'Train_Recall',
              'Train_F1', 'Test_Loss', 'Test_Acc', 'Test_Precision', 'Test_Recall', 'Test_F1']
LOG_COLUMNS = KD_COLUMNS + ['Robust_Acc', 'Teacher_Agreement', 'Train_Clean_Acc', 'Train_Adv_Acc', 'LR',
                            'Epoch_Seconds', 'AT_Attack', 'AT_eps', 'AT_steps', 'AT_lambda', 'AT_bn', 'Robust_N']


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
# 1. KD loss with the teacher pinned to the clean batch
# =====================================================
def kd_batch_loss_fn(distiller, teacher_logits):
    """loss_fn(student_logits, y): KD.forward_train's objective for one batch.

    ce_weight * ce_criterion(s, y) + kd_weight * kd_loss(s, t_clean, T), with the
    teacher logits fixed to the ones computed on the CLEAN images, so the same
    function scores both f_s(x) and f_s(x_adv).
    """
    def loss_fn(logits, target):
        return (distiller.ce_weight * distiller.ce_criterion(logits, target)
                + distiller.kd_weight * kd_loss(logits, teacher_logits, distiller.temperature))
    return loss_fn


class Normalizer:
    """(x - mean) / std for raw [0,1] batches, on the right device."""
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)


# =====================================================
# 2. One epoch: L_kd(x) + lambda * L_kd(x_adv), shared BN policy
# =====================================================
def kd_at_one_epoch(distiller, student_w, normalize, train_loader, optimizer, adv_generator, device='cuda',
                    clean_weight=1.0, adv_weight=1.0, bn_policy="both", max_batches=None,
                    clip_grad=CLIP_GRAD_NORM):
    """One KD epoch with the added adversarial term.

    `student_w` = NormalizedModel(KDLogitsOnly(distiller.student)), fed raw
    [0,1] images; mode switches on it propagate to the student. The teacher is
    kept in eval mode and evaluated once per batch on the normalised clean
    images. Train_Loss is the summed objective; Train_Acc / precision / recall
    / F1 are the clean student predictions vs y, as util.train_one_epoch_kd
    reports them (adv_weight == 0 with bn_policy="both" reproduces that
    function; test_kd_at.py checks it).
    """
    running_loss, total, clean_correct, adv_correct = 0.0, 0, 0, 0
    all_preds, all_targets = [], []
    teacher = distiller.teacher
    teacher.eval()
    for batch_idx, (images, targets) in enumerate(train_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.no_grad():
            t_out = teacher(normalize(images))              # CLEAN x only
            teacher_logits = t_out[0] if isinstance(t_out, (tuple, list)) else t_out
        loss_fn = kd_batch_loss_fn(distiller, teacher_logits)

        optimizer.zero_grad(set_to_none=True)
        out = mixed_adversarial_step(student_w, images, targets, loss_fn, adv_generator,
                                     clean_weight, adv_weight, bn_policy=bn_policy)
        out["loss"].backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(distiller.get_learnable_parameters(), max_norm=clip_grad)
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
# 3. Naming, preflight
# =====================================================
def kd_at_scenario_name(base_scenario, method, seed, r, attack_name, attack_kwargs, lam, bn_policy, run_tag):
    return kd_scenario_name(base_scenario, method, seed, r) + at_suffix(attack_name, attack_kwargs, lam, run_tag, bn_policy)


def expand_plan(yaml_path, seeds, rates, methods, run_tag, eps_filter=None):
    with Path(yaml_path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    attacks, _, lam, bn_policy = parse_additive_at_config(cfg)
    attacks = filter_attacks(attacks, eps_filter)
    plan_methods = [d.get("name") for d in cfg.get("Distillation", []) or []]
    for m in methods:
        if m not in plan_methods:
            raise ValueError(f"{Path(yaml_path).name}: no Distillation entry for method {m}")
        if m != "KD":
            raise NotImplementedError(f"main_kd_at.py supports the KD method only (got {m})")
    names = [kd_at_scenario_name(cfg["Scenario_Name"], m, seed, r, name, kwargs, lam, bn_policy, run_tag)
             for seed in seeds for r in rates for m in methods for _, name, kwargs in attacks]
    return cfg, names


# =====================================================
# 4. Main
# =====================================================
def main_kd_at(seed, r, yaml_file_path, *, methods=("KD",), run_tag="v1", eps_filter=None,
               epochs_override=None, max_batches=None, robust_n=None, force=False):
    print(f"Device: {device}")
    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_kd_setup(exp_yaml)
    attack_configs, clean_w, adv_w, bn_policy = parse_additive_at_config(exp_yaml)
    attack_configs = filter_attacks(attack_configs, eps_filter)
    num_epochs = int(epochs_override) if epochs_override is not None else int(exp_yaml["Epochs"])
    print(f"Additive AT: L = {clean_w:g} * L_kd(x) + {adv_w:g} * L_kd(x_adv); attacks: {len(attack_configs)}; "
          f"bn_policy={bn_policy}; epochs={num_epochs}; rate={r}")

    # ----- Data (identical index files to main_kd; raw images for the student) -----
    set_seed(seed, deterministic=DETERMINISTIC)
    g = torch.Generator()
    g.manual_seed(seed)
    dataset_obj = exp_setup["Dataset"]
    train_set = dataset_obj.train_set
    group_A = create_or_load_group_A(dataset=train_set, save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
                                     group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"],
                                     seed=42, force_rebuild=False)
    group_B = create_or_load_group_B(dataset=train_set, save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
                                     group_A_indices=group_A, group_size=exp_setup["GroupSize"],
                                     num_classes=exp_setup["NumClasses"], overlap_rate=r, seed=42,
                                     force_rebuild=False)
    print(f"[DATA] transfer set: overlap_rate={r}, size={len(group_B)}, overlap with group_A={len(set(group_B) & set(group_A))}")
    normalize = Normalizer(dataset_obj.mean, dataset_obj.std)
    norm_clean = dataset_obj.subset("train", group_B, clean=True)
    raw_clean = dataset_obj.subset("raw_train_clean", group_B)
    for i in [0, len(group_B) // 2, len(group_B) - 1]:
        (xn, yn), (xr, yr) = norm_clean[i], raw_clean[i]
        assert int(yn) == int(yr) and torch.allclose(normalize(xr.unsqueeze(0))[0], xn, atol=1e-5), \
            f"raw/normalised transfer data misaligned at {i}"
    trainloader = DataLoader(dataset_obj.subset("raw_train", group_B), batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=4, worker_init_fn=seed_worker, generator=g, persistent_workers=True,
                             pin_memory=True)
    testloader = DataLoader(dataset_obj.test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    raw_test_set = dataset_obj.raw_test_set
    if robust_n is not None and robust_n < len(raw_test_set):
        raw_test_set = Subset(raw_test_set, list(range(robust_n)))
    robust_n_used = len(raw_test_set)
    raw_testloader = DataLoader(raw_test_set, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    criterion = nn.CrossEntropyLoss()
    os.makedirs(KD_AT_LOG_ROOT, exist_ok=True)

    for method_name in methods:
        if method_name != "KD":
            raise NotImplementedError("main_kd_at.py supports the KD method only")
        kd_cfg = exp_setup["KD_Setups"][method_name]
        for attack_fn, attack_name, attack_kwargs in attack_configs:
            scenario_name = kd_at_scenario_name(exp_yaml["Scenario_Name"], method_name, seed, r, attack_name,
                                                attack_kwargs, adv_w, bn_policy, run_tag)
            save_dir = Path(KD_AT_ROOT) / scenario_name
            best_ckpt_path = save_dir / 'best_epoch.pth'
            log_file = Path(KD_AT_LOG_ROOT) / f"training_log_{scenario_name}.csv"
            if (best_ckpt_path.is_file() and best_ckpt_path.stat().st_size > 0) or log_file.exists():
                if not force:
                    print(f"[SKIP] {scenario_name}: outputs already exist (use --force or a new --run-tag)")
                    continue
                print(f"[FORCE] overwriting outputs of {scenario_name}")
                if log_file.exists():
                    log_file.unlink()
            print(f"\n{'=' * 60}\nMethod: {method_name} | {attack_name} {attack_kwargs} | lambda={adv_w:g} | bn={bn_policy}"
                  f"\nScenario: {scenario_name}\n{'=' * 60}")

            # ----- Fresh distiller (teacher checkpoint + student init), as main_kd -----
            set_seed(seed, deterministic=DETERMINISTIC)
            g.manual_seed(seed)
            distiller = kd_cfg["Builder"]().to(device)
            distiller.train()                                     # student train, teacher pinned to eval
            student_w = NormalizedModel(KDLogitsOnly(distiller.student), dataset_obj.mean, dataset_obj.std).to(device)
            adv_generator = make_adv_generator(attack_fn, attack_kwargs)

            trainable = distiller.get_learnable_parameters()
            optimizer = getattr(optim, kd_cfg["Optimizer_Name"])(trainable, **kd_cfg["Optimizer_Params"])
            scheduler = None
            if kd_cfg["Scheduler_Name"] is not None:
                s_params = kd_cfg["Scheduler_Params"].copy()
                if s_params.get("T_max") == "auto":
                    s_params["T_max"] = num_epochs
                if kd_cfg["Scheduler_Name"] == "WarmupCosineAnnealingLR":
                    scheduler = build_warmup_cosine_scheduler(
                        optimizer, total_epochs=int(s_params.get("T_max", num_epochs)),
                        warmup_epochs=int(s_params.get("warmup_epochs", 10)),
                        warmup_start_factor=float(s_params.get("warmup_start_factor", 0.1)),
                        eta_min=float(s_params.get("eta_min", 1e-6)))
                else:
                    scheduler = getattr(lr_sched, kd_cfg["Scheduler_Name"])(optimizer, **s_params)

            os.makedirs(save_dir, exist_ok=True)
            with open(log_file, 'w', newline='') as f:
                csv.writer(f).writerow(LOG_COLUMNS)
            teacher_acc = evaluate1(distiller.teacher, testloader, criterion, device)["test_acc"]
            print(f"    Teacher Test Acc: {teacher_acc:.2f}%")

            best_test_acc, best_rob_acc = -1.0, -1.0
            for epoch in range(num_epochs):
                t0 = time.time()
                train_result = kd_at_one_epoch(distiller, student_w, normalize, trainloader, optimizer, adv_generator,
                                               device, clean_w, adv_w, bn_policy, max_batches)
                wait_for_cool_gpu(threshold=89.5)
                lr_now = optimizer.param_groups[0]['lr']
                if scheduler is not None:
                    scheduler.step()
                test_result = evaluate1(distiller.student, testloader, criterion, device)
                rob_acc = 100.0 * compute_robust_test_accuracy(student_w, raw_testloader, attack_fn, attack_kwargs)
                agreement = evaluate_fidelity(distiller.teacher, distiller.student, testloader, device)
                seconds = time.time() - t0
                print(f"  [Epoch {epoch}/{num_epochs}] LR {lr_now:.5f} | Loss {train_result['train_loss']:.4f}"
                      f" | Train clean {train_result['train_clean_acc']:.2f}% adv {train_result['train_adv_acc']:.2f}%"
                      f" | Test clean {test_result['test_acc']:.2f}% | Test robust {rob_acc:.2f}%"
                      f" | Agree(teacher) {agreement:.2f}% | {seconds:.0f}s")
                with open(log_file, 'a', newline='') as f:
                    csv.writer(f).writerow([
                        scenario_name, epoch,
                        train_result["train_loss"], train_result["train_acc"], train_result["train_precision"],
                        train_result["train_recall"], train_result["train_f1"],
                        test_result["test_loss"], test_result["test_acc"], test_result["test_precision"],
                        test_result["test_recall"], test_result["test_f1"],
                        round(rob_acc, 4), round(agreement, 4), round(train_result["train_clean_acc"], 4),
                        round(train_result["train_adv_acc"], 4), lr_now, round(seconds, 1),
                        attack_name, attack_kwargs.get("eps"), attack_kwargs.get("steps"), adv_w, bn_policy,
                        robust_n_used,
                    ])
                if test_result["test_acc"] > best_test_acc:
                    best_test_acc = test_result["test_acc"]
                    torch.save(distiller.student.state_dict(), best_ckpt_path)
                if rob_acc > best_rob_acc:
                    best_rob_acc = rob_acc
                    torch.save(distiller.student.state_dict(), save_dir / 'best_rob_epoch.pth')

            torch.save(distiller.student.state_dict(), save_dir / f'epoch_{epoch}.pth')
            print(f"==> Finished {scenario_name}. Best clean {best_test_acc:.2f}% | best robust {best_rob_acc:.2f}%"
                  f" | teacher {teacher_acc:.2f}%")
            del distiller, student_w, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# =====================================================
# 5. Entry point
# =====================================================
def main(argv=None):
    parser = argparse.ArgumentParser(description="Knowledge distillation with an added adversarial loss term.")
    parser.add_argument("--plans", nargs="+", type=Path, help="YAML plans; default: all saved_exp_plan/kd_at_plan/*.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, help="KD seeds; default SEED_START..SEED_END.")
    parser.add_argument("--rates", nargs="+", type=float, default=[0.0], help="Transfer-set overlap rates (0.0 = Cell C).")
    parser.add_argument("--methods", nargs="+", default=["KD"])
    parser.add_argument("--eps", nargs="+", type=float, help="Only run the plan's attack configs with these eps.")
    parser.add_argument("--run-tag", default="v1")
    parser.add_argument("--epochs", type=int, help="Override the YAML epoch budget (T_max: auto follows).")
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

    print(f"Plans: {[str(p) for p in yaml_files]}; seeds: {seeds}; rates: {args.rates}; methods: {args.methods}; "
          f"eps: {args.eps or 'all'}; run tag: {args.run_tag}")
    for path in yaml_files:
        _, names = expand_plan(path, seeds, args.rates, args.methods, args.run_tag, args.eps)
        for n in names:
            exists = (Path(KD_AT_ROOT) / n / 'best_epoch.pth').is_file()
            print(f"  {'[exists] ' if exists else ''}{n}")
    if args.check_only:
        print("Preflight passed. No training started.")
        return

    torch.multiprocessing.set_start_method("spawn", force=True)
    for yaml_path in yaml_files:
        for seed in seeds:
            for r in args.rates:
                print(f"\n>>> seed {seed}, rate {r} for {yaml_path.name}")
                main_kd_at(seed, r, str(yaml_path), methods=tuple(args.methods), run_tag=args.run_tag,
                           eps_filter=args.eps, epochs_override=args.epochs, max_batches=args.max_batches,
                           robust_n=args.robust_n, force=args.force)


if __name__ == "__main__":
    main()
