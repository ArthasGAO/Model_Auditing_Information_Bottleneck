"""Clean-accuracy-preserving post-hoc adversarial fine-tuning of finished derived models.

Same two-stage setting as main_at.py: the attack (FT / KD / extraction ...) is
finished; the attacker then adversarially fine-tunes the resulting model on
its own auxiliary images (group_B 0.0 = the dataset's other 25000 training
images, disjoint from the victim's group_A) with ground-truth labels. The
recipe is changed to keep as much clean accuracy as possible:

    L = CE(f(x), y) + lambda * CE(f(x_adv[:k]), y[:k])     weights as given, no normalisation
    x_adv = PGD (L_inf, eps in [0,1] pixel units) against the model being tuned, built for the
            first k = round(adv_fraction * B) images of each shuffled batch (adv_fraction = 1: all)
  adv_fraction < 1 tilts the step toward clean data twice over: fewer adversarial examples in
  the loss and a smaller adversarial share k / (B + k) of the BatchNorm batch (Kurakin et al.'s
  "k adversarial out of m" mixed batch).

  * clean term kept (main_at.py's at_final used the pure adversarial loss);
    the template is the one main_ft_at.py / main_prune_at.py use, so post-hoc
    and one-stage models are trained on the identical per-batch objective;
  * BatchNorm (AdversarialTraining.bn_policy, POSTHOC_BN_POLICIES):
      "joint"  (default) ONE train-mode forward on cat([x, x_adv]); both halves
               share the batch statistics, so the clean/adversarial statistics
               "switch" seen with "both" on CIFAR-10 cannot be learned, and the
               normalisation stays scale-invariant;
      "frozen" every BN in eval mode (the ft_at / pruning_at choice). NOT usable on
               the CIFAR-100 sources: their pre-BN scale is tiny (median running_var
               4e-5 after wd 1e-3 x 360 ep), so eval-mode BN turns lr 1e-3 into a
               divergence within 20 batches and lr 1e-4 learns no robustness;
      "both"   two train-mode forwards (at_family_common);
  * small learning rate (CIFAR-100 plan: 3e-4 cosine -> 1e-6, 30 epochs; at_final used 1e-2);
  * checkpoint selection best_clean_epoch.pth = highest clean test accuracy
    from epoch int(0.2 * epochs) on (main_at.py's window); best_rob_epoch.pth
    and the last epoch are kept as well.
  Pilot numbers behind these choices are in the CIFAR-100 plan's header.

Reused unchanged: util_adv.build_at_dataset_from_yaml (the group_B 0.0 indices
main_at.py uses, raw [0,1] crop+flip images), util_adv.initialize_optimizer_scheduler,
compute_robust_test_accuracy / pgd_attack_v2 (via parse_attack_configs),
main_ft_at.ft_at_one_epoch (strategy FT-AL: every layer trains; used for "frozen" / "both",
"joint" has its own epoch function with the same logged fields) and
at_family_common's config parser. Source checkpoints are bare state_dicts
(ft_final / kd_final / extraction_final best_epoch.pth); a KD student saved
from ResNet18_dist loads into ResNet18 with identical logits (checked
2026-09-23 on the CIFAR-100 DKD student, max |diff| 0.0).

Outputs (checkpoints UNWRAPPED, same keys as the source checkpoint)
  saved_models/at_posthoc/<scenario>/best_clean_epoch.pth, best_rob_epoch.pth and, with
      SAVE_ALL_EPOCHS (default), epoch_-1.pth (= the source) ... epoch_<E-1>.pth for trajectory
      plots (33 files, ~1.5 GB per 30-epoch run); otherwise only epoch_<E-1>.pth
  saved_logs/at_posthoc/Performance/training_log_<scenario>.csv   (ft_at column layout)
  <scenario> = <source folder>_<attack>_<k=v sorted>_bn=<policy>_atn=<n>_atepochs=<E>
               _lr=<lr>_lambda=<l>_advfrac=<f>[_mask=kept]_run=<tag>_atseed=<s>_mix=add
  A Model_Path entry may be a mapping {Path, Alias, Keep_Zero_Mask} (parse_sources): Alias replaces
  <source folder> (long pruning names exceed the Windows path limit), <run>/source.json records the
  real source; Keep_Zero_Mask keeps the source's pruned (exactly zero) conv/linear weights at zero.
A run counts as finished when epoch_<E-1>.pth exists (then it is skipped);
partial outputs stop the run unless --force.

Usage (from E:\\Experiment, pytorch_env)
  python main_at_posthoc.py --check-only
  python main_at_posthoc.py --epochs 2 --max-batches 20 --robust-n 500 --run-tag smoke --force
  python main_at_posthoc.py                                  # every source x eps in the plan
  python main_at_posthoc.py --select FT-AL --eps 0.031373    # a subset
"""
import argparse
import csv
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

from at_family_common import (AT_EXTRA_COLUMNS, batch_accuracy, filter_attacks, make_adv_generator,
                              parse_additive_at_config)
from main_ft_at import FT_COLUMNS, ft_at_one_epoch
from util import build_dataset_from_yaml, evaluate1, process_yaml_file, wait_for_cool_gpu
from util_adv import (NormalizedModel, build_at_dataset_from_yaml, build_model, compute_robust_test_accuracy,
                      initialize_optimizer_scheduler)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = Path(__file__).resolve().parent
DETERMINISTIC = False
DEFAULT_PLAN_DIR = ROOT / "saved_exp_plan/at_posthoc_plan"
OUT_MODEL_ROOT = Path("./saved_models/at_posthoc")
OUT_LOG_ROOT = Path("./saved_logs/at_posthoc/Performance")
SOURCE_CKPT = "best_epoch.pth"
BEST_CKPT_START_FRAC = 0.2           # main_at.py's selection window
SAVE_ALL_EPOCHS = True               # epoch_-1 (source) ... epoch_<E-1>: 45 MB each, ~1.5 GB per 30-epoch run
WINDOWS_MAX_PATH = 259               # longest absolute file path Windows accepts without LongPathsEnabled
LOG_COLUMNS = FT_COLUMNS + AT_EXTRA_COLUMNS
# "joint" is local to this script: ONE train-mode forward on cat([x, x_adv]), so the clean
# and adversarial halves always share the same batch statistics (no clean/adv statistics
# switch is learnable) and normalisation stays scale-invariant. "frozen" / "both" delegate
# to at_family_common.mixed_adversarial_step through main_ft_at.ft_at_one_epoch.
POSTHOC_BN_POLICIES = ("joint", "frozen", "both")


def parse_posthoc_config(exp_yaml):
    """parse_additive_at_config plus the local "joint" policy -> (attacks, clean_w, adv_w, bn_policy)."""
    at_cfg = dict(exp_yaml.get("AdversarialTraining") or {})
    bn_policy = str(at_cfg.get("bn_policy", "joint")).lower()
    if bn_policy not in POSTHOC_BN_POLICIES:
        raise ValueError(f"AdversarialTraining.bn_policy must be one of {POSTHOC_BN_POLICIES}")
    at_cfg["bn_policy"] = "frozen" if bn_policy == "joint" else bn_policy   # validated separately above
    attacks, clean_w, adv_w, _ = parse_additive_at_config(dict(exp_yaml, AdversarialTraining=at_cfg))
    if bn_policy == "joint" and (clean_w <= 0 or adv_w <= 0):
        raise ValueError("bn_policy=joint needs both loss terms (lambda > 0)")
    return attacks, clean_w, adv_w, bn_policy


def parse_adv_fraction(exp_yaml):
    """AdversarialTraining.adv_fraction in (0, 1]: share of each batch that gets an adversarial copy.

    1.0 = every image (B clean + B adversarial in the joint forward, BN share 50%);
    0.25 = the first B/4 images of the shuffled batch (B clean + B/4 adversarial, BN share 20%).
    Only the "joint" policy supports values below 1.
    """
    at_cfg = exp_yaml.get("AdversarialTraining") or {}
    frac = float(at_cfg.get("adv_fraction", 1.0))
    if not 0.0 < frac <= 1.0:
        raise ValueError("AdversarialTraining.adv_fraction must be in (0, 1]")
    if frac < 1.0 and str(at_cfg.get("bn_policy", "joint")).lower() != "joint":
        raise ValueError("adv_fraction < 1 is only implemented for bn_policy=joint")
    return frac


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
# 1. Naming, sources, data checks
# =====================================================
def posthoc_scenario_name(model_path, attack_name, attack_kwargs, *, bn_policy, train_size, epochs, lr, lam,
                          run_tag, at_seed, adv_fraction=1.0, base=None, keep_mask=False):
    """<base>_<attack>_<params>_bn=_atn=_atepochs=_lr=_lambda=_advfrac=[_mask=kept]_run=_atseed=_mix=add.

    base = the source folder name, or the plan's Alias for that source (Windows path limit).
    """
    if not re.fullmatch(r"[A-Za-z0-9-]{1,32}", run_tag):
        raise ValueError("run_tag must contain 1-32 letters, digits or hyphens.")
    if train_size <= 0 or epochs <= 0 or lr <= 0 or lam < 0 or at_seed < 0 or not 0 < adv_fraction <= 1:
        raise ValueError("train_size, epochs, lr positive; lambda, at_seed non-negative; adv_fraction in (0, 1].")
    base = base or model_path.rstrip("/").split("/")[-1]
    params = "_".join(f"{k}={v}" for k, v in sorted(attack_kwargs.items()))
    mask = "_mask=kept" if keep_mask else ""
    return (f"{base}_{attack_name}_{params}_bn={bn_policy}_atn={train_size}_atepochs={epochs}"
            f"_lr={lr:g}_lambda={lam:g}_advfrac={adv_fraction:g}{mask}_run={run_tag}_atseed={at_seed}_mix=add")


def parse_sources(cfg):
    """Model_Path entries -> [(path, alias or None, keep_mask)].

    An entry is either a path string (old form) or a mapping
      {Path: /pruning_final/<folder>, Alias: <short name for output folders>, Keep_Zero_Mask: true}
    Alias only shortens the output folder name (the real path is written to <run>/source.json);
    Keep_Zero_Mask re-zeroes, after every optimiser step, the conv / linear weights that are exactly
    zero in the source (a pruned model stays pruned).
    """
    out = []
    for entry in cfg.get("Model_Path") or []:
        if isinstance(entry, str):
            out.append((entry, None, False))
        elif isinstance(entry, dict) and entry.get("Path"):
            alias = entry.get("Alias")
            if alias is not None and not re.fullmatch(r"[A-Za-z0-9=._-]{1,80}", str(alias)):
                raise ValueError(f"Alias must be 1-80 of [A-Za-z0-9=._-]: {alias!r}")
            out.append((entry["Path"], alias, bool(entry.get("Keep_Zero_Mask", False))))
        else:
            raise ValueError(f"Model_Path entry must be a path or a mapping with Path: {entry!r}")
    paths = [p for p, _, _ in out]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate Model_Path entries")
    return out


def zero_masks(net):
    """{param name: bool mask of non-zero entries} for every conv / linear weight that contains zeros."""
    masks = {}
    for name, p in net.named_parameters():
        if name.endswith("weight") and p.ndim in (2, 4):
            nz = p.detach() != 0
            if not bool(nz.all()):
                masks[name] = nz
    return masks


def keep_zero_mask(optimizer, net, masks):
    """Wrap optimizer.step so masked (pruned) weights are reset to zero after every update."""
    params = dict(net.named_parameters())
    original_step = optimizer.step

    def step(*args, **kwargs):
        out = original_step(*args, **kwargs)
        with torch.no_grad():
            for name, m in masks.items():
                params[name].mul_(m)
        return out
    optimizer.step = step
    return optimizer


def conv_linear_sparsity(net):
    total = zero = 0
    for name, p in net.named_parameters():
        if name.endswith("weight") and p.ndim in (2, 4):
            total += p.numel()
            zero += int((p.detach() == 0).sum())
    return zero / total if total else 0.0


def source_checkpoint(model_path):
    return Path("./saved_models" + model_path) / SOURCE_CKPT


def load_state(path, map_location="cpu"):
    """Bare state_dict; NormalizedModel dicts (base_model. prefix + mean/std) are unwrapped."""
    state = torch.load(path, map_location=map_location, weights_only=False)
    if any(k.startswith("base_model.") for k in state):
        state = {k[len("base_model."):]: v for k, v in state.items() if k.startswith("base_model.")}
    return state


def load_source(model_name, num_classes, model_path, dataset_obj):
    """Fresh backbone with the source weights (strict), plus its NormalizedModel wrapper."""
    net = build_model(model_name, num_classes).to(device)
    net.load_state_dict(load_state(source_checkpoint(model_path), device), strict=True)
    model = NormalizedModel(net, dataset_obj.mean, dataset_obj.std).to(device)
    return net, model


def check_disjoint_split(ds_cfg):
    """group_B 0.0 (AT data) must be group_size unique indices disjoint from the victim's group_A."""
    d = Path("Indices") / ds_cfg["name"]
    n = ds_cfg["group_size"]
    group_b = np.load(d / f"group_B_25000_0.0_{n}_seed42.npy", allow_pickle=False)
    group_a = np.load(d / f"group_A_{n}_seed42.npy", allow_pickle=False)
    if group_b.shape != (n,) or np.unique(group_b).size != n:
        raise ValueError(f"AT indices are not {n} unique entries")
    overlap = np.intersect1d(group_a, group_b).size
    if overlap:
        raise ValueError(f"AT indices overlap the victim's group_A in {overlap} images")
    return n


def bn_running_stats(net):
    return {k: v.detach().clone() for k, v in net.state_dict().items()
            if k.endswith("running_mean") or k.endswith("running_var")}


def expand_plan(yaml_path, at_seeds, run_tag, *, eps=None, select=None, epochs=None, lr=None):
    """Read-only expansion: [(model_path, attack_fn, attack_name, kwargs, at_seed, scenario)]."""
    cfg = process_yaml_file(str(yaml_path)) if not isinstance(yaml_path, dict) else yaml_path
    attacks, _, lam, bn_policy = parse_posthoc_config(cfg)
    adv_fraction = parse_adv_fraction(cfg)
    attacks = filter_attacks(attacks, eps)
    sources = parse_sources(cfg)
    if select:
        sources = [s for s in sources if any(t in s[0] or t in (s[1] or "") for t in select)]
    if not sources:
        raise ValueError("No Model_Path entry left (plan empty or --select matched nothing).")
    n_epochs = int(epochs if epochs is not None else cfg["Optimizer"]["Epochs"])
    lr_value = float(lr if lr is not None else cfg["Optimizer"]["params"]["lr"])
    runs = []
    for path, alias, keep_mask in sources:
        for at_seed in at_seeds:
            for attack_fn, attack_name, kwargs in attacks:
                name = posthoc_scenario_name(path, attack_name, kwargs, bn_policy=bn_policy,
                                             train_size=cfg["Dataset"]["group_size"], epochs=n_epochs,
                                             lr=lr_value, lam=lam, run_tag=run_tag, at_seed=at_seed,
                                             adv_fraction=adv_fraction, base=alias, keep_mask=keep_mask)
                runs.append((path, attack_fn, attack_name, kwargs, at_seed, name))
    return cfg, runs


def check_path_lengths(scenarios):
    """Fail early on Windows when the longest output file path exceeds MAX_PATH."""
    if os.name != "nt" or not scenarios:
        return
    longest = max(scenarios, key=len)
    paths = [(OUT_MODEL_ROOT / longest / "best_clean_epoch.pth").resolve(),
             (OUT_LOG_ROOT / f"training_log_{longest}.csv").resolve()]
    worst = max(len(str(p)) for p in paths)
    if worst > WINDOWS_MAX_PATH:
        raise OSError(f"Output paths reach {worst} characters (> {WINDOWS_MAX_PATH}); move the project to a "
                      f"directory at most {len(str(ROOT)) - (worst - WINDOWS_MAX_PATH)} characters long "
                      f"(now {len(str(ROOT))}: {ROOT}) or enable Windows LongPathsEnabled.")
    print(f"Longest output path: {worst} characters (limit {WINDOWS_MAX_PATH}).")


def run_status(scenario, epochs):
    """'done' (last checkpoint exists), 'partial' (some output exists) or 'new'."""
    save_dir = OUT_MODEL_ROOT / scenario
    if (save_dir / f"epoch_{epochs - 1}.pth").is_file():
        return "done"
    if save_dir.exists() or (OUT_LOG_ROOT / f"training_log_{scenario}.csv").exists():
        return "partial"
    return "new"


# =====================================================
# 2. Epoch functions
# =====================================================
def joint_adversarial_step(model, images, target, loss_fn, adv_generator, clean_weight, adv_weight,
                           adv_fraction=1.0):
    """x_adv from the eval-mode model, then ONE train-mode forward on cat([x, x_adv]).

    x_adv is built for the first k = round(adv_fraction * B) images of the (shuffled) batch
    (k = B by default). Loss = clean_weight * CE(all B clean) + adv_weight * CE(k adversarial),
    each a batch mean, i.e. with k = B the same objective as mixed_adversarial_step; only the
    BatchNorm statistics differ (shared over the B + k mixed batch; running statistics track
    that mixture, adversarial share k / (B + k)).
    """
    b = images.size(0)
    k = max(1, round(adv_fraction * b))
    adv_target = target[:k]
    model.eval()
    adv_images = adv_generator(model, images[:k], adv_target).detach()
    model.train()
    logits = model(torch.cat([images, adv_images]))
    clean_logits, adv_logits = logits[:b], logits[b:]
    clean_loss, adv_loss = loss_fn(clean_logits, target), loss_fn(adv_logits, adv_target)
    return {"loss": clean_weight * clean_loss + adv_weight * adv_loss, "clean_logits": clean_logits,
            "adv_logits": adv_logits, "clean_loss": clean_loss, "adv_loss": adv_loss, "adv_target": adv_target}


def posthoc_one_epoch(model, net, train_loader, optimizer, criterion, adv_generator, clean_weight, adv_weight,
                      device='cuda', max_batches=None, bn_policy="joint", adv_fraction=1.0):
    """One epoch; returns ft_at_one_epoch's dict (Train_Acc etc. on the clean half,
    train_adv_acc over the adversarial copies)."""
    if bn_policy != "joint":
        if adv_fraction != 1.0:
            raise ValueError("adv_fraction < 1 is only implemented for bn_policy=joint")
        return ft_at_one_epoch(model, net, train_loader, optimizer, criterion, adv_generator, "FT-AL",
                               clean_weight, adv_weight, device, max_batches, bn_policy)
    running_loss, total, adv_total, clean_correct, adv_correct = 0.0, 0, 0, 0, 0
    all_preds, all_targets = [], []
    for batch_idx, (images, targets) in enumerate(train_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = joint_adversarial_step(model, images, targets, criterion, adv_generator, clean_weight, adv_weight,
                                     adv_fraction)
        out["loss"].backward()
        optimizer.step()
        n = targets.size(0)
        running_loss += out["loss"].item() * n
        total += n
        adv_total += out["adv_target"].size(0)
        clean_correct += batch_accuracy(out["clean_logits"], targets)
        adv_correct += batch_accuracy(out["adv_logits"], out["adv_target"])
        all_preds.append(out["clean_logits"].argmax(1).detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
    all_preds, all_targets = np.concatenate(all_preds), np.concatenate(all_targets)
    from sklearn.metrics import f1_score, precision_score, recall_score
    return {
        "train_loss": running_loss / total,
        "train_acc": 100.0 * clean_correct / total,
        "train_precision": precision_score(all_targets, all_preds, average="weighted", zero_division=0),
        "train_recall": recall_score(all_targets, all_preds, average="weighted", zero_division=0),
        "train_f1": f1_score(all_targets, all_preds, average="weighted", zero_division=0),
        "train_clean_acc": 100.0 * clean_correct / total,
        "train_adv_acc": 100.0 * adv_correct / adv_total,
    }


# =====================================================
# 3. One run = one source x attack x AT seed
# =====================================================
def run_one(exp_yaml, dataset_obj, num_classes, at_train, raw_test_set, model_path, attack_fn, attack_name,
            attack_kwargs, at_seed, scenario, *, clean_w, adv_w, bn_policy, epochs, lr=None, max_batches=None,
            robust_n=None, adv_fraction=1.0, keep_mask=False):
    set_seed(at_seed, deterministic=DETERMINISTIC)
    g = torch.Generator().manual_seed(at_seed)
    save_dir = OUT_MODEL_ROOT / scenario
    log_file = OUT_LOG_ROOT / f"training_log_{scenario}.csv"
    print(f"\n{'=' * 70}\nSource : {model_path}\nAttack : {attack_name} {attack_kwargs} | lambda={adv_w:g}"
          f" | adv_fraction={adv_fraction:g} | bn={bn_policy} | epochs={epochs}\nOutput : {scenario}\n{'=' * 70}")

    net, model = load_source(exp_yaml.get("Model", "ResNet-18"), num_classes, model_path, dataset_obj)
    source_stats = bn_running_stats(net)
    masks = zero_masks(net) if keep_mask else {}
    source_sparsity = conv_linear_sparsity(net)
    if keep_mask:
        print(f"  Keep_Zero_Mask: {len(masks)} weight tensors, conv/linear sparsity {source_sparsity:.4f} kept")
    adv_generator = make_adv_generator(attack_fn, attack_kwargs)
    criterion = nn.CrossEntropyLoss()

    trainloader = DataLoader(at_train, batch_size=exp_yaml.get("BatchSize", 128), shuffle=True, num_workers=4,
                             worker_init_fn=seed_worker, generator=g, persistent_workers=True, pin_memory=True)
    testloader = DataLoader(dataset_obj.test_set, batch_size=256, shuffle=False, num_workers=2,
                            persistent_workers=True, pin_memory=True)
    rob_set = raw_test_set
    if robust_n and robust_n < len(raw_test_set):
        rob_set = Subset(raw_test_set, list(range(robust_n)))
    robust_n_used = len(rob_set)
    raw_testloader = DataLoader(rob_set, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(OUT_LOG_ROOT, exist_ok=True)
    with open(save_dir / "source.json", "w", encoding="utf-8") as f:     # provenance (folder may use an Alias)
        json.dump({"source_path": model_path, "source_checkpoint": str(source_checkpoint(model_path)),
                   "keep_zero_mask": keep_mask, "source_conv_linear_sparsity": source_sparsity,
                   "attack": attack_name, "attack_kwargs": attack_kwargs, "at_seed": at_seed}, f, indent=2)
    with open(log_file, 'w', newline='') as f:
        csv.writer(f).writerow(LOG_COLUMNS)

    def log_row(epoch, tr, te, rob, lr_now, seconds):
        tr = tr or {}
        with open(log_file, 'a', newline='') as f:
            csv.writer(f).writerow([
                scenario, epoch, tr.get("train_loss", 0), tr.get("train_acc", 0), tr.get("train_precision", 0),
                tr.get("train_recall", 0), tr.get("train_f1", 0),
                te["test_loss"], te["test_acc"], te["test_precision"], te["test_recall"], te["test_f1"],
                round(rob, 4), round(tr.get("train_clean_acc", float('nan')), 4),
                round(tr.get("train_adv_acc", float('nan')), 4), lr_now, round(seconds, 1),
                attack_name, attack_kwargs.get("eps"), attack_kwargs.get("steps"), adv_w, bn_policy, robust_n_used,
            ])

    model.eval()
    pre = evaluate1(net, testloader, criterion, device)
    pre_rob = 100.0 * compute_robust_test_accuracy(model, raw_testloader, attack_fn, attack_kwargs)
    print(f"  [Epoch -1] source: clean {pre['test_acc']:.2f}% | robust {pre_rob:.2f}% (n={robust_n_used})")
    log_row(-1, None, pre, pre_rob, 0, 0)
    if SAVE_ALL_EPOCHS:
        torch.save(net.state_dict(), save_dir / "epoch_-1.pth")        # the source itself: trajectory start

    opt_yaml = dict(exp_yaml, Optimizer=dict(exp_yaml["Optimizer"], Epochs=epochs,
                                             params=dict(exp_yaml["Optimizer"]["params"],
                                                         **({"lr": lr} if lr is not None else {}))))
    optimizer, scheduler, _ = initialize_optimizer_scheduler(opt_yaml, net)
    if keep_mask:
        keep_zero_mask(optimizer, net, masks)

    best_clean, best_rob, best_clean_ep, best_rob_ep = -1.0, -1.0, None, None
    best_from = int(epochs * BEST_CKPT_START_FRAC)
    for epoch in range(epochs):
        t0 = time.time()
        tr = posthoc_one_epoch(model, net, trainloader, optimizer, criterion, adv_generator,
                               clean_w, adv_w, device, max_batches, bn_policy, adv_fraction)
        wait_for_cool_gpu(threshold=89.5)
        lr_now = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()
        model.eval()
        te = evaluate1(net, testloader, criterion, device)
        rob = 100.0 * compute_robust_test_accuracy(model, raw_testloader, attack_fn, attack_kwargs)
        seconds = time.time() - t0
        print(f"  [Epoch {epoch}/{epochs}] LR {lr_now:.6f} | Loss {tr['train_loss']:.4f}"
              f" | Train clean {tr['train_clean_acc']:.2f}% adv {tr['train_adv_acc']:.2f}%"
              f" | Test clean {te['test_acc']:.2f}% robust {rob:.2f}% | {seconds:.0f}s", flush=True)
        log_row(epoch, tr, te, rob, lr_now, seconds)
        if epoch >= best_from and te["test_acc"] > best_clean:
            best_clean, best_clean_ep = te["test_acc"], epoch
            torch.save(net.state_dict(), save_dir / "best_clean_epoch.pth")
        if epoch >= best_from and rob > best_rob:
            best_rob, best_rob_ep = rob, epoch
            torch.save(net.state_dict(), save_dir / "best_rob_epoch.pth")
        # epoch_<E-1>.pth is written last: its existence marks the run as finished (run_status)
        if SAVE_ALL_EPOCHS or epoch == epochs - 1:
            torch.save(net.state_dict(), save_dir / f"epoch_{epoch}.pth")

    # ----- Post-run checks: frozen stats untouched, best_clean reloads to the logged accuracy -----
    if bn_policy == "frozen":
        drift = max((bn_running_stats(net)[k] - v).abs().max().item() for k, v in source_stats.items())
        print(f"  BN running statistics max drift vs source: {drift:.2e}")
        if drift > 0:
            raise RuntimeError("bn_policy=frozen but running statistics changed")
    if keep_mask:
        params = dict(net.named_parameters())
        regrown = sum(int((params[n].detach()[~m] != 0).sum()) for n, m in masks.items())
        print(f"  Zero mask: {regrown} regrown weights, sparsity {conv_linear_sparsity(net):.4f} "
              f"(source {source_sparsity:.4f})")
        if regrown:
            raise RuntimeError("Keep_Zero_Mask set but pruned weights became non-zero")
    check = build_model(exp_yaml.get("Model", "ResNet-18"), num_classes).to(device)
    check.load_state_dict(torch.load(save_dir / "best_clean_epoch.pth", map_location=device), strict=True)
    reloaded = evaluate1(check, testloader, criterion, device)["test_acc"]
    print(f"==> done. source clean {pre['test_acc']:.2f}% -> best_clean {best_clean:.2f}% @ epoch {best_clean_ep}"
          f" (reload {reloaded:.2f}%) | best robust {best_rob:.2f}% @ epoch {best_rob_ep} | window from epoch {best_from}")
    return {"source_clean": pre["test_acc"], "source_robust": pre_rob, "best_clean": best_clean,
            "best_robust": best_rob, "reloaded_clean": reloaded, "best_clean_epoch": best_clean_ep,
            "best_rob_epoch": best_rob_ep}


# =====================================================
# 4. Entry point
# =====================================================
def main(argv=None):
    p = argparse.ArgumentParser(description="Clean-preserving post-hoc adversarial fine-tuning.")
    p.add_argument("--plans", nargs="+", type=Path, help="default: saved_exp_plan/at_posthoc_plan/*.yaml")
    p.add_argument("--select", nargs="+", help="keep Model_Path entries containing any of these substrings")
    p.add_argument("--eps", nargs="+", type=float, help="subset of the plan's eps grid")
    p.add_argument("--seeds", nargs="+", type=int, default=[0], help="AT seeds (default 0)")
    p.add_argument("--run-tag", default="v1")
    p.add_argument("--lr", type=float, help="override Optimizer.params.lr (pilots); recorded in the name")
    p.add_argument("--epochs", type=int, help="override Optimizer.Epochs (smoke / pilots); recorded in the name")
    p.add_argument("--max-batches", type=int, help="smoke tests: at most N batches per epoch")
    p.add_argument("--robust-n", type=int, help="robust accuracy on the first N test images (default all 10000)")
    p.add_argument("--force", action="store_true", help="overwrite finished or partial outputs")
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args(argv)
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        p.error("--seeds must be distinct non-negative integers")

    os.chdir(ROOT)
    yaml_files = args.plans or sorted(DEFAULT_PLAN_DIR.glob("*.yaml"))
    if not yaml_files:
        p.error(f"No YAML plans in {DEFAULT_PLAN_DIR}")
    todo = []
    for path in yaml_files:
        cfg, runs = expand_plan(path, args.seeds, args.run_tag, eps=args.eps, select=args.select,
                                epochs=args.epochs, lr=args.lr)
        check_disjoint_split(cfg["Dataset"])
        epochs = int(args.epochs or cfg["Optimizer"]["Epochs"])
        print(f"Plan {path.name}: {len(runs)} run(s), {epochs} epochs, lr {args.lr or cfg['Optimizer']['params']['lr']}")
        for model_path, *_rest, scenario in runs:
            if not source_checkpoint(model_path).is_file():
                raise FileNotFoundError(f"Missing source checkpoint {source_checkpoint(model_path)}")
            status = run_status(scenario, epochs)
            if status == "partial" and not args.force and not args.check_only:
                raise FileExistsError(f"Partial outputs for {scenario}; rerun with --force to overwrite")
            print(f"  [{status}] {scenario}")
        todo.append((path, cfg, runs, epochs))
    check_path_lengths([r[-1] for _, _, runs, _ in todo for r in runs])
    if args.check_only:
        print("Preflight passed. No training started.")
        return

    torch.multiprocessing.set_start_method("spawn", force=True)
    summary = []
    for path, cfg, runs, epochs in todo:
        _, clean_w, adv_w, bn_policy = parse_posthoc_config(cfg)
        adv_fraction = parse_adv_fraction(cfg)
        keep = {path: km for path, _alias, km in parse_sources(cfg)}
        dataset_obj, num_classes, _ = build_dataset_from_yaml(cfg["Dataset"])
        at_train, raw_test_set = build_at_dataset_from_yaml(dataset_obj, cfg["Dataset"], rate=0.0)
        for model_path, attack_fn, attack_name, kwargs, at_seed, scenario in runs:
            if run_status(scenario, epochs) == "done" and not args.force:
                print(f"[SKIP] {scenario}: finished")
                continue
            res = run_one(cfg, dataset_obj, num_classes, at_train, raw_test_set, model_path, attack_fn, attack_name,
                          kwargs, at_seed, scenario, clean_w=clean_w, adv_w=adv_w, bn_policy=bn_policy,
                          epochs=epochs, lr=args.lr, max_batches=args.max_batches, robust_n=args.robust_n,
                          adv_fraction=adv_fraction, keep_mask=keep[model_path])
            summary.append((scenario, res))
    print("\n===== Summary (clean % source -> best_clean @epoch, best robust % @epoch) =====")
    for scenario, r in summary:
        print(f"  {r['source_clean']:6.2f} -> {r['best_clean']:6.2f} @{r['best_clean_epoch']}"
              f"  rob {r['best_robust']:6.2f} @{r['best_rob_epoch']}  {scenario}")


if __name__ == "__main__":
    main()
