"""main_dfms_extraction.py

DFMS-HL model extraction, framework-aligned with main_knockoff_extraction.py
and main_jba_extraction.py.

    Sanyal, Addepalli, Babu. "Towards Data-Free Model Stealing in a Hard
    Label Setting", CVPR 2022.  github.com/val-iisc/Hard-Label-Model-Stealing

Threat model
    The attacker sees only argmax labels from the victim, never its training
    data. It owns an unlabelled proxy set (natural images from unrelated
    classes, or synthetic shapes) that seeds a DCGAN; generator and clone are
    then trained in alternation. The generator never queries the victim: its
    gradient comes from the clone (class-diversity loss) and from a
    discriminator that keeps it on the proxy manifold. Every victim query is
    spent labelling clone training data.

Pipeline, one process, every stage cached in the output folder
    S1  train DCGAN on the proxy                              -> netG_dcgan.pth
    S2  20k G samples + victim labels = attacker's val set    -> val_dcgan.pt
    S3  clone from scratch on victim-labelled proxy + G data  -> clone_stage3_last.pth
    S4  DivGAN: G vs frozen S3 clone, lambda_div = 10         -> netG_divgan.pth
    S5  clone from scratch on victim-labelled proxy + DivGAN  -> clone_stage5_last.pth
    S6  20k DivGAN samples + victim labels                     -> val_divgan.pt
    S7  alternating G / clone training, lambda_div = 500|100  -> best_epoch.pth ...
    S7 is where the query budget goes: batch_size per iteration, i.e.
    epochs * ceil(N_proxy / 64) * 64  (8M for the paper's CIFAR-10 setting).

Framework mapping
    victim      saved_models/vanilla/{CNN,Transformer}_Models/<Model_Name>_<seed>_1.0
    proxy       the proxy dataset's group_B (disjoint from any victim's group_A),
                optionally restricted to the paper's 40 unrelated CIFAR-100
                classes, or a locally generated synthetic-shape set
    clone       trained in the victim dataset's normalisation, so the checkpoint
                is consumed unchanged by calculate_MI_extraction.py and
                build_mea_matrices.py
    outputs     saved_models/extraction_final/<Scenario>_<seed>_1.0/
                saved_logs/extraction_final/Performance/
                    (flat, no Transformer_Models level: the layout the knockoff
                    scripts and calculate_MI_extraction.py's best path now use.
                    Runs before 2026-09-20 went to extraction_vanilla/ with a
                    Transformer_Models/ level for DeiT clones.)
                    training_log_<Scenario>_<seed>_1.0.csv   (clone stages, same
                    leading columns as the knockoff log, plus attack columns)
                    gan_log_<Scenario>_<seed>_1.0.csv        (GAN stages)

Checkpoint selection
    best_epoch.pth      best test accuracy, the convention shared with the
                        other MEA scripts and read downstream
    best_val_epoch.pth  best balanced agreement on the DivGAN val set, the
                        attacker-legal criterion the reference code uses
    final_round.pth     clone at the end of S7

Deliberate deviations from the reference code
    * The EMA / "tau" weight-averaging bookkeeping is dropped; the run
      scripts never switch on the flags that consume it.
    * S7 warmup is a clean linear ramp; the reference sets lr = 0.001*epoch
      after the epoch ends, so its epoch 1 runs at lr 0.
    * Proxy labels are queried once and cached (an attacker would); the
      reference re-queries them in S5, which only inflates the count.
    * RandomCrop / Flip / AutoAugment run on GPU tensors; padding fill is
      pixel 0, as everywhere else in this repo.
    * S3/S5 lock-step over separate proxy / GAN loaders is kept, including
      its consequence that only min(steps) batches of each pool are seen per
      epoch. mix_ratio in the plan controls the per-step composition.

Seeds
    MODEL_SEED (default 42) selects the victim; SEED_START / SEED_END
    (default 0 / 1) are the attacker seeds, named _<seed>_1.0 in every output,
    matching the knockoff convention. DFMS_SMOKE=1 shrinks every stage to a
    few iterations and redirects outputs to the scratchpad.
"""
import os
import glob
import json
import csv
import sys
import time
import random
from pathlib import Path

# util.load_best_checkpoint prints an emoji and a Windows console / redirected
# stdout defaults to cp1252, which raised UnicodeEncodeError mid-run. Doing it
# here means no PYTHONIOENCODING=utf-8 is needed on the command line.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader, TensorDataset

from util import (process_yaml_file, build_dataset_from_yaml,
                  create_or_load_group_A, create_or_load_group_B,
                  load_best_checkpoint, evaluate1, evaluate_fidelity,
                  build_deit_student, build_warmup_cosine_scheduler, get_targets)
from Model.ResNet_18 import ResNet18
from Model.VGG16 import ModifiedVGG16
from Model.MLP import MNIST_MLP
from Model.DCGAN import Generator, Discriminator, weights_init
from AdvAttack.dfms_hl import (ImageSpace, VictimOracle, ProxyBank, BatchAutoAugment,
                               resolve_class_filter, natural_proxy_uint8,
                               generate_synthetic_shapes, balanced_agreement, predict,
                               sample_generator, make_val_set, train_dcgan,
                               train_clone_offline, train_divgan, alternate_train,
                               logits_of)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
DETERMINISTIC = False          # cudnn.benchmark matters over 125k GAN iterations

MODEL_SEED = int(os.environ.get("MODEL_SEED", 42))
SEED_START = int(os.environ.get("SEED_START", 0))
SEED_END = int(os.environ.get("SEED_END", 1))
SMOKE = os.environ.get("DFMS_SMOKE", "0") == "1"
# A shell that once exported SEED_END (e.g. =5 for the A1 tier's seeds 0..4)
# keeps it for the life of that window and silently multiplies every later run,
# so the banner has to say which values did not come from the defaults above.
SEED_OVERRIDES = {k: os.environ[k] for k in ("MODEL_SEED", "SEED_START", "SEED_END")
                  if k in os.environ}
# DFMS_OUT_ROOT redirects saved_models/ and saved_logs/ (smoke tests, timing
# pilots). Unset = the production locations shared with the knockoff scripts.
OUT_ROOT = os.environ.get("DFMS_OUT_ROOT", None)
if SMOKE and OUT_ROOT is None:
    OUT_ROOT = os.path.join(os.environ.get("TEMP", "."), "dfms_smoke")

# Output tree, matching main_knockoff_extraction{,_deit}.py: one flat folder for
# every substitute architecture. calculate_MI_extraction.py's best path
# (EXTRACTION_BEST_MODEL_DIR) globs exactly this layout.
# Old location, used by runs before 2026-09-20:
#   EXTRACTION_SUBDIR = 'extraction_vanilla' + ('/Transformer_Models' if is_deit_clone else '')
EXTRACTION_SUBDIR = 'extraction_final'
EXTRACTION_LOG_SUBDIR = 'extraction_final'
# A finished run is one that left a non-empty best_epoch.pth. Set False to
# force a recompute (state.json still caches the individual stages).
SKIP_EXISTING = True

CLONE_LOG_HEADER = ['Scenario', 'Stage', 'Epoch', 'Queries',
                    'Train_Loss', 'Train_Acc', 'Train_Precision', 'Train_Recall', 'Train_F1',
                    'Test_Loss', 'Test_Acc', 'Test_Precision', 'Test_Recall', 'Test_F1',
                    'Fidelity', 'Val_DCGAN_Agree', 'Val_DivGAN_Agree', 'Proxy_Agree',
                    'Label_Entropy', 'LR', 'Epoch_Time']
GAN_LOG_HEADER = ['Scenario', 'Stage', 'Epoch', 'Queries',
                  'Loss_D_real', 'Loss_D_fake', 'Loss_G_adv', 'Loss_Div', 'Loss_Ent',
                  'D_x', 'D_G_z', 'Acc_D', 'Label_Entropy', 'Epoch_Time']


# =====================================================
# 1. Global setup
# =====================================================
def set_seed(seed, deterministic=DETERMINISTIC):
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


def build_model(model_name, num_classes):
    if model_name == "MLP":
        return MNIST_MLP()
    if model_name == "ResNet-18":
        return ResNet18(num_classes=num_classes)
    if model_name == "VGG16":
        return ModifiedVGG16(num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_name}")


def build_model_any(model_cfg, num_classes):
    """str -> CNN from Model/, dict -> timm DeiT via build_deit_student."""
    if isinstance(model_cfg, dict):
        return build_deit_student({"Model": model_cfg}, num_classes)
    return build_model(model_cfg, num_classes)


def make_optimizer(cfg, params, default):
    cfg = cfg or default
    return getattr(optim, cfg["name"])(params, **cfg.get("params", {}))


def make_scheduler(cfg, optimizer, total_epochs):
    """Length-bound to the stage: cosine T_max is always the stage's epoch
    count (rescale, never truncate)."""
    cfg = cfg or {"name": "CosineAnnealingLR", "params": {}}
    name = cfg.get("name", "CosineAnnealingLR")
    p = dict(cfg.get("params", {}))
    if name == "WarmupCosineAnnealingLR":
        warmup = min(int(p.get("warmup_epochs", 10)), max(0, total_epochs - 1))
        return build_warmup_cosine_scheduler(optimizer, total_epochs, warmup,
                                             float(p.get("warmup_start_factor", 0.1)),
                                             float(p.get("eta_min", 0.0)))
    if name == "CosineAnnealingLR":
        p["T_max"] = total_epochs
        return lr_sched.CosineAnnealingLR(optimizer, **p)
    return getattr(lr_sched, name)(optimizer, **p)


def append_row(path, header, row):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def fmt(v):
    return f"{v:.2f}" if isinstance(v, float) else str(v)


# =====================================================
# 2. Main DFMS-HL extraction logic
# =====================================================
def main_dfms(model_seed, extract_seed, yaml_file_path):
    print(device)
    exp_yaml = process_yaml_file(yaml_file_path)

    # ----- Whole-run skip (same guard as main_knockoff_extraction.py) -----
    # state.json already caches the seven stages, but reaching that check still
    # costs a victim load, a dataset build, the proxy labelling and the closing
    # evaluation -- minutes per finished run. A shipped bundle that carries
    # results computed elsewhere wants those runs gone instantly, so a complete
    # best_epoch.pth short-circuits before any of that work.
    if SKIP_EXISTING and not SMOKE:
        _scen = exp_yaml["Scenario_Name"] + f"_{extract_seed}_{1.0}"
        _root = OUT_ROOT if OUT_ROOT else "."
        _best = os.path.join(_root, "saved_models", EXTRACTION_SUBDIR, _scen, "best_epoch.pth")
        if os.path.isfile(_best) and os.path.getsize(_best) > 0:
            print(f"[SKIP] {_scen}: best_epoch.pth already exists -> {_best}")
            return

    dfms = dict(exp_yaml.get("DFMS", {}))
    epochs = dict(dfms.get("epochs", {}))
    gan_cfg = dict(dfms.get("gan", {}))
    max_iters = dfms.get("max_iters_per_epoch", None)
    if SMOKE:
        epochs = {"dcgan": 1, "clone_offline": 1, "divgan": 1, "alternate": 2}
        dfms["val_samples"] = 256
        dfms["max_samples"] = 512
        max_iters = 4
        print(f"[SMOKE] epochs={epochs}, 4 iterations per epoch, outputs under {OUT_ROOT}")

    nz = int(gan_cfg.get("nz", 100))
    gan_bs = int(gan_cfg.get("batch_size", 64))
    gan_lr = float(gan_cfg.get("lr", 2e-4))
    gan_beta1 = float(gan_cfg.get("beta1", 0.5))
    lambda_div = float(dfms.get("lambda_div", 500))
    lambda_cls = float(dfms.get("lambda_cls", 0))
    divgan_lambda_div = float(dfms.get("divgan_lambda_div", 10))
    temp = float(dfms.get("temp", 1.0))
    max_samples = int(dfms.get("max_samples", 50000))
    mix = dict(dfms.get("mix_ratio", {"proxy": 1.0, "gan": 0.5}))
    offline_bs = int(dfms.get("offline_batch_size", 128))
    val_samples = int(dfms.get("val_samples", 20000))
    grad_clip = dfms.get("grad_clip", None)
    autoaug = BatchAutoAugment("cifar10") if bool(dfms.get("autoaugment", True)) else None

    # ----- Victim -----
    victim_cfg = exp_yaml["Victim"]
    victim_ds_cfg = victim_cfg["Dataset"]
    victim_obj, num_classes, _ = build_dataset_from_yaml(victim_ds_cfg)
    victim_model_cfg = victim_cfg.get("Model", "ResNet-18")

    print('==> Loading victim model..')
    victim_net = build_model_any(victim_model_cfg, num_classes).to(device)
    victim_folder = victim_cfg["Model_Name"] + f"_{model_seed}_{1.0}"
    default_root = ('./saved_models/vanilla/Transformer_Models' if isinstance(victim_model_cfg, dict)
                    else './saved_models/vanilla/CNN_Models')
    victim_dir = Path(victim_cfg.get("Model_Dir", default_root)) / victim_folder
    ckpt_path, _ = load_best_checkpoint(victim_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"No victim checkpoint in {victim_dir}")
    victim_net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    victim_net.eval()
    for p in victim_net.parameters():
        p.requires_grad_(False)
    print(f'   Loaded: {ckpt_path}')

    space = ImageSpace(victim_obj.mean, victim_obj.std, device)
    oracle = VictimOracle(victim_net, device)

    # ----- Victim's test set, materialised once (evaluated every epoch for 100s of epochs) -----
    xs, ys = [], []
    for x, y in DataLoader(victim_obj.test_set, batch_size=1000, shuffle=False, num_workers=0):
        xs.append(x); ys.append(y)
    test_loader = DataLoader(TensorDataset(torch.cat(xs), torch.cat(ys)), batch_size=500, shuffle=False)
    eval_criterion = nn.CrossEntropyLoss()
    victim_test = evaluate1(victim_net, test_loader, eval_criterion, device)
    print(f'   Victim Test Acc: {victim_test["test_acc"]:.2f}%')

    # ----- Proxy -----
    proxy_cfg = exp_yaml["Proxy"]
    proxy_type = proxy_cfg.get("type", "natural")
    print(f'==> Preparing proxy ({proxy_type})..')
    if proxy_type == "natural":
        pds_cfg = proxy_cfg["Dataset"]
        proxy_obj, proxy_K, proxy_group = build_dataset_from_yaml(pds_cfg)
        train_set = proxy_obj.train_set
        idx_dir = f'./Indices/{pds_cfg["name"]}/'
        group_A = create_or_load_group_A(dataset=train_set, save_dir=idx_dir, group_size=proxy_group,
                                         num_classes=proxy_K, seed=42, force_rebuild=False)
        group_B = create_or_load_group_B(dataset=train_set, save_dir=idx_dir, group_A_indices=group_A,
                                         group_size=proxy_group, num_classes=proxy_K,
                                         overlap_rate=0.0, seed=42, force_rebuild=False)
        split = proxy_cfg.get("split", "group_B")
        if split != "group_B":
            raise ValueError("Proxy.split must be group_B: the attacker's pool is by construction "
                             "disjoint from every victim's group_A")
        pool = np.asarray(group_B)
        cls = resolve_class_filter(proxy_cfg.get("class_filter", "all"),
                                   getattr(train_set, "class_to_idx", {}))
        if cls is not None:
            pool = pool[np.isin(get_targets(train_set)[pool], cls)]
        proxy_uint8 = natural_proxy_uint8(proxy_obj, pool)
        proxy_desc = f'{pds_cfg["name"]} group_B, class_filter={proxy_cfg.get("class_filter", "all")}'
    elif proxy_type == "synthetic":
        s = dict(proxy_cfg.get("synthetic", {}))
        n_syn = int(s.get("num_images", 50000)) if not SMOKE else 512
        tag = (f'shapes_n{n_syn}_s{s.get("num_shapes", 50)}_{s.get("min_size", 5)}-{s.get("max_size", 10)}'
               f'_c{s.get("canvas", 100)}_{"grey" if s.get("grey", True) else "rgb"}_seed{s.get("seed", 0)}.pt')
        cache = os.path.join("./data/dfms_synthetic", tag)
        if os.path.exists(cache):
            proxy_uint8 = torch.load(cache)
        else:
            t0 = time.time()
            proxy_uint8 = generate_synthetic_shapes(n_syn, int(s.get("num_shapes", 50)),
                                                    int(s.get("min_size", 5)), int(s.get("max_size", 10)),
                                                    int(s.get("canvas", 100)), bool(s.get("grey", True)),
                                                    int(s.get("seed", 0)))
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            torch.save(proxy_uint8, cache)
            print(f'   generated {n_syn} synthetic images in {time.time() - t0:.0f}s -> {cache}')
        proxy_desc = f'synthetic {tag}'
    else:
        raise ValueError(f"Unknown Proxy.type: {proxy_type}")
    bank = ProxyBank(proxy_uint8, device)
    iters_per_epoch = bank.num_batches(gan_bs)
    print(f'   proxy: {proxy_desc} -> {len(bank)} images, {iters_per_epoch} iterations/epoch at bs {gan_bs}')
    print(f'   S7 budget: {epochs.get("alternate", 0)} epochs x {iters_per_epoch} x {gan_bs} '
          f'= {epochs.get("alternate", 0) * iters_per_epoch * gan_bs / 1e6:.2f}M queries')

    # ----- Clone -----
    sub_cfg = exp_yaml["Substitute"].get("Model", victim_model_cfg)
    is_deit_clone = isinstance(sub_cfg, dict)

    # ----- Output locations (knockoff convention) -----
    scenario_name = exp_yaml["Scenario_Name"] + f"_{extract_seed}_{1.0}"
    sub_root = EXTRACTION_SUBDIR
    root = OUT_ROOT if OUT_ROOT else "."
    model_dir = os.path.join(root, "saved_models", sub_root, scenario_name)
    log_dir = os.path.join(root, "saved_logs", EXTRACTION_LOG_SUBDIR, "Performance")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    clone_log = os.path.join(log_dir, f"training_log_{scenario_name}.csv")
    gan_log = os.path.join(log_dir, f"gan_log_{scenario_name}.csv")
    P = lambda name: os.path.join(model_dir, name)

    state_path = P("state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {"queries": 0, "done": []}
    oracle.queries = int(state["queries"])

    def stage_done(stage, output):
        state["queries"] = oracle.queries
        if stage not in state["done"]:
            state["done"].append(stage)
        json.dump(state, open(state_path, "w"), indent=2)
        print(f'   [{stage}] done -> {output}   (queries so far: {oracle.queries:,})')

    def cached(stage, output):
        if os.path.exists(P(output)) and stage in state["done"]:
            print(f'==> [{stage}] cached, skipping ({output})')
            return True
        return False

    # ----- Attacker-side validation sets and cached proxy labels -----
    vals = {"dcgan": None, "divgan": None}
    proxy_off = {"x": None, "y": None}

    def load_val(name):
        if vals[name] is None and os.path.exists(P(f"val_{name}.pt")):
            d = torch.load(P(f"val_{name}.pt"))
            vals[name] = (d["x"], d["y"])

    def evaluate_clone(clone):
        r = evaluate1(clone, test_loader, eval_criterion, device)
        out = {k: r[k] for k in ("test_loss", "test_acc", "test_precision", "test_recall", "test_f1")}
        out["fidelity"] = evaluate_fidelity(victim_net, clone, test_loader, device)
        for name in ("dcgan", "divgan"):
            load_val(name)
            out[f"val_{name}"] = (balanced_agreement(predict(clone, vals[name][0], space), vals[name][1], num_classes)
                                  if vals[name] is not None else "")
        out["proxy_agree"] = (100.0 * (predict(clone, proxy_off["x"], space) == proxy_off["y"]).float().mean().item()
                              if proxy_off["y"] is not None else "")
        return out

    def log_clone_epoch(stage, epoch, st, clone):
        ev = evaluate_clone(clone)
        row = [scenario_name, stage, epoch, oracle.queries,
               st.get("train_loss", ""), st.get("train_acc", ""), st.get("train_precision", ""),
               st.get("train_recall", ""), st.get("train_f1", ""),
               ev["test_loss"], ev["test_acc"], ev["test_precision"], ev["test_recall"], ev["test_f1"],
               ev["fidelity"], ev["val_dcgan"], ev["val_divgan"], ev["proxy_agree"],
               st.get("label_entropy", ""), st.get("lr", ""), st.get("epoch_time", "")]
        append_row(clone_log, CLONE_LOG_HEADER, row)
        print(f'   [{stage}] ep {epoch:4d} | train {fmt(st.get("train_loss", 0.0))}/{fmt(st.get("train_acc", 0.0))}% '
              f'| test {ev["test_acc"]:.2f}% fid {ev["fidelity"]:.2f}% '
              f'| val dcgan {fmt(ev["val_dcgan"])} divgan {fmt(ev["val_divgan"])} proxy {fmt(ev["proxy_agree"])} '
              f'| Hlab {fmt(st.get("label_entropy", ""))} | lr {st.get("lr", 0):.2e} | q {oracle.queries:,} '
              f'| {st.get("epoch_time", 0):.0f}s')
        return ev

    def log_gan_epoch(stage, epoch, st):
        row = [scenario_name, stage, epoch, oracle.queries,
               st.get("loss_d_real", ""), st.get("loss_d_fake", ""), st.get("loss_g_adv", ""),
               st.get("loss_div", ""), st.get("loss_ent", ""), st.get("D_x", ""), st.get("D_G_z", ""),
               st.get("acc_d", ""), st.get("label_entropy", ""), st.get("epoch_time", "")]
        append_row(gan_log, GAN_LOG_HEADER, row)
        print(f'   [{stage}] ep {epoch:4d} | D {fmt(st.get("loss_d_real", 0.0))}+{fmt(st.get("loss_d_fake", 0.0))} '
              f'G {fmt(st.get("loss_g_adv", 0.0))} div {fmt(st.get("loss_div", ""))} '
              f'| D(x) {fmt(st.get("D_x", 0.0))} D(G(z)) {fmt(st.get("D_G_z", 0.0))} accD {fmt(st.get("acc_d", 0.0))} '
              f'| Hlab {fmt(st.get("label_entropy", ""))} | {st.get("epoch_time", 0):.0f}s')

    def new_gan():
        G = Generator(nz=nz, ngf=int(gan_cfg.get("ngf", 64))).to(device); G.apply(weights_init)
        D = Discriminator(ndf=int(gan_cfg.get("ndf", 64))).to(device); D.apply(weights_init)
        return G, D

    def new_clone():
        return build_model_any(sub_cfg, num_classes).to(device)

    def offline_parts(G):
        """Victim-labelled proxy + generator samples for S3/S5 (train_student.py)."""
        n_proxy = min(len(bank), int(mix["proxy"] * max_samples))
        n_gan = int(mix["gan"] * max_samples)
        if proxy_off["y"] is None:
            if os.path.exists(P("proxy_labels.pt")):
                d = torch.load(P("proxy_labels.pt"))
                proxy_off["x"], proxy_off["y"] = bank.x[d["idx"].to(device)], d["y"].to(device)
            else:
                sel = torch.randperm(len(bank), device=device)[:n_proxy] if n_proxy < len(bank) \
                    else torch.arange(len(bank), device=device)
                x = bank.x[sel]
                y = oracle.hard_labels(space.to_victim(x.float().div(255.0)))
                torch.save({"idx": sel.cpu(), "y": y.cpu()}, P("proxy_labels.pt"))
                proxy_off["x"], proxy_off["y"] = x, y
                print(f'   labelled {len(y)} proxy images (queries {oracle.queries:,})')
        gan_x = sample_generator(G, n_gan, nz, device)
        gan_y = oracle.hard_labels(space.to_victim(gan_x))
        print(f'   labelled {n_gan} generator samples (queries {oracle.queries:,})')
        return [{"x": proxy_off["x"], "y": proxy_off["y"], "ratio": float(mix["proxy"])},
                {"x": gan_x, "y": gan_y, "ratio": float(mix["gan"])}]

    def run_offline(stage, G, out_last, out_bestval):
        clone = new_clone()
        opt = make_optimizer(exp_yaml.get("Optimizer"), clone.parameters(),
                             {"name": "SGD", "params": {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4}})
        sched = make_scheduler(exp_yaml.get("Scheduler"), opt, epochs["clone_offline"])
        best = {"val": -1.0}

        def on_epoch(epoch, st):
            ev = log_clone_epoch(stage, epoch, st, clone)
            v = ev["val_dcgan"] if ev["val_dcgan"] != "" else ev["test_acc"]
            if v > best["val"]:
                best["val"] = v
                torch.save(clone.state_dict(), P(out_bestval))

        parts = offline_parts(G)
        train_clone_offline(clone, parts, offline_bs, epochs["clone_offline"], opt, sched, space, device,
                            on_epoch, grad_clip=grad_clip, max_iters=max_iters)
        torch.save(clone.state_dict(), P(out_last))
        del parts
        return clone

    t_start = time.time()
    print(f'\n==> Scenario {scenario_name}  |  clone {"DeiT" if is_deit_clone else sub_cfg}  |  K={num_classes}')

    # ===== S1: DCGAN on the proxy =====
    if not cached("S1", "netG_dcgan.pth"):
        print('==> [S1] DCGAN pre-training on the proxy..')
        G, D = new_gan()
        train_dcgan(G, D, bank, space, epochs["dcgan"], gan_bs, nz, gan_lr, gan_beta1, device,
                    lambda e, st: log_gan_epoch("S1_dcgan", e, st), max_iters=max_iters)
        torch.save(G.state_dict(), P("netG_dcgan.pth"))
        stage_done("S1", "netG_dcgan.pth")
        del D
    G = Generator(nz=nz, ngf=int(gan_cfg.get("ngf", 64))).to(device)
    G.load_state_dict(torch.load(P("netG_dcgan.pth"), map_location=device))

    # ===== S2: DCGAN val set =====
    if not cached("S2", "val_dcgan.pt"):
        print('==> [S2] Victim-labelled DCGAN val set..')
        x, y = make_val_set(G, oracle, space, val_samples, nz, device)
        torch.save({"x": x, "y": y}, P("val_dcgan.pt"))
        stage_done("S2", "val_dcgan.pt")
    load_val("dcgan")

    # ===== S3: clone from scratch on proxy + DCGAN samples =====
    if not cached("S3", "clone_stage3_last.pth"):
        print('==> [S3] Offline clone training (proxy + DCGAN samples)..')
        run_offline("S3_clone_offline", G, "clone_stage3_last.pth", "clone_stage3_bestval.pth")
        stage_done("S3", "clone_stage3_last.pth")
    if proxy_off["y"] is None and os.path.exists(P("proxy_labels.pt")):
        d = torch.load(P("proxy_labels.pt"))
        proxy_off["x"], proxy_off["y"] = bank.x[d["idx"].to(device)], d["y"].to(device)

    # ===== S4: DivGAN against the frozen S3 clone =====
    if not cached("S4", "netG_divgan.pth"):
        print(f'==> [S4] DivGAN (lambda_div={divgan_lambda_div}) against the S3 clone..')
        clone = new_clone()
        clone.load_state_dict(torch.load(P("clone_stage3_last.pth"), map_location=device))
        _, D = new_gan()                     # fresh discriminator, as in the reference
        train_divgan(G, D, clone, bank, space, epochs["divgan"], gan_bs, nz, gan_lr, gan_beta1,
                     divgan_lambda_div, lambda_cls, temp, num_classes, device,
                     lambda e, st: log_gan_epoch("S4_divgan", e, st), max_iters=max_iters)
        torch.save(G.state_dict(), P("netG_divgan.pth"))
        stage_done("S4", "netG_divgan.pth")
        del clone, D
    G.load_state_dict(torch.load(P("netG_divgan.pth"), map_location=device))

    # ===== S5: clone from scratch on proxy + DivGAN samples =====
    if not cached("S5", "clone_stage5_last.pth"):
        print('==> [S5] Offline clone training (proxy + DivGAN samples)..')
        run_offline("S5_clone_offline", G, "clone_stage5_last.pth", "clone_stage5_bestval.pth")
        stage_done("S5", "clone_stage5_last.pth")

    # ===== S6: DivGAN val set =====
    if not cached("S6", "val_divgan.pt"):
        print('==> [S6] Victim-labelled DivGAN val set..')
        x, y = make_val_set(G, oracle, space, val_samples, nz, device)
        torch.save({"x": x, "y": y}, P("val_divgan.pt"))
        stage_done("S6", "val_divgan.pt")
    load_val("divgan")

    # ===== S7: alternating training =====
    if not cached("S7", "final_round.pth"):
        print(f'==> [S7] Alternating G / clone training (lambda_div={lambda_div}, '
              f'{epochs["alternate"]} epochs, autoaugment={autoaug is not None})..')
        clone = new_clone()
        clone.load_state_dict(torch.load(P("clone_stage5_last.pth"), map_location=device))
        _, D = new_gan()
        alt = exp_yaml.get("Alternate", {})
        optC = make_optimizer(alt.get("Optimizer"), clone.parameters(),
                              {"name": "SGD", "params": {"lr": 0.01, "momentum": 0.9, "weight_decay": 5e-4}})
        schedC = make_scheduler(alt.get("Scheduler", {"name": "WarmupCosineAnnealingLR",
                                                       "params": {"warmup_epochs": 10, "warmup_start_factor": 0.1}}),
                                optC, epochs["alternate"])
        best = {"test": -1.0, "val": -1.0}

        def on_epoch(epoch, st):
            ev = log_clone_epoch("S7_alternate", epoch, st, clone)
            log_gan_epoch("S7_alternate", epoch, st)
            if ev["test_acc"] > best["test"]:
                best["test"] = ev["test_acc"]
                torch.save(clone.state_dict(), P("best_epoch.pth"))
            if ev["val_divgan"] != "" and ev["val_divgan"] > best["val"]:
                best["val"] = ev["val_divgan"]
                torch.save(clone.state_dict(), P("best_val_epoch.pth"))
            if (epoch + 1) % 10 == 0:
                torch.save(G.state_dict(), P("netG_last.pth"))
                torch.save(clone.state_dict(), P("clone_last.pth"))

        alternate_train(G, D, clone, oracle, bank, space, epochs["alternate"], gan_bs, nz,
                        gan_lr, gan_beta1, lambda_div, lambda_cls, temp, optC, schedC, autoaug,
                        num_classes, device, on_epoch, grad_clip=grad_clip, max_iters=max_iters)
        torch.save(clone.state_dict(), P("final_round.pth"))
        torch.save(G.state_dict(), P("netG_final.pth"))
        stage_done("S7", "final_round.pth")

    # ===== Final: reload best checkpoint so downstream metrics refer to the same weights =====
    clone = new_clone()
    clone.load_state_dict(torch.load(P("best_epoch.pth"), map_location=device, weights_only=False))
    clone.eval()
    best_test = evaluate1(clone, test_loader, eval_criterion, device)["test_acc"]
    fidelity = evaluate_fidelity(victim_net, clone, test_loader, device)
    acc_recovery = 100.0 * best_test / victim_test["test_acc"]
    summary = {
        "scenario": scenario_name, "victim": str(victim_dir), "proxy": proxy_desc,
        "victim_test_acc": victim_test["test_acc"], "substitute_best_test_acc": best_test,
        "accuracy_recovery": acc_recovery, "fidelity_test": fidelity,
        "total_queries": oracle.queries, "epochs": epochs, "lambda_div": lambda_div,
        "iterations_per_epoch": iters_per_epoch, "gan_batch_size": gan_bs,
        "wall_time_s": time.time() - t_start,
    }
    if os.path.exists(P("best_val_epoch.pth")):
        clone.load_state_dict(torch.load(P("best_val_epoch.pth"), map_location=device, weights_only=False))
        summary["best_val_ckpt_test_acc"] = evaluate1(clone, test_loader, eval_criterion, device)["test_acc"]
        summary["best_val_ckpt_fidelity"] = evaluate_fidelity(victim_net, clone, test_loader, device)
    json.dump(summary, open(P("summary.json"), "w"), indent=2)

    print(f'\n==> DFMS-HL extraction complete.')
    print(f"Victim Test Acc:      {victim_test['test_acc']:.2f}%")
    print(f"Substitute Best Acc:  {best_test:.2f}%   (best_epoch.pth, by test acc)")
    if "best_val_ckpt_test_acc" in summary:
        print(f"Attacker-legal pick:  {summary['best_val_ckpt_test_acc']:.2f}%   "
              f"(best_val_epoch.pth, by DivGAN-val agreement)")
    print(f"Accuracy Recovery:    {acc_recovery:.1f}%")
    print(f"Fidelity:             {fidelity:.1f}%")
    print(f"Total queries:        {oracle.queries:,}")


# =====================================================
# 3. Entry point (matches main_knockoff_extraction.py structure)
# =====================================================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    # DFMS_PLAN_DIR only exists so a smoke test can point at a scratch folder;
    # production runs use the default and the matrix/ subfolder is never globbed.
    exp_folder = os.environ.get("DFMS_PLAN_DIR", "./saved_exp_plan/dfms_plan")
    yaml_files = sorted(glob.glob(os.path.join(exp_folder, "*.yaml")))
    if not yaml_files:
        print(f"No YAML files found in {exp_folder}")
    else:
        # Spell out victim vs clone: the file name carries both and they are
        # easy to read the wrong way round (RES18_..._CrossDeiT = RN18 victim,
        # DeiT clone). Finished plans are parked in dfms_plan/done/.
        import yaml as _yaml          # plain load: process_yaml_file dumps the
                                      # whole blueprint and would bury this
        print(f"Found {len(yaml_files)} DFMS experiment plan(s):")
        for f in yaml_files:
            try:
                _y = _yaml.safe_load(open(f, encoding="utf-8"))
                _s = _y["Substitute"].get("Model", _y["Victim"].get("Model"))
                _s = _s.get("model_name", "DeiT") if isinstance(_s, dict) else _s
                _v = _y["Victim"].get("Model")
                _v = "DeiT" if isinstance(_v, dict) else _v
                print(f" - {os.path.basename(f)}")
                print(f"     victim (black box, frozen) : {_y['Victim']['Model_Name']}  [{_v}]")
                print(f"     clone  (the one trained)   : {_s}")
            except Exception as _e:
                print(f" - {f}   (could not preview: {_e})")
    n_runs = (SEED_END - SEED_START) * len(yaml_files)
    print(f"Victim seed: {MODEL_SEED}   |   attacker seeds: [{SEED_START}, {SEED_END}) "
          f"-> {SEED_END - SEED_START} run(s) per plan" + ("   [SMOKE]" if SMOKE else ""))
    if SEED_OVERRIDES:
        print("  !! seed range comes from the ENVIRONMENT, not the defaults in this file: "
              + ", ".join(f"{k}={v}" for k, v in SEED_OVERRIDES.items()))
        print("     clear it with  Remove-Item Env:SEED_END  (PowerShell)  if that is not intended")
    print(f"TOTAL: {n_runs} run(s) = {len(yaml_files)} plan(s) x {SEED_END - SEED_START} seed(s)")

    for yaml_path in yaml_files:
        print(f"\n========== {yaml_path} ==========")
        for extraction_seed in range(SEED_START, SEED_END):
            print(f"\n>>> model_seed={MODEL_SEED} extract_seed={extraction_seed}")
            set_seed(extraction_seed)
            main_dfms(MODEL_SEED, extraction_seed, yaml_path)
