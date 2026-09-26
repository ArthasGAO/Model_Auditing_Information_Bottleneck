"""Build the 3x3 victim-x-surrogate metric matrices for the CIFAR-10 knockoff grid.

Rows = victim architecture, columns = surrogate architecture, and every cell
carries two numbers: the Same10 (CIFAR-10 queries) and Cross100 (CIFAR-100
queries) auxiliary-domain results.

Three metrics, each reported on TWO evaluation sets:
    Accuracy  - substitute top-1 (%)
    Recovery  - 100 * substitute_acc / victim_acc
    Fidelity  - % of samples where substitute and victim predict the SAME label
                (ground truth is ignored - this measures functional agreement)

    eval set "test"  - the victim's test set (generalisation view)
    eval set "train" - the victim's OWN training data: group_A, 25000 images,
                       clean transforms. This is byte-identical to the probe
                       that calculate_MI*.py uses for the In-sample MI, so these
                       rows line up one-to-one with the MI table.

Why this recomputes instead of reading logs: main_knockoff_extraction*.py PRINT
fidelity and recovery but never persist them - the per-epoch CSV only stores
accuracy. So the only faithful source is the saved best_epoch.pth checkpoints,
which is what this script evaluates. Definitions match the training scripts
exactly (best checkpoint, victim's own test set, evaluate_fidelity).

Results are cached to CACHE_CSV; delete that file (or set FORCE_RECOMPUTE) to
re-evaluate. Rendering from cache is instant.
"""
import os
import glob
import csv

import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader

from util import (build_dataset_from_yaml, load_best_checkpoint, evaluate1,
                  evaluate_fidelity, build_deit_student, create_or_load_group_A)
from Model.ResNet_18 import ResNet18
from Model.VGG16 import ModifiedVGG16
import torch.nn as nn

# ---------------------------------------------------------------- settings
PLAN_GLOB = "saved_exp_plan/**/*.yaml"
CACHE_CSV = "./saved_logs/extraction_vanilla/MEA_metrics.csv"
FORCE_RECOMPUTE = False

SEED = 0                     # attacker seed shown in the matrices
VICTIM_SEED, RATE = 42, 1.0  # victim checkpoint selector

ARCH_ORDER = ["ResNet-18", "VGG16", "DeiT"]
EVAL_SETS = ["test", "train"]   # "train" = the group_A MI probe; drop one to skip it
MI_GROUP_SEED = 42              # must match calculate_MI*.py so the probe matches
DOMAINS = [("Same10", "CIFAR-10"), ("Cross100", "CIFAR-100")]

VICTIM_ROOTS = ["./saved_models/vanilla/CNN_Models",
                "./saved_models/vanilla/Transformer_Models"]
SUB_ROOTS = ["./saved_models/extraction_vanilla",
             "./saved_models/extraction_vanilla/Transformer_Models"]

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- helpers
def arch_of(model_cfg):
    """A dict-valued Model means a timm DeiT; a string means a CNN."""
    return "DeiT" if isinstance(model_cfg, dict) else model_cfg


def build_model_any(model_cfg, num_classes):
    if isinstance(model_cfg, dict):
        return build_deit_student({"Model": model_cfg}, num_classes)
    if model_cfg == "ResNet-18":
        return ResNet18(num_classes=num_classes)
    if model_cfg == "VGG16":
        return ModifiedVGG16(num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_cfg}")


def find_ckpt(roots, folder_name):
    """Locate a checkpoint folder across the CNN / Transformer output roots."""
    for r in roots:
        p = Path(r) / folder_name
        if p.is_dir():
            ck, _ = load_best_checkpoint(p)
            if ck is not None:
                return ck
    return None


def discover_cells():
    """Enumerate the CIFAR-10 knockoff cells straight from the plan YAMLs."""
    cells = []
    for f in glob.glob(PLAN_GLOB, recursive=True):
        txt = open(f, encoding="utf-8").read()
        if "Knockoff" not in txt or "Substitute" not in txt or "JBA" in txt:
            continue
        d = yaml.safe_load(txt)
        if "Victim" not in d or "Auxiliary_Dataset" not in d or d.get("Epochs") is None:
            continue
        vds = d["Victim"]["Dataset"]
        if (vds["name"] if isinstance(vds, dict) else vds) != "CIFAR-10":
            continue
        cells.append(d)
    # de-duplicate on scenario name (a plan may exist in several folders)
    return list({d["Scenario_Name"]: d for d in cells}.values())


def build_eval_loaders(ds_cfg):
    """Return ({eval_set: DataLoader}, num_classes).

    "test"  -> the victim's test set.
    "train" -> the victim's own training data, reproduced EXACTLY as
               calculate_MI*.py builds its In-sample probe: in_sample_set
               (clean/test transforms, no augmentation) restricted to group_A
               at seed 42. Matching this matters - it is what makes these
               numbers comparable with the MI table row for row.
    """
    ds_obj, ncls, gsize = build_dataset_from_yaml(ds_cfg)
    out = {}
    if "test" in EVAL_SETS:
        out["test"] = DataLoader(ds_obj.test_set, batch_size=256, shuffle=False,
                                 num_workers=0, pin_memory=True)
    if "train" in EVAL_SETS:
        group_A = create_or_load_group_A(
            dataset=ds_obj.in_sample_set,
            save_dir=f'./Indices/{ds_cfg["name"]}/',
            group_size=gsize, num_classes=ncls,
            seed=MI_GROUP_SEED, force_rebuild=False,
        )
        probe = ds_obj.subset("train", group_A, clean=True)
        out["train"] = DataLoader(probe, batch_size=256, shuffle=False,
                                  num_workers=0, pin_memory=True)
    return out, ncls


# ---------------------------------------------------------------- compute
def compute_rows(cells):
    victim_cache, loader_cache, victim_acc_cache = {}, {}, {}
    rows = []

    for d in sorted(cells, key=lambda x: x["Scenario_Name"]):
        scen = d["Scenario_Name"]
        vcfg = d["Victim"]
        v_arch = arch_of(vcfg["Model"])
        s_arch = arch_of(d["Substitute"]["Model"])
        aux = d["Auxiliary_Dataset"]["name"]

        ds_key = (vcfg["Dataset"]["name"], vcfg["Dataset"].get("normalization"))
        if ds_key not in loader_cache:
            loader_cache[ds_key] = build_eval_loaders(vcfg["Dataset"])
        loaders, num_classes = loader_cache[ds_key]

        # ---- victim (3 distinct ones, so cache them) ----
        vkey = (v_arch, vcfg["Model_Name"])
        if vkey not in victim_cache:
            folder = f'{vcfg["Model_Name"]}_{VICTIM_SEED}_{RATE}'
            ck = find_ckpt(VICTIM_ROOTS, folder)
            if ck is None:
                print(f"  [skip] victim checkpoint missing: {folder}")
                continue
            net = build_model_any(vcfg["Model"], num_classes).to(device)
            net.load_state_dict(torch.load(ck, map_location=device, weights_only=False))
            net.eval()
            victim_cache[vkey] = net
            victim_acc_cache[vkey] = {
                es: evaluate1(net, loaders[es], nn.CrossEntropyLoss(),
                              device)["test_acc"]
                for es in EVAL_SETS
            }
        victim_net = victim_cache[vkey]

        # ---- every attacker seed we have a checkpoint for ----
        for seed in range(0, 10):
            folder = f"{scen}_{seed}_{RATE}"
            ck = find_ckpt(SUB_ROOTS, folder)
            if ck is None:
                continue
            sub = build_model_any(d["Substitute"]["Model"], num_classes).to(device)
            sub.load_state_dict(torch.load(ck, map_location=device, weights_only=False))
            sub.eval()

            msg = [f"  {scen:<52} seed{seed}"]
            for es in EVAL_SETS:
                v_acc = victim_acc_cache[vkey][es]
                s_acc = evaluate1(sub, loaders[es], nn.CrossEntropyLoss(),
                                  device)["test_acc"]
                fid = evaluate_fidelity(victim_net, sub, loaders[es], device)
                rows.append(dict(scenario=scen, victim=v_arch, surrogate=s_arch,
                                 aux=aux, seed=seed, eval_set=es,
                                 victim_acc=round(v_acc, 4), acc=round(s_acc, 4),
                                 recovery=round(100.0 * s_acc / v_acc, 4),
                                 fidelity=round(fid, 4)))
                msg.append(f"[{es}] acc {s_acc:6.2f} rec {100.0*s_acc/v_acc:6.2f} "
                           f"fid {fid:6.2f}")
            print("  ".join(msg))
            del sub
            torch.cuda.empty_cache()
    return rows


def load_or_compute():
    if os.path.exists(CACHE_CSV) and not FORCE_RECOMPUTE:
        with open(CACHE_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # A cache written before EVAL_SETS existed (or missing one of the
        # requested sets) is unusable - fall through to recompute.
        have_sets = {r.get("eval_set") for r in rows} if rows else set()
        if not rows or not have_sets >= set(EVAL_SETS):
            print(f"cache does not cover eval sets {EVAL_SETS} - recomputing")
            rows = []
        for r in rows:
            for k in ("victim_acc", "acc", "recovery", "fidelity"):
                r[k] = float(r[k])
            r["seed"] = int(r["seed"])
        if rows:
            print(f"loaded {len(rows)} cached rows from {CACHE_CSV}")
            return rows

    print("evaluating checkpoints (this is the slow path)...")
    rows = compute_rows(discover_cells())
    os.makedirs(os.path.dirname(CACHE_CSV), exist_ok=True)
    with open(CACHE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {CACHE_CSV}")
    return rows


# ---------------------------------------------------------------- render
EVAL_LABEL = {"test": "victim's TEST set",
              "train": "victim's TRAINING data (group_A, = the MI In-probe)"}


def render(rows, metric, eval_set="test", seed=SEED):
    """One 3x3 matrix; each cell shows Same10 / Cross100."""
    idx = {(r["victim"], r["surrogate"], r["aux"]): r
           for r in rows if r["seed"] == seed and r["eval_set"] == eval_set}

    title = {"acc": "ACCURACY  (substitute top-1, %)",
             "recovery": "RECOVERY  (100 x substitute_acc / victim_acc, %)",
             "fidelity": "FIDELITY  (agreement with victim's predictions, %)"}[metric]

    print("\n" + "=" * 78)
    print(f"{title}   on {EVAL_LABEL[eval_set]}   [seed {seed}]")
    print("rows = VICTIM   columns = SURROGATE   cell = Same10 / Cross100")
    print("=" * 78)
    print(f"{'victim \\ surrogate':>20} | " + " | ".join(f"{a:^16}" for a in ARCH_ORDER))
    print("-" * 78)
    for v in ARCH_ORDER:
        cells = []
        for s in ARCH_ORDER:
            vals = []
            for _, auxname in DOMAINS:
                r = idx.get((v, s, auxname))
                vals.append(f"{r[metric]:.2f}" if r else "  --  ")
            cells.append(f"{vals[0]:>7} /{vals[1]:>7}")
        print(f"{v:>20} | " + " | ".join(cells))
    print("-" * 78)


def render_seed_spread(rows, metric, eval_set="test"):
    """Supplementary: mean +/- std for cells that have more than one seed."""
    from statistics import mean, stdev
    groups = {}
    for r in rows:
        if r["eval_set"] != eval_set:
            continue
        groups.setdefault((r["victim"], r["surrogate"], r["aux"]), []).append(r[metric])
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if not multi:
        return
    print(f"\ncells with >1 seed ({metric}):")
    for (v, s, a), vals in sorted(multi.items()):
        print(f"    {v:>10} -> {s:<10} {a:<10} n={len(vals)}  "
              f"{mean(vals):6.2f} +/- {stdev(vals):.2f}")


if __name__ == "__main__":
    rows = load_or_compute()
    total = 9 * len(DOMAINS)
    for es in EVAL_SETS:
        print("\n\n" + "#" * 78)
        print(f"#  EVALUATED ON: {EVAL_LABEL[es].upper()}")
        print("#" * 78)
        for m in ("acc", "recovery", "fidelity"):
            render(rows, m, eval_set=es)
            render_seed_spread(rows, m, eval_set=es)
        n = len({(r["victim"], r["surrogate"], r["aux"]) for r in rows
                 if r["seed"] == SEED and r["eval_set"] == es})
        print(f"\ncells populated at seed {SEED} on '{es}': {n}/{total}")
