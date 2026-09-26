"""Generate the CIFAR-10 fine-tuning and pruning plans (3 archs, 3 seeds).

    python saved_exp_plan/make_c10_ft_prune_plans.py

Sibling of make_c100_ft_prune_plans.py, same structure but a shorter FT epoch
budget (see below). Deliberately a separate
file rather than a `dataset` switch inside that one: regenerating the CIFAR-100
plans would change their bytes, and their plan_sha256 is recorded in every row
of saved_logs/ft_final/MI_master_table_ft.csv.

Writes four folders, split CNN / DeiT because main_ft.py and main_prune.py
drive those through different functions:

    saved_exp_plan/ft_plan_c10/         ResNet-18, VGG16  -> main_ft
    saved_exp_plan/ft_plan_c10_deit/    DeiT-Ti plain     -> main_ft_deit
    saved_exp_plan/prune_plan_c10/      ResNet-18, VGG16  -> main_prune
    saved_exp_plan/prune_plan_c10_deit/ DeiT-Ti plain     -> main_prune_deit

------------------------------------------------------------------------------
Design
------------------------------------------------------------------------------
Source model = the CIFAR-10 victim of each architecture (seed 42, rate 1.0,
trained on group_A):
    saved_models/vanilla/CNN_Models/CIFAR-10_ResNet-18_25000_42_1.0
    saved_models/vanilla/CNN_Models/CIFAR-10_VGG16_25000_42_1.0
    saved_models/vanilla/Transformer_Models/CIFAR-10_DeiT_Plain_25000_42_1.0

Fine-tuning data = the OTHER 25000 CIFAR-10 images, matching the CIFAR-100
setup. FT_Dataset.name is the dataset's own name, which sends
util.determine_ft_dataset down its create_or_load_group_B(overlap_rate=0.0)
branch - group_A's disjoint complement. `note: same` is documentation; the
branch keys off the NAME. This is NOT the pseudo-labelled ImageNet subset the
older CIFAR-10 runs used, so the scenario names (_Same_25000) do not collide
with the _PseudoLabel_25000 models already in ft_final / pruning_final.

DeiT here is the PLAIN variant (CIFAR-10's negative pool is DeiT_Plain), unlike
CIFAR-100 where the only transformer pool is DeiT_Distill. Transforms copy the
CIFAR-10 DeiT_Plain pool recipe: RandAugment magnitude 7, RandomErasing p=0.15,
drop_path 0.1.

Epoch budget (shortened 2026-09-21 from the CIFAR-100 row's 30/50/80):
    FT-LL  10   only the classifier head trains - a linear probe, converges fast
    FT-AL  25   all layers, small lr
    RT-AL  50   head re-initialised, so it still needs the longest recovery
    prune  50 (CNN) / 30 (DeiT) per sparsity, FT-AL only - UNCHANGED
NOTE this no longer matches the CIFAR-100 FT budget, which is already trained
and MI-computed at 30/50/80. Each cell is judged against its own architecture's
null, so that is fine within a dataset, but the two datasets' FT rows are no
longer epoch-matched - say so if you compare them directly.
Learning rates are per architecture, taken from the validated CIFAR-10 plans in
ft_plan/old_plan/ and prune_plan/; only the epoch counts are harmonised.
"""
import hashlib
import os
from pathlib import Path

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

GROUP_SIZE = 25000
FT_GROUP_SIZE = 25000          # the remaining half of CIFAR-10
SPARSITIES = [0.2, 0.8]

TEST_TF = [{"name": "ToTensor"}, {"name": "Normalize"}]
CNN_TF = [
    {"name": "RandomCrop", "params": {"size": 32, "padding": 4}},
    {"name": "RandomHorizontalFlip", "params": {"p": 0.5}},
    {"name": "ToTensor"},
    {"name": "Normalize"},
]
# CIFAR-10 DeiT_Plain pool recipe - NOT CIFAR-100's Distill one (m9 / p0.20).
DEIT_TF = [
    {"name": "RandomResizedCrop", "params": {"size": 32, "scale": [0.85, 1.0], "ratio": [0.9, 1.1]}},
    {"name": "RandomHorizontalFlip", "params": {"p": 0.5}},
    {"name": "RandAugment", "params": {"num_ops": 2, "magnitude": 7}},
    {"name": "ToTensor"},
    {"name": "Normalize"},
    {"name": "RandomErasing", "params": {"p": 0.15, "scale": [0.02, 0.25], "ratio": [0.3, 3.3]}},
]
DEIT_MODEL = {"model_name": "deit_tiny_patch16_224", "img_size": 32,
              "patch_size": 4, "pretrained": False, "drop_path_rate": 0.1}

SCHED = {"name": "CosineAnnealingLR", "params": {"T_max": "auto", "eta_min": 0.000001}}


# No CIFAR-10 FT/prune plan has been through an MI run yet, so nothing is
# frozen. AFTER calculate_MI_ft.py / calculate_MI_prune.py have written rows
# for these plans, add each file's sha256 here: from then on its bytes are
# part of those rows' identity and rewriting it invalidates them.
# See make_c100_ft_prune_plans.py for the incident this prevents.
FROZEN = {}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_frozen(path):
    """(skip?, message) for a target plan path -- see FROZEN above."""
    path = Path(path)
    key = (path.parent.name, path.name)
    expected = FROZEN.get(key)
    if expected is None or not path.exists():
        return False, ""
    actual = _sha256(path)
    if actual == expected:
        return True, f"[FROZEN] {key[0]}/{key[1]}: bytes are in an MI table, left untouched"
    return True, (f"[FROZEN-DRIFT] {key[0]}/{key[1]}: on disk {actual[:16]}, "
                  f"MI tables expect {expected[:16]}. NOT overwriting; restore the "
                  f"recorded bytes or move the MI rows to a new CSV.")


def sgd(lr, wd):
    return {"lr": lr, "momentum": 0.9, "weight_decay": wd, "nesterov": True}


def adamw(lr, wd):
    return {"lr": lr, "weight_decay": wd}


ARCHS = {
    "RES18": {
        "victim": "CIFAR-10_ResNet-18_25000", "model": "ResNet-18",
        "tf": CNN_TF, "deit": False, "file": "CIFAR10_RES18",
        "ft": {"FT-LL": ("SGD", sgd(0.01, 0.0005), 10),
               "FT-AL": ("SGD", sgd(0.001, 0.0005), 25),
               "RT-AL": ("SGD", sgd(0.005, 0.0005), 50)},
        "prune": ("SGD", sgd(0.001, 0.0005), 50),
    },
    "VGG16": {
        "victim": "CIFAR-10_VGG16_25000", "model": "VGG16",
        "tf": CNN_TF, "deit": False, "file": "CIFAR10_VGG16",
        "ft": {"FT-LL": ("SGD", sgd(0.01, 0.0005), 10),
               "FT-AL": ("SGD", sgd(0.0005, 0.001), 25),
               "RT-AL": ("SGD", sgd(0.005, 0.0005), 50)},
        "prune": ("SGD", sgd(0.001, 0.0005), 50),
    },
    "DEIT": {
        "victim": "CIFAR-10_DeiT_Plain_25000", "model": DEIT_MODEL,
        "tf": DEIT_TF, "deit": True, "file": "CIFAR10_DEIT_PLAIN",
        "ft": {"FT-LL": ("AdamW", adamw(0.0005, 0.03), 10),
               "FT-AL": ("AdamW", adamw(0.0001, 0.03), 25),
               "RT-AL": ("AdamW", adamw(0.0003, 0.03), 50)},
        "prune": ("AdamW", adamw(0.0001, 0.03), 30),
    },
}
FT_ORDER = ["FT-LL", "FT-AL", "RT-AL"]


def _blocks(a):
    ds = {"name": "CIFAR-10", "normalization": "cifar10", "img_size": 32,
          "train_transforms": a["tf"], "test_transforms": TEST_TF,
          "group_size": GROUP_SIZE}
    ft = dict(ds, group_size=FT_GROUP_SIZE)
    ft["note"] = "same"
    return ds, ft


def build_ft(a):
    ds, ft = _blocks(a)
    return {
        "Model_Name": a["victim"],
        "Scenario_Name": f"{a['victim']}_Same_{FT_GROUP_SIZE}",
        "Dataset": ds,
        "FT_Dataset": ft,
        "Model": a["model"],
        "Optimizers": [
            {"name": a["ft"][s][0], "strategy": s, "params": a["ft"][s][1],
             "Epochs": a["ft"][s][2]}
            for s in FT_ORDER
        ],
        "Scheduler": SCHED,
        "BatchSize": 128,
    }


def build_prune(a):
    ds, ft = _blocks(a)
    name, params, epochs = a["prune"]
    return {
        "Model_Name": a["victim"],
        "Scenario_Name": f"{a['victim']}_Same_{FT_GROUP_SIZE}",
        "Dataset": ds,
        "FT_Dataset": ft,
        "Model": a["model"],
        "Optimizers": [
            {"name": name, "sparsity": sp, "params": params, "Epochs": epochs}
            for sp in SPARSITIES
        ],
        "Scheduler": SCHED,
        "BatchSize": 128,
    }


def header(a, kind):
    src = ("saved_models/vanilla/Transformer_Models" if a["deit"]
           else "saved_models/vanilla/CNN_Models")
    if kind == "ft":
        what = "  ".join(f"{s} {a['ft'][s][2]}ep lr={a['ft'][s][1]['lr']}" for s in FT_ORDER)
        driver = "main_ft_deit" if a["deit"] else "main_ft"
    else:
        what = f"sparsity {SPARSITIES} x FT-AL {a['prune'][2]}ep lr={a['prune'][1]['lr']}"
        driver = "main_prune_deit" if a["deit"] else "main_prune"
    lines = [
        "# " + "=" * 70,
        f"# CIFAR-10 {'fine-tuning' if kind == 'ft' else 'pruning'} "
        f"- source model {a['victim']}",
        f"#   checkpoint : {src}/{a['victim']}_42_1.0/best_epoch.pth",
        f"#   driver     : {driver}",
        f"#   {what}",
        "#",
        "# FT data = the OTHER 25000 CIFAR-10 images, matching the CIFAR-100 row.",
        "# FT_Dataset.name is the dataset's own name, which sends",
        "# util.determine_ft_dataset down its create_or_load_group_B(0.0) branch -",
        "# group_A's disjoint complement. This is NOT the pseudo-labelled ImageNet",
        "# subset the older CIFAR-10 runs used, so _Same_25000 does not collide",
        "# with the _PseudoLabel_25000 models already on disk.",
    ]
    if a["deit"]:
        lines += [
            "#",
            "# PLAIN DeiT: CIFAR-10's transformer negative pool is DeiT_Plain (CIFAR-100's",
            "# is DeiT_Distill), so the suspect must be that variant to have a null.",
            "# Transforms copy that pool's recipe (RandAugment m7, erase p0.15, dp 0.1).",
        ]
    lines += [
        "# Generated by saved_exp_plan/make_c10_ft_prune_plans.py - edit there.",
        "# " + "=" * 70,
        "",
    ]
    return "\n".join(lines)


def dump(path, head, plan):
    """Write the plan, unless FROZEN says its bytes belong to an MI table."""
    skip, why = is_frozen(path)
    if skip:
        print(why)
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(head)
        yaml.safe_dump(plan, fh, sort_keys=False, default_flow_style=False)
    return True


def main():
    made = []
    for key, a in ARCHS.items():
        for kind, build in (("ft", build_ft), ("prune", build_prune)):
            base = f"{kind}_plan_c10" + ("_deit" if a["deit"] else "")
            tag = "FT" if kind == "ft" else "PRUNE"
            fname = f"{a['file']}_{tag}_Same_{FT_GROUP_SIZE}.yaml"
            wrote = dump(os.path.join(HERE, base, fname), header(a, kind), build(a))
            made.append((base, fname, wrote))

    width = max(len(b) for b, _, _ in made)
    print(f"{'folder':<{width}}  file")
    print("-" * (width + 46))
    for base, fname, wrote in made:
        print(f"{base:<{width}}  {fname}   {'written' if wrote else 'frozen, skipped'}")
    n = sum(1 for m in made if m[-1])
    print(f"\n{n} plan(s) written, {len(made) - n} frozen (unchanged)")
    print(f"  FT   : 3 arch x {len(FT_ORDER)} strategies {FT_ORDER} x 3 seeds = 27 points")
    print(f"  Prune: 3 arch x {len(SPARSITIES)} sparsities {SPARSITIES} x 3 seeds = 18 points")


if __name__ == "__main__":
    main()
