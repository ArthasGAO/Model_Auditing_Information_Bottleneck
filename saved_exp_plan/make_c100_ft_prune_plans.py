"""Generate the CIFAR-100 fine-tuning and pruning plans (3 archs, 3 seeds).

    python saved_exp_plan/make_c100_ft_prune_plans.py

Writes four folders, split CNN / DeiT because main_ft.py and main_prune.py
drive those through different functions (main_ft vs main_ft_deit, main_prune vs
main_prune_deit):

    saved_exp_plan/ft_plan_c100/         ResNet-18, VGG16   -> main_ft
    saved_exp_plan/ft_plan_c100_deit/    DeiT-Ti distilled  -> main_ft_deit
    saved_exp_plan/prune_plan_c100/      ResNet-18, VGG16   -> main_prune
    saved_exp_plan/prune_plan_c100_deit/ DeiT-Ti distilled  -> main_prune_deit

They are NEW folders rather than additions to ft_plan/ and prune_plan/, which
are globbed wholesale by four different entry points and still hold CIFAR-10
work that must not be re-run.

------------------------------------------------------------------------------
Design
------------------------------------------------------------------------------
Source model = the CIFAR-100 victim of each architecture (seed 42, rate 1.0,
trained on group_A):
    saved_models/vanilla/CNN_Models/CIFAR-100_ResNet-18_25000_42_1.0
    saved_models/vanilla/CNN_Models/CIFAR-100_VGG16_25000_42_1.0
    saved_models/vanilla/Transformer_Models/CIFAR-100_DeiT_Distill_25000_42_1.0

Fine-tuning data = the OTHER 25000 CIFAR-100 images. Setting FT_Dataset.name to
the dataset's own name (not PseudoLabelCIFAR-10 / CIFARNet) sends
util.determine_ft_dataset down its `else` branch, which is
create_or_load_group_B(overlap_rate=0.0) - group_A's disjoint complement. The
`note: same` key is documentation only; the branch keys off the NAME.

DeiT is the DISTILLED variant, because that is the only CIFAR-100 transformer
negative pool and therefore the only null a DeiT suspect can be tested against.
Its Dataset/FT_Dataset transforms copy the DeiT_Distill pool recipe (RandAugment
magnitude 9, RandomErasing p=0.20, drop_path 0.15), not the Plain one.
util.setup_finetune already treats `head` and `head_dist` together as "the last
layer", so FT-LL / RT-AL are well defined on it.

Epoch budget, uniform across architectures so the three are comparable:
    FT-LL  30   only the classifier head trains, so it converges early
    FT-AL  50   all layers, small lr
    RT-AL  80   head re-initialised, so it needs the longest recovery
    prune  50 (CNN) / 30 (DeiT) per sparsity, FT-AL only
Learning rates are taken per architecture from the already-validated plans in
ft_plan/old_plan/ and prune_plan/; only the epoch counts are harmonised.
"""
import hashlib
import os
from pathlib import Path

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

GROUP_SIZE = 25000
FT_GROUP_SIZE = 25000          # the remaining half of CIFAR-100
SPARSITIES = [0.2, 0.8]

TEST_TF = [{"name": "ToTensor"}, {"name": "Normalize"}]
CNN_TF = [
    {"name": "RandomCrop", "params": {"size": 32, "padding": 4}},
    {"name": "RandomHorizontalFlip", "params": {"p": 0.5}},
    {"name": "ToTensor"},
    {"name": "Normalize"},
]
# CIFAR-100 DeiT_Distill pool recipe - NOT the Plain one (m7 / p0.15).
DEIT_TF = [
    {"name": "RandomResizedCrop", "params": {"size": 32, "scale": [0.85, 1.0], "ratio": [0.9, 1.1]}},
    {"name": "RandomHorizontalFlip", "params": {"p": 0.5}},
    {"name": "RandAugment", "params": {"num_ops": 2, "magnitude": 9}},
    {"name": "ToTensor"},
    {"name": "Normalize"},
    {"name": "RandomErasing", "params": {"p": 0.20, "scale": [0.02, 0.20], "ratio": [0.3, 3.3]}},
]
DEIT_MODEL = {"model_name": "deit_tiny_distilled_patch16_224", "img_size": 32,
              "patch_size": 4, "pretrained": False, "drop_path_rate": 0.15}

SCHED = {"name": "CosineAnnealingLR", "params": {"T_max": "auto", "eta_min": 0.000001}}


# ---------------------------------------------------------------------------
# Plans whose bytes are already recorded in an MI master table's plan_sha256
# column. calculate_MI_ft.py / calculate_MI_prune.py compare that hash before
# reusing rows, so rewriting one of these files -- even if only a comment
# changes -- makes every row computed from it fail with "identity mismatch".
# That is exactly what happened on 2026-09-21: this generator was re-run, the
# CIFAR-100 RES18 prune plan picked up a new header, and the 1400 rows in
# saved_logs/pruning_final/MI_master_table_prune.csv became unusable. The
# original bytes were recovered from saved_exp_plan/prune_plan/ and restored
# here, which is why that one file does not look like the generator's output.
#
# dump() therefore SKIPS these files. To change a frozen plan for real: write
# it to a NEW folder, point PRUNE_PLAN_DIR / FT_PLAN_DIR at it, and give the
# MI run a new master CSV.
FROZEN = {
    ("ft_plan_c100", "CIFAR100_RES18_FT_Same_25000.yaml"):
        "d9153ca529f7c2cb25c0141db0e1e724489bff714f76e5f4afa85905ce973631",
    ("ft_plan_c100", "CIFAR100_VGG16_FT_Same_25000.yaml"):
        "bee5880712ac1aa38ba818333f57b1b7454e39275d831e8569469dd4ddb95121",
    ("ft_plan_c100_deit", "CIFAR100_DEIT_DISTILL_FT_Same_25000.yaml"):
        "96a0ff190c7f57ddc574f4726e45c0fc06ce6b67d3a1962469c589ec6f0c275a",
    ("prune_plan_c100", "CIFAR100_RES18_PRUNE_Same_25000.yaml"):
        "7aac4fae507fcd8b090583dc086752051fe04bb0b189331b364d9881a5f6eaca",
    ("prune_plan_c100", "CIFAR100_VGG16_PRUNE_Same_25000.yaml"):
        "9de9593f6871c1384f63706e69120aa0c294aa85ca967ec1d15463a5285803a3",
    ("prune_plan_c100_deit", "CIFAR100_DEIT_DISTILL_PRUNE_Same_25000.yaml"):
        "87aa3d1cd165b048b2123a1c10fc7b5dfc3c83b720b644d20ea1e7ea80538af9",
}


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


# arch key -> everything that differs between the three rows.
#   victim   : Model_Name, i.e. the checkpoint folder stem under saved_models/vanilla
#   model    : the plan's `Model` value (str for CNN, dict for DeiT)
#   ft       : {strategy: (optimizer name, params, epochs)}
#   prune    : (optimizer name, params, epochs) used at every sparsity
ARCHS = {
    "RES18": {
        "victim": "CIFAR-100_ResNet-18_25000", "model": "ResNet-18",
        "tf": CNN_TF, "deit": False, "file": "CIFAR100_RES18",
        "ft": {"FT-LL": ("SGD", sgd(0.01, 0.0005), 30),
               "FT-AL": ("SGD", sgd(0.0001, 0.0002), 50),
               "RT-AL": ("SGD", sgd(0.0005, 0.0002), 80)},
        "prune": ("SGD", sgd(0.0001, 0.0002), 50),
    },
    "VGG16": {
        "victim": "CIFAR-100_VGG16_25000", "model": "VGG16",
        "tf": CNN_TF, "deit": False, "file": "CIFAR100_VGG16",
        "ft": {"FT-LL": ("SGD", sgd(0.01, 0.0005), 30),
               "FT-AL": ("SGD", sgd(0.0003, 0.0005), 50),
               "RT-AL": ("SGD", sgd(0.0005, 0.0005), 80)},
        "prune": ("SGD", sgd(0.0001, 0.0005), 50),
    },
    "DEIT": {
        "victim": "CIFAR-100_DeiT_Distill_25000", "model": DEIT_MODEL,
        "tf": DEIT_TF, "deit": True, "file": "CIFAR100_DEIT_DISTILL",
        "ft": {"FT-LL": ("AdamW", adamw(0.0005, 0.05), 30),
               "FT-AL": ("AdamW", adamw(0.00015, 0.05), 50),
               "RT-AL": ("AdamW", adamw(0.0002, 0.05), 80)},
        "prune": ("AdamW", adamw(0.00015, 0.05), 30),
    },
}
FT_ORDER = ["FT-LL", "FT-AL", "RT-AL"]


def _blocks(a):
    ds = {"name": "CIFAR-100", "normalization": "cifar100", "img_size": 32,
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
        what = (f"sparsity {SPARSITIES} x FT-AL {a['prune'][2]}ep "
                f"lr={a['prune'][1]['lr']}")
        driver = "main_prune_deit" if a["deit"] else "main_prune"
    lines = [
        "# " + "=" * 70,
        f"# CIFAR-100 {'fine-tuning' if kind == 'ft' else 'pruning'} "
        f"- source model {a['victim']}",
        f"#   checkpoint : {src}/{a['victim']}_42_1.0/best_epoch.pth",
        f"#   driver     : {driver}",
        f"#   {what}",
        "#",
        "# FT data = the OTHER 25000 CIFAR-100 images. FT_Dataset.name is the",
        "# dataset's own name, which sends util.determine_ft_dataset down its",
        "# create_or_load_group_B(overlap_rate=0.0) branch - group_A's disjoint",
        "# complement. `note: same` is documentation; the branch keys off the name.",
    ]
    if a["deit"]:
        lines += [
            "#",
            "# DISTILLED DeiT: the only CIFAR-100 transformer negative pool is",
            "# DeiT_Distill, so the suspect must be that variant to have a null.",
            "# Transforms copy that pool's recipe (RandAugment m9, erase p0.20,",
            "# drop_path 0.15). util.setup_finetune treats head + head_dist together",
            "# as the last layer, so FT-LL / RT-AL are well defined.",
        ]
    lines += [
        "# Generated by saved_exp_plan/make_c100_ft_prune_plans.py - edit there.",
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
            base = f"{kind}_plan_c100" + ("_deit" if a["deit"] else "")
            tag = "FT" if kind == "ft" else "PRUNE"
            fname = f"{a['file']}_{tag}_Same_{FT_GROUP_SIZE}.yaml"
            path = os.path.join(HERE, base, fname)
            wrote = dump(path, header(a, kind), build(a))
            made.append((base, fname, key, kind, wrote))

    width = max(len(b) for b, _, _, _, _ in made)
    print(f"{'folder':<{width}}  file")
    print("-" * (width + 48))
    for base, fname, key, kind, wrote in made:
        print(f"{base:<{width}}  {fname}   {'written' if wrote else 'frozen, skipped'}")
    n = sum(1 for m in made if m[-1])
    print(f"\n{n} plan(s) written, {len(made) - n} frozen (unchanged)")
    print(f"  FT   : 3 arch x {len(FT_ORDER)} strategies {FT_ORDER}")
    print(f"  Prune: 3 arch x {len(SPARSITIES)} sparsities {SPARSITIES} (FT-AL only)")


if __name__ == "__main__":
    main()
