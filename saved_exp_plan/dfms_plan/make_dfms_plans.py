"""Generate the DFMS-HL experiment matrix as one YAML per cell.

    python saved_exp_plan/dfms_plan/make_dfms_plans.py

Writes every cell into saved_exp_plan/dfms_plan/matrix/ and copies the tier-A1
plan (CIFAR-10, ResNet-18 -> ResNet-18, 40-class proxy) into
saved_exp_plan/dfms_plan/ where main_dfms_extraction.py picks it up. Move
other cells up one level to run them; the matrix folder is never globbed.

Matrix = 2 victim datasets x 2 proxies x 3 victim archs x 3 clone archs, with
the synthetic proxy limited to the same-arch diagonal (22 plans):

    tier  victim      proxy                 pairs
    A1    CIFAR-10    CIFAR-100 40 classes  RN18 -> RN18          (seeds 0..4)
    A2    CIFAR-10    CIFAR-100 40 classes  the other 8 pairs
    B1    CIFAR-10    synthetic shapes      RN18 -> RN18
    B2    CIFAR-10    synthetic shapes      VGG16 -> VGG16, DeiT -> DeiT
    C1    CIFAR-100   CIFAR-10 (all)        RN18 -> RN18
    C2    CIFAR-100   CIFAR-10 (all)        the other 8 pairs
    D1    CIFAR-100   synthetic shapes      RN18 -> RN18

Settings per (victim dataset, proxy) come from the reference run scripts for
CIFAR-10 and from the paper text for CIFAR-100 (no script shipped for it).
"""
import copy
import os
import shutil

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = os.path.join(HERE, "matrix")

CNN_TRAIN_TF = [
    {"name": "RandomCrop", "params": {"size": 32, "padding": 4}},
    {"name": "RandomHorizontalFlip", "params": {"p": 0.5}},
    {"name": "ToTensor"},
    {"name": "Normalize"},
]
DEIT_TRAIN_TF = [
    {"name": "RandomResizedCrop", "params": {"size": 32, "scale": [0.85, 1.0], "ratio": [0.9, 1.1]}},
    {"name": "RandomHorizontalFlip", "params": {"p": 0.5}},
    {"name": "RandAugment", "params": {"num_ops": 2, "magnitude": 7}},
    {"name": "ToTensor"},
    {"name": "Normalize"},
    {"name": "RandomErasing", "params": {"p": 0.15, "scale": [0.02, 0.25], "ratio": [0.3, 3.3]}},
]
TEST_TF = [{"name": "ToTensor"}, {"name": "Normalize"}]

DEIT_MODEL = {"model_name": "deit_tiny_patch16_224", "img_size": 32, "patch_size": 4,
              "pretrained": False, "drop_path_rate": 0.1}

DATASETS = {
    "CIFAR-10": {"file": "CIFAR10", "norm": "cifar10", "K": 10},
    "CIFAR-100": {"file": "CIFAR100", "norm": "cifar100", "K": 100},
}

ARCHS = {
    "RES18": {"tag": "18", "model": "ResNet-18", "folder": "ResNet-18"},
    "VGG16": {"tag": "16", "model": "VGG16", "folder": "VGG16"},
    "DEIT":  {"tag": "DeiT", "model": DEIT_MODEL, "folder": "DeiT_Plain"},
}

# Clone recipes. CNN = the reference code's SGD schedule. DeiT = this repo's
# DeiT knockoff recipe for the offline stages; the online (S7) learning rate
# has no precedent in the paper and follows the CNN's 10x reduction.
CNN_RECIPE = {
    "Optimizer": {"name": "SGD", "params": {"lr": 0.1, "momentum": 0.9, "weight_decay": 0.0005}},
    "Scheduler": {"name": "CosineAnnealingLR", "params": {"eta_min": 0.0}},
    "Alternate": {
        "Optimizer": {"name": "SGD", "params": {"lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005}},
        "Scheduler": {"name": "WarmupCosineAnnealingLR",
                      "params": {"warmup_epochs": 10, "warmup_start_factor": 0.1, "eta_min": 0.0}},
    },
    "grad_clip": None,
}
DEIT_RECIPE = {
    "Optimizer": {"name": "AdamW", "params": {"lr": 0.0007, "betas": [0.9, 0.999], "eps": 1.0e-8,
                                              "weight_decay": 0.03}},
    "Scheduler": {"name": "WarmupCosineAnnealingLR",
                  "params": {"warmup_epochs": 10, "warmup_start_factor": 0.1, "eta_min": 1.0e-6}},
    "Alternate": {
        "Optimizer": {"name": "AdamW", "params": {"lr": 0.00007, "betas": [0.9, 0.999], "eps": 1.0e-8,
                                                  "weight_decay": 0.03}},
        "Scheduler": {"name": "WarmupCosineAnnealingLR",
                      "params": {"warmup_epochs": 10, "warmup_start_factor": 0.1, "eta_min": 1.0e-6}},
    },
    "grad_clip": 1.0,
}

# (victim dataset, proxy key) -> attack settings.
#   C100-40C : reference run_cifar40_classes_*.sh, lambda 500, 400 ep on 20k proxy.
#              Our pool is group_B ∩ 40 classes = 10k, so 800 epochs keep the
#              same 125k iterations / 8.0M queries; mix 0.5/0.5 keeps the same
#              157 steps and ~10k generator samples per offline epoch.
#   Synth    : reference run_synthetic_*.sh (0.5 / 0.5 mix, 150 alternating epochs).
#   C10      : paper Sec. 4.2 / App. D.2, lambda 100, 10M queries; 25k CIFAR-10
#              group_B at bs 64 = 391 it/epoch -> 400 epochs. Mix by analogy
#              with the 40-class script.
SETTINGS = {
    ("CIFAR-10", "C100-40C"): dict(lambda_div=500, alternate=800, mix={"proxy": 0.5, "gan": 0.5},
                                   budget="8.0M", paper="ResNet-18 92.06 / AlexNet 76.02 (victims 93.65 / ~80)"),
    ("CIFAR-10", "Synth"):    dict(lambda_div=500, alternate=150, mix={"proxy": 0.5, "gan": 0.5},
                                   budget="7.5M", paper="ResNet-18 84.51 / AlexNet 67.03"),
    ("CIFAR-100", "C10"):     dict(lambda_div=100, alternate=400, mix={"proxy": 1.0, "gan": 0.5},
                                   budget="10.0M", paper="ResNet-18 72.83 (victim 78.52)"),
    ("CIFAR-100", "Synth"):   dict(lambda_div=100, alternate=200, mix={"proxy": 0.5, "gan": 0.5},
                                   budget="10.0M", paper="ResNet-18 43.56 (victim 78.52)"),
}

PROXIES = {
    "C100-40C": {"scenario": "Cross100-40C", "desc": "CIFAR-100 group_B, the paper's 40 unrelated classes (10,000 images)",
                 "block": {"type": "natural", "split": "group_B", "class_filter": "cifar100_40_unrelated",
                           "Dataset": {"name": "CIFAR-100", "normalization": "cifar100", "img_size": 32,
                                       "train_transforms": CNN_TRAIN_TF, "test_transforms": TEST_TF,
                                       "group_size": 25000}}},
    "C10": {"scenario": "Cross10", "desc": "CIFAR-10 group_B, all 10 classes (25,000 images), as the paper uses CIFAR-10 for its CIFAR-100 victim",
            "block": {"type": "natural", "split": "group_B", "class_filter": "all",
                      "Dataset": {"name": "CIFAR-10", "normalization": "cifar10", "img_size": 32,
                                  "train_transforms": CNN_TRAIN_TF, "test_transforms": TEST_TF,
                                  "group_size": 25000}}},
    "Synth": {"scenario": "Synth", "desc": "50,000 synthetic shape images (grey), no natural data at all",
              "block": {"type": "synthetic",
                        "synthetic": {"num_images": 50000, "num_shapes": 50, "min_size": 5, "max_size": 10,
                                      "canvas": 100, "grey": True, "seed": 0}}},
}

TIERS = []
for v in ("RES18", "VGG16", "DEIT"):
    for c in ("RES18", "VGG16", "DEIT"):
        TIERS.append(("A1" if (v, c) == ("RES18", "RES18") else "A2", "CIFAR-10", "C100-40C", v, c))
for a in ("RES18", "VGG16", "DEIT"):
    TIERS.append(("B1" if a == "RES18" else "B2", "CIFAR-10", "Synth", a, a))
for v in ("RES18", "VGG16", "DEIT"):
    for c in ("RES18", "VGG16", "DEIT"):
        TIERS.append(("C1" if (v, c) == ("RES18", "RES18") else "C2", "CIFAR-100", "C10", v, c))
TIERS.append(("D1", "CIFAR-100", "Synth", "RES18", "RES18"))


def make_plan(tier, ds, proxy_key, v, c):
    # deep copies so PyYAML writes plain lists instead of &id anchors
    D = DATASETS[ds]; V = copy.deepcopy(ARCHS[v]); C = copy.deepcopy(ARCHS[c])
    S = SETTINGS[(ds, proxy_key)]; PX = copy.deepcopy(PROXIES[proxy_key])
    rel = ("Same" if v == c else "Cross") + C["tag"]
    scenario = f'{ds}_{V["folder"]}_25000_DFMS_{PX["scenario"]}_{rel}'
    fname = f'{D["file"]}_{v}_DFMS_{proxy_key}_{rel}.yaml'
    recipe = DEIT_RECIPE if c == "DEIT" else CNN_RECIPE

    plan = {
        "Scenario_Name": scenario,
        "Victim": {
            "Model_Name": f'{ds}_{V["folder"]}_25000',
            "Model": V["model"],
            "Dataset": {"name": ds, "normalization": D["norm"], "img_size": 32,
                        "train_transforms": copy.deepcopy(DEIT_TRAIN_TF if v == "DEIT" else CNN_TRAIN_TF),
                        "test_transforms": copy.deepcopy(TEST_TF), "group_size": 25000},
        },
        "Substitute": {"Model": C["model"]},
        "Proxy": PX["block"],
        "DFMS": {
            "lambda_div": S["lambda_div"],
            "lambda_cls": 0,
            "temp": 1.0,
            "divgan_lambda_div": 10,
            "gan": {"nz": 100, "ngf": 64, "ndf": 64, "lr": 0.0002, "beta1": 0.5, "batch_size": 64},
            "epochs": {"dcgan": 200, "clone_offline": 200, "divgan": 100, "alternate": S["alternate"]},
            "max_samples": 50000,
            "mix_ratio": S["mix"],
            "offline_batch_size": 128,
            "val_samples": 20000,
            "autoaugment": True,
            "grad_clip": recipe["grad_clip"],
        },
        "Optimizer": copy.deepcopy(recipe["Optimizer"]),
        "Scheduler": copy.deepcopy(recipe["Scheduler"]),
        "Alternate": copy.deepcopy(recipe["Alternate"]),
    }

    header = [
        "# ======================================================",
        f"# DFMS-HL model extraction  |  tier {tier}",
        f"# Victim:     {ds} + {V['folder']} (saved_models/vanilla/.../{ds}_{V['folder']}_25000_<seed>_1.0)",
        f"# Clone:      {C['folder']} ({'same' if v == c else 'different'} architecture family)",
        f"# Proxy:      {PX['desc']}",
        f"# Budget:     {S['alternate']} alternating epochs x ceil(N_proxy/64) x 64 ~= {S['budget']} hard-label queries",
        f"# Paper ref:  {S['paper']}",
        "# Run with:   main_dfms_extraction.py (SEED_START/SEED_END pick attacker seeds)",
    ]
    if tier == "A1":
        header.append("# Seeds:      run attacker seeds 0..4 to match the knockoff H1 pool")
    if c == "DEIT":
        header.append("# NOTE:       DeiT clone is outside the paper; offline recipe = this repo's DeiT knockoff plan,")
        header.append("#             S7 lr = offline lr / 10 by analogy with the CNN schedule. Pilot before trusting.")
    if proxy_key == "C100-40C":
        header.append("# NOTE:       mix_ratio 0.5/0.5 (paper: 1.0/0.5) because group_B holds 10k of the 40 classes,")
        header.append("#             half the paper's 20k; this keeps 157 steps and ~10k G samples per offline epoch.")
    if ds == "CIFAR-100":
        header.append("# NOTE:       no reference run script for a CIFAR-100 victim; lambda_div and the 10M budget")
        header.append("#             come from the paper, mix_ratio and epoch split by analogy with the 40-class script.")
    header.append("# ======================================================")
    body = yaml.safe_dump(plan, sort_keys=False, default_flow_style=None, width=110)
    return fname, "\n".join(header) + "\n\n" + body


if __name__ == "__main__":
    os.makedirs(MATRIX, exist_ok=True)
    written = []
    for tier, ds, proxy_key, v, c in TIERS:
        fname, text = make_plan(tier, ds, proxy_key, v, c)
        path = os.path.join(MATRIX, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        written.append((tier, fname))
        if tier == "A1":
            shutil.copyfile(path, os.path.join(HERE, fname))
    for tier, fname in written:
        print(f"{tier:3s} {fname}")
    print(f"\n{len(written)} plans in {MATRIX}; tier A1 also copied to {HERE}")
