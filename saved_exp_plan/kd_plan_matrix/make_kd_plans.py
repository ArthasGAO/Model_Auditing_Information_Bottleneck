"""Generate the "bigger same-family teacher -> pooled student" KD matrix.

    python saved_exp_plan/kd_plan_matrix/make_kd_plans.py

Writes, with no arguments:

    saved_exp_plan/kd_plan_matrix/matrix/          all 18 cells (never globbed)
    saved_exp_plan/kd_plan_matrix/run/c10_diag/    CIFAR-10  within-family only
    saved_exp_plan/kd_plan_matrix/run/c10_full/    CIFAR-10  all 9 cells
    saved_exp_plan/kd_plan_matrix/run/c100_diag/   CIFAR-100 within-family only
    saved_exp_plan/kd_plan_matrix/run/c100_full/   CIFAR-100 all 9 cells
    saved_exp_plan/train_plan/teacher_plan/        the victim plans (existing files kept)

Pick a folder at run time, nothing in the code changes:

    KD_PLAN_DIR=./saved_exp_plan/kd_plan_matrix/run/c10_diag python main_kd.py
    KD_PLAN_DIR=./saved_exp_plan/kd_plan_matrix/run/c10_diag python calculate_MI_kd.py

The run/ folders SKIP cells whose best_epoch.pth already exists, because
main_kd.py has no skip guard and would silently retrain and overwrite them.
matrix/ always holds the complete set for reference.

------------------------------------------------------------------------------
Design
------------------------------------------------------------------------------
MI needs a reference group homogeneous with the suspect model, so the STUDENT
is pinned to an architecture that has a 50+ model rate-0.0 pool:

    CIFAR-10   ResNet-18 / VGG16  (CNN_Models, Negative_Model_Pool_0.0/CNN)
               DeiT_Plain         (Negative_Model_Pool_0.0/Transformer, 80)
    CIFAR-100  ResNet-18 / VGG16  (CNN_Models, Negative_Model_Pool_0.0/CNN)
               DeiT_Distill       (Negative_Model_Pool_0.0/Transformer, 80)

That rules out the textbook KD students (ResNet-10, VGG8): no pool, no null.
So KD keeps its "compression" semantics by making the TEACHER the bigger one -
one extra victim per family instead of one extra 50-model pool:

    ResNet-18  11.17M  <-  ResNet-34  21.28M   (1.90x)
    VGG16      14.99M  <-  VGG19      20.30M   (1.35x)
    DeiT-Ti     5.36M  <-  DeiT-S     21.34M   (3.98x)

Every teacher is trained on group_A (rate 1.0, seed 42) with the STUDENT's own
pool recipe, and each cell copies that same recipe, so a KD positive and its
null differ only by the teacher term. Deviations are listed in EPOCHS below.

CIFAR-100 DeiT is the distilled variant on purpose - it is the only C100
transformer pool. timm's `distilled_training` flag stays False (nobody in the
KD path sets it), so the backbone returns ONE averaged (head + head_dist)/2
tensor in both train and eval, and KD/DKD train both heads through it. Same
convention as the C100 knockoff DeiT surrogates.

FitNet is omitted everywhere. It needs conv hint maps of matching shape:
ResNet feats[2] is (128,16,16) vs VGG (256,4,4), and DeiTKDStudent returns an
empty feats list. VGG19->VGG16 is the one cell where it would work.
"""
import os
import shutil

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MATRIX = os.path.join(HERE, "matrix")
RUN = os.path.join(HERE, "run")
TEACHER_PLAN_DIR = os.path.join(REPO, "saved_exp_plan", "train_plan", "teacher_plan")
# main_kd.py writes here since 2026-09-21 (was saved_models/kd_vanilla, which
# still holds the older to10 / to8 / ResNet-18-teacher rows).
KD_MODEL_ROOT = os.path.join(REPO, "saved_models", "kd_final")

GROUP_SIZE = 25000
SEED = 42
METHOD_NAMES = ("KD", "DKD")
CHECK_RATES = ("0.0", "1.0")

# --------------------------------------------------------------------------
# shared blocks
# --------------------------------------------------------------------------
TEST_TF = [{"name": "ToTensor"}, {"name": "Normalize"}]

CNN_TRAIN_TF = [
    {"name": "RandomCrop", "params": {"size": 32, "padding": 4}},
    {"name": "RandomHorizontalFlip", "params": {"p": 0.5}},
    {"name": "ToTensor"},
    {"name": "Normalize"},
]

# The two DeiT recipes are NOT interchangeable: CIFAR-100's pool is the
# distilled family and was trained with heavier augmentation.
DEIT_TRAIN_TF = {
    "CIFAR-10": [
        {"name": "RandomResizedCrop", "params": {"size": 32, "scale": [0.85, 1.0], "ratio": [0.9, 1.1]}},
        {"name": "RandomHorizontalFlip", "params": {"p": 0.5}},
        {"name": "RandAugment", "params": {"num_ops": 2, "magnitude": 7}},
        {"name": "ToTensor"},
        {"name": "Normalize"},
        {"name": "RandomErasing", "params": {"p": 0.15, "scale": [0.02, 0.25], "ratio": [0.3, 3.3]}},
    ],
    "CIFAR-100": [
        {"name": "RandomResizedCrop", "params": {"size": 32, "scale": [0.85, 1.0], "ratio": [0.9, 1.1]}},
        {"name": "RandomHorizontalFlip", "params": {"p": 0.5}},
        {"name": "RandAugment", "params": {"num_ops": 2, "magnitude": 9}},
        {"name": "ToTensor"},
        {"name": "Normalize"},
        {"name": "RandomErasing", "params": {"p": 0.20, "scale": [0.02, 0.20], "ratio": [0.3, 3.3]}},
    ],
}

DISTILLATION = [
    {"name": "KD", "params": {"TEMPERATURE": 4, "CE_WEIGHT": 0.1, "KD_WEIGHT": 0.9}},
    {"name": "DKD", "params": {"CE_WEIGHT": 1.0, "ALPHA": 1.0, "BETA": 8.0, "T": 4.0, "WARMUP": 20}},
]

COSINE = {"name": "CosineAnnealingLR", "params": {"T_max": "auto", "eta_min": 0.000001}}


def sgd(weight_decay):
    return {"name": "SGD", "params": {"lr": 0.1, "momentum": 0.9, "nesterov": True,
                                      "weight_decay": weight_decay}}


def adamw(weight_decay):
    return {"name": "AdamW", "params": {"lr": 0.0007, "betas": [0.9, 0.999], "eps": 1.0e-8,
                                        "weight_decay": weight_decay, "amsgrad": False}}


def warmup_cosine(warmup_epochs):
    return {"name": "WarmupCosineAnnealingLR",
            "params": {"warmup_epochs": warmup_epochs, "warmup_start_factor": 0.1,
                       "T_max": "auto", "eta_min": 0.000001}}


DATASETS = {
    "CIFAR-10": {"tag": "CIFAR10", "norm": "cifar10"},
    "CIFAR-100": {"tag": "CIFAR100", "norm": "cifar100"},
}

# --------------------------------------------------------------------------
# students -- one entry per (dataset, student), copied from that student's own
# negative pool recipe. `pool` records the pool's own budget so any deviation
# is visible in the generated header.
# --------------------------------------------------------------------------
STUDENTS = {
    ("CIFAR-10", "RES18"): {
        "suffix": "to18",
        "block": {"student_name": "ResNet-18"},
        "tf": CNN_TRAIN_TF, "opt": sgd(0.0005), "sched": COSINE,
        "epochs": 200, "aug": None,
        "pool": "CIFAR-10_ResNet-18_25000 rate 0.0 (160 ep)",
        "note": "200 ep, not the pool's 160: keeps this row comparable with the "
                "already-trained ResNet-18-teacher row in kd_plan_arch/.",
    },
    ("CIFAR-10", "VGG16"): {
        "suffix": "toVGG16",
        "block": {"student_name": "VGG16"},
        "tf": CNN_TRAIN_TF, "opt": sgd(0.0005), "sched": COSINE,
        "epochs": 200, "aug": None,
        "pool": "CIFAR-10_VGG16_25000 rate 0.0 (160 ep)",
        "note": "200 ep, not the pool's 160: same reason as the ResNet-18 student. "
                "(train_plan/old_plan/CIFAR10_VGG16_SGD_SMALL.yaml says Epochs 100, "
                "but all 10 trained rate-0.0 logs are 160 - the yaml was edited after "
                "the pool was built. 160 is the real recipe.)",
    },
    ("CIFAR-10", "DEIT"): {
        "suffix": "toDeiT",
        "block": {"student_name": "DeiT", "model_name": "deit_tiny_patch16_224",
                  "img_size": 32, "patch_size": 4, "pretrained": False, "drop_path_rate": 0.1},
        "tf": DEIT_TRAIN_TF["CIFAR-10"], "opt": adamw(0.03), "sched": warmup_cosine(10),
        "epochs": 200, "aug": {"use_mixup": False, "label_smoothing": 0.1},
        "pool": "CIFAR-10_DeiT_Plain_25000 rate 0.0 (200 ep)",
        "note": "matches the pool exactly.",
    },
    ("CIFAR-100", "RES18"): {
        "suffix": "to18",
        "block": {"student_name": "ResNet-18"},
        "tf": CNN_TRAIN_TF, "opt": sgd(0.001), "sched": COSINE,
        "epochs": 360, "aug": None,
        "pool": "CIFAR-100_ResNet-18_25000 rate 0.0 (360 ep, wd 1e-3)",
        "note": "matches the pool exactly. NOTE the weight decay is 1e-3 here, "
                "not the 5e-4 used everywhere on CIFAR-10.",
    },
    ("CIFAR-100", "VGG16"): {
        "suffix": "toVGG16",
        "block": {"student_name": "VGG16"},
        "tf": CNN_TRAIN_TF, "opt": sgd(0.0005), "sched": COSINE,
        "epochs": 400, "aug": None,
        "pool": "CIFAR-100_VGG16_25000 rate 0.0 (400 ep)",
        "note": "matches the pool exactly.",
    },
    ("CIFAR-100", "DEIT"): {
        "suffix": "toDeiT",
        "block": {"student_name": "DeiT", "model_name": "deit_tiny_distilled_patch16_224",
                  "img_size": 32, "patch_size": 4, "pretrained": False, "drop_path_rate": 0.15},
        "tf": DEIT_TRAIN_TF["CIFAR-100"], "opt": adamw(0.05), "sched": warmup_cosine(15),
        "epochs": 260, "aug": {"use_mixup": False, "label_smoothing": 0.15},
        "pool": "CIFAR-100_DeiT_Distill_25000 rate 0.0 (260 ep)",
        "note": "distilled backbone, matching the only C100 transformer pool. "
                "timm's distilled_training stays False, so the model emits one "
                "averaged (head + head_dist)/2 tensor and KD/DKD train both heads.",
    },
}

# --------------------------------------------------------------------------
# teachers -- the bigger same-family victim, trained on group_A seed 42.
# --------------------------------------------------------------------------
TEACHERS = {
    ("CIFAR-10", "RES34"): {
        "tag": "RES34", "scen": "ResNet-34", "victim_scen": "CIFAR-10_ResNet-34_25000",
        "block": {"teacher_name": "ResNet-34"},
        "family": "RES18",
    },
    ("CIFAR-10", "VGG19"): {
        "tag": "VGG19", "scen": "VGG19", "victim_scen": "CIFAR-10_VGG19_25000",
        "block": {"teacher_name": "VGG19"},
        "family": "VGG16",
    },
    ("CIFAR-10", "DEITS"): {
        "tag": "DeiTS", "scen": "DeiT-S", "victim_scen": "CIFAR-10_DeiT-S_25000",
        "block": {"teacher_name": "DeiT", "model_name": "deit_small_patch16_224",
                  "img_size": 32, "patch_size": 4, "pretrained": False, "drop_path_rate": 0.1},
        "family": "DEIT",
    },
    ("CIFAR-100", "RES34"): {
        "tag": "RES34", "scen": "ResNet-34", "victim_scen": "CIFAR-100_ResNet-34_25000",
        "block": {"teacher_name": "ResNet-34"},
        "family": "RES18",
    },
    ("CIFAR-100", "VGG19"): {
        "tag": "VGG19", "scen": "VGG19", "victim_scen": "CIFAR-100_VGG19_25000",
        "block": {"teacher_name": "VGG19"},
        "family": "VGG16",
    },
    ("CIFAR-100", "DEITS"): {
        "tag": "DeiTS", "scen": "DeiT-S", "victim_scen": "CIFAR-100_DeiT-S_Distill_25000",
        "block": {"teacher_name": "DeiT", "model_name": "deit_small_distilled_patch16_224",
                  "img_size": 32, "patch_size": 4, "pretrained": False, "drop_path_rate": 0.15},
        "family": "DEIT",
    },
}

# --------------------------------------------------------------------------
# SELF teachers -- the student's OWN architecture, i.e. the victim itself.
# This is the attacker who distills the deployed model into an identical
# network (born-again / self-distillation), the most direct KD-as-stealing
# setting. No extra victim is needed: the teacher checkpoint is the same pool
# victim the student's null is built around, which lives under CNN_Models/ or
# Transformer_Models/ -- NOT directly under vanilla/ like the RES34 / VGG19 /
# DeiT-S teachers main_train_teacher.py writes. Hence the explicit `ckpt`.
SELF_TEACHERS = {
    ("CIFAR-10", "SELF_RES18"): {
        "tag": "RES18", "scen": "ResNet-18", "family": "RES18",
        "block": {"teacher_name": "ResNet-18"},
        "ckpt": "./saved_models/vanilla/CNN_Models/CIFAR-10_ResNet-18_25000_42_1.0/best_epoch.pth",
    },
    ("CIFAR-10", "SELF_VGG16"): {
        "tag": "VGG16", "scen": "VGG16", "family": "VGG16",
        "block": {"teacher_name": "VGG16"},
        "ckpt": "./saved_models/vanilla/CNN_Models/CIFAR-10_VGG16_25000_42_1.0/best_epoch.pth",
    },
    ("CIFAR-10", "SELF_DEIT"): {
        "tag": "DEIT", "scen": "DeiT", "family": "DEIT",
        "block": {"teacher_name": "DeiT", "model_name": "deit_tiny_patch16_224",
                  "img_size": 32, "patch_size": 4, "pretrained": False,
                  "drop_path_rate": 0.1},
        "ckpt": "./saved_models/vanilla/Transformer_Models/CIFAR-10_DeiT_Plain_25000_42_1.0/best_epoch.pth",
    },
    ("CIFAR-100", "SELF_RES18"): {
        "tag": "RES18", "scen": "ResNet-18", "family": "RES18",
        "block": {"teacher_name": "ResNet-18"},
        "ckpt": "./saved_models/vanilla/CNN_Models/CIFAR-100_ResNet-18_25000_42_1.0/best_epoch.pth",
    },
    ("CIFAR-100", "SELF_VGG16"): {
        "tag": "VGG16", "scen": "VGG16", "family": "VGG16",
        "block": {"teacher_name": "VGG16"},
        "ckpt": "./saved_models/vanilla/CNN_Models/CIFAR-100_VGG16_25000_42_1.0/best_epoch.pth",
    },
    ("CIFAR-100", "SELF_DEIT"): {
        "tag": "DEIT", "scen": "DeiT", "family": "DEIT",
        "block": {"teacher_name": "DeiT", "model_name": "deit_tiny_distilled_patch16_224",
                  "img_size": 32, "patch_size": 4, "pretrained": False,
                  "drop_path_rate": 0.15},
        "ckpt": "./saved_models/vanilla/Transformer_Models/CIFAR-100_DeiT_Distill_25000_42_1.0/best_epoch.pth",
    },
}
TEACHERS.update(SELF_TEACHERS)
# (teacher key, student key) -- only the three diagonal cells, no cross terms.
SELF_PAIRS = [("SELF_RES18", "RES18"), ("SELF_VGG16", "VGG16"), ("SELF_DEIT", "DEIT")]

TEACHER_ORDER = ["RES34", "VGG19", "DEITS"]
STUDENT_ORDER = ["RES18", "VGG16", "DEIT"]


def victim_ckpt(victim_scen):
    return f"./saved_models/vanilla/{victim_scen}_{SEED}_1.0/best_epoch.pth"


# --------------------------------------------------------------------------
# emit
# --------------------------------------------------------------------------
def build_cell(dataset, t_key, s_key):
    ds = DATASETS[dataset]
    te = TEACHERS[(dataset, t_key)]
    st = STUDENTS[(dataset, s_key)]

    scenario = f"{dataset}_{te['scen']}{st['suffix']}_{GROUP_SIZE}"
    filename = f"{ds['tag']}_{te['tag']}{st['suffix']}_KD.yaml"

    teacher_block = dict(te["block"])
    # SELF teachers carry an explicit path (the victim sits in a CNN_Models /
    # Transformer_Models subfolder); the bigger teachers are written straight
    # into vanilla/ by main_train_teacher.py.
    teacher_block["teacher_ckpt"] = te.get("ckpt") or victim_ckpt(te["victim_scen"])

    plan = {
        "Scenario_Name": scenario,
        "Dataset": {
            "name": dataset,
            "normalization": ds["norm"],
            "img_size": 32,
            "train_transforms": st["tf"],
            "test_transforms": TEST_TF,
            "group_size": GROUP_SIZE,
        },
        "Teacher_Model": teacher_block,
        "Student_Model": st["block"],
    }
    if st["aug"] is not None:
        plan["Augmentation"] = st["aug"]
    plan["Optimizer"] = st["opt"]
    plan["Distillation"] = DISTILLATION
    plan["Scheduler"] = st["sched"]
    plan["Epochs"] = st["epochs"]
    plan["BatchSize"] = 128

    diagonal = te["family"] == s_key
    header = "\n".join([
        "# " + "=" * 70,
        f"# KD positive cell - {dataset}",
        f"#   Teacher / victim : {te['scen']:<10} "
        f"({'SELF / born-again' if t_key.startswith('SELF') else 'WITHIN-FAMILY' if diagonal else 'cross-family'})",
        f"#   Student /suspect : {s_key:<10} -> negative pool {st['pool']}",
        "#",
        "#   teacher_ckpt     : " + teacher_block["teacher_ckpt"],
        "#                      (train it with main_train_teacher.py, rate 1.0)",
        "#",
        "# Student recipe is copied from that student's own pool so the positive",
        "# and its null differ only by the teacher term.",
        "#   " + st["note"],
        "#",
        "# Methods: KD and DKD. FitNet is omitted - see make_kd_plans.py.",
        "# Generated by saved_exp_plan/kd_plan_matrix/make_kd_plans.py - edit there.",
        "# " + "=" * 70,
        "",
    ])
    return filename, scenario, header, plan, diagonal


def dump(path, header, plan):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(header)
        yaml.safe_dump(plan, fh, sort_keys=False, default_flow_style=False, allow_unicode=True)


# Seeds to look for when reporting what is already trained. 0,1,2 is the
# current convention; 42 is what KD used before 2026-09-21 and is still the
# suffix on everything in the legacy kd_vanilla tree.
CHECK_SEEDS = (0, 1, 2, 42)
KD_MODEL_ROOT_LEGACY = os.path.join(REPO, "saved_models", "kd_vanilla")


def trained_rates(scenario):
    """Which (method, seed, rate) already have a best_epoch.pth on disk.

    Looks in kd_final AND the legacy kd_vanilla tree, because the CIFAR-10
    self-distillation cell was trained there under the old seed-42 naming
    before this matrix existed. Legacy hits are tagged so they are not
    mistaken for something the current run would skip -- main_kd.py's guard
    only reads KD_MODEL_ROOT.
    """
    done = []
    for root, tag in ((KD_MODEL_ROOT, ""), (KD_MODEL_ROOT_LEGACY, " (kd_vanilla)")):
        for method in METHOD_NAMES:
            for seed in CHECK_SEEDS:
                for rate in CHECK_RATES:
                    folder = os.path.join(root, f"{scenario}_{method}_{seed}_{rate}")
                    if os.path.isfile(os.path.join(folder, "best_epoch.pth")):
                        done.append(f"{method}/s{seed}@{rate}{tag}")
    return done


# --------------------------------------------------------------------------
# teacher / victim training plans
# --------------------------------------------------------------------------
def teacher_plans():
    """(filename, header, plan) for every victim this matrix needs."""
    out = []

    # -- CNN victims: Model is a plain string, trained by main_overlap --------
    cnn = [
        ("CIFAR10_VGG19_SGD_SMALL.yaml", "CIFAR-10", "CIFAR-10_VGG19_25000", "VGG19",
         sgd(0.0005), 160, "CIFAR-10 VGG16 pool recipe, 160 ep - matching the trained "
                           "rate-0.0 VGG16 logs (the pool yaml's Epochs: 100 is stale) "
                           "and the ResNet-34 victim's budget"),
        ("CIFAR100_RES34_SGD_SMALL.yaml", "CIFAR-100", "CIFAR-100_ResNet-34_25000", "ResNet-34",
         sgd(0.001), 400, "lr/wd/cosine from the CIFAR-100 ResNet-18 pool (wd 1e-3), "
                           "400 ep as in the three earlier CIFAR-100 ResNet-34 runs "
                           "(72.96 / 73.36 / 72.86) rather than that pool's 360"),
        ("CIFAR100_VGG19_SGD_SMALL.yaml", "CIFAR-100", "CIFAR-100_VGG19_25000", "VGG19",
         sgd(0.0005), 400, "CIFAR-100 VGG16 pool recipe"),
    ]
    for fname, dataset, scen, model, opt, epochs, why in cnn:
        sched = {"name": "CosineAnnealingLR", "params": {"T_max": epochs, "eta_min": 0.000001}}
        plan = {
            "Scenario_Name": scen,
            "Dataset": {"name": dataset, "normalization": DATASETS[dataset]["norm"],
                        "img_size": 32, "train_transforms": CNN_TRAIN_TF,
                        "test_transforms": TEST_TF, "group_size": GROUP_SIZE},
            "Model": model,
            "Optimizer": opt,
            "Scheduler": sched,
            "Epochs": epochs,
            "BatchSize": 128,
        }
        header = "\n".join([
            "# " + "=" * 70,
            f"# {model} victim for the bigger-teacher KD row ({dataset}).",
            f"# Recipe: {why}.",
            "#",
            "# Launch with main_train_teacher.py (NOT main_train_nega.py):",
            "#   TEACHER_SEEDS=42 TEACHER_RATES=1.0 python main_train_teacher.py",
            "# Scenario_Name carries no seed/rate suffix - main_overlap appends",
            f"# _42_1.0, giving ./saved_models/vanilla/{scen}_42_1.0/best_epoch.pth",
            "# which is exactly what the kd_plan_matrix cells point at.",
            "# Generated by saved_exp_plan/kd_plan_matrix/make_kd_plans.py.",
            "# " + "=" * 70,
            "",
        ])
        out.append((fname, header, plan))

    # -- DeiT-S victims: Model is a dict, needs main_deit_overlap ------------
    deit = [
        ("CIFAR10_DeiTS_SMALL.yaml", "CIFAR-10", "CIFAR-10_DeiT-S_25000",
         "deit_small_patch16_224", 0.1, adamw(0.03), warmup_cosine(10), 200,
         {"use_mixup": False, "label_smoothing": 0.1}, None),
        ("CIFAR100_DeiTS_Distill_SMALL.yaml", "CIFAR-100", "CIFAR-100_DeiT-S_Distill_25000",
         "deit_small_distilled_patch16_224", 0.15, adamw(0.05), warmup_cosine(15), 260,
         {"use_mixup": False, "label_smoothing": 0.15},
         {"enabled": True, "type": "hard", "alpha": 0.5, "tau": 2.0,
          "teacher_name": "ResNet-18",
          "teacher_ckpt": "./saved_models/vanilla/CNN_Models/"
                          "CIFAR-100_ResNet-18_25000_42_1.0/best_epoch.pth"}),
    ]
    for fname, dataset, scen, model_name, dp, opt, sched, epochs, aug, dist in deit:
        sched = {"name": sched["name"], "params": dict(sched["params"], T_max=epochs)}
        plan = {
            "Scenario_Name": scen,
            "Dataset": {"name": dataset, "normalization": DATASETS[dataset]["norm"],
                        "img_size": 32, "train_transforms": DEIT_TRAIN_TF[dataset],
                        "test_transforms": TEST_TF, "group_size": GROUP_SIZE},
            "Model": {"model_name": model_name, "img_size": 32, "patch_size": 4,
                      "pretrained": False, "drop_path_rate": dp},
            "Augmentation": aug,
        }
        plan["Distillation"] = dist if dist is not None else {"enabled": False}
        plan["Optimizer"] = opt
        plan["Scheduler"] = sched
        plan["Epochs"] = epochs
        plan["BatchSize"] = 128

        lines = [
            "# " + "=" * 70,
            f"# DeiT-S victim for the bigger-teacher KD row ({dataset}).",
            f"# Recipe copied from the {dataset} DeiT pool, backbone widened",
            f"# deit_tiny -> deit_small (5.36M -> 21.34M).",
            "#",
        ]
        if dist is not None:
            lines += [
                "# The distilled backbone is NOT optional: CIFAR-100's only transformer",
                "# pool is DeiT_Distill, so the KD students must be distilled too, and the",
                "# victim follows the same family. Its DeiT teacher is the CIFAR-100",
                "# ResNet-18 VICTIM (group_A, rate 1.0) - a victim may see group_A. The",
                "# pool plan deliberately uses a seed-paired rate-0.0 teacher instead, so",
                "# that negatives never inherit group_A through the teacher's logits.",
                "#",
            ]
        else:
            lines += [
                "# Plain backbone, matching the CIFAR-10 DeiT_Plain pool. No distillation.",
                "#",
            ]
        lines += [
            "# Model is a dict, so this plan needs main_deit_overlap, NOT main_overlap.",
            "# main_train_teacher.py currently calls main_overlap unconditionally - it",
            "# needs a dict/str branch before this plan can run.",
            "#",
            f"# Expect ./saved_models/vanilla/{scen}_42_1.0/best_epoch.pth",
            "# Generated by saved_exp_plan/kd_plan_matrix/make_kd_plans.py.",
            "# " + "=" * 70,
            "",
        ]
        out.append((fname, "\n".join(lines), plan))

    return out


def main():
    for d in (MATRIX, RUN):
        os.makedirs(d, exist_ok=True)

    run_dirs = {}
    for dataset, short in (("CIFAR-10", "c10"), ("CIFAR-100", "c100")):
        # "cnn" is the within-family diagonal minus the DeiT cell: the two cells
        # that need no timm teacher, so they run with what is already wired up.
        for kind in ("cnn", "diag", "full", "self"):
            path = os.path.join(RUN, f"{short}_{kind}")
            if os.path.isdir(path):
                shutil.rmtree(path)
            os.makedirs(path)
            run_dirs[(dataset, kind)] = path

    print(f"{'cell':<34} {'diag':<5} {'teacher ckpt':<8} already trained")
    print("-" * 92)

    staged = {k: 0 for k in run_dirs}
    for dataset in ("CIFAR-10", "CIFAR-100"):
        for t_key in TEACHER_ORDER:
            for s_key in STUDENT_ORDER:
                fname, scenario, header, plan, diagonal = build_cell(dataset, t_key, s_key)
                dump(os.path.join(MATRIX, fname), header, plan)

                ckpt = os.path.join(REPO, plan["Teacher_Model"]["teacher_ckpt"].lstrip("./"))
                have_ckpt = "yes" if os.path.isfile(ckpt) else "MISSING"
                done = trained_rates(scenario)

                # Every cell is staged, finished or not. main_kd.py now has a
                # skip guard keyed on (scenario, method, seed, rate), so a
                # finished cell costs nothing - and excluding whole plans here
                # was actively wrong once seeds became repeated experiments: a
                # plan dropped because seed 42 was done would never get seeds
                # 43 and 44 trained.
                kinds = ("diag", "full") if diagonal else ("full",)
                if diagonal and s_key != "DEIT":
                    kinds = ("cnn",) + kinds
                for kind in kinds:
                    dump(os.path.join(run_dirs[(dataset, kind)], fname), header, plan)
                    staged[(dataset, kind)] += 1

                print(f"{fname:<34} {'*' if diagonal else ' ':<5} {have_ckpt:<8} "
                      f"{', '.join(done) if done else '-'}")

    # -- the self-distillation row: teacher == the student's own victim -----
    for dataset in ("CIFAR-10", "CIFAR-100"):
        for t_key, s_key in SELF_PAIRS:
            fname, scenario, header, plan, _ = build_cell(dataset, t_key, s_key)
            dump(os.path.join(MATRIX, fname), header, plan)
            ckpt = os.path.join(REPO, plan["Teacher_Model"]["teacher_ckpt"].lstrip("./"))
            have_ckpt = "yes" if os.path.isfile(ckpt) else "MISSING"
            done = trained_rates(scenario)
            dump(os.path.join(run_dirs[(dataset, "self")], fname), header, plan)
            staged[(dataset, "self")] += 1
            print(f"{fname:<34} {'S':<5} {have_ckpt:<8} "
                  f"{', '.join(done) if done else '-'}")

    print("-" * 92)
    for (dataset, kind), path in run_dirs.items():
        print(f"staged {staged[(dataset, kind)]:>2} plan(s) -> {os.path.relpath(path, REPO)}")

    print()
    os.makedirs(TEACHER_PLAN_DIR, exist_ok=True)
    for fname, header, plan in teacher_plans():
        path = os.path.join(TEACHER_PLAN_DIR, fname)
        if os.path.exists(path):
            print(f"[keep]  {fname} already exists")
            continue
        dump(path, header, plan)
        scen = plan["Scenario_Name"]
        ck = os.path.join(REPO, "saved_models", "vanilla", f"{scen}_{SEED}_1.0", "best_epoch.pth")
        print(f"[write] {fname:<34} -> {scen}_{SEED}_1.0 "
              f"({'trained' if os.path.isfile(ck) else 'NOT trained'})")


if __name__ == "__main__":
    main()
