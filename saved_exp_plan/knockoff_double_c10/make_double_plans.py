"""Generate the CIFAR-10 "double knockoff" cells: victim -> S1 -> S2.

    python saved_exp_plan/knockoff_double_c10/make_double_plans.py

Writes, with no arguments:

    saved_exp_plan/knockoff_double_c10/matrix/       all 6 cells (never globbed)
    saved_exp_plan/knockoff_double_c10/run/chain_a/  the 2 RN18->RN18->RN18 cells
    saved_exp_plan/knockoff_double_c10/run/chain_b/  the 2 RN18->VGG16->DeiT cells
    saved_exp_plan/knockoff_double_c10/run/chain_c/  the 2 RN18->DeiT->VGG16 cells

Pick a folder at run time; nothing in the code changes:

    KNOCKOFF2_PLAN_DIR=./saved_exp_plan/knockoff_double_c10/run/chain_a
    python main_knockoff_extraction_double.py

run/ is split by GPU cost (160 ep SGD vs 200 ep AdamW) so the two halves can
occupy two sessions; main_knockoff_extraction_double.py also takes a basename
filter, so any other split works without regenerating.

------------------------------------------------------------------------------
The question
------------------------------------------------------------------------------
Every positive in this project is ONE hop from the victim. A real laundering
pipeline does not stop there -- the thief's surrogate is served as an API and
stolen again. Does the ownership signal survive a second hop?

    hop 0   victim   CIFAR-10 ResNet-18, group_A, seed 42
    hop 1   S1       knockoff on the victim, queried with CIFAR-10 group_B
                     -- ALREADY TRAINED, seeds 0..4, MI already on the grid
    hop 2   S2       knockoff on S1  -- these six cells
                     S2 is the suspect that gets measured

Three chains x two hop-2 query sets:

    chain A   S1 = ResNet-18  (homogeneous)    S2 = ResNet-18, stays homogeneous
    chain B   S1 = VGG16      (heterogeneous)  S2 = DeiT-Ti, crosses again
    chain C   S1 = DeiT-Ti    (heterogeneous)  S2 = VGG16, crosses back to a CNN

    hop-2 query set   CIFAR-10  group_B   (the same pool hop 1 used)
                      CIFAR-100 group_B   (out of distribution, disjoint)

Chain C is the reverse of chain B and the one with the least to lose: its
intermediate is already down at d = 4.78 after one hop. If two hops are ever
going to put a suspect back inside the null, this is where it happens.

------------------------------------------------------------------------------
Why the hop-2 query set is a factor and not a detail
------------------------------------------------------------------------------
Measured agreement between the hop-1 surrogates and the victim, by region
(seed 0, top-1 / KL(V||S1)):

    S1          group_B (S1 trained here)   test (unseen)    group_A (probe)
    RN18          99.31%  /  0.0073         93.70% / 0.144   94.22% / 0.172
    VGG16         99.05%  /  0.0253         92.34% / 0.193   92.60% / 0.235

ON group_B the surrogate IS the victim, to within KL 0.007-0.025; off it the
divergence is 6-20x larger. So the CIFAR-10 arm does not really measure a
second hop -- S2 gets back, to within ~1%, the very soft labels hop 1 got from
the victim, and the chain collapses to "one hop with 1% label noise". Read it
as the defender-favourable bound: if the signal dies there, it dies anywhere.

The CIFAR-100 arm is the real second hop: S1 is queried where it deviates from
the victim, so the fingerprint is genuinely filtered through S1's own
generalisation. CIFAR-10 train is exhausted (group_A is the victim's, group_B
is hop 1's), so an OOD pool is the only disjoint 25 000 images available.

------------------------------------------------------------------------------
Which baseline each cell is read against -- THE trap in this design
------------------------------------------------------------------------------
Changing the query set alone moves d more than adding a hop plausibly will.
Measured on the shared grid (bins=50, in_size=25000), 1-hop surrogates of this
same victim:

    surrogate   query set            d/p95     I(X;T)     I(T;Y)
    RN18        CIFAR-10  group_B    27.00     +37.74     +13.53   (5 seeds)
    RN18        CIFAR-100 group_B    61.37     +87.52     +13.75   (3 seeds)
    VGG16       CIFAR-10  group_B    17.73     +23.57     +13.98   (5 seeds)
    VGG16       CIFAR-100 group_B    55.76     +71.60     +15.47   (3 seeds)
    DeiT-Ti     CIFAR-10  group_B     4.78      +1.79      +5.92   (3 seeds)
    DeiT-Ti     CIFAR-100 group_B    22.19     +25.50     +12.93   (1 seed)

An OOD-queried surrogate is 2-5x FURTHER from the null while being 5-7 points
LESS accurate (87.2% vs 92.3% for RN18). The extra distance is almost entirely
I(X;T): a model trained on CIFAR-100 sees group_A as out of distribution, so
its logits there are unsaturated and spread out. That axis tracks "how far the
training data is from the probe", not "whose model this came from".

So a Cross100 hop-2 cell must be read against the Cross100 1-hop number, never
against the Same10 one, or two hops will look like they STRENGTHENED the
signal. Each cell's header names its own baseline. Prefer the I(T;Y) column
when comparing across query sets: it is the inherited-group_A-competence axis
and it is the one that is stable under a query-set change.

------------------------------------------------------------------------------
Seeds
------------------------------------------------------------------------------
Chain seed s pairs hop-1 seed s with hop-2 seed s:  S2(s) = knockoff(S1(s)),
s in {0,1,2}. The three averaged points are three end-to-end laundering runs,
so the spread includes hop-1 variance, as the hop-1 baseline's own spread does.

That is what main_knockoff_extraction_double.py means by passing the same
integer as BOTH model_seed and extract_seed: model_seed picks the hop-1 folder
suffix, extract_seed names the hop-2 output and seeds its training.

------------------------------------------------------------------------------
Naming
------------------------------------------------------------------------------
    CIFAR-10_ResNet-18_25000_Knockoff2_Same10_Via18_Cross100_CrossDeiT
    |______________________|_________|______|_____|________|_________|
     original victim (fixed) 2 hops   hop-1  hop-1 hop-2     hop-2 arch, named
                                      query  arch  query     relative to the
                                                             ORIGINAL victim

Alternating (query set, architecture) pairs, left to right, one pair per hop.
Same18/Cross16/CrossDeiT keep their one-hop meaning -- same as / different from
the VICTIM's architecture. `Knockoff2` separates the scenario namespace from
the one-hop cells in the same flat model tree.

Note for whoever reads the MI table: `victim_model` holds arch(S1), not
ResNet-18, because S1 is the model the attack queried; `aux_dataset` holds the
HOP-2 query set, which is what distinguishes the two arms. The `Scenario`
string is the authority on the chain. Those columns are frozen (the header is
validated on every append), so the chain could not be given one of its own.

No code change was needed for the OOD arm: main_knockoff already builds the
auxiliary group_B from whatever `Auxiliary_Dataset.name` says, and
Indices/CIFAR-100/group_B_25000_0.0_25000_seed42.npy already exists and is
disjoint from CIFAR-100 group_A.
"""
import copy
import os
import shutil

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MATRIX_DIR = os.path.join(HERE, "matrix")
RUN_ROOT = os.path.join(HERE, "run")
EXTRACTION_ROOT = os.path.join(REPO, "saved_models", "extraction_final")

CHAIN_SEEDS = (0, 1, 2)
VICTIM_SCENARIO = "CIFAR-10_ResNet-18_25000"

# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------
DEIT = {
    "model_name": "deit_tiny_patch16_224",
    "img_size": 32,
    "patch_size": 4,
    "pretrained": False,
    "drop_path_rate": 0.1,
}

CNN_AUG = [
    {"name": "RandomCrop", "params": {"size": 32, "padding": 4}},
    {"name": "RandomHorizontalFlip", "params": {"p": 0.5}},
    {"name": "ToTensor"},
    {"name": "Normalize"},
]

DEIT_AUG = [
    {"name": "RandomResizedCrop",
     "params": {"size": 32, "scale": [0.85, 1.0], "ratio": [0.9, 1.1]}},
    {"name": "RandomHorizontalFlip", "params": {"p": 0.5}},
    {"name": "RandAugment", "params": {"num_ops": 2, "magnitude": 7}},
    {"name": "ToTensor"},
    {"name": "Normalize"},
    {"name": "RandomErasing",
     "params": {"p": 0.15, "scale": [0.02, 0.25], "ratio": [0.3, 3.3]}},
]

CLEAN = [{"name": "ToTensor"}, {"name": "Normalize"}]


def dataset_block(name, normalization, train_transforms):
    # deepcopy, not list(): the same transform dicts appear in more than one
    # block of one plan, and yaml.safe_dump emits &id001/*id001 anchors for any
    # object it has already seen. That loads back identically but is
    # unreadable, and editing one block would silently edit the other.
    return {
        "name": name,
        "normalization": normalization,
        "img_size": 32,
        "train_transforms": copy.deepcopy(list(train_transforms)),
        "test_transforms": copy.deepcopy(list(CLEAN)),
        "group_size": 25000,
    }


# Surrogate recipes, copied verbatim from the one-hop plans so a two-hop cell
# and its one-hop baseline differ only by which model was queried.
CNN_RECIPE = {
    "Optimizer": {"name": "SGD", "params": {
        "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0005, "nesterov": True}},
    "Scheduler": {"name": "CosineAnnealingLR", "params": {
        "T_max": 160, "eta_min": 0.000001}},
    "Epochs": 160,
    "BatchSize": 128,
}

DEIT_RECIPE = {
    "Optimizer": {"name": "AdamW", "params": {
        "lr": 0.0007, "betas": [0.9, 0.999], "eps": 1.0e-8,
        "weight_decay": 0.03, "amsgrad": False}},
    "Scheduler": {"name": "WarmupCosineAnnealingLR", "params": {
        "warmup_epochs": 10, "warmup_start_factor": 0.1,
        "T_max": 200, "eta_min": 0.000001}},
    "Epochs": 200,
    "BatchSize": 128,
}

# --------------------------------------------------------------------------
# the pieces of a chain
# --------------------------------------------------------------------------
# hop-1 surrogates: trained, on disk with seeds 0..4, MI already on the shared
# grid. `aug` is the recipe THAT model was trained under -- it goes into
# Victim.Dataset.train_transforms as documentation only (the attack reads
# test_set / in_sample_set, both clean).
HOP1 = {
    "Via18": dict(
        scenario=VICTIM_SCENARIO + "_Knockoff_Same10_Same18",
        model="ResNet-18", label="ResNet-18", aug=CNN_AUG,
        d1="27.00", acc="92.3-92.4%"),
    "Via16": dict(
        scenario=VICTIM_SCENARIO + "_Knockoff_Same10_Cross16",
        model="VGG16", label="VGG16", aug=CNN_AUG,
        d1="17.73", acc="90.9-91.2%"),
    # The weakest intermediate of the three: one hop into a transformer already
    # costs most of the signal (d 26.67 -> 4.78), which is what makes the chain
    # through it worth measuring -- it starts with the least to lose.
    "ViaDeiT": dict(
        scenario=VICTIM_SCENARIO + "_Knockoff_Same10_CrossDeiT",
        model=DEIT, label="DeiT-Ti", aug=DEIT_AUG,
        d1="4.78", acc="83.8-84.5%"),
}

# hop-2 query sets. Both resolve to that dataset's group_B at overlap_rate 0.0,
# which main_knockoff builds unchanged; both index files already exist.
QUERY = {
    "Same10": dict(name="CIFAR-10", norm="cifar10",
                   note="CIFAR-10 group_B -- the SAME 25 000 images hop 1 used"),
    "Cross100": dict(name="CIFAR-100", norm="cifar100",
                     note="CIFAR-100 group_B -- 25 000 OOD images, disjoint from "
                          "every CIFAR-10 group"),
}

# hop-2 surrogates, named relative to the ORIGINAL victim. Each must have an
# 80-model rate-0.0 CIFAR-10 pool -- it is the model MI is read from.
HOP2 = {
    "Same18": dict(model="ResNet-18", label="ResNet-18", aug=CNN_AUG,
                   recipe=CNN_RECIPE,
                   pool="Negative_Model_Pool_0.0/CNN  (ResNet-18, 80)"),
    "Cross16": dict(model="VGG16", label="VGG16", aug=CNN_AUG,
                    recipe=CNN_RECIPE,
                    pool="Negative_Model_Pool_0.0/CNN  (VGG16, 80)"),
    "CrossDeiT": dict(model=DEIT, label="DeiT-Ti", aug=DEIT_AUG,
                      recipe=DEIT_RECIPE,
                      pool="Negative_Model_Pool_0.0/Transformer  (DeiT_Plain, 80)"),
}

# The 1-hop surrogate of the VICTIM with the same (query set, architecture) as
# this cell's hop 2. That -- not the hop-1 model of the chain -- is what a
# two-hop number must be compared against, because it holds the query set and
# the suspect's architecture fixed and differs only in the number of hops.
BASELINE = {
    ("Same10", "Same18"): ("CIFAR-10_ResNet-18_25000_Knockoff_Same10_Same18",
                           "27.00", "+13.53", "5 seeds, extraction_final/"),
    ("Cross100", "Same18"): ("CIFAR-10_ResNet-18_25000_Knockoff_Cross100_Same18",
                             "61.37", "+13.75", "3 seeds, extraction_vanilla/"),
    ("Same10", "Cross16"): ("CIFAR-10_ResNet-18_25000_Knockoff_Same10_Cross16",
                            "17.73", "+13.98", "5 seeds, extraction_final/"),
    ("Cross100", "Cross16"): ("CIFAR-10_ResNet-18_25000_Knockoff_Cross100_Cross16",
                              "55.76", "+15.47", "3 seeds, extraction_vanilla/"),
    ("Same10", "CrossDeiT"): ("CIFAR-10_ResNet-18_25000_Knockoff_Same10_CrossDeiT",
                              "4.78", "+5.92", "3 seeds, extraction_final/"),
    ("Cross100", "CrossDeiT"): ("CIFAR-10_ResNet-18_25000_Knockoff_Cross100_CrossDeiT",
                                "22.19", "+12.93",
                                "1 seed only, extraction_vanilla/Transformer_Models/"),
}

# (run folder, hop-1 key, hop-2 query key, hop-2 architecture key)
CELLS = [
    ("chain_a", "Via18", "Same10", "Same18"),
    ("chain_a", "Via18", "Cross100", "Same18"),
    ("chain_b", "Via16", "Same10", "CrossDeiT"),
    ("chain_b", "Via16", "Cross100", "CrossDeiT"),
    ("chain_c", "ViaDeiT", "Same10", "Cross16"),
    ("chain_c", "ViaDeiT", "Cross100", "Cross16"),
]


# --------------------------------------------------------------------------
# emit
# --------------------------------------------------------------------------
def build_cell(via_key, q_key, sub_key):
    via, q, sub = HOP1[via_key], QUERY[q_key], HOP2[sub_key]
    base_sc, base_d, base_ity, base_n = BASELINE[(q_key, sub_key)]
    scenario = (VICTIM_SCENARIO + "_Knockoff2_Same10_" + via_key
                + "_" + q_key + "_" + sub_key)

    plan = {
        "Scenario_Name": scenario,
        # Documentation only: nothing reads this block. It is here because the
        # Victim block below names S1, so without it the plan would record
        # neither who the actual owner is nor which number to compare against.
        "Chain": {
            "hop0_victim": VICTIM_SCENARIO + "  (ResNet-18, group_A, seed 42, rate 1.0)",
            "hop1_surrogate": via["scenario"] + "  (" + via["label"]
                              + ", queried with CIFAR-10 group_B, test acc "
                              + via["acc"] + ", d = " + via["d1"] + " p95)",
            "hop2_surrogate": scenario + "  (" + sub["label"] + ")  <- the suspect",
            "hop2_query_pool": q["note"],
            "measured_against": sub["pool"],
            "compare_with": base_sc,
            "compare_with_d_over_p95": base_d,
            "compare_with_I_T_Y_over_p95": base_ity,
            "compare_with_note": base_n,
        },
        # `Victim` is S1: the model this attack queries. Model_Dir points at
        # extraction_final/ instead of vanilla/, and Model_Name + _{seed}_1.0
        # resolves the hop-1 folder -- which is why the driver passes the chain
        # seed as model_seed.
        "Victim": {
            "Model_Name": via["scenario"],
            "Model_Dir": "./saved_models/extraction_final",
            "Model": copy.deepcopy(via["model"]),
            "Dataset": dataset_block("CIFAR-10", "cifar10", via["aug"]),
        },
        "Substitute": {"Model": copy.deepcopy(sub["model"])},
        # train_transforms here = the HOP-2 surrogate's own augmentation,
        # applied to the hop-2 query images. The query itself runs clean=True,
        # so the stolen soft labels do not depend on it.
        "Auxiliary_Dataset": dataset_block(q["name"], q["norm"], sub["aug"]),
        "Knockoff": {"sampling_size": 1.0},
    }
    plan.update(sub["recipe"])

    rule = "# " + "=" * 70
    header = (
        rule + "\n"
        "# Double knockoff extraction -- hop 2 of 2   (GENERATED, do not hand-edit)\n"
        "#   make_double_plans.py  ->  " + via_key + "_" + q_key + "_" + sub_key + "\n"
        + rule + "\n"
        "# chain    : CIFAR-10 ResNet-18 (victim)  ->  " + via["label"]
        + " (S1)  ->  " + sub["label"] + " (S2)\n"
        "# hop 1    : CIFAR-10 group_B   " + via["scenario"] + "\n"
        "#            already trained, seeds 0..4, test acc " + via["acc"]
        + ", d = " + via["d1"] + " p95\n"
        "# hop 2    : " + q["note"] + "\n"
        "# suspect  : S2, measured against " + sub["pool"] + "\n"
        "#\n"
        "# COMPARE AGAINST, and nothing else:\n"
        "#   " + base_sc + "\n"
        "#   d = " + base_d + " p95   I(T;Y) = " + base_ity + " p95   (" + base_n + ")\n"
        "# That is the 1-hop surrogate of the VICTIM with this cell's query set\n"
        "# AND architecture, so it differs from this cell only in hop count.\n"
        "# Comparing a Cross100 cell against a Same10 number instead would show\n"
        "# two hops 'strengthening' the signal -- the query set alone moves d by\n"
        "# 2-5x, almost all of it on the I(X;T) axis. Prefer I(T;Y) when reading\n"
        "# across query sets; it is the inherited-competence axis and is stable.\n"
        "#\n"
        "# query    : soft labels, KL objective, clean transforms -- same attack\n"
        "#            as hop 1; only the queried model (and, in the Cross100\n"
        "#            arm, the query pool) differ\n"
        "# recipe   : how a CIFAR-10 " + sub["label"] + " is normally trained, unchanged\n"
        "#            from the one-hop plan\n"
        "# run      : main_knockoff_extraction_double.py   (NOT the one-hop drivers:\n"
        "#            main_knockoff_extraction.py ignores Model_Dir and would load\n"
        "#            the ORIGINAL victim from vanilla/, silently redoing hop 1)\n"
        "# MI       : calculate_MI_extraction.py -- listed in EXTRACTION_BEST_PLANS\n"
        + rule + "\n"
    )
    fname = "CIFAR10_RES18_" + scenario[len(VICTIM_SCENARIO) + 1:] + ".yaml"
    return fname, header, plan


def dump(path, header, plan):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(header)
        yaml.safe_dump(plan, fh, sort_keys=False, default_flow_style=False,
                       allow_unicode=True, width=100)


def _done(scenario, root=EXTRACTION_ROOT):
    """Chain seeds whose best_epoch.pth is on disk for this scenario."""
    return [s for s in CHAIN_SEEDS
            if os.path.isfile(os.path.join(
                root, scenario + "_" + str(s) + "_1.0", "best_epoch.pth"))]


def main():
    if os.path.isdir(MATRIX_DIR):
        shutil.rmtree(MATRIX_DIR)
    os.makedirs(MATRIX_DIR)
    if os.path.isdir(RUN_ROOT):
        shutil.rmtree(RUN_ROOT)

    print("victim: " + VICTIM_SCENARIO + "   chain seeds: " + str(list(CHAIN_SEEDS)))
    print()
    ready = {}
    for via_key in HOP1:
        ready[via_key] = _done(HOP1[via_key]["scenario"])
        missing = [s for s in CHAIN_SEEDS if s not in ready[via_key]]
        note = "" if not missing else "   MISSING hop-1 seeds " + str(missing)
        print("  hop-1 %-7s %-10s seeds on disk %s%s"
              % (via_key, HOP1[via_key]["label"], ready[via_key], note))
    print()

    written = staged = 0
    for tier, via_key, q_key, sub_key in CELLS:
        fname, header, plan = build_cell(via_key, q_key, sub_key)
        dump(os.path.join(MATRIX_DIR, fname), header, plan)
        written += 1

        done = _done(plan["Scenario_Name"])
        todo = [s for s in CHAIN_SEEDS if s not in done]
        blocked = [s for s in todo if s not in ready[via_key]]

        chain = ("RN18 -> " + HOP1[via_key]["label"] + " -> " + HOP2[sub_key]["label"])
        flag = ""
        if done:
            flag += "  done=" + str(done)
        if blocked:
            flag += "  BLOCKED (no hop-1) " + str(blocked)
        if todo and not blocked:
            folder = os.path.join(RUN_ROOT, tier)
            os.makedirs(folder, exist_ok=True)
            shutil.copyfile(os.path.join(MATRIX_DIR, fname),
                            os.path.join(folder, fname))
            staged += 1
        print("  %-8s %-60s %-27s %-10s%s"
              % (tier, fname, chain, "hop2=" + q_key, flag))

    print()
    print(str(written) + " plan(s) written to matrix/, "
          + str(staged) + " staged into run/")
    print("  matrix : " + MATRIX_DIR)
    print("  run    : " + RUN_ROOT)


if __name__ == "__main__":
    main()
