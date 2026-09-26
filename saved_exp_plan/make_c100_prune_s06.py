"""Derive single-sparsity 0.60 pruning plans from the existing CIFAR-100 ones.

    python saved_exp_plan/make_c100_prune_s06.py

Reads each frozen 0.2/0.8 plan, keeps EVERYTHING as it is, and emits a copy
whose Optimizers list holds exactly one entry: the same optimizer block with
sparsity 0.60. Nothing else -- dataset, transforms, model, scheduler, epochs,
lr -- is touched, and the source plans are never rewritten.

    saved_exp_plan/prune_plan_c100_s06/       ResNet-18, VGG16  -> main_prune
    saved_exp_plan/prune_plan_c100_s06_deit/  DeiT-Ti distilled -> main_prune_deit

ONE SPARSITY PER PLAN, and 0.2 is deliberately absent. main_prune.py's skip
guard works per (plan, ft_seed) and only fires when EVERY sparsity in that plan
already has best_epoch.pth -- so a plan holding {0.2, 0.6} would report "not
done" and retrain 0.2 from scratch. Splitting the level out is what keeps the
existing 0.2 models untouched.

Same reason the file must not later grow a second level: once
calculate_MI_prune.py writes rows from it, its sha256 is part of those rows'
identity (see the 2026-09-21 "Prune-MI identity mismatch" incident).
"""
import copy
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
NEW_SPARSITY = 0.6

SOURCES = [
    ("prune_plan_c100/CIFAR100_RES18_PRUNE_Same_25000.yaml",
     "prune_plan_c100_s06/CIFAR100_RES18_PRUNE_Sparsity060_Same_25000.yaml"),
    ("prune_plan_c100/CIFAR100_VGG16_PRUNE_Same_25000.yaml",
     "prune_plan_c100_s06/CIFAR100_VGG16_PRUNE_Sparsity060_Same_25000.yaml"),
    ("prune_plan_c100_deit/CIFAR100_DEIT_DISTILL_PRUNE_Same_25000.yaml",
     "prune_plan_c100_s06_deit/CIFAR100_DEIT_DISTILL_PRUNE_Sparsity060_Same_25000.yaml"),
]

HEADER = """# ======================================================================
# CIFAR-100 pruning at sparsity {sp} -- {model}
#   derived by saved_exp_plan/make_c100_prune_s06.py from
#   saved_exp_plan/{src}
#   ONLY the sparsity differs. Optimizer, epochs, scheduler, dataset,
#   transforms and Scenario_Name are copied verbatim from that plan, so the
#   only variable between this cell and the 0.2 cell is the sparsity.
#
#   {opt} lr={lr} x {ep} ep, FT-AL
#   driver : {driver}
#
# Single level on purpose. main_prune.py's skip guard is per (plan, ft_seed)
# and trains EVERY sparsity the plan lists, so keeping 0.2 in here would
# retrain the 0.2 models that are already done. They live in the frozen source
# plan and are not re-entered from this folder.
#
# Do not add a second sparsity to this file later: calculate_MI_prune.py
# records its sha256 in every MI row, and changing the bytes invalidates them.
# Copy it to a new _SparsityNNN_ file instead.
# ======================================================================
"""


def main():
    for src_rel, dst_rel in SOURCES:
        src = os.path.join(HERE, src_rel)
        with open(src, "r", encoding="utf-8") as fh:
            plan = yaml.safe_load(fh)

        opts = plan.get("Optimizers") or []
        if not opts:
            raise ValueError(f"{src_rel}: no Optimizers block")
        # Take the existing block verbatim and only swap the sparsity. Every
        # level in these plans carries identical params/Epochs (they share a
        # YAML anchor), so index 0 is representative -- assert that.
        base = copy.deepcopy(opts[0])
        for o in opts[1:]:
            a = {k: v for k, v in o.items() if k != "sparsity"}
            b = {k: v for k, v in base.items() if k != "sparsity"}
            if a != b:
                raise ValueError(f"{src_rel}: levels differ beyond sparsity; "
                                 f"pick one explicitly instead of opts[0]")
        base["sparsity"] = NEW_SPARSITY
        plan["Optimizers"] = [base]

        dst = os.path.join(HERE, dst_rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        head = HEADER.format(
            sp=NEW_SPARSITY, model=plan["Model_Name"], src=src_rel,
            opt=base["name"], lr=base["params"]["lr"], ep=base["Epochs"],
            driver=("main_prune_deit" if isinstance(plan.get("Model"), dict)
                    else "main_prune"),
        )
        with open(dst, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(head)
            yaml.safe_dump(plan, fh, sort_keys=False, default_flow_style=False)
        print(f"{dst_rel}\n    {base['name']} lr={base['params']['lr']} "
              f"{base['Epochs']}ep  sparsity={NEW_SPARSITY}  "
              f"Scenario={plan['Scenario_Name']}")


if __name__ == "__main__":
    main()
