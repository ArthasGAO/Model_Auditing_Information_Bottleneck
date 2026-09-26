"""Two-hop ("double") knockoff extraction: victim -> S1 -> S2.

    KNOCKOFF2_PLAN_DIR=./saved_exp_plan/knockoff_double_c10/run/chain_a \
    KNOCKOFF2_SEEDS=0,1,2 python main_knockoff_extraction_double.py [name-filter ...]

The attack itself is NOT reimplemented here. This file is only an entry point:
the per-cell work is `main_knockoff_extraction_deit.main_knockoff`, unchanged,
because that fork already does the two things a second hop needs --

  * `Victim.Model_Dir` overrides the victim root, so the queried model can live
    in saved_models/extraction_final/ instead of saved_models/vanilla/;
  * `Victim.Model` / `Substitute.Model` may each be a CNN string or a DeiT
    dict, so both the CNN and the DeiT hop-2 cells build.

so double extraction needed no change to the attack, only a different loop.

------------------------------------------------------------------------------
The one thing that differs from the one-hop drivers
------------------------------------------------------------------------------
They call main_knockoff(model_seed=42, extract_seed=s): 42 is the victim's own
seed, which is what `vanilla/CIFAR-10_ResNet-18_25000_42_1.0` is suffixed with.

Here the queried model is a hop-1 SURROGATE, whose folder is suffixed with the
hop-1 attacker seed, so this driver calls

    main_knockoff(model_seed=s, extract_seed=s)

with the same chain seed on both sides: model_seed resolves
`extraction_final/<hop1 scenario>_s_1.0` and extract_seed names the hop-2
output `extraction_final/<hop2 scenario>_s_1.0` and seeds its training. The
three chains are therefore end to end independent -- seed s of hop 2 is always
built on seed s of hop 1 -- so the spread over seeds includes hop-1 variance
rather than hiding it. See make_double_plans.py for why that was chosen.

Because the same string arrives as both arguments, a plan run through the WRONG
driver fails loudly rather than quietly: main_knockoff_extraction.py ignores
`Model_Dir`, so it would look for `vanilla/CNN_Models/<hop1 scenario>_42_1.0`
and raise FileNotFoundError. It cannot silently redo hop 1.

------------------------------------------------------------------------------
Outputs
------------------------------------------------------------------------------
The same flat tree as every other extraction run -- saved_models/extraction_final/
and saved_logs/extraction_final/Performance/ -- because the scenario namespace
already separates them (`Knockoff2` vs `Knockoff`). One MI sweep therefore
covers one-hop and two-hop cells together, which is the point: the hop-1 number
each chain starts from is already in the master table.

SKIP_EXISTING is inherited from the imported module: a cell whose
best_epoch.pth exists is skipped before the victim is even loaded. As
everywhere else in this repo that guard cannot tell an interrupted run from a
finished one -- cross-check `epoch_<N-1>.pth` and the training-log row count
after any interruption (audit_kd_runs.py does this for the KD tree).
"""
import glob
import os
import sys

import torch

from main_knockoff_extraction_deit import (  # noqa: F401  (SKIP_EXISTING is read there)
    EXTRACTION_ROOT, SKIP_EXISTING, main_knockoff, set_seed,
)
from util import process_yaml_file

DEFAULT_PLAN_DIR = "./saved_exp_plan/knockoff_double_c10/run/chain_a"


def _seeds_from_env():
    raw = os.environ.get("KNOCKOFF2_SEEDS", "0,1,2")
    return [int(tok) for tok in raw.replace(",", " ").split()]


def _hop1_dir(exp_yaml, seed):
    """Where main_knockoff will look for the queried (hop-1) model."""
    victim = exp_yaml["Victim"]
    root = victim.get("Model_Dir", "./saved_models/vanilla/CNN_Models")
    return os.path.join(root, victim["Model_Name"] + "_" + str(seed) + "_1.0")


def _survey(yaml_files, seeds):
    """Per-(plan, seed) status, printed before anything is trained.

    Three states matter and are easy to confuse: a cell already finished, a
    cell ready to run, and a cell whose hop-1 model is absent -- the last one
    would otherwise surface 160 epochs into the run as a FileNotFoundError from
    a completely different file.
    """
    rows = []
    for path in yaml_files:
        exp_yaml = process_yaml_file(path)
        scenario = exp_yaml["Scenario_Name"]
        for seed in seeds:
            out = os.path.join(EXTRACTION_ROOT, scenario + "_" + str(seed) + "_1.0",
                               "best_epoch.pth")
            hop1 = os.path.join(_hop1_dir(exp_yaml, seed), "best_epoch.pth")
            if os.path.isfile(out) and os.path.getsize(out) > 0:
                state = "done"
            elif not os.path.isfile(hop1):
                state = "no-hop1"
            else:
                state = "to-train"
            rows.append((path, scenario, seed, state, hop1))
    return rows


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    plan_dir = os.environ.get("KNOCKOFF2_PLAN_DIR", DEFAULT_PLAN_DIR)
    seeds = _seeds_from_env()
    filters = [arg.lower() for arg in sys.argv[1:]]

    yaml_files = sorted(glob.glob(os.path.join(plan_dir, "*.yaml")))
    if filters:
        yaml_files = [f for f in yaml_files
                      if any(x in os.path.basename(f).lower() for x in filters)]

    print("[ENV] KNOCKOFF2_PLAN_DIR = " + plan_dir)
    print("[ENV] KNOCKOFF2_SEEDS    = " + str(seeds))
    print("[ENV] filters            = " + (str(filters) if filters else "(none)"))
    print("[ENV] SKIP_EXISTING      = " + str(SKIP_EXISTING))

    if not yaml_files:
        print("No YAML files found in " + plan_dir)
        sys.exit(1)

    print("Found " + str(len(yaml_files)) + " experiment plan(s):")
    for f in yaml_files:
        print(" -", f)

    rows = _survey(yaml_files, seeds)
    blocked = [r for r in rows if r[3] == "no-hop1"]
    todo = [r for r in rows if r[3] == "to-train"]
    done = [r for r in rows if r[3] == "done"]
    print("\nTOTAL: " + str(len(rows)) + " cell(s), " + str(len(done))
          + " already trained, " + str(len(todo)) + " to train, "
          + str(len(blocked)) + " blocked")
    for _, scenario, seed, _, hop1 in blocked:
        print("  [NO HOP-1] " + scenario + " seed " + str(seed) + " needs " + hop1)
    if blocked and not todo:
        print("Nothing runnable; train the hop-1 surrogates first.")
        sys.exit(1)

    for yaml_path in yaml_files:
        print("\n========== Starting experiments from " + yaml_path + " ==========")
        for chain_seed in seeds:
            print("\n>>> chain seed " + str(chain_seed)
                  + " (hop-1 model seed = hop-2 attacker seed = " + str(chain_seed)
                  + ") for " + os.path.basename(yaml_path))
            set_seed(chain_seed)
            main_knockoff(chain_seed, chain_seed, yaml_path)
