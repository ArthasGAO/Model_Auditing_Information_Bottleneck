"""Hypothesis test for the double-knockoff chains, 1 hop vs 2 hops.

    python run_hypothesis_test_knockoff2.py

Nothing statistical is re-derived here. This file only supplies a
POSITIVE_FAMILIES configuration to `run_hypothesis_test_positives` and calls
its `main()`, so every suspect is judged against exactly the reference sets the
negative run used: the same frozen manifest (80-model pool, 50 rounds,
k in {5,10,15,20,25,30}), the same mu/Sigma estimator, and the same
chi2 / exact-F laws.

Why four families and not one
-----------------------------
A family carries ONE MI table and ONE H0 scenario, and this experiment needs
two of each:

  * H0 is per architecture, because the null has to hold architecture fixed.
    Chain A's suspects are ResNet-18, chain B's are DeiT-Ti, so they are scored
    against their own 80-model pools.
  * The 1-hop CIFAR-100-queried surrogates predate extraction_final/ and still
    live in extraction_vanilla/; their MI on the shared 70-cell grid is in a
    separate table (MI_master_table_extraction_grid70.csv) rather than appended
    to the single-writer extraction_final table.

Within a family `group_by=("Scenario",)` reports every cell separately, so the
six dirs give twelve numbers: 1 hop and 2 hops, for each of the three chains,
for each of the two hop-2 query sets.

Reading the result
------------------
chi2 underflows to exactly 0 on a strong suspect; the exact F law keeps
resolution in the same cells, so read `p_F`. Report it against the matched
1-hop cell -- same architecture, same query set -- not across query sets.
"""
from pathlib import Path

import run_hypothesis_test_positives as H

BASE = Path(__file__).resolve().parent
PLANS = BASE / "saved_exp_plan/hypo_knockoff2"
CSV_FINAL = BASE / "saved_logs/extraction_final/MI_master_table_extraction.csv"
CSV_OOD1 = BASE / "saved_logs/extraction_vanilla/MI_master_table_extraction_grid70.csv"

H.POSITIVE_FAMILIES = [
    # chain A: suspects are ResNet-18, scored against the ResNet-18 pool
    {"label": "chainA_rn18",
     "plan_dir": PLANS / "rn18_final",
     "mi_csv": CSV_FINAL,
     "h0_scenario": "CIFAR-10_ResNet-18_25000",
     "group_by": ("Scenario",)},
    {"label": "chainA_rn18_ood1hop",
     "plan_dir": PLANS / "rn18_ood1",
     "mi_csv": CSV_OOD1,
     "h0_scenario": "CIFAR-10_ResNet-18_25000",
     "group_by": ("Scenario",)},
    # chain B: suspects are DeiT-Ti, scored against the DeiT_Plain pool
    {"label": "chainB_deit",
     "plan_dir": PLANS / "deit_final",
     "mi_csv": CSV_FINAL,
     "h0_scenario": "CIFAR-10_DeiT_Plain_25000",
     "group_by": ("Scenario",)},
    {"label": "chainB_deit_ood1hop",
     "plan_dir": PLANS / "deit_ood1",
     "mi_csv": CSV_OOD1,
     "h0_scenario": "CIFAR-10_DeiT_Plain_25000",
     "group_by": ("Scenario",)},
    # chain C: suspects are VGG16, scored against the VGG16 pool
    {"label": "chainC_vgg16",
     "plan_dir": PLANS / "vgg16_final",
     "mi_csv": CSV_FINAL,
     "h0_scenario": "CIFAR-10_VGG16_25000",
     "group_by": ("Scenario",)},
    {"label": "chainC_vgg16_ood1hop",
     "plan_dir": PLANS / "vgg16_ood1",
     "mi_csv": CSV_OOD1,
     "h0_scenario": "CIFAR-10_VGG16_25000",
     "group_by": ("Scenario",)},
]

# One configuration: the operating point every d and every plot in this
# experiment already uses. Widen IN_SIZE_RATES / BINS to sweep the grid; each
# combination writes its own result directory.
H.IN_SIZE_RATES = [1.00]
H.BINS = [50]
H.MI_KIND = "In"
H.ALPHAS = [0.05, 0.01]
H.RUN_MODE = "evaluate"
# main() refuses to write into a directory that already holds a combined
# summary, so a re-run after adding cells needs a new tag. v1 = the first four
# cells; v2 = all six.
TAG = "v2"
H.OUTPUT_DIR = BASE / f"saved_logs/vanilla/Hypo_Test_Knockoff2_{TAG}"

if __name__ == "__main__":
    H.main()
