# Model Auditing with Information-Bottleneck Statistics

Does a suspect model derive from a victim model? This repository measures how much a
model's output retains about the victim's training data, I(X;T) and I(T;Y) estimated on
nested subsets of the victim's training set, and tests the suspect's statistics against a
reference set of independently trained models with a Hotelling T² / exact F test.
Everything needed to reproduce the CIFAR-10 × ResNet-18 and CIFAR-100 × VGG16 cases is here.

## Setup

Python 3.12; the packages and the versions the results were produced with are listed in
`requirements.txt` (newer releases work).

```bash
pip install -r requirements.txt
```

CIFAR-10 / CIFAR-100 download automatically into `./data` (or `--data-root <dir>`).
A GPU is assumed; every command falls back to CPU.

## Layout

| File | Role |
|---|---|
| `main_victim.py` | train the victim of a scenario on the split group_A |
| `main_negative.py` | train independent negatives (reference pool) on the disjoint split group_B |
| `main_fine_tune.py` | positives: FT-LL, FT-AL, RT-AL fine-tuning of the victim |
| `main_prune.py` | positives: global L1 pruning (0.2, 0.8) + FT-AL recovery |
| `main_distillation.py` | positives: KD and DKD students of the victim |
| `main_extraction.py` | positives: Knockoff substitutes trained on the victim's soft labels |
| `calculate_MI.py` | MI grid of any trained model; also called by every trainer after training |
| `hypothesis_test.py` | reference split, false-positive rate on negatives, detection rate on positives |
| `util.py` | plans, naming, data splits, models, training loops, attack primitives |
| `saved_exp_plan/<stage>/*.yaml` | one plan per (stage, scenario); `hypothesis/default.yaml` configures the test |

Each trainer reads the plans of its stage, trains every (plan, seed) cell that has no
`best_epoch.pth` yet, and appends the model's MI grid to the stage table. All outputs are
written next to the code (or under `--output-root <dir>`):

```
saved_models/<stage>/<model_name>/best_epoch.pth      selected by test accuracy
saved_logs/<stage>/Performance/training_log_<model_name>.csv
saved_logs/<stage>/MI_<stage>.csv                      one row per (model, in_size, bins)
saved_logs/hypothesis/                                 reference_splits.json, negatives/, positives/
Indices/<dataset>/                                     frozen data splits and MI probe subsets
```

## Reproducing a case

The two scenarios are `CIFAR10_ResNet18` and `CIFAR100_VGG16`; every stage has a plan of
each name, so one `--plans` value selects the whole case. The commands below run the
CIFAR-10 case; swap the plan name for CIFAR-100.

```bash
# 1. victim (seed 42, trained on group_A)
python main_victim.py --plans CIFAR10_ResNet18

# 2. reference pool: negatives trained on group_B
python main_negative.py --plans CIFAR10_ResNet18

# 3. positives derived from the victim
python main_fine_tune.py    --plans CIFAR10_ResNet18      # FT-LL, FT-AL, RT-AL
python main_prune.py        --plans CIFAR10_ResNet18      # sparsity 0.2, 0.6
python main_distillation.py --plans CIFAR10_ResNet18      # KD, DKD
python main_extraction.py   --plans CIFAR10_ResNet18      # Knockoff

# 4. hypothesis test on the MI tables
python hypothesis_test.py all
```

Step 4 prints, per MI cell, the false-positive rate over the evaluated negatives and the
detection rate per positive group, and writes `summary_all_cells.csv` under
`saved_logs/hypothesis/negatives/` and `positives/`. Both scenarios can be trained before
running the test once; it handles every case it finds victim plans for.

MI is measured automatically at the end of each training run. To measure models trained
with `--no-mi`, or after changing the grid, run `python calculate_MI.py --stage <stage>`;
it fills in only the missing cells.

## How the pieces fit

- **Splits.** group_A (class balanced, seed 42) is the victim's training set and the MI
  probe set; group_B is its disjoint complement and is what every negative and positive
  trains on. Both are cached under `Indices/<dataset>/` on first use.
- **MI grid.** For a model, the logits on nested subsets of group_A of size
  `rate × |group_A|` (rates 0.01 … 1.0) are soft-maxed, bucketised into `bins`
  (5 … 200) and I(X;T), I(T;Y) are estimated per cell: 70 cells per model, one row each.
- **Test.** For each case the negatives found on disk form the pool. One seeded shuffle
  (`reference.seed`) puts the first `reference.k` models into the reference set; every
  other negative is evaluated to give the false-positive rate. A positive is tested against
  the reference set of the victim's dataset and *its own* architecture, so a distilled or
  extracted model of another architecture needs that architecture's victim plan and negatives.
  The statistic is the predictive Hotelling T² of (I(X;T), I(T;Y)) with its exact F law.

## Options

All trainers share `--plans`, `--seeds` (`42`, `0,1,2` or `42:122`), `--workers`,
`--data-root`, `--output-root`, `--no-mi`, `--no-skip` (retrain existing cells) and
`--epochs`. Stage specific:
`--strategies` (fine-tune), `--sparsities` (prune), `--methods` and `--rate` (distillation).
`hypothesis_test.py` takes `--config`, `--stages`, `--rates`, `--bins`, `--output-root`.


## Adding a scenario

Copy the two plans of an existing scenario in each stage folder and change
`Scenario_Name`, `Model` / `Teacher_Model` / `Student_Model` / `Substitute`, and the
`Dataset` block. Positive plans name their victim with `Victim_Scenario`, which must equal
the victim plan's `Scenario_Name`. Model folders, MI rows and test cases are derived from
these names, so nothing else needs configuring.

## Notes

- Victims, negatives, fine-tuning and pruning run with cuDNN autotuning (non-deterministic);
  distillation, extraction and every MI measurement run in deterministic mode.
- Batch sizes are fixed in the entry points (128; 256 for distillation).
- An MI table is append-only with a fixed header; a table written by an older version is
  refused rather than mixed.
