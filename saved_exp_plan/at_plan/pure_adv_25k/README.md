# 25k pure-adversarial fine-tuning

Run the top-level `saved_exp_plan/at_plan/*.yaml` plans with `main_at.py`
from the project PyTorch environment. The unified entry point runs the entire
30-epoch group before the 50-epoch group. Both
groups start independently from the original extraction checkpoints.

| Setting | Value |
| --- | --- |
| Models | DFMS Cross100-40C, Knockoff Cross100, Knockoff Same10 (ResNet-18) |
| AT data | 25,000 unique CIFAR-10 training images, includes the prior 10,000 |
| AT seeds | 0, 1, 2 by default |
| Attack | PGD-10, eps=0.031373 (~8/255), random start, default step ~2/255 |
| Mixing | Off: adversarial CE only; adversarial train forward updates BN |
| Optimizer | SGD, lr=0.01, momentum=0.9, weight_decay=0.0005, Nesterov |
| Scheduler | CosineAnnealingLR over 30 or 50 epochs, eta_min=1e-6 |
| Batch size | 128 |

## Commands

```powershell
# Validate the configs, indices, checkpoint paths and output names only.
python .\main_at.py --seeds 0 1 2 --check-only

# Run all 18 cases: 30 epochs first, then 50 epochs.
python -u .\main_at.py --seeds 0 1 2

# Or launch each group separately, in this order.
python -u .\main_at.py --epochs 30 --seeds 0 1 2
python -u .\main_at.py --epochs 50 --seeds 0 1 2

# Run just one seed, or label an independent repetition.
python -u .\main_at.py --epochs 30 --seeds 0 --run-tag v2

# Restrict execution to one exact YAML if other plans are in the same directory.
python -u .\main_at.py --plans .\saved_exp_plan\at_plan\CIFAR10_RES18_Extraction_PGD_25k_30epochs_off.yaml --seeds 0
```

If Python is not on PATH, invoke the configured environment explicitly:

```powershell
& 'C:\Users\louj\python_env\pytorch_env\Scripts\python.exe' -u 'E:\Experiment\main_at.py' --seeds 0 1 2
```

The script resolves project paths relative to its own location. By default it
scans **all top-level YAML files** in `saved_exp_plan/at_plan`; use `--plans` to
select exact files. `--epochs` filters on the YAML's `Optimizer.Epochs`, without
overriding it. There is no hard-coded restriction on dataset size, epsilon,
attack, or mixing: these remain controlled by each YAML.

Default seeds are 0, 1, 2. Existing `SEED_START`/`SEED_END` environment variables
override this default range; explicit `--seeds` takes precedence over both.
Each model case is reseeded so changing
the duration of a previous model does not change this model's starting RNG
stream. DETERMINISTIC=False is retained, so bitwise reproducibility is not promised.
`run_at_25k.py` is now only a compatibility alias selecting the two 25k YAMLs;
it delegates all execution to `main_at.py` and has no separate training logic.

## Outputs and plotting

Names retain the notebook parser's final `_atseed=..._mix=...` fields:

```text
..._PGD_eps=0.031373_steps=10_bn=adv_atn=25000_atepochs=30_run=v1_atseed=0_mix=off
..._PGD_eps=0.031373_steps=10_bn=adv_atn=25000_atepochs=50_run=v1_atseed=0_mix=off
```

CSV files go to `saved_logs/at_evasion/Performance`; checkpoints go to
`saved_models/at_evasion`. Existing tagged CSVs or checkpoint directories cause
an error before training. Use a new `--run-tag` for a fresh repetition; the
entry point does not resume partial training. This integration does not emit
the former standalone launcher's JSON manifests. Data size and epoch budget
are included in both CSV and checkpoint names even when `main_at_pos()` is
called directly without a run tag. If other settings change (e.g. learning
rate), use a new run tag; names are not a complete configuration fingerprint.

After defining the existing notebook plot functions, use:

```python
CASE_CONTAINS = "_atn=25000_atepochs=30_run=v1"
EPSILON = 0.031373
ATTACK_METHOD = "PGD"
STEPS = 10
MIX_ORDER = ("off",)
summary_30 = plot_at_mix_summary(show_table=False)

CASE_CONTAINS = "_atn=25000_atepochs=50_run=v1"
summary_50 = plot_at_mix_summary(show_table=False)
```

Each call produces a separate figure per model. Different epoch budgets and
run tags remain separate overview cases, so seeds are not pooled across them.

The existing checkpoint-selection rule in main_at.py is retained (currently
starts at 20% of the epoch budget). The final checkpoints are epoch_29.pth and
epoch_49.pth. The existing evaluator uses the CIFAR-10 test set every epoch;
these comparisons are exploratory and are not an independent validation-set
selection protocol.
