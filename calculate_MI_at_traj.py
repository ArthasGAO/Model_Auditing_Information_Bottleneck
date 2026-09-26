"""Per-epoch MI for the post-hoc AT trajectories (saved_models/at_evasion).

    python calculate_MI_at_traj.py                       # every epoch of every cell
    python calculate_MI_at_traj.py --select Prune20      # a subset of the cells
    python calculate_MI_at_traj.py --epochs 0 5 10 29    # a subset of the epochs
    python calculate_MI_at_traj.py --check-only

Nothing about the MI estimate is re-derived here. This is a thin driver over
`calculate_MI_at.main_at_best`, which already measures one AT checkpoint on the
victim's own nested subsets (group_A seed 42, subset seed 42) and writes the
shared identity columns. It gained a checkpoint-filename resolver
(`at_ckpt_filename`) so a `ckpt_kind` may be `epoch_<N>` as well as
`best_clean` / `best_rob`; that is the whole of what per-epoch support needed.
`best_clean` / `best_rob` behave exactly as before.

Why a separate CSV
------------------
`saved_logs/at_final/MI_master_table_at.csv` is the curated at_final table: two
checkpoints per model, the full 70-cell grid. A trajectory is the opposite
shape -- 30 checkpoints per model, one grid cell -- and it reads a different
model tree (at_evasion, where main_at.py writes). Mixing the two into one
single-writer table would make "how many rows should this model have"
unanswerable, so the trajectory gets its own file with the identical header.

Why one grid cell by default
----------------------------
A trajectory needs the operating point, not the grid: 12 cells x 30 epochs is
360 checkpoints, which is 360 rows at (in_size=25000, bins=50) and 25 200 rows
over the full 70-cell grid. The single cell is minutes; the full grid is hours
and answers a question nobody asked of every intermediate epoch. Widen with
--bins / --in-sizes when a specific epoch deserves it; the endpoints are the
usual candidates.

The `epoch` column holds the integer N, so a plot can sort and join on it.
`ckpt_kind` holds `epoch_<N>`, which is what makes the rows filterable against
the `best_clean` / `best_rob` rows in the at_final table.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import calculate_MI_at as M

ROOT = Path(__file__).resolve().parent
TRAJ_MODEL_DIR = ROOT / "saved_models/at_evasion"
TRAJ_MASTER_CSV = ROOT / "saved_logs/at_evasion/MI_master_table_at_traj.csv"
# A glob, not a fixed list, so the same file works in the full repo (two halves:
# FTAL_Prune20 and DKD_Knockoff) and in a bundle that carries only one of them.
TRAJ_PLAN_GLOB = "saved_exp_plan/at_plan/CIFAR10_RES18_PostAT_MixOff_*.yaml"
# run=traj was the first pass, which kept only the last epoch (the per-epoch save
# in main_at.py was outside the loop). run=traj2 is the pass with
# --save-all-epochs. The older rows stay readable: pass --run-tag traj.
TRAJ_RUN_TAG = "traj2"
TRAJ_AT_SEEDS = [0]          # one seed per case
TRAJ_IN_SIZES = [25000]      # the operating point every d in this project uses
TRAJ_BINS = [50]


def main(argv=None):
    p = argparse.ArgumentParser(description="Per-epoch MI for at_evasion trajectories.")
    p.add_argument("--plans", nargs="+", type=Path,
                   default=sorted(ROOT.glob(TRAJ_PLAN_GLOB)))
    p.add_argument("--run-tag", default=TRAJ_RUN_TAG)
    p.add_argument("--seeds", nargs="+", type=int, default=TRAJ_AT_SEEDS,
                   help="the --seeds that were passed to main_at.py")
    p.add_argument("--select", nargs="+",
                   help="keep only scenarios containing one of these substrings")
    p.add_argument("--epochs", nargs="+", type=int,
                   help="measure only these epochs (default: every epoch on disk)")
    p.add_argument("--in-sizes", nargs="+", type=int, default=TRAJ_IN_SIZES)
    p.add_argument("--bins", nargs="+", type=int, default=TRAJ_BINS)
    p.add_argument("--model-dir", type=Path, default=TRAJ_MODEL_DIR)
    p.add_argument("--out", type=Path, default=TRAJ_MASTER_CSV)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args(argv)
    if not args.plans:
        p.error(f"No plans matched {TRAJ_PLAN_GLOB}")

    os.chdir(ROOT)
    jobs, absent = [], []
    for plan in args.plans:
        for scenario in M.expand_at_plan(plan, at_seeds=args.seeds, run_tag=args.run_tag):
            if args.select and not any(s in scenario for s in args.select):
                continue
            folder = args.model_dir / scenario
            if not folder.is_dir():
                absent.append(scenario)
                continue
            kinds = M.at_epoch_kinds(args.model_dir, scenario)
            if args.epochs is not None:
                wanted = {f"epoch_{n}" for n in args.epochs}
                missing = sorted(wanted - set(kinds))
                if missing:
                    raise FileNotFoundError(
                        f"{scenario}: no checkpoint for {missing}")
                kinds = [k for k in kinds if k in wanted]
            if not kinds:
                absent.append(scenario + "   (folder present, no epoch_*.pth)")
                continue
            jobs.append((plan, scenario, kinds))

    print(f"[ENV] model_dir : {args.model_dir}")
    print(f"[ENV] output    : {args.out}")
    print(f"[ENV] run_tag   : {args.run_tag}   seeds {args.seeds}")
    print(f"[ENV] grid      : in_sizes={args.in_sizes}  bins={args.bins}")
    total = sum(len(k) for _, _, k in jobs)
    print(f"TOTAL: {len(jobs)} scenario(s), {total} checkpoint(s) "
          f"-> at most {total * len(args.in_sizes) * len(args.bins)} rows")
    for _, scenario, kinds in jobs:
        span = f"{kinds[0]}..{kinds[-1]}" if len(kinds) > 1 else kinds[0]
        print(f"  {len(kinds):>3} ckpt  {span:<20} {scenario}")
    for scenario in absent:
        print(f"  [ABSENT] {scenario}")
    if not jobs:
        print("Nothing to measure.")
        return 1
    if args.check_only:
        print("Preflight passed. No MI computed.")
        return 0

    written = 0
    for plan, scenario, kinds in jobs:
        written += M.main_at_best(
            scenario, plan,
            in_sizes=args.in_sizes, num_intervals_list=args.bins,
            master_csv_path=args.out, model_dir=args.model_dir,
            ckpt_kinds=kinds,
        )
        print(f"[OK] {scenario}: {written} rows so far")
    print(f"\nappended {written} row(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
