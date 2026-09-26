"""Audit which kd_final cells main_kd would skip, and whether they are COMPLETE.

    python audit_kd_runs.py                                   # both self rows
    python audit_kd_runs.py ./saved_exp_plan/kd_plan_matrix/run/c100_self
    KD_SEEDS=0 python audit_kd_runs.py <folder>               # narrow the grid

Why this exists: main_kd.kd_run_done only tests that best_epoch.pth exists and
is non-empty, and main_kd writes that file on EVERY test-accuracy improvement
starting at epoch 0. A run killed part-way therefore leaves a checkpoint the
guard skips forever, silently filing a half-trained student as finished. This
cross-checks two things the guard does not look at:

    epoch_<Epochs-1>.pth   written only after the final epoch completes
    the training log       row count vs the plan's Epochs

A cell flagged PARTIAL-AND-SKIPPED is fixed by deleting its folder under
saved_models/kd_final (and its log under saved_logs/kd_final/Performance, which
main_kd appends to rather than truncates) and re-running.
"""
import glob, os, sys, yaml
sys.path.insert(0, os.getcwd())
import main_kd as MK

LOGS = "./saved_logs/kd_final/Performance"
FOLDERS = sys.argv[1:] or ["./saved_exp_plan/kd_plan_matrix/run/c100_self",
                           "./saved_exp_plan/kd_plan_matrix/run/c10_self"]
SEEDS = [int(x) for x in os.environ.get("KD_SEEDS", "0,1,2").split(",")]
RATES = [float(x) for x in os.environ.get("KD_RATES", "0.0").split(",")]
print(f"KD_MODEL_ROOT={MK.KD_MODEL_ROOT}  seeds={SEEDS}  rates={RATES}\n")

for folder in FOLDERS:
    print(f"=== {folder} ===")
    print(f"{'cell':<44} {'guard':<7} {'final .pth':<11} {'log ep':<10} verdict")
    print("-" * 92)
    skip = train = suspect = 0
    for p in sorted(glob.glob(os.path.join(folder, "*.yaml"))):
        n_ep = yaml.safe_load(open(p, encoding="utf-8"))["Epochs"]
        scen, methods = MK.kd_plan_methods(p)
        for seed in SEEDS:
            for m in methods:
                for r in RATES:
                    name = MK.kd_scenario_name(scen, m, seed, r)
                    done = MK.kd_run_done(scen, m, seed, r)
                    final = os.path.join(MK.KD_MODEL_ROOT, name, f"epoch_{n_ep - 1}.pth")
                    has_final = os.path.isfile(final) and os.path.getsize(final) > 0
                    log = os.path.join(LOGS, f"training_log_{name}.csv")
                    rows = (sum(1 for _ in open(log, encoding="utf-8")) - 1
                            if os.path.isfile(log) else 0)
                    if done and has_final and rows >= n_ep:
                        v, skip = "complete, will SKIP", skip + 1
                    elif done:
                        v, suspect = ("PARTIAL but the guard SKIPS it "
                                      "-> delete the folder to redo"), suspect + 1
                    else:
                        v, train = "will train", train + 1
                    print(f"{name:<44} {'skip' if done else 'run':<7} "
                          f"{('yes' if has_final else 'no'):<11} {f'{rows}/{n_ep}':<10} {v}")
    print(f"-> {skip} complete, {train} to train, {suspect} PARTIAL-AND-SKIPPED\n")
