"""Stage and package DFMS-HL extraction work so it can run on another machine.

A DFMS run is driven entirely by DFMS_PLAN_DIR, so a "job" here is just a
folder of plan yamls. Two groupings are supported:

    cases/<case>/        one cell           (--stage / --pack)
    victims/<VICTIM>/    one victim's row   (--stage-victims / --pack-victim)

`--pack-victim` is the one used to farm the CIFAR-10 3x3 out to three
machines: all three cells of a row share the same victim checkpoint, so one
bundle carries it once and runs 3 cells x SEEDS seeds. Results already on this
machine are copied in, and main_dfms_extraction.py's SKIP_EXISTING guard makes
those runs exit instantly instead of re-deriving them.

    python pack_dfms_case.py --list
    python pack_dfms_case.py --stage-victims
    python pack_dfms_case.py --pack-victim RN18 --out D:/ship --zip
    python pack_dfms_case.py --pack-victim all  --out D:/ship --zip --seeds 3

What a run reads (walked from the import graph and the runtime paths):
  code     main_dfms_extraction.py, util.py and the four package dirs
           AdvAttack/ Model/ Dataset/ KnowledgeDistillation/ -- copied whole,
           because cherry-picking by import once missed base_distiller.py and
           two __init__.py files.
  plan     the yamls in the job folder
  victim   saved_models/vanilla/{CNN_Models,Transformer_Models}/<Model_Name>_42_1.0/
           best_epoch.pth   (which of the two comes from the victim being a dict)
  indices  Indices/CIFAR-100/group_A_25000_seed42.npy and
           group_B_25000_0.0_25000_seed42.npy -- the frozen proxy split. These
           are COPIED, never regenerated on the far machine.
  data     data/cifar-10-batches-py/ and data/cifar-100-python/ (~356 MB);
           --no-data leaves them out for a machine that already has them.
Third-party: torch, torchvision, timm, numpy, pandas, scikit-learn, opencv,
pillow, pyyaml.
"""
import argparse
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
MATRIX = ROOT / "saved_exp_plan/dfms_plan/matrix"
CASES = ROOT / "saved_exp_plan/dfms_plan/cases"
VICTIMS = ROOT / "saved_exp_plan/dfms_plan/victims"
OUT_MODELS = ROOT / "saved_models/extraction_final"
OUT_LOGS = ROOT / "saved_logs/extraction_final/Performance"

CODE_FILES = ["main_dfms_extraction.py", "util.py"]
CODE_DIRS = ["AdvAttack", "Model", "Dataset", "KnowledgeDistillation"]
INDEX_FILES = [
    "Indices/CIFAR-100/group_A_25000_seed42.npy",
    "Indices/CIFAR-100/group_B_25000_0.0_25000_seed42.npy",
]
DATA_DIRS = ["data/cifar-10-batches-py", "data/cifar-100-python"]
GRID = sorted(p.name[:-5] for p in MATRIX.glob("CIFAR10_*_DFMS_C100-40C_*.yaml"))
# plan-filename token -> bundle name
VICTIM_KEY = {"RES18": "RN18", "VGG16": "VGG16", "DEIT": "DeiT"}


def plan_of(case):
    p = MATRIX / (case + ".yaml")
    if not p.is_file():
        raise SystemExit("unknown case %r; run --list" % case)
    return p, yaml.safe_load(p.read_text(encoding="utf-8"))


def victim_key(case):
    return VICTIM_KEY[case.split("_")[1]]


def victim_ckpt(y):
    vm = y["Victim"].get("Model")
    sub = "Transformer_Models" if isinstance(vm, dict) else "CNN_Models"
    return "saved_models/vanilla/%s/%s_42_1.0/best_epoch.pth" % (sub, y["Victim"]["Model_Name"])


def describe(case):
    _, y = plan_of(case)
    vm, sm = y["Victim"].get("Model"), y["Substitute"]["Model"]
    return (y["Scenario_Name"],
            "DeiT" if isinstance(vm, dict) else vm,
            "DeiT" if isinstance(sm, dict) else sm)


def rows_of(vkey):
    return [c for c in GRID if victim_key(c) == vkey]


def done_seeds(scen, seeds):
    return [s for s in seeds
            if (OUT_MODELS / ("%s_%d_1.0" % (scen, s)) / "best_epoch.pth").is_file()]


def cmd_list(seeds):
    print("%-46s %-10s %-10s %-10s %-12s %s"
          % ("case", "victim", "clone", "bundle", "done seeds", "victim ckpt"))
    print("-" * 118)
    for c in GRID:
        scen, v, s = describe(c)
        d = done_seeds(scen, seeds)
        ck = victim_ckpt(plan_of(c)[1])
        print("%-46s %-10s %-10s %-10s %-12s %s"
              % (c, v, s, victim_key(c), (",".join(map(str, d)) or "-"),
                 "ok" if (ROOT / ck).is_file() else "MISSING " + ck))
    print("-" * 118)
    for k in ("RN18", "VGG16", "DeiT"):
        cs = rows_of(k)
        tot = len(cs) * len(seeds)
        got = sum(len(done_seeds(describe(c)[0], seeds)) for c in cs)
        print("   bundle %-6s %d cell(s) x %d seed(s) = %2d run(s), %d already done -> %2d to train"
              % (k, len(cs), len(seeds), tot, got, tot - got))


def _stage(folder, cases):
    folder.mkdir(parents=True, exist_ok=True)
    for c in cases:
        src, dst = MATRIX / (c + ".yaml"), folder / (c + ".yaml")
        if dst.is_file() and dst.read_bytes() == src.read_bytes():
            print("   unchanged  %s" % dst.relative_to(ROOT).as_posix())
        else:
            shutil.copy2(src, dst)
            print("   staged     %s" % dst.relative_to(ROOT).as_posix())


def cmd_stage():
    for c in GRID:
        _stage(CASES / c, [c])


def cmd_stage_victims():
    for k in ("RN18", "VGG16", "DeiT"):
        _stage(VICTIMS / k, rows_of(k))


def copy(src_rel, out, label, optional=False):
    src, dst = ROOT / src_rel, out / src_rel
    if not src.exists():
        if optional:
            return 0
        raise SystemExit("missing %s: %s" % (label, src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        n = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
    else:
        shutil.copy2(src, dst)
        n = dst.stat().st_size
    print("   %-9s %-62s %7.1f MB" % (label, Path(src_rel).as_posix(), n / 1048576))
    return n


def cmd_pack_victim(vkey, out_root, seeds, with_data, make_zip):
    cases = rows_of(vkey)
    if not cases:
        raise SystemExit("no cases for victim %r" % vkey)
    if not (VICTIMS / vkey).is_dir():
        raise SystemExit("victim folder missing; run --stage-victims first")
    name = "DFMS_C10_victim_%s" % vkey
    out = Path(out_root).resolve() / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    print("packing %s  (%d cells x %d seeds)" % (name, len(cases), len(seeds)))
    for c in cases:
        scen, v, s = describe(c)
        print("   %-46s clone %-10s done seeds %s"
              % (c, s, done_seeds(scen, seeds) or "-"))
    print("   into %s\n" % out)

    total = 0
    for f in CODE_FILES:
        total += copy(f, out, "code")
    for d in CODE_DIRS:
        total += copy(d, out, "code")
    total += copy("saved_exp_plan/dfms_plan/victims/" + vkey, out, "plans")
    # all three cells of a row share one victim
    total += copy(victim_ckpt(plan_of(cases[0])[1]), out, "victim")
    for f in INDEX_FILES:
        total += copy(f, out, "indices")

    # results already computed here -> SKIP_EXISTING makes them free on arrival
    n_done = 0
    for c in cases:
        scen = describe(c)[0]
        for sd in done_seeds(scen, seeds):
            folder = "%s_%d_1.0" % (scen, sd)
            total += copy("saved_models/extraction_final/%s/best_epoch.pth" % folder, out, "done")
            for extra in ("best_val_epoch.pth", "summary.json"):
                total += copy("saved_models/extraction_final/%s/%s" % (folder, extra),
                              out, "done", optional=True)
            for kind in ("training_log", "gan_log"):
                total += copy("saved_logs/extraction_final/Performance/%s_%s.csv" % (kind, folder),
                              out, "done", optional=True)
            n_done += 1
    if with_data:
        for d in DATA_DIRS:
            total += copy(d, out, "data")

    lo, hi = min(seeds), max(seeds) + 1
    plan_dir = "./saved_exp_plan/dfms_plan/victims/%s" % vkey
    (out / "run.ps1").write_text(PS1.format(plan_dir=plan_dir, lo=lo, hi=hi), encoding="utf-8")
    # run.sh must keep LF: write_text on Windows translates \n to \r\n, and a
    # shebang ending in \r makes Linux/Git Bash fail with
    #   /usr/bin/env: 'bash\r': No such file or directory
    (out / "run.sh").write_text(SH.format(plan_dir=plan_dir, lo=lo, hi=hi),
                                encoding="utf-8", newline="\n")
    (out / "RUN.md").write_text(README.format(
        name=name, vkey=vkey, plan_dir=plan_dir, lo=lo, hi=hi,
        n_runs=len(cases) * len(seeds), n_done=n_done,
        n_todo=len(cases) * len(seeds) - n_done,
        cells="\n".join("  - `%s`  ->  clone **%s**, scenario `%s`"
                        % (c, describe(c)[2], describe(c)[0]) for c in cases),
        scens="\n".join("    saved_models/extraction_final/%s_<seed>_1.0/best_epoch.pth"
                        % describe(c)[0] for c in cases),
        data=("bundled under data/" if with_data
              else "NOT bundled -- provide data/cifar-10-batches-py and data/cifar-100-python"),
    ), encoding="utf-8")

    print("\n   total %.0f MB in %s" % (total / 1048576, out))
    if make_zip:
        print("   zipping ...")
        z = shutil.make_archive(str(Path(out_root).resolve() / name), "zip",
                                root_dir=str(out.parent), base_dir=name)
        print("   -> %s  (%.0f MB)" % (z, Path(z).stat().st_size / 1048576))
        shutil.rmtree(out)
    return out


PS1 = """# Run every plan in this bundle. Explicit env so nothing leaks in from the shell.
$env:DFMS_PLAN_DIR = "{plan_dir}"
$env:SEED_START    = "{lo}"
$env:SEED_END      = "{hi}"
python main_dfms_extraction.py
"""

SH = """#!/usr/bin/env bash
# Run every plan in this bundle. Explicit env so nothing leaks in from the shell.
set -e
DFMS_PLAN_DIR="{plan_dir}" SEED_START={lo} SEED_END={hi} python main_dfms_extraction.py
"""

README = """# {name}

DFMS-HL data-free extraction, CIFAR-10, victim **{vkey}** -- all three clone
architectures. {n_runs} runs total ({n_done} already computed and bundled, **{n_todo} to train**).

{cells}

## Run

Needs Python 3.11+ with torch (CUDA), torchvision, timm, numpy, pandas,
scikit-learn, opencv-python, pillow, pyyaml. Unzip, then from the bundle root:

    ./run.sh            # bash
    .\\run.ps1           # PowerShell

Both set DFMS_PLAN_DIR / SEED_START / SEED_END explicitly, so a stale SEED_END
left in the shell cannot change how many runs happen. Equivalent one-liner:

    DFMS_PLAN_DIR={plan_dir} SEED_START={lo} SEED_END={hi} python main_dfms_extraction.py

Check the banner: it must say `TOTAL: {n_runs} run(s)`, and the victim / clone
lines must match the cells listed above.

The {n_done} bundled result(s) print `[SKIP] ... best_epoch.pth already exists`
and cost nothing. Interrupted runs resume from `state.json`, so re-running the
same command is always safe.

Budget on one RTX 5090: ~2.1 h per CNN clone, ~2.9 h per DeiT clone.

Datasets: {data}

## Send back

Per finished run, only these (~60 MB each):

{scens}
    ...same folder: best_val_epoch.pth, summary.json
    saved_logs/extraction_final/Performance/training_log_<scenario>_<seed>_1.0.csv
    saved_logs/extraction_final/Performance/gan_log_<scenario>_<seed>_1.0.csv

Drop them into the same relative paths in the main repo. The generator
checkpoints and val_*.pt (~560 MB per run) are resume state only -- leave them.

On the main machine the MI step needs no code change; all nine DFMS plans are
already listed in EXTRACTION_BEST_PLANS:

    python calculate_MI_extraction.py DFMS
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--stage", action="store_true", help="one folder per cell")
    ap.add_argument("--stage-victims", action="store_true", help="one folder per victim row")
    ap.add_argument("--pack", metavar="CASE")
    ap.add_argument("--pack-victim", metavar="RN18|VGG16|DeiT|all")
    ap.add_argument("--out", default="./dfms_bundles")
    ap.add_argument("--seeds", type=int, default=3, help="attacker seeds 0..N-1 (default 3)")
    ap.add_argument("--no-data", action="store_true", help="leave CIFAR-10/100 out (-356 MB)")
    ap.add_argument("--zip", action="store_true", help="zip the bundle and delete the folder")
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    if a.list:
        cmd_list(seeds)
    elif a.stage:
        cmd_stage()
    elif a.stage_victims:
        cmd_stage_victims()
    elif a.pack_victim:
        keys = ["RN18", "VGG16", "DeiT"] if a.pack_victim == "all" else [a.pack_victim]
        for k in keys:
            cmd_pack_victim(k, a.out, seeds, not a.no_data, a.zip)
            print("")
    elif a.pack:
        raise SystemExit("--pack (single cell) is superseded by --pack-victim; "
                         "use --stage + DFMS_PLAN_DIR for one-off local runs")
    else:
        ap.print_help()
