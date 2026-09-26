"""
Copy finished (round, case) rows of one repeated-DI run into another suite's
output directory, so a case that moves between suites is not evaluated again.

Valid because a row depends only on the seed namespace, the victim, the case,
the round and the protocol settings -- never on the suite or on the other cases
(main_DI_repeat.derive_seed). Checked on real n=1000 data 2026-09-23: the 9 FT-AL
rows of C10_one_stage_at and C10_ft_at are identical (max |diff| 0.0).

Works for both main_DI_repeat.py and main_DI_repeat_paper.py outputs (detected
from run_config.json). The tool
  * refuses when the target run already exists with different identity settings;
  * copies only rows whose Case_ID the target suite's plans declare;
  * for rows already in the target, checks they are identical instead of copying;
  * copies each row's scores npz and its round's regressor file (sha256-checked when
    the target already has one), rewriting Suite / Regressor_Path to the target;
  * records the migration in the target's run_config.json.

Usage (from E:\\Experiment):
  python migrate_di_repeat_rows.py --src saved_logs/di_repeat_paper/C10_one_stage_at_per_round_n1000-1000_m10x100 --dst-suite C10_kd_at --dry-run
  python migrate_di_repeat_rows.py --src saved_logs/di_repeat_paper/C10_one_stage_at_per_round_n1000-1000_m10x100 --dst-suite C10_kd_at
"""
import argparse
import csv
import glob
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from main_DI_eval import _parse_epoch_from_ckpt
from main_DI_repeat import PLAN_ROOT, score_dir_name, sha256_file

STAT_COLUMNS = ("Mean_Private", "Mean_Public", "Delta", "T_Stat", "P_Value", "P_Value_Full",
                "P_Value_M", "Mean_Diff_M", "Regressor_Seed", "Walk_Seed", "M_Seed")


def plan_case_ids(plan_file):
    """Case IDs a plan declares, by the rule of main_DI_eval._iter_suspects_by_paths."""
    cfg = yaml.safe_load(Path(plan_file).read_text(encoding="utf-8"))
    ids = set()
    for key in ("Positive", "Negative Suspect"):
        blocks = cfg.get(key) or []
        for block in (blocks if isinstance(blocks, list) else [blocks]):
            for path in block.get("Model_Path") or []:
                leaf = path.rstrip("/").split("/")[-1]
                if block.get("Checkpoint"):
                    stems = [Path(block["Checkpoint"]).stem]
                elif block.get("State", "best") == "all":
                    stems = [Path(p).stem for p in glob.glob(str(Path("saved_models") / path.lstrip("/") / "epoch_*.pth"))]
                else:
                    stems = ["best_epoch"]
                for stem in stems:
                    epoch = _parse_epoch_from_ckpt(stem)
                    ids.add(f"{leaf}_epoch={epoch}" if epoch is not None else leaf)
    victim = cfg.get("Victim", {}).get("Model_Name")
    if victim:
        ids.add(f"{victim}_42_1.0__victim_self")
    return ids


def default_dst_dir(src_cfg, suite, paper):
    mode, n_train, n_test = src_cfg["regressor_mode"], src_cfg["n_train"], src_cfg["n_test"]
    if paper:
        return Path("./saved_logs/di_repeat_paper") / f"{suite}_{mode}_n{n_train}-{n_test}_m{src_cfg['m']}x{src_cfg['m_reps']}"
    return Path("./saved_logs/di_repeat") / f"{suite}_{mode}_n{n_train}-{n_test}"


def read_csv(path):
    if not path.exists():
        return None, []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def append_rows(path, header, rows):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def same_stats(a, b):
    for col in STAT_COLUMNS:
        if col in a and col in b and a[col] != "" and b[col] != "":
            if not np.isclose(float(a[col]), float(b[col]), rtol=0, atol=0):
                return False, col
    return True, None


def migrate(src, dst_suite, dst_dir=None, plan_dir=None, dry_run=False):
    src = Path(src)
    src_cfg = json.loads((src / "run_config.json").read_text(encoding="utf-8"))
    paper = "m" in src_cfg
    if paper:
        from main_DI_repeat_paper import IDENTITY_KEYS
    else:
        from main_DI_repeat import IDENTITY_KEYS
    plan_dir = Path(plan_dir) if plan_dir else PLAN_ROOT / dst_suite
    plan_files = sorted(Path(p) for p in glob.glob(str(plan_dir / "*.yaml")))
    if not plan_files:
        raise FileNotFoundError(f"No plans in {plan_dir}")
    wanted = set().union(*(plan_case_ids(p) for p in plan_files))
    dst = Path(dst_dir) if dst_dir else default_dst_dir(src_cfg, dst_suite, paper)
    print(f"src {src}  ({'paper' if paper else 'original'} protocol)\ndst {dst}  (suite {dst_suite}, "
          f"{len(wanted)} declared case id(s))")

    dst_cfg_path = dst / "run_config.json"
    if dst_cfg_path.exists():
        dst_cfg = json.loads(dst_cfg_path.read_text(encoding="utf-8"))
        diff = {k: (src_cfg.get(k), dst_cfg.get(k)) for k in IDENTITY_KEYS if src_cfg.get(k) != dst_cfg.get(k)}
        if diff:
            raise ValueError(f"src and dst runs differ in identity settings {diff}; refusing to mix rows.")
    else:
        dst_cfg = {**src_cfg, "suite": dst_suite,
                   "plans": {str(p): sha256_file(p) for p in plan_files},
                   "created": datetime.now().isoformat(timespec="seconds")}

    header, src_rows = read_csv(src / "rounds.csv")
    dst_header, dst_rows = read_csv(dst / "rounds.csv")
    if dst_header is not None and dst_header != header:
        raise ValueError("rounds.csv column layouts differ between src and dst.")
    have = {(r["Round"], r["Case_ID"]): r for r in dst_rows}
    to_copy, identical = [], 0
    for row in src_rows:
        if row["Case_ID"] not in wanted:
            continue
        key = (row["Round"], row["Case_ID"])
        if key in have:
            ok, col = same_stats(row, have[key])
            if not ok:
                raise ValueError(f"{key} exists in dst with a different {col}; not touching dst.")
            identical += 1
            continue
        to_copy.append(row)
    print(f"rows: {len(to_copy)} to copy, {identical} already in dst and identical, "
          f"{sum(r['Case_ID'] not in wanted for r in src_rows)} src rows belong to other suites")

    # Regressor files for every (victim, regressor file) the copied rows use.
    regs = sorted({(r["Victim_Model"], Path(r["Regressor_Path"]).name) for r in to_copy})
    for victim, name in regs:
        s, d = src / "regressors" / victim / name, dst / "regressors" / victim / name
        if not s.is_file():
            raise FileNotFoundError(s)
        if d.is_file() and sha256_file(d) != sha256_file(s):
            raise ValueError(f"{d} exists with different content than {s}.")
    print(f"regressor files: {len(regs)} ({', '.join(f'{v}/{n}' for v, n in regs) or '-'})")
    missing = [r for r in to_copy if not (src / "scores" / score_dir_name(r["Case_ID"]) /
                                          f"round_{int(r['Round']):03d}.npz").is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} copied row(s) have no scores npz in src, e.g. {missing[0]['Case_ID']}")
    if dry_run or not to_copy:
        print("dry run, nothing written." if dry_run else "nothing to copy.")
        return dst

    dst.mkdir(parents=True, exist_ok=True)
    for victim, name in regs:
        d = dst / "regressors" / victim / name
        if not d.is_file():
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / "regressors" / victim / name, d)
    _, src_regs = read_csv(src / "regressors.csv")
    reg_header, dst_regs = read_csv(dst / "regressors.csv")
    have_regs = {(r["Victim_Model"], Path(r["Regressor_Path"]).name) for r in dst_regs}
    new_regs = []
    for r in src_regs:
        k = (r["Victim_Model"], Path(r["Regressor_Path"]).name)
        if k in regs and k not in have_regs:
            new_regs.append({**r, "Suite": dst_suite,
                             "Regressor_Path": str(dst / "regressors" / k[0] / k[1])})
    if new_regs:
        append_rows(dst / "regressors.csv", list(new_regs[0].keys()) if reg_header is None else reg_header, new_regs)
    for r in to_copy:
        rel = Path("scores") / score_dir_name(r["Case_ID"]) / f"round_{int(r['Round']):03d}.npz"
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, dst / rel)
    append_rows(dst / "rounds.csv", header,
                [{**r, "Suite": dst_suite,
                  "Regressor_Path": str(dst / "regressors" / r["Victim_Model"] / Path(r["Regressor_Path"]).name)}
                 for r in to_copy])
    dst_cfg.setdefault("migrated", []).append({
        "from": str(src), "rows": len(to_copy), "rounds": sorted({int(r["Round"]) for r in to_copy}),
        "at": datetime.now().isoformat(timespec="seconds")})
    dst_cfg_path.write_text(json.dumps(dst_cfg, indent=2), encoding="utf-8")
    print(f"copied {len(to_copy)} row(s), {len(new_regs)} regressor row(s) -> {dst}")
    return dst


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy finished repeated-DI rows into another suite's output.")
    parser.add_argument("--src", required=True, help="source run directory (holds run_config.json, rounds.csv)")
    parser.add_argument("--dst-suite", required=True, help=f"target suite (plan folder under {PLAN_ROOT})")
    parser.add_argument("--dst-dir", help="target run directory; default = the entry script's default for that suite")
    parser.add_argument("--plan-dir", help="explicit plan folder of the target suite")
    parser.add_argument("--dry-run", action="store_true")
    cli = parser.parse_args()
    migrate(cli.src, cli.dst_suite, dst_dir=cli.dst_dir, plan_dir=cli.plan_dir, dry_run=cli.dry_run)
