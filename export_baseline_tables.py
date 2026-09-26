"""Split the four-baseline master into one independent table per baseline.

    python export_baseline_tables.py                       # every complete baseline
    python export_baseline_tables.py --methods IPGuard     # just one
    python export_baseline_tables.py --include-pending     # keep rows with no value yet

Why a per-baseline table at all
-------------------------------
`master.csv` is one wide row per checkpoint with every baseline's columns side
by side. That is the right shape for the driver (one resumable cell grid) and
the wrong shape for analysis: the columns of a baseline that has not run yet are
empty, the metric names all carry a prefix, and plotting one baseline means
remembering which prefix and which status column guard it. One table per
baseline is narrow, self-describing, and can be produced the moment that
baseline finishes -- the others keep running.

Safe to run while the driver is working. `run_four_baselines_posthoc.write_text`
replaces master.csv atomically, so a concurrent read sees either the previous
or the next complete file, never a torn one. Re-run whenever a baseline
finishes; each export is rewritten from scratch.

What each table carries
-----------------------
The identity of the checkpoint, then that baseline's own columns with the
prefix stripped (IPGuard_TR -> TR), plus Status / Seconds / Updated_At / Error.
`Case_ID` is the join key back to master.csv and across baselines.

AT_Eps / AT_Seed / Source / Family are RE-DERIVED here from Model_Dir rather
than copied. Case sets declared with `Model_Path` (which is how the CIFAR-10
trajectory is declared) wrote those four empty until the driver was fixed, and
`merge_rows` will not refresh an existing row, so the master still holds the
blanks for rows written before the fix. Deriving them keeps the exported tables
correct regardless of when a row was written.
"""
import argparse
import csv
import io
import re
from pathlib import Path

from run_four_baselines_posthoc import METHODS, ROOT, read_master, write_text

# These two live HERE, not in the driver, on purpose. The driver's code sha256
# is part of its output directory's config fingerprint, so editing it refuses
# every later run against results produced by the previous version -- which is
# the guard doing its job, but it makes a metadata-only fix cost a full re-run.
# Deriving the metadata at export time instead leaves the driver untouched.


def at_meta_from_name(folder):
    """(source, eps, at_seed) read off an AT folder name; "" for anything absent.

    Both AT namers (main_at.build_at_pos_scenario_name and
    main_at_posthoc's) put the attack block after the source:
        <source>_<attack>_<k=v ...>_..._atseed=<i>_mix=<m>
    so the source is everything before the attack tag and eps / atseed are
    ordinary k=v pairs. A folder that is not an AT run matches nothing and
    keeps its own name as the source.
    """
    leaf = folder.rstrip("/").split("/")[-1]
    eps = re.search(r"_eps=([0-9.]+)", leaf)
    seed = re.search(r"_atseed=(\d+)", leaf)
    attack = re.search(r"_(PGD|FGSM|CW|AutoAttack)_", leaf)
    source = leaf[: attack.start()] if attack else leaf
    return source, (eps.group(1) if eps else ""), (seed.group(1) if seed else "")


def family_of(name):
    """Attack family of a source name.

    Pruning is tested first: its folder carries the recovery strategy too
    ("Prune20-FT-AL", "sparsity=0.2_FT-AL"), so an FT-AL test would claim it --
    which is why the driver's own family_of puts those rows in "other".
    """
    for key, label in (("Prune20", "Pruning"), ("sparsity=", "Pruning"),
                       ("_FT-AL_", "FT-AL"), ("_FT-LL_", "FT-LL"), ("_RT-AL_", "RT-AL"),
                       ("_DKD_", "DKD"), ("_KD_", "KD"), ("Knockoff", "Knockoff")):
        if key in name:
            return label
    return "other"

DEFAULT_MASTER = ROOT / "saved_logs/at_eval/posthoc_c10_traj_4baselines/master.csv"
IDENTITY = ["Case_ID", "Case_Set", "Stage", "Family", "Source", "AT_Eps", "AT_Seed",
            "Epoch", "Model_Dir", "Checkpoint", "Checkpoint_Path", "Checkpoint_SHA256",
            "Suspect_Arch"]
BOOKKEEPING = ["Status", "Seconds", "Updated_At", "Error"]


def metric_columns(rows, method):
    """That baseline's metric columns, prefix stripped, bookkeeping last."""
    prefix = f"{method}_"
    book = {f"{prefix}{b}" for b in BOOKKEEPING}
    metrics = [c for c in rows[0] if c.startswith(prefix) and c not in book]
    return metrics


def export(master_path, method, out_dir, include_pending=False):
    rows = read_master(master_path)
    if not rows:
        raise FileNotFoundError(f"no rows in {master_path}")
    status_col = f"{method}_Status"
    if status_col not in rows[0]:
        raise KeyError(f"{master_path} has no {status_col} column")
    metrics = metric_columns(rows, method)

    kept, skipped = [], 0
    for r in rows:
        if not include_pending and r[status_col] != "complete":
            skipped += 1
            continue
        src, eps, at_seed = at_meta_from_name(r["Model_Dir"])
        out = {c: r.get(c, "") for c in IDENTITY}
        out.update(Source=src or r.get("Source", ""), AT_Eps=eps, AT_Seed=at_seed,
                   Family=family_of(src or r.get("Source", "")))
        out["Baseline"] = method
        for c in metrics:
            out[c[len(method) + 1:]] = r[c]
        for b in BOOKKEEPING:
            out[b] = r.get(f"{method}_{b}", "")
        kept.append(out)

    if not kept:
        print(f"[{method}] nothing complete yet ({skipped} pending); no table written")
        return None
    fields = (["Baseline"] + IDENTITY
              + [c[len(method) + 1:] for c in metrics] + BOOKKEEPING)
    sio = io.StringIO(newline="")
    w = csv.DictWriter(sio, fieldnames=fields)
    w.writeheader()
    w.writerows(kept)
    out_path = Path(out_dir) / f"{method}.csv"
    write_text(out_path, sio.getvalue())
    eps_seen = sorted({r["AT_Eps"] for r in kept if r["AT_Eps"]})
    print(f"[{method}] {len(kept)} rows -> {out_path}"
          + (f"   ({skipped} not complete yet)" if skipped else ""))
    print(f"           metrics: {[c[len(method)+1:] for c in metrics]}")
    print(f"           eps values: {eps_seen or '(none: no AT rows)'}")
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description="One table per baseline from the four-baseline master.")
    p.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    p.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    p.add_argument("--out-dir", type=Path,
                   help="default: <master's directory>/by_baseline")
    p.add_argument("--include-pending", action="store_true",
                   help="keep rows whose Status is not yet 'complete'")
    args = p.parse_args(argv)
    out_dir = args.out_dir or args.master.parent / "by_baseline"
    for method in args.methods:
        export(args.master, method, out_dir, args.include_pending)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
