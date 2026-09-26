"""Add the per-law statistic and summary threshold columns to written results.

Two backfills, both derived from data the files already hold:
  * per_model.csv gains `stat_chi2` and `stat_F`.
  * summary.csv and summary_all_configs.csv gain `n_eval_total` and, for each
    reference law, the spread of that cell's p-values: min / p05 / median /
    p95 / max / mean, aggregated from that configuration's per_model.csv. The
    decision rule is p < alpha, so no k-dependent threshold column is needed.
    An earlier revision wrote a T2 spread and `crit_T2_*` thresholds instead;
    those columns are removed when encountered.

Why this is not a re-run
-----------------------
`stat_chi2` and `stat_F` are pure functions of columns the old files already
carry: stat_chi2 is T2 itself, and stat_F is (k-2)/(2(k-1)) * T2 with k taken
from the k_ref column. Nothing is recomputed from models or MI tables, so the
numbers are identical to what a fresh run would write.

Safety
------
Every existing field is copied verbatim as text, so no stored value is
reformatted; only the two new columns are produced. Each file is written to a
temporary file in the same directory and then atomically replaced. A file that
already has the new header is skipped, so the script is idempotent and can be
interrupted and resumed. Before replacing a file the script checks that the
derived statistics reproduce that file's own stored p-values; a mismatch aborts
without touching anything.

run_metadata.json gains a `laws` block and a `backfilled_utc` stamp, because the
recorded `script_sha256` belongs to the run that wrote the original columns, not
to this edit.

Usage (from E:\\Experiment)
    python backfill_law_statistics.py --check      # report only, write nothing
    python backfill_law_statistics.py              # backfill
    python backfill_law_statistics.py --root DIR   # limit to one results tree
    python backfill_law_statistics.py --only summary   # or: per_model
"""
import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist, f as f_dist

BASE = Path(__file__).resolve().parent
DEFAULT_ROOTS = [
    BASE / "saved_logs/vanilla/Hypo_Test_FixedSplits",
    BASE / "saved_logs/vanilla/Hypo_Test_Positives",
]
P = 2                      # dimension of the MI vector
TOLERANCE = 1e-9           # p-values are round-tripped through text
LAWS = {
    "chi2": "p = sf(stat_chi2, df=2), stat_chi2 = T2",
    "F": "p = sf(stat_F, dfn=2, dfd=k-2), stat_F = (k-2)/(2(k-1)) * T2",
}


def law_columns(frame):
    """(stat_chi2 text, stat_F text) for one per_model frame read as strings."""
    t2 = frame["T2"].to_numpy(dtype=float)
    k = frame["k_ref"].to_numpy(dtype=float)
    if not np.isfinite(t2).all():
        raise ValueError("nonfinite T2")
    if not (k > P).all():
        raise ValueError(f"k_ref must exceed p={P}")
    stat_f = t2 * (k - P) / (P * (k - 1.0))
    # stat_chi2 IS T2, so reuse its stored text rather than reformatting it.
    return frame["T2"].to_numpy(), np.array([str(float(v)) for v in stat_f]), t2, k, stat_f


def verify(frame, t2, k, stat_f):
    """The derived statistics must reproduce the p-values already in the file."""
    stored_chi2 = frame["p_chi2"].to_numpy(dtype=float)
    stored_f = frame["p_F"].to_numpy(dtype=float)
    worst_chi2 = float(np.abs(chi2_dist.sf(t2, P) - stored_chi2).max())
    worst_f = float(np.abs(f_dist.sf(stat_f, P, k - P) - stored_f).max())
    if worst_chi2 > TOLERANCE or worst_f > TOLERANCE:
        raise ValueError(
            f"derived statistics do not reproduce the stored p-values "
            f"(chi2 {worst_chi2:.2e}, F {worst_f:.2e}); file left untouched")
    return worst_chi2, worst_f


def reorder(columns):
    """Place each statistic immediately before its own p-value."""
    out = []
    for name in columns:
        if name == "p_chi2":
            out.append("stat_chi2")
        elif name == "p_F":
            out.append("stat_F")
        out.append(name)
    return out


def backfill_file(path, write=True):
    """Returns (status, rows, worst_chi2, worst_f)."""
    with path.open(encoding="utf-8") as stream:
        header = stream.readline().strip().split(",")
    if "stat_chi2" in header and "stat_F" in header:
        return "already-done", 0, 0.0, 0.0
    for required in ("T2", "k_ref", "p_chi2", "p_F"):
        if required not in header:
            raise ValueError(f"{path}: column {required!r} missing; cannot backfill")

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    stat_chi2, stat_f_text, t2, k, stat_f = law_columns(frame)
    worst_chi2, worst_f = verify(frame, t2, k, stat_f)
    if not write:
        return "would-write", len(frame), worst_chi2, worst_f

    frame["stat_chi2"] = stat_chi2
    frame["stat_F"] = stat_f_text
    frame = frame[reorder([c for c in frame.columns
                           if c not in ("stat_chi2", "stat_F")])]
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".csv")
    os.close(handle)
    temporary = Path(temporary)
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return "written", len(frame), worst_chi2, worst_f


def stamp_metadata(path, write=True):
    if not path.is_file():
        return "no-metadata"
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("laws") == LAWS and "backfilled_utc" in meta:
        return "already-done"
    if not write:
        return "would-write"
    meta["laws"] = LAWS
    meta["backfilled_utc"] = datetime.now(timezone.utc).isoformat()
    meta["backfill_note"] = (
        "stat_chi2 and stat_F were added to per_model.csv after the run by "
        "backfill_law_statistics.py; they are derived from the stored T2 and "
        "k_ref, so script_sha256 above is the run that wrote the other columns.")
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return "written"


# One block per reference law, means last: the mean is a poor summary of a
# quantity spanning many orders of magnitude, so it follows the quantiles.
SUMMARY_LAWS = ["chi2", "F"]
SUMMARY_PARTS = ["min", "p05", "median", "p95", "max", "mean"]
SUMMARY_NEW = ["n_eval_total"] + [f"p_{law}_{part}"
                                  for law in SUMMARY_LAWS for part in SUMMARY_PARTS]
# Written by the first revision of this script; dropped on sight.
SUPERSEDED_PREFIXES = ("f_scale", "T2_", "crit_T2_")


def summary_key_columns(columns):
    """The columns identifying one summary row inside a configuration."""
    if "family" in columns:
        return ["family", "group", "k_ref"]      # positives
    return ["scenario", "k_ref"]                 # negatives


def superseded(columns):
    return [c for c in columns if c.startswith(SUPERSEDED_PREFIXES)]


def cell_statistics(per_model, key_columns):
    """{key tuple: {column: text}} aggregated from one per_model.csv."""
    needed = key_columns + [f"p_{law}" for law in SUMMARY_LAWS]
    frame = pd.read_csv(per_model, usecols=needed, dtype=str, keep_default_na=False)
    out = {}
    for key, block in frame.groupby(key_columns, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        stats = {}
        for law in SUMMARY_LAWS:
            values = block[f"p_{law}"].to_numpy(dtype=float)
            stats.setdefault("n_eval_total", str(int(values.size)))
            stats[f"p_{law}_min"] = str(float(values.min()))
            stats[f"p_{law}_p05"] = str(float(np.percentile(values, 5)))
            stats[f"p_{law}_median"] = str(float(np.median(values)))
            stats[f"p_{law}_p95"] = str(float(np.percentile(values, 95)))
            stats[f"p_{law}_max"] = str(float(values.max()))
            stats[f"p_{law}_mean"] = str(float(values.mean()))
        out[key] = stats
    return out


def write_frame(frame, path):
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".csv")
    os.close(handle)
    temporary = Path(temporary)
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def apply_summary(frame, stats, key_columns):
    """Drop superseded columns and write the current ones, in the fixed order."""
    keys = [tuple(row) for row in frame[key_columns].to_numpy().tolist()]
    absent = [k for k in keys if k not in stats]
    if absent:
        raise ValueError(f"no per_model rows for {absent[:3]}")
    frame = frame.drop(columns=superseded(frame.columns))
    for column in SUMMARY_NEW:
        frame[column] = [stats[k][column] for k in keys]
    return frame


def backfill_summary(directory, write=True, collect=None):
    """Patch one configuration's summary.csv, collecting rows for the combined
    file keyed by (in_size_rate, bins) + the summary key."""
    summary = directory / "summary.csv"
    per_model = directory / "per_model.csv"
    if not summary.is_file():
        return "no-summary", 0
    if not per_model.is_file():
        return "no-per-model", 0
    frame = pd.read_csv(summary, dtype=str, keep_default_na=False)
    key_columns = summary_key_columns(frame.columns)
    stats = cell_statistics(per_model, key_columns)
    if collect is not None:
        rate, bins = frame["in_size_rate"].iloc[0], frame["bins"].iloc[0]
        for key, values in stats.items():
            collect[(rate, bins) + key] = values
    current = (all(c in frame.columns for c in SUMMARY_NEW)
               and not superseded(frame.columns))
    if current:
        return "already-done", 0
    if not write:
        return "would-write", len(frame)
    write_frame(apply_summary(frame, stats, key_columns), summary)
    return "written", len(frame)


def backfill_combined(root, collected, write=True):
    """Patch summary_all_configs.csv from the per-configuration statistics."""
    path = root / "summary_all_configs.csv"
    if not path.is_file():
        return "no-combined", 0
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    key_columns = summary_key_columns(frame.columns)
    if (all(c in frame.columns for c in SUMMARY_NEW)
            and not superseded(frame.columns)):
        return "already-done", 0
    if not write:
        return "would-write", len(frame)
    columns = ["in_size_rate", "bins"] + key_columns
    keys = [tuple(row) for row in frame[columns].to_numpy().tolist()]
    absent = [k for k in keys if k not in collected]
    if absent:
        raise ValueError(f"{path}: {len(absent)} row(s) have no configuration "
                         f"statistics, e.g. {absent[0]}")
    frame = frame.drop(columns=superseded(frame.columns))
    for column in SUMMARY_NEW:
        frame[column] = [collected[k][column] for k in keys]
    write_frame(frame, path)
    return "written", len(frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="report what would change and write nothing")
    parser.add_argument("--root", action="append", type=Path,
                        help="a results tree; repeatable. Default: both.")
    parser.add_argument("--only", choices=["per_model", "summary"],
                        help="run just one of the two backfills")
    args = parser.parse_args()
    roots = args.root or DEFAULT_ROOTS
    write = not args.check

    total = {"written": 0, "already-done": 0, "would-write": 0}
    rows = 0
    worst_chi2 = worst_f = 0.0
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            print(f"[SKIP] not a directory: {root}")
            continue
        files = sorted(root.glob("*/per_model.csv"))
        print(f"{root}: {len(files)} configuration(s)")
        if args.only != "summary":
            for path in files:
                status, n, wc, wf = backfill_file(path, write=write)
                total[status] = total.get(status, 0) + 1
                rows += n
                worst_chi2, worst_f = max(worst_chi2, wc), max(worst_f, wf)
                stamp_metadata(path.parent / "run_metadata.json", write=write)
                if status != "already-done":
                    print(f"  per_model  {status:<12} {n:>7} rows  "
                          f"{path.parent.name}", flush=True)
        if args.only != "per_model":
            collected = {}
            for path in files:
                status, n = backfill_summary(path.parent, write=write,
                                             collect=collected)
                total[status] = total.get(status, 0) + 1
                if status != "already-done":
                    print(f"  summary    {status:<12} {n:>7} rows  "
                          f"{path.parent.name}", flush=True)
            status, n = backfill_combined(root, collected, write=write)
            total[status] = total.get(status, 0) + 1
            print(f"  summary_all_configs.csv: {status}, {n} rows")
    print()
    print(f"written      : {total.get('written', 0)}")
    print(f"would write  : {total.get('would-write', 0)}")
    print(f"already done : {total.get('already-done', 0)}")
    print(f"rows touched : {rows}")
    print(f"max deviation from the stored p-values: "
          f"chi2 {worst_chi2:.2e}, F {worst_f:.2e}")
    if args.check:
        print("\n--check: nothing was written.")


if __name__ == "__main__":
    sys.exit(main())
