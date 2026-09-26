"""
Negative-pool extraction from the MI CSV.

The CSV stores one row per (model, in_size, bins) combination. A single
model — identified by (seed, rate) — therefore has multiple rows across
different bins/in_size choices. This module collapses the CSV down to
one MI vector per (rate, seed) for a fixed (in_size, bins) and the
chosen In/Out mode, and returns it in a metadata-bearing pool dict that
the downstream split function will consume.
"""
import re

import numpy as np
import pandas as pd

# Column names used in the CSV.  Centralised here in case the schema
# Negative
COL_SCENARIO = "Scenario"
COL_SEED = "seed"
COL_RATE = "rate"
COL_IN_SIZE = "in_size"
COL_BINS = "bins"
COL_IXT_IN = "I(X;T)-In"
COL_ITY_IN = "I(T;Y)-In"
COL_IXT_OUT = "I(X;T)-Out"
COL_ITY_OUT = "I(T;Y)-Out"

# Positive
COL_MODEL    = "model_name"
COL_EPOCH    = "epoch"

RATE_DECIMALS = 1


def extract_victim_mi(csv_path, scenario, in_size, bins, mi_kind="In",
                      victim_rate=1.0, victim_seed=42):
    """
    Extract the victim model's MI vector for Gate 1.
    """
    import pandas as pd
    if mi_kind == "In":
        col_ixt, col_ity = COL_IXT_IN, COL_ITY_IN
    else:
        col_ixt, col_ity = COL_IXT_OUT, COL_ITY_OUT
 
    df = pd.read_csv(csv_path)
    df[COL_RATE] = df[COL_RATE].astype(float).round(RATE_DECIMALS)
    df[COL_SEED] = df[COL_SEED].astype(int)
 
    sel = df[(df[COL_SCENARIO] == scenario) &
             (df[COL_RATE] == round(float(victim_rate), RATE_DECIMALS)) &
             (df[COL_SEED] == int(victim_seed)) &
             (df[COL_IN_SIZE] == in_size) &
             (df[COL_BINS] == bins)]
 
    if len(sel) == 0:
        raise ValueError(
            f"Victim not found: scenario={scenario}, rate={victim_rate}, "
            f"seed={victim_seed}, in_size={in_size}, bins={bins}."
        )
    if len(sel) > 1:
        raise ValueError(
            f"Victim ambiguous ({len(sel)} rows) for scenario={scenario}, "
            f"rate={victim_rate}, seed={victim_seed}, in_size={in_size}, "
            f"bins={bins}. Please deduplicate."
        )
    row = sel.iloc[0]
    return np.array([row[col_ixt], row[col_ity]], dtype=float)


def extract_neg_pool(csv_path,
                     scenario,
                     rate_range,
                     seed_range,
                     in_size,
                     bins,
                     mi_kind="In",
                     return_summary=False):
    """
    Extract a clean per-model MI-vector pool from the negatives CSV.
    """
    # ---- validate inputs ------------------------------------------------
    # Verify requested rates are already on the RATE_DECIMALS grid before
    # we round.  Otherwise a typo like 0.55 would silently snap to 0.6.
    rate_list = []
    for r in rate_range:
        r_f = float(r)
        r_rounded = round(r_f, RATE_DECIMALS)
        if abs(r_f - r_rounded) > 1e-9:
            raise ValueError(
                f"Requested rate {r_f} is not on the {RATE_DECIMALS}-decimal "
                f"grid expected by the CSV.  Nearest grid value is {r_rounded}."
            )
        rate_list.append(r_rounded)
    seed_list = [int(s) for s in seed_range]
 
    if len(rate_list) == 0:
        raise ValueError("rate_range is empty — nothing to extract.")
    if len(seed_list) == 0:
        raise ValueError("seed_range is empty — nothing to extract.")
    if mi_kind not in ("In", "Out"):
        raise ValueError(f"mi_kind must be 'In' or 'Out', got {mi_kind!r}.")
    if not isinstance(scenario, str) or len(scenario) == 0:
        raise ValueError(f"scenario must be a non-empty string, got {scenario!r}.")
 
    if mi_kind == "In":
        col_ixt, col_ity = COL_IXT_IN, COL_ITY_IN
    else:
        col_ixt, col_ity = COL_IXT_OUT, COL_ITY_OUT
 
    # ---- load CSV -------------------------------------------------------
    df = pd.read_csv(csv_path)
 
    required_cols = {COL_SCENARIO, COL_SEED, COL_RATE, COL_IN_SIZE, COL_BINS,
                     col_ixt, col_ity}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"CSV is missing expected columns: {sorted(missing_cols)}. "
            f"Found: {sorted(df.columns)}"
        )
 
    # Filter by scenario first — fail fast if the tag isn't present.
    available_scenarios = set(df[COL_SCENARIO].unique())
    if scenario not in available_scenarios:
        raise ValueError(
            f"scenario={scenario!r} not present in CSV. "
            f"Available scenarios: {sorted(available_scenarios)}."
        )
    df = df[df[COL_SCENARIO] == scenario].copy()
 
    # Normalise rate to RATE_DECIMALS so float-precision noise doesn't
    # break equality matching downstream.
    df[COL_RATE] = df[COL_RATE].astype(float).round(RATE_DECIMALS)
    df[COL_SEED] = df[COL_SEED].astype(int)
 
    # ---- apply (in_size, bins) filter once ------------------------------
    df_filt = df[(df[COL_IN_SIZE] == in_size) & (df[COL_BINS] == bins)]
    if len(df_filt) == 0:
        raise ValueError(
            f"No rows match scenario={scenario!r}, in_size={in_size}, "
            f"bins={bins}.  Within this scenario, available in_size values: "
            f"{sorted(df[COL_IN_SIZE].unique())}; available bins values: "
            f"{sorted(df[COL_BINS].unique())}."
        )
 
    # ---- build the pool, validating each (rate, seed) cell --------------
    pool = {}
    summary = {}
 
    available_rates = set(df_filt[COL_RATE].unique())
    missing_rates = [r for r in rate_list if r not in available_rates]
    if missing_rates:
        raise ValueError(
            f"Requested rates not present in CSV for scenario={scenario!r} "
            f"after in_size/bins filter: {missing_rates}. "
            f"Available: {sorted(available_rates)}."
        )
 
    for rate in rate_list:
        rate_rows = df_filt[df_filt[COL_RATE] == rate]
        pool[rate] = []
 
        for seed in seed_list:
            cell = rate_rows[rate_rows[COL_SEED] == seed]
 
            if len(cell) == 0:
                raise ValueError(
                    f"No row found for scenario={scenario!r}, rate={rate}, "
                    f"seed={seed}, in_size={in_size}, bins={bins}."
                )
            if len(cell) > 1:
                raise ValueError(
                    f"Expected exactly one row for scenario={scenario!r}, "
                    f"rate={rate}, seed={seed}, in_size={in_size}, bins={bins}, "
                    f"but found {len(cell)}.  CSV likely has duplicates — "
                    f"please deduplicate."
                )
 
            row = cell.iloc[0]
            mi_vec = np.array([row[col_ixt], row[col_ity]], dtype=float)
 
            pool[rate].append({"seed": seed, "mi": mi_vec})
 
        summary[rate] = len(pool[rate])
 
    if return_summary:
        return pool, summary
    return pool


def pool_to_array(pool):
    """
    Flatten a pool dict back to a plain (N, 2) MI array and a parallel
    list of (rate, seed) metadata tuples. Convenience helper for quick
    inspection / plotting; the main pipeline keeps the pool structure.
    """
    rows = []
    meta = []
    for rate in sorted(pool.keys()):
        for entry in pool[rate]:
            rows.append(entry["mi"])
            meta.append((rate, entry["seed"]))
    return np.array(rows), meta


import random
import copy


def split_h0_heldout(pool, selection_spec, seed):
    """
    Split a negative pool into H0 fit set and held-out set.
    """
    rng = random.Random(seed)
 
    h0_pool = {}
    heldout_pool = {}
 
    # Iterate in sorted rate order so the rng draws are deterministic
    # given the seed, regardless of how pool's keys were ordered.
    for rate in sorted(pool.keys()):
        entries = pool[rate]
        n_pick = selection_spec.get(rate, 0)
 
        if n_pick == 0:
            # Nothing into H0; everything into held-out.
            heldout_pool[rate] = [copy.deepcopy(e) for e in entries]
            continue
 
        # Pick n_pick entries by index; the rest go to held-out.
        n_total = len(entries)
        picked_idx = set(rng.sample(range(n_total), n_pick))
 
        h0_entries = []
        heldout_entries = []
        for i, e in enumerate(entries):
            target = h0_entries if i in picked_idx else heldout_entries
            target.append(copy.deepcopy(e))
 
        h0_pool[rate] = h0_entries
        if heldout_entries:
            heldout_pool[rate] = heldout_entries
 
    return h0_pool, heldout_pool


def select_subset_from_heldout(pool, selection_spec, seed):
    """
    Thin alias of split_h0_heldout for the case where the input is a
    held-out pool (or any pool) and you want to carve a chosen subset
    out of it. Mechanically identical to split_h0_heldout — same
    spec-driven, seeded, stratified picking — only the return names
    read more naturally in this context.
    """
    return split_h0_heldout(pool, selection_spec, seed)


import numpy as np
from scipy.stats import chi2
from sklearn.covariance import ledoit_wolf


def hotelling_T2_raw_fit(aux_matrix, ddof=1, shrinkage="auto"):
    """
    Fits the Hotelling T2 baseline, calculating both the raw sample
    covariance and an optionally shrunk covariance matrix for comparison.
    """
    X = np.asarray(aux_matrix, dtype=float)
    k, p = X.shape
    mu = X.mean(axis=0)
 
    # 1. Calculate Standard (Raw) Covariance and its diagnostics
    S_raw = np.cov(X, rowvar=False, ddof=ddof)
    evals_raw = np.linalg.eigvalsh(S_raw)
    cond_raw = float(evals_raw.max() / max(evals_raw.min(), 1e-12))
 
    # Initialize final variables (defaults to raw if no shrinkage is applied)
    S_final = S_raw.copy()
    evals_final = evals_raw.copy()
    cond_final = cond_raw
    applied_lambda = 0.0
 
    # 2. Apply Shrinkage if requested
    if shrinkage == "auto":
        # Ledoit-Wolf finds the mathematically optimal lambda
        _, optimal_lambda = ledoit_wolf(X)
        applied_lambda = optimal_lambda
 
    elif isinstance(shrinkage, float) and shrinkage > 0.0:
        # Manual shrinkage
        print("Manual shrinkage applied.")
        applied_lambda = min(max(shrinkage, 0.0), 1.0)
 
    # 3. Compute the Final Shrunk Matrix (if lambda > 0)
    if applied_lambda > 0.0:
        avg_var = np.trace(S_raw) / p
        target = avg_var * np.eye(p)
        S_final = (1.0 - applied_lambda) * S_raw + applied_lambda * target
 
        # Re-calculate diagnostics for the stabilized matrix
        evals_final = np.linalg.eigvalsh(S_final)
        cond_final = float(evals_final.max() / max(evals_final.min(), 1e-12))
 
    # 4. Store EVERYTHING in the info dictionary for later comparison
    info = {
        "mu": mu,
        "k": k,
        "p": p,
        "shrinkage_lambda": applied_lambda,
        # Final (Used) Statistics
        "S": S_final,
        "eigvals": evals_final,
        "cond": cond_final,
        # Raw (Original) Statistics for Comparison
        "S_raw": S_raw,
        "eigvals_raw": evals_raw,
        "cond_raw": cond_raw,
    }
    return mu, S_final, info


def hotelling_T2_raw_score(mu, S, x, k=None, scaling="none"):
    """
    scaling:
      - "none": T2 = md2
      - "paper": T2 = (k/(k+1)) * md2   (Eq.(6)-style single-sample correction)
      - "k": T2 = k * md2               (some codebases do this; not Eq.(6))
    """
    x = np.asarray(x, dtype=float).ravel()
    diff = x - mu
 
    Sinv = np.linalg.inv(S)
    md2 = float(diff @ Sinv @ diff)
 
    if scaling == "none":
        T2 = md2
    elif scaling == "paper":
        if k is None:
            raise ValueError("k must be provided for scaling='paper'")
        T2 = float((k / (k + 1.0)) * md2)
    elif scaling == "k":
        if k is None:
            raise ValueError("k must be provided for scaling='k'")
        T2 = float(k * md2)
    else:
        raise ValueError("scaling must be in {'none','paper','k'}")
 
    p_value = 1.0 - chi2.cdf(T2, df=len(mu))
    return T2, p_value, {"diff": diff, "md2": md2, "scaling": scaling}
 
 
def decisions_from_p(p_value, alphas=(0.10, 0.05, 0.01, 0.001)):
    """
    Reject H0 if p < alpha.
    Returns dict: {alpha: True/False}
    """
    return {a: (p_value < a) for a in alphas}


def passes_gate1(x, victim_mi):
    """
    Gate 1 — necessary condition for "stolen".
 
    A stolen suspect must lie LOWER-RIGHT of the victim in the
    information plane:
        I(X;T)_S > I(X;T)_V   AND   I(T;Y)_S < I(T;Y)_V
    i.e. it carries more input information and less label information
    than the victim. A suspect failing this cannot be stolen, so it is
    short-circuited to negative (p=1) without running Hotelling.
    """
    x = np.asarray(x, float).ravel()
    v = np.asarray(victim_mi, float).ravel()
    return bool((x[0] >= v[0]) and (x[1] <= v[1]))


def evaluate_group(name, samples, mu, S, alphas=(0.10, 0.05, 0.01, 0.001),
                   verbose=False, k=None, scaling="none",
                   victim_mi=None, apply_gate1=False):
    """
    samples: iterable of 2D points like (ix,iy) or array-like shape (2,)
    Returns per-sample records and a structured summary of rejection rates.
 
    Gate 1 (optional): if apply_gate1=True and victim_mi is given, each
    sample first checks the necessary condition (lower-right of victim).
    Failing samples are short-circuited to p=1.0, T2=0.0 (judged
    negative) WITHOUT running Hotelling. Each record gets a "gate1"
    field: "pass" (ran Hotelling), "fail" (short-circuited), or "n/a"
    (gate not applied).
    """
    if apply_gate1 and victim_mi is None:
        raise ValueError("apply_gate1=True requires victim_mi.")
 
    records = []
    for idx, x in enumerate(samples):
        if apply_gate1:
            if passes_gate1(x, victim_mi):
                T2, pval, extra = hotelling_T2_raw_score(mu, S, x, k, scaling)
                gate = "pass"
            else:
                # necessary condition fails -> cannot be stolen -> negative
                T2, pval = 0.0, 1.0
                gate = "fail"
        else:
            T2, pval, extra = hotelling_T2_raw_score(mu, S, x, k, scaling)
            gate = "n/a"
 
        dec = decisions_from_p(pval, alphas=alphas)
        rec = {"group": name, "idx": idx, "x": np.asarray(x, float),
               "T2": T2, "p": pval, "gate1": gate, **dec}
        records.append(rec)
        if verbose:
            print(f"[{name} #{idx}] x={rec['x']}  gate1={gate}  "
                  f"T2={T2:.6f}  p={pval:.6g}  " +
                  " ".join([f"a={a}:{'REJ' if rec[a] else 'OK'}" for a in alphas]))
 
    n = len(records)
 
    # Store counts and rates in dictionaries with the alpha floats as keys
    counts = {}
    rates = {}
 
    for a in alphas:
        rej = sum(1 for r in records if r[a])
        counts[a] = rej
        rates[a] = rej / max(n, 1)
 
    # how many were short-circuited by Gate 1 (if applied)
    n_gate1_fail = sum(1 for r in records if r["gate1"] == "fail")
    n_gate1_pass = sum(1 for r in records if r["gate1"] == "pass")
 
    # Structured summary
    summary = {
        "group": name,
        "n": n,
        "counts": counts,
        "rates": rates,
        "n_gate1_fail": n_gate1_fail,
        "n_gate1_pass": n_gate1_pass,
    }
 
    return records, summary


def evaluate_pool(name, pool, mu, S, alphas=(0.10, 0.05, 0.01, 0.001),
                  verbose=False, k=None, scaling="none",
                  victim_mi=None, apply_gate1=False):
    """
    Pool-aware wrapper around evaluate_group.
 
    Accepts a pool dict — {rate: [{"seed": int, "mi": np.ndarray(2,)}, ...]} —
    instead of a raw array, scores every entry, and carries the
    (rate, seed) provenance into each per-sample record.
 
    Gate 1 (optional): pass apply_gate1=True and victim_mi to enforce the
    necessary condition (suspect lower-right of victim) before Hotelling;
    failing samples are judged negative (p=1) without scoring. Records
    gain a "gate1" field and the per-rate breakdown gains gate1 counts.
 
    Returns
    -------
    records : list[dict]
        Same per-sample fields as evaluate_group, plus "rate", "seed".
    summary : dict
        Overall counts/rates, n_gate1_fail/pass, plus "rate_breakdown"
        with per-rate counts/rates and per-rate gate1 tallies.
    """
    # Flatten the pool in a deterministic order (sorted rate, then the
    # pool's own seed order) and keep parallel provenance.
    flat = []  # list of (rate, seed, mi)
    for rate in sorted(pool.keys()):
        for entry in pool[rate]:
            flat.append((rate, entry["seed"], np.asarray(entry["mi"], float)))
 
    samples = [mi for (_r, _s, mi) in flat]
 
    # Reuse the core scorer for the overall pass.
    records, summary = evaluate_group(
        name=name, samples=samples, mu=mu, S=S,
        alphas=alphas, verbose=verbose, k=k, scaling=scaling,
        victim_mi=victim_mi, apply_gate1=apply_gate1,
    )
 
    # Attach provenance to each record.
    for rec, (rate, seed, _mi) in zip(records, flat):
        rec["rate"] = rate
        rec["seed"] = seed
 
    # Per-rate breakdown of rejection statistics.
    rate_breakdown = {}
    for rate in sorted(pool.keys()):
        rate_recs = [r for r in records if r["rate"] == rate]
        n_r = len(rate_recs)
        counts_r = {a: sum(1 for r in rate_recs if r[a]) for a in alphas}
        rates_r = {a: counts_r[a] / max(n_r, 1) for a in alphas}
        rate_breakdown[rate] = {
            "n": n_r, "counts": counts_r, "rates": rates_r,
            "n_gate1_fail": sum(1 for r in rate_recs if r["gate1"] == "fail"),
            "n_gate1_pass": sum(1 for r in rate_recs if r["gate1"] == "pass"),
        }
 
    summary["rate_breakdown"] = rate_breakdown
 
    return records, summary


def h0_pool_to_matrix(h0_pool):
    """
    Flatten an h0_pool dict into an MI matrix and parallel provenance.

    Iterates rates in sorted order, and seeds within each rate in the
    order they appear in the pool (which split_h0_heldout keeps sorted
    by seed). The row order of the returned matrix is therefore
    deterministic and matches `meta` element-for-element.
    """
    rows = []
    meta = []
    for rate in sorted(h0_pool.keys()):
        for entry in h0_pool[rate]:
            rows.append(np.asarray(entry["mi"], dtype=float))
            meta.append((rate, entry["seed"]))
    aux_matrix = np.vstack(rows)
    return aux_matrix, meta


def build_h0(h0_pool, ddof=1, shrinkage="auto"):
    """
    Build the H0 Hotelling statistic from an h0_pool.

    Flattens the pool, fits (mu, S) via hotelling_T2_raw_fit, and
    augments the returned fit_info with the provenance of the rows so
    you can later see exactly which (rate, seed) models defined this H0
    and how many came from each rate.
    """
    aux_matrix, meta = h0_pool_to_matrix(h0_pool)

    mu, S, fit_info = hotelling_T2_raw_fit(
        aux_matrix, ddof=ddof, shrinkage=shrinkage
    )

    # Attach provenance so the fitted statistic is always traceable.
    fit_info["meta"] = meta
    rate_breakdown = {}
    for rate, _seed in meta:
        rate_breakdown[rate] = rate_breakdown.get(rate, 0) + 1
    fit_info["rate_breakdown"] = rate_breakdown

    return mu, S, fit_info


import os
import datetime
import numpy as np


def _fmt_vec(v, nd=6):
    return "[" + ", ".join(f"{x:.{nd}f}" for x in np.asarray(v).ravel()) + "]"
 
 
def _fmt_mat(M, nd=6, indent="         "):
    M = np.asarray(M)
    lines = []
    for row in M:
        lines.append(indent + "[" + ", ".join(f"{x:.{nd}f}" for x in row) + "]")
    return "\n".join(lines)
 
 
def log_split_section(
    log_path,
    scenario,
    split_seed,
    selection_spec,
    h0_pool,
    fit_info,
    heldout_records,
    heldout_summary,
    shrinkage,
    in_size=None,
    bins=None,
    mi_kind="In",
    alphas=(0.05, 0.01),
    split_index=None,
    float_nd=6,
):
    """
    Append one split section to the txt log.
    """
    sep = "=" * 64
    lines = []
 
    # ---- header ---------------------------------------------------------
    header = "SPLIT" if split_index is None else f"SPLIT #{split_index}"
    lines.append(sep)
    lines.append(header)
    lines.append(sep)
 
    # ---- reproducibility ------------------------------------------------
    lines.append("[Reproducibility]")
    lines.append(f"  timestamp       : {datetime.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"  scenario        : {scenario}")
    lines.append(f"  split_seed      : {split_seed}")
    spec_str = "{" + ", ".join(f"{r}:{n}" for r, n in sorted(selection_spec.items())) + "}"
    lines.append(f"  selection_spec  : {spec_str}")
    lines.append(f"  shrinkage       : {shrinkage}")
    lines.append(f"  in_size / bins  : {in_size} / {bins}")
    lines.append(f"  mi_kind         : {mi_kind}")
    lines.append("")
 
    # ---- H0 composition -------------------------------------------------
    k = fit_info.get("k", sum(len(v) for v in h0_pool.values()))
    lines.append(f"[H0 composition]   k = {k}")
    for rate in sorted(h0_pool.keys()):
        seeds = [e["seed"] for e in h0_pool[rate]]
        lines.append(f"  rate={rate:<4} : seeds {seeds}")
    lines.append("")
 
    # ---- H0 statistic ---------------------------------------------------
    lines.append("[H0 statistic]")
    if "mu" in fit_info:
        lines.append(f"  mu     = {_fmt_vec(fit_info['mu'], float_nd)}")
    if "S" in fit_info:
        lines.append("  S      =")
        lines.append(_fmt_mat(fit_info["S"], float_nd))
    if "shrinkage_lambda" in fit_info:
        lines.append(f"  lambda = {fit_info['shrinkage_lambda']:.{float_nd}f}")
    if "cond" in fit_info:
        lines.append(f"  cond   = {fit_info['cond']:.4f}")
    lines.append("")
 
    # ---- held-out evaluation -------------------------------------------
    n = heldout_summary.get("n", len(heldout_records))
    lines.append(f"[Held-out evaluation]   n = {n}")
 
    rates_summary = heldout_summary.get("rates", {})
    counts_summary = heldout_summary.get("counts", {})
    for a in alphas:
        rate_val = rates_summary.get(a, float("nan"))
        cnt = counts_summary.get(a, None)
        if cnt is None:
            cnt = sum(1 for r in heldout_records if r.get(a, False))
        lines.append(f"  FPR @{a} = {rate_val:.2%} ({cnt}/{n})")
 
    # per-rate FPR — list EVERY rate present in held-out (including
    # those with 0 FP), each with its denominator, so the log is fully
    # self-describing for downstream per-rate aggregation. Under Option A
    # this covers all rates in the pool, including rates NOT selected
    # into H0 (which appear in held-out with their full seed count).
    rate_bd = heldout_summary.get("rate_breakdown", {})
    if rate_bd:
        # which rates were actually selected into H0 (for the * marker)
        selected_rates = set(selection_spec.keys())
        for a in alphas:
            lines.append(f"  per-rate FPR @{a}:")
            for rate in sorted(rate_bd.keys()):
                bd = rate_bd[rate]
                r_rate = bd["rates"].get(a, 0.0)
                r_cnt = bd["counts"].get(a, 0)
                r_n = bd["n"]
                # mark rates NOT used to build H0 (held-out only)
                mark = "" if rate in selected_rates else " *"
                lines.append(
                    f"    rate={rate:<4} : {r_rate:.0%} ({r_cnt}/{r_n}){mark}"
                )
        lines.append("    (* = rate not selected into H0)")
 
    # explicit FP cases — list the exact (rate, seed) that were rejected
    lines.append("")
    lines.append("  False-positive cases (rejected held-out negatives):")
    for a in alphas:
        fp_recs = [r for r in heldout_records if r.get(a, False)]
        if not fp_recs:
            lines.append(f"    @{a}: none")
        else:
            lines.append(f"    @{a}: {len(fp_recs)} case(s)")
            # sort for stable, readable output
            fp_recs_sorted = sorted(
                fp_recs, key=lambda r: (r.get("rate", 0), r.get("seed", 0))
            )
            for r in fp_recs_sorted:
                lines.append(
                    f"        rate={r.get('rate')}, seed={r.get('seed')}, "
                    f"x={_fmt_vec(r['x'], float_nd)}, "
                    f"T2={r['T2']:.{float_nd}f}, p={float(r['p']):.{float_nd}f}"
                )
    lines.append(sep)
    lines.append("")
    lines.append("")
 
    # ---- append to file -------------------------------------------------
    with open(log_path, "a") as f:
        f.write("\n".join(lines))
 
    return log_path


def _discover_rates(csv_path, scenario, in_size, bins):
    """
    Return the sorted list of all rates present in the CSV for the given
    scenario under the (in_size, bins) filter. Used as the default
    rate_range so the pool covers every available rate (Option A).
    """
    df = pd.read_csv(csv_path)
    df = df[df[COL_SCENARIO] == scenario]
    df = df[(df[COL_IN_SIZE] == in_size) & (df[COL_BINS] == bins)]
    rates = sorted({round(float(r), RATE_DECIMALS) for r in df[COL_RATE].unique()})
    return rates


def run_split_sweep_neg(
        csv_path,
        scenario,
        selection_spec,
        split_seeds,
        log_path,
        rate_range=None,
        seed_range=range(42, 52),
        in_size=10000,
        bins=100,
        mi_kind="In",
        alphas=(0.05, 0.01),
        shrinkage_modes=(None, "auto"),
        scaling="paper",
        ddof=1,
        fresh_log=True,
        verbose=True,
        apply_gate1=False,
        victim_scenario=None,
        victim_rate=1.0,
        victim_seed=42,
        suspect_csv_path=None,
        suspect_scenario=None,
        suspect_rate_range=None,
        suspect_seed_range=range(42, 52),
):
    """
    Run one fixed split logic across many split seeds, logging each.

    Gate 1 (optional): apply_gate1 + victim_scenario enforce the
    necessary condition before Hotelling. Cross-architecture (optional):
    suspect_scenario draws suspects from a DIFFERENT scenario's full
    negative pool (H0 from `scenario`, suspects from `suspect_scenario`).
    Both default off -> original same-architecture behaviour.
    """
    if rate_range is None:
        rate_range = _discover_rates(csv_path, scenario, in_size, bins)
        print(rate_range)

    # 1) Extract the H0 pool ONCE — identical for every split seed.
    pool, pool_summary = extract_neg_pool(
        csv_path=csv_path,
        scenario=scenario,
        rate_range=rate_range,
        seed_range=seed_range,
        in_size=in_size,
        bins=bins,
        mi_kind=mi_kind,
        return_summary=True,
    )

    if verbose:
        total = sum(pool_summary.values())
        print(f"Extracted pool: scenario={scenario}, "
              f"{len(pool_summary)} rates, {total} models total.")
        print(f"Fixed selection_spec: "
              f"{ {r: selection_spec[r] for r in sorted(selection_spec)} }")
        print(f"Sweeping {len(list(split_seeds))} split seeds "
              f"x {len(shrinkage_modes)} shrinkage modes.\n")
    split_seeds = list(split_seeds)

    # Gate 1 setup: extract victim MI once if requested. Victim defaults
    # to the H0 scenario's (victim_rate, victim_seed) anchor.
    victim_mi = None
    if apply_gate1:
        vscen = victim_scenario if victim_scenario is not None else scenario
        victim_mi = extract_victim_mi(
            csv_path=csv_path, scenario=vscen,
            in_size=in_size, bins=bins, mi_kind=mi_kind,
            victim_rate=victim_rate, victim_seed=victim_seed,
        )
        if verbose:
            print(f"Gate 1 ON. Victim ({vscen}, rate={victim_rate}, "
                  f"seed={victim_seed}) MI = {victim_mi}")

    # Cross-architecture suspect pool (optional). Built once; suspects
    # come from a DIFFERENT scenario's full negative pool. When None,
    # suspects are the held-out split (original behaviour).
    cross_suspect_pool = None
    if suspect_scenario is not None:
        s_csv = suspect_csv_path if suspect_csv_path is not None else csv_path
        s_rates = suspect_rate_range
        if s_rates is None:
            s_rates = _discover_rates(s_csv, suspect_scenario, in_size, bins)
        cross_suspect_pool = extract_neg_pool(
            csv_path=s_csv,
            scenario=suspect_scenario,
            rate_range=s_rates,
            seed_range=suspect_seed_range,
            in_size=in_size,
            bins=bins,
            mi_kind=mi_kind,
        )
        if verbose:
            n_susp = sum(len(v) for v in cross_suspect_pool.values())
            print(f"Cross-arch suspects: scenario={suspect_scenario}, "
                  f"{len(cross_suspect_pool)} rates, {n_susp} models "
                  f"(evaluated against H0 from {scenario}).")

    if fresh_log and os.path.exists(log_path):
        os.remove(log_path)

    results = []
    section_counter = 0

    for split_seed in split_seeds:
        h0_pool, heldout_pool = split_h0_heldout(
            pool, selection_spec, seed=split_seed
        )

        for shrinkage in shrinkage_modes:
            section_counter += 1

            mu, S, fit_info = build_h0(
                h0_pool, ddof=ddof, shrinkage=shrinkage
            )

            # Suspects: cross-arch pool if given, else the held-out split.
            suspect_pool = (cross_suspect_pool
                            if cross_suspect_pool is not None
                            else heldout_pool)
            suspect_name = ("cross_suspect_pool"
                            if cross_suspect_pool is not None
                            else "heldout_pool")
            recs, summ = evaluate_pool(
                name=suspect_name,
                pool=suspect_pool,
                mu=mu, S=S,
                alphas=alphas,
                k=fit_info["k"],
                scaling=scaling,
                victim_mi=victim_mi,
                apply_gate1=apply_gate1,
            )

            log_split_section(
                log_path=log_path,
                scenario=scenario,
                split_seed=split_seed,
                selection_spec=selection_spec,
                h0_pool=h0_pool,
                fit_info=fit_info,
                heldout_records=recs,
                heldout_summary=summ,
                shrinkage=shrinkage,
                in_size=in_size,
                bins=bins,
                mi_kind=mi_kind,
                alphas=alphas,
                split_index=f"{section_counter}  (split_seed={split_seed}, "
                            f"shrinkage={shrinkage})",
            )

            row = {
                "split_seed": split_seed,
                "shrinkage": "raw" if shrinkage is None else str(shrinkage),
                "k": fit_info["k"],
                "cond": fit_info.get("cond"),
                "lambda": fit_info.get("shrinkage_lambda"),
                "n_heldout": summ["n"],
            }
            for a in alphas:
                row[f"fpr@{a}"] = summ["rates"][a]
                row[f"nfp@{a}"] = summ["counts"][a]
            results.append(row)

            if verbose:
                fpr_str = ", ".join(
                    f"FPR@{a}={summ['rates'][a]:.1%}({summ['counts'][a]}/{summ['n']})"
                    for a in alphas
                )
                cond_str = (f"cond={fit_info['cond']:.1f}"
                            if fit_info.get("cond") is not None else "cond=NA")
                print(f"  seed={split_seed:<6} "
                      f"shrink={row['shrinkage']:<5} "
                      f"{fpr_str}  {cond_str}")

    if verbose:
        print(f"\nWrote {section_counter} sections to {log_path}")

    return results


''' Positive Group'''


def _normalize_epochs(epochs, available):
    """
    Normalize the flexible `epochs` argument into a sorted list of ints.
    """
    available_sorted = sorted(set(int(e) for e in available))

    if epochs is None:
        return available_sorted

    # Single scalar int (note: bool is an int subclass — reject it).
    if isinstance(epochs, (int, np.integer)) and not isinstance(epochs, bool):
        wanted = [int(epochs)]
    else:
        # Assume iterable of ints.
        try:
            wanted = [int(e) for e in epochs]
        except TypeError:
            raise ValueError(
                f"epochs must be None, an int, or an iterable of ints; "
                f"got {epochs!r}."
            )

    # Sort ascending and de-duplicate so trajectories are always in
    # epoch (time) order regardless of how the user listed them.
    wanted = sorted(set(wanted))
    return wanted


def extract_pos_trajectory(csv_path,
                           model_name,
                           in_size,
                           bins,
                           mi_kind="In",
                           epochs=None):
    """
    Extract one positive case's MI trajectory across epochs.
    """
    if mi_kind not in ("In", "Out"):
        raise ValueError(f"mi_kind must be 'In' or 'Out', got {mi_kind!r}.")
    if not isinstance(model_name, str) or len(model_name) == 0:
        raise ValueError(f"model_name must be a non-empty string, got {model_name!r}.")
 
    if mi_kind == "In":
        col_ixt, col_ity = COL_IXT_IN, COL_ITY_IN
    else:
        col_ixt, col_ity = COL_IXT_OUT, COL_ITY_OUT
 
    df = pd.read_csv(csv_path)
 
    required_cols = {COL_MODEL, COL_EPOCH, COL_IN_SIZE, COL_BINS,
                     col_ixt, col_ity}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV missing expected columns: {sorted(missing)}. "
            f"Found: {sorted(df.columns)}"
        )
 
    # Filter to this case.
    df_case = df[df[COL_MODEL] == model_name]
    if len(df_case) == 0:
        # Help the user spot near-misses (e.g. trailing spaces, typos).
        sample = df[COL_MODEL].dropna().unique()[:5]
        raise ValueError(
            f"model_name not found (exact match): {model_name!r}. "
            f"Example model_names in CSV: {list(sample)}"
        )
 
    # Apply (in_size, bins) filter.
    df_case = df_case[(df_case[COL_IN_SIZE] == in_size) &
                      (df_case[COL_BINS] == bins)]
    if len(df_case) == 0:
        raise ValueError(
            f"No rows for model_name={model_name!r} with "
            f"in_size={in_size}, bins={bins}. For this case, available "
            f"in_size: {sorted(df[df[COL_MODEL]==model_name][COL_IN_SIZE].unique())}; "
            f"available bins: {sorted(df[df[COL_MODEL]==model_name][COL_BINS].unique())}."
        )
 
    df_case = df_case.copy()
    df_case[COL_EPOCH] = df_case[COL_EPOCH].astype(int)
 
    # Decide which epochs to take (flexible: None / int / iterable).
    available_epochs = sorted(df_case[COL_EPOCH].unique())
    epochs = _normalize_epochs(epochs, available_epochs)
 
    missing_epochs = [e for e in epochs if e not in set(available_epochs)]
    if missing_epochs:
        raise ValueError(
            f"Requested epochs missing for model_name={model_name!r} "
            f"(in_size={in_size}, bins={bins}): {missing_epochs}. "
            f"Available epochs: {available_epochs}."
        )
 
    # Build the epoch-ordered trajectory, validating one row per epoch.
    trajectory = []
    for ep in epochs:
        cell = df_case[df_case[COL_EPOCH] == ep]
        if len(cell) == 0:
            raise ValueError(
                f"No row for model_name={model_name!r}, epoch={ep}, "
                f"in_size={in_size}, bins={bins}."
            )
        if len(cell) > 1:
            raise ValueError(
                f"Expected one row for model_name={model_name!r}, epoch={ep}, "
                f"in_size={in_size}, bins={bins}, but found {len(cell)}. "
                f"CSV likely has duplicates — please deduplicate."
            )
        row = cell.iloc[0]
        mi_vec = np.array([row[col_ixt], row[col_ity]], dtype=float)
        trajectory.append({"epoch": ep, "mi": mi_vec})
 
    return {
        "model_name": model_name,
        "in_size": in_size,
        "bins": bins,
        "mi_kind": mi_kind,
        "trajectory": trajectory,
        "epochs": [t["epoch"] for t in trajectory],
    }


def _fmt(x, nd=6):
    try:
        return f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def log_positive_section(
        log_path,
        scenario,
        model_name,
        selection_spec,
        split_seeds,
        epoch_results,
        shrinkage,
        in_size=None,
        bins=None,
        mi_kind="In",
        alphas=(0.05, 0.01),
        section_label=None,
        float_nd=6,
):
    """
    Append one positive-case section to the txt log.
    """
    sep = "=" * 72
    lines = []

    header = "POSITIVE CASE" if section_label is None else f"POSITIVE CASE  [{section_label}]"
    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    # ---- reproducibility ----
    seeds = list(split_seeds)
    if seeds:
        seed_summary = f"{min(seeds)}..{max(seeds)} (n={len(seeds)})"
    else:
        seed_summary = "(none)"

    lines.append("[Reproducibility]")
    lines.append(f"  timestamp       : {datetime.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"  scenario        : {scenario}")
    lines.append(f"  model_name      : {model_name}")
    spec_str = "{" + ", ".join(f"{r}:{n}" for r, n in sorted(selection_spec.items())) + "}"
    lines.append(f"  selection_spec  : {spec_str}")
    lines.append(f"  H0 ensemble     : split_seeds {seed_summary}")
    lines.append(f"  shrinkage       : {shrinkage}")
    lines.append(f"  in_size / bins  : {in_size} / {bins}")
    lines.append(f"  mi_kind         : {mi_kind}")
    lines.append("")

    # ---- per-epoch trajectory (aggregated across H0 ensemble) ----
    n_h0 = epoch_results[0]["n_h0"] if epoch_results else 0
    lines.append(f"[Epoch trajectory]   (aggregated across {n_h0} H0 splits)")
    lines.append("  Each row: one epoch, statistics taken ACROSS the H0 ensemble.")
    lines.append("")

    # Column header
    head = (f"  {'epoch':>5}  {'I(X;T)':>9} {'I(T;Y)':>9}  "
            f"{'T2_mean':>9} {'T2_std':>9}  {'p_mean':>9} {'p_std':>9}")
    for a in alphas:
        head += f"  {'TPR@' + str(a):>9}"
    lines.append(head)
    lines.append("  " + "-" * (len(head) - 2))

    for er in epoch_results:
        ixt, ity = er["mi"][0], er["mi"][1]
        row = (f"  {er['epoch']:>5}  {_fmt(ixt, float_nd):>9} {_fmt(ity, float_nd):>9}  "
               f"{_fmt(er['T2_mean'], float_nd):>9} {_fmt(er['T2_std'], float_nd):>9}  "
               f"{_fmt(er['p_mean'], float_nd):>9} {_fmt(er['p_std'], float_nd):>9}")
        for a in alphas:
            row += f"  {_fmt(er['tpr'].get(a, float('nan')), 4):>9}"
        lines.append(row)

    lines.append("")
    lines.append(sep)
    lines.append("")
    lines.append("")

    with open(log_path, "a") as f:
        f.write("\n".join(lines))

    return log_path


def evaluate_positive_case(
    neg_csv_path,
    pos_csv_path,
    neg_scenario,
    model_name,
    selection_spec,
    split_seeds=range(1000, 1050),
    log_path=None,
    pos_scenario=None,
    neg_rate_range=None,
    neg_seed_range=range(42, 52),
    in_size=25000,
    bins=50,
    mi_kind="In",
    epochs=None,
    alphas=(0.05, 0.01),
    shrinkage=None,
    scaling="paper",
    ddof=1,
    section_label=None,
    fresh_log=False,
    verbose=True,
    apply_gate1=False,
    victim_scenario=None,
    victim_rate=1.0,
    victim_seed=42,
):
    """
    Score one positive case's epoch trajectory against the H0 ensemble.
    """
    # pos_scenario is informational only; default it for the log header.
    if pos_scenario is None:
        pos_scenario = neg_scenario
 
    # 1) Build the negative pool ONCE (filtered by neg_scenario).
    if neg_rate_range is None:
        # discover all rates present for the scenario under (in_size,bins)
        _df = pd.read_csv(neg_csv_path)
        _df = _df[_df[COL_SCENARIO] == neg_scenario]
        _df = _df[(_df[COL_IN_SIZE] == in_size) & (_df[COL_BINS] == bins)]
        neg_rate_range = sorted({round(float(r), RATE_DECIMALS)
                                 for r in _df[COL_RATE].unique()})
 
    neg_pool = extract_neg_pool(
        csv_path=neg_csv_path,
        scenario=neg_scenario,
        rate_range=neg_rate_range,
        seed_range=neg_seed_range,
        in_size=in_size,
        bins=bins,
        mi_kind=mi_kind,
    )
 
    # 2) Extract the positive case's epoch trajectory.
    case = extract_pos_trajectory(
        csv_path=pos_csv_path,
        model_name=model_name,
        in_size=in_size,
        bins=bins,
        mi_kind=mi_kind,
        epochs=epochs,
    )
    trajectory = case["trajectory"]   # [{"epoch", "mi"}, ...] ascending
 
    split_seeds = list(split_seeds)
 
    # Gate 1 setup (optional): victim = (victim_rate, victim_seed) of the
    # victim scenario, which defaults to neg_scenario (the anchor whose
    # H0 manifold we test against). Same (in_size, bins, mi_kind) as H0.
    victim_mi = None
    if apply_gate1:
        vscen = victim_scenario if victim_scenario is not None else neg_scenario
        victim_mi = extract_victim_mi(
            csv_path=neg_csv_path, scenario=vscen,
            in_size=in_size, bins=bins, mi_kind=mi_kind,
            victim_rate=victim_rate, victim_seed=victim_seed,
        )
        if verbose:
            print(f"Gate 1 ON. Victim ({vscen}, rate={victim_rate}, "
                  f"seed={victim_seed}) MI = {victim_mi}")
 
    # 3) Build the H0 ensemble: one (mu, Sigma) per split_seed.
    #    Score every epoch against every H0; collect per-epoch across H0.
    #    matrix[epoch_idx][h0_idx] = (T2, p)
    n_epochs = len(trajectory)
    T2_grid = [[] for _ in range(n_epochs)]
    p_grid = [[] for _ in range(n_epochs)]
 
    for s in split_seeds:
        h0_pool, _heldout = split_h0_heldout(neg_pool, selection_spec, seed=s)
        mu, S, fit_info = build_h0(h0_pool, ddof=ddof, shrinkage=shrinkage)
        k = fit_info["k"]
 
        for i, step in enumerate(trajectory):
            # Gate 1: a suspect not lower-right of the victim cannot be
            # stolen -> p=1 (negative), skip Hotelling. Note this is
            # per-epoch: early epochs may sit near the victim and fail
            # Gate 1, only entering Hotelling once they escape.
            if apply_gate1 and not passes_gate1(step["mi"], victim_mi):
                T2, pval = 0.0, 1.0
            else:
                T2, pval, _extra = hotelling_T2_raw_score(
                    mu, S, step["mi"], k=k, scaling=scaling
                )
            T2_grid[i].append(T2)
            p_grid[i].append(pval)
 
    # 4) Aggregate PER EPOCH across the H0 ensemble.
    epoch_results = []
    n_h0 = len(split_seeds)
    for i, step in enumerate(trajectory):
        T2_arr = np.array(T2_grid[i], dtype=float)
        p_arr = np.array(p_grid[i], dtype=float)
 
        tpr = {}
        for a in alphas:
            tpr[a] = float(np.mean(p_arr < a))   # fraction of H0 that reject
 
        er = {
            "epoch": step["epoch"],
            "mi": step["mi"],
            "T2_mean": float(T2_arr.mean()),
            "T2_std": float(T2_arr.std(ddof=1)) if n_h0 > 1 else 0.0,
            "p_mean": float(p_arr.mean()),
            "p_std": float(p_arr.std(ddof=1)) if n_h0 > 1 else 0.0,
            "tpr": tpr,
            "n_h0": n_h0,
            "T2_all": T2_arr.tolist(),
            "p_all": p_arr.tolist(),
        }
        epoch_results.append(er)
 
        if verbose:
            tpr_str = ", ".join(f"TPR@{a}={tpr[a]:.0%}" for a in alphas)
            print(f"  epoch={er['epoch']:>3}  "
                  f"T2={er['T2_mean']:.3f}+/-{er['T2_std']:.3f}  "
                  f"p={er['p_mean']:.4f}+/-{er['p_std']:.4f}  {tpr_str}")
 
    # 5) Log.
    if log_path is not None:
        if fresh_log:
            import os
            if os.path.exists(log_path):
                os.remove(log_path)
        log_positive_section(
            log_path=log_path,
            scenario=pos_scenario,
            model_name=model_name,
            selection_spec=selection_spec,
            split_seeds=split_seeds,
            epoch_results=epoch_results,
            shrinkage=shrinkage,
            in_size=in_size,
            bins=bins,
            mi_kind=mi_kind,
            alphas=alphas,
            section_label=section_label,
        )
 
    return epoch_results


# ftseed is the trailing "_ftseed=<int>" of the model_name.
_RE_FTSEED = re.compile(r"_ftseed=(?P<v>\d+)\s*$")


def case_of(model_name):
    """Return (case_name_without_ftseed, ftseed_int_or_None)."""
    m = _RE_FTSEED.search(model_name)
    if not m:
        return model_name, None
    ftseed = int(m.group("v"))
    case = model_name[: m.start()]  # strip the "_ftseed=N" suffix
    return case, ftseed


def discover_cases(pos_csv_path, scenario, in_size=None, bins=None):
    """
    Scan the positive CSV for a scenario and group model_names by case.

    Returns
    -------
    cases : dict[str, list[tuple[int|None, str]]]
        {case_name: [(ftseed, full_model_name), ...]}  sorted by ftseed.
    """
    df = pd.read_csv(pos_csv_path)
    df = df[df[COL_SCENARIO] == scenario]
    # in_size / bins filtering is optional here (we just need the set of
    # model_names present); the per-case extraction will filter properly.
    names = sorted(df[COL_MODEL].dropna().unique())

    cases = {}
    for name in names:
        case, ftseed = case_of(name)
        cases.setdefault(case, []).append((ftseed, name))

    for case in cases:
        cases[case].sort(key=lambda t: (t[0] is None, t[0]))  # ftseed asc
    return cases


def _safe_filename(s):
    """Make a case string safe-ish for a filename (it already is, but
    guard against path separators / equals just in case)."""
    return s.replace("/", "_").replace("\\", "_")



''' Pool-0.0 negative test '''
# ===========================================================================
# Pool-0.0 negative test: 50 rate-0.0 models per scenario, k / (50-k) splits
#
# Reference (H0) = k models drawn at random from the pool; suspects = the
# remaining 50-k. All suspects are true negatives, so every rejection is a
# false positive. Each suspect gets TWO p-values from the same statistic
#     T2 = k/(k+1) * (x - mu)^T S^-1 (x - mu)
#   chi2 : T2 ~ chi2(p)                         (existing path, large-k limit)
#   F    : T2 ~ p(k-1)/(k-p) * F(p, k-p)        (exact small-sample law)
# and after every reference fit the reference set itself is checked against
# the bivariate-normal H0 it is supposed to follow (Mardia skewness /
# kurtosis, Shapiro-Wilk marginals, leave-one-out T2 against the exact law).
# The suspect T2 values are then checked against both laws (KS test,
# empirical FPR vs nominal alpha), which is the two-sided calibration view.
#
# Everything below is additive: split_h0_heldout, build_h0,
# h0_pool_to_matrix, passes_gate1 and extract_victim_mi are reused as-is.
# ===========================================================================
import csv
from scipy.stats import f as f_dist, norm, shapiro, kstest

POOL0_CSV      = "./saved_logs/vanilla/MI_master_table_neg_pool0.csv"
# Victim (seed 42, rate 1.0) MI on the pool-0.0 grid, from calculate_MI_victim.py.
# The old default "./saved_logs/vanilla/MI_master_table.csv" only has the old
# in_size grid {1000, 5000, ..., 25000} and raises "Victim not found" elsewhere.
POOL0_VICTIM_CSV = "./saved_logs/vanilla/MI_master_table_victim.csv"
POOL0_LOG_ROOT = "./saved_logs/vanilla/Hypo_Test_Pool0"
POOL0_RATE     = 0.0
POOL0_P        = 2          # dimension of the MI vector [I(X;T), I(T;Y)]


# ---------------------------------------------------------------------------
# Pool extraction
# ---------------------------------------------------------------------------

def extract_pool0(csv_path, scenario, in_size, bins, mi_kind="In",
                  rate=POOL0_RATE, seed_range=None, expected_n=None):
    """
    Pool dict {rate: [{"seed", "mi"}, ...]} for ONE overlap rate.

    Seeds are discovered from the CSV (for this scenario / rate / in_size /
    bins) unless seed_range is given. expected_n, when set, makes an
    incomplete pool an error instead of a silently smaller experiment.
    Delegates the per-cell validation to extract_neg_pool.
    """
    rate = round(float(rate), RATE_DECIMALS)
    if seed_range is None:
        df = pd.read_csv(csv_path)
        df = df[df[COL_SCENARIO] == scenario].copy()
        df[COL_RATE] = df[COL_RATE].astype(float).round(RATE_DECIMALS)
        df = df[(df[COL_RATE] == rate) &
                (df[COL_IN_SIZE] == in_size) & (df[COL_BINS] == bins)]
        seed_range = sorted(int(s) for s in df[COL_SEED].unique())
    seed_range = [int(s) for s in seed_range]
    if expected_n is not None and len(seed_range) != expected_n:
        raise ValueError(
            f"{scenario}: pool at rate={rate}, in_size={in_size}, bins={bins} "
            f"has {len(seed_range)} seeds, expected {expected_n}. "
            f"Seeds present: {seed_range}"
        )
    return extract_neg_pool(
        csv_path=csv_path, scenario=scenario, rate_range=[rate],
        seed_range=seed_range, in_size=in_size, bins=bins, mi_kind=mi_kind,
    )


# ---------------------------------------------------------------------------
# Exact Hotelling law and scoring
# ---------------------------------------------------------------------------

def hotelling_T2_cdf(t, m, p=POOL0_P):
    """
    CDF of T2 = m/(m+1) * md2 for a NEW point scored against a reference of
    m points (sample mean, sample covariance with ddof=1) under H0:
        T2 ~ p(m-1)/(m-p) * F(p, m-p).
    """
    if m <= p:
        raise ValueError(f"need m > p for the Hotelling law, got m={m}, p={p}")
    t = np.asarray(t, dtype=float)
    return f_dist.cdf(t * (m - p) / (p * (m - 1.0)), p, m - p)


def hotelling_T2_sf(t, m, p=POOL0_P):
    """Upper tail (p-value) of the exact law in hotelling_T2_cdf."""
    if m <= p:
        raise ValueError(f"need m > p for the Hotelling law, got m={m}, p={p}")
    t = np.asarray(t, dtype=float)
    return f_dist.sf(t * (m - p) / (p * (m - 1.0)), p, m - p)


def _md2(x, mu, S):
    d = np.asarray(x, dtype=float).ravel() - np.asarray(mu, dtype=float).ravel()
    return float(d @ np.linalg.solve(S, d))


def score_pool0_suspects(suspect_pool, mu, S, k, alphas=(0.05, 0.01),
                         victim_mi=None, apply_gate1=False, p=POOL0_P):
    """
    Score every suspect with T2 = k/(k+1)*md2 and both p-values.

    Returns one record per suspect: rate, seed, x, gate1, md2, T2, p_chi2,
    p_F, and decision flags "chi2@a" / "F@a" for each alpha. Gate 1 is
    optional and, when it fails, short-circuits to p=1 like evaluate_group.
    """
    if apply_gate1 and victim_mi is None:
        raise ValueError("apply_gate1=True requires victim_mi.")
    records = []
    for rate in sorted(suspect_pool.keys()):
        for e in suspect_pool[rate]:
            x = np.asarray(e["mi"], dtype=float)
            gate = "n/a"
            if apply_gate1:
                gate = "pass" if passes_gate1(x, victim_mi) else "fail"
            if gate == "fail":
                md2 = T2 = 0.0
                p_chi2 = p_F = 1.0
            else:
                md2 = _md2(x, mu, S)
                T2 = k / (k + 1.0) * md2
                p_chi2 = float(chi2.sf(T2, p))
                p_F = float(hotelling_T2_sf(T2, k, p))
            rec = {"rate": rate, "seed": e["seed"], "x": x, "gate1": gate,
                   "md2": md2, "T2": T2, "p_chi2": p_chi2, "p_F": p_F}
            for a in alphas:
                rec[f"chi2@{a}"] = bool(p_chi2 < a)
                rec[f"F@{a}"] = bool(p_F < a)
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Reference-side diagnostics: does the H0 set look bivariate normal?
# ---------------------------------------------------------------------------

def mardia_test(X):
    """
    Mardia's multivariate skewness and kurtosis tests.

    Skewness: n*b1p/6 (with Mardia's small-sample factor) ~ chi2(p(p+1)(p+2)/6).
    Kurtosis: (b2p - p(p+2)(n-1)/(n+1)) / sqrt(8p(p+2)/n) ~ N(0,1), two-sided.
    Low power at n = 10..25; treat as a screen, not a verdict.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    S_ml = Xc.T @ Xc / n
    D = Xc @ np.linalg.solve(S_ml, Xc.T)        # n x n generalised inner products
    b1p = float((D ** 3).sum() / n ** 2)
    b2p = float((np.diag(D) ** 2).sum() / n)

    df_skew = p * (p + 1) * (p + 2) / 6.0
    small_n = ((p + 1) * (n + 1) * (n + 3)) / (n * ((n + 1) * (p + 1) - 6.0))
    skew_stat = n * b1p / 6.0 * small_n
    skew_p = float(chi2.sf(skew_stat, df_skew))

    kurt_mean = p * (p + 2) * (n - 1.0) / (n + 1.0)
    kurt_sd = np.sqrt(8.0 * p * (p + 2) / n)
    kurt_z = float((b2p - kurt_mean) / kurt_sd)
    kurt_p = float(2.0 * norm.sf(abs(kurt_z)))
    return {"b1p": b1p, "skew_stat": skew_stat, "skew_df": df_skew, "skew_p": skew_p,
            "b2p": b2p, "kurt_z": kurt_z, "kurt_p": kurt_p}


def loo_reference_T2(X, ddof=1):
    """
    Leave-one-out T2 of every reference point against the other n-1:
        T2_i = (n-1)/n * md2(x_i; mean_-i, S_-i)
    Under H0 each T2_i follows the exact law with m = n-1 reference points
    (hotelling_T2_cdf(., n-1)); the n values are only weakly dependent.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    out = np.empty(n)
    for i in range(n):
        Xi = np.delete(X, i, axis=0)
        mu_i = Xi.mean(axis=0)
        S_i = np.cov(Xi, rowvar=False, ddof=ddof)
        out[i] = (n - 1.0) / n * _md2(X[i], mu_i, S_i)
    return out


def reference_diagnostics(aux_matrix, alphas=(0.05, 0.01), ddof=1, p=POOL0_P):
    """
    Bundle of H0-fit diagnostics on the reference matrix (k x p):
      mardia        : skewness / kurtosis tests
      shapiro_p     : Shapiro-Wilk p per marginal
      loo_T2        : leave-one-out T2 values
      loo_ks_p_F    : KS p of loo_T2 vs the exact law (m = k-1)
      loo_ks_p_chi2 : KS p of loo_T2 vs chi2(p)
      loo_rej_F / loo_rej_chi2 : #reference points that would be rejected
                                 at each alpha under each law
    """
    X = np.asarray(aux_matrix, dtype=float)
    k = X.shape[0]
    if k < p + 2:
        raise ValueError(f"reference too small for LOO diagnostics: k={k}, need >= {p + 2}")
    out = {"k": k, "mardia": mardia_test(X),
           "shapiro_p": [float(shapiro(X[:, j]).pvalue) for j in range(X.shape[1])]}
    loo = loo_reference_T2(X, ddof=ddof)
    out["loo_T2"] = loo
    out["loo_ks_p_F"] = float(kstest(loo, lambda t: hotelling_T2_cdf(t, k - 1, p)).pvalue)
    out["loo_ks_p_chi2"] = float(kstest(loo, lambda t: chi2.cdf(t, p)).pvalue)
    pF = hotelling_T2_sf(loo, k - 1, p)
    pc = chi2.sf(loo, p)
    out["loo_rej_F"] = {a: int((pF < a).sum()) for a in alphas}
    out["loo_rej_chi2"] = {a: int((pc < a).sum()) for a in alphas}
    return out


# ---------------------------------------------------------------------------
# Suspect-side calibration: do the negatives' T2 follow the assumed law?
# ---------------------------------------------------------------------------

def suspect_calibration(records, k, alphas=(0.05, 0.01), p=POOL0_P):
    """
    Empirical FPR under both laws, KS of the suspect T2 values against
    chi2(p) and against the exact law with m = k, and the mean p-value
    (0.5 expected for a calibrated law). Suspects are scored against the
    same (mu, S), so the KS p-values are approximate.
    """
    n = len(records)
    scored = [r for r in records if r["gate1"] != "fail"]
    out = {"n": n, "n_scored": len(scored)}
    for a in alphas:
        for law in ("chi2", "F"):
            nfp = sum(1 for r in records if r[f"{law}@{a}"])
            out[f"nfp_{law}@{a}"] = nfp
            out[f"fpr_{law}@{a}"] = nfp / max(n, 1)
    if len(scored) >= 2:
        T2 = np.array([r["T2"] for r in scored], dtype=float)
        out["ks_p_chi2"] = float(kstest(T2, lambda t: chi2.cdf(t, p)).pvalue)
        out["ks_p_F"] = float(kstest(T2, lambda t: hotelling_T2_cdf(t, k, p)).pvalue)
        out["mean_p_chi2"] = float(np.mean([r["p_chi2"] for r in scored]))
        out["mean_p_F"] = float(np.mean([r["p_F"] for r in scored]))
    else:
        out.update({"ks_p_chi2": float("nan"), "ks_p_F": float("nan"),
                    "mean_p_chi2": float("nan"), "mean_p_F": float("nan")})
    return out


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_pool0_split_section(log_path, scenario, csv_path, split_seed, k_ref,
                            h0_pool, fit_info, diag, records, cal, shrinkage,
                            in_size, bins, mi_kind, rate, alphas,
                            split_index=None, float_nd=6, apply_gate1=False):
    """Append one split section (reference fit, diagnostics, suspects) to the txt log."""
    sep = "=" * 72
    n_susp = cal["n"]
    L = []
    L.append(sep)
    L.append(f"SPLIT #{split_index}  (split_seed={split_seed}, k_ref={k_ref}, n_susp={n_susp})")
    L.append(sep)
    L.append("[Reproducibility]")
    L.append(f"  timestamp       : {datetime.datetime.now().isoformat(timespec='seconds')}")
    L.append(f"  scenario        : {scenario}")
    L.append(f"  pool csv        : {csv_path}")
    L.append(f"  split_seed      : {split_seed}")
    L.append(f"  k_ref / n_susp  : {k_ref} / {n_susp}")
    L.append(f"  shrinkage       : {shrinkage}")
    L.append(f"  in_size / bins  : {in_size} / {bins}")
    L.append(f"  mi_kind / rate  : {mi_kind} / {rate}")
    L.append(f"  gate1           : {'on' if apply_gate1 else 'off'}")
    L.append("")

    k = fit_info["k"]
    L.append(f"[Reference composition]   k = {k}")
    for r in sorted(h0_pool.keys()):
        L.append(f"  rate={r:<4} : seeds {[e['seed'] for e in h0_pool[r]]}")
    L.append("")

    L.append("[H0 statistic]")
    L.append(f"  mu     = {_fmt_vec(fit_info['mu'], float_nd)}")
    L.append("  S      =")
    L.append(_fmt_mat(fit_info["S"], float_nd))
    L.append(f"  lambda = {fit_info.get('shrinkage_lambda', 0.0):.{float_nd}f}")
    L.append(f"  cond   = {fit_info['cond']:.4f}")
    L.append("")

    m = diag["mardia"]
    L.append("[Reference diagnostics]   (is the reference set consistent with a bivariate-normal H0?)")
    L.append(f"  Mardia skewness : b1p={m['b1p']:.4f}  stat={m['skew_stat']:.4f} ~ chi2({m['skew_df']:.0f})  p={m['skew_p']:.4f}")
    L.append(f"  Mardia kurtosis : b2p={m['b2p']:.4f}  z={m['kurt_z']:+.4f}               p={m['kurt_p']:.4f}")
    L.append(f"  Shapiro-Wilk    : I(X;T) p={diag['shapiro_p'][0]:.4f}   I(T;Y) p={diag['shapiro_p'][1]:.4f}")
    L.append(f"  LOO T2 of the {k} reference points (each vs the other {k - 1}):")
    L.append(f"     KS vs exact T2 law (m={k - 1}) : p={diag['loo_ks_p_F']:.4f}")
    L.append(f"     KS vs chi2(2)                 : p={diag['loo_ks_p_chi2']:.4f}")
    for a in alphas:
        L.append(f"     rejected @{a}: F {diag['loo_rej_F'][a]}/{k}   chi2 {diag['loo_rej_chi2'][a]}/{k}")
    L.append("")

    L.append(f"[Suspect evaluation]   n = {n_susp}   (all true negatives -> rejections are false positives)")
    if apply_gate1:
        n_fail = sum(1 for r in records if r["gate1"] == "fail")
        L.append(f"  gate1 fail (judged negative without scoring): {n_fail}/{n_susp}")
    for a in alphas:
        L.append(f"  FPR @{a:<5}: chi2 = {cal[f'fpr_chi2@{a}']:.2%} ({cal[f'nfp_chi2@{a}']}/{n_susp})   "
                 f"F = {cal[f'fpr_F@{a}']:.2%} ({cal[f'nfp_F@{a}']}/{n_susp})")
    L.append(f"  KS of suspect T2 : vs chi2(2) p={cal['ks_p_chi2']:.4f}   vs exact T2 law (m={k}) p={cal['ks_p_F']:.4f}")
    L.append(f"  mean p-value     : chi2 {cal['mean_p_chi2']:.4f}   F {cal['mean_p_F']:.4f}   (0.5 expected if calibrated)")
    L.append("")
    a0 = alphas[0]
    fp = [r for r in records if r[f"chi2@{a0}"] or r[f"F@{a0}"]]
    L.append(f"  False-positive cases @{a0} (rejected by either law):")
    if not fp:
        L.append("    none")
    for r in sorted(fp, key=lambda r: r["seed"]):
        L.append(f"    seed={r['seed']:<3} x={_fmt_vec(r['x'], float_nd)}  T2={r['T2']:.{float_nd}f}  "
                 f"p_chi2={r['p_chi2']:.{float_nd}f} [{'REJ' if r[f'chi2@{a0}'] else 'ok '}]  "
                 f"p_F={r['p_F']:.{float_nd}f} [{'REJ' if r[f'F@{a0}'] else 'ok '}]")
    L.append(sep)
    L.append("")
    L.append("")
    with open(log_path, "a") as fh:
        fh.write("\n".join(L))
    return log_path


def log_pool0_aggregate_section(log_path, agg, seed_tally, alphas, top_n=10):
    """Append the across-splits aggregate section to the txt log."""
    sep = "=" * 72
    L = [sep,
         f"AGGREGATE over {agg['n_splits']} splits   "
         f"({agg['scenario']}, k_ref={agg['k_ref']}, n_susp={agg['n_susp']})",
         sep]
    for a in alphas:
        L.append(f"  FPR @{a:<5}: chi2 mean={agg[f'mean_fpr_chi2@{a}']:.2%} sd={agg[f'std_fpr_chi2@{a}']:.2%} "
                 f"pooled={agg[f'pooled_fpr_chi2@{a}']:.2%}   |   "
                 f"F mean={agg[f'mean_fpr_F@{a}']:.2%} sd={agg[f'std_fpr_F@{a}']:.2%} "
                 f"pooled={agg[f'pooled_fpr_F@{a}']:.2%}")
    L.append(f"  mean cond(S)  : {agg['mean_cond']:.2f}")
    L.append("")
    L.append("  Reference diagnostics, fraction of splits rejecting normality/consistency at 0.05:")
    L.append(f"     Mardia skew {agg['frac_mardia_skew_rej']:.0%}   Mardia kurt {agg['frac_mardia_kurt_rej']:.0%}   "
             f"Shapiro (either marginal) {agg['frac_shapiro_rej']:.0%}")
    L.append(f"     LOO-KS vs exact law {agg['frac_loo_ks_F_rej']:.0%}   LOO-KS vs chi2 {agg['frac_loo_ks_chi2_rej']:.0%}")
    L.append("  Suspect calibration, fraction of splits with KS p<0.05:")
    L.append(f"     vs chi2(2) {agg['frac_susp_ks_chi2_rej']:.0%} (median p {agg['median_susp_ks_p_chi2']:.3f})   "
             f"vs exact law {agg['frac_susp_ks_F_rej']:.0%} (median p {agg['median_susp_ks_p_F']:.3f})")
    L.append(f"     mean p-value across splits: chi2 {agg['mean_p_chi2']:.3f}   F {agg['mean_p_F']:.3f}")
    L.append("")
    a0 = alphas[0]
    L.append(f"  Most-rejected suspects @{a0} (rejections / times drawn as suspect):")
    rows = sorted(seed_tally.items(),
                  key=lambda kv: (-kv[1]["rej_F"], -kv[1]["rej_chi2"], kv[0]))[:top_n]
    for seed, t in rows:
        if t["rej_F"] == 0 and t["rej_chi2"] == 0:
            break
        L.append(f"     seed={seed:<3} F {t['rej_F']}/{t['n_susp']}   chi2 {t['rej_chi2']}/{t['n_susp']}")
    L.append(sep)
    L.append("")
    with open(log_path, "a") as fh:
        fh.write("\n".join(L))
    return log_path


# ---------------------------------------------------------------------------
# Aggregation and the sweep driver
# ---------------------------------------------------------------------------

def aggregate_pool0_results(scenario, k_ref, results, alphas=(0.05, 0.01)):
    """One summary dict per (scenario, k_ref) from the per-split result rows."""
    n = len(results)
    agg = {"scenario": scenario, "k_ref": k_ref,
           "n_susp": results[0]["n_susp"] if results else 0, "n_splits": n,
           "mean_cond": float(np.mean([r["cond"] for r in results])) if n else float("nan")}
    for a in alphas:
        for law in ("chi2", "F"):
            v = np.array([r[f"fpr_{law}@{a}"] for r in results], dtype=float)
            agg[f"mean_fpr_{law}@{a}"] = float(v.mean()) if n else float("nan")
            agg[f"std_fpr_{law}@{a}"] = float(v.std(ddof=1)) if n > 1 else 0.0
            tot_fp = sum(r[f"nfp_{law}@{a}"] for r in results)
            tot_n = sum(r["n_susp"] for r in results)
            agg[f"pooled_fpr_{law}@{a}"] = tot_fp / max(tot_n, 1)

    def frac(key, thr=0.05):
        return float(np.mean([r[key] < thr for r in results])) if n else float("nan")

    agg["frac_mardia_skew_rej"] = frac("mardia_skew_p")
    agg["frac_mardia_kurt_rej"] = frac("mardia_kurt_p")
    agg["frac_shapiro_rej"] = (float(np.mean([min(r["shapiro_p_ixt"], r["shapiro_p_ity"]) < 0.05
                                              for r in results])) if n else float("nan"))
    agg["frac_loo_ks_F_rej"] = frac("loo_ks_p_F")
    agg["frac_loo_ks_chi2_rej"] = frac("loo_ks_p_chi2")
    agg["frac_susp_ks_chi2_rej"] = frac("susp_ks_p_chi2")
    agg["frac_susp_ks_F_rej"] = frac("susp_ks_p_F")
    agg["median_susp_ks_p_chi2"] = float(np.median([r["susp_ks_p_chi2"] for r in results])) if n else float("nan")
    agg["median_susp_ks_p_F"] = float(np.median([r["susp_ks_p_F"] for r in results])) if n else float("nan")
    agg["mean_p_chi2"] = float(np.mean([r["mean_p_chi2"] for r in results])) if n else float("nan")
    agg["mean_p_F"] = float(np.mean([r["mean_p_F"] for r in results])) if n else float("nan")
    return agg


def write_dict_rows_csv(rows, path):
    """Write a list of flat dicts to CSV (union of keys, first-seen order)."""
    if not rows:
        return path
    cols = []
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def run_pool0_split_sweep(
        csv_path,
        scenario,
        k_ref,
        split_seeds,
        log_path,
        per_split_csv=None,
        in_size=25000,
        bins=50,
        mi_kind="In",
        rate=POOL0_RATE,
        seed_range=None,
        expected_pool_size=None,
        alphas=(0.05, 0.01),
        shrinkage=None,
        ddof=1,
        fresh_log=True,
        verbose=True,
        apply_gate1=False,
        victim_csv_path=POOL0_VICTIM_CSV,   # was "./saved_logs/vanilla/MI_master_table.csv" (old grid)
        victim_scenario=None,
        victim_rate=1.0,
        victim_seed=42,
):
    """
    k_ref reference / (pool - k_ref) suspect sweep over many split seeds.

    Per split: draw k_ref models into the reference (split_h0_heldout with
    spec {rate: k_ref}), fit (mu, S) with build_h0, run the reference
    diagnostics, score the remaining models under both laws, run the
    suspect-side calibration, log a section. Then an aggregate section and
    (optionally) a per-split CSV. Returns (results, aggregate).

    The exact F law assumes the plain sample covariance; with shrinkage it
    is only approximate (the chi2 path is unaffected either way).
    """
    p = POOL0_P
    pool = extract_pool0(csv_path, scenario, in_size, bins, mi_kind=mi_kind,
                         rate=rate, seed_range=seed_range, expected_n=expected_pool_size)
    rate = round(float(rate), RATE_DECIMALS)
    n_pool = len(pool[rate])
    if k_ref < p + 2 or k_ref >= n_pool:
        raise ValueError(f"k_ref={k_ref} must satisfy {p + 2} <= k_ref < pool size {n_pool}")

    victim_mi = None
    if apply_gate1:
        vscen = victim_scenario if victim_scenario is not None else scenario
        victim_mi = extract_victim_mi(csv_path=victim_csv_path, scenario=vscen,
                                      in_size=in_size, bins=bins, mi_kind=mi_kind,
                                      victim_rate=victim_rate, victim_seed=victim_seed)

    split_seeds = list(split_seeds)
    if verbose:
        print(f"[{scenario}] pool={n_pool} models at rate {rate}; k_ref={k_ref}, "
              f"n_susp={n_pool - k_ref}; {len(split_seeds)} splits; shrinkage={shrinkage}"
              + (f"; gate1 on, victim MI={victim_mi}" if apply_gate1 else ""))
    if fresh_log and os.path.exists(log_path):
        os.remove(log_path)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    results = []
    seed_tally = {e["seed"]: {"n_susp": 0, "rej_F": 0, "rej_chi2": 0} for e in pool[rate]}
    a0 = alphas[0]

    for idx, split_seed in enumerate(split_seeds, start=1):
        h0_pool, susp_pool = split_h0_heldout(pool, {rate: k_ref}, seed=split_seed)
        mu, S, fit_info = build_h0(h0_pool, ddof=ddof, shrinkage=shrinkage)
        k = fit_info["k"]
        aux, _meta = h0_pool_to_matrix(h0_pool)

        diag = reference_diagnostics(aux, alphas=alphas, ddof=ddof, p=p)
        records = score_pool0_suspects(susp_pool, mu, S, k, alphas=alphas,
                                       victim_mi=victim_mi, apply_gate1=apply_gate1, p=p)
        cal = suspect_calibration(records, k, alphas=alphas, p=p)

        for r in records:
            t = seed_tally[r["seed"]]
            t["n_susp"] += 1
            t["rej_F"] += int(r[f"F@{a0}"])
            t["rej_chi2"] += int(r[f"chi2@{a0}"])

        log_pool0_split_section(
            log_path, scenario, csv_path, split_seed, k_ref, h0_pool, fit_info,
            diag, records, cal, shrinkage, in_size, bins, mi_kind, rate, alphas,
            split_index=idx, apply_gate1=apply_gate1,
        )

        row = {"scenario": scenario, "split_seed": split_seed, "k_ref": k,
               "n_susp": cal["n"], "cond": fit_info["cond"],
               "lambda": fit_info.get("shrinkage_lambda", 0.0)}
        for a in alphas:
            for law in ("chi2", "F"):
                row[f"nfp_{law}@{a}"] = cal[f"nfp_{law}@{a}"]
                row[f"fpr_{law}@{a}"] = cal[f"fpr_{law}@{a}"]
        row.update({
            "mardia_skew_p": diag["mardia"]["skew_p"],
            "mardia_kurt_p": diag["mardia"]["kurt_p"],
            "shapiro_p_ixt": diag["shapiro_p"][0],
            "shapiro_p_ity": diag["shapiro_p"][1],
            "loo_ks_p_F": diag["loo_ks_p_F"],
            "loo_ks_p_chi2": diag["loo_ks_p_chi2"],
            f"loo_rej_F@{a0}": diag["loo_rej_F"][a0],
            f"loo_rej_chi2@{a0}": diag["loo_rej_chi2"][a0],
            "susp_ks_p_chi2": cal["ks_p_chi2"],
            "susp_ks_p_F": cal["ks_p_F"],
            "mean_p_chi2": cal["mean_p_chi2"],
            "mean_p_F": cal["mean_p_F"],
        })
        results.append(row)

        if verbose:
            fp_str = "  ".join(
                f"@{a}: chi2 {cal[f'fpr_chi2@{a}']:.1%}({cal[f'nfp_chi2@{a}']}) "
                f"F {cal[f'fpr_F@{a}']:.1%}({cal[f'nfp_F@{a}']})" for a in alphas)
            print(f"  seed={split_seed:<5} k={k:<3} {fp_str}  "
                  f"mardia p=({diag['mardia']['skew_p']:.2f},{diag['mardia']['kurt_p']:.2f})  "
                  f"KS susp p=(chi2 {cal['ks_p_chi2']:.2f}, F {cal['ks_p_F']:.2f})  "
                  f"cond={fit_info['cond']:.0f}")

    agg = aggregate_pool0_results(scenario, k_ref, results, alphas=alphas)
    log_pool0_aggregate_section(log_path, agg, seed_tally, alphas)
    if per_split_csv is not None:
        write_dict_rows_csv(results, per_split_csv)
    if verbose:
        for a in alphas:
            print(f"  => @{a}: pooled FPR chi2 {agg[f'pooled_fpr_chi2@{a}']:.2%}  "
                  f"F {agg[f'pooled_fpr_F@{a}']:.2%}")
        print(f"  wrote {len(results)} sections + aggregate to {log_path}")
    return results, agg



if __name__ == "__main__":
    # ---- Pool-0.0 negative test: k_ref reference / (50 - k_ref) suspects -----
    # Pool: saved_logs/vanilla/MI_master_table_neg_pool0.csv (50 rate-0.0
    # models per CNN scenario). One txt log + one per-split CSV per
    # (scenario, k_ref), plus summary_pool0.csv across all of them.
    POOL0_SCENARIOS = [
        "CIFAR-10_ResNet-18_25000",
        "CIFAR-10_VGG16_25000",
        "CIFAR-100_ResNet-18_25000",
        "CIFAR-100_VGG16_25000",
        "CIFAR-10_DeiT_Plain_25000",       # DeiT pools: skipped until 50 models are in the CSV
        "CIFAR-100_DeiT_Distill_25000",
    ]
    POOL0_K_REF = [10, 15, 20, 25]
    POOL0_SPLIT_SEEDS = range(1000, 1050)      # 50 random reference/suspect splits
    POOL0_EXPECTED_POOL = 50

    os.makedirs(POOL0_LOG_ROOT, exist_ok=True)
    summary_rows = []
    for scenario in POOL0_SCENARIOS:
        log_dir = os.path.join(POOL0_LOG_ROOT, scenario)
        os.makedirs(log_dir, exist_ok=True)
        for k_ref in POOL0_K_REF:
            stem = f"{scenario}_pool0_k{k_ref}"
            try:
                _, agg = run_pool0_split_sweep(
                    csv_path=POOL0_CSV,
                    scenario=scenario,
                    k_ref=k_ref,
                    split_seeds=POOL0_SPLIT_SEEDS,
                    log_path=os.path.join(log_dir, f"{stem}_Neg_detection_log.txt"),
                    per_split_csv=os.path.join(log_dir, f"{stem}_per_split.csv"),
                    in_size=25000, bins=50, mi_kind="In",
                    expected_pool_size=POOL0_EXPECTED_POOL,
                    alphas=(0.05, 0.01),
                    shrinkage=None,
                )
            except ValueError as err:
                # e.g. the pool for this scenario is not complete yet
                print(f"[SKIP] {scenario} k_ref={k_ref}: {err}")
                break
            summary_rows.append(agg)
    write_dict_rows_csv(summary_rows, os.path.join(POOL0_LOG_ROOT, "summary_pool0.csv"))
    print(f"\nSummary ({len(summary_rows)} rows) -> {os.path.join(POOL0_LOG_ROOT, 'summary_pool0.csv')}")


    """ Previous entry (DeiT negatives via run_split_sweep_neg) -- kept for reference.
    neg_csv_path = "./saved_logs/vanilla/MI_master_table.csv"

    RATE_LOGICS = {
        "all_rates": [round(r * 0.1, 1) for r in range(11)],  # 11 rates
        "even_rates": [round(r * 0.1, 1) for r in [0, 2, 4, 6, 8, 10]],  # 6 rates
        "endpoints_mid": [round(r * 0.1, 1) for r in [0, 5, 10]],  # 3 rates
    }
    n_values = list(range(1, 10))

    # split logic: H0 construction.
    scenario_ls = ["CIFAR-100_DeiT_Plain_25000"]#"CIFAR-10_ResNet-18_25000", "CIFAR-10_VGG16_25000",
                   #"CIFAR-100_ResNet-18_25000", "CIFAR-100_VGG16_25000"]

    for scenario in scenario_ls:
        neg_log_dir = f"./saved_logs/vanilla/Hypo_Test/{scenario}"
        os.makedirs(neg_log_dir, exist_ok=True)

        for logic_name, rates in RATE_LOGICS.items():
            for n in n_values:
                spec = {r: n for r in rates}
                log_path = os.path.join(
                    neg_log_dir,
                    f"{scenario}_{logic_name}_n{n}_Neg_detection_log.txt",
                )

                results = run_split_sweep_neg(
                    csv_path=neg_csv_path,
                    scenario=scenario,
                    selection_spec=spec,
                    split_seeds=range(1000, 1050),  # 50 different splits
                    log_path=log_path,
                    in_size=25000, bins=50, mi_kind="In",
                    alphas=(0.05, 0.01),
                    shrinkage_modes=(None,),
                )
    """

    '''pos_csv_path = "./saved_logs/ft_vanilla/MI_master_table_ft.csv"


    ft_scenario_names = ["CIFAR-100_DeiT_Plain_25000_Same_10000_42_1.0"]#, "CIFAR-10_ResNet-18_25000_Same_10000_45_0.5"]#CIFAR-10_VGG16_25000_PseudoLabel_10000_42_1.0"]#"CIFAR-100_ResNet-18_25000_Same_10000_42_1.0"]


    ft_strategy = ["FT-LL_ftsize=10000",
                   "FT-AL_ftsize=10000",
                   "RT-AL_ftsize=10000"]

    ft_seeds = range(0, 1)

    for scenario_name in ft_scenario_names:
        for strategy in ft_strategy:
            for ft_seed in ft_seeds:
                model_name = f"{scenario_name}_{strategy}_ftseed={ft_seed}"

                pos_log_dir = f"./saved_logs/ft_vanilla/Hypo_Test/{model_name}"
                os.makedirs(pos_log_dir, exist_ok=True)

                for logic_name, rates in RATE_LOGICS.items():
                    for n in n_values:
                        spec = {r: n for r in rates}

                        log_path = os.path.join(
                            pos_log_dir,
                            f"{model_name}_{logic_name}_n{n}_Pos_detection_log.txt",
                        )

                        evaluate_positive_case(
                            neg_csv_path=neg_csv_path,
                            pos_csv_path=pos_csv_path,
                            neg_scenario="CIFAR-100_DeiT_Plain_25000",  # negative CSV 的标签
                            model_name=model_name,  # positive 只认这个
                            pos_scenario="...",  # 可选,log 记录用
                            selection_spec=spec,
                            split_seeds=range(1000, 1050),
                            log_path=log_path,
                            in_size=25000, bins=50, mi_kind="In",
                            epochs=None,
                            shrinkage=None,
                            fresh_log=True,
                        )'''
    
    # Cross-arch Eval
    '''
    "CIFAR-10_ResNet-18_25000",
        "CIFAR-10_DeiT_Plain_25000",
        "CIFAR-10_VGG16_25000",'''
    '''neg_csv_path = "./saved_logs/vanilla/MI_master_table.csv"

    ALL_SCENARIOS = [
        "CIFAR-10_ResNet-18_25000",
        "CIFAR-10_ResNet-18_10000_test",
        
    ]

    RATE_LOGICS = {
        "all_rates":     [round(r*0.1, 1) for r in range(11)],
        "even_rates":    [round(r*0.1, 1) for r in [0, 2, 4, 6, 8, 10]],
        "endpoints_mid": [round(r*0.1, 1) for r in [0, 5, 10]],
    }

    for h0_scenario in ALL_SCENARIOS:
        # suspect = 除 H0 之外的其余架构
        suspect_scenarios = [s for s in ALL_SCENARIOS if s != h0_scenario]

        for suspect_scen in suspect_scenarios:
            for gate_on in (False, True):
                gate_tag = "gate1" if gate_on else "nogate"
                cross_dir = f"./saved_logs/vanilla/CrossArch/H0={h0_scenario}"
                os.makedirs(cross_dir, exist_ok=True)
                for logic_name, rates in RATE_LOGICS.items():
                    for n in range(1, 10):
                        spec = {r: n for r in rates}
                        log_path = os.path.join(
                            cross_dir,
                            f"SUSP={suspect_scen}_{logic_name}_n{n}_{gate_tag}_CrossArch_log.txt",
                        )
                        run_split_sweep_neg(
                            csv_path=neg_csv_path,
                            scenario=h0_scenario,            # H0 来源（动态）
                            selection_spec=spec,
                            split_seeds=range(1000, 1050),
                            log_path=log_path,
                            in_size=25000, bins=50, mi_kind="In",
                            alphas=(0.05, 0.01),
                            shrinkage_modes=(None,),
                            suspect_scenario=suspect_scen,   # suspect 来源（别的架构）
                            suspect_csv_path=neg_csv_path,
                            apply_gate1=gate_on,
                            victim_scenario=h0_scenario,     # victim = 当前 H0
                        )'''







