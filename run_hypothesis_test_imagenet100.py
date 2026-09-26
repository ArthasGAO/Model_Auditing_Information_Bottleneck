"""ImageNet-100 Swin-T hypothesis test on the frozen 50-round reference splits.

Run with the project interpreter from E:\\Experiment:
    python run_hypothesis_test_imagenet100.py

What it does
------------
* Negative pool: the 80 rate-0.0 Swin-T models (seeds 42..121) in
  MI_pool0/MI_master_table_neg_pool0.csv.  Their seed identifiers are the same
  80 integers as the CIFAR pool, so the CIFAR split template
  (saved_logs/vanilla/fixed_splits/pool0_80_models_v1.json: 50 rounds, 50
  evaluation negatives + 30 ordered H0 candidates, nested k = 5..30) is reused
  verbatim.  This makes the ImageNet-100 run paired with the CIFAR runs.  A new
  manifest document is written that carries the same rounds and a single
  ImageNet-100 case; nothing is sampled here.
* Positives (LS01 policy): FT-LL / FT-AL / RT-AL from the LS01 plan, pruning
  20% / 80% *final* checkpoints from the LS01 plan, KD / DKD (student recipe ==
  pool recipe, label smoothing 0.1) and the single Knockoff surrogate (label
  smoothing 0.0; kept because there is no LS01 variant).  The victim is scored
  as an extra reference group.
* Statistics: exactly the code path of run_hypothesis_test_fixed_splits.evaluate
  and run_hypothesis_test_positives.evaluate_positives (same T2, chi2 and
  exact-F p-values, same diagnostics); this file only prepares the inputs.
* Grid: every (bins, in_size) cell present in the pool table (11 bins x 7
  sizes = 77 cells) x k in {5,10,15,20,25,30} x 50 rounds.
* Round selection: per (cell, k) the round with the minimum number of F-law
  false positives at alpha 0.01 over its 50 evaluation negatives (tie -> round
  44 if tied, else the smallest round id: the rule of
  run_hypothesis_test_same_arch_best_fpr_per_k.py).  Additionally the rounds
  selected at the canonical cell (in_size 63342, bins 50) are applied to every
  other cell so that the transfer of one selection across the grid is visible.

Outputs (OUTPUT_ROOT)
---------------------
  manifest_in100.json                    the reused-rounds manifest (hash-checked)
  negatives/In_size<S>_bins<B>/          per cell: per_model / per_split / summary
  positives/In_size<S>_bins<B>/          per cell: per_model / per_split / summary
  summary_negatives_all_cells.csv        concat of the per-cell negative summaries
  summary_positives_all_cells.csv        concat of the per-cell positive summaries
  round_selection_all_cells.csv          best round per (cell, k) + tie info
  best_round_summary.csv                 FPR and per-method detection at the best round
  best_round_canonical_transfer.csv      canonical-cell rounds applied to every cell
  grid_overview.csv                      one row per (cell, k): 50-round means + best round
  cells_status.csv                       feasible k per cell; k skipped for a singular reference
  failed_attempts/                       directories left by an aborted cell (moved, not deleted)
  run_log.txt                            progress / failures

Singular references: small probes with many bins saturate the estimator
(I(T;Y) -> H(Y) for every model, only a handful of distinct values in the
pool), so a k=5 reference can have zero variance on that axis.  Such k values
are skipped for that cell and listed in cells_status.csv; all other k values
of the cell are evaluated normally.  Re-running the script skips completed
cells and only recomputes missing ones, then re-aggregates everything.
"""
import csv
import hashlib
import io
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import run_hypothesis_test_fixed_splits as neg
import run_hypothesis_test_positives as pos

BASE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- inputs
CIFAR_MANIFEST = BASE / "saved_logs/vanilla/fixed_splits/pool0_80_models_v1.json"
POOL_CSV = BASE / "MI_pool0/MI_master_table_neg_pool0.csv"
VICTIM_CSV = BASE / "MI_pool0/MI_master_table_victim.csv"
FT_CSV = BASE / "imagenet100_attacks/ft_final/MI_master_table_ft.csv"
PRUNE_CSV = BASE / "imagenet100_attacks/pruning_final/MI_master_table_prune.csv"
KD_CSV = BASE / "imagenet100_attacks/kd_final/MI_master_table_kd.csv"
KNOCKOFF_CSV = BASE / "imagenet100_attacks/extraction_final/MI_master_table_knockoff.csv"

CASE = "in100_swin_tiny_scratch_half"
TRAINING_SIZE = 63342           # group_A size of the victim / pool recipe
VICTIM_NAME = "in100_swin_tiny_scratch_half_neg_B_ov1p0_s42_r0"
MI_KIND = "In"
ALPHAS = [0.05, 0.01]
K_VALUES = [5, 10, 15, 20, 25, 30]
SELECT_LAW, SELECT_ALPHA, PREFERRED_TIE_ROUND = "F", 0.01, 44
CANONICAL_CELL = (63342, 50)    # (in_size, bins) whose selection is transferred

# (label, csv, Scenario, extra row filters)  -- LS01 policy
POSITIVES = [
    ("FT-LL", FT_CSV, "in100_swin_tiny_scratch_half_Same_63342_LS01",
     {"strategy": "FT-LL"}),
    ("FT-AL", FT_CSV, "in100_swin_tiny_scratch_half_Same_63342_LS01",
     {"strategy": "FT-AL"}),
    ("RT-AL", FT_CSV, "in100_swin_tiny_scratch_half_Same_63342_LS01",
     {"strategy": "RT-AL"}),
    ("P-20%", PRUNE_CSV, "in100_swin_tiny_scratch_half_Same_63342_LS01",
     {"sparsity": "0.2", "ckpt_kind": "final"}),
    ("P-80%", PRUNE_CSV, "in100_swin_tiny_scratch_half_Same_63342_LS01",
     {"sparsity": "0.8", "ckpt_kind": "final"}),
    ("KD", KD_CSV, "in100_swin_tiny_scratch_half_SwinToSwin_63342", {"method": "KD"}),
    ("DKD", KD_CSV, "in100_swin_tiny_scratch_half_SwinToSwin_63342", {"method": "DKD"}),
    ("Knockoff", KNOCKOFF_CSV, "in100_swin_tiny_scratch_half_Knockoff_Same100_SameSwin",
     {"attack": "Knockoff"}),
    ("Victim", VICTIM_CSV, CASE, {"model_name": VICTIM_NAME}),
]
METHOD_ORDER = [label for label, *_ in POSITIVES]

OUTPUT_ROOT = BASE / "saved_logs/imagenet100/Hypo_Test_IN100_LS01_v1"
MANIFEST_OUT = OUTPUT_ROOT / "manifest_in100.json"


# ---------------------------------------------------------------- helpers
def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def log(message, stream):
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    stream.write(line + "\n")
    stream.flush()


def build_in100_document(cifar_document, pool_rows):
    """Reuse the CIFAR rounds verbatim; only the case inventory changes."""
    payload = cifar_document["payload"]
    models = {}
    for row in pool_rows:
        name, seed = row["model_name"], int(row["seed"])
        neg.require(row["Scenario"] == CASE and float(row["rate"]) == 0.0,
                    f"Unexpected pool row: {name}")
        entry = {"model_name": name, "checkpoint": row["checkpoint"],
                 "checkpoint_sha256": row["checkpoint_sha256"]}
        if str(seed) in models:
            neg.require(models[str(seed)] == entry, f"Inconsistent pool metadata: {name}")
        models[str(seed)] = entry
    neg.require(sorted(map(int, models)) == payload["pool_seeds"],
                "ImageNet-100 pool seeds differ from the CIFAR split template")
    new_payload = {
        "schema_version": 1,
        "design": "paired_nested_h0_fixed_50_negatives (rounds reused from CIFAR template)",
        "pool_size": 80, "evaluation_size": 50, "n_rounds": payload["n_rounds"],
        "k_values": payload["k_values"], "pool_seeds": payload["pool_seeds"],
        "rate": 0.0, "master_seed": payload["master_seed"],
        "generator": payload["generator"], "generator_python": payload["generator_python"],
        "source_manifest_sha256": cifar_document["manifest_sha256"],
        "cases": {CASE: models}, "rounds": payload["rounds"],
    }
    return {"manifest_sha256": neg.payload_hash(new_payload), "payload": new_payload}


def select_rows(rows, *, in_size, bins, scenario=None, filters=None):
    out = []
    for row in rows:
        if float(row["in_size"]) != in_size or float(row["bins"]) != bins:
            continue
        if scenario is not None and row["Scenario"] != scenario:
            continue
        if any(row[c] != v for c, v in (filters or {}).items()):
            continue
        out.append(row)
    return out


def preflight_pool(pool_rows, in_size, bins):
    """Exactly one finite MI row per pool model at this cell."""
    fields = [f"I(X;T)-{MI_KIND}", f"I(T;Y)-{MI_KIND}"]
    found = {}
    for row in select_rows(pool_rows, in_size=in_size, bins=bins, scenario=CASE):
        name = row["model_name"]
        neg.require(name not in found, f"Duplicate pool MI row: {name} @ {in_size}/{bins}")
        vector = [float(row[f]) for f in fields]
        neg.require(all(math.isfinite(v) for v in vector), f"Nonfinite pool MI: {name}")
        found[name] = vector
    neg.require(len(found) == 80, f"Pool has {len(found)} models at {in_size}/{bins}, expected 80")
    return found


def preflight_positives(tables, hashes, in_size, bins):
    """One family, one group per method; each group holds exactly one model."""
    fields = [f"I(X;T)-{MI_KIND}", f"I(T;Y)-{MI_KIND}"]
    groups, values, group_of, scenario_of, report = {}, {}, {}, {}, {}
    for label, csv_path, scenario, filters in POSITIVES:
        rows = select_rows(tables[csv_path], in_size=in_size, bins=bins,
                           scenario=scenario, filters=filters)
        neg.require(len(rows) == 1,
                    f"{label}: expected exactly one row at {in_size}/{bins}, got {len(rows)}")
        row = rows[0]
        name = row["model_name"]
        vector = [float(row[f]) for f in fields]
        neg.require(all(math.isfinite(v) for v in vector), f"{label}: nonfinite MI")
        neg.require(name not in values, f"{label}: model {name} selected twice")
        values[name] = vector
        key = (scenario, label)
        groups[key] = [name]
        group_of[name] = "|".join(key)
        scenario_of[name] = scenario
        report[label] = {"model_name": name, "csv": str(csv_path),
                         "csv_sha256": hashes[csv_path], "filters": filters,
                         "checkpoint_sha256": row.get("checkpoint_sha256")}
    family = {"label": "in100_ls01", "plan_dir": Path(__file__), "mi_csv": FT_CSV,
              "h0_scenario": CASE, "group_by": ("Scenario", "method"),
              "_groups": groups, "_values": values,
              "_hash": sha256_bytes(json.dumps(sorted(hashes.values())).encode()),
              "_report": {"selection_mode": "explicit_in100_ls01_case_map",
                          "n_positive": len(values), "methods": report},
              "_group_of": group_of, "_scenario_of": scenario_of}
    return family


def cell_tag(in_size, bins):
    return f"In_size{in_size}_bins{bins}"


def feasible_k_values(document, h0_values):
    """k values whose reference covariance is positive definite in all 50 rounds.

    Small probes with many bins saturate the MI estimator (I(T;Y) -> H(Y) for
    every model), so a k-subset of the pool can have zero variance on one axis.
    `score` refuses such a reference; instead of abandoning the cell, the cell
    is run on the k values that are feasible in every round and the skipped k
    values are recorded.  Nothing is resampled.
    """
    import numpy as np

    def positive_definite(x):
        return np.linalg.eigvalsh(np.cov(x, rowvar=False, ddof=1)).min() > 0

    feasible, skipped = [], {}
    for k in K_VALUES:
        bad = None
        for r in range(document["payload"]["n_rounds"]):
            split = neg.get_split(document, CASE, r, k)
            x = np.array([h0_values[m["model_name"]] for m in split["h0"]], dtype=float)
            # the full reference (score) and every leave-one-out reference
            # (diagnostics) must be positive definite, as evaluate requires
            if not positive_definite(x) or not all(
                    positive_definite(np.delete(x, i, axis=0)) for i in range(k)):
                bad = r
                break
        if bad is None:
            feasible.append(k)
        else:
            skipped[k] = bad
    return feasible, skipped


# ---------------------------------------------------------------- aggregation
def choose_round(neg_split_rows, law=SELECT_LAW, alpha=SELECT_ALPHA):
    """Minimum false positives under law@alpha; tie -> 44 else smallest round id."""
    column = f"nfp_{law}@{alpha}"
    by_round = {int(r["round_id"]): int(r[column]) for r in neg_split_rows}
    neg.require(set(by_round) == set(range(50)), "Expected 50 unique rounds")
    minimum = min(by_round.values())
    ties = sorted(r for r, v in by_round.items() if v == minimum)
    chosen = PREFERRED_TIE_ROUND if PREFERRED_TIE_ROUND in ties else ties[0]
    return chosen, minimum, ties


def aggregate(cells, out_root, stream):
    import pandas as pd

    neg_summary, pos_summary, selection, best, transfer, overview = [], [], [], [], [], []
    neg_splits, pos_splits, pos_models = {}, {}, {}
    for (in_size, bins, rate) in cells:
        tag = cell_tag(in_size, bins)
        ndir, pdir = out_root / "negatives" / tag, out_root / "positives" / tag
        ns = pd.read_csv(ndir / "summary.csv")
        ps = pd.read_csv(pdir / "summary.csv")
        neg_summary.append(ns)
        pos_summary.append(ps)
        neg_splits[(in_size, bins)] = pd.read_csv(ndir / "per_split.csv")
        pos_splits[(in_size, bins)] = pd.read_csv(pdir / "per_split.csv")
        pos_models[(in_size, bins)] = pd.read_csv(pdir / "per_model.csv")

    def method_of(group):
        return group.split("|")[-1]

    canonical = {}
    for (in_size, bins, rate) in cells:
        nsp, psp, pm = (neg_splits[(in_size, bins)], pos_splits[(in_size, bins)],
                        pos_models[(in_size, bins)])
        psp = psp.assign(method=psp["group"].map(method_of))
        pm = pm.assign(method=pm["group"].map(method_of))
        ks_present = sorted(int(k) for k in nsp.k_ref.unique())
        for k in ks_present:
            rows = nsp[nsp.k_ref == k].to_dict("records")
            chosen, minimum, ties = choose_round(rows)
            alt_chosen, alt_min, alt_ties = choose_round(rows, "F", 0.05)
            selection.append({"in_size": in_size, "in_size_rate": rate, "bins": bins,
                              "k_ref": k, "round_id": chosen,
                              f"nfp_{SELECT_LAW}@{SELECT_ALPHA}": minimum,
                              f"fpr_{SELECT_LAW}@{SELECT_ALPHA}": minimum / 50,
                              "n_tied_minimum_rounds": len(ties),
                              "tied_round_ids": json.dumps(ties),
                              "alt_round_id_F@0.05": alt_chosen,
                              "alt_nfp_F@0.05": alt_min,
                              "alt_n_tied_minimum_rounds_F@0.05": len(alt_ties)})
            if (in_size, bins) == CANONICAL_CELL:
                canonical[k] = chosen

            def record_round(rid, target, kind):
                nrow = nsp[(nsp.k_ref == k) & (nsp.round_id == rid)].iloc[0]
                rec = {"in_size": in_size, "in_size_rate": rate, "bins": bins,
                       "k_ref": k, "selection": kind, "round_id": int(rid)}
                for law in ("F", "chi2"):
                    for a in ALPHAS:
                        rec[f"fpr_{law}@{a}"] = float(nrow[f"fpr_{law}@{a}"])
                rec["condition_number"] = float(nrow["condition_number"])
                rec["mardia_skew_p"] = float(nrow["mardia_skew_p"])
                rec["mardia_kurt_p"] = float(nrow["mardia_kurt_p"])
                sub = pm[(pm.k_ref == k) & (pm.round_id == rid)]
                for m in METHOD_ORDER:
                    r = sub[sub.method == m]
                    neg.require(len(r) == 1, f"{tag} k={k} round={rid}: method {m} rows={len(r)}")
                    r = r.iloc[0]
                    rec[f"p_F[{m}]"] = float(r["p_F"])
                    rec[f"det_F@0.01[{m}]"] = int(r["p_F"] < 0.01)
                    rec[f"det_F@0.05[{m}]"] = int(r["p_F"] < 0.05)
                    rec[f"det_chi2@0.01[{m}]"] = int(r["p_chi2"] < 0.01)
                attack = [m for m in METHOD_ORDER if m != "Victim"]
                rec["n_attacks_det_F@0.01"] = sum(rec[f"det_F@0.01[{m}]"] for m in attack)
                rec["n_attacks_det_F@0.05"] = sum(rec[f"det_F@0.05[{m}]"] for m in attack)
                target.append(rec)

            record_round(chosen, best, "best_per_cell")
            if alt_chosen != chosen:
                record_round(alt_chosen, best, "best_per_cell_alt_F@0.05")

            # 50-round overview
            nrow = nsp[nsp.k_ref == k]
            ov = {"in_size": in_size, "in_size_rate": rate, "bins": bins, "k_ref": k,
                  "n_rounds": int(len(nrow))}
            for law in ("F", "chi2"):
                for a in ALPHAS:
                    ov[f"mean_fpr_{law}@{a}"] = float(nrow[f"fpr_{law}@{a}"].mean())
            ov["min_fpr_F@0.01"] = float(nrow["fpr_F@0.01"].min())
            ov["n_rounds_zero_fp_F@0.01"] = int((nrow["nfp_F@0.01"] == 0).sum())
            ov["median_cond"] = float(nrow["condition_number"].median())
            ov["frac_rounds_mardia_skew_p<0.05"] = float((nrow["mardia_skew_p"] < 0.05).mean())
            sub = psp[psp.k_ref == k]
            for m in METHOD_ORDER:
                r = sub[sub.method == m]
                ov[f"tpr_F@0.01[{m}]"] = float(r["tpr_F@0.01"].mean())
                ov[f"tpr_F@0.05[{m}]"] = float(r["tpr_F@0.05"].mean())
            attack = [m for m in METHOD_ORDER if m != "Victim"]
            ov["mean_tpr_F@0.01_attacks"] = float(sum(ov[f"tpr_F@0.01[{m}]"] for m in attack) / len(attack))
            ov["mean_tpr_F@0.05_attacks"] = float(sum(ov[f"tpr_F@0.05[{m}]"] for m in attack) / len(attack))
            pmk = pm[pm.k_ref == k]
            for m in METHOD_ORDER:
                ov[f"median_p_F[{m}]"] = float(pmk[pmk.method == m]["p_F"].median())
            chosen_rec = next(b for b in reversed(best)
                              if b["selection"] == "best_per_cell" and b["k_ref"] == k
                              and b["in_size"] == in_size and b["bins"] == bins)
            ov["best_round_id"] = chosen
            ov["best_round_fpr_F@0.01"] = minimum / 50
            ov["best_round_fpr_F@0.05"] = chosen_rec["fpr_F@0.05"]
            ov["best_round_n_attacks_det_F@0.01"] = chosen_rec["n_attacks_det_F@0.01"]
            ov["best_round_n_attacks_det_F@0.05"] = chosen_rec["n_attacks_det_F@0.05"]
            overview.append(ov)

    neg.require(set(canonical) == set(K_VALUES), "Canonical cell missing from the grid")
    for (in_size, bins, rate) in cells:
        nsp, pm = neg_splits[(in_size, bins)], pos_models[(in_size, bins)]
        pm = pm.assign(method=pm["group"].map(method_of))
        for k in sorted(int(k) for k in nsp.k_ref.unique()):
            rid = canonical[k]
            nrow = nsp[(nsp.k_ref == k) & (nsp.round_id == rid)].iloc[0]
            rec = {"in_size": in_size, "in_size_rate": rate, "bins": bins, "k_ref": k,
                   "canonical_round_id": rid}
            for law in ("F", "chi2"):
                for a in ALPHAS:
                    rec[f"fpr_{law}@{a}"] = float(nrow[f"fpr_{law}@{a}"])
            sub = pm[(pm.k_ref == k) & (pm.round_id == rid)]
            attack = [m for m in METHOD_ORDER if m != "Victim"]
            for m in METHOD_ORDER:
                rec[f"det_F@0.01[{m}]"] = int(sub[sub.method == m].iloc[0]["p_F"] < 0.01)
            rec["n_attacks_det_F@0.01"] = sum(rec[f"det_F@0.01[{m}]"] for m in attack)
            transfer.append(rec)

    pd.concat(neg_summary).to_csv(out_root / "summary_negatives_all_cells.csv", index=False)
    pd.concat(pos_summary).to_csv(out_root / "summary_positives_all_cells.csv", index=False)
    pd.DataFrame(selection).to_csv(out_root / "round_selection_all_cells.csv", index=False)
    pd.DataFrame(best).to_csv(out_root / "best_round_summary.csv", index=False)
    pd.DataFrame(transfer).to_csv(out_root / "best_round_canonical_transfer.csv", index=False)
    pd.DataFrame(overview).to_csv(out_root / "grid_overview.csv", index=False)
    log(f"aggregated {len(cells)} cells -> grid_overview.csv ({len(overview)} rows)", stream)


# ---------------------------------------------------------------- main
def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "run_log.txt").open("a", encoding="utf-8") as stream:
        log(f"start; python={sys.version.split()[0]}", stream)
        cifar = neg.load_manifest(CIFAR_MANIFEST)
        pool_rows = read_rows(POOL_CSV)
        pool_hash = sha256_bytes(POOL_CSV.read_bytes())
        document = build_in100_document(cifar, pool_rows)
        if MANIFEST_OUT.exists():
            existing = json.loads(MANIFEST_OUT.read_text(encoding="utf-8"))
            neg.require(existing["manifest_sha256"] == document["manifest_sha256"],
                        "Existing ImageNet-100 manifest differs; choose a new OUTPUT_ROOT")
        else:
            neg.write_json(MANIFEST_OUT, document)
        log(f"manifest_in100 sha256={document['manifest_sha256'][:16]} "
            f"(rounds from CIFAR template {cifar['manifest_sha256'][:16]})", stream)

        tables = {p: read_rows(p) for p in {c for _, c, _, _ in POSITIVES}}
        hashes = {p: sha256_bytes(p.read_bytes()) for p in tables}
        # provenance strings recorded by evaluate_positives' metadata
        pos.H0_CSV, pos.ROOT, pos.VICTIM_CSV = POOL_CSV, BASE / "MI_pool0", VICTIM_CSV

        cells = sorted({(int(float(r["in_size"])), int(float(r["bins"])),
                         float(r["in_size_rate"])) for r in pool_rows},
                       key=lambda c: (c[1], c[0]))
        log(f"grid: {len(cells)} cells; bins={sorted({b for _, b, _ in cells})}; "
            f"in_sizes={sorted({s for s, _, _ in cells})}", stream)

        def move_aside(path):
            """A directory left by a failed attempt is moved, never deleted."""
            if path.exists() and not (path / "summary.csv").exists():
                target = OUTPUT_ROOT / "failed_attempts" / f"{path.parent.name}_{path.name}_{time.strftime('%Y%m%dT%H%M%S')}"
                target.parent.mkdir(parents=True, exist_ok=True)
                path.rename(target)
                log(f"moved incomplete {path.parent.name}/{path.name} -> {target.relative_to(OUTPUT_ROOT)}", stream)

        done, failed, status = [], [], []
        for index, (in_size, bins, rate) in enumerate(cells, 1):
            tag = cell_tag(in_size, bins)
            ndir, pdir = OUTPUT_ROOT / "negatives" / tag, OUTPUT_ROOT / "positives" / tag
            h0_values = preflight_pool(pool_rows, in_size, bins)
            feasible, skipped = feasible_k_values(document, h0_values)
            status.append({"in_size": in_size, "in_size_rate": rate, "bins": bins,
                           "feasible_k": json.dumps(feasible),
                           "skipped_k_singular_reference": json.dumps(
                               {str(k): f"first singular round {r}" for k, r in skipped.items()})})
            if (ndir / "summary.csv").exists() and (pdir / "summary.csv").exists():
                done.append((in_size, bins, rate))
                continue
            move_aside(ndir)
            move_aside(pdir)
            started = time.time()
            try:
                neg.require(bool(feasible), f"no k value has a positive-definite reference: {skipped}")
                family = preflight_positives(tables, hashes, in_size, bins)
                sizes = {CASE: in_size}
                if not (ndir / "summary.csv").exists():
                    neg.evaluate(document, h0_values, pool_hash, output_dir=ndir,
                                 csv_path=POOL_CSV, model_root=BASE / "MI_pool0",
                                 in_size=sizes, bins=bins, mi_kind=MI_KIND, alphas=ALPHAS,
                                 in_size_rate=rate, training_sizes={CASE: TRAINING_SIZE},
                                 k_values=feasible)
                if not (pdir / "summary.csv").exists():
                    pos.evaluate_positives(document, [family], h0_values, pool_hash,
                                           output_dir=pdir, in_size=sizes, bins=bins,
                                           mi_kind=MI_KIND, alphas=ALPHAS, in_size_rate=rate,
                                           training_sizes={CASE: TRAINING_SIZE},
                                           k_values=feasible)
                done.append((in_size, bins, rate))
                note = f"; skipped k={sorted(skipped)} (singular reference)" if skipped else ""
                log(f"[{index}/{len(cells)}] {tag} ok ({time.time() - started:.0f}s){note}", stream)
            except Exception as exc:  # keep the sweep going, record the failure
                failed.append((in_size, bins, rate, str(exc)))
                log(f"[{index}/{len(cells)}] {tag} FAILED: {exc}", stream)
        with (OUTPUT_ROOT / "cells_status.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(status[0]))
            writer.writeheader()
            writer.writerows(status)
        log(f"cells done={len(done)} failed={len(failed)}", stream)
        for in_size, bins, rate, why in failed:
            log(f"  failed {cell_tag(in_size, bins)}: {why}", stream)
        aggregate(sorted(done, key=lambda c: (c[1], c[0])), OUTPUT_ROOT, stream)
        log("finished", stream)


if __name__ == "__main__":
    main()
