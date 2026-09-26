"""Positive-suspect hypothesis testing on the frozen negative splits.

Edit the global configuration below, then run:
python run_hypothesis_test_positives.py

Relation to run_hypothesis_test_fixed_splits.py
-----------------------------------------------
This script does not re-derive any statistics. It imports `load_manifest`,
`get_split`, `score` and `diagnostics` from the negative driver, so a positive
suspect is judged against exactly the reference set the negative run used for
the same (H0 scenario, round, k): same manifest, same 50 rounds, same nested
H0 seeds, same mu / Sigma estimator, same chi2 and exact-F p-values. Nothing
here samples models. Like that script it needs only Python, NumPy and SciPy;
no Torch, no checkpoints are loaded.

Case selection is plan-based
----------------------------
Each entry of POSITIVE_FAMILIES names a plan DIRECTORY. Every *.yaml in it
contributes its `Scenario_Name`, and the family's positives are the rows of its
MI table carrying those scenarios. Adding models to saved_logs does not enlarge
a run; adding a plan to the directory does. A plan whose scenario has no rows is
reported, and a table row outside the declared scenarios is ignored (and
counted), so the two can be reconciled.

Metric
------
For each (family, group, k, round): fit (mu, Sigma) on the round's k reference
models, score every positive of the group, and record the rejection fraction.
That is the TPR of that round. The headline number is the mean over the 50
rounds, with the sample standard deviation describing round-to-round variation
of the fixed pool, not a confidence interval. The positive set is identical in
every round; only the reference moves.
"""
import csv
import io
import json
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from run_hypothesis_test_fixed_splits import (
    apply_gate1, diagnostics, digest, get_split, law_statistics, load_manifest,
    preflight, preflight_victim, rate_configuration_grid, require, score,
    summary_statistics, write_json,
)

# ============================================================================
# Global configuration: edit this section, then run the file directly.
# ============================================================================
# "validate": check plans, manifest and MI rows only. "evaluate": also score.
RUN_MODE = "evaluate"
IN_SIZE_RATES = [0.05, 0.10, 0.20, 0.50, 0.75, 1.00]
BINS = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]
MI_KIND = "In"              # "In" or "Out"
ALPHAS = [0.05, 0.01]

BASE = Path(__file__).resolve().parent
# The reference side is the negative run's, unchanged.
ROOT = BASE / "saved_models/vanilla/Negative_Model_Pool_0.0"
H0_CSV = BASE / "saved_logs/vanilla/MI_master_table_neg_pool0.csv"
MANIFEST = BASE / "saved_logs/vanilla/fixed_splits/pool0_80_models_v1.json"
OUTPUT_DIR = BASE / "saved_logs/vanilla/Hypo_Test_Positives"

# Denominators for IN_SIZE_RATES, per H0 scenario (the probe set is the
# victim's, so the positives inherit the same absolute in_size).
TRAINING_SIZES = {
    "CIFAR-10_ResNet-18_25000": 25000,
    "CIFAR-10_VGG16_25000": 25000,
    "CIFAR-100_ResNet-18_25000": 25000,
    "CIFAR-100_VGG16_25000": 25000,
    "CIFAR-10_DeiT_Plain_25000": 25000,
    "CIFAR-100_DeiT_Distill_25000": 25000,
}

# One entry per positive family.
#   label        short name used in the output files
#   plan_dir     every *.yaml here contributes its Scenario_Name
#   mi_csv       the family's MI master table
#   h0_scenario  which manifest case supplies the reference models
#   filters      optional {column: [allowed values]} applied on top of the
#                scenario selection, e.g. {"ckpt_kind": ["best_clean"]}
#   group_by     table columns that split the family into reported groups;
#                every column must exist in mi_csv. ("Scenario",) is the
#                minimum and is always prepended if absent.
POSITIVE_FAMILIES = [
    {"label": "ft",
     "plan_dir": BASE / "saved_exp_plan/ft_plan",
     "mi_csv": BASE / "saved_logs/ft_final/MI_master_table_ft.csv",
     "h0_scenario": "CIFAR-10_ResNet-18_25000",
     "group_by": ("Scenario", "strategy")},
    {"label": "extraction",
     "plan_dir": BASE / "saved_exp_plan/extraction_plan",
     "mi_csv": BASE / "saved_logs/extraction_final/MI_master_table_extraction.csv",
     "h0_scenario": "CIFAR-10_ResNet-18_25000",
     "group_by": ("Scenario",)},
    {"label": "at",
     "plan_dir": BASE / "saved_exp_plan/at_plan",
     "mi_csv": BASE / "saved_logs/at_final/MI_master_table_at.csv",
     "h0_scenario": "CIFAR-10_ResNet-18_25000",
     "group_by": ("Scenario", "eps", "ckpt_kind")},
]

# Gate 1 (optional, default OFF): identical rule and victim table as the
# negative driver. A positive failing the gate is short-circuited to T2=0, p=1,
# i.e. counted as NOT detected, which is the honest accounting for a necessary
# condition that the suspect does not meet.
APPLY_GATE1 = False
VICTIM_CSV = BASE / "saved_logs/vanilla/MI_master_table_victim.csv"


def scenarios_from_plan(path):
    """The Scenario values one plan declares.

    Two plan shapes are understood, matching the MI scripts:
      * `Scenario_Name` directly (FT, extraction, victim plans).
      * `Model_Path` only (AT plans): the scenario is each base model's name
        minus its trailing _<seed>_<rate>, which is what calculate_MI_at writes
        into the Scenario column.
    Selection is therefore at scenario granularity; use a family's `filters` to
    narrow further on any column of its MI table (eps, run_tag, ckpt_kind, ...).
    """
    import yaml as _yaml

    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        plan = _yaml.safe_load(stream)
    require(isinstance(plan, dict), f"Plan is not a mapping: {path}")
    if plan.get("Scenario_Name"):
        return [str(plan["Scenario_Name"])]
    paths = plan.get("Model_Path") or []
    require(bool(paths),
            f"Plan declares neither Scenario_Name nor Model_Path: {path}")
    scenarios = []
    for entry in paths:
        base = str(entry).rstrip("/").split("/")[-1]
        parts = base.split("_")
        require(len(parts) >= 3,
                f"Model_Path entry {base!r} carries no _<seed>_<rate> suffix: {path}")
        scenarios.append("_".join(parts[:-2]))
    return scenarios


def scenarios_from_plan_dir(plan_dir):
    """{scenario: plan path} for every *.yaml in the directory."""
    plan_dir = Path(plan_dir)
    require(plan_dir.is_dir(), f"Plan directory not found: {plan_dir}")
    files = sorted(plan_dir.glob("*.yaml"))
    require(bool(files), f"No *.yaml plans in {plan_dir}")
    scenarios = {}
    for path in files:
        for name in scenarios_from_plan(path):
            require(name not in scenarios,
                    f"Two plans in {plan_dir} declare scenario {name!r}: "
                    f"{scenarios.get(name)} and {path}")
            scenarios[name] = path
    return scenarios


def normalize_family(family):
    """Validate one POSITIVE_FAMILIES entry and fill in its defaults."""
    require(isinstance(family, dict), "Each positive family must be a dict")
    missing = {"label", "plan_dir", "mi_csv", "h0_scenario"} - set(family)
    require(not missing, f"Positive family is missing keys: {sorted(missing)}")
    group_by = tuple(family.get("group_by") or ("Scenario",))
    if "Scenario" not in group_by:
        group_by = ("Scenario",) + group_by
    require(len(set(group_by)) == len(group_by),
            f"{family['label']}: group_by contains duplicates")
    return {**family, "group_by": group_by,
            "plan_dir": Path(family["plan_dir"]), "mi_csv": Path(family["mi_csv"])}


def preflight_positives(family, in_size, bins, mi_kind):
    """Select the family's positives and read exactly one MI row for each.

    Returns (groups, values, csv_sha256, report) where `groups` maps a group key
    tuple to its sorted model names and `values` maps model name to [ixt, ity].
    """
    import math

    declared = scenarios_from_plan_dir(family["plan_dir"])
    raw = family["mi_csv"].read_bytes()
    fields = [f"I(X;T)-{mi_kind}", f"I(T;Y)-{mi_kind}"]

    values, groups, seen_scenarios, table_scenarios = {}, {}, set(), set()
    outside = 0
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    require(reader.fieldnames is not None, f"Empty MI table: {family['mi_csv']}")
    for column in tuple(family["group_by"]) + tuple(fields) + ("model_name", "in_size", "bins"):
        require(column in reader.fieldnames,
                f"{family['label']}: column {column!r} missing from {family['mi_csv']}")
    filters = {c: {str(v) for v in vals} for c, vals in (family.get("filters") or {}).items()}
    for column in filters:
        require(column in reader.fieldnames,
                f"{family['label']}: filter column {column!r} missing from {family['mi_csv']}")
    filtered_out = 0
    for row in reader:
        if float(row["in_size"]) != in_size or float(row["bins"]) != bins:
            continue
        table_scenarios.add(row["Scenario"])
        if row["Scenario"] not in declared:
            outside += 1
            continue
        if any(row[c] not in allowed for c, allowed in filters.items()):
            filtered_out += 1
            continue
        seen_scenarios.add(row["Scenario"])
        name = row["model_name"]
        require(name not in values,
                f"{family['label']}: duplicate MI row for {name}, {in_size}, {bins}")
        vector = [float(row[f]) for f in fields]
        require(all(math.isfinite(v) for v in vector),
                f"{family['label']}: nonfinite MI for {name}")
        values[name] = vector
        groups.setdefault(tuple(row[c] for c in family["group_by"]), []).append(name)

    empty = sorted(set(declared) - seen_scenarios)
    shown_filters = {c: sorted(v) for c, v in filters.items()}
    require(values,
            f"{family['label']}: no positive rows at in_size={in_size}, bins={bins}.\n"
            f"  plan dir   : {family['plan_dir']}\n"
            f"  declares   : {sorted(declared)}\n"
            f"  table has  : {sorted(table_scenarios)}\n"
            f"  filters    : {shown_filters}  (removed {filtered_out} row(s))\n"
            f"  table      : {family['mi_csv']}\n"
            "  Point plan_dir at the plans that produced this table, relax the "
            "filters, or compute the declared scenarios first.")
    for key in groups:
        groups[key] = sorted(groups[key])
    report = {"declared_scenarios": sorted(declared), "scenarios_without_rows": empty,
              "rows_outside_declared_scenarios": outside, "n_positive": len(values),
              "filters": {c: sorted(v) for c, v in filters.items()},
              "rows_removed_by_filters": filtered_out,
              "plans": {k: str(v) for k, v in declared.items()}}
    return groups, values, digest(raw), report


def evaluate_positives(document, families, h0_values, h0_hash, *, output_dir,
                       in_size, bins, mi_kind, alphas, in_size_rate=None,
                       training_sizes=None, victims=None, victim_hash=None,
                       round_ids=None, k_values=None, round_by_h0_k=None):
    """Score positives on frozen references, optionally selecting existing rounds/k.

    Defaults preserve the full run. Selection never changes the manifest or
    renumbers round IDs. Single-round results have no sample split SD.
    """
    import numpy as np
    import scipy

    payload = document["payload"]
    rounds = list(range(payload["n_rounds"])) if round_ids is None else list(round_ids)
    ks = list(payload["k_values"]) if k_values is None else list(k_values)
    require(bool(rounds) and all(type(r) is int and 0 <= r < payload["n_rounds"] for r in rounds)
            and len(set(rounds)) == len(rounds), "Invalid or duplicate round_ids")
    require(bool(ks) and all(type(k) is int and k in payload["k_values"] for k in ks)
            and len(set(ks)) == len(ks), "Invalid or duplicate k_values")
    if round_by_h0_k is not None:
        require(round_ids is None, "Cannot combine round_ids with round_by_h0_k")
        expected = {(family["h0_scenario"], k) for family in families for k in ks}
        require(set(round_by_h0_k) == expected,
                "round_by_h0_k must cover every selected H0 scenario/k cell exactly")
        require(all(type(r) is int and 0 <= r < payload["n_rounds"]
                    for r in round_by_h0_k.values()), "Invalid selected round ID")
    sizes = in_size if isinstance(in_size, dict) else {c: in_size for c in payload["cases"]}
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)

    metadata = {
        "status": "running", "manifest_sha256": document["manifest_sha256"],
        "h0_csv": str(Path(H0_CSV).resolve()), "h0_csv_sha256": h0_hash,
        "script_sha256": digest(Path(__file__).read_bytes()),
        "model_root": str(Path(ROOT).resolve()),
        "in_sizes_by_case": sizes, "in_size_rate": in_size_rate,
        "training_sizes": training_sizes or {}, "bins": bins, "mi_kind": mi_kind,
        "alphas": alphas, "ddof": 1, "shrinkage": None,
        "round_ids": rounds if round_by_h0_k is None else None,
        "k_values": ks, "n_rounds": len(rounds) if round_by_h0_k is None else 1,
        "gate1": victims is not None,
        "laws": {"chi2": "p = sf(stat_chi2, df=2), stat_chi2 = T2",
                 "F": "p = sf(stat_F, dfn=2, dfd=k-2), "
                      "stat_F = (k-2)/(2(k-1)) * T2"},
        "families": [{"label": f["label"], "plan_dir": str(f["plan_dir"]),
                      "mi_csv": str(f["mi_csv"].resolve()),
                      "mi_csv_sha256": f["_hash"], "h0_scenario": f["h0_scenario"],
                      "group_by": list(f["group_by"]), **f["_report"]}
                     for f in families],
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            f"Mean TPR over {len(rounds) if round_by_h0_k is None else 1} selected paired reference split(s) of a fixed pool. "
            "SD is split variability, not a confidence interval, and is unavailable for one round. "
            "The positive set is identical in every selected round."),
    }
    if round_by_h0_k is not None:
        metadata["selected_round_by_h0_k"] = {
            scenario: {str(k): round_by_h0_k[(scenario, k)] for k in ks}
            for scenario in sorted({family["h0_scenario"] for family in families})}
    if victims is not None:
        metadata.update({"victim_csv": str(Path(VICTIM_CSV).resolve()),
                         "victim_csv_sha256": victim_hash,
                         "victim_mi_by_case": victims,
                         "gate1_rule": "pass iff I(X;T) >= victim and I(T;Y) <= victim "
                                       "(non-strict); failures get T2=0, p=1"})
    write_json(out / "run_metadata.json", metadata)
    write_json(out / "manifest.json", document)

    model_fields = ["manifest_sha256", "in_size_rate", "training_size", "in_size", "bins",
                    "mi_kind", "family", "group", "scenario", "model_name", "h0_scenario",
                    "round_id", "k_ref", "ixt", "ity", "T2",
                    "stat_chi2", "p_chi2", "stat_F", "p_F"]
    if victims is not None:
        model_fields.append("gate1")

    per_split, summary = [], []
    try:
        with (out / "per_model.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=model_fields)
            writer.writeheader()
            for family in families:
                h0_scenario = family["h0_scenario"]
                training_size = (training_sizes or {}).get(h0_scenario)
                probe = sizes[h0_scenario]
                names = sorted(family["_values"])
                matrix = np.array([family["_values"][n] for n in names], dtype=float)
                index = {n: i for i, n in enumerate(names)}
                for k in ks:
                    cell_rounds = (rounds if round_by_h0_k is None
                                   else [round_by_h0_k[(h0_scenario, k)]])
                    first = len(per_split)
                    cell_p = {key: {"chi2": [], "F": []}
                              for key in family["_groups"]}
                    for r in cell_rounds:
                        split = get_split(document, h0_scenario, r, k)
                        reference = [h0_values[m["model_name"]] for m in split["h0"]]
                        try:
                            mu, cov, t2, p_chi2, p_f = score(reference, matrix)
                            diag = diagnostics(reference, t2)
                            flags = None
                            if victims is not None:
                                t2, p_chi2, p_f, flags = apply_gate1(
                                    matrix, victims[h0_scenario], t2, p_chi2, p_f)
                        except (ValueError, np.linalg.LinAlgError) as exc:
                            raise ValueError(
                                f"{family['label']}, round={r}, N={k}: {exc}") from exc
                        stat_chi2, stat_f = law_statistics(t2, k)
                        for group_key, members in family["_groups"].items():
                            picks = [index[n] for n in members]
                            cell_p[group_key]["chi2"].append(
                                np.asarray(p_chi2, dtype=float)[picks])
                            cell_p[group_key]["F"].append(
                                np.asarray(p_f, dtype=float)[picks])
                        for i, name in enumerate(names):
                            writer.writerow(dict(zip(model_fields, [
                                document["manifest_sha256"], in_size_rate, training_size,
                                probe, bins, mi_kind, family["label"],
                                family["_group_of"][name], family["_scenario_of"][name],
                                name, h0_scenario, r, k,
                                matrix[i, 0], matrix[i, 1], float(t2[i]),
                                float(stat_chi2[i]), float(p_chi2[i]),
                                float(stat_f[i]), float(p_f[i]),
                                *(["pass" if flags[i] else "fail"] if flags is not None else []),
                            ])))
                        for group_key, members in sorted(family["_groups"].items()):
                            rows = [index[n] for n in members]
                            record = {
                                "manifest_sha256": document["manifest_sha256"],
                                "in_size_rate": in_size_rate, "training_size": training_size,
                                "in_size": probe, "bins": bins, "mi_kind": mi_kind,
                                "family": family["label"], "group": "|".join(group_key),
                                "h0_scenario": h0_scenario, "round_id": r, "k_ref": k,
                                "n_positive": len(members),
                                "h0_seeds": json.dumps([m["seed"] for m in split["h0"]]),
                                "mu": json.dumps(mu.tolist()),
                                "covariance": json.dumps(cov.tolist()),
                                "condition_number": float(np.linalg.cond(cov)),
                            }
                            # Reference-side diagnostics only: the suspect KS
                            # statistics of the negative driver assume null
                            # draws, which positives are not.
                            record.update({k2: v for k2, v in diag.items()
                                           if not k2.startswith("susp_")})
                            if flags is not None:
                                record["n_gate1_fail"] = int(
                                    sum(1 for i in rows if not flags[i]))
                            for law, probs in [("chi2", p_chi2), ("F", p_f)]:
                                for alpha in alphas:
                                    hits = int(sum(1 for i in rows if probs[i] < alpha))
                                    record[f"ntp_{law}@{alpha}"] = hits
                                    record[f"tpr_{law}@{alpha}"] = hits / len(members)
                            per_split.append(record)
                    for group_key in sorted(family["_groups"]):
                        label = "|".join(group_key)
                        block = [row for row in per_split[first:] if row["group"] == label]
                        agg = {"manifest_sha256": document["manifest_sha256"],
                               "family": family["label"], "group": label,
                               "h0_scenario": h0_scenario, "in_size_rate": in_size_rate,
                               "training_size": training_size, "in_size": probe,
                               "bins": bins, "mi_kind": mi_kind, "k_ref": k,
                               "n_rounds": len(cell_rounds),
                               "n_positive": block[0]["n_positive"]}
                        for law in ["chi2", "F"]:
                            for alpha in alphas:
                                rates = [row[f"tpr_{law}@{alpha}"] for row in block]
                                agg[f"mean_tpr_{law}@{alpha}"] = float(np.mean(rates))
                                agg[f"std_tpr_{law}@{alpha}"] = (
                                    float(np.std(rates, ddof=1)) if len(rates) > 1 else None)
                        agg.update(summary_statistics(cell_p[group_key]))
                        if victims is not None:
                            agg["mean_n_gate1_fail"] = float(
                                np.mean([row["n_gate1_fail"] for row in block]))
                        summary.append(agg)
                    print(f"Completed {family['label']}, N={k}: "
                          f"{len(cell_rounds)} rounds x {len(names)} positives", flush=True)
        for name, rows in [("per_split.csv", per_split), ("summary.csv", summary)]:
            with (out / name).open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        metadata["status"] = "complete"
        metadata["completed_splits"] = len(per_split)
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = str(exc)
        raise
    finally:
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (out / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Results: {out}")
    return summary


def main():
    """Run the operation selected in the global configuration; no CLI arguments."""
    require(RUN_MODE in ("validate", "evaluate"),
            "RUN_MODE must be 'validate' or 'evaluate'")
    require(MI_KIND in ("In", "Out"), "MI_KIND must be 'In' or 'Out'")
    require(type(APPLY_GATE1) is bool, "APPLY_GATE1 must be True or False")
    require(bool(ALPHAS) and len(set(ALPHAS)) == len(ALPHAS)
            and all(0 < a < 1 for a in ALPHAS),
            "ALPHAS must be nonempty, unique and between 0 and 1")
    require(bool(POSITIVE_FAMILIES), "POSITIVE_FAMILIES is empty")
    families_cfg = [normalize_family(f) for f in POSITIVE_FAMILIES]
    labels = [f["label"] for f in families_cfg]
    require(len(set(labels)) == len(labels), "Positive family labels must be unique")

    document = load_manifest(Path(MANIFEST))
    cases = document["payload"]["cases"]
    for family in families_cfg:
        require(family["h0_scenario"] in cases,
                f"{family['label']}: h0_scenario {family['h0_scenario']!r} is not a "
                f"manifest case; available: {sorted(cases)}")
    configurations = rate_configuration_grid(
        IN_SIZE_RATES, BINS, TRAINING_SIZES, list(cases))

    output_root = Path(OUTPUT_DIR)
    combined_path = output_root / "summary_all_configs.csv"
    if RUN_MODE == "evaluate":
        require(not combined_path.exists(),
                f"Combined summary already exists; choose a new OUTPUT_DIR: {combined_path}")

    prepared = []
    for rate, sizes, bins in configurations:
        rate_tag = format(Decimal(str(rate)).normalize(), "f")
        destination = output_root / f"{MI_KIND}_rate{rate_tag}_bins{bins}"
        if RUN_MODE == "evaluate":
            require(not destination.exists(),
                    f"Result directory already exists: {destination}")
        # The reference side is validated exactly as the negative driver does.
        h0_values, h0_hash = preflight(document, Path(ROOT), Path(H0_CSV),
                                       sizes, bins, MI_KIND)
        victims = victim_hash = None
        if APPLY_GATE1:
            victims, victim_hash = preflight_victim(
                document, Path(VICTIM_CSV), sizes, bins, MI_KIND)
        resolved = []
        for family in families_cfg:
            probe = sizes[family["h0_scenario"]]
            groups, values, family_hash, report = preflight_positives(
                family, probe, bins, MI_KIND)
            group_of = {n: "|".join(key) for key, members in groups.items() for n in members}
            scenario_of = {n: key[0] for key, members in groups.items() for n in members}
            resolved.append({**family, "_groups": groups, "_values": values,
                             "_hash": family_hash, "_report": report,
                             "_group_of": group_of, "_scenario_of": scenario_of})
            note = ""
            if report["scenarios_without_rows"]:
                note += f"; NO ROWS for {report['scenarios_without_rows']}"
            if report["rows_outside_declared_scenarios"]:
                note += (f"; ignored {report['rows_outside_declared_scenarios']} row(s) "
                         "outside the declared scenarios")
            print(f"  [{family['label']}] {len(values)} positive(s) in "
                  f"{len(groups)} group(s) from {len(report['declared_scenarios'])} "
                  f"declared scenario(s){note}")
        if prepared:
            require(h0_hash == prepared[0]["h0_hash"],
                    "H0 MI CSV changed during preflight; retry with a stable input")
        prepared.append({"rate": rate, "sizes": sizes, "bins": bins,
                         "h0_values": h0_values, "h0_hash": h0_hash,
                         "families": resolved, "destination": destination,
                         "victims": victims, "victim_hash": victim_hash})
        print(f"Validated rate={rate:.0%}, sizes={sizes[list(cases)[0]]}, bins={bins}; "
              f"manifest={document['manifest_sha256']}"
              + ("; gate1 ON" if victims else ""))

    if RUN_MODE == "validate":
        print(f"\nValidation only: {len(configurations)} configuration(s) ready.")
        return

    combined = []
    for item in prepared:
        combined.extend(evaluate_positives(
            document, item["families"], item["h0_values"], item["h0_hash"],
            output_dir=item["destination"], in_size=item["sizes"], bins=item["bins"],
            mi_kind=MI_KIND, alphas=ALPHAS, in_size_rate=item["rate"],
            training_sizes=TRAINING_SIZES, victims=item["victims"],
            victim_hash=item["victim_hash"],
        ))
    with combined_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    print(f"Completed {len(configurations)} configurations. "
          f"Combined summary: {combined_path}")


if __name__ == "__main__":
    main()
