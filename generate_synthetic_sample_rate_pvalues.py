"""Densify measured sample-rate curves without recomputing MI.

Scope
-----
Only the 24 measured combinations used by the four-curve figures are kept:
CF10/CF100 x RN18/VGG16/DeiT x FT-AL/P-20%/DKD/Knockoff. DKD and Knockoff
are exact same-architecture cases. The 1% observation is intentionally
excluded. Real anchors at 5, 10, 20, 50, 75 and 100 percent are copied
unchanged; missing 5-point-grid rates are interpolated per measured model and
per k in log(p) space.

For the exact predictive law used by the source experiment,

    p = SF_F((k-2)/(2(k-1)) * T2; 2, k-2)

the dfn=2 survival function has a closed form, so the inverse is

    T2 = (k-1) * (p**(-2/(k-2)) - 1).

No synthetic MI values, model identities, seeds, cross-architecture cases, or
random perturbations are introduced.
"""

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


BASE = Path(__file__).resolve().parent
SOURCE_ROOT = BASE / "saved_logs/vanilla/Hypo_Test_UnknownArch_RawMI_Sweeps_BestFPR_PerK"
OUTPUT_ROOT = BASE / "saved_logs/vanilla/Hypo_Test_UnknownArch_RawMI_SyntheticSampleRates_BestFPR_PerK"
STAGING_ROOT = OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + ".staging")

DATASETS = ("CF10", "CF100")
ARCHITECTURES = ("RN18", "VGG16", "DeiT")
POSITIVE_FORMS = ("FT-AL", "P-20%", "DKD", "Knockoff")
K_VALUES = (5, 10, 15, 20, 25, 30)
ANCHOR_RATES = (5, 10, 20, 50, 75, 100)
TARGET_RATES = tuple(range(5, 101, 5))
SYNTHETIC_RATES = tuple(rate for rate in TARGET_RATES if rate not in ANCHOR_RATES)
TRAINING_SIZE = 25000
FIXED_BINS = 50
ALPHA = 0.01

EXPECTED_MODEL_COUNTS = {
    ("CF10", "RN18", "FT-AL"): 3,
    ("CF10", "RN18", "P-20%"): 3,
    ("CF10", "RN18", "DKD"): 1,
    ("CF10", "RN18", "Knockoff"): 5,
    ("CF10", "VGG16", "FT-AL"): 3,
    ("CF10", "VGG16", "P-20%"): 3,
    ("CF10", "VGG16", "DKD"): 1,
    ("CF10", "VGG16", "Knockoff"): 3,
    ("CF10", "DeiT", "FT-AL"): 3,
    ("CF10", "DeiT", "P-20%"): 3,
    ("CF10", "DeiT", "DKD"): 1,
    ("CF10", "DeiT", "Knockoff"): 3,
    ("CF100", "RN18", "FT-AL"): 3,
    ("CF100", "RN18", "P-20%"): 5,
    ("CF100", "RN18", "DKD"): 1,
    ("CF100", "RN18", "Knockoff"): 3,
    ("CF100", "VGG16", "FT-AL"): 3,
    ("CF100", "VGG16", "P-20%"): 3,
    ("CF100", "VGG16", "DKD"): 1,
    ("CF100", "VGG16", "Knockoff"): 3,
    ("CF100", "DeiT", "FT-AL"): 3,
    ("CF100", "DeiT", "P-20%"): 3,
    ("CF100", "DeiT", "DKD"): 1,
    ("CF100", "DeiT", "Knockoff"): 3,
}

PER_MODEL_FIELDS = (
    "source_manifest_sha256", "dataset", "operating_point_id",
    "in_size_rate", "training_size", "in_size", "bins", "mi_kind",
    "family", "case_id", "positive_form", "group", "case_parameters",
    "suspect_architecture", "suspect_scenario", "k_ref", "seed",
    "model_name", "p_F_max", "stat_F", "T2_composite_min",
    "T2_at_max_p_F", "reject_F@0.01", "decision_F",
    "classification_outcome_F", "data_origin", "interpolation_method",
    "lower_anchor_rate", "upper_anchor_rate", "lower_anchor_p_F_max",
    "upper_anchor_p_F_max", "source_operating_point_id",
)

SUMMARY_FIELDS = (
    "source_manifest_sha256", "dataset", "operating_point_id",
    "in_size_rate", "training_size", "in_size", "bins", "mi_kind",
    "family", "case_id", "positive_form", "group", "case_parameters",
    "suspect_architecture", "suspect_scenario", "k_ref", "truth",
    "data_origin", "n_suspects", "n_rejected_F@0.01",
    "rejection_rate_F@0.01", "ntp_F@0.01", "tpr_F@0.01",
    "fnr_F@0.01", "T2_composite_min_min", "T2_composite_min_p05",
    "T2_composite_min_median", "T2_composite_min_p95",
    "T2_composite_min_max", "T2_composite_min_mean", "p_F_max_min",
    "p_F_max_p05", "p_F_max_median", "p_F_max_p95", "p_F_max_max",
    "p_F_max_mean",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    require(rows, f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classify_case(case_id):
    if re.search(r"_Same_25000\|strategy=FT-AL$", case_id):
        return "FT-AL"
    if re.search(r"_Same_25000\|sparsity=0\.2\|ckpt_kind=best$", case_id):
        return "P-20%"
    if re.search(r"(ResNet-18to18|VGG16toVGG16|DeiTtoDeiT)_25000\|method=DKD$",
                 case_id):
        return "DKD"
    if re.search(r"Knockoff_Same(10|100)_Same(18|16|DeiT)$", case_id):
        return "Knockoff"
    return None


def inverse_exact_f_p_to_t2(p_value, k):
    require(0.0 < p_value <= 1.0, f"p outside (0,1]: {p_value}")
    require(k > 2, f"Exact F law requires k>2: {k}")
    return (k - 1.0) * math.expm1((-2.0 / (k - 2.0)) * math.log(p_value))


def exact_f_p_from_t2(t2, k):
    require(t2 >= 0.0 and math.isfinite(t2), f"Invalid T2: {t2}")
    return math.exp(-0.5 * (k - 2.0) * math.log1p(t2 / (k - 1.0)))


def percentile_summary(values, prefix):
    array = np.asarray(values, dtype=float)
    require(array.size > 0 and np.isfinite(array).all(), f"Invalid {prefix} values")
    p05, median, p95 = np.quantile(array, [0.05, 0.5, 0.95])
    return {
        f"{prefix}_min": float(array.min()),
        f"{prefix}_p05": float(p05),
        f"{prefix}_median": float(median),
        f"{prefix}_p95": float(p95),
        f"{prefix}_max": float(array.max()),
        f"{prefix}_mean": float(array.mean()),
    }


def anchors_around(rate):
    lower = max(anchor for anchor in ANCHOR_RATES if anchor <= rate)
    upper = min(anchor for anchor in ANCHOR_RATES if anchor >= rate)
    return lower, upper


def validate_source_case_inventory(selected_rows):
    cases_by_combo = defaultdict(set)
    models_by_combo = defaultdict(set)
    for row in selected_rows:
        combo = (row["dataset"], row["suspect_architecture"], row["positive_form"])
        cases_by_combo[combo].add(row["case_id"])
        models_by_combo[combo].add((row["model_name"], row["seed"]))
    require(set(cases_by_combo) == set(EXPECTED_MODEL_COUNTS),
            f"Wrong 24-combination inventory: {sorted(set(EXPECTED_MODEL_COUNTS)-set(cases_by_combo))}")
    for combo, expected in EXPECTED_MODEL_COUNTS.items():
        require(len(cases_by_combo[combo]) == 1,
                f"Combination maps to multiple cases: {combo}, {cases_by_combo[combo]}")
        require(len(models_by_combo[combo]) == expected,
                f"Wrong model count for {combo}: {len(models_by_combo[combo])} != {expected}")


def load_selected_source_rows():
    selected = []
    source_info = {}
    for dataset in DATASETS:
        path = SOURCE_ROOT / dataset / "positives/per_model.csv"
        rows = read_csv(path)
        source_info[dataset] = {"path": str(path.resolve()), "sha256": sha256(path)}
        for row in rows:
            positive_form = classify_case(row["case_id"])
            if positive_form is None:
                continue
            if row["suspect_architecture"] not in ARCHITECTURES:
                continue
            if row["in_sample_sweep"] != "True" or int(row["bins"]) != FIXED_BINS:
                continue
            rate = int(round(float(row["in_size_rate"]) * 100.0))
            if rate not in ANCHOR_RATES:
                continue
            copied = dict(row)
            copied["positive_form"] = positive_form
            copied["rate_pct"] = rate
            selected.append(copied)
    validate_source_case_inventory(selected)
    return selected, source_info


def validate_real_law(row):
    k = int(row["k_ref"])
    p_value = float(row["p_F_max"])
    t2 = float(row["T2_at_max_p_F"])
    inverse = inverse_exact_f_p_to_t2(p_value, k)
    require(math.isclose(inverse, t2, rel_tol=2e-12, abs_tol=2e-12),
            f"Source p/T2 law mismatch: {row['case_id']}, {row['model_name']}, "
            f"k={k}, rate={row['rate_pct']}, inverse={inverse}, source={t2}")


def densify_per_model(source_rows):
    groups = defaultdict(list)
    for row in source_rows:
        validate_real_law(row)
        key = (row["dataset"], row["case_id"], int(row["k_ref"]),
               row["model_name"], row["seed"])
        groups[key].append(row)

    expected_groups = sum(EXPECTED_MODEL_COUNTS.values()) * len(K_VALUES)
    require(len(groups) == expected_groups,
            f"Wrong model-k group count: {len(groups)} != {expected_groups}")

    dense = []
    for key, rows in sorted(groups.items()):
        by_rate = {int(row["rate_pct"]): row for row in rows}
        require(set(by_rate) == set(ANCHOR_RATES),
                f"Missing/duplicate anchors for {key}: {sorted(by_rate)}")
        template = rows[0]
        for rate in TARGET_RATES:
            lower_rate, upper_rate = anchors_around(rate)
            lower_row, upper_row = by_rate[lower_rate], by_rate[upper_rate]
            lower_p = float(lower_row["p_F_max"])
            upper_p = float(upper_row["p_F_max"])
            require(lower_p > 0.0 and upper_p > 0.0,
                    f"Cannot interpolate nonpositive p for {key}")

            if rate in by_rate:
                source = by_rate[rate]
                p_text = source["p_F_max"]
                t2_text = source["T2_at_max_p_F"]
                p_value = float(p_text)
                t2 = float(t2_text)
                origin = "real"
                method = "measured"
                source_operating_point = source["operating_point_id"]
            else:
                weight = (rate - lower_rate) / (upper_rate - lower_rate)
                log_p = ((1.0 - weight) * math.log(lower_p)
                         + weight * math.log(upper_p))
                p_value = math.exp(log_p)
                t2 = inverse_exact_f_p_to_t2(p_value, int(template["k_ref"]))
                p_text = repr(p_value)
                t2_text = repr(t2)
                origin = "synthetic"
                method = "piecewise_linear_log_p"
                source_operating_point = ""
                lower_log, upper_log = math.log(lower_p), math.log(upper_p)
                require(min(lower_log, upper_log) <= log_p <= max(lower_log, upper_log),
                        f"Synthetic log-p overshoot for {key}, rate={rate}")

            k = int(template["k_ref"])
            roundtrip_p = exact_f_p_from_t2(t2, k)
            require(math.isclose(roundtrip_p, p_value, rel_tol=2e-12, abs_tol=0.0),
                    f"p/T2 roundtrip failed for {key}, rate={rate}")
            reject = p_value < ALPHA
            in_size = TRAINING_SIZE * rate // 100
            dense.append({
                "source_manifest_sha256": template["manifest_sha256"],
                "dataset": template["dataset"],
                "operating_point_id": f"N{in_size}_B{FIXED_BINS}",
                "in_size_rate": rate / 100.0,
                "training_size": TRAINING_SIZE,
                "in_size": in_size,
                "bins": FIXED_BINS,
                "mi_kind": template["mi_kind"],
                "family": template["family"],
                "case_id": template["case_id"],
                "positive_form": template["positive_form"],
                "group": template["group"],
                "case_parameters": template["case_parameters"],
                "suspect_architecture": template["suspect_architecture"],
                "suspect_scenario": template["suspect_scenario"],
                "k_ref": k,
                "seed": template["seed"],
                "model_name": template["model_name"],
                "p_F_max": p_text,
                "stat_F": t2 * (k - 2.0) / (2.0 * (k - 1.0)),
                "T2_composite_min": t2_text,
                "T2_at_max_p_F": t2_text,
                "reject_F@0.01": reject,
                "decision_F": "stolen" if reject else "independent",
                "classification_outcome_F": "true_positive" if reject else "false_negative",
                "data_origin": origin,
                "interpolation_method": method,
                "lower_anchor_rate": lower_rate / 100.0,
                "upper_anchor_rate": upper_rate / 100.0,
                "lower_anchor_p_F_max": repr(lower_p),
                "upper_anchor_p_F_max": repr(upper_p),
                "source_operating_point_id": source_operating_point,
            })
    return dense


def summarize(dense_rows):
    groups = defaultdict(list)
    for row in dense_rows:
        key = (row["dataset"], row["case_id"], int(row["k_ref"]),
               float(row["in_size_rate"]))
        groups[key].append(row)

    summaries = []
    for key, rows in sorted(groups.items()):
        template = rows[0]
        origins = {row["data_origin"] for row in rows}
        require(len(origins) == 1, f"Mixed real/synthetic summary cell: {key}")
        p_values = [float(row["p_F_max"]) for row in rows]
        t2_values = [float(row["T2_composite_min"]) for row in rows]
        rejected = sum(value < ALPHA for value in p_values)
        n = len(rows)
        summaries.append({
            "source_manifest_sha256": template["source_manifest_sha256"],
            "dataset": template["dataset"],
            "operating_point_id": template["operating_point_id"],
            "in_size_rate": template["in_size_rate"],
            "training_size": template["training_size"],
            "in_size": template["in_size"],
            "bins": template["bins"],
            "mi_kind": template["mi_kind"],
            "family": template["family"],
            "case_id": template["case_id"],
            "positive_form": template["positive_form"],
            "group": template["group"],
            "case_parameters": template["case_parameters"],
            "suspect_architecture": template["suspect_architecture"],
            "suspect_scenario": template["suspect_scenario"],
            "k_ref": template["k_ref"],
            "truth": "positive",
            "data_origin": next(iter(origins)),
            "n_suspects": n,
            "n_rejected_F@0.01": rejected,
            "rejection_rate_F@0.01": rejected / n,
            "ntp_F@0.01": rejected,
            "tpr_F@0.01": rejected / n,
            "fnr_F@0.01": 1.0 - rejected / n,
            **percentile_summary(t2_values, "T2_composite_min"),
            **percentile_summary(p_values, "p_F_max"),
        })
    return summaries


def validate_dense(source_rows, dense_rows, summaries):
    expected_real = sum(EXPECTED_MODEL_COUNTS.values()) * len(K_VALUES) * len(ANCHOR_RATES)
    expected_synthetic = (sum(EXPECTED_MODEL_COUNTS.values()) * len(K_VALUES)
                          * len(SYNTHETIC_RATES))
    require(sum(row["data_origin"] == "real" for row in dense_rows) == expected_real,
            "Wrong real-row count")
    require(sum(row["data_origin"] == "synthetic" for row in dense_rows) == expected_synthetic,
            "Wrong synthetic-row count")
    require(len(dense_rows) == sum(EXPECTED_MODEL_COUNTS.values()) * len(K_VALUES)
            * len(TARGET_RATES), "Wrong dense per-model row count")
    require(len(summaries) == len(EXPECTED_MODEL_COUNTS) * len(K_VALUES)
            * len(TARGET_RATES), "Wrong dense summary row count")

    source_map = {
        (row["dataset"], row["case_id"], int(row["k_ref"]), row["model_name"],
         row["seed"], int(row["rate_pct"])): row
        for row in source_rows
    }
    for row in dense_rows:
        p_value = float(row["p_F_max"])
        t2 = float(row["T2_at_max_p_F"])
        require(0.0 < p_value <= 1.0 and math.isfinite(p_value), "Invalid dense p")
        require(t2 >= 0.0 and math.isfinite(t2), "Invalid dense T2")
        if row["data_origin"] == "real":
            key = (row["dataset"], row["case_id"], int(row["k_ref"]),
                   row["model_name"], row["seed"],
                   int(round(float(row["in_size_rate"]) * 100.0)))
            source = source_map[key]
            require(row["p_F_max"] == source["p_F_max"],
                    f"Real p changed: {key}")
            require(row["T2_at_max_p_F"] == source["T2_at_max_p_F"],
                    f"Real T2 changed: {key}")

    summary_map = defaultdict(dict)
    for row in summaries:
        key = (row["dataset"], row["suspect_architecture"], row["positive_form"],
               int(row["k_ref"]))
        rate = int(round(float(row["in_size_rate"]) * 100.0))
        summary_map[key][rate] = row
    require(all(set(curve) == set(TARGET_RATES) for curve in summary_map.values()),
            "Incomplete dense summary curve")

    # Each synthetic aggregate mean must remain inside its two real aggregate
    # endpoint means. This verifies the final plotted curves gain no new extrema.
    for key, curve in summary_map.items():
        for rate in SYNTHETIC_RATES:
            lower, upper = anchors_around(rate)
            for field in ("p_F_max_mean", "T2_composite_min_mean"):
                value = float(curve[rate][field])
                endpoints = [float(curve[lower][field]), float(curve[upper][field])]
                lo, hi = min(endpoints), max(endpoints)
                tolerance = max(abs(lo), abs(hi), 1.0) * 2e-12
                require(lo - tolerance <= value <= hi + tolerance,
                        f"Aggregate shape overshoot: {key}, rate={rate}, field={field}, "
                        f"value={value}, endpoints={endpoints}")


def manifest_rows(dense_rows):
    by_combo = defaultdict(lambda: {"case_ids": set(), "models": set()})
    for row in dense_rows:
        combo = (row["dataset"], row["suspect_architecture"], row["positive_form"])
        by_combo[combo]["case_ids"].add(row["case_id"])
        by_combo[combo]["models"].add((row["model_name"], row["seed"]))
    rows = []
    for combo, info in sorted(by_combo.items()):
        require(len(info["case_ids"]) == 1, f"Multiple cases in manifest combo: {combo}")
        rows.append({
            "dataset": combo[0],
            "suspect_architecture": combo[1],
            "positive_form": combo[2],
            "case_id": next(iter(info["case_ids"])),
            "n_real_models": len(info["models"]),
            "k_values": json.dumps(K_VALUES, separators=(",", ":")),
            "real_anchor_rates_percent": json.dumps(ANCHOR_RATES, separators=(",", ":")),
            "synthetic_rates_percent": json.dumps(SYNTHETIC_RATES, separators=(",", ":")),
            "target_rates_percent": json.dumps(TARGET_RATES, separators=(",", ":")),
        })
    return rows


def main():
    require(not OUTPUT_ROOT.exists(), f"Output exists; refusing overwrite: {OUTPUT_ROOT}")
    require(not STAGING_ROOT.exists(), f"Staging exists; inspect first: {STAGING_ROOT}")
    source_rows, source_info = load_selected_source_rows()
    dense_rows = densify_per_model(source_rows)
    summaries = summarize(dense_rows)
    validate_dense(source_rows, dense_rows, summaries)
    cases = manifest_rows(dense_rows)

    STAGING_ROOT.mkdir(parents=True)
    for dataset in DATASETS:
        dataset_dense = [row for row in dense_rows if row["dataset"] == dataset]
        dataset_summary = [row for row in summaries if row["dataset"] == dataset]
        out = STAGING_ROOT / dataset / "positives"
        write_csv(out / "per_model_sample_rate_dense.csv", dataset_dense, PER_MODEL_FIELDS)
        write_csv(out / "summary_sample_rate_dense.csv", dataset_summary, SUMMARY_FIELDS)
    write_csv(STAGING_ROOT / "case_manifest.csv", cases, tuple(cases[0]))

    metadata = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_info,
        "scope": {
            "datasets": list(DATASETS),
            "architectures": list(ARCHITECTURES),
            "positive_forms": list(POSITIVE_FORMS),
            "same_arch_only": ["DKD", "Knockoff"],
            "n_base_combinations": len(EXPECTED_MODEL_COUNTS),
            "n_real_models": sum(EXPECTED_MODEL_COUNTS.values()),
            "k_values": list(K_VALUES),
            "fixed_bins": FIXED_BINS,
            "training_size": TRAINING_SIZE,
            "excluded_rate_percent": 1,
            "real_anchor_rates_percent": list(ANCHOR_RATES),
            "synthetic_rates_percent": list(SYNTHETIC_RATES),
            "target_rates_percent": list(TARGET_RATES),
        },
        "method": {
            "interpolation": "per-real-model piecewise linear interpolation in natural-log p",
            "randomness": None,
            "real_anchor_policy": "copied exactly; never replaced or moved",
            "shape_constraint": "within each adjacent real-anchor interval; no overshoot",
            "inverse_exact_F": "T2=(k-1)*expm1((-2/(k-2))*log(p))",
            "forward_exact_F": "p=(1+T2/(k-1))^(-(k-2)/2)",
            "alpha": ALPHA,
        },
        "counts": {
            "per_model_real_rows": sum(row["data_origin"] == "real" for row in dense_rows),
            "per_model_synthetic_rows": sum(row["data_origin"] == "synthetic" for row in dense_rows),
            "per_model_total_rows": len(dense_rows),
            "summary_real_rows": sum(row["data_origin"] == "real" for row in summaries),
            "summary_synthetic_rows": sum(row["data_origin"] == "synthetic" for row in summaries),
            "summary_total_rows": len(summaries),
        },
        "important_limit": (
            "Synthetic rows interpolate hypothesis-test outputs only. They are not recomputed MI "
            "measurements and must remain labeled data_origin=synthetic."
        ),
    }
    with (STAGING_ROOT / "run_metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")

    STAGING_ROOT.replace(OUTPUT_ROOT)
    print(OUTPUT_ROOT)
    print(json.dumps(metadata["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
