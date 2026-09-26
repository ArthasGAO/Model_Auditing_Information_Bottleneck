"""Unknown-architecture Hotelling tests for every measured raw positive case.

This is the sweep counterpart of ``run_hypothesis_test_unknown_arch.py``.  It
keeps that experiment's statistical design unchanged:

* frozen Pool-0 manifest and evaluated-negative identities;
* the minimum-FPR round selected at (in_size=25000, bins=50) separately for
  every dataset / reference architecture / k;
* separate RN18, VGG16 and DeiT reference moments (never a pooled H0);
* composite union-null p-value ``max(p_RN18, p_VGG16, p_DeiT)``;
* exact predictive F decision at alpha=0.01, with Gate 1 disabled.

Only the positive source and operating-point grid differ.  The positives are
discovered from the four measured (non-``multiple``) master tables.  The grid
is the union of two one-dimensional sweeps:

* sample-size sweep: every measured in_size with bins fixed at 50;
* bin sweep: every measured bins value with in_size fixed at 25000.

The shared (25000, 50) anchor is evaluated once.  Detailed rows retain the
three component Hotelling T^2 values.  They additionally store
``T2_composite_min`` / ``T2_at_max_p_F``: because all three component tests use
the same k and F law, the maximum composite p-value is attained by the minimum
component T^2.  Summary rows include distributions of that composite T^2 and
of every architecture-specific T^2, rather than reporting p-values alone.

The established result directory is never modified.  A new run is written to
a staging directory, validated, and atomically promoted.
"""

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import run_hypothesis_test_fixed_splits as neg
import run_hypothesis_test_same_arch_best_fpr_per_k as same


BASE = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE / "saved_logs/vanilla/Hypo_Test_UnknownArch_RawMI_Sweeps_BestFPR_PerK"
STAGING_ROOT = OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + ".staging")

RAW_TABLES = {
    "ft": BASE / "saved_logs/ft_final/MI_master_table_ft.csv",
    "prune": BASE / "saved_logs/pruning_final/MI_master_table_prune.csv",
    "kd": BASE / "saved_logs/kd_final/MI_master_table_kd.csv",
    "extraction": BASE / "saved_logs/extraction_final/MI_master_table_extraction.csv",
}

IN_SIZES = (250, 1250, 2500, 5000, 12500, 18750, 25000)
BIN_VALUES = (5, 10, 15, 20, 30, 50, 75, 100, 150, 200)
FIXED_BINS_FOR_SAMPLE_SWEEP = 50
FIXED_IN_SIZE_FOR_BIN_SWEEP = 25000

ALPHA = same.ALPHA
K_VALUES = tuple(same.K_VALUES)
DATASETS = tuple(same.DATASETS)
CANDIDATE_ARCHITECTURES = ("RN18", "VGG16", "DeiT")
EXPECTED_CASE_COUNTS = {"ft": 21, "prune": 28, "kd": 24, "extraction": 24}
EXPECTED_MODEL_COUNTS = {"ft": 69, "prune": 100, "kd": 32, "extraction": 64}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_text(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=False)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    require(rows, f"Refusing to write an empty CSV: {path}")
    fields = list(rows[0])
    require(all(list(row) == fields for row in rows), f"Inconsistent CSV fields: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def operating_points():
    points = []
    for in_size in IN_SIZES:
        is_anchor = in_size == FIXED_IN_SIZE_FOR_BIN_SWEEP
        points.append({
            "operating_point_id": f"N{in_size}_B{FIXED_BINS_FOR_SAMPLE_SWEEP}",
            "sweep_axis": "anchor" if is_anchor else "sample_size",
            "in_size_rate": in_size / 25000.0,
            "in_size": in_size,
            "bins": FIXED_BINS_FOR_SAMPLE_SWEEP,
            "in_sample_sweep": True,
            "in_bin_sweep": is_anchor,
        })
    for bins in BIN_VALUES:
        if bins == FIXED_BINS_FOR_SAMPLE_SWEEP:
            continue
        points.append({
            "operating_point_id": f"N{FIXED_IN_SIZE_FOR_BIN_SWEEP}_B{bins}",
            "sweep_axis": "bin_size",
            "in_size_rate": 1.0,
            "in_size": FIXED_IN_SIZE_FOR_BIN_SWEEP,
            "bins": bins,
            "in_sample_sweep": False,
            "in_bin_sweep": True,
        })
    require(len(points) == len(IN_SIZES) + len(BIN_VALUES) - 1 == 16,
            "Operating-point union is not the expected 16 cells")
    require(len({(p["in_size"], p["bins"]) for p in points}) == len(points),
            "Duplicate operating point")
    return points


OPERATING_POINTS = operating_points()
POINT_BY_PAIR = {(p["in_size"], p["bins"]): p for p in OPERATING_POINTS}
FULL_RAW_GRID = {(n, b) for n in IN_SIZES for b in BIN_VALUES}


def distribution_summary(values, prefix):
    values = np.asarray(values, dtype=float)
    require(values.ndim == 1 and values.size > 0 and np.isfinite(values).all(),
            f"Invalid values for {prefix} summary")
    return {
        f"{prefix}_min": float(values.min()),
        f"{prefix}_p05": float(np.percentile(values, 5)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_mean": float(values.mean()),
    }


def dataset_of(scenario):
    if scenario.startswith("CIFAR-10_"):
        return "CF10"
    if scenario.startswith("CIFAR-100_"):
        return "CF100"
    raise RuntimeError(f"Cannot infer dataset from Scenario: {scenario}")


def normalize_architecture(value):
    value = str(value).lower()
    if "vgg16" in value:
        return "VGG16"
    if "deit" in value:
        return "DeiT"
    if ("resnet-18" in value or value.endswith("18") or "to18" in value
            or value.startswith("18_")):
        return "RN18"
    raise RuntimeError(f"Cannot normalize suspect architecture: {value}")


def case_identity(family, row):
    scenario = row["Scenario"]
    if family == "ft":
        params = {"strategy": row["strategy"]}
        group = f"{scenario}|strategy={row['strategy']}"
        arch_source = scenario
    elif family == "prune":
        params = {"sparsity": row["sparsity"], "ckpt_kind": row["ckpt_kind"]}
        group = (f"{scenario}|sparsity={row['sparsity']}|"
                 f"ckpt_kind={row['ckpt_kind']}")
        arch_source = scenario
    elif family == "kd":
        method = "DKD" if "_DKD_" in row["model_name"] else "KD"
        params = {"method": method}
        group = f"{scenario}|method={method}"
        arch_source = scenario.split("to", 1)[-1]
    elif family == "extraction":
        params = {"attack": row["attack"]}
        group = scenario
        arch_source = row["substitute_model"]
    else:
        raise RuntimeError(f"Unknown family: {family}")
    return {
        "case_id": f"{family}::{group}",
        "family": family,
        "group": group,
        "case_parameters": params,
        "dataset": dataset_of(scenario),
        "suspect_architecture": normalize_architecture(arch_source),
        "suspect_scenario": scenario,
    }


def discover_raw_cases():
    cases = {}
    table_hashes = {}
    counts = {}
    for family, path in RAW_TABLES.items():
        require("multiple" not in path.name.lower(), f"Synthetic/multiple table is forbidden: {path}")
        rows = read_csv(path)
        require(rows, f"Empty raw positive table: {path}")
        table_hashes[family] = sha256(path)
        per_model_pairs = defaultdict(set)
        seen_cells = set()
        family_case_ids = set()
        family_models = set()
        for row in rows:
            identity = case_identity(family, row)
            case_id = identity["case_id"]
            if case_id not in cases:
                cases[case_id] = {
                    **identity,
                    "source_csv": str(path),
                    "source_csv_sha256": table_hashes[family],
                    "models": {},
                    "values": defaultdict(dict),
                    "achieved_sparsity": set(),
                }
            case = cases[case_id]
            require(all(case[key] == identity[key] for key in
                        ("family", "group", "dataset", "suspect_architecture",
                         "suspect_scenario")), f"Case metadata drift: {case_id}")
            name = row["model_name"]
            pair = (int(row["in_size"]), int(row["bins"]))
            cell = (name, *pair)
            require(cell not in seen_cells, f"Duplicate raw MI cell: {family}, {cell}")
            seen_cells.add(cell)
            per_model_pairs[name].add(pair)
            seed = int(float(row["seed"]))
            prior = case["models"].setdefault(name, seed)
            require(prior == seed, f"Seed drift for {name}")
            vector = (float(row["I(X;T)-In"]), float(row["I(T;Y)-In"]))
            require(all(math.isfinite(value) for value in vector),
                    f"Nonfinite raw MI: {name}, {pair}")
            if pair in POINT_BY_PAIR:
                require(name not in case["values"][pair],
                        f"Duplicate selected operating point: {name}, {pair}")
                case["values"][pair][name] = vector
            if family == "prune":
                case["achieved_sparsity"].add(float(row["achieved_sparsity"]) / 100.0)
            family_case_ids.add(case_id)
            family_models.add(name)

        bad = {name: sorted(FULL_RAW_GRID - pairs)
               for name, pairs in per_model_pairs.items() if pairs != FULL_RAW_GRID}
        require(not bad, f"Incomplete raw MI grids in {family}: {list(bad.items())[:3]}")
        require(len(family_case_ids) == EXPECTED_CASE_COUNTS[family],
                f"{family}: expected {EXPECTED_CASE_COUNTS[family]} cases, got {len(family_case_ids)}")
        require(len(family_models) == EXPECTED_MODEL_COUNTS[family],
                f"{family}: expected {EXPECTED_MODEL_COUNTS[family]} models, got {len(family_models)}")
        counts[family] = {"cases": len(family_case_ids), "models": len(family_models),
                          "rows": len(rows)}

    for case in cases.values():
        names = set(case["models"])
        require(set(case["values"]) == set(POINT_BY_PAIR),
                f"Missing selected operating points: {case['case_id']}")
        require(all(set(values) == names for values in case["values"].values()),
                f"Model identities change across sweep: {case['case_id']}")
        if case["family"] == "prune":
            achieved = sorted(case.pop("achieved_sparsity"))
            case["case_parameters"] = {
                **case["case_parameters"], "achieved_sparsity": achieved}
        else:
            case.pop("achieved_sparsity")
        case["case_parameters_json"] = json_text(case["case_parameters"])
    require(len(cases) == sum(EXPECTED_CASE_COUNTS.values()) == 97,
            f"Expected 97 cases, got {len(cases)}")
    require(sum(len(case["models"]) for case in cases.values()) == 265,
            "Expected 265 case-local model instances")
    return sorted(cases.values(), key=lambda c: c["case_id"]), table_hashes, counts


def architecture_maps(dataset):
    configs = same.ARCHES[dataset]
    require(tuple(configs) == CANDIDATE_ARCHITECTURES,
            f"Unexpected architecture order for {dataset}: {tuple(configs)}")
    return configs


def point_fields(point):
    return {
        "operating_point_id": point["operating_point_id"],
        "sweep_axis": point["sweep_axis"],
        "in_sample_sweep": point["in_sample_sweep"],
        "in_bin_sweep": point["in_bin_sweep"],
        "in_size_rate": point["in_size_rate"],
        "training_size": 25000,
        "in_size": point["in_size"],
        "bins": point["bins"],
        "mi_kind": "In",
    }


def load_h0_values(document):
    expected_hash = sha256(neg.CSV)
    by_point = {}
    for point in OPERATING_POINTS:
        sizes = {scenario: point["in_size"] for scenario in document["payload"]["cases"]}
        values, checked_hash = neg.preflight(
            document, neg.ROOT, neg.CSV, sizes, point["bins"], "In")
        require(checked_hash == expected_hash, "H0 CSV changed during sweep preflight")
        by_point[(point["in_size"], point["bins"])] = values
    return by_point, expected_hash


def build_reference_pools(dataset, document, h0_values, selected, point):
    configs = architecture_maps(dataset)
    pools, rows = {}, []
    for k in K_VALUES:
        for arch in CANDIDATE_ARCHITECTURES:
            scenario = configs[arch]["h0"]
            round_id = selected[(scenario, k)]
            split = neg.get_split(document, scenario, round_id, k)
            names = [item["model_name"] for item in split["h0"]]
            seeds = [int(item["seed"]) for item in split["h0"]]
            matrix = np.asarray([h0_values[name] for name in names], dtype=float)
            mu = matrix.mean(axis=0)
            covariance = np.cov(matrix, rowvar=False, ddof=1)
            eig = np.linalg.eigvalsh(covariance)
            require(float(eig.min()) > 0,
                    f"Non-positive covariance: {point['operating_point_id']}, "
                    f"{dataset}, {arch}, k={k}")
            diag = neg.diagnostics(matrix, np.asarray([0.0], dtype=float))
            diag = {key: value for key, value in diag.items()
                    if not key.startswith("susp_")}
            pools[(k, arch)] = {
                "architecture": arch,
                "scenario": scenario,
                "round_id": round_id,
                "split": split,
                "names": names,
                "seeds": seeds,
                "matrix": matrix,
                "mu": mu,
                "covariance": covariance,
            }
            rows.append({
                "manifest_sha256": document["manifest_sha256"],
                "dataset": dataset,
                **point_fields(point),
                "reference_architecture": arch,
                "h0_scenario": scenario,
                "round_id": round_id,
                "k_ref": k,
                "n_reference": k,
                "h0_seeds": json_text(seeds),
                "h0_model_names": json_text(names),
                "mu": json_text(mu.tolist()),
                "covariance": json_text(covariance.tolist()),
                "condition_number": float(np.linalg.cond(covariance)),
                **diag,
            })
    return pools, rows


def score_all_references(k, names, matrix, pools, common):
    require(len(names) == len(matrix) and len(names) > 0, "Empty/misaligned suspects")
    components = {}
    component_rows = []
    for ref_arch in CANDIDATE_ARCHITECTURES:
        pool = pools[(k, ref_arch)]
        mu, covariance, t2, p_chi2, p_f = neg.score(pool["matrix"], matrix)
        stat_chi2, stat_f = neg.law_statistics(t2, k)
        components[ref_arch] = {
            "T2": np.asarray(t2, dtype=float),
            "stat_chi2": np.asarray(stat_chi2, dtype=float),
            "p_chi2": np.asarray(p_chi2, dtype=float),
            "stat_F": np.asarray(stat_f, dtype=float),
            "p_F": np.asarray(p_f, dtype=float),
        }
        for index, name in enumerate(names):
            component_rows.append({
                **common(index, name),
                "reference_architecture": ref_arch,
                "h0_scenario": pool["scenario"],
                "reference_round_id": pool["round_id"],
                "h0_seeds": json_text(pool["seeds"]),
                "reference_mu": json_text(mu.tolist()),
                "reference_covariance": json_text(covariance.tolist()),
                "reference_condition_number": float(np.linalg.cond(covariance)),
                "T2": float(t2[index]),
                "stat_chi2": float(stat_chi2[index]),
                "p_chi2": float(p_chi2[index]),
                "stat_F": float(stat_f[index]),
                "p_F": float(p_f[index]),
                f"reject_chi2@{ALPHA}": bool(p_chi2[index] < ALPHA),
                f"reject_F@{ALPHA}": bool(p_f[index] < ALPHA),
            })

    combined_rows = []
    for index, name in enumerate(names):
        p_f_by_arch = {arch: float(components[arch]["p_F"][index])
                       for arch in CANDIDATE_ARCHITECTURES}
        p_chi2_by_arch = {arch: float(components[arch]["p_chi2"][index])
                          for arch in CANDIDATE_ARCHITECTURES}
        t2_by_arch = {arch: float(components[arch]["T2"][index])
                      for arch in CANDIDATE_ARCHITECTURES}
        p_f_max = max(p_f_by_arch.values())
        p_chi2_max = max(p_chi2_by_arch.values())
        t2_min = min(t2_by_arch.values())
        max_f_arches = [arch for arch, value in p_f_by_arch.items() if value == p_f_max]
        max_chi2_arches = [arch for arch, value in p_chi2_by_arch.items()
                           if value == p_chi2_max]
        min_t2_arches = [arch for arch, value in t2_by_arch.items() if value == t2_min]
        require(set(max_f_arches) == set(min_t2_arches),
                f"max-p_F/min-T2 mismatch for {name}, k={k}")
        row = common(index, name)
        for arch in CANDIDATE_ARCHITECTURES:
            metric = components[arch]
            row.update({
                f"{arch}_h0_scenario": pools[(k, arch)]["scenario"],
                f"{arch}_reference_round_id": pools[(k, arch)]["round_id"],
                f"T2_{arch}": float(metric["T2"][index]),
                f"stat_chi2_{arch}": float(metric["stat_chi2"][index]),
                f"p_chi2_{arch}": float(metric["p_chi2"][index]),
                f"stat_F_{arch}": float(metric["stat_F"][index]),
                f"p_F_{arch}": float(metric["p_F"][index]),
            })
        reject_f = bool(p_f_max < ALPHA)
        reject_chi2 = bool(p_chi2_max < ALPHA)
        row.update({
            "T2_composite_min": t2_min,
            "T2_at_max_p_F": t2_min,
            "min_T2_reference_architectures": json_text(min_t2_arches),
            "p_chi2_max": p_chi2_max,
            "max_p_chi2_reference_architectures": json_text(max_chi2_arches),
            f"reject_chi2@{ALPHA}": reject_chi2,
            "p_F_max": p_f_max,
            "max_p_F_reference_architectures": json_text(max_f_arches),
            f"reject_F@{ALPHA}": reject_f,
            "decision_F": "stolen" if reject_f else "independent",
        })
        combined_rows.append(row)
    return component_rows, combined_rows, components


def summarize_group(base, combined_rows, components, truth):
    p_f_max = np.asarray([row["p_F_max"] for row in combined_rows], dtype=float)
    p_chi2_max = np.asarray([row["p_chi2_max"] for row in combined_rows], dtype=float)
    t2_composite = np.asarray([row["T2_composite_min"] for row in combined_rows],
                              dtype=float)
    reject_f = p_f_max < ALPHA
    reject_chi2 = p_chi2_max < ALPHA
    n = len(combined_rows)
    summary = {
        **base,
        "truth": truth,
        "n_suspects": n,
        f"n_rejected_chi2@{ALPHA}": int(reject_chi2.sum()),
        f"rejection_rate_chi2@{ALPHA}": float(reject_chi2.mean()),
        f"n_rejected_F@{ALPHA}": int(reject_f.sum()),
        f"rejection_rate_F@{ALPHA}": float(reject_f.mean()),
    }
    if truth == "negative":
        summary.update({
            f"nfp_chi2@{ALPHA}": int(reject_chi2.sum()),
            f"fpr_chi2@{ALPHA}": float(reject_chi2.mean()),
            f"nfp_F@{ALPHA}": int(reject_f.sum()),
            f"fpr_F@{ALPHA}": float(reject_f.mean()),
        })
    else:
        summary.update({
            f"ntp_chi2@{ALPHA}": int(reject_chi2.sum()),
            f"tpr_chi2@{ALPHA}": float(reject_chi2.mean()),
            f"fnr_chi2@{ALPHA}": float(1.0 - reject_chi2.mean()),
            f"ntp_F@{ALPHA}": int(reject_f.sum()),
            f"tpr_F@{ALPHA}": float(reject_f.mean()),
            f"fnr_F@{ALPHA}": float(1.0 - reject_f.mean()),
        })
    summary.update(distribution_summary(t2_composite, "T2_composite_min"))
    summary.update(distribution_summary(p_chi2_max, "p_chi2_max"))
    summary.update(distribution_summary(p_f_max, "p_F_max"))
    for arch in CANDIDATE_ARCHITECTURES:
        t2 = components[arch]["T2"]
        p_chi2 = components[arch]["p_chi2"]
        p_f = components[arch]["p_F"]
        summary.update({
            f"n_rejected_chi2_vs_{arch}@{ALPHA}": int((p_chi2 < ALPHA).sum()),
            f"rejection_rate_chi2_vs_{arch}@{ALPHA}": float((p_chi2 < ALPHA).mean()),
            f"n_rejected_F_vs_{arch}@{ALPHA}": int((p_f < ALPHA).sum()),
            f"rejection_rate_F_vs_{arch}@{ALPHA}": float((p_f < ALPHA).mean()),
        })
        summary.update(distribution_summary(t2, f"T2_{arch}"))
        summary.update(distribution_summary(p_chi2, f"p_chi2_{arch}"))
        summary.update(distribution_summary(p_f, f"p_F_{arch}"))
    return summary


def evaluate_negatives(dataset, document, h0_values, selected, pools, point):
    configs = architecture_maps(dataset)
    component_rows, combined_rows, summaries = [], [], []
    for source_arch in CANDIDATE_ARCHITECTURES:
        scenario = configs[source_arch]["h0"]
        for k in K_VALUES:
            source_round = selected[(scenario, k)]
            split = neg.get_split(document, scenario, source_round, k)
            models = split["evaluation_negative"]
            names = [model["model_name"] for model in models]
            matrix = np.asarray([h0_values[name] for name in names], dtype=float)

            def common(index, name):
                return {
                    "manifest_sha256": document["manifest_sha256"],
                    "dataset": dataset,
                    **point_fields(point),
                    "truth": "negative",
                    "suspect_architecture": source_arch,
                    "suspect_scenario": scenario,
                    "source_round_id": source_round,
                    "k_ref": k,
                    "seed": int(models[index]["seed"]),
                    "model_name": name,
                    "ixt": float(matrix[index, 0]),
                    "ity": float(matrix[index, 1]),
                }

            long_rows, model_rows, components = score_all_references(
                k, names, matrix, pools, common)
            for row in model_rows:
                row["classification_outcome_F"] = (
                    "false_positive" if row[f"reject_F@{ALPHA}"] else "true_negative")
            component_rows.extend(long_rows)
            combined_rows.extend(model_rows)
            summaries.append(summarize_group({
                "manifest_sha256": document["manifest_sha256"],
                "dataset": dataset,
                **point_fields(point),
                "suspect_architecture": source_arch,
                "suspect_scenario": scenario,
                "source_round_id": source_round,
                "k_ref": k,
            }, model_rows, components, "negative"))
    return component_rows, combined_rows, summaries


def evaluate_positives(dataset, document, cases, pools, point):
    component_rows, combined_rows, summaries = [], [], []
    pair = (point["in_size"], point["bins"])
    for case in cases:
        if case["dataset"] != dataset:
            continue
        names = sorted(case["models"])
        values = case["values"][pair]
        matrix = np.asarray([values[name] for name in names], dtype=float)
        for k in K_VALUES:

            def common(index, name):
                return {
                    "manifest_sha256": document["manifest_sha256"],
                    "dataset": dataset,
                    **point_fields(point),
                    "truth": "positive",
                    "family": case["family"],
                    "case_id": case["case_id"],
                    "group": case["group"],
                    "case_parameters": case["case_parameters_json"],
                    "suspect_architecture": case["suspect_architecture"],
                    "suspect_scenario": case["suspect_scenario"],
                    "k_ref": k,
                    "seed": case["models"][name],
                    "model_name": name,
                    "ixt": float(matrix[index, 0]),
                    "ity": float(matrix[index, 1]),
                }

            long_rows, model_rows, components = score_all_references(
                k, names, matrix, pools, common)
            for row in model_rows:
                row["classification_outcome_F"] = (
                    "true_positive" if row[f"reject_F@{ALPHA}"] else "false_negative")
            component_rows.extend(long_rows)
            combined_rows.extend(model_rows)
            summaries.append(summarize_group({
                "manifest_sha256": document["manifest_sha256"],
                "dataset": dataset,
                **point_fields(point),
                "family": case["family"],
                "case_id": case["case_id"],
                "group": case["group"],
                "case_parameters": case["case_parameters_json"],
                "suspect_architecture": case["suspect_architecture"],
                "suspect_scenario": case["suspect_scenario"],
                "k_ref": k,
            }, model_rows, components, "positive"))
    return component_rows, combined_rows, summaries


def validate_component_links(kind, component_rows, model_rows):
    if kind == "negatives":
        key_fields = ("operating_point_id", "suspect_scenario", "source_round_id",
                      "k_ref", "model_name")
    else:
        key_fields = ("operating_point_id", "case_id", "k_ref", "model_name")
    components_by_key = {}
    for row in component_rows:
        key = tuple(str(row[field]) for field in key_fields)
        bucket = components_by_key.setdefault(key, {})
        require(row["reference_architecture"] not in bucket,
                f"Duplicate component row: {key}, {row['reference_architecture']}")
        bucket[row["reference_architecture"]] = row
    for row in model_rows:
        key = tuple(str(row[field]) for field in key_fields)
        candidates = components_by_key.get(key, {})
        require(set(candidates) == set(CANDIDATE_ARCHITECTURES),
                f"Missing candidate architectures: {key}")
        p_f = {arch: float(candidates[arch]["p_F"]) for arch in CANDIDATE_ARCHITECTURES}
        t2 = {arch: float(candidates[arch]["T2"]) for arch in CANDIDATE_ARCHITECTURES}
        require(math.isclose(float(row["p_F_max"]), max(p_f.values()),
                             rel_tol=0, abs_tol=0), f"Wrong composite p_F: {key}")
        require(math.isclose(float(row["T2_composite_min"]), min(t2.values()),
                             rel_tol=0, abs_tol=0), f"Wrong composite T2: {key}")
        require(math.isclose(float(row["T2_at_max_p_F"]), min(t2.values()),
                             rel_tol=0, abs_tol=0), f"Wrong p-linked T2: {key}")
        require(bool(row[f"reject_F@{ALPHA}"]) == all(value < ALPHA for value in p_f.values()),
                f"Wrong composite F decision: {key}")


def persist_dataset(root, dataset, reference_rows, negative_data, positive_data,
                    n_positive_cases, n_positive_models):
    expected_points = len(OPERATING_POINTS)
    expected_negative_models = expected_points * 3 * len(K_VALUES) * 50
    expected_negative_summaries = expected_points * 3 * len(K_VALUES)
    expected_positive_models = expected_points * len(K_VALUES) * n_positive_models
    expected_positive_summaries = expected_points * len(K_VALUES) * n_positive_cases
    expected_reference_rows = expected_points * len(K_VALUES) * 3

    require(len(reference_rows) == expected_reference_rows,
            f"{dataset}: wrong reference-pool row count")
    negative_components, negative_models, negative_summaries = negative_data
    positive_components, positive_models, positive_summaries = positive_data
    require(len(negative_models) == expected_negative_models and
            len(negative_components) == expected_negative_models * 3 and
            len(negative_summaries) == expected_negative_summaries,
            f"{dataset}: wrong negative output counts")
    require(len(positive_models) == expected_positive_models and
            len(positive_components) == expected_positive_models * 3 and
            len(positive_summaries) == expected_positive_summaries,
            f"{dataset}: wrong positive output counts")
    validate_component_links("negatives", negative_components, negative_models)
    validate_component_links("positives", positive_components, positive_models)

    dataset_root = root / dataset
    write_csv(dataset_root / "reference_pools.csv", reference_rows)
    for kind, rows in (
        ("negatives/per_reference.csv", negative_components),
        ("negatives/per_model.csv", negative_models),
        ("negatives/summary.csv", negative_summaries),
        ("positives/per_reference.csv", positive_components),
        ("positives/per_model.csv", positive_models),
        ("positives/summary.csv", positive_summaries),
    ):
        write_csv(dataset_root / kind, rows)
    return {
        "reference_pools": len(reference_rows),
        "negative_per_reference": len(negative_components),
        "negative_per_model": len(negative_models),
        "negative_summary": len(negative_summaries),
        "positive_cases": n_positive_cases,
        "positive_models": n_positive_models,
        "positive_per_reference": len(positive_components),
        "positive_per_model": len(positive_models),
        "positive_summary": len(positive_summaries),
    }


def validate_anchor_against_existing(root):
    """The anchor's negative results must reproduce the established run exactly.

    Positive source tables differ intentionally (raw measured rows versus the
    old 50-model ``multiple`` tables), so positive agreement is checked only
    where the same model_name/family/group/k tuple exists.
    """
    old_root = BASE / "saved_logs/vanilla/Hypo_Test_UnknownArch_BestFPR_PerK"
    require(old_root.is_dir(), f"Established unknown-architecture result missing: {old_root}")
    report = {}
    for dataset in DATASETS:
        old_neg = read_csv(old_root / dataset / "negatives/per_model.csv")
        new_neg = [row for row in read_csv(root / dataset / "negatives/per_model.csv")
                   if row["operating_point_id"] == "N25000_B50"]
        old_by_key = {(r["suspect_scenario"], r["source_round_id"], r["k_ref"],
                       r["model_name"]): r for r in old_neg}
        require(len(old_by_key) == len(old_neg) == len(new_neg),
                f"{dataset}: anchor negative count differs from established run")
        checked = 0
        for row in new_neg:
            key = (row["suspect_scenario"], row["source_round_id"], row["k_ref"],
                   row["model_name"])
            prior = old_by_key.get(key)
            require(prior is not None, f"{dataset}: missing established anchor negative {key}")
            for field in ("T2_RN18", "p_F_RN18", "T2_VGG16", "p_F_VGG16",
                          "T2_DeiT", "p_F_DeiT", "p_F_max"):
                require(math.isclose(float(row[field]), float(prior[field]),
                                     rel_tol=1e-13, abs_tol=1e-15),
                        f"{dataset}: anchor mismatch {key}, {field}")
            checked += 1
        report[dataset] = {"negative_anchor_rows_matched": checked}
    return report


def output_hashes(root):
    paths = sorted(root.rglob("*.csv"))
    require(len(paths) == 15, f"Expected 15 result CSVs, got {len(paths)}")
    return {str(path.relative_to(root)).replace("\\", "/"): sha256(path)
            for path in paths}


def main():
    document = neg.load_manifest(neg.MANIFEST)
    cases, table_hashes, source_counts = discover_raw_cases()
    h0_by_point, h0_hash = load_h0_values(document)
    selected, selection_report = same.archived_min_fpr_rounds(document, h0_hash)
    require(sha256(same.OUTPUT_ROOT / "round_selection.csv") ==
            sha256(BASE / "saved_logs/vanilla/Hypo_Test_UnknownArch_BestFPR_PerK/round_selection.csv"),
            "Same-arch and unknown-arch round-selection files differ")

    require(not OUTPUT_ROOT.exists(), f"Output already exists; refusing to overwrite: {OUTPUT_ROOT}")
    require(not STAGING_ROOT.exists(),
            f"Staging directory already exists; inspect before retrying: {STAGING_ROOT}")
    STAGING_ROOT.mkdir(parents=True)

    metadata = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "design": "unknown_suspect_architecture_raw_MI_two_axis_sweeps",
        "candidate_architectures": list(CANDIDATE_ARCHITECTURES),
        "combination_rule": "p_combined = max(p_RN18, p_VGG16, p_DeiT)",
        "composite_statistic": (
            "T2_composite_min = min(T2_RN18, T2_VGG16, T2_DeiT); equals "
            "T2_at_max_p_F because all components use the same k/F law"),
        "decision_rule": f"reject independent training iff p_combined < {ALPHA}",
        "reference_policy": "separate architecture-specific moments and F calibration; never pooled",
        "round_selection_policy": (
            "reuse minimum-FPR rounds selected at in_size=25000, bins=50; "
            "same identities for every sweep operating point"),
        "sample_size_sweep": {"in_sizes": list(IN_SIZES),
                              "fixed_bins": FIXED_BINS_FOR_SAMPLE_SWEEP},
        "bin_size_sweep": {"bins": list(BIN_VALUES),
                           "fixed_in_size": FIXED_IN_SIZE_FOR_BIN_SWEEP},
        "operating_points": OPERATING_POINTS,
        "mi_kind": "In",
        "alpha": ALPHA,
        "k_values": list(K_VALUES),
        "gate1": False,
        "manifest_sha256": document["manifest_sha256"],
        "h0_csv": str(neg.CSV),
        "h0_csv_sha256": h0_hash,
        "positive_csv_sha256": table_hashes,
        "positive_source_counts": source_counts,
        "n_positive_cases": len(cases),
        "n_positive_models": sum(len(case["models"]) for case in cases),
        "round_selection_source": str(same.OUTPUT_ROOT / "round_selection.csv"),
        "round_selection_source_sha256": sha256(same.OUTPUT_ROOT / "round_selection.csv"),
        "script_sha256": sha256(__file__),
        "python": sys.version,
        "numpy": np.__version__,
    }
    (STAGING_ROOT / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    neg.write_json(STAGING_ROOT / "manifest.json", document)

    try:
        write_csv(STAGING_ROOT / "round_selection.csv", selection_report)
        dataset_buffers = {
            dataset: {"references": [], "neg_components": [], "neg_models": [],
                      "neg_summaries": [], "pos_components": [], "pos_models": [],
                      "pos_summaries": []}
            for dataset in DATASETS}
        for point in OPERATING_POINTS:
            pair = (point["in_size"], point["bins"])
            h0_values = h0_by_point[pair]
            for dataset in DATASETS:
                pools, references = build_reference_pools(
                    dataset, document, h0_values, selected, point)
                negatives = evaluate_negatives(
                    dataset, document, h0_values, selected, pools, point)
                positives = evaluate_positives(
                    dataset, document, cases, pools, point)
                buffer = dataset_buffers[dataset]
                buffer["references"].extend(references)
                buffer["neg_components"].extend(negatives[0])
                buffer["neg_models"].extend(negatives[1])
                buffer["neg_summaries"].extend(negatives[2])
                buffer["pos_components"].extend(positives[0])
                buffer["pos_models"].extend(positives[1])
                buffer["pos_summaries"].extend(positives[2])
            print(f"Completed operating point {point['operating_point_id']}", flush=True)

        dataset_counts = {}
        for dataset in DATASETS:
            buffer = dataset_buffers[dataset]
            dataset_cases = [case for case in cases if case["dataset"] == dataset]
            dataset_counts[dataset] = persist_dataset(
                STAGING_ROOT, dataset, buffer["references"],
                (buffer["neg_components"], buffer["neg_models"], buffer["neg_summaries"]),
                (buffer["pos_components"], buffer["pos_models"], buffer["pos_summaries"]),
                len(dataset_cases), sum(len(case["models"]) for case in dataset_cases))

        anchor_validation = validate_anchor_against_existing(STAGING_ROOT)
        hashes = output_hashes(STAGING_ROOT)
        metadata.update({
            "status": "complete",
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_counts": dataset_counts,
            "anchor_validation": anchor_validation,
            "output_csv_sha256": hashes,
        })
        (STAGING_ROOT / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8")
        STAGING_ROOT.rename(OUTPUT_ROOT)
    except Exception as exc:
        metadata.update({
            "status": "failed",
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
        })
        (STAGING_ROOT / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8")
        raise
    print(f"Complete: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
