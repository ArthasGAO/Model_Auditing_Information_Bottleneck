"""Composite-null MI hypothesis test for an unknown suspect architecture.

For every dataset and k, this driver reuses the frozen reference/evaluated-
negative identities selected by ``run_hypothesis_test_same_arch_best_fpr_per_k``.
Every suspect is scored separately against the RN18, VGG16 and DeiT reference
pools.  The architecture-unknown p-value is the maximum of the three
architecture-specific p-values, so the F-test rejects independent training only
when all three candidate-reference tests reject it.

The existing same-architecture results are never modified.  A completed output
directory is validated and skipped; a new run is first written to a staging
directory and promoted only after all invariants pass.
"""

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import run_hypothesis_test_fixed_splits as neg
import run_hypothesis_test_same_arch_best_fpr_per_k as same


BASE = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE / "saved_logs/vanilla/Hypo_Test_UnknownArch_BestFPR_PerK"
STAGING_ROOT = OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + ".staging")

IN_SIZE = same.IN_SIZE
BINS = same.BINS
ALPHA = same.ALPHA
K_VALUES = same.K_VALUES
DATASETS = same.DATASETS
CANDIDATE_ARCHITECTURES = ("RN18", "VGG16", "DeiT")


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
    require(all(list(row) == fields for row in rows),
            f"Inconsistent CSV fields: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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


def architecture_maps(dataset):
    configs = same.ARCHES[dataset]
    require(tuple(configs) == CANDIDATE_ARCHITECTURES,
            f"Unexpected architecture order for {dataset}: {tuple(configs)}")
    by_scenario = {config["h0"]: arch for arch, config in configs.items()}
    require(len(by_scenario) == len(CANDIDATE_ARCHITECTURES),
            f"Duplicate H0 scenarios for {dataset}")
    return configs, by_scenario


def build_reference_pools(dataset, document, h0_values, selected):
    configs, _ = architecture_maps(dataset)
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
            require(float(eig.min()) > 0, f"Non-positive covariance: {dataset}, {arch}, k={k}")
            # Reference-only diagnostics are independent of the evaluated suspects.
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
            row = {
                "manifest_sha256": document["manifest_sha256"],
                "dataset": dataset,
                "reference_architecture": arch,
                "h0_scenario": scenario,
                "round_id": round_id,
                "k_ref": k,
                "in_size": IN_SIZE,
                "bins": BINS,
                "mi_kind": "In",
                "n_reference": k,
                "h0_seeds": json_text(seeds),
                "h0_model_names": json_text(names),
                "mu": json_text(mu.tolist()),
                "covariance": json_text(covariance.tolist()),
                "condition_number": float(np.linalg.cond(covariance)),
                **diag,
            }
            rows.append(row)
    require(len(rows) == len(K_VALUES) * len(CANDIDATE_ARCHITECTURES),
            f"Wrong reference-pool count for {dataset}")
    return pools, rows


def score_all_references(dataset, k, names, matrix, pools, common):
    """Return long component rows and one composite row per suspect."""
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
        p_f_max = max(p_f_by_arch.values())
        p_chi2_max = max(p_chi2_by_arch.values())
        max_f_arches = [arch for arch, value in p_f_by_arch.items() if value == p_f_max]
        max_chi2_arches = [arch for arch, value in p_chi2_by_arch.items()
                           if value == p_chi2_max]
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
    summary.update(distribution_summary(p_chi2_max, "p_chi2_max"))
    summary.update(distribution_summary(p_f_max, "p_F_max"))
    for arch in CANDIDATE_ARCHITECTURES:
        p_chi2 = components[arch]["p_chi2"]
        p_f = components[arch]["p_F"]
        summary.update({
            f"n_rejected_chi2_vs_{arch}@{ALPHA}": int((p_chi2 < ALPHA).sum()),
            f"rejection_rate_chi2_vs_{arch}@{ALPHA}": float((p_chi2 < ALPHA).mean()),
            f"n_rejected_F_vs_{arch}@{ALPHA}": int((p_f < ALPHA).sum()),
            f"rejection_rate_F_vs_{arch}@{ALPHA}": float((p_f < ALPHA).mean()),
        })
        summary.update(distribution_summary(p_chi2, f"p_chi2_{arch}"))
        summary.update(distribution_summary(p_f, f"p_F_{arch}"))
    return summary


def evaluate_negatives(dataset, document, h0_values, selected, pools):
    configs, _ = architecture_maps(dataset)
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
                    "truth": "negative",
                    "suspect_architecture": source_arch,
                    "suspect_scenario": scenario,
                    "source_round_id": source_round,
                    "k_ref": k,
                    "seed": int(models[index]["seed"]),
                    "model_name": name,
                    "in_size": IN_SIZE,
                    "bins": BINS,
                    "mi_kind": "In",
                    "ixt": float(matrix[index, 0]),
                    "ity": float(matrix[index, 1]),
                }

            long_rows, model_rows, components = score_all_references(
                dataset, k, names, matrix, pools, common)
            for row in model_rows:
                row["classification_outcome_F"] = (
                    "false_positive" if row[f"reject_F@{ALPHA}"] else "true_negative")
            component_rows.extend(long_rows)
            combined_rows.extend(model_rows)
            summaries.append(summarize_group({
                "manifest_sha256": document["manifest_sha256"],
                "dataset": dataset,
                "suspect_architecture": source_arch,
                "suspect_scenario": scenario,
                "source_round_id": source_round,
                "k_ref": k,
                "in_size": IN_SIZE,
                "bins": BINS,
                "mi_kind": "In",
            }, model_rows, components, "negative"))
    return component_rows, combined_rows, summaries


def evaluate_positives(dataset, document, families, pools, scenario_to_arch):
    component_rows, combined_rows, summaries = [], [], []
    for family in families:
        suspect_arch = scenario_to_arch[family["h0_scenario"]]
        names = sorted(family["_values"])
        matrix = np.asarray([family["_values"][name] for name in names], dtype=float)
        require(len(family["_groups"]) == 1,
                f"Expected one group per resolved family: {family['label']}")
        group_key, group_members = next(iter(family["_groups"].items()))
        require(sorted(group_members) == names, f"Family/group mismatch: {family['label']}")
        group = "|".join(group_key)
        for k in K_VALUES:

            def common(index, name):
                return {
                    "manifest_sha256": document["manifest_sha256"],
                    "dataset": dataset,
                    "truth": "positive",
                    "family": family["label"],
                    "group": group,
                    "suspect_architecture": suspect_arch,
                    "suspect_scenario": family["_scenario_of"][name],
                    "k_ref": k,
                    "model_name": name,
                    "in_size": IN_SIZE,
                    "bins": BINS,
                    "mi_kind": "In",
                    "ixt": float(matrix[index, 0]),
                    "ity": float(matrix[index, 1]),
                }

            long_rows, model_rows, components = score_all_references(
                dataset, k, names, matrix, pools, common)
            for row in model_rows:
                row["classification_outcome_F"] = (
                    "true_positive" if row[f"reject_F@{ALPHA}"] else "false_negative")
            component_rows.extend(long_rows)
            combined_rows.extend(model_rows)
            summaries.append(summarize_group({
                "manifest_sha256": document["manifest_sha256"],
                "dataset": dataset,
                "family": family["label"],
                "group": group,
                "suspect_architecture": suspect_arch,
                "suspect_scenario": next(iter(family["_scenario_of"].values())),
                "k_ref": k,
                "in_size": IN_SIZE,
                "bins": BINS,
                "mi_kind": "In",
            }, model_rows, components, "positive"))
    return component_rows, combined_rows, summaries


def validate_rows(dataset, kind, component_rows, model_rows, summaries):
    expected_groups = 3 if kind == "negatives" else 27
    expected_models = 50
    expected_model_rows = expected_groups * len(K_VALUES) * expected_models
    expected_component_rows = expected_model_rows * len(CANDIDATE_ARCHITECTURES)
    require(len(model_rows) == expected_model_rows,
            f"{dataset} {kind}: expected {expected_model_rows} model rows, got {len(model_rows)}")
    require(len(component_rows) == expected_component_rows,
            f"{dataset} {kind}: expected {expected_component_rows} component rows, "
            f"got {len(component_rows)}")
    require(len(summaries) == expected_groups * len(K_VALUES),
            f"{dataset} {kind}: wrong summary count")

    key_fields = (["suspect_scenario", "source_round_id", "k_ref", "model_name"]
                  if kind == "negatives"
                  else ["family", "group", "k_ref", "model_name"])
    components_by_key = {}
    for row in component_rows:
        key = tuple(str(row[field]) for field in key_fields)
        require(row["reference_architecture"] not in components_by_key.setdefault(key, {}),
                f"Duplicate component row: {key}, {row['reference_architecture']}")
        components_by_key[key][row["reference_architecture"]] = row
    for row in model_rows:
        key = tuple(str(row[field]) for field in key_fields)
        candidates = components_by_key.get(key, {})
        require(set(candidates) == set(CANDIDATE_ARCHITECTURES),
                f"Missing candidate architectures: {key}")
        p_f = [float(candidates[arch]["p_F"]) for arch in CANDIDATE_ARCHITECTURES]
        p_chi2 = [float(candidates[arch]["p_chi2"]) for arch in CANDIDATE_ARCHITECTURES]
        require(math.isclose(float(row["p_F_max"]), max(p_f), rel_tol=0, abs_tol=0),
                f"Wrong max F p-value: {key}")
        require(math.isclose(float(row["p_chi2_max"]), max(p_chi2), rel_tol=0, abs_tol=0),
                f"Wrong max chi2 p-value: {key}")
        require(bool(row[f"reject_F@{ALPHA}"]) == all(value < ALPHA for value in p_f),
                f"Wrong composite F decision: {key}")
        require(bool(row[f"reject_chi2@{ALPHA}"]) == all(value < ALPHA for value in p_chi2),
                f"Wrong composite chi2 decision: {key}")


def validate_against_same_arch(dataset, kind, model_rows):
    path = same.OUTPUT_ROOT / dataset / kind / "per_model.csv"
    old_rows = read_csv(path)
    if kind == "negatives":
        old_key = lambda row: (row["scenario"], int(row["round_id"]),
                               int(row["k_ref"]), row["model_name"])
        new_key = lambda row: (row["suspect_scenario"], int(row["source_round_id"]),
                               int(row["k_ref"]), row["model_name"])
        arch_key = "suspect_architecture"
    else:
        old_key = lambda row: (row["family"], row["group"],
                               int(row["k_ref"]), row["model_name"])
        new_key = old_key
        arch_key = "suspect_architecture"
    old = {old_key(row): row for row in old_rows}
    require(len(old) == len(old_rows) == len(model_rows),
            f"Existing same-arch row count differs: {dataset} {kind}")
    for row in model_rows:
        prior = old.get(new_key(row))
        require(prior is not None, f"No same-arch counterpart: {new_key(row)}")
        arch = row[arch_key]
        for old_field, new_field in (("T2", f"T2_{arch}"),
                                     ("stat_chi2", f"stat_chi2_{arch}"),
                                     ("p_chi2", f"p_chi2_{arch}"),
                                     ("stat_F", f"stat_F_{arch}"),
                                     ("p_F", f"p_F_{arch}")):
            require(math.isclose(float(prior[old_field]), float(row[new_field]),
                                 rel_tol=1e-13, abs_tol=1e-15),
                    f"Matched-architecture metric differs: {dataset}, {kind}, "
                    f"{new_key(row)}, {old_field}")


def persist_dataset(root, dataset, reference_rows, negative_data, positive_data):
    dataset_root = root / dataset
    write_csv(dataset_root / "reference_pools.csv", reference_rows)
    counts = {"reference_pools": len(reference_rows)}
    for kind, data in (("negatives", negative_data), ("positives", positive_data)):
        components, models, summaries = data
        validate_rows(dataset, kind, components, models, summaries)
        validate_against_same_arch(dataset, kind, models)
        write_csv(dataset_root / kind / "per_reference.csv", components)
        write_csv(dataset_root / kind / "per_model.csv", models)
        write_csv(dataset_root / kind / "summary.csv", summaries)
        counts[kind] = {
            "per_reference_rows": len(components),
            "per_model_rows": len(models),
            "summary_rows": len(summaries),
        }
    return counts


def validate_saved_tree(root, expected_hashes):
    csv_paths = sorted(root.rglob("*.csv"))
    require(len(csv_paths) == 15, f"Expected 15 result CSVs, got {len(csv_paths)}")
    actual_hashes = {str(path.relative_to(root)).replace("\\", "/"): sha256(path)
                     for path in csv_paths}
    if expected_hashes is not None:
        require(actual_hashes == expected_hashes, "Existing output CSV hashes differ")
    for path in csv_paths:
        rows = read_csv(path)
        require(rows, f"Empty output CSV: {path}")
        require(len(rows[0]) == len(set(rows[0])), f"Duplicate headers: {path}")
    return actual_hashes


def load_and_validate_inputs():
    document = neg.load_manifest(neg.MANIFEST)
    h0_hash = sha256(neg.CSV)
    selected, selection_report = same.archived_min_fpr_rounds(document, h0_hash)
    sizes = {scenario: IN_SIZE for scenario in document["payload"]["cases"]}
    h0_values, checked_hash = neg.preflight(document, neg.ROOT, neg.CSV,
                                             sizes, BINS, "In")
    require(h0_hash == checked_hash, "H0 CSV changed during preflight")
    table_rows = {key: same.read_csv(path) for key, path in same.TABLES.items()}
    table_hashes = {key: sha256(path) for key, path in same.TABLES.items()}
    families = {dataset: same.resolved_families(dataset, table_rows, table_hashes)
                for dataset in DATASETS}
    for key, path in same.TABLES.items():
        require(sha256(path) == table_hashes[key], f"Positive table changed: {path}")
    return document, h0_values, h0_hash, selected, selection_report, families, table_hashes


def verify_existing(metadata, document, h0_hash, table_hashes):
    require(metadata.get("status") == "complete", "Existing result is incomplete")
    require(metadata.get("manifest_sha256") == document["manifest_sha256"],
            "Existing manifest hash differs")
    require(metadata.get("h0_csv_sha256") == h0_hash, "Existing H0 CSV hash differs")
    require(metadata.get("positive_csv_sha256") == table_hashes,
            "Existing positive-table hashes differ")
    require(metadata.get("script_sha256") == sha256(__file__),
            "Existing result was produced by different code")
    validate_saved_tree(OUTPUT_ROOT, metadata.get("output_csv_sha256"))


def main():
    (document, h0_values, h0_hash, selected, selection_report,
     families, table_hashes) = load_and_validate_inputs()

    if OUTPUT_ROOT.exists():
        metadata_path = OUTPUT_ROOT / "run_metadata.json"
        require(metadata_path.is_file(), "Existing output lacks run_metadata.json")
        verify_existing(json.loads(metadata_path.read_text(encoding="utf-8")),
                        document, h0_hash, table_hashes)
        print(f"VALID existing result: {OUTPUT_ROOT}")
        return
    require(not STAGING_ROOT.exists(),
            f"Staging directory already exists; inspect it before retrying: {STAGING_ROOT}")
    STAGING_ROOT.mkdir(parents=True)

    metadata = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "design": "unknown_suspect_architecture_composite_union_null",
        "candidate_architectures": list(CANDIDATE_ARCHITECTURES),
        "combination_rule": "p_combined = max(p_RN18, p_VGG16, p_DeiT)",
        "decision_rule": f"reject independent training iff p_combined < {ALPHA}",
        "reference_policy": "separate architecture-specific moments and F calibration; never pooled",
        "evaluated_negative_policy": "reuse the fixed evaluated negatives from the selected source-architecture round for each k",
        "positive_policy": "reuse the unchanged 50-positive groups from the same-architecture run",
        "manifest_sha256": document["manifest_sha256"],
        "h0_csv": str(neg.CSV),
        "h0_csv_sha256": h0_hash,
        "positive_csv_sha256": table_hashes,
        "round_selection_source": str(same.OUTPUT_ROOT / "round_selection.csv"),
        "round_selection_source_sha256": sha256(same.OUTPUT_ROOT / "round_selection.csv"),
        "script_sha256": sha256(__file__),
        "in_size": IN_SIZE,
        "bins": BINS,
        "mi_kind": "In",
        "alpha": ALPHA,
        "k_values": list(K_VALUES),
        "gate1": False,
    }
    (STAGING_ROOT / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    neg.write_json(STAGING_ROOT / "manifest.json", document)

    try:
        write_csv(STAGING_ROOT / "round_selection.csv", selection_report)
        dataset_counts = {}
        for dataset in DATASETS:
            configs, scenario_to_arch = architecture_maps(dataset)
            pools, reference_rows = build_reference_pools(
                dataset, document, h0_values, selected)
            negative_data = evaluate_negatives(
                dataset, document, h0_values, selected, pools)
            positive_data = evaluate_positives(
                dataset, document, families[dataset], pools, scenario_to_arch)
            dataset_counts[dataset] = persist_dataset(
                STAGING_ROOT, dataset, reference_rows, negative_data, positive_data)
            print(f"Completed {dataset}: unknown-architecture composite tests", flush=True)

        output_hashes = validate_saved_tree(STAGING_ROOT, None)
        metadata.update({
            "status": "complete",
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_counts": dataset_counts,
            "output_csv_sha256": output_hashes,
        })
        (STAGING_ROOT / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8")
        STAGING_ROOT.rename(OUTPUT_ROOT)
        verify_existing(metadata, document, h0_hash, table_hashes)
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
