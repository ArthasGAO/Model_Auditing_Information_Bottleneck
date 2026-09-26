"""One frozen minimum-FPR round per dataset/architecture/k, then all same-arch positives.

The selection is made only from the archived 50-round negative F-test at
in_size=25000, bins=50, alpha=0.01. Ties prefer the previously plotted round
44; otherwise the smallest round ID wins. No MI or split identities are sampled.
Run this file directly with the project environment's Python interpreter.
"""

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import run_hypothesis_test_fixed_splits as neg
import run_hypothesis_test_positives as pos


BASE = Path(__file__).resolve().parent
SOURCE_ROUNDS = BASE / "saved_logs/vanilla/Hypo_Test_FixedSplits/In_rate1_bins50"
OUTPUT_ROOT = BASE / "saved_logs/vanilla/Hypo_Test_SameArch_BestFPR_PerK"
PREFERRED_TIE_ROUND = 44
IN_SIZE, BINS, ALPHA = 25000, 50, 0.01
K_VALUES = (5, 10, 15, 20, 25, 30)
DATASETS = ("CF10", "CF100")

TABLES = {
    "ft": BASE / "saved_logs/ft_final/MI_master_table_ft_multiple.csv",
    "prune": BASE / "saved_logs/pruning_final/MI_master_table_prune_multiple.csv",
    "prune_illustrative": BASE / "saved_logs/pruning_final/MI_master_table_prune_multiple1.csv",
    "kd": BASE / "saved_logs/kd_final/MI_master_table_kd_multiple.csv",
    "extraction": BASE / "saved_logs/extraction_final/MI_master_table_extraction_multiple.csv",
    "extraction_illustrative": BASE / "saved_logs/extraction_final/MI_master_table_extraction_multiple1.csv",
}

ARCHES = {
    "CF10": {
        "RN18": {
            "h0": "CIFAR-10_ResNet-18_25000",
            "same": "CIFAR-10_ResNet-18_25000_Same_25000",
            "kd": "CIFAR-10_ResNet-18to18_25000",
            "knockoff": "CIFAR-10_ResNet-18_25000_Knockoff_Same10_Same18",
            "hl": "CIFAR-10_ResNet-18_25000_DFMS_Cross100-40C_Same18",
        },
        "VGG16": {
            "h0": "CIFAR-10_VGG16_25000",
            "same": "CIFAR-10_VGG16_25000_Same_25000",
            "kd": "CIFAR-10_VGG16toVGG16_25000",
            "knockoff": "CIFAR-10_VGG16_25000_Knockoff_Same10_Same16",
            "hl": "CIFAR-10_VGG16_25000_DFMS_Cross100-40C_Same16",
            "prune_illustrative": {
                "P-20%": "CIFAR-10_VGG16_25000_Illustrative_Prune20_FromRN18Same",
                "P-80%": "CIFAR-10_VGG16_25000_Illustrative_Prune80_FromRN18Same",
            },
        },
        "DeiT": {
            "h0": "CIFAR-10_DeiT_Plain_25000",
            "same": "CIFAR-10_DeiT_Plain_25000_Same_25000",
            "kd": "CIFAR-10_DeiTtoDeiT_25000",
            "knockoff": "CIFAR-10_DeiT_25000_Knockoff_Same10_SameDeiT",
            "hl": "CIFAR-10_DeiT_25000_DFMS_Illustrative_SameDeiT",
            "hl_illustrative": True,
        },
    },
    "CF100": {
        "RN18": {
            "h0": "CIFAR-100_ResNet-18_25000",
            "same": "CIFAR-100_ResNet-18_25000_Same_25000",
            "kd": "CIFAR-100_ResNet-18to18_25000",
            "knockoff": "CIFAR-100_ResNet-18_25000_Knockoff_Same100_Same18",
            "hl": "CIFAR-100_ResNet-18_25000_DFMS_Illustrative_Same18",
            "hl_illustrative": True,
        },
        "VGG16": {
            "h0": "CIFAR-100_VGG16_25000",
            "same": "CIFAR-100_VGG16_25000_Same_25000",
            "kd": "CIFAR-100_VGG16toVGG16_25000",
            "knockoff": "CIFAR-100_VGG16_25000_Knockoff_Same100_Same16",
            "hl": "CIFAR-100_VGG16_25000_DFMS_Illustrative_Same16",
            "hl_illustrative": True,
        },
        "DeiT": {
            "h0": "CIFAR-100_DeiT_Distill_25000",
            "same": "CIFAR-100_DeiT_Distill_25000_Same_25000",
            "kd": "CIFAR-100_DeiTtoDeiT_25000",
            "knockoff": "CIFAR-100_DeiT_25000_Knockoff_Same100_SameDeiT",
            "hl": "CIFAR-100_DeiT_25000_DFMS_Illustrative_SameDeiT",
            "hl_illustrative": True,
        },
    },
}

METHODS = ("FT-LL", "FT-AL", "RT-AL", "P-20%", "P-80%",
           "KD", "DKD", "Knockoff", "HL")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def archived_min_fpr_rounds(document, h0_hash):
    metadata = json.loads((SOURCE_ROUNDS / "run_metadata.json").read_text(encoding="utf-8"))
    neg.require(metadata["status"] == "complete", "Archived negative run is incomplete")
    neg.require(metadata["manifest_sha256"] == document["manifest_sha256"],
                "Archived negative manifest differs")
    neg.require(metadata["mi_csv_sha256"] == h0_hash,
                "Archived negative MI differs from the current table")
    neg.require(metadata["bins"] == BINS and metadata["mi_kind"] == "In"
                and metadata["gate1"] is False and ALPHA in metadata["alphas"],
                "Archived negative settings differ")
    neg.require(all(metadata["in_sizes_by_case"][scenario] == IN_SIZE
                    for cases in ARCHES.values() for config in cases.values()
                    for scenario in [config["h0"]]), "Archived in_size differs")
    grouped = defaultdict(list)
    for row in read_csv(SOURCE_ROUNDS / "per_split.csv"):
        key = (row["scenario"], int(row["k_ref"]))
        if key[0] in document["payload"]["cases"] and key[1] in K_VALUES:
            grouped[key].append(row)
    expected = {(config["h0"], k) for cases in ARCHES.values()
                for config in cases.values() for k in K_VALUES}
    neg.require(set(grouped) == expected, "Archived negative cells are incomplete or extraneous")
    selected, report = {}, []
    for dataset in DATASETS:
        for arch, config in ARCHES[dataset].items():
            scenario = config["h0"]
            for k in K_VALUES:
                rows = grouped[(scenario, k)]
                neg.require(len(rows) == 50 and {int(row["round_id"]) for row in rows} == set(range(50)),
                            f"Expected 50 unique rounds for {scenario}, k={k}")
                min_fp = min(int(row[f"nfp_F@{ALPHA}"]) for row in rows)
                ties = [row for row in rows if int(row[f"nfp_F@{ALPHA}"]) == min_fp]
                tie_ids = sorted(int(row["round_id"]) for row in ties)
                chosen_id = (PREFERRED_TIE_ROUND if PREFERRED_TIE_ROUND in tie_ids
                             else tie_ids[0])
                chosen = next(row for row in ties if int(row["round_id"]) == chosen_id)
                neg.require(int(chosen["n_eval"]) == 50
                            and math.isclose(float(chosen[f"fpr_F@{ALPHA}"]), min_fp / 50),
                            f"Invalid FPR denominator for {scenario}, k={k}")
                selected[(scenario, k)] = chosen_id
                report.append({
                    "dataset": dataset, "architecture": arch, "h0_scenario": scenario,
                    "k_ref": k, "round_id": chosen_id, "nfp_F@0.01": min_fp,
                    "fpr_F@0.01": min_fp / 50, "n_tied_minimum_rounds": len(tie_ids),
                    "tied_round_ids": json.dumps(tie_ids),
                    "h0_seeds": chosen["h0_seeds"],
                    "eval_negative_seeds": chosen["eval_negative_seeds"],
                })
    return selected, report


def selected_rows(table_rows, config, method):
    if method in ("FT-LL", "FT-AL", "RT-AL"):
        table_key, scenario = "ft", config["same"]
        accept = lambda row: row["strategy"] == method and row["rate"] == "1.0"
    elif method in ("P-20%", "P-80%"):
        use_illustrative = method in config.get("prune_illustrative", {})
        table_key = "prune_illustrative" if use_illustrative else "prune"
        scenario = (config["prune_illustrative"][method] if use_illustrative
                    else config["same"])
        sparsity = "0.2" if method == "P-20%" else "0.8"
        expected_ckpt = "synthetic" if use_illustrative else "best"
        accept = lambda row: (row["sparsity"] == sparsity and row["strategy"] == "FT-AL"
                              and row["ckpt_kind"] == expected_ckpt and row["rate"] == "1.0")
    elif method in ("KD", "DKD"):
        table_key, scenario = "kd", config["kd"]
        accept = lambda row: f"_{method}_" in row["model_name"] and row["rate"] == "0.0"
    else:
        illustrative = method == "HL" and config.get("hl_illustrative", False)
        table_key = "extraction_illustrative" if illustrative else "extraction"
        scenario = config["hl"] if method == "HL" else config["knockoff"]
        attack = "DFMS" if method == "HL" else "Knockoff"
        accept = lambda row: row["attack"] == attack and row["rate"] == "1.0"
    chosen = [row for row in table_rows[table_key]
              if row["Scenario"] == scenario and row["bins"] == str(BINS)
              and row["in_size"] == str(IN_SIZE) and accept(row)]
    neg.require(len(chosen) == 50, f"Expected 50 positive rows: {scenario}, {method}; got {len(chosen)}")
    seeds = [int(row["seed"]) for row in chosen]
    neg.require(sorted(seeds) == list(range(50)), f"Wrong seed identities: {scenario}, {method}")
    names = [row["model_name"] for row in chosen]
    neg.require(len(set(names)) == 50, f"Duplicate model names: {scenario}, {method}")
    values = {}
    for row in chosen:
        vector = [float(row["I(X;T)-In"]), float(row["I(T;Y)-In"])]
        neg.require(all(math.isfinite(value) for value in vector),
                    f"Nonfinite positive MI: {scenario}, {method}")
        values[row["model_name"]] = vector
    return table_key, scenario, values


def resolved_families(dataset, rows_by_table, hashes):
    families = []
    for arch, config in ARCHES[dataset].items():
        for method in METHODS:
            table_key, scenario, values = selected_rows(rows_by_table, config, method)
            label = f"{dataset}_{arch}_{method}"
            names = sorted(values)
            group = (scenario, method)
            families.append({
                "label": label, "plan_dir": Path(__file__),
                "mi_csv": TABLES[table_key], "h0_scenario": config["h0"],
                "group_by": ("Scenario", "method"),
                "_groups": {group: names}, "_values": values,
                "_hash": hashes[table_key],
                "_report": {
                    "selection_mode": "explicit_same_arch_case_map",
                    "source_scenario": scenario, "method": method,
                    "illustrative_table": table_key.endswith("illustrative"),
                    "n_positive": 50, "seed_range": [0, 49],
                },
                "_group_of": {name: "|".join(group) for name in names},
                "_scenario_of": {name: scenario for name in names},
            })
    neg.require(len(families) == 27 and len({f["label"] for f in families}) == 27,
                f"Expected 27 method groups for {dataset}")
    return families


def ensure_existing_result(path, *, document, h0_hash, selected, families=None):
    if not path.exists():
        return False
    metadata_path = path / "run_metadata.json"
    neg.require(metadata_path.is_file(), f"Existing result lacks metadata: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    neg.require(metadata["status"] == "complete" and
                metadata["manifest_sha256"] == document["manifest_sha256"] and
                metadata["bins"] == BINS and metadata["mi_kind"] == "In" and
                metadata["alphas"] == [ALPHA] and metadata["gate1"] is False,
                f"Existing result has different settings: {path}")
    scenarios = ({family["h0_scenario"] for family in families} if families else
                 {scenario for scenario, _ in selected})
    selected_json = {scenario: {str(k): selected[(scenario, k)] for k in K_VALUES}
                     for scenario in sorted(scenarios)}
    field = "selected_round_by_h0_k" if families else "selected_round_by_scenario_k"
    neg.require(metadata.get(field) == selected_json, f"Existing round map differs: {path}")
    hash_field = "h0_csv_sha256" if families else "mi_csv_sha256"
    neg.require(metadata[hash_field] == h0_hash, f"Existing H0 MI differs: {path}")
    if families:
        expected_hashes = {f["label"]: f["_hash"] for f in families}
        actual_hashes = {f["label"]: f["mi_csv_sha256"] for f in metadata["families"]}
        neg.require(expected_hashes == actual_hashes, f"Existing positive MI differs: {path}")
    for file_name in ("per_model.csv", "per_split.csv", "summary.csv", "manifest.json"):
        neg.require((path / file_name).is_file(), f"Existing result missing {file_name}: {path}")
    return True


def save_selection(report, document, h0_hash):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_ROOT / "round_selection.csv"
    metadata_path = OUTPUT_ROOT / "round_selection.metadata.json"
    metadata = {
        "source": str(SOURCE_ROUNDS / "per_split.csv"),
        "source_sha256": sha256(SOURCE_ROUNDS / "per_split.csv"),
        "h0_csv_sha256": h0_hash,
        "manifest_sha256": document["manifest_sha256"],
        "in_size": IN_SIZE, "bins": BINS, "law": "F", "alpha": ALPHA,
        "gate1": False, "preferred_tie_round": PREFERRED_TIE_ROUND,
        "tie_rule": "minimum integer false-positive count; prefer round 44 on ties, otherwise smallest round_id",
    }
    if csv_path.exists() or metadata_path.exists():
        neg.require(csv_path.is_file() and metadata_path.is_file(),
                    "Existing round selection is incomplete")
        neg.require(read_csv(csv_path) == [{k: str(v) for k, v in row.items()}
                                           for row in report], "Existing round selection differs")
        neg.require(json.loads(metadata_path.read_text(encoding="utf-8")) == metadata,
                    "Existing round-selection metadata differs")
        return
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    neg.write_json(metadata_path, metadata)


def main():
    document = neg.load_manifest(neg.MANIFEST)
    h0_hash = sha256(neg.CSV)
    selected, report = archived_min_fpr_rounds(document, h0_hash)
    sizes = {scenario: IN_SIZE for scenario in document["payload"]["cases"]}
    h0_values, checked_hash = neg.preflight(document, neg.ROOT, neg.CSV,
                                             sizes, BINS, "In")
    neg.require(h0_hash == checked_hash, "H0 CSV changed during preflight")
    rows_by_table = {key: read_csv(path) for key, path in TABLES.items()}
    hashes = {key: sha256(path) for key, path in TABLES.items()}
    families = {dataset: resolved_families(dataset, rows_by_table, hashes)
                for dataset in DATASETS}
    for key, path in TABLES.items():
        neg.require(sha256(path) == hashes[key], f"MI table changed during preflight: {path}")

    save_selection(report, document, h0_hash)
    for dataset in DATASETS:
        cases = [config["h0"] for config in ARCHES[dataset].values()]
        dataset_rounds = {(scenario, k): selected[(scenario, k)]
                          for scenario in cases for k in K_VALUES}
        negative_dir = OUTPUT_ROOT / dataset / "negatives"
        if ensure_existing_result(negative_dir, document=document, h0_hash=h0_hash,
                                  selected=dataset_rounds):
            print(f"SKIP existing {negative_dir}")
        else:
            neg.evaluate(document, h0_values, h0_hash, output_dir=negative_dir,
                         csv_path=neg.CSV, model_root=neg.ROOT,
                         in_size=sizes, bins=BINS, mi_kind="In", alphas=[ALPHA],
                         in_size_rate=1.0, training_sizes=neg.TRAINING_SIZES,
                         scenarios=cases, k_values=list(K_VALUES),
                         round_by_scenario_k=dataset_rounds)
        positive_dir = OUTPUT_ROOT / dataset / "positives"
        if ensure_existing_result(positive_dir, document=document, h0_hash=h0_hash,
                                  selected=dataset_rounds, families=families[dataset]):
            print(f"SKIP existing {positive_dir}")
        else:
            pos.evaluate_positives(
                document, families[dataset], h0_values, h0_hash,
                output_dir=positive_dir, in_size=sizes, bins=BINS,
                mi_kind="In", alphas=[ALPHA], in_size_rate=1.0,
                training_sizes=pos.TRAINING_SIZES, k_values=list(K_VALUES),
                round_by_h0_k=dataset_rounds)
    print(f"Complete: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
