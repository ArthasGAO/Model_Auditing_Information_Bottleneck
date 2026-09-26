"""Read-only integrity audit for the selected-round same-architecture results."""

import csv
import json
import math
from collections import defaultdict

from scipy.stats import f as f_distribution

import run_hypothesis_test_same_arch_best_fpr_per_k as run


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def same_float(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-14)


def audit():
    document = run.neg.load_manifest(run.neg.MANIFEST)
    selected, expected_selection = run.archived_min_fpr_rounds(document, run.sha256(run.neg.CSV))
    archived = {(row["scenario"], int(row["k_ref"]), int(row["round_id"])): row
                for row in rows(run.SOURCE_ROUNDS / "per_split.csv")}
    stored_selection = rows(run.OUTPUT_ROOT / "round_selection.csv")
    assert len(stored_selection) == len(expected_selection) == 36
    assert stored_selection == [{key: str(value) for key, value in record.items()}
                                for record in expected_selection]
    assert sum(round_id != 44 for round_id in selected.values()) == 1
    table_rows = {key: run.read_csv(path) for key, path in run.TABLES.items()}

    for dataset in run.DATASETS:
        base = run.OUTPUT_ROOT / dataset
        negative_models = rows(base / "negatives/per_model.csv")
        negative_splits = rows(base / "negatives/per_split.csv")
        negative_summary = rows(base / "negatives/summary.csv")
        positive_models = rows(base / "positives/per_model.csv")
        positive_splits = rows(base / "positives/per_split.csv")
        positive_summary = rows(base / "positives/summary.csv")
        assert (len(negative_models), len(negative_splits), len(negative_summary)) == (900, 18, 18)
        assert (len(positive_models), len(positive_splits), len(positive_summary)) == (8100, 162, 162)
        assert len({(row["family"], row["k_ref"], row["model_name"])
                    for row in positive_models}) == 8100
        assert all(int(row["n_rounds"]) == 1 and row["std_tpr_F@0.01"] == ""
                   for row in positive_summary)
        assert all(int(row["n_rounds"]) == 1 and row["std_fpr_F@0.01"] == ""
                   for row in negative_summary)

        n_by_key = defaultdict(list)
        for row in negative_models:
            key = (row["scenario"], int(row["k_ref"]))
            assert int(row["round_id"]) == selected[key]
            assert same_float(row["p_F"], f_distribution.sf(float(row["stat_F"]), 2, key[1] - 2))
            n_by_key[key].append(row)
        n_split_by_key = {(row["scenario"], int(row["k_ref"])): row
                          for row in negative_splits}
        assert len(n_split_by_key) == 18
        for key, members in n_by_key.items():
            assert len(members) == 50
            nfp = sum(float(row["p_F"]) < run.ALPHA for row in members)
            split = n_split_by_key[key]
            old = archived[(key[0], key[1], selected[key])]
            assert int(split["nfp_F@0.01"]) == nfp
            assert same_float(split["fpr_F@0.01"], nfp / 50)
            assert split["h0_seeds"] == old["h0_seeds"]
            assert split["eval_negative_seeds"] == old["eval_negative_seeds"]
            assert same_float(split["fpr_F@0.01"], old["fpr_F@0.01"])
            frozen = run.neg.get_split(document, key[0], selected[key], key[1])
            assert json.loads(split["h0_seeds"]) == [m["seed"] for m in frozen["h0"]]
            assert json.loads(split["eval_negative_seeds"]) == [m["seed"] for m in frozen["evaluation_negative"]]

        p_by_key = defaultdict(list)
        for row in positive_models:
            key = (row["family"], int(row["k_ref"]))
            assert int(row["round_id"]) == selected[(row["h0_scenario"], key[1])]
            assert same_float(row["p_F"], f_distribution.sf(float(row["stat_F"]), 2, key[1] - 2))
            p_by_key[key].append(row)
        p_split_by_key = {(row["family"], int(row["k_ref"])): row
                          for row in positive_splits}
        assert len(p_split_by_key) == 162
        for key, members in p_by_key.items():
            assert len(members) == 50
            split = p_split_by_key[key]
            ntp = sum(float(row["p_F"]) < run.ALPHA for row in members)
            assert int(split["ntp_F@0.01"]) == ntp
            assert same_float(split["tpr_F@0.01"], ntp / 50)
            n_split = n_split_by_key[(split["h0_scenario"], key[1])]
            assert split["h0_seeds"] == n_split["h0_seeds"]
            assert split["mu"] == n_split["mu"]
            assert split["covariance"] == n_split["covariance"]

        for arch, config in run.ARCHES[dataset].items():
            for method in run.METHODS:
                _, _, source = run.selected_rows(table_rows, config, method)
                label = f"{dataset}_{arch}_{method}"
                for model in [row for row in positive_models
                              if row["family"] == label and row["k_ref"] == "30"]:
                    actual = source[model["model_name"]]
                    assert same_float(model["ixt"], actual[0])
                    assert same_float(model["ity"], actual[1])
        print(f"VERIFIED {dataset}: 18 negative cells, 162 positive cells, matched H0 and source MI")


if __name__ == "__main__":
    audit()
