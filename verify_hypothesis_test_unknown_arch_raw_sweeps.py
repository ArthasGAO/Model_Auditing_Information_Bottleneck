"""Read-only integrity audit for the raw-MI unknown-architecture sweeps."""

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import f as f_distribution

import run_hypothesis_test_unknown_arch_raw_sweeps as run


def rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, *, rel=1e-12, abs_=1e-14):
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


def audit():
    root = run.OUTPUT_ROOT
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "complete"
    assert metadata["n_positive_cases"] == 97
    assert metadata["n_positive_models"] == 265
    assert metadata["k_values"] == list(run.K_VALUES)
    assert len(metadata["operating_points"]) == 16
    assert metadata["script_sha256"] == digest(run.__file__)
    assert metadata["h0_csv_sha256"] == digest(run.neg.CSV)
    assert metadata["positive_csv_sha256"] == {
        key: digest(path) for key, path in run.RAW_TABLES.items()}
    for relative, expected in metadata["output_csv_sha256"].items():
        assert digest(root / relative) == expected

    expected_points = {(p["operating_point_id"], str(p["in_size"]), str(p["bins"]))
                       for p in run.OPERATING_POINTS}
    overlap_total = 0
    for dataset in run.DATASETS:
        base = root / dataset
        reference = rows(base / "reference_pools.csv")
        neg_components = rows(base / "negatives/per_reference.csv")
        neg_models = rows(base / "negatives/per_model.csv")
        neg_summary = rows(base / "negatives/summary.csv")
        pos_components = rows(base / "positives/per_reference.csv")
        pos_models = rows(base / "positives/per_model.csv")
        pos_summary = rows(base / "positives/summary.csv")
        counts = metadata["dataset_counts"][dataset]
        assert len(reference) == counts["reference_pools"]
        assert len(neg_components) == counts["negative_per_reference"]
        assert len(neg_models) == counts["negative_per_model"]
        assert len(neg_summary) == counts["negative_summary"]
        assert len(pos_components) == counts["positive_per_reference"]
        assert len(pos_models) == counts["positive_per_model"]
        assert len(pos_summary) == counts["positive_summary"]
        assert {(r["operating_point_id"], r["in_size"], r["bins"])
                for r in pos_summary} == expected_points

        case_ids = {r["case_id"] for r in pos_summary}
        assert len(case_ids) == counts["positive_cases"]
        expected_summary_keys = {
            (point["operating_point_id"], case_id, str(k))
            for point in run.OPERATING_POINTS for case_id in case_ids for k in run.K_VALUES}
        actual_summary_keys = {(r["operating_point_id"], r["case_id"], r["k_ref"])
                               for r in pos_summary}
        assert actual_summary_keys == expected_summary_keys
        assert len(actual_summary_keys) == len(pos_summary)

        def verify_models(kind, models, components):
            key_fields = (("operating_point_id", "case_id", "k_ref", "model_name")
                          if kind == "positive" else
                          ("operating_point_id", "suspect_scenario", "source_round_id",
                           "k_ref", "model_name"))
            by_key = defaultdict(dict)
            for row in components:
                key = tuple(row[field] for field in key_fields)
                assert row["reference_architecture"] not in by_key[key]
                by_key[key][row["reference_architecture"]] = row
                k = int(row["k_ref"])
                expected_p = f_distribution.sf(float(row["stat_F"]), 2, k - 2)
                assert close(row["p_F"], expected_p)
                assert close(row["T2"], row["stat_chi2"], rel=0, abs_=0)
                assert close(row["stat_F"],
                             float(row["T2"]) * (k - 2) / (2 * (k - 1)))
            assert len(by_key) == len(models)
            for row in models:
                key = tuple(row[field] for field in key_fields)
                refs = by_key[key]
                assert set(refs) == set(run.CANDIDATE_ARCHITECTURES)
                p_values = [float(refs[a]["p_F"]) for a in run.CANDIDATE_ARCHITECTURES]
                t2_values = [float(refs[a]["T2"]) for a in run.CANDIDATE_ARCHITECTURES]
                assert close(row["p_F_max"], max(p_values), rel=0, abs_=0)
                assert close(row["T2_composite_min"], min(t2_values), rel=0, abs_=0)
                assert close(row["T2_at_max_p_F"], min(t2_values), rel=0, abs_=0)
                assert (row["reject_F@0.01"].lower() == "true") == all(
                    value < run.ALPHA for value in p_values)

        verify_models("negative", neg_models, neg_components)
        verify_models("positive", pos_models, pos_components)

        models_by_summary = defaultdict(list)
        for row in pos_models:
            models_by_summary[(row["operating_point_id"], row["case_id"],
                               row["k_ref"])].append(row)
        for summary in pos_summary:
            key = (summary["operating_point_id"], summary["case_id"], summary["k_ref"])
            members = models_by_summary[key]
            assert len(members) == int(summary["n_suspects"])
            t2 = np.asarray([float(r["T2_composite_min"]) for r in members])
            assert close(summary["T2_composite_min_min"], t2.min())
            assert close(summary["T2_composite_min_p05"], np.percentile(t2, 5))
            assert close(summary["T2_composite_min_median"], np.median(t2))
            assert close(summary["T2_composite_min_p95"], np.percentile(t2, 95))
            assert close(summary["T2_composite_min_max"], t2.max())
            assert close(summary["T2_composite_min_mean"], t2.mean())

        # Any measured raw model also present in the established 50-model table
        # must reproduce its default-point component scores exactly.
        old = rows(BASE_OLD / dataset / "positives/per_model.csv")
        old_by_key = defaultdict(list)
        for row in old:
            old_by_key[(row["model_name"], row["k_ref"])].append(row)
        assert all(len(group) == 1 for group in old_by_key.values())
        overlap = 0
        for row in pos_models:
            if row["operating_point_id"] != "N25000_B50":
                continue
            prior_group = old_by_key.get((row["model_name"], row["k_ref"]))
            if not prior_group:
                continue
            prior = prior_group[0]
            for field in ("T2_RN18", "p_F_RN18", "T2_VGG16", "p_F_VGG16",
                          "T2_DeiT", "p_F_DeiT", "p_F_max"):
                assert close(row[field], prior[field], rel=1e-13, abs_=1e-15)
            overlap += 1
        assert overlap > 0
        overlap_total += overlap
        print(f"VERIFIED {dataset}: {len(case_ids)} cases, {len(pos_summary)} summary "
              f"cells, {overlap} measured anchor rows matched")
    print(f"VERIFIED total measured anchor overlaps: {overlap_total}")


BASE_OLD = (run.BASE /
            "saved_logs/vanilla/Hypo_Test_UnknownArch_BestFPR_PerK")


if __name__ == "__main__":
    audit()
