"""Evaluate the exact notebook grid positives through the standard positive driver.

No MI, figure, notebook or existing hypothesis result is overwritten. Run once
per new OUTPUT_DIR; then author_fig1_grid_pvalues.mjs writes the presentation CSV.
"""
import csv
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import run_hypothesis_test_positives as driver

BASE = Path(__file__).resolve().parent
OUTPUT_DIR = BASE / "saved_logs/vanilla/Hypo_Test_Positives_Fig1_grid_round44_k30"
FAMILIES = {1: "rn18_ft", 2: "rn18_prune", 4: "rn18_extraction",
            5: "deit_ft", 6: "deit_prune", 8: "deit_extraction"}


def load_grid():
    """Execute only the existing framework/config/grid cells, suppressing exports."""
    book = json.loads((BASE / "distribution_check.ipynb").read_text(encoding="utf-8"))
    module = types.ModuleType("fig1_grid_positive_audit")
    sys.modules[module.__name__] = module
    ns = module.__dict__
    framework = next("".join(c["source"]) for c in reversed(book["cells"])
                     if "class TableGroupSpec:" in "".join(c.get("source", [])))
    exec(compile(framework, "MI-plot-framework", "exec"), ns)
    for identity in ("b6bf5527-38a7-43f3-b5c0-ab17d70cb169", "fixed-h0-grid-config", "fixed-h0-grid-render"):
        source = next("".join(c["source"]) for c in book["cells"] if c.get("id") == identity)
        source = "\n".join(line for line in source.splitlines() if not line.startswith("%"))
        with patch("matplotlib.figure.Figure.savefig"), patch("matplotlib.pyplot.show"):
            exec(compile(source, identity, "exec"), ns)
    plt.close("all")
    return ns


def main():
    driver.require(not OUTPUT_DIR.exists(), f"Choose a new OUTPUT_DIR; refusing overwrite: {OUTPUT_DIR}")
    ns = load_grid()
    driver.require((ns["BINS"], ns["IN_SIZE"], ns["H0_K"], ns["H0_ROUND_ID"], ns["H0_ALPHA"])
                   == (50, 25000, 30, 44, .01), "Grid settings changed; review this named run first")
    document = driver.load_manifest(driver.MANIFEST)
    sizes = {case: ns["IN_SIZE"] for case in document["payload"]["cases"]}
    h0, h0_hash = driver.preflight(document, driver.ROOT, Path(ns["CSV_NEG0"]), sizes, ns["BINS"], "In")
    driver.require(Path(ns["CSV_NEG0"]).resolve() == driver.H0_CSV.resolve(), "Driver/grid H0 CSV differs")
    sources = {str(p.resolve()): driver.digest(p.read_bytes()) for p in (
        BASE/"distribution_check.ipynb", BASE/"plot_fixed_split_h0.py",
        BASE/"run_hypothesis_test_positives.py", BASE/"run_hypothesis_test_fixed_splits.py",
        Path(__file__), driver.MANIFEST, driver.H0_CSV, Path(ns["CSV_VICTIM"]))}
    families, panels = [], []
    plotted = ns["grid_point_results"]
    for panel_id, panel in enumerate(ns["PLOT_PANELS"], 1):
        caption = ns["GRID_CAPTIONS"][panel_id-1]
        info = dict(panel=panel_id, case=caption, status="no_positive_data", points=[])
        panels.append(info)
        if not panel["positive_groups"]:
            driver.require(panel_id in (3, 7), "Unexpected missing group")
            continue
        driver.require(len(panel["positive_groups"]) == 1, "Review multiple groups before running")
        group = panel["positive_groups"][0]
        spec = group["spec"]
        points, xy = ns["fixed_h0"].selected_points(spec, ns, ns["BINS"], ns["IN_SIZE"])
        driver.require(len(points) == 50 and sorted(p["seed"] for p in points) == list(range(50)),
                       "Positive identity coverage differs from grid")
        source = Path(spec.csv_path).resolve()
        data_kind = "illustrative_affine_transfer" if "multiple1" in source.name else "mixed_measured_and_synthetic"
        family = driver.normalize_family(dict(
            label=FAMILIES[panel_id], plan_dir=BASE/"saved_exp_plan/fig1_grid_positive_audit"/FAMILIES[panel_id],
            mi_csv=source, h0_scenario=panel["negative_groups"][group["h0"]]["scenario"],
            group_by=("Scenario",), filters={"model_name": [p["model_name"] for p in points]}))
        groups, values, source_hash, report = driver.preflight_positives(family, ns["IN_SIZE"], ns["BINS"], "In")
        driver.require(set(values) == {p["model_name"] for p in points}, "Driver/grid selection differs")
        for p in points:
            np.testing.assert_array_equal(values[p["model_name"]], [p["ix"], p["iy"]])
        families.append({**family, "_groups": groups, "_values": values, "_hash": source_hash,
                         "_report": {**report, "data_kind": data_kind, "panel": panel_id},
                         "_group_of": {n: "|".join(k) for k, ms in groups.items() for n in ms},
                         "_scenario_of": {n: k[0] for k, ms in groups.items() for n in ms}})
        info.update(status="evaluated", family=family["label"], source_csv=str(source),
                    source_sha256=source_hash, data_kind=data_kind, points=points)
        sources[str(source)] = source_hash
        for plan in family["plan_dir"].glob("*.yaml"):
            sources[str(plan)] = driver.digest(plan.read_bytes())
    OUTPUT_DIR.mkdir()
    driver.evaluate_positives(
        document, families, h0, h0_hash, output_dir=OUTPUT_DIR/"raw_driver_results",
        in_size=sizes, bins=ns["BINS"], mi_kind="In", alphas=[ns["H0_ALPHA"]],
        in_size_rate=1., training_sizes=driver.TRAINING_SIZES,
        round_ids=[ns["H0_ROUND_ID"]], k_values=[ns["H0_K"]], victims=None)
    with (OUTPUT_DIR/"raw_driver_results/per_model.csv").open(newline="", encoding="utf-8") as stream:
        raw_results = list(csv.DictReader(stream))
    driver.require(len(raw_results) == 300, "Expected six groups of 50 points")
    index = {(r["family"], r["model_name"]): r for r in raw_results}
    driver.require(len(index) == 300, "Duplicate result keys")
    rows, summaries = [], []
    fields = ["panel", "case", "status", "seed", "ixt", "ity", "p_F", "alpha", "reject_H0_F",
              "decision_F", "data_kind", "round_id", "k_ref", "in_size", "bins", "mi_kind",
              "gate1", "T2", "stat_F", "p_chi2", "stat_chi2", "h0_scenario", "model_name",
              "source_csv", "source_sha256", "manifest_sha256"]
    for panel in panels:
        common = {k: panel[k] for k in ("panel", "case", "status")}
        common.update(alpha=.01, round_id=44, k_ref=30, in_size=25000, bins=50,
                      mi_kind="In", gate1=False, manifest_sha256=document["manifest_sha256"])
        if panel["status"] == "no_positive_data":
            rows.append({**dict.fromkeys(fields), **common, "decision_F": "not_evaluated"})
            summaries.append({"panel": panel["panel"], "case": panel["case"], "n": 0,
                              "p_F_min": None, "p_F_max": None, "rejected": None})
            continue
        values = []
        for point in sorted(panel["points"], key=lambda p: p["seed"]):
            scored = index[(panel["family"], point["model_name"])]
            numerical = {k: float(scored[k]) for k in ("ixt", "ity", "p_F", "p_chi2", "T2", "stat_F", "stat_chi2")}
            p_f = numerical["p_F"]
            driver.require(int(scored["round_id"]) == 44 and int(scored["k_ref"]) == 30, "Wrong reference selection")
            match = plotted[(plotted.panel == panel["panel"]) & (plotted.role == "positive") &
                            (plotted.model_name == point["model_name"])]
            driver.require(len(match) == 1, "Grid match not unique")
            np.testing.assert_allclose(p_f, match.iloc[0].p_F, rtol=1e-11, atol=0)
            # Independent closed-form F(2, k-2) survival check; never round tiny p-values to zero.
            expected = np.exp(-14*np.log1p(2*numerical["stat_F"]/28))
            np.testing.assert_allclose(p_f, expected, rtol=1e-11, atol=0)
            rejected = p_f < .01
            rows.append({**common, **numerical, "seed": int(point["seed"]),
                         "reject_H0_F": rejected, "decision_F": "reject_H0" if rejected else "do_not_reject_H0",
                         **{k: panel[k] for k in ("data_kind", "source_csv", "source_sha256")},
                         "h0_scenario": scored["h0_scenario"], "model_name": point["model_name"]})
            values.append(p_f)
        summaries.append(dict(panel=panel["panel"], case=panel["case"], n=len(values),
                              p_F_min=min(values), p_F_max=max(values),
                              p_F_median=float(np.median(values)), rejected=sum(p < .01 for p in values)))
    for source, expected_hash in sources.items():
        driver.require(driver.digest(Path(source).read_bytes()) == expected_hash, f"Input changed: {source}")
    result = dict(columns=fields, rows=rows, panel_summary=summaries, source_hashes=sources,
                  output_csv=str(OUTPUT_DIR/"grid_positive_pvalues.csv"),
                  description="Single plotted round only. Mixed and illustrative synthetic groups are not independent empirical TPR/FNR evidence. Missing DKD positives are not assigned p=0 or p=1. chi2 values can underflow to zero; use exact-F values.")
    driver.write_json(OUTPUT_DIR/"result_bundle.json", result)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
