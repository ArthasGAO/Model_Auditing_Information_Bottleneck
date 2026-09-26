"""Recompute one case with the existing evaluator; safely replace mixed outputs.

Run with the project's NumPy/SciPy environment. No model sampling or MI changes.
Old files are retained outside the active result tree for recovery. All outputs
are staged and checked before any active result is replaced. A commit failure
rolls back files already replaced. The full immutable split manifest is reused.
"""
import csv
import io
import json
import os
from pathlib import Path
import shutil
from datetime import datetime, timezone

import numpy as np
from scipy.stats import chi2, f
import run_hypothesis_test_fixed_splits as experiment

CASE = "CIFAR-10_VGG16_25000"
BASE = Path(__file__).resolve().parent
ACTIVE = BASE / "saved_logs/vanilla/Hypo_Test_FixedSplits"
RECOVERY = BASE / "saved_logs/repair_backups"


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def key(row):
    return tuple(row.get(c, "") for c in
                 ("scenario", "in_size", "bins", "mi_kind", "k_ref", "round_id", "seed"))


def merge_case(old_path, new_rows, out_path, expected, values=None):
    """Preserve other scenarios byte-for-byte, including row order and header."""
    replacements = {key(r): r for r in new_rows}
    assert len(replacements) == len(new_rows) == expected
    original_hash = experiment.digest(old_path.read_bytes())
    before, after = [], []
    seen = set()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with old_path.open("rb") as source, out_path.open("xb") as dest:
        header = source.readline()
        fields = next(csv.reader([header.decode("utf-8-sig")]))
        index = fields.index("scenario")
        dest.write(header)
        for raw in source:
            # Identity columns contain no commas/quotes/newlines in this schema.
            case = raw.split(b",", index + 1)[index].decode()
            if case != CASE:
                if values is not None:
                    cols = raw.decode().rstrip("\r\n").split(",")
                    name = cols[fields.index("model_name")]
                    assert [float(cols[fields.index(c)]) for c in ("ixt", "ity")] == values[name], name
                before.append(raw)
                dest.write(raw)
                after.append(raw)
                continue
            row = dict(zip(fields, next(csv.reader([raw.decode()]))))
            identity = key(row)
            assert identity in replacements and identity not in seen, identity
            seen.add(identity)
            replacement = replacements[identity]
            assert set(replacement) == set(fields)
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=fields,
                                    lineterminator="\r\n" if header.endswith(b"\r\n") else "\n")
            writer.writerow(replacement)
            dest.write(buffer.getvalue().encode())
    assert seen == set(replacements)
    assert before == after
    return {"old_sha256": original_hash, "new_sha256": experiment.digest(out_path.read_bytes()),
            "replaced_rows": expected, "preserved_rows": len(before),
            "preserved_rows_sha256": experiment.digest(b"".join(before))}


def validate_new(folder, doc, values, metadata):
    models = read_rows(folder / "per_model.csv")
    splits = read_rows(folder / "per_split.csv")
    summaries = read_rows(folder / "summary.csv")
    assert (len(models), len(splits), len(summaries)) == (15000, 300, 6)
    groups = {}
    for row in models:
        assert row["scenario"] == CASE and row["manifest_sha256"] == doc["manifest_sha256"]
        assert [float(row[c]) for c in ("ixt", "ity")] == values[row["model_name"]]
        group = groups.setdefault((int(row["round_id"]), int(row["k_ref"])), {})
        assert int(row["seed"]) not in group
        group[int(row["seed"])] = row
    for row in splits:
        rid, k = int(row["round_id"]), int(row["k_ref"])
        split = experiment.get_split(doc, CASE, rid, k)
        assert json.loads(row["h0_seeds"]) == [m["seed"] for m in split["h0"]]
        assert json.loads(row["eval_negative_seeds"]) == [m["seed"] for m in split["evaluation_negative"]]
        group = groups[(rid, k)]
        assert set(group) == {m["seed"] for m in split["evaluation_negative"]}
        x = np.array([values[m["model_name"]] for m in split["h0"]])
        y = np.array([values[m["model_name"]] for m in split["evaluation_negative"]])
        mu, cov = x.mean(0), np.cov(x, rowvar=False, ddof=1)
        np.testing.assert_array_equal(mu, json.loads(row["mu"]))
        np.testing.assert_array_equal(cov, json.loads(row["covariance"]))
        delta = y - mu
        t2 = k / (k + 1) * np.einsum("ij,ji->i", delta, np.linalg.solve(cov, delta.T))
        records = [group[m["seed"]] for m in split["evaluation_negative"]]
        np.testing.assert_allclose(t2, [float(r["T2"]) for r in records], rtol=1e-13)
        for law, probs in (("chi2", chi2.sf(t2, 2)),
                           ("F", f.sf(t2 * (k - 2)/(2*(k - 1)), 2, k - 2))):
            np.testing.assert_allclose(probs, [float(r["p_" + law]) for r in records], rtol=1e-12)
            for alpha in metadata["alphas"]:
                nfp = int(sum(probs < alpha))
                assert nfp == int(row[f"nfp_{law}@{alpha}"])
                assert nfp / 50 == float(row[f"fpr_{law}@{alpha}"])
    for summary in summaries:
        rows = [r for r in splits if r["k_ref"] == summary["k_ref"]]
        for law in ("F", "chi2"):
            for alpha in metadata["alphas"]:
                rates = [float(r[f"fpr_{law}@{alpha}"]) for r in rows]
                assert float(summary[f"mean_fpr_{law}@{alpha}"]) == float(np.mean(rates))
                assert float(summary[f"std_fpr_{law}@{alpha}"]) == float(np.std(rates, ddof=1))
    return models, splits, summaries


def resume_saved(run):
    """Commit a verified rolled-back run without evaluating statistics again.

    Rebuilt CSV bytes must match the previously validated output hashes.
    Metadata gets a fresh commit timestamp; the original report stays intact.
    """
    run = Path(run).resolve()
    assert run.is_relative_to(RECOVERY.resolve()) and run.is_dir()
    previous = json.loads((run / "repair_report.json").read_text())
    assert previous["status"] == "rolled_back" and previous["scenario"] == CASE
    doc = experiment.load_manifest(experiment.MANIFEST)
    manifest_bytes = experiment.MANIFEST.read_bytes()
    assert doc["manifest_sha256"] == previous["manifest_sha256"]
    assert experiment.digest(experiment.CSV.read_bytes()) == previous["mi_csv_sha256"]
    assert len(previous["files"]) == 241
    for name, stats in previous["files"].items():
        active, backup = ACTIVE / name, run / "original" / name
        assert active.resolve().is_relative_to(ACTIVE.resolve())
        assert backup.resolve().is_relative_to(run)
        assert experiment.digest(active.read_bytes()) == stats["old_sha256"], name
        assert experiment.digest(backup.read_bytes()) == stats["old_sha256"], name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    attempt = run / ("resume_" + stamp)
    attempt.mkdir(exist_ok=False)
    report = {"status": "staging", "scenario": CASE,
              "mi_csv_sha256": previous["mi_csv_sha256"],
              "manifest_sha256": doc["manifest_sha256"],
              "resumed_from": str(run / "repair_report.json"),
              "repair_script_sha256": experiment.digest(Path(__file__).read_bytes()),
              "statistics_recomputed": False, "files": {}}
    commits = []
    combined = []
    folders = sorted((run / "recomputed").iterdir())
    assert len(folders) == 60
    for index, stage in enumerate(folders, 1):
        folder = ACTIVE / stage.name
        meta = json.loads((folder / "run_metadata.json").read_text())
        fresh = json.loads((stage / "run_metadata.json").read_text())
        assert fresh["status"] == "complete" and fresh["completed_splits"] == 300
        assert fresh["evaluated_scenarios"] == [CASE]
        assert fresh["mi_csv_sha256"] == previous["mi_csv_sha256"]
        assert fresh["manifest_sha256"] == doc["manifest_sha256"]
        assert json.loads((stage / "manifest.json").read_text()) == doc
        assert json.loads((folder / "manifest.json").read_text()) == doc
        for name, count in (("per_model.csv", 15000), ("per_split.csv", 300), ("summary.csv", 6)):
            rows = read_rows(stage / name)
            assert all(r["scenario"] == CASE for r in rows)
            relative = Path(stage.name) / name
            replacement = attempt / relative
            stats = merge_case(folder / name, rows, replacement, count)
            assert stats == previous["files"][str(relative)], relative
            report["files"][str(relative)] = stats
            commits.append((folder / name, replacement, relative))
            if name == "summary.csv":
                combined.extend(rows)
        per_case = dict(meta.get("case_run_metadata", {case: dict(meta) for case in doc["payload"]["cases"]}))
        per_case[CASE] = fresh
        updated = dict(meta)
        updated.update({"mi_csv_sha256": previous["mi_csv_sha256"],
                        "case_run_metadata": per_case,
                        "provenance_mode": "mixed_case_runs; see case_run_metadata for execution provenance",
                        "last_repair_utc": datetime.now(timezone.utc).isoformat(),
                        "repair_backup": str(run), "last_repaired_scenarios": [CASE],
                        "unchanged_case_mi_verified_against_current_csv": True,
                        "repair_commit_report": str(attempt / "commit_report.json")})
        relative = Path(stage.name) / "run_metadata.json"
        replacement = attempt / relative
        experiment.write_json(replacement, updated)
        report["files"][str(relative)] = {
            "old_sha256": previous["files"][str(relative)]["old_sha256"],
            "new_sha256": experiment.digest(replacement.read_bytes())}
        commits.append((folder / relative.name, replacement, relative))
        if index % 10 == 0:
            print(f"RESTAGED_VERIFIED {index}/60", flush=True)
    relative = Path("summary_all_configs.csv")
    replacement = attempt / relative
    stats = merge_case(ACTIVE / relative, combined, replacement, 360)
    assert stats == previous["files"][str(relative)]
    report["files"][str(relative)] = stats
    # The previously locked combined file is attempted first this time.
    commits.insert(0, (ACTIVE / relative, replacement, relative))
    assert experiment.digest(experiment.CSV.read_bytes()) == previous["mi_csv_sha256"]
    assert experiment.MANIFEST.read_bytes() == manifest_bytes
    for active, replacement, relative in commits:
        assert active.resolve().is_relative_to(ACTIVE.resolve())
        assert replacement.resolve().is_relative_to(attempt.resolve())
        assert experiment.digest(active.read_bytes()) == report["files"][str(relative)]["old_sha256"]
    report_path = attempt / "commit_report.json"
    experiment.write_json(report_path, report)
    committed = []
    try:
        for active, replacement, relative in commits:
            assert experiment.digest(active.read_bytes()) == report["files"][str(relative)]["old_sha256"]
            os.replace(replacement, active)
            committed.append((active, relative))
            assert experiment.digest(active.read_bytes()) == report["files"][str(relative)]["new_sha256"]
        assert experiment.digest(experiment.CSV.read_bytes()) == previous["mi_csv_sha256"]
        assert experiment.MANIFEST.read_bytes() == manifest_bytes
    except BaseException:
        for active, relative in reversed(committed):
            shutil.copy2(run / "original" / relative, active)
        report["status"] = "rolled_back"
        raise
    else:
        report.update(status="complete", configurations=60, replaced_model_rows=900000,
                      replaced_split_rows=18000, replaced_summary_rows=360,
                      untouched_cases_preserved=True, manifest_unchanged=True)
    finally:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("RESUME_COMPLETE", report_path, flush=True)
    return report_path


def main():
    doc = experiment.load_manifest(experiment.MANIFEST)
    manifest_bytes = experiment.MANIFEST.read_bytes()
    csv_hash = experiment.digest(experiment.CSV.read_bytes())
    configurations = sorted(p for p in ACTIVE.iterdir() if p.is_dir())
    assert len(configurations) == 60
    prepared = []
    # All configurations pass the original input/checkpoint validation first.
    for folder in configurations:
        meta = json.loads((folder / "run_metadata.json").read_text())
        assert meta["status"] == "complete" and not meta["gate1"]
        assert meta["manifest_sha256"] == doc["manifest_sha256"]
        assert meta["ddof"] == 1 and meta["shrinkage"] is None
        assert json.loads((folder / "manifest.json").read_text()) == doc
        values, checked_hash = experiment.preflight(
            doc, experiment.ROOT, experiment.CSV, meta["in_sizes_by_case"], meta["bins"], meta["mi_kind"])
        assert checked_hash == csv_hash
        prepared.append((folder, meta, values))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = RECOVERY / (stamp + "_cifar10_vgg16_fixed_splits")
    run.mkdir(parents=True, exist_ok=False)
    report = {"status": "staging", "scenario": CASE, "mi_csv_sha256": csv_hash,
              "manifest_sha256": doc["manifest_sha256"], "files": {},
              "repair_script_sha256": experiment.digest(Path(__file__).read_bytes()),
              "reason": "User corrected MI coordinates; recompute only CIFAR-10 VGG16."}
    combined = []
    commits = []
    for index, (folder, meta, values) in enumerate(prepared, 1):
        stage = run / "recomputed" / folder.name
        experiment.evaluate(
            doc, values, csv_hash, output_dir=stage, csv_path=experiment.CSV,
            model_root=experiment.ROOT, in_size=meta["in_sizes_by_case"],
            bins=meta["bins"], mi_kind=meta["mi_kind"], alphas=meta["alphas"],
            in_size_rate=meta["in_size_rate"], training_sizes=meta["training_sizes"], scenarios=[CASE])
        models, splits, summaries = validate_new(stage, doc, values, meta)
        combined.extend(summaries)
        for name, rows in (("per_model.csv", models), ("per_split.csv", splits), ("summary.csv", summaries)):
            relative = Path(folder.name) / name
            replacement = run / "replacement" / relative
            report["files"][str(relative)] = merge_case(folder/name, rows, replacement, len(rows),
                                                      values if name == "per_model.csv" else None)
            commits.append((folder/name, replacement, relative))
        # Mixed results retain the historical provenance of untouched cases.
        updated = dict(meta)
        fresh = json.loads((stage/"run_metadata.json").read_text())
        per_case = dict(meta.get("case_run_metadata", {case: dict(meta) for case in doc["payload"]["cases"]}))
        per_case[CASE] = fresh
        updated.update({"mi_csv_sha256": csv_hash, "case_run_metadata": per_case,
                        "provenance_mode": "mixed_case_runs; see case_run_metadata for execution provenance",
                        "last_repair_utc": datetime.now(timezone.utc).isoformat(),
                        "repair_backup": str(run), "last_repaired_scenarios": [CASE],
                        "unchanged_case_mi_verified_against_current_csv": True})
        relative = Path(folder.name)/"run_metadata.json"
        replacement = run/"replacement"/relative
        experiment.write_json(replacement, updated)
        report["files"][str(relative)] = {"old_sha256": experiment.digest((folder/relative.name).read_bytes()),
                                         "new_sha256": experiment.digest(replacement.read_bytes())}
        commits.append((folder/relative.name, replacement, relative))
        print(f"STAGED_AND_VERIFIED {index}/60: {folder.name}", flush=True)
    relative = Path("summary_all_configs.csv")
    replacement = run/"replacement"/relative
    report["files"][str(relative)] = merge_case(ACTIVE/relative, combined, replacement, 360)
    commits.append((ACTIVE/relative, replacement, relative))
    assert experiment.digest(experiment.CSV.read_bytes()) == csv_hash
    assert experiment.MANIFEST.read_bytes() == manifest_bytes
    # Validate every destructive target and back up ALL originals before commit.
    for active, replacement, relative in commits:
        assert active.resolve().is_relative_to(ACTIVE.resolve()) and active.is_file()
        assert replacement.resolve().is_relative_to(run.resolve())
        assert experiment.digest(active.read_bytes()) == report["files"][str(relative)]["old_sha256"]
        backup = run/"original"/relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(active, backup)
        assert experiment.digest(backup.read_bytes()) == report["files"][str(relative)]["old_sha256"]
    report_path = run/"repair_report.json"
    experiment.write_json(report_path, report)
    committed = []
    try:
        for active, replacement, relative in commits:
            assert experiment.digest(active.read_bytes()) == report["files"][str(relative)]["old_sha256"]
            os.replace(replacement, active)
            committed.append((active, relative))
            assert experiment.digest(active.read_bytes()) == report["files"][str(relative)]["new_sha256"]
    except BaseException:
        for active, relative in reversed(committed):
            shutil.copy2(run/"original"/relative, active)
        report["status"] = "rolled_back"
        raise
    else:
        report.update(status="complete", configurations=60, replaced_model_rows=900000,
                      replaced_split_rows=18000, replaced_summary_rows=360,
                      untouched_cases_preserved=True, manifest_unchanged=True)
    finally:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("REPAIR_COMPLETE", report_path, flush=True)


if __name__ == "__main__":
    main()
