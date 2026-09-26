"""Reproducible paired negative-model evaluation, independent of the legacy script.

Edit the global configuration below, then run:
python run_hypothesis_test_fixed_splits.py

Generate once. Archive/reuse the JSON manifest for positives and other baselines.
Evaluation never samples models. Paths stored in the manifest are relative to
the model root; model identity is (scenario, seed, model_name), not seed alone.

Gate 1 (APPLY_GATE1, default off) reads the victim MI from a separate table
(VICTIM_CSV) computed on the same grid as the pool and short-circuits negatives
that do not lie lower-right of the victim to T2=0, p=1 before Hotelling.
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import sys
from datetime import datetime, timezone
from decimal import Decimal

# ============================================================================
# Global configuration: edit this section, then run the file directly.
# ============================================================================
# "generate": create a new manifest only (refuses to overwrite an existing one).
# "validate": check the saved manifest, checkpoints and selected MI rows only.
# "evaluate": load the saved manifest and run the negative evaluation.
RUN_MODE = "evaluate"
MASTER_SEED = 20260914       # Only used in generate mode; never resamples on load.
# Fractions of each case's training size; 0.05 means 5%, not 5.
IN_SIZE_RATES = [0.05, 0.10, 0.20, 0.50, 0.75, 1.00]
BINS = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]
MI_KIND = "In"              # "In" or "Out"
ALPHAS = [0.05, 0.01]

BASE = Path(__file__).resolve().parent
ROOT = BASE / "saved_models/vanilla/Negative_Model_Pool_0.0"
CSV = BASE / "saved_logs/vanilla/MI_master_table_neg_pool0.csv"
MANIFEST = BASE / "saved_logs/vanilla/fixed_splits/pool0_80_models_v1.json"
# Each combination gets its own child directory; existing results are protected.
OUTPUT_DIR = BASE / "saved_logs/vanilla/Hypo_Test_FixedSplits"
# Gate 1 (optional, default OFF): necessary condition "suspect lies lower-right of
# the victim in the information plane" (I(X;T) >= victim and I(T;Y) <= victim,
# non-strict so an exact copy passes) checked before Hotelling. Failing negatives
# are short-circuited to T2=0, p=1 (judged negative). The victim vector (seed
# VICTIM_SEED, rate VICTIM_RATE) is read from a SEPARATE table computed on the
# same (in_size, bins) grid as the pool by calculate_MI_victim.py, never from CSV.
# Every manifest case needs exactly one victim row; a missing victim is an error,
# not a silent skip. All results so far were produced with the gate off, and
# off leaves per_model/per_split/summary byte-identical.
APPLY_GATE1 = False
VICTIM_CSV = BASE / "saved_logs/vanilla/MI_master_table_victim.csv"
VICTIM_SEED = 42
VICTIM_RATE = 1.0
# These describe the saved experiment design. Changing them requires a matching
# new manifest, rather than silently changing the identity of an existing run.
K_VALUES = [5, 10, 15, 20, 25, 30]
SCENARIOS = ["CIFAR-10_ResNet-18_25000", "CIFAR-10_VGG16_25000",
             "CIFAR-100_ResNet-18_25000", "CIFAR-100_VGG16_25000",
             "CIFAR-10_DeiT_Plain_25000", "CIFAR-100_DeiT_Distill_25000"]

# Explicit denominators, based on group_size in the six training plans.
# Do not infer training size from model names. Adjust a case here if its training
# data size changes; results always record the denominator and absolute MI size.
TRAINING_SIZES = {
    "CIFAR-10_ResNet-18_25000": 25000,
    "CIFAR-10_VGG16_25000": 25000,
    "CIFAR-100_ResNet-18_25000": 25000,
    "CIFAR-100_VGG16_25000": 25000,
    "CIFAR-10_DeiT_Plain_25000": 25000,
    "CIFAR-100_DeiT_Distill_25000": 25000,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def payload_hash(payload):
    return digest(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode())


def write_json(path, value):
    # Exclusive creation: never silently replace an established manifest/run.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def discover_models(root):
    cases = {}
    for scenario in SCENARIOS:
        models = {}
        for folder in sorted(root.rglob(scenario + "_*_0.0")):
            if not folder.is_dir():
                continue
            suffix = folder.name[len(scenario) + 1:-4]
            require(suffix.isdigit(), f"Invalid model directory: {folder}")
            seed = int(suffix)
            require(str(seed) not in models, f"Duplicate model seed: {scenario}/{seed}")
            checkpoint = folder / "best_epoch.pth"
            require(checkpoint.is_file() and checkpoint.stat().st_size > 0,
                    f"Missing/empty checkpoint: {checkpoint}")
            models[str(seed)] = {"model_name": folder.name,
                                 "relative_dir": folder.relative_to(root).as_posix()}
        require(len(models) == 80, f"{scenario}: expected 80 models, got {len(models)}")
        cases[scenario] = models
    return cases


def generate_manifest(root, master_seed):
    cases = discover_models(root)
    seeds = sorted(map(int, next(iter(cases.values()))))
    require(all(sorted(map(int, models)) == seeds for models in cases.values()),
            "Cases must share the same 80 seed identifiers for paired templates")
    rng = random.Random(master_seed)
    rounds = []
    for index in range(50):
        shuffled = rng.sample(seeds, len(seeds))
        evaluation, candidates = shuffled[:50], shuffled[50:]
        rounds.append({"round_id": index, "eval_negative_seeds": evaluation,
                       "h0_candidate_seeds_ordered": candidates,
                       "by_k": {str(k): {"h0_seeds": candidates[:k],
                                         "unused_seeds": candidates[k:]}
                                for k in K_VALUES}})
    payload = {"schema_version": 1, "design": "paired_nested_h0_fixed_50_negatives",
               "pool_size": 80, "evaluation_size": 50, "n_rounds": 50,
               "k_values": K_VALUES, "pool_seeds": seeds,
               "rate": 0.0, "master_seed": master_seed,
               "generator": "python.random.Random.sample",
               "generator_python": sys.version, "cases": cases, "rounds": rounds}
    return {"manifest_sha256": payload_hash(payload), "payload": payload}


def validate_manifest(document):
    p = document["payload"]
    require(document["manifest_sha256"] == payload_hash(p), "Manifest hash mismatch")
    require(p["schema_version"] == 1 and p["pool_size"] == 80
            and p["evaluation_size"] == 50 and p["n_rounds"] == 50
            and p["k_values"] == K_VALUES and p["rate"] == 0.0,
            "Unsupported experiment design")
    seeds = p["pool_seeds"]
    require(len(seeds) == 80 and len(set(seeds)) == 80
            and all(type(s) is int for s in seeds), "Invalid pool seeds")
    require(set(p["cases"]) == set(SCENARIOS), "Unexpected case inventory")
    for scenario, models in p["cases"].items():
        require(set(models) == {str(s) for s in seeds}, f"Seed mismatch: {scenario}")
        for seed, model in models.items():
            name = f"{scenario}_{seed}_0.0"
            path = Path(model["relative_dir"])
            require(model["model_name"] == name and path.name == name
                    and not path.is_absolute() and ".." not in path.parts,
                    f"Invalid model identity/path: {model}")
    require(len(p["rounds"]) == 50, "Expected 50 rounds")
    for i, item in enumerate(p["rounds"]):
        e, h = item["eval_negative_seeds"], item["h0_candidate_seeds_ordered"]
        require(item["round_id"] == i and len(e) == 50 and len(h) == 30,
                f"Invalid round dimensions: {i}")
        require(len(set(e + h)) == 80 and set(e + h) == set(seeds)
                and all(type(s) is int for s in e + h), f"Overlap/invalid seeds: round {i}")
        require(set(item["by_k"]) == set(map(str, K_VALUES)), f"Invalid k keys: {i}")
        for k in K_VALUES:
            split = item["by_k"][str(k)]
            require(split["h0_seeds"] == h[:k] and split["unused_seeds"] == h[k:],
                    f"Invalid nested H0: round={i}, k={k}")
    return p


def load_manifest(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(document)
    return document


def get_split(document, scenario, round_id, k):
    """Public adapter for future positive/baseline drivers; performs NO sampling.

    Returns model identity dictionaries with seed, model_name, relative_dir.
    Positives reuse 'h0'; their independently frozen positive list is separate.
    """
    p = document["payload"]
    require(scenario in p["cases"] and k in p["k_values"]
            and 0 <= round_id < p["n_rounds"], "Unknown case/round/k")
    item = p["rounds"][round_id]
    def resolve(seeds):
        return [{"seed": s, **p["cases"][scenario][str(s)]} for s in seeds]
    return {"h0": resolve(item["by_k"][str(k)]["h0_seeds"]),
            "evaluation_negative": resolve(item["eval_negative_seeds"]),
            "unused": resolve(item["by_k"][str(k)]["unused_seeds"])}


def preflight(document, root, csv_path, in_size, bins, mi_kind):
    """Validate exactly one finite MI row for every selected model/configuration."""
    import math
    p = validate_manifest(document)
    require(discover_models(root) == p["cases"], "Model inventory differs from manifest")
    raw = csv_path.read_bytes()
    fields = [f"I(X;T)-{mi_kind}", f"I(T;Y)-{mi_kind}"]
    wanted = {m["model_name"]: (scenario, int(seed))
              for scenario, models in p["cases"].items() for seed, m in models.items()}
    sizes = in_size if isinstance(in_size, dict) else {case: in_size for case in p["cases"]}
    require(set(sizes) == set(p["cases"]), "MI sizes must specify every manifest case")
    found = {}
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
        name = row["model_name"]
        if name not in wanted or float(row["in_size"]) != sizes[wanted[name][0]] or float(row["bins"]) != bins:
            continue
        require(name not in found, f"Duplicate MI row: {name}, {in_size}, {bins}")
        scenario, seed = wanted[name]
        require(row["Scenario"] == scenario and float(row["seed"]) == seed
                and float(row["rate"]) == 0.0, f"MI metadata mismatch: {name}")
        values = [float(row[f]) for f in fields]
        require(all(math.isfinite(v) for v in values), f"Nonfinite MI: {name}")
        found[name] = values
    missing = sorted(set(wanted) - set(found))
    require(not missing, f"Missing MI rows: {len(missing)} models; bins={bins}, "
            f"requested sizes={sizes}. Compute these exact MI sizes first. "
            "Examples:\n" + "\n".join(missing[:8]))
    return found, digest(raw)


def preflight_victim(document, csv_path, in_size, bins, mi_kind,
                     victim_seed=VICTIM_SEED, victim_rate=VICTIM_RATE):
    """Validate exactly one finite victim MI row per manifest case (Gate 1 input).

    Same contract as preflight: exact (in_size, bins) match with no fallback size,
    duplicate/missing/nonfinite rows are errors. Returns ({case: [ixt, ity]}, sha256).
    """
    import math
    p = validate_manifest(document)
    raw = csv_path.read_bytes()
    fields = [f"I(X;T)-{mi_kind}", f"I(T;Y)-{mi_kind}"]
    sizes = in_size if isinstance(in_size, dict) else {case: in_size for case in p["cases"]}
    require(set(sizes) == set(p["cases"]), "MI sizes must specify every manifest case")
    found = {}
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
        case = row["Scenario"]
        if (case not in sizes or float(row["seed"]) != victim_seed
                or float(row["rate"]) != victim_rate
                or float(row["in_size"]) != sizes[case] or float(row["bins"]) != bins):
            continue
        require(case not in found, f"Duplicate victim MI row: {case}, {sizes[case]}, {bins}")
        expected = f"{case}_{victim_seed}_{victim_rate}"
        require(row["model_name"] == expected,
                f"Victim model_name mismatch: {row['model_name']} != {expected}")
        values = [float(row[f]) for f in fields]
        require(all(math.isfinite(v) for v in values), f"Nonfinite victim MI: {case}")
        found[case] = values
    missing = sorted(set(sizes) - set(found))
    require(not missing, f"Missing victim MI rows: {missing}; bins={bins}, "
            f"requested sizes={sizes}, seed={victim_seed}, rate={victim_rate}. "
            "Compute them with calculate_MI_victim.py or set APPLY_GATE1 = False.")
    return found, digest(raw)


def score(reference, evaluation):
    import numpy as np
    from scipy.stats import chi2, f
    x = np.asarray(reference, dtype=float)
    y = np.asarray(evaluation, dtype=float)
    k = len(x)
    mu, covariance = x.mean(axis=0), np.cov(x, rowvar=False, ddof=1)
    require(np.linalg.eigvalsh(covariance).min() > 0,
            "Singular/non-positive reference covariance; no split is replaced")
    delta = y - mu
    md2 = np.einsum("ij,ji->i", delta, np.linalg.solve(covariance, delta.T))
    t2 = k / (k + 1.0) * md2
    require(np.isfinite(t2).all() and (t2 >= 0).all(), "Invalid T2 scores")
    return mu, covariance, t2, chi2.sf(t2, 2), f.sf(t2 * (k - 2) / (2 * (k - 1)), 2, k - 2)


def law_statistics(t2, k, p=2):
    """The value each reference law is actually applied to.

    The two laws are not fed the same number. The asymptotic chi2 law takes the
    raw Hotelling T2; the exact predictive law takes (k-p)/(p(k-1)) * T2, which
    is what is compared against F(p, k-p). Returning both keeps the CSV
    self-explanatory instead of leaving the scaling implicit.
    """
    import numpy as np
    t2 = np.asarray(t2, dtype=float)
    return t2, t2 * (k - p) / (p * (k - 1.0))


def summary_statistics(p_by_law):
    """Spread of the p-values one summary cell produced, per reference law.

    A summary row aggregates many evaluations, so no single test statistic
    belongs there. The decision rule is p < alpha, so the p-value spread is what
    makes a cell readable, and it needs no k-dependent threshold: 0.05 and 0.01
    are the only numbers to compare against. The low tail says how close the
    negatives came to a false rejection; the high tail says the worst case among
    the positives. The mean is reported last for completeness, but it is a poor
    summary of a quantity spanning many orders of magnitude; read the median and
    the tails first.

    The two laws are not equally resolved. chi2 underflows to exactly 0 for a
    strong suspect, so its columns saturate on positives; the exact F law keeps
    full resolution in the same cells. Values must come from the same (possibly
    gated) p-values the rejection rates were computed from.

    `p_by_law` maps a law name to the per-round blocks of that law's p-values.
    """
    import numpy as np

    out, size = {}, None
    for law, blocks in p_by_law.items():
        values = [np.asarray(b, dtype=float).ravel() for b in blocks]
        if not values or sum(v.size for v in values) == 0:
            raise ValueError(f"summary_statistics needs at least one {law} p-value")
        p = np.concatenate(values)
        if size is None:
            size = int(p.size)
            out["n_eval_total"] = size
        elif p.size != size:
            raise ValueError("every law must cover the same evaluations")
        out[f"p_{law}_min"] = float(p.min())
        out[f"p_{law}_p05"] = float(np.percentile(p, 5))
        out[f"p_{law}_median"] = float(np.median(p))
        out[f"p_{law}_p95"] = float(np.percentile(p, 95))
        out[f"p_{law}_max"] = float(p.max())
        out[f"p_{law}_mean"] = float(p.mean())
    return out


def passes_gate1(x, victim):
    """Gate 1: suspect must lie lower-right of the victim, I(X;T) >= and I(T;Y) <=.

    Non-strict on purpose (an exact copy of the victim must pass); the same rule
    as run_hypothesis_test.passes_gate1, whose docstring says strict but code is not.
    """
    return bool(x[0] >= victim[0] and x[1] <= victim[1])


def apply_gate1(evaluation, victim, t2, p_chi2, p_f):
    """Short-circuit gate failures to T2=0, p=1 (legacy convention).

    Returns new arrays (inputs untouched) and the per-sample pass flags.
    """
    import numpy as np
    flags = [passes_gate1(x, victim) for x in evaluation]
    keep = np.asarray(flags, dtype=bool)
    gated_t2 = np.where(keep, np.asarray(t2, dtype=float), 0.0)
    gated_chi2 = np.where(keep, np.asarray(p_chi2, dtype=float), 1.0)
    gated_f = np.where(keep, np.asarray(p_f, dtype=float), 1.0)
    return gated_t2, gated_chi2, gated_f, flags


def diagnostics(reference, t2):
    """Legacy normality/LOO diagnostics, descriptive only (KS scores are dependent).

    These never filter or replace a split and are not calibration certificates.
    """
    import numpy as np
    from scipy.stats import chi2, f, norm, shapiro, kstest
    x = np.asarray(reference, dtype=float)
    n, p = x.shape
    centered = x - x.mean(axis=0)
    ml_cov = centered.T @ centered / n
    d = centered @ np.linalg.solve(ml_cov, centered.T)
    b1 = float((d ** 3).sum() / n ** 2)
    b2 = float((np.diag(d) ** 2).mean())
    correction = (p + 1) * (n + 1) * (n + 3) / (n * ((n + 1) * (p + 1) - 6))
    skew_p = float(chi2.sf(n * b1 / 6 * correction, p * (p + 1) * (p + 2) / 6))
    kurt_z = (b2 - p * (p + 2) * (n - 1) / (n + 1)) / np.sqrt(8 * p * (p + 2) / n)
    loo = np.array([score(np.delete(x, i, axis=0), x[i:i + 1])[2][0] for i in range(n)])
    def exact_cdf(t, m):
        return f.cdf(t * (m - p) / (p * (m - 1)), p, m - p)
    return {"mardia_skew_p": skew_p, "mardia_kurt_p": float(2 * norm.sf(abs(kurt_z))),
            "shapiro_p_ixt": float(shapiro(x[:, 0]).pvalue),
            "shapiro_p_ity": float(shapiro(x[:, 1]).pvalue),
            "loo_ks_p_F_descriptive": float(kstest(loo, lambda t: exact_cdf(t, n - 1)).pvalue),
            "loo_ks_p_chi2_descriptive": float(kstest(loo, lambda t: chi2.cdf(t, p)).pvalue),
            "susp_ks_p_F_descriptive": float(kstest(t2, lambda t: exact_cdf(t, n)).pvalue),
            "susp_ks_p_chi2_descriptive": float(kstest(t2, lambda t: chi2.cdf(t, p)).pvalue)}


def evaluate(document, values, csv_hash, *, output_dir, csv_path, model_root,
             in_size, bins, mi_kind, alphas, in_size_rate=None, training_sizes=None,
             victims=None, victim_csv=None, victim_csv_hash=None, scenarios=None,
             round_ids=None, k_values=None, round_by_scenario_k=None):
    """Explicit inputs keep this evaluator reusable by future experiment drivers.

    victims: {case: [ixt, ity]} from preflight_victim enables Gate 1; None (default)
    leaves every output identical to a run without the gate.
    scenarios, round_ids and k_values optionally select existing manifest entries
    without changing the frozen manifest or its hash. Omission retains the
    original all-case, all-round behavior. round_by_scenario_k selects one
    existing round separately for each (scenario, k) cell.
    """
    import numpy as np
    import scipy
    p = document["payload"]
    selected = list(p["cases"]) if scenarios is None else list(scenarios)
    require(bool(selected) and len(selected) == len(set(selected))
            and set(selected) <= set(p["cases"]), "Invalid evaluation scenarios")
    rounds = list(range(p["n_rounds"])) if round_ids is None else list(round_ids)
    ks = list(p["k_values"]) if k_values is None else list(k_values)
    require(bool(rounds) and all(type(r) is int and 0 <= r < p["n_rounds"] for r in rounds)
            and len(set(rounds)) == len(rounds), "Invalid or duplicate round_ids")
    require(bool(ks) and all(type(k) is int and k in p["k_values"] for k in ks)
            and len(set(ks)) == len(ks), "Invalid or duplicate k_values")
    if round_by_scenario_k is not None:
        require(round_ids is None, "Cannot combine round_ids with round_by_scenario_k")
        expected = {(scenario, k) for scenario in selected for k in ks}
        require(set(round_by_scenario_k) == expected,
                "round_by_scenario_k must cover every selected scenario/k cell exactly")
        require(all(type(r) is int and 0 <= r < p["n_rounds"]
                    for r in round_by_scenario_k.values()), "Invalid selected round ID")
    sizes = in_size if isinstance(in_size, dict) else {case: in_size for case in p["cases"]}
    bases = training_sizes or {case: None for case in p["cases"]}
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    metadata = {"status": "running", "manifest_sha256": document["manifest_sha256"],
                "mi_csv_sha256": csv_hash, "script_sha256": digest(Path(__file__).read_bytes()),
                "mi_csv": str(Path(csv_path).resolve()), "model_root": str(Path(model_root).resolve()),
                "in_sizes_by_case": sizes, "in_size_rate": in_size_rate,
                "training_sizes": bases, "bins": bins, "mi_kind": mi_kind,
                "alphas": alphas, "ddof": 1, "shrinkage": None, "gate1": victims is not None,
                "round_ids": rounds if round_by_scenario_k is None else None,
                "k_values": ks, "n_rounds": len(rounds) if round_by_scenario_k is None else 1,
                "laws": {"chi2": "p = sf(stat_chi2, df=2), stat_chi2 = T2",
                         "F": "p = sf(stat_F, dfn=2, dfd=k-2), "
                              "stat_F = (k-2)/(2(k-1)) * T2"},
                "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "interpretation": (f"Mean FPR over {len(rounds) if round_by_scenario_k is None else 1} selected paired split(s) of a fixed pool; "
                                   "SD is split variability, not a confidence interval, "
                                   "and is unavailable for one round.")}
    if round_by_scenario_k is not None:
        metadata["selected_round_by_scenario_k"] = {
            scenario: {str(k): round_by_scenario_k[(scenario, k)] for k in ks}
            for scenario in selected}
    if scenarios is not None:
        metadata["evaluated_scenarios"] = selected
    if victims is not None:
        require(set(victims) == set(p["cases"]),
                "Gate 1 requires a victim MI vector for every manifest case")
        metadata.update({"victim_csv": str(Path(victim_csv).resolve()) if victim_csv else None,
                         "victim_csv_sha256": victim_csv_hash, "victim_seed": VICTIM_SEED,
                         "victim_rate": VICTIM_RATE, "victim_mi_by_case": victims,
                         "gate1_rule": "pass iff I(X;T) >= victim and I(T;Y) <= victim "
                                       "(non-strict); failures get T2=0, p=1"})
    write_json(out / "run_metadata.json", metadata)
    write_json(out / "manifest.json", document)
    per_split = []
    summary = []
    sample_fields = ["manifest_sha256", "in_size_rate", "training_size", "in_size", "bins", "mi_kind", "scenario", "round_id", "k_ref", "seed",
                     "model_name", "ixt", "ity", "T2",
                     "stat_chi2", "p_chi2", "stat_F", "p_F"] + (["gate1"] if victims is not None else [])
    try:
        with (out / "per_model.csv").open("x", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=sample_fields)
            writer.writeheader()
            for scenario in selected:
                in_size = sizes[scenario]
                training_size = bases[scenario]
                for k in ks:
                    cell_rounds = (rounds if round_by_scenario_k is None
                                   else [round_by_scenario_k[(scenario, k)]])
                    cell_p = {"chi2": [], "F": []}
                    first = len(per_split)
                    for r in cell_rounds:
                        split = get_split(document, scenario, r, k)
                        reference = [values[m["model_name"]] for m in split["h0"]]
                        evaluation = [values[m["model_name"]] for m in split["evaluation_negative"]]
                        try:
                            mu, cov, t2, pc, pf = score(reference, evaluation)
                            diag = diagnostics(reference, t2)   # raw T2: law check before gating
                            flags = None
                            if victims is not None:
                                t2, pc, pf, flags = apply_gate1(evaluation, victims[scenario], t2, pc, pf)
                        except (ValueError, np.linalg.LinAlgError) as exc:
                            raise ValueError(f"{scenario}, round={r}, N={k}: {exc}") from exc
                        stat_chi2, stat_f = law_statistics(t2, k)
                        cell_p["chi2"].append(np.asarray(pc, dtype=float))
                        cell_p["F"].append(np.asarray(pf, dtype=float))
                        record = {"manifest_sha256": document["manifest_sha256"],
                                  "in_size_rate": in_size_rate, "training_size": training_size,
                                  "in_size": in_size, "bins": bins, "mi_kind": mi_kind,
                                  "scenario": scenario, "round_id": r, "k_ref": k, "n_eval": 50,
                                  "h0_seeds": json.dumps([m["seed"] for m in split["h0"]]),
                                  "eval_negative_seeds": json.dumps([m["seed"] for m in split["evaluation_negative"]]),
                                  "mu": json.dumps(mu.tolist()), "covariance": json.dumps(cov.tolist()),
                                  "condition_number": float(np.linalg.cond(cov))}
                        record.update(diag)
                        if flags is not None:
                            record["n_gate1_fail"] = int(flags.count(False))
                        for law, probs in [("chi2", pc), ("F", pf)]:
                            for alpha in alphas:
                                fp = int((probs < alpha).sum())
                                record[f"nfp_{law}@{alpha}"] = fp
                                record[f"fpr_{law}@{alpha}"] = fp / 50
                        per_split.append(record)
                        for j, model in enumerate(split["evaluation_negative"]):
                            writer.writerow(dict(zip(sample_fields, [document["manifest_sha256"],
                                in_size_rate, training_size, in_size, bins, mi_kind, scenario, r, k, model["seed"], model["model_name"],
                                *evaluation[j], float(t2[j]),
                                float(stat_chi2[j]), float(pc[j]),
                                float(stat_f[j]), float(pf[j]),
                                *(["pass" if flags[j] else "fail"] if flags is not None else [])])))
                    group = per_split[first:]
                    agg = {"manifest_sha256": document["manifest_sha256"], "scenario": scenario,
                           "in_size_rate": in_size_rate, "training_size": training_size,
                                  "in_size": in_size, "bins": bins, "mi_kind": mi_kind,
                           "k_ref": k, "n_rounds": len(cell_rounds), "n_eval_per_round": 50}
                    for law in ["chi2", "F"]:
                        for alpha in alphas:
                            rates = [row[f"fpr_{law}@{alpha}"] for row in group]
                            agg[f"mean_fpr_{law}@{alpha}"] = float(np.mean(rates))
                            agg[f"std_fpr_{law}@{alpha}"] = (
                                float(np.std(rates, ddof=1)) if len(rates) > 1 else None)
                    agg.update(summary_statistics(cell_p))
                    if victims is not None:
                        agg["mean_n_gate1_fail"] = float(np.mean([row["n_gate1_fail"] for row in group]))
                    summary.append(agg)
                    print(f"Completed {scenario}, N={k}: {len(cell_rounds)} rounds x 50 negatives", flush=True)
        for name, rows in [("per_split.csv", per_split), ("summary.csv", summary)]:
            with (out / name).open("x", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
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
        (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Results: {out}")
    return summary


def configuration_grid(in_sizes, bins_values):
    """Normalize global settings and reject empty/duplicate/invalid combinations."""
    def normalize(value, name):
        items = [value] if type(value) is int else value
        require(isinstance(items, (list, tuple)) and bool(items),
                f"{name} must be a positive integer or a nonempty list/tuple")
        require(all(type(v) is int and v > 0 for v in items),
                f"{name} must contain positive integers")
        require(len(set(items)) == len(items), f"{name} contains duplicates")
        return items
    return [(size, count) for size in normalize(in_sizes, "IN_SIZE")
            for count in normalize(bins_values, "BINS")]


def rate_configuration_grid(rates, bins_values, training_sizes, scenarios):
    """Resolve exact rate x training-size counts without silent rounding."""
    require(isinstance(rates, (list, tuple)) and bool(rates), "IN_SIZE_RATES must be nonempty")
    require(all(type(r) in (int, float) and 0 < r <= 1 for r in rates),
            "IN_SIZE_RATES must contain fractions in (0, 1], e.g. 0.05 for 5%")
    require(len(set(rates)) == len(rates), "IN_SIZE_RATES contains duplicates")
    require(set(training_sizes) == set(scenarios), "TRAINING_SIZES must specify every case")
    require(all(type(n) is int and n > 0 for n in training_sizes.values()),
            "TRAINING_SIZES must contain positive integers")
    counts = [b for _, b in configuration_grid([1], bins_values)]
    grid = []
    for rate in rates:
        sizes = {}
        for case in scenarios:
            size = Decimal(str(rate)) * training_sizes[case]
            require(size == size.to_integral_value() and size >= 1,
                    f"{case}: training size {training_sizes[case]} x rate {rate} = {size}; "
                    "choose a rate producing an integer sample count (no silent rounding)")
            sizes[case] = int(size)
        for bins in counts:
            grid.append((float(rate), sizes, bins))
    return grid


def main():
    """Run the operation selected in the global configuration; no CLI arguments."""
    require(RUN_MODE in ("generate", "validate", "evaluate"),
            "RUN_MODE must be 'generate', 'validate' or 'evaluate'")
    require(MI_KIND in ("In", "Out"), "MI_KIND must be 'In' or 'Out'")
    require(type(APPLY_GATE1) is bool, "APPLY_GATE1 must be True or False")
    configurations = rate_configuration_grid(IN_SIZE_RATES, BINS, TRAINING_SIZES, SCENARIOS)
    require(bool(ALPHAS) and len(set(ALPHAS)) == len(ALPHAS)
            and all(0 < a < 1 for a in ALPHAS),
            "ALPHAS must be nonempty, unique and between 0 and 1")
    manifest_path, model_root, csv_path = Path(MANIFEST), Path(ROOT), Path(CSV)
    if RUN_MODE == "generate":
        require(not manifest_path.exists(), f"Manifest already exists; reuse it: {manifest_path}")
        document = generate_manifest(model_root, MASTER_SEED)
    else:
        document = load_manifest(manifest_path)
    output_root = Path(OUTPUT_DIR)
    combined_path = output_root / "summary_all_configs.csv"
    prepared = []
    # Validate every requested combination before writing any experiment output.
    # Cache the exact checked values so the evaluation uses that CSV snapshot.
    if RUN_MODE == "evaluate":
        require(not combined_path.exists(),
                f"Combined summary already exists; choose a new OUTPUT_DIR: {combined_path}")
    for rate, sizes, bins in configurations:
        rate_tag = format(Decimal(str(rate)).normalize(), "f")
        destination = output_root / f"{MI_KIND}_rate{rate_tag}_bins{bins}"
        if RUN_MODE == "evaluate":
            require(not destination.exists(), f"Result directory already exists: {destination}")
        values, csv_hash = preflight(document, model_root, csv_path, sizes, bins, MI_KIND)
        victims = victim_hash = None
        if APPLY_GATE1:
            victims, victim_hash = preflight_victim(document, Path(VICTIM_CSV), sizes, bins, MI_KIND)
        if prepared:
            require(csv_hash == prepared[0][4], "MI CSV changed during preflight; retry with a stable input")
            require(victim_hash == prepared[0][7], "Victim MI CSV changed during preflight; retry with a stable input")
        prepared.append((rate, sizes, bins, values, csv_hash, destination, victims, victim_hash))
        print(f"Validated {len(values)} models: rate={rate:.0%}, sizes={sizes}, bins={bins}; "
              f"manifest={document['manifest_sha256']}"
              + (f"; gate1 ON, victims={victims}" if victims else ""))
    if RUN_MODE == "generate":
        write_json(manifest_path, document)
        print(f"Saved immutable split manifest: {manifest_path}")
    elif RUN_MODE == "evaluate":
        combined = []
        for rate, sizes, bins, values, csv_hash, destination, victims, victim_hash in prepared:
            combined.extend(evaluate(document, values, csv_hash, output_dir=destination,
                                     csv_path=csv_path, model_root=model_root, in_size=sizes,
                                     bins=bins, mi_kind=MI_KIND, alphas=ALPHAS,
                                     in_size_rate=rate, training_sizes=TRAINING_SIZES,
                                     victims=victims,
                                     victim_csv=Path(VICTIM_CSV) if victims is not None else None,
                                     victim_csv_hash=victim_hash))
        with combined_path.open("x", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(combined[0]))
            writer.writeheader()
            writer.writerows(combined)
        print(f"Completed {len(configurations)} configurations. Combined summary: {combined_path}")



if __name__ == "__main__":
    main()
