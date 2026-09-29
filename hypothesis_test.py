import argparse
import csv
import json
import math
import random
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import f as f_dist

import util

DEFAULT_CONFIG = Path("saved_exp_plan/hypothesis/default.yaml")
NEGATIVE_STAGE = "negative"
POSITIVE_STAGES = ("fine_tune", "prune", "distillation", "extraction")
MI_FIELDS = ("I(X;T)", "I(T;Y)")


def require(condition, message):
    if not condition:
        raise ValueError(message)


# ===========================================================================
# Configuration and cases
# ===========================================================================
def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    require(isinstance(cfg, dict), f"Config is not a mapping: {path}")
    ref, grid = cfg.setdefault("reference", {}), cfg.setdefault("grid", {})
    ref["k"] = int(ref.get("k", 20))
    require(ref["k"] >= 3, "reference.k must be >= 3 (the F law needs k-2 >= 1)")
    ref["seed"] = int(ref.get("seed", 0))
    grid["in_size_rates"] = [float(r) for r in grid.get("in_size_rates", [0.05, 0.1, 0.2, 0.5, 0.75, 1.0])]
    grid["bins"] = [int(b) for b in grid.get("bins", [5, 10, 15, 20, 30, 50, 75, 100, 150, 200])]
    grid["alphas"] = [float(a) for a in grid.get("alphas", [0.05, 0.01])]
    require(grid["alphas"] and all(0 < a < 1 for a in grid["alphas"]), "grid.alphas must lie in (0, 1)")
    cfg.setdefault("positives", {s: {} for s in POSITIVE_STAGES})
    for stage, spec in cfg["positives"].items():
        require(stage in POSITIVE_STAGES, f"positives: unknown stage {stage!r}; known {POSITIVE_STAGES}")
        group_by = tuple(spec.get("group_by") or ("Scenario",))
        spec["group_by"] = group_by if "Scenario" in group_by else ("Scenario",) + group_by
    cfg.setdefault("output_dir", "saved_logs/hypothesis")
    cfg["victim_plan_dir"] = cfg.get("victim_plan_dir") or str(util.stage_plan_dir("victim"))
    return cfg


def load_cases(victim_plan_dir):
    """{scenario: {dataset, arch, training_size, plan}} from the victim plans."""
    files = sorted(Path(victim_plan_dir).glob("*.yaml"))
    require(bool(files), f"No victim plans in {victim_plan_dir}")
    cases = {}
    for path in files:
        with path.open(encoding="utf-8") as stream:
            plan = yaml.safe_load(stream)
        ds = plan["Dataset"]
        require(plan["Scenario_Name"] not in cases, f"Two victim plans declare {plan['Scenario_Name']!r}")
        cases[plan["Scenario_Name"]] = {"dataset": ds["name"] if isinstance(ds, dict) else str(ds),
                                        "arch": plan["Model"], "training_size": int(ds.get("group_size", 50000)),
                                        "plan": path.name}
    return cases


def case_for(cases, dataset, arch, training_size):
    """The case whose pool a suspect of (dataset, arch, training_size) is tested against."""
    hits = [s for s, c in cases.items()
            if c["dataset"] == dataset and c["arch"] == arch and c["training_size"] == int(training_size)]
    require(len(hits) <= 1, f"Ambiguous reference case for ({dataset}, {arch}, {training_size}): {hits}")
    require(hits, f"No reference pool for ({dataset}, {arch}, {training_size}): no victim plan declares that "
                  "scenario. Add it to saved_exp_plan/victim/ and train its negatives with main_negative.py.")
    return hits[0]


def grid_cells(cfg, cases):
    """[(rate, {case: in_size}, bins)] with exact integer sample counts."""
    cells = []
    for rate in cfg["grid"]["in_size_rates"]:
        sizes = {}
        for case, info in cases.items():
            size = Decimal(str(rate)) * info["training_size"]
            require(size == size.to_integral_value() and size >= 1,
                    f"{case}: {info['training_size']} x {rate} is not an integer sample count")
            sizes[case] = int(size)
        cells.extend((rate, sizes, bins) for bins in cfg["grid"]["bins"])
    return cells


def cell_tag(rate, bins):
    return f"rate{format(Decimal(str(rate)).normalize(), 'f')}_bins{bins}"


# ===========================================================================
# MI tables
# ===========================================================================
def read_table(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def index_table(rows):
    """{(model_name, in_size, bins): [ixt, ity]}; a duplicate cell is an error."""
    index = {}
    for row in rows:
        key = (row["model_name"], int(float(row["in_size"])), int(float(row["bins"])))
        require(key not in index, f"duplicate MI row for {key}")
        vector = [float(row[c]) for c in MI_FIELDS]
        require(all(math.isfinite(v) for v in vector), f"nonfinite MI for {key}")
        index[key] = vector
    return index


# ===========================================================================
# Part 1: reference group
# ===========================================================================
def negative_pool(case):
    """Names of the trained negatives of a case, sorted by seed (folders <case>_<seed>_0.0)."""
    root = util.stage_model_root(NEGATIVE_STAGE)
    prefix, suffix = f"{case}_", f"_{round(float(util.NEGATIVE_RATE), 2)}"
    pool = []
    if root.is_dir():
        for folder in root.iterdir():
            name = folder.name
            seed = name[len(prefix):-len(suffix)] if name.startswith(prefix) and name.endswith(suffix) else ""
            if seed.isdigit() and util.checkpoint_exists(folder / util.BEST_CHECKPOINT):
                pool.append((int(seed), name))
    return [name for _, name in sorted(pool)]


def check_pool(cases, cfg):
    """Every case needs at least k+1 trained negatives, each measured on the whole grid.
    Returns {case: [model_name]} and the negative MI index."""
    k = cfg["reference"]["k"]
    table = util.mi_table_path(NEGATIVE_STAGE)
    index = index_table(read_table(table)) if table.is_file() else {}
    pools, problems = {}, []
    for case, info in cases.items():
        pool = negative_pool(case)
        cells = [(int(Decimal(str(r)) * info["training_size"]), b)
                 for r in cfg["grid"]["in_size_rates"] for b in cfg["grid"]["bins"]]
        unmeasured = [n for n in pool if any((n,) + c not in index for c in cells)]
        if len(pool) < k + 1:
            first = 42 if not pool else max(int(n.rsplit("_", 2)[1]) for n in pool) + 1
            problems.append(f"  {case}: {len(pool)} trained negative(s), need at least k+1 = {k + 1}"
                            f"\n      -> python main_negative.py --plans {info['plan']} "
                            f"--seeds {first}:{first + k + 1 - len(pool)}")
        if unmeasured:
            problems.append(f"  {case}: {len(unmeasured)} negative(s) lack MI cells, e.g. {unmeasured[:3]}"
                            f"\n      -> python calculate_MI.py --stage negative --plans {info['plan']}")
        pools[case] = pool
    require(not problems, "The reference pool is incomplete; train or measure these first:\n" + "\n".join(problems))
    return pools, index


def reference_split(pools, cfg):
    """{case: {reference, evaluated}}: ONE seeded shuffle of each pool; the first k models are H0."""
    ref = cfg["reference"]
    rng = random.Random(ref["seed"])
    splits = {}
    for case in pools:
        order = rng.sample(pools[case], len(pools[case]))
        splits[case] = {"reference": order[:ref["k"]], "evaluated": order[ref["k"]:]}
    return splits


def build_reference(cfg):
    cases = load_cases(cfg["victim_plan_dir"])
    print(f"[reference] {len(cases)} case(s): " + ", ".join(cases))
    pools, index = check_pool(cases, cfg)
    splits = reference_split(pools, cfg)
    out = Path(cfg["output_dir"]) / "reference_splits.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"k": cfg["reference"]["k"], "seed": cfg["reference"]["seed"],
                               "cases": cases, "splits": splits}, indent=1), encoding="utf-8")
    for case, s in splits.items():
        print(f"[reference] {case}: {len(pools[case])} negatives -> {len(s['reference'])} reference, "
              f"{len(s['evaluated'])} evaluated")
    print(f"[reference] split saved to {out}")
    return cases, splits, index


def hotelling_f(reference, suspects):
    """Predictive Hotelling T2 of each suspect against the reference sample and its exact F law."""
    x = np.asarray(reference, dtype=float)
    y = np.atleast_2d(np.asarray(suspects, dtype=float))
    k = len(x)
    mu, cov = x.mean(axis=0), np.cov(x, rowvar=False, ddof=1)
    require(np.linalg.eigvalsh(cov).min() > 0, "singular reference covariance")
    delta = y - mu
    t2 = k / (k + 1.0) * np.einsum("ij,ji->i", delta, np.linalg.solve(cov, delta.T))
    stat_f = t2 * (k - 2) / (2 * (k - 1.0))
    return t2, stat_f, f_dist.sf(stat_f, 2, k - 2)


def write_rows(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, key_fields, rate_name, alphas, k):
    """One row per key: rejection rate at each alpha over the models of the group."""
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[c] for c in key_fields), []).append(row)
    out = []
    for key, members in groups.items():
        agg = dict(zip(key_fields, key))
        agg["k"], agg["n"] = k, len(members)
        ps = np.array([m["p_F"] for m in members])
        for alpha in alphas:
            agg[f"{rate_name}@{alpha}"] = float(np.mean(ps < alpha))
        agg["p_F_mean"], agg["p_F_median"] = float(ps.mean()), float(np.median(ps))
        out.append(agg)
    return out


# ===========================================================================
# Part 2: negatives
# ===========================================================================
def evaluate_negatives(cfg, cases, splits, index):
    """Score every evaluated negative against its case's reference set; FPR per case and cell."""
    alphas, k = cfg["grid"]["alphas"], cfg["reference"]["k"]
    root = Path(cfg["output_dir"]) / "negatives"
    combined = []
    for rate, sizes, bins in grid_cells(cfg, cases):
        rows = []
        for case in cases:
            h0, held_out = splits[case]["reference"], splits[case]["evaluated"]
            require(not set(h0) & set(held_out), f"{case}: reference and evaluated sets overlap")
            vec = {n: index[(n, sizes[case], bins)] for n in h0 + held_out}
            t2, stat_f, p_f = hotelling_f([vec[n] for n in h0], [vec[n] for n in held_out])
            for n, t, s, p in zip(held_out, t2, stat_f, p_f):
                rows.append({"in_size_rate": rate, "in_size": sizes[case], "bins": bins, "case": case,
                             "model_name": n, "ixt": vec[n][0], "ity": vec[n][1],
                             "T2": float(t), "stat_F": float(s), "p_F": float(p)})
        summary = summarize(rows, ["case", "in_size_rate", "in_size", "bins"], "fpr", alphas, k)
        out = root / cell_tag(rate, bins)
        write_rows(out / "per_model.csv", rows)
        write_rows(out / "summary.csv", summary)
        combined.extend(summary)
        print(f"[negatives] {cell_tag(rate, bins)}: {len(rows)} negatives evaluated -> {out}")
    write_rows(root / "summary_all_cells.csv", combined)
    return combined


# ===========================================================================
# Part 3: positives
# ===========================================================================
def positive_models(stage):
    root = util.stage_model_root(stage)
    return sorted(d.name for d in root.iterdir() if util.checkpoint_exists(d / util.BEST_CHECKPOINT)) if root.is_dir() else []


def collect_positives(stage, spec, cases):
    """{model_name: {case, victim, scenario, group, training_size}} plus the stage's MI index."""
    names = positive_models(stage)
    require(names, f"{stage}: no trained models under {util.stage_model_root(stage)}")
    table = util.mi_table_path(stage)
    require(table.is_file(), f"{stage}: no MI table {table}; run calculate_MI.py --stage {stage}")
    rows = read_table(table)
    meta = {}
    for row in rows:
        meta.setdefault(row["model_name"], row)
    missing = [n for n in names if n not in meta]
    require(not missing, f"{stage}: {len(missing)} trained model(s) have no MI rows, e.g. {missing[:5]}\n"
                         f"  -> python calculate_MI.py --stage {stage}")
    positives = {}
    for n in names:
        m = meta[n]
        victim, training_size = m["victim_scenario"], int(float(m["training_size"]))
        require(victim in cases, f"{n}: victim_scenario {victim!r} has no victim plan")
        positives[n] = {"case": case_for(cases, cases[victim]["dataset"], m["suspect_arch"], training_size),
                        "victim": victim, "scenario": m["Scenario"], "training_size": training_size,
                        "group": "|".join(m[c] for c in spec["group_by"])}
    return positives, index_table(rows)


def evaluate_positives(cfg, cases, splits, index, stages=None):
    """Score each stage's positives against the reference set of their own architecture."""
    alphas, k = cfg["grid"]["alphas"], cfg["reference"]["k"]
    stages = list(cfg["positives"]) if stages is None else list(stages)
    collected = {stage: collect_positives(stage, cfg["positives"][stage], cases) for stage in stages}
    root = Path(cfg["output_dir"]) / "positives"
    combined = []
    for rate, sizes, bins in grid_cells(cfg, cases):
        rows = []
        for stage in stages:
            positives, pos_index = collected[stage]
            for case in sorted({p["case"] for p in positives.values()}):
                names = sorted(n for n, p in positives.items() if p["case"] == case)
                vectors = []
                for n in names:
                    in_size = int(Decimal(str(rate)) * positives[n]["training_size"])
                    require((n, in_size, bins) in pos_index,
                            f"{stage}: {n} has no MI row at in_size={in_size}, bins={bins}; run calculate_MI.py --stage {stage}")
                    vectors.append(pos_index[(n, in_size, bins)])
                h0 = splits[case]["reference"]
                t2, stat_f, p_f = hotelling_f([index[(n, sizes[case], bins)] for n in h0], vectors)
                for n, v, t, s, p in zip(names, vectors, t2, stat_f, p_f):
                    rows.append({"in_size_rate": rate, "in_size": sizes[case], "bins": bins,
                                 "stage": stage, "group": positives[n]["group"], "scenario": positives[n]["scenario"],
                                 "model_name": n, "victim_scenario": positives[n]["victim"], "reference_case": case,
                                 "ixt": v[0], "ity": v[1], "T2": float(t), "stat_F": float(s), "p_F": float(p)})
        summary = summarize(rows, ["stage", "group", "reference_case", "in_size_rate", "in_size", "bins"],
                            "tpr", alphas, k)
        out = root / cell_tag(rate, bins)
        write_rows(out / "per_model.csv", rows)
        write_rows(out / "summary.csv", summary)
        combined.extend(summary)
        print(f"[positives] {cell_tag(rate, bins)}: {len(rows)} positives evaluated -> {out}")
    write_rows(root / "summary_all_cells.csv", combined)
    return combined


def main(argv=None):
    parser = argparse.ArgumentParser(description="Hypothesis test on the MI tables (reference / negatives / positives)")
    parser.add_argument("part", choices=("reference", "negatives", "positives", "all"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--stages", nargs="*", default=None, help="positives: subset of the configured stages")
    parser.add_argument("--output-root", default=None, help="where saved_models/, saved_logs/, Indices/ live")
    parser.add_argument("--victim-plan-dir", default=None, help="override the victim plan folder that defines the cases")
    parser.add_argument("--rates", default=None, help="override grid.in_size_rates, e.g. 0.05,1.0")
    parser.add_argument("--bins", default=None, help="override grid.bins, e.g. 50")
    args = parser.parse_args(argv)
    if args.output_root:
        util.set_output_root(args.output_root)
    cfg = load_config(args.config)
    if args.victim_plan_dir:
        cfg["victim_plan_dir"] = args.victim_plan_dir
    if args.rates:
        cfg["grid"]["in_size_rates"] = [float(x) for x in args.rates.split(",")]
    if args.bins:
        cfg["grid"]["bins"] = [int(x) for x in args.bins.split(",")]
    if args.output_root and not Path(cfg["output_dir"]).is_absolute():
        cfg["output_dir"] = str(Path(args.output_root) / cfg["output_dir"])
    try:
        cases, splits, index = build_reference(cfg)
        if args.part in ("negatives", "all"):
            evaluate_negatives(cfg, cases, splits, index)
        if args.part in ("positives", "all"):
            evaluate_positives(cfg, cases, splits, index, args.stages)
    except ValueError as exc:
        print(f"\n[stop] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
