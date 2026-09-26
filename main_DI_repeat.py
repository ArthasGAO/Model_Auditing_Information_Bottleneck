"""
Repeated Dataset Inference with the SELF-IMPLEMENTED path (AdvAttack/DI.py).

DI is a random procedure (Blind Walk noise directions, the regressor's init and
minibatch order), so one p-value per model is one draw. This entry re-runs the
whole detection R times (default 50) on a suite of cases and records every
draw, so each case gets a detection RATE and a p-value spread instead of a
single decision.

Suite = every *.yaml in saved_exp_plan/di_repeat_plan/<suite>/ (one victim per
plan, same schema as di_eval_plan: Victim + DI + Positive). The shipped suites
C10 and C100 each hold 3 victims (ResNet-18 / VGG16 / DeiT) x 9 knockoff clones
from saved_models/extraction_final (3 clone archs x seeds 0..2) = 27 cases.

One round, per victim:
  1. regressor g_V: Blind Walk on the victim over the first N_Train private
     (group_A subset, raw_train_clean_set) and public (raw_test_set) images,
     then the 2-layer tanh regressor -- DatasetInferencePipeline.train_regressor
     exactly as main_DI_eval.py calls it (30 epochs, Adam 1e-3).
     --regressor per_round (default): retrained every round under its own seed,
     so a round repeats the full detection. --regressor fixed: trained once and
     reused, so only the suspect-side walks vary.
  2. every suspect: DatasetInferencePipeline.verify_suspect on the first N_Test
     private / public images (the same images as step 1, as in main_DI_eval.py),
     one-sided Welch t-test, Stolen = p < Alpha.

Seeds: every random step is reseeded from a hash of (namespace, role, victim,
[case,] round), so a row does not depend on which other cases ran before it, on
resuming, or on whether the regressor was trained or loaded from disk. Walk
seeds differ from regressor seeds: suspects are probed along fresh directions,
not the ones g_V was trained on. The images themselves are fixed.

Outputs (saved_logs/di_repeat/<suite>_<mode>_n<train>-<test>/):
  rounds.csv            one row per (round, case); append-only, resumable
  regressors.csv        one row per trained regressor, with its train-set separation
  regressors/<victim>/  round_XXX.pt (per_round) or fixed.pt
  scores/<case>/        round_XXX.npz, raw g_V scores (private, public)
  summary_by_case.csv   per case over rounds: detection rate, p-value spread
  summary_by_cell.csv   per (victim, clone arch), the 3 clone seeds pooled
  summary_by_round.csv  per round: how many cases were detected
  run_config.json       settings a resumed run must match

Usage (from E:\\Experiment):
  python main_DI_repeat.py --suite C100                  # 50 rounds x 27 cases
  python main_DI_repeat.py --suite C10
  python main_DI_repeat.py --suite C100 --regressor fixed
  python main_DI_repeat.py --suite C100 --summarize-only
An interrupted run resumes where it stopped; a larger --rounds extends a run.
"""
import argparse
import contextlib
import csv
import glob
import hashlib
import io
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

from main_DI_eval import (
    _setup_victim_context, _detect_suspect_block, _iter_suspects, set_seed, device,
)
from AdvAttack.DI import DatasetInferencePipeline


# =====================================================
# 1. Configuration
# =====================================================
PLAN_ROOT = Path("./saved_exp_plan/di_repeat_plan")
OUT_ROOT = Path("./saved_logs/di_repeat")
ROUNDS = 50
SEED_NAMESPACE = "di_repeat_v1"
REGRESSOR_EPOCHS = 30            # as main_DI_eval._get_or_train_di_pipeline
SUMMARY_ALPHAS = (0.05, 0.01)
ARCH_ORDER = ["ResNet-18", "VGG16", "DeiT"]
PRIVATE_INDICES = "./Indices/{dataset}/group_A_subset_10000_from_25000_seed42.npy"
DI_SOURCE = Path(__file__).parent / "AdvAttack" / "DI.py"

ROUND_COLUMNS = [
    "Run_Timestamp", "Suite", "Dataset", "Round", "Case_Type",
    "Victim_Model", "Victim_Arch", "Suspect_Arch", "Suspect_Model", "Clone_Seed", "Case_ID",
    "Checkpoint_Path", "Checkpoint_SHA256",
    "Regressor_Mode", "Regressor_Seed", "Regressor_Path", "Walk_Seed", "Seed_Namespace",
    "N_Train", "N_Test", "Alpha",
    "Mean_Private", "Mean_Public", "Delta", "T_Stat", "P_Value", "Stolen",
    "Capped_Private_Pct", "Capped_Public_Pct", "Seconds",
]
REGRESSOR_COLUMNS = [
    "Run_Timestamp", "Suite", "Victim_Model", "Victim_Arch", "Round", "Regressor_Mode",
    "Regressor_Seed", "Regressor_Path", "Regressor_SHA256", "N_Train", "Epochs",
    "Train_Loss", "Train_Mean_Private", "Train_Mean_Public",
    "Capped_Private_Pct", "Capped_Public_Pct", "Seconds",
]
# Settings that must not change inside one output directory (rows would mix).
IDENTITY_KEYS = ("n_train", "n_test", "alpha", "regressor_mode", "seed_namespace",
                 "regressor_epochs", "walk", "di_py_sha256")

_CAPPED_RE = re.compile(r"Overall did-not-flip fraction: ([\d.]+)%")
_EPOCH_RE = re.compile(r"Epoch (\d+)/(\d+)\s+loss=(-?[\d.]+)\s+"
                       r"mean\(g_V \| private\)=(-?[\d.]+)\s+mean\(g_V \| public\)=(-?[\d.]+)")
_CLONE_SEED_RE = re.compile(r"_(\d+)_1\.0$")


# =====================================================
# 2. Small helpers
# =====================================================
def derive_seed(namespace, *parts):
    """Stable 31-bit seed from a tuple of labels (order-independent of the run)."""
    text = "|".join(str(p) for p in (namespace,) + parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little") & 0x7FFFFFFF


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def run_captured(fn, verbose):
    """Run fn with its prints captured (DI.py is chatty); echo them if verbose."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn()
    text = buf.getvalue()
    if verbose:
        sys.stdout.write(text)
    return out, text


def parse_capped(text):
    """Did-not-flip percentages printed by compute_embeddings, in call order."""
    return [float(v) for v in _CAPPED_RE.findall(text)]


def append_row(path, columns, row):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if new:
            writer.writeheader()
        writer.writerow(row)


def check_header(path, columns):
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if header != columns:
            raise ValueError(f"{path} has a different column layout; use a new --out-dir.")


SCORE_DIR_LIMIT = 64   # Windows MAX_PATH (LongPathsEnabled = 0 on this machine)


def score_dir_name(case_id, limit=SCORE_DIR_LIMIT):
    """Folder name under scores/ for a case. IDs up to `limit` chars are used as-is
    (every extraction-suite ID); longer ones (post-hoc AT runs, ~190 chars) become
    first 48 chars + sha1[:8], and the full ID is stored inside each npz."""
    if len(case_id) <= limit:
        return case_id
    return f"{case_id[:48]}__{hashlib.sha1(case_id.encode('utf-8')).hexdigest()[:8]}"


def case_label(case_id, width=58):
    """Console label: padded as before for short IDs, full text for long ones
    (post-hoc AT IDs share a 58-char prefix, so truncation would hide the case)."""
    return f"{case_id:{width}s}" if len(case_id) <= width else case_id


_AT_EPS_RE = re.compile(r"_(?:AT)?PGD_eps=([\d.]+)_")   # post-hoc (_PGD_) and one-stage (_ATPGD_) AT names
_AT_FAMILIES = ("Knockoff", "FT-AL", "FT-LL", "RTAL", "DKD", "DFMS", "JBA", "KD")


def at_family_eps(case_id):
    """(family, eps) for AT case IDs (<source>_PGD_eps=E_... or <source>_ATPGD_eps=E_...), else (None, None)."""
    m = _AT_EPS_RE.search(case_id)
    if not m:
        return None, None
    source = case_id[:m.start()] + "_"
    sparsity = re.search(r"_sparsity=([\d.]+)_", source)
    if sparsity:   # pruning runs also carry _FT-AL_ (their recovery step); label them by the pruning
        return f"Prune(s={sparsity.group(1)})", float(m.group(1))
    family = next((k for k in _AT_FAMILIES if f"_{k}_" in source), case_id[:m.start()])
    return family, float(m.group(1))


_AT_SEED_KEYS = ("atseed", "epoch", "ckpt")   # vary between repeats of one setting; pooled in the grid


def at_source_setting(case_id):
    """(source, setting tokens) of an AT case ID: the model the AT started from, and the AT
    hyper-parameters after the eps (key=value tokens, seeds / epoch / checkpoint dropped)."""
    m = _AT_EPS_RE.search(case_id)
    if not m:
        return None, ()
    tokens = re.findall(r"([A-Za-z]+=[^_]+)", case_id[m.end():])
    # The source's own seeds (ftseed=N, the clone / student seed in ..._<seed>_<rate>) are repeats of
    # one source, not different sources: blank them so they pool like atseed does.
    source = re.sub(r"(ftseed|kdseed|seed)=\d+", r"\1=*", case_id[:m.start()])
    source = re.sub(r"_\d+_(\d+(?:\.\d+)?)$", r"_*_\1", source)
    return source, tuple(t for t in tokens if t.split("=")[0] not in _AT_SEED_KEYS)


def _at_row_labels(d):
    """Row label per case: the family, plus the source when a family has several sources in
    the table (e.g. Knockoff Cross100 vs Same10), plus the AT setting when one row would
    otherwise mix several settings. Suites with one source and one setting per family keep
    the plain family label."""
    src_tag = {}
    for fam, grp in d.groupby("Family"):
        sources = sorted(grp["Source"].unique())
        for s in sources:
            tail = s.split(f"_{fam}_", 1)[1].split("_")[0] if f"_{fam}_" in s else s
            src_tag[(fam, s)] = fam if len(sources) == 1 else f"{fam}({tail})"
    common = set.intersection(*(set(t) for t in d["Setting"])) if len(d) else set()
    base = [src_tag[(f, s)] for f, s in zip(d["Family"], d["Source"])]
    labels = []
    for b, setting in zip(base, d["Setting"]):
        n_settings = d.loc[[x == b for x in base], "Setting"].nunique()
        extra = ",".join(t for t in setting if t not in common) or "default"
        labels.append(b if n_settings == 1 else f"{b} [{extra}]")
    return labels


def summarize_at_grid(df, p_cols, out_dir, alphas=SUMMARY_ALPHAS):
    """Family x eps view for suites of AT models; no-op for other suites.

    p_cols: [(label, column)], e.g. [("", "P_Value")] or [("M", "P_Value_M"), ("Full", "P_Value_Full")].
    Writes summary_by_family_eps.csv and prints one row x eps matrix per p column. Rows are the
    family, split by source and AT setting only when a family has several (see _at_row_labels);
    seeds (atseed / ftseed repeats of the same setting) and rounds are pooled.
    """
    fe = [at_family_eps(c) for c in df["Case_ID"]]
    if all(f is None for f, _ in fe):
        return None
    ss = [at_source_setting(c) for c in df["Case_ID"]]
    d = df.assign(Family=[f for f, _ in fe], Eps=[e for _, e in fe],
                  Source=[s for s, _ in ss], Setting=[t for _, t in ss]).dropna(subset=["Family"])
    d = d.assign(Row=_at_row_labels(d), Setting=[",".join(t) for t in d["Setting"]])
    agg = {"N_Cases": ("Case_ID", "nunique"), "N_Rows": ("Round", "size")}
    for label, col in p_cols:
        tag = f"_{label}" if label else ""
        for a in alphas:
            agg[f"Detect_Rate{tag}@{a}"] = (col, lambda p, a=a: (p < a).mean())
        agg[f"P{tag}_Median"] = (col, "median")
    tab = (d.groupby(["Victim_Model", "Row", "Family", "Source", "Setting", "Eps"], sort=True)
           .agg(**agg).reset_index())
    tab.to_csv(Path(out_dir) / "summary_by_family_eps.csv", index=False)
    eps_label = lambda e: f"{round(e * 255)}/255" if abs(e * 255 - round(e * 255)) < 0.01 else f"{e:g}"
    for label, col in p_cols:
        tag = f"_{label}" if label else ""
        for a in alphas:
            mat = tab.pivot_table(index="Row", columns="Eps", values=f"Detect_Rate{tag}@{a}")
            mat.columns = [eps_label(c) for c in mat.columns]
            print(f"\nAT models: Detect_Rate{tag}@{a} (rows = attack family, cols = PGD eps; "
                  f"cases x rounds pooled)")
            print(mat.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"  -> {Path(out_dir) / 'summary_by_family_eps.csv'}")
    return tab


def _single(value, key):
    values = value if isinstance(value, (list, tuple)) else [value]
    if len(values) != 1:
        raise ValueError(f"DI.{key} must hold ONE value for a repeat run, got {values}.")
    return int(values[0])


# =====================================================
# 3. Suite loading (all models stay on the GPU for the whole run)
# =====================================================
def load_suite(plan_files, include_victim_check=False, case_filter=None):
    """Parse every plan, load its victim and suspects once. Returns (victims, di)."""
    victims, di_settings = [], set()
    pattern = re.compile(case_filter) if case_filter else None
    for plan in plan_files:
        print(f"\n==> Plan {plan}")
        ctx = _setup_victim_context(plan, build_raw_loader=False)
        exp_yaml, ds = ctx["exp_yaml"], ctx["victim_dataset_obj"]
        di_cfg = exp_yaml.get("DI", {})
        di_settings.add((_single(di_cfg.get("N_Train_Samples", 1000), "N_Train_Samples"),
                         _single(di_cfg.get("N_Test_Samples", 1000), "N_Test_Samples"),
                         float(di_cfg.get("Alpha", 0.05))))
        s_type, suspect_cfg = _detect_suspect_block(exp_yaml)
        if s_type != "positive":
            raise ValueError(f"{plan}: a repeat suite plan declares Positive suspects only.")

        victim_cfg = ctx["victim_cfg"]
        key = victim_cfg["Model_Name"]
        victim = {
            "key": key, "arch": ctx["victim_arch"], "dataset": ctx["victim_ds_cfg"]["name"],
            "plan": str(plan), "ds": ds, "model": ctx["victim_model"],
            "pipeline": DatasetInferencePipeline(ctx["victim_model"], device=device),
            "private_idx": np.load(PRIVATE_INDICES.format(dataset=ctx["victim_ds_cfg"]["name"])),
            "regressor_tag": None, "cases": [],
        }
        blocks = suspect_cfg if isinstance(suspect_cfg, list) else [suspect_cfg]
        for block in blocks:
            model_cfg = block.get("Model_Config") or {}
            for rec in _iter_suspects(s_type, block, ds, ctx["victim_num_classes"],
                                      ctx["victim_arch"], ctx["victim_id"]):
                case_id = rec["scenario_name"]
                if pattern and not pattern.search(case_id):
                    continue
                m = _CLONE_SEED_RE.search(case_id)
                victim["cases"].append({
                    "case_id": case_id, "type": "positive", "model": rec["model"],
                    "suspect_arch": rec["suspect_arch"],
                    "suspect_model": model_cfg.get("model_name", rec["suspect_arch"]),
                    "clone_seed": int(m.group(1)) if m else "",
                    "ckpt_path": str(Path(rec["ckpt_path"]).resolve()),
                    "ckpt_sha256": sha256_file(rec["ckpt_path"]),
                })
        if include_victim_check:
            case_id = f"{key}_42_1.0__victim_self"
            if not pattern or pattern.search(case_id):
                victim_model_cfg = victim_cfg.get("Model_Config") or {}
                victim["cases"].append({
                    "case_id": case_id, "type": "victim", "model": ctx["victim_model"],
                    "suspect_arch": ctx["victim_arch"],
                    "suspect_model": victim_model_cfg.get("model_name", ctx["victim_arch"]),
                    "clone_seed": "", "ckpt_path": "", "ckpt_sha256": "",
                })
        if victim["cases"]:
            victims.append(victim)
        print(f"  {key}: {len(victim['cases'])} case(s)")
    if len(di_settings) != 1:
        raise ValueError(f"Plans of one suite must share N_Train / N_Test / Alpha, got {di_settings}.")
    ids = [c["case_id"] for v in victims for c in v["cases"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate Case_ID in the suite.")
    return victims, di_settings.pop()


# =====================================================
# 4. One regressor / one detection
# =====================================================
def ensure_regressor(victim, round_idx, run):
    """Load or train this round's g_V for `victim`. Returns (seed, path)."""
    fixed = run["regressor_mode"] == "fixed"
    tag = "fixed" if fixed else f"round_{round_idx:03d}"
    seed = derive_seed(run["seed_namespace"], "regressor", victim["key"], tag)
    path = run["out_dir"] / "regressors" / victim["key"] / f"{tag}.pt"
    if victim["regressor_tag"] == tag:
        return seed, path
    pipeline, ds = victim["pipeline"], victim["ds"]
    if path.is_file():
        run_captured(lambda: pipeline.load_regressor(path), run["verbose"])
    else:
        set_seed(seed)
        t0 = time.time()
        _, text = run_captured(lambda: pipeline.train_regressor(
            private_dataset=ds.raw_train_clean_set, public_dataset=ds.raw_test_set,
            private_indices=victim["private_idx"], n_train_samples=run["n_train"],
            regressor_epochs=REGRESSOR_EPOCHS,
        ), run["verbose"])
        seconds = time.time() - t0
        run_captured(lambda: pipeline.save_regressor(path), run["verbose"])
        capped = parse_capped(text) + [None, None]
        epochs = _EPOCH_RE.findall(text)
        last = epochs[-1] if epochs else ("", "", "", "", "")
        append_row(run["out_dir"] / "regressors.csv", REGRESSOR_COLUMNS, {
            "Run_Timestamp": datetime.now().isoformat(timespec="seconds"),
            "Suite": run["suite"], "Victim_Model": victim["key"], "Victim_Arch": victim["arch"],
            "Round": "fixed" if fixed else round_idx, "Regressor_Mode": run["regressor_mode"],
            "Regressor_Seed": seed, "Regressor_Path": str(path), "Regressor_SHA256": sha256_file(path),
            "N_Train": run["n_train"], "Epochs": REGRESSOR_EPOCHS,
            "Train_Loss": last[2], "Train_Mean_Private": last[3], "Train_Mean_Public": last[4],
            "Capped_Private_Pct": capped[0], "Capped_Public_Pct": capped[1],
            "Seconds": round(seconds, 2),
        })
        print(f"  [g_V] {victim['key']} {tag}: trained in {seconds:.0f}s "
              f"(train mean private {last[3]} / public {last[4]})")
    victim["regressor_tag"] = tag
    return seed, path


def run_case(victim, case, round_idx, reg_seed, reg_path, run):
    """One detection of one case in one round -> one rounds.csv row."""
    ds = victim["ds"]
    walk_seed = derive_seed(run["seed_namespace"], "walk", victim["key"], case["case_id"], round_idx)
    set_seed(walk_seed)
    t0 = time.time()
    result, text = run_captured(lambda: victim["pipeline"].verify_suspect(
        suspect_model=case["model"],
        private_dataset=ds.raw_train_clean_set, public_dataset=ds.raw_test_set,
        private_indices=victim["private_idx"], n_test_samples=run["n_test"], alpha=run["alpha"],
    ), run["verbose"])
    seconds = time.time() - t0
    stats = ("mean_private", "mean_public", "delta", "t_stat", "p_value")
    if not all(np.isfinite(result[k]) for k in stats):
        raise ValueError(f"Non-finite DI statistics for {case['case_id']} round {round_idx}: "
                         f"{ {k: result[k] for k in stats} }")

    # Scores first, then the row: a row on disk always has its scores.
    score_dir = score_dir_name(case["case_id"])
    score_path = run["out_dir"] / "scores" / score_dir / f"round_{round_idx:03d}.npz"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    extra = {} if score_dir == case["case_id"] else {"case_id": np.array(case["case_id"])}
    np.savez_compressed(score_path,
                        private=np.asarray(result["scores_private"], dtype=np.float32),
                        public=np.asarray(result["scores_public"], dtype=np.float32), **extra)
    capped = parse_capped(text) + [None, None]
    row = {
        "Run_Timestamp": datetime.now().isoformat(timespec="seconds"),
        "Suite": run["suite"], "Dataset": victim["dataset"], "Round": round_idx,
        "Case_Type": case["type"],
        "Victim_Model": victim["key"], "Victim_Arch": victim["arch"],
        "Suspect_Arch": case["suspect_arch"], "Suspect_Model": case["suspect_model"],
        "Clone_Seed": case["clone_seed"], "Case_ID": case["case_id"],
        "Checkpoint_Path": case["ckpt_path"], "Checkpoint_SHA256": case["ckpt_sha256"],
        "Regressor_Mode": run["regressor_mode"], "Regressor_Seed": reg_seed,
        "Regressor_Path": str(reg_path), "Walk_Seed": walk_seed,
        "Seed_Namespace": run["seed_namespace"],
        "N_Train": run["n_train"], "N_Test": run["n_test"], "Alpha": run["alpha"],
        "Mean_Private": round(result["mean_private"], 6),
        "Mean_Public": round(result["mean_public"], 6),
        "Delta": round(result["delta"], 6),
        "T_Stat": round(result["t_stat"], 6),
        "P_Value": f"{result['p_value']:.6e}",
        "Stolen": int(result["stolen"]),
        "Capped_Private_Pct": capped[0], "Capped_Public_Pct": capped[1],
        "Seconds": round(seconds, 2),
    }
    append_row(run["out_dir"] / "rounds.csv", ROUND_COLUMNS, row)
    return row


# =====================================================
# 5. Driver
# =====================================================
def build_run_config(suite, plan_files, n_train, n_test, alpha, regressor_mode, seed_namespace):
    probe = DatasetInferencePipeline(None, device=device)
    return {
        "suite": suite,
        "plans": {str(p): sha256_file(p) for p in plan_files},
        "n_train": n_train, "n_test": n_test, "alpha": alpha,
        "regressor_mode": regressor_mode, "seed_namespace": seed_namespace,
        "regressor_epochs": REGRESSOR_EPOCHS,
        "walk": {"n_samples": probe.n_samples, "noise_uniform": probe.noise_uniform,
                 "noise_gaussian": probe.noise_gaussian, "noise_laplace": probe.noise_laplace,
                 "max_steps": probe.max_steps, "point_batch_size": probe.point_batch_size},
        "di_py_sha256": sha256_file(DI_SOURCE),
        "private_indices": PRIVATE_INDICES,
        "created": datetime.now().isoformat(timespec="seconds"),
    }


def check_or_write_config(out_dir, config):
    path = out_dir / "run_config.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        diff = {k: (old.get(k), config[k]) for k in IDENTITY_KEYS if old.get(k) != config[k]}
        if diff:
            raise ValueError(f"{out_dir} was started with different settings {diff}; "
                             "resume with the original ones or pick a new --out-dir.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def read_done(out_dir):
    path = out_dir / "rounds.csv"
    if not path.exists():
        return set(), []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {(int(r["Round"]), r["Case_ID"]) for r in rows}, rows


def main_di_repeat(plan_dir, rounds=ROUNDS, regressor_mode="per_round", out_dir=None,
                   n_train=None, n_test=None, seed_namespace=SEED_NAMESPACE,
                   victim_check=False, case_filter=None, verbose=False):
    if regressor_mode not in ("per_round", "fixed"):
        raise ValueError("regressor_mode must be 'per_round' or 'fixed'.")
    plan_dir = Path(plan_dir)
    plan_files = sorted(Path(p) for p in glob.glob(str(plan_dir / "*.yaml")))
    if not plan_files:
        raise FileNotFoundError(f"No plans in {plan_dir}")
    suite = plan_dir.name

    victims, (plan_n_train, plan_n_test, alpha) = load_suite(plan_files, victim_check, case_filter)
    n_train = int(n_train or plan_n_train)
    n_test = int(n_test or plan_n_test)
    out_dir = Path(out_dir) if out_dir else OUT_ROOT / f"{suite}_{regressor_mode}_n{n_train}-{n_test}"
    check_or_write_config(out_dir, build_run_config(
        suite, plan_files, n_train, n_test, alpha, regressor_mode, seed_namespace))
    for name, cols in (("rounds.csv", ROUND_COLUMNS), ("regressors.csv", REGRESSOR_COLUMNS)):
        check_header(out_dir / name, cols)

    run = {"suite": suite, "out_dir": out_dir, "n_train": n_train, "n_test": n_test,
           "alpha": alpha, "regressor_mode": regressor_mode,
           "seed_namespace": seed_namespace, "verbose": verbose}
    n_cases = sum(len(v["cases"]) for v in victims)
    done, rows = read_done(out_dir)
    todo_total = sum(1 for r in range(rounds) for v in victims for c in v["cases"]
                     if (r, c["case_id"]) not in done)
    print(f"\n==> Suite {suite}: {len(victims)} victim(s), {n_cases} case(s) per round, "
          f"{rounds} round(s); {todo_total} detection(s) to run, "
          f"{rounds * n_cases - todo_total} already in {out_dir / 'rounds.csv'}")
    print(f"    n_train={n_train} n_test={n_test} alpha={alpha} regressor={regressor_mode} "
          f"namespace={seed_namespace}")

    t_start, finished = time.time(), 0
    for r in range(rounds):
        pending = [(v, [c for c in v["cases"] if (r, c["case_id"]) not in done]) for v in victims]
        if not any(cases for _, cases in pending):
            continue
        print(f"\n=== Round {r} ({r + 1}/{rounds}) ===")
        for victim, cases in pending:
            if not cases:
                continue
            reg_seed, reg_path = ensure_regressor(victim, r, run)
            for case in cases:
                row = run_case(victim, case, r, reg_seed, reg_path, run)
                done.add((r, case["case_id"]))
                rows.append(row)
                finished += 1
                print(f"  r{r:02d} {case_label(case['case_id'])} delta={row['Delta']:+.4f} "
                      f"t={row['T_Stat']:7.2f} p={float(row['P_Value']):.2e} "
                      f"{'STOLEN' if row['Stolen'] else '------'} ({row['Seconds']:.0f}s)")
        this_round = [x for x in rows if int(x["Round"]) == r and x["Case_Type"] == "positive"]
        detected = sum(int(x["Stolen"]) for x in this_round)
        elapsed = time.time() - t_start
        eta = elapsed / finished * (todo_total - finished) if finished else 0.0
        print(f"--- round {r}: {detected}/{len(this_round)} positives detected at alpha={alpha} "
              f"| elapsed {elapsed / 60:.1f} min, ETA {eta / 60:.1f} min")

    summarize(out_dir)
    return out_dir


# =====================================================
# 6. Summaries
# =====================================================
def _rate(p, a):
    return (p < a).mean()


def summarize(out_dir):
    """Write summary_by_case / _by_cell / _by_round next to rounds.csv and print the matrix."""
    out_dir = Path(out_dir)
    path = out_dir / "rounds.csv"
    if not path.exists():
        print(f"No {path}; nothing to summarize.")
        return None
    df = pd.read_csv(path)
    if df.empty:
        print(f"{path} is empty; nothing to summarize.")
        return None
    df["Log10_P"] = np.log10(df["P_Value"].clip(lower=1e-300))

    case_keys = ["Case_Type", "Victim_Model", "Victim_Arch", "Suspect_Arch", "Suspect_Model",
                 "Clone_Seed", "Case_ID"]
    agg = {"N_Rounds": ("Round", "nunique")}
    for a in SUMMARY_ALPHAS:
        agg[f"Detect_Rate@{a}"] = ("P_Value", lambda p, a=a: _rate(p, a))
    agg.update({
        "P_Median": ("P_Value", "median"),
        "P_Q25": ("P_Value", lambda p: p.quantile(0.25)),
        "P_Q75": ("P_Value", lambda p: p.quantile(0.75)),
        "P_Min": ("P_Value", "min"), "P_Max": ("P_Value", "max"),
        "Log10P_Mean": ("Log10_P", "mean"), "Log10P_Std": ("Log10_P", "std"),
        "T_Mean": ("T_Stat", "mean"), "T_Std": ("T_Stat", "std"),
        "Delta_Mean": ("Delta", "mean"), "Delta_Std": ("Delta", "std"),
    })
    by_case = df.groupby(case_keys, dropna=False, sort=False).agg(**agg).reset_index()
    by_case.to_csv(out_dir / "summary_by_case.csv", index=False)

    pos = df[df["Case_Type"] == "positive"]
    pos_cases = by_case[by_case["Case_Type"] == "positive"]
    first_alpha = f"Detect_Rate@{SUMMARY_ALPHAS[0]}"
    cell_keys = ["Victim_Model", "Victim_Arch", "Suspect_Arch"]
    cell_agg = {"N_Cases": ("Case_ID", "nunique"), "N_Rows": ("Round", "size")}
    for a in SUMMARY_ALPHAS:
        cell_agg[f"Detect_Rate@{a}"] = ("P_Value", lambda p, a=a: _rate(p, a))
    cell_agg.update({"P_Median": ("P_Value", "median"),
                     "Log10P_Mean": ("Log10_P", "mean"), "Log10P_Std": ("Log10_P", "std"),
                     "T_Mean": ("T_Stat", "mean"), "T_Std": ("T_Stat", "std")})
    by_cell = pos.groupby(cell_keys, sort=False).agg(**cell_agg).reset_index()
    stability = pos_cases.groupby(cell_keys, sort=False)[first_alpha].agg(
        Cases_Always=lambda r: int((r == 1).sum()),
        Cases_Never=lambda r: int((r == 0).sum()),
        Cases_Mixed=lambda r: int(((r > 0) & (r < 1)).sum()),
    ).reset_index()
    by_cell = by_cell.merge(stability, on=cell_keys, how="left")
    by_cell.to_csv(out_dir / "summary_by_cell.csv", index=False)

    round_agg = {"N_Cases": ("Case_ID", "nunique")}
    for a in SUMMARY_ALPHAS:
        round_agg[f"Detected@{a}"] = ("P_Value", lambda p, a=a: int((p < a).sum()))
    round_agg.update({"P_Median": ("P_Value", "median"), "P_Max": ("P_Value", "max")})
    by_round = pos.groupby("Round").agg(**round_agg).reset_index()
    by_round.to_csv(out_dir / "summary_by_round.csv", index=False)

    rounds_seen = df["Round"].nunique()
    print(f"\n==> Summary of {out_dir} ({rounds_seen} round(s), {len(pos_cases)} positive case(s))")
    for a in SUMMARY_ALPHAS:
        mat = by_cell.pivot_table(index="Victim_Arch", columns="Suspect_Arch",
                                  values=f"Detect_Rate@{a}")
        mat = mat.reindex(index=[x for x in ARCH_ORDER if x in mat.index],
                          columns=[x for x in ARCH_ORDER if x in mat.columns])
        print(f"\nDetection rate at alpha={a} (rows = victim, cols = clone; 3 seeds x rounds pooled)")
        print(mat.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nPer-case stability at alpha={SUMMARY_ALPHAS[0]} "
          f"(always / never / mixed over rounds, out of N_Cases):")
    print(by_cell[cell_keys + ["N_Cases", "Cases_Always", "Cases_Never", "Cases_Mixed"]]
          .to_string(index=False))
    victims_rows = by_case[by_case["Case_Type"] == "victim"]
    if not victims_rows.empty:
        print("\nVictim self-check:")
        print(victims_rows[["Victim_Model", "N_Rounds", first_alpha, "P_Median", "P_Max"]]
              .to_string(index=False))
    summarize_at_grid(pos, [("", "P_Value")], out_dir)
    print(f"\n  -> {out_dir / 'summary_by_case.csv'}\n  -> {out_dir / 'summary_by_cell.csv'}"
          f"\n  -> {out_dir / 'summary_by_round.csv'}")
    return by_case, by_cell, by_round


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repeat the self-implemented DI detection R times.")
    parser.add_argument("--suite", default="C100", help=f"plan folder under {PLAN_ROOT} (C10 or C100)")
    parser.add_argument("--plan-dir", help="explicit plan folder; overrides --suite")
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--regressor", choices=("per_round", "fixed"), default="per_round",
                        help="retrain g_V every round (full repeat) or train once")
    parser.add_argument("--out-dir", help="default: saved_logs/di_repeat/<suite>_<mode>_n<train>-<test>")
    parser.add_argument("--n-train", type=int, help="override the plans' DI.N_Train_Samples")
    parser.add_argument("--n-test", type=int, help="override the plans' DI.N_Test_Samples")
    parser.add_argument("--seed-namespace", default=SEED_NAMESPACE)
    parser.add_argument("--victim-check", action="store_true",
                        help="also test each victim against its own g_V every round (reference row)")
    parser.add_argument("--case-filter", help="regex on Case_ID; run only matching cases")
    parser.add_argument("--verbose", action="store_true", help="echo DI.py's per-walk prints")
    parser.add_argument("--summarize-only", action="store_true")
    cli = parser.parse_args()

    torch.multiprocessing.set_start_method("spawn", force=True)
    plan_dir = Path(cli.plan_dir) if cli.plan_dir else PLAN_ROOT / cli.suite
    if cli.summarize_only:
        if cli.out_dir:
            target = Path(cli.out_dir)
        else:
            matches = sorted(OUT_ROOT.glob(f"{plan_dir.name}_{cli.regressor}_n*"))
            if len(matches) != 1:
                raise SystemExit(f"Pass --out-dir; found {[str(m) for m in matches]}")
            target = matches[0]
        summarize(target)
    else:
        main_di_repeat(plan_dir, rounds=cli.rounds, regressor_mode=cli.regressor,
                       out_dir=cli.out_dir, n_train=cli.n_train, n_test=cli.n_test,
                       seed_namespace=cli.seed_namespace, victim_check=cli.victim_check,
                       case_filter=cli.case_filter, verbose=cli.verbose)
