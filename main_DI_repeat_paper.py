"""
Repeated Dataset Inference, PAPER-ALIGNED protocol (AdvAttack/DI_paper.py).

Same suites, rounds, seeding and resume logic as main_DI_repeat.py, with the
three changes of AdvAttack/DI_paper.py (Maini et al., ICLR 2021):

  1. g_V has a tanh output (bounded scores, as in the official notebook);
  2. g_V is trained on the victim's embeddings of the TRAIN images and every
     suspect is tested on disjoint TEST images:
        train = private group_A-subset positions [0, N_Train)       + test-set images [0, N_Train)
        test  = private group_A-subset positions [N_Train, N_Train+N_Test) + test-set images [same]
  3. decision by the m-sample protocol: m = 10 revealed samples per side,
     100 draws, harmonic mean of the one-sided Welch p-values
     (P_Value_M / Stolen_M). The full-sample Welch test on all N_Test held-out
     images is kept as P_Value_Full / Stolen_Full.
Everything else (Blind Walk, 30-dim embedding, standardization, hidden layer,
Adam 1e-3, 30 epochs, N_Train / N_Test / Alpha from the plans) is unchanged.

Independence from main_DI_repeat.py: AdvAttack/DI.py, main_DI_repeat.py and
util_adv.py are imported, never modified; outputs, regressor files and scores
live under saved_logs/di_repeat_paper/, so neither run can touch the other's
values. The default seed namespace is the SAME as main_DI_repeat.py's, so round
r trains g_V from the same victim walks (same images, same noise) in both runs:
the two regressors differ only by the tanh head, which makes round-by-round
comparisons paired. Pass --seed-namespace to draw independent noise instead.

Outputs (saved_logs/di_repeat_paper/<suite>_<mode>_n<train>-<test>_m<m>x<reps>/):
  rounds.csv, regressors.csv, regressors/, scores/ (private, public, p_values_m),
  summary_by_case.csv, summary_by_cell.csv, summary_by_round.csv, run_config.json

Usage (from E:\\Experiment):
  python main_DI_repeat_paper.py --suite C100            # 50 rounds x 27 cases
  python main_DI_repeat_paper.py --suite C10
  python main_DI_repeat_paper.py --suite C100 --summarize-only
"""
import argparse
import glob
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

from main_DI_eval import set_seed, device
from main_DI_repeat import (
    PLAN_ROOT, ROUNDS, SEED_NAMESPACE, REGRESSOR_EPOCHS, ARCH_ORDER, SUMMARY_ALPHAS,
    DI_SOURCE, PRIVATE_INDICES, _EPOCH_RE,
    load_suite, derive_seed, sha256_file, run_captured, parse_capped,
    append_row, check_header, read_done,
    score_dir_name, case_label, summarize_at_grid,
)
from AdvAttack.DI_paper import PaperDIPipeline, PROTOCOL_TAG

OUT_ROOT = Path("./saved_logs/di_repeat_paper")
DI_PAPER_SOURCE = Path(__file__).parent / "AdvAttack" / "DI_paper.py"
M_REVEALED = 10          # paper Table 1 / official generate_table selected_m
M_REPS = 100             # paper: 100 repetitions, harmonic mean

ROUND_COLUMNS = [
    "Run_Timestamp", "Suite", "Dataset", "Round", "Case_Type",
    "Victim_Model", "Victim_Arch", "Suspect_Arch", "Suspect_Model", "Clone_Seed", "Case_ID",
    "Checkpoint_Path", "Checkpoint_SHA256",
    "Regressor_Mode", "Regressor_Seed", "Regressor_Path", "Walk_Seed", "M_Seed", "Seed_Namespace",
    "N_Train", "N_Test", "Train_Range", "Test_Range", "M", "M_Reps", "Alpha",
    "P_Value_M", "Mean_Diff_M", "Stolen_M", "N_Degenerate_M",
    "Mean_Private", "Mean_Public", "Delta", "T_Stat", "P_Value_Full", "Stolen_Full",
    "Saturated_Frac", "Capped_Private_Pct", "Capped_Public_Pct", "Seconds",
]
REGRESSOR_COLUMNS = [
    "Run_Timestamp", "Suite", "Victim_Model", "Victim_Arch", "Round", "Regressor_Mode", "Head",
    "Regressor_Seed", "Regressor_Path", "Regressor_SHA256", "N_Train", "Train_Range", "Epochs",
    "Train_Loss", "Train_Mean_Private", "Train_Mean_Public",
    "Capped_Private_Pct", "Capped_Public_Pct", "Seconds",
]
IDENTITY_KEYS = ("n_train", "n_test", "alpha", "regressor_mode", "seed_namespace",
                 "regressor_epochs", "walk", "m", "m_reps", "protocol_tag",
                 "di_py_sha256", "di_paper_sha256")


# =====================================================
# 1. Index sets
# =====================================================
def split_indices(private_all, n_train, n_test, n_public):
    """Positions [0, n_train) train g_V, [n_train, n_train + n_test) test suspects."""
    end = n_train + n_test
    if end > len(private_all) or end > n_public:
        raise ValueError(f"n_train + n_test = {end} exceeds the private ({len(private_all)}) "
                         f"or public ({n_public}) pool.")
    return {
        "train_private": [int(i) for i in private_all[:n_train]],
        "train_public": list(range(n_train)),
        "test_private": [int(i) for i in private_all[n_train:end]],
        "test_public": list(range(n_train, end)),
        "train_range": f"0-{n_train - 1}",
        "test_range": f"{n_train}-{end - 1}",
    }


# =====================================================
# 2. One regressor / one detection
# =====================================================
def ensure_regressor(victim, round_idx, run):
    fixed = run["regressor_mode"] == "fixed"
    tag = "fixed" if fixed else f"round_{round_idx:03d}"
    seed = derive_seed(run["seed_namespace"], "regressor", victim["key"], tag)
    path = run["out_dir"] / "regressors" / victim["key"] / f"{tag}.pt"
    if victim["regressor_tag"] == tag:
        return seed, path
    pipeline, ds, split = victim["pipeline"], victim["ds"], victim["split"]
    if path.is_file():
        run_captured(lambda: pipeline.load_regressor(path), run["verbose"])
    else:
        set_seed(seed)
        t0 = time.time()
        _, text = run_captured(lambda: pipeline.train_regressor(
            private_dataset=ds.raw_train_clean_set, public_dataset=ds.raw_test_set,
            private_idx=split["train_private"], public_idx=split["train_public"],
            regressor_epochs=REGRESSOR_EPOCHS,
        ), run["verbose"])
        seconds = time.time() - t0
        pipeline.save_regressor(path)
        capped = parse_capped(text) + [None, None]
        epochs = _EPOCH_RE.findall(text)
        last = epochs[-1] if epochs else ("", "", "", "", "")
        append_row(run["out_dir"] / "regressors.csv", REGRESSOR_COLUMNS, {
            "Run_Timestamp": datetime.now().isoformat(timespec="seconds"),
            "Suite": run["suite"], "Victim_Model": victim["key"], "Victim_Arch": victim["arch"],
            "Round": "fixed" if fixed else round_idx, "Regressor_Mode": run["regressor_mode"],
            "Head": "tanh", "Regressor_Seed": seed, "Regressor_Path": str(path),
            "Regressor_SHA256": sha256_file(path), "N_Train": run["n_train"],
            "Train_Range": split["train_range"], "Epochs": REGRESSOR_EPOCHS,
            "Train_Loss": last[2], "Train_Mean_Private": last[3], "Train_Mean_Public": last[4],
            "Capped_Private_Pct": capped[0], "Capped_Public_Pct": capped[1],
            "Seconds": round(seconds, 2),
        })
        print(f"  [g_V tanh] {victim['key']} {tag}: trained in {seconds:.0f}s "
              f"(train mean private {last[3]} / public {last[4]})")
    victim["regressor_tag"] = tag
    return seed, path


def run_case(victim, case, round_idx, reg_seed, reg_path, run):
    ds, split = victim["ds"], victim["split"]
    walk_seed = derive_seed(run["seed_namespace"], "walk", victim["key"], case["case_id"], round_idx)
    m_seed = derive_seed(run["seed_namespace"], "mtest", victim["key"], case["case_id"], round_idx)
    set_seed(walk_seed)
    generator = torch.Generator().manual_seed(m_seed)
    t0 = time.time()
    result, text = run_captured(lambda: victim["pipeline"].verify_suspect(
        suspect_model=case["model"],
        private_dataset=ds.raw_train_clean_set, public_dataset=ds.raw_test_set,
        private_idx=split["test_private"], public_idx=split["test_public"],
        alpha=run["alpha"], m=run["m"], reps=run["m_reps"], m_generator=generator,
    ), run["verbose"])
    seconds = time.time() - t0
    stats = ("mean_private", "mean_public", "delta", "t_stat", "p_value", "mean_diff_m")
    if not all(np.isfinite(result[k]) for k in stats):
        raise ValueError(f"Non-finite DI statistics for {case['case_id']} round {round_idx}: "
                         f"{ {k: result[k] for k in stats} }")

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
        "Regressor_Path": str(reg_path), "Walk_Seed": walk_seed, "M_Seed": m_seed,
        "Seed_Namespace": run["seed_namespace"],
        "N_Train": run["n_train"], "N_Test": run["n_test"],
        "Train_Range": split["train_range"], "Test_Range": split["test_range"],
        "M": run["m"], "M_Reps": run["m_reps"], "Alpha": run["alpha"],
        "P_Value_M": f"{result['p_value_m']:.6e}",
        "Mean_Diff_M": round(result["mean_diff_m"], 6),
        "Stolen_M": int(result["stolen_m"]),
        "N_Degenerate_M": result["n_degenerate_m"],
        "Mean_Private": round(result["mean_private"], 6),
        "Mean_Public": round(result["mean_public"], 6),
        "Delta": round(result["delta"], 6),
        "T_Stat": round(result["t_stat"], 6),
        "P_Value_Full": f"{result['p_value']:.6e}",
        "Stolen_Full": int(result["stolen"]),
        "Saturated_Frac": round(result["saturated_frac"], 6),
        "Capped_Private_Pct": capped[0], "Capped_Public_Pct": capped[1],
        "Seconds": round(seconds, 2),
    }
    append_row(run["out_dir"] / "rounds.csv", ROUND_COLUMNS, row)
    return row


# =====================================================
# 3. Driver
# =====================================================
def build_run_config(suite, plan_files, n_train, n_test, alpha, regressor_mode, seed_namespace,
                     m, m_reps, split):
    probe = PaperDIPipeline(None, device=device)
    return {
        "suite": suite,
        "plans": {str(p): sha256_file(p) for p in plan_files},
        "n_train": n_train, "n_test": n_test, "alpha": alpha,
        "regressor_mode": regressor_mode, "seed_namespace": seed_namespace,
        "regressor_epochs": REGRESSOR_EPOCHS,
        "walk": {"n_samples": probe.n_samples, "noise_uniform": probe.noise_uniform,
                 "noise_gaussian": probe.noise_gaussian, "noise_laplace": probe.noise_laplace,
                 "max_steps": probe.max_steps, "point_batch_size": probe.point_batch_size},
        "m": m, "m_reps": m_reps, "protocol_tag": PROTOCOL_TAG,
        "train_range": split["train_range"], "test_range": split["test_range"],
        "di_py_sha256": sha256_file(DI_SOURCE),
        "di_paper_sha256": sha256_file(DI_PAPER_SOURCE),
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


def main_di_repeat_paper(plan_dir, rounds=ROUNDS, regressor_mode="per_round", out_dir=None,
                         n_train=None, n_test=None, seed_namespace=SEED_NAMESPACE,
                         m=M_REVEALED, m_reps=M_REPS,
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
    if not 2 <= m <= n_test:
        raise ValueError(f"m={m} must lie in [2, n_test={n_test}].")
    for v in victims:
        v["pipeline"] = PaperDIPipeline(v["model"], device=device)   # replaces the DI.py pipeline
        v["split"] = split_indices(v["private_idx"], n_train, n_test, len(v["ds"].raw_test_set))
    split0 = victims[0]["split"]

    out_dir = (Path(out_dir) if out_dir else
               OUT_ROOT / f"{suite}_{regressor_mode}_n{n_train}-{n_test}_m{m}x{m_reps}")
    check_or_write_config(out_dir, build_run_config(
        suite, plan_files, n_train, n_test, alpha, regressor_mode, seed_namespace, m, m_reps, split0))
    for name, cols in (("rounds.csv", ROUND_COLUMNS), ("regressors.csv", REGRESSOR_COLUMNS)):
        check_header(out_dir / name, cols)

    run = {"suite": suite, "out_dir": out_dir, "n_train": n_train, "n_test": n_test,
           "alpha": alpha, "regressor_mode": regressor_mode, "seed_namespace": seed_namespace,
           "m": m, "m_reps": m_reps, "verbose": verbose}
    n_cases = sum(len(v["cases"]) for v in victims)
    done, rows = read_done(out_dir)
    todo_total = sum(1 for r in range(rounds) for v in victims for c in v["cases"]
                     if (r, c["case_id"]) not in done)
    print(f"\n==> [paper protocol] Suite {suite}: {len(victims)} victim(s), {n_cases} case(s) per round, "
          f"{rounds} round(s); {todo_total} detection(s) to run, "
          f"{rounds * n_cases - todo_total} already in {out_dir / 'rounds.csv'}")
    print(f"    g_V on positions {split0['train_range']}, suspects tested on {split0['test_range']}; "
          f"m={m} x {m_reps} draws (harmonic mean); alpha={alpha}; regressor={regressor_mode}; "
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
                print(f"  r{r:02d} {case_label(case['case_id'])} p_m={float(row['P_Value_M']):.2e} "
                      f"{'STOLEN' if row['Stolen_M'] else '------'} | full p={float(row['P_Value_Full']):.2e} "
                      f"delta={row['Delta']:+.4f} ({row['Seconds']:.0f}s)")
        this_round = [x for x in rows if int(x["Round"]) == r and x["Case_Type"] == "positive"]
        det_m = sum(int(x["Stolen_M"]) for x in this_round)
        det_full = sum(int(x["Stolen_Full"]) for x in this_round)
        elapsed = time.time() - t_start
        eta = elapsed / finished * (todo_total - finished) if finished else 0.0
        print(f"--- round {r}: detected {det_m}/{len(this_round)} (m-protocol), "
              f"{det_full}/{len(this_round)} (full Welch) at alpha={alpha} "
              f"| elapsed {elapsed / 60:.1f} min, ETA {eta / 60:.1f} min")

    summarize(out_dir)
    return out_dir


# =====================================================
# 4. Summaries
# =====================================================
def _rate(p, a):
    return (p < a).mean()


def summarize(out_dir):
    out_dir = Path(out_dir)
    path = out_dir / "rounds.csv"
    if not path.exists():
        print(f"No {path}; nothing to summarize.")
        return None
    df = pd.read_csv(path)
    if df.empty:
        print(f"{path} is empty; nothing to summarize.")
        return None
    df["Log10_P_M"] = np.log10(df["P_Value_M"].clip(lower=1e-300))
    df["Log10_P_Full"] = np.log10(df["P_Value_Full"].clip(lower=1e-300))

    def rate_aggs():
        out = {}
        for a in SUMMARY_ALPHAS:
            out[f"Detect_Rate_M@{a}"] = ("P_Value_M", lambda p, a=a: _rate(p, a))
        for a in SUMMARY_ALPHAS:
            out[f"Detect_Rate_Full@{a}"] = ("P_Value_Full", lambda p, a=a: _rate(p, a))
        return out

    case_keys = ["Case_Type", "Victim_Model", "Victim_Arch", "Suspect_Arch", "Suspect_Model",
                 "Clone_Seed", "Case_ID"]
    agg = {"N_Rounds": ("Round", "nunique"), **rate_aggs()}
    agg.update({
        "P_M_Median": ("P_Value_M", "median"),
        "P_M_Q25": ("P_Value_M", lambda p: p.quantile(0.25)),
        "P_M_Q75": ("P_Value_M", lambda p: p.quantile(0.75)),
        "P_M_Min": ("P_Value_M", "min"), "P_M_Max": ("P_Value_M", "max"),
        "Log10P_M_Mean": ("Log10_P_M", "mean"), "Log10P_M_Std": ("Log10_P_M", "std"),
        "Mean_Diff_M_Mean": ("Mean_Diff_M", "mean"), "Mean_Diff_M_Std": ("Mean_Diff_M", "std"),
        "P_Full_Median": ("P_Value_Full", "median"),
        "Log10P_Full_Mean": ("Log10_P_Full", "mean"), "Log10P_Full_Std": ("Log10_P_Full", "std"),
        "T_Mean": ("T_Stat", "mean"), "T_Std": ("T_Stat", "std"),
        "Delta_Mean": ("Delta", "mean"), "Delta_Std": ("Delta", "std"),
        "Degenerate_Draws": ("N_Degenerate_M", "sum"),
        "Saturated_Frac_Mean": ("Saturated_Frac", "mean"),
    })
    by_case = df.groupby(case_keys, dropna=False, sort=False).agg(**agg).reset_index()
    by_case.to_csv(out_dir / "summary_by_case.csv", index=False)

    pos = df[df["Case_Type"] == "positive"]
    pos_cases = by_case[by_case["Case_Type"] == "positive"]
    primary = f"Detect_Rate_M@{SUMMARY_ALPHAS[0]}"
    cell_keys = ["Victim_Model", "Victim_Arch", "Suspect_Arch"]
    cell_agg = {"N_Cases": ("Case_ID", "nunique"), "N_Rows": ("Round", "size"), **rate_aggs()}
    cell_agg.update({"P_M_Median": ("P_Value_M", "median"),
                     "Log10P_M_Mean": ("Log10_P_M", "mean"), "Log10P_M_Std": ("Log10_P_M", "std"),
                     "P_Full_Median": ("P_Value_Full", "median")})
    by_cell = pos.groupby(cell_keys, sort=False).agg(**cell_agg).reset_index()
    stability = pos_cases.groupby(cell_keys, sort=False)[primary].agg(
        Cases_Always=lambda r: int((r == 1).sum()),
        Cases_Never=lambda r: int((r == 0).sum()),
        Cases_Mixed=lambda r: int(((r > 0) & (r < 1)).sum()),
    ).reset_index()
    by_cell = by_cell.merge(stability, on=cell_keys, how="left")
    by_cell.to_csv(out_dir / "summary_by_cell.csv", index=False)

    round_agg = {"N_Cases": ("Case_ID", "nunique")}
    for a in SUMMARY_ALPHAS:
        round_agg[f"Detected_M@{a}"] = ("P_Value_M", lambda p, a=a: int((p < a).sum()))
    for a in SUMMARY_ALPHAS:
        round_agg[f"Detected_Full@{a}"] = ("P_Value_Full", lambda p, a=a: int((p < a).sum()))
    round_agg.update({"P_M_Median": ("P_Value_M", "median"), "P_M_Max": ("P_Value_M", "max")})
    by_round = pos.groupby("Round").agg(**round_agg).reset_index()
    by_round.to_csv(out_dir / "summary_by_round.csv", index=False)

    print(f"\n==> Summary of {out_dir} ({df['Round'].nunique()} round(s), "
          f"{len(pos_cases)} positive case(s)) -- paper protocol")
    shown = [f"Detect_Rate_M@{a}" for a in SUMMARY_ALPHAS] + [f"Detect_Rate_Full@{SUMMARY_ALPHAS[0]}"]
    for col in shown:
        mat = by_cell.pivot_table(index="Victim_Arch", columns="Suspect_Arch", values=col)
        mat = mat.reindex(index=[x for x in ARCH_ORDER if x in mat.index],
                          columns=[x for x in ARCH_ORDER if x in mat.columns])
        print(f"\n{col} (rows = victim, cols = clone; 3 seeds x rounds pooled)")
        print(mat.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nPer-case stability of the m-protocol decision at alpha={SUMMARY_ALPHAS[0]}:")
    print(by_cell[cell_keys + ["N_Cases", "Cases_Always", "Cases_Never", "Cases_Mixed"]]
          .to_string(index=False))
    if int(df["N_Degenerate_M"].sum()):
        print(f"\n[NOTE] {int(df['N_Degenerate_M'].sum())} m-draw(s) had p = 0 or nan "
              "(zero-variance samples); see Degenerate_Draws in summary_by_case.csv.")
    victims_rows = by_case[by_case["Case_Type"] == "victim"]
    if not victims_rows.empty:
        print("\nVictim self-check:")
        print(victims_rows[["Victim_Model", "N_Rounds", primary, "P_M_Median", "P_Full_Median"]]
              .to_string(index=False))
    summarize_at_grid(pos, [("M", "P_Value_M"), ("Full", "P_Value_Full")], out_dir)
    print(f"\n  -> {out_dir / 'summary_by_case.csv'}\n  -> {out_dir / 'summary_by_cell.csv'}"
          f"\n  -> {out_dir / 'summary_by_round.csv'}")
    return by_case, by_cell, by_round


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repeated DI with the paper-aligned protocol.")
    parser.add_argument("--suite", default="C100", help=f"plan folder under {PLAN_ROOT} (C10 or C100)")
    parser.add_argument("--plan-dir", help="explicit plan folder; overrides --suite")
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--regressor", choices=("per_round", "fixed"), default="per_round")
    parser.add_argument("--out-dir", help="default: saved_logs/di_repeat_paper/<suite>_<mode>_n<train>-<test>_m<m>x<reps>")
    parser.add_argument("--n-train", type=int, help="override the plans' DI.N_Train_Samples")
    parser.add_argument("--n-test", type=int, help="override the plans' DI.N_Test_Samples")
    parser.add_argument("--m", type=int, default=M_REVEALED, help="revealed samples per side (paper: 10)")
    parser.add_argument("--m-reps", type=int, default=M_REPS, help="draws aggregated by harmonic mean (paper: 100)")
    parser.add_argument("--seed-namespace", default=SEED_NAMESPACE,
                        help="default = main_DI_repeat.py's, which pairs the two runs round by round")
    parser.add_argument("--victim-check", action="store_true")
    parser.add_argument("--case-filter", help="regex on Case_ID; run only matching cases")
    parser.add_argument("--verbose", action="store_true")
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
        main_di_repeat_paper(plan_dir, rounds=cli.rounds, regressor_mode=cli.regressor,
                             out_dir=cli.out_dir, n_train=cli.n_train, n_test=cli.n_test,
                             seed_namespace=cli.seed_namespace, m=cli.m, m_reps=cli.m_reps,
                             victim_check=cli.victim_check, case_filter=cli.case_filter,
                             verbose=cli.verbose)
