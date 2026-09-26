"""Four-baseline detection (IPGuard, DeepJudge, ADV_TRA, DI) of plan-declared suspect checkpoints.

Built for the CIFAR-100 post-hoc AT models of main_at_posthoc.py, but the victim, the baseline
settings and the case sets all come from the plan (saved_exp_plan/at_posthoc_eval_plan/*.yaml).
Every baseline is evaluated with the SAME call and the SAME cached victim-side asset as its
YAML entry point, so rows are comparable with the existing CIFAR-100 results:

  IPGuard   AdvAttack.IP_Guard.verify_fingerprint on the cached fingerprint sets
            Indices/<ds>/IPGuard/victim=<arch>_<victim_id>_k=<k>_size=<n>/{TR,TL,RR,RL}.pt
            (main_IPGUARD_eval layout; never regenerated here); all four matching rates and their
            mean are recorded. The entry point reports scores only; the decision used here is
            <Decision statistic> > Tau, statistic = IPGuard.Decision (default TR, the only config
            main_IPGUARD_eval.IPGUARD_CONFIGS still evaluates), Tau = its largest value over the
            victim's overlap-0.0 negatives already in the IPGuard master (same k / size), i.e.
            zero false positives on that pool.
  DeepJudge main_DEEPJUDGE_eval: adversarial set build_adv_save_path(Victim, Attack) (cached),
            compute_robustness / compute_robd, compute_jsd(victim, suspect, build_raw_test_loader),
            cached thresholds deepjudge_threshold_path("RobD" | "JSD"); Stolen = both votes
            (P_Copy > 0.5), as main_robd_jsd.
  ADV_TRA   the fingerprint root and build_args values of main_adv_tra.main_adv_tra_pos. A missing
            fingerprint is NOT extracted during an evaluation run: `--prepare-advtra` extracts it
            once by calling main_adv_tra_pos itself with an empty suspect list (on the CIFAR-100
            victim this failed on 2026-09-23: 0 of 50 base samples completed the 9-hop, length-8
            trajectory, every attempt aborted at "step size > 1.0"; see the plan header).
            Verification: adv_tra_adapter.run_verification_pretty with the adapter's short
            default dummy path (main_adv_tra_pos writes _pos_<folder>_<ckpt>.pth, which exceeds the
            Windows path limit for the post-hoc AT folder names). Stolen = detection rate > threshold.
  DI        main_DI_eval._get_or_train_di_pipeline (clean protocol, cached regressor
            saved_models/di_regressor/victim=<arch>_<victim_id>_n=<n>_clean.pt) and
            pipeline.verify_suspect with main_di's arguments; Stolen = p < alpha.
  Utility   clean test accuracy of the checkpoint (util_adv.compute_clean_accuracy, raw test set).

main_DI_eval.set_seed(42) runs before every (checkpoint, baseline) evaluation, so each result is
independent of the order and of the other cases in the run.

Case sets (plan Case_Sets): Model_Path lists and/or From_Training_Plan (names rebuilt with
main_at_posthoc.expand_plan, the function that named the training folders); Checkpoint: <file>
or State: all (every epoch_*.pth, epoch_-1 = the source first). A training run counts only once
its last epoch checkpoint exists (main_at_posthoc's finished marker); unfinished or absent runs
are listed as missing and picked up by a later invocation.

Outputs (plan Evaluation_Output): master.csv (one row per checkpoint; per baseline Status /
Updated_At / Seconds / Error / metrics), manifest.json (settings + sha256 of assets and code; a
changed configuration refuses to reuse the directory), status.json, run.log. Resumable: finished
(checkpoint, baseline) cells are skipped, new checkpoints are appended, a checkpoint whose sha256
changed since it was evaluated stops the run.

The plan's Methods list is the default method set (the CIFAR-100 plan leaves ADV_TRA out until a
fingerprint exists); --methods overrides it.

Usage (from E:\\Experiment, pytorch_env)
  python run_four_baselines_posthoc.py --preflight
  python run_four_baselines_posthoc.py --prepare-advtra        # one-off fingerprint extraction
  python run_four_baselines_posthoc.py                        # sets: sources + best_clean
  python run_four_baselines_posthoc.py --sets trajectory      # epoch_-1 ... epoch_29 of every run
  python run_four_baselines_posthoc.py --methods IPGuard DeepJudge --select DKD
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # main_DI_eval.set_seed is deterministic
ROOT = Path(__file__).resolve().parent
DEFAULT_PLAN = ROOT / "saved_exp_plan/at_posthoc_eval_plan/CIFAR100_RES18_PostAT_4Baselines.yaml"
DEFAULT_SETS = ("sources", "best_clean")
METHODS = ("Utility", "IPGuard", "DeepJudge", "ADV_TRA", "DI")
METRICS = {
    "Utility": ["Clean_Acc"],
    "IPGuard": ["TR", "TL", "RR", "RL", "Mean", "Decision", "Tau", "Stolen"],
    "DeepJudge": ["Rob_Victim", "Rob_Suspect", "RobD", "JSD", "Tau_RobD", "Tau_JSD", "RobD_Vote", "JSD_Vote",
                  "P_Copy", "Stolen"],
    "ADV_TRA": ["Detection_Rate", "Mean_Mutation_Rate", "Num_Trajectories", "Threshold", "Stolen"],
    "DI": ["Mean_Private", "Mean_Public", "Delta", "T_Stat", "P_Value", "Alpha", "Stolen"],
}
IDENTITY = ["Case_ID", "Case_Set", "Stage", "Family", "Source", "AT_Eps", "AT_Seed", "Checkpoint", "Epoch",
            "Model_Dir", "Checkpoint_Path", "Checkpoint_SHA256", "Suspect_Arch"]
COLUMNS = IDENTITY + [f"{m}_{k}" for m in METHODS for k in ["Status", "Updated_At", "Seconds", "Error", *METRICS[m]]]
IPGUARD_TAGS = ("TR", "TL", "RR", "RL")
ADVTRA_MIN_TRAJECTORIES = 20          # main_adv_tra_pos's MIN_ACCEPTABLE_TRAJECTORIES
CODE_FILES = ["run_four_baselines_posthoc.py", "main_IPGUARD_eval.py", "AdvAttack/IP_Guard.py",
              "main_DEEPJUDGE_eval.py", "main_adv_tra.py", "AdvAttack/advtra/adv_tra_adapter.py",
              "AdvAttack/advtra/advtra_vendored/adv_gen.py", "main_DI_eval.py", "AdvAttack/DI.py", "util_adv.py"]


# =====================================================
# 1. Small helpers
# =====================================================
def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_text(path, text):
    """Atomic replace; retries while another reader (Excel, a notebook) holds the file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(120):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 119:
                raise
            time.sleep(0.5)


def write_master(path, rows):
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    write_text(path, buf.getvalue())


def read_master(path):
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def family_of(name):
    for key, label in (("_FT-AL_", "FT-AL"), ("_FT-LL_", "FT-LL"), ("_RT-AL_", "RT-AL"), ("_DKD_", "DKD"),
                       ("_KD_", "KD"), ("Knockoff", "Knockoff")):
        if key in name:
            return label
    return "other"


def parse_epoch(stem):
    if not stem.startswith("epoch_"):
        return None
    try:
        return int(stem.split("_")[-1])
    except ValueError:
        return None


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
            s.flush()
        return len(text)

    def flush(self):
        for s in self.streams:
            s.flush()


# =====================================================
# 2. Plan -> cases
# =====================================================
def load_plan(path):
    import yaml
    with Path(path).open(encoding="utf-8") as f:
        plan = yaml.safe_load(f)
    for key in ("Victim", "Baselines", "Case_Sets", "Evaluation_Output"):
        if key not in plan:
            raise KeyError(f"{Path(path).name}: missing '{key}'")
    return plan


def expand_cases(plan, sets, select=None, models_root=None):
    """-> (cases, missing). A case is one checkpoint file of one ready model directory."""
    models_root = Path(models_root or ROOT / "saved_models")
    cases, missing = [], []
    for set_name in sets:
        if set_name not in plan["Case_Sets"]:
            raise KeyError(f"Case set '{set_name}' not in plan (have {sorted(plan['Case_Sets'])})")
        spec = plan["Case_Sets"][set_name]
        if bool(spec.get("Checkpoint")) == (spec.get("State") == "all"):
            raise ValueError(f"Case set '{set_name}': give exactly one of Checkpoint: <file> or State: all")
        dirs = []                                          # (rel dir, ready, meta)
        if spec.get("From_Training_Plan"):
            import main_at_posthoc as mp
            cfg, runs = mp.expand_plan(ROOT / spec["From_Training_Plan"], list(spec.get("AT_Seeds", [0])),
                                       spec.get("Run_Tag", "v1"), eps=spec.get("Eps"))
            last = int(cfg["Optimizer"]["Epochs"]) - 1
            for model_path, _fn, _name, kwargs, at_seed, scenario in runs:
                rel = f"{mp.OUT_MODEL_ROOT.name}/{scenario}"
                dirs.append((rel, (models_root / rel / f"epoch_{last}.pth").is_file(),
                             dict(Source=model_path.rstrip("/").split("/")[-1], AT_Eps=kwargs.get("eps", ""),
                                  AT_Seed=at_seed)))
        for p in spec.get("Model_Path") or []:
            rel = p.strip("/")
            dirs.append((rel, (models_root / rel).is_dir(), dict(Source=rel.split("/")[-1], AT_Eps="", AT_Seed="")))
        for rel, ready, meta in dirs:
            if select and not any(s in rel for s in select):
                continue
            if not ready:
                missing.append((set_name, rel))
                continue
            d = models_root / rel
            if spec.get("Checkpoint"):
                files = [d / spec["Checkpoint"]]
            else:
                files = sorted(d.glob("epoch_*.pth"), key=lambda f: int(f.stem.split("_")[-1]))
            for f in files:
                if not f.is_file():
                    missing.append((set_name, f"{rel}/{f.name}"))
                    continue
                leaf = rel.split("/")[-1]
                epoch = parse_epoch(f.stem)
                cases.append(dict(
                    Case_ID=f"{set_name}:{leaf}:{f.stem}", Case_Set=set_name, Stage=spec.get("Stage", set_name),
                    Family=family_of(meta["Source"]), Source=meta["Source"], AT_Eps=meta["AT_Eps"],
                    AT_Seed=meta["AT_Seed"], Checkpoint=f.stem, Epoch="" if epoch is None else epoch,
                    Model_Dir=rel, Checkpoint_Path=f.relative_to(models_root.parent).as_posix()
                    if models_root.parent in f.parents else f.as_posix(),
                    Suspect_Arch=spec.get("Model", "ResNet-18")))
    ids = [c["Case_ID"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate Case_IDs across the selected case sets")
    return cases, missing


def merge_rows(existing, cases, root=None):
    """Existing master rows + new cases (pending). Refuses a checkpoint whose sha256 changed."""
    root = Path(root or ROOT)
    by_id = {r["Case_ID"]: r for r in existing}
    rows = list(existing)
    for case in cases:
        digest = sha256(root / case["Checkpoint_Path"])
        if case["Case_ID"] in by_id:
            old = by_id[case["Case_ID"]]
            if old["Checkpoint_SHA256"] and old["Checkpoint_SHA256"] != digest:
                raise RuntimeError(f"{case['Case_ID']}: checkpoint changed since it was evaluated "
                                   f"(sha256 {old['Checkpoint_SHA256'][:12]} -> {digest[:12]}). Use a new output.")
            continue
        row = {**dict.fromkeys(COLUMNS, ""), **case, "Checkpoint_SHA256": digest}
        row.update({f"{m}_Status": "pending" for m in METHODS})
        rows.append(row)
        by_id[case["Case_ID"]] = row
    return rows


# =====================================================
# 3. Victim-side assets (paths built with the entry points' own helpers)
# =====================================================
def victim_assets(plan):
    from main_DEEPJUDGE_eval import build_adv_save_path, build_attack_suffix, deepjudge_threshold_path
    from main_DI_eval import di_regressor_path
    from main_IPGUARD_eval import _build_victim_id
    victim = plan["Victim"]
    ds_cfg = victim["Dataset"]
    arch = victim["Model"]
    victim_id = _build_victim_id(victim, ds_cfg)
    b = plan["Baselines"]
    ip = b["IPGuard"]
    ip_dir = ROOT / f"Indices/{ds_cfg['name']}/IPGuard/victim={arch}_{victim_id}_k={ip['k']}_size={ip['Size']}"
    attack = dict(b["DeepJudge"]["Attack"])
    attack_name = attack.pop("name")
    fp_root = b["ADV_TRA"]["Fingerprint_Root"]
    return dict(
        victim_id=victim_id, arch=arch,
        victim_ckpt=ROOT / f"saved_models/vanilla/CNN_Models/{victim['Model_Name']}_{victim.get('Seed', 42)}"
                           f"_{victim.get('Overlap', 1.0)}/best_epoch.pth",
        ipguard={t: ip_dir / f"{t}.pt" for t in IPGUARD_TAGS},
        ipguard_negatives=ROOT / ip.get("Negative_Master", "saved_logs/at_eval/IPGuard/master.csv"),
        adv_examples=ROOT / build_adv_save_path(victim, attack_name, attack),
        tau_robd=ROOT / deepjudge_threshold_path("RobD", arch, victim_id, build_attack_suffix(attack_name, attack)),
        tau_jsd=ROOT / deepjudge_threshold_path("JSD", arch, victim_id, "clean"),
        advtra_root=Path("./results/advtra1") / fp_root,
        advtra_dir=ROOT / "results/advtra1" / fp_root / "fingerprints" / "cifar10" / "trajectory_8",
        di_regressor=ROOT / di_regressor_path(arch, victim_id, int(b["DI"]["N_Train_Samples"]), "clean"),
        di_indices=ROOT / f"Indices/{ds_cfg['name']}/group_A_subset_10000_from_25000_seed42.npy",
    )


def advtra_count(assets):
    d = assets["advtra_dir"]
    return sum(1 for s in d.iterdir() if s.is_dir() and (s / "tra_log.pth").exists()) if d.exists() else 0


def ipguard_decision(plan):
    """Statistic the IPGuard decision uses: one config tag (default TR) or Mean of the four."""
    stat = str(plan["Baselines"]["IPGuard"].get("Decision", "TR"))
    if stat not in (*IPGUARD_TAGS, "Mean"):
        raise ValueError(f"IPGuard.Decision must be one of {IPGUARD_TAGS} or Mean, got {stat!r}")
    return stat


def ipguard_tau(assets, plan):
    """Largest decision statistic over the victim's overlap-0.0 negatives in the IPGuard master.

    Decision TR (default; main_IPGUARD_eval.IPGUARD_CONFIGS has evaluated TR only since 2026-09-13):
    per-model TR matching rate. Decision Mean: per-model mean of the four configs (all required).
    """
    ip = plan["Baselines"]["IPGuard"]
    stat = ipguard_decision(plan)
    rows = read_master(assets["ipguard_negatives"])
    prefix = f"{assets['victim_id']}_seed="
    per_model = {}
    for r in rows:
        if (r.get("Suspect_Type") == "negative" and r.get("Victim_Arch") == assets["arch"]
                and r.get("Suspect_Arch") == assets["arch"] and r["Scenario_Name"].startswith(prefix)
                and r["Scenario_Name"].endswith("_overlap=0.0") and r.get("FP_Config") in IPGUARD_TAGS
                and float(r["k_Param"]) == float(ip["k"]) and int(float(r["Size"])) == int(ip["Size"])):
            per_model.setdefault(r["Scenario_Name"], {})[r["FP_Config"]] = float(r["Matching_Rate"])
    if stat == "Mean":
        values = [sum(v.values()) / 4 for v in per_model.values() if set(v) == set(IPGUARD_TAGS)]
    else:
        values = [v[stat] for v in per_model.values() if stat in v]
    if not values:
        return None, 0
    return max(values), len(values)


def advtra_args(assets, num_classes, device, threshold):
    """The build_args values of main_adv_tra.main_adv_tra_pos (same fingerprint layout)."""
    from AdvAttack.advtra.adv_tra_adapter import build_args
    root = assets["advtra_root"]
    return build_args(dataset_name="cifar10", num_classes=num_classes, data_path=str(root / "data"),
                      model_path=str(root / "_model_paths"), fingerprint_path=str(root / "fingerprints"),
                      num_trajectories=100, length=8, tra_classes=10, max_iteration=1000, initial_stepsize=0.05,
                      tra_lr=0.05, factor_lc=0.9, factor_re=0.95, threshold=threshold, device=device)


def extract_advtra_fingerprint(plan, output):
    """Run main_adv_tra_pos on an empty suspect list: extraction + source-vs-source sanity only."""
    import yaml
    import main_adv_tra
    fp_root = plan["Baselines"]["ADV_TRA"]["Fingerprint_Root"]
    prep = dict(Scenario_Name=fp_root, Victim=plan["Victim"],
                Positive=dict(Model=plan["Victim"]["Model"], State="best", Model_Path=[]),
                ADV_TRA=dict(Fingerprint_Root=fp_root, Log_Dir=str(Path(output) / "ADV_TRA_prepare")))
    path = Path(output) / "advtra_prepare.yaml"
    write_text(path, yaml.safe_dump(prep, sort_keys=False))
    main_adv_tra.main_adv_tra_pos(str(path))


def asset_config(plan, assets):
    files = [assets["victim_ckpt"], *assets["ipguard"].values(), assets["adv_examples"], assets["tau_robd"],
             assets["tau_jsd"], assets["di_regressor"], assets["di_indices"]]
    files += sorted(assets["advtra_dir"].glob("*/tra_log.pth")) + sorted(assets["advtra_dir"].glob("*/pred_log.pth"))
    tau, n_neg = ipguard_tau(assets, plan)
    return dict(protocol="four_baselines_posthoc_v1", victim=plan["Victim"], baselines=plan["Baselines"],
                ipguard_decision=ipguard_decision(plan), ipguard_tau=tau, ipguard_tau_n_negatives=n_neg,
                assets={Path(p).relative_to(ROOT).as_posix(): sha256(p) for p in files},
                code={f: sha256(ROOT / f) for f in CODE_FILES})


# =====================================================
# 4. Evaluators (one per baseline; built once, reused for every checkpoint)
# =====================================================
def build_evaluators(methods, plan, assets, ctx, output):
    import numpy as np
    import torch
    device = ctx["device"]
    ds, victim = ctx["dataset"], ctx["victim"]
    b = plan["Baselines"]
    evaluators = {}
    for method in methods:
        if method == "Utility":
            from torch.utils.data import DataLoader
            from util_adv import compute_clean_accuracy
            loader = DataLoader(ds.raw_test_set, batch_size=500, shuffle=False, num_workers=0)
            evaluators[method] = lambda m, _l=loader: dict(Clean_Acc=round(100.0 * compute_clean_accuracy(m, _l), 4))
        elif method == "IPGuard":
            from AdvAttack.IP_Guard import IPGuardGenerator, verify_fingerprint
            fps = {t: IPGuardGenerator.load_fingerprints(p) for t, p in assets["ipguard"].items()}
            tau, _ = ipguard_tau(assets, plan)
            stat = ipguard_decision(plan)

            def ev_ip(m, _fps=fps, _tau=tau, _stat=stat):
                out = {t: verify_fingerprint(m, fp, device=device)["matching_rate"] for t, fp in _fps.items()}
                out["Mean"] = float(np.mean([out[t] for t in IPGUARD_TAGS]))
                out["Decision"] = _stat
                out["Tau"] = "" if _tau is None else _tau
                # rates are multiples of 1/size stored as float32 (0.05 -> 0.0500000007) while Tau comes
                # from the master CSV (rounded to 4 decimals): compare rounded values, so a suspect
                # that ties the most similar negative is NOT flagged
                out["Stolen"] = "" if _tau is None else int(round(out[_stat], 6) > round(_tau, 6))
                return out
            evaluators[method] = ev_ip
        elif method == "DeepJudge":
            import main_DEEPJUDGE_eval as dj
            from util_adv import load_adv_examples
            adv = load_adv_examples(str(assets["adv_examples"]))
            tau_r = dj.load_deepjudge_threshold(assets["tau_robd"])["tau"]
            tau_j = dj.load_deepjudge_threshold(assets["tau_jsd"])["tau"]
            batch = int(b["DeepJudge"].get("Batch_Size", 256))
            raw_loader = dj.build_raw_test_loader(ds)
            rob_v = dj.compute_robustness(victim, adv, batch_size=batch)

            def ev_dj(m):
                rob_s = dj.compute_robustness(m, adv, batch_size=batch)
                robd = dj.compute_robd(rob_v, rob_s)
                jsd = dj.compute_jsd(victim, m, raw_loader)
                vr, vj = int(robd <= tau_r), int(jsd <= tau_j)
                return dict(Rob_Victim=rob_v, Rob_Suspect=rob_s, RobD=robd, JSD=jsd, Tau_RobD=tau_r, Tau_JSD=tau_j,
                            RobD_Vote=vr, JSD_Vote=vj, P_Copy=(vr + vj) / 2, Stolen=int((vr + vj) / 2 > 0.5))
            evaluators[method] = ev_dj
        elif method == "ADV_TRA":
            from AdvAttack.advtra.adv_tra_adapter import run_verification_pretty
            threshold = float(b["ADV_TRA"].get("Threshold", 0.5))
            if advtra_count(assets) < ADVTRA_MIN_TRAJECTORIES:
                raise FileNotFoundError(f"ADV_TRA: {advtra_count(assets)} trajectories under {assets['advtra_dir']} "
                                        f"(need >= {ADVTRA_MIN_TRAJECTORIES}); run --prepare-advtra first")
            args = advtra_args(assets, ctx["num_classes"], device, threshold)
            sanity = run_verification_pretty(args, victim)
            print(f"  ADV_TRA source-vs-source sanity: {sanity.detection_rate:.3f} "
                  f"({sanity.num_trajectories} trajectories)", flush=True)
            if sanity.detection_rate < 0.99:
                raise RuntimeError(f"ADV_TRA victim sanity failed: {sanity}")

            def ev_adv(m, _args=args):
                r = run_verification_pretty(_args, m)
                return dict(Detection_Rate=r.detection_rate, Mean_Mutation_Rate=r.mean_mutation_rate,
                            Num_Trajectories=r.num_trajectories, Threshold=r.threshold,
                            Stolen=int(r.detection_rate > r.threshold))
            evaluators[method] = ev_adv
        elif method == "DI":
            from main_DI_eval import _get_or_train_di_pipeline
            di = b["DI"]
            pipeline, raw_train, raw_test, indices = _get_or_train_di_pipeline(
                victim, ds, plan["Victim"]["Dataset"], assets["arch"], assets["victim_id"], int(di["N_Train_Samples"]))
            n_test, alpha = int(di["N_Test_Samples"]), float(di["Alpha"])

            def ev_di(m):
                r = pipeline.verify_suspect(m, private_dataset=raw_train, public_dataset=raw_test,
                                            private_indices=indices, n_test_samples=n_test, alpha=alpha)
                return dict(Mean_Private=r["mean_private"], Mean_Public=r["mean_public"], Delta=r["delta"],
                            T_Stat=r["t_stat"], P_Value=r["p_value"], Alpha=alpha, Stolen=int(r["stolen"]))
            evaluators[method] = ev_di
    return evaluators


# =====================================================
# 5. Entry point
# =====================================================
def preflight(plan, cases, missing, methods, assets):
    import torch
    from util import build_dataset_from_yaml
    from util_adv import load_positive_suspect
    print(f"Victim {assets['victim_id']} ({assets['arch']}): {assets['victim_ckpt'].relative_to(ROOT)} "
          f"{'OK' if assets['victim_ckpt'].is_file() else 'MISSING'}")
    status = {
        "IPGuard": all(p.is_file() for p in assets["ipguard"].values()),
        "DeepJudge": all(p.is_file() for p in (assets["adv_examples"], assets["tau_robd"], assets["tau_jsd"])),
        "ADV_TRA": advtra_count(assets) >= ADVTRA_MIN_TRAJECTORIES,
        "DI": assets["di_regressor"].is_file() and assets["di_indices"].is_file(),
        "Utility": True,
    }
    tau, n_neg = ipguard_tau(assets, plan)
    notes = {"IPGuard": f"fingerprints cached; decision {ipguard_decision(plan)} > Tau = {tau} "
                        f"(max over {n_neg} overlap-0.0 negatives)",
             "DeepJudge": "adversarial set + RobD/JSD thresholds cached",
             "ADV_TRA": f"{advtra_count(assets)} trajectories cached" if status["ADV_TRA"]
             else f"{advtra_count(assets)} trajectories -> cells fail until --prepare-advtra succeeds",
             "DI": "clean regressor cached" if status["DI"] else "regressor missing -> trained on first run",
             "Utility": "clean test accuracy"}
    for m in methods:
        hard_fail = m in ("IPGuard", "DeepJudge") and not status[m]
        print(f"  {m:<9s} {'OK ' if status[m] else ('MISSING' if hard_fail else 'LATER')}  {notes[m]}")
        if hard_fail:
            raise FileNotFoundError(f"{m}: cached victim asset missing (never regenerated here)")
    if "IPGuard" in methods and tau is None:
        print("  [WARN] no IPGuard negatives found: IPGuard_Stolen stays empty (scores only)")
    by_set = {}
    for c in cases:
        by_set[c["Case_Set"]] = by_set.get(c["Case_Set"], 0) + 1
    print(f"Ready checkpoints: {by_set or 'none'}")
    for set_name, rel in missing:
        print(f"  [missing] {set_name}: {rel}")
    ds, ncls, _ = build_dataset_from_yaml(plan["Victim"]["Dataset"])
    x = torch.stack([ds.raw_test_set[i][0] for i in range(4)]).cuda()
    for c in cases:
        net = load_positive_suspect(c["Suspect_Arch"], ncls, ds, ROOT / c["Checkpoint_Path"])
        with torch.no_grad():
            out = net(x)
        if out.shape != (4, ncls) or not torch.isfinite(out).all():
            raise RuntimeError(f"{c['Case_ID']}: bad logits {tuple(out.shape)}")
        del net
    print(f"Load check passed for {len(cases)} checkpoint(s). Preflight done; nothing written.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    p.add_argument("--sets", nargs="+", default=list(DEFAULT_SETS), help="case sets of the plan")
    p.add_argument("--methods", nargs="+", choices=list(METHODS), help="default: the plan's Methods list")
    p.add_argument("--select", nargs="+", help="keep model directories containing any of these substrings")
    p.add_argument("--output", type=Path, help="override the plan's Evaluation_Output")
    p.add_argument("--preflight", action="store_true", help="check assets and load every ready checkpoint")
    p.add_argument("--prepare-advtra", action="store_true",
                   help="extract the victim's ADV_TRA fingerprint through main_adv_tra_pos, then stop")
    args = p.parse_args(argv)
    os.chdir(ROOT)
    plan = load_plan(args.plan)
    output = (args.output or ROOT / plan["Evaluation_Output"]).resolve()
    wanted = args.methods or plan.get("Methods") or list(METHODS)
    unknown = set(wanted) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown methods {sorted(unknown)}")
    methods = [m for m in METHODS if m in wanted]
    assets = victim_assets(plan)
    if args.prepare_advtra:
        from main_DI_eval import set_seed
        set_seed(42)
        output.mkdir(parents=True, exist_ok=True)
        extract_advtra_fingerprint(plan, output)
        print(f"ADV_TRA trajectories now cached: {advtra_count(assets)} under {assets['advtra_dir']}")
        return
    cases, missing = expand_cases(plan, args.sets, args.select)
    print(f"Plan {args.plan.name}: sets {args.sets}, methods {methods}; output {output}")
    if args.preflight:
        preflight(plan, cases, missing, methods, assets)
        return

    import torch
    from main_DI_eval import set_seed
    from util import build_dataset_from_yaml
    from util_adv import load_positive_suspect, load_victim_model
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=True)
    log = (output / "run.log").open("a", encoding="utf-8", buffering=1)
    sys.stdout, sys.stderr = Tee(sys.__stdout__, log), Tee(sys.__stderr__, log)
    print(f"\n===== {datetime.now().isoformat(timespec='seconds')} =====")
    for set_name, rel in missing:
        print(f"[missing, skipped] {set_name}: {rel}")
    if not cases:
        print("No ready checkpoints for the selected sets.")
        return

    set_seed(42)
    ds, ncls, _ = build_dataset_from_yaml(plan["Victim"]["Dataset"])
    victim = load_victim_model(plan["Victim"], ds, ncls, plan["Victim"].get("Seed", 42))
    ctx = dict(device="cuda", dataset=ds, num_classes=ncls, victim=victim)

    master = output / "master.csv"
    rows = merge_rows(read_master(master), cases)
    selected = {c["Case_ID"] for c in cases}
    pending = {m: [r for r in rows if r["Case_ID"] in selected and r[f"{m}_Status"] != "complete"] for m in methods}
    need = [m for m in methods if pending[m]]
    evaluators, prep_errors = {}, {}
    for m in need:                                   # a failing baseline must not stop the others
        try:
            set_seed(42)
            evaluators.update(build_evaluators([m], plan, assets, ctx, output))
        except Exception:
            prep_errors[m] = traceback.format_exc()
            print(prep_errors[m], flush=True)

    config = asset_config(plan, assets)
    config_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    manifest = output / "manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old["config_id"] != config_id:
            raise RuntimeError(f"{output} was produced with different settings/assets/code "
                               f"({old['config_id']} vs {config_id}); choose a new --output.")
    else:
        write_text(manifest, json.dumps(dict(config_id=config_id, config=config), indent=2))
    write_master(master, rows)

    def status(state, **extra):
        done = sum(r[f"{m}_Status"] == "complete" for r in rows if r["Case_ID"] in selected for m in methods)
        write_text(output / "status.json", json.dumps(dict(
            state=state, pid=os.getpid(), updated_at=datetime.now().isoformat(timespec="seconds"),
            config_id=config_id, completed_cells=done, total_cells=len(selected) * len(methods), **extra), indent=2))

    for m, err in prep_errors.items():
        for r in pending[m]:
            r[f"{m}_Status"], r[f"{m}_Error"] = "failed", err
    write_master(master, rows)
    todo = [r for r in rows if r["Case_ID"] in selected and any(r[f"{m}_Status"] != "complete" for m in evaluators)]
    print(f"Config {config_id}: {len(selected)} checkpoint(s), {len(todo)} with pending cells; master {master}")
    for i, row in enumerate(todo, 1):
        model = load_positive_suspect(row["Suspect_Arch"], ncls, ds, ROOT / row["Checkpoint_Path"])
        print(f"\n[{i}/{len(todo)}] {row['Case_ID']}", flush=True)
        for m, evaluate in evaluators.items():
            if row[f"{m}_Status"] == "complete":
                continue
            start = time.monotonic()
            row[f"{m}_Status"] = "running"
            status("running", case=row["Case_ID"], method=m)
            try:
                set_seed(42)
                result = evaluate(model)
                row.update({f"{m}_{k}": v for k, v in result.items()})
                row[f"{m}_Status"], row[f"{m}_Error"] = "complete", ""
                print(f"  {m}: {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items()})}",
                      flush=True)
            except Exception:
                row[f"{m}_Status"], row[f"{m}_Error"] = "failed", traceback.format_exc()
                print(row[f"{m}_Error"], flush=True)
            finally:
                row[f"{m}_Seconds"] = round(time.monotonic() - start, 2)
                row[f"{m}_Updated_At"] = datetime.now().isoformat(timespec="seconds")
                write_master(master, rows)
        del model
        torch.cuda.empty_cache()
    failed = sum(r[f"{m}_Status"] != "complete" for r in rows if r["Case_ID"] in selected for m in methods)
    status("completed_with_errors" if failed else "complete")
    print(f"\nFinished: {len(selected) * len(methods) - failed}/{len(selected) * len(methods)} cells complete. {master}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    import torch.multiprocessing as tmp
    tmp.set_start_method("spawn", force=True)
    main()
