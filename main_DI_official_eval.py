"""
Dataset Inference (Maini et al., ICLR 2021) evaluated through the OFFICIAL code.

Everything after the model / data objects is upstream code, vendored under
AdvAttack/di_official/di_vendored (see SOURCE.md there for hashes and the
exact list of modifications). Pipeline:

  1. features : official `get_random_label_only` -> `rand_steps` (Blind Walk)
                on `num_images` private images (victim's group_A subset,
                raw_train_clean_set, no augmentation) and `num_images` public
                images (raw_test_set), for the victim ("teacher") and every
                suspect declared in the YAML. Cached in upstream's layout:
                saved_logs/di_official/files/victim=<arch>_<id>/<DATASET>/model_<name>_normalized/
  2. decision : official notebook protocol. Victim features -> standardize ->
                .T.reshape(num_images, 30) -> rows [:split_index] train the
                2-layer Tanh regressor (SGD 0.1, 1000 full-batch epochs) ->
                rows [split_index:] of every model are scored ->
                  P_Value_Welch : one Welch t-test, public > private (cells 16/18)
                  P_Value_M     : selected_m rows per side, inner_rep random
                                  draws, harmonic mean (generate_table; the
                                  protocol behind the paper's tables)
  3. master   : saved_logs/di_official/master.csv, one row per (model, setting).
                Stolen_* = P_Value_* < Alpha is THIS script's reading of the
                p-value; upstream prints p-values and draws 0.05 / 0.01 lines.

YAML plans: saved_exp_plan/di_official_plan/*.yaml, same Victim / Positive /
"Negative Suspect" schema as main_DI_eval.py (explicit Model_Path lists).

Usage (PowerShell, from E:/Experiment):
    python main_DI_official_eval.py                          # every plan in --plan-dir
    python main_DI_official_eval.py --yaml saved_exp_plan/di_official_plan/CIFAR10_RES18_DIofficial_NegRes.yaml
    python main_DI_official_eval.py --yaml ... --all-cached  # also score every cached model of this victim
    python main_DI_official_eval.py --yaml ... --protocol-only   # no extraction: re-score all cached models

Reminder on what DI tests: H0 is "the suspect was not trained on the victim's
private data". A negative that shares group_A (overlap 1.0) is expected to be
flagged by DI; only overlap-0.0 negatives correspond to upstream's
"independent" model.
"""
import argparse
import glob
import hashlib
import os
import re
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from main_DI_eval import (
    _setup_victim_context, _detect_suspect_block, _iter_suspects,
    append_master_row, set_seed,
)
from AdvAttack.di_official.di_official_adapter import (
    build_args, official_dataset_name, make_loader, extract_and_save,
    feature_dir, features_exist, load_meta, list_cached_names,
    run_notebook_protocol, sha256_file,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

MASTER_CSV_PATH = Path("./saved_logs/di_official/master.csv")
FEATURES_ROOT = Path("./saved_logs/di_official/files")

MASTER_COLUMNS = [
    "Run_Timestamp", "Scenario_Name",
    "Victim_Arch", "Suspect_Arch", "Suspect_Type", "Checkpoint",
    "Feature_Name", "Num_Images", "Batch_Size", "Split_Index",
    "Feature_Seed", "Protocol_Seed", "Selected_M", "Inner_Rep", "Regressor_Epochs", "Alpha",
    "Mean_Diff_Welch", "P_Value_Welch", "Stolen_Welch",
    "Mean_Diff_M", "P_Value_M", "Stolen_M",
    "Walk_Steps", "Attacks_SHA256",
    "Checkpoint_Path", "Checkpoint_SHA256", "Features_Dir",
]
DEDUP_KEY = (
    "Scenario_Name", "Victim_Arch", "Suspect_Arch", "Suspect_Type", "Checkpoint",
    "Num_Images", "Split_Index", "Feature_Seed", "Protocol_Seed",
    "Selected_M", "Inner_Rep", "Regressor_Epochs", "Alpha",
    "Walk_Steps", "Attacks_SHA256",
)

_VENDORED_ATTACKS = Path(__file__).parent / "AdvAttack" / "di_official" / "di_vendored" / "attacks.py"


def vendored_walk_settings():
    """(steps, sha256[:12]) of the vendored rand_steps.

    The walk budget is a literal inside the upstream function, so the only
    way to tell a 50-step table from a 200-step one is the file itself: the
    default `steps = N` literal is parsed for readability and the file hash
    is recorded as the authoritative identifier. Both enter the dedup key so
    rows produced under different walk settings never overwrite each other.
    """
    text = _VENDORED_ATTACKS.read_text(encoding="utf-8")
    m = re.search(r"uni, std, scale = \([^)]*\); steps = (\d+)", text)
    steps = int(m.group(1)) if m else -1
    return steps, sha256_file(_VENDORED_ATTACKS)[:12]


def feature_name_for(suspect_arch, scenario_name):
    """Filesystem-safe, unique name for a suspect's feature directory.

    Long scenario names are truncated and suffixed with a short hash so the
    full path stays inside the Windows 260-character limit (the upstream
    layout adds `model_` + `_normalized/train_rand_vulnerability.pt` around
    the name); meta.json keeps the untruncated scenario name.
    """
    base = f"{suspect_arch}__{scenario_name}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
    safe = "".join(c if (c.isalnum() or c in "-_=.") else "_" for c in base)
    return f"{safe[:48]}__{digest}"


def main_di_official(
    yaml_path, *,
    num_images=1000, batch_size=500, split_index=500,
    feature_seed=0, protocol_seed=0, selected_m=10, inner_rep=100,
    regressor_epochs=1000, alpha=0.05,
    force_features=False, protocol_only=False, all_cached=False, deterministic=True,
    master_path=MASTER_CSV_PATH, features_root=FEATURES_ROOT,
    suspect_seeds=None, overlap_rates=None,
):
    if not torch.cuda.is_available():
        raise RuntimeError("upstream attacks.py builds CUDA tensors at import; a GPU is required.")
    if not (0 < split_index < num_images):
        raise ValueError("split_index must lie strictly between 0 and num_images.")
    master_path = Path(master_path)
    features_root = Path(features_root)

    # Deterministic-algorithms flag for the GPU walks; the adapter reseeds
    # before every model's extraction, this call only sets the mode.
    set_seed(feature_seed, deterministic=deterministic)

    ctx = _setup_victim_context(yaml_path, build_raw_loader=False, load_victim=not protocol_only)
    exp_yaml      = ctx["exp_yaml"]
    victim_ds_cfg = ctx["victim_ds_cfg"]
    victim_ds_obj = ctx["victim_dataset_obj"]
    num_classes   = ctx["victim_num_classes"]
    victim_model  = ctx["victim_model"]
    victim_arch   = ctx["victim_arch"]
    victim_id     = ctx["victim_id"]

    ds_name = victim_ds_cfg["name"]
    ds_off  = official_dataset_name(ds_name)
    root    = features_root / f"victim={victim_arch}_{victim_id}"
    # Blind Walk noise is drawn once per model under `feature_seed` (upstream
    # `--seed`, default 0). A repeat of the whole extraction under another
    # seed must not silently reuse the seed-0 cache, so non-zero seeds get
    # their own sub-root (seed 0 keeps the original path).
    if feature_seed != 0:
        root = root / f"fseed={feature_seed}"
    args    = build_args(ds_off, batch_size=batch_size)

    # ---- the two image sets every model is probed on (upstream: first
    #      num_images of an unshuffled train loader / test loader) ----
    private_all = np.load(f"./Indices/{ds_name}/group_A_subset_10000_from_25000_seed42.npy")
    if len(private_all) < num_images:
        raise ValueError("Not enough private indices for num_images.")
    private_idx = [int(i) for i in private_all[:num_images]]
    public_idx  = list(range(num_images))
    private_loader = make_loader(victim_ds_obj.raw_train_clean_set, private_idx, batch_size)
    public_loader  = make_loader(victim_ds_obj.raw_test_set, public_idx, batch_size)

    common_meta = {
        "victim_arch": victim_arch, "victim_id": victim_id, "dataset_framework": ds_name,
        "private_split": "raw_train_clean_set (victim group_A subset, ToTensor only)",
        "public_split": "raw_test_set (ToTensor only)",
        "private_indices": private_idx, "public_indices": public_idx,
    }
    walk_steps, attacks_sha = vendored_walk_settings()
    print(f"  Features root: {root}")
    print(f"  num_images={num_images} batch_size={batch_size} split_index={split_index} "
          f"feature_seed={feature_seed} protocol_seed={protocol_seed} "
          f"selected_m={selected_m} inner_rep={inner_rep} alpha={alpha}")
    print(f"  vendored rand_steps: steps={walk_steps} attacks.py sha256={attacks_sha}"
          + ("" if walk_steps == 50 else "   [NOTE: not the upstream default of 50]"))

    # ---- victim = "teacher" ---------------------------------------------
    rows_meta = {}
    if protocol_only:
        if not features_exist(root, ds_off, "teacher"):
            raise FileNotFoundError(f"--protocol-only but no cached victim features under {root}")
    else:
        extract_and_save(
            root=root, dataset_name=ds_off, name="teacher", model=victim_model,
            private_loader=private_loader, public_loader=public_loader, args=args, device=device,
            num_images=num_images, seed=feature_seed, force=force_features,
            meta_extra={**common_meta, "scenario_name": victim_id, "suspect_arch": victim_arch,
                        "suspect_type": "victim", "checkpoint": "best",
                        "checkpoint_path": "", "checkpoint_sha256": ""},
        )
    rows_meta["teacher"] = dict(scenario_name=victim_id, suspect_arch=victim_arch,
                                suspect_type="victim", checkpoint="best",
                                checkpoint_path="", checkpoint_sha256="")

    # ---- suspects from the YAML ------------------------------------------
    if not protocol_only:
        s_type, suspect_cfg = _detect_suspect_block(exp_yaml)
        print(f"  Suspect block: {s_type}")
        for rec in _iter_suspects(s_type, suspect_cfg, victim_ds_obj, num_classes,
                                  victim_arch, victim_id, suspect_seeds, overlap_rates):
            name = feature_name_for(rec["suspect_arch"], rec["scenario_name"])
            ckpt_path = str(Path(rec["ckpt_path"]).resolve()) if rec.get("ckpt_path") else ""
            ckpt_sha  = sha256_file(rec["ckpt_path"]) if rec.get("ckpt_path") else ""
            meta_extra = {**common_meta,
                          "scenario_name": rec["scenario_name"], "suspect_arch": rec["suspect_arch"],
                          "suspect_type": rec["type"], "checkpoint": rec["eval_mode"],
                          "checkpoint_path": ckpt_path, "checkpoint_sha256": ckpt_sha}
            extract_and_save(
                root=root, dataset_name=ds_off, name=name, model=rec["model"],
                private_loader=private_loader, public_loader=public_loader, args=args, device=device,
                num_images=num_images, seed=feature_seed, force=force_features, meta_extra=meta_extra,
            )
            rows_meta[name] = dict(scenario_name=rec["scenario_name"], suspect_arch=rec["suspect_arch"],
                                   suspect_type=rec["type"], checkpoint=rec["eval_mode"],
                                   checkpoint_path=ckpt_path, checkpoint_sha256=ckpt_sha)
            del rec["model"]
            torch.cuda.empty_cache()

    # ---- optionally every cached model of this victim ---------------------
    if all_cached or protocol_only:
        for name in list_cached_names(root, ds_off):
            if name in rows_meta:
                continue
            m = load_meta(root, ds_off, name)
            if m.get("num_images", num_images) != num_images:
                print(f"  [SKIP cached] {name}: extracted with num_images={m.get('num_images')}")
                continue
            rows_meta[name] = dict(scenario_name=m.get("scenario_name", name),
                                   suspect_arch=m.get("suspect_arch", ""),
                                   suspect_type=m.get("suspect_type", ""),
                                   checkpoint=m.get("checkpoint", ""),
                                   checkpoint_path=m.get("checkpoint_path", ""),
                                   checkpoint_sha256=m.get("checkpoint_sha256", ""))

    names = ["teacher"] + [n for n in rows_meta if n != "teacher"]
    print(f"\n==> Notebook protocol over {len(names)} model(s) (victim + {len(names) - 1} suspect(s))")
    results = run_notebook_protocol(
        root=root, dataset_name=ds_off, names=names, split_index=split_index, seed=protocol_seed,
        selected_m=selected_m, total_inner_rep=inner_rep, num_images=num_images,
        regressor_epochs=regressor_epochs,
    )

    # ---- master table -------------------------------------------------------
    stamp = datetime.now().isoformat(timespec="seconds")
    print(f"\n{'Scenario':60s} {'type':9s} {'mean_diff':>9s} {'p_welch':>10s} {'mean_m':>8s} {'p_m':>10s}")
    for name in names:
        r = results[name]
        meta = rows_meta[name]
        row = {
            "Run_Timestamp": stamp,
            "Scenario_Name": meta["scenario_name"],
            "Victim_Arch": victim_arch,
            "Suspect_Arch": meta["suspect_arch"],
            "Suspect_Type": meta["suspect_type"],
            "Checkpoint": meta["checkpoint"],
            "Feature_Name": name,
            "Num_Images": num_images, "Batch_Size": batch_size, "Split_Index": split_index,
            "Feature_Seed": feature_seed, "Protocol_Seed": protocol_seed,
            "Selected_M": selected_m, "Inner_Rep": inner_rep,
            "Regressor_Epochs": regressor_epochs, "Alpha": alpha,
            "Mean_Diff_Welch": round(r["mean_diff_welch"], 6),
            "P_Value_Welch": f"{r['p_value_welch']:.6e}",
            "Stolen_Welch": int(r["p_value_welch"] < alpha),
            "Mean_Diff_M": round(r["mean_diff_m"], 6),
            "P_Value_M": f"{r['p_value_m']:.6e}",
            "Stolen_M": int(r["p_value_m"] < alpha),
            "Walk_Steps": walk_steps,
            "Attacks_SHA256": attacks_sha,
            "Checkpoint_Path": meta["checkpoint_path"],
            "Checkpoint_SHA256": meta["checkpoint_sha256"],
            "Features_Dir": str(feature_dir(root, ds_off, name)),
        }
        append_master_row(master_path, row, MASTER_COLUMNS, DEDUP_KEY)
        print(f"{meta['scenario_name'][:60]:60s} {meta['suspect_type']:9s} "
              f"{r['mean_diff_welch']:9.4f} {r['p_value_welch']:10.3e} "
              f"{r['mean_diff_m']:8.4f} {r['p_value_m']:10.3e}")
    print(f"\n  Results -> {master_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Inference with the official code path.")
    parser.add_argument("--yaml", action="append", help="Plan to run; repeat for several. Default: every *.yaml in --plan-dir.")
    parser.add_argument("--plan-dir", default="./saved_exp_plan/di_official_plan")
    parser.add_argument("--num-images", type=int, default=1000, help="upstream num_images (private and public each)")
    parser.add_argument("--batch-size", type=int, default=500, help="upstream README value; must divide --num-images")
    parser.add_argument("--split-index", type=int, default=500, help="notebook split_index (rows used to train the regressor)")
    parser.add_argument("--feature-seed", type=int, default=0, help="upstream --seed for feature generation")
    parser.add_argument("--protocol-seed", type=int, default=0, help="notebook seed (cell 3)")
    parser.add_argument("--selected-m", type=int, default=10, help="generate_table selected_m")
    parser.add_argument("--inner-rep", type=int, default=100, help="generate_table total_inner_rep")
    parser.add_argument("--regressor-epochs", type=int, default=1000, help="notebook cell 12 epochs")
    parser.add_argument("--alpha", type=float, default=0.05, help="threshold applied to the p-values for Stolen_*")
    parser.add_argument("--force-features", action="store_true", help="re-extract even if cached")
    parser.add_argument("--protocol-only", action="store_true", help="skip extraction; score every cached model of the victim")
    parser.add_argument("--all-cached", action="store_true", help="after extraction, also score every cached model of the victim")
    parser.add_argument("--nondeterministic", action="store_true", help="do not enable torch deterministic algorithms")
    parser.add_argument("--master", default=str(MASTER_CSV_PATH))
    parser.add_argument("--features-root", default=str(FEATURES_ROOT))
    cli = parser.parse_args()

    yaml_files = cli.yaml or sorted(glob.glob(os.path.join(cli.plan_dir, "*.yaml")))
    if not yaml_files:
        print(f"No YAML files found in {cli.plan_dir}")
    else:
        print(f"Found {len(yaml_files)} experiment plan(s):")
        for f in yaml_files:
            print(" -", f)

    for yaml_path in yaml_files:
        print(f"\n{'=' * 60}\n  {yaml_path}\n{'=' * 60}")
        main_di_official(
            yaml_path,
            num_images=cli.num_images, batch_size=cli.batch_size, split_index=cli.split_index,
            feature_seed=cli.feature_seed, protocol_seed=cli.protocol_seed,
            selected_m=cli.selected_m, inner_rep=cli.inner_rep,
            regressor_epochs=cli.regressor_epochs, alpha=cli.alpha,
            force_features=cli.force_features, protocol_only=cli.protocol_only,
            all_cached=cli.all_cached, deterministic=not cli.nondeterministic,
            master_path=Path(cli.master), features_root=Path(cli.features_root),
        )
