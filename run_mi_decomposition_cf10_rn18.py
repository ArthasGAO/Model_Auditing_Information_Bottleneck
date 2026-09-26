"""Seed-0, paired equation (4)/(5) sweep on the existing owner MI probe grid.

Run in the project PyTorch environment. Configuration lives below. Existing
checkpoints, indices and master tables are read-only. --preflight loads every
checkpoint strictly; --plot-only redraws already computed results.
"""
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from MI_check import collect_logits, mi_from_logits, set_seed, POOL0_BINS, POOL0_IN_SIZE_RATES
from mi_pool_support import sizes_from_rates
from mi_information_decomposition import decomposition_from_states, decomposition_from_logits
from Model.ResNet_18 import ResNet18
from Model.kd_eval import build_kd_student
from util import build_dataset_from_yaml, process_yaml_file

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/mi_decomposition_cf10_rn18_seed0"
SIZES = sizes_from_rates(25000, POOL0_IN_SIZE_RATES)
BINS = list(POOL0_BINS)
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ATTACKS = ["FT-AL", "PR-80%", "DKD", "Knockoff"]
TERMS = ["G_X", "L_X", "G_Y", "L_Y"]
SPECS = {
    "Victim": {
        "folder": "vanilla/CNN_Models/CIFAR-10_ResNet-18_25000_42_1.0",
        "plan": "train_plan/old_plan/CIFAR10_RES18_SGD_SMALL.yaml",
        "table": "vanilla/MI_master_table_victim.csv", "seed": 42,
    },
    "FT-AL": {
        "folder": "ft_final/CIFAR-10_ResNet-18_25000_Same_25000_42_1.0_FT-AL_ftsize=25000_ftseed=0",
        "plan": "ft_plan_c10/CIFAR10_RES18_FT_Same_25000.yaml",
        "table": "ft_final/MI_master_table_ft.csv", "seed": 0,
    },
    "PR-80%": {
        "folder": "pruning_final/CIFAR-10_ResNet-18_25000_Same_25000_42_1.0_sparsity=0.8_FT-AL_ftsize=25000_ftseed=0",
        "plan": "prune_plan_c10/CIFAR10_RES18_PRUNE_Same_25000.yaml",
        "table": "pruning_final/MI_master_table_prune.csv", "seed": 0,
    },
    "DKD": {
        "folder": "kd_final/CIFAR-10_ResNet-18to18_25000_DKD_0_0.0",
        "plan": "kd_plan_matrix/matrix/CIFAR10_RES18to18_KD.yaml",
        "table": "kd_final/MI_master_table_kd.csv", "seed": 0,
    },
    "Knockoff": {
        "folder": "extraction_final/CIFAR-10_ResNet-18_25000_Knockoff_Same10_Same18_0_1.0",
        "plan": "knockoff_3x3_c10_same10/CIFAR10_RES18_KNOCKOFF_Same10_Same18.yaml",
        "table": "extraction_final/MI_master_table_extraction.csv", "seed": 0,
    },
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def checkpoint(spec):
    return ROOT / "saved_models" / spec["folder"] / "best_epoch.pth"


def load_model(name):
    spec = SPECS[name]
    net = build_kd_student("ResNet-18", 10) if name == "DKD" else ResNet18(num_classes=10)
    state = torch.load(checkpoint(spec), map_location="cpu", weights_only=True)
    if "model" in state:
        state = state["model"]
    elif "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state, strict=True)
    net.eval()
    if name == "PR-80%":
        from calculate_MI_prune import measured_sparsity
        zero_pct = measured_sparsity(net, "cnn")
        if abs(zero_pct - 80) > 0.01:
            raise ValueError(f"Pruning sparsity mismatch: {zero_pct}")
        print(f"[PRUNE] measured sparsity={zero_pct:.6f}%", flush=True)
    return net.to(DEVICE)


def preflight():
    torch.set_num_threads(4)
    set_seed(42)
    indices_path = ROOT / "Indices/CIFAR-10/nested_subsets_seed42.npz"
    group_path = ROOT / "Indices/CIFAR-10/group_A_25000_seed42.npy"
    with np.load(indices_path, allow_pickle=False) as archive:
        subsets = {n: archive[f"size_{n}"].copy() for n in SIZES}
    ids = subsets[max(SIZES)]
    group = np.load(group_path, allow_pickle=False)
    assert len(np.unique(ids)) == 25000 and np.array_equal(np.sort(ids), np.sort(group))
    for n in SIZES:
        assert len(subsets[n]) == n and len(np.unique(subsets[n])) == n
        assert np.isin(subsets[n], ids).all()
    with contextlib.redirect_stdout(io.StringIO()):
        configs = {name: process_yaml_file(ROOT / "saved_exp_plan" / s["plan"]) for name, s in SPECS.items()}
    ds_cfg = configs["Victim"]["Dataset"].copy()
    for name, plan in configs.items():
        other = plan["Victim"]["Dataset"] if name == "Knockoff" else plan["Dataset"]
        for key in ("name", "normalization", "img_size", "test_transforms", "group_size"):
            assert other[key] == ds_cfg[key], (name, key)
    assert (ROOT / configs["DKD"]["Teacher_Model"]["teacher_ckpt"]).resolve() == checkpoint(SPECS["Victim"]).resolve()
    assert configs["FT-AL"]["Model_Name"] == configs["Victim"]["Scenario_Name"]
    assert configs["PR-80%"]["Model_Name"] == configs["Victim"]["Scenario_Name"]
    assert configs["Knockoff"]["Victim"]["Model_Name"] == configs["Victim"]["Scenario_Name"]
    ds_cfg["download"] = False
    ds_cfg["root_dir"] = str(ROOT / "data")
    dataset, classes, _ = build_dataset_from_yaml(ds_cfg)
    assert classes == 10
    labels = np.asarray(dataset.in_sample_set.targets)[ids]
    for n, sub in subsets.items():
        counts = np.bincount(np.asarray(dataset.in_sample_set.targets)[sub], minlength=10)
        assert counts.max() - counts.min() <= 1
    identity = {
        "scenario": "CIFAR-10_ResNet-18_25000", "split": "owner_train_clean",
        "sizes": SIZES, "bins": BINS, "batch_size": BATCH_SIZE,
        "subset_seed": 42, "attack_seed": 0, "victim_seed": 42,
        "indices_sha256": sha(indices_path), "group_A_sha256": sha(group_path),
        "preprocessing": repr(dataset.in_sample_set.transform),
        "data_sha256": {p.name: sha(p) for p in sorted((ROOT / "data/cifar-10-batches-py").glob("data_batch_*"))},
        "code_sha256": {str(p): sha(ROOT / p) for p in [
            "MI_check.py", "mi_information_decomposition.py", "Model/ResNet_18.py",
            "Model/ResNet_18_dist.py", "Model/kd_eval.py", "Dataset/CIFAR_10.py", "util.py",
        ]},
        "models": {},
    }
    for name, spec in SPECS.items():
        path = checkpoint(spec)
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        net = load_model(name)
        probe = torch.stack([dataset.in_sample_set[int(i)][0] for i in ids[:4]]).to(DEVICE)
        with torch.no_grad():
            logits = net(probe)
            if isinstance(logits, tuple):
                logits = logits[0]
        assert logits.shape == (4, 10) and torch.isfinite(logits).all()
        identity["models"][name] = {
            **spec, "checkpoint": str(path), "checkpoint_sha256": sha(path),
            "plan_sha256": sha(ROOT / "saved_exp_plan" / spec["plan"]),
            "parameter_count": sum(p.numel() for p in net.parameters()),
        }
        print(f"[PREFLIGHT] {name}: strict checkpoint load and native logits OK", flush=True)
        del net
    # Exercise the real torch/MI_check integration, including exact same-model terms.
    onehot = torch.nn.functional.one_hot(torch.as_tensor(labels[:4], device=DEVICE), 10).float()
    with contextlib.redirect_stdout(io.StringIO()):
        test = decomposition_from_logits(logits, logits, onehot, onehot,
            victim_sample_ids=ids[:4], suspect_sample_ids=ids[:4], num_intervals=50)
    assert max(abs(test[k]) for k in TERMS) < 1e-10
    print(f"[GRID] sizes={SIZES}; bins={BINS}; 4 x {len(SIZES)*len(BINS)} = {4*len(SIZES)*len(BINS)} paired cells", flush=True)
    return identity, dataset, ids, subsets, labels


def cached_logits(name, identity, dataset, ids, labels):
    suffix = "" if len(ids) == max(SIZES) else f"_size{len(ids)}"
    cache = OUT / "logits" / f"{name.replace('%', 'pct')}{suffix}.npz"
    expected = identity["models"][name]["checkpoint_sha256"]
    if cache.exists():
        with np.load(cache, allow_pickle=False) as a:
            assert str(a["checkpoint_sha256"]) == expected
            assert np.array_equal(a["sample_ids"], ids) and np.array_equal(a["labels"], labels)
            logits = torch.from_numpy(a["logits"].copy())
        print(f"[CACHE] {name}, N={len(ids)}", flush=True)
    else:
        net = load_model(name)
        loader = DataLoader(Subset(dataset.in_sample_set, ids.tolist()), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0, pin_memory=(DEVICE == "cuda"))
        print(f"[INFER] {name}: {len(ids)} owner inputs", flush=True)
        logits, y = collect_logits(net, loader, DEVICE)
        assert np.array_equal(y.argmax(1).cpu().numpy(), labels)
        assert sha(checkpoint(SPECS[name])) == expected, "Checkpoint changed during inference"
        logits = logits.cpu()
        assert torch.isfinite(logits).all()
        temporary = cache.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, logits=logits.numpy(), labels=labels, sample_ids=ids,
                            checkpoint_sha256=expected)
        temporary.replace(cache)
        del net
        torch.cuda.empty_cache()
    accuracy = float((logits.argmax(1).numpy() == labels).mean())
    print(f"[ACCURACY] {name}: {accuracy:.6f} (owner probe)", flush=True)
    return logits, accuracy


def measure(logits, labels, n, b):
    with contextlib.redirect_stdout(io.StringIO()):
        x, y, debug = mi_from_logits(logits, labels, num_intervals=b, verbose=True)
    return {"I_X": x, "I_Y": y, "states": debug["inverse_idx"]}


def compare_old_tables(marginals, identity):
    report = {}
    for name, spec in SPECS.items():
        path = ROOT / "saved_logs" / spec["table"]
        target_name = Path(spec["folder"]).name + ("_ckpt=best" if name == "PR-80%" else "")
        with path.open(newline="", encoding="utf-8-sig") as stream:
            matches = [r for r in csv.DictReader(stream) if r["model_name"] == target_name]
        grid = {}
        for row in matches:
            key = (int(row["in_size"]), int(row["bins"]))
            if key in grid:
                raise ValueError(f"Duplicate reference row: {name}, {key}")
            if row.get("checkpoint_sha256"):
                assert row["checkpoint_sha256"] == identity["models"][name]["checkpoint_sha256"]
            grid[key] = row
        diffs = []
        absent = []
        for n in SIZES:
            for b in BINS:
                if (n, b) not in grid:
                    absent.append([n, b])
                    continue
                old = grid[n, b]
                current = marginals[name][n, b]
                diffs.append({"size": n, "bins": b,
                    "abs_I_X_error": abs(current["I_X"] - float(old["I(X;T)-In"])),
                    "abs_I_Y_error": abs(current["I_Y"] - float(old["I(T;Y)-In"]))})
        report[name] = {"table": str(path), "table_sha256": sha(path), "matched_cells": len(diffs),
                        "absent_cells": absent,
                        "max_abs_error": max((max(r["abs_I_X_error"], r["abs_I_Y_error"]) for r in diffs), default=None),
                        "cells": diffs}
    return report


def plot_results(records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    cube = np.empty((4, 4, len(SIZES), len(BINS)))
    keyed = {(r["attack"], r["N"], r["bins"]): r for r in records}
    for ai, attack in enumerate(ATTACKS):
        for ti, term in enumerate(TERMS):
            cube[ai, ti] = [[keyed[attack, n, b][term] for b in BINS] for n in SIZES]
    max_x, max_y = cube[:, :2].max(), cube[:, 2:].max()
    norms = [Normalize(0, max_x)] * 2 + [Normalize(0, max_y)] * 2
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(4, 4, figsize=(20.8, 12.6))
    fig.subplots_adjust(left=.14, right=.985, top=.87, bottom=.18, hspace=.23, wspace=.13)
    titles = [r"$I(X;T_s\mid T_v)$", r"$I(X;T_v\mid T_s)$",
              r"$I(T_s;Y\mid T_v)$", r"$I(T_v;Y\mid T_s)$"]
    subtitles = ["Input gain", "Input loss", "Label gain", "Label loss"]
    for ai, attack in enumerate(ATTACKS):
        for ti, term in enumerate(TERMS):
            ax = axes[ai, ti]
            values = cube[ai, ti]
            ax.imshow(values, cmap=cmap, norm=norms[ti], aspect="auto", origin="lower")
            ax.set_xticks(range(len(BINS)), [str(b) for b in BINS])
            ax.set_yticks(range(len(SIZES)), [f"{n:,}" for n in SIZES])
            ax.tick_params(length=0, labelsize=9, labelleft=ti == 0, labelbottom=ai == 3)
            if ai == 0:
                ax.set_title(titles[ti] + "\n" + subtitles[ti], fontsize=14, pad=12)
            if ai == 3:
                ax.set_xlabel("Bins per probability dimension", labelpad=9)
            if ti == 0:
                ax.set_ylabel("MI sample size", labelpad=7)
                pos = ax.get_position()
                fig.text(.013, (pos.y0+pos.y1)/2, attack, ha="left", va="center", fontsize=15, weight="bold")
            for yi in range(len(SIZES)):
                for xi in range(len(BINS)):
                    value = values[yi, xi]
                    color = "#102328" if norms[ti](value) > .59 else "white"
                    label = f"{max(value, 0):.3f}" if ti < 2 else f"{max(value, 0):.4f}"
                    ax.text(xi, yi, label, ha="center", va="center", color=color, fontsize=7.0)
            ax.add_patch(Rectangle((BINS.index(50)-.5, SIZES.index(25000)-.5), 1, 1,
                                   fill=False, edgecolor="#ff953f", linewidth=2.3))
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle("CIFAR-10 / ResNet-18: paired information gain and loss", x=.53, y=.986, fontsize=21, weight="bold")
    fig.text(.53, .95, "Attack seed 0 · common victim (seed 42) · clean owner training inputs · best checkpoints", ha="center", fontsize=12, color="#475569")
    centers = [(axes[0, i].get_position().x0 + axes[0, i+1].get_position().x1)/2 for i in [0, 2]]
    fig.text(centers[0], .91, "Equation (4)", ha="center", fontsize=13, weight="bold")
    fig.text(centers[1], .91, "Equation (5)", ha="center", fontsize=13, weight="bold")
    for start, width, norm, label in [(.15, .385, norms[0], "Input information (bits): shared scale for gain and loss"),
                                    (.59, .375, norms[2], "Label information (bits): shared scale for gain and loss")]:
        cax = fig.add_axes([start, .102, width, .014])
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
        cb.set_label(label, labelpad=4)
        cb.ax.tick_params(labelsize=9)
    fig.text(.53, .013, "Each cell is one measured (sample size, bins) setting. Orange outline: N = 25,000, bins = 50.\nPR-80% includes recovery fine-tuning; T denotes the quantized full softmax vector. Display values are rounded.",
             ha="center", va="bottom", fontsize=10, color="#475569")
    fig.savefig(OUT / "equations_4_5_grid.png", dpi=200, facecolor="white")
    fig.savefig(OUT / "equations_4_5_grid.pdf", facecolor="white")
    plt.close(fig)


def main():
    os.chdir(ROOT)
    if "--plot-only" in sys.argv:
        plot_results(json.loads((OUT / "results.json").read_text(encoding="utf-8"))["records"])
        return
    identity, dataset, ids, subsets, labels = preflight()
    if "--preflight" in sys.argv:
        print(json.dumps(identity["models"], indent=2))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logits").mkdir(exist_ok=True)
    manifest = OUT / "manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text(encoding="utf-8"))["identity"] == identity, "Input identity changed; use a new output directory"
    else:
        save_json(manifest, {"identity": identity, "created_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": {"python": sys.version, "torch": torch.__version__, "device": DEVICE,
                        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None},
            "runner_sha256": sha(__file__)})
        np.savez_compressed(OUT / "probe_indices.npz", **{f"size_{n}": a for n, a in subsets.items()})
    cache = {}
    accuracies = {}
    for name in SPECS:
        cache[name], accuracies[name] = cached_logits(name, identity, dataset, ids, labels)
        save_json(OUT / "status.json", {"stage": "inference", "complete_models": list(cache)})
    locations = {int(sample_id): i for i, sample_id in enumerate(ids)}
    marginals = {name: {} for name in SPECS}
    records = []
    for n in SIZES:
        positions = np.array([locations[int(i)] for i in subsets[n]])
        y = labels[positions]
        onehot = torch.nn.functional.one_hot(torch.as_tensor(y, device=DEVICE), 10).float()
        for name in SPECS:
            # Match legacy inference batches for EVERY size. Slicing the full-N
            # cache can move boundary-adjacent outputs across quantization bins
            # because CUDA arithmetic can depend on the final batch shape.
            if n == max(SIZES):
                logits = cache[name].to(DEVICE)
            else:
                logits, _ = cached_logits(name, identity, dataset, subsets[n], y)
                logits = logits.to(DEVICE)
            for b in BINS:
                marginals[name][n, b] = measure(logits, onehot, n, b)
        for attack in ATTACKS:
            for b in BINS:
                v, s = marginals["Victim"][n, b], marginals[attack][n, b]
                r = decomposition_from_states(v["states"], s["states"], y,
                    victim_sample_ids=subsets[n], suspect_sample_ids=subsets[n])
                residuals = {"I_X_V": r["I_X_V"]-v["I_X"], "I_X_S": r["I_X_S"]-s["I_X"],
                             "I_V_Y": r["I_V_Y"]-v["I_Y"], "I_S_Y": r["I_S_Y"]-s["I_Y"]}
                assert max(abs(d) for d in residuals.values()) < 1e-5
                records.append({"attack": attack, "attack_seed": 0, "bins": b, **r,
                                "MI_check_marginal_residuals": residuals})
        print(f"[MEASURED] size={n}: {len(records)}/280 paired cells", flush=True)
        save_json(OUT / "status.json", {"stage": "measurement", "complete_cells": len(records)})
    assert len(records) == len(ATTACKS)*len(SIZES)*len(BINS)
    comparisons = compare_old_tables(marginals, identity)
    save_json(OUT / "reference_mi_comparison.json", comparisons)
    for name, reference in comparisons.items():
        assert not reference["absent_cells"], f"Missing reference cells: {name}"
        assert reference["max_abs_error"] < 1e-5, f"Reference MI mismatch: {name}"
    summary = {}
    for attack in ATTACKS:
        rows = [r for r in records if r["attack"] == attack]
        summary[attack] = {"grid_cells": len(rows),
            "input_gain_gt_loss": sum(r["delta_X"] > 1e-8 for r in rows),
            "label_loss_gt_gain": sum(r["delta_Y"] < -1e-8 for r in rows),
            "both_strict_directions": sum(r["delta_X"] > 1e-8 and r["delta_Y"] < -1e-8 for r in rows),
            "default_cell": next(r for r in rows if r["N"] == 25000 and r["bins"] == 50)}
    payload = {"units": "bits", "representation": "quantized full softmax vector",
               "inference_policy": "Independent forward pass per size using cached legacy sample order and batch size 128",
               "runner_sha256": sha(__file__),
               "sizes": SIZES, "bins": BINS, "owner_probe_accuracy": accuracies,
               "summary": summary, "records": records}
    save_json(OUT / "results.json", payload)
    plot_results(records)
    save_json(OUT / "status.json", {"stage": "complete", "complete_cells": len(records),
        "max_eq4_residual": max(abs(r["eq4_residual"]) for r in records),
        "max_eq5_residual": max(abs(r["eq5_residual"]) for r in records),
        "max_MI_check_marginal_residual": max(abs(d) for r in records for d in r["MI_check_marginal_residuals"].values()),
        "reference_comparison": {k: {f: v[f] for f in ["matched_cells", "max_abs_error", "absent_cells"]} for k, v in comparisons.items()}})
    print(json.dumps({k: {a: b for a, b in v.items() if a != "default_cell"} for k, v in summary.items()}, indent=2), flush=True)
    print(f"[DONE] {OUT}", flush=True)


if __name__ == "__main__":
    main()
