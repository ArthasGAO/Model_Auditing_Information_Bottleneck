"""All existing seeds for CF10 RN18 cases; mean of per-model MI.

Existing seed-0 outputs and master tables are read-only. Selection is frozen
by exact directory names and checkpoint hashes. Each MI size is inferred
separately, using the original sample order and batch size. --plot-only uses
the already saved mean results. --expanded adds FT-LL, RT-AL, PR-20%, KD,
and HL in a separate output directory. All user-facing configuration is below.
"""
import contextlib
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys

import run_mi_decomposition_cf10_rn18 as base
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from mi_information_decomposition import decomposition_from_states

ROOT = base.ROOT
EXPANDED_CASES = "--expanded" in sys.argv
OUT = ROOT / ("outputs/mi_decomposition_cf10_rn18_nine_cases_all_seeds" if EXPANDED_CASES
              else "outputs/mi_decomposition_cf10_rn18_all_seeds")
PREVIOUS = ROOT / "outputs/mi_decomposition_cf10_rn18_seed0"
PREVIOUS_ALL = ROOT / "outputs/mi_decomposition_cf10_rn18_all_seeds"
SIZES, BINS = list(base.SIZES), list(base.BINS)
ATTACKS, TERMS = list(base.ATTACKS), list(base.TERMS)
SPECS = {k: dict(v) for k, v in base.SPECS.items()}
if EXPANDED_CASES:
    ATTACKS = ["FT-LL", "FT-AL", "RT-AL", "PR-20%", "PR-80%", "KD", "DKD", "Knockoff", "HL"]
    for attack in ["FT-LL", "RT-AL"]:
        SPECS[attack] = {**SPECS["FT-AL"],
            "folder": SPECS["FT-AL"]["folder"].replace("_FT-AL_", f"_{attack}_")}
    SPECS["PR-20%"] = {**SPECS["PR-80%"],
        "folder": SPECS["PR-80%"]["folder"].replace("sparsity=0.8", "sparsity=0.2")}
    SPECS["KD"] = {**SPECS["DKD"], "folder": SPECS["DKD"]["folder"].replace("_DKD_", "_KD_")}
    SPECS["HL"] = {
        "folder": "extraction_final/CIFAR-10_ResNet-18_25000_DFMS_Cross100-40C_Same18_0_1.0",
        "plan": "dfms_plan/matrix/CIFAR10_RES18_DFMS_C100-40C_Same18.yaml",
        "table": "extraction_final/MI_master_table_extraction.csv", "seed": 0,
    }
BATCH_SIZE = base.BATCH_SIZE
DEVICE = base.DEVICE
TOL = 1e-8
MEAN_FIELDS = [
    "G_X", "L_X", "G_Y", "L_Y", "delta_X", "delta_Y",
    "I_X_V", "I_X_S", "I_V_Y", "I_S_Y", "I_VS_Y",
    "H_Y", "H_VS", "H_Y_given_V", "H_Y_given_S", "H_Y_given_VS",
]
sha, save_json = base.sha, base.save_json


def snapshot_tree(path):
    return {p.relative_to(path).as_posix(): sha(p) for p in sorted(path.rglob("*")) if p.is_file()}


def discover_models():
    entries = []
    for attack in ATTACKS:
        spec = SPECS[attack]
        parent = ROOT / "saved_models" / Path(spec["folder"]).parent
        template = Path(spec["folder"]).name
        if "ftseed=0" in template:
            regex = re.escape(template).replace("ftseed=0", r"ftseed=(\d+)")
        elif attack in {"KD", "DKD"}:
            regex = re.escape(template).replace(f"_{attack}_0_", f"_{attack}_" + r"(\d+)_")
        else:
            regex = re.escape(template).replace("_Same18_0_", r"_Same18_(\d+)_")
        found = []
        for directory in parent.iterdir():
            match = re.fullmatch(regex, directory.name)
            if not directory.is_dir() or not match:
                continue
            seed = int(match[1])
            cp = directory / "best_epoch.pth"
            if not cp.is_file() or cp.stat().st_size == 0:
                raise FileNotFoundError(f"Existing case has no usable best checkpoint: {cp}")
            found.append({**spec, "attack": attack, "seed": seed,
                "key": f"{attack.replace('%', 'pct')}_seed{seed}",
                "folder": directory.relative_to(ROOT / "saved_models").as_posix(),
                "model_name": directory.name, "checkpoint": str(cp),
                "checkpoint_sha256": sha(cp),
                "plan_sha256": sha(ROOT / "saved_exp_plan" / spec["plan"])})
        if not found:
            raise ValueError(f"No checkpoint exists for {attack}")
        found.sort(key=lambda x: x["seed"])
        assert len({r["seed"] for r in found}) == len(found)
        print(f"[INVENTORY] {attack}: seeds={[r['seed'] for r in found]}", flush=True)
        entries.extend(found)
    return entries


def load_model(entry):
    attack = entry["attack"]
    net = base.build_kd_student("ResNet-18", 10) if attack in {"KD", "DKD"} else base.ResNet18(num_classes=10)
    state = torch.load(entry["checkpoint"], map_location="cpu", weights_only=True)
    if "model" in state:
        state = state["model"]
    elif "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state, strict=True)
    net.eval()
    if attack.startswith("PR-"):
        from calculate_MI_prune import measured_sparsity
        with contextlib.redirect_stdout(io.StringIO()):
            sparsity = measured_sparsity(net, "cnn")
        assert abs(sparsity - float(attack[3:-1])) < .01, (entry["key"], sparsity)
    return net.to(DEVICE)


def collect(entry, dataset, sample_ids, labels):
    cache_path = OUT / "logits" / f"{entry['key']}_size{len(sample_ids)}.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as a:
            assert str(a["checkpoint_sha256"]) == entry["checkpoint_sha256"]
            assert np.array_equal(a["sample_ids"], sample_ids)
            assert np.array_equal(a["labels"], labels)
            logits = torch.from_numpy(a["logits"].copy())
    else:
        net = load_model(entry)
        loader = DataLoader(Subset(dataset.in_sample_set, sample_ids.tolist()),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(DEVICE == "cuda"))
        with contextlib.redirect_stdout(io.StringIO()):
            logits, y = base.collect_logits(net, loader, DEVICE)
        assert np.array_equal(y.argmax(1).cpu().numpy(), labels)
        assert sha(entry["checkpoint"]) == entry["checkpoint_sha256"], "Checkpoint changed during inference"
        logits = logits.cpu()
        assert torch.isfinite(logits).all()
        temporary = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, logits=logits.numpy(), labels=labels, sample_ids=sample_ids,
                            checkpoint_sha256=entry["checkpoint_sha256"])
        temporary.replace(cache_path)
        del net
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return logits.to(DEVICE), float((logits.argmax(1).numpy() == labels).mean())


def reference_check(entry, measured):
    path = ROOT / "saved_logs" / entry["table"]
    target = entry["model_name"] + ("_ckpt=best" if entry["attack"].startswith("PR-") else "")
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["model_name"] == target]
    reference = {}
    for row in rows:
        k = (int(row["in_size"]), int(row["bins"]))
        assert k not in reference, f"Duplicate source row: {entry['key']}, {k}"
        if row.get("checkpoint_sha256"):
            assert row["checkpoint_sha256"] == entry["checkpoint_sha256"], f"Source checkpoint changed: {entry['key']}"
        reference[k] = row
    differences, missing = [], []
    for n in SIZES:
        for b in BINS:
            if (n, b) not in reference:
                missing.append([n, b])
                continue
            old, new = reference[n, b], measured[n, b]
            differences.append({"N": n, "bins": b,
                "abs_I_X_error": abs(new["I_X"] - float(old["I(X;T)-In"])),
                "abs_I_Y_error": abs(new["I_Y"] - float(old["I(T;Y)-In"]))})
    maximum = max((max(d["abs_I_X_error"], d["abs_I_Y_error"]) for d in differences), default=0.0)
    return {"table": str(path), "table_sha256": sha(path), "model_name": target,
            "matched_cells": len(differences), "missing_cells": missing,
            "max_abs_error": maximum, "cells": differences}


def aggregate(records, seeds_by_attack):
    index = {(r["attack"], r["seed"], r["N"], r["bins"]): r for r in records}
    expected = sum(map(len, seeds_by_attack.values())) * len(SIZES) * len(BINS)
    assert len(records) == len(index) == expected
    means = []
    for attack in ATTACKS:
        seeds = seeds_by_attack[attack]
        for n in SIZES:
            for b in BINS:
                group = [index[attack, seed, n, b] for seed in seeds]
                mean = {k: statistics.fmean(r[k] for r in group) for k in MEAN_FIELDS}
                for k in MEAN_FIELDS:
                    assert abs(mean[k] - float(np.mean([r[k] for r in group], dtype=np.float64))) < 1e-12
                row = {"attack": attack, "N": n, "bins": b, "n_seeds": len(seeds), "seeds": seeds,
                    **mean,
                    "std_across_seeds": {k: statistics.stdev(r[k] for r in group) if len(group) > 1 else None for k in MEAN_FIELDS},
                    "min_across_seeds": {k: min(r[k] for r in group) for k in MEAN_FIELDS},
                    "max_across_seeds": {k: max(r[k] for r in group) for k in MEAN_FIELDS},
                    "per_seed_both_strict": sum(r["delta_X"] > TOL and r["delta_Y"] < -TOL for r in group)}
                row["eq4_residual"] = mean["delta_X"] - (mean["G_X"] - mean["L_X"])
                row["eq5_residual"] = mean["delta_Y"] - (mean["G_Y"] - mean["L_Y"])
                assert max(abs(row["eq4_residual"]), abs(row["eq5_residual"])) < 1e-12
                means.append(row)
    return means


def summarize(means, records, seeds_by_attack):
    summary = {}
    for attack in ATTACKS:
        rows = [r for r in means if r["attack"] == attack]
        single = [r for r in records if r["attack"] == attack]
        def directions(r):
            return {"cells": len(r), "input_positive": sum(x["delta_X"] > TOL for x in r),
                    "label_negative": sum(x["delta_Y"] < -TOL for x in r),
                    "both_strict": sum(x["delta_X"] > TOL and x["delta_Y"] < -TOL for x in r),
                    "input_reverse": sum(x["delta_X"] < -TOL for x in r),
                    "label_reverse": sum(x["delta_Y"] > TOL for x in r)}
        summary[attack] = {"seeds": seeds_by_attack[attack], "n_seeds": len(seeds_by_attack[attack]),
            "mean_directions": directions(rows), "per_seed_directions": directions(single),
            "default_cell": next(r for r in rows if r["N"] == 25000 and r["bins"] == 50)}
    return summary


def plot_results(payload):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    rows = {(r["attack"], r["N"], r["bins"]): r for r in payload["records"]}
    cube = np.array([[[[rows[a, n, b][t] for b in BINS] for n in SIZES] for t in TERMS] for a in ATTACKS])
    norms = [Normalize(0, cube[:, :2].max())] * 2 + [Normalize(0, cube[:, 2:].max())] * 2
    cmap = plt.get_cmap("viridis")
    row_count = len(ATTACKS)
    height = 12.6 + 2.3 * (row_count - 4)
    fig, axes = plt.subplots(row_count, 4, figsize=(20.8, height))
    fig.subplots_adjust(left=.14, right=.985, top=1-1.65/height, bottom=2.27/height, hspace=.23, wspace=.13)
    titles = [r"$I(X;T_s\mid T_v)$", r"$I(X;T_v\mid T_s)$", r"$I(T_s;Y\mid T_v)$", r"$I(T_v;Y\mid T_s)$"]
    subtitles = ["Mean input gain", "Mean input loss", "Mean label gain", "Mean label loss"]
    for ai, attack in enumerate(ATTACKS):
        for ti, term in enumerate(TERMS):
            ax = axes[ai, ti]
            ax.imshow(cube[ai, ti], cmap=cmap, norm=norms[ti], aspect="auto", origin="lower")
            ax.set_xticks(range(len(BINS)), [str(b) for b in BINS])
            ax.set_yticks(range(len(SIZES)), [f"{n:,}" for n in SIZES])
            ax.tick_params(length=0, labelsize=9, labelleft=ti == 0, labelbottom=ai == row_count - 1)
            if ai == 0:
                ax.set_title(titles[ti] + "\n" + subtitles[ti], fontsize=14, pad=12)
            if ai == row_count - 1:
                ax.set_xlabel("Bins per probability dimension", labelpad=9)
            if ti == 0:
                ax.set_ylabel("MI sample size", labelpad=7)
                pos = ax.get_position()
                cy = (pos.y0 + pos.y1) / 2
                fig.text(.013, cy, attack, ha="left", va="center", fontsize=15, weight="bold")
            for yi in range(len(SIZES)):
                for xi in range(len(BINS)):
                    value = cube[ai, ti, yi, xi]
                    label = f"{max(value, 0):.3f}" if ti < 2 else f"{max(value, 0):.4f}"
                    color = "#102328" if norms[ti](value) > .59 else "white"
                    ax.text(xi, yi, label, ha="center", va="center", color=color, fontsize=7.0)
            ax.add_patch(Rectangle((BINS.index(50)-.5, SIZES.index(25000)-.5), 1, 1,
                                  fill=False, edgecolor="#ff953f", linewidth=2.3))
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle("CIFAR-10 / ResNet-18: mean paired information gain and loss", x=.53, y=1-.18/height, fontsize=21, weight="bold")
    fig.text(.53, 1-.63/height, "Equal-weight mean of per-model information on common owner inputs", ha="center", fontsize=12, color="#475569")
    for first, title in [(0, "Equation (4)"), (2, "Equation (5)")]:
        cx = (axes[0, first].get_position().x0 + axes[0, first+1].get_position().x1) / 2
        fig.text(cx, 1-1.13/height, title, ha="center", fontsize=13, weight="bold")
    for start, width, norm, label in [(.15, .385, norms[0], "Input information (bits): shared scale for gain and loss"),
                                    (.59, .375, norms[2], "Label information (bits): shared scale for gain and loss")]:
        cax = fig.add_axes([start, 1.285/height, width, .1764/height])
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
        cb.set_label(label, labelpad=4)
        cb.ax.tick_params(labelsize=9)
    fig.text(.53, .16/height, "Each cell is the arithmetic mean of per-model conditional information. Orange outline: N = 25,000, bins = 50.\nClean owner inputs; pruning includes recovery fine-tuning. T denotes the quantized full softmax vector.",
             ha="center", va="bottom", fontsize=10, color="#475569")
    fig.savefig(OUT / "equations_4_5_mean_grid.png", dpi=200, facecolor="white")
    fig.savefig(OUT / "equations_4_5_mean_grid.pdf", facecolor="white")
    plt.close(fig)


def write_readme(payload, validation):
    lines = ["# CF10 RN18：全部现有 seed 的条件信息均值", "",
        "先对每个真实 checkpoint 分别计算四项，再在每个 attack 的每个 (MI size, bins) 下等权取算术均值。没有平均 logits、没有混合模型状态、没有使用合成 CSV seed。", "",
        "## 模型数量", "", "| Attack | Seeds | 模型数 |", "|---|---|---:|"]
    for attack, summary in payload["summary"].items():
        lines.append(f"| {attack} | {', '.join(map(str, summary['seeds']))} | {summary['n_seeds']} |")
    lines += ["", "共同 victim 为 seed 42，保持上一轮同一 checkpoint。剪枝 case 使用 FT-AL 恢复训练后的 best checkpoint。仅一个实际 seed 的 case，其标准差标为 null（不可估计）。HL 对应 DFMS-HL 的 CIFAR-100 40 类代理数据设置。图中不显示 seed 数量。", "",
        f"MI sizes：{SIZES}；bins：{BINS}。", "",
        f"逐 seed 共 {validation['per_seed_cells']} 组结果；均值共 {validation['mean_cells']} 组结果。所有 size 使用原缓存的有序 owner training 子集，batch size=128，逐 size 独立前向。", "",
        "## N=25,000、bins=50 的均值（bits）", "",
        "| Attack | 输入增益 G_X | 输入损失 L_X | 标签增益 G_Y | 标签损失 L_Y |", "|---|---:|---:|---:|---:|"]
    for attack, s in payload["summary"].items():
        r = s["default_cell"]
        lines.append(f"| {attack} | " + " | ".join(f"{max(r[k], 0):.9f}" for k in TERMS) + " |")
    lines += ["", "## 均值方向性", "", "| Attack | 输入均值净增 > 0 | 标签均值净损失 > 0 | 两者均严格成立 |", "|---|---:|---:|---:|"]
    for attack, s in payload["summary"].items():
        d = s["mean_directions"]
        lines.append(f"| {attack} | {d['input_positive']}/70 | {d['label_negative']}/70 | {d['both_strict']}/70 |")
    lines += ["", "零容差为 1e-8 bits。均值方向不等于每个 seed 都具有同一方向；逐 seed 数值和方向统计保存在结果中。70 个参数格不是 70 次独立训练重复。", "",
        "## 输出与核验", "", f"- `equations_4_5_mean_grid.png` / `.pdf`：{len(ATTACKS)} 行 × 4 列均值图。",
        f"- `mean_results.json`：{validation['mean_cells']} 个均值格子，含均值、逐 seed 标准差（ddof=1）、min/max 和 seed 数。",
        "- `per_seed_results.json`：全部逐模型结果、状态占用、边际核对残差。",
        "- `manifest.json`：固定 checkpoint 路径/哈希、完整 seed 清单、数据/配置/代码哈希。",
        "- `logits/`、`probe_indices.npz`：本目录独立保存的逐 size 输出和输入清单。",
        "- `reference_mi_comparison.json`、`validation.json`：原 MI 表对照、seed 0 回归和聚合检查。", "",
        f"原始目录文件哈希核对：{validation['previous_directory_unchanged']}；本次没有改写之前的 seed-0 结果。",
        f"原 MI 表匹配 {validation['reference_matched_cells']} 组，缺失 {validation['reference_missing_cells']} 组；最大边际差为 {validation['reference_max_abs_error']:.3g} bits。",
        f"seed-0 与上一轮四项的最大差为 {validation['seed0_max_abs_error']:.3g} bits。",
        f"均值恒等式最大残差为 {validation['mean_max_equation_residual']:.3g} bits。", "",
        "本结果描述固定 owner 经验分布上的分箱 MI。共同 victim 的 H(Y|T_v)=0 使标签增益为零；该零值不应单独解释为模型窃取的特异性证据。", ""]
    if EXPANDED_CASES:
        lines += [f"此前四类 all-seed 输出目录文件哈希核对：{validation['previous_all_seed_directory_unchanged']}。",
                  f"此前四类的全部逐 seed 四项最大差：{validation['previous_all_seed_max_abs_error']:.3g} bits。", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    os.chdir(ROOT)
    if "--plot-only" in sys.argv:
        plot_results(json.loads((OUT / "mean_results.json").read_text(encoding="utf-8")))
        return
    entries = discover_models()
    seeds_by_attack = {a: [e["seed"] for e in entries if e["attack"] == a] for a in ATTACKS}
    previous_snapshot = snapshot_tree(PREVIOUS)
    previous_all_snapshot = snapshot_tree(PREVIOUS_ALL) if EXPANDED_CASES else None
    with contextlib.redirect_stdout(io.StringIO()):
        seed0_identity, dataset, _, subsets, _ = base.preflight()
    with contextlib.redirect_stdout(io.StringIO()):
        configs = {attack: base.process_yaml_file(ROOT / "saved_exp_plan" / spec["plan"])
                   for attack, spec in SPECS.items()}
    ds_cfg = configs["Victim"]["Dataset"]
    for attack in ATTACKS:
        cfg = configs[attack]
        other = cfg["Victim"]["Dataset"] if attack in {"Knockoff", "HL"} else cfg["Dataset"]
        for key in ("name", "normalization", "img_size", "test_transforms", "group_size"):
            assert other[key] == ds_cfg[key], (attack, key)
        if attack in {"KD", "DKD"}:
            assert (ROOT / cfg["Teacher_Model"]["teacher_ckpt"]).resolve() == base.checkpoint(SPECS["Victim"]).resolve()
            assert cfg["Student_Model"]["student_name"] == "ResNet-18"
        elif attack in {"Knockoff", "HL"}:
            assert cfg["Victim"]["Model_Name"] == configs["Victim"]["Scenario_Name"]
            assert cfg["Substitute"]["Model"] == "ResNet-18"
        else:
            assert cfg["Model_Name"] == configs["Victim"]["Scenario_Name"]
            assert cfg["Model"] == "ResNet-18"
    victim = {**seed0_identity["models"]["Victim"], "attack": "Victim", "key": "Victim_seed42",
              "model_name": Path(base.SPECS["Victim"]["folder"]).name}
    probe_ids = subsets[max(SIZES)][:4]
    probe = torch.stack([dataset.in_sample_set[int(i)][0] for i in probe_ids]).to(DEVICE)
    for entry in entries:
        net = load_model(entry)
        with torch.no_grad():
            logits = net(probe)
            if isinstance(logits, tuple):
                logits = logits[0]
        assert logits.shape == (4, 10) and torch.isfinite(logits).all()
        assert sha(entry["checkpoint"]) == entry["checkpoint_sha256"]
        print(f"[PREFLIGHT] {entry['key']}: strict load and native logits OK", flush=True)
        del net
    assert len({e["checkpoint_sha256"] for e in [victim] + entries}) == len(entries) + 1
    common_identity = {k: v for k, v in seed0_identity.items() if k not in {"models", "attack_seed"}}
    identity = {**common_identity, "models": [victim] + entries, "seeds_by_attack": seeds_by_attack,
                "aggregation": "arithmetic mean of per-seed information, equal weight per seed",
                "all_seed_runner_sha256": sha(__file__), "base_runner_sha256": sha(Path(base.__file__))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logits").mkdir(exist_ok=True)
    if (OUT / "manifest.json").exists():
        old_manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
        assert old_manifest["identity"] == identity, "Selection or inputs changed: use a new directory"
        assert old_manifest["previous_directory_snapshot"] == previous_snapshot
        assert old_manifest.get("previous_all_seed_directory_snapshot") == previous_all_snapshot
    else:
        save_json(OUT / "manifest.json", {"identity": identity, "previous_directory_snapshot": previous_snapshot,
            "previous_all_seed_directory_snapshot": previous_all_snapshot,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": {"python": sys.version, "torch": torch.__version__, "device": DEVICE,
                        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None}})
        np.savez_compressed(OUT / "probe_indices.npz", **{f"size_{n}": ids for n, ids in subsets.items()})
    victim_mi = {}
    comparisons, records, accuracies = {}, [], {}
    expected = len(entries) * len(SIZES) * len(BINS)
    for entry in [victim] + entries:
        measured = {}
        accuracies[entry["key"]] = {}
        for n in SIZES:
            ids = subsets[n]
            y = np.asarray(dataset.in_sample_set.targets)[ids]
            logits, accuracy = collect(entry, dataset, ids, y)
            accuracies[entry["key"]][n] = accuracy
            onehot = torch.nn.functional.one_hot(torch.as_tensor(y, device=DEVICE), 10).float()
            for b in BINS:
                s = base.measure(logits, onehot, n, b)
                measured[n, b] = {"I_X": s["I_X"], "I_Y": s["I_Y"]}
                if entry["attack"] == "Victim":
                    victim_mi[n, b] = s
                    continue
                v = victim_mi[n, b]
                r = decomposition_from_states(v["states"], s["states"], y,
                    victim_sample_ids=ids, suspect_sample_ids=ids)
                residuals = {"I_X_V": r["I_X_V"] - v["I_X"], "I_X_S": r["I_X_S"] - s["I_X"],
                             "I_V_Y": r["I_V_Y"] - v["I_Y"], "I_S_Y": r["I_S_Y"] - s["I_Y"]}
                assert max(map(abs, residuals.values())) < 1e-5
                records.append({"attack": entry["attack"], "seed": entry["seed"], "model_key": entry["key"],
                    "bins": b, **r, "MI_check_marginal_residuals": residuals})
            print(f"[MEASURED] {entry['key']}, N={n}: {len(records)}/{expected} paired cells", flush=True)
            save_json(OUT / "status.json", {"stage": "measuring", "model": entry["key"], "N": n,
                "completed_per_seed_cells": len(records), "expected_per_seed_cells": expected})
        comparisons[entry["key"]] = reference_check(entry, measured)
        save_json(OUT / "per_seed_results.json", {"units": "bits", "complete": len(records) == expected,
            "sizes": SIZES, "bins": BINS, "seeds_by_attack": seeds_by_attack,
            "owner_probe_accuracy": accuracies, "records": records})
        save_json(OUT / "reference_mi_comparison.json", comparisons)
    means = aggregate(records, seeds_by_attack)
    summary = summarize(means, records, seeds_by_attack)
    payload = {"units": "bits", "representation": "quantized full softmax vector",
        "aggregation": "arithmetic mean of per-seed information, equal weight per seed",
        "sizes": SIZES, "bins": BINS, "seeds_by_attack": seeds_by_attack,
        "summary": summary, "records": means}
    save_json(OUT / "mean_results.json", payload)
    plot_results(payload)
    old_rows = json.loads((PREVIOUS / "results.json").read_text(encoding="utf-8"))["records"]
    old_index = {(r["attack"], r["N"], r["bins"]): r for r in old_rows}
    seed0_error = max(abs(r[k] - old_index[r["attack"], r["N"], r["bins"]][k])
                      for r in records if r["seed"] == 0 and r["attack"] in base.ATTACKS for k in TERMS)
    assert seed0_error < 1e-10, "Seed 0 differs from the previous verified run"
    final_entries = discover_models()
    assert final_entries == entries, "Checkpoint inventory changed while running"
    assert sha(victim["checkpoint"]) == victim["checkpoint_sha256"]
    assert snapshot_tree(PREVIOUS) == previous_snapshot, "Previous output directory changed"
    previous_all_error = None
    if EXPANDED_CASES:
        assert snapshot_tree(PREVIOUS_ALL) == previous_all_snapshot, "Previous all-seed output directory changed"
        previous_all_rows = json.loads((PREVIOUS_ALL / "per_seed_results.json").read_text(encoding="utf-8"))["records"]
        current_index = {(r["attack"], r["seed"], r["N"], r["bins"]): r for r in records}
        previous_all_error = max(abs(r[k] - current_index[r["attack"], r["seed"], r["N"], r["bins"]][k])
                                 for r in previous_all_rows for k in TERMS)
        assert previous_all_error < 1e-10
    validation = {"previous_directory_unchanged": True, "checkpoint_inventory_unchanged": True,
        "positive_models": len(entries), "per_seed_cells": len(records), "mean_cells": len(means),
        "seed0_max_abs_error": seed0_error,
        "reference_matched_cells": sum(c["matched_cells"] for c in comparisons.values()),
        "reference_missing_cells": sum(len(c["missing_cells"]) for c in comparisons.values()),
        "reference_max_abs_error": max(c["max_abs_error"] for c in comparisons.values()),
        "max_MI_check_marginal_residual": max(abs(v) for r in records for v in r["MI_check_marginal_residuals"].values()),
        "per_seed_max_equation_residual": max(abs(r[k]) for r in records for k in ["eq4_residual", "eq5_residual"]),
        "mean_max_equation_residual": max(abs(r[k]) for r in means for k in ["eq4_residual", "eq5_residual"]),
        "means_verified_with_two_implementations": True,
        "max_abs_label_gain": max(abs(r["G_Y"]) for r in records)}
    if EXPANDED_CASES:
        validation.update(previous_all_seed_directory_unchanged=True,
                          previous_all_seed_max_abs_error=previous_all_error)
    save_json(OUT / "validation.json", validation)
    write_readme(payload, validation)
    save_json(OUT / "status.json", {"stage": "complete", **validation})
    print(json.dumps(validation, indent=2), flush=True)
    print(json.dumps({a: {k: s['default_cell'][k] for k in TERMS} for a, s in summary.items()}, indent=2), flush=True)
    print(f"[DONE] {OUT}", flush=True)


if __name__ == "__main__":
    main()
