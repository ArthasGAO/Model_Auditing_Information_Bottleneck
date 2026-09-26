"""Standalone plot of saved equation (4)/(5) means; no model or MI imports.

Run this file, or use ``main(custom_config)`` from a notebook. Edit CONFIG
below for case/grid selection and presentation. Source results are read-only.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parent
CONFIG = {
    "source": ROOT / "outputs/mi_decomposition_cf10_rn18_nine_cases_all_seeds/mean_results.json",
    "output_dir": ROOT / "outputs/mi_decomposition_cf10_rn18_selected_plot",
    "output_name": "equations_4_5_selected_rates_v7_tight",
    "attacks": ["RT-AL", "PR-80%", "DKD", "Knockoff"],
    "display_labels": {},  # Presentation-only aliases; keep the source keys above.
    "attack_label_rotation": 90,  # 左侧 attack 名称竖排；设为 0 可恢复横排。
    "attack_label_x": .024,  # 整张图的相对横坐标，调小可向左移动。
    "bins": [10, 20, 50, 100, 200],
    "sample_rates": [.05, .20, .50, .75, 1.00],
    "owner_sample_count": 25000,  # Denominator of sample rate, not test-set size.
    "figsize": (17.2, 11.2),
    "font_family": "Times New Roman",  # 全图字体，包含公式；字号单位均为 pt。
    "font_sizes": {
        "cell": 15,       # 每个 cell 内的数值
        "label": 20,        # 外侧轴标题：Bins、MI sample rate
        "tick": 18,         # 横纵轴刻度：10/20/...、5%/20%/...
        "column": 25,       # 顶部四个 metric 公式
        "attack": 25,       # 左侧 RT-AL、PR-80%、DKD、Knockoff
        "colorbar_tick": 18,  # 底部 color bar 的数值刻度
    },
    "cmap": "viridis",
    "decimals": {"G_X": 2, "L_X": 2, "G_Y": 2, "L_Y": 2},  # Text only; colors use full precision.
    "color_max": {"input": None, "label": None},  # None: shared maximum of selected cells.
    "highlight": {"enabled": False, "sample_rate": 1.00, "bins": 50, "color": "#ff953f"},
    "png_dpi": 240,
    "bbox_inches": "tight",
    "pad_inches": 0.05,
    "save_pdf": True,
    "show_inline": False,
    "zero_tolerance": 1e-8,  # Only roundoff negatives are clipped in the display.
}

TERMS = ("G_X", "L_X", "G_Y", "L_Y")
# Times New Roman lacks U+2223 (\mid). Its own ASCII bar with relation-sized
# spacing denotes the same conditioning operation without a fallback font.
TITLES = (r"$I(X;T_s\;|\;T_v)$", r"$I(X;T_v\;|\;T_s)$",
          r"$I(T_s;Y\;|\;T_v)$", r"$I(T_v;Y\;|\;T_s)$")

# TITLES = (
#     r"$\mathbfit{I(X;T_s\;|\;T_v)}$",
#     r"$\mathbfit{I(X;T_v\;|\;T_s)}$",
#     r"$\mathbfit{I(T_s;Y\;|\;T_v)}$",
#     r"$\mathbfit{I(T_v;Y\;|\;T_s)}$",
# )


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False, default=str), encoding="utf-8")


def plot_style(config):
    """Apply the same text/math fonts during creation, rendering and export."""
    family = config["font_family"]
    return {
        "font.family": family, "font.size": config["font_sizes"]["label"],
        "mathtext.fontset": "custom", "mathtext.rm": family,
        "mathtext.it": family + ":italic", "mathtext.bf": family + ":bold",
        "mathtext.bfit": family + ":italic:bold", "mathtext.sf": family,
        "mathtext.tt": family, "mathtext.cal": family, "mathtext.fallback": None,
        "pdf.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": False,
    }


def load_selection(config):
    """Select exact saved cells, preserving the per-model arithmetic means."""
    source = Path(config["source"]).resolve()
    raw = source.read_bytes()
    payload = json.loads(raw)
    if payload["units"] != "bits":
        raise ValueError("Expected saved MI in bits")
    if payload["aggregation"] != "arithmetic mean of per-seed information, equal weight per seed":
        raise ValueError("Unexpected source aggregation")
    attacks, bins, rates = config["attacks"], config["bins"], config["sample_rates"]
    for name, values in [("attacks", attacks), ("bins", bins), ("sample_rates", rates)]:
        if not values or len(set(values)) != len(values):
            raise ValueError(f"Empty or duplicate {name}")
    if rates != sorted(rates) or any(not 0 < r <= 1 for r in rates):
        raise ValueError("Sample rates must be increasing fractions in (0, 1]")
    sizes = [int(round(config["owner_sample_count"] * rate)) for rate in rates]
    if any(abs(n - config["owner_sample_count"] * r) > 1e-8 for n, r in zip(sizes, rates)):
        raise ValueError("A sample rate does not map to an integer sample count")
    if max(payload["sizes"]) != config["owner_sample_count"]:
        raise ValueError("Source full owner size differs from the rate denominator")
    index = {}
    for row in payload["records"]:
        key = row["attack"], row["N"], row["bins"]
        if key in index:
            raise ValueError(f"Duplicate source cell: {key}")
        index[key] = row
    selected = []
    for attack in attacks:
        for rate, n in zip(rates, sizes):
            for b in bins:
                key = attack, n, b
                if key not in index:
                    raise ValueError(f"Requested cell does not exist: {key}")
                row = index[key]
                values = np.array([row[term] for term in TERMS], dtype=float)
                if not np.isfinite(values).all() or values.min() < -config["zero_tolerance"]:
                    raise ValueError(f"Invalid conditional information: {key}")
                if row["seeds"] != payload["seeds_by_attack"][attack] or row["n_seeds"] != len(row["seeds"]):
                    raise ValueError(f"Inconsistent saved model population: {key}")
                selected.append({**row, "sample_rate": rate})
    cube = np.array([[[[index[a, n, b][term] for b in bins] for n in sizes]
                      for term in TERMS] for a in attacks])
    return {
        "source": str(source), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "units": payload["units"], "aggregation": payload["aggregation"],
        "attacks": attacks, "bins": bins, "sample_rates": rates, "sizes": sizes,
        "owner_sample_count": config["owner_sample_count"],
        "records": selected,
    }, cube


def draw_figure(cube, config):
    """Draw four paired columns with shared scales within each equation."""
    attacks, bins, rates = config["attacks"], config["bins"], config["sample_rates"]
    fonts = config["font_sizes"]
    group_max = [config["color_max"][name] for name in ("input", "label")]
    for i in range(2):
        if group_max[i] is None:
            group_max[i] = max(float(cube[:, i*2:i*2+2].max()), 1e-12)
        if group_max[i] <= 0 or cube[:, i*2:i*2+2].max() > group_max[i] + config["zero_tolerance"]:
            raise ValueError("Color maximum must be positive and cover every selected value")
    norms = [Normalize(0, group_max[0])] * 2 + [Normalize(0, group_max[1])] * 2
    cmap = plt.get_cmap(config["cmap"])
    with plt.rc_context(plot_style(config)):
        fig, axes = plt.subplots(len(attacks), 4, figsize=config["figsize"], squeeze=False)
        fig.subplots_adjust(left=.135, right=.985, top=.954, bottom=.15, hspace=.20, wspace=.12)
        for ai, attack in enumerate(attacks):
            for ti, term in enumerate(TERMS):
                ax = axes[ai, ti]
                # A vector mesh also keeps the PDF cells sharp at any zoom level.
                ax.pcolormesh(np.arange(len(bins)+1)-.5, np.arange(len(rates)+1)-.5,
                              np.maximum(cube[ai, ti], 0), cmap=cmap, norm=norms[ti],
                              shading="flat", edgecolors=(1, 1, 1, .12), linewidth=.3,
                              rasterized=False)
                ax.set_xticks(range(len(bins)), [str(b) for b in bins])
                ax.set_yticks(range(len(rates)), [f"{r:.0%}" for r in rates])
                ax.tick_params(length=0, pad=5, labelsize=fonts["tick"],
                               labelleft=ti == 0, labelbottom=ai == len(attacks)-1)
                if ai == 0:
                    ax.set_title(TITLES[ti], fontsize=fonts["column"], pad=12)
                if ai == len(attacks)-1:
                    ax.set_xlabel("Number of Bins", fontsize=fonts["label"], labelpad=9)
                if ti == 0:
                    ax.set_ylabel("Query Set Rate", fontsize=fonts["label"], labelpad=9)
                    pos = ax.get_position()
                    fig.text(config["attack_label_x"], (pos.y0+pos.y1)/2,
                             config["display_labels"].get(attack, attack),
                             ha="center", va="center", fontsize=fonts["attack"], weight="bold",
                             rotation=config["attack_label_rotation"], rotation_mode="anchor")
                for yi in range(len(rates)):
                    for xi in range(len(bins)):
                        value = max(float(cube[ai, ti, yi, xi]), 0)
                        rgb = cmap(norms[ti](value))[:3]
                        luminance = np.dot(rgb, [.2126, .7152, .0722])
                        ax.text(xi, yi, f"{value:.{config['decimals'][term]}f}", ha="center", va="center",
                                color="#102328" if luminance > .53 else "white", fontsize=fonts["cell"])
                highlight = config["highlight"]
                if highlight["enabled"] and highlight["sample_rate"] in rates and highlight["bins"] in bins:
                    ax.add_patch(Rectangle((bins.index(highlight["bins"])-.5, rates.index(highlight["sample_rate"])-.5),
                                           1, 1, fill=False, edgecolor=highlight["color"], linewidth=2.2))
                for spine in ax.spines.values():
                    spine.set_visible(False)
        for first in (0, 2):
            x0, x1 = axes[0, first].get_position().x0, axes[0, first+1].get_position().x1
            cax = fig.add_axes([x0+.012, .047, x1-x0-.024, .016])
            cb = fig.colorbar(ScalarMappable(norm=norms[first], cmap=cmap), cax=cax, orientation="horizontal")
            cb.solids.set_rasterized(False)
            cb.ax.tick_params(labelsize=fonts.get("colorbar_tick", fonts["tick"]), length=3)
            cb.outline.set_linewidth(.6)
    return fig, {"input": group_max[0], "label": group_max[1]}


def main(config=None):
    config = deepcopy(CONFIG if config is None else config)
    selection, cube = load_selection(config)
    out = Path(config["output_dir"]).resolve()
    source_dir = Path(config["source"]).resolve().parent
    if out == source_dir or source_dir in out.parents:
        raise ValueError("Keep plot outputs separate from the saved calculation directory")
    out.mkdir(parents=True, exist_ok=True)
    fig, color_max = draw_figure(cube, config)
    stem = out / config["output_name"]
    with plt.rc_context(plot_style(config)):
        fig.savefig(stem.with_suffix(".png"), dpi=config["png_dpi"], facecolor="white",
                    bbox_inches=config["bbox_inches"], pad_inches=config["pad_inches"])
        if config["save_pdf"]:
            fig.savefig(stem.with_suffix(".pdf"), facecolor="white",
                        bbox_inches=config["bbox_inches"], pad_inches=config["pad_inches"])
    write_json(out / "selected_mean_results.json", selection)
    if sha256(config["source"]) != selection["source_sha256"]:
        raise RuntimeError("Source changed while plotting")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "plot_script_sha256": sha256(__file__), "source_sha256": selection["source_sha256"],
        "source_unchanged": True, "selected_cells": len(selection["records"]),
        "displayed_values": int(cube.size), "cube_shape": list(cube.shape),
        "rate_to_size": {f"{r:.0%}": n for r, n in zip(selection["sample_rates"], selection["sizes"])},
        "color_max": color_max,
        "roundoff_negative_displayed_as_zero": int((cube < 0).sum()),
    }
    write_json(out / "plot_manifest.json", manifest)
    (out / "README.md").write_text(
        "# CF10 RN18：四类 case 的独立绘图\n\n"
        "仅筛选既有均值结果，不加载模型、不重新计算 MI，不重新汇总 seed。\n\n"
        f"源文件：`{selection['source']}`\n\n"
        f"Cases：{', '.join(config['attacks'])}。Bins：{config['bins']}。\n\n"
        f"Sample rate → 样本数：{manifest['rate_to_size']}。\n\n"
        "每行一个 attack；四列依次为公式 (4) 输入增益、输入损失，以及公式 (5) 标签增益、标签损失。"
        f"每个热图为 {len(config['sample_rates'])}×{len(config['bins'])} 参数格。图中保留逐模型计算后取均值的数值，不显示 seed 数量。\n\n"
        "四列仅以公式标识；没有总标题、Equation 标记、metric 文字说明或色条下方说明。"
        "格内数字保留两位小数，颜色和色条仍按未经舍入的原始均值绘制。\n\n"
        f"全图字体：{config['font_family']}，包括公式；字号在顶部 `CONFIG['font_sizes']` 中调整（单位 pt）。\n\n"
        "`cell`：格内数字；`label`：轴标题；`tick`：横纵轴刻度；`column`：公式；"
        "`attack`：左侧 case 名称；`colorbar_tick`：色条数值刻度。\n\n"
        "- PNG/PDF：图像和矢量版本。\n"
        f"- `selected_mean_results.json`：精确选取的 {len(selection['records'])} 组均值记录及 rate。\n"
        "- `plot_manifest.json`：筛选配置、色标上界、输入和脚本哈希。\n\n"
        "调整 `plot_mi_decomposition_cf10_rn18.py` 顶部 `CONFIG` 后运行即可重画。"
        "在 notebook 中可调用 `main(config)`，或 `load_selection(config)` 与 `draw_figure(cube, config)`。\n",
        encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "selected_cells": len(selection["records"]),
                      "displayed_values": int(cube.size), "rate_to_size": manifest["rate_to_size"],
                      "source_unchanged": True}, indent=2))
    if config["show_inline"]:
        from IPython.display import display
        with plt.rc_context(plot_style(config)):
            display(fig)
    plt.close(fig)
    return selection, manifest


if __name__ == "__main__":
    main()
