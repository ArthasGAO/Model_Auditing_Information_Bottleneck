"""Two-column signed differences of saved MI means, using the current style.

The four-column script supplies data selection, font handling and default
configuration. This script does not change that script or its output files.
Run directly, or call main(custom_config) from a notebook.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import plot_mi_decomposition_cf10_rn18 as base
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np


# Inherit the current source, attacks, bins, rates, figure dimensions, Times
# New Roman font sizes, vertical attack labels, viridis map and tight export.
CONFIG = deepcopy(base.CONFIG)
CONFIG.update({
    "output_dir": base.ROOT / "outputs/mi_decomposition_cf10_rn18_net_plot",
    "output_name": "equations_4_5_net_selected_rates",
    "decimals": {"delta_X": 2, "delta_Y": 2},
    # None: linear range covering the original signed values and zero.
    # Each column shares its scale across all four attacks.
    "color_range": {"delta_X": None, "delta_Y": None},
    "xlabel": "Number of Bins",
    "ylabel": "Query Set Rate",
})
CONFIG.pop("color_max", None)  # Four-column nonnegative bounds do not apply.
# Optional local overrides, e.g. CONFIG["font_sizes"]["cell"] = 16.

TERMS = ("delta_X", "delta_Y")
TITLES = (
    r"$I(X;T_s\;|\;T_v)-I(X;T_v\;|\;T_s)$",
    r"$I(T_s;Y\;|\;T_v)-I(T_v;Y\;|\;T_s)$",
)


def load_net_selection(config):
    selection, original = base.load_selection(config)
    # Subtract full-precision means, not the rounded cell annotations.
    net = np.stack((original[:, 0] - original[:, 1],
                    original[:, 2] - original[:, 3]), axis=1)
    records, residual = [], 0.0
    for row in selection["records"]:
        dx, dy = row["G_X"] - row["L_X"], row["G_Y"] - row["L_Y"]
        residual = max(residual, abs(dx-row["delta_X"]), abs(dy-row["delta_Y"]))
        records.append({**row, "net_input": dx, "net_label": dy})
    if residual > 1e-12 or not np.isfinite(net).all():
        raise ValueError("Signed differences disagree with the saved MI changes")
    stats = {}
    for ai, attack in enumerate(config["attacks"]):
        stats[attack] = {}
        for ti, term in enumerate(TERMS):
            values, tol = net[ai, ti], config["zero_tolerance"]
            stats[attack][term] = {
                "positive": int((values > tol).sum()),
                "negative": int((values < -tol).sum()),
                "approximately_zero": int((np.abs(values) <= tol).sum()),
                "min": float(values.min()), "max": float(values.max()),
            }
    return {**selection, "records": records,
            "definitions": {"net_input": "G_X - L_X", "net_label": "G_Y - L_Y"},
            "direction_counts": stats, "max_saved_delta_residual": residual}, net


def draw_figure(net, config):
    attacks, bins, rates = config["attacks"], config["bins"], config["sample_rates"]
    fonts, tol = config["font_sizes"], config["zero_tolerance"]
    cmap = plt.get_cmap(config["cmap"])
    norms, ranges = [], {}
    for ti, term in enumerate(TERMS):
        values = net[:, ti]
        limits = config["color_range"][term]
        if limits is None:
            limits = (min(float(values.min()), 0.0), max(float(values.max()), 0.0))
            if limits[0] == limits[1]:
                limits = (-tol, tol)
        lo, hi = map(float, limits)
        if not np.isfinite([lo, hi]).all() or not lo < hi or not lo <= 0 <= hi:
            raise ValueError(f"Invalid signed color range for {term}: {limits}")
        if values.min() < lo or values.max() > hi:
            raise ValueError(f"Color range hides values in {term}")
        ranges[term] = [lo, hi]
        norms.append(Normalize(lo, hi))
    with plt.rc_context(base.plot_style(config)):
        fig, axes = plt.subplots(len(attacks), 2, figsize=config["figsize"], squeeze=False)
        fig.subplots_adjust(left=.135, right=.985, top=.954, bottom=.15, hspace=.20, wspace=.12)
        for ai, attack in enumerate(attacks):
            for ti, term in enumerate(TERMS):
                ax = axes[ai, ti]
                # Preserve signed full-precision values in the mesh and colorbar.
                ax.pcolormesh(np.arange(len(bins)+1)-.5, np.arange(len(rates)+1)-.5,
                              net[ai, ti], cmap=cmap, norm=norms[ti], shading="flat",
                              edgecolors=(1, 1, 1, .12), linewidth=.3, rasterized=False)
                ax.set_xticks(range(len(bins)), [str(b) for b in bins])
                ax.set_yticks(range(len(rates)), [f"{r:.0%}" for r in rates])
                ax.tick_params(length=0, pad=5, labelsize=fonts["tick"],
                               labelleft=ti == 0, labelbottom=ai == len(attacks)-1)
                if ai == 0:
                    ax.set_title(TITLES[ti], fontsize=fonts["column"], pad=12)
                if ai == len(attacks)-1:
                    ax.set_xlabel(config["xlabel"], fontsize=fonts["label"], labelpad=9)
                if ti == 0:
                    ax.set_ylabel(config["ylabel"], fontsize=fonts["label"], labelpad=9)
                    pos = ax.get_position()
                    fig.text(config["attack_label_x"], (pos.y0+pos.y1)/2,
                             config["display_labels"].get(attack, attack),
                             ha="center", va="center", fontsize=fonts["attack"], weight="bold",
                             rotation=config["attack_label_rotation"], rotation_mode="anchor")
                for yi in range(len(rates)):
                    for xi in range(len(bins)):
                        value = float(net[ai, ti, yi, xi])
                        # Suppress a minus sign only for numerical zero, not small
                        # real negative changes that round to -0.00 at two decimals.
                        shown = 0.0 if abs(value) <= tol else value
                        rgb = cmap(norms[ti](value))[:3]
                        luminance = np.dot(rgb, [.2126, .7152, .0722])
                        ax.text(xi, yi, f"{shown:.{config['decimals'][term]}f}",
                                ha="center", va="center", fontsize=fonts["cell"],
                                color="#102328" if luminance > .53 else "white")
                h = config["highlight"]
                if h["enabled"] and h["sample_rate"] in rates and h["bins"] in bins:
                    ax.add_patch(Rectangle((bins.index(h["bins"])-.5, rates.index(h["sample_rate"])-.5),
                                           1, 1, fill=False, edgecolor=h["color"], linewidth=2.2))
                for spine in ax.spines.values():
                    spine.set_visible(False)
        for ti in range(2):
            pos = axes[0, ti].get_position()
            cax = fig.add_axes([pos.x0+.012, .047, pos.width-.024, .016])
            cb = fig.colorbar(ScalarMappable(norm=norms[ti], cmap=cmap), cax=cax, orientation="horizontal")
            cb.solids.set_rasterized(False)
            cb.ax.tick_params(labelsize=fonts.get("colorbar_tick", fonts["tick"]), length=3)
            cb.outline.set_linewidth(.6)
    return fig, ranges


def main(config=None):
    config = deepcopy(CONFIG if config is None else config)
    base_script_hash = base.sha256(base.__file__)
    selection, net = load_net_selection(config)
    out = Path(config["output_dir"]).resolve()
    protected = (Path(config["source"]).resolve().parent, Path(base.CONFIG["output_dir"]).resolve())
    if any(out == p or p in out.parents for p in protected):
        raise ValueError("Use a separate directory for the two-column outputs")
    out.mkdir(parents=True, exist_ok=True)
    fig, ranges = draw_figure(net, config)
    stem = out / config["output_name"]
    with plt.rc_context(base.plot_style(config)):
        fig.savefig(stem.with_suffix(".png"), dpi=config["png_dpi"], facecolor="white",
                    bbox_inches=config["bbox_inches"], pad_inches=config["pad_inches"])
        if config["save_pdf"]:
            fig.savefig(stem.with_suffix(".pdf"), facecolor="white",
                        bbox_inches=config["bbox_inches"], pad_inches=config["pad_inches"])
    assert base.sha256(config["source"]) == selection["source_sha256"]
    assert base.sha256(base.__file__) == base_script_hash
    base.write_json(out / "net_mean_results.json", selection)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "source_sha256": selection["source_sha256"], "source_unchanged": True,
        "plot_script_sha256": base.sha256(__file__), "base_plot_script_sha256": base_script_hash,
        "selected_cells": len(selection["records"]), "displayed_values": int(net.size),
        "cube_shape": list(net.shape), "color_ranges": ranges,
        "max_saved_delta_residual": selection["max_saved_delta_residual"],
        "direction_counts": selection["direction_counts"],
    }
    base.write_json(out / "plot_manifest.json", manifest)
    (out / "README.md").write_text(
        "# CF10 RN18：条件信息净变化\n\n"
        "第一列为 G_X - L_X，等于 I(X;T_s) - I(X;T_v)；"
        "第二列为 G_Y - L_Y，等于 I(T_s;Y) - I(T_v;Y)。\n\n"
        "使用已有逐模型 MI 算术均值的完整精度进行相减，并与已存 delta_X/delta_Y 交叉核对。"
        "没有重新推理模型或重新估计 MI。\n\n"
        "沿用四列图的当前筛选、画布、Times New Roman、字号、轴标签、竖排 case 名称、"
        "两位小数、viridis、无橘框、无总标题及说明文字、tight 裁边及 0.05 英寸留白。"
        "两列各自共享一个线性色阶，刻度保留正负号，原始差值参与着色；色条无文字说明。\n\n"
        "两位小数中的 -0.00 表示真实的小幅负变化四舍五入后的显示；"
        "绝对值不超过 1e-8 bits 的数值误差显示为 0.00。完整精度在 JSON 中保留。\n\n"
        "在所选 100 个参数格中，输入净变化全部为正；标签净变化 95 格为负，"
        "PR-80% 的其余 5 格约为零。该汇总使用 1e-8 bits 的零容差。\n\n"
        "- `net_mean_results.json`：原四项均值、两项完整精度差值和方向统计。\n"
        "- `plot_manifest.json`：全部绘图配置、色阶范围、代码和输入哈希。\n\n"
        "运行 `plot_mi_decomposition_net_cf10_rn18.py` 即可重画；顶部 CONFIG 可覆盖原图的任意配置。\n",
        encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "selected_cells": len(selection["records"]),
                      "displayed_values": int(net.size), "color_ranges": ranges,
                      "max_saved_delta_residual": selection["max_saved_delta_residual"],
                      "direction_counts": selection["direction_counts"]}, indent=2))
    if config["show_inline"]:
        from IPython.display import display
        with plt.rc_context(base.plot_style(config)):
            display(fig)
    plt.close(fig)
    return selection, manifest


if __name__ == "__main__":
    main()
