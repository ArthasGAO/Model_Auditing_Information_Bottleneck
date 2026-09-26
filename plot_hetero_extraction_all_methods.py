"""Plot heterogeneous Knockoff and DFMS-HL MI clouds with matched H0 pools.

This module reads existing MI tables. It never synthesizes or rewrites points.
Each figure has one dataset, one victim architecture, and one *different*
suspect architecture. The negative/reference pool follows the suspect arch.
"""

from __future__ import annotations

import importlib
import re
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Bbox

import plot_fixed_split_h0 as fixed_h0


BASE = Path(__file__).resolve().parent
ARCHES = ("RN18", "VGG16", "DeiT")
MODEL_LABEL = {
    "RN18": "ResNet-18",
    "VGG16": "VGG16",
    ("CF10", "DeiT"): "deit_tiny_patch16_224",
    ("CF100", "DeiT"): "deit_tiny_distilled_patch16_224",
}
ARCH_SCENARIO = {"RN18": "ResNet-18", "VGG16": "VGG16", "DeiT": "DeiT"}
SUSPECT_SUFFIX = {"RN18": "Cross18", "VGG16": "Cross16", "DeiT": "CrossDeiT"}


def _model_label(dataset, arch):
    return MODEL_LABEL[(dataset, arch)] if arch == "DeiT" else MODEL_LABEL[arch]


def _victim_scenario(dataset, arch):
    number = dataset.removeprefix("CF")
    if arch == "DeiT":
        variant = "Plain" if dataset == "CF10" else "Distill"
        return f"CIFAR-{number}_DeiT_{variant}_25000"
    return f"CIFAR-{number}_{ARCH_SCENARIO[arch]}_25000"


def _attack_prefix(dataset, victim):
    return f"CIFAR-{dataset.removeprefix('CF')}_{ARCH_SCENARIO[victim]}_25000"


def _method_source(dataset, victim, suspect, csv_paths):
    prefix = _attack_prefix(dataset, victim)
    suffix = SUSPECT_SUFFIX[suspect]
    number = dataset.removeprefix("CF")
    knockoff = f"{prefix}_Knockoff_Same{number}_{suffix}"
    if dataset == "CF10" and victim != "DeiT":
        hl = f"{prefix}_DFMS_Cross100-40C_{suffix}"
        hl_source = csv_paths["multiple"]
        hl_illustrative = False
    else:
        hl = f"{prefix}_DFMS_Illustrative_{suffix}"
        hl_source = csv_paths["multiple1"]
        hl_illustrative = True
    return [
        dict(method="Knockoff", scenario=knockoff, source=csv_paths["multiple"],
             pattern=rf"^{re.escape(knockoff)}_\d+_1\.0$", illustrative=False),
        dict(method="HL", scenario=hl, source=hl_source,
             pattern=(rf"^{re.escape(hl)}_synthetic_seed=\d+$" if hl_illustrative
                      else rf"^{re.escape(hl)}_\d+_1\.0$"),
             illustrative=hl_illustrative),
    ]


def _preflight_source(source, dataset, victim, suspect, bins, in_size):
    df = pd.read_csv(source["source"])
    selected = df.loc[
        df["Scenario"].eq(source["scenario"])
        & df["model_name"].str.fullmatch(source["pattern"].removeprefix("^").removesuffix("$"), na=False)
        & df["bins"].eq(bins)
        & df["in_size"].eq(in_size)
        & df["rate"].eq(1.0)
    ]
    where = f"{dataset} {victim}->{suspect} {source['method']}"
    assert len(selected) == 50, f"{where}: expected exactly 50 rows, got {len(selected)}"
    assert sorted(selected["seed"].tolist()) == list(range(50)), f"{where}: seed coverage mismatch"
    assert selected["victim_model"].eq(_model_label(dataset, victim)).all(), f"{where}: victim_model mismatch"
    assert selected["substitute_model"].eq(_model_label(dataset, suspect)).all(), f"{where}: substitute_model mismatch"
    xy = selected[["I(X;T)-In", "I(T;Y)-In"]].to_numpy(dtype=float)
    assert np.isfinite(xy).all(), f"{where}: non-finite MI"
    return dict(dataset=dataset, victim=victim, suspect=suspect,
                method=source["method"], source=str(source["source"]),
                scenario=source["scenario"], n=len(selected),
                illustrative=source["illustrative"])


def _square_box(fig, bbox, pad):
    box = bbox.padded(pad)
    side = max(box.width, box.height)
    cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
    return Bbox.from_extents(cx-side/2, cy-side/2, cx+side/2, cy+side/2)


def _style_dual_negative_groups(ax, cfg, victim, suspect):
    """Color two *unchanged* fixed-H0 groups and replace the single-pool legend.

    plot_fixed_split_single draws suspect negatives/references first, victim
    negatives/references second, then the two positive clouds and victim. Both
    boundary lines come directly from their own frozen reference split.
    """
    assert len(ax.collections) == 7 and len(ax.lines) == 2, (
        "Expected two negative/reference pairs, two positives, and one victim")
    colors = cfg["dual_null_colors"]
    assert colors["suspect"] != colors["victim"]
    marker = MarkerStyle("+")
    plus_path = marker.get_path().transformed(marker.get_transform())
    ref_area = cfg["reference_marker"]["area"]
    ref_width = cfg["reference_marker"]["linewidth"]
    for index, role in enumerate(("suspect", "victim")):
        color = colors[role]
        negatives, references = ax.collections[index*2:index*2+2]
        negatives.set_facecolor(color)
        negatives.set_edgecolor("none")
        references.set_paths([plus_path])
        references.set_facecolors("none")
        references.set_edgecolors(color)
        references.set_linewidths(ref_width)
        references.set_sizes([ref_area])
        references.set_zorder(cfg["reference_marker"]["zorder"])
        boundary = ax.lines[index]
        boundary.set_color(color)
        boundary.set_linestyle(cfg["boundary_style"]["linestyle"])
        boundary.set_linewidth(cfg["boundary_style"]["linewidth"])
        boundary.set_zorder(cfg["boundary_style"]["zorder"])

    marker_size = np.sqrt(cfg["role_styles"]["positive_style"]["s"])
    neg_size = np.sqrt(cfg["role_styles"]["negative_style"]["s"])
    handles = [
        Line2D([], [], color="none", marker="s", markersize=np.sqrt(cfg["role_styles"]["victim_style"]["s"]),
               markerfacecolor="black", markeredgecolor="none", label=f"Victim model ({victim})"),
        Line2D([], [], color="none", marker="^", markersize=marker_size,
               markerfacecolor=cfg["method_colors"]["Knockoff"], markeredgecolor="none", label="Knockoff"),
        Line2D([], [], color="none", marker="^", markersize=marker_size,
               markerfacecolor=cfg["method_colors"]["HL"], markeredgecolor="none", label="HL"),
    ]
    for role, arch in (("suspect", suspect), ("victim", victim)):
        color = colors[role]
        handles.append(Line2D([], [], color=color, linestyle="--", linewidth=cfg["boundary_style"]["linewidth"],
                              marker="^", markersize=neg_size, markerfacecolor=color,
                              markeredgecolor="none", label=f"{arch} negatives + H0"))
        handles.append(Line2D([], [], color="none", marker="+", markersize=np.sqrt(ref_area),
                              markeredgecolor=color, markeredgewidth=ref_width,
                              label=f"{arch} references"))
    legend_options = {**cfg["legend_options"], **cfg.get("dual_legend_options", {})}
    ax.legend(handles=handles,
              prop=FontProperties(family=cfg["font_family"], size=cfg["font_sizes"]["legend"]),
              **legend_options)
    ax.figure.canvas.draw()


def _pad_dual_view_to_all_data(ax, fraction=.07):
    """Keep the extra victim-model square visible without changing any data/H0."""
    xy = [np.asarray(collection.get_offsets(), dtype=float)
          for collection in ax.collections]
    xy += [np.asarray(line.get_xydata(), dtype=float) for line in ax.lines]
    stacked = np.vstack([points for points in xy if points.size])
    assert stacked.shape[1] == 2 and np.isfinite(stacked).all()
    low, high = stacked.min(axis=0), stacked.max(axis=0)
    pad = np.maximum(high-low, 1e-6)*fraction
    current_x, current_y = ax.get_xlim(), ax.get_ylim()
    ax.set_xlim(min(current_x[0], low[0]-pad[0]), max(current_x[1], high[0]+pad[0]))
    ax.set_ylim(min(current_y[0], low[1]-pad[1]), max(current_y[1], high[1]+pad[1]))


def render(config, framework):
    """Preflight all enabled architecture pairs, then save square figures.

    ``framework`` is the notebook globals after running its MI plotting
    framework cell. Returns (coverage_dataframe, output_paths, figures).
    """
    importlib.reload(fixed_h0)
    cfg = deepcopy(config)
    required = ("TableGroupSpec", "VictimSpec", "points_from_tablegroupspec",
                "_resolve_victim_point", "clear_csv_cache")
    assert all(key in framework for key in required), "Run the notebook's MI plotting framework cell first"
    assert cfg["bins"] == 50 and cfg["in_size"] == 25000
    assert cfg["k"] == 30 and cfg["round_id"] == 44 and cfg["alpha"] == .01
    assert len(cfg["figsize"]) == 2 and cfg["figsize"][0] == cfg["figsize"][1]
    assert set(cfg["datasets"]) <= {"CF10", "CF100"}
    assert cfg["datasets"]
    pairs = [(v, s) for v, s in cfg["pairs"] if v != s]
    assert pairs and len(set(pairs)) == len(pairs)
    assert all(v in ARCHES and s in ARCHES for v, s in pairs)
    paths = {key: BASE / value for key, value in cfg["csv_paths"].items()}
    output_dir = BASE / cfg["output_dir"]
    result_dir = BASE / cfg["result_dir"]
    rows, figures, outputs = [], [], []
    try:
        for dataset in cfg["datasets"]:
            for victim, suspect in pairs:
                methods = _method_source(dataset, victim, suspect, paths)
                rows.extend(_preflight_source(source, dataset, victim, suspect,
                                              cfg["bins"], cfg["in_size"])
                            for source in methods)
        for dataset in cfg["datasets"]:
            for victim, suspect in pairs:
                victim_scenario = _victim_scenario(dataset, victim)
                h0_scenario = _victim_scenario(dataset, suspect)
                key = f"{dataset}_{suspect}"
                methods = _method_source(dataset, victim, suspect, paths)
                specs = []
                colors = {}
                for source in methods:
                    label = source["method"]
                    spec = framework["TableGroupSpec"](
                        label=label, csv_path=source["source"], scenario=source["scenario"],
                        model_name=re.compile(source["pattern"]), seeds=list(range(50)),
                        rates=[1.0], bins=cfg["bins"], in_size=cfg["in_size"],
                        domain="in", mode="points", style=dict(cfg["role_styles"]["positive_style"]),
                    )
                    points, _ = fixed_h0.selected_points(spec, framework, cfg["bins"], cfg["in_size"])
                    assert sorted(p["seed"] for p in points) == list(range(50))
                    specs.append(dict(h0=key, synthetic=True, spec=spec))
                    colors[label] = cfg["method_colors"][label]
                negative_groups = {key: dict(scenario=h0_scenario,
                                             label=f"{dataset} {suspect}",
                                             color=cfg["role_styles"]["negative_style"]["color"],
                                             seeds=None)}
                if cfg.get("dual_negative_groups", False):
                    victim_key = f"{dataset}_{victim}"
                    assert victim_key != key
                    negative_groups[victim_key] = dict(
                        scenario=victim_scenario, label=f"{dataset} {victim}",
                        color=cfg["dual_null_colors"]["victim"], seeds=None)
                panel = dict(
                    negative_groups=negative_groups,
                    positive_groups=specs,
                    victims=[framework["VictimSpec"](
                        csv_path=paths["victim"], scenario=victim_scenario,
                        seeds=[42], rates=[1.0], bins=cfg["bins"], in_size=cfg["in_size"],
                        domain="in", label=f"{dataset} {victim} victim",
                        style=dict(cfg["role_styles"]["victim_style"]),
                    )],
                )
                fig, ax, summary, point_results = fixed_h0.plot_fixed_split_single(
                    panel, framework=framework, figsize=cfg["figsize"],
                    font_family=cfg["font_family"], font_sizes=cfg["font_sizes"],
                    role_styles=cfg["role_styles"], legend_options=cfg["legend_options"],
                    legend_labels=cfg["legend_labels"], axes_options=cfg["axes_options"],
                    boundary_style=cfg["boundary_style"],
                    positive_group_colors=colors, show=False,
                    bins=cfg["bins"], in_size=cfg["in_size"], k=cfg["k"],
                    round_id=cfg["round_id"], alpha=cfg["alpha"],
                    negative_csv=paths["negative"], result_dir=result_dir,
                )
                if cfg.get("dual_negative_groups", False):
                    _style_dual_negative_groups(ax, cfg, victim, suspect)
                    _pad_dual_view_to_all_data(ax)
                elif cfg["reference_open"]:
                    fixed_h0.style_reference_open_marker(ax, **cfg["reference_marker"])
                limits = cfg.get("axis_limits", {}).get(f"{dataset}_{victim}_to_{suspect}", {})
                if limits.get("x") is not None:
                    ax.set_xlim(*limits["x"])
                if limits.get("y") is not None:
                    ax.set_ylim(*limits["y"])
                assert len(point_results.query("role == 'negative'")) == (
                    100 if cfg.get("dual_negative_groups", False) else 50)
                if cfg.get("dual_negative_groups", False):
                    assert set(summary.query("role == 'negative'")["h0"]) == {key, victim_key}
                    assert summary.query("role == 'negative'")["n"].eq(50).all()
                    assert all(group["h0"] == key for group in panel["positive_groups"])
                assert all(summary.query("role == 'positive'")["n"].eq(50))
                assert panel["victims"][0].scenario == victim_scenario
                assert panel["negative_groups"][key]["scenario"] == h0_scenario
                if cfg.get("dual_negative_groups", False):
                    assert panel["negative_groups"][victim_key]["scenario"] == victim_scenario
                figures.append((dataset, victim, suspect, fig))
        coverage = pd.DataFrame(rows)
        if cfg["save_pdf"] or cfg["save_png"]:
            output_dir.mkdir(parents=True, exist_ok=True)
            for dataset in cfg["datasets"]:
                subset = [(v, s, fig) for d, v, s, fig in figures if d == dataset]
                for _, _, fig in subset:
                    fig.canvas.draw()
                tight = [fig.get_tightbbox(fig.canvas.get_renderer()) for _, _, fig in subset]
                shared = Bbox.union(tight) if cfg["shared_crop"] else None
                for (victim, suspect, fig), own_bbox in zip(subset, tight):
                    bbox = _square_box(fig, shared if shared is not None else own_bbox,
                                       cfg["pad_inches"]) if cfg["export_square"] else "tight"
                    stem = f"{dataset}_{victim}_to_{suspect}_Knockoff_HL"
                    if cfg.get("dual_negative_groups", False):
                        stem += "_dual_H0"
                    for fmt, enabled in (("pdf", cfg["save_pdf"]), ("png", cfg["save_png"])):
                        if enabled:
                            path = output_dir / f"{stem}.{fmt}"
                            fixed_h0.save_paper_figure(fig, path, dpi=cfg["dpi"],
                                                       bbox_inches=bbox,
                                                       pad_inches=0 if cfg["export_square"] else cfg["pad_inches"])
                            outputs.append(path)
        if not cfg["show_inline"]:
            for _, _, _, fig in figures:
                plt.close(fig)
        return coverage, outputs, figures
    except Exception:
        for _, _, _, fig in figures:
            plt.close(fig)
        raise
