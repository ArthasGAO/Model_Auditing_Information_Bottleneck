"""Same-architecture MI planes: one mean per suspect group, all 15 H0 references.

Produces three independent figures for each of CIFAR-10 and CIFAR-100. The
negative and positive means are descriptive coordinates only; the F boundary
is the unmodified selected-round predictive F-test at alpha=0.01. Each figure
can optionally overlay the RN18, VGG16 and DeiT negative layers for its dataset,
while keeping the victim and positive means specific to the figure architecture.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
import numpy as np
import pandas as pd

import plot_fixed_split_h0 as fixed_h0
import run_hypothesis_test_same_arch_best_fpr_per_k as audit


ROOT = Path(__file__).resolve().parent
ALL_ARCHITECTURES = ("RN18", "VGG16", "DeiT")
METHOD_COLORS = {
    "FT-LL": "#C35B65", "FT-AL": "#D8823F", "RT-AL": "#A65A82",
    "P-20%": "#C2A03D", "P-80%": "#95602F",
    "KD": "#765CA5", "DKD": "#B668B1",
    "Knockoff": "#D14D3E", "HL": "#704C82",
}

POSITIVE_GROUPS = {
    "FT-LL": "Finetune",
    "FT-AL": "Finetune",
    "RT-AL": "Finetune",

    "P-20%": "Pruning",
    "P-80%": "Pruning",

    "KD": "Distillation",
    "DKD": "Distillation",

    "Knockoff": "Extraction",
    "HL": "Extraction",
}

GROUP_COLORS = {
    "Finetune": "#FCAD38",      # 例如一类共用烟红色
    "Pruning": "#EB7F31",       # 例如土黄色
    "Distillation": "#E45742",  # 例如烟紫色
    "Extraction": "#972828",    # 例如烟棕色
}

METHOD_MARKERS = {
    "FT-LL": "o",
    "FT-AL": "^",
    "RT-AL": "p",
    "P-20%": "D",
    "P-80%": "P",
    "KD": "X",
    "DKD": "v",
    "Knockoff": ">",
    "HL": "<",
}

METHOD_SIZE_SCALE = {
    "FT-LL": 1.00,
    "FT-AL": 1.05,
    "RT-AL": 1.20,
    "P-20%": 0.90,
    "P-80%": 1.05,
    "KD": 1.10,
    "DKD": 1.05,
    "Knockoff": 1.05,
    "HL": 1.05,
}

def default_config():
    """Editable plot settings for standalone runs or notebook imports."""
    return {
        "datasets": ["CF10", "CF100"],
        "architectures": ["RN18", "VGG16", "DeiT"],
        "methods": list(audit.METHODS),
        "k": 15, "alpha": .01, "bins": 50, "in_size": 25000,
        "selection_csv": ROOT / "saved_logs/vanilla/Hypo_Test_SameArch_BestFPR_PerK/round_selection.csv",
        "result_root": ROOT / "saved_logs/vanilla/Hypo_Test_SameArch_BestFPR_PerK",
        "output_dir": ROOT / "saved_plots/same_arch_mean_k15_all_arch_negatives_no_legend_auto_ticks",
        "figsize": (5.2, 5.2), "font_family": "Times New Roman",
        "font_sizes": {"xlabel": 15, "ylabel": 15, "ticks": 13, "legend": 8},
        "tick_count": 3, "tick_margin_fraction": .08,
        "colors": {
                    "victim": "#85A947",
                    "negative": "#1686D9",
                    "reference": "#000000",
                    "boundary": "#4C4C4C",
                    "positive_groups": deepcopy(GROUP_COLORS),
                },
        "markers": {
            "reference": "o",
            "victim": "*",
            "negative": "s",
            "positive": deepcopy(METHOD_MARKERS),
        },
        "sizes": {"victim": 200, "negative": 80, "reference": 8,
                  "positive": 80},
        "alpha_points": {"negative": 1.0, "reference": 1.0, "positive": 1.0},
        "reference_marker": "o", "reference_facecolor": "#000000",
        "reference_linewidth": 0.2,
        "boundary_linewidth": 1.0, "boundary_linestyle": "--",
        "show_all_arch_negative_overlays": True,
        "show_nonmatching_negative_means": True,
        "boundary_display_scales": {arch: 1.0 for arch in ALL_ARCHITECTURES},
        "grid_alpha": .30, "grid_linewidth": .55,
        "show_legend": False,
        "legend": {"loc": "best", "ncol": 2, "frameon": True,
                   "framealpha": .95, "columnspacing": .65,
                   "labelspacing": .25, "borderpad": .3,
                   "handletextpad": .3, "handlelength": 1.0},
        "margins": {"left": .15, "right": .97, "bottom": .13, "top": .97,
                    "data": .055},
        "axis_limits": {},  # e.g. {"CF10_RN18": {"x": (3, 7), "y": (3.08, 3.34)}}
        "save_pdf": True, "save_png": True, "png_dpi": 260,
        "show_inline": False,
    }


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _spaced_ticks(limits, count, margin_fraction):
    lo, hi = limits
    span = hi - lo
    require(np.isfinite([lo, hi]).all() and span > 0, "Invalid axis limits")
    require(isinstance(count, (int, np.integer)) and count >= 2,
            "tick_count must be an integer of at least two")
    require(0 < margin_fraction < .5, "Tick margin must be between zero and one-half")
    ticks = np.linspace(lo + margin_fraction * span,
                        hi - margin_fraction * span, count)
    tick_step = span * (1 - 2 * margin_fraction) / (count - 1)
    decimals = max(0, 1 - int(np.floor(np.log10(tick_step))))
    rounded = np.round(ticks, decimals)
    if len(np.unique(rounded)) == count and lo < rounded[0] < rounded[-1] < hi:
        return rounded
    return ticks


def _victim(victims, scenario, config):
    chosen = victims[(victims.Scenario == scenario) & (victims.seed == 42)
                     & (victims.rate == 1) & (victims.bins == config["bins"])
                     & (victims.in_size == config["in_size"])]
    require(len(chosen) == 1, f"Expected one victim MI row for {scenario}")
    return chosen[["I(X;T)-In", "I(T;Y)-In"]].iloc[0].to_numpy(dtype=float)


def _plot_one(dataset, arch, config, table_rows, positive_models, victims, selections):
    scenario = audit.ARCHES[dataset][arch]["h0"]
    result_dir = Path(config["result_root"]) / dataset / "negatives"
    overlay_arches = (list(ALL_ARCHITECTURES)
                      if config["show_all_arch_negative_overlays"] else [arch])
    # Draw the figure's own negative layer last. With the initial shared style
    # this preserves its boundary and reference points where layers overlap.
    overlay_arches = [name for name in overlay_arches if name != arch] + [arch]
    nulls_by_arch = {}
    boundary_rounds = {}
    for overlay_arch in overlay_arches:
        boundary_scenario = audit.ARCHES[dataset][overlay_arch]["h0"]
        choice = selections[(dataset, overlay_arch, config["k"])]
        require(choice["h0_scenario"] == boundary_scenario,
                "Selection/H0 architecture mismatch")
        boundary_round_id = int(choice["round_id"])
        boundary_null = fixed_h0.load_nulls(
            [boundary_scenario], bins=config["bins"], in_size=config["in_size"],
            k=config["k"], round_id=boundary_round_id, alpha=config["alpha"],
            result_dir=result_dir)[boundary_scenario]
        require(len(boundary_null.reference) == config["k"]
                and len(boundary_null.evaluation) == 50,
                f"{dataset} {overlay_arch}: wrong reference/evaluation group size")
        nulls_by_arch[overlay_arch] = boundary_null
        boundary_rounds[overlay_arch] = boundary_round_id
    xy_cols = ["I(X;T)-In", "I(T;Y)-In"]
    negative_layers_by_arch = {}
    for overlay_arch in overlay_arches:
        overlay_scenario = audit.ARCHES[dataset][overlay_arch]["h0"]
        overlay_null = nulls_by_arch[overlay_arch]
        overlay_round = boundary_rounds[overlay_arch]
        exact_boundary = overlay_null.boundary()
        boundary_display_scale = float(
            config["boundary_display_scales"][overlay_arch])
        displayed_boundary = (
            overlay_null.mu
            + boundary_display_scale * (exact_boundary - overlay_null.mu)
        )
        negative_layers_by_arch[overlay_arch] = {
            "scenario": overlay_scenario,
            "round_id": overlay_round,
            "null": overlay_null,
            "boundary": displayed_boundary,
            "boundary_display_scale": boundary_display_scale,
            "reference": overlay_null.reference[xy_cols].to_numpy(dtype=float),
            "negative_mean": overlay_null.evaluation[xy_cols].to_numpy(dtype=float).mean(axis=0),
        }
    current_negative_layer = negative_layers_by_arch[arch]
    null = current_negative_layer["null"]
    round_id = current_negative_layer["round_id"]
    victim_xy = _victim(victims, scenario, config)
    positive_means = {}
    for method in config["methods"]:
        _, _, values = audit.selected_rows(table_rows, audit.ARCHES[dataset][arch], method)
        xy = np.array(list(values.values()), dtype=float)
        require(xy.shape == (50, 2), f"{dataset} {arch} {method}: need 50 coordinates")
        label = f"{dataset}_{arch}_{method}"
        tested = positive_models[
            (positive_models.family == label) & (positive_models.k_ref == config["k"])
            & (positive_models.round_id == round_id)]
        require(len(tested) == 50 and set(tested.model_name) == set(values),
                f"{label}: saved positive test does not match source models")
        tested_mean = tested[["ixt", "ity"]].to_numpy(dtype=float).mean(axis=0)
        require(np.allclose(tested_mean, xy.mean(axis=0), rtol=0, atol=1e-12),
                f"{label}: saved positive MI differs from plotted source")
        positive_means[method] = xy.mean(axis=0)

    font = config["font_family"]
    findfont(FontProperties(family=font), fallback_to_default=False)
    fig, ax = plt.subplots(figsize=config["figsize"], layout=None)
    fig.set_layout_engine(None)
    fig.subplots_adjust(**{k: config["margins"][k] for k in ("left", "right", "bottom", "top")})
    ax.set_box_aspect(1)
    ax.set_title("")
    for overlay_arch in overlay_arches:
        layer = negative_layers_by_arch[overlay_arch]
        boundary_xy = layer["boundary"]
        ax.plot(boundary_xy[:, 0], boundary_xy[:, 1],
                color=config["colors"]["boundary"],
                linewidth=config["boundary_linewidth"],
                linestyle=config["boundary_linestyle"], zorder=2)
        reference_xy = layer["reference"]
        ax.scatter(reference_xy[:, 0], reference_xy[:, 1],
                   marker=config["reference_marker"],
                   facecolors=config["reference_facecolor"],
                   edgecolors=config["colors"]["reference"],
                   s=config["sizes"]["reference"], linewidths=config["reference_linewidth"],
                   alpha=config["alpha_points"]["reference"], zorder=4)
        plot_negative_mean = (
            config["show_nonmatching_negative_means"] or overlay_arch == arch
        )
        layer["negative_mean_plotted"] = bool(plot_negative_mean)
        if plot_negative_mean:
            negative_mean = layer["negative_mean"]
            ax.scatter(negative_mean[0], negative_mean[1], marker="s",
                       color=config["colors"]["negative"], s=config["sizes"]["negative"],
                       alpha=config["alpha_points"]["negative"], edgecolors="none",
                       linewidths=0, zorder=5)
    for method in config["methods"]:
        xy = positive_means[method]
        group = POSITIVE_GROUPS[method]
        ax.scatter(
            xy[0],
            xy[1],
            marker=config["markers"]["positive"][method],
            color=config["colors"]["positive_groups"][group],
            s=config["sizes"]["positive"] * METHOD_SIZE_SCALE[method],
            alpha=config["alpha_points"]["positive"],
            edgecolors="none",
            linewidths=0,
            zorder=6,
        )
    ax.scatter(victim_xy[0], victim_xy[1], marker="*",
               color=config["colors"]["victim"], s=config["sizes"]["victim"],
               edgecolors="none", linewidths=0, zorder=7)
    all_xy = np.vstack([
        *[
            component
            for layer in negative_layers_by_arch.values()
            for component in (
                [layer["boundary"], layer["reference"]]
                + ([layer["negative_mean"][None]]
                   if layer["negative_mean_plotted"] else [])
            )
        ],
        victim_xy[None],
        *[xy[None] for xy in positive_means.values()],
    ])
    span = np.ptp(all_xy, axis=0)
    padding = np.maximum(span * config["margins"]["data"], [1e-3, 1e-3])
    ax.set_xlim(all_xy[:, 0].min() - padding[0], all_xy[:, 0].max() + padding[0])
    ax.set_ylim(all_xy[:, 1].min() - padding[1], all_xy[:, 1].max() + padding[1])
    limits = config["axis_limits"].get(f"{dataset}_{arch}", {})
    if limits.get("x") is not None:
        ax.set_xlim(limits["x"])
    if limits.get("y") is not None:
        ax.set_ylim(limits["y"])
    if config["tick_count"] is not None:
        ax.xaxis.set_major_locator(FixedLocator(
            _spaced_ticks(ax.get_xlim(), config["tick_count"],
                          config["tick_margin_fraction"])))
        ax.yaxis.set_major_locator(FixedLocator(
            _spaced_ticks(ax.get_ylim(), config["tick_count"],
                          config["tick_margin_fraction"])))
    ax.set_xlabel("I (X;T)", fontsize=config["font_sizes"]["xlabel"], fontfamily=font)
    ax.set_ylabel("I (T;Y)", fontsize=config["font_sizes"]["ylabel"], fontfamily=font)
    ax.tick_params(axis="both", labelsize=config["font_sizes"]["ticks"],
                   length=3, width=.6)
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.grid(alpha=config["grid_alpha"], linewidth=config["grid_linewidth"])
    for tick in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        tick.set_fontfamily(font)
    handles = [
        Line2D([], [], marker="s", linestyle="None", color=config["colors"]["victim"],
               markersize=7, label="Victim model"),
        Line2D([], [], marker="^", linestyle="None", color=config["colors"]["negative"],
               markersize=7, label="negative suspects (mean)"),
        Line2D([], [], marker=config["reference_marker"], linestyle="None",
               markerfacecolor=config["reference_facecolor"],
               markeredgecolor=config["colors"]["reference"], markersize=7,
               markeredgewidth=config["reference_linewidth"], label="reference models"),
        Line2D([], [], linestyle=config["boundary_linestyle"],
               color=config["colors"]["boundary"], linewidth=config["boundary_linewidth"],
               label=("H0 boundaries" if config["show_all_arch_negative_overlays"]
                      else "H0 boundary")),
    ]
    handles += [Line2D(
                        [], [],
                        marker=config["markers"]["positive"][method],
                        linestyle="None",
                        color=config["colors"]["positive_groups"][POSITIVE_GROUPS[method]],
                        markersize=7,
                        label=f"{method} ({POSITIVE_GROUPS[method]})"
                    )
                    for method in config["methods"]
                ]
    if config["show_legend"]:
        ax.legend(handles=handles, prop=FontProperties(family=font,
                  size=config["font_sizes"]["legend"]), **config["legend"])
    fig.canvas.draw()
    require(np.isclose(ax.bbox.width, ax.bbox.height, rtol=0, atol=2),
            f"{dataset} {arch}: plot area is not square")
    if config["tick_count"] is not None:
        require(len(ax.get_xticks()) == config["tick_count"]
                and len(ax.get_yticks()) == config["tick_count"],
                f"{dataset} {arch}: unexpected major tick count")
    details = {
        "dataset": dataset, "architecture": arch, "h0_scenario": scenario,
        "k": config["k"], "round_id": round_id, "alpha": config["alpha"],
        "h0_boundaries": [
            {
                "architecture": overlay_arch,
                "h0_scenario": audit.ARCHES[dataset][overlay_arch]["h0"],
                "round_id": boundary_rounds[overlay_arch],
                "display_scale": negative_layers_by_arch[overlay_arch]["boundary_display_scale"],
            }
            for overlay_arch in overlay_arches
        ],
        "negative_layers": {
            overlay_arch: {
                "h0_scenario": layer["scenario"],
                "round_id": layer["round_id"],
                "negative_mean": layer["negative_mean"].tolist(),
                "negative_mean_plotted": layer["negative_mean_plotted"],
                "boundary_display_scale": layer["boundary_display_scale"],
                "reference_count": len(layer["reference"]),
                "negative_mean_p_F": float(
                    layer["null"].p_values([layer["negative_mean"]])[0]),
            }
            for overlay_arch, layer in negative_layers_by_arch.items()
        },
        "victim": victim_xy.tolist(),
        "negative_mean": current_negative_layer["negative_mean"].tolist(),
        "reference_count": len(current_negative_layer["reference"]),
        "x_ticks": ax.get_xticks().tolist(), "y_ticks": ax.get_yticks().tolist(),
        "positive_means": {key: xy.tolist() for key, xy in positive_means.items()},
        "mean_point_p_F": {
            "negative": float(null.p_values([current_negative_layer["negative_mean"]])[0]),
            **{key: float(null.p_values([xy])[0]) for key, xy in positive_means.items()},
        },
    }
    return fig, details


def render(config=None):
    config = deepcopy(default_config() if config is None else config)
    require(config["k"] in audit.K_VALUES and config["k"] == 15,
            "This first layout is configured for k=15")
    require(config["bins"] == 50 and config["in_size"] == 25000 and config["alpha"] == .01,
            "Selected-round results require bins=50, in_size=25000, alpha=.01")
    require(np.isclose(*config["figsize"]), "Figure canvas must be square")
    require(config["datasets"] and config["architectures"] and config["methods"],
            "Choose at least one dataset, architecture and method")
    require(set(config["datasets"]) <= set(audit.DATASETS), "Unknown dataset")
    require(set(config["architectures"]) <= set(ALL_ARCHITECTURES), "Unknown architecture")
    require(set(config["methods"]) <= set(audit.METHODS), "Unknown method")
    require(isinstance(config["show_all_arch_negative_overlays"], (bool, np.bool_)),
            "show_all_arch_negative_overlays must be boolean")
    require(isinstance(config["show_nonmatching_negative_means"], (bool, np.bool_)),
            "show_nonmatching_negative_means must be boolean")
    require(set(config["boundary_display_scales"]) == set(ALL_ARCHITECTURES),
            "boundary_display_scales must define RN18, VGG16 and DeiT exactly")
    require(all(np.isfinite(float(scale)) and float(scale) > 0
                for scale in config["boundary_display_scales"].values()),
            "Every boundary display scale must be finite and positive")
    selection_path = Path(config["selection_csv"])
    selections = {(row["dataset"], row["architecture"], int(row["k_ref"])): row
                  for row in audit.read_csv(selection_path)}
    require(len(selections) == 36, "Selected-round table must have 36 unique cells")
    table_rows = {key: audit.read_csv(path) for key, path in audit.TABLES.items()}
    victims = pd.read_csv(ROOT / "saved_logs/vanilla/MI_master_table_victim.csv")
    positive_models = {
        dataset: pd.read_csv(Path(config["result_root"]) / dataset / "positives/per_model.csv")
        for dataset in config["datasets"]}
    source_paths = [selection_path, ROOT / "saved_logs/vanilla/MI_master_table_victim.csv",
                    *audit.TABLES.values(),
                    *[Path(config["result_root"]) / dataset / role / name
                      for dataset in config["datasets"] for role in ("negatives", "positives")
                      for name in ("per_model.csv", "per_split.csv", "run_metadata.json")]]
    source_hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_paths}
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    output_paths, details, figures = [], [], []
    for dataset in config["datasets"]:
        for arch in config["architectures"]:
            fig, detail = _plot_one(dataset, arch, config, table_rows,
                                    positive_models[dataset], victims, selections)
            stem = f"{dataset}_{arch}_same_arch_mean_k{config['k']}"
            if config["save_pdf"]:
                pdf_path = out / f"{stem}.pdf"
                fixed_h0.save_square_figure(fig, pdf_path)
                output_paths.append(pdf_path)
            if config["save_png"]:
                png_path = out / f"{stem}.png"
                fixed_h0.save_square_figure(fig, png_path, dpi=config["png_dpi"])
                output_paths.append(png_path)
            details.append(detail)
            if config["show_inline"]:
                figures.append((dataset, arch, fig))
            else:
                plt.close(fig)
            layer_count = len(detail["negative_layers"])
            plotted_negative_count = sum(
                layer["negative_mean_plotted"]
                for layer in detail["negative_layers"].values())
            print(f"{stem}: round={detail['round_id']}, {layer_count} negative layers, "
                  f"{len(detail['positive_means'])} positive means, "
                  f"{plotted_negative_count} plotted negative means, "
                  f"{sum(layer['reference_count'] for layer in detail['negative_layers'].values())} "
                  "references, 1 victim")
    require(all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
                for path, digest in source_hashes.items()), "A source file changed while plotting")
    manifest = {
        "description": "One mean per positive/evaluated-negative group; every selected H0 reference point shown",
        "negative_overlay_mode": ("all_architectures_same_dataset"
                                  if config["show_all_arch_negative_overlays"]
                                  else "figure_architecture_only"),
        "negative_mean_mode": ("all_h0_layers"
                               if config["show_nonmatching_negative_means"]
                               else "figure_architecture_only"),
        "boundary_display_scales": {
            arch: float(scale)
            for arch, scale in config["boundary_display_scales"].items()
        },
        "k": config["k"], "alpha": config["alpha"], "bins": config["bins"],
        "in_size": config["in_size"], "selection_csv": str(selection_path.resolve()),
        "source_hashes": source_hashes, "figures": details,
        "note": "Synthetic MI seeds are descriptive, not independent trained-model evidence. "
                "F tests and p-values use the exact predictive boundary; rendered boundary "
                "curves may be scaled about their fitted mu for display only, as recorded in "
                "boundary_display_scales.",
    }
    manifest_path = out / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    return output_paths, details, figures


if __name__ == "__main__":
    render()
