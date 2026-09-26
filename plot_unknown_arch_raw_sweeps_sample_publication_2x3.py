"""Publication-size sample-rate panels for a 2x3 LaTeX grid.

This script is intentionally separate from
``plot_unknown_arch_raw_sweeps_sample.py``.  It reuses that script's data
loading, scenario validation, curve definitions and case selection, while
rendering each panel close to its final physical size in a full-width paper
figure.  This prevents LaTeX from shrinking a large square source figure and
making every label, marker and legend entry unreadably small.

The default export contains the six p-value panels at k=15, ordered as two
dataset rows (CIFAR-10, CIFAR-100) by three suspect-architecture columns
(ResNet-18, VGG-16, DeiT).  Set METRICS_TO_PLOT to ("pvalue", "t2") if a
second 2x3 grid of Hotelling T^2 panels is needed later.
"""

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, MaxNLocator, NullLocator

import plot_unknown_arch_raw_sweeps_sample as base
from plot_unknown_arch_sweeps_shared_tight import (
    DEFAULT_OUTPUT_ROOT, SHARED_PAD_INCHES, prepare_shared_export,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "sample_rate"

# Editable publication settings.
METRICS_TO_PLOT = ("pvalue","t2")
PANEL_FIGSIZE_INCHES = (2.25, 2.25)
LATEX_PANEL_WIDTH = r"0.31\linewidth"
PNG_DPI = 400

# Publication layout switch.  Keep panel legends disabled when the LaTeX
# figure supplies one shared legend on the right of the complete 2x3 grid.
SHOW_LEGEND = False

FONT_SIZES = {
    "xlabel": 12,
    "ylabel": 12,
    "ticks": 8.0,
    "legend": 6.4,
}

LINE_SIZES = {
    "curve_width": 1.2,
    "marker_size": 3.0,
    "marker_edge_width": 0.25,
    "threshold_width": 1.15,
    "grid_width": 0.42,
    "spine_width": 0.72,
    "tick_length": 2.4,
    "tick_width": 0.65,
}

# Fixed square axes boxes for every sweep framework.  Both metric layouts have
# a horizontal and vertical span of exactly 0.735 on the 2.25 x 2.25 canvas.
# The enlarged left margin accommodates 12-point y-axis labels without
# clipping; the extra top margin also protects the uppermost tick label.
PANEL_MARGINS_BY_METRIC = {
    "pvalue": {
        "left": 0.240,
        "right": 0.975,
        "bottom": 0.190,
        "top": 0.925,
    },
    "t2": {
        "left": 0.250,
        "right": 0.985,
        "bottom": 0.190,
        "top": 0.925,
    },
}

P_VALUE_LEGEND_BBOX = {
    "CF10": (0.97, 0.70),
    "CF100": (0.97, 0.765),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_fonts():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [base.FONT_FAMILY, "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def p_axis_spec(curves):
    """Return a natural, uniformly spaced log tick grid for one panel."""
    positive_values = []
    for positive_form in base.POSITIVE_FORMS:
        positive_values.extend(
            base.values(curves[positive_form], "p_F_max_mean").tolist()
        )
    positive_floor = min(value for value in positive_values if value > 0)
    lower_exp = math.floor(math.log10(positive_floor))
    upper_exp = 0
    tick_step = max(1, math.ceil((upper_exp - lower_exp) / 5))
    lowest_tick_exp = -(abs(lower_exp) // tick_step) * tick_step
    tick_exponents = list(range(upper_exp, lowest_tick_exp - 1, -tick_step))
    return lower_exp, tick_exponents


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(
        True,
        which="major",
        color=base.GRID,
        linestyle=(0, (2, 2)),
        linewidth=LINE_SIZES["grid_width"],
        alpha=0.8,
    )
    ax.tick_params(
        direction="out",
        length=LINE_SIZES["tick_length"],
        width=LINE_SIZES["tick_width"],
        colors="black",
        labelsize=FONT_SIZES["ticks"],
        pad=1.8,
    )
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(LINE_SIZES["spine_width"])


def plot_panel(ax, curves, metric, dataset):
    if metric == "pvalue":
        mean_key = "p_F_max_mean"
        ylabel = "Auditing score"
    elif metric == "t2":
        mean_key = "T2_composite_min_mean"
        ylabel = r"Hotelling's $\mathrm{T}^2$"
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    x = np.asarray(base.TARGET_RATES, dtype=float)
    for positive_form in base.POSITIVE_FORMS:
        rows = curves[positive_form]
        curve_style = base.CURVE_STYLES[positive_form]
        ax.plot(
            x,
            base.values(rows, mean_key),
            color=curve_style["color"],
            marker=curve_style["marker"],
            linestyle=curve_style["linestyle"],
            markersize=LINE_SIZES["marker_size"],
            linewidth=LINE_SIZES["curve_width"],
            markerfacecolor=curve_style["color"],
            markeredgecolor="white",
            markeredgewidth=LINE_SIZES["marker_edge_width"],
            label=positive_form,
            zorder=2,
        )

    if metric == "pvalue":
        lower_exp, tick_exponents = p_axis_spec(curves)
        ax.set_yscale("log")
        ax.set_ylim(10.0 ** lower_exp / 2.0, 2.0)
        ax.set_yticks([10.0 ** exponent for exponent in tick_exponents])
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.axhline(
            base.ALPHA,
            color=base.THRESHOLD,
            linestyle="-",
            linewidth=LINE_SIZES["threshold_width"],
            label=rf"$\alpha={base.ALPHA:g}$",
            zorder=1,
        )
        legend_loc = "center right"
        legend_bbox = P_VALUE_LEGEND_BBOX[dataset]
    else:
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        ax.yaxis.set_minor_locator(NullLocator())
        legend_loc = "upper left"
        legend_bbox = None

    ax.set_xlim(2.5, 102.5)
    ax.set_xticks((25, 50, 75, 100))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(
        "Dataset usage rate (%)",
        fontsize=FONT_SIZES["xlabel"],
        labelpad=2.0,
    )
    ax.set_ylabel(ylabel, fontsize=FONT_SIZES["ylabel"], labelpad=2.0)
    style_axis(ax)
    ax.set_box_aspect(1)
    if SHOW_LEGEND:
        ax.legend(
            loc=legend_loc,
            bbox_to_anchor=legend_bbox,
            ncol=2,
            frameon=True,
            fancybox=False,
            edgecolor="#A8A8A8",
            framealpha=0.94,
            fontsize=FONT_SIZES["legend"],
            markerscale=1.0,
            columnspacing=0.55,
            handlelength=1.55,
            handletextpad=0.35,
            labelspacing=0.30,
            borderpad=0.30,
            borderaxespad=0.20,
        )


def safe_token(value):
    return value.lower().replace("-", "").replace(" ", "_")


def create_panel(dataset, architecture, curves, metric):
    """Build the same uncropped figure for measurement and final export."""
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE_INCHES)
    plot_panel(ax, curves, metric, dataset)
    fig.subplots_adjust(**PANEL_MARGINS_BY_METRIC[metric])
    return fig


def save_panel(dataset, architecture, k, curves, metric, shared_export=None):
    if shared_export is None:
        shared_export = prepare_shared_export({"sample_rate": sys.modules[__name__]})
    fig = create_panel(dataset, architecture, curves, metric)
    savefig_kwargs = shared_export.savefig_kwargs(metric)

    subdir = OUTPUT_DIR / f"k{k}" / metric
    subdir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{dataset.lower()}_{safe_token(architecture)}_k{k}_"
        f"{metric}_sample_rate_publication_2x3"
    )
    pdf_path = subdir / f"{stem}.pdf"
    png_path = subdir / f"{stem}.png"

    # Use the same padded tight union as the bin and reference-pool sweeps.
    fig.savefig(
        pdf_path,
        facecolor="white",
        metadata={
            "Title": f"{dataset} {architecture} k={k} {metric}",
            "Subject": "Publication-size 2x3 sample-rate panel",
        },
        **savefig_kwargs,
    )
    fig.savefig(
        png_path,
        dpi=PNG_DPI,
        facecolor="white",
        **savefig_kwargs,
    )
    plt.close(fig)
    return pdf_path, png_path


def main(shared_export=None):
    configure_fonts()
    if shared_export is None:
        shared_export = prepare_shared_export({"sample_rate": sys.modules[__name__]})
    scenarios, sources = shared_export.data["sample_rate"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = []
    for metric in METRICS_TO_PLOT:
        for dataset in base.DATASETS_TO_PLOT:
            for architecture in base.ARCHITECTURES_TO_PLOT:
                for k in base.K_VALUES_TO_PLOT:
                    scenario = (dataset, architecture, k)
                    base.require(scenario in scenarios, f"Missing scenario: {scenario}")
                    pdf_path, png_path = save_panel(
                        dataset,
                        architecture,
                        k,
                        scenarios[scenario],
                        metric,
                        shared_export=shared_export,
                    )
                    outputs.append({
                        "dataset": dataset,
                        "suspect_architecture": architecture,
                        "k_ref": k,
                        "metric": metric,
                        "pdf_path": str(pdf_path.resolve()),
                        "png_path": str(png_path.resolve()),
                        "pdf_sha256": sha256(pdf_path),
                    })

    metadata = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "render_script": str(Path(__file__).resolve()),
        "render_script_sha256": sha256(__file__),
        "data_and_validation_script": str(Path(base.__file__).resolve()),
        "source_tables": sources,
        "selection": {
            "datasets": list(base.DATASETS_TO_PLOT),
            "architectures": list(base.ARCHITECTURES_TO_PLOT),
            "k_values": list(base.K_VALUES_TO_PLOT),
            "metrics": list(METRICS_TO_PLOT),
            "positive_forms": list(base.POSITIVE_FORMS),
        },
        "publication_layout": {
            "grid": "2x3",
            "row_order": [base.DISPLAY_DATASETS[key]
                          for key in base.DATASETS_TO_PLOT],
            "column_order": [base.DISPLAY_ARCHITECTURES[key]
                             for key in base.ARCHITECTURES_TO_PLOT],
            "panel_figsize_inches": list(PANEL_FIGSIZE_INCHES),
            "latex_panel_width": LATEX_PANEL_WIDTH,
            "font_sizes_points": FONT_SIZES,
            "line_sizes_points": LINE_SIZES,
            "panel_margins_by_metric": PANEL_MARGINS_BY_METRIC,
            "axes_box_aspect": 1.0,
            "show_legend": SHOW_LEGEND,
            "bbox_inches": "shared_tight",
            "pad_inches": SHARED_PAD_INCHES,
            "shared_crop": shared_export.metadata,
            "fixed_canvas": False,
            "png_dpi": PNG_DPI,
        },
        "outputs": outputs,
    }
    metadata_path = OUTPUT_DIR / "publication_manifest.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(OUTPUT_DIR)
    print(json.dumps({
        "pdf_files": len(outputs),
        "png_files": len(outputs),
        "metrics": list(METRICS_TO_PLOT),
    }, sort_keys=True))
    for output in outputs:
        print(output["pdf_path"])


if __name__ == "__main__":
    sys.modules.setdefault("plot_unknown_arch_raw_sweeps_sample_publication_2x3", sys.modules[__name__])
    main()
