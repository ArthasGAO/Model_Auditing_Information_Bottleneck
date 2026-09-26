"""Publication-size k-number sweep panels for 2x3 LaTeX grids.

Each exported panel fixes the measured operating point at N=25,000 and
bin=50, then varies the number of reference models over
``k = {5, 10, 15, 20, 25, 30}``.  One panel corresponds to one dataset and
suspect architecture and contains the four positive forms FT-AL, P-20%, DKD,
and Knockoff.  P-value and Hotelling T^2 panels are exported independently.

The drawing parameters intentionally match the latest sample-rate/bin
publication framework: identical physical canvas, typography, line/marker
styles, grids, threshold legend entry, and compact fixed margins.  All plotted
values come from the original raw-sweep summary; no synthetic values are used.
"""

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, MaxNLocator, NullLocator

import plot_unknown_arch_raw_sweeps_sample as base
import plot_unknown_arch_raw_sweeps_sample_publication_2x3 as publication
from plot_unknown_arch_sweeps_shared_tight import (
    DEFAULT_OUTPUT_ROOT, SHARED_PAD_INCHES, prepare_shared_export,
)


ROOT = Path(__file__).resolve().parent
RAW_RESULT_ROOT = (
    ROOT
    / "saved_logs"
    / "vanilla"
    / "Hypo_Test_UnknownArch_RawMI_Sweeps_BestFPR_PerK"
)
DENSE_RATE_ROOT = (
    ROOT
    / "saved_logs"
    / "vanilla"
    / "Hypo_Test_UnknownArch_RawMI_SyntheticSampleRates_BestFPR_PerK"
)
OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "reference_pool"

METRICS_TO_PLOT = ("pvalue", "t2")
K_VALUES = (5, 10, 15, 20, 25, 30)
FIXED_OPERATING_POINT = "N25000_B50"
FIXED_SAMPLE_SIZE = 25000
FIXED_SAMPLE_RATE = 1.0
FIXED_BINS = 50
PNG_DPI = publication.PNG_DPI
SHOW_LEGEND = publication.SHOW_LEGEND

# Reuse the same square axes geometry as the sample-rate and bin sweeps.
PANEL_MARGINS_BY_METRIC = publication.PANEL_MARGINS_BY_METRIC
AXIS_LABEL_PAD = 2.0

# Per-panel placements were chosen after inspecting the rendered PDFs.  Every
# p-value curve decreases strongly with k, leaving a clean wedge at the upper
# right between the alpha threshold and the measured trajectories.  Small
# vertical changes keep each box clear of its closest curve rather than using
# one global location for all six panels.
P_VALUE_LEGEND_LAYOUT = {
    ("CF10", "RN18"): {"loc": "upper right", "bbox": (1.000, 0.900)},
    ("CF10", "VGG16"): {"loc": "upper right", "bbox": (1.000, 0.910)},
    ("CF10", "DeiT"): {"loc": "upper right", "bbox": (1.000, 0.900)},
    ("CF100", "RN18"): {"loc": "upper right", "bbox": (1.000, 0.920)},
    ("CF100", "VGG16"): {"loc": "upper right", "bbox": (1.000, 0.920)},
    ("CF100", "DeiT"): {"loc": "upper right", "bbox": (1.000, 0.920)},
}

T2_LEGEND_LAYOUT = {
    (dataset, architecture): {"loc": "upper center", "bbox": (0.500, 0.995)}
    for dataset in base.DATASETS_TO_PLOT
    for architecture in base.ARCHITECTURES_TO_PLOT
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def values(rows, key):
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def load_case_form_map(dataset):
    """Load the audited case-id labels without using dense synthetic values."""
    path = DENSE_RATE_ROOT / dataset / "positives" / "summary_sample_rate_dense.csv"
    require(path.is_file(), f"Missing case mapping table: {path}")
    mapping_sets = defaultdict(set)
    for row in read_csv(path):
        mapping_sets[row["case_id"]].add(row["positive_form"])
    ambiguous = {
        case_id: sorted(forms)
        for case_id, forms in mapping_sets.items()
        if len(forms) != 1
    }
    require(not ambiguous, f"Ambiguous positive-form mappings: {ambiguous}")
    mapping = {case_id: next(iter(forms)) for case_id, forms in mapping_sets.items()}
    require(
        set(mapping.values()) == set(base.POSITIVE_FORMS),
        f"Positive-form mapping mismatch for {dataset}: {set(mapping.values())}",
    )
    return mapping, path


def load_k_scenarios():
    """Return six dataset/architecture scenarios with four measured k curves."""
    grouped = defaultdict(lambda: defaultdict(list))
    sources = {}

    for dataset in base.DATASETS_TO_PLOT:
        case_form_map, mapping_path = load_case_form_map(dataset)
        summary_path = RAW_RESULT_ROOT / dataset / "positives" / "summary.csv"
        require(summary_path.is_file(), f"Missing raw sweep summary: {summary_path}")
        sources[dataset] = {
            "raw_k_summary": {
                "path": str(summary_path.resolve()),
                "sha256": sha256(summary_path),
            },
            "case_form_mapping": {
                "path": str(mapping_path.resolve()),
                "sha256": sha256(mapping_path),
                "usage": "case_id to positive_form only; synthetic rows are not plotted",
            },
        }

        for row in read_csv(summary_path):
            if row["sweep_axis"] != "anchor":
                continue
            if row["operating_point_id"] != FIXED_OPERATING_POINT:
                continue
            if row["in_sample_sweep"] != "True" or row["in_bin_sweep"] != "True":
                continue
            if int(row["in_size"]) != FIXED_SAMPLE_SIZE:
                continue
            if not np.isclose(float(row["in_size_rate"]), FIXED_SAMPLE_RATE):
                continue
            if int(row["bins"]) != FIXED_BINS:
                continue
            k = int(row["k_ref"])
            if k not in K_VALUES:
                continue
            case_id = row["case_id"]
            if case_id not in case_form_map:
                continue
            architecture = row["suspect_architecture"]
            if architecture not in base.ARCHITECTURES_TO_PLOT:
                continue
            require(row["truth"] == "positive", f"Unexpected truth label: {case_id}")
            positive_form = case_form_map[case_id]
            grouped[(dataset, architecture)][positive_form].append(row)

    expected_scenarios = {
        (dataset, architecture)
        for dataset in base.DATASETS_TO_PLOT
        for architecture in base.ARCHITECTURES_TO_PLOT
    }
    require(
        set(grouped) == expected_scenarios,
        f"K-sweep scenario coverage mismatch; "
        f"missing={sorted(expected_scenarios-set(grouped))}, "
        f"extra={sorted(set(grouped)-expected_scenarios)}",
    )

    scenarios = {}
    for scenario, forms in grouped.items():
        require(
            set(forms) == set(base.POSITIVE_FORMS),
            f"Missing positive forms for {scenario}: "
            f"{set(base.POSITIVE_FORMS)-set(forms)}",
        )
        checked = {}
        for positive_form in base.POSITIVE_FORMS:
            curve = sorted(forms[positive_form], key=lambda row: int(row["k_ref"]))
            k_values = tuple(int(row["k_ref"]) for row in curve)
            require(
                k_values == K_VALUES,
                f"Wrong k grid for {scenario}/{positive_form}: {k_values}",
            )
            require(
                len({row["case_id"] for row in curve}) == 1,
                f"Multiple case IDs for {scenario}/{positive_form}",
            )
            require(
                len({(row["case_id"], row["k_ref"]) for row in curve}) == len(K_VALUES),
                f"Duplicate k rows for {scenario}/{positive_form}",
            )
            require(
                all(row["operating_point_id"] == FIXED_OPERATING_POINT for row in curve),
                f"Operating-point drift for {scenario}/{positive_form}",
            )
            for field in ("p_F_max_mean", "T2_composite_min_mean"):
                array = values(curve, field)
                require(
                    np.isfinite(array).all(),
                    f"Nonfinite {field} for {scenario}/{positive_form}",
                )
                if field.startswith("p_"):
                    require(
                        (array > 0).all() and (array <= 1).all(),
                        f"Invalid p-value for {scenario}/{positive_form}",
                    )
                else:
                    require(
                        (array >= 0).all(),
                        f"Negative T2 for {scenario}/{positive_form}",
                    )
            checked[positive_form] = curve
        scenarios[scenario] = checked
    return scenarios, sources


def p_axis_spec(curves):
    """Return a readable panel-specific log scale using major ticks only."""
    positive_values = []
    for positive_form in base.POSITIVE_FORMS:
        positive_values.extend(values(curves[positive_form], "p_F_max_mean").tolist())
    positive_floor = min(positive_values)
    lower_exp = math.floor(math.log10(positive_floor))
    upper_exp = 0
    tick_step = max(1, math.ceil((upper_exp - lower_exp) / 5))
    lowest_tick_exp = -(abs(lower_exp) // tick_step) * tick_step
    tick_exponents = list(range(upper_exp, lowest_tick_exp - 1, -tick_step))
    return lower_exp, tick_exponents


def plot_panel(ax, curves, metric, dataset, architecture):
    if metric == "pvalue":
        mean_key = "p_F_max_mean"
        ylabel = "Auditing score"
    elif metric == "t2":
        mean_key = "T2_composite_min_mean"
        ylabel = r"Hotelling's $\mathrm{T}^2$"
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    x = np.asarray(K_VALUES, dtype=float)
    for positive_form in base.POSITIVE_FORMS:
        curve_style = base.CURVE_STYLES[positive_form]
        ax.plot(
            x,
            values(curves[positive_form], mean_key),
            color=curve_style["color"],
            marker=curve_style["marker"],
            linestyle=curve_style["linestyle"],
            markersize=publication.LINE_SIZES["marker_size"],
            linewidth=publication.LINE_SIZES["curve_width"],
            markerfacecolor=curve_style["color"],
            markeredgecolor="white",
            markeredgewidth=publication.LINE_SIZES["marker_edge_width"],
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
            linewidth=publication.LINE_SIZES["threshold_width"],
            label=rf"$\alpha={base.ALPHA:g}$",
            zorder=1,
        )
        legend_layout = P_VALUE_LEGEND_LAYOUT[(dataset, architecture)]
    else:
        # Reserve a clean strip above the largest measured value for the
        # two-row legend.  Without this headroom an upper-left legend hides
        # the k=5 point in several T2 panels.
        t2_max = max(
            float(np.max(values(curves[positive_form], mean_key)))
            for positive_form in base.POSITIVE_FORMS
        )
        ax.set_ylim(0, t2_max * 1.35)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        ax.yaxis.set_minor_locator(NullLocator())
        legend_layout = T2_LEGEND_LAYOUT[(dataset, architecture)]

    ax.set_xlim(4.0, 31.0)
    ax.set_xticks(K_VALUES, labels=K_VALUES)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(
        r"Reference pool size", #($k$)",
        fontsize=publication.FONT_SIZES["xlabel"],
        labelpad=AXIS_LABEL_PAD,
    )
    ax.set_ylabel(
        ylabel,
        fontsize=publication.FONT_SIZES["ylabel"],
        labelpad=AXIS_LABEL_PAD,
    )
    publication.style_axis(ax)
    ax.set_box_aspect(1)
    if SHOW_LEGEND:
        ax.legend(
            loc=legend_layout["loc"],
            bbox_to_anchor=legend_layout["bbox"],
            ncol=2,
            frameon=True,
            fancybox=False,
            edgecolor="#A8A8A8",
            framealpha=0.94,
            fontsize=publication.FONT_SIZES["legend"],
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
    fig, ax = plt.subplots(figsize=publication.PANEL_FIGSIZE_INCHES)
    plot_panel(ax, curves, metric, dataset, architecture)
    fig.subplots_adjust(**PANEL_MARGINS_BY_METRIC[metric])
    return fig


def save_panel(dataset, architecture, curves, metric, shared_export=None):
    if shared_export is None:
        shared_export = prepare_shared_export({"reference_pool": sys.modules[__name__]})
    fig = create_panel(dataset, architecture, curves, metric)
    savefig_kwargs = shared_export.savefig_kwargs(metric)

    subdir = OUTPUT_DIR / metric
    subdir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{dataset.lower()}_{safe_token(architecture)}_"
        f"{metric}_k_number_publication_2x3"
    )
    pdf_path = subdir / f"{stem}.pdf"
    png_path = subdir / f"{stem}.png"
    fig.savefig(
        pdf_path,
        facecolor="white",
        metadata={
            "Title": f"{dataset} {architecture} {metric} k-number sweep",
            "Subject": "Publication-size 2x3 k-number sweep panel",
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
    publication.configure_fonts()
    if shared_export is None:
        shared_export = prepare_shared_export({"reference_pool": sys.modules[__name__]})
    scenarios, sources = shared_export.data["reference_pool"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = []
    for metric in METRICS_TO_PLOT:
        for dataset in base.DATASETS_TO_PLOT:
            for architecture in base.ARCHITECTURES_TO_PLOT:
                scenario = (dataset, architecture)
                pdf_path, png_path = save_panel(
                    dataset, architecture, scenarios[scenario], metric,
                    shared_export=shared_export,
                )
                outputs.append({
                    "dataset": dataset,
                    "suspect_architecture": architecture,
                    "metric": metric,
                    "fixed_operating_point": FIXED_OPERATING_POINT,
                    "fixed_sample_size": FIXED_SAMPLE_SIZE,
                    "fixed_sample_rate": FIXED_SAMPLE_RATE,
                    "fixed_bins": FIXED_BINS,
                    "k_values": list(K_VALUES),
                    "case_ids": {
                        form: scenarios[scenario][form][0]["case_id"]
                        for form in base.POSITIVE_FORMS
                    },
                    "pdf_path": str(pdf_path.resolve()),
                    "png_path": str(png_path.resolve()),
                    "pdf_sha256": sha256(pdf_path),
                })

    metadata = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "render_script": str(Path(__file__).resolve()),
        "render_script_sha256": sha256(__file__),
        "style_source_script": str(Path(publication.__file__).resolve()),
        "style_source_script_sha256": sha256(publication.__file__),
        "source_tables": sources,
        "selection": {
            "datasets": list(base.DATASETS_TO_PLOT),
            "architectures": list(base.ARCHITECTURES_TO_PLOT),
            "metrics": list(METRICS_TO_PLOT),
            "positive_forms": list(base.POSITIVE_FORMS),
            "k_values": list(K_VALUES),
            "sweep_axis": "anchor",
            "fixed_operating_point": FIXED_OPERATING_POINT,
            "fixed_sample_size": FIXED_SAMPLE_SIZE,
            "fixed_sample_rate": FIXED_SAMPLE_RATE,
            "fixed_bins": FIXED_BINS,
            "synthetic_values_used": False,
        },
        "publication_layout": {
            "grid": "2x3 per metric",
            "row_order": [
                base.DISPLAY_DATASETS[key] for key in base.DATASETS_TO_PLOT
            ],
            "column_order": [
                base.DISPLAY_ARCHITECTURES[key]
                for key in base.ARCHITECTURES_TO_PLOT
            ],
            "panel_figsize_inches": list(publication.PANEL_FIGSIZE_INCHES),
            "latex_panel_width": publication.LATEX_PANEL_WIDTH,
            "font_sizes_points": publication.FONT_SIZES,
            "line_sizes_points": publication.LINE_SIZES,
            "panel_margins_by_metric": PANEL_MARGINS_BY_METRIC,
            "axes_box_aspect": 1.0,
            "axis_label_pad": AXIS_LABEL_PAD,
            "show_legend": SHOW_LEGEND,
            "pvalue_legend_layout": {
                f"{dataset}_{architecture}": layout
                for (dataset, architecture), layout
                in P_VALUE_LEGEND_LAYOUT.items()
            },
            "t2_legend_layout": {
                f"{dataset}_{architecture}": layout
                for (dataset, architecture), layout
                in T2_LEGEND_LAYOUT.items()
            },
            "bbox_inches": "shared_tight",
            "pad_inches": SHARED_PAD_INCHES,
            "shared_crop": shared_export.metadata,
            "fixed_canvas": False,
            "png_dpi": PNG_DPI,
        },
        "counts": {
            "scenarios": len(scenarios),
            "pdf_files": len(outputs),
            "png_files": len(outputs),
            "curves_per_panel": len(base.POSITIVE_FORMS),
            "points_per_curve": len(K_VALUES),
        },
        "outputs": outputs,
    }
    metadata_path = OUTPUT_DIR / "publication_manifest.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(OUTPUT_DIR)
    print(json.dumps(metadata["counts"], sort_keys=True))
    for output in outputs:
        print(output["pdf_path"])


if __name__ == "__main__":
    main()
