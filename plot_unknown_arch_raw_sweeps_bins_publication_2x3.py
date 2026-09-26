"""Publication-size bin-sweep panels for 2x3 LaTeX grids.

The plotting style is inherited from the latest sample-rate publication
script so the rate and bin sweeps use the same physical panel size, fonts,
line widths, markers, fixed canvas, grids, and legend placement.  Only the
sweep axis and its data source differ.

Bin-sweep values are read from the original hypothesis-test summary with
sample size fixed at 100% and k=15.  The dense sample-rate table is used only
as an audited case_id-to-positive-form mapping; no synthetic values enter the
bin curves.
"""

import csv
import hashlib
import json
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
OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "bins"

METRICS_TO_PLOT = ("pvalue", "t2")
BIN_VALUES = (5, 10, 15, 20, 30, 50, 75, 100, 150, 200)
# Tick locations are a clean, uniformly spaced visual guide; they do not need
# to coincide with every measured bin value.  The observations themselves are
# still plotted at their true bin counts from 5 through 200.
BIN_TICK_VALUES = (50, 100, 150, 200)
BIN_X_LIMITS = (2.5, 202.5)
FIXED_SAMPLE_RATE = 1.0
PNG_DPI = publication.PNG_DPI
SHOW_LEGEND = publication.SHOW_LEGEND

# Reuse the same square axes geometry as the sample-rate and k sweeps.
PANEL_MARGINS_BY_METRIC = publication.PANEL_MARGINS_BY_METRIC
AXIS_LABEL_PAD = 2.0


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
    require(set(mapping.values()) == set(base.POSITIVE_FORMS),
            f"Positive-form mapping mismatch for {dataset}: {set(mapping.values())}")
    return mapping, path


def load_bin_scenarios():
    grouped = defaultdict(lambda: defaultdict(list))
    sources = {}

    for dataset in base.DATASETS_TO_PLOT:
        case_form_map, mapping_path = load_case_form_map(dataset)
        summary_path = RAW_RESULT_ROOT / dataset / "positives" / "summary.csv"
        require(summary_path.is_file(), f"Missing raw sweep summary: {summary_path}")
        sources[dataset] = {
            "raw_bin_summary": {
                "path": str(summary_path.resolve()),
                "sha256": sha256(summary_path),
            },
            "case_form_mapping": {
                "path": str(mapping_path.resolve()),
                "sha256": sha256(mapping_path),
                "usage": "case_id to positive_form only",
            },
        }

        for row in read_csv(summary_path):
            is_measured_bin_sweep = row["sweep_axis"] == "bin_size"
            is_shared_default_anchor = (
                row["sweep_axis"] == "anchor"
                and row["in_bin_sweep"] == "True"
                and row["in_sample_sweep"] == "True"
                and int(row["bins"]) == 50
                and row["operating_point_id"] == "N25000_B50"
            )
            if not (is_measured_bin_sweep or is_shared_default_anchor):
                continue
            if int(row["k_ref"]) not in base.K_VALUES_TO_PLOT:
                continue
            if not np.isclose(float(row["in_size_rate"]), FIXED_SAMPLE_RATE):
                continue
            case_id = row["case_id"]
            if case_id not in case_form_map:
                continue
            architecture = row["suspect_architecture"]
            if architecture not in base.ARCHITECTURES_TO_PLOT:
                continue
            require(row["truth"] == "positive",
                    f"Unexpected truth label for bin curve: {case_id}")
            positive_form = case_form_map[case_id]
            grouped[(dataset, architecture, int(row["k_ref"]))][positive_form].append(row)

    expected_scenarios = {
        (dataset, architecture, k)
        for dataset in base.DATASETS_TO_PLOT
        for architecture in base.ARCHITECTURES_TO_PLOT
        for k in base.K_VALUES_TO_PLOT
    }
    require(set(grouped) == expected_scenarios,
            f"Bin scenario coverage mismatch; missing={sorted(expected_scenarios-set(grouped))}, "
            f"extra={sorted(set(grouped)-expected_scenarios)}")

    scenarios = {}
    for scenario, forms in grouped.items():
        require(set(forms) == set(base.POSITIVE_FORMS),
                f"Missing positive forms for {scenario}: "
                f"{set(base.POSITIVE_FORMS)-set(forms)}")
        checked = {}
        for positive_form in base.POSITIVE_FORMS:
            curve = sorted(forms[positive_form], key=lambda row: int(row["bins"]))
            bins = tuple(int(row["bins"]) for row in curve)
            require(bins == BIN_VALUES,
                    f"Wrong bin grid for {scenario}/{positive_form}: {bins}")
            require(len({row["case_id"] for row in curve}) == 1,
                    f"Multiple case IDs for {scenario}/{positive_form}")
            require(all(np.isclose(float(row["in_size_rate"]), FIXED_SAMPLE_RATE)
                        for row in curve),
                    f"Non-100% sample size in {scenario}/{positive_form}")
            anchor_rows = [
                row for row in curve
                if int(row["bins"]) == 50 and row["sweep_axis"] == "anchor"
            ]
            require(len(anchor_rows) == 1,
                    f"Missing/duplicate shared bin=50 anchor for "
                    f"{scenario}/{positive_form}")
            require(all(row["sweep_axis"] == "bin_size"
                        for row in curve if int(row["bins"]) != 50),
                    f"Unexpected non-bin sweep row for {scenario}/{positive_form}")
            for field in ("p_F_max_mean", "T2_composite_min_mean"):
                array = values(curve, field)
                require(np.isfinite(array).all(),
                        f"Nonfinite {field} for {scenario}/{positive_form}")
                if field.startswith("p_"):
                    require((array > 0).all() and (array <= 1).all(),
                            f"Invalid p-value for {scenario}/{positive_form}")
                else:
                    require((array >= 0).all(),
                            f"Negative T2 for {scenario}/{positive_form}")
            checked[positive_form] = curve
        scenarios[scenario] = checked
    return scenarios, sources


def p_axis_spec(curves):
    positive_values = []
    for positive_form in base.POSITIVE_FORMS:
        positive_values.extend(values(curves[positive_form], "p_F_max_mean").tolist())
    positive_floor = min(value for value in positive_values if value > 0)
    lower_exp = int(np.floor(np.log10(positive_floor)))
    upper_exp = 0
    tick_step = max(1, int(np.ceil((upper_exp - lower_exp) / 5)))
    lowest_tick_exp = -(abs(lower_exp) // tick_step) * tick_step
    tick_exponents = list(range(upper_exp, lowest_tick_exp - 1, -tick_step))
    return lower_exp, tick_exponents


def plot_panel(ax, curves, metric, dataset):
    if metric == "pvalue":
        mean_key = "p_F_max_mean"
        ylabel = "Auditing score"
    elif metric == "t2":
        mean_key = "T2_composite_min_mean"
        ylabel = r"Hotelling's $\mathrm{T}^2$"
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Use the measured bin counts directly.  A log x-axis preserves their
    # multiplicative spacing without inventing intermediate observations.
    x = np.asarray(BIN_VALUES, dtype=float)
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
        legend_loc = "center right"
        legend_bbox = publication.P_VALUE_LEGEND_BBOX[dataset]
    else:
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        ax.yaxis.set_minor_locator(NullLocator())
        legend_loc = "upper left"
        legend_bbox = None

    ax.set_xscale("linear")
    ax.set_xlim(*BIN_X_LIMITS)
    ax.set_xticks(BIN_TICK_VALUES, labels=BIN_TICK_VALUES)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(
        "Number of bins",
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
            loc=legend_loc,
            bbox_to_anchor=legend_bbox,
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
    plot_panel(ax, curves, metric, dataset)
    fig.subplots_adjust(**PANEL_MARGINS_BY_METRIC[metric])
    return fig


def save_panel(dataset, architecture, k, curves, metric, shared_export=None):
    if shared_export is None:
        shared_export = prepare_shared_export({"bins": sys.modules[__name__]})
    fig = create_panel(dataset, architecture, curves, metric)
    savefig_kwargs = shared_export.savefig_kwargs(metric)

    subdir = OUTPUT_DIR / f"k{k}" / metric
    subdir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{dataset.lower()}_{safe_token(architecture)}_k{k}_"
        f"{metric}_bins_publication_2x3"
    )
    pdf_path = subdir / f"{stem}.pdf"
    png_path = subdir / f"{stem}.png"
    fig.savefig(
        pdf_path,
        facecolor="white",
        metadata={
            "Title": f"{dataset} {architecture} k={k} {metric} bin sweep",
            "Subject": "Publication-size 2x3 bin-sweep panel",
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
        shared_export = prepare_shared_export({"bins": sys.modules[__name__]})
    scenarios, sources = shared_export.data["bins"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = []
    for metric in METRICS_TO_PLOT:
        for dataset in base.DATASETS_TO_PLOT:
            for architecture in base.ARCHITECTURES_TO_PLOT:
                for k in base.K_VALUES_TO_PLOT:
                    scenario = (dataset, architecture, k)
                    pdf_path, png_path = save_panel(
                        dataset, architecture, k, scenarios[scenario], metric,
                        shared_export=shared_export,
                    )
                    outputs.append({
                        "dataset": dataset,
                        "suspect_architecture": architecture,
                        "k_ref": k,
                        "metric": metric,
                        "fixed_sample_rate": FIXED_SAMPLE_RATE,
                        "bin_values": list(BIN_VALUES),
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
            "k_values": list(base.K_VALUES_TO_PLOT),
            "metrics": list(METRICS_TO_PLOT),
            "positive_forms": list(base.POSITIVE_FORMS),
            "sweep_axis": "bin_size plus shared default anchor",
            "shared_default_anchor": {
                "sweep_axis": "anchor",
                "operating_point_id": "N25000_B50",
                "bins": 50,
                "in_size_rate": FIXED_SAMPLE_RATE,
                "in_sample_sweep": True,
                "in_bin_sweep": True,
            },
            "x_axis_scale": "linear",
            "x_axis_limits": list(BIN_X_LIMITS),
            "x_tick_values": list(BIN_TICK_VALUES),
            "bin_values": list(BIN_VALUES),
            "fixed_sample_rate": FIXED_SAMPLE_RATE,
        },
        "publication_layout": {
            "grid": "2x3 per metric",
            "panel_figsize_inches": list(publication.PANEL_FIGSIZE_INCHES),
            "latex_panel_width": publication.LATEX_PANEL_WIDTH,
            "font_sizes_points": publication.FONT_SIZES,
            "line_sizes_points": publication.LINE_SIZES,
            "panel_margins_by_metric": PANEL_MARGINS_BY_METRIC,
            "axes_box_aspect": 1.0,
            "axis_label_pad": AXIS_LABEL_PAD,
            "show_legend": SHOW_LEGEND,
            "pvalue_legend_bbox": publication.P_VALUE_LEGEND_BBOX,
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
            "points_per_curve": len(BIN_VALUES),
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
