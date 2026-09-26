"""Generate standalone dense sample-rate plots for every measured scenario.

One scenario is one fixed dataset, suspect architecture and k. Each scenario
produces two independent figures: one for composite Hotelling T^2 and one for
the composite exact-F p-value. Every figure contains the same four positive
forms: FT-AL, P-20%, DKD and Knockoff.

The data source contains measured anchors and explicitly labelled synthetic
interpolation points on a 5%-spaced usage-rate grid. This script does not
recalculate, interpolate or otherwise alter those values.
"""

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, MaxNLocator, NullLocator


BASE = Path(__file__).resolve().parent
RESULT_ROOT = (
    BASE
    / "saved_logs/vanilla/"
    / "Hypo_Test_UnknownArch_RawMI_SyntheticSampleRates_BestFPR_PerK"
)
OUTPUT_DIR = (
    BASE
    / "saved_plots/unknown_arch_dense_sample_rates_all_scenarios_k15_draft_v2"
)

# Editable top-level selection. Use (5, 10, 15, 20, 25, 30) after the k=15
# content and visual format have been approved.
K_VALUES_TO_PLOT = (15,)
DATASETS_TO_PLOT = ("CF10", "CF100")
ARCHITECTURES_TO_PLOT = ("RN18", "VGG16", "DeiT")
POSITIVE_FORMS = ("FT-AL", "P-20%", "DKD", "Knockoff")
TARGET_RATES = tuple(range(5, 101, 5))
REAL_RATES = (5, 10, 20, 50, 75, 100)
SYNTHETIC_RATES = tuple(rate for rate in TARGET_RATES if rate not in REAL_RATES)
ALPHA = 0.01

DISPLAY_DATASETS = {"CF10": "CIFAR-10", "CF100": "CIFAR-100"}
DISPLAY_ARCHITECTURES = {"RN18": "ResNet-18", "VGG16": "VGG-16", "DeiT": "DeiT"}

CURVE_STYLES = {
    "FT-AL": {"color": "#006199", "marker": "o", "linestyle": "-"},
    "P-20%": {"color": "#F2842F", "marker": "s", "linestyle": "--"},
    "DKD": {"color": "#2A7C13", "marker": "^", "linestyle": "-."},
    "Knockoff": {"color": "#9564DD", "marker": "D", "linestyle": ":"},
}

GRID = "#C7C7C7"
THRESHOLD = "#000000"
FONT_FAMILY = "Times New Roman"
PNG_DPI = 300


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


def rate_percent(row):
    return int(round(float(row["in_size_rate"]) * 100.0))


def configure_fonts():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [FONT_FAMILY, "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_all_rows():
    rows = []
    sources = {}
    for dataset in DATASETS_TO_PLOT:
        path = RESULT_ROOT / dataset / "positives/summary_sample_rate_dense.csv"
        require(path.is_file(), f"Missing dense summary table: {path}")
        sources[dataset] = {"path": str(path.resolve()), "sha256": sha256(path)}
        dataset_rows = read_csv(path)
        require(all(row["dataset"] == dataset for row in dataset_rows),
                f"Dataset mismatch inside {path}")
        rows.extend(dataset_rows)
    return rows, sources


def build_scenarios(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        k = int(row["k_ref"])
        if k not in K_VALUES_TO_PLOT:
            continue
        dataset = row["dataset"]
        architecture = row["suspect_architecture"]
        positive_form = row["positive_form"]
        if dataset not in DATASETS_TO_PLOT or architecture not in ARCHITECTURES_TO_PLOT:
            continue
        require(positive_form in POSITIVE_FORMS,
                f"Unexpected positive form: {positive_form}")
        grouped[(dataset, architecture, k)][positive_form].append(row)

    expected_keys = {
        (dataset, architecture, k)
        for dataset in DATASETS_TO_PLOT
        for architecture in ARCHITECTURES_TO_PLOT
        for k in K_VALUES_TO_PLOT
    }
    require(set(grouped) == expected_keys,
            f"Scenario coverage mismatch; missing={sorted(expected_keys-set(grouped))}, "
            f"extra={sorted(set(grouped)-expected_keys)}")

    scenarios = {}
    for scenario, curves in grouped.items():
        require(set(curves) == set(POSITIVE_FORMS),
                f"Missing positive form in {scenario}: {set(POSITIVE_FORMS)-set(curves)}")
        checked = {}
        for positive_form in POSITIVE_FORMS:
            curve = sorted(curves[positive_form], key=rate_percent)
            rates = tuple(rate_percent(row) for row in curve)
            require(rates == TARGET_RATES,
                    f"Wrong rate grid for {scenario}/{positive_form}: {rates}")
            require(len({row["case_id"] for row in curve}) == 1,
                    f"Multiple case IDs for {scenario}/{positive_form}")
            origins = {rate_percent(row): row["data_origin"] for row in curve}
            require(all(origins[rate] == ("real" if rate in REAL_RATES else "synthetic")
                        for rate in TARGET_RATES),
                    f"Wrong real/synthetic labels for {scenario}/{positive_form}")
            for field in ("p_F_max_mean", "p_F_max_min", "p_F_max_max",
                          "T2_composite_min_mean", "T2_composite_min_min",
                          "T2_composite_min_max"):
                array = values(curve, field)
                require(np.isfinite(array).all(),
                        f"Nonfinite {field} for {scenario}/{positive_form}")
                if field.startswith("p_"):
                    require((array > 0).all() and (array <= 1).all(),
                            f"Invalid p-value in {scenario}/{positive_form}/{field}")
                else:
                    require((array >= 0).all(),
                            f"Negative T2 in {scenario}/{positive_form}/{field}")
            checked[positive_form] = curve
        scenarios[scenario] = checked
    return scenarios


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color=GRID, linestyle=(0, (2, 2)),
            linewidth=0.65, alpha=0.8)
    ax.tick_params(direction="out", length=3.0, width=0.75, colors="black",
                   labelsize=11, pad=2.5)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.9)


def plot_metric(ax, curves, metric):
    if metric == "t2":
        mean_key = "T2_composite_min_mean"
        ylabel = r"Composite Hotelling's $T^2$"
        log_y = False
    else:
        mean_key = "p_F_max_mean"
        ylabel = "Auditing score"
        log_y = True

    x = np.asarray(TARGET_RATES, dtype=float)
    p_floor_candidates = []
    for positive_form in POSITIVE_FORMS:
        rows = curves[positive_form]
        style = CURVE_STYLES[positive_form]
        mean = values(rows, mean_key)
        ax.plot(
            x,
            mean,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markersize=5,
            linewidth=1.15,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.40,
            label=positive_form,
        )
        if log_y:
            p_floor_candidates.extend(mean.tolist())

    if log_y:
        ax.set_yscale("log")
        positive_floor = min(value for value in p_floor_candidates if value > 0)
        lower_exp = math.floor(math.log10(positive_floor))
        upper_exp = 0
        tick_step = max(1, math.ceil((upper_exp - lower_exp) / 5))
        # Anchor logarithmic ticks at 10^0 and move downward with a uniform
        # exponent interval. This avoids an odd final gap such as
        # 10^0, 10^-1, 10^-4, 10^-7, ... when the data floor is 10^-13.
        lowest_tick_exp = -(abs(lower_exp) // tick_step) * tick_step
        tick_exponents = list(
            range(upper_exp, lowest_tick_exp - 1, -tick_step)
        )
        ax.set_ylim(10.0 ** lower_exp / 2.0, 2.0)
        ax.axhline(
            ALPHA,
            color=THRESHOLD,
            linestyle="-",
            linewidth=1.5,
            zorder=1,
            label=rf"$\alpha={ALPHA:g}$",
        )
        '''ax.text(0.015, ALPHA, rf"$\alpha={ALPHA:g}$  ", color=THRESHOLD,
                transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                fontsize=8.2)'''
        ax.set_yticks([10.0 ** exponent for exponent in tick_exponents])
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ax.yaxis.set_minor_locator(NullLocator())
        # All p-value panels reserve this right-middle empty region for the
        # five-entry legend (four positive forms plus the alpha threshold).
        legend_loc = "center right"
        legend_bbox = (0.985, 0.66)
    else:
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        ax.yaxis.set_minor_locator(NullLocator())
        legend_loc = "upper left"
        legend_bbox = None

    ax.set_xlim(2.5, 102.5)
    ax.set_xticks((25, 50, 75, 100))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Dataset usage rate (%)", fontsize=20, labelpad=4)
    ax.set_ylabel(ylabel, fontsize=20, labelpad=4)
    style_axis(ax)
    ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox,
              ncol=2, frameon=True, fancybox=False,
              edgecolor="#B0B0B0", framealpha=0.92, fontsize=10,
              columnspacing=0.9, handlelength=2.4, handletextpad=0.5)


def scenario_caption(dataset, architecture, k):
    return (f"{DISPLAY_DATASETS[dataset]} · "
            f"{DISPLAY_ARCHITECTURES[architecture]} suspect · k={k}")


def safe_token(value):
    return value.lower().replace("-", "").replace(" ", "_")


def save_figure(dataset, architecture, k, curves, metric):
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    plot_metric(ax, curves, metric)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.15, top=0.975)
    

    subdir = OUTPUT_DIR / f"k{k}"
    subdir.mkdir(parents=True, exist_ok=True)
    stem = (f"{dataset.lower()}_{safe_token(architecture)}_k{k}_"
            f"{metric}_sample_rate_dense")
    png = subdir / f"{stem}.png"
    pdf = subdir / f"{stem}.pdf"
    fig.savefig(png, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.04,
                facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    return png, pdf


def main():
    configure_fonts()
    rows, sources = load_all_rows()
    scenarios = build_scenarios(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []
    for (dataset, architecture, k), curves in sorted(scenarios.items()):
        case_ids = {form: curves[form][0]["case_id"] for form in POSITIVE_FORMS}
        for metric in ("t2", "pvalue"):
            png, pdf = save_figure(dataset, architecture, k, curves, metric)
            manifest.append({
                "dataset": dataset,
                "suspect_architecture": architecture,
                "k_ref": k,
                "metric": metric,
                "n_curves": len(POSITIVE_FORMS),
                "points_per_curve": len(TARGET_RATES),
                "real_rates_percent": json.dumps(REAL_RATES, separators=(",", ":")),
                "synthetic_rates_percent": json.dumps(SYNTHETIC_RATES,
                                                       separators=(",", ":")),
                "case_ids_by_positive_form": json.dumps(case_ids, sort_keys=True,
                                                         separators=(",", ":")),
                "pdf_path": str(pdf.resolve()),
                "png_path": str(png.resolve()),
            })

    manifest_path = OUTPUT_DIR / "plot_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    metadata = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_tables": sources,
        "selection": {
            "datasets": list(DATASETS_TO_PLOT),
            "architectures": list(ARCHITECTURES_TO_PLOT),
            "k_values": list(K_VALUES_TO_PLOT),
            "positive_forms": list(POSITIVE_FORMS),
            "target_rates_percent": list(TARGET_RATES),
        },
        "counts": {
            "scenarios": len(scenarios),
            "pdf_files": len(manifest),
            "png_files": len(manifest),
            "curves_per_figure": len(POSITIVE_FORMS),
            "points_per_curve": len(TARGET_RATES),
        },
        "plotting": {
            "curve_summary": "mean only; min-max bands removed",
            "x_major_ticks_percent": [25, 50, 75, 100],
            "p_y_major_ticks": "at most about six logarithmic ticks",
            "t2_y_major_ticks": "about five linear ticks",
            "grid": "major ticks only",
        },
        "note": "Draft v2 export with sparse axes and mean-only curves.",
    }
    with (OUTPUT_DIR / "run_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")

    print(OUTPUT_DIR)
    print(json.dumps(metadata["counts"], sort_keys=True))
    for row in manifest:
        print(row["pdf_path"])


if __name__ == "__main__":
    main()
