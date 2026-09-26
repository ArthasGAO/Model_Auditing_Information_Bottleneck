"""Exact-F Q-Q diagnostics as independent publication panels.

Each output PDF contains one scenario and one reference-set size ``k``.  The
script reads the already-computed, ungated negative-model statistics at fixed
dataset usage rate 100% and 50 MI bins.  For every frozen split it compares
the 50 empirical predictive Hotelling statistics

    F* = (k - p) / (p (k - 1)) * T^2,  p = 2,

against F(p, k-p).  The plotted curve is the median empirical order statistic
across the 50 fixed rounds.  A gray pointwise 95% null envelope is obtained by
parametric simulation under N_2(0, I), replaying the exact frozen 80-model,
50-round reference/evaluation split design.  The optional blue split band is
descriptive only and is disabled for the publication-style sample.
"""

import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, NullLocator
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
from scipy.stats import f


ROOT = Path(__file__).resolve().parent
INPUT_CSV = (
    ROOT
    / "saved_logs"
    / "vanilla"
    / "Hypo_Test_FixedSplits"
    / "In_rate1_bins50"
    / "per_model.csv"
)
SPLIT_MANIFEST = (
    ROOT
    / "saved_logs"
    / "vanilla"
    / "fixed_splits"
    / "pool0_80_models_v1.json"
)
OUTPUT_DIR = (
    ROOT
    / "saved_plots"
    / "h0_exact_f_qq_k_sweep_all_cases_null_envelope_shared_tight_v5"
)

SCENARIO_STEMS = {
    "CIFAR-10_ResNet-18_25000": "cf10_rn18",
    "CIFAR-10_VGG16_25000": "cf10_vgg16",
    "CIFAR-10_DeiT_Plain_25000": "cf10_deit",
    "CIFAR-100_ResNet-18_25000": "cf100_rn18",
    "CIFAR-100_VGG16_25000": "cf100_vgg16",
    "CIFAR-100_DeiT_Distill_25000": "cf100_deit",
}
SCENARIOS_TO_PLOT = tuple(SCENARIO_STEMS)
SCENARIO_LABELS = {
    "cf10_rn18": "CF10 / RN18", "cf10_vgg16": "CF10 / VGG16", "cf10_deit": "CF10 / DeiT",
    "cf100_rn18": "CF100 / RN18", "cf100_vgg16": "CF100 / VGG16", "cf100_deit": "CF100 / DeiT",
}

K_VALUES = (5, 10, 15, 20, 25, 30)
P_DIMENSION = 2
EXPECTED_ROUNDS = 50
EXPECTED_EVALUATIONS_PER_ROUND = 50
ALPHA = 0.01
PLOTTING_POSITION_OFFSET = 0.5
SPLIT_BAND_QUANTILES = (0.10, 0.90)
NULL_ENVELOPE_QUANTILES = (0.025, 0.975)
N_NULL_SIMULATIONS = 10_000
NULL_SIMULATION_BATCH_SIZE = 100
NULL_SIMULATION_SEED = 20260924

# Visual switches.  The files intentionally contain no subplot letters or
# titles: k and scenario will be supplied by LaTeX subcaptions later.
SHOW_NULL_ENVELOPE = True
SHOW_SPLIT_BAND = False
SHOW_K_ANNOTATION = False
SHOW_METRICS = False

PANEL_FIGSIZE_INCHES = (2.25, 2.25)
PANEL_MARGINS = {
    "left": 0.280,
    "right": 0.975,
    "bottom": 0.200,
    "top": 0.895,
}
PNG_DPI = 400
SAVE_BBOX_INCHES = "tight"
SAVE_PAD_INCHES = 0.05
# Keep tight export, but give every panel the same content boundary so LaTeX
# applies the same scale to each PDF.  False restores independent tight crops.
SHARE_TIGHT_BBOX = True

FONT_SIZES = {
    "xlabel": 10.5,
    "ylabel": 10.5,
    "ticks": 8.0,
    "annotation": 7.5,
}
LINE_SIZES = {
    "qq_width": 1.25,
    "qq_marker_size": 3.0,
    "qq_marker_edge_width": 0.30,
    "identity_width": 1.0,
    "grid_width": 0.42,
    "spine_width": 0.72,
    "tick_length": 2.4,
    "tick_width": 0.65,
}

QQ_COLOR = "#006199"
QQ_BAND_COLOR = "#72B7DF"
NULL_ENVELOPE_COLOR = "#B8B8B8"
IDENTITY_COLOR = "#202020"
GRID_COLOR = "#C7C7C7"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_fonts():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_rows():
    require(INPUT_CSV.is_file(), f"Missing fixed-split table: {INPUT_CSV}")
    with INPUT_CSV.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    require(rows, f"No rows in {INPUT_CSV}")
    required = {
        "in_size_rate", "in_size", "bins", "scenario", "round_id",
        "k_ref", "T2", "stat_F", "p_F", "manifest_sha256", "seed", "model_name",
        "mi_kind", "training_size",
    }
    missing = required - set(rows[0])
    require(not missing, f"Missing required columns: {sorted(missing)}")
    require("gate1" not in rows[0],
            "Q-Q diagnostics must use raw ungated H0 statistics")
    return rows


def validate_source_identities(rows):
    """Check exact evaluation identities against the frozen split manifest."""
    document = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    payload = document["payload"]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode()
    require(hashlib.sha256(canonical).hexdigest() == document["manifest_sha256"],
            "Split-manifest payload hash mismatch")
    evaluations = {int(r["round_id"]): set(map(int, r["eval_negative_seeds"]))
                   for r in payload["rounds"]}
    require(set(evaluations) == set(range(EXPECTED_ROUNDS)), "Invalid manifest round ids")
    seen = set()
    for row in rows:
        scenario, k = row["scenario"], int(row["k_ref"])
        if scenario not in SCENARIOS_TO_PLOT or k not in K_VALUES:
            continue
        round_id, seed = int(row["round_id"]), int(row["seed"])
        key = scenario, k, round_id, seed
        require(key not in seen, f"Duplicate evaluation identity: {key}")
        seen.add(key)
        require(row["manifest_sha256"] == document["manifest_sha256"], f"Wrong manifest: {key}")
        require(round_id in evaluations and seed in evaluations[round_id], f"Wrong evaluation seed: {key}")
        require(row["model_name"] == payload["cases"][scenario][str(seed)]["model_name"],
                f"Wrong evaluation model: {key}")
        require(row["mi_kind"] == "In" and int(row["training_size"]) == 25000,
                f"Wrong MI population: {key}")
    expected = len(SCENARIOS_TO_PLOT) * len(K_VALUES) * EXPECTED_ROUNDS * EXPECTED_EVALUATIONS_PER_ROUND
    require(len(seen) == expected, f"Expected {expected} unique evaluation rows, found {len(seen)}")
    return {"unique_evaluation_rows": len(seen), "duplicate_evaluation_rows": 0,
            "all_model_identities_match_manifest": True,
            "all_evaluation_seeds_match_manifest": True}


def load_split_design():
    require(SPLIT_MANIFEST.is_file(),
            f"Missing fixed-split manifest: {SPLIT_MANIFEST}")
    manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    payload = manifest["payload"]
    pool_seeds = [int(seed) for seed in payload["pool_seeds"]]
    rounds = payload["rounds"]
    require(len(pool_seeds) == 80,
            f"Expected 80 pool seeds, found {len(pool_seeds)}")
    require(len(set(pool_seeds)) == len(pool_seeds),
            "Duplicate pool seeds in split manifest")
    require(len(rounds) == EXPECTED_ROUNDS,
            f"Expected {EXPECTED_ROUNDS} rounds, found {len(rounds)}")
    seed_to_index = {seed: index for index, seed in enumerate(pool_seeds)}

    designs = {}
    for k in K_VALUES:
        reference_indices = []
        evaluation_indices = []
        for expected_round_id, round_spec in enumerate(rounds):
            require(int(round_spec["round_id"]) == expected_round_id,
                    "Split-manifest round ids are not contiguous from zero")
            reference_seeds = [
                int(seed) for seed in round_spec["by_k"][str(k)]["h0_seeds"]
            ]
            evaluation_seeds = [
                int(seed) for seed in round_spec["eval_negative_seeds"]
            ]
            require(len(reference_seeds) == k,
                    f"round={expected_round_id}, k={k}: wrong reference count")
            require(len(evaluation_seeds) == EXPECTED_EVALUATIONS_PER_ROUND,
                    f"round={expected_round_id}: wrong evaluation count")
            require(len(set(reference_seeds)) == k and len(set(evaluation_seeds)) == len(evaluation_seeds),
                    f"Duplicate split seed: round={expected_round_id}, k={k}")
            require(not set(reference_seeds) & set(evaluation_seeds),
                    f"round={expected_round_id}, k={k}: reference/eval overlap")
            require(set(reference_seeds) <= set(seed_to_index),
                    "Unknown reference seed in split manifest")
            require(set(evaluation_seeds) <= set(seed_to_index),
                    "Unknown evaluation seed in split manifest")
            reference_indices.append([
                seed_to_index[seed] for seed in reference_seeds
            ])
            evaluation_indices.append([
                seed_to_index[seed] for seed in evaluation_seeds
            ])
        designs[k] = {
            "reference_indices": np.asarray(reference_indices, dtype=int),
            "evaluation_indices": np.asarray(evaluation_indices, dtype=int),
        }
    return pool_seeds, designs


def simulate_null_envelopes(pool_size, designs):
    """Simulate pointwise null envelopes for the median fixed-split Q-Q curve."""
    envelopes = {}
    for k in K_VALUES:
        reference_indices = designs[k]["reference_indices"]
        evaluation_indices = designs[k]["evaluation_indices"]
        simulated_medians = np.empty(
            (N_NULL_SIMULATIONS, EXPECTED_EVALUATIONS_PER_ROUND),
            dtype=float,
        )
        rng = np.random.default_rng(NULL_SIMULATION_SEED + k)
        start = 0
        while start < N_NULL_SIMULATIONS:
            stop = min(start + NULL_SIMULATION_BATCH_SIZE,
                       N_NULL_SIMULATIONS)
            batch_size = stop - start
            pool = rng.standard_normal((batch_size, pool_size, P_DIMENSION))

            references = pool[:, reference_indices, :]
            reference_mean = references.mean(axis=2)
            centered = references - reference_mean[:, :, None, :]
            covariance = np.einsum(
                "brki,brkj->brij", centered, centered, optimize=True
            ) / (k - 1.0)

            evaluations = pool[:, evaluation_indices, :]
            delta = evaluations - reference_mean[:, :, None, :]
            solved = np.linalg.solve(
                covariance, np.swapaxes(delta, -1, -2)
            )
            mahalanobis_sq = np.einsum(
                "brni,brin->brn", delta, solved, optimize=True
            )
            t2 = (k / (k + 1.0)) * mahalanobis_sq
            stat_f = t2 * (k - P_DIMENSION) / (
                P_DIMENSION * (k - 1.0)
            )
            order_statistics = np.sort(stat_f, axis=2)
            simulated_medians[start:stop] = np.median(
                order_statistics, axis=1
            )
            start = stop
            if start % 2000 == 0 or start == N_NULL_SIMULATIONS:
                print(f"[NULL] k={k}: {start}/{N_NULL_SIMULATIONS} simulations", flush=True)

        envelopes[k] = {
            "lower": np.quantile(
                simulated_medians, NULL_ENVELOPE_QUANTILES[0], axis=0
            ),
            "median": np.median(simulated_medians, axis=0),
            "upper": np.quantile(
                simulated_medians, NULL_ENVELOPE_QUANTILES[1], axis=0
            ),
        }
    return envelopes


def validate_and_group(rows):
    grouped = defaultdict(list)
    for row in rows:
        scenario = row["scenario"]
        if scenario not in SCENARIOS_TO_PLOT:
            continue
        k = int(row["k_ref"])
        if k not in K_VALUES:
            continue
        require(math.isclose(float(row["in_size_rate"]), 1.0,
                             rel_tol=0, abs_tol=1e-12),
                f"Wrong sample rate in {scenario}, k={k}")
        require(int(row["in_size"]) == 25000,
                f"Wrong in_size in {scenario}, k={k}")
        require(int(row["bins"]) == 50,
                f"Wrong bins in {scenario}, k={k}")

        t2 = float(row["T2"])
        stat_f = float(row["stat_F"])
        p_f = float(row["p_F"])
        require(all(math.isfinite(value) for value in (t2, stat_f, p_f)),
                f"Nonfinite statistic in {scenario}, k={k}")
        require(t2 >= 0 and stat_f >= 0 and 0 <= p_f <= 1,
                f"Out-of-range statistic in {scenario}, k={k}")

        expected_stat_f = t2 * (k - P_DIMENSION) / (
            P_DIMENSION * (k - 1.0)
        )
        require(math.isclose(stat_f, expected_stat_f,
                             rel_tol=1e-10, abs_tol=1e-14),
                f"T2/stat_F mismatch in {scenario}, k={k}")
        expected_p_f = float(f.sf(stat_f, P_DIMENSION, k - P_DIMENSION))
        require(math.isclose(p_f, expected_p_f,
                             rel_tol=1e-10, abs_tol=1e-14),
                f"stat_F/p_F mismatch in {scenario}, k={k}")

        grouped[(scenario, k, int(row["round_id"]))].append(stat_f)

    for scenario in SCENARIOS_TO_PLOT:
        require(scenario in SCENARIO_STEMS,
                f"Missing output stem for scenario: {scenario}")
        for k in K_VALUES:
            round_ids = sorted(
                round_id for (found_scenario, found_k, round_id) in grouped
                if found_scenario == scenario and found_k == k
            )
            require(len(round_ids) == EXPECTED_ROUNDS,
                    f"{scenario}, k={k}: expected {EXPECTED_ROUNDS} rounds, "
                    f"found {len(round_ids)}")
            require(len(set(round_ids)) == EXPECTED_ROUNDS,
                    f"Duplicate round ids in {scenario}, k={k}")
            for round_id in round_ids:
                values = grouped[(scenario, k, round_id)]
                require(len(values) == EXPECTED_EVALUATIONS_PER_ROUND,
                        f"{scenario}, k={k}, round={round_id}: expected "
                        f"{EXPECTED_EVALUATIONS_PER_ROUND} evaluations, "
                        f"found {len(values)}")
    return grouped


def qq_summary(grouped, scenario, k):
    round_ids = sorted(
        round_id for (found_scenario, found_k, round_id) in grouped
        if found_scenario == scenario and found_k == k
    )
    n = EXPECTED_EVALUATIONS_PER_ROUND
    probabilities = (
        np.arange(1, n + 1, dtype=float) - PLOTTING_POSITION_OFFSET
    ) / n
    theoretical = f.ppf(probabilities, P_DIMENSION, k - P_DIMENSION)
    empirical_by_round = np.asarray([
        np.sort(np.asarray(grouped[(scenario, k, round_id)], dtype=float))
        for round_id in round_ids
    ])

    lower = np.quantile(empirical_by_round, SPLIT_BAND_QUANTILES[0], axis=0)
    median = np.median(empirical_by_round, axis=0)
    upper = np.quantile(empirical_by_round, SPLIT_BAND_QUANTILES[1], axis=0)
    scale = max(float(theoretical[-1] - theoretical[0]), np.finfo(float).eps)
    qq_nrmse = float(np.sqrt(np.mean((median - theoretical) ** 2)) / scale)

    all_statistics = empirical_by_round.ravel()
    p_values = f.sf(all_statistics, P_DIMENSION, k - P_DIMENSION)
    mean_fpr = float(np.mean(p_values < ALPHA))
    per_round_fpr = np.mean(
        f.sf(empirical_by_round, P_DIMENSION, k - P_DIMENSION) < ALPHA,
        axis=1,
    )
    return {
        "probabilities": probabilities,
        "theoretical": theoretical,
        "lower": lower,
        "median": median,
        "upper": upper,
        "qq_nrmse": qq_nrmse,
        "mean_fpr_at_0.01": mean_fpr,
        "median_round_fpr_at_0.01": float(np.median(per_round_fpr)),
        "round_fpr_p10": float(np.quantile(per_round_fpr, 0.10)),
        "round_fpr_p90": float(np.quantile(per_round_fpr, 0.90)),
        "n_rounds": len(round_ids),
        "n_evaluations_per_round": n,
    }


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(
        True,
        which="major",
        color=GRID_COLOR,
        linestyle=(0, (2, 2)),
        linewidth=LINE_SIZES["grid_width"],
        alpha=0.8,
    )
    ax.tick_params(
        direction="out",
        length=LINE_SIZES["tick_length"],
        width=LINE_SIZES["tick_width"],
        labelsize=FONT_SIZES["ticks"],
        pad=1.8,
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(LINE_SIZES["spine_width"])


def build_panel(scenario, k, summary, null_envelope):
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE_INCHES)
    theoretical = summary["theoretical"]

    if SHOW_NULL_ENVELOPE:
        ax.fill_between(
            theoretical,
            null_envelope["lower"],
            null_envelope["upper"],
            color=NULL_ENVELOPE_COLOR,
            alpha=0.55,
            linewidth=0,
            zorder=1,
        )
    if SHOW_SPLIT_BAND:
        ax.fill_between(
            theoretical,
            summary["lower"],
            summary["upper"],
            color=QQ_BAND_COLOR,
            alpha=0.28,
            linewidth=0,
            zorder=1,
        )
    ax.plot(
        theoretical,
        summary["median"],
        color=QQ_COLOR,
        marker="o",
        linestyle="-",
        linewidth=LINE_SIZES["qq_width"],
        markersize=LINE_SIZES["qq_marker_size"],
        markerfacecolor=QQ_COLOR,
        markeredgecolor="white",
        markeredgewidth=LINE_SIZES["qq_marker_edge_width"],
        zorder=3,
    )

    upper_limit = 1.06 * max(
        float(np.max(theoretical)),
        float(np.max(summary["upper"] if SHOW_SPLIT_BAND
                     else summary["median"])),
        float(np.max(null_envelope["upper"])),
    )
    ax.plot(
        [0, upper_limit],
        [0, upper_limit],
        color=IDENTITY_COLOR,
        linestyle="--",
        linewidth=LINE_SIZES["identity_width"],
        zorder=2,
    )
    ax.set_xlim(0, upper_limit)
    ax.set_ylim(0, upper_limit)
    ax.set_xlabel(
        r"Theoretical $F$ quantiles",
        fontsize=FONT_SIZES["xlabel"],
        labelpad=2.0,
    )
    ax.set_ylabel(
        r"Empirical $F$ quantiles",
        fontsize=FONT_SIZES["ylabel"],
        labelpad=2.0,
    )
    style_axis(ax)
    ax.set_box_aspect(1)

    if SHOW_K_ANNOTATION:
        ax.text(
            0.05, 0.94, rf"$k={k}$",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=FONT_SIZES["annotation"],
        )
    if SHOW_METRICS:
        ax.text(
            0.95, 0.05,
            rf"NRMSE={summary['qq_nrmse']:.3f}" "\n"
            rf"FPR$_{{.01}}$={summary['mean_fpr_at_0.01']:.3f}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=FONT_SIZES["annotation"],
        )

    fig.subplots_adjust(**PANEL_MARGINS)
    return fig, upper_limit


def measure_panel_geometry(fig):
    """Measure actual PNG and PDF renderers; their font extents can differ."""
    measurements = {}

    def record(renderer, format_name):
        measurements[format_name] = {
            "content_bounds_inches": list(fig.get_tightbbox(renderer).extents),
            "axes_bounds_inches": list(
                fig.axes[0].get_window_extent(renderer)
                .transformed(fig.dpi_scale_trans.inverted()).bounds
            ),
        }

    original_dpi = fig.dpi
    fig.set_dpi(PNG_DPI)
    fig.canvas.draw()
    record(fig.canvas.get_renderer(), "png")
    callback_id = fig.canvas.mpl_connect(
        "draw_event", lambda event: record(event.renderer, "pdf")
    )
    try:
        # Measure the PDF backend before cropping; this draft stays in memory.
        with io.BytesIO() as buffer, matplotlib.rc_context({"savefig.bbox": None}):
            fig.savefig(buffer, format="pdf", bbox_inches=None)
    finally:
        fig.canvas.mpl_disconnect(callback_id)
        fig.set_dpi(original_dpi)
    require("pdf" in measurements, "PDF renderer did not report its bounds")
    return measurements


def compute_shared_tight_bbox(summaries, null_envelopes):
    """Union all selected panels in physical inches, before common padding."""
    require(SAVE_BBOX_INCHES == "tight", "Shared tight bounds require tight export")
    bounds = []
    measurements = []
    common_axes = None
    for scenario in SCENARIOS_TO_PLOT:
        for k in K_VALUES:
            fig, _ = build_panel(scenario, k, summaries[scenario, k], null_envelopes[k])
            try:
                geometry = measure_panel_geometry(fig)
            finally:
                plt.close(fig)
            for backend in geometry.values():
                bounds.append(Bbox.from_extents(*backend["content_bounds_inches"]))
                axes_bounds = backend["axes_bounds_inches"]
                if common_axes is None:
                    common_axes = axes_bounds
                require(np.allclose(axes_bounds, common_axes, rtol=0, atol=1e-10),
                        "Panel axes geometry differs; shared cropping cannot align it")
            measurements.append({"scenario": scenario, "k_ref": k, **geometry})
    shared_bbox = Bbox.union(bounds).frozen()
    require(np.isfinite(shared_bbox.extents).all(), "Non-finite shared tight bounds")
    print(f"[LAYOUT] Shared tight content bounds (inches): {shared_bbox.extents.tolist()}",
          flush=True)
    return shared_bbox, {
        "scope": "all selected scenarios and k values, PNG and PDF renderers",
        "content_bounds_inches": shared_bbox.extents.tolist(),
        "page_size_points": [
            72 * (shared_bbox.width + 2 * SAVE_PAD_INCHES),
            72 * (shared_bbox.height + 2 * SAVE_PAD_INCHES),
        ],
        "axes_bounds_inches_before_crop": list(common_axes),
        "axes_bounds_points_on_page": [
            72 * (common_axes[0] - shared_bbox.x0 + SAVE_PAD_INCHES),
            72 * (common_axes[1] - shared_bbox.y0 + SAVE_PAD_INCHES),
            72 * common_axes[2], 72 * common_axes[3],
        ],
        "measurements": measurements,
    }


def render_panel(scenario, k, summary, null_envelope, shared_tight_bbox=None):
    fig, upper_limit = build_panel(scenario, k, summary, null_envelope)
    extra_artists = None
    if shared_tight_bbox is not None:
        # An unpainted rectangle contributes the common content bounds to the
        # normal tight calculation.  It does not resize axes or alter data.
        crop_anchor = Rectangle(
            (shared_tight_bbox.x0, shared_tight_bbox.y0),
            shared_tight_bbox.width, shared_tight_bbox.height,
            transform=fig.dpi_scale_trans, facecolor="none", edgecolor="none",
            linewidth=0, clip_on=False,
        )
        fig.add_artist(crop_anchor)
        extra_artists = (crop_anchor,)
    scenario_dir = OUTPUT_DIR / SCENARIO_STEMS[scenario]
    scenario_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{SCENARIO_STEMS[scenario]}_k{k}_exact_f_qq"
    pdf_path = scenario_dir / f"{stem}.pdf"
    png_path = scenario_dir / f"{stem}.png"
    fig.savefig(
        pdf_path,
        facecolor="white",
        bbox_inches=SAVE_BBOX_INCHES,
        pad_inches=SAVE_PAD_INCHES,
        bbox_extra_artists=extra_artists,
        metadata={
            "Title": f"{scenario}, k={k}, exact-F Q-Q diagnostic",
            "Subject": "Fixed N=25000 and bins=50 H0 distribution diagnostic",
        },
    )
    fig.savefig(
        png_path,
        dpi=PNG_DPI,
        facecolor="white",
        bbox_inches=SAVE_BBOX_INCHES,
        pad_inches=SAVE_PAD_INCHES,
        bbox_extra_artists=extra_artists,
    )
    plt.close(fig)
    return pdf_path, png_path, upper_limit


def render_preview(outputs, scenario):
    """Create a review-only 2x3 contact sheet; publication PDFs stay separate."""
    selected = [o for o in outputs if o["scenario"] == scenario]
    require([o["k_ref"] for o in selected] == list(K_VALUES), "Incomplete scenario preview")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    for ax, output in zip(axes.ravel(), selected):
        ax.imshow(plt.imread(output["png"]))
        ax.set_title(rf"$k={output['k_ref']}$", fontsize=10, pad=2)
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.95,
                        wspace=0.02, hspace=0.10)
    preview_path = OUTPUT_DIR / f"{SCENARIO_STEMS[scenario]}_exact_f_qq_null_envelope_preview_2x3.png"
    fig.savefig(preview_path, dpi=180, facecolor="white",
                bbox_inches=SAVE_BBOX_INCHES, pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)
    return preview_path


def render_all_cases_preview(outputs):
    """Review sheet: scenario rows and k columns, with every exported panel."""
    index = {(o["scenario"], o["k_ref"]): o for o in outputs}
    fig, axes = plt.subplots(len(SCENARIOS_TO_PLOT), len(K_VALUES), figsize=(13.8, 13.6), squeeze=False)
    fig.subplots_adjust(left=.077, right=.995, top=.965, bottom=.005, wspace=.01, hspace=.07)
    for si, scenario in enumerate(SCENARIOS_TO_PLOT):
        for ki, k in enumerate(K_VALUES):
            ax = axes[si, ki]
            ax.imshow(plt.imread(index[scenario, k]["png"]))
            ax.axis("off")
            if si == 0:
                ax.set_title(rf"$k={k}$", fontsize=12, pad=4)
        pos = axes[si, 0].get_position()
        label = SCENARIO_LABELS[SCENARIO_STEMS[scenario]].replace(" / ", "\n")
        fig.text(.013, (pos.y0 + pos.y1)/2, label, ha="left", va="center", fontsize=11, weight="bold")
    path = OUTPUT_DIR / "all_cases_exact_f_qq_null_envelope_preview_6x6.png"
    fig.savefig(path, dpi=180, facecolor="white", bbox_inches=SAVE_BBOX_INCHES, pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)
    return path


def main():
    configure_fonts()
    input_hash, split_hash = sha256(INPUT_CSV), sha256(SPLIT_MANIFEST)
    rows = load_rows()
    identity_validation = validate_source_identities(rows)
    grouped = validate_and_group(rows)
    pool_seeds, split_designs = load_split_design()
    print(f"[VALIDATED] {len(SCENARIOS_TO_PLOT)} scenarios x {len(K_VALUES)} k values; "
          f"{identity_validation['unique_evaluation_rows']} unique evaluation rows", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    null_envelopes = simulate_null_envelopes(
        len(pool_seeds), split_designs
    )
    np.savez_compressed(
        OUTPUT_DIR / "null_envelopes.npz",
        **{
            f"k{k}_{component}": values
            for k, envelope in null_envelopes.items()
            for component, values in envelope.items()
        },
    )

    summaries = {(scenario, k): qq_summary(grouped, scenario, k)
                 for scenario in SCENARIOS_TO_PLOT for k in K_VALUES}
    shared_bbox, shared_layout = (None, None)
    if SHARE_TIGHT_BBOX:
        shared_bbox, shared_layout = compute_shared_tight_bbox(summaries, null_envelopes)

    outputs = []
    for scenario in SCENARIOS_TO_PLOT:
        for k in K_VALUES:
            summary = summaries[scenario, k]
            pdf_path, png_path, upper_limit = render_panel(
                scenario, k, summary, null_envelopes[k], shared_bbox
            )
            outputs.append({
                "scenario": scenario,
                "scenario_stem": SCENARIO_STEMS[scenario],
                "k_ref": k,
                "theoretical_distribution": f"F({P_DIMENSION},{k-P_DIMENSION})",
                "n_rounds": summary["n_rounds"],
                "n_evaluations_per_round": summary["n_evaluations_per_round"],
                "qq_nrmse": summary["qq_nrmse"],
                "mean_fpr_at_0.01": summary["mean_fpr_at_0.01"],
                "median_round_fpr_at_0.01": summary["median_round_fpr_at_0.01"],
                "round_fpr_p10": summary["round_fpr_p10"],
                "round_fpr_p90": summary["round_fpr_p90"],
                "axis_upper_limit": upper_limit,
                "pdf": str(pdf_path.resolve()),
                "png": str(png_path.resolve()),
                "pdf_sha256": sha256(pdf_path),
            })
        print(f"[RENDERED] {SCENARIO_STEMS[scenario]}: {len(K_VALUES)} PDF/PNG panels", flush=True)

    summary_fields = [
        "scenario", "scenario_stem", "k_ref", "theoretical_distribution",
        "n_rounds", "n_evaluations_per_round", "qq_nrmse",
        "mean_fpr_at_0.01", "median_round_fpr_at_0.01",
        "round_fpr_p10", "round_fpr_p90", "axis_upper_limit",
        "pdf", "png", "pdf_sha256",
    ]
    with (OUTPUT_DIR / "qq_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(outputs)

    previews = {scenario: str(render_preview(outputs, scenario).resolve()) for scenario in SCENARIOS_TO_PLOT}
    preview_path = render_all_cases_preview(outputs)
    require(len(outputs) == len(SCENARIOS_TO_PLOT) * len(K_VALUES), "Incomplete panel grid")
    require(sha256(INPUT_CSV) == input_hash and sha256(SPLIT_MANIFEST) == split_hash,
            "Input table or split manifest changed during this run")

    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Independent exact-F Q-Q panels over reference-set size at fixed "
            "in_size=25000 and bins=50"
        ),
        "render_script": str(Path(__file__).resolve()),
        "render_script_sha256": sha256(__file__),
        "input_csv": str(INPUT_CSV.resolve()),
        "input_csv_sha256": input_hash,
        "split_manifest": str(SPLIT_MANIFEST.resolve()),
        "split_manifest_sha256": split_hash,
        "validation": {**identity_validation, "input_files_unchanged": True, "complete_panel_grid": True},
        "selection": {
            "scenarios": list(SCENARIOS_TO_PLOT),
            "k_values": list(K_VALUES),
            "in_size_rate": 1.0,
            "in_size": 25000,
            "bins": 50,
            "p_dimension": P_DIMENSION,
            "alpha": ALPHA,
        },
        "qq_definition": {
            "empirical_statistic": "stat_F = (k-2)/(2(k-1)) * T2",
            "theoretical_distribution": "F(2, k-2)",
            "plotting_positions": "(i - 0.5) / n, n=50",
            "central_curve": "median empirical order statistic across 50 rounds",
            "null_envelope": (
                "pointwise 95% envelope for the median 50-round Q-Q curve; "
                "10,000 parametric N_2(0,I) simulations replaying the exact "
                "frozen 80-model reference/evaluation split structure"
            ),
            "null_envelope_quantiles": list(NULL_ENVELOPE_QUANTILES),
            "null_simulations": N_NULL_SIMULATIONS,
            "null_simulation_seed_base": NULL_SIMULATION_SEED,
            "null_simulation_seed_rule": "base + k",
            "null_model": "N_2(0,I); valid by affine invariance",
            "split_band": (
                "10th-90th percentile across observed rounds; descriptive, "
                "not a confidence interval; disabled in this export"
            ),
            "gate1_applied": False,
            "shrinkage_applied": False,
        },
        "layout": {
            "one_k_per_file": True,
            "panel_figsize_inches": list(PANEL_FIGSIZE_INCHES),
            "bbox_inches": SAVE_BBOX_INCHES,
            "pad_inches": SAVE_PAD_INCHES,
            "page_size_policy": (
                "union of all panel content bounds plus common padding"
                if SHARE_TIGHT_BBOX else "independent content bounds plus padding"
            ),
            "share_tight_bbox": SHARE_TIGHT_BBOX,
            "shared_tight_layout": shared_layout,
            "panel_margins": PANEL_MARGINS,
            "axes_box_aspect": 1.0,
            "show_null_envelope": SHOW_NULL_ENVELOPE,
            "show_split_band": SHOW_SPLIT_BAND,
            "show_k_annotation": SHOW_K_ANNOTATION,
            "show_metrics": SHOW_METRICS,
        },
        "review_preview_png": str(preview_path.resolve()),
        "scenario_previews": previews,
        "outputs": outputs,
    }
    manifest_path = OUTPUT_DIR / "qq_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = ["# Exact-F Q-Q：全部 case 与 k 扫描", "",
             f"共 {len(SCENARIOS_TO_PLOT)} 个 case × {len(K_VALUES)} 个 k，每组独立输出 PDF 和 PNG。",
             f"k = {list(K_VALUES)}；MI size = 25,000（100%），bins = 50。", "",
             "蓝线是 50 个固定 round 的中位数 Q-Q 曲线；每个 round 使用 50 个固定评估模型。",
             "灰带为 10,000 次 H0 模拟得到的 95% pointwise null envelope（逐点零假设包络）；虚线为 y=x。",
             "所有 case 采用相同固定划分设计，故同一 k 的零假设包络共用。", "",
             f"所有输出使用 bbox_inches='{SAVE_BBOX_INCHES}'、pad_inches={SAVE_PAD_INCHES}；每个面板坐标轴仍保持正方形。", "",
             ("独立面板共享全部 case/k 在 PDF、PNG 渲染器下的 tight 内容边界并集；页面尺寸、坐标框位置及大小一致。"
              if SHARE_TIGHT_BBOX else "独立面板分别 tight 裁剪，页面尺寸可能不同。"), "",
             f"[六个 case 总览]({preview_path.name})", "", "| Case | 2×3 预览 | 独立 PDF/PNG |", "|---|---|---|"]
    for scenario in SCENARIOS_TO_PLOT:
        stem = SCENARIO_STEMS[scenario]
        lines.append(f"| {SCENARIO_LABELS[stem]} | [预览]({Path(previews[scenario]).name}) | [{stem}/]({stem}/) |")
    lines += ["", f"已核对 {identity_validation['unique_evaluation_rows']:,} 条唯一评估记录及模型身份、固定划分、T2/F/p 值。",
              "qq_summary.csv 保存每组诊断摘要；qq_manifest.json 保存输入、配置和文件身份；null_envelopes.npz 保存六个 k 的模拟包络。", ""]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")

    print(OUTPUT_DIR.resolve())
    print(json.dumps({
        "pdf_files": len(outputs),
        "png_files": len(outputs),
        "scenarios": list(SCENARIOS_TO_PLOT),
        "k_values": list(K_VALUES),
    }, sort_keys=True))
    for output in outputs:
        print(output["pdf"])


if __name__ == "__main__":
    main()
