# %% [markdown]
# Adversarial-training epoch summary: one row x three mix-rate panels.
# Copy this entire file into one Jupyter notebook code cell, or run it as a
# percent-format notebook cell in VS Code/JupyterLab.

# %%
from pathlib import Path
import re
import textwrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration: these are the only values normally needing adjustment.
# -----------------------------------------------------------------------------
LOG_DIR = Path(
    r"E:\Experiment\saved_logs\at_evasion\Performance"
)

# A substring from the model/scenario portion of the filename. Set to None to
# draw every matching model as a separate 1 x 3 figure.
CASE_CONTAINS = "DFMS_Cross100-40C_Same18_0_1.0"

ATTACK_METHOD = "PGD"       # e.g. "PGD" or "FGSM"; None disables this filter
EPSILON = None               # None shows every epsilon as a separate figure row
STEPS = 10                   # None disables the step-count filter

MIX_ORDER = ("off", "0.5", "0.8")
INCLUDE_PRE_AT = True        # include logged epoch -1 (the original model)
SHOW_INDIVIDUAL_SEEDS = True
SHOW_STD_BAND = True         # meaningful once at least two seeds are available
LABEL_SEED_ENDPOINTS = False # useful for a few seeds, cluttered for many seeds

# Set this to a directory Path to save PNGs as well as displaying them.
SAVE_DIR = None              # example: Path(r"E:\Experiment\saved_plots\at_mix")


REQUIRED_COLUMNS = {"AT_Epoch", "Clean_Acc", "Robust_Acc"}
METRICS = {
    "Clean_Acc": {"label": "Clean", "color": "#1f77b4"},
    "Robust_Acc": {"label": "Robust", "color": "#d62728"},
}


def normalize_mix_label(value):
    """Normalize values such as 0.50 -> '0.5' while preserving 'off'."""
    text = str(value).strip().lower()
    if text == "off":
        return "off"
    try:
        return f"{float(text):g}"
    except ValueError:
        return text


def parse_log_filename(path):
    """Return the experiment key, seed, and mix label encoded in a log name."""
    stem = path.stem
    if stem.startswith("at_log_"):
        stem = stem[len("at_log_"):]

    if "_atseed=" not in stem or "_mix=" not in stem:
        return None

    case_key, tail = stem.rsplit("_atseed=", 1)
    seed_text, mix_text = tail.split("_mix=", 1)
    try:
        seed = int(seed_text)
    except ValueError:
        return None

    return {
        "case_key": case_key,
        "seed": seed,
        "mix": normalize_mix_label(mix_text),
    }


def make_overview_key(case_key):
    """Remove the swept epsilon value so all epsilons share one figure."""
    return re.sub(r"_eps=[^_]+", "", case_key, count=1)


def format_epsilon(value):
    """Use a readable fraction such as 8/255 when the value is close to one."""
    if value is None:
        return "not logged"

    numerator = int(round(value * 255))
    if numerator > 0 and np.isclose(value, numerator / 255, atol=1e-6, rtol=0):
        return f"{numerator}/255 ({value:g})"
    return f"{value:g}"


def keep_latest_logged_run(frame):
    """
    The logger appends when a CSV already exists. If the file contains multiple
    epoch -1 rows, retain only the most recently appended run.
    """
    starts = frame.index[frame["AT_Epoch"].eq(-1)].tolist()
    if len(starts) > 1:
        frame = frame.loc[starts[-1]:].copy()

    # Defensively retain the last value if an epoch was logged twice.
    return frame.drop_duplicates(subset="AT_Epoch", keep="last")


def load_matching_runs():
    runs = []
    skipped = []

    for path in sorted(LOG_DIR.glob("at_log_*.csv")):
        parsed = parse_log_filename(path)
        if parsed is None:
            continue

        if CASE_CONTAINS and CASE_CONTAINS.lower() not in parsed["case_key"].lower():
            continue

        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            skipped.append((path.name, f"read failure: {exc}"))
            continue

        if not REQUIRED_COLUMNS.issubset(frame.columns):
            skipped.append((path.name, "missing required columns"))
            continue

        for column in ["AT_Epoch", "Clean_Acc", "Robust_Acc", "AT_eps", "AT_steps"]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame = frame.dropna(subset=list(REQUIRED_COLUMNS)).copy()
        frame["AT_Epoch"] = frame["AT_Epoch"].astype(int)
        frame = keep_latest_logged_run(frame).sort_values("AT_Epoch")

        if not INCLUDE_PRE_AT:
            frame = frame[frame["AT_Epoch"] >= 0]
        if frame.empty:
            continue

        attack_value = None
        if "Attack_Method" in frame.columns and frame["Attack_Method"].notna().any():
            attack_value = str(frame.loc[frame["Attack_Method"].notna(), "Attack_Method"].iloc[0])
        if ATTACK_METHOD and attack_value != ATTACK_METHOD:
            continue

        eps_value = None
        if "AT_eps" in frame.columns and frame["AT_eps"].notna().any():
            eps_value = float(frame.loc[frame["AT_eps"].notna(), "AT_eps"].iloc[0])
        if EPSILON is not None:
            if eps_value is None or not np.isclose(eps_value, EPSILON, atol=1e-6, rtol=0):
                continue

        steps_value = None
        if "AT_steps" in frame.columns and frame["AT_steps"].notna().any():
            steps_value = int(frame.loc[frame["AT_steps"].notna(), "AT_steps"].iloc[0])
        if STEPS is not None and steps_value != STEPS:
            continue

        runs.append({
            **parsed,
            "overview_key": make_overview_key(parsed["case_key"]),
            "path": path,
            "frame": frame,
            "attack": attack_value,
            "eps": eps_value,
            "steps": steps_value,
        })

    if skipped:
        print(f"Skipped {len(skipped)} malformed/unreadable file(s):")
        for name, reason in skipped[:10]:
            print(f"  - {name}: {reason}")

    return runs


def draw_metric(ax, runs, metric_name):
    style = METRICS[metric_name]
    seed_series = []

    for run in sorted(runs, key=lambda item: item["seed"]):
        frame = run["frame"]
        epochs = frame["AT_Epoch"].to_numpy()
        values = 100.0 * frame[metric_name].to_numpy(dtype=float)

        series = pd.Series(values, index=epochs, name=run["seed"])
        seed_series.append(series)

        if SHOW_INDIVIDUAL_SEEDS:
            ax.plot(
                epochs,
                values,
                color=style["color"],
                linewidth=1.0,
                alpha=0.25,
                zorder=1,
            )

        if LABEL_SEED_ENDPOINTS and len(epochs):
            ax.annotate(
                f"{style['label'][0]} s{run['seed']}",
                xy=(epochs[-1], values[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=7,
                color=style["color"],
                va="center",
            )

    aligned = pd.concat(seed_series, axis=1).sort_index()
    mean = aligned.mean(axis=1)
    std = aligned.std(axis=1, ddof=1).fillna(0.0)

    ax.plot(
        mean.index,
        mean.values,
        color=style["color"],
        linewidth=2.6,
        label=f"{style['label']} mean",
        zorder=3,
    )

    if SHOW_STD_BAND and aligned.shape[1] >= 2:
        lower = np.clip(mean - std, 0.0, 100.0)
        upper = np.clip(mean + std, 0.0, 100.0)
        ax.fill_between(
            mean.index.to_numpy(),
            lower.to_numpy(),
            upper.to_numpy(),
            color=style["color"],
            alpha=0.12,
            linewidth=0,
            zorder=0,
        )


def plot_at_mix_summary(show_table=True):
    """
    Load logs using the current configuration variables and draw one overview
    figure per selected model/attack case:

        rows    = epsilon values
        columns = mix-rate values

    Each panel overlays all matching seeds. The function optionally displays a
    per-seed summary table and always returns that summary as a DataFrame.

    Notebook usage after running the definitions once:

        CASE_CONTAINS = "DFMS_Cross100-40C_Same18_0_1.0"
        EPSILON = None  # all available epsilons
        summary = plot_at_mix_summary(show_table=False)

    Because this function reads the configuration globals at call time, settings
    changed in a later notebook cell take effect immediately. Set show_table to
    False when only the figures should be displayed. The summary DataFrame is
    returned either way.
    """
    runs = load_matching_runs()
    if not runs:
        raise FileNotFoundError(
            "No logs matched the configuration. Check LOG_DIR, CASE_CONTAINS, "
            "ATTACK_METHOD, EPSILON, and STEPS."
        )

    mix_order = tuple(normalize_mix_label(value) for value in MIX_ORDER)
    overview_keys = sorted({run["overview_key"] for run in runs})

    print(
        f"Loaded {len(runs)} run file(s) across "
        f"{len(overview_keys)} overview case(s)."
    )

    summary_rows = []

    for overview_key in overview_keys:
        case_runs = [run for run in runs if run["overview_key"] == overview_key]
        epsilon_values = sorted({run["eps"] for run in case_runs if run["eps"] is not None})
        if not epsilon_values:
            epsilon_values = [None]

        n_rows = len(epsilon_values)
        n_cols = len(mix_order)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5.8 * n_cols, 3.7 * n_rows + 0.8),
            sharex=True,
            sharey=True,
            squeeze=False,
        )

        for row_index, epsilon in enumerate(epsilon_values):
            for col_index, mix in enumerate(mix_order):
                ax = axes[row_index, col_index]
                panel_runs = [
                    run
                    for run in case_runs
                    if run["mix"] == mix
                    and (
                        run["eps"] is epsilon
                        or (
                            run["eps"] is not None
                            and epsilon is not None
                            and np.isclose(run["eps"], epsilon, atol=1e-12, rtol=0)
                        )
                    )
                ]

                if not panel_runs:
                    ax.text(
                        0.5,
                        0.5,
                        "No matching logs",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        color="0.45",
                    )
                else:
                    for metric_name in METRICS:
                        draw_metric(ax, panel_runs, metric_name)

                    seeds = sorted({run["seed"] for run in panel_runs})
                    ax.text(
                        0.98,
                        0.96,
                        f"{len(seeds)} seed(s): {seeds}",
                        ha="right",
                        va="top",
                        transform=ax.transAxes,
                        fontsize=8,
                        color="0.35",
                    )

                    for run in panel_runs:
                        trained = run["frame"][run["frame"]["AT_Epoch"] >= 0]
                        if trained.empty:
                            continue
                        last = trained.loc[trained["AT_Epoch"].idxmax()]
                        summary_rows.append({
                            "Case": overview_key,
                            "Epsilon": epsilon,
                            "Mix": mix,
                            "Seed": run["seed"],
                            "Final epoch": int(last["AT_Epoch"]),
                            "Final clean (%)": 100.0 * float(last["Clean_Acc"]),
                            "Final robust (%)": 100.0 * float(last["Robust_Acc"]),
                            "Best clean (%)": 100.0 * float(trained["Clean_Acc"].max()),
                            "Best robust (%)": 100.0 * float(trained["Robust_Acc"].max()),
                        })

                if row_index == 0:
                    ax.set_title(f"mix = {mix}", fontsize=11, fontweight="semibold")
                if row_index == n_rows - 1:
                    ax.set_xlabel("Adversarial-training epoch")
                if col_index == 0:
                    ax.set_ylabel(
                        f"epsilon = {format_epsilon(epsilon)}\nAccuracy (%)"
                    )

                ax.set_ylim(0, 100)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

        title = textwrap.fill(overview_key, width=115)
        fig.suptitle(title, fontsize=12, fontweight="semibold", y=0.995)
        legend_handles = [
            Line2D([0], [0], color=style["color"], linewidth=2.6, label=style["label"])
            for style in METRICS.values()
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=len(legend_handles),
            frameon=False,
        )
        fig.text(
            0.5,
            0.012,
            "Thin curves: individual seeds. Thick curves: seed mean. "
            "Shading: mean ± 1 SD.",
            ha="center",
            fontsize=9,
            color="0.35",
        )
        fig.tight_layout(rect=(0, 0.045, 1, 0.91))

        if SAVE_DIR is not None:
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", overview_key).strip("_")
            output_path = SAVE_DIR / f"{safe_name}_epsilon_mix_overview.png"
            fig.savefig(output_path, dpi=180, bbox_inches="tight")
            print(f"Saved: {output_path}")

        plt.show()

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["Case", "Epsilon", "Mix", "Seed"]
        ).reset_index(drop=True)
        if show_table:
            try:
                display(summary.round(2))
            except NameError:
                print(summary.round(2).to_string(index=False))

    return summary


# Do not call plot_at_mix_summary() here. Run this definitions cell once, then
# change the configuration and call plot_at_mix_summary() from any later cell.
