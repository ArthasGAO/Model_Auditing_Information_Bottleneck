"""Render one checkpoint-aligned 3D AT trajectory example.

Example fixed for review:
  baseline = IPGuard, metric = TR
  AT source = FT-AL, epsilon = 2/255
  epochs = 0..29

The x/y coordinates are the in-distribution information-plane coordinates.
The z coordinate is IPGuard's actual decision statistic.  Joins use immutable
checkpoint SHA256 values rather than folder-name parsing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MI_TRAJ = ROOT / "saved_logs/at_evasion/MI_master_table_at_traj.csv"
MI_NEG = ROOT / "saved_logs/vanilla/MI_master_table_neg_pool0.csv"
MI_PRE = ROOT / "saved_logs/ft_final/MI_master_table_ft.csv"
BASELINE = (
    ROOT
    / "saved_logs/at_eval/posthoc_c10_traj_4baselines/by_baseline/IPGuard.csv"
)
OUTPUT_DIR = ROOT / "saved_plots/at_traj_3d_sample_ipguard_ftal_eps2_255"

FAMILY = "FT-AL"
EPS = 0.007843
EPS_LABEL = "2/255"
RUN_TAG = "traj2"
BINS = 50
IN_SIZE = 25000
METRIC = "TR"
PRE_MODEL = (
    "CIFAR-10_ResNet-18_25000_Same_25000_42_1.0_"
    "FT-AL_ftsize=25000_ftseed=0"
)
NEG_SCENARIO = "CIFAR-10_ResNet-18_25000"
XCOL, YCOL = "I(X;T)-In", "I(T;Y)-In"
PLOT_STYLE = "trihedral"  # trihedral | boxed
SHOW_NEGATIVES = False
GENERATE_INTERACTIVE = False


def load_data():
    """Return the 30-point trajectory, 80 negatives, source, and real tau."""
    mi = pd.read_csv(MI_TRAJ)
    mi = mi[
        (mi["run_tag"] == RUN_TAG)
        & (mi["bins"] == BINS)
        & (mi["in_size"] == IN_SIZE)
        & (mi["base_model"] == "C10_RN18_Same_FT-AL_0_1.0")
        & (mi["eps"].round(6) == EPS)
    ].copy()

    baseline = pd.read_csv(BASELINE)
    traj_b = baseline[
        (baseline["Case_Set"] == "trajectory")
        & (baseline["Family"] == FAMILY)
        & (baseline["AT_Eps"].round(6) == EPS)
        & (baseline["Status"] == "complete")
    ].copy()

    traj = traj_b.merge(
        mi,
        left_on="Checkpoint_SHA256",
        right_on="checkpoint_sha256",
        how="inner",
        validate="one_to_one",
    ).sort_values("Epoch")
    if len(traj) != 30 or traj["Epoch"].astype(int).tolist() != list(range(30)):
        raise ValueError("expected exactly one matched trajectory for epochs 0..29")

    neg = pd.DataFrame(columns=[XCOL, YCOL, METRIC])
    if SHOW_NEGATIVES:
        neg_mi = pd.read_csv(MI_NEG)
        neg_mi = neg_mi[
            (neg_mi["Scenario"] == NEG_SCENARIO)
            & (neg_mi["rate"] == 0.0)
            & (neg_mi["bins"] == BINS)
            & (neg_mi["in_size"] == IN_SIZE)
        ].copy()
        neg_b = baseline[
            (baseline["Case_Set"] == "negatives")
            & (baseline["Status"] == "complete")
        ].copy()
        neg_b["seed_join"] = (
            neg_b["Model_Dir"].str.extract(r"_25000_(\d+)_0\.0$")[0].astype(int)
        )
        neg = neg_mi.merge(
            neg_b[["seed_join", METRIC]],
            left_on="seed",
            right_on="seed_join",
            how="inner",
            validate="one_to_one",
        )
        if len(neg) != 80:
            raise ValueError(f"expected 80 matched negatives, got {len(neg)}")

    pre_mi = pd.read_csv(MI_PRE)
    pre_mi = pre_mi[
        (pre_mi["model_name"] == PRE_MODEL)
        & (pre_mi["bins"] == BINS)
        & (pre_mi["in_size"] == IN_SIZE)
    ].copy()
    pre_b = baseline[
        (baseline["Case_Set"] == "sources")
        & (baseline["Source"] == PRE_MODEL)
        & (baseline["Status"] == "complete")
    ].copy()
    pre = pre_b.merge(
        pre_mi,
        left_on="Checkpoint_SHA256",
        right_on="checkpoint_sha256",
        how="inner",
        validate="one_to_one",
    )
    if len(pre) != 1:
        raise ValueError(f"expected one SHA-matched pre-AT source, got {len(pre)}")

    tau_values = baseline.loc[baseline["Status"] == "complete", "Tau"].dropna().unique()
    if len(tau_values) != 1:
        raise ValueError(f"expected one IPGuard threshold, got {tau_values}")
    return traj, neg, pre.iloc[0], float(tau_values[0])


def bounds(traj, neg, pre):
    xs = np.concatenate([traj[XCOL].to_numpy(), neg[XCOL].to_numpy(), [pre[XCOL]]])
    ys = np.concatenate([traj[YCOL].to_numpy(), neg[YCOL].to_numpy(), [pre[YCOL]]])
    xpad = 0.06 * (xs.max() - xs.min())
    ypad = 0.14 * (ys.max() - ys.min())
    return (xs.min() - xpad, xs.max() + xpad), (ys.min() - ypad, ys.max() + ypad)


def render_static(traj, neg, pre, tau, output_dir):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 9,
        }
    )
    fig = plt.figure(figsize=(12, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    xlim, ylim = bounds(traj, neg, pre)

    gx, gy = np.meshgrid(np.linspace(*xlim, 2), np.linspace(*ylim, 2))
    ax.plot_surface(
        gx,
        gy,
        np.zeros_like(gx),
        color="#EAF1F8",
        alpha=0.28,
        shade=False,
        linewidth=0,
    )
    ax.plot_surface(
        gx,
        gy,
        np.full_like(gx, tau),
        color="#E15759",
        alpha=0.18,
        shade=False,
        linewidth=0,
    )
    for x0, x1, y0, y1 in [
        (xlim[0], xlim[1], ylim[0], ylim[0]),
        (xlim[0], xlim[1], ylim[1], ylim[1]),
        (xlim[0], xlim[0], ylim[0], ylim[1]),
        (xlim[1], xlim[1], ylim[0], ylim[1]),
    ]:
        ax.plot([x0, x1], [y0, y1], [tau, tau], color="#D62728", ls="--", lw=1.2)

    ax.scatter(
        neg[XCOL],
        neg[YCOL],
        neg[METRIC],
        marker="s",
        s=25,
        color="#3B73B9",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.35,
        label=f"H0 negatives (n={len(neg)})",
    )
    ax.scatter(
        [pre[XCOL]],
        [pre[YCOL]],
        [pre[METRIC]],
        marker="o",
        s=95,
        color="#D84A5B",
        edgecolor="white",
        linewidth=0.8,
        depthshade=False,
        label="Positive suspect (pre-AT)",
    )

    x, y, z = (traj[c].to_numpy(float) for c in (XCOL, YCOL, METRIC))
    epoch = traj["Epoch"].to_numpy(int)
    pts = np.column_stack([x, y, z])
    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    norm = colors.Normalize(vmin=0, vmax=29)
    cmap = plt.get_cmap("viridis")
    line = Line3DCollection(segments, cmap=cmap, norm=norm, linewidth=2.2, alpha=0.95)
    line.set_array((epoch[:-1] + epoch[1:]) / 2)
    ax.add_collection3d(line)
    scatter = ax.scatter(
        x,
        y,
        z,
        c=epoch,
        cmap=cmap,
        norm=norm,
        marker="D",
        s=50,
        edgecolor="white",
        linewidth=0.55,
        depthshade=False,
        label=f"FT-AL AT, ε={EPS_LABEL}",
    )
    ax.scatter([x[0]], [y[0]], [z[0]], marker="D", s=92, facecolor="none", edgecolor="black", lw=1.0)
    ax.scatter([x[-1]], [y[-1]], [z[-1]], marker="D", s=92, facecolor="none", edgecolor="#F5C542", lw=1.4)
    ax.text(x[0], y[0], z[0] + 0.035, "epoch 0", fontsize=8)
    ax.text(x[-1], y[-1], z[-1] + 0.035, "epoch 29", fontsize=8)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(-0.035, 1.02)
    ax.set_xlabel(r"$I(X;T)$", labelpad=10)
    ax.set_ylabel(r"$I(T;Y)$", labelpad=10)
    ax.set_zlabel("IPGuard matching rate (TR)", labelpad=9)
    ax.set_title(
        "FT-AL post-hoc AT trajectory (ε = 2/255) · IPGuard\n"
        "information plane on the floor; colour shows epoch 0 → 29",
        pad=18,
    )
    ax.view_init(elev=22, azim=-56)
    ax.set_box_aspect((1.35, 1.0, 0.92))
    ax.grid(True, alpha=0.35)
    ax.xaxis.pane.set_facecolor((0.96, 0.96, 0.96, 0.7))
    ax.yaxis.pane.set_facecolor((0.96, 0.96, 0.96, 0.7))
    ax.zaxis.pane.set_facecolor((0.98, 0.98, 0.98, 0.5))

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="#E15759", alpha=0.22, edgecolor="#D62728", label=f"τ = {tau:.2f}"))
    labels.append(f"τ = {tau:.2f}")
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.70, 0.98), frameon=True)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.025, pad=0.08, shrink=0.70)
    cbar.set_label("Epoch")
    cbar.set_ticks([0, 5, 10, 15, 20, 25, 29])

    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.04, top=0.90)
    png = output_dir / "ipguard_ftal_eps2_255_3d.png"
    pdf = output_dir / "ipguard_ftal_eps2_255_3d.pdf"
    fig.savefig(png, dpi=240, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, pdf


def render_static_trihedral(traj, pre, tau, output_dir):
    """Reference-inspired three-plane coordinate frame without negatives."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 13,
            "legend.fontsize": 9,
        }
    )
    fig = plt.figure(figsize=(11.2, 8.2), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    tx = traj[XCOL].to_numpy(float)
    ty = traj[YCOL].to_numpy(float)
    tz = traj[METRIC].to_numpy(float)
    epoch = traj["Epoch"].to_numpy(int)
    all_x = np.append(tx, float(pre[XCOL]))
    all_y = np.append(ty, float(pre[YCOL]))
    xspan = all_x.max() - all_x.min()
    yspan = all_y.max() - all_y.min()
    xmin, xmax = all_x.min() - 0.08 * xspan, all_x.max() + 0.08 * xspan
    ymin, ymax = all_y.min() - 0.13 * yspan, all_y.max() + 0.13 * yspan
    zmin, zmax = 0.0, 1.02

    # Three planes meet at the lower-left origin, matching the reference's
    # diagrammatic coordinate frame while retaining the real data scales.
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, 2), np.linspace(ymin, ymax, 2))
    ax.plot_surface(
        gx,
        gy,
        np.zeros_like(gx),
        color="#F5A9A9",
        alpha=0.28,
        shade=False,
        linewidth=0,
    )
    gx, gz = np.meshgrid(np.linspace(xmin, xmax, 2), np.linspace(zmin, zmax, 2))
    ax.plot_surface(
        gx,
        np.full_like(gx, ymin),
        gz,
        color="#B9D7ED",
        alpha=0.30,
        shade=False,
        linewidth=0,
    )
    gy, gz = np.meshgrid(np.linspace(ymin, ymax, 2), np.linspace(zmin, zmax, 2))
    ax.plot_surface(
        np.full_like(gy, xmin),
        gy,
        gz,
        color="#C9C9C9",
        alpha=0.32,
        shade=False,
        linewidth=0,
    )

    # Custom arrow axes replace Matplotlib's boxed 3D frame.
    xend, yend, zend = xmax + 0.045 * xspan, ymax + 0.055 * yspan, zmax + 0.045
    ax.plot([xmin, xend], [ymin, ymin], [0, 0], color="black", lw=1.6)
    ax.plot([xmin, xmin], [ymin, yend], [0, 0], color="black", lw=1.6)
    ax.plot([xmin, xmin], [ymin, ymin], [0, zend], color="black", lw=1.6)
    ax.scatter([xend], [ymin], [0], marker=">", s=28, color="black", depthshade=False)
    ax.scatter([xmin], [yend], [0], marker="<", s=28, color="black", depthshade=False)
    ax.scatter([xmin], [ymin], [zend], marker="^", s=28, color="black", depthshade=False)

    # The threshold is a height in this coordinate system.  Showing it on the
    # two walls keeps the trajectory unobstructed while preserving its meaning.
    ax.plot(
        [xmin, xmax],
        [ymin, ymin],
        [tau, tau],
        color="#D62728",
        ls="--",
        lw=1.75,
        alpha=0.95,
    )
    ax.plot(
        [xmin, xmin],
        [ymin, ymax],
        [tau, tau],
        color="#D62728",
        ls="--",
        lw=1.75,
        alpha=0.95,
    )
    ax.text(xmax - 0.12 * xspan, ymin, tau + 0.025, rf"$\tau={tau:.2f}$", color="#B22222", fontsize=9)

    # Pre-AT source and its projection onto the information plane.
    px, py, pz = float(pre[XCOL]), float(pre[YCOL]), float(pre[METRIC])
    ax.plot([px, px], [py, py], [0, pz], color="#D84A5B", ls="--", lw=1.25, alpha=0.9)
    ax.scatter(
        [px],
        [py],
        [0],
        marker="o",
        s=62,
        facecolor="white",
        edgecolor="#D84A5B",
        linewidth=1.4,
        depthshade=False,
    )
    ax.scatter(
        [px],
        [py],
        [pz],
        marker="o",
        s=105,
        color="#D84A5B",
        edgecolor="white",
        linewidth=0.8,
        depthshade=False,
    )

    pts = np.column_stack([tx, ty, tz])
    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    norm = colors.Normalize(vmin=0, vmax=29)
    cmap = plt.get_cmap("viridis")
    line = Line3DCollection(segments, cmap=cmap, norm=norm, linewidth=3.4, alpha=1.0)
    line.set_array((epoch[:-1] + epoch[1:]) / 2)
    ax.add_collection3d(line)
    scatter = ax.scatter(
        tx,
        ty,
        tz,
        c=epoch,
        cmap=cmap,
        norm=norm,
        marker="D",
        s=64,
        edgecolor="white",
        linewidth=0.6,
        depthshade=False,
    )
    ax.scatter([tx[0]], [ty[0]], [tz[0]], marker="D", s=105, facecolor="none", edgecolor="black", lw=1.1)
    ax.scatter([tx[-1]], [ty[-1]], [tz[-1]], marker="D", s=105, facecolor="none", edgecolor="#F3C623", lw=1.5)
    ax.text(tx[0], ty[0], tz[0] + 0.045, "epoch 0", fontsize=8)
    ax.text(tx[-1], ty[-1], tz[-1] + 0.045, "epoch 29", fontsize=8)

    # Labels are presentation-only; the underlying axes retain real MI/TR data.
    ax.text(xend + 0.025 * xspan, ymin, 0.005, r"$I(X;T)$", fontsize=13, ha="left", va="center")
    ax.text(xmin, yend + 0.025 * yspan, 0.005, r"$I(T;Y)$", fontsize=13, ha="center", va="bottom")
    ax.text(
        xmin,
        ymin,
        zend + 0.025,
        "IPGuard\nmatching rate (TR)",
        fontsize=10.5,
        ha="left",
        va="bottom",
    )
    ax.text(
        xmax - 0.22 * xspan,
        ymax - 0.10 * yspan,
        0.012,
        "Information plane",
        fontsize=9,
        color="#7A1F1F",
        ha="center",
    )

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#D84A5B", markeredgecolor="white", markersize=9, label="Positive suspect (pre-AT)"),
        Line2D([0], [0], marker="D", color="#5F8F8D", markerfacecolor="#3BAF8C", markeredgecolor="white", markersize=8, lw=2.2, label=f"FT-AL AT, ε={EPS_LABEL}"),
        Line2D([0], [0], color="#D62728", ls="--", lw=1.4, label=rf"Threshold $\tau={tau:.2f}$"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.99, 0.98), frameon=False)

    ax.set_xlim(xmin, xend + 0.07 * xspan)
    ax.set_ylim(ymin, yend + 0.06 * yspan)
    ax.set_zlim(-0.02, zend + 0.08)
    ax.set_box_aspect((1.15, 0.88, 1.0))
    ax.view_init(elev=24, azim=-54)
    ax.set_axis_off()
    ax.set_title(
        "FT-AL post-hoc AT trajectory (ε = 2/255) · IPGuard",
        pad=12,
    )
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.026, pad=0.025, shrink=0.57)
    cbar.set_label("Epoch")
    cbar.set_ticks([0, 5, 10, 15, 20, 25, 29])

    fig.subplots_adjust(left=0.015, right=0.90, bottom=0.02, top=0.92)
    png = output_dir / "ipguard_ftal_eps2_255_3d_trihedral.png"
    pdf = output_dir / "ipguard_ftal_eps2_255_3d_trihedral.pdf"
    fig.savefig(png, dpi=240, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, pdf


def render_interactive(traj, neg, pre, tau, output_dir):
    import plotly.graph_objects as go

    xlim, ylim = bounds(traj, neg, pre)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=neg[XCOL],
            y=neg[YCOL],
            z=neg[METRIC],
            mode="markers",
            marker=dict(size=4, color="#3B73B9", opacity=0.72, symbol="square"),
            name=f"H0 negatives (n={len(neg)})",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[pre[XCOL]],
            y=[pre[YCOL]],
            z=[pre[METRIC]],
            mode="markers",
            marker=dict(size=8, color="#D84A5B", line=dict(color="white", width=1)),
            name="Positive suspect (pre-AT)",
        )
    )
    hover = [
        f"epoch={int(e)}<br>I(X;T)={x:.4f}<br>I(T;Y)={y:.4f}<br>TR={z:.3f}"
        for e, x, y, z in zip(traj["Epoch"], traj[XCOL], traj[YCOL], traj[METRIC])
    ]
    fig.add_trace(
        go.Scatter3d(
            x=traj[XCOL],
            y=traj[YCOL],
            z=traj[METRIC],
            mode="lines+markers",
            line=dict(color="#777777", width=4),
            marker=dict(
                size=6,
                symbol="diamond",
                color=traj["Epoch"],
                colorscale="Viridis",
                cmin=0,
                cmax=29,
                line=dict(color="white", width=0.7),
                colorbar=dict(title="Epoch"),
            ),
            text=hover,
            hoverinfo="text",
            name=f"FT-AL AT, ε={EPS_LABEL}",
        )
    )
    gx, gy = np.meshgrid(np.linspace(*xlim, 2), np.linspace(*ylim, 2))
    fig.add_trace(
        go.Surface(
            x=gx,
            y=gy,
            z=np.full_like(gx, tau),
            showscale=False,
            opacity=0.20,
            colorscale=[[0, "#D62728"], [1, "#D62728"]],
            name=f"τ = {tau:.2f}",
            showlegend=True,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=(
            "FT-AL post-hoc AT trajectory (ε = 2/255) · IPGuard TR"
            "<br><sub>floor = information plane; colour = epoch 0 → 29; "
            f"actual decision threshold τ = {tau:.2f}</sub>"
        ),
        scene=dict(
            xaxis_title="I(X;T)",
            yaxis_title="I(T;Y)",
            zaxis_title="IPGuard matching rate (TR)",
            xaxis=dict(range=list(xlim)),
            yaxis=dict(range=list(ylim)),
            zaxis=dict(range=[-0.035, 1.02]),
            camera=dict(eye=dict(x=1.65, y=-1.55, z=0.92)),
            aspectratio=dict(x=1.35, y=1.0, z=0.92),
        ),
        width=1050,
        height=760,
        legend=dict(x=0.68, y=0.98),
        margin=dict(l=30, r=30, t=85, b=25),
    )
    html = output_dir / "ipguard_ftal_eps2_255_3d_interactive.html"
    fig.write_html(html, include_plotlyjs=True, full_html=True)
    return html


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    traj, neg, pre, tau = load_data()
    if PLOT_STYLE == "trihedral":
        png, pdf = render_static_trihedral(traj, pre, tau, OUTPUT_DIR)
    elif PLOT_STYLE == "boxed":
        png, pdf = render_static(traj, neg if SHOW_NEGATIVES else neg.iloc[0:0], pre, tau, OUTPUT_DIR)
    else:
        raise ValueError(f"unknown PLOT_STYLE: {PLOT_STYLE}")
    html = None
    if GENERATE_INTERACTIVE:
        try:
            html = render_interactive(traj, neg, pre, tau, OUTPUT_DIR)
        except ModuleNotFoundError as exc:
            if exc.name != "plotly":
                raise
    print(f"trajectory rows: {len(traj)}; negatives shown: {len(neg) if SHOW_NEGATIVES else 0}; tau: {tau}")
    print(f"epoch range: {int(traj.Epoch.min())}..{int(traj.Epoch.max())}")
    print(f"outputs: {png}\n         {pdf}")
    if html is not None:
        print(f"         {html}")
    elif GENERATE_INTERACTIVE:
        print("interactive HTML skipped: plotly is not installed in the headless runtime")


if __name__ == "__main__":
    main()
