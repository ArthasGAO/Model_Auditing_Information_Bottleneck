%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    clear_csv_cache()
except NameError:
    pass

# Standalone: a trajectory is a path with a direction, which TableGroupSpec has
# no way to express, so this cell draws on the axes directly instead of going
# through plot_information_plane. Same plane, same units, same null.
CSV_TRAJ = "./saved_logs/at_evasion/MI_master_table_at_traj.csv"
CSV_NEG = "./saved_logs/vanilla/MI_master_table_neg_pool0.csv"
CSV_VIC = "./saved_logs/vanilla/MI_master_table_victim.csv"

BINS, IN_SIZE = 50, 25000
RUN_TAG = "traj2"          # traj = the first pass, last epoch only. Not this.
POOL = "CIFAR-10_ResNet-18_25000"   # all four sources are ResNet-18
ONE_FIGURE_PER_CASE = True          # False -> one 2x2 grid instead

# One case per figure. `pre_at` is where that source sat BEFORE any adversarial
# fine-tuning: its own row in its own family's table, the same checkpoint the AT
# run started from (epoch -1, which main_at.py logs but does not checkpoint).
CASES = [
    dict(label="FT-AL", base="C10_RN18_Same_FT-AL_0_1.0",
         pre_csv="./saved_logs/ft_final/MI_master_table_ft.csv",
         pre_model="CIFAR-10_ResNet-18_25000_Same_25000_42_1.0_FT-AL_ftsize=25000_ftseed=0",
         note="fine-tuned victim, all layers"),
    dict(label="Pruning 20%", base="C10_RN18_Same_Prune20-FT-AL_0_1.0",
         pre_csv="./saved_logs/pruning_final/MI_master_table_prune.csv",
         pre_model="CIFAR-10_ResNet-18_25000_Same_25000_42_1.0_sparsity=0.2_FT-AL_ftsize=25000_ftseed=0_ckpt=best",
         note="20% global L1 pruning + FT-AL recovery"),
    dict(label="DKD", base="CIFAR-10_ResNet-18to18_25000_DKD_0_0.0",
         pre_csv="./saved_logs/kd_final/MI_master_table_kd.csv",
         pre_model="CIFAR-10_ResNet-18to18_25000_DKD_0_0.0",
         note="decoupled KD, RN18 -> RN18"),
    dict(label="Knockoff", base="CIFAR-10_ResNet-18_25000_Knockoff_Same10_Same18_0_1.0",
         pre_csv="./saved_logs/extraction_final/MI_master_table_extraction.csv",
         pre_model="CIFAR-10_ResNet-18_25000_Knockoff_Same10_Same18_0_1.0",
         note="knockoff extraction, RN18 -> RN18"),
]

# eps is an ORDERED scale, so it takes a one-hue ramp, not categorical hues.
# Validated light->dark: monotone L, adjacent dL 0.237/0.142 (floor 0.06),
# lightest step 2.06:1 against the surface (floor 2.0).
EPS = [
    (0.007843, "2/255", "#86b6ef"),
    (0.015686, "4/255", "#256abf"),
    (0.031373, "8/255", "#104281"),
]
S_FIRST, S_LAST = 14, 100      # marker area: grows with epoch, so age is visible
LABEL_EPOCHS = (0, 10, 20, 29)  # selective direct labels, never one per point
LABEL_ON = 0.031373            # ...and only on one series, or they collide
_C = ["I(X;T)-In", "I(T;Y)-In"]


def _sel(path, **eq):
    d = pd.read_csv(path)
    d = d[(d.bins == BINS) & (d.in_size == IN_SIZE)]
    for k, v in eq.items():
        d = d[d[k] == v]
    return d


_traj = _sel(CSV_TRAJ, run_tag=RUN_TAG)
_neg = _sel(CSV_NEG, Scenario=POOL)
_neg = _neg[_neg.rate == 0.0]
_vic = _sel(CSV_VIC, Scenario=POOL)

P = _neg[_C].to_numpy()
if len(P) < 3:
    raise ValueError(f"negative pool {POOL!r} has {len(P)} rows -- check the name")
MU = P.mean(0)
SI = np.linalg.inv(np.cov(P.T))
_r = np.sort([np.sqrt((p - MU) @ SI @ (p - MU)) for p in P])
P95 = _r[int(0.95 * len(_r))]


def d95(xy):
    """Mahalanobis distance from the null, in units of the null's own p95."""
    v = np.asarray(xy, float) - MU
    return float(np.sqrt(v @ SI @ v)) / P95


def draw(ax, case):
    ax.scatter(P[:, 0], P[:, 1], s=34, c="#BBBBBB", alpha=0.75,
               edgecolors="none", zorder=1)

    xs = [P[:, 0].min(), P[:, 0].max()]
    ys = [P[:, 1].min(), P[:, 1].max()]

    # where this source sat before any AT: read first, so every trajectory can
    # be joined back to it. Epoch 0 is already one full epoch of AT away, and
    # on this recipe that first step is the largest one of the whole run --
    # without the connector the plot silently drops it.
    pre = _sel(case["pre_csv"])
    pre = pre[pre.model_name == case["pre_model"]]
    if pre.empty:
        raise ValueError(f"no pre-AT row for {case['label']}: {case['pre_model']}")
    pxy = pre[_C].to_numpy()[0]
    xs.append(pxy[0])
    ys.append(pxy[1])

    for eps, eps_label, colour in EPS:
        g = _traj[(_traj.base_model == case["base"]) & (_traj.eps == eps)]
        g = g.sort_values("epoch")
        if g.empty:
            print(f"[skip] no rows for {case['label']} eps={eps_label}")
            continue
        xy = g[_C].to_numpy()
        ep = g["epoch"].to_numpy()
        xs += [xy[:, 0].min(), xy[:, 0].max()]
        ys += [xy[:, 1].min(), xy[:, 1].max()]

        ax.plot([pxy[0], xy[0, 0]], [pxy[1], xy[0, 1]], ":", color=colour,
                lw=1.3, alpha=0.65, zorder=2)
        ax.plot(xy[:, 0], xy[:, 1], "-", color=colour, lw=1.6, alpha=0.85, zorder=3)
        sizes = np.linspace(S_FIRST, S_LAST, len(xy))
        ax.scatter(xy[:, 0], xy[:, 1], s=sizes, facecolors=colour,
                   edgecolors="white", linewidths=0.6, zorder=4)
        # a hollow ring marks epoch 0, so the start is findable without reading text
        ax.scatter(*xy[0], s=95, facecolors="none", edgecolors=colour,
                   linewidths=1.6, zorder=5)
        # direction: an arrowhead on the last step
        ax.annotate("", xy=xy[-1], xytext=xy[-2], zorder=5,
                    arrowprops=dict(arrowstyle="-|>", color=colour,
                                    lw=1.8, mutation_scale=16))
        for e in (LABEL_EPOCHS if eps == LABEL_ON else ()):
            hit = np.where(ep == e)[0]
            if len(hit):
                i = hit[0]
                ax.annotate(f"{e}", xy=xy[i], xytext=(5, 5),
                            textcoords="offset points", fontsize=8,
                            color="#444444", zorder=6)

    ax.scatter(*pxy, s=170, marker="D", facecolors="#F0E442",
               edgecolors="black", linewidths=1.0, zorder=7)
    ax.annotate(f"pre-AT  d={d95(pxy):.1f}", xy=pxy, xytext=(8, -14),
                textcoords="offset points", fontsize=9, color="#222222", zorder=7)

    vxy = _vic[_C].to_numpy()[0]
    xs += [vxy[0]]
    ys += [vxy[1]]
    ax.scatter(*vxy, s=330, marker="*", facecolors="#CC79A7",
               edgecolors="black", linewidths=0.8, zorder=7)

    padx = 0.05 * (max(xs) - min(xs))
    pady = 0.05 * (max(ys) - min(ys))
    ax.set_xlim(min(xs) - padx, max(xs) + padx)
    ax.set_ylim(min(ys) - pady, max(ys) + pady)
    ax.set_xlabel("I(X;T)  on group_A", fontsize=12)
    ax.set_ylabel("I(T;Y)  on group_A", fontsize=12)
    ax.set_title(f"{case['label']}   ({case['note']})", fontsize=13)
    ax.grid(True, lw=0.5, alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    handles = [Line2D([], [], color=c, lw=2.0, marker="o", ms=6,
                      markerfacecolor=c, markeredgecolor="white",
                      label=f"eps {lab}   d {d95(_traj[(_traj.base_model == case['base']) & (_traj.eps == e)].sort_values('epoch')[_C].to_numpy()[-1]):.1f} at ep29")
               for e, lab, c in EPS
               if not _traj[(_traj.base_model == case["base"]) & (_traj.eps == e)].empty]
    handles += [
        Line2D([], [], ls="none", marker="D", ms=10, markerfacecolor="#F0E442",
               markeredgecolor="black", label="pre-AT source (epoch -1)"),
        Line2D([], [], ls=":", color="#777777", lw=1.3, marker="o", ms=7,
               markerfacecolor="none", markeredgecolor="#777777",
               label="first AT epoch; ring = epoch 0"),
        Line2D([], [], ls="none", marker="*", ms=17, markerfacecolor="#CC79A7",
               markeredgecolor="black", label=f"victim   d={d95(vxy):.1f}"),
        Line2D([], [], ls="none", marker="o", ms=7, markerfacecolor="#BBBBBB",
               markeredgecolor="none", label=f"H0: negative pool (n={len(P)})"),
    ]
    ax.legend(handles=handles, fontsize=8.5, loc="best", framealpha=0.92)


SUB = (f"CIFAR-10 post-hoc adversarial training, mix=off · epoch runs ring (0) → arrow (29), "
       f"marker grows with it; dotted = the first AT epoch, from the pre-AT source\n"
       f"epoch numbers are on the 8/255 path only;  d = Mahalanobis distance from the null "
       f"in units of its own p95 ({P95:.2f} raw);  bins={BINS}, in_size={IN_SIZE}")

if ONE_FIGURE_PER_CASE:
    for case in CASES:
        fig, ax = plt.subplots(figsize=(8.2, 6.4))
        draw(ax, case)
        fig.suptitle(SUB, fontsize=10, y=1.005, color="#444444")
        fig.tight_layout()
        plt.show()
else:
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 12.5))
    for ax, case in zip(axes.ravel(), CASES):
        draw(ax, case)
    fig.suptitle(SUB, fontsize=12, y=1.01)
    fig.tight_layout()
    plt.show()
