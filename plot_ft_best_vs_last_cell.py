%matplotlib inline
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

clear_csv_cache()

CSV_FT  = "./saved_logs/ft_final/MI_master_table_ft.csv"
CSV_NEG = "./saved_logs/vanilla/MI_master_table_neg_pool0.csv"
CSV_VIC = "./saved_logs/vanilla/MI_master_table_victim.csv"

BINS, IN_SIZE = 50, 25000
MODEL_SEED, RATE, FT_SIZE = 42, 1.0, 25000
POOL_SEEDS = None

# Last-epoch MI, computed off-table: calculate_MI_ft.main_best was run UNCHANGED
# against a temp tree in which best_epoch.pth was hardlinked to each run's final
# epoch_<N>.pth. Same probe splits, same estimator, same nested subsets as the
# best-epoch rows in MI_master_table_ft.csv, so the two are directly comparable.
# Deliberately NOT written to the master table: its contract is one row per best
# checkpoint. Three ft_seeds per entry, (I(X;T)-In, I(T;Y)-In) at bins=50.
LAST_EPOCH = {
    ("CIFAR-100_ResNet-18_25000_Same_25000", "FT-LL"):
        [(7.781857, 6.643856), (7.820348, 6.643856), (7.767933, 6.643856)],
    ("CIFAR-100_ResNet-18_25000_Same_25000", "FT-AL"):
        [(8.063826, 6.631776), (8.070492, 6.630247), (8.050405, 6.627578)],
    ("CIFAR-100_ResNet-18_25000_Same_25000", "RT-AL"):
        [(9.157015, 6.441788), (9.115259, 6.435734), (9.157992, 6.445947)],
    ("CIFAR-100_VGG16_25000_Same_25000", "FT-LL"):
        [(7.776247, 6.643857), (7.805005, 6.643856), (7.798089, 6.643856)],
    ("CIFAR-100_VGG16_25000_Same_25000", "FT-AL"):
        [(9.539268, 6.561971), (9.496889, 6.564040), (9.614084, 6.567244)],
    ("CIFAR-100_VGG16_25000_Same_25000", "RT-AL"):
        [(9.427189, 6.368152), (9.467363, 6.367387), (9.451964, 6.373328)],
    ("CIFAR-100_DeiT_Distill_25000_Same_25000", "FT-LL"):
        [(10.546978, 6.643856), (10.578814, 6.643856), (10.572436, 6.643856)],
    ("CIFAR-100_DeiT_Distill_25000_Same_25000", "FT-AL"):
        [(8.341223, 6.627364), (8.336140, 6.628430), (8.338350, 6.630648)],
    ("CIFAR-100_DeiT_Distill_25000_Same_25000", "RT-AL"):
        [(9.544468, 6.554817), (9.525362, 6.548853), (9.576925, 6.549665)],
}
LAST_EP_IDX = {"FT-LL": 29, "FT-AL": 49, "RT-AL": 79}

ARCH = {
    "ResNet-18":         ("CIFAR-100_ResNet-18_25000",
                          "CIFAR-100_ResNet-18_25000_Same_25000"),
    "VGG16":             ("CIFAR-100_VGG16_25000",
                          "CIFAR-100_VGG16_25000_Same_25000"),
    "DeiT-Ti distilled": ("CIFAR-100_DeiT_Distill_25000",
                          "CIFAR-100_DeiT_Distill_25000_Same_25000"),
}
STRATEGY = {
    "FT-LL": ("#0072B2", "o"),
    "FT-AL": ("#009E73", "s"),
    "RT-AL": ("#D55E00", "^"),
}


def _sel(path):
    d = pd.read_csv(path)
    return d[(d.bins == BINS) & (d.in_size == IN_SIZE)]

_neg, _ft, _vic = _sel(CSV_NEG), _sel(CSV_FT), _sel(CSV_VIC)
if POOL_SEEDS is not None:
    _neg = _neg[_neg.seed.isin(POOL_SEEDS)]
_C = ["I(X;T)-In", "I(T;Y)-In"]


def _null(pool_scen):
    P = _neg[_neg.Scenario == pool_scen][_C].to_numpy()
    if len(P) < 3:
        raise ValueError(f"negative pool {pool_scen!r} has {len(P)} rows")
    mu, Si = P.mean(0), np.linalg.inv(np.cov(P.T))
    r = np.sort([np.sqrt((p - mu) @ Si @ (p - mu)) for p in P])
    return mu, Si, r[int(0.95 * len(r))], len(P)


def _d(mu, Si, p):
    v = np.asarray(p, float) - mu
    return float(np.sqrt(v @ Si @ v))


def best_re(scen, strategy):
    return (rf"{re.escape(scen)}_{MODEL_SEED}_{re.escape(str(RATE))}"
            rf"_{strategy}_ftsize={FT_SIZE}_ftseed=\d+$")


rows = []
fig, axes = plt.subplots(1, len(ARCH), figsize=(20, 5.9))
for ax, (arch, (pool_scen, ft_scen)) in zip(np.atleast_1d(axes), ARCH.items()):
    mu, Si, p95, npool = _null(pool_scen)

    specs = [TableGroupSpec(
        label=f"Negative pool  (n={npool})",
        csv_path=CSV_NEG, scenario=pool_scen, rates=[0.0], seeds=POOL_SEEDS,
        bins=BINS, in_size=IN_SIZE, domain="in", mode="points",
        style={"marker": "o", "color": "#BBBBBB", "s": 46,
               "alpha": 0.75, "edgecolors": "none", "zorder": 1},
    )]
    best_mean = {}
    for strat, (colour, marker) in STRATEGY.items():
        G = _ft[_ft.model_name.str.match(best_re(ft_scen, strat), na=False)][_C].to_numpy()
        if len(G) == 0:
            print(f"[skip] no best-epoch rows for {ft_scen} {strat}")
            continue
        best_mean[strat] = G.mean(0)
        specs.append(TableGroupSpec(
            label=f"{strat}  best  (n={len(G)}, d={_d(mu, Si, G.mean(0)):.0f})",
            csv_path=CSV_FT, model_name=re.compile(best_re(ft_scen, strat)),
            bins=BINS, in_size=IN_SIZE, domain="in", mode="points",
            style={"marker": marker, "color": colour, "s": 120, "alpha": 0.95,
                   "edgecolors": "black", "linewidths": 0.6, "zorder": 3},
        ))

    v = _vic[_vic.Scenario == pool_scen][_C].to_numpy()
    victim = VictimSpec(
        csv_path=CSV_VIC, scenario=pool_scen, seeds=[42], rates=[1.0],
        bins=BINS, in_size=IN_SIZE, domain="in",
        label=f"victim  (d={_d(mu, Si, v[0]):.0f})" if len(v) else "victim",
        style={"marker": "*", "color": "#CC79A7", "s": 320,
               "edgecolors": "black", "linewidths": 0.8, "zorder": 4},
    )

    plot_information_plane(
        specs, title=f"{arch}      null = its own rate-0.0 pool",
        victim=[victim], ax=ax, show=False, legend_outside=False,
        font_sizes={"title": 13, "xlabel": 13, "ylabel": 13,
                    "xtick": 11, "ytick": 11, "legend": 8},
    )

    # ---- overlay the last-epoch seeds, hollow, mean joined to the best mean --
    for strat, (colour, marker) in STRATEGY.items():
        pts = LAST_EPOCH.get((ft_scen, strat))
        if not pts or strat not in best_mean:
            continue
        L = np.asarray(pts, float)
        ax.scatter(L[:, 0], L[:, 1], marker=marker, s=150, facecolors="none",
                   edgecolors=colour, linewidths=2.0, zorder=5)
        bx, by = best_mean[strat]
        lx, ly = L.mean(0)
        ax.plot([bx, lx], [by, ly], color=colour, lw=1.4, ls="--", alpha=0.85, zorder=2)
        delta = np.array([lx, ly]) - np.array([bx, by])
        rows.append((arch, strat, len(L), LAST_EP_IDX[strat],
                     bx, by, _d(mu, Si, (bx, by)),
                     lx, ly, _d(mu, Si, (lx, ly)),
                     float(np.sqrt(delta @ Si @ delta)), p95))

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="o", color="#444444", markerfacecolor="none",
                          markeredgecolor="#444444", markeredgewidth=2.0,
                          markersize=10, linestyle="--"))
    labels.append("hollow = last epoch")
    ax.legend(handles, labels, fontsize=8, loc="best")

fig.suptitle(
    "Fine-tuning, CIFAR-100, 3 ft_seeds  -  best_epoch.pth (filled) vs the final "
    "epoch (hollow)\n"
    "dashed line joins the two group means;  d = Mahalanobis distance from the "
    f"arch's own null;  bins={BINS}, in_size={IN_SIZE}",
    fontsize=14, y=1.06)
fig.tight_layout()
plt.show()

print(f"{'arch':<18} {'strat':<6} {'n':>2} {'lastep':>6} "
      f"{'best mean (IXT, ITY)':<23} {'d':>6}  "
      f"{'last mean (IXT, ITY)':<23} {'d':>6} {'shift':>8} {'%p95':>7}")
print("-" * 112)
for arch, strat, n, lep, bx, by, db, lx, ly, dl, shift, p95 in rows:
    flag = "  <== unstable" if shift > 3 * p95 else ""
    print(f"{arch:<18} {strat:<6} {n:>2} {lep:>6} "
          f"({bx:8.4f}, {by:7.4f})  {db:6.1f}  "
          f"({lx:8.4f}, {ly:7.4f})  {dl:6.1f} {shift:8.2f} {100*shift/p95:6.0f}%{flag}")
