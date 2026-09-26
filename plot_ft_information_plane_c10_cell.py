%matplotlib inline
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

clear_csv_cache()

CSV_FT  = "./saved_logs/ft_final/MI_master_table_ft.csv"
CSV_NEG = "./saved_logs/vanilla/MI_master_table_neg_pool0.csv"
CSV_VIC = "./saved_logs/vanilla/MI_master_table_victim.csv"

BINS, IN_SIZE = 50, 25000
POOL_SEEDS = None            # all 80 per pool
MODEL_SEED, RATE, FT_SIZE = 42, 1.0, 25000
# CIFAR-10 was trained on the shortened budget (FT-LL 10 / FT-AL 25 /
# RT-AL 50 ep); the CIFAR-100 row used 30/50/80. The two datasets' FT
# distances are therefore NOT equally hard evasion attempts.

# One panel per ARCHITECTURE: fine-tuning does not change the architecture, so
# each suspect is tested against its own family's pool - the same pool its
# unmodified victim belongs to.
# arch -> (pool scenario == victim scenario, FT Scenario_Name)
ARCH = {
    "ResNet-18":     ("CIFAR-10_ResNet-18_25000",
                      "CIFAR-10_ResNet-18_25000_Same_25000"),
    "VGG16":         ("CIFAR-10_VGG16_25000",
                      "CIFAR-10_VGG16_25000_Same_25000"),
    # CIFAR-10's transformer pool is the PLAIN DeiT, not the distilled one.
    "DeiT-Ti plain": ("CIFAR-10_DeiT_Plain_25000",
                      "CIFAR-10_DeiT_Plain_25000_Same_25000"),
}
# strategy -> (colour, marker, what it touches)
STRATEGY = {
    "FT-LL": ("#0072B2", "o", "head only"),
    "FT-AL": ("#009E73", "s", "all layers"),
    "RT-AL": ("#D55E00", "^", "head re-init + all layers"),
}


def _sel(path):
    d = pd.read_csv(path)
    return d[(d.bins == BINS) & (d.in_size == IN_SIZE)]

_neg = _sel(CSV_NEG)
if POOL_SEEDS is not None:
    _neg = _neg[_neg.seed.isin(POOL_SEEDS)]
_ft = _sel(CSV_FT)
_vic = _sel(CSV_VIC)
_C = ["I(X;T)-In", "I(T;Y)-In"]


def _null(pool_scen):
    P = _neg[_neg.Scenario == pool_scen][_C].to_numpy()
    if len(P) < 3:
        raise ValueError(f"negative pool {pool_scen!r} has {len(P)} rows -- check the name")
    mu, Si = P.mean(0), np.linalg.inv(np.cov(P.T))
    r = np.sort([np.sqrt((p - mu) @ Si @ (p - mu)) for p in P])
    return mu, Si, r[int(0.95 * len(r))], len(P)


def model_re(scen, strategy):
    """Must match main_ft / main_ft_deit's scenario_name byte for byte."""
    return (rf"{re.escape(scen)}_{MODEL_SEED}_{re.escape(str(RATE))}"
            rf"_{strategy}_ftsize={FT_SIZE}_ftseed=\d+$")


def maha(mu, Si, frame, rex):
    G = frame[frame.model_name.str.match(rex, na=False)][_C].to_numpy()
    if len(G) == 0:
        return None, 0
    d = G.mean(0) - mu
    return float(np.sqrt(d @ Si @ d)), len(G)


p95s = [_null(v[0])[2] for v in ARCH.values()]

fig, axes = plt.subplots(1, len(ARCH), figsize=(20, 5.8))
for ax, (arch, (pool_scen, ft_scen)) in zip(np.atleast_1d(axes), ARCH.items()):
    mu, Si, _, npool = _null(pool_scen)
    specs = [TableGroupSpec(
        label=f"Negative pool  (n={npool})",
        csv_path=CSV_NEG, scenario=pool_scen, rates=[0.0], seeds=POOL_SEEDS,
        bins=BINS, in_size=IN_SIZE, domain="in", mode="points",
        style={"marker": "o", "color": "#BBBBBB", "s": 46,
               "alpha": 0.75, "edgecolors": "none", "zorder": 1},
    )]
    for strat, (colour, marker, what) in STRATEGY.items():
        rex = model_re(ft_scen, strat)
        d, n = maha(mu, Si, _ft, rex)
        if n == 0:
            print(f"[skip] no FT rows for {ft_scen} {strat} "
                  f"at bins={BINS}, in_size={IN_SIZE}")
            continue
        specs.append(TableGroupSpec(
            label=f"{strat}  {what}  (n={n}, d={d:.0f})",
            csv_path=CSV_FT, model_name=re.compile(rex),
            bins=BINS, in_size=IN_SIZE, domain="in", mode="points",
            style={"marker": marker, "color": colour, "s": 115, "alpha": 0.95,
                   "edgecolors": "black", "linewidths": 0.6, "zorder": 3},
        ))

    v = _vic[_vic.Scenario == pool_scen][_C].to_numpy()
    dv = float(np.sqrt((v[0] - mu) @ Si @ (v[0] - mu))) if len(v) else float("nan")
    victim = VictimSpec(
        csv_path=CSV_VIC, scenario=pool_scen,
        seeds=[42], rates=[1.0], bins=BINS, in_size=IN_SIZE, domain="in",
        label=f"victim (not fine-tuned)  (d={dv:.0f})",
        style={"marker": "*", "color": "#CC79A7", "s": 320,
               "edgecolors": "black", "linewidths": 0.8, "zorder": 4},
    )

    plot_information_plane(
        specs, title=f"{arch}      null = its own rate-0.0 pool",
        victim=[victim], ax=ax, show=False, legend_outside=False,
        font_sizes={"title": 13, "xlabel": 13, "ylabel": 13,
                    "xtick": 11, "ytick": 11, "legend": 9},
    )

fig.suptitle(
    f"Fine-tuning on the other 25000 CIFAR-10 images  -  3 architectures x 3 strategies\n"
    f"d = Mahalanobis distance of the group mean from its own null "
    f"(pool p95 radius {min(p95s):.1f}-{max(p95s):.1f});  bins={BINS}, in_size={IN_SIZE}",
    fontsize=14, y=1.06)
fig.tight_layout()
plt.show()
