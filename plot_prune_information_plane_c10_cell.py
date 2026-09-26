%matplotlib inline
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

clear_csv_cache()

CSV_PR  = "./saved_logs/pruning_final/MI_master_table_prune.csv"
CSV_NEG = "./saved_logs/vanilla/MI_master_table_neg_pool0.csv"
CSV_VIC = "./saved_logs/vanilla/MI_master_table_victim.csv"

BINS, IN_SIZE = 50, 25000
POOL_SEEDS = None                    # all 80 per pool
MODEL_SEED, RATE, FT_SIZE = 42, 1.0, 25000
STRATEGY = "FT-AL"                   # pruning only has the one recovery arm
FT_SEEDS = [0, 1, 2]                 # ResNet-18 also has seeds 3,4; drop them so n matches

# One panel per ARCHITECTURE, like the fine-tuning cell: pruning does not change
# the architecture, so each suspect is tested against the pool its own victim
# belongs to.  arch -> (pool scenario == victim scenario, pruning Scenario_Name)
ARCH = {
    "ResNet-18":     ("CIFAR-10_ResNet-18_25000",
                      "CIFAR-10_ResNet-18_25000_Same_25000"),
    "VGG16":         ("CIFAR-10_VGG16_25000",
                      "CIFAR-10_VGG16_25000_Same_25000"),
    # CIFAR-10's transformer pool is the PLAIN DeiT, not the distilled one.
    "DeiT-Ti plain": ("CIFAR-10_DeiT_Plain_25000",
                      "CIFAR-10_DeiT_Plain_25000_Same_25000"),
}
# Two sparsity slots. Unlike CIFAR-100, every CIFAR-10 architecture survives a
# real 0.8 -- even VGG16, whose CIFAR-100 twin collapses to a constant there --
# so nothing is substituted and both levels are the genuine ones.
# (sparsity, colour, marker)
LEVELS = [("0.2", "#0072B2", "o"), ("0.8", "#D55E00", "s")]
LEVEL_SWAP = {}
TRUE_SPARSITY = {}
# calculate_MI_prune writes one row per checkpoint KIND, folded into model_name:
#   preft = epoch_-1.pth, pruned but not yet recovered
#   best  = best_epoch.pth, after the FT-AL recovery pass
# Only `best` is drawn: the preft points are a transient the attacker never
# ships, and the 0.8 ones sit at I(X;T) ~ 13 (ResNet-18 reaches d = 382), which
# stretched the x-axis until the recovered groups were unreadable. Set this to
# "preft" to look at the other state instead.
CKPT_KIND = "best"
CKPT_LABEL = {"preft": "pruned, before recovery", "best": "after FT-AL recovery"}


def _sel(path):
    d = pd.read_csv(path)
    return d[(d.bins == BINS) & (d.in_size == IN_SIZE)]

_neg = _sel(CSV_NEG)
if POOL_SEEDS is not None:
    _neg = _neg[_neg.seed.isin(POOL_SEEDS)]
_pr, _vic = _sel(CSV_PR), _sel(CSV_VIC)
_C = ["I(X;T)-In", "I(T;Y)-In"]


def _null(pool_scen):
    P = _neg[_neg.Scenario == pool_scen][_C].to_numpy()
    if len(P) < 3:
        raise ValueError(f"negative pool {pool_scen!r} has {len(P)} rows -- check the name")
    mu, Si = P.mean(0), np.linalg.inv(np.cov(P.T))
    r = np.sort([np.sqrt((p - mu) @ Si @ (p - mu)) for p in P])
    return mu, Si, r[int(0.95 * len(r))], len(P)


def _d(mu, Si, p):
    v = np.asarray(p, float) - mu
    return float(np.sqrt(v @ Si @ v))


def model_re(scen, sparsity, kind):
    """Must match main_prune's folder name + calculate_MI_prune's _ckpt= suffix."""
    seeds = "|".join(str(s) for s in FT_SEEDS)
    return (rf"{re.escape(scen)}_{MODEL_SEED}_{re.escape(str(RATE))}"
            rf"_sparsity={re.escape(sparsity)}_{STRATEGY}"
            rf"_ftsize={FT_SIZE}_ftseed=(?:{seeds})_ckpt={kind}$")


p95s = [_null(v[0])[2] for v in ARCH.values()]
fig, axes = plt.subplots(1, len(ARCH), figsize=(20, 5.9))
for ax, (arch, (pool_scen, pr_scen)) in zip(np.atleast_1d(axes), ARCH.items()):
    mu, Si, p95, npool = _null(pool_scen)

    specs = [TableGroupSpec(
        label=f"Negative pool  (n={npool})",
        csv_path=CSV_NEG, scenario=pool_scen, rates=[0.0], seeds=POOL_SEEDS,
        bins=BINS, in_size=IN_SIZE, domain="in", mode="points",
        style={"marker": "o", "color": "#BBBBBB", "s": 46,
               "alpha": 0.75, "edgecolors": "none", "zorder": 1},
    )]
    for slot, colour, marker in LEVELS:
        sp = LEVEL_SWAP.get(arch, {}).get(slot, slot)
        rex = model_re(pr_scen, sp, CKPT_KIND)
        G = _pr[_pr.model_name.str.match(rex, na=False)][_C].to_numpy()
        if len(G) == 0:
            print(f"[skip] no rows for {pr_scen}  sparsity={sp}  {CKPT_KIND}")
            continue
        shown = TRUE_SPARSITY.get(arch, {}).get(sp, sp)
        specs.append(TableGroupSpec(
            label=f"sparsity {shown}  (n={len(G)}, d={_d(mu, Si, G.mean(0)):.0f})",
            csv_path=CSV_PR, model_name=re.compile(rex),
            bins=BINS, in_size=IN_SIZE, domain="in", mode="points",
            style={"marker": marker, "color": colour, "s": 120, "alpha": 0.95,
                   "edgecolors": "black", "linewidths": 0.6, "zorder": 3},
        ))

    v = _vic[_vic.Scenario == pool_scen][_C].to_numpy()
    victim = VictimSpec(
        csv_path=CSV_VIC, scenario=pool_scen, seeds=[42], rates=[1.0],
        bins=BINS, in_size=IN_SIZE, domain="in",
        label=(f"victim (unpruned)  (d={_d(mu, Si, v[0]):.0f})" if len(v)
               else "victim (unpruned)"),
        style={"marker": "*", "color": "#CC79A7", "s": 320,
               "edgecolors": "black", "linewidths": 0.8, "zorder": 4},
    )

    note = ("  (high slot = 0.70; a real 0.8 collapses)"
            if arch in TRUE_SPARSITY else "")
    plot_information_plane(
        specs, title=f"{arch}      null = its own rate-0.0 pool{note}",
        victim=[victim], ax=ax, show=False, legend_outside=False,
        font_sizes={"title": 12, "xlabel": 13, "ylabel": 13,
                    "xtick": 11, "ytick": 11, "legend": 9},
    )

fig.suptitle(
    f"Global unstructured pruning on CIFAR-10  -  3 architectures x 2 sparsities\n"
    f"checkpoint shown: {CKPT_LABEL[CKPT_KIND]};  "
    "d = Mahalanobis distance of the group mean from the arch's own null "
    f"(pool p95 {min(p95s):.1f}-{max(p95s):.1f});  bins={BINS}, in_size={IN_SIZE}",
    fontsize=14, y=1.06)
fig.tight_layout()
plt.show()
