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
POOL_SEEDS = None            # all 80 per pool

# Which pruning row to draw. The pruned model keeps the victim's architecture,
# so the victim's own pool is the null - no per-panel pool switching needed.
#   scenario   : Scenario_Name in the pruning table
#   pool/victim: scenario in the neg-pool / victim tables
#   ft_size    : the ftsize= field in the folder name
ROW = dict(
    label="CIFAR-100  ResNet-18",
    scenario="CIFAR-100_ResNet-18_25000_Same_25000",
    pool="CIFAR-100_ResNet-18_25000",
    victim="CIFAR-100_ResNet-18_25000",
    ft_size=25000,
    note="fine-tuned on the other 25000 CIFAR-100 images",
)
# The CIFAR-10 row is also computed; swap it in to look at that instead:
# ROW = dict(label="CIFAR-10  ResNet-18",
#            scenario="CIFAR-10_ResNet-18_25000_PseudoLabel_25000",
#            pool="CIFAR-10_ResNet-18_25000", victim="CIFAR-10_ResNet-18_25000",
#            ft_size=25000, note="fine-tuned on a pseudo-labelled ImageNet subset")

# calculate_MI_prune writes one row per checkpoint KIND, folded into model_name:
#   preft = epoch_-1.pth, pruned but not yet recovered
#   best  = best_epoch.pth, after FT-AL recovery
KINDS = [("preft", "pruned, before recovery"), ("best", "after FT-AL recovery")]
# sparsity -> (colour, marker)
SPARSITY = {
    "0.2": ("#0072B2", "o"),
    "0.8": ("#D55E00", "s"),
}
STRATEGY, MODEL_SEED, RATE = "FT-AL", 42, 1.0


def _sel(path):
    d = pd.read_csv(path)
    return d[(d.bins == BINS) & (d.in_size == IN_SIZE)]

_neg = _sel(CSV_NEG)
if POOL_SEEDS is not None:
    _neg = _neg[_neg.seed.isin(POOL_SEEDS)]
_pr = _sel(CSV_PR)
_C = ["I(X;T)-In", "I(T;Y)-In"]

P = _neg[_neg.Scenario == ROW["pool"]][_C].to_numpy()
if len(P) < 3:
    raise ValueError(f"negative pool {ROW['pool']!r} has {len(P)} rows -- check the name")
MU, SI = P.mean(0), np.linalg.inv(np.cov(P.T))
_r = np.sort([np.sqrt((p - MU) @ SI @ (p - MU)) for p in P])
P95, NPOOL = _r[int(0.95 * len(_r))], len(P)


def model_re(sparsity, kind):
    """Must match main_prune's folder name plus calculate_MI_prune's _ckpt= suffix."""
    return (rf"{re.escape(ROW['scenario'])}_{MODEL_SEED}_{re.escape(str(RATE))}"
            rf"_sparsity={re.escape(sparsity)}_{STRATEGY}"
            rf"_ftsize={ROW['ft_size']}_ftseed=\d+_ckpt={kind}$")


def maha(rex):
    G = _pr[_pr.model_name.str.match(rex, na=False)][_C].to_numpy()
    if len(G) == 0:
        raise ValueError(f"no pruning rows match {rex!r} -- check sparsity / ft_size")
    d = G.mean(0) - MU
    return float(np.sqrt(d @ SI @ d)), len(G)


fig, axes = plt.subplots(1, len(KINDS), figsize=(14.5, 5.8), sharex=True, sharey=True)
for ax, (kind, kind_label) in zip(np.atleast_1d(axes), KINDS):
    specs = [TableGroupSpec(
        label=f"Negative pool  (n={NPOOL})",
        csv_path=CSV_NEG, scenario=ROW["pool"], rates=[0.0], seeds=POOL_SEEDS,
        bins=BINS, in_size=IN_SIZE, domain="in", mode="points",
        style={"marker": "o", "color": "#BBBBBB", "s": 46,
               "alpha": 0.75, "edgecolors": "none", "zorder": 1},
    )]
    for sp, (colour, marker) in SPARSITY.items():
        rex = model_re(sp, kind)
        d, n = maha(rex)
        specs.append(TableGroupSpec(
            label=f"sparsity {sp}  (n={n}, d={d:.0f})",
            csv_path=CSV_PR, model_name=re.compile(rex),
            bins=BINS, in_size=IN_SIZE, domain="in", mode="points",
            style={"marker": marker, "color": colour, "s": 115, "alpha": 0.95,
                   "edgecolors": "black", "linewidths": 0.6, "zorder": 3},
        ))

    v = _sel(CSV_VIC)
    v = v[v.Scenario == ROW["victim"]][_C].to_numpy()
    dv = float(np.sqrt((v[0] - MU) @ SI @ (v[0] - MU))) if len(v) else float("nan")
    victim = VictimSpec(
        csv_path=CSV_VIC, scenario=ROW["victim"],
        seeds=[42], rates=[1.0], bins=BINS, in_size=IN_SIZE, domain="in",
        label=f"victim (unpruned)  (d={dv:.0f})",
        style={"marker": "*", "color": "#CC79A7", "s": 320,
               "edgecolors": "black", "linewidths": 0.8, "zorder": 4},
    )

    plot_information_plane(
        specs, title=f"{kind}  -  {kind_label}",
        victim=[victim], ax=ax, show=False, legend_outside=False,
        font_sizes={"title": 13, "xlabel": 13, "ylabel": 13,
                    "xtick": 11, "ytick": 11, "legend": 9},
    )

fig.suptitle(
    f"Global unstructured pruning, {ROW['label']}  -  {ROW['note']}\n"
    f"5 ft_seeds per group;  d = Mahalanobis distance of the group mean from the "
    f"{ROW['pool']} null (p95 radius {P95:.1f});  bins={BINS}, in_size={IN_SIZE}",
    fontsize=14, y=1.05)
fig.tight_layout()
plt.show()
