"""
Blind Walk diagnostics for the official-code DI path.

For every cached feature set under --features-root (all victims, all
fseed=/steps= sub-roots) report, per model and per split (private / public):

  unflipped %   walks that never crossed the boundary within the step budget
                (exact counts from meta.json when the adapter recorded them
                ["exact"]; otherwise a threshold on the distance: >= --cap-frac
                x the model's largest distance of that family ["threshold"]).
                The threshold is reliable for L_inf and, at the upstream budget
                of 50 steps, for L2/L1 as well (capped distances vary < 1.5%
                across images). With large budgets pixel clipping makes capped
                L2/L1 distances image-dependent and the threshold UNDERCOUNTS
                them; prefer rows marked "exact" (features extracted after
                2026-09-10 record the counts parsed from rand_steps' prints).
  step1 %       L_inf walks that "flipped" at the first step, i.e. images the
                model already misclassifies (distance <= 1.02 x noise scale)
  mean dist     mean recorded distance per family

Prints one table per victim root and writes a CSV (--out).

    python summarize_di_official_walks.py
    python summarize_di_official_walks.py --features-root saved_logs/di_official/files --out saved_logs/di_official/walk_summary.csv
"""
import argparse
import csv
import json
from pathlib import Path

import torch

FAMILIES = ("linf", "l2", "l1")
UNI_SCALE = 0.005   # upstream uniform-noise scale for CIFAR (0.01 for SVHN)


def load_dir(d: Path):
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").is_file() else {}
    priv = torch.load(d / "train_rand_vulnerability.pt")   # [N, 10, 3]
    pub = torch.load(d / "test_rand_vulnerability.pt")
    return meta, priv, pub


def split_stats(f: torch.Tensor, cap_ref: torch.Tensor, cap_frac: float, exact: dict | None):
    """f: [N, 10, 3] for one split; cap_ref: [3] largest distance per family for the model."""
    n_walks = f.shape[0] * f.shape[1]
    out = {}
    for k, fam in enumerate(FAMILIES):
        v = f[:, :, k]
        if exact and "unflipped" in exact:
            unflipped = int(exact["unflipped"][fam])
            src = "exact"
        else:
            unflipped = int((v >= cap_frac * cap_ref[k]).sum())
            src = "threshold"
        out[f"unflipped_{fam}"] = unflipped
        out[f"unflipped_{fam}_pct"] = 100.0 * unflipped / n_walks
        out[f"mean_{fam}"] = float(v.mean())
    out["unflipped_total_pct"] = 100.0 * sum(out[f"unflipped_{fam}"] for fam in FAMILIES) / (3 * n_walks)
    out["step1_pct"] = 100.0 * float((f[:, :, 0] <= 1.02 * UNI_SCALE).float().mean())
    out["walks_per_family"] = n_walks
    out["source"] = src
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-root", default="saved_logs/di_official/files")
    ap.add_argument("--out", default="saved_logs/di_official/walk_summary.csv")
    ap.add_argument("--cap-frac", type=float, default=0.95)
    args = ap.parse_args()

    root = Path(args.features_root)
    dirs = sorted(p for p in root.rglob("model_*_normalized")
                  if (p / "train_rand_vulnerability.pt").is_file() and (p / "test_rand_vulnerability.pt").is_file())
    if not dirs:
        print(f"no cached features under {root}")
        return

    rows = []
    by_root = {}
    for d in dirs:
        meta, priv, pub = load_dir(d)
        cap_ref = torch.maximum(priv.amax(dim=(0, 1)), pub.amax(dim=(0, 1)))
        sub_root = str(d.parent.parent.relative_to(root))          # victim=.../[fseed=k|steps=n]
        name = d.name[len("model_"):-len("_normalized")]
        base = {
            "root": sub_root,
            "feature_name": name,
            "scenario": meta.get("scenario_name", name),
            "suspect_arch": meta.get("suspect_arch", ""),
            "type": meta.get("suspect_type", "victim" if name == "teacher" else ""),
            "num_images": int(priv.shape[0]),
            "feature_seed": meta.get("feature_seed", ""),
            "attacks_sha256": (meta.get("vendored_attacks_sha256") or "")[:12],
        }
        for split, f, exact in (("private", priv, meta.get("walks_private")),
                                ("public", pub, meta.get("walks_public"))):
            rows.append({**base, "split": split, **split_stats(f, cap_ref, args.cap_frac, exact)})
        by_root.setdefault(sub_root, []).append(rows[-2:])

    # ---- terminal tables -------------------------------------------------
    for sub_root, pairs in by_root.items():
        print(f"\n=== {sub_root}  ({len(pairs)} models, cap threshold = {args.cap_frac:.2f} x max, "
              f"'exact' = counts parsed from rand_steps prints) ===")
        hdr = (f"{'scenario':52s} {'type':8s} {'split':7s} | {'Linf%':>6s} {'L2%':>6s} {'L1%':>6s} {'all%':>6s} "
               f"| {'step1%':>6s} | {'mLinf':>6s} {'mL2':>6s} {'mL1':>6s} | src")
        print(hdr)
        print("-" * len(hdr))
        tot = {fam: 0 for fam in FAMILIES}
        tot_walks = 0
        for pr, pu in pairs:
            for r in (pr, pu):
                print(f"{r['scenario'][:52]:52s} {r['type'][:8]:8s} {r['split']:7s} | "
                      f"{r['unflipped_linf_pct']:6.1f} {r['unflipped_l2_pct']:6.1f} {r['unflipped_l1_pct']:6.1f} "
                      f"{r['unflipped_total_pct']:6.1f} | {r['step1_pct']:6.1f} | "
                      f"{r['mean_linf']:6.3f} {r['mean_l2']:6.2f} {r['mean_l1']:6.0f} | {r['source']}")
                for fam in FAMILIES:
                    tot[fam] += r[f"unflipped_{fam}"]
                tot_walks += r["walks_per_family"]
        print("-" * len(hdr))
        print(f"{'ALL MODELS, BOTH SPLITS':52s} {'':8s} {'':7s} | "
              f"{100 * tot['linf'] / tot_walks:6.1f} {100 * tot['l2'] / tot_walks:6.1f} {100 * tot['l1'] / tot_walks:6.1f} "
              f"{100 * sum(tot.values()) / (3 * tot_walks):6.1f} |        |   ({3 * tot_walks} walks)")
        if any(r["source"] == "threshold" for pr, pu in pairs for r in (pr, pu)):
            print("  note: 'threshold' rows are approximate; with budgets above 50 steps the L2/L1 columns "
                  "undercount capped walks (clipping). Re-extract with the current adapter for exact counts.")

    # ---- csv ---------------------------------------------------------------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nCSV -> {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
