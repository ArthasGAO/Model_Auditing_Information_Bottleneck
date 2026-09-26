"""
Parity test: batched boundary-band estimator vs the original per-query loop.

Float64 is the rigorous check: the two paths are BITWISE identical there (verified
on cuda), because the math is the same and float64 kernels don't reorder. Float32
differs only by cuDNN batch-1-vs-batch-N kernel noise (reported, loosely bounded);
on a randomly initialized net this noise is relatively large because margins are
differences of nearly-equal logits, while on a trained model margins are O(1-10)
and the noise is negligible (checked against the real checkpoint when present).

Run:  python test_boundary_band_batch.py
"""

import time
from pathlib import Path

import torch

from Model.ResNet_18 import ResNet18
from AdvAttack.boundary_band import (
    collect_boundary_stats, collect_boundary_stats_batched,
    first_order_batch, _as_logits,
    posthoc_verify_batch,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}\n")


def check_close(name, a, b, rtol, atol):
    a, b = a.float(), b.float()
    assert torch.equal(torch.isfinite(a), torch.isfinite(b)), f"{name}: inf/nan mismatch"
    fin = torch.isfinite(a)
    diff = (a[fin] - b[fin]).abs()
    ok = torch.allclose(a[fin], b[fin], rtol=rtol, atol=atol)
    print(f"  {name:14s} max_abs_diff={diff.max().item():.3e}  {'OK' if ok else 'FAIL'}")
    assert ok, f"{name} mismatch beyond rtol={rtol}, atol={atol}"


def check_equal(name, a, b):
    ok = torch.equal(a, b)
    print(f"  {name:14s} {'exact match OK' if ok else 'FAIL'}")
    assert ok, f"{name} not identical"


def make_model(dtype=torch.float32):
    torch.manual_seed(0)
    return ResNet18(num_classes=10).to(device=device, dtype=dtype).eval()


torch.manual_seed(1)
X = torch.rand(64, 3, 32, 32)
Y = torch.randint(0, 10, (64,))

# ---------------------------------------------------------------
# 1: exact parity in float64 (first order, both topk modes, + curvature)
# ---------------------------------------------------------------
model64 = make_model(torch.float64)
X64, Y64 = X[:32].double(), Y[:32]
for topk in (None, 3):
    print(f"[float64 exact parity, topk={topk}, curvature=True]")
    old = collect_boundary_stats(model64, (X64, Y64), device=device, topk=topk,
                                 compute_curvature=True, verbose=False)
    new = collect_boundary_stats_batched(model64, (X64, Y64), device=device, topk=topk,
                                         compute_curvature=True, verbose=False)
    for k in ("y_pred", "j_star", "label", "ill"):
        check_equal(k, old[k], new[k])
    # float32-cast outputs of an identical float64 computation must match bitwise
    for k in ("delta1", "margin_star", "gradnorm_star", "delta2", "kappa", "correction"):
        check_equal(k, old[k], new[k])
    print()

# ---------------------------------------------------------------
# 2: float32 noise report (kernel-order effects only; argmins must still agree)
# ---------------------------------------------------------------
print("[float32, topk=None -- batch-1 vs batch-64 cuDNN kernel noise]")
model = make_model()
old = collect_boundary_stats(model, (X, Y), device=device, verbose=False)
new = collect_boundary_stats_batched(model, (X, Y), device=device, verbose=False)
for k in ("y_pred", "j_star", "ill"):
    check_equal(k, old[k], new[k])
check_close("delta1", old["delta1"], new["delta1"], rtol=0.5, atol=1e-2)
check_close("gradnorm_star", old["gradnorm_star"], new["gradnorm_star"], rtol=0.05, atol=1e-2)
print()

# ---------------------------------------------------------------
# 3: post-hoc verification vs naive per-sample reimplementation
# ---------------------------------------------------------------
print("[posthoc_verify_batch vs naive loop, float64]")
Xp = X64.to(device)
fo = first_order_batch(model64, Xp, topk=None)
ph = posthoc_verify_batch(model64, Xp, fo["y"], fo["j_star"],
                          fo["margin_star"], fo["grad_star"])

naive_min, naive_resid, naive_nviol = [], [], []
with torch.no_grad():
    for i in range(Xp.shape[0]):
        g, grad = fo["margin_star"][i], fo["grad_star"][i]
        r = -(g / grad.flatten().pow(2).sum()) * grad
        logits = _as_logits(model64((Xp[i] + r).unsqueeze(0)))[0]
        y, j = int(fo["y"][i]), int(fo["j_star"][i])
        g_all = logits[y] - logits
        others = [c for c in range(10) if c not in (y, j)]
        naive_min.append(min(float(g_all[c]) for c in others))
        naive_resid.append(float(g_all[j]))
        naive_nviol.append(sum(float(g_all[c]) < 0 for c in others))

check_close("min_other", ph["min_other"].cpu(), torch.tensor(naive_min), rtol=1e-8, atol=1e-10)
check_close("lin_resid", ph["lin_resid"].cpu(), torch.tensor(naive_resid), rtol=1e-8, atol=1e-10)
check_equal("n_viol", ph["n_viol"].cpu(), torch.tensor(naive_nviol))
check_equal("ok", ph["ok"].cpu(), torch.tensor(naive_min) >= 0)
print(f"  pass rate: {ph['ok'].float().mean().item():.2f}, "
      f"mean |lin_resid|: {ph['lin_resid'].abs().mean().item():.4f}")
print()

# ---------------------------------------------------------------
# 4: real trained checkpoint, if present (realistic margins + timing)
# ---------------------------------------------------------------
CKPT = Path("./saved_models/ft_vanilla/"
            "CIFAR-10_ResNet-18_25000_Same_10000_42_1.0_FT-AL_ftsize=10000_ftseed=0/"
            "epoch_49.pth")
if CKPT.exists():
    print(f"[trained checkpoint parity, {CKPT.parent.name}]")
    net = ResNet18(num_classes=10).to(device)
    net.load_state_dict(torch.load(CKPT, map_location=device))
    try:
        from util_adv import NormalizedModel
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        net = NormalizedModel(net, mean, std).to(device)
    except Exception as e:
        print(f"  (NormalizedModel unavailable, testing raw net: {e})")
    net.eval()

    torch.manual_seed(2)
    Xr = torch.rand(256, 3, 32, 32)
    old = collect_boundary_stats(net, Xr, device=device, verbose=False)
    new = collect_boundary_stats_batched(net, Xr, device=device, verbose=False)
    check_equal("y_pred", old["y_pred"], new["y_pred"])
    check_equal("j_star", old["j_star"], new["j_star"])
    # float32 batch-1 vs batch-256 kernel noise; bound it and report the spread
    rel = ((old["delta1"] - new["delta1"]).abs()
           / old["delta1"].abs().clamp_min(1e-8))
    worst = int(rel.argmax())
    print(f"  delta1 rel err: median={rel.median().item():.2e} "
          f"p99={rel.quantile(0.99).item():.2e} max={rel.max().item():.2e}")
    print(f"  (max is at delta1={old['delta1'][worst].item():.2e}, "
          f"margin={old['margin_star'][worst].item():.2e} -- relative error is "
          f"cancellation-dominated for near-boundary samples; abs diff there is "
          f"{(old['delta1'][worst] - new['delta1'][worst]).abs().item():.2e})")
    assert rel.quantile(0.99).item() < 0.05, "delta1 float32 kernel noise p99 exceeds 5%"
    assert (old["delta1"] - new["delta1"]).abs().max().item() < 5e-3, \
        "delta1 float32 absolute kernel noise exceeds 5e-3"
    print()

    print("[timing, 256 queries, topk=None, trained net]")
    times = {}
    for tag, fn, data in [("loop", collect_boundary_stats, Xr),
                          ("batched-128", collect_boundary_stats_batched,
                           torch.utils.data.DataLoader(
                               torch.utils.data.TensorDataset(Xr), batch_size=128))]:
        fn(net, Xr[:4], device=device, verbose=False)  # warm-up
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(net, data, device=device, verbose=False)
        if device == "cuda":
            torch.cuda.synchronize()
        times[tag] = time.perf_counter() - t0
        print(f"  {tag:12s} {times[tag]:.3f}s")
    print(f"  speedup: {times['loop'] / times['batched-128']:.1f}x")
else:
    print(f"[skipped trained-checkpoint test: {CKPT} not found]")

print("\nAll parity checks passed.")
