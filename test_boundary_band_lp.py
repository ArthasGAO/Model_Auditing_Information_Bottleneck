"""
Lp generalization tests for boundary_band.py, per
Lp_generalization_implementation_guide.md's mandatory sanity checks.

  1. S1 regression (p=2): first_order_batch_lp == first_order_batch element-wise
     (delta1, j_star, gradnorm, margin, ill). Float64 -> bitwise; float32 -> noise.
  2. S2 direction: r at p=2 == L2 residual -(g/||a||^2)a;
                   r at p=inf == FGSM direction -g sign(a)/||a||_1;
                   r at p=1  == single-coordinate (mass on argmax|a|).
  3. S2 identity: <a, r> == -margin for every p (the linearized-boundary condition).
  4. S3 post-hoc regression (p=2): posthoc_verify_batch_lp == posthoc_verify_batch.
  5. S1 distance identity: delta1 == margin / ||a||_q for the winner, every p.
  6. S6 metrics: lp_separability_metrics sane; AUROC(self,self)~0.5.

Run:  python test_boundary_band_lp.py
"""

import torch

from Model.ResNet_18 import ResNet18
from AdvAttack.boundary_band import (
    first_order_batch, first_order_batch_lp,
    posthoc_verify_batch, posthoc_verify_batch_lp,
    collect_boundary_stats_batched_lp, lp_separability_metrics,
    _dual_q, _dual_norm_rows,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}\n")


def ok(name, cond, extra=""):
    print(f"  {name:36s} {'OK' if cond else 'FAIL'}  {extra}")
    assert cond, name


def make_model(dtype=torch.float32):
    torch.manual_seed(0)
    return ResNet18(num_classes=10).to(device=device, dtype=dtype).eval()


torch.manual_seed(1)
X = torch.rand(48, 3, 32, 32, device=device)
Xd = X.double()

# ---------------------------------------------------------------
# 1: S1 regression at p=2 (float64 -> exact)
# ---------------------------------------------------------------
print("[S1 regression: first_order_batch_lp(p=2) == first_order_batch, float64]")
m64 = make_model(torch.float64)
old = first_order_batch(m64, Xd, topk=None)
new = first_order_batch_lp(m64, Xd, p=2.0, topk=None)
ok("y", torch.equal(old["y"], new["y"]))
ok("j_star", torch.equal(old["j_star"], new["j_star"]))
ok("ill", torch.equal(old["ill"], new["ill"]))
ok("margin_star", torch.equal(old["margin_star"], new["margin_star"]))
# delta1 / gradnorm: same math, but vector_norm(ord=2) vs .norm(p=2) differ by a
# float64 ULP in reduction order -> allclose, not bitwise. Report the max error.
dd = (old["delta1"] - new["delta1"]).abs().max().item()
gg = (old["gradnorm_star"] - new["gradnorm_star"]).abs().max().item()
ok("delta1 (allclose)", torch.allclose(old["delta1"], new["delta1"], rtol=1e-12, atol=1e-14),
   f"max|err|={dd:.2e}")
ok("gradnorm_star (allclose)", torch.allclose(old["gradnorm_star"], new["gradnorm_star"], rtol=1e-12, atol=1e-14),
   f"max|err|={gg:.2e}")
print()

# ---------------------------------------------------------------
# 2 + 3: S2 direction forms and the <a,r> = -margin identity
# ---------------------------------------------------------------
print("[S2 direction: closed forms per p, float64]")
for p in (2.0, float("inf"), 1.0):
    fo = first_order_batch_lp(m64, Xd, p=p, topk=None)
    a = fo["grad_star"].flatten(1)                  # winner's margin gradient
    b = fo["margin_star"]                            # >= 0
    r = fo["r_star"].flatten(1)
    q = _dual_q(p)

    # <a, r> == -b  (linearized boundary; guide S2)
    inner = (a * r).sum(dim=1)
    ok(f"p={p}: <a,r> == -margin", torch.allclose(inner, -b, atol=1e-8, rtol=1e-6),
       f"max|err|={(inner + b).abs().max().item():.2e}")

    if p == 2.0:
        r_l2 = -(b / a.pow(2).sum(dim=1)).unsqueeze(1) * a
        ok("p=2: r == -b a/||a||_2^2", torch.allclose(r, r_l2, atol=1e-9))
    elif p == float("inf"):
        r_fgsm = -(b / a.abs().sum(dim=1)).unsqueeze(1) * a.sign()
        ok("p=inf: r == -b sign(a)/||a||_1", torch.allclose(r, r_fgsm, atol=1e-9))
    elif p == 1.0:
        nz = (r != 0).sum(dim=1)                     # exactly one nonzero coord/sample
        idx_r = r.abs().argmax(dim=1)
        idx_a = a.abs().argmax(dim=1)
        ok("p=1: single nonzero coordinate", torch.all(nz == 1))
        ok("p=1: coord == argmax|a|", torch.equal(idx_r, idx_a))
print()

# ---------------------------------------------------------------
# 5: S1 distance identity  delta1 == margin / ||a||_q  (winner), every p
# ---------------------------------------------------------------
print("[S1 identity: delta1 == margin / ||grad g||_q, float64]")
for p in (2.0, 3.0, float("inf"), 1.0):
    fo = first_order_batch_lp(m64, Xd, p=p, topk=None)
    q = _dual_q(p)
    gnorm_q = _dual_norm_rows(fo["grad_star"].flatten(1), q)
    recomputed = fo["margin_star"] / gnorm_q.clamp_min(1e-12)
    ok(f"p={p}: delta1 matches", torch.allclose(fo["delta1"], recomputed, atol=1e-8, rtol=1e-6),
       f"max|err|={(fo['delta1'] - recomputed).abs().max().item():.2e}")
print()

# ---------------------------------------------------------------
# 4: S3 post-hoc regression at p=2
# ---------------------------------------------------------------
print("[S3 post-hoc: posthoc_verify_batch_lp(p=2) == posthoc_verify_batch, float64]")
fo2 = first_order_batch_lp(m64, Xd, p=2.0, topk=None)
ph_old = posthoc_verify_batch(m64, Xd, fo2["y"], fo2["j_star"],
                              fo2["margin_star"], fo2["grad_star"])
ph_new = posthoc_verify_batch_lp(m64, Xd, fo2["y"], fo2["j_star"], fo2["r_star"])
ok("ok", torch.equal(ph_old["ok"], ph_new["ok"]))
ok("n_viol", torch.equal(ph_old["n_viol"], ph_new["n_viol"]))
ok("min_other", torch.allclose(ph_old["min_other"], ph_new["min_other"], atol=1e-8))
ok("lin_resid", torch.allclose(ph_old["lin_resid"], ph_new["lin_resid"], atol=1e-8))
print()

# ---------------------------------------------------------------
# 6: S6 dimensionless metrics sanity (float32, realistic path)
# ---------------------------------------------------------------
print("[S6 separability metrics + cross-p unit-shift, float32]")
m32 = make_model()
Y = torch.randint(0, 10, (48,))
test = collect_boundary_stats_batched_lp(m32, (X[:32], Y[:32]), p=2.0, device=device,
                                         topk=None, posthoc=True, verbose=False)
fp = collect_boundary_stats_batched_lp(m32, (X[32:], Y[32:]), p=2.0, device=device,
                                       topk=None, verbose=False)
mt = lp_separability_metrics(test["delta1"], fp["delta1"])
ok("auroc in [0,1]", 0.0 <= mt["auroc"] <= 1.0, f"auroc={mt['auroc']:.3f}")
ok("recall in [0,1]", 0.0 <= mt["recall_at_fpr"] <= 1.0, f"recall@1%={mt['recall_at_fpr']:.3f}")
ok("AUROC(self,self) ~ 0.5",
   abs(lp_separability_metrics(test["delta1"], test["delta1"])["auroc"] - 0.5) < 1e-6)
ok("ph_* present under posthoc", "ph_ok" in test and test["ph_ok"].dtype == torch.bool)

# guide S6: Linf delta systematically smaller than L2 (unit shift, not separability)
d2 = collect_boundary_stats_batched_lp(m32, (X[:32], Y[:32]), p=2.0, device=device,
                                       topk=None, verbose=False)["delta1"]
dinf = collect_boundary_stats_batched_lp(m32, (X[:32], Y[:32]), p=float("inf"),
                                         device=device, topk=None, verbose=False)["delta1"]
ok("p=inf delta < p=2 delta (unit shift)",
   bool((dinf.median() < d2.median()).item()),
   f"median: L2={d2.median():.4f} Linf={dinf.median():.4f}")

print("\nAll Lp checks passed.")
