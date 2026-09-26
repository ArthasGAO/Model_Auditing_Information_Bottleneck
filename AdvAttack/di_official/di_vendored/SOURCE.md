# Provenance of the vendored Dataset Inference code

Upstream: https://github.com/cleverhans-lab/dataset-inference
Commit:   24baf4610d53677a02d1212e66279a424aea9c30 ("Adding Table Generator", 2022-10-10)
Local clone used for the copy: `E:\dataset-inference`
Paper: Maini, Yaghini, Papernot. *Dataset Inference: Ownership Resolution in Machine Learning*. ICLR 2021.

Upstream environment (requirements.txt): python 3.8, torch 1.8.1, scipy 1.5.2, numpy 1.19.2.
Evaluation environment here: python 3.12, torch 2.9.0+cu128, scipy 1.16.2, numpy 2.2.6.

| Vendored file | Upstream file | Upstream sha256 | Status |
|---|---|---|---|
| `attacks.py` | `src/attacks.py` | `c60a9d4c26699b17aeac5d3f208a372d4f2439e05447e6fbb3ca6fcf2f251714` | byte-level copy with ONE torch-2.x patch applied at two lines (below); vendored sha256 `ef6fb66a2ae3aa2880bb105c5f5727e84ec6725d08dd9d179cc00dc73f137669` |
| `generate_features.py` | `src/generate_features.py` | `63e2d9d1b456ec7c5b9fcf558212641f9a33ff6d828604e1dad5924d3d604f6b` | `get_random_label_only` copied verbatim; other functions and CLI omitted; imports rewritten; `device` injected (see file header) |
| `notebook_protocol.py` | `src/notebooks/CIFAR10_rand.ipynb` | `52190de66a4a2901b017f37b3c012496e483958c46eac5d79a9ae9c1e9856f1c` | cells 3, 8-18, 20, 29 transcribed verbatim into functions (see file header, items a-g) |
| `notebook_protocol.py` (`generate_table`) | `src/notebooks/utils.py` | `0c28cd0bc59d38f237f81763ac3b77487d6c00956adf1c17dded3b82f1bb8d82` | verbatim except `tqdm.notebook` -> `tqdm.auto`, `ipdb.set_trace()` -> `raise` |

## The single patch in `attacks.py` (torch version compatibility)

`diff -u src/attacks.py di_vendored/attacks.py`:

```diff
@@ -153,7 +153,7 @@            (rand_steps, the Blind Walk used here)
             if t>0: 
                 preds = model(X_r+delta_r)
                 new_remaining = (preds.max(1)[1] == y[remaining])
-                remaining[remaining] = new_remaining
+                remaining[remaining.clone()] = new_remaining
@@ -188,7 +188,7 @@            (mingd, white-box path, not used by this evaluation)
             remaining_temp = remaining.clone()
-            remaining[remaining] = new_remaining
+            remaining[remaining.clone()] = new_remaining
```

Why: torch 1.8.1 accepted a masked in-place write whose mask is the tensor being
written (`remaining[remaining] = ...`). torch >= 1.10 raises
`RuntimeError: unsupported operation: some elements of the input tensor and the
written-to tensor refer to a single memory location. Please clone() the tensor`.
Cloning the mask reproduces the 1.8 semantics exactly (the mask is evaluated
before the write in both versions); no number changes. The same line in `mingd`
is patched for consistency although only `rand_steps` is called.

## Environment-level shims (no file changes)

* `ipdb` is not installed. `attacks.py` imports it at module top and only
  references it in commented-out lines, so `di_vendored/__init__.py` registers
  a placeholder module before the import. If `ipdb` is installed the real
  package is used.
* `attacks.py` creates CUDA tensors at import time (`mu`, `std`, `none_std`)
  and `rand_steps` calls `.cuda()` for the Gaussian noise. Evaluation therefore
  requires a CUDA device and runs on `cuda:0`.
* `generate_features.device` is set by the adapter before
  `get_random_label_only` is called (upstream assigns it in `__main__`).
* The adapter wraps the feature call in `torch.no_grad()` (upstream runs the
  two extra `model(X)` / `model(X+delta)` evaluations with autograd on but never
  uses the graph; the walk itself is under `no_grad` upstream).

## Upstream quirks that are preserved on purpose

* `train_{rand}_vulnerability.pt` naming: upstream `generate_features.py`
  saves `*_vulnerability_2.pt` while the notebook loads `*_vulnerability.pt`.
  The adapter writes the notebook's name.
* Notebook cell 9 flattens the `[1000, 10, 3]` feature tensor with
  `.T.reshape(1000, 30)`. Because `.T` reverses all dimensions, each of the
  1000 resulting rows holds the distances of 30 consecutive images under ONE
  noise family (rows 0-332: L_inf, 333-665: L2, 666-999: L1), not the
  30-dimensional per-image vector described in the paper. Rows `[:split_index]`
  train the regressor, rows `[split_index:]` are scored. This is what the
  released code does and it is kept unchanged.
* `rand_steps` increments the step multiplier once more after the last check,
  so walks that never flip end at 51x the base noise, not 50x.
* `rand_steps` draws the Laplace noise with `np.random.laplace` (NumPy RNG)
  while the uniform / Gaussian noise use the torch RNG. Upstream seeds only
  torch; the adapter seeds both.
* The notebook draws `generate_table`'s random row subsets from the global
  torch RNG in `names` order. The adapter resets the seed before each model's
  draws (`per_name_reseed=True`) so a row does not depend on which other
  models are in the same run; pass `per_name_reseed=False` for the notebook's
  sequential stream.
