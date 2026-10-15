# `binary_cross_entropy_with_logits` returns exactly 0 for large negative logits (the softplus tail is cancelled away)

**Component:** PyTorch 2.12.1+cu130, CPU: `torch.nn.functional.binary_cross_entropy_with_logits` and
`torch.nn.BCEWithLogitsLoss`, `float32` and `float64`.
**Severity:** silent wrong loss value.  For a confidently-correct prediction on the negative side the
loss is the softplus `log1p(exp(-|z|))`, and PyTorch returns **exactly `0.0`** for `|z| >= 20`
(float32) and `|z| >= 40` (float64) although the exact values (`2.06e-09`, `4.25e-18`) are normal
numbers; near the cutoff the reported loss is quantised at the ULP of `|z|` (0.83% high at `|z| = 10`).
The **gradient is the sigmoid and is correct**, so training is unaffected - the *reported loss* is
wrong, which matters for loss weighting, multi-term objectives, thresholds, early stopping and
hard-example mining by loss value.
**Status:** present in 2.12.1+cu130 on CPU, both widths, and **asymmetric**: `z = +20, y = 1` returns
the exact `2.06115369216775e-09`, so only the negative side cancels.  TensorFlow's
`tf.nn.sigmoid_cross_entropy_with_logits` is exact at every tested point in both widths, and
`torch.nn.functional.softplus(-|z|)` is exact in the same framework, so a correct implementation is
available.
**Found with:** the wave-229 logit-tail oracle (the BCE-with-logits loss against the closed form
`log1p(exp(-|z|))`, with `softplus` and the cancelled form as controls and TensorFlow as the
cross-framework reference); certified reproducer `repro.py` in this directory
(`--replays 2` -> `RESULT: 0 failing replay(s).`, 43 checks).

---

## Summary

```python
import torch, torch.nn.functional as F, math
z, y = -10.0, 0.0
F.binary_cross_entropy_with_logits(torch.tensor([z]), torch.tensor([y]))   # 4.57763671875e-05
F.softplus(torch.tensor([z]))                                             # 4.5398901420412585e-05
math.log1p(math.exp(-abs(z)))                                             # 4.539889921686465e-05

F.binary_cross_entropy_with_logits(torch.tensor([-20.0]), torch.tensor([0.0]))  # 0.0
math.log1p(math.exp(-20.0))                                                     # 2.061153620314381e-09

F.binary_cross_entropy_with_logits(torch.tensor([20.0]), torch.tensor([1.0]))   # 2.06115369216775e-09
```

| z (y on the correct side) | exact `log1p(exp(-|z|))` | PyTorch loss | TensorFlow loss |
|---|---|---|---|
| -1 | 3.1326168751822286e-01 | 3.132617473602295e-01 (1.9e-07) | exact |
| -5 | 6.715348489118068e-03 | 6.715297698974609e-03 (7.6e-06) | exact |
| -10 | 4.539889921686465e-05 | **4.57763671875e-05 (8.3e-03)** | exact |
| -20 | 2.061153620314381e-09 | **0.0 (relative error 1)** | exact |
| -30 | 9.357622968839737e-14 | **0.0** | exact |
| -40 | 4.248354255291589e-18 | **0.0** (float32 **and float64**) | exact |
| +20 (y=1) | 2.061153620314381e-09 | 2.06115369216775e-09 (accurate) | exact |
| +40 (y=1) | 4.248354255291589e-18 | 4.24835413113866e-18 (accurate) | exact |

The softplus computed directly is exact at every one of those points:
`F.softplus(torch.tensor([-40.0])) = 4.24835413113866e-18` (fp32) and
`F.softplus(torch.tensor([-40.0], dtype=torch.float64)) = 4.248354255291589e-18`.

## Root cause

At `|z| >= 5` the negative-side loss is **bit-for-bit** `softplus(|z|) - |z|` (for example
`z = -10`: the loss and `F.softplus(tensor([10.0])) - 10.0` are both exactly
`4.57763671875e-05`), i.e. the kernel evaluates the softplus term through a cancelled expression
instead of `log1p(exp(-|z|))`.  The small term is rounded away inside `|z| + small` before the
subtraction, so it is lost as soon as `exp(-|z|)` is below half an ULP of `|z|`: `|z| >= 20` in
float32 (ULP(20) = 1.9e-06) and `|z| >= 40` in float64 (ULP(40) = 7.1e-15), both exactly where the
observed zeros start.  Between the exact region and the zero region the subtraction quantises the
result at that same ULP, which is why `|z| = 10` is 0.83% high (an absolute error of ~9.5e-07-ish on
a `4.5e-05` result).

The positive side does not cancel (the kernel's `max_val` term is zero there), which is why
`z = +20, y = 1` keeps the tail - the same mathematical quantity is computed two different ways
inside one operator.

## Impact

* A confidently-correct prediction on the negative side reports **exactly zero loss**, and between
  `|z| = 10` and `20` the reported value is a staircase rather than a smooth function of the logit.
  `binary_cross_entropy_with_logits` is the standard numerically-stable binary loss, so any code that
  weights samples by loss, mines hard examples, thresholds on loss, combines the BCE term with other
  terms, or logs per-sample losses sees a wrong (and asymmetric) number.
* The framework is inconsistent with itself: the mathematically identical
  `F.softplus(-|z|)`/manual `log1p(exp(-|z|))` formula is exact at every affected point, and
  TensorFlow is exact at every point in both widths.
* The gradient is correct, so a gradient check passes and training converges normally - the defect is
  invisible to the usual sanity checks.

## Suggested fix

Compute the softplus term as `log1p(exp(-|z|))` (a `log1p`-based kernel, the way
`torch.nn.functional.softplus` already does) instead of evaluating a cancelled form, and add a
regression test on the tail:

```python
z = torch.tensor([-20.0]); y = torch.tensor([0.0])
assert float(F.binary_cross_entropy_with_logits(z, y)) == float(F.softplus(z))   # currently 0.0 vs 2.06e-09
```

## Controls (all in `repro.py`, 43 checks, 0 failing on each of 2 replays)

* The exact reference is the closed form `log1p(exp(-|z|))`; every affected point is asserted to be a
  normal number, so the zeros are not underflow.
* The bit-for-bit identity with the cancelled form `softplus(|z|) - |z|` is asserted at `z = -5, -10,
  -20`, and the difference from it at `z = -1` is asserted to be rounding-level only.
* `F.softplus(-|z|)` is asserted accurate to `1e-6` relative at `z = -1 ... -40` in both widths, so the
  loss path (not the arithmetic) is the defect.
* The positive side is asserted to keep the tail (`z = +20, +40` within `1e-6`), which is what makes
  the defect asymmetric.
* The gradient is asserted to be the exact sigmoid at `z = -20, -40`, isolating the forward value.
* The controls `|z| = 1` (1e-6) and `|z| = 5` (1e-4) show the formula is fine away from the tail, and
  `nn.BCEWithLogitsLoss` plus all three reductions are checked to show the same value.
* `float32` and `float64` are both covered.

## Novelty

PyTorch tracker searches for `binary_cross_entropy_with_logits zero` / `underflow` / `wrong`,
`BCEWithLogitsLoss precision large logits` and `bce_with_logits 0` return nothing on this observable
(the only hits are an unrelated 2018 memory leak and a 2017 TypeError).  The campaign's ledger has no
PyTorch BCE-with-logits record; its neighbouring entries are MS-031 (a *different* defect: the
single-element `binary_cross_entropy` reduction doubling, found in the same week) and the wave-11
`bce fp64` support note.  MindSpore's `binary_cross_entropy_with_logits` shows the same cancelled
tail (and also loses it on the positive side), which is recorded in this wave's ledger entry as a
shared-class observation rather than a separate draft.

## Environment

PyTorch 2.12.1+cu130 (CPU execution), NumPy 2.5.3, Python 3.12.13, WSL2 (Linux 6.6.87.2) x86_64.
TensorFlow 2.21.0 (CPU) for the cross-framework reference.  Full dump in `collect_env.txt`;
transcripts in `evidence.txt`.

---

## Anticipated objections and responses

* **"A zero loss for a confidently correct prediction is desirable."**  The gradient is zero anyway,
  so the *training* effect is nil; the defect is the reported value: it is exactly 0 while the true
  value is `2.06e-09`, it is 0.83% wrong at `|z| = 10`, it is asymmetric, TensorFlow returns the exact
  value, and `F.softplus` in the same framework returns the exact value.  Any loss-based weighting or
  thresholding inherits the error.
* **"This is just float32 underflow."**  `2.06e-09` and `4.25e-18` are normal float32/float64 numbers;
  the zeros appear where `exp(-|z|)` falls below an ULP of `|z|`, which is a cancellation in the
  kernel's expression, not an exponent-range limit.
* **"Only the tail is affected."**  The tail is exactly zero, and the region just above it is
  quantised at the ULP of `|z|` - the 0.83% error at `|z| = 10` is well inside the range where
  softplus is normally used.
* **"MindSpore does the same."**  Yes, and that is recorded - it does not change what PyTorch returns;
  TensorFlow shows an exact implementation is straightforward.
