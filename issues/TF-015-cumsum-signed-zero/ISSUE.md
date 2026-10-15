# `tf.math.cumsum` drops the sign of IEEE `-0.0` (real, complex, eager/`tf.function`/reverse, even single-element)

**Component:** TensorFlow 2.21.0, CPU, `tf.math.cumsum` on `float32`/`float64` and `complex128`,
eager, `tf.function` and `reverse=True`
**Severity:** silent wrong sign of zero — even a **single-element** input, where no addition takes
place, comes back as `+0.0`; downstream `signbit`-sensitive code takes the opposite branch
**Status:** present in tensorflow 2.21.0 (CPU); NumPy (and MindSpore) preserve the sign
**Found with:** the wave-181 reduction grid, minimised to the sign-bit witnesses below; certified
reproducer `repro.py` in this directory (`--replays 2` -> `RESULT: 0 failing replay(s).`)

---

## Summary

```python
import numpy as np, tensorflow as tf
t = tf.constant([-0.0])
np.cumsum(np.asarray(t))                    # [-0.0]
tf.math.cumsum(t).numpy()                   # [ 0.0]     <- no addition happened, yet the sign is gone
tf.math.cumsum(tf.constant([-0.0, -0.0])).numpy()   # [0.0, 0.0]  (numpy [-0.0, -0.0])
```

| input | NumPy (sign bits) | `tf.math.cumsum` eager | `tf.function` | `reverse=True` |
|---|---|---|---|---|
| `[-0.0]` | `[1]` | `[0]` | `[0]` | `[0]` |
| `[-0.0, -0.0]` | `[1, 1]` | `[0, 0]` | `[0, 0]` | `[0, 0]` |
| `[-0.0, -0.0, -0.0]` | `[1, 1, 1]` | `[0, 0, 0]` | `[0, 0, 0]` | `[0, 0, 0]` |
| `[-0.0, 0.0, -0.0]` | `[1, 0, 0]` | `[0, 0, 0]` | `[0, 0, 0]` | `[0, 0, 0]` |
| `[-0.0, 1.0, -0.0]` | `[1, 0, 0]` | `[0, 0, 0]` | `[0, 0, 0]` | `[0, 0, 0]` |
| `float32 [-0.0, -0.0]` | `[1, 1]` | `[0, 0]` | `[0, 0]` | - |
| complex `[-0-0j]` | `[(1,1)]` | `[(0,0)]` | `[(0,0)]` | - |
| complex `[+0-0j, +0-0j]` | `[(0,1), (0,1)]` | `[(0,0), (0,0)]` | `[(0,0), (0,0)]` | - |

The single-element case is the sharpest: with one element the cumulative sum is the identity, so
returning `+0.0` is a pure copy/packing defect rather than an accumulation-order convention.

## Scope and controls

* Non-zero inputs (`[1, 2, 3]`, `[1.5, -2.5, 0.25]`, `[0, 0, 0]`) match NumPy exactly, and
  `[0.0, -0.0] -> [0.0, 0.0]` is correct (opposite-signed zeros add to `+0.0`).
* `reverse=True` also loses a sign, so both scan directions are affected; `tf.function` behaves like
  eager.
* `tf.math.cumsum` on `complex128` cannot be XLA-compiled in this build
  (`InvalidArgumentError: ... Cumsum ... attributes didn't match`, only `COMPLEX64` is registered),
  so the complex compiled lane is a separate coverage gap and is recorded rather than compared.

## The product scan shows the same identity loss for complex inputs

`tf.math.cumprod` is correct for real inputs (all eight real zero-sign cases match NumPy), but the
complex identity element loses the real sign: `cumprod([(-0-0j)])` returns `[(0,1)]` where NumPy
gives `[(1,1)]`, eagerly and under `tf.function`, and the same happens for the first element of a
two-element `[(-0-0j), (-0-0j)]` input.  A single-element cumulative product is the identity, so this
is a copy defect of the same class as the inclusive `cumsum` one.

## Impact

Any pipeline that cumulatively sums data containing negative zeros - differences of equal values,
products of zeros, explicitly padded `-0.0` - gets `+0.0` prefixes, so `signbit` tests, `1/prefix`,
`angle` and branch-cut-sensitive functions silently take the wrong branch.  Because even the identity
case (one element) is wrong, the defect is reachable with the smallest possible input.

## Novelty

Searched the TensorFlow tracker for `cumsum negative zero`, `cumsum signed zero`, `cumsum -0.0` and
`scan negative zero sign`: no report exists.  The two cumsum issues found (`#58790` ragged-inf
propagation, `#61975` "weird behaviour") are unrelated value/ragged cases.  PyTorch has the same
class of defect in `torch.cumsum` (pytorch#195851, open) but that is a different project, and the
campaign's TensorFlow drafts (TF-001…014) do not mention `cumsum` or zero signs in it.

## Environment

TensorFlow 2.21.0 CPU wheel, NumPy 2.5.1, Python 3.12.13, WSL2 (Linux 6.6.87.2) x86_64.
Full dump in `collect_env.txt`; transcripts in `evidence.txt`.

---

## Anticipated objections and responses

* **"Zero signs are not part of the contract."**  NumPy and MindSpore preserve them, TensorFlow's own
  `tf.math.add`/`reduce_sum` paths do too, and the single-element `cumsum` is required to be the
  identity; silently returning `+0.0` there is a defect independent of any accumulation convention.
* **"Use `tf.math.cumsum(x, exclusive=True)`."**  That intentionally starts at `+0.0`, but it does not
  explain the inclusive identity case above.
