# `tf.experimental.numpy.cbrt(-0.0)` returns `+0.0` — the sign of zero is lost

**Component:** TensorFlow 2.21.0, `tensorflow/python/ops/numpy_ops/np_math_ops.py`, `cbrt`
(eager and `tf.function`; `float16`, `float32`, `float64`)
**Severity:** silent wrong value at a documented IEEE boundary. `cbrt` is an odd function, so
`cbrt(-0.0)` must be `-0.0`; the op returns `+0.0`, which flips any downstream use of the sign
(`1/cbrt(-0.0)` is `+inf` instead of `-inf`).
**Status:** present in TensorFlow 2.21.0 **and still present on `master`**
(`4acac40ffb50`, 2026-09-11) — see "Novelty" below.
**Found with:** Rondo's `tf.experimental.numpy` vs `tf.math` differential oracle
(`artifact/hunt-v2/tools_tf_numpy_diff.py`), which sweeps both implementations of the same op over
signed zeros, subnormals, infinities and NaN; the case was then minimised and certified with a
per-item reproducer.

---

## Summary

`np.cbrt` is odd: `cbrt(-x) == -cbrt(x)`, and for the two zeros `np.cbrt(-0.0)` is `-0.0`
(its sign bit is set). TF-NumPy implements `cbrt` as

```python
def cbrt(x):
  def f(x):
    # __pow__ can't handle negative base, so we use `abs` here.
    rt = math_ops.abs(x) ** (1.0 / 3)
    return array_ops.where_v2(x < 0, -rt, rt)
  return _scalar(f, x, True)
```

The sign is restored through the predicate `x < 0`. For `x = -0.0`, `-0.0 < 0` is **False** —
`-0.0` equals `0`, it is not less than it — so the negation is skipped and `rt = abs(-0.0) ** (1/3)
= +0.0` is returned unchanged.

```python
import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp

tnp.cbrt(tf.constant(-0.0, tf.float64))   # 0.0     <-- sign bit cleared
np.cbrt(np.float64(-0.0))                 # -0.0    <-- correct
1.0 / tnp.cbrt(tf.constant(-0.0))         # +inf    <-- wrong branch
1.0 / np.cbrt(np.float64(-0.0))           # -inf
```

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing replay(s).`
(40 checks per replay, each asserting the observed behaviour or a control that must stay correct.)

| dtype | `cbrt(-0.0)` | sign | `cbrt(+0.0)` | NumPy `cbrt(-0.0)` |
|---|---|---|---|---|
| `float16` | `0.0` | **`+`** | `0.0` | (no NumPy dtype; oddness broken) |
| `float32` | `0.0` | **`+`** | `0.0` | `-0.0` |
| `float64` | `0.0` | **`+`** | `0.0` | `-0.0` |

The defect is present on every execution lane, so it is not a scalar fast path:

| lane (`float64`) | `cbrt(-0.0)` |
|---|---|
| eager | `+0.0` |
| `tf.function` (no XLA) | `+0.0` |
| XLA (`jit_compile=True`) | `+0.0` |
| vector lane (`[-0.0, 0.0, -8.0, 8.0]`) | `[+0.0, +0.0, -2.0, +2.0]` |
| root cause: `tf.math.less(-0.0, 0.0)` | `False` |

Controls (all correct, on every dtype): `cbrt(8) = 2`, `cbrt(-8) = -2`, `cbrt(27) = 3`,
`cbrt(-27) = -3`, `cbrt(1) = 1`, `cbrt(+0.0) = +0.0`. The defect is therefore specific to the
sign of a negative zero, not a magnitude problem.

One downstream consequence makes the severity concrete: `tf.math.reciprocal(tnp.cbrt(-0.0))` is
`+inf`, whereas the reference (`1/np.cbrt(-0.0)`) is `-inf`. Any expression such as
`1 / cbrt(x - a)` silently takes the wrong-sided infinity exactly at the root `x == a`.

## Root cause (pointer)

`tensorflow/python/ops/numpy_ops/np_math_ops.py`, `cbrt`: the sign is taken from the *comparison*
`x < 0`, which cannot see a negative zero, instead of from the IEEE-754 **sign bit**. The same
mistake was fixed for `signbit` in this file (PR #125954, "Use the IEEE sign bit in
experimental.numpy.signbit", merged 2026-08-27) by bitcasting the float to a same-width integer and
testing the sign bit; `cbrt` was not updated and still carries the pattern. A dtype-preserving
`-rt` selection keyed on the sign bit (`array_ops.bitcast` + `math_ops.less`) fixes it the same way,
or the negation can be applied whenever `x` is negative *or* negative-zero.

## Novelty

* The defect is **still present on upstream `master`** at commit `4acac40ffb50` (2026-09-11); the
  current master `cbrt` body is byte-identical in the relevant line:
  `return array_ops.where_v2(x < 0, -rt, rt)`.
* `gh search issues/prs --repo tensorflow/tensorflow` for `cbrt` finds no report of a sign-of-zero
  problem; the only `cbrt` PRs are XLA/StableHLO accuracy and lowering work.
* The sibling defects that *were* fixed upstream are different code paths and do not cover this one:
  `signbit` (PR #125954, sign bit), `hypot` overflow/NaN (PR #124888, rewritten with scaling),
  `atan2` on subnormal float32 ([#118643](https://github.com/tensorflow/tensorflow/issues/118643),
  open), CPU `floor` for negative float32 subnormals ([#124151](https://github.com/tensorflow/tensorflow/issues/124151),
  PR #124232 open), and `tf.math.sign` on subnormals (own TF-004 draft).

## Suggested fix

```python
_SIGN_BIT_DTYPES = {dtypes.bfloat16: dtypes.int16, dtypes.float16: dtypes.int16,
                    dtypes.float32: dtypes.int32, dtypes.float64: dtypes.int64}

def cbrt(x):
  def f(x):
    rt = math_ops.abs(x) ** (1.0 / 3)
    negative = math_ops.less(array_ops.bitcast(x, _SIGN_BIT_DTYPES[x.dtype]), 0)
    return array_ops.where_v2(negative, -rt, rt)
  return _scalar(f, x, True)
```

(A regression test is simply `assertTensorsEqual(np.cbrt(-0.0), cbrt(-0.0))` including the sign
bit, across `float16`/`float32`/`float64`.)
