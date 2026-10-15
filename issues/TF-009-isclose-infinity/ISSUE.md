# `tf.experimental.numpy.isclose` / `allclose` invert the comparison of infinities

**Component:** TensorFlow 2.21.0, `tensorflow/python/ops/numpy_ops/np_math_ops.py`, `isclose` and
`allclose` (eager, `tf.function`, XLA; float32 and float64)
**Severity:** silent wrong boolean. `isclose(inf, inf)` returns **False** and `isclose(-inf, inf)`
returns **True** — both are the *opposite* of NumPy's documented behaviour, and the wrong answers are
plausible enough to pass unnoticed in validation, deduplication and test-helper code.
**Status:** present in TensorFlow 2.21.0 **and still present on `master`**
(`4acac40ffb50`, 2026-09-11) — see "Novelty" below.
**Found with:** Rondo's second `tf.experimental.numpy` differential surface
(`artifact/hunt-v2/tools_tf_numpy_misc_probe.py`), which sweeps comparisons, NaN-aware reductions and
array helpers against NumPy; minimised and certified with a self-checking reproducer.

---

## Summary

NumPy's `isclose` is documented as applying its tolerance formula **to finite values**:

> For finite values, isclose uses the following equation … `absolute(a - b) <= (atol + rtol * absolute(b))`

Non-finite values are settled first: a value is close to another when they are equal, and an infinity
equals only an infinity of the same sign.  So `np.isclose(inf, inf)` is `True`,
`np.isclose(-inf, inf)` is `False`, and these hold for `rtol=atol=0` as well.

TF-NumPy applies the tolerance formula unconditionally:

```python
def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
  def f(a, b):
    dtype = a.dtype
    if np.issubdtype(dtype.as_numpy_dtype, np.inexact):
      rtol_ = ops.convert_to_tensor(rtol, dtype.real_dtype)
      atol_ = ops.convert_to_tensor(atol, dtype.real_dtype)
      result = math_ops.abs(a - b) <= atol_ + rtol_ * math_ops.abs(b)
      ...
```

With infinities the formula produces the wrong answer in both directions:

| cell | why the formula fails | TF-NumPy | NumPy |
|---|---|---|---|
| `isclose(inf, inf)` | `abs(inf - inf) = nan`, and `nan <= inf` is False | **False** | `True` |
| `isclose(-inf, -inf)` | same | **False** | `True` |
| `isclose(-inf, inf)` | `abs(-inf - inf) = inf <= atol + rtol * inf = inf` | **True** | `False` |
| `isclose(inf, -inf)` | same | **True** | `False` |
| `isclose(1e308, inf)` | right-hand side is `atol + rtol * inf = inf` | **True** | `False` |

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing replay(s).`
(117 checks per replay; both sides of every cell are pinned — the live NumPy reference *and* the
recorded TF-NumPy answer — and finite inputs must agree everywhere.)

```python
import numpy as np, tensorflow as tf
import tensorflow.experimental.numpy as tnp

tnp.isclose(tf.constant(np.inf), tf.constant(np.inf))    # False   <-- should be True
tnp.isclose(tf.constant(-np.inf), tf.constant(np.inf))   # True    <-- should be False
np.isclose(np.inf, np.inf), np.isclose(-np.inf, np.inf)  # (True, False)
```

The scope of the defect, all asserted in the reproducer:

| dimension | values | result |
|---|---|---|
| dtype | `float32`, `float64` | inverted in both |
| input shape | scalars, vectors | inverted in both (`[inf, -inf, 1.0]` vs `[inf, inf, 1.0]` → `[False, True, True]`, NumPy `[True, False, True]`) |
| execution lane | eager, `tf.function`, XLA (`jit_compile=True`) | identical wrong answers |
| tolerance | defaults, `equal_nan=True` | inverted (`equal_nan` does not help: `nan` handling is already correct, `inf` is not) |
| tolerance | `rtol=0, atol=0` | still inverted for same-sign infinities |
| propagation | `allclose([inf], [inf])` → `False`; `allclose([-inf], [inf])` → `True` | both wrong |
| controls | `(1.0, 1.0)`, `(0.0, -0.0)`, `(0.0, 1e-12)`, `(1.0, 2.0)`, `(nan, nan)`, `(1e10, 1e10+1)` | all match NumPy, with and without tolerances |

The three root-cause checks in the reproducer show the mechanism directly:
`tf.abs(inf - inf)` is `nan`, `nan <= inf` is `False`, and `tf.abs(-inf - inf) <= inf` is `True`.

## Root cause (pointer)

`tensorflow/python/ops/numpy_ops/np_math_ops.py`, `isclose`: the float branch is
`math_ops.abs(a - b) <= atol_ + rtol_ * math_ops.abs(b)` with no non-finite guard, so the comparison
inherits IEEE's `inf - inf = nan` and `inf <= inf` instead of NumPy's "an infinity equals only
itself" rule.  `allclose` simply reduces `isclose`, so it inherits the defect.

A fix that matches the ufunc:

```python
same = math_ops.equal(a, b)                     # covers inf == inf, -inf == -inf, and finite equality
finite_pair = math_ops.is_finite(a) & math_ops.is_finite(b)
within_tol = math_ops.abs(a - b) <= atol_ + rtol_ * math_ops.abs(b)
result = same | (finite_pair & within_tol)
```

## Novelty

* The defect is **still present on upstream `master`** at commit `4acac40ffb50` (2026-09-11): the
  float branch of `isclose` is unchanged; the only recent change added an *integer* branch.
* `gh search issues/prs --repo tensorflow/tensorflow` for `isclose`, `allclose`, `isclose inf` and
  `isclose wrong result` finds no report of the infinity behaviour.
  [#108657](https://github.com/tensorflow/tensorflow/issues/108657) (closed) is
  "`tf.experimental.numpy.isclose` returns incorrect results for integer inputs" — a different
  branch of the same function, fixed by the integer path that is already on master.
* The sibling divergences found in the same sweep are **not** filed here because upstream already
  tracks them: `sort`/`argsort` place NaN first instead of NumPy's documented last
  ([#58496](https://github.com/tensorflow/tensorflow/issues/58496), closed), and the `clip`
  `a_min > a_max` rule is the `tf.clip_by_value` behaviour of
  [#57151](https://github.com/tensorflow/tensorflow/issues/57151) (closed).
