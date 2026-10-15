# `tf.experimental.numpy.heaviside` returns `x2` for NaN input instead of NaN

**Component:** TensorFlow 2.21.0, `tensorflow/python/ops/numpy_ops/np_math_ops.py`, `heaviside`
(eager, `tf.function`, and XLA; `float32`, `float64`)
**Severity:** silent wrong value — a NaN in the first argument is replaced by whatever the caller
passed as the second argument, so the NaN disappears instead of propagating. For the common
`heaviside(x, 0.5)` formulation a NaN silently becomes `0.5`.
**Status:** present in TensorFlow 2.21.0 **and still present on `master`**
(`4acac40ffb50`, 2026-09-11) — see "Novelty" below.
**Found with:** Rondo's `tf.experimental.numpy` vs `tf.math` differential oracle
(`artifact/hunt-v2/tools_tf_numpy_diff.py`), which compares the two independent implementations of
each op over signed zeros, subnormals, infinities and NaN; the case was then minimised and
certified with a per-item reproducer.

---

## Summary

`np.heaviside(x1, x2)` emits `0` where `x1 < 0`, `1` where `x1 > 0`, and `x2` where `x1 == 0`.
NumPy's ufunc has an explicit NaN clause: `np.heaviside(nan, x2)` is **always NaN**, for every `x2`
(verified for `0.0`, `-0.0`, `0.5`, `1.0`, `-2.0`, `2.0`, `1e-38`, `3.4e38`).

TF-NumPy implements only the three-branch formula:

```python
def heaviside(x1, x2):
  def f(x1, x2):
    return array_ops.where_v2(
        x1 < 0,
        constant_op.constant(0, dtype=x2.dtype),
        array_ops.where_v2(x1 > 0, constant_op.constant(1, dtype=x2.dtype), x2),
    )
  y = _bin_op(f, x1, x2)
  ...
```

Both `nan < 0` and `nan > 0` are **False**, so a NaN takes the `else` branch and the second argument
is returned verbatim.

```python
import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp

tnp.heaviside(tf.constant(float("nan")), tf.constant(0.5))   # 0.5    <-- NaN swallowed
np.heaviside(float("nan"), 0.5)                              # nan    <-- correct
```

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing replay(s).`
(40 checks per replay, each asserting the observed behaviour or a control that must stay correct.)

| `x2` | TF-NumPy `heaviside(nan, x2)` | NumPy |
|---|---|---|
| `0.0` | `0.0` | `nan` |
| `-0.0` | `-0.0` | `nan` |
| `0.5` | `0.5` | `nan` |
| `1.0` | `1.0` | `nan` |
| `-2.0` | `-2.0` | `nan` |
| `2.0` | `2.0` | `nan` |
| `1e-38` | `1e-38` | `nan` |
| `3.4e38` | `3.4e38` | `nan` |

Every lane shows it (vectors of `[nan, nan, -1, 0, -0, 1]` with `x2 = 0.5`):

| lane | result |
|---|---|
| eager | `[0.5, 0.5, 0.0, 0.5, 0.5, 1.0]` |
| `tf.function` | `[0.5, 0.5, 0.0, 0.5, 0.5, 1.0]` |
| XLA (`jit_compile=True`) | `[0.5, 0.5, 0.0, 0.5, 0.5, 1.0]` |
| NumPy | `[nan, nan, 0.0, 0.5, 0.5, 1.0]` |
| root cause: `less(nan, 0)` / `greater(nan, 0)` | `False` / `False` |

Controls (all correct, `float32` and `float64`): `heaviside(-1.0, 0.5) = 0`, `heaviside(1.0, 0.5) = 1`,
`heaviside(0.0, 0.5) = 0.5`, `heaviside(-0.0, 0.5) = 0.5`, `heaviside(-1e-30, 0.5) = 0`. The
non-NaN branches therefore match NumPy exactly; only the NaN case is wrong.

## Root cause (pointer)

`tensorflow/python/ops/numpy_ops/np_math_ops.py`, `heaviside`: the nested `where_v2` treats "neither
`< 0` nor `> 0`" as "equal to zero". NaN satisfies neither comparison but is not zero, so it falls
into the `x2` branch. An explicit NaN guard is needed before the two `where_v2` calls, e.g.
`array_ops.where_v2(math_ops.is_nan(x1), x1, <existing selection>)`, which also keeps the NaN's own
sign/dtype.

## Novelty

* The defect is **still present on upstream `master`** at commit `4acac40ffb50` (2026-09-11); the
  master `heaviside` body is still the two nested `where_v2` calls shown above.
* `gh search issues/prs --repo tensorflow/tensorflow` for `heaviside` finds nothing about NaN
  handling (the only hits are unrelated: an activation-function feature request, a gradient issue
  for `tf.math.floor`, and an Autograph failure).
* The one recent `heaviside` commit on this file (2026-08-26, "Fix AttributeError in tnp around,
  heaviside and average") changed only the `astype` → `math_ops.cast` fallback — it does not touch
  the NaN branch, and the tests it added cover non-NaN inputs.
* NumPy's NaN clause is not a rounding detail: it is the difference between a NaN that stops a
  pipeline and a plausible-looking constant that silently continues it.
