# The eager scalar cache conflates `+0.0` and `-0.0`: `tf.constant(-0.0)` returns `+0.0` if any `+0.0` scalar was created earlier in the process

**Component:** TensorFlow 2.21.0 (CPU), eager execution; `tf.constant`, `tf.convert_to_tensor`,
`tf.fill`, `tf.Variable` with scalar Python-`float` / `numpy.float64` values
**Severity:** silent wrong value with process-history dependence. `tf.constant(-0.0)` returns `+0.0`
once an unrelated `+0.0` scalar exists (and `tf.constant(0.0)` returns `-0.0` in the reverse order),
so `1.0 / tf.constant(-0.0)` becomes `+inf` instead of `-inf` and
`tf.math.sign(tf.constant(-0.0))` becomes `+0.0` instead of `-0.0`. Whether a program sees the
correct value depends on what other code created a zero scalar earlier, which makes results
non-reproducible across process histories.
**Status:** present in TensorFlow 2.21.0 (CPU, eager)
**Found with:** Rondo's cross-framework signed-zero differential - a control in the `tf.math.floormod`
probe disagreed with NumPy depending on the order in which the probe's constants were built.  The
standalone probe shipped next to this file reproduces the table above.

---

## Summary

TensorFlow's eager runtime keeps a **scalar cache** so that repeated scalar conversions return the
same `EagerTensor`.  The cache is reached from `convert_to_eager_tensor`
(`tensorflow/python/framework/constant_op.py`) and keyed by the scalar's value, for which
`-0.0 == 0.0` and `hash(-0.0) == hash(0.0)`.  The two zeros therefore share one entry, and the first
one created wins:

| call, in this process order | result |
|---|---|
| `tf.constant(-0.0)` (nothing else created) | `-0.0` |
| `a = tf.constant(0.0)` then `tf.constant(-0.0)` | `+0.0`  <-- wrong |
| `b = tf.constant(-0.0)` then `tf.constant(0.0)` | `-0.0`  <-- wrong |
| `a = tf.constant(3.5)` then `tf.constant(-0.0)` | `-0.0` (only the two zeros collide) |

`pywrap_tfe.TFE_ClearScalarCache()` restores the sign, which identifies the cache as the cause:

```python
from tensorflow.python import pywrap_tfe
tf.constant(0.0)
float(tf.constant(-0.0))              # 0.0    <-- wrong, numpy gives -0.0
pywrap_tfe.TFE_ClearScalarCache()
float(tf.constant(-0.0))              # -0.0
```

The observable consequence is a sign flip that survives into arithmetic:

```python
import numpy as np, tensorflow as tf
tf.constant(0.0)                      # any unrelated +0.0 scalar
x = tf.constant(-0.0)
float(x)                              # 0.0   (numpy: -0.0)
float(1.0 / x)                        # inf   (numpy: -inf)
float(tf.math.sign(x))                # 0.0   (numpy: -0.0)
```

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing replay(s).`
(26 checks per replay, 2 fresh processes, plus four order-dependent cases each run in its **own**
interpreter.)

What the reproducer pins:

| dimension | result |
|---|---|
| order dependence | in one process (cache cleared between rows, via `TFE_ClearScalarCache`): no prime → `-0.0`; prime `+0.0` → `+0.0`; prime `-0.0` then request `+0.0` → `-0.0`; prime `3.5` → `-0.0` |
| fresh-process oracle | four cases (`neg_only`, `pos_then_neg`, `neg_then_pos`, `nonzero_first`) each executed by a new interpreter, so the collision is a property of the process, not of the probe |
| lanes | `tf.constant(-0.0)`, `tf.constant(np.float64(-0.0))`, `tf.convert_to_tensor(-0.0)`, `tf.fill([1], -0.0)` and `tf.Variable(-0.0)` all collide after a same-type `+0.0` scalar; each returns `+0.0` |
| impact | `1.0 / tf.constant(-0.0)` is `+inf` (reference `-inf`), `tf.math.sign(tf.constant(-0.0))` is `+0.0` (reference `-0.0`) |
| mechanism | `pywrap_tfe.TFE_ClearScalarCache()` restores `-0.0`; the cache is type-scoped (`np.float64` collides with an `np.float64` prime but not with a Python-`float` prime) |
| controls | `tf.constant([-0.0])`, `tf.constant(np.array([-0.0]))` and `tf.constant(np.float32(-0.0))` keep the sign; `tf.constant(-0.0)` inside `@tf.function` (graph `Const`) keeps the sign and `1.0/tf.constant(-0.0)` there is `-inf`; a non-zero prime (`3.5`) does not interfere |

## Root cause (pointer)

The C++ eager scalar cache (the one exposed as `TFE_ClearScalarCache` /
`pywrap_tfe.TFE_ClearScalarCache`, used by `EagerContext` for scalar `EagerTensor` interning) keys
entries by the scalar value through equality/hash, and IEEE `-0.0 == 0.0` with
`hash(-0.0) == hash(0.0)` makes the two zeros indistinguishable.  `convert_to_eager_tensor`
(`tensorflow/python/framework/constant_op.py:75`) hands the Python `float`/`numpy.float64` straight
to `ops.EagerTensor`, so every scalar constant path (`tf.constant`, `tf.convert_to_tensor`, `tf.fill`,
`tf.Variable(0.0)`) is affected, while non-scalar and graph-constant paths are not.  A fix is to key
the cache on the bit pattern (e.g. compare `std::bit_cast<uint64_t>(value)` / `memcmp`) or to bypass
the cache when the value compares equal to zero; a regression test is
`signbit(tf.constant(-0.0)) == True` after creating `tf.constant(0.0)` in the same process.

## Additional user-visible symptom: gradients through a scaled input

The same cache corrupts a `-0.0` **arithmetic operand**, and the corruption is observable in a
gradient:

```python
def g(fn, x=1.0):
    xs = tf.Variable(tf.constant(x, tf.float64))
    with tf.GradientTape() as tape:
        y = fn(xs)
    return float(tape.gradient(y, xs).numpy())

g(lambda x: x * (-0.0))        # fresh process: -0.0 (correct; d/dx x*c = c = -0.0)
g(lambda x: x * 0.0)           # first call with a +0.0 operand
g(lambda x: x * (-0.0))        # -0.0 has become +0.0: the gradient is now +0.0
```

`tf.function` and XLA always return `-0.0` (the constant is built at trace time), and PyTorch, JAX
and MindSpore all return `-0.0` for the same expression, so this is another face of the eager
scalar-cache conflation rather than a gradient defect.

## Novelty

* `gh search issues --repo tensorflow/tensorflow` for `scalar cache`, `eager scalar cache`,
  `tf.constant negative zero`, `signed zero constant`, `convert_to_tensor negative zero`,
  `constant cache float` returns nothing on this mechanism; the closest hits are
  [#92103](https://github.com/tensorflow/tensorflow/issues/92103) (closed, GPU memory retained by
  rank-0 tensors - the cache as a *memory* issue), and the open signed-zero reports
  [#99759](https://github.com/tensorflow/tensorflow/issues/99759) (`tf.clip_by_value` on `-0.0`,
  CPU/GPU) and [#124151](https://github.com/tensorflow/tensorflow/issues/124151) (`tf.math.floor` on
  negative subnormals), which are per-operator and do not mention constant creation or the cache.
* A web search for the behaviour surfaces only API documentation pages for `tf.constant`, no issue.
* The campaign's other TensorFlow drafts cover per-operator kernels (`xlogy`/`xlog1py`,
  `maximum`/`minimum`, `sign`, complex `sqrt`/`reciprocal`, `cbrt`, `heaviside`, `isclose`,
  `erfinv`) or forward-mode AD; none is about the eager constant path.
