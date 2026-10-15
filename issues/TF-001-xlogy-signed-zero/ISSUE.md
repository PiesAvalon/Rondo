# `tf.math.xlogy` / `tf.math.xlog1py` return a **size-dependent sign of zero** for `-0.0` (scalar path returns `+0.0`, vectorized path returns `-0.0`, XLA returns `+0.0`)

**Component:** TensorFlow CPU kernels — `tensorflow/core/kernels/cwise_ops.h`
(`xlogy_op`, `xlog1py_op`), eager and `tf.function` / `tf.function(jit_compile=True)`
**Severity:** silent wrong bit-level result, **depending on tensor size** (low–medium; it changes
`1/x`, `signbit`, `copysign`, complex phase and any downstream sign-sensitive computation)
**Status:** present in TensorFlow 2.21.0 (CPU)
**Found with:** Rondo identity/edge-value differential hunt (`artifact/hunt-v2`)

---

## Summary

`tf.math.xlogy(-0.0, y)` should be `-0.0 * log(y)`; `-0.0 * log(2.0) = -0.0`, which is what
NumPy/SciPy return.  TensorFlow returns

* `+0.0` when the tensor is smaller than one SIMD packet (scalar code path), and
* `-0.0` when the tensor fills at least one packet (vectorized code path),

so the *sign* of the result depends on the number of elements.  `tf.function(jit_compile=True)`
(XLA) consistently returns `+0.0`, disagreeing with the vectorized eager path for the same input.

```python
import numpy as np, tensorflow as tf
for n in (4, 8):
    x = tf.constant([-0.0] * n, tf.float32)
    print(n, np.signbit(tf.math.xlogy(x, tf.constant([2.0] * n, tf.float32)).numpy()).tolist())
# 4 [False, False, False, False]     <-- +0.0
# 8 [True, True, True, True, True, True, True, True]   <-- -0.0
```

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing check(s).`

The failing checks are the *self-checks of the bug*: the repro passes when the size-dependent sign
change reproduces (and fails if a fixed build starts returning a single consistent answer).

| check | expected under the buggy build |
|---|---|
| `xlogy(-0.0, 2.0)`, n=4, fp32 | all elements `+0.0` (scalar path) |
| `xlogy(-0.0, 2.0)`, n=8, fp32 | all elements `-0.0` (packet path) |
| size threshold per dtype | fp16/bf16/fp32 flip at n=8, fp64 at n=4 — exactly one packet |
| `xlog1py(-0.0, 1.0)` | identical size dependence |
| `reciprocal(xlogy(-0.0, 2.0))` | `+inf` for n=4 but `-inf` for n=8 |
| `tf.function(jit_compile=True)` at n=8 | `+0.0` (XLA disagrees with vectorized eager) |
| control `xlogy(+0.0, 2.0)` | `+0.0` for every size (no divergence) |

Dtype/packet-width sweep (negative-sign counts out of `n`):

| dtype | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---|---|---|---|---|---|---|---|
| float16 | 0/1 | 0/2 | 0/4 | 8/8 | 16/16 | 32/32 | 64/64 |
| bfloat16 | 0/1 | 0/2 | 0/4 | 8/8 | 16/16 | 32/32 | 64/64 |
| float32 | 0/1 | 0/2 | 0/4 | 8/8 | 16/16 | 32/32 | 64/64 |
| float64 | 0/1 | 0/2 | 4/4 | 8/8 | 16/16 | 32/32 | 64/64 |

## Root cause

The scalar functor and the vectorized functor disagree (TensorFlow 2.21.0,
`tensorflow/core/kernels/cwise_ops.h`):

```cpp
template <typename Scalar>
struct xlogy_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& x, const Scalar& y) const {
    if (x == Scalar(0.)) {
      return Scalar(0.);          // <-- -0.0 == 0.0 is true, so this returns +0.0
    }
    return x * numext::log(y);
  }
  template <typename Packet>
  ... packetOp(const Packet& x, const Packet& y) const {
    ...
    Packet mask = pcmp_eq(x, zeros);
    ...
    return pselect(mask, x, x_log_y);   // <-- returns x, i.e. preserves -0.0
  }
};
```

`xlog1py_op` has exactly the same shape (`return Scalar(0.);` in `operator()`,
`pselect(mask, x, x_log1p_y)` in `packetOp`).  The special case exists to avoid `0 * log(0)`
(`0 * -inf = nan`), but it discards the sign of the zero, while the vectorized path keeps it.

Eigen chooses the packet path when the tensor is at least one packet wide, which is why the
threshold in the table is exactly the SIMD packet width for each dtype (8 for 16/32-bit,
4 for 64-bit).

## Suggested fix

Return `x` instead of a literal zero in both scalar functors, matching the vectorized path:

```cpp
if (x == Scalar(0.)) {
  return x;        // preserves -0.0, still avoids 0 * log(0) = nan
}
```

A regression test can be written directly from the table above: for each dtype, evaluate
`xlogy([-0.0] * n, [2.0] * n)` for `n` just below and just above the packet width and require the
same sign, and require XLA and eager to agree.

## Impact

The value is numerically unchanged (`+0.0 == -0.0`), but the sign is observable and the result
depends on the *number of elements in the tensor*: a model can produce `+inf` from
`1 / xlogy(x, y)` for a 4-element batch and `-inf` for an 8-element batch, and `signbit`/
`copysign`/complex phase see the difference.  Size-dependent behaviour is also harder to catch in
tests, which tend to use one shape.

## Novelty assessment

* Tracker searches for `xlogy negative zero`, `xlogy sign of zero`, `tf.math.xlogy -0.0` and
  `xlog1py negative zero` return no matching report.
* The nearby `xlogy` issues are about **gradients** at zero (#119476 closed, #127240 open,
  #56873 closed) and about `raw_ops.Xlogy` under `jit_compile=True` (#62235, #62262 — closed and
  about the op compilation, not the sign), not about the forward value's sign of zero or the
  size-dependent scalar/vectorized split.
* NumPy/SciPy `xlogy(-0.0, 2.0)` returns `-0.0`; `torch.xlogy(-0.0, 2.0)` returns `-0.0` as well,
  so TensorFlow's `+0.0` is the outlier.

## Files

* `repro.py` — self-checking reproducer (scalar/vectorized/XLA comparison, per-dtype threshold
  table, downstream `reciprocal` impact, `+0.0` control)
* `evidence.txt` — raw observations and the kernel source snippet
* `collect_env.txt` — environment

## Anticipated objections and responses

* **"`+0.0 == -0.0`, so this is cosmetic."**  The defect is not the sign itself but that it is
  **size-dependent**: the same value and the same op return `+0.0` for a tensor shorter than one
  SIMD packet and `-0.0` at or above the packet width, and XLA returns `+0.0` for both.  The result
  therefore depends on how much data is processed together, which breaks reproducibility and cannot
  be waved away as a convention.
* **"Which layer should change?"**  The report points at both functors in `cwise_ops.h`: the scalar
  one returns a literal `Scalar(0.)`, the packet one returns `x` via `pselect`.  Making the scalar
  path return `x` is a one-line change that preserves the existing `0 * log(0)` protection.
* **"Is it already known?"**  Searches for `xlogy negative zero`, `xlogy sign of zero`,
  `tf.math.xlogy -0.0` and `xlog1py negative zero` return no match; the nearby `xlogy` issues are
  about gradients at zero or about `raw_ops` under `jit_compile`.
