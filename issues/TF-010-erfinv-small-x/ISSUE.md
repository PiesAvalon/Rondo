# `tf.math.erfinv` returns exactly `0.0` for small `|x|` (both dtypes), and is 11-19 % off just above the clamp

**Component:** TensorFlow 2.21.0, `tf.math.erfinv`, CPU, float32 and float64, eager and
`tf.function` (`jit_compile=False`); XLA (`jit_compile=True`) is correct
**Severity:** silent wrong value. Every input with `|x| <= eps/2` maps to `0.0`, although
`erfinv(x) = (sqrt(pi)/2) x + O(x^3)` is non-zero and representable there; immediately above the
clamp the same branch is 11 % (fp64) to 19 % (fp32) off.
**Status:** present in TensorFlow 2.21.0 (CPU)
**Found with:** a `torch.special` / `tf.math` vs SciPy boundary differential over unary special
functions (`artifact/hunt-v2/tools_special_cross_check.py`), then minimised to the clamp thresholds.

---

## Summary

For small `x` the inverse error function is `erfinv(x) = (sqrt(pi)/2) x + O(x^3)` with
`sqrt(pi)/2 = 0.8862269254527580`, so the result is a normal floating-point number for every input
down to the smallest subnormal.  TensorFlow's eager kernel clamps to zero below **half an epsilon** of
the input dtype and uses a coarse approximation above it:

| input | dtype | TF eager | SciPy 1.18 | relative error |
|---|---|---|---|---|
| `1e-16` | float64 | `0.0` | `8.86226925452758e-17` | 100 % |
| `1e-17` | float64 | `0.0` | `8.86226925452758e-18` | 100 % |
| `1e-20` | float64 | `0.0` | `8.86226925452758e-21` | 100 % |
| `1e-100` | float64 | `0.0` | `8.862269254527582e-101` | 100 % |
| `1e-300` | float64 | `0.0` | `8.86226925452758e-301` | 100 % |
| `1e-8` | float32 | `0.0` | `8.862269140763601e-09` | 100 % |
| `1e-15` | float64 | `9.839095376804144e-16` | `8.862269254527582e-16` | 11 % |
| `1e-8` | float64 | `8.862269200667397e-09` | `8.86226925452758e-09` | 6.1e-9 |
| `1e-7` | float32 | `1.0564647823230189e-07` | `8.862269140763601e-08` | 19 % |
| `2e-7` | float32 | `2.1129295646460378e-07` | `1.7724538281527202e-07` | 19 % |
| `0.5` | both | `0.4769362762044699` | `0.4769362762044699` | 0 (control) |

The clamp is exactly half an epsilon:

```python
tf.math.erfinv(1.1102230246251565e-16)   # 0.0       (2**-53, float64)
tf.math.erfinv(np.nextafter(2**-53, 1)) # 8.86e-17  (first unclamped input)
tf.math.erfinv(np.float32(5.9604645e-08))  # 0.0     (2**-24, float32)
```

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing replay(s).`
(55 checks per replay: the clamped values, both thresholds, the analytic reference, the lane
disagreement, the lost sign and controls.)

```python
import numpy as np, tensorflow as tf
tf.math.erfinv(tf.constant(1e-20, tf.float64))          # 0.0   <-- should be 8.86e-21
tf.math.erfinv(tf.constant(1e-8,  tf.float32))          # 0.0   <-- should be 8.86e-09
tf.function(tf.math.erfinv)(tf.constant(1e-20, tf.float64))               # 0.0
tf.function(tf.math.erfinv, jit_compile=True)(tf.constant(1e-20, tf.float64))  # 8.86e-21
```

Scope, all asserted in the reproducer:

| dimension | result |
|---|---|
| float64 clamp | `0.0` for `|x| <= 2**-53`, non-zero at the next representable input above |
| float32 clamp | `0.0` for `|x| <= 2**-24` (`erfinv(1e-8)` is `0.0`) |
| above the clamp | 11 % (fp64 at `1e-15`) and 19 % (fp32 at `1e-7`, `2e-7`) too high; still 6.1e-9 relative at fp64 `1e-8` |
| lanes | eager and `tf.function(jit_compile=False)` give `0.0`; XLA `jit_compile=True` gives the correct `8.86226925452758e-21` |
| sign | `erfinv(-1e-20)` is `0.0`, so the (negative) sign is lost as well |
| controls | `erfcinv(1e-16)`, `erfcinv(1e-8)`, `ndtri(1e-16)`, `ndtri(1e-8)`, `erfinv(0.5)`, `erfinv(1e-3)`, `erfinv(0.0)` all match SciPy |

## Root cause (pointer)

The CPU `Erfinv` kernel's small-`|x|` branch (an approximation selected below a fixed cutoff) both
clamps to zero below `eps/2` and uses a low-order approximation above it.  The cutoffs are visible as
the exact `2**-53` / `2**-24` boundaries, and the XLA lowering - which evaluates the same function
without that branch - returns the correct value, so the defect is in the eager/non-XLA kernel rather
than in the mathematical definition.  The small-argument regime has a cheap float64-accurate form,
`x * (sqrt(pi)/2) * (1 + (pi/12) x^2 + ...)`, which is what SciPy uses.

## Novelty

* `gh search issues --repo tensorflow/tensorflow` for `erfinv small`, `erfinv tiny`, `erfinv zero`,
  `erfinv inaccurate`, `erfinv precision` returns only:
  * [#121502](https://github.com/tensorflow/tensorflow/issues/121502) (open) — `inf` for the largest
    float32 value **below 1.0**, i.e. the `x -> 1` edge;
  * [#121629](https://github.com/tensorflow/tensorflow/issues/121629) (closed) — eager fp32 loses
    precision **near `x = ±1`** while XLA is accurate.
  Both are the opposite end of the domain; neither mentions the small-`|x|` clamp or a `0.0` result.
* `erf`'s own edge bugs are separate and already filed: `erf(±inf) = nan`
  ([#108483](https://github.com/tensorflow/tensorflow/issues/108483)) and `erf` saturating to NaN for
  very large finite float64 inputs
  ([#124773](https://github.com/tensorflow/tensorflow/issues/124773)) — found by the same sweep, and
  **not** drafted here because they are known.  The same applies to
  `torch.special.ndtr` flushing the lower tail
  ([pytorch#192295](https://github.com/pytorch/pytorch/issues/192295)) and to
  `tf.math.digamma(±0) = nan` ([tensorflow#121498](https://github.com/tensorflow/tensorflow/issues/121498)).
