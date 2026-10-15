# `[autodiff]` forward-mode JVP of division and `pow(x, -1)` is `NaN` / loses the zero sign where reverse-mode is correct

**Component:** TensorFlow 2.21.0, forward-mode autodiff (`tf.autodiff.ForwardAccumulator`)
**Severity:** silent wrong derivative (a `NaN` Jacobian or a sign-flipped zero) at boundary points
**Status:** present in TensorFlow 2.21.0 (CPU, eager and `tf.function`)
**Found with:** Rondo AD-consistency fuzz (`artifact/hunt-v2/ad_fuzz.py`)

---

## Summary

Forward-mode and reverse-mode autodiff are independent implementations of the same derivative, so
they must agree.  For `1 / x`, `tf.math.divide(1.0, x)` and `x ** -1`, reverse mode returns the
correct limit `-1/x²` while forward mode returns `NaN` at `x = 0` and (for `1/x`) the wrong zero
sign at `x = +inf`:

```python
import tensorflow as tf
x = tf.constant(0.0, tf.float64)

with tf.GradientTape() as tape:            # reverse mode
    tape.watch(x); y = 1 / x
tape.gradient(y, x)                        # -inf      (correct: d/dx 1/x = -1/x**2)

with tf.autodiff.ForwardAccumulator(x, tf.ones_like(x)) as acc:   # forward mode
    y = 1 / x
acc.jvp(y)                                 # nan       <-- wrong
```

The dedicated ops `tf.math.reciprocal` and `tf.raw_ops.Inv` are **correct in both modes**
(`-inf` at 0), and so are `log`, `sqrt`, `tan` and `sigmoid`, so the defect is in the forward-mode
(JVP) rules of the division / power operations rather than in the derivative of `1/x` itself.

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing check(s).`

Float64, eager, `x = 0.0` / `-0.0` / `+inf`:

| expression | `x` | reverse mode | forward mode | analytic `-1/x²` | |
|---|---|---|---|---|---|
| `1 / x` | `0.0` | `-inf` | **`nan`** | `-inf` | JVP wrong |
| `1 / x` | `-0.0` | `-inf` | **`nan`** | `-inf` | JVP wrong |
| `1 / x` | `+inf` | `-0.0` | **`+0.0`** | `-0.0` | sign lost |
| `tf.math.divide(1.0, x)` | `0.0` | `-inf` | **`nan`** | `-inf` | JVP wrong |
| `tf.math.divide(1.0, x)` | `+inf` | `-0.0` | **`+0.0`** | `-0.0` | sign lost |
| `x ** -1` | `0.0` | `-inf` | **`nan`** | `-inf` | JVP wrong |
| `x ** -1` | `+inf` | `-0.0` | **`nan`** | `-0.0` | JVP wrong |
| **control** `tf.math.reciprocal(x)` | `0.0` | `-inf` | `-inf` | `-inf` | agrees |
| **control** `tf.raw_ops.Inv(x=x)` | `0.0` | `-inf` | `-inf` | `-inf` | agrees |
| **control** `tf.math.reciprocal(x)` | `+inf` | `-0.0` | `-0.0` | `-0.0` | agrees |
| **control** `tf.math.log(x)` | `0.0` | `inf` | `inf` | — | agrees |
| **control** `tf.math.sqrt(x)` | `0.0` | `inf` | `inf` | — | agrees |
| **control** `tf.math.sigmoid(x)` | `0.0` | `0.25` | `0.25` | — | agrees |
| **control** `tf.math.tan(x)` | `0.0` | `1` | `1` | — | agrees |

The divergence reproduces for **float32 and float64**, in **eager and inside `tf.function`** (the
same numbers in both), and for `x = +0.0` and `x = -0.0` alike.

## Root cause (pointer)

Forward mode evaluates a JVP rule per primitive instead of the reverse-mode gradient function.
For `RealDiv` / `Pow` with a negative exponent the JVP is formed from products such as
`-x_dot * y / x` (and `y * x**(y-1) * x_dot`), which produce `0 * inf` / `inf - inf` at the
boundary and therefore `NaN`, whereas the reverse-mode rule simplifies to `-y² = -1/x²` and takes
the limit correctly.  `tf.math.reciprocal` has its own (correct) JVP rule, which is why the
dedicated op does not show the bug but the mathematically identical expression does.

## Impact

Mixed-mode differentiation (`ForwardAccumulator` used for Jacobian-vector products, e.g. in
optimisation, ODE integrators, or Hessian-vector products layered on top of `GradientTape`) gives a
`NaN` where reverse mode gives `-inf`, and `NaN` propagates through everything downstream.  The
sign-of-zero loss at `+inf` additionally changes `1/derivative` and `signbit` for the same input
depending on the differentiation mode.  Two mathematically identical spellings (`1/x` vs
`tf.math.reciprocal`) also disagree, so the bug is easy to hit accidentally by refactoring.

## Novelty assessment

* Tracker searches for `ForwardAccumulator reciprocal`, `truediv forward gradient`,
  `divide jvp nan` and `pow forward gradient nan` return no matching report.
* Two issues filed on the same day (2026-09-11) are in the same *category* but different ops and
  different mechanisms and were checked explicitly: **#127243** (`cumulative_logsumexp` NaN second
  derivative under nested forward-mode) and **#127233** (`segment_mean` NaN JVP for an empty
  segment).  Neither mentions division, `reciprocal`, `inv` or `x ** -1`.
* The closed #57002 (`segment_max` forward-mode NaN, 2022) is also a different op; it does show
  that maintainers treat "forward mode returns NaN where the derivative is well-defined" as a bug.

## Files

* `repro.py` — self-checking reproducer (both modes, analytic `-1/x²` reference, eager +
  `tf.function`, float32/float64, `reciprocal`/`Inv`/`log`/`sqrt`/`sigmoid`/`tan` controls,
  `--scan` grid)
* `evidence.txt` — raw observations
* `collect_env.txt` — environment
