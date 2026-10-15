# `tf.linalg.expm` collapses the exponential of a finite nilpotent matrix towards zero

**Component:** TensorFlow 2.21.0 (CPU), `tf.linalg.expm`, in eager mode and under
`tf.function(jit_compile=True)` (XLA), `float64` (and the same shape in `float32`).
**Severity:** silent wrong values, up to a total loss of the result.  For the nilpotent matrix
`A = [[0, b], [0, 0]]` (`A @ A == 0`) the exponential is exactly `I + A = [[1, b], [0, 1]]` for
**every** finite `b`; TensorFlow returns that up to `b ~ 1e6`, then the whole factor is progressively
scaled down and at `b >= 1e20` it is the **zero matrix** (or all-NaN for the transposed matrix).  At
`b = 1e18` the result is `[[1.13e-07, 1.13e+11], [0, 1.13e-07]]` instead of `[[1, 1e18], [0, 1]]` -
the off-diagonal is wrong by seven orders of magnitude while every entry is finite.
**Status:** present in 2.21.0 in eager and XLA (both agree, so this is the implementation rather than
a compiler divergence).  `scipy.linalg.expm` returns the exact factor at every scale up to `1e200`.
**Found with:** the wave-365 Rondo matrix-function probe (the finite-extreme-value method behind
TF-025/TF-026/MS-034/JAX-018), using the closed-form nilpotent witness; certified reproducer
`repro.py` in this directory (`--replays 2` -> `RESULT: 0 failing replay(s).`, 45 checks).

---

## Summary

```python
import numpy as np, tensorflow as tf
b = 1e18
A = tf.constant([[0.0, b], [0.0, 0.0]], dtype=tf.float64)   # A @ A == 0 exactly
tf.linalg.expm(A).numpy()
# [[1.12535161e-07, 1.12535161e+11], [0.0, 1.12535161e-07]]   <- everything ~1e-7 of the truth
# exact: [[1.0, 1e18], [0.0, 1.0]]
```

Error growth on `A = [[0, b], [0, 0]]` (exact `expm = [[1, b], [0, 1]]`); the diagonal and the
relative off-diagonal error are listed, and the XLA lane is identical at every row:

| `b` | diagonal | off-diagonal relative error | XLA diagonal |
|---|---|---|---|
| `1e6` | `0.999999999985` | `1.5e-11` | same |
| `1e8` | `0.999999998137` | `1.9e-09` | same |
| `1e10` | `0.999999880791` | `1.2e-07` | same |
| `1e11` | `0.999998092653` | `1.9e-06` | same |
| `1e12` | `0.999984741327` | `1.5e-05` | same |
| `1e14` | `0.998048781093` | `2.0e-03` | same |
| `1e15` | `0.984496436891` | `1.6e-02` | same |
| `1e16` | `0.882496901764` | `1.2e-01` | same |
| `1e17` | `0.135335281223` | `8.6e-01` | same |
| `1e18` | `1.12535161e-07` | `1.0` | same |
| `1e19` | `2.57220692e-56` | `1.0` | same |
| `1e20` | `0` (zero matrix) | `1.0` | same |
| `1e200` | `0` (zero matrix) | `1.0` | same |

Other witnesses and controls:

* the transposed nilpotent `[[0, 0], [1e20, 0]]` returns **all-NaN** (the exact value is
  `[[1, 0], [1e20, 1]]`), and at `1e15` its two diagonal entries disagree (`[0,0] = 0.984`,
  `[1,1] = 1.0`) although both must be `1`;
* the Jordan block `[[5, 1e20], [0, 5]]` returns the zero matrix (exact
  `e^5 * [[1, 1e20], [0, 1]]`); a 3x3 nilpotent shift collapses the same way;
* **not** a blanket large-norm failure: the rotation `[[0, 100], [-100, 0]]` matches
  `[[cos 100, sin 100], [-sin 100, cos 100]]` to 1e-12, and `[[0, 1e10], [-1e10, 0]]` stays finite
  inside `[-1, 1]` (its argument-reduction conditioning is a separate matter);
* `scipy.linalg.expm` returns the exact `I + A` at `1e12`, `1e18`, `1e20` and `1e200`, and the exact
  Jordan factor at `1e20`.

## Root cause

The nilpotent witness makes the defect a pure scaling/squaring arithmetic error: `A @ A == 0`, so the
exact exponential is `I + A` with no cancellation or precision loss anywhere, yet TensorFlow's result
is `expm(A)` multiplied by a factor that decreases with `b`.  Reading the diagonal at the sampled
points, the factor is a power of `e` at integer exponents (`0.135335281223 = e^-2`,
`1.12535161e-07 = e^-16`, `2.57220692e-56 = e^-128`), i.e. TensorFlow is effectively computing
`expm(A - c I)` for a `c` that grows with `log2(b)` instead of `expm(A)` - the scaling exponent is
being applied to the identity term (or the squaring loop is being replaced by a subtraction of
`2^s I`).  Whatever the exact code path, the observable is a systematic, finite, large error on a
matrix whose exact exponential is trivial, and the XLA lane inherits it because it shares the
implementation.

The relevant input range is not extreme: the relative error is already `1.9e-6` at `b = 1e11` and
`1.5e-2` at `b = 1e15`, both routine magnitudes for stiff ODE systems, rates matrices and
`expm`-based attention/normalising-flow models, and the failure becomes total at `1e20`.

## Impact

* `expm` is used for stiff linear ODEs, continuous-time Markov chains, matrix functions in
  normalising flows and (in the recent literature) attention kernels.  A matrix with a large norm but
  a well-behaved exponential - the nilpotent/Jordan cases are the clearest - silently loses the
  identity term, so the result is a scaled-down version of the truth or zero, with no NaN to flag it.
* The all-NaN transpose case and the zero-matrix case are visible, but the middle range
  (`1e11`-`1e17`) returns plausible finite numbers with errors from 1e-6 to 86%, which is the
  dangerous part for downstream use.

## Suggested fix

Correct the scaling/squaring (or Padé) implementation so the identity term is not rescaled with the
exponent: `expm(A) = expm(A / 2^s)^(2^s)` with the exact `I` in the base, and clamp/handle the
squaring count explicitly rather than subtracting it from the diagonal.  `scipy.linalg.expm` (and the
closed form) is a reference; a regression test is the nilpotent witness above at
`b = 1e12 ... 1e200`, which must return `[[1, b], [0, 1]]`.

## Controls (all in `repro.py`, 45 checks, 0 failing on each of 2 replays)

* **Exactness control** - `expm([[0,b],[0,0]]) == [[1,b],[0,1]]` to 1e-12 at `b = 1, 1e3` and to
  1e-8 at `b = 1e6`, with `A @ A == 0` asserted at every scale.
* **Common-factor control** - at every failing scale the diagonal and the off-diagonal error track
  each other to 1e-12 relative, i.e. the whole matrix is scaled by one wrong factor (which is what
  makes the mechanism a scaling error rather than an accuracy loss).
* **Category control** - the rotation matrices (and the fp32 controls at `1e8`/`1e10`) are exact, so
  large norms alone are not the trigger; the collapse needs the nilpotent/Jordan structure.
* **Lane control** - eager and XLA agree (to 1e-9) at `1e15`, `1e18` and `1e20`, so this is the shared
  implementation, not a compiler bridge.
* **Reference** - `scipy.linalg.expm` returns the exact factor at every scale up to `1e200`, and the
  exact-value checks use the closed form `I + A` rather than another framework's output.

## Novelty

Searched the TensorFlow tracker for `tf.linalg.expm` + wrong/zeros/large values/overflow and
`matrix exponential` + overflow/nan: the only related items are **#122575** (a float16-support fix,
different dtype) and **#15465** (a 2017 missing-gradient issue); no report of magnitude-dependent
wrong values.  The campaign's own TF linalg differential row covered `expm` only at degenerate or
moderate inputs (its explanations concern `norm`/`logm`/`cond`/`cholesky`, not large-norm `expm`), and
the neighbouring finite-extreme drafts are TF-025 (`inv`/`solve` determinant overflow), TF-026
(`tf.linalg.qr`) and TF-024 (XLA `cholesky` with an `inf` diagonal) - all different operators and
different failure modes.

## Environment

```
tensorflow 2.21.0 (CPU), numpy 2.5.3, Python 3.12.13, WSL2; scipy reference from the
jax/jaxlib 0.10.2 interpreter - see collect_env.txt
```

## Anticipated objections and responses

* **"Extreme norms are out of contract for `expm`."**  The witness is the mildest possible case: a
  nilpotent matrix, where `expm(A) = I + A` by definition with no cancellation.  The failure starts at
  `1e11` (1.9e-6 error) and `scipy` is exact to `1e200`.
* **"This is just an accuracy limitation."**  At `1e18` the off-diagonal is off by seven orders of
  magnitude and at `1e20` the entire factor is zero - a structural loss of the result, not rounding,
  and the diagonal/off-diagonal share one wrong factor.
* **"The rotation at `1e10` is also wrong, so it is a general large-norm issue."**  The rotation
  values are conditioning-limited by trigonometric argument reduction; the nilpotent witness has no
  such conditioning (its exact exponential is a polynomial in `b`).
* **"Use `tf.function`/XLA and it will be fine."**  The XLA lane returns the same wrong values.
